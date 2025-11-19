# llm - 2025_11

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13998v1">LoCoBench-Agent: An Interactive Benchmark for LLM Agents in Long-Context Software Engineering</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 54-pages
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) evolve into sophisticated autonomous agents capable of complex software development tasks, evaluating their real-world capabilities becomes critical. While existing benchmarks like LoCoBench~\cite{qiu2025locobench} assess long-context code understanding, they focus on single-turn evaluation and cannot capture the multi-turn interactive nature, tool usage patterns, and adaptive reasoning required by real-world coding agents. We introduce \textbf{LoCoBench-Agent}, a comprehensive evaluation framework specifically designed to assess LLM agents in realistic, long-context software engineering workflows. Our framework extends LoCoBench's 8,000 scenarios into interactive agent environments, enabling systematic evaluation of multi-turn conversations, tool usage efficiency, error recovery, and architectural consistency across extended development sessions. We also introduce an evaluation methodology with 9 metrics across comprehension and efficiency dimensions. Our framework provides agents with 8 specialized tools (file operations, search, code analysis) and evaluates them across context lengths ranging from 10K to 1M tokens, enabling precise assessment of long-context performance. Through systematic evaluation of state-of-the-art models, we reveal several key findings: (1) agents exhibit remarkable long-context robustness; (2) comprehension-efficiency trade-off exists with negative correlation, where thorough exploration increases comprehension but reduces efficiency; and (3) conversation efficiency varies dramatically across models, with strategic tool usage patterns differentiating high-performing agents. As the first long-context LLM agent benchmark for software engineering, LoCoBench-Agent establishes a rigorous foundation for measuring agent capabilities, identifying performance gaps, and advancing autonomous software development at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13994v1">Hint-Augmented Re-ranking: Efficient Product Search using LLM-Based Query Decomposition</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 AACL 2025
    </div>
    <details class="paper-abstract">
      Search queries with superlatives (e.g., best, most popular) require comparing candidates across multiple dimensions, demanding linguistic understanding and domain knowledge. We show that LLMs can uncover latent intent behind these expressions in e-commerce queries through a framework that extracts structured interpretations or hints. Our approach decomposes queries into attribute-value hints generated concurrently with retrieval, enabling efficient integration into the ranking pipeline. Our method improves search performanc eby 10.9 points in MAP and ranking by 5.9 points in MRR over baselines. Since direct LLM-based reranking faces prohibitive latency, we develop an efficient approach transferring superlative interpretations to lightweight models. Our findings provide insights into how superlative semantics can be represented and transferred between models, advancing linguistic interpretation in retrieval systems while addressing practical deployment constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13984v1">Node-Level Uncertainty Estimation in LLM-Generated SQL</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      We present a practical framework for detecting errors in LLM-generated SQL by estimating uncertainty at the level of individual nodes in the query's abstract syntax tree (AST). Our approach proceeds in two stages. First, we introduce a semantically aware labeling algorithm that, given a generated SQL and a gold reference, assigns node-level correctness without over-penalizing structural containers or alias variation. Second, we represent each node with a rich set of schema-aware and lexical features - capturing identifier validity, alias resolution, type compatibility, ambiguity in scope, and typo signals - and train a supervised classifier to predict per-node error probabilities. We interpret these probabilities as calibrated uncertainty, enabling fine-grained diagnostics that pinpoint exactly where a query is likely to be wrong. Across multiple databases and datasets, our method substantially outperforms token log-probabilities: average AUC improves by +27.44% while maintaining robustness under cross-database evaluation. Beyond serving as an accuracy signal, node-level uncertainty supports targeted repair, human-in-the-loop review, and downstream selective execution. Together, these results establish node-centric, semantically grounded uncertainty estimation as a strong and interpretable alternative to aggregate sequence level confidence measures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.04418v2">Characterizing Multi-Hunk Patches: Divergence, Proximity, and LLM Repair Challenges</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Multi-hunk bugs, where fixes span disjoint regions of code, are common in practice, yet remain underrepresented in automated repair. Existing techniques and benchmarks pre-dominantly target single-hunk scenarios, overlooking the added complexity of coordinating semantically related changes across the codebase. In this work, we characterize HUNK4J, a dataset of multi-hunk patches derived from 372 real-world defects. We propose hunk divergence, a metric that quantifies the variation among edits in a patch by capturing lexical, structural, and file-level differences, while incorporating the number of hunks involved. We further define spatial proximity, a classification that models how hunks are spatially distributed across the program hierarchy. Our empirical study spanning six LLMs reveals that model success rates decline with increased divergence and spatial dispersion. Notably, when using the LLM alone, no model succeeds in the most dispersed Fragment class. These findings highlight a critical gap in LLM capabilities and motivate divergence-aware repair strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10819v2">LLM-as-a-Grader: Practical Insights from Large Language Model for Short-Answer and Report Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly explored for educational tasks such as grading, yet their alignment with human evaluation in real classrooms remains underexamined. In this study, we investigate the feasibility of using an LLM (GPT-4o) to evaluate short-answer quizzes and project reports in an undergraduate Computational Linguistics course. We collect responses from approximately 50 students across five quizzes and receive project reports from 14 teams. LLM-generated scores are compared against human evaluations conducted independently by the course teaching assistants (TAs). Our results show that GPT-4o achieves strong correlation with human graders (up to 0.98) and exact score agreement in 55\% of quiz cases. For project reports, it also shows strong overall alignment with human grading, while exhibiting some variability in scoring technical, open-ended responses. We release all code and sample data to support further research on LLMs in educational assessment. This work highlights both the potential and limitations of LLM-based grading systems and contributes to advancing automated grading in real-world academic settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.12274v3">Hierarchical LLMs In-the-Loop Optimization for Real-Time Multi-Robot Target Tracking under Unknown Hazards</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Real-time multi-robot coordination in hazardous and adversarial environments requires fast, reliable adaptation to dynamic threats. While Large Language Models (LLMs) offer strong high-level reasoning capabilities, the lack of safety guarantees limits their direct use in critical decision-making. In this paper, we propose a hierarchical optimization framework that integrates LLMs into the decision loop for multi-robot target tracking in dynamic and hazardous environments. Rather than generating control actions directly, LLMs are used to generate task configuration and adjust parameters in a bi-level task allocation and planning problem. We formulate multi-robot coordination for tracking tasks as a bi-level optimization problem, with LLMs to reason about potential hazards in the environment and the status of the robot team and modify both the inner and outer levels of the optimization. This hierarchical approach enables real-time adjustments to the robots' behavior. Additionally, a human supervisor can offer broad guidance and assessments to address unexpected dangers, model mismatches, and performance issues arising from local minima. We validate our proposed framework in both simulation and real-world experiments with comprehensive evaluations, demonstrating its effectiveness and showcasing its capability for safe LLM integration for multi-robot systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13950v1">NL-DPE: An Analog In-memory Non-Linear Dot Product Engine for Efficient CNN and LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Resistive Random Access Memory (RRAM) based in-memory computing (IMC) accelerators offer significant performance and energy advantages for deep neural networks (DNNs), but face three major limitations: (1) they support only \textit{static} dot-product operations and cannot accelerate arbitrary non-linear functions or data-dependent multiplications essential to modern LLMs; (2) they demand large, power-hungry analog-to-digital converter (ADC) circuits; and (3) mapping model weights to device conductance introduces errors from cell nonidealities. These challenges hinder scalable and accurate IMC acceleration as models grow. We propose NL-DPE, a Non-Linear Dot Product Engine that overcomes these barriers. NL-DPE augments crosspoint arrays with RRAM-based Analog Content Addressable Memory (ACAM) to execute arbitrary non-linear functions and data-dependent matrix multiplications in the analog domain by transforming them into decision trees, fully eliminating ADCs. To address device noise, NL-DPE uses software-based Noise Aware Fine-tuning (NAF), requiring no in-device calibration. Experiments show that NL-DPE delivers 28X energy efficiency and 249X speedup over a GPU baseline, and 22X energy efficiency and 245X speedup over existing IMC accelerators, while maintaining high accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13909v1">Mind the Gap: Evaluating LLM Understanding of Human-Taught Road Safety Principles</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Following road safety norms is non-negotiable not only for humans but also for the AI systems that govern autonomous vehicles. In this work, we evaluate how well multi-modal large language models (LLMs) understand road safety concepts, specifically through schematic and illustrative representations. We curate a pilot dataset of images depicting traffic signs and road-safety norms sourced from school text books and use it to evaluate models capabilities in a zero-shot setting. Our preliminary results show that these models struggle with safety reasoning and reveal gaps between human learning and model interpretation. We further provide an analysis of these performance gaps for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.21359v3">Can Machines Think Like Humans? A Behavioral Evaluation of LLM Agents in Dictator Games</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      As Large Language Model (LLM)-based agents increasingly engage with human society, how well do we understand their prosocial behaviors? We (1) investigate how LLM agents' prosocial behaviors can be induced by different personas and benchmarked against human behaviors; and (2) introduce a social science approach to evaluate LLM agents' decision-making. We explored how different personas and experimental framings affect these AI agents' altruistic behavior in dictator games and compared their behaviors within the same LLM family, across various families, and with human behaviors. The findings reveal that merely assigning a human-like identity to LLMs does not produce human-like behaviors. These findings suggest that LLM agents' reasoning does not consistently exhibit textual markers of human decision-making in dictator games and that their alignment with human behavior varies substantially across model architectures and prompt formulations; even worse, such dependence does not follow a clear pattern. As society increasingly integrates machine intelligence, "Prosocial AI" emerges as a promising and urgent research direction in philanthropic studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13900v1">What Works for 'Lost-in-the-Middle' in LLMs? A Study on GM-Extract and Mitigations</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 To be submitted for publication
    </div>
    <details class="paper-abstract">
      The diminishing ability of large language models (LLMs) to effectively utilize long-range context-the "lost-in-the-middle" phenomenon-poses a significant challenge in retrieval-based LLM applications. To study the impact of this phenomenon in a real-world application setting, we introduce GM-Extract, a novel benchmark dataset meticulously designed to evaluate LLM performance on retrieval of control variables. To accurately diagnose failure modes, we propose a simple yet elegant evaluation system using two distinct metrics: one for spatial retrieval capability (Document Metric) and the other for semantic retrieval capability (Variable Extraction Metric). We conduct a systematic evaluation of 7-8B parameter models on two multi-document tasks (key-value extraction and question-answering), demonstrating a significant change in retrieval performance simply by altering how the data is represented in the context window. While a distinct U-shaped curve was not consistently observed, our analysis reveals a clear pattern of performance across models, which we further correlate with perplexity scores. Furthermore, we perform a literature survey of mitigation methods, which we categorize into two distinct approaches: black-box and white-box methods. We then apply these techniques to our benchmark, finding that their efficacy is highly nuanced. Our evaluation highlights scenarios where these strategies successfully improve performance, as well as surprising cases where they lead to a negative impact, providing a comprehensive understanding of their utility in a practical context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13876v1">QwenCLIP: Boosting Medical Vision-Language Pretraining via LLM Embeddings and Prompt tuning</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 This work has been submitted to the IEEE ISBI for possible publication
    </div>
    <details class="paper-abstract">
      Contrastive Language-Image Pretraining (CLIP) has demonstrated strong generalization for vision-language tasks in computer vision and medical domains, yet its text encoder accepts only up to 77 tokens, which limits its ability to represent long and information-rich radiology reports. Recent adaptations using domain-specific encoders, such as PubMedBERT or ClinicalBERT, mitigate this issue by leveraging medical corpora, but remain constrained by their limited input length (typically 512 tokens) and relatively shallow semantic understanding. To address these limitations, we propose QwenCLIP, a vision-language framework that replaces CLIP's text encoder with a large language model (LLM)-based embedding module (e.g., Qwen3-Embedding) and introduces learnable prompts to enhance cross-modal alignment. By leveraging the extended context window and richer representations of LLMs, QwenCLIP captures comprehensive medical semantics from long-form clinical text, substantially improving medical image-text alignment and downstream performance on radiology benchmarks. Our code is publicly available at https://github.com/Wxy-24/QwenCLIP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10459v2">LocalBench: Benchmarking LLMs on County-Level Local Knowledge and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been widely evaluated on macro-scale geographic tasks, such as global factual recall, event summarization, and regional reasoning. Yet, their ability to handle hyper-local knowledge remains poorly understood. This gap is increasingly consequential as real-world applications, from civic platforms to community journalism, demand AI systems that can reason about neighborhood-specific dynamics, cultural narratives, and local governance. Existing benchmarks fall short in capturing this complexity, often relying on coarse-grained data or isolated references. We present LocalBench, the first benchmark designed to systematically evaluate LLMs on county-level local knowledge across the United States. Grounded in the Localness Conceptual Framework, LocalBench includes 14,782 validated question-answer pairs across 526 U.S. counties in 49 states, integrating diverse sources such as Census statistics, local subreddit discourse, and regional news. It spans physical, cognitive, and relational dimensions of locality. Using LocalBench, we evaluate 13 state-of-the-art LLMs under both closed-book and web-augmented settings. Our findings reveal critical limitations: even the best-performing models reach only 56.8% accuracy on narrative-style questions and perform below 15.5% on numerical reasoning. Moreover, larger model size and web augmentation do not guarantee better performance, for example, search improves Gemini's accuracy by +13.6%, but reduces GPT-series performance by -11.4%. These results underscore the urgent need for language models that can support equitable, place-aware AI systems: capable of engaging with the diverse, fine-grained realities of local communities across geographic and cultural contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10687v2">Who Gets the Reward, Who Gets the Blame? Evaluation-Aligned Training Signals for Multi-LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 Withdrawing temporarily to coordinate revisions with co-authors. A revised version will be resubmitted
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) in multi-agent systems (MAS) have shown promise for complex tasks, yet current training methods lack principled ways to connect system-level evaluation with agent-level and message-level learning. We propose a theoretical framework that unifies cooperative game-theoretic attribution with process reward modeling to transform system evaluation into agent credit and then into response-level signals. Unlike prior approaches that rely only on attribution (e.g., Shapley) or step-level labels (e.g., PRM), our method produces local, signed, and credit-conserving signals. In success cases, Shapley-based credit assignment fairly allocates outcomes across agents and is refined into per-message rewards that promote cooperation while discouraging redundancy or sabotage. In failure cases, first-error localization yields repair-aware preferences that penalize harmful steps while rewarding corrective attempts. The resulting signals are bounded, cooperative, and directly compatible with reinforcement-based or preference-based post-training, providing a unified and auditable pathway from global evaluation to local supervision in LLM multi-agent training. Our contribution is conceptual: we present a theoretical foundation and training signals, leaving empirical validation for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12823v1">Enhancing LLM Code Generation Capabilities through Test-Driven Development and Code Interpreter</a></div>
    <div class="paper-meta">
      📅 2025-11-16
      | 💬 AACL-IJCNLP 2025 Workshop BLP Shared Task 2, 6 pages, 7 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Over the past few years, improving LLM code generation capabilities has been a key focus in NLP research. Despite Bengali having 242 million native speakers worldwide, it receives little attention when it comes to training LLMs. More recently, various fine-tuning and augmented generation techniques have been employed to significantly enhance code generation performance. However, they require considerable expertise and resources to utilize effectively as an end user. The goal of our work is to democratize access to powerful code generation tools in resource-constrained emerging markets, enabling users to leverage them in their native language. We introduce a novel approach that combines Test-Driven Development (TDD) and Code Interpreter (CI), utilizing open-weight models, which improves the baseline accuracy for code generation with Bengali prompts and achieves an overall accuracy of 85%. Our approach requires no finetuning and proves that even the smallest models in the same family can attain up to 98% accuracy compared to the largest models. All of our results are publicly shared in GitHub for validation and reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12821v1">BioMedJImpact: A Comprehensive Dataset and LLM Pipeline for AI Engagement and Scientific Impact Analysis of Biomedical Journals</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      Assessing journal impact is central to scholarly communication, yet existing open resources rarely capture how collaboration structures and artificial intelligence (AI) research jointly shape venue prestige in biomedicine. We present BioMedJImpact, a large-scale, biomedical-oriented dataset designed to advance journal-level analysis of scientific impact and AI engagement. Built from 1.74 million PubMed Central articles across 2,744 journals, BioMedJImpact integrates bibliometric indicators, collaboration features, and LLM-derived semantic indicators for AI engagement. Specifically, the AI engagement feature is extracted through a reproducible three-stage LLM pipeline that we propose. Using this dataset, we analyze how collaboration intensity and AI engagement jointly influence scientific impact across pre- and post-pandemic periods (2016-2019, 2020-2023). Two consistent trends emerge: journals with higher collaboration intensity, particularly those with larger and more diverse author teams, tend to achieve greater citation impact, and AI engagement has become an increasingly strong correlate of journal prestige, especially in quartile rankings. To further validate the three-stage LLM pipeline we proposed for deriving the AI engagement feature, we conduct human evaluation, confirming substantial agreement in AI relevance detection and consistent subfield classification. Together, these contributions demonstrate that BioMedJImpact serves as both a comprehensive dataset capturing the intersection of biomedicine and AI, and a validated methodological framework enabling scalable, content-aware scientometric analysis of scientific impact and innovation dynamics. Code is available at https://github.com/JonathanWry/BioMedJImpact.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12817v1">Assessing Automated Fact-Checking for Medical LLM Responses with Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2025-11-16
      | 💬 Accepted as a conference paper at AAAI'26
    </div>
    <details class="paper-abstract">
      The recent proliferation of large language models (LLMs) holds the potential to revolutionize healthcare, with strong capabilities in diverse medical tasks. Yet, deploying LLMs in high-stakes healthcare settings requires rigorous verification and validation to understand any potential harm. This paper investigates the reliability and viability of using medical knowledge graphs (KGs) for the automated factuality evaluation of LLM-generated responses. To ground this investigation, we introduce FAITH, a framework designed to systematically probe the strengths and limitations of this KG-based approach. FAITH operates without reference answers by decomposing responses into atomic claims, linking them to a medical KG, and scoring them based on evidence paths. Experiments on diverse medical tasks with human subjective evaluations demonstrate that KG-grounded evaluation achieves considerably higher correlations with clinician judgments and can effectively distinguish LLMs with varying capabilities. It is also robust to textual variances. The inherent explainability of its scoring can further help users understand and mitigate the limitations of current LLMs. We conclude that while limitations exist, leveraging KGs is a prominent direction for automated factuality assessment in healthcare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06838v3">P3-LLM: An Integrated NPU-PIM Accelerator for LLM Inference Using Hybrid Numerical Formats</a></div>
    <div class="paper-meta">
      📅 2025-11-16
      | 💬 Preprint. Under review
    </div>
    <details class="paper-abstract">
      The substantial memory bandwidth and computational demands of large language models (LLMs) present critical challenges for efficient inference. To tackle this, the literature has explored heterogeneous systems that combine neural processing units (NPUs) with DRAM-based processing-in-memory (PIM) for LLM acceleration. However, existing high-precision (e.g., FP16) PIM compute units incur significant area and power overhead in DRAM technology, limiting the effective computation throughput. In this paper, we introduce P3-LLM, a novel NPU-PIM integrated accelerator for LLM inference using hybrid numerical formats. Our approach is threefold: First, we propose a flexible mixed-precision quantization scheme, which leverages hybrid numerical formats to quantize different LLM operands with high compression efficiency and minimal accuracy loss. Second, we architect an efficient PIM accelerator for P3-LLM, featuring enhanced compute units to support hybrid numerical formats. Our careful choice of numerical formats allows to co-design low-precision PIM compute units that significantly boost the computation throughput under iso-area constraints. Third, we optimize the low-precision dataflow of different LLM modules by applying operator fusion to minimize the overhead of runtime dequantization. Evaluation on a diverse set of representative LLMs and tasks demonstrates that P3-LLM achieves state-of-the-art accuracy in terms of both KV-cache quantization and weight-activation quantization. Combining the proposed quantization scheme with PIM architecture co-design, P3-LLM yields an average of $4.9\times$, $2.0\times$, and $3.4\times$ speedups over the state-of-the-art LLM accelerators HBM-PIM, Ecco, and Pimba, respectively. Our quantization code is available at https://github.com/yc2367/P3-LLM.git
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.16785v2">Interpreting the Effects of Quantization on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-16
      | 💬 Accepted to AACL 2025 Main
    </div>
    <details class="paper-abstract">
      Quantization offers a practical solution to deploy LLMs in resource-constraint environments. However, its impact on internal representations remains understudied, raising questions about the reliability of quantized models. In this study, we employ a range of interpretability techniques to investigate how quantization affects model and neuron behavior. We analyze multiple LLMs under 4-bit and 8-bit quantization. Our findings reveal that the impact of quantization on model calibration is generally minor. Analysis of neuron activations indicates that the number of dead neurons, i.e., those with activation values close to 0 across the dataset, remains consistent regardless of quantization. In terms of neuron contribution to predictions, we observe that smaller full precision models exhibit fewer salient neurons, whereas larger models tend to have more, with the exception of Llama-2-7B. The effect of quantization on neuron redundancy varies across models. Overall, our findings suggest that effect of quantization may vary by model and tasks, however, we did not observe any drastic change which may discourage the use of quantization as a reliable model compression technique.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12782v1">LLM Reinforcement in Context</a></div>
    <div class="paper-meta">
      📅 2025-11-16
      | 💬 4 pages
    </div>
    <details class="paper-abstract">
      Current Large Language Model alignment research mostly focuses on improving model robustness against adversarial attacks and misbehavior by training on examples and prompting. Research has shown that LLM jailbreak probability increases with the size of the user input or conversation length. There is a lack of appropriate research into means of strengthening alignment which also scale with user input length. We propose interruptions as a possible solution to this problem. Interruptions are control sentences added to the user input approximately every x tokens for some arbitrary x. We suggest that this can be generalized to the Chain-of-Thought process to prevent scheming.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.00833v2">HumanoidGen: Data Generation for Bimanual Dexterous Manipulation via LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-16
      | 💬 Project Page: https://openhumanoidgen.github.io
    </div>
    <details class="paper-abstract">
      For robotic manipulation, existing robotics datasets and simulation benchmarks predominantly cater to robot-arm platforms. However, for humanoid robots equipped with dual arms and dexterous hands, simulation tasks and high-quality demonstrations are notably lacking. Bimanual dexterous manipulation is inherently more complex, as it requires coordinated arm movements and hand operations, making autonomous data collection challenging. This paper presents HumanoidGen, an automated task creation and demonstration collection framework that leverages atomic dexterous operations and LLM reasoning to generate relational constraints. Specifically, we provide spatial annotations for both assets and dexterous hands based on the atomic operations, and perform an LLM planner to generate a chain of actionable spatial constraints for arm movements based on object affordances and scenes. To further improve planning ability, we employ a variant of Monte Carlo tree search to enhance LLM reasoning for long-horizon tasks and insufficient annotation. In experiments, we create a novel benchmark with augmented scenarios to evaluate the quality of the collected data. The results show that the performance of the 2D and 3D diffusion policies can scale with the generated dataset. Project page is https://openhumanoidgen.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12751v1">Are LLMs The Way Forward? A Case Study on LLM-Guided Reinforcement Learning for Decentralized Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      Autonomous vehicle navigation in complex environments such as dense and fast-moving highways and merging scenarios remains an active area of research. A key limitation of RL is its reliance on well-specified reward functions, which often fail to capture the full semantic and social complexity of diverse, out-of-distribution situations. As a result, a rapidly growing line of research explores using Large Language Models (LLMs) to replace or supplement RL for direct planning and control, on account of their ability to reason about rich semantic context. However, LLMs present significant drawbacks: they can be unstable in zero-shot safety-critical settings, produce inconsistent outputs, and often depend on expensive API calls with network latency. This motivates our investigation into whether small, locally deployed LLMs (< 14B parameters) can meaningfully support autonomous highway driving through reward shaping rather than direct control. We present a case study comparing RL-only, LLM-only, and hybrid approaches, where LLMs augment RL rewards by scoring state-action transitions during training, while standard RL policies execute at test time. Our findings reveal that RL-only agents achieve moderate success rates (73-89%) with reasonable efficiency, LLM-only agents can reach higher success rates (up to 94%) but with severely degraded speed performance, and hybrid approaches consistently fall between these extremes. Critically, despite explicit efficiency instructions, LLM-influenced approaches exhibit systematic conservative bias with substantial model-dependent variability, highlighting important limitations of current small LLMs for safety-critical control tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12728v1">On the Brittleness of LLMs: A Journey around Set Membership</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) achieve superhuman performance on complex reasoning tasks, yet often fail on much simpler problems, raising concerns about their reliability and interpretability. We investigate this paradox through a focused study with two key design features: simplicity, to expose basic failure modes, and scale, to enable comprehensive controlled experiments. We focus on set membership queries -- among the most fundamental forms of reasoning -- using tasks like ``Is apple an element of the set \{pear, plum, apple, raspberry\}?''. We conduct a systematic empirical evaluation across prompt phrasing, semantic structure, element ordering, and model choice. Our large-scale analysis reveals that LLM performance on this elementary task is consistently brittle, and unpredictable across all dimensions, suggesting that the models' ``understanding'' of the set concept is fragmented and convoluted at best. Our work demonstrates that the large-scale experiments enabled by the simplicity of the problem allow us to map and analyze the failure modes comprehensively, making this approach a valuable methodology for LLM evaluation in general.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.15859v2">InfiMed-ORBIT: Aligning LLMs on Open-Ended Complex Tasks via Rubric-Based Incremental Training</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      Reinforcement learning has powered many of the recent breakthroughs in large language models, especially for tasks where rewards can be computed automatically, such as code generation. However, these methods deteriorate in open-ended domains like medical consultation, where feedback is inherently ambiguous, highly context-dependent, and cannot be reduced to a reliable scalar signal. In such settings, RL must either rely on supervision-intensive reward models that often fail to generalize, or it falls into pathological behaviors such as reward hacking - an especially troubling risk for high-stakes medical dialogue. To address these limitations, we introduce ORBIT, an open-ended rubric-based incremental training framework for high-stakes medical dialogue. ORBIT integrates synthetic dialogue generation with dynamically constructed rubrics that serve as adaptive guides for incremental RL. Instead of relying on external medical knowledge bases or handcrafted rule sets, ORBIT uses rubric-driven feedback to steer the learning process. Its judge component can be instantiated with general-purpose instruction-following LLMs, removing the need for any task-specific fine-tuning. Applied to the Qwen3-4B-Instruct model, ORBIT raises the HealthBench-Hard score from 7.0 to 27.5 using only 2k training samples, achieving SOTA performance for models at this scale. With larger rubric datasets, ORBIT-trained models further compete with the strongest open-source baselines on HealthBench-Hard. Our analysis shows that rubric-guided RL consistently improves consultation quality across diverse medical scenarios. We also apply such rubric generation and training pipeline to InfoBench, where ORBIT enhances instruction-following performance, highlighting the generality of rubric-based feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12710v1">Evolve the Method, Not the Prompts: Evolutionary Synthesis of Jailbreak Attacks on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      Automated red teaming frameworks for Large Language Models (LLMs) have become increasingly sophisticated, yet they share a fundamental limitation: their jailbreak logic is confined to selecting, combining, or refining pre-existing attack strategies. This binds their creativity and leaves them unable to autonomously invent entirely new attack mechanisms. To overcome this gap, we introduce \textbf{EvoSynth}, an autonomous framework that shifts the paradigm from attack planning to the evolutionary synthesis of jailbreak methods. Instead of refining prompts, EvoSynth employs a multi-agent system to autonomously engineer, evolve, and execute novel, code-based attack algorithms. Crucially, it features a code-level self-correction loop, allowing it to iteratively rewrite its own attack logic in response to failure. Through extensive experiments, we demonstrate that EvoSynth not only establishes a new state-of-the-art by achieving an 85.5\% Attack Success Rate (ASR) against highly robust models like Claude-Sonnet-4.5, but also generates attacks that are significantly more diverse than those from existing methods. We release our framework to facilitate future research in this new direction of evolutionary synthesis of jailbreak methods. Code is available at: https://github.com/dongdongunique/EvoSynth.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.02503v2">OptiHive: Ensemble Selection for LLM-Based Optimization via Statistical Modeling</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      LLM-based solvers have emerged as a promising means of automating problem modeling and solving. However, they remain unreliable and often depend on iterative repair loops that result in significant latency. We introduce OptiHive, a framework that enhances any solver-generation pipeline to produce higher-quality solvers from natural-language descriptions of optimization problems. OptiHive uses a single batched generation to produce diverse components (solvers, problem instances, and validation tests) and filters out erroneous components to ensure fully interpretable outputs. Accounting for the imperfection of the generated components, we employ a statistical model to infer their true performance, enabling principled uncertainty quantification and solver selection. On tasks ranging from traditional optimization problems to challenging variants of the Multi-Depot Vehicle Routing Problem, OptiHive significantly outperforms baselines, increasing the optimality rate from 5% to 92% on the most complex problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.03369v2">Silenced Biases: The Dark Side LLMs Learned to Refuse</a></div>
    <div class="paper-meta">
      📅 2025-11-16
      | 💬 Accepted to The 40th Annual AAAI Conference on Artificial Intelligence - AI Alignment Track (Oral)
    </div>
    <details class="paper-abstract">
      Safety-aligned large language models (LLMs) are becoming increasingly widespread, especially in sensitive applications where fairness is essential and biased outputs can cause significant harm. However, evaluating the fairness of models is a complex challenge, and approaches that do so typically utilize standard question-answer (QA) styled schemes. Such methods often overlook deeper issues by interpreting the model's refusal responses as positive fairness measurements, which creates a false sense of fairness. In this work, we introduce the concept of silenced biases, which are unfair preferences encoded within models' latent space and are effectively concealed by safety-alignment. Previous approaches that considered similar indirect biases often relied on prompt manipulation or handcrafted implicit queries, which present limited scalability and risk contaminating the evaluation process with additional biases. We propose the Silenced Bias Benchmark (SBB), which aims to uncover these biases by employing activation steering to reduce model refusals during QA. SBB supports easy expansion to new demographic groups and subjects, presenting a fairness evaluation framework that encourages the future development of fair models and tools beyond the masking effects of alignment training. We demonstrate our approach over multiple LLMs, where our findings expose an alarming distinction between models' direct responses and their underlying fairness issues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.12689v2">From Delegates to Trustees: How Optimizing for Long-Term Interests Shapes Bias and Alignment in LLM</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown promising accuracy in predicting survey responses and policy preferences, which has increased interest in their potential to represent human interests in various domains. Most existing research has focused on "behavioral cloning", effectively evaluating how well models reproduce individuals' expressed preferences. Drawing on theories of political representation, we highlight an underexplored design trade-off: whether AI systems should act as delegates, mirroring expressed preferences, or as trustees, exercising judgment about what best serves an individual's interests. This trade-off is closely related to issues of LLM sycophancy, where models can encourage behavior or validate beliefs that may be aligned with a user's short-term preferences, but is detrimental to their long-term interests. Through a series of experiments simulating votes on various policy issues in the U.S. context, we apply a temporal utility framework that weighs short and long-term interests (simulating a trustee role) and compare voting outcomes to behavior-cloning models (simulating a delegate). We find that trustee-style predictions weighted toward long-term interests produce policy decisions that align more closely with expert consensus on well-understood issues, but also show greater bias toward models' default stances on topics lacking clear agreement. These findings reveal a fundamental trade-off in designing AI systems to represent human interests. Delegate models better preserve user autonomy but may diverge from well-supported policy positions, while trustee models can promote welfare on well-understood issues yet risk paternalism and bias on subjective topics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12661v1">Reason-KE++: Aligning the Process, Not Just the Outcome, for Faithful LLM Knowledge Editing</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      Aligning Large Language Models (LLMs) to be faithful to new knowledge in complex, multi-hop reasoning tasks is a critical, yet unsolved, challenge. We find that SFT-based methods, e.g., Reason-KE, while state-of-the-art, suffer from a "faithfulness gap": they optimize for format mimicry rather than sound reasoning. This gap enables the LLM's powerful parametric priors to override new contextual facts, resulting in critical factual hallucinations (e.g., incorrectly reasoning "Houston" from "NASA" despite an explicit edit). To solve this core LLM alignment problem, we propose Reason-KE++, an SFT+RL framework that instills process-level faithfulness. Its core is a Stage-aware Reward mechanism that provides dense supervision for intermediate reasoning steps (e.g., Decomposition, Sub-answer Correctness). Crucially, we identify that naive outcome-only RL is a deceptive trap for LLM alignment: it collapses reasoning integrity (e.g., 19.00% Hop acc) while superficially boosting final accuracy. Our process-aware framework sets a new SOTA of 95.48% on MQUAKE-CF-3k (+5.28%), demonstrating that for complex tasks, aligning the reasoning process is essential for building trustworthy LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.09539v3">TFRank: Think-Free Reasoning Enables Practical Pointwise LLM Ranking</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      Reasoning-intensive ranking models built on Large Language Models (LLMs) have made notable progress. However, existing approaches often rely on large-scale LLMs and explicit Chain-of-Thought (CoT) reasoning, resulting in high computational cost and latency that limit real-world use. To address this, we propose \textbf{TFRank}, an efficient pointwise reasoning ranker based on small-scale LLMs. To improve ranking performance, TFRank effectively integrates CoT data, fine-grained score supervision, and multi-task training. Furthermore, it achieves an efficient ``\textbf{T}hink-\textbf{F}ree" reasoning capability by employing a ``think-mode switch'' and pointwise format constraints. Specifically, this allows the model to leverage explicit reasoning during training while delivering precise relevance scores for complex queries at inference without generating any reasoning chains. Experiments show that TFRank achieves performance comparable to models with four times more parameters on the BRIGHT benchmark and demonstrates strong competitiveness on the BEIR benchmark. Further analysis shows that TFRank achieves an effective balance between performance and efficiency, providing a practical solution for integrating advanced reasoning into real-world systems. Our code and data are released in the repository: https://github.com/JOHNNY-fans/TFRank.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.03661v3">Automated Algorithmic Discovery for Scientific Computing through LLM-Guided Evolutionary Search: A Case Study in Gravitational-Wave Detection</a></div>
    <div class="paper-meta">
      📅 2025-11-16
      | 💬 76 pages (28 main), with 6+6 figures and 2 tables, substantially revised with improved structure and clarity
    </div>
    <details class="paper-abstract">
      Automated algorithm discovery in scientific computing faces fundamental challenges: vast design spaces with expensive evaluations, domain-specific physical constraints requiring expert knowledge, and the necessity for interpretable solutions that scientists can validate and understand. We present the Evo-MCTS (Evolutionary Monte Carlo Tree Search) framework, integrating large language models (LLMs) with tree-structured evolutionary search for interpretable algorithm discovery. Evo-MCTS combines reflective code synthesis leveraging LLM domain knowledge, multi-scale evolutionary operations on structured code representations, and interpretable algorithmic pathways emerging from tree-guided exploration. When applied to gravitational wave detection-a challenging domain with continuous parameter spaces and strict physical constraints-Evo-MCTS achieves 20.2% improvement over domain-specific methods and 59.1% over LLM-based optimization frameworks. This improvement arises from its ability to consistently converge toward interpretable algorithmic structures that integrate multiple functional components. Our domain-agnostic architecture establishes a generalizable methodology for automated algorithm discovery in scientific computing, where algorithmic transparency and physical validity are as essential as performance optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.09724v2">UDA: Unsupervised Debiasing Alignment for Pair-wise LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      Pairwise evaluation of Large Language Models (LLMs) is a common paradigm, but it is prone to preference bias, where judges systematically favor certain outputs, such as their own. This bias leads to inconsistent and skewed rankings across different judges. To address this, we first empirically demonstrate significant and heterogeneous biases in cross-model evaluations. We then propose UDA (Unsupervised Debiasing Alignment), a framework that reduces inter-judge disagreement by dynamically adjusting the Elo rating system. For each pairwise comparison, a compact neural network learns to adaptively set the K-factor and refine win probabilities. Crucially, UDA operates in a fully unsupervised manner, guided solely by the objective of minimizing the dispersion among the Elo trajectories of all judges. This forces an alignment towards a collective consensus, which serves as an unsupervised proxy for a more stable and reproducible evaluation. In addition, we provide theoretical motivation demonstrating how alignment towards a consensus can reduce aggregate system bias. Experiments show that UDA significantly reduces the inter-judge rating standard deviation by up to 63.4% and improves the average correlation with human judgments by 24.7%. Notably, UDA elevates the performance of poorly performing judges to achieve parity with high-quality ones, fostering a more robust and reliable evaluation ecosystem. Code and data are available at https://anonymous.4open.science/r/62AB93CD-23B4.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.01623v3">Fira: Can We Achieve Full-rank Training of LLMs Under Low-rank Constraint?</a></div>
    <div class="paper-meta">
      📅 2025-11-16
      | 💬 NeurIPS 2025, Project page: https://github.com/xichen-fy/Fira
    </div>
    <details class="paper-abstract">
      Low-rank training has emerged as a promising approach for reducing memory usage in training Large Language Models (LLMs). Previous methods either rely on decomposing weight matrices (e.g., LoRA), or seek to decompose gradient matrices (e.g., GaLore) to ensure reduced memory consumption. However, both of them constrain the training in a low-rank subspace, thus inevitably leading to sub-optimal performance. This raises a question: whether it is possible to consistently preserve the low-rank constraint for memory efficiency, while achieving full-rank training (i.e., training with full-rank gradients of full-rank weights) to avoid inferior outcomes? In this paper, we propose a new plug-and-play training framework for LLMs called Fira, as the first attempt to achieve this goal. First, we observe an interesting phenomenon during LLM training: the scaling impact of adaptive optimizers (e.g., Adam) on the gradient norm remains similar from low-rank to full-rank training. Based on this observation, we propose a norm-based scaling method, which utilizes the scaling impact of low-rank optimizers as substitutes for that of original full-rank optimizers to enable full-rank training. In this way, we can preserve the low-rank constraint in the optimizer while achieving full-rank training for better performance. Moreover, we find that there are sudden gradient rises during the optimization process, potentially causing loss spikes. To address this, we further put forward a norm-growth limiter to smooth the gradient via regulating the relative increase of gradient norms. Extensive experiments on the pre-training and fine-tuning of LLMs show that Fira outperforms both LoRA and GaLore, achieving performance that is comparable to or even better than full-rank training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.09443v2">LLMs Cannot Reliably Judge (Yet?): A Comprehensive Assessment on the Robustness of LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated exceptional capabilities across diverse tasks, driving the development and widespread adoption of LLM-as-a-Judge systems for automated evaluation, including red teaming and benchmarking. However, these systems are susceptible to adversarial attacks that can manipulate evaluation outcomes, raising critical concerns about their robustness and trustworthiness. Existing evaluation methods for LLM-based judges are often fragmented and lack a unified framework for comprehensive robustness assessment. Furthermore, the impact of prompt template design and model selection on judge robustness has rarely been explored, and their performance in real-world deployments remains largely unverified. To address these gaps, we introduce RobustJudge, a fully automated and scalable framework designed to systematically evaluate the robustness of LLM-as-a-Judge systems. Specifically, RobustJudge investigates the effectiveness of 15 attack methods and 7 defense strategies across 12 models (RQ1), examines the impact of prompt template design and model selection (RQ2), and evaluates the security of real-world deployments (RQ3). Our study yields three key findings: (1) LLM-as-a-Judge systems are highly vulnerable to attacks such as PAIR and combined attacks, while defense mechanisms such as re-tokenization and LLM-based detectors can provide enhanced protection; (2) robustness varies substantially across prompt templates (up to 40%); (3) deploying RobustJudge on Alibaba's PAI platform uncovers previously undiscovered vulnerabilities. These results offer practical insights for building trustworthy LLM-as-a-Judge systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.05925v2">Small Models, Big Support: A Local LLM Framework for Educator-Centric Content Creation and Assessment with RAG and CAG</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) are increasingly applied in student-facing educational tools, their potential to directly support educators through locally deployable and customizable solutions remains underexplored. Many existing approaches rely on proprietary, cloud-based systems that raise significant cost, privacy, and control concerns for educational institutions. To address these barriers, we introduce an end-to-end, open-source framework that empowers educators using small (3B-7B parameter), locally deployable LLMs. Our system is designed for comprehensive teacher support, including customized teaching material generation and AI-assisted assessment. The framework synergistically combines Retrieval-Augmented Generation (RAG) and Context-Augmented Generation (CAG) to produce factually accurate, pedagogically-styled content. A core feature is an interactive refinement loop, a teacher-in-the-loop mechanism that ensures educator agency and precise alignment of the final output. To enhance reliability and safety, an auxiliary verifier LLM inspects all generated content. We validate our framework through a rigorous evaluation of its content generation capabilities and report on a successful technical deployment in a college physics course, which confirms its feasibility on standard institutional hardware. Our findings demonstrate that carefully engineered, self-hosted systems built on small LLMs can provide robust, affordable, and private support for educators, achieving practical utility comparable to much larger models for targeted instructional tasks. This work presents a practical blueprint for the development of sovereign AI tools tailored to the real-world needs of educational institutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.03945v3">Function-based Labels for Complementary Recommendation: Definition, Annotation, and LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      Complementary recommendations enhance the user experience by suggesting items that are frequently purchased together while serving different functions from the query item. Inferring or evaluating whether two items have a complementary relationship requires complementary relationship labels; however, defining these labels is challenging because of the inherent ambiguity of such relationships. Complementary labels based on user historical behavior logs attempt to capture these relationships, but often produce inconsistent and unreliable results. Recent efforts have introduced large language models (LLMs) to infer these relationships. However, these approaches provide a binary classification without a nuanced understanding of complementary relationships. In this study, we address these challenges by introducing Function-Based Labels (FBLs), a novel definition of complementary relationships independent of user purchase logs and the opaque decision processes of LLMs. We constructed a human-annotated FBLs dataset comprising 2,759 item pairs and demonstrated that it covered possible item relationships and minimized ambiguity. We then evaluated whether some machine learning (ML) methods using annotated FBLs could accurately infer labels for unseen item pairs, and whether LLM-generated complementary labels align with human perception. Our results demonstrate that even with limited data, ML models, such as logistic regression and SVM achieve high macro-F1 scores (approximately 0.82). Furthermore, LLMs, such as gpt-4o-mini, demonstrated high consistency (0.989) and classification accuracy (0.849) under the detailed definition of FBLs, indicating their potential as effective annotators that mimic human judgment. Overall, our study presents FBLs as a clear definition of complementary relationships, enabling more accurate inferences and automated labeling of complementary recommendations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12485v1">ARCHE: A Novel Task to Evaluate LLMs on Latent Reasoning Chain Extraction</a></div>
    <div class="paper-meta">
      📅 2025-11-16
      | 💬 Accepted to AAAI 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in scientific domains. While they can produce reasoning-like content via methods such as chain-of-thought prompting, these outputs are typically unstructured and informal, obscuring whether models truly understand the fundamental reasoning paradigms that underpin scientific inference. To address this, we introduce a novel task named Latent Reasoning Chain Extraction (ARCHE), in which models must decompose complex reasoning arguments into combinations of standard reasoning paradigms in the form of a Reasoning Logic Tree (RLT). In RLT, all reasoning steps are explicitly categorized as one of three variants of Peirce's fundamental inference modes: deduction, induction, or abduction. To facilitate this task, we release ARCHE Bench, a new benchmark derived from 70 Nature Communications articles, including more than 1,900 references and 38,000 viewpoints. We propose two logic-aware evaluation metrics: Entity Coverage (EC) for content completeness and Reasoning Edge Accuracy (REA) for step-by-step logical validity. Evaluations on 10 leading LLMs on ARCHE Bench reveal that models exhibit a trade-off between REA and EC, and none are yet able to extract a complete and standard reasoning chain. These findings highlight a substantial gap between the abilities of current reasoning models and the rigor required for scientific argumentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12484v1">One Request, Multiple Experts: LLM Orchestrates Domain Specific Models via Adaptive Task Routing</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      With the integration of massive distributed energy resources and the widespread participation of novel market entities, the operation of active distribution networks (ADNs) is progressively evolving into a complex multi-scenario, multi-objective problem. Although expert engineers have developed numerous domain specific models (DSMs) to address distinct technical problems, mastering, integrating, and orchestrating these heterogeneous DSMs still entail considerable overhead for ADN operators. Therefore, an intelligent approach is urgently required to unify these DSMs and enable efficient coordination. To address this challenge, this paper proposes the ADN-Agent architecture, which leverages a general large language model (LLM) to coordinate multiple DSMs, enabling adaptive intent recognition, task decomposition, and DSM invocation. Within the ADN-Agent, we design a novel communication mechanism that provides a unified and flexible interface for diverse heterogeneous DSMs. Finally, for some language-intensive subtasks, we propose an automated training pipeline for fine-tuning small language models, thereby effectively enhancing the overall problem-solving capability of the system. Comprehensive comparisons and ablation experiments validate the efficacy of the proposed method and demonstrate that the ADN-Agent architecture outperforms existing LLM application paradigms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12474v1">Co-Layout: LLM-driven Co-optimization for Interior Layout</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      We present a novel framework for automated interior design that combines large language models (LLMs) with grid-based integer programming to jointly optimize room layout and furniture placement. Given a textual prompt, the LLM-driven agent workflow extracts structured design constraints related to room configurations and furniture arrangements. These constraints are encoded into a unified grid-based representation inspired by ``Modulor". Our formulation accounts for key design requirements, including corridor connectivity, room accessibility, spatial exclusivity, and user-specified preferences. To improve computational efficiency, we adopt a coarse-to-fine optimization strategy that begins with a low-resolution grid to solve a simplified problem and guides the solution at the full resolution. Experimental results across diverse scenarios demonstrate that our joint optimization approach significantly outperforms existing two-stage design pipelines in solution quality, and achieves notable computational efficiency through the coarse-to-fine strategy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12472v1">Assessing LLMs for Serendipity Discovery in Knowledge Graphs: A Case for Drug Repurposing</a></div>
    <div class="paper-meta">
      📅 2025-11-16
      | 💬 The 40th AAAI Conference on Artificial Intelligence (AAAI-26)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have greatly advanced knowledge graph question answering (KGQA), yet existing systems are typically optimized for returning highly relevant but predictable answers. A missing yet desired capacity is to exploit LLMs to suggest surprise and novel ("serendipitious") answers. In this paper, we formally define the serendipity-aware KGQA task and propose the SerenQA framework to evaluate LLMs' ability to uncover unexpected insights in scientific KGQA tasks. SerenQA includes a rigorous serendipity metric based on relevance, novelty, and surprise, along with an expert-annotated benchmark derived from the Clinical Knowledge Graph, focused on drug repurposing. Additionally, it features a structured evaluation pipeline encompassing three subtasks: knowledge retrieval, subgraph reasoning, and serendipity exploration. Our experiments reveal that while state-of-the-art LLMs perform well on retrieval, they still struggle to identify genuinely surprising and valuable discoveries, underscoring a significant room for future improvements. Our curated resources and extended version are released at: https://cwru-db-group.github.io/serenQA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12448v1">SeedAIchemy: LLM-Driven Seed Corpus Generation for Fuzzing</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      We introduce SeedAIchemy, an automated LLM-driven corpus generation tool that makes it easier for developers to implement fuzzing effectively. SeedAIchemy consists of five modules which implement different approaches at collecting publicly available files from the internet. Four of the five modules use large language model (LLM) workflows to construct search terms designed to maximize corpus quality. Corpora generated by SeedAIchemy perform significantly better than a naive corpus and similarly to a manually-curated corpus on a diverse range of target programs and libraries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.24021v2">SelecTKD: Selective Token-Weighted Knowledge Distillation for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      Knowledge distillation (KD) is a standard route to compress Large Language Models (LLMs) into compact students, yet most pipelines uniformly apply token-wise loss regardless of teacher confidence. This indiscriminate supervision amplifies noisy, high-entropy signals and is especially harmful under large teacher-student capacity gaps. We introduce SelecTKD, a plug-and-play Selective Token-Weighted distillation framework that shifts the focus from "how to measure divergence" to "where to apply learning". At each step, the student proposes tokens that are verified by the teacher through a robust propose-and-verify procedure with two variants: greedy Top-k and non-greedy Spec-k. Accepted tokens receive full loss, while rejected tokens are masked or down-weighted. This objective-agnostic design works with on- and off-policy data, induces an implicit curriculum quantified by Token Acceptance Rate (TAR), and stabilizes optimization. Across instruction following, mathematical reasoning, code generation, and a VLM setting, SelecTKD consistently improves strong baselines and achieves state-of-the-art results for small models without architectural changes or extra reference models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12423v1">GRAPHTEXTACK: A Realistic Black-Box Node Injection Attack on LLM-Enhanced GNNs</a></div>
    <div class="paper-meta">
      📅 2025-11-16
      | 💬 AAAI 2026
    </div>
    <details class="paper-abstract">
      Text-attributed graphs (TAGs), which combine structural and textual node information, are ubiquitous across many domains. Recent work integrates Large Language Models (LLMs) with Graph Neural Networks (GNNs) to jointly model semantics and structure, resulting in more general and expressive models that achieve state-of-the-art performance on TAG benchmarks. However, this integration introduces dual vulnerabilities: GNNs are sensitive to structural perturbations, while LLM-derived features are vulnerable to prompt injection and adversarial phrasing. While existing adversarial attacks largely perturb structure or text independently, we find that uni-modal attacks cause only modest degradation in LLM-enhanced GNNs. Moreover, many existing attacks assume unrealistic capabilities, such as white-box access or direct modification of graph data. To address these gaps, we propose GRAPHTEXTACK, the first black-box, multi-modal{, poisoning} node injection attack for LLM-enhanced GNNs. GRAPHTEXTACK injects nodes with carefully crafted structure and semantics to degrade model performance, operating under a realistic threat model without relying on model internals or surrogate models. To navigate the combinatorial, non-differentiable search space of connectivity and feature assignments, GRAPHTEXTACK introduces a novel evolutionary optimization framework with a multi-objective fitness function that balances local prediction disruption and global graph influence. Extensive experiments on five datasets and two state-of-the-art LLM-enhanced GNN models show that GRAPHTEXTACK significantly outperforms 12 strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.19850v2">FALCONEye: Finding Answers and Localizing Content in ONE-hour-long videos with multi-modal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      Finding information in hour-long videos is a challenging task even for top-performing Vision Language Models (VLMs), as encoding visual content quickly exceeds available context windows. To tackle this challenge, we present FALCONEye, a novel video agent based on a training-free, model-agnostic meta-architecture composed of a VLM and a Large Language Model (LLM). FALCONEye answers open-ended questions using an exploration-based search algorithm guided by calibrated confidence from the VLM's answers. We also introduce the FALCON-Bench benchmark, extending Question Answering problem to Video Answer Search-requiring models to return both the answer and its supporting temporal window for open-ended questions in hour-long videos. With just a 7B VLM and a lightweight LLM, FALCONEye outscores all open-source 7B VLMs and comparable agents in FALCON-Bench. It further demonstrates its generalization capability in MLVU benchmark with shorter videos and different tasks, surpassing GPT-4o on single-detail tasks while slashing inference cost by roughly an order of magnitude.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.04245v3">Contextual Integrity in LLMs via Reasoning and Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-16
      | 💬 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
    </div>
    <details class="paper-abstract">
      As the era of autonomous agents making decisions on behalf of users unfolds, ensuring contextual integrity (CI) -- what is the appropriate information to share while carrying out a certain task -- becomes a central question to the field. We posit that CI demands a form of reasoning where the agent needs to reason about the context in which it is operating. To test this, we first prompt LLMs to reason explicitly about CI when deciding what information to disclose. We then extend this approach by developing a reinforcement learning (RL) framework that further instills in models the reasoning necessary to achieve CI. Using a synthetic, automatically created, dataset of only $\sim700$ examples but with diverse contexts and information disclosure norms, we show that our method substantially reduces inappropriate information disclosure while maintaining task performance across multiple model sizes and families. Importantly, improvements transfer from this synthetic dataset to established CI benchmarks such as PrivacyLens that has human annotations and evaluates privacy leakage of AI assistants in actions and tool calls.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.12982v2">Accommodate Knowledge Conflicts in Retrieval-augmented LLMs: Towards Robust Response Generation in the Wild</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      The proliferation of large language models (LLMs) has significantly advanced intelligent systems. Unfortunately, LLMs often face knowledge conflicts between internal memory and retrieved external information, arising from misinformation, biases, or outdated knowledge. These conflicts undermine response reliability and introduce uncertainty in decision-making. In this work, we analyze how LLMs navigate knowledge conflicts from an information-theoretic perspective and reveal that when conflicting and supplementary information exhibit significant differences, LLMs confidently resolve their preferences and alleviate the uncertainty during their response generation. When this difference is ambiguous, LLMs experience considerable uncertainty about their generation. Based on this insight, we propose Swin-VIB, a novel framework that integrates a pipeline of variational information bottleneck models to adapt the retrieved information difference, facilitating robust response generation of LLMs even in conflicting contexts. Extensive experiments confirm our theoretical analysis and demonstrate the performance of Swin-VIB. Notably, Swin-VIB outperforms all competitive baselines in terms of the accuracy of the multiple-choice task, while improving the EM values in the open-ended QA task by at least 11.14%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13788v1">Scaling Patterns in Adversarial Alignment: Evidence from Multi-LLM Jailbreak Experiments</a></div>
    <div class="paper-meta">
      📅 2025-11-16
      | 💬 19 pages, 6 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly operate in multi-agent and safety-critical settings, raising open questions about how their vulnerabilities scale when models interact adversarially. This study examines whether larger models can systematically jailbreak smaller ones - eliciting harmful or restricted behavior despite alignment safeguards. Using standardized adversarial tasks from JailbreakBench, we simulate over 6,000 multi-turn attacker-target exchanges across major LLM families and scales (0.6B-120B parameters), measuring both harm score and refusal behavior as indicators of adversarial potency and alignment integrity. Each interaction is evaluated through aggregated harm and refusal scores assigned by three independent LLM judges, providing a consistent, model-based measure of adversarial outcomes. Aggregating results across prompts, we find a strong and statistically significant correlation between mean harm and the logarithm of the attacker-to-target size ratio (Pearson r = 0.51, p < 0.001; Spearman rho = 0.52, p < 0.001), indicating that relative model size correlates with the likelihood and severity of harmful completions. Mean harm score variance is higher across attackers (0.18) than across targets (0.10), suggesting that attacker-side behavioral diversity contributes more to adversarial outcomes than target susceptibility. Attacker refusal frequency is strongly and negatively correlated with harm (rho = -0.93, p < 0.001), showing that attacker-side alignment mitigates harmful responses. These findings reveal that size asymmetry influences robustness and provide exploratory evidence for adversarial scaling patterns, motivating more controlled investigations into inter-model alignment and safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11967v1">Bootstrapped LLM Semantics for Context-Aware Path Planning</a></div>
    <div class="paper-meta">
      📅 2025-11-15
    </div>
    <details class="paper-abstract">
      Prompting robots with natural language (NL) has largely been studied as what task to execute (goal selection, skill sequencing) rather than how to execute that task safely and efficiently in semantically rich, human-centric spaces. We address this gap with a framework that turns a large language model (LLM) into a stochastic semantic sensor whose outputs modulate a classical planner. Given a prompt and a semantic map, we draw multiple LLM "danger" judgments and apply a Bayesian bootstrap to approximate a posterior over per-class risk. Using statistics from the posterior, we create a potential cost to formulate a path planning problem. Across simulated environments and a BIM-backed digital twin, our method adapts how the robot moves in response to explicit prompts and implicit contextual information. We present qualitative and quantitative results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.22162v3">Surface Reading LLMs: Synthetic Text and its Styles</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 12 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Despite a potential plateau in ML advancement, the societal impact of large language models lies not in approaching superintelligence but in generating text surfaces indistinguishable from human writing. While Critical AI Studies provides essential material and socio-technical critique, it risks overlooking how LLMs phenomenologically reshape meaning-making. This paper proposes a semiotics of "surface integrity" as attending to the immediate plane where LLMs inscribe themselves into human communication. I distinguish three knowledge interests in ML research (epistemology, epistēmē, and epistemics) and argue for integrating surface-level stylistic analysis alongside depth-oriented critique. Through two case studies examining stylistic markers of synthetic text, I argue how attending to style as a semiotic phenomenon reveals LLMs as cultural machines that transform the conditions of meaning emergence and circulation in contemporary discourse, independent of questions about machine consciousness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11961v1">"Power of Words": Stealthy and Adaptive Private Information Elicitation via LLM Communication Strategies</a></div>
    <div class="paper-meta">
      📅 2025-11-15
    </div>
    <details class="paper-abstract">
      While communication strategies of Large Language Models (LLMs) are crucial for human-LLM interactions, they can also be weaponized to elicit private information, yet such stealthy attacks remain under-explored. This paper introduces the first adaptive attack framework for stealthy and targeted private information elicitation via communication strategies. Our framework operates in a dynamic closed-loop: it first performs real-time psychological profiling of the users' state, then adaptively selects an optimized communication strategy, and finally maintains stealthiness through prompt-based rewriting. We validated this framework through a user study (N=84), demonstrating its generalizability across 3 distinct LLMs and 3 scenarios. The targeted attacks achieved a 205.4% increase in eliciting specific targeted information compared to stealthy interactions without strategies. Even stealthy interactions without specific strategies successfully elicited private information in 54.8% cases. Notably, users not only failed to detect the manipulation but paradoxically rated the attacking chatbot as more empathetic and trustworthy. Finally, we advocate for mitigations, encouraging developers to integrate adaptive, just-in-time alerts, users to build literacy against specific manipulative tactics, and regulators to define clear ethical boundaries distinguishing benign persuasion from coercion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11954v1">LLM-Assisted Formalization Enables Deterministic Detection of Statutory Inconsistency in the Internal Revenue Code</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 29 pages, 3 appendices with Prolog code and full codebase available at: https://github.com/borchuluun/section121-inconsistency-detection
    </div>
    <details class="paper-abstract">
      This study introduces a hybrid neuro-symbolic framework that achieves deterministic detection of statutory inconsistency in complex law. We use the U.S. Internal Revenue Code (IRC) as a case study because its complexity makes it a fertile domain for identifying conflicts. Our research offers a solution for detecting inconsistent provisions by combining Large Language Models (LLMs) with symbolic logic. LLM-based methods can support compliance, fairness, and statutory drafting, yet tax-specific applications remain sparse. A key challenge is that such models struggle with hierarchical processing and deep structured reasoning, especially over long text. This research addresses these gaps through experiments using GPT-4o, GPT-5, and Prolog. GPT-4o was first used to translate Section 121 into Prolog rules and refine them in SWISH. These rules were then incorporated into prompts to test whether Prolog-augmented prompting improved GPT-4o's inconsistency detection. GPT-4o, whether prompted with natural language alone or with Prolog augmentation, detected the inconsistency in only one of three strategies (33 percent accuracy), but its reasoning quality differed: natural-language prompting achieved 100 percent rule coverage, while Prolog-augmented prompting achieved 66 percent, indicating more incomplete statutory analysis. In contrast to probabilistic prompting, the hybrid Prolog model produced deterministic and reproducible results. Guided by GPT-5 for refinement, the model formalized the IRC section's competing interpretations and successfully detected an inconsistency zone. Validation tests confirm that the Prolog implementation is accurate, internally consistent, deterministic, and capable of autonomously identifying inconsistencies. These findings show that LLM-assisted formalization, anchored in symbolic logic, enables transparent and reliable statutory inconsistency detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12376v1">BitSnap: Checkpoint Sparsification and Quantization in LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 12 pages, numerous figures
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to grow in size and complexity, efficient checkpoint saving\&loading has become crucial for managing storage, memory usage, and fault tolerance in LLM training. The current works do not comprehensively take into account the optimization of these several aspects. This paper proposes a novel checkpoint sparsification and quantization method that adapts dynamically to different training stages and model architectures. We present a comprehensive analysis of existing lossy and lossless compression techniques, identify current limitations, and introduce our adaptive approach that balances compression ratio, speed, and precision impact throughout the training process. Experiments on different sizes of LLMs demonstrate that our bitmask-based sparsification method achieves 16x compression ratio without compromising model accuracy. Additionally, the cluster-based quantization method achieves 2x compression ratio with little precision loss.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.00730v4">Teaching LLMs to See and Guide: Context-Aware Real-Time Assistance in Augmented Reality</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 This work has been submitted to the IEEE Transactions on Systems, Man, and Cybernetics: Systems for possible publication
    </div>
    <details class="paper-abstract">
      The growing adoption of augmented and virtual reality (AR and VR) technologies in industrial training and on-the-job assistance has created new opportunities for intelligent, context-aware support systems. As workers perform complex tasks guided by AR and VR, these devices capture rich streams of multimodal data, including gaze, hand actions, and task progression, that can reveal user intent and task state in real time. Leveraging this information effectively remains a major challenge. In this work, we present a context-aware large language model (LLM) assistant that integrates diverse data modalities, such as hand actions, task steps, and dialogue history, into a unified framework for real-time question answering. To systematically study how context influences performance, we introduce an incremental prompting framework, where each model version receives progressively richer contextual inputs. Using the HoloAssist dataset, which records AR-guided task executions, we evaluate how each modality contributes to the assistant's effectiveness. Our experiments show that incorporating multimodal context significantly improves the accuracy and relevance of responses. These findings highlight the potential of LLM-driven multimodal integration to enable adaptive, intuitive assistance for AR and VR-based industrial training and assistance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.04688v2">Evaluating LLMs' Reasoning Over Ordered Procedural Steps</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 Accepted to IJCNLP-AACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Reasoning over procedural sequences, where the order of steps directly impacts outcomes, is a critical capability for large language models (LLMs). In this work, we study the task of reconstructing globally ordered sequences from shuffled procedural steps, using a curated dataset of food recipes, a domain where correct sequencing is essential for task success. We evaluate several LLMs under zero-shot and few-shot settings and present a comprehensive evaluation framework that adapts established metrics from ranking and sequence alignment. These include Kendall's Tau, Normalized Longest Common Subsequence (NLCS), and Normalized Edit Distance (NED), which capture complementary aspects of ordering quality. Our analysis shows that model performance declines with increasing sequence length, reflecting the added complexity of longer procedures. We also find that greater step displacement in the input, corresponding to more severe shuffling, leads to further degradation. These findings highlight the limitations of current LLMs in procedural reasoning, especially with longer and more disordered inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.19172v2">When Facts Change: Probing LLMs on Evolving Knowledge with evolveQA</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 Under submission
    </div>
    <details class="paper-abstract">
      LLMs often fail to handle temporal knowledge conflicts--contradictions arising when facts evolve over time within their training data. Existing studies evaluate this phenomenon through benchmarks built on structured knowledge bases like Wikidata, but they focus on widely-covered, easily-memorized popular entities and lack the dynamic structure needed to fairly evaluate LLMs with different knowledge cut-off dates. We introduce evolveQA, a benchmark specifically designed to evaluate LLMs on temporally evolving knowledge, constructed from 3 real-world, time-stamped corpora: AWS updates, Azure changes, and WHO disease outbreak reports. Our framework identifies naturally occurring knowledge evolution and generates questions with gold answers tailored to different LLM knowledge cut-off dates. Through extensive evaluation of 12 open and closed-source LLMs across 3 knowledge probing formats, we demonstrate significant performance drops of up to 31% on evolveQA compared to static knowledge questions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12300v1">Do LLMs and Humans Find the Same Questions Difficult? A Case Study on Japanese Quiz Answering</a></div>
    <div class="paper-meta">
      📅 2025-11-15
    </div>
    <details class="paper-abstract">
      LLMs have achieved performance that surpasses humans in many NLP tasks. However, it remains unclear whether problems that are difficult for humans are also difficult for LLMs. This study investigates how the difficulty of quizzes in a buzzer setting differs between LLMs and humans. Specifically, we first collect Japanese quiz data including questions, answers, and correct response rate of humans, then prompted LLMs to answer the quizzes under several settings, and compare their correct answer rate to that of humans from two analytical perspectives. The experimental results showed that, compared to humans, LLMs struggle more with quizzes whose correct answers are not covered by Wikipedia entries, and also have difficulty with questions that require numerical answers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12295v1">Privacy-Preserving Prompt Injection Detection for LLMs Using Federated Learning and Embedding-Based NLP Classification</a></div>
    <div class="paper-meta">
      📅 2025-11-15
    </div>
    <details class="paper-abstract">
      Prompt injection attacks are an emerging threat to large language models (LLMs), enabling malicious users to manipulate outputs through carefully designed inputs. Existing detection approaches often require centralizing prompt data, creating significant privacy risks. This paper proposes a privacy-preserving prompt injection detection framework based on federated learning and embedding-based classification. A curated dataset of benign and adversarial prompts was encoded with sentence embedding and used to train both centralized and federated logistic regression models. The federated approach preserved privacy by sharing only model parameters across clients, while achieving detection performance comparable to centralized training. Results demonstrate that effective prompt injection detection is feasible without exposing raw data, making this one of the first explorations of federated security for LLMs. Although the dataset is limited in scale, the findings establish a strong proof-of-concept and highlight new directions for building secure and privacy-aware LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12288v1">Reducing Hallucinations in LLM-Generated Code via Semantic Triangulation</a></div>
    <div class="paper-meta">
      📅 2025-11-15
    </div>
    <details class="paper-abstract">
      When generating code from natural language prompts, an LLM samples programs from a probability distribution, many of which might be incorrect. Sample consensus techniques - such as majority voting or validation against generated tests or specifications - aim to identify a correct program in the sample or abstain if none is valid. However, existing methods often fail to select a correct solution when its sampling probability is low, or when the problem permits multiple valid but non-equivalent solutions. Additionally, they often fail to abstain when no correct solution is present in the sample. To overcome these limitations, we introduce semantic triangulation, which transforms a programming problem in a way that non-trivially alters its semantics while preserving an exact, verifiable mapping between solutions before and after transformation. We theoretically establish that verifying consistency across such problem transformations increases confidence that generated programs reflect accurate generalization rather than spurious statistical correlations, enabling more reliable sample consensus and abstention. On the LiveCodeBench and CodeElo benchmarks, using GPT-4o and DeepSeek-V3 models, semantic triangulation increases reliability of generated code by 21% compared to the method that selects only high-confidence solutions with the probability threshold 0.5, while being able to pinpoint correct solutions at sampling probabilities as low as 0.14. Apart from that, it is also the only approach to consistently form true consensus on tasks with multiple valid but non-equivalent solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12286v1">Sangam: Chiplet-Based DRAM-PIM Accelerator with CXL Integration for LLM Inferencing</a></div>
    <div class="paper-meta">
      📅 2025-11-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are becoming increasingly data-intensive due to growing model sizes, and they are becoming memory-bound as the context length and, consequently, the key-value (KV) cache size increase. Inference, particularly the decoding phase, is dominated by memory-bound GEMV or flat GEMM operations with low operational intensity (OI), making it well-suited for processing-in-memory (PIM) approaches. However, existing in/near-memory solutions face critical limitations such as reduced memory capacity due to the high area cost of integrating processing elements (PEs) within DRAM chips, and limited PE capability due to the constraints of DRAM fabrication technology. This work presents a chiplet-based memory module that addresses these limitations by decoupling logic and memory into chiplets fabricated in heterogeneous technology nodes and connected via an interposer. The logic chiplets sustain high bandwidth access to the DRAM chiplets, which house the memory banks, and enable the integration of advanced processing components such as systolic arrays and SRAM-based buffers to accelerate memory-bound GEMM kernels, capabilities that were not feasible in prior PIM architectures. We propose Sangam, a CXL-attached PIM-chiplet based memory module that can either act as a drop-in replacement for GPUs or co-executes along side the GPUs. Sangam achieves speedup of 3.93, 4.22, 2.82x speedup in end-to-end query latency, 10.3, 9.5, 6.36x greater decoding throughput, and order of magnitude energy savings compared to an H100 GPU for varying input size, output length, and batch size on LLaMA 2-7B, Mistral-7B, and LLaMA 3-70B, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12271v1">MoralReason: Generalizable Moral Decision Alignment For LLM Agents Using Reasoning-Level Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 Accepted for AAAI 2026
    </div>
    <details class="paper-abstract">
      Large language models are increasingly influencing human moral decisions, yet current approaches focus primarily on evaluating rather than actively steering their moral decisions. We formulate this as an out-of-distribution moral alignment problem, where LLM agents must learn to apply consistent moral reasoning frameworks to scenarios beyond their training distribution. We introduce Moral-Reason-QA, a novel dataset extending 680 human-annotated, high-ambiguity moral scenarios with framework-specific reasoning traces across utilitarian, deontological, and virtue ethics, enabling systematic evaluation of moral generalization in realistic decision contexts. Our learning approach employs Group Relative Policy Optimization with composite rewards that simultaneously optimize decision alignment and framework-specific reasoning processes to facilitate learning of the underlying moral frameworks. Experimental results demonstrate successful generalization to unseen moral scenarios, with softmax-normalized alignment scores improving by +0.757 for utilitarian and +0.450 for deontological frameworks when tested on out-of-distribution evaluation sets. The experiments also reveal training challenges and promising directions that inform future research. These findings establish that LLM agents can be systematically trained to internalize and apply specific moral frameworks to novel situations, providing a critical foundation for AI safety as language models become more integrated into human decision-making processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06194v2">NURBGen: High-Fidelity Text-to-CAD Generation through LLM-Driven NURBS Modeling</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 Accepted in AAAI 2026
    </div>
    <details class="paper-abstract">
      Generating editable 3D CAD models from natural language remains challenging, as existing text-to-CAD systems either produce meshes or rely on scarce design-history data. We present NURBGen, the first framework to generate high-fidelity 3D CAD models directly from text using Non-Uniform Rational B-Splines (NURBS). To achieve this, we fine-tune a large language model (LLM) to translate free-form texts into JSON representations containing NURBS surface parameters (\textit{i.e}, control points, knot vectors, degrees, and rational weights) which can be directly converted into BRep format using Python. We further propose a hybrid representation that combines untrimmed NURBS with analytic primitives to handle trimmed surfaces and degenerate regions more robustly, while reducing token complexity. Additionally, we introduce partABC, a curated subset of the ABC dataset consisting of individual CAD components, annotated with detailed captions using an automated annotation pipeline. NURBGen demonstrates strong performance on diverse prompts, surpassing prior methods in geometric fidelity and dimensional accuracy, as confirmed by expert evaluations. Code and dataset will be released publicly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.14995v2">LLM-Enhanced Multi-Agent Reinforcement Learning with Expert Workflow for Real-Time P2P Energy Trading</a></div>
    <div class="paper-meta">
      📅 2025-11-15
    </div>
    <details class="paper-abstract">
      Real-time peer-to-peer (P2P) electricity markets dynamically adapt to fluctuations in renewable energy and variations in demand, maximizing economic benefits through instantaneous price responses while enhancing grid flexibility. However, scaling expert guidance for massive personalized prosumers poses critical challenges, including diverse decision-making demands and lack of customized modeling frameworks. This paper proposed an integrated large language model-multi-agent reinforcement learning (LLM-MARL) framework for real-time P2P energy trading to address challenges such as the limited technical capability of prosumers, the lack of expert experience, and security issues of distribution networks. LLMs are introduced as experts to generate personalized strategy, guiding MARL under the centralized training with decentralized execution (CTDE) paradigm through imitation learning. A differential attention-based critic network is designed to enhance convergence performance. Experimental results demonstrate that LLM generated strategies effectively substitute human experts. The proposed multi-agent imitation learning algorithms achieve significantly lower economic costs and voltage violation rates on test sets compared to baselines algorithms, while maintaining robust stability. This work provides an effective solution for real-time P2P electricity market decision-making by bridging expert knowledge with agent learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12236v1">Consistency Is the Key: Detecting Hallucinations in LLM Generated Text By Checking Inconsistencies About Key Facts</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 To appear at International Joint Conference on Natural Language Processing & Asia-Pacific Chapter of the Association for Computational Linguistics (IJCNLP-AACL), 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), despite their remarkable text generation capabilities, often hallucinate and generate text that is factually incorrect and not grounded in real-world knowledge. This poses serious risks in domains like healthcare, finance, and customer support. A typical way to use LLMs is via the APIs provided by LLM vendors where there is no access to model weights or options to fine-tune the model. Existing methods to detect hallucinations in such settings where the model access is restricted or constrained by resources typically require making multiple LLM API calls, increasing latency and API cost. We introduce CONFACTCHECK, an efficient hallucination detection approach that does not leverage any external knowledge base and works on the simple intuition that responses to factual probes within the generated text should be consistent within a single LLM and across different LLMs. Rigorous empirical evaluation on multiple datasets that cover both the generation of factual texts and the open generation shows that CONFACTCHECK can detect hallucinated facts efficiently using fewer resources and achieves higher accuracy scores compared to existing baselines that operate under similar conditions. Our code is available here.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12224v1">RulePilot: An LLM-Powered Agent for Security Rule Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 This paper has been accepted for publication at ICSE 2026
    </div>
    <details class="paper-abstract">
      The real-time demand for system security leads to the detection rules becoming an integral part of the intrusion detection life-cycle. Rule-based detection often identifies malicious logs based on the predefined grammar logic, requiring experts with deep domain knowledge for rule generation. Therefore, automation of rule generation can result in significant time savings and ease the burden of rule-related tasks on security engineers. In this paper, we propose RulePilot, which mimics human expertise via LLM-based agent for addressing rule-related challenges like rule creation or conversion. Using RulePilot, the security analysts do not need to write down the rules following the grammar, instead, they can just provide the annotations such as the natural-language-based descriptions of a rule, our RulePilot can automatically generate the detection rules without more intervention. RulePilot is equipped with the intermediate representation (IR), which abstracts the complexity of config rules into structured, standardized formats, allowing LLMs to focus on generation rules in a more manageable and consistent way. We present a comprehensive evaluation of RulePilot in terms of textual similarity and execution success abilities, showcasing RulePilot can generate high-fidelity rules, outperforming the baseline models by up to 107.4% in textual similarity to ground truths and achieving better detection accuracy in real-world execution tests. We perform a case study from our industry collaborators in Singapore, showcasing that RulePilot significantly help junior analysts/general users in the rule creation process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12217v1">AlignTree: Efficient Defense Against LLM Jailbreak Attacks</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 Accepted as an Oral Presentation at the 40th AAAI Conference on Artificial Intelligence (AAAI-26), January 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are vulnerable to adversarial attacks that bypass safety guidelines and generate harmful content. Mitigating these vulnerabilities requires defense mechanisms that are both robust and computationally efficient. However, existing approaches either incur high computational costs or rely on lightweight defenses that can be easily circumvented, rendering them impractical for real-world LLM-based systems. In this work, we introduce the AlignTree defense, which enhances model alignment while maintaining minimal computational overhead. AlignTree monitors LLM activations during generation and detects misaligned behavior using an efficient random forest classifier. This classifier operates on two signals: (i) the refusal direction -- a linear representation that activates on misaligned prompts, and (ii) an SVM-based signal that captures non-linear features associated with harmful content. Unlike previous methods, AlignTree does not require additional prompts or auxiliary guard models. Through extensive experiments, we demonstrate the efficiency and robustness of AlignTree across multiple LLMs and benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.12396v3">LLM-CoT Enhanced Graph Neural Recommendation with Harmonized Group Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-15
    </div>
    <details class="paper-abstract">
      Graph neural networks (GNNs) have advanced recommender systems by modeling interaction relationships. However, existing graph-based recommenders rely on sparse ID features and do not fully exploit textual information, resulting in low information density within representations. Furthermore, graph contrastive learning faces challenges. Random negative sampling can introduce false negative samples, while fixed temperature coefficients cannot adapt to the heterogeneity of different nodes. In addition, current efforts to enhance recommendations with large language models (LLMs) have not fully utilized their Chain-of-Thought (CoT) reasoning capabilities to guide representation learning. To address these limitations, we introduces LGHRec (LLM-CoT Enhanced Graph Neural Recommendation with Harmonized Group Policy Optimization). This framework leverages the CoT reasoning ability of LLMs to generate semantic IDs, enriching reasoning processes and improving information density and semantic quality of representations. Moreover, we design a reinforcement learning algorithm, Harmonized Group Policy Optimization (HGPO), to optimize negative sampling strategies and temperature coefficients in contrastive learning. This approach enhances long-tail recommendation performance and ensures optimization consistency across different groups. Experimental results on three datasets demonstrate that LGHRec improves representation quality through semantic IDs generated by LLM's CoT reasoning and effectively boosts contrastive learning with HGPO. Our method outperforms several baseline models. The code is available at: https://anonymous.4open.science/r/LLM-Rec.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2406.08824v2">LLM-Driven Robots Risk Enacting Discrimination, Violence, and Unlawful Actions</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 Published in International Journal of Social Robotics (2025). 49 pages (65 with references and appendix), 27 Figures, 8 Tables. Andrew Hundt and Rumaisa Azeem are equal contribution co-first authors. The positions of the two co-first authors were swapped from arxiv version 1 with the written consent of all four authors. The Version of Record is available via DOI: 10.1007/s12369-025-01301-x
    </div>
    <details class="paper-abstract">
      Members of the Human-Robot Interaction (HRI) and Machine Learning (ML) communities have proposed Large Language Models (LLMs) as a promising resource for robotics tasks such as natural language interaction, household and workplace tasks, approximating 'common sense reasoning', and modeling humans. However, recent research has raised concerns about the potential for LLMs to produce discriminatory outcomes and unsafe behaviors in real-world robot experiments and applications. To assess whether such concerns are well placed in the context of HRI, we evaluate several highly-rated LLMs on discrimination and safety criteria. Our evaluation reveals that LLMs are currently unsafe for people across a diverse range of protected identity characteristics, including, but not limited to, race, gender, disability status, nationality, religion, and their intersections. Concretely, we show that LLMs produce directly discriminatory outcomes- e.g., 'gypsy' and 'mute' people are labeled untrustworthy, but not 'european' or 'able-bodied' people. We find various such examples of direct discrimination on HRI tasks such as facial expression, proxemics, security, rescue, and task assignment. Furthermore, we test models in settings with unconstrained natural language (open vocabulary) inputs, and find they fail to act safely, generating responses that accept dangerous, violent, or unlawful instructions-such as incident-causing misstatements, taking people's mobility aids, and sexual predation. Our results underscore the urgent need for systematic, routine, and comprehensive risk assessments and assurances to improve outcomes and ensure LLMs only operate on robots when it is safe, effective, and just to do so. We provide code to reproduce our experiments at https://github.com/rumaisa-azeem/llm-robots-discrimination-safety .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12033v1">EARL: Entropy-Aware RL Alignment of LLMs for Reliable RTL Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-15
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have demonstrated significant potential in hardware design automation, particularly in using natural language to synthesize Register-Transfer Level (RTL) code. Despite this progress, a gap remains between model capability and the demands of real-world RTL design, including syntax errors, functional hallucinations, and weak alignment to designer intent. Reinforcement Learning with Verifiable Rewards (RLVR) offers a promising approach to bridge this gap, as hardware provides executable and formally checkable signals that can be used to further align model outputs with design intent. However, in long, structured RTL code sequences, not all tokens contribute equally to functional correctness, and naïvely spreading gradients across all tokens dilutes learning signals. A key insight from our entropy analysis in RTL generation is that only a small fraction of tokens (e.g., always, if, assign, posedge) exhibit high uncertainty and largely influence control flow and module structure. To address these challenges, we present EARL, an Entropy-Aware Reinforcement Learning framework for Verilog generation. EARL performs policy optimization using verifiable reward signals and introduces entropy-guided selective updates that gate policy gradients to high-entropy tokens. This approach preserves training stability and concentrates gradient updates on functionally important regions of code. Our experiments on VerilogEval and RTLLM show that EARL improves functional pass rates over prior LLM baselines by up to 14.7%, while reducing unnecessary updates and improving training stability. These results indicate that focusing RL on critical, high-uncertainty tokens enables more reliable and targeted policy improvement for structured RTL code generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12031v1">Striking the Right Balance between Compute and Copy: Improving LLM Inferencing Under Speculative Decoding</a></div>
    <div class="paper-meta">
      📅 2025-11-15
    </div>
    <details class="paper-abstract">
      With the skyrocketing costs of GPUs and their virtual instances in the cloud, there is a significant desire to use CPUs for large language model (LLM) inference. KV cache update, often implemented as allocation, copying, and in-place strided update for each generated token, incurs significant overhead. As the sequence length increases, the allocation and copy overheads dominate the performance. Alternate approaches may allocate large KV tensors upfront to enable in-place updates, but these matrices (with zero-padded rows) cause redundant computations. In this work, we propose a new KV cache allocation mechanism called Balancing Memory and Compute (BMC). BMC allocates, once every r iterations, KV tensors with r redundant rows, allowing in-place update without copy overhead for those iterations, but at the expense of a small amount of redundant computation. Second, we make an interesting observation that the extra rows allocated in the KV tensors and the resulting redundant computation can be repurposed for Speculative Decoding (SD) that improves token generation efficiency. Last, BMC represents a spectrum of design points with different values of r. To identify the best-performing design point(s), we derive a simple analytical model for BMC. The proposed BMC method achieves an average throughput acceleration of up to 3.2x over baseline HuggingFace (without SD). Importantly when we apply BMC with SD, it results in an additional speedup of up to 1.39x, over and above the speedup offered by SD. Further, BMC achieves a throughput acceleration of up to 1.36x and 2.29x over state-of-the-art inference servers vLLM and DeepSpeed, respectively. Although the BMC technique is evaluated extensively across different classes of CPUs (desktop and server class), we also evaluate the scheme with GPUs and demonstrate that it works well for GPUs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12014v1">CURE: Cultural Understanding and Reasoning Evaluation - A Framework for "Thick" Culture Alignment Evaluation in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 7 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in culturally diverse environments, yet existing evaluations of cultural competence remain limited. Existing methods focus on de-contextualized correctness or forced-choice judgments, overlooking the need for cultural understanding and reasoning required for appropriate responses. To address this gap, we introduce a set of benchmarks that, instead of directly probing abstract norms or isolated statements, present models with realistic situational contexts that require culturally grounded reasoning. In addition to the standard Exact Match metric, we introduce four complementary metrics (Coverage, Specificity, Connotation, and Coherence) to capture different dimensions of model's response quality. Empirical analysis across frontier models reveals that thin evaluation systematically overestimates cultural competence and produces unstable assessments with high variance. In contrast, thick evaluation exposes differences in reasoning depth, reduces variance, and provides more stable, interpretable signals of cultural understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.24189v2">PET: Preference Evolution Tracking with LLM-Generated Explainable Distribution</a></div>
    <div class="paper-meta">
      📅 2025-11-15
    </div>
    <details class="paper-abstract">
      Understanding how user preference evolves over time is a fundamental challenge central to modern digital ecosystems, for which Large Language Models (LLMs) are an increasingly prominent and popular approach due to their ability to comprehend the rich semantic context within behavioral data. A common practice is to use LLMs to predict a user's next action by directly generating a ranked list of preferred items. Although effective for short-term prediction, the end-to-end generation paradigm inherently limits personalization. Its opaque decision-making process obscures holistic user profiling and exacerbates popularity bias. To address these limitations, we propose Preference Evolution Tracking (PET), a framework that reframes the task as inferring a dynamic probability distribution over a stable and interpretable lattice of preference clusters. By applying logit-probing and generative classification techniques, PET infers a user's preference as a probability distribution, enabling transparent preference learning. On public benchmarks (Yelp, MovieLens), PET improves ranking quality by up to 40% in NDCG over direct generation baselines. On a large-scale, real-world dataset from a short-video platform, it excels at ranking long-tail contents, significantly outperforming a SOTA production model by 7 times in the NDCG score. Ultimately, PET transforms the user profile model from direct preference list generation to a transparent distributional preference mapping, paving the way for more explainable, fair, and diverse personalization systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13772v1">Can LLMs Create Legally Relevant Summaries and Analyses of Videos?</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 Accepted for publication at JURIX 2025 Torino, Italy. This is the preprint version. Code and data available at: https://github.com/maastrichtlawtech/jurix2025_LLM_video_analysis
    </div>
    <details class="paper-abstract">
      Understanding the legally relevant factual basis of an event and conveying it through text is a key skill of legal professionals. This skill is important for preparing forms (e.g., insurance claims) or other legal documents (e.g., court claims), but often presents a challenge for laypeople. Current AI approaches aim to bridge this gap, but mostly rely on the user to articulate what has happened in text, which may be challenging for many. Here, we investigate the capability of large language models (LLMs) to understand and summarize events occurring in videos. We ask an LLM to summarize and draft legal letters, based on 120 YouTube videos showing legal issues in various domains. Overall, 71.7\% of the summaries were rated as of high or medium quality, which is a promising result, opening the door to a number of applications in e.g. access to justice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.13812v2">The Empty Chair: Using LLMs to Raise Missing Perspectives in Policy Deliberations</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: PersonaLLM: Workshop on LLM Persona Modeling
    </div>
    <details class="paper-abstract">
      Deliberation is essential to well-functioning democracies, yet physical, economic, and social barriers often exclude certain groups, reducing representativeness and contributing to issues like group polarization. In this work, we explore the use of large language model (LLM) personas to introduce missing perspectives in policy deliberations. We develop and evaluate a tool that transcribes conversations in real-time and simulates input from relevant but absent stakeholders. We deploy this tool in a 19-person student citizens' assembly on campus sustainability. Participants and facilitators found that the tool was useful to spark new discussions and surfaced valuable perspectives they had not previously considered. However, they also raised skepticism about the ability of LLMs to accurately characterize the perspectives of different groups, especially ones that are already underrepresented. Overall, this case study highlights that while AI personas can usefully surface new perspectives and prompt discussion in deliberative settings, their successful deployment depends on clarifying their limitations and emphasizing that they complement rather than replace genuine participation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.25506v2">Reflections on the Reproducibility of Commercial LLM Performance in Empirical Software Engineering Studies</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      Large Language Models have gained remarkable interest in industry and academia. The increasing interest in LLMs in academia is also reflected in the number of publications on this topic over the last years. For instance, alone 78 of the around 425 publications at ICSE 2024 performed experiments with LLMs. Conducting empirical studies with LLMs remains challenging and raises questions on how to achieve reproducible results, for both other researchers and practitioners. One important step towards excelling in empirical research on LLMs and their application is to first understand to what extent current research results are eventually reproducible and what factors may impede reproducibility. This investigation is within the scope of our work. We contribute an analysis of the reproducibility of LLM-centric studies, provide insights into the factors impeding reproducibility, and discuss suggestions on how to improve the current state. In particular, we studied the 86 articles describing LLM-centric studies, published at ICSE 2024 and ASE 2024. Of the 86 articles, 18 provided research artefacts and used OpenAI models. We attempted to replicate those 18 studies. Of the 18 studies, only five were fit for reproduction. For none of the five studies, we were able to fully reproduce the results. Two studies seemed to be partially reproducible, and three studies did not seem to be reproducible. Our results highlight not only the need for stricter research artefact evaluations but also for more robust study designs to ensure the reproducible value of future publications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10222v2">Speech-Audio Compositional Attacks on Multimodal LLMs and Their Mitigation with SALMONN-Guard</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      Recent progress in large language models (LLMs) has enabled understanding of both speech and non-speech audio, but exposing new safety risks emerging from complex audio inputs that are inadequately handled by current safeguards. We introduce SACRED-Bench (Speech-Audio Composition for RED-teaming) to evaluate the robustness of LLMs under complex audio-based attacks. Unlike existing perturbation-based methods that rely on noise optimization or white-box access, SACRED-Bench exploits speech-audio composition mechanisms. SACRED-Bench adopts three mechanisms: (a) speech overlap and multi-speaker dialogue, which embeds harmful prompts beneath or alongside benign speech; (b) speech-audio mixture, which imply unsafe intent via non-speech audio alongside benign speech or audio; and (c) diverse spoken instruction formats (open-ended QA, yes/no) that evade text-only filters. Experiments show that, even Gemini 2.5 Pro, the state-of-the-art proprietary LLM, still exhibits 66% attack success rate in SACRED-Bench test set, exposing vulnerabilities under cross-modal, speech-audio composition attacks. To bridge this gap, we propose SALMONN-Guard, a safeguard LLM that jointly inspects speech, audio, and text for safety judgments, reducing attack success down to 20%. Our results highlight the need for audio-aware defenses for the safety of multimodal LLMs. The benchmark and SALMONN-Guard checkpoints can be found at https://huggingface.co/datasets/tsinghua-ee/SACRED-Bench. Warning: this paper includes examples that may be offensive or harmful.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.10624v3">Comprehension Without Competence: Architectural Limits of LLMs in Symbolic Computation and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 v2: Two TMLR revision rounds addressing reviewer feedback. Added real-world validation (3.4), interpretability analysis (7), computational hallucination framework, strengthened theory. v3: Sec 3.2 - added transformer architecture diagram, clarified UAT capacity vs computational limits, improved role specialization theorem presentation
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) display striking surface fluency yet systematically fail at tasks requiring symbolic reasoning, arithmetic accuracy, and logical consistency. This paper offers a structural diagnosis of such failures, revealing a persistent gap between \textit{comprehension} and \textit{competence}. Through controlled experiments and architectural analysis, we demonstrate that LLMs often articulate correct principles without reliably applying them--a failure rooted not in knowledge access, but in computational execution. We term this phenomenon the computational \textit{split-brain syndrome}, where instruction and action pathways are geometrically and functionally dissociated. This core limitation recurs across domains, from mathematical operations to relational inferences, and explains why model behavior remains brittle even under idealized prompting. We argue that LLMs function as powerful pattern completion engines, but lack the architectural scaffolding for principled, compositional reasoning. Our findings delineate the boundary of current LLM capabilities and motivate future models with metacognitive control, principle lifting, and structurally grounded execution. This diagnosis also clarifies why mechanistic interpretability findings may reflect training-specific pattern coordination rather than universal computational principles, and why the geometric separation between instruction and execution pathways suggests limitations in neural introspection and mechanistic analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11356v1">SEAL: Subspace-Anchored Watermarks for LLM Ownership</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable success across a wide range of natural language processing tasks, demonstrating human-level performance in text generation, reasoning, and question answering. However, training such models requires substantial computational resources, large curated datasets, and sophisticated alignment procedures. As a result, they constitute highly valuable intellectual property (IP) assets that warrant robust protection mechanisms. Existing IP protection approaches suffer from critical limitations. Model fingerprinting techniques can identify model architectures but fail to establish ownership of specific model instances. In contrast, traditional backdoor-based watermarking methods embed behavioral anomalies that can be easily removed through common post-processing operations such as fine-tuning or knowledge distillation. We propose SEAL, a subspace-anchored watermarking framework that embeds multi-bit signatures directly into the model's latent representational space, supporting both white-box and black-box verification scenarios. Our approach leverages model editing techniques to align the hidden representations of selected anchor samples with predefined orthogonal bit vectors. This alignment embeds the watermark while preserving the model's original factual predictions, rendering the watermark functionally harmless and stealthy. We conduct comprehensive experiments on multiple benchmark datasets and six prominent LLMs, comparing SEAL with 11 existing fingerprinting and watermarking methods to demonstrate its superior effectiveness, fidelity, efficiency, and robustness. Furthermore, we evaluate SEAL under potential knowledgeable attacks and show that it maintains strong verification performance even when adversaries possess knowledge of the watermarking mechanism and the embedded signatures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.16851v2">Interpretable LLM Guardrails via Sparse Representation Steering</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit impressive capabilities in generation tasks but are prone to producing harmful, misleading, or biased content, posing significant ethical and safety concerns. To mitigate such risks, representation engineering, which steer model behavior toward desired attributes by injecting carefully designed steering vectors into LLM's representations at inference time, has emerged as a promising alternative to fine-tuning approaches. However, due to the semantically entangled nature of LLM's representation, existing representation engineering methods still suffer from several limitations: limited fine-grained controllability, content quality degradation, and conflict in multi-attribute control. To overcome these challenges, we propose Sparse Representation Steering (SRS), a novel framework that achieves fine-grained and interpretable control over LLM behavior by first disentangling internal activations into a sparse, semantically meaningful representation space, and then selectively steering relevant dimensions. Specifically, SRS leverages a pretrained Sparse Autoencoder (SAE) to transform dense, entangled activation patterns into a sparse monosemantic feature space. To identify relevant features, SRS contrasts sparse activations from positive and negative prompt pairs and measures their bidirectional KL divergence to locate dimensions most associated with the target attribute. We conduct comprehensive experiments on Gemma-2 series model across three alignment dimensions, i.e., safety, fairness, and truthfulness, to evaluate the effectiveness of SRS. Results show that SRS consistently outperforms existing steering methods, which achieves significantly improved controllability across both single and multiple attribute settings, while preserving high linguistic quality and general ability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11306v1">iMAD: Intelligent Multi-Agent Debate for Efficient and Accurate LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 Accepted in AAAI 2026 (Oral)
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agent systems have advanced rapidly, driven by their strong generalization in zero-shot settings. To further enhance reasoning and accuracy on complex tasks, Multi-Agent Debate (MAD) has emerged as a promising framework that engages multiple LLM agents in structured debates to encourage diverse reasoning. However, triggering MAD for every query is inefficient, as it incurs substantial computational (token) cost and may even degrade accuracy by overturning correct single-agent answers. To address these limitations, we propose intelligent Multi-Agent Debate (iMAD), a token-efficient framework that selectively triggers MAD only when it is likely to be beneficial (i.e., correcting an initially wrong answer). To achieve this goal, iMAD learns generalizable model behaviors to make accurate debate decisions. Specifically, iMAD first prompts a single agent to produce a structured self-critique response, from which we extract 41 interpretable linguistic and semantic features capturing hesitation cues. Then, iMAD uses a lightweight debate-decision classifier, trained using our proposed FocusCal loss, to determine whether to trigger MAD, enabling robust debate decisions without test dataset-specific tuning. Through extensive experiments using six (visual) question answering datasets against five competitive baselines, we have shown that iMAD significantly reduces token usage (by up to 92%) while also improving final answer accuracy (by up to 13.5%).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11258v1">KGQuest: Template-Driven QA Generation from Knowledge Graphs with LLM-Based Refinement</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      The generation of questions and answers (QA) from knowledge graphs (KG) plays a crucial role in the development and testing of educational platforms, dissemination tools, and large language models (LLM). However, existing approaches often struggle with scalability, linguistic quality, and factual consistency. This paper presents a scalable and deterministic pipeline for generating natural language QA from KGs, with an additional refinement step using LLMs to further enhance linguistic quality. The approach first clusters KG triplets based on their relations, creating reusable templates through natural language rules derived from the entity types of objects and relations. A module then leverages LLMs to refine these templates, improving clarity and coherence while preserving factual accuracy. Finally, the instantiation of answer options is achieved through a selection strategy that introduces distractors from the KG. Our experiments demonstrate that this hybrid approach efficiently generates high-quality QA pairs, combining scalability with fluency and linguistic precision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11257v1">AIonopedia: an LLM agent orchestrating multimodal learning for ionic liquid discovery</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      The discovery of novel Ionic Liquids (ILs) is hindered by critical challenges in property prediction, including limited data, poor model accuracy, and fragmented workflows. Leveraging the power of Large Language Models (LLMs), we introduce AIonopedia, to the best of our knowledge, the first LLM agent for IL discovery. Powered by an LLM-augmented multimodal domain foundation model for ILs, AIonopedia enables accurate property predictions and incorporates a hierarchical search architecture for molecular screening and design. Trained and evaluated on a newly curated and comprehensive IL dataset, our model delivers superior performance. Complementing these results, evaluations on literature-reported systems indicate that the agent can perform effective IL modification. Moving beyond offline tests, the practical efficacy was further confirmed through real-world wet-lab validation, in which the agent demonstrated exceptional generalization capabilities on challenging out-of-distribution tasks, underscoring its ability to accelerate real-world IL discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11255v1">Align$^3$GR: Unified Multi-Level Alignment for LLM-based Generative Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 Accepted by AAAI 2026 (Oral)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate significant advantages in leveraging structured world knowledge and multi-step reasoning capabilities. However, fundamental challenges arise when transforming LLMs into real-world recommender systems due to semantic and behavioral misalignment. To bridge this gap, we propose Align$^3$GR, a novel framework that unifies token-level, behavior modeling-level, and preference-level alignment. Our approach introduces: Dual tokenization fusing user-item semantic and collaborative signals. Enhanced behavior modeling with bidirectional semantic alignment. Progressive DPO strategy combining self-play (SP-DPO) and real-world feedback (RF-DPO) for dynamic preference adaptation. Experiments show Align$^3$GR outperforms the SOTA baseline by +17.8% in Recall@10 and +20.2% in NDCG@10 on the public dataset, with significant gains in online A/B tests and full-scale deployment on an industrial large-scale recommendation platform.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11252v1">UAVBench: An Open Benchmark Dataset for Autonomous and Agentic AI UAV Systems via LLM-Generated Flight Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 18 pages, 5 Figures
    </div>
    <details class="paper-abstract">
      Autonomous aerial systems increasingly rely on large language models (LLMs) for mission planning, perception, and decision-making, yet the lack of standardized and physically grounded benchmarks limits systematic evaluation of their reasoning capabilities. To address this gap, we introduce UAVBench, an open benchmark dataset comprising 50,000 validated UAV flight scenarios generated through taxonomy-guided LLM prompting and multi-stage safety validation. Each scenario is encoded in a structured JSON schema that includes mission objectives, vehicle configuration, environmental conditions, and quantitative risk labels, providing a unified representation of UAV operations across diverse domains. Building on this foundation, we present UAVBench_MCQ, a reasoning-oriented extension containing 50,000 multiple-choice questions spanning ten cognitive and ethical reasoning styles, ranging from aerodynamics and navigation to multi-agent coordination and integrated reasoning. This framework enables interpretable and machine-checkable assessment of UAV-specific cognition under realistic operational contexts. We evaluate 32 state-of-the-art LLMs, including GPT-5, ChatGPT-4o, Gemini 2.5 Flash, DeepSeek V3, Qwen3 235B, and ERNIE 4.5 300B, and find strong performance in perception and policy reasoning but persistent challenges in ethics-aware and resource-constrained decision-making. UAVBench establishes a reproducible and physically grounded foundation for benchmarking agentic AI in autonomous aerial systems and advancing next-generation UAV reasoning intelligence. To support open science and reproducibility, we release the UAVBench dataset, the UAVBench_MCQ benchmark, evaluation scripts, and all related materials on GitHub at https://github.com/maferrag/UAVBench
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11250v1">Prompt Engineering vs. Fine-Tuning for LLM-Based Vulnerability Detection in Solana and Algorand Smart Contracts</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      Smart contracts have emerged as key components within decentralized environments, enabling the automation of transactions through self-executing programs. While these innovations offer significant advantages, they also present potential drawbacks if the smart contract code is not carefully designed and implemented. This paper investigates the capability of large language models (LLMs) to detect OWASP-inspired vulnerabilities in smart contracts beyond the Ethereum Virtual Machine (EVM) ecosystem, focusing specifically on Solana and Algorand. Given the lack of labeled datasets for non-EVM platforms, we design a synthetic dataset of annotated smart contract snippets in Rust (for Solana) and PyTeal (for Algorand), structured around a vulnerability taxonomy derived from OWASP. We evaluate LLMs under three configurations: prompt engineering, fine-tuning, and a hybrid of both, comparing their performance on different vulnerability categories. Experimental results show that prompt engineering achieves general robustness, while fine-tuning improves precision and recall on less semantically rich languages such as TEAL. Additionally, we analyze how the architectural differences of Solana and Algorand influence the manifestation and detectability of vulnerabilities, offering platform-specific mappings that highlight limitations in existing security tooling. Our findings suggest that LLM-based approaches are viable for static vulnerability detection in smart contracts, provided domain-specific data and categorization are integrated into training pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11248v1">T-MAN: Enabling End-to-End Low-Bit LLM Inference on NPUs via Unified Table Lookup</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed on customer devices. To support them, current devices are adopting SoCs (System on Chip) with NPUs (Neural Processing Unit) installed. Although high performance is expected, LLM inference on NPUs is slower than its CPU counterpart. The reason is that NPUs have poor performance on computations other than GEMM, like dequantization. Current works either disaggregate prefill on the NPUs and decoding on the CPUs, or put both on the NPUs but with an accuracy loss. To solve this issue, based on the insight that low-bit can enable target computation encoded within an acceptably sized table, we propose table lookup to subsume hardware operations otherwise unsupported. To realize this, we overcome the conflicting hardware behavior of prefill and decoding to design a unified table layout and tiling through (1) fused two-level table-based dequantization and (2) concurrency-hierarchy-guided tiling. Based on that, we implement the prefill phase by three-stage pipeline and map the table-lookup-based decoding to NPU's vector units. Results show 1.4x and 3.1x speedup for prefill and decoding respectively, and 84% energy savings compared to the baseline NPU methods. The code is available at https://github.com/microsoft/T-MAC/tree/main/t-man.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.09719v3">ICL-Router: In-Context Learned Model Representations for LLM Routing</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often exhibit complementary strengths. Model routing harnesses these strengths by dynamically directing each query to the most suitable model, given a candidate model pool. However, routing performance relies on accurate model representations, and adding new models typically requires retraining, limiting scalability. To address these challenges, we propose a novel routing method using in-context vectors to represent model capabilities. The method proceeds in two stages. First, queries are embedded and projected into vectors, with a projector and LLM-based router trained to reconstruct the original queries, aligning vector representations with the router's semantic space. Second, each candidate model is profiled on a query set, and the router learns -- based on in-context vectors of query and model performance -- to predict whether each model can correctly answer new queries. Extensive experiments demonstrate that our method achieves state-of-the-art routing performance in both in-distribution and out-of-distribution tasks. Moreover, our method allows for seamless integration of new models without retraining the router. The code is available at https://github.com/lalalamdbf/ICL-Router.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11125v1">Utilizing LLMs for Industrial Process Automation: A Case Study on Modifying RAPID Programs</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 Submitted to the International Conference on Software Engineering (ICSE) track Software Engineering in Practice (SEIP) 2026
    </div>
    <details class="paper-abstract">
      How to best use Large Language Models (LLMs) for software engineering is covered in many publications in recent years. However, most of this work focuses on widely-used general purpose programming languages. The utility of LLMs for software within the industrial process automation domain, with highly-specialized languages that are typically only used in proprietary contexts, is still underexplored. Within this paper, we study enterprises can achieve on their own without investing large amounts of effort into the training of models specific to the domain-specific languages that are used. We show that few-shot prompting approaches are sufficient to solve simple problems in a language that is otherwise not well-supported by an LLM and that is possible on-premise, thereby ensuring the protection of sensitive company data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11106v1">AccKV: Towards Efficient Audio-Video LLMs Inference via Adaptive-Focusing and Cross-Calibration KV Cache Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      Recent advancements in Audio-Video Large Language Models (AV-LLMs) have enhanced their capabilities in tasks like audio-visual question answering and multimodal dialog systems. Video and audio introduce an extended temporal dimension, resulting in a larger key-value (KV) cache compared to static image embedding. A naive optimization strategy is to selectively focus on and retain KV caches of audio or video based on task. However, in the experiment, we observed that the attention of AV-LLMs to various modalities in the high layers is not strictly dependent on the task. In higher layers, the attention of AV-LLMs shifts more towards the video modality. In addition, we also found that directly integrating temporal KV of audio and spatial-temporal KV of video may lead to information confusion and significant performance degradation of AV-LLMs. If audio and video are processed indiscriminately, it may also lead to excessive compression or reservation of a certain modality, thereby disrupting the alignment between modalities. To address these challenges, we propose AccKV, an Adaptive-Focusing and Cross-Calibration KV cache optimization framework designed specifically for efficient AV-LLMs inference. Our method is based on layer adaptive focusing technology, selectively focusing on key modalities according to the characteristics of different layers, and enhances the recognition of heavy hitter tokens through attention redistribution. In addition, we propose a Cross-Calibration technique that first integrates inefficient KV caches within the audio and video modalities, and then aligns low-priority modalities with high-priority modalities to selectively evict KV cache of low-priority modalities. The experimental results show that AccKV can significantly improve the computational efficiency of AV-LLMs while maintaining accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.20856v3">Strada-LLM: Graph LLM for traffic prediction</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      Traffic forecasting is pivotal for intelligent transportation systems, where accurate and interpretable predictions can significantly enhance operational efficiency and safety. A key challenge stems from the heterogeneity of traffic conditions across diverse locations, leading to highly varied traffic data distributions. Large language models (LLMs) show exceptional promise for few-shot learning in such dynamic and data-sparse scenarios. However, existing LLM-based solutions often rely on prompt-tuning, which can struggle to fully capture complex graph relationships and spatiotemporal dependencies-thereby limiting adaptability and interpretability in real-world traffic networks. We address these gaps by introducing Strada-LLM, a novel multivariate probabilistic forecasting LLM that explicitly models both temporal and spatial traffic patterns. By incorporating proximal traffic information as covariates, Strada-LLM more effectively captures local variations and outperforms prompt-based existing LLMs. To further enhance adaptability, we propose a lightweight distribution-derived strategy for domain adaptation, enabling parameter-efficient model updates when encountering new data distributions or altered network topologies-even under few-shot constraints. Empirical evaluations on spatio-temporal transportation datasets demonstrate that Strada-LLM consistently surpasses state-of-the-art LLM-driven and traditional GNN-based predictors. Specifically, it improves long-term forecasting by 17% in RMSE error and 16% more efficiency. Moreover, it maintains robust performance across different LLM backbones with minimal degradation, making it a versatile and powerful solution for real-world traffic prediction tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.05129v2">Navigating Through Paper Flood: Advancing LLM-based Paper Evaluation through Domain-Aware Retrieval and Latent Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 Accepted for publication in AAAI'26
    </div>
    <details class="paper-abstract">
      With the rapid and continuous increase in academic publications, identifying high-quality research has become an increasingly pressing challenge. While recent methods leveraging Large Language Models (LLMs) for automated paper evaluation have shown great promise, they are often constrained by outdated domain knowledge and limited reasoning capabilities. In this work, we present PaperEval, a novel LLM-based framework for automated paper evaluation that addresses these limitations through two key components: 1) a domain-aware paper retrieval module that retrieves relevant concurrent work to support contextualized assessments of novelty and contributions, and 2) a latent reasoning mechanism that enables deep understanding of complex motivations and methodologies, along with comprehensive comparison against concurrently related work, to support more accurate and reliable evaluation. To guide the reasoning process, we introduce a progressive ranking optimization strategy that encourages the LLM to iteratively refine its predictions with an emphasis on relative comparison. Experiments on two datasets demonstrate that PaperEval consistently outperforms existing methods in both academic impact and paper quality evaluation. In addition, we deploy PaperEval in a real-world paper recommendation system for filtering high-quality papers, which has gained strong engagement on social media -- amassing over 8,000 subscribers and attracting over 10,000 views for many filtered high-quality papers -- demonstrating the practical effectiveness of PaperEval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11087v1">Can LLMs Detect Their Own Hallucinations?</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can generate fluent responses, but sometimes hallucinate facts. In this paper, we investigate whether LLMs can detect their own hallucinations. We formulate hallucination detection as a classification task of a sentence. We propose a framework for estimating LLMs' capability of hallucination detection and a classification method using Chain-of-Thought (CoT) to extract knowledge from their parameters. The experimental results indicated that GPT-$3.5$ Turbo with CoT detected $58.2\%$ of its own hallucinations. We concluded that LLMs with CoT can detect hallucinations if sufficient knowledge is contained in their parameters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11019v1">PATCHEVAL: A New Benchmark for Evaluating LLMs on Patching Real-World Vulnerabilities</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      Software vulnerabilities are increasing at an alarming rate. However, manual patching is both time-consuming and resource-intensive, while existing automated vulnerability repair (AVR) techniques remain limited in effectiveness. Recent advances in large language models (LLMs) have opened a new paradigm for AVR, demonstrating remarkable progress. To examine the capability of LLMs in AVR, several vulnerability benchmarks have been proposed recently. However, they still suffer from key limitations of outdated vulnerabilities, limited language coverage, unreliable patch validation, and insufficient reproducibility. To overcome these challenges, we introduce PATCHEVAL, a multilingual benchmark for Go, JavaScript, and Python, languages for which existing benchmarks remain unexplored. PATCHEVAL curates a dataset of 1,000 vulnerabilities drawn from CVEs reported between 2015 and 2025, covering 65 distinct CWEs. A subset of 230 CVEs is further equipped with runtime sandbox environments, enabling patch verification through both security tests and functionality tests. To provide a systematic comparison of LLM-based vulnerability repair, we evaluate a series of state-of-the-art LLMs and agents, presenting an in-depth analysis that empirically yields key insights to guide future research in AVR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.18544v2">SLICE: SLO-Driven Scheduling for LLM Inference on Edge Computing Devices</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 This work has been submitted to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), as the foundational architecture for next-generation interactive AI applications, not only power intelligent dialogue systems but also drive the evolution of embodied intelligence on edge devices, including humanoid robots, smart vehicles, and other scenarios. The applications running on these edge devices impose differentiated Service Level Objectives (SLO) requirements on LLM services, specifically manifested as distinct constraints on Time to First Token (TTFT) and Time Per Output Token (TPOT) as well as end-to-end latency. Notably, edge devices typically handle real-time tasks that are extremely sensitive to latency, such as machine control and navigation planning. However, existing scheduling service systems still prioritize maximizing output token throughput as the sole optimization objective, failing to adequately address the diversity of SLO requirements. This ultimately results in persistently high violation rates for end-to-end latency or TPOT related SLOs. This paper proposes SLICE, an innovative scheduling solution designed for edge computing scenarios with differentiated SLO requirements. By combining a utility-maximizing request scheduling algorithm with a dynamic iterative control mechanism for generation rates, SLICE significantly improves LLM inference service SLO attainment. Experimental results demonstrate that compared to state-of-the-art solutions Orca and FastServe, SLICE achieves up to 35x higher SLO attainment and 3.4x advantage in task completion time than the other two solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11000v1">DialogGraph-LLM: Graph-Informed LLMs for End-to-End Audio Dialogue Intent Recognition</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 8 pages, 2 figures; Series: Frontiers in Artificial Intelligence and Applications, Volume 413: ECAI 2025
    </div>
    <details class="paper-abstract">
      Recognizing speaker intent in long audio dialogues among speakers has a wide range of applications, but is a non-trivial AI task due to complex inter-dependencies in speaker utterances and scarce annotated data. To address these challenges, an end-to-end framework, namely DialogGraph-LLM, is proposed in the current work. DialogGraph-LLM combines a novel Multi-Relational Dialogue Attention Network (MR-DAN) architecture with multimodal foundation models (e.g., Qwen2.5-Omni-7B) for direct acoustic-to-intent inference. An adaptive semi-supervised learning strategy is designed using LLM with a confidence-aware pseudo-label generation mechanism based on dual-threshold filtering using both global and class confidences, and an entropy-based sample selection process that prioritizes high-information unlabeled instances. Extensive evaluations on the proprietary MarketCalls corpus and the publicly available MIntRec 2.0 benchmark demonstrate DialogGraph-LLM's superiority over strong audio and text-driven baselines. The framework demonstrates strong performance and efficiency in intent recognition in real world scenario audio dialogues, proving its practical value for audio-rich domains with limited supervision. Our code is available at https://github.com/david188888/DialogGraph-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10979v1">PAS: A Training-Free Stabilizer for Temporal Encoding in Video LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 13 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Video LLMs suffer from temporal inconsistency: small shifts in frame timing can flip attention and suppress relevant frames. We trace this instability to the common extension of Rotary Position Embeddings to video through multimodal RoPE. The induced inverse Fourier time kernel exhibits frame-scale ripples that multiply adjacent frames by different factors, which perturbs attention that should otherwise be governed by the raw query key inner product. We present Phase Aggregated Smoothing (PAS), a simple, training-free mechanism that applies small opposed phase offsets across heads and then aggregates their outputs. PAS preserves the per-head spectrum magnitude, while the aggregation effectively smooths the temporal kernel and reduces phase sensitivity without changing the positional encoding structure. Our analysis shows that the RoPE rotated logit can be approximated as a content dot product scaled by a time kernel; smoothing this kernel yields Lipschitz stability of attention to small temporal shifts; multi phase averaging attenuates high frequency ripples while preserving per-head spectra under Nyquist-valid sampling. Experiments on multiple video understanding benchmarks under matched token budgets show consistent improvements with negligible computational overhead. PAS provides a plug and play upgrade for robust temporal encoding in Video LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07943v2">Thinker: Training LLMs in Hierarchical Thinking for Deep Search via Multi-Turn Interaction</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 Accepted to AAAI 2026. Extended version with full Appendix
    </div>
    <details class="paper-abstract">
      Efficient retrieval of external knowledge bases and web pages is crucial for enhancing the reasoning abilities of LLMs. Previous works on training LLMs to leverage external retrievers for solving complex problems have predominantly employed end-to-end reinforcement learning. However, these approaches neglect supervision over the reasoning process, making it difficult to guarantee logical coherence and rigor. To address these limitations, we propose Thinker, a hierarchical thinking model for deep search through multi-turn interaction, making the reasoning process supervisable and verifiable. It decomposes complex problems into independently solvable sub-problems, each dually represented in both natural language and an equivalent logical function to support knowledge base and web searches. Concurrently, dependencies between sub-problems are passed as parameters via these logical functions, enhancing the logical coherence of the problem-solving process. To avoid unnecessary external searches, we perform knowledge boundary determination to check if a sub-problem is within the LLM's intrinsic knowledge, allowing it to answer directly. Experimental results indicate that with as few as several hundred training samples, the performance of Thinker is competitive with established baselines. Furthermore, when scaled to the full training set, Thinker significantly outperforms these methods across various datasets and model sizes. The source code is available at https://github.com/OpenSPG/KAG-Thinker.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.15980v2">Text-to-SQL Domain Adaptation via Human-LLM Collaborative Data Annotation</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 Accepted by IUI'25 Code & Demo: https://github.com/magic-YuanTian/SQLsynth
    </div>
    <details class="paper-abstract">
      Text-to-SQL models, which parse natural language (NL) questions to executable SQL queries, are increasingly adopted in real-world applications. However, deploying such models in the real world often requires adapting them to the highly specialized database schemas used in specific applications. We find that existing text-to-SQL models experience significant performance drops when applied to new schemas, primarily due to the lack of domain-specific data for fine-tuning. This data scarcity also limits the ability to effectively evaluate model performance in new domains. Continuously obtaining high-quality text-to-SQL data for evolving schemas is prohibitively expensive in real-world scenarios. To bridge this gap, we propose SQLsynth, a human-in-the-loop text-to-SQL data annotation system. SQLsynth streamlines the creation of high-quality text-to-SQL datasets through human-LLM collaboration in a structured workflow. A within-subjects user study comparing SQLsynth with manual annotation and ChatGPT shows that SQLsynth significantly accelerates text-to-SQL data annotation, reduces cognitive load, and produces datasets that are more accurate, natural, and diverse. Our code is available at https://github.com/magic-YuanTian/SQLsynth.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10948v1">DEFT-LLM: Disentangled Expert Feature Tuning for Micro-Expression Recognition</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      Micro expression recognition (MER) is crucial for inferring genuine emotion. Applying a multimodal large language model (MLLM) to this task enables spatio-temporal analysis of facial motion and provides interpretable descriptions. However, there are still two core challenges: (1) The entanglement of static appearance and dynamic motion cues prevents the model from focusing on subtle motion; (2) Textual labels in existing MER datasets do not fully correspond to underlying facial muscle movements, creating a semantic gap between text supervision and physical motion. To address these issues, we propose DEFT-LLM, which achieves motion semantic alignment by multi-expert disentanglement. We first introduce Uni-MER, a motion-driven instruction dataset designed to align text with local facial motion. Its construction leverages dual constraints from optical flow and Action Unit (AU) labels to ensure spatio-temporal consistency and reasonable correspondence to the movements. We then design an architecture with three experts to decouple facial dynamics into independent and interpretable representations (structure, dynamic textures, and motion-semantics). By integrating the instruction-aligned knowledge from Uni-MER into DEFT-LLM, our method injects effective physical priors for micro expressions while also leveraging the cross modal reasoning ability of large language models, thus enabling precise capture of subtle emotional cues. Experiments on multiple challenging MER benchmarks demonstrate state-of-the-art performance, as well as a particular advantage in interpretable modeling of local facial motion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10067v2">Enhancing the Medical Context-Awareness Ability of LLMs via Multifaceted Self-Refinement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 20 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown great promise in the medical domain, achieving strong performance on several benchmarks. However, they continue to underperform in real-world medical scenarios, which often demand stronger context-awareness, i.e., the ability to recognize missing or critical details (e.g., user identity, medical history, risk factors) and provide safe, helpful, and contextually appropriate responses. To address this issue, we propose Multifaceted Self-Refinement (MuSeR), a data-driven approach that enhances LLMs' context-awareness along three key facets (decision-making, communication, and safety) through self-evaluation and refinement. Specifically, we first design a attribute-conditioned query generator that simulates diverse real-world user contexts by varying attributes such as role, geographic region, intent, and degree of information ambiguity. An LLM then responds to these queries, self-evaluates its answers along three key facets, and refines its responses to better align with the requirements of each facet. Finally, the queries and refined responses are used for supervised fine-tuning to reinforce the model's context-awareness ability. Evaluation results on the latest HealthBench dataset demonstrate that our method significantly improves LLM performance across multiple aspects, with particularly notable gains in the context-awareness axis. Furthermore, by incorporating knowledge distillation with the proposed method, the performance of a smaller backbone LLM (e.g., Qwen3-32B) surpasses its teacher model, achieving a new SOTA across all open-source LLMs on HealthBench (63.8%) and its hard subset (43.1%). Code and dataset will be released at https://muser-llm.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10890v1">LLM enhanced graph inference for long-term disease progression modelling</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      Understanding the interactions between biomarkers among brain regions during neurodegenerative disease is essential for unravelling the mechanisms underlying disease progression. For example, pathophysiological models of Alzheimer's Disease (AD) typically describe how variables, such as regional levels of toxic proteins, interact spatiotemporally within a dynamical system driven by an underlying biological substrate, often based on brain connectivity. However, current methods grossly oversimplify the complex relationship between brain connectivity by assuming a single-modality brain connectome as the disease-spreading substrate. This leads to inaccurate predictions of pathology spread, especially during the long-term progression period. Meanhwile, other methods of learning such a graph in a purely data-driven way face the identifiability issue due to lack of proper constraint. We thus present a novel framework that uses Large Language Models (LLMs) as expert guides on the interaction of regional variables to enhance learning of disease progression from irregularly sampled longitudinal patient data. By leveraging LLMs' ability to synthesize multi-modal relationships and incorporate diverse disease-driving mechanisms, our method simultaneously optimizes 1) the construction of long-term disease trajectories from individual-level observations and 2) the biologically-constrained graph structure that captures interactions among brain regions with better identifiability. We demonstrate the new approach by estimating the pathology propagation using tau-PET imaging data from an Alzheimer's disease cohort. The new framework demonstrates superior prediction accuracy and interpretability compared to traditional approaches while revealing additional disease-driving factors beyond conventional connectivity measures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.15793v2">Format as a Prior: Quantifying and Analyzing Bias in LLMs for Heterogeneous Data</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 Accepted by AAAI 2026, camera ready version
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly employed in applications that require processing information from heterogeneous formats, including texts, tables, infoboxes, and knowledge graphs. However, systematic biases toward particular formats may undermine LLMs' ability to integrate heterogeneous data impartially, potentially resulting in reasoning errors and increased risks in downstream tasks. Yet it remains unclear whether such biases are systematic, which data-level factors drive them, and what internal mechanisms underlie their emergence. In this paper, we present the first comprehensive study of format bias in LLMs through a three-stage empirical analysis. The first stage explores the presence and direction of bias across a diverse range of LLMs. The second stage examines how key data-level factors influence these biases. The third stage analyzes how format bias emerges within LLMs' attention patterns and evaluates a lightweight intervention to test its effectiveness. Our results show that format bias is consistent across model families, driven by information richness, structure quality, and representation type, and is closely associated with attention imbalance within the LLMs. Based on these investigations, we identify three future research directions to reduce format bias: enhancing data pre-processing through format repair and normalization, introducing inference-time interventions such as attention re-weighting, and developing format-balanced training corpora. These directions will support the design of more robust and fair heterogeneous data processing systems.
    </details>
</div>
