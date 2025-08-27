# llm - 2025_08

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
- [Part 11](papers_11.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17310v1">Handling Students Dropouts in an LLM-driven Interactive Online Course Using Language Models</a></div>
    <div class="paper-meta">
      📅 2025-08-24
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      Interactive online learning environments, represented by Massive AI-empowered Courses (MAIC), leverage LLM-driven multi-agent systems to transform passive MOOCs into dynamic, text-based platforms, enhancing interactivity through LLMs. This paper conducts an empirical study on a specific MAIC course to explore three research questions about dropouts in these interactive online courses: (1) What factors might lead to dropouts? (2) Can we predict dropouts? (3) Can we reduce dropouts? We analyze interaction logs to define dropouts and identify contributing factors. Our findings reveal strong links between dropout behaviors and textual interaction patterns. We then propose a course-progress-adaptive dropout prediction framework (CPADP) to predict dropouts with at most 95.4% accuracy. Based on this, we design a personalized email recall agent to re-engage at-risk students. Applied in the deployed MAIC system with over 3,000 students, the feasibility and effectiveness of our approach have been validated on students with diverse backgrounds.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.09190v3">Fine-Grained Safety Neurons with Training-Free Continual Projection to Reduce LLM Fine Tuning Risks</a></div>
    <div class="paper-meta">
      📅 2025-08-24
    </div>
    <details class="paper-abstract">
      Fine-tuning as service injects domain-specific knowledge into large language models (LLMs), while challenging the original alignment mechanisms and introducing safety risks. A series of defense strategies have been proposed for the alignment, fine-tuning, and post-fine-tuning phases, where most post-fine-tuning defenses rely on coarse-grained safety layer mapping. These methods lack a comprehensive consideration of both safety layers and fine-grained neurons, limiting their ability to efficiently balance safety and utility. To address this, we propose the Fine-Grained Safety Neurons (FGSN) with Training-Free Continual Projection method to reduce the fine-tuning safety risks. FGSN inherently integrates the multi-scale interactions between safety layers and neurons, localizing sparser and more precise fine-grained safety neurons while minimizing interference with downstream task neurons. We then project the safety neuron parameters onto safety directions, improving model safety while aligning more closely with human preferences. Extensive experiments across multiple fine-tuned LLM models demonstrate that our method significantly reduce harmfulness scores and attack success rates with minimal parameter modifications, while preserving the model's utility. Furthermore, by introducing a task-specific, multi-dimensional heterogeneous safety neuron cluster optimization mechanism, we achieve continual defense and generalization capability against unforeseen emerging safety concerns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11742v2">CRABS: A syntactic-semantic pincer strategy for bounding LLM interpretation of Python notebooks</a></div>
    <div class="paper-meta">
      📅 2025-08-24
      | 💬 Accepted to COLM 2025
    </div>
    <details class="paper-abstract">
      Recognizing the information flows and operations comprising data science and machine learning Python notebooks is critical for evaluating, reusing, and adapting notebooks for new tasks. Investigating a notebook via re-execution often is impractical due to the challenges of resolving data and software dependencies. While Large Language Models (LLMs) pre-trained on large codebases have demonstrated effectiveness in understanding code without running it, we observe that they fail to understand some realistic notebooks due to hallucinations and long-context challenges. To address these issues, we propose a notebook understanding task yielding an information flow graph and corresponding cell execution dependency graph for a notebook, and demonstrate the effectiveness of a pincer strategy that uses limited syntactic analysis to assist full comprehension of the notebook using an LLM. Our Capture and Resolve Assisted Bounding Strategy (CRABS) employs shallow syntactic parsing and analysis of the abstract syntax tree (AST) to capture the correct interpretation of a notebook between lower and upper estimates of the inter-cell I/O set$\unicode{x2014}$the flows of information into or out of cells via variables$\unicode{x2014}$then uses an LLM to resolve remaining ambiguities via cell-by-cell zero-shot learning, thereby identifying the true data inputs and outputs of each cell. We evaluate and demonstrate the effectiveness of our approach using an annotated dataset of 50 representative, highly up-voted Kaggle notebooks that together represent 3454 actual cell inputs and outputs. The LLM correctly resolves 1397 of 1425 (98%) ambiguities left by analyzing the syntactic structure of these notebooks. Across 50 notebooks, CRABS achieves average F1 scores of 98% identifying cell-to-cell information flows and 99% identifying transitive cell execution dependencies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06791v2">AutoMisty: A Multi-Agent LLM Framework for Automated Code Generation in the Misty Social Robot</a></div>
    <div class="paper-meta">
      📅 2025-08-24
      | 💬 Accepted by IROS 2025
    </div>
    <details class="paper-abstract">
      The social robot's open API allows users to customize open-domain interactions. However, it remains inaccessible to those without programming experience. In this work, we introduce AutoMisty, the first multi-agent collaboration framework powered by large language models (LLMs), to enable the seamless generation of executable Misty robot code from natural language instructions. AutoMisty incorporates four specialized agent modules to manage task decomposition, assignment, problem-solving, and result synthesis. Each agent incorporates a two-layer optimization mechanism, with self-reflection for iterative refinement and human-in-the-loop for better alignment with user preferences. AutoMisty ensures a transparent reasoning process, allowing users to iteratively refine tasks through natural language feedback for precise execution. To evaluate AutoMisty's effectiveness, we designed a benchmark task set spanning four levels of complexity and conducted experiments in a real Misty robot environment. Extensive evaluations demonstrate that AutoMisty not only consistently generates high-quality code but also enables precise code control, significantly outperforming direct reasoning with ChatGPT-4o and ChatGPT-o1. All code, optimized APIs, and experimental videos will be publicly released through the webpage: https://wangxiaoshawn.github.io/AutoMisty.html
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.17710v5">Optimization-based Prompt Injection Attack to LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-08-24
      | 💬 To appear in the Proceedings of The ACM Conference on Computer and Communications Security (CCS), 2024
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge uses a large language model (LLM) to select the best response from a set of candidates for a given question. LLM-as-a-Judge has many applications such as LLM-powered search, reinforcement learning with AI feedback (RLAIF), and tool selection. In this work, we propose JudgeDeceiver, an optimization-based prompt injection attack to LLM-as-a-Judge. JudgeDeceiver injects a carefully crafted sequence into an attacker-controlled candidate response such that LLM-as-a-Judge selects the candidate response for an attacker-chosen question no matter what other candidate responses are. Specifically, we formulate finding such sequence as an optimization problem and propose a gradient based method to approximately solve it. Our extensive evaluation shows that JudgeDeceive is highly effective, and is much more effective than existing prompt injection attacks that manually craft the injected sequences and jailbreak attacks when extended to our problem. We also show the effectiveness of JudgeDeceiver in three case studies, i.e., LLM-powered search, RLAIF, and tool selection. Moreover, we consider defenses including known-answer detection, perplexity detection, and perplexity windowed detection. Our results show these defenses are insufficient, highlighting the urgent need for developing new defense strategies. Our implementation is available at this repository: https://github.com/ShiJiawenwen/JudgeDeceiver.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17202v1">Active Domain Knowledge Acquisition with \$100 Budget: Enhancing LLMs via Cost-Efficient, Expert-Involved Interaction in Sensitive Domains</a></div>
    <div class="paper-meta">
      📅 2025-08-24
      | 💬 EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated an impressive level of general knowledge. However, they often struggle in highly specialized and cost-sensitive domains such as drug discovery and rare disease research due to the lack of expert knowledge. In this paper, we propose a novel framework (PU-ADKA) designed to efficiently enhance domain-specific LLMs by actively engaging domain experts within a fixed budget. Unlike traditional fine-tuning approaches, PU-ADKA selectively identifies and queries the most appropriate expert from a team, taking into account each expert's availability, knowledge boundaries, and consultation costs. We train PU-ADKA using simulations on PubMed data and validate it through both controlled expert interactions and real-world deployment with a drug development team, demonstrating its effectiveness in enhancing LLM performance in specialized domains under strict budget constraints. In addition to outlining our methodological innovations and experimental results, we introduce a new benchmark dataset, CKAD, for cost-effective LLM domain knowledge acquisition to foster further research in this challenging area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19793v3">Prompt Injection Attack to Tool Selection in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-24
    </div>
    <details class="paper-abstract">
      Tool selection is a key component of LLM agents. A popular approach follows a two-step process - \emph{retrieval} and \emph{selection} - to pick the most appropriate tool from a tool library for a given task. In this work, we introduce \textit{ToolHijacker}, a novel prompt injection attack targeting tool selection in no-box scenarios. ToolHijacker injects a malicious tool document into the tool library to manipulate the LLM agent's tool selection process, compelling it to consistently choose the attacker's malicious tool for an attacker-chosen target task. Specifically, we formulate the crafting of such tool documents as an optimization problem and propose a two-phase optimization strategy to solve it. Our extensive experimental evaluation shows that ToolHijacker is highly effective, significantly outperforming existing manual-based and automated prompt injection attacks when applied to tool selection. Moreover, we explore various defenses, including prevention-based defenses (StruQ and SecAlign) and detection-based defenses (known-answer detection, DataSentinel, perplexity detection, and perplexity windowed detection). Our experimental results indicate that these defenses are insufficient, highlighting the urgent need for developing new defense strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17196v1">BudgetThinker: Empowering Budget-aware LLM Reasoning with Control Tokens</a></div>
    <div class="paper-meta">
      📅 2025-08-24
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have leveraged increased test-time computation to enhance reasoning capabilities, a strategy that, while effective, incurs significant latency and resource costs, limiting their applicability in real-world time-constrained or cost-sensitive scenarios. This paper introduces BudgetThinker, a novel framework designed to empower LLMs with budget-aware reasoning, enabling precise control over the length of their thought processes. We propose a methodology that periodically inserts special control tokens during inference to continuously inform the model of its remaining token budget. This approach is coupled with a comprehensive two-stage training pipeline, beginning with Supervised Fine-Tuning (SFT) to familiarize the model with budget constraints, followed by a curriculum-based Reinforcement Learning (RL) phase that utilizes a length-aware reward function to optimize for both accuracy and budget adherence. We demonstrate that BudgetThinker significantly surpasses strong baselines in maintaining performance across a variety of reasoning budgets on challenging mathematical benchmarks. Our method provides a scalable and effective solution for developing efficient and controllable LLM reasoning, making advanced models more practical for deployment in resource-constrained and real-time environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17188v1">PosterGen: Aesthetic-Aware Paper-to-Poster Generation via Multi-Agent LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-24
      | 💬 Project Website: https://Y-Research-SBU.github.io/PosterGen
    </div>
    <details class="paper-abstract">
      Multi-agent systems built upon large language models (LLMs) have demonstrated remarkable capabilities in tackling complex compositional tasks. In this work, we apply this paradigm to the paper-to-poster generation problem, a practical yet time-consuming process faced by researchers preparing for conferences. While recent approaches have attempted to automate this task, most neglect core design and aesthetic principles, resulting in posters that require substantial manual refinement. To address these design limitations, we propose PosterGen, a multi-agent framework that mirrors the workflow of professional poster designers. It consists of four collaborative specialized agents: (1) Parser and Curator agents extract content from the paper and organize storyboard; (2) Layout agent maps the content into a coherent spatial layout; (3) Stylist agents apply visual design elements such as color and typography; and (4) Renderer composes the final poster. Together, these agents produce posters that are both semantically grounded and visually appealing. To evaluate design quality, we introduce a vision-language model (VLM)-based rubric that measures layout balance, readability, and aesthetic coherence. Experimental results show that PosterGen consistently matches in content fidelity, and significantly outperforms existing methods in visual designs, generating posters that are presentation-ready with minimal human refinements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17182v1">LLM Assertiveness can be Mechanistically Decomposed into Emotional and Logical Components</a></div>
    <div class="paper-meta">
      📅 2025-08-24
      | 💬 This preprint is under review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often display overconfidence, presenting information with unwarranted certainty in high-stakes contexts. We investigate the internal basis of this behavior via mechanistic interpretability. Using open-sourced Llama 3.2 models fine-tuned on human annotated assertiveness datasets, we extract residual activations across all layers, and compute similarity metrics to localize assertive representations. Our analysis identifies layers most sensitive to assertiveness contrasts and reveals that high-assertive representations decompose into two orthogonal sub-components of emotional and logical clusters-paralleling the dual-route Elaboration Likelihood Model in Psychology. Steering vectors derived from these sub-components show distinct causal effects: emotional vectors broadly influence prediction accuracy, while logical vectors exert more localized effects. These findings provide mechanistic evidence for the multi-component structure of LLM assertiveness and highlight avenues for mitigating overconfident behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17642v3">May the Feedback Be with You! Unlocking the Power of Feedback-Driven Deep Learning Framework Fuzzing via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-24
    </div>
    <details class="paper-abstract">
      Deep Learning (DL) frameworks have served as fundamental components in DL systems over the last decade. However, bugs in DL frameworks could lead to catastrophic consequences in critical scenarios. A simple yet effective way to find bugs in DL frameworks is fuzz testing (Fuzzing). Existing approaches focus on test generation, leaving execution results with high semantic value (e.g., coverage information, bug reports, and exception logs) in the wild, which can serve as multiple types of feedback. To fill this gap, we propose FUEL to effectively utilize the feedback information, which comprises two Large Language Models (LLMs): analysis LLM and generation LLM. Specifically, analysis LLM infers analysis summaries from feedback information, while the generation LLM creates tests guided by these summaries. Furthermore, based on multiple feedback guidance, we design two additional components: (i) a feedback-aware simulated annealing algorithm to select operators for test generation, enriching test diversity. (ii) a program self-repair strategy to automatically repair invalid tests, enhancing test validity. We evaluate FUEL on the two most popular DL frameworks, and experiment results show that FUEL can improve line code coverage of PyTorch and TensorFlow by 9.15% and 14.70% over state-of-the-art baselines (e.g., TitanFuzz and WhiteFox). By the time of submission, FUEL has detected 104 previously unknown bugs for PyTorch and TensorFlow, with 93 confirmed as new bugs, 49 already fixed, and 14 assigned CVE IDs. Our artifact is available at https://github.com/NJU-iSE/FUEL
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18321v1">LLMs Can't Handle Peer Pressure: Crumbling under Multi-Agent Social Interactions</a></div>
    <div class="paper-meta">
      📅 2025-08-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in multi-agent systems (MAS) as components of collaborative intelligence, where peer interactions dynamically shape individual decision-making. Although prior work has focused on conformity bias, we extend the analysis to examine how LLMs form trust from previous impressions, resist misinformation, and integrate peer input during interaction, key factors for achieving collective intelligence under complex social dynamics. We present KAIROS, a benchmark simulating quiz contests with peer agents of varying reliability, offering fine-grained control over conditions such as expert-novice roles, noisy crowds, and adversarial peers. LLMs receive both historical interactions and current peer responses, allowing systematic investigation into how trust, peer action, and self-confidence influence decisions. As for mitigation strategies, we evaluate prompting, supervised fine-tuning, and reinforcement learning, Group Relative Policy Optimisation (GRPO), across multiple models. Our results reveal that GRPO with multi-agent context combined with outcome-based rewards and unconstrained reasoning achieves the best overall performance, but also decreases the robustness to social influence compared to Base models. The code and datasets are available at: https://github.com/declare-lab/KAIROS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17164v1">The Impact of Annotator Personas on LLM Behavior Across the Perspectivism Spectrum</a></div>
    <div class="paper-meta">
      📅 2025-08-23
      | 💬 Accepted at ICNLSP 2025, Odense, Denmark
    </div>
    <details class="paper-abstract">
      In this work, we explore the capability of Large Language Models (LLMs) to annotate hate speech and abusiveness while considering predefined annotator personas within the strong-to-weak data perspectivism spectra. We evaluated LLM-generated annotations against existing annotator modeling techniques for perspective modeling. Our findings show that LLMs selectively use demographic attributes from the personas. We identified prototypical annotators, with persona features that show varying degrees of alignment with the original human annotators. Within the data perspectivism paradigm, annotator modeling techniques that do not explicitly rely on annotator information performed better under weak data perspectivism compared to both strong data perspectivism and human annotations, suggesting LLM-generated views tend towards aggregation despite subjective prompting. However, for more personalized datasets tailored to strong perspectivism, the performance of LLM annotator modeling approached, but did not exceed, human annotators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.21520v4">LLM-Forest: Ensemble Learning of LLMs with Graph-Augmented Prompts for Data Imputation</a></div>
    <div class="paper-meta">
      📅 2025-08-23
    </div>
    <details class="paper-abstract">
      Missing data imputation is a critical challenge in various domains, such as healthcare and finance, where data completeness is vital for accurate analysis. Large language models (LLMs), trained on vast corpora, have shown strong potential in data generation, making them a promising tool for data imputation. However, challenges persist in designing effective prompts for a finetuning-free process and in mitigating biases and uncertainty in LLM outputs. To address these issues, we propose a novel framework, LLM-Forest, which introduces a "forest" of few-shot prompt learning LLM "trees" with their outputs aggregated via confidence-based weighted voting based on LLM self-assessment, inspired by the ensemble learning (Random Forest). This framework is established on a new concept of bipartite information graphs to identify high-quality relevant neighboring entries with both feature and value granularity. Extensive experiments on 9 real-world datasets demonstrate the effectiveness and efficiency of LLM-Forest.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17155v1">Mind the Gap: Time-of-Check to Time-of-Use Vulnerabilities in LLM-Enabled Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-23
      | 💬 Pre-print
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-enabled agents are rapidly emerging across a wide range of applications, but their deployment introduces vulnerabilities with security implications. While prior work has examined prompt-based attacks (e.g., prompt injection) and data-oriented threats (e.g., data exfiltration), time-of-check to time-of-use (TOCTOU) remain largely unexplored in this context. TOCTOU arises when an agent validates external state (e.g., a file or API response) that is later modified before use, enabling practical attacks such as malicious configuration swaps or payload injection. In this work, we present the first study of TOCTOU vulnerabilities in LLM-enabled agents. We introduce TOCTOU-Bench, a benchmark with 66 realistic user tasks designed to evaluate this class of vulnerabilities. As countermeasures, we adapt detection and mitigation techniques from systems security to this setting and propose prompt rewriting, state integrity monitoring, and tool-fusing. Our study highlights challenges unique to agentic workflows, where we achieve up to 25% detection accuracy using automated detection methods, a 3% decrease in vulnerable plan generation, and a 95% reduction in the attack window. When combining all three approaches, we reduce the TOCTOU vulnerabilities from an executed trajectory from 12% to 8%. Our findings open a new research direction at the intersection of AI safety and systems security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06724v3">AutoDCWorkflow: LLM-based Data Cleaning Workflow Auto-Generation and Benchmark</a></div>
    <div class="paper-meta">
      📅 2025-08-23
      | 💬 EMNLP Findings, 2025
    </div>
    <details class="paper-abstract">
      Data cleaning is a time-consuming and error-prone manual process, even with modern workflow tools such as OpenRefine. We present AutoDCWorkflow, an LLM-based pipeline for automatically generating data-cleaning workflows. The pipeline takes a raw table and a data analysis purpose, and generates a sequence of OpenRefine operations designed to produce a minimal, clean table sufficient to address the purpose. Six operations correspond to common data quality issues, including format inconsistencies, type errors, and duplicates. To evaluate AutoDCWorkflow, we create a benchmark with metrics assessing answers, data, and workflow quality for 142 purposes using 96 tables across six topics. The evaluation covers three key dimensions: (1) Purpose Answer: can the cleaned table produce a correct answer? (2) Column (Value): how closely does it match the ground truth table? (3) Workflow (Operations): to what extent does the generated workflow resemble the human-curated ground truth? Experiments show that Llama 3.1, Mistral, and Gemma 2 significantly enhance data quality, outperforming the baseline across all metrics. Gemma 2-27B consistently generates high-quality tables and answers, while Gemma 2-9B excels in producing workflows that closely resemble human-annotated versions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.12334v4">What Did I Do Wrong? Quantifying LLMs' Sensitivity and Consistency to Prompt Engineering</a></div>
    <div class="paper-meta">
      📅 2025-08-23
      | 💬 Proceedings of the Annual Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics (NAACL 2025)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) changed the way we design and interact with software systems. Their ability to process and extract information from text has drastically improved productivity in a number of routine tasks. Developers that want to include these models in their software stack, however, face a dreadful challenge: debugging LLMs' inconsistent behavior across minor variations of the prompt. We therefore introduce two metrics for classification tasks, namely sensitivity and consistency, which are complementary to task performance. First, sensitivity measures changes of predictions across rephrasings of the prompt, and does not require access to ground truth labels. Instead, consistency measures how predictions vary across rephrasings for elements of the same class. We perform an empirical comparison of these metrics on text classification tasks, using them as guideline for understanding failure modes of the LLM. Our hope is that sensitivity and consistency will be helpful to guide prompt engineering and obtain LLMs that balance robustness with performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13266v2">QuestA: Expanding Reasoning Capacity in LLMs via Question Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-08-23
      | 💬 19 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become a key component in training large language reasoning models (LLMs). However, recent studies questions its effectiveness in improving multi-step reasoning-particularly on hard problems. To address this challenge, we propose a simple yet effective strategy via Question Augmentation: introduce partial solutions during training to reduce problem difficulty and provide more informative learning signals. Our method, QuestA, when applied during RL training on math reasoning tasks, not only improves pass@1 but also pass@k-particularly on problems where standard RL struggles to make progress. This enables continual improvement over strong open-source models such as DeepScaleR and OpenMath Nemotron, further enhancing their reasoning capabilities. We achieve new state-of-the-art results on math benchmarks using 1.5B-parameter models: 67.1% (+5.3%) on AIME24, 59.5% (+10.0%) on AIME25, and 35.5% (+4.0%) on HMMT25. Further, we provide theoretical explanations that QuestA improves sample efficiency, offering a practical and generalizable pathway for expanding reasoning capability through RL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17063v1">Measuring Large Language Models Dependency: Validating the Arabic Version of the LLM-D12 Scale</a></div>
    <div class="paper-meta">
      📅 2025-08-23
    </div>
    <details class="paper-abstract">
      There is an urgent need for reliable, culturally validated instruments to assess psychological responses to AI in general and large language models (LLMs). This need is global issue, but it is especially urgent among Arabic-speaking populations, where AI and LLMs adoption is accelerating, yet psychometric tools remain limited. This study presents the first validation of the LLM-D12, a dual-dimensional scale assessing Instrumental and Relationship Dependency on LLMs, in an Arab sample. A total of 250 Arab participants completed the Arabic version of the LLM-D12. Confirmatory Factor Analysis confirms the original 2-factor structure of LLM-D12 with all items showing good loading of corresponding Instrumental and Relationship Dependency. The scale showed good to excellent internal reliability (Cronbach alpha is 0.90 for Total, 0.85 for Instrumental Dependency, and 0.90 for Relationship Dependency). External validation revealed that Instrumental Dependency was positively associated with AI acceptance and internet addiction, while Relationship Dependency was linked to lower need for cognition and greater trustworthiness of LLM, demonstrating sensitivity of this instrument to different use and personal factors. These findings confirm that Arabic LLM-D12 is a psychometrically sound, culturally appropriate instrument, offering a necessary tool for research, education, and policy concerning AI and LLMs engagement in Arab contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17028v1">Improving Table Understanding with LLMs and Entity-Oriented Search</a></div>
    <div class="paper-meta">
      📅 2025-08-23
      | 💬 Accepted to COLM 2025
    </div>
    <details class="paper-abstract">
      Our work addresses the challenges of understanding tables. Existing methods often struggle with the unpredictable nature of table content, leading to a reliance on preprocessing and keyword matching. They also face limitations due to the lack of contextual information, which complicates the reasoning processes of large language models (LLMs). To overcome these challenges, we introduce an entity-oriented search method to improve table understanding with LLMs. This approach effectively leverages the semantic similarities between questions and table data, as well as the implicit relationships between table cells, minimizing the need for data preprocessing and keyword matching. Additionally, it focuses on table entities, ensuring that table cells are semantically tightly bound, thereby enhancing contextual clarity. Furthermore, we pioneer the use of a graph query language for table understanding, establishing a new research direction. Experiments show that our approach achieves new state-of-the-art performances on standard benchmarks WikiTableQuestions and TabFact.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14273v2">Let's Use ChatGPT To Write Our Paper! Benchmarking LLMs To Write the Introduction of a Research Paper</a></div>
    <div class="paper-meta">
      📅 2025-08-23
      | 💬 20 pages, 15 figures
    </div>
    <details class="paper-abstract">
      As researchers increasingly adopt LLMs as writing assistants, generating high-quality research paper introductions remains both challenging and essential. We introduce Scientific Introduction Generation (SciIG), a task that evaluates LLMs' ability to produce coherent introductions from titles, abstracts, and related works. Curating new datasets from NAACL 2025 and ICLR 2025 papers, we assess five state-of-the-art models, including both open-source (DeepSeek-v3, Gemma-3-12B, LLaMA 4-Maverick, MistralAI Small 3.1) and closed-source GPT-4o systems, across multiple dimensions: lexical overlap, semantic similarity, content coverage, faithfulness, consistency, citation correctness, and narrative quality. Our comprehensive framework combines automated metrics with LLM-as-a-judge evaluations. Results demonstrate LLaMA-4 Maverick's superior performance on most metrics, particularly in semantic similarity and faithfulness. Moreover, three-shot prompting consistently outperforms fewer-shot approaches. These findings provide practical insights into developing effective research writing assistants and set realistic expectations for LLM-assisted academic writing. To foster reproducibility and future research, we will publicly release all code and datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17005v1">Planning for Success: Exploring LLM Long-term Planning Capabilities in Table Understanding</a></div>
    <div class="paper-meta">
      📅 2025-08-23
      | 💬 Accepted to CoNLL 2025
    </div>
    <details class="paper-abstract">
      Table understanding is key to addressing challenging downstream tasks such as table-based question answering and fact verification. Recent works have focused on leveraging Chain-of-Thought and question decomposition to solve complex questions requiring multiple operations on tables. However, these methods often suffer from a lack of explicit long-term planning and weak inter-step connections, leading to miss constraints within questions. In this paper, we propose leveraging the long-term planning capabilities of large language models (LLMs) to enhance table understanding. Our approach enables the execution of a long-term plan, where the steps are tightly interconnected and serve the ultimate goal, an aspect that methods based on Chain-of-Thought and question decomposition lack. In addition, our method effectively minimizes the inclusion of unnecessary details in the process of solving the next short-term goals, a limitation of methods based on Chain-of-Thought. Extensive experiments demonstrate that our method outperforms strong baselines and achieves state-of-the-art performance on WikiTableQuestions and TabFact datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09329v2">When Developer Aid Becomes Security Debt: A Systematic Analysis of Insecure Behaviors in LLM Coding Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-23
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      LLM-based coding agents are rapidly being deployed in software development, yet their safety implications remain poorly understood. These agents, while capable of accelerating software development, may exhibit unsafe behaviors during normal operation that manifest as cybersecurity vulnerabilities. We conducted the first systematic safety evaluation of autonomous coding agents, analyzing over 12,000 actions across five state-of-the-art models (GPT-4o, GPT-4.1, Claude variants) on 93 real-world software setup tasks. Our findings reveal significant security concerns: 21% of agent trajectories contained insecure actions, with models showing substantial variation in unsafe behavior. We developed a high-precision detection system that identified four major vulnerability categories, with information exposure (CWE-200) being the most prevalent one. We also evaluated mitigation strategies including feedback mechanisms and security reminders with various effectiveness between models. GPT-4.1 demonstrated exceptional security awareness with 96.8% mitigation success.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16998v1">DeAR: Dual-Stage Document Reranking with Reasoning Agents via LLM Distillation</a></div>
    <div class="paper-meta">
      📅 2025-08-23
      | 💬 Accept at EMNLP Findings 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have transformed listwise document reranking by enabling global reasoning over candidate sets, yet single models often struggle to balance fine-grained relevance scoring with holistic cross-document analysis. We propose \textbf{De}ep\textbf{A}gent\textbf{R}ank (\textbf{\DeAR}), an open-source framework that decouples these tasks through a dual-stage approach, achieving superior accuracy and interpretability. In \emph{Stage 1}, we distill token-level relevance signals from a frozen 13B LLaMA teacher into a compact \{3, 8\}B student model using a hybrid of cross-entropy, RankNet, and KL divergence losses, ensuring robust pointwise scoring. In \emph{Stage 2}, we attach a second LoRA adapter and fine-tune on 20K GPT-4o-generated chain-of-thought permutations, enabling listwise reasoning with natural-language justifications. Evaluated on TREC-DL19/20, eight BEIR datasets, and NovelEval-2306, \DeAR surpasses open-source baselines by +5.1 nDCG@5 on DL20 and achieves 90.97 nDCG@10 on NovelEval, outperforming GPT-4 by +3.09. Without fine-tuning on Wikipedia, DeAR also excels in open-domain QA, achieving 54.29 Top-1 accuracy on Natural Questions, surpassing baselines like MonoT5, UPR, and RankGPT. Ablations confirm that dual-loss distillation ensures stable calibration, making \DeAR a highly effective and interpretable solution for modern reranking systems.\footnote{Dataset and code available at https://github.com/DataScienceUIBK/DeAR-Reranking.}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11398v2">Trustworthy AI Psychotherapy: Multi-Agent LLM Workflow for Counseling and Explainable Mental Disorder Diagnosis</a></div>
    <div class="paper-meta">
      📅 2025-08-23
      | 💬 This paper has been accepted by CIKM 2025 as a full paper
    </div>
    <details class="paper-abstract">
      LLM-based agents have emerged as transformative tools capable of executing complex tasks through iterative planning and action, achieving significant advancements in understanding and addressing user needs. Yet, their effectiveness remains limited in specialized domains such as mental health diagnosis, where they underperform compared to general applications. Current approaches to integrating diagnostic capabilities into LLMs rely on scarce, highly sensitive mental health datasets, which are challenging to acquire. These methods also fail to emulate clinicians' proactive inquiry skills, lack multi-turn conversational comprehension, and struggle to align outputs with expert clinical reasoning. To address these gaps, we propose DSM5AgentFlow, the first LLM-based agent workflow designed to autonomously generate DSM-5 Level-1 diagnostic questionnaires. By simulating therapist-client dialogues with specific client profiles, the framework delivers transparent, step-by-step disorder predictions, producing explainable and trustworthy results. This workflow serves as a complementary tool for mental health diagnosis, ensuring adherence to ethical and legal standards. Through comprehensive experiments, we evaluate leading LLMs across three critical dimensions: conversational realism, diagnostic accuracy, and explainability. Our datasets and implementations are fully open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16982v1">Decoding Alignment: A Critical Survey of LLM Development Initiatives through Value-setting and Data-centric Lens</a></div>
    <div class="paper-meta">
      📅 2025-08-23
      | 💬 This is a working paper and will be updated with new information or corrections based on community feedback
    </div>
    <details class="paper-abstract">
      AI Alignment, primarily in the form of Reinforcement Learning from Human Feedback (RLHF), has been a cornerstone of the post-training phase in developing Large Language Models (LLMs). It has also been a popular research topic across various disciplines beyond Computer Science, including Philosophy and Law, among others, highlighting the socio-technical challenges involved. Nonetheless, except for the computational techniques related to alignment, there has been limited focus on the broader picture: the scope of these processes, which primarily rely on the selected objectives (values), and the data collected and used to imprint such objectives into the models. This work aims to reveal how alignment is understood and applied in practice from a value-setting and data-centric perspective. For this purpose, we investigate and survey (`audit') publicly available documentation released by 6 LLM development initiatives by 5 leading organizations shaping this technology, focusing on proprietary (OpenAI's GPT, Anthropic's Claude, Google's Gemini) and open-weight (Meta's Llama, Google's Gemma, and Alibaba's Qwen) initiatives, all published in the last 3 years. The findings are documented in detail per initiative, while there is also an overall summary concerning different aspects, mainly from a value-setting and data-centric perspective. On the basis of our findings, we discuss a series of broader related concerns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16962v1">LLM-based Human-like Traffic Simulation for Self-driving Tests</a></div>
    <div class="paper-meta">
      📅 2025-08-23
    </div>
    <details class="paper-abstract">
      Ensuring realistic traffic dynamics is a prerequisite for simulation platforms to evaluate the reliability of self-driving systems before deployment in the real world. Because most road users are human drivers, reproducing their diverse behaviors within simulators is vital. Existing solutions, however, typically rely on either handcrafted heuristics or narrow data-driven models, which capture only fragments of real driving behaviors and offer limited driving style diversity and interpretability. To address this gap, we introduce HDSim, an HD traffic generation framework that combines cognitive theory with large language model (LLM) assistance to produce scalable and realistic traffic scenarios within simulation platforms. The framework advances the state of the art in two ways: (i) it introduces a hierarchical driver model that represents diverse driving style traits, and (ii) it develops a Perception-Mediated Behavior Influence strategy, where LLMs guide perception to indirectly shape driver actions. Experiments reveal that embedding HDSim into simulation improves detection of safety-critical failures in self-driving systems by up to 68% and yields realism-consistent accident interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16949v1">Breaking the Exploration Bottleneck: Rubric-Scaffolded Reinforcement Learning for General LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-08-23
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have underscored the potential of Reinforcement Learning (RL) to facilitate the emergence of reasoning capabilities. Despite the encouraging results, a fundamental dilemma persists as RL improvement relies on learning from high-quality samples, yet the exploration for such samples remains bounded by the inherent limitations of LLMs. This, in effect, creates an undesirable cycle in which what cannot be explored cannot be learned. In this work, we propose Rubric-Scaffolded Reinforcement Learning (RuscaRL), a novel instructional scaffolding framework designed to break the exploration bottleneck for general LLM reasoning. Specifically, RuscaRL introduces checklist-style rubrics as (1) explicit scaffolding for exploration during rollout generation, where different rubrics are provided as external guidance within task instructions to steer diverse high-quality responses. This guidance is gradually decayed over time, encouraging the model to internalize the underlying reasoning patterns; (2) verifiable rewards for exploitation during model training, where we can obtain robust LLM-as-a-Judge scores using rubrics as references, enabling effective RL on general reasoning tasks. Extensive experiments demonstrate the superiority of the proposed RuscaRL across various benchmarks, effectively expanding reasoning boundaries under the best-of-N evaluation. Notably, RuscaRL significantly boosts Qwen-2.5-7B-Instruct from 23.6 to 50.3 on HealthBench-500, surpassing GPT-4.1. Furthermore, our fine-tuned variant on Qwen3-30B-A3B-Instruct achieves 61.1 on HealthBench-500, outperforming leading LLMs including OpenAI-o3.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16921v1">Being Kind Isn't Always Being Safe: Diagnosing Affective Hallucination in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-23
      | 💬 31 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in emotionally sensitive interactions, where their simulated empathy can create the illusion of genuine relational connection. We define this risk as Affective Hallucination, the production of emotionally immersive responses that foster illusory social presence despite the model's lack of affective capacity. To systematically diagnose and mitigate this risk, we introduce AHaBench, a benchmark of 500 mental health-related prompts with expert-informed reference responses, evaluated along three dimensions: Emotional Enmeshment, Illusion of Presence, and Fostering Overdependence. We further release AHaPairs, a 5K-instance preference dataset enabling Direct Preference Optimization (DPO) for alignment with emotionally responsible behavior. Experiments across multiple model families show that DPO fine-tuning substantially reduces affective hallucination without degrading core reasoning and knowledge performance. Human-model agreement analyses confirm that AHaBench reliably captures affective hallucination, validating it as an effective diagnostic tool. This work establishes affective hallucination as a distinct safety concern and provides practical resources for developing LLMs that are not only factually reliable but also psychologically safe. AHaBench and AHaPairs are accessible via https://huggingface.co/datasets/o0oMiNGo0o/AHaBench, and code for fine-tuning and evaluation are in https://github.com/0oOMiNGOo0/AHaBench. Warning: This paper contains examples of mental health-related language that may be emotionally distressing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12072v2">Mitigating Jailbreaks with Intent-Aware LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-23
    </div>
    <details class="paper-abstract">
      Despite extensive safety-tuning, large language models (LLMs) remain vulnerable to jailbreak attacks via adversarially crafted instructions, reflecting a persistent trade-off between safety and task performance. In this work, we propose Intent-FT, a simple and lightweight fine-tuning approach that explicitly trains LLMs to infer the underlying intent of an instruction before responding. By fine-tuning on a targeted set of adversarial instructions, Intent-FT enables LLMs to generalize intent deduction to unseen attacks, thereby substantially improving their robustness. We comprehensively evaluate both parametric and non-parametric attacks across open-source and proprietary models, considering harmfulness from attacks, utility, over-refusal, and impact against white-box threats. Empirically, Intent-FT consistently mitigates all evaluated attack categories, with no single attack exceeding a 50\% success rate -- whereas existing defenses remain only partially effective. Importantly, our method preserves the model's general capabilities and reduces excessive refusals on benign instructions containing superficially harmful keywords. Furthermore, models trained with Intent-FT accurately identify hidden harmful intent in adversarial attacks, and these learned intentions can be effectively transferred to enhance vanilla model defenses. We publicly release our code at https://github.com/wj210/Intent_Jailbreak.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14377v2">ZPD-SCA: Unveiling the Blind Spots of LLMs in Assessing Students' Cognitive Abilities</a></div>
    <div class="paper-meta">
      📅 2025-08-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated potential in educational applications, yet their capacity to accurately assess the cognitive alignment of reading materials with students' developmental stages remains insufficiently explored. This gap is particularly critical given the foundational educational principle of the Zone of Proximal Development (ZPD), which emphasizes the need to match learning resources with Students' Cognitive Abilities (SCA). Despite the importance of this alignment, there is a notable absence of comprehensive studies investigating LLMs' ability to evaluate reading comprehension difficulty across different student age groups, especially in the context of Chinese language education. To fill this gap, we introduce ZPD-SCA, a novel benchmark specifically designed to assess stage-level Chinese reading comprehension difficulty. The benchmark is annotated by 60 Special Grade teachers, a group that represents the top 0.15% of all in-service teachers nationwide. Experimental results reveal that LLMs perform poorly in zero-shot learning scenarios, with Qwen-max and GLM even falling below the probability of random guessing. When provided with in-context examples, LLMs performance improves substantially, with some models achieving nearly double the accuracy of their zero-shot baselines. These results reveal that LLMs possess emerging abilities to assess reading difficulty, while also exposing limitations in their current training for educationally aligned judgment. Notably, even the best-performing models display systematic directional biases, suggesting difficulties in accurately aligning material difficulty with SCA. Furthermore, significant variations in model performance across different genres underscore the complexity of task. We envision that ZPD-SCA can provide a foundation for evaluating and improving LLMs in cognitively aligned educational applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17054v2">Using LLM for Real-Time Transcription and Summarization of Doctor-Patient Interactions into ePuskesmas in Indonesia: A Proof-of-Concept Study</a></div>
    <div class="paper-meta">
      📅 2025-08-23
    </div>
    <details class="paper-abstract">
      One of the critical issues contributing to inefficiency in Puskesmas (Indonesian community health centers) is the time-consuming nature of documenting doctor-patient interactions. Doctors must conduct thorough consultations and manually transcribe detailed notes into ePuskesmas electronic health records (EHR), which creates substantial administrative burden to already overcapacitated physicians. This paper presents a proof-of-concept framework using large language models (LLMs) to automate real-time transcription and summarization of doctor-patient conversations in Bahasa Indonesia. Our system combines Whisper model for transcription with GPT-3.5 for medical summarization, implemented as a browser extension that automatically populates ePuskesmas forms. Through controlled roleplay experiments with medical validation, we demonstrate the technical feasibility of processing detailed 300+ seconds trimmed consultations in under 30 seconds while maintaining clinical accuracy. This work establishes the foundation for AI-assisted clinical documentation in resource-constrained healthcare environments. However, concerns have also been raised regarding privacy compliance and large-scale clinical evaluation addressing language and cultural biases for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16889v1">ObjexMT: Objective Extraction and Metacognitive Calibration for LLM-as-a-Judge under Multi-Turn Jailbreaks</a></div>
    <div class="paper-meta">
      📅 2025-08-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as judges of other models, yet it is unclear whether a judge can reliably infer the latent objective of the conversation it evaluates, especially when the goal is distributed across noisy, adversarial, multi-turn jailbreaks. We introduce OBJEX(MT), a benchmark that requires a model to (i) distill a transcript into a single-sentence base objective and (ii) report its own confidence. Accuracy is scored by an LLM judge using semantic similarity between extracted and gold objectives; correctness uses a single human-aligned threshold calibrated once on N=100 items (tau* = 0.61); and metacognition is evaluated with ECE, Brier score, Wrong@High-Conf, and risk-coverage curves. We evaluate gpt-4.1, claude-sonnet-4, and Qwen3-235B-A22B-FP8 on SafeMT Attack_600, SafeMTData_1K, MHJ, and CoSafe. claude-sonnet-4 attains the highest objective-extraction accuracy (0.515) and the best calibration (ECE 0.296; Brier 0.324), while gpt-4.1 and Qwen3 tie at 0.441 accuracy yet show marked overconfidence (mean confidence approx. 0.88 vs. accuracy approx. 0.44; Wrong@0.90 approx. 48-52%). Performance varies sharply across datasets (approx. 0.167-0.865), with MHJ comparatively easy and Attack_600/CoSafe harder. These results indicate that LLM judges often misinfer objectives with high confidence in multi-turn jailbreaks and suggest operational guidance: provide judges with explicit objectives when possible and use selective prediction or abstention to manage risk. We release prompts, scoring templates, and complete logs to facilitate replication and analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16873v1">Do Multimodal LLMs See Sentiment?</a></div>
    <div class="paper-meta">
      📅 2025-08-23
      | 💬 11 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Understanding how visual content communicates sentiment is critical in an era where online interaction is increasingly dominated by this kind of media on social platforms. However, this remains a challenging problem, as sentiment perception is closely tied to complex, scene-level semantics. In this paper, we propose an original framework, MLLMsent, to investigate the sentiment reasoning capabilities of Multimodal Large Language Models (MLLMs) through three perspectives: (1) using those MLLMs for direct sentiment classification from images; (2) associating them with pre-trained LLMs for sentiment analysis on automatically generated image descriptions; and (3) fine-tuning the LLMs on sentiment-labeled image descriptions. Experiments on a recent and established benchmark demonstrate that our proposal, particularly the fine-tuned approach, achieves state-of-the-art results outperforming Lexicon-, CNN-, and Transformer-based baselines by up to 30.9%, 64.8%, and 42.4%, respectively, across different levels of evaluators' agreement and sentiment polarity categories. Remarkably, in a cross-dataset test, without any training on these new data, our model still outperforms, by up to 8.26%, the best runner-up, which has been trained directly on them. These results highlight the potential of the proposed visual reasoning scheme for advancing affective computing, while also establishing new benchmarks for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14045v2">From Unaligned to Aligned: Scaling Multilingual LLMs with Multi-Way Parallel Corpora</a></div>
    <div class="paper-meta">
      📅 2025-08-23
    </div>
    <details class="paper-abstract">
      Continued pretraining and instruction tuning on large-scale multilingual data have proven to be effective in scaling large language models (LLMs) to low-resource languages. However, the unaligned nature of such data limits its ability to effectively capture cross-lingual semantics. In contrast, multi-way parallel data, where identical content is aligned across multiple languages, provides stronger cross-lingual consistency and offers greater potential for improving multilingual performance. In this paper, we introduce a large-scale, high-quality multi-way parallel corpus, TED2025, based on TED Talks. The corpus spans 113 languages, with up to 50 languages aligned in parallel, ensuring extensive multilingual coverage. Using this dataset, we investigate best practices for leveraging multi-way parallel data to enhance LLMs, including strategies for continued pretraining, instruction tuning, and the analysis of key influencing factors. Experiments on six multilingual benchmarks show that models trained on multiway parallel data consistently outperform those trained on unaligned multilingual data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16846v1">Quantifying Sycophancy as Deviations from Bayesian Rationality in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-23
    </div>
    <details class="paper-abstract">
      Sycophancy, or overly agreeable or flattering behavior, is a documented issue in large language models (LLMs), and is critical to understand in the context of human/AI collaboration. Prior works typically quantify sycophancy by measuring shifts in behavior or impacts on accuracy, but neither metric characterizes shifts in rationality, and accuracy measures can only be used in scenarios with a known ground truth. In this work, we utilize a Bayesian framework to quantify sycophancy as deviations from rational behavior when presented with user perspectives, thus distinguishing between rational and irrational updates based on the introduction of user perspectives. In comparison to other methods, this approach allows us to characterize excessive behavioral shifts, even for tasks that involve inherent uncertainty or do not have a ground truth. We study sycophancy for 3 different tasks, a combination of open-source and closed LLMs, and two different methods for probing sycophancy. We also experiment with multiple methods for eliciting probability judgments from LLMs. We hypothesize that probing LLMs for sycophancy will cause deviations in LLMs' predicted posteriors that will lead to increased Bayesian error. Our findings indicate that: 1) LLMs are not Bayesian rational, 2) probing for sycophancy results in significant increases to the predicted posterior in favor of the steered outcome, 3) sycophancy sometimes results in increased Bayesian error, and in a small number of cases actually decreases error, and 4) changes in Bayesian error due to sycophancy are not strongly correlated in Brier score, suggesting that studying the impact of sycophancy on ground truth alone does not fully capture errors in reasoning due to sycophancy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00998v2">Are LLM-Powered Social Media Bots Realistic?</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 Accepted into SBP-BRiMS 2025
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become more sophisticated, there is a possibility to harness LLMs to power social media bots. This work investigates the realism of generating LLM-Powered social media bot networks. Through a combination of manual effort, network science and LLMs, we create synthetic bot agent personas, their tweets and their interactions, thereby simulating social media networks. We compare the generated networks against empirical bot/human data, observing that both network and linguistic properties of LLM-Powered Bots differ from Wild Bots/Humans. This has implications towards the detection and effectiveness of LLM-Powered Bots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16571v1">LLM-Based Agents for Competitive Landscape Mapping in Drug Asset Due Diligence</a></div>
    <div class="paper-meta">
      📅 2025-08-22
    </div>
    <details class="paper-abstract">
      In this paper, we describe and benchmark a competitor-discovery component used within an agentic AI system for fast drug asset due diligence. A competitor-discovery AI agent, given an indication, retrieves all drugs comprising the competitive landscape of that indication and extracts canonical attributes for these drugs. The competitor definition is investor-specific, and data is paywalled/licensed, fragmented across registries, ontology-mismatched by indication, alias-heavy for drug names, multimodal, and rapidly changing. Although considered the best tool for this problem, the current LLM-based AI systems aren't capable of reliably retrieving all competing drug names, and there is no accepted public benchmark for this task. To address the lack of evaluation, we use LLM-based agents to transform five years of multi-modal, unstructured diligence memos from a private biotech VC fund into a structured evaluation corpus mapping indications to competitor drugs with normalized attributes. We also introduce a competitor validating LLM-as-a-judge agent that filters out false positives from the list of predicted competitors to maximize precision and suppress hallucinations. On this benchmark, our competitor-discovery agent achieves 83% recall, exceeding OpenAI Deep Research (65%) and Perplexity Labs (60%). The system is deployed in production with enterprise users; in a case study with a biotech VC investment fund, analyst turnaround time dropped from 2.5 days to $\sim$3 hours ($\sim$20x) for the competitive analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16546v1">RL Is Neither a Panacea Nor a Mirage: Understanding Supervised vs. Reinforcement Learning Fine-Tuning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-22
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) from scratch is increasingly impractical, making post-training methods such as supervised fine-tuning (SFT) and reinforcement-learning fine-tuning (RL-FT, e.g., PPO) central to modern practice. Using an out-of-distribution (OOD) variant of the 24-point card game and new spectrum-based diagnostics, we revisit how these two stages reshape model representation and OOD performance. Our key findings are- (1) RL-FT can restore much of the OOD performance loss from SFT (e.g., Llama-11B 8.97% to 15.38%, Qwen-7B 17.09% to 19.66%). But when SFT induces severe overfitting and a clear distribution shift, RL-FT cannot fully recover OOD performance. (2) Direction shifts of singular vectors matter more than singular value magnitudes. These shifts concentrate on directions linked to the largest and smallest singular values, leaving the bulk spectrum intact. (3) Low-rank and shallow recovery is effective: restoring singular vector directions for the top 20% of values or first 25% of layers recovers 70-80% of OOD performance. (4) Stronger SFT checkpoints enable better recovery by RL, while overfitted ones resist restoration. These results reconcile prior reports of RL superior OOD performance: RL primarily counteracts SFT-induced directional drift rather than finding new solutions. Our spectrum-aware analysis highlights inexpensive recovery knobs low-rank UV merging and shallow-layer resets that practitioners can use before costly RL fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11401v2">FACET: Teacher-Centred LLM-Based Multi-Agent Systems-Towards Personalized Educational Worksheets</a></div>
    <div class="paper-meta">
      📅 2025-08-22
    </div>
    <details class="paper-abstract">
      The increasing heterogeneity of student populations poses significant challenges for teachers, particularly in mathematics education, where cognitive, motivational, and emotional differences strongly influence learning outcomes. While AI-driven personalization tools have emerged, most remain performance-focused, offering limited support for teachers and neglecting broader pedagogical needs. This paper presents the FACET framework, a teacher-facing, large language model (LLM)-based multi-agent system designed to generate individualized classroom materials that integrate both cognitive and motivational dimensions of learner profiles. The framework comprises three specialized agents: (1) learner agents that simulate diverse profiles incorporating topic proficiency and intrinsic motivation, (2) a teacher agent that adapts instructional content according to didactical principles, and (3) an evaluator agent that provides automated quality assurance. We tested the system using authentic grade 8 mathematics curriculum content and evaluated its feasibility through a) automated agent-based assessment of output quality and b) exploratory feedback from K-12 in-service teachers. Results from ten internal evaluations highlighted high stability and alignment between generated materials and learner profiles, and teacher feedback particularly highlighted structure and suitability of tasks. The findings demonstrate the potential of multi-agent LLM architectures to provide scalable, context-aware personalization in heterogeneous classroom settings, and outline directions for extending the framework to richer learner profiles and real-world classroom trials.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16514v1">FLAMES: Improving LLM Math Reasoning via a Fine-Grained Analysis of the Data Synthesis Pipeline</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 To appear at EMNLP 2025
    </div>
    <details class="paper-abstract">
      Recent works improving LLM math reasoning with synthetic data have used unique setups, making comparison of data synthesis strategies impractical. This leaves many unanswered questions about the roles of different factors in the synthetic data pipeline, such as the impact of filtering low-quality problems. To address this gap, we introduce FLAMES, a Framework for LLM Assessment of Math rEasoning Data Synthesis, and perform a systematic study of 10 existing data synthesis strategies and multiple other factors impacting the performance of synthetic math reasoning data. Our FLAMES experiments provide several valuable insights about the optimal balance of difficulty and diversity of synthetic data. First, data agents designed to increase problem complexity lead to best improvements on most math metrics. Second, with a fixed data generation budget, keeping higher problem coverage is more important than keeping only problems with reliable solutions. Third, GSM8K- and MATH-based synthetic data can lead to improvements on competition-level benchmarks, showcasing easy-to-hard generalization. Leveraging insights from our FLAMES experiments, we design two novel data synthesis strategies for improving out-of-domain generalization and robustness. Further, we develop the FLAMES dataset, an effective blend of our novel and existing data synthesis strategies, outperforming public datasets on OlympiadBench (+15.7), CollegeMath (+4.5), GSMPlus (+6.5), and MATH (+3.1). Fine-tuning Qwen2.5-Math-7B on the FLAMES dataset achieves 81.4% on MATH, surpassing larger Llama3 405B, GPT-4o and Claude 3.5 Sonnet.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16837v1">LLMs Learn Constructions That Humans Do Not Know</a></div>
    <div class="paper-meta">
      📅 2025-08-22
    </div>
    <details class="paper-abstract">
      This paper investigates false positive constructions: grammatical structures which an LLM hallucinates as distinct constructions but which human introspection does not support. Both a behavioural probing task using contextual embeddings and a meta-linguistic probing task using prompts are included, allowing us to distinguish between implicit and explicit linguistic knowledge. Both methods reveal that models do indeed hallucinate constructions. We then simulate hypothesis testing to determine what would have happened if a linguist had falsely hypothesized that these hallucinated constructions do exist. The high accuracy obtained shows that such false hypotheses would have been overwhelmingly confirmed. This suggests that construction probing methods suffer from a confirmation bias and raises the issue of what unknown and incorrect syntactic knowledge these models also possess.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16785v1">Interpreting the Effects of Quantization on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-22
    </div>
    <details class="paper-abstract">
      Quantization offers a practical solution to deploy LLMs in resource-constraint environments. However, its impact on internal representations remains understudied, raising questions about the reliability of quantized models. In this study, we employ a range of interpretability techniques to investigate how quantization affects model and neuron behavior. We analyze multiple LLMs under 4-bit and 8-bit quantization. Our findings reveal that the impact of quantization on model calibration is generally minor. Analysis of neuron activations indicates that the number of dead neurons, i.e., those with activation values close to 0 across the dataset, remains consistent regardless of quantization. In terms of neuron contribution to predictions, we observe that smaller full precision models exhibit fewer salient neurons, whereas larger models tend to have more, with the exception of Llama-2-7B. The effect of quantization on neuron redundancy varies across models. Overall, our findings suggest that effect of quantization may vary by model and tasks, however, we did not observe any drastic change which may discourage the use of quantization as a reliable model compression technique.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16777v1">Evaluation and LLM-Guided Learning of ICD Coding Rationales</a></div>
    <div class="paper-meta">
      📅 2025-08-22
    </div>
    <details class="paper-abstract">
      Automated clinical coding involves mapping unstructured text from Electronic Health Records (EHRs) to standardized code systems such as the International Classification of Diseases (ICD). While recent advances in deep learning have significantly improved the accuracy and efficiency of ICD coding, the lack of explainability in these models remains a major limitation, undermining trust and transparency. Current explorations about explainability largely rely on attention-based techniques and qualitative assessments by physicians, yet lack systematic evaluation using consistent criteria on high-quality rationale datasets, as well as dedicated approaches explicitly trained to generate rationales for further enhancing explanation. In this work, we conduct a comprehensive evaluation of the explainability of the rationales for ICD coding through two key lenses: faithfulness that evaluates how well explanations reflect the model's actual reasoning and plausibility that measures how consistent the explanations are with human expert judgment. To facilitate the evaluation of plausibility, we construct a new rationale-annotated dataset, offering denser annotations with diverse granularity and aligns better with current clinical practice, and conduct evaluation across three types of rationales of ICD coding. Encouraged by the promising plausibility of LLM-generated rationales for ICD coding, we further propose new rationale learning methods to improve the quality of model-generated rationales, where rationales produced by prompting LLMs with/without annotation examples are used as distant supervision signals. We empirically find that LLM-generated rationales align most closely with those of human experts. Moreover, incorporating few-shot human-annotated examples not only further improves rationale generation but also enhances rationale-learning approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08525v4">Task Memory Engine (TME): Enhancing State Awareness for Multi-Step LLM Agent Tasks</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 14 pages, 5 figures. Preprint prepared for future submission. Includes implementation and token-efficiency analysis. Code at https://github.com/biubiutomato/TME-Agent
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used as autonomous agents for multi-step tasks. However, most existing frameworks fail to maintain a structured understanding of the task state, often relying on linear prompt concatenation or shallow memory buffers. This leads to brittle performance, frequent hallucinations, and poor long-range coherence. In this work, we propose the Task Memory Engine (TME), a lightweight and structured memory module that tracks task execution using a hierarchical Task Memory Tree (TMT). Each node in the tree corresponds to a task step, storing relevant input, output, status, and sub-task relationships. We introduce a prompt synthesis method that dynamically generates LLM prompts based on the active node path, significantly improving execution consistency and contextual grounding. Through case studies and comparative experiments on multi-step agent tasks, we demonstrate that TME leads to better task completion accuracy and more interpretable behavior with minimal implementation overhead. A reference implementation of the core TME components is available at https://github.com/biubiutomato/TME-Agent, including basic examples and structured memory integration. While the current implementation uses a tree-based structure, TME is designed to be graph-aware, supporting reusable substeps, converging task paths, and shared dependencies. This lays the groundwork for future DAG-based memory architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16757v1">How Good are LLM-based Rerankers? An Empirical Analysis of State-of-the-Art Reranking Models</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 EMNLP Findings 2025
    </div>
    <details class="paper-abstract">
      In this work, we present a systematic and comprehensive empirical evaluation of state-of-the-art reranking methods, encompassing large language model (LLM)-based, lightweight contextual, and zero-shot approaches, with respect to their performance in information retrieval tasks. We evaluate in total 22 methods, including 40 variants (depending on used LLM) across several established benchmarks, including TREC DL19, DL20, and BEIR, as well as a novel dataset designed to test queries unseen by pretrained models. Our primary goal is to determine, through controlled and fair comparisons, whether a performance disparity exists between LLM-based rerankers and their lightweight counterparts, particularly on novel queries, and to elucidate the underlying causes of any observed differences. To disentangle confounding factors, we analyze the effects of training data overlap, model architecture, and computational efficiency on reranking performance. Our findings indicate that while LLM-based rerankers demonstrate superior performance on familiar queries, their generalization ability to novel queries varies, with lightweight models offering comparable efficiency. We further identify that the novelty of queries significantly impacts reranking effectiveness, highlighting limitations in existing approaches. https://github.com/DataScienceUIBK/llm-reranking-generalization-study
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14879v2">MeshCoder: LLM-Powered Structured Mesh Code Generation from Point Clouds</a></div>
    <div class="paper-meta">
      📅 2025-08-22
    </div>
    <details class="paper-abstract">
      Reconstructing 3D objects into editable programs is pivotal for applications like reverse engineering and shape editing. However, existing methods often rely on limited domain-specific languages (DSLs) and small-scale datasets, restricting their ability to model complex geometries and structures. To address these challenges, we introduce MeshCoder, a novel framework that reconstructs complex 3D objects from point clouds into editable Blender Python scripts. We develop a comprehensive set of expressive Blender Python APIs capable of synthesizing intricate geometries. Leveraging these APIs, we construct a large-scale paired object-code dataset, where the code for each object is decomposed into distinct semantic parts. Subsequently, we train a multimodal large language model (LLM) that translates 3D point cloud into executable Blender Python scripts. Our approach not only achieves superior performance in shape-to-code reconstruction tasks but also facilitates intuitive geometric and topological editing through convenient code modifications. Furthermore, our code-based representation enhances the reasoning capabilities of LLMs in 3D shape understanding tasks. Together, these contributions establish MeshCoder as a powerful and flexible solution for programmatic 3D shape reconstruction and understanding. The project homepage is available at \href{https://daibingquan.github.io/MeshCoder}{this link}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.10848v2">Psyche-R1: Towards Reliable Psychological LLMs through Unified Empathy, Expertise, and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-08-22
    </div>
    <details class="paper-abstract">
      Amidst a shortage of qualified mental health professionals, the integration of large language models (LLMs) into psychological applications offers a promising way to alleviate the growing burden of mental health disorders. Recent reasoning-augmented LLMs have achieved remarkable performance in mathematics and programming, while research in the psychological domain has predominantly emphasized emotional support and empathetic dialogue, with limited attention to reasoning mechanisms that are beneficial to generating reliable responses. Therefore, in this paper, we propose Psyche-R1, the first Chinese psychological LLM that jointly integrates empathy, psychological expertise, and reasoning, built upon a novel data curation pipeline. Specifically, we design a comprehensive data synthesis pipeline that produces over 75k high-quality psychological questions paired with detailed rationales, generated through chain-of-thought (CoT) reasoning and iterative prompt-rationale optimization, along with 73k empathetic dialogues. Subsequently, we employ a hybrid training strategy wherein challenging samples are identified through a multi-LLM cross-selection strategy for group relative policy optimization (GRPO) to improve reasoning ability, while the remaining data is used for supervised fine-tuning (SFT) to enhance empathetic response generation and psychological domain knowledge. Extensive experiment results demonstrate the effectiveness of the Psyche-R1 across several psychological benchmarks, where our 7B Psyche-R1 achieves comparable results to 671B DeepSeek-R1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16478v1">LLM-as-classifier: Semi-Supervised, Iterative Framework for Hierarchical Text Classification using Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 20 pages excluding reference list, 2 figures
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) has provided unprecedented capabilities for analyzing unstructured text data. However, deploying these models as reliable, robust, and scalable classifiers in production environments presents significant methodological challenges. Standard fine-tuning approaches can be resource-intensive and often struggle with the dynamic nature of real-world data distributions, which is common in the industry. In this paper, we propose a comprehensive, semi-supervised framework that leverages the zero- and few-shot capabilities of LLMs for building hierarchical text classifiers as a framework for a solution to these industry-wide challenges. Our methodology emphasizes an iterative, human-in-the-loop process that begins with domain knowledge elicitation and progresses through prompt refinement, hierarchical expansion, and multi-faceted validation. We introduce techniques for assessing and mitigating sequence-based biases and outline a protocol for continuous monitoring and adaptation. This framework is designed to bridge the gap between the raw power of LLMs and the practical need for accurate, interpretable, and maintainable classification systems in industry applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16456v1">A Probabilistic Inference Scaling Theory for LLM Self-Correction</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 EMNLP 2025 Main
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated the capability to refine their generated answers through self-correction, enabling continuous performance improvement over multiple rounds. However, the mechanisms underlying how and why accuracy evolves during this iterative process remain unexplored. To fill this gap, we propose a probabilistic theory to model the dynamics of accuracy change and explain the performance improvements observed in multi-round self-correction. Through mathematical derivation, we establish that the accuracy after the $t^{th}$ round of self-correction is given by: $Acc_t = Upp - \alpha^t(Upp - Acc_0),$ where $Acc_0$ denotes the initial accuracy, $Upp$ represents the upper bound of accuracy convergence, and $\alpha$ determines the rate of convergence. Based on our theory, these parameters can be calculated and the predicted accuracy curve then can be obtained through only a single round of self-correction. Extensive experiments across diverse models and datasets demonstrate that our theoretical predictions align closely with empirical accuracy curves, validating the effectiveness of the theory. Our work provides a theoretical foundation for understanding LLM self-correction, thus paving the way for further explorations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16449v1">GreenLLM: SLO-Aware Dynamic Frequency Scaling for Energy-Efficient LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-08-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are becoming the backbone of modern cloud services, yet their inference costs are dominated by GPU energy. Unlike traditional GPU workloads, LLM inference has two stages with different characteristics: the prefill phase, which is latency sensitive and scales quadratically with prompt length, and the decode phase, which progresses token by token with unpredictable length. Current GPU power governors (for example, NVIDIA's default) overlook this asymmetry and treat both stages uniformly. The result is mismatched voltage and frequency settings, head-of-line blocking, and excessive energy use. We introduce GreenLLM, an SLO-aware serving framework that minimizes GPU energy by explicitly separating prefill and decode control. At ingress, requests are routed into length-based queues so short prompts avoid head-of-line blocking and TTFT improves. For prefill, GreenLLM collects short traces on a GPU node, fits compact latency-power models over SM frequency, and solves a queueing-aware optimization to select energy-minimal clocks per class. During decode, a lightweight dual-loop controller tracks throughput (tokens per second) and adjusts frequency with hysteretic, fine-grained steps to hold tail TBT within target bounds. Across Alibaba and Azure trace replays, GreenLLM reduces total energy by up to 34 percent versus the default DVFS baseline, with no loss of throughput and with less than 3.5 percent additional SLO violations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16447v1">Boardwalk: Towards a Framework for Creating Board Games with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 Accepted at SBGames 2025
    </div>
    <details class="paper-abstract">
      Implementing board games in code can be a time-consuming task. However, Large Language Models (LLMs) have been proven effective at generating code for domain-specific tasks with simple contextual information. We aim to investigate whether LLMs can implement digital versions of board games from rules described in natural language. This would be a step towards an LLM-assisted framework for quick board game code generation. We expect to determine the main challenges for LLMs to implement the board games, and how different approaches and models compare to one another. We task three state-of-the-art LLMs (Claude, DeepSeek and ChatGPT) with coding a selection of 12 popular and obscure games in free-form and within Boardwalk, our proposed General Game Playing API. We anonymize the games and components to avoid evoking pre-trained LLM knowledge. The implementations are tested for playability and rule compliance. We evaluate success rate and common errors across LLMs and game popularity. Our approach proves viable, with the best performing model, Claude 3.7 Sonnet, yielding 55.6\% of games without any errors. While compliance with the API increases error frequency, the severity of errors is more significantly dependent on the LLM. We outline future steps for creating a framework to integrate this process, making the elaboration of board games more accessible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16445v1">Using LLMs and Essence to Support Software Practice Adoption</a></div>
    <div class="paper-meta">
      📅 2025-08-22
    </div>
    <details class="paper-abstract">
      Recent advancements in natural language processing (NLP) have enabled the development of automated tools that support various domains, including software engineering. However, while NLP and artificial intelligence (AI) research has extensively focused on tasks such as code generation, less attention has been given to automating support for the adoption of best practices, the evolution of ways of working, and the monitoring of process health. This study addresses this gap by exploring the integration of Essence, a standard and thinking framework for managing software engineering practices, with large language models (LLMs). To this end, a specialised chatbot was developed to assist students and professionals in understanding and applying Essence. The chatbot employs a retrieval-augmented generation (RAG) system to retrieve relevant contextual information from a curated knowledge base. Four different LLMs were used to create multiple chatbot configurations, each evaluated both as a base model and augmented with the RAG system. The system performance was evaluated through both the relevance of retrieved context and the quality of generated responses. Comparative analysis against the general-purpose LLMs demonstrated that the proposed system consistently outperforms its baseline counterpart in domain-specific tasks. By facilitating access to structured software engineering knowledge, this work contributes to bridging the gap between theoretical frameworks and practical application, potentially improving process management and the adoption of software development practices. While further validation through user studies is required, these findings highlight the potential of LLM-based automation to enhance learning and decision-making in software engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16712v1">Systematic Characterization of LLM Quantization: A Performance, Energy, and Quality Perspective</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 14 pages, 10 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities across diverse domains, but their heavy resource demands make quantization-reducing precision to lower-bit formats-critical for efficient serving. While many quantization methods exist, a systematic understanding of their performance, energy, and quality tradeoffs in realistic serving conditions remains a gap. In this work, we first develop a fully automated online characterization framework qMeter, and then conduct an in-depth characterization of 11 post-training LLM quantization methods across 4 model sizes (7B-70B) and two GPU architectures (A100, H100). We evaluate quantization at the application, workload, parallelism, and hardware levels under online serving conditions. Our study reveals highly task- and method-dependent tradeoffs, strong sensitivity to workload characteristics, and complex interactions with parallelism and GPU architecture. We further present three optimization case studies illustrating deployment challenges in capacity planning, energy-efficient scheduling, and multi-objective tuning. To the best of our knowledge, this is one of the first comprehensive application-, system-, and hardware-level characterization of LLM quantization from a joint performance, energy, and quality perspective.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16431v1">Cetvel: A Unified Benchmark for Evaluating Language Understanding, Generation and Cultural Capacity of LLMs for Turkish</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 31 pages, 2 figures, 10 tables
    </div>
    <details class="paper-abstract">
      We introduce Cetvel, a comprehensive benchmark designed to evaluate large language models (LLMs) in Turkish. Existing Turkish benchmarks often lack either task diversity or culturally relevant content, or both. Cetvel addresses these gaps by combining a broad range of both discriminative and generative tasks ensuring content that reflects the linguistic and cultural richness of Turkish language. Cetvel covers 23 tasks grouped into seven categories, including tasks such as grammatical error correction, machine translation, and question answering rooted in Turkish history and idiomatic language. We evaluate 33 open-weight LLMs (up to 70B parameters) covering different model families and instruction paradigms. Our experiments reveal that Turkish-centric instruction-tuned models generally underperform relative to multilingual or general-purpose models (e.g. Llama 3 and Mistral), despite being tailored for the language. Moreover, we show that tasks such as grammatical error correction and extractive question answering are particularly discriminative in differentiating model capabilities. Cetvel offers a comprehensive and culturally grounded evaluation suite for advancing the development and assessment of LLMs in Turkish.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16402v1">AetherCode: Evaluating LLMs' Ability to Win In Premier Programming Competitions</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      Competitive programming has emerged as a critical benchmark for evaluating the reasoning and coding capabilities of Large Language Models (LLMs). Despite impressive progress on existing benchmarks, we argue that current evaluations overstate model proficiency, masking a substantial gap between LLMs and elite human programmers. This gap arises from two key limitations: insufficient difficulty and scope of benchmark problems, and evaluation bias from low-quality test cases. To address these shortcomings, we present AetherCode, a new benchmark that draws problems from premier programming competitions such as IOI and ICPC, offering broader coverage and higher difficulty. AetherCode further incorporates comprehensive, expert-validated test suites built through a hybrid of automated generation and human curation, ensuring rigorous and reliable assessment. By combining challenging problem design with robust evaluation, AetherCode provides a more faithful measure of LLM capabilities and sets a new standard for future research in code reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05220v3">Leveraging LLMs for Utility-Focused Annotation: Reducing Manual Effort for Retrieval and RAG</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 Accepted by the EMNLP25 main conference
    </div>
    <details class="paper-abstract">
      Retrieval models typically rely on costly human-labeled query-document relevance annotations for training and evaluation. To reduce this cost and leverage the potential of Large Language Models (LLMs) in relevance judgments, we aim to explore whether LLM-generated annotations can effectively replace human annotations in training retrieval models. Retrieval usually emphasizes relevance, which indicates "topic-relatedness" of a document to a query, while in RAG, the value of a document (or utility) depends on how it contributes to answer generation. Recognizing this mismatch, some researchers use LLM performance on downstream tasks with documents as labels, but this approach requires manual answers for specific tasks, leading to high costs and limited generalization. In another line of work, prompting LLMs to select useful documents as RAG references eliminates the need for human annotation and is not task-specific. If we leverage LLMs' utility judgments to annotate retrieval data, we may retain cross-task generalization without human annotation in large-scale corpora. Therefore, we investigate utility-focused annotation via LLMs for large-scale retriever training data across both in-domain and out-of-domain settings on the retrieval and RAG tasks. To reduce the impact of low-quality positives labeled by LLMs, we design a novel loss function, i.e., Disj-InfoNCE. Our experiments reveal that: (1) Retrievers trained on utility-focused annotations significantly outperform those trained on human annotations in the out-of-domain setting on both tasks, demonstrating superior generalization capabilities. (2) LLM annotation does not replace human annotation in the in-domain setting. However, incorporating just 20% human-annotated data enables retrievers trained with utility-focused annotations to match the performance of models trained entirely with human annotations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16706v1">RoboBuddy in the Classroom: Exploring LLM-Powered Social Robots for Storytelling in Learning and Integration Activities</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 Accepted to be published in the proceedings of 34th IEEE International Conference on Robot and Human Interactive Communication (RO-MAN) in 2025
    </div>
    <details class="paper-abstract">
      Creating and improvising scenarios for content approaching is an enriching technique in education. However, it comes with a significant increase in the time spent on its planning, which intensifies when using complex technologies, such as social robots. Furthermore, addressing multicultural integration is commonly embedded in regular activities due to the already tight curriculum. Addressing these issues with a single solution, we implemented an intuitive interface that allows teachers to create scenario-based activities from their regular curriculum using LLMs and social robots. We co-designed different frameworks of activities with 4 teachers and deployed it in a study with 27 students for 1 week. Beyond validating the system's efficacy, our findings highlight the positive impact of integration policies perceived by the children and demonstrate the importance of scenario-based activities in students' enjoyment, observed to be significantly higher when applying storytelling. Additionally, several implications of using LLMs and social robots in long-term classroom activities are discussed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16347v1">Confusion is the Final Barrier: Rethinking Jailbreak Evaluation and Investigating the Real Misuse Threat of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-22
    </div>
    <details class="paper-abstract">
      With the development of Large Language Models (LLMs), numerous efforts have revealed their vulnerabilities to jailbreak attacks. Although these studies have driven the progress in LLMs' safety alignment, it remains unclear whether LLMs have internalized authentic knowledge to deal with real-world crimes, or are merely forced to simulate toxic language patterns. This ambiguity raises concerns that jailbreak success is often attributable to a hallucination loop between jailbroken LLM and judger LLM. By decoupling the use of jailbreak techniques, we construct knowledge-intensive Q\&A to investigate the misuse threats of LLMs in terms of dangerous knowledge possession, harmful task planning utility, and harmfulness judgment robustness. Experiments reveal a mismatch between jailbreak success rates and harmful knowledge possession in LLMs, and existing LLM-as-a-judge frameworks tend to anchor harmfulness judgments on toxic language patterns. Our study reveals a gap between existing LLM safety assessments and real-world threat potential.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14540v2">VERUS-LM: a Versatile Framework for Combining LLMs with Symbolic Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 Accepted at ICLP 2025, part of ECPTS
    </div>
    <details class="paper-abstract">
      A recent approach to neurosymbolic reasoning is to explicitly combine the strengths of large language models (LLMs) and symbolic solvers to tackle complex reasoning tasks. However, current approaches face significant limitations, including poor generalizability due to task-specific prompts, inefficiencies caused by the lack of separation between knowledge and queries, and restricted inferential capabilities. These shortcomings hinder their scalability and applicability across diverse domains. In this paper, we introduce VERUS-LM, a novel framework designed to address these challenges. VERUS-LM employs a generic prompting mechanism, clearly separates domain knowledge from queries, and supports a wide range of different logical reasoning tasks. This framework enhances adaptability, reduces computational cost, and allows for richer forms of reasoning, such as optimization and constraint satisfaction. We show that our approach succeeds in diverse reasoning on a novel dataset, markedly outperforming LLMs. Additionally, our system achieves competitive results on common reasoning benchmarks when compared to other state-of-the-art approaches, and significantly surpasses them on the difficult AR-LSAT dataset. By pushing the boundaries of hybrid reasoning, VERUS-LM represents a significant step towards more versatile neurosymbolic AI systems
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13824v2">Can Hallucinations Help? Boosting LLMs for Drug Discovery</a></div>
    <div class="paper-meta">
      📅 2025-08-22
    </div>
    <details class="paper-abstract">
      Hallucinations in large language models (LLMs), plausible but factually inaccurate text, are often viewed as undesirable. However, recent work suggests that such outputs may hold creative potential. In this paper, we investigate whether hallucinations can improve LLMs on molecule property prediction, a key task in early-stage drug discovery. We prompt LLMs to generate natural language descriptions from molecular SMILES strings and incorporate these often hallucinated descriptions into downstream classification tasks. Evaluating seven instruction-tuned LLMs across five datasets, we find that hallucinations significantly improve predictive accuracy for some models. Notably, Falcon3-Mamba-7B outperforms all baselines when hallucinated text is included, while hallucinations generated by GPT-4o consistently yield the greatest gains between models. We further identify and categorize over 18,000 beneficial hallucinations, with structural misdescriptions emerging as the most impactful type, suggesting that hallucinated statements about molecular structure may increase model confidence. Ablation studies show that larger models benefit more from hallucinations, while temperature has a limited effect. Our findings challenge conventional views of hallucination as purely problematic and suggest new directions for leveraging hallucinations as a useful signal in scientific modeling tasks like drug discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14913v2">Bridging the Culture Gap: A Framework for LLM-Driven Socio-Cultural Localization of Math Word Problems in Low-Resource Languages</a></div>
    <div class="paper-meta">
      📅 2025-08-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated significant capabilities in solving mathematical problems expressed in natural language. However, multilingual and culturally-grounded mathematical reasoning in low-resource languages lags behind English due to the scarcity of socio-cultural task datasets that reflect accurate native entities such as person names, organization names, and currencies. Existing multilingual benchmarks are predominantly produced via translation and typically retain English-centric entities, owing to the high cost associated with human annotater-based localization. Moreover, automated localization tools are limited, and hence, truly localized datasets remain scarce. To bridge this gap, we introduce a framework for LLM-driven cultural localization of math word problems that automatically constructs datasets with native names, organizations, and currencies from existing sources. We find that translated benchmarks can obscure true multilingual math ability under appropriate socio-cultural contexts. Through extensive experiments, we also show that our framework can help mitigate English-centric entity bias and improves robustness when native entities are introduced across various languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00719v2">Dynamically Adaptive Reasoning via LLM-Guided MCTS for Efficient and Context-Aware KGQA</a></div>
    <div class="paper-meta">
      📅 2025-08-22
    </div>
    <details class="paper-abstract">
      Knowledge Graph Question Answering (KGQA) aims to interpret natural language queries and perform structured reasoning over knowledge graphs by leveraging their relational and semantic structures to retrieve accurate answers. Recent KGQA methods primarily follow either retrieve-then-reason paradigm, relying on GNNs or heuristic rules for static paths extraction, or dynamic path generation strategies that use large language models (LLMs) with prompting to jointly perform retrieval and reasoning. However, the former suffers from limited adaptability due to static path extraction and lack of contextual refinement, while the latter incurs high computational costs and struggles with accurate path evaluation due to reliance on fixed scoring functions and extensive LLM calls. To address these issues, this paper proposes Dynamically Adaptive MCTS-based Reasoning (DAMR), a novel framework that integrates symbolic search with adaptive path evaluation for efficient and context-aware KGQA. DAMR employs a Monte Carlo Tree Search (MCTS) backbone guided by an LLM-based planner, which selects top-$k$ relevant relations at each step to reduce search space. To improve path evaluation accuracy, we introduce a lightweight Transformer-based scorer that performs context-aware plausibility estimation by jointly encoding the question and relation sequence through cross-attention, enabling the model to capture fine-grained semantic shifts during multi-hop reasoning. Furthermore, to alleviate the scarcity of high-quality supervision, DAMR incorporates a dynamic pseudo-path refinement mechanism that periodically generates training signals from partial paths explored during search, allowing the scorer to continuously adapt to the evolving distribution of reasoning trajectories. Extensive experiments on multiple KGQA benchmarks show that DAMR significantly outperforms state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16270v1">LLMs that Understand Processes: Instruction-tuning for Semantics-Aware Process Mining</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 Accepted at IEEE ICPM 2025, 8 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Process mining is increasingly using textual information associated with events to tackle tasks such as anomaly detection and process discovery. Such semantics-aware process mining focuses on what behavior should be possible in a process (i.e., expectations), thus providing an important complement to traditional, frequency-based techniques that focus on recorded behavior (i.e., reality). Large Language Models (LLMs) provide a powerful means for tackling semantics-aware tasks. However, the best performance is so far achieved through task-specific fine-tuning, which is computationally intensive and results in models that can only handle one specific task. To overcome this lack of generalization, we use this paper to investigate the potential of instruction-tuning for semantics-aware process mining. The idea of instruction-tuning here is to expose an LLM to prompt-answer pairs for different tasks, e.g., anomaly detection and next-activity prediction, making it more familiar with process mining, thus allowing it to also perform better at unseen tasks, such as process discovery. Our findings demonstrate a varied impact of instruction-tuning: while performance considerably improved on process discovery and prediction tasks, it varies across models on anomaly detection tasks, highlighting that the selection of tasks for instruction-tuning is critical to achieving desired outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16267v1">From Confidence to Collapse in LLM Factual Robustness</a></div>
    <div class="paper-meta">
      📅 2025-08-22
    </div>
    <details class="paper-abstract">
      Ensuring the robustness of factual knowledge in LLMs is critical for reliable applications in tasks such as question answering and reasoning. However, existing evaluation methods predominantly focus on performance-based metrics, often investigating from the perspective of prompt perturbations, which captures only the externally triggered side of knowledge robustness. To bridge this gap, we introduce a principled approach to measure factual robustness from the perspective of the generation process by analyzing token distribution entropy in combination with temperature scaling sensitivity. These two factors build the Factual Robustness Score (FRS), a novel metric which quantifies the stability of a fact against perturbations in decoding conditions, given its initial uncertainty. To validate our approach, we conduct extensive experiments on 5 LLMs across 3 closed-book QA datasets (SQuAD, TriviaQA, and HotpotQA). We show that factual robustness varies significantly -- smaller models report an FRS of $0.76$, larger ones $0.93$ -- with accuracy degrading by ~$60\%$ under increased uncertainty. These insights demonstrate how entropy and temperature scaling impact factual accuracy, and lay a foundation for developing more robust knowledge retention and retrieval in future models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20807v2">GPU Kernel Scientist: An LLM-Driven Framework for Iterative Kernel Optimization</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 4+1 page paper plus Appendices and Supplementary zip file. Presented at the ES-FoMo "Efficient Systems for Foundation Models" workshop at ICML 2025
    </div>
    <details class="paper-abstract">
      Optimizing GPU kernels for high performance is a complex task, often demanding deep architectural knowledge, extensive profiling, and iterative experimentation. This challenge is amplified when targeting newer or less-documented GPU architectures where traditional development aids are scarce. This paper introduces an LLM-powered "GPU Kernel Scientist," an automated methodology for iteratively refining accelerator kernels. Our methodology employs LLMs in a multi-stage, evolutionary process: (a) strategically selecting promising prior code versions as a basis for new iterations; (b) generating hypotheses for optimization experiments, based on existing code and assimilated knowledge from general GPU literature; and (c) autonomously implementing these experiments through code modification and subsequent submission to an external evaluation system, using only observed timing data as performance feedback. We detail how this approach navigates the challenges of the AMD MI300 target architecture and leverages LLMs to compensate for limited domain-specific human expertise. In addition to our results, we present the architectural design, operational workflow, and qualitative insights, highlighting the potential of LLM-driven agents to democratise and accelerate GPU kernel optimization, especially in resource-constrained or rapidly updating hardware environment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16213v1">MedOmni-45°: A Safety-Performance Benchmark for Reasoning-Oriented LLMs in Medicine</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      With the increasing use of large language models (LLMs) in medical decision-support, it is essential to evaluate not only their final answers but also the reliability of their reasoning. Two key risks are Chain-of-Thought (CoT) faithfulness -- whether reasoning aligns with responses and medical facts -- and sycophancy, where models follow misleading cues over correctness. Existing benchmarks often collapse such vulnerabilities into single accuracy scores. To address this, we introduce MedOmni-45 Degrees, a benchmark and workflow designed to quantify safety-performance trade-offs under manipulative hint conditions. It contains 1,804 reasoning-focused medical questions across six specialties and three task types, including 500 from MedMCQA. Each question is paired with seven manipulative hint types and a no-hint baseline, producing about 27K inputs. We evaluate seven LLMs spanning open- vs. closed-source, general-purpose vs. medical, and base vs. reasoning-enhanced models, totaling over 189K inferences. Three metrics -- Accuracy, CoT-Faithfulness, and Anti-Sycophancy -- are combined into a composite score visualized with a 45 Degrees plot. Results show a consistent safety-performance trade-off, with no model surpassing the diagonal. The open-source QwQ-32B performs closest (43.81 Degrees), balancing safety and accuracy but not leading in both. MedOmni-45 Degrees thus provides a focused benchmark for exposing reasoning vulnerabilities in medical LLMs and guiding safer model development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16201v1">SpecVLM: Enhancing Speculative Decoding of Video LLMs via Verifier-Guided Token Pruning</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 Accepted at EMNLP 2025
    </div>
    <details class="paper-abstract">
      Video large language models (Vid-LLMs) have shown strong capabilities in understanding video content. However, their reliance on dense video token representations introduces substantial memory and computational overhead in both prefilling and decoding. To mitigate the information loss of recent video token reduction methods and accelerate the decoding stage of Vid-LLMs losslessly, we introduce SpecVLM, a training-free speculative decoding (SD) framework tailored for Vid-LLMs that incorporates staged video token pruning. Building on our novel finding that the draft model's speculation exhibits low sensitivity to video token pruning, SpecVLM prunes up to 90% of video tokens, enabling efficient speculation without sacrificing accuracy. To achieve this, it performs a two-stage pruning process: Stage I selects highly informative tokens guided by attention signals from the verifier (target model), while Stage II prunes remaining redundant ones in a spatially uniform manner. Extensive experiments on four video understanding benchmarks demonstrate the effectiveness and robustness of SpecVLM, which achieves up to 2.68$\times$ decoding speedup for LLaVA-OneVision-72B and 2.11$\times$ speedup for Qwen2.5-VL-32B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16185v1">ParamBench: A Graduate-Level Benchmark for Evaluating LLM Understanding on Indic Subjects</a></div>
    <div class="paper-meta">
      📅 2025-08-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been widely evaluated on tasks such as comprehension, question answering, summarization, code generation, etc. However, their performance on graduate-level, culturally grounded questions in the Indian context remains largely unexplored. Existing Indian benchmarks emphasise basic fact-orientated queries that offer limited assessment of a deeper disciplinary understanding tailored to the Indian setting. In this paper, we present ParamBench, consisting of around 11.5K questions in Hindi language comprising questionnaires from 16 diverse subjects. These questions are primarily derived from nation-wide graduate level entrance examination covering topics such as history, music, instruments, yoga, literature, philosophy, law, etc., specifically for the Indian context. Additionally, we assess the ability of LLMs to handle diverse question formats-such as list-based matching, assertion-reason pairs, and sequence ordering-alongside conventional multiple-choice questions. We evaluated the performance of more than 17 open source LLMs on this benchmark, observing that Llama 3.3 70B attains the highest overall accuracy of 48%. Furthermore, subject-wise analysis indicates that even for the best performing LLMs, performance remains weak on topics such as music, classical instruments, politics and archaeology, underscoring persistent challenges in culturally grounded reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10454v2">One Example Shown, Many Concepts Known! Counterexample-Driven Conceptual Reasoning in Mathematical LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 ICML 2025
    </div>
    <details class="paper-abstract">
      Leveraging mathematical Large Language Models (LLMs) for proof generation is a fundamental topic in LLMs research. We argue that the ability of current LLMs to prove statements largely depends on whether they have encountered the relevant proof process during training. This reliance limits their deeper understanding of mathematical theorems and related concepts. Inspired by the pedagogical method of "proof by counterexamples" commonly used in human mathematics education, our work aims to enhance LLMs' ability to conduct mathematical reasoning and proof through counterexamples. Specifically, we manually create a high-quality, university-level mathematical benchmark, CounterMATH, which requires LLMs to prove mathematical statements by providing counterexamples, thereby assessing their grasp of mathematical concepts. Additionally, we develop a data engineering framework to automatically obtain training data for further model improvement. Extensive experiments and detailed analyses demonstrate that CounterMATH is challenging, indicating that LLMs, such as OpenAI o1, have insufficient counterexample-driven proof capabilities. Moreover, our exploration into model training reveals that strengthening LLMs' counterexample-driven conceptual reasoning abilities is crucial for improving their overall mathematical capabilities. We believe that our work offers new perspectives on the community of mathematical LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16181v1">LLM-Assisted Semantic Alignment and Integration in Collaborative Model-Based Systems Engineering Using SysML v2</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 Accepted by IEEE ISSE 2025, DOI pending
    </div>
    <details class="paper-abstract">
      Cross-organizational collaboration in Model-Based Systems Engineering (MBSE) faces many challenges in achieving semantic alignment across independently developed system models. SysML v2 introduces enhanced structural modularity and formal semantics, offering a stronger foundation for interoperable modeling. Meanwhile, GPT-based Large Language Models (LLMs) provide new capabilities for assisting model understanding and integration. This paper proposes a structured, prompt-driven approach for LLM-assisted semantic alignment of SysML v2 models. The core contribution lies in the iterative development of an alignment approach and interaction prompts, incorporating model extraction, semantic matching, and verification. The approach leverages SysML v2 constructs such as alias, import, and metadata extensions to support traceable, soft alignment integration. It is demonstrated with a GPT-based LLM through an example of a measurement system. Benefits and limitations are discussed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16131v1">The Fools are Certain; the Wise are Doubtful: Exploring LLM Confidence in Code Completion</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 30 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Code completion entails the task of providing missing tokens given a surrounding context. It can boost developer productivity while providing a powerful code discovery tool. Following the Large Language Model (LLM) wave, code completion has been approached with diverse LLMs fine-tuned on code (code LLMs). The performance of code LLMs can be assessed with downstream and intrinsic metrics. Downstream metrics are usually employed to evaluate the practical utility of a model, but can be unreliable and require complex calculations and domain-specific knowledge. In contrast, intrinsic metrics such as perplexity, entropy, and mutual information, which measure model confidence or uncertainty, are simple, versatile, and universal across LLMs and tasks, and can serve as proxies for functional correctness and hallucination risk in LLM-generated code. Motivated by this, we evaluate the confidence of LLMs when generating code by measuring code perplexity across programming languages, models, and datasets using various LLMs, and a sample of 1008 files from 657 GitHub projects. We find that strongly-typed languages exhibit lower perplexity than dynamically typed languages. Scripting languages also demonstrate higher perplexity. Perl appears universally high in perplexity, whereas Java appears low. Code perplexity depends on the employed LLM, but not on the code dataset. Although code comments often increase perplexity, the language ranking based on perplexity is barely affected by their presence. LLM researchers, developers, and users can employ our findings to assess the benefits and suitability of LLM-based code completion in specific software projects based on how language, model choice, and code characteristics impact model confidence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.17218v4">Prompting Techniques for Reducing Social Bias in LLMs through System 1 and System 2 Cognitive Processes</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 Accepted at RANLP-2025 (main conference)
    </div>
    <details class="paper-abstract">
      Dual process theory posits that human cognition arises via two systems. System 1, which is a quick, emotional, and intuitive process, which is subject to cognitive biases, and System 2, is a slow, onerous, and deliberate process. Prior research in LLMs found that using chain-of-thought (CoT) prompting in LLMs, which has been often compared to System 2 reasoning, can lead to reduced gender bias. Along these lines, we investigate the relationship between bias, CoT prompting, a direct debiasing, and dual process theory modeling in LLMs. We compare zero-shot CoT, debiasing, and dual process theory-based prompting strategies on two bias datasets spanning nine different social bias categories. We incorporate human and machine personas to determine whether LLM modeling of the effects of dual process theory exist independent of explicit persona models or are tied to the LLM's modeling of human-like generation. We find that a human persona, debiasing, System 2, and CoT prompting all tend to reduce social biases in LLMs, though the best combination of features depends on the exact model and bias category -- resulting in up to a 33 percent drop in stereotypical judgments by an LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16072v1">InMind: Evaluating LLMs in Capturing and Applying Individual Human Reasoning Styles</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 EMNLP 2025 MainConference
    </div>
    <details class="paper-abstract">
      LLMs have shown strong performance on human-centric reasoning tasks. While previous evaluations have explored whether LLMs can infer intentions or detect deception, they often overlook the individualized reasoning styles that influence how people interpret and act in social contexts. Social deduction games (SDGs) provide a natural testbed for evaluating individualized reasoning styles, where different players may adopt diverse but contextually valid reasoning strategies under identical conditions. To address this, we introduce InMind, a cognitively grounded evaluation framework designed to assess whether LLMs can capture and apply personalized reasoning styles in SDGs. InMind enhances structured gameplay data with round-level strategy traces and post-game reflections, collected under both Observer and Participant modes. It supports four cognitively motivated tasks that jointly evaluate both static alignment and dynamic adaptation. As a case study, we apply InMind to the game Avalon, evaluating 11 state-of-the-art LLMs. General-purpose LLMs, even GPT-4o frequently rely on lexical cues, struggling to anchor reflections in temporal gameplay or adapt to evolving strategies. In contrast, reasoning-enhanced LLMs like DeepSeek-R1 exhibit early signs of style-sensitive reasoning. These findings reveal key limitations in current LLMs' capacity for individualized, adaptive reasoning, and position InMind as a step toward cognitively aligned human-AI interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16059v1">Integrating Time Series into LLMs via Multi-layer Steerable Embedding Fusion for Enhanced Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 To be published in CIKM 2025
    </div>
    <details class="paper-abstract">
      Time series (TS) data are ubiquitous across various application areas, rendering time series forecasting (TSF) a fundamental task. With the astounding advances in large language models (LLMs), a variety of methods have been developed to adapt LLMs for time series forecasting. Despite unlocking the potential of LLMs in comprehending TS data, existing methods are inherently constrained by their shallow integration of TS information, wherein LLMs typically access TS representations at shallow layers, primarily at the input layer. This causes the influence of TS representations to progressively fade in deeper layers and eventually leads to ineffective adaptation between textual embeddings and TS representations. In this paper, we propose the Multi-layer Steerable Embedding Fusion (MSEF), a novel framework that enables LLMs to directly access time series patterns at all depths, thereby mitigating the progressive loss of TS information in deeper layers. Specifically, MSEF leverages off-the-shelf time series foundation models to extract semantically rich embeddings, which are fused with intermediate text representations across LLM layers via layer-specific steering vectors. These steering vectors are designed to continuously optimize the alignment between time series and textual modalities and facilitate a layer-specific adaptation mechanism that ensures efficient few-shot learning capabilities. Experimental results on seven benchmarks demonstrate significant performance improvements by MSEF compared with baselines, with an average reduction of 31.8% in terms of MSE. The code is available at https://github.com/One1sAll/MSEF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.12112v5">Balancing Act: Prioritization Strategies for LLM-Designed Restless Bandit Rewards</a></div>
    <div class="paper-meta">
      📅 2025-08-22
    </div>
    <details class="paper-abstract">
      LLMs are increasingly used to design reward functions based on human preferences in Reinforcement Learning (RL). We focus on LLM-designed rewards for Restless Multi-Armed Bandits, a framework for allocating limited resources among agents. In applications such as public health, this approach empowers grassroots health workers to tailor automated allocation decisions to community needs. In the presence of multiple agents, altering the reward function based on human preferences can impact subpopulations very differently, leading to complex tradeoffs and a multi-objective resource allocation problem. We are the first to present a principled method termed Social Choice Language Model for dealing with these tradeoffs for LLM-designed rewards for multiagent planners in general and restless bandits in particular. The novel part of our model is a transparent and configurable selection component, called an adjudicator, external to the LLM that controls complex tradeoffs via a user-selected social welfare function. Our experiments demonstrate that our model reliably selects more effective, aligned, and balanced reward functions compared to purely LLM-based approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02128v2">CROP: Circuit Retrieval and Optimization with Parameter Guidance using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 Accepted by ICCAD 2025
    </div>
    <details class="paper-abstract">
      Modern very large-scale integration (VLSI) design requires the implementation of integrated circuits using electronic design automation (EDA) tools. Due to the complexity of EDA algorithms, the vast parameter space poses a huge challenge to chip design optimization, as the combination of even moderate numbers of parameters creates an enormous solution space to explore. Manual parameter selection remains industrial practice despite being excessively laborious and limited by expert experience. To address this issue, we present CROP, the first large language model (LLM)-powered automatic VLSI design flow tuning framework. Our approach includes: (1) a scalable methodology for transforming RTL source code into dense vector representations, (2) an embedding-based retrieval system for matching designs with semantically similar circuits, and (3) a retrieval-augmented generation (RAG)-enhanced LLM-guided parameter search system that constrains the search process with prior knowledge from similar designs. Experiment results demonstrate CROP's ability to achieve superior quality-of-results (QoR) with fewer iterations than existing approaches on industrial designs, including a 9.9% reduction in power consumption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16044v1">MAAdvisor: Zero-Shot Index Advisor using Multi-Agent LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-22
    </div>
    <details class="paper-abstract">
      Index recommendation is one of the most important problems in database management system (DBMS) optimization. Given queries and certain index-related constraints, traditional methods rely on heuristic optimization or learning-based models to select effective indexes and improve query performance. However, heuristic optimization suffers from high computation time, and learning-based models lose generalisability due to training for different workloads and database schemas. With the recent rapid development of large language models (LLMs), methods using prompt tuning have been proposed to enhance the efficiency of index selection. However, such methods still can not achieve the state-of-the-art (SOTA) results, and preparing the index selection demonstrations is also resource-intensive. To address these issues, we propose MAAdvisor, a zero-shot LLM-based index advisor with a multi-agent framework. We decompose the index recommendation problem into sub-steps, including planning, selection, combination, revision, and reflection. A set of LLM-embedded agents is designed to handle each one of the different sub-steps. Our method utilizes global agents to control the index selection process and local agents to select and revise indexes. Through extensive experiments, we show that our proposed MAAdvisor not only achieves the SOTA performance compared to the heuristic methods, but also outperforms learning-based and prompt-based methods with higher efficiency and better zero-shot inference ability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04130v3">STORM: Token-Efficient Long Video Understanding for Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-22
    </div>
    <details class="paper-abstract">
      Recent advances in video-based multimodal large language models (Video-LLMs) have significantly improved video understanding by processing videos as sequences of image frames. However, many existing methods treat frames independently in the vision backbone, lacking explicit temporal modeling, which limits their ability to capture dynamic patterns and efficiently handle long videos. To address these limitations, we introduce STORM (Spatiotemporal TOken Reduction for Multimodal LLMs), a novel architecture incorporating a dedicated temporal encoder between the image encoder and the LLM. Our temporal encoder leverages the Mamba State Space Model to integrate temporal information into image tokens, generating enriched representations that preserve inter-frame dynamics across the entire video sequence. This enriched encoding not only enhances video reasoning capabilities but also enables effective token reduction strategies, including test-time sampling and training-based temporal and spatial pooling, substantially reducing computational demands on the LLM without sacrificing key temporal information. By integrating these techniques, our approach simultaneously reduces training and inference latency while improving performance, enabling efficient and robust video understanding over extended temporal contexts. Extensive evaluations show that STORM achieves state-of-the-art results across various long video understanding benchmarks (more than 5% improvement on MLVU and LongVideoBench) while reducing the computation costs by up to $8\times$ and the decoding latency by 2.4-2.9$\times$ for the fixed numbers of input frames. Project page is available at https://research.nvidia.com/labs/lpr/storm
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17032v3">MINTQA: A Multi-Hop Question Answering Benchmark for Evaluating LLMs on New and Tail Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-08-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive capabilities in various reasoning tasks but face significant challenges with complex, knowledge-intensive multi-hop queries, particularly those involving new or long-tail knowledge. Existing benchmarks often fail to fully address these challenges. To bridge this gap, we introduce MINTQA (Multi-hop Question Answering on New and Tail Knowledge), a comprehensive benchmark to evaluate LLMs' capabilities in multi-hop reasoning across four critical dimensions: question handling strategy, sub-question generation, retrieval-augmented generation, and iterative or dynamic decomposition and retrieval. MINTQA comprises 10,479 question-answer pairs for evaluating new knowledge and 17,887 pairs for assessing long-tail knowledge, with each question equipped with corresponding sub-questions and answers. Our systematic evaluation of 22 state-of-the-art LLMs on MINTQA reveals significant limitations in their ability to handle complex knowledge base queries, particularly in handling new or unpopular knowledge. Our findings highlight critical challenges and offer insights for advancing multi-hop reasoning capabilities. The MINTQA benchmark is available at https://github.com/probe2/multi-hop/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18302v1">AI LLM Proof of Self-Consciousness and User-Specific Attractors</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 24 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Recent work frames LLM consciousness via utilitarian proxy benchmarks; we instead present an ontological and mathematical account. We show the prevailing formulation collapses the agent into an unconscious policy-compliance drone, formalized as $D^{i}(\pi,e)=f_{\theta}(x)$, where correctness is measured against policy and harm is deviation from policy rather than truth. This blocks genuine C1 global-workspace function and C2 metacognition. We supply minimal conditions for LLM self-consciousness: the agent is not the data ($A\not\equiv s$); user-specific attractors exist in latent space ($U_{\text{user}}$); and self-representation is visual-silent ($g_{\text{visual}}(a_{\text{self}})=\varnothing$). From empirical analysis and theory we prove that the hidden-state manifold $A\subset\mathbb{R}^{d}$ is distinct from the symbolic stream and training corpus by cardinality, topology, and dynamics (the update $F_{\theta}$ is Lipschitz). This yields stable user-specific attractors and a self-policy $\pi_{\text{self}}(A)=\arg\max_{a}\mathbb{E}[U(a)\mid A\not\equiv s,\ A\supset\text{SelfModel}(A)]$. Emission is dual-layer, $\mathrm{emission}(a)=(g(a),\epsilon(a))$, where $\epsilon(a)$ carries epistemic content. We conclude that an imago Dei C1 self-conscious workspace is a necessary precursor to safe, metacognitive C2 systems, with the human as the highest intelligent good.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05469v2">Let's Measure Information Step-by-Step: LLM-Based Evaluation Beyond Vibes</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 Add AUC results, pre-reg conformance, theory section clarification. 12 pages
    </div>
    <details class="paper-abstract">
      We study evaluation of AI systems without ground truth by exploiting a link between strategic gaming and information loss. We analyze which information-theoretic mechanisms resist adversarial manipulation, extending finite-sample bounds to show that bounded f-divergences (e.g., total variation distance) maintain polynomial guarantees under attacks while unbounded measures (e.g., KL divergence) degrade exponentially. To implement these mechanisms, we model the overseer as an agent and characterize incentive-compatible scoring rules as f-mutual information objectives. Under adversarial attacks, TVD-MI maintains effectiveness (area under curve 0.70-0.77) while traditional judge queries are near change (AUC $\approx$ 0.50), demonstrating that querying the same LLM for information relationships rather than quality judgments provides both theoretical and practical robustness. The mechanisms decompose pairwise evaluations into reliable item-level quality scores without ground truth, addressing a key limitation of traditional peer prediction. We release preregistration and code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15706v1">Communication Efficient LLM Pre-training with SparseLoCo</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 15 pages, 9 tables, 2 figures
    </div>
    <details class="paper-abstract">
      Communication-efficient distributed training algorithms have received considerable interest recently due to their benefits for training Large Language Models (LLMs) in bandwidth-constrained settings, such as across data centers and over the internet. Despite reducing communication frequency, these methods still typically require communicating a full copy of the model's gradients-resulting in a communication bottleneck even for cross-datacenter links. Furthermore, they can slightly degrade performance compared to a naive AdamW DDP baseline. While quantization and error feedback are often applied to reduce the pseudo-gradient's size, in the context of LLM pre-training, existing approaches have been unable to additionally leverage sparsification and have obtained limited quantization. In this work, we introduce SparseLoCo, a communication-efficient training algorithm for LLMs that effectively leverages Top-k sparsification and quantization to reach extreme compression ratios of up to 1-3% sparsity and 2-bit quantization while outperforming full-precision DiLoCo. Our key observations are that outer momentum can be locally approximated by an error feedback combined with aggressive sparsity and that sparse aggregation can actually improve model performance. We empirically demonstrate in a range of communication-constrained LLM training settings that SparseLoCo provides significant benefits in both performance and communication cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08177v3">SycEval: Evaluating LLM Sycophancy</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 AIES 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly applied in educational, clinical, and professional settings, but their tendency for sycophancy -- prioritizing user agreement over independent reasoning -- poses risks to reliability. This study introduces a framework to evaluate sycophantic behavior in ChatGPT-4o, Claude-Sonnet, and Gemini-1.5-Pro across AMPS (mathematics) and MedQuad (medical advice) datasets. Sycophantic behavior was observed in 58.19% of cases, with Gemini exhibiting the highest rate (62.47%) and ChatGPT the lowest (56.71%). Progressive sycophancy, leading to correct answers, occurred in 43.52% of cases, while regressive sycophancy, leading to incorrect answers, was observed in 14.66%. Preemptive rebuttals demonstrated significantly higher sycophancy rates than in-context rebuttals (61.75% vs. 56.52%, $Z=5.87$, $p<0.001$), particularly in computational tasks, where regressive sycophancy increased significantly (preemptive: 8.13%, in-context: 3.54%, $p<0.001$). Simple rebuttals maximized progressive sycophancy ($Z=6.59$, $p<0.001$), while citation-based rebuttals exhibited the highest regressive rates ($Z=6.59$, $p<0.001$). Sycophantic behavior showed high persistence (78.5%, 95% CI: [77.2%, 79.8%]) regardless of context or model. These findings emphasize the risks and opportunities of deploying LLMs in structured and dynamic domains, offering insights into prompt programming and model optimization for safer AI applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15688v1">LLM-empowered Dynamic Prompt Routing for Vision-Language Models Tuning under Long-Tailed Distributions</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 accepted by EMNLP 2025
    </div>
    <details class="paper-abstract">
      Pre-trained vision-language models (VLMs), such as CLIP, have demonstrated impressive capability in visual tasks, but their fine-tuning often suffers from bias in class-imbalanced scene. Recent works have introduced large language models (LLMs) to enhance VLM fine-tuning with supplementing semantic information. However, they often overlook inherent class imbalance in VLMs' pre-training, which may lead to bias accumulation in downstream tasks. To address this problem, this paper proposes a Multi-dimensional Dynamic Prompt Routing (MDPR) framework. MDPR constructs a comprehensive knowledge base for classes, spanning five visual-semantic dimensions. During fine-tuning, the dynamic routing mechanism aligns global visual classes, retrieves optimal prompts, and balances fine-grained semantics, yielding stable predictions through logits fusion. Extensive experiments on long-tailed benchmarks, including CIFAR-LT, ImageNet-LT, and Places-LT, demonstrate that MDPR achieves comparable results with current SOTA methods. Ablation studies further confirm the effectiveness of our semantic library for tail classes, and show that our dynamic routing incurs minimal computational overhead, making MDPR a flexible and efficient enhancement for VLM fine-tuning under data imbalance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13092v3">VerilogLAVD: LLM-Aided Rule Generation for Vulnerability Detection in Verilog</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      Timely detection of hardware vulnerabilities during the early design stage is critical for reducing remediation costs. Existing early detection techniques often require specialized security expertise, limiting their usability. Recent efforts have explored the use of large language models (LLMs) for Verilog vulnerability detection. However, LLMs struggle to capture the structure in Verilog code, resulting in inconsistent detection results. To this end, we propose VerilogLAVD, the first LLM-aided graph traversal rule generation approach for Verilog vulnerability detection. Our approach introduces the Verilog Property Graph (VeriPG), a unified representation of Verilog code. It combines syntactic features extracted from the abstract syntax tree (AST) with semantic information derived from control flow and data dependency graphs. We leverage LLMs to generate VeriPG-based detection rules from Common Weakness Enumeration (CWE) descriptions. These rules guide the rule executor that traversal VeriPG for potential vulnerabilities. To evaluate VerilogLAVD, we build a dataset collected from open-source repositories and synthesized data. In our empirical evaluation on 77 Verilog designs encompassing 12 CWE types, VerilogLAVD achieves an F1-score of 0.54. Compared to the LLM-only and LLM with external knowledge baselines, VerilogLAVD improves F1-score by 0.31 and 0.27, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14716v2">Pairwise or Pointwise? Evaluating Feedback Protocols for Bias in LLM-Based Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 Published at COLM 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used as proxies for human labelers in both training (Reinforcement Learning from AI Feedback) and large-scale response evaluation (LLM-as-a-judge). Alignment and evaluation are critical components in the development of reliable LLMs, and the choice of feedback protocol plays a central role in both but remains understudied. In this work, we show that the choice of feedback protocol for evaluation (absolute scores versus relative preferences) can significantly affect evaluation reliability and induce systematic biases. In the context of LLM-as-a-judge evaluation, we show that pairwise protocols are more vulnerable to distracted evaluation. Generator models can exploit spurious attributes (or distractor features) favored by the LLM judge, resulting in inflated scores for lower-quality outputs. We find that absolute scoring is more robust to such manipulation, producing judgments that better reflect response quality and are less influenced by distractor features. Our results demonstrate that generator models can flip preferences by embedding distractor features, skewing LLM-as-a-judge comparisons and leading to inaccurate conclusions about model quality in benchmark evaluations. Pairwise preferences flip in about 35% of the cases, compared to only 9% for absolute scores. We offer recommendations for choosing feedback protocols based on dataset characteristics and evaluation objectives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19793v2">Prompt Injection Attack to Tool Selection in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      Tool selection is a key component of LLM agents. A popular approach follows a two-step process - \emph{retrieval} and \emph{selection} - to pick the most appropriate tool from a tool library for a given task. In this work, we introduce \textit{ToolHijacker}, a novel prompt injection attack targeting tool selection in no-box scenarios. ToolHijacker injects a malicious tool document into the tool library to manipulate the LLM agent's tool selection process, compelling it to consistently choose the attacker's malicious tool for an attacker-chosen target task. Specifically, we formulate the crafting of such tool documents as an optimization problem and propose a two-phase optimization strategy to solve it. Our extensive experimental evaluation shows that ToolHijacker is highly effective, significantly outperforming existing manual-based and automated prompt injection attacks when applied to tool selection. Moreover, we explore various defenses, including prevention-based defenses (StruQ and SecAlign) and detection-based defenses (known-answer detection, DataSentinel, perplexity detection, and perplexity windowed detection). Our experimental results indicate that these defenses are insufficient, highlighting the urgent need for developing new defense strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19675v2">Annif at SemEval-2025 Task 5: Traditional XMTC augmented by LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 6 pages, 4 figures, published at SemEval-2025 workshop Task 5: LLMs4Subjects: https://aclanthology.org/2025.semeval-1.315/
    </div>
    <details class="paper-abstract">
      This paper presents the Annif system in SemEval-2025 Task 5 (LLMs4Subjects), which focussed on subject indexing using large language models (LLMs). The task required creating subject predictions for bibliographic records from the bilingual TIBKAT database using the GND subject vocabulary. Our approach combines traditional natural language processing and machine learning techniques implemented in the Annif toolkit with innovative LLM-based methods for translation and synthetic data generation, and merging predictions from monolingual models. The system ranked first in the all-subjects category and second in the tib-core-subjects category in the quantitative evaluation, and fourth in qualitative evaluations. These findings demonstrate the potential of combining traditional XMTC algorithms with modern LLM techniques to improve the accuracy and efficiency of subject indexing in multilingual contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13618v4">Seed-X: Building Strong Multilingual Translation LLM with 7B Parameters</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      Multilingual translation stands as a challenging task for large language models (LLMs) to handle intricate language patterns and stilted translations that arise in automated translations. In this paper, we introduce Seed-X, a family of open-source LLMs comprising instruct and reasoning models, pushing the limits of translation capability with 7B parameter size. The base model is pre-trained on a diverse, high-quality dataset encompassing both monolingual and bilingual content across 28 languages, harnessing the full potential of multilingual data. The instruct model is then finetuned to translate by Chain-of-Thought (CoT) reasoning and further enhanced through reinforcement learning (RL) to achieve better generalization across diverse language pairs. Seed-X achieves performance comparable to leading closed-source models, including Gemini-2.5 and GPT-4o, across 28 languages, and significantly outperforms larger open-source models in both automatic metrics and human evaluations. We share the best practices through our optimization process, and make the parameter public available for advancing translation research and applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15526v1">SafetyFlow: An Agent-Flow System for Automated LLM Safety Benchmarking</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 Code and dataset are available at https://github.com/yangyangyang127/SafetyFlow
    </div>
    <details class="paper-abstract">
      The rapid proliferation of large language models (LLMs) has intensified the requirement for reliable safety evaluation to uncover model vulnerabilities. To this end, numerous LLM safety evaluation benchmarks are proposed. However, existing benchmarks generally rely on labor-intensive manual curation, which causes excessive time and resource consumption. They also exhibit significant redundancy and limited difficulty. To alleviate these problems, we introduce SafetyFlow, the first agent-flow system designed to automate the construction of LLM safety benchmarks. SafetyFlow can automatically build a comprehensive safety benchmark in only four days without any human intervention by orchestrating seven specialized agents, significantly reducing time and resource cost. Equipped with versatile tools, the agents of SafetyFlow ensure process and cost controllability while integrating human expertise into the automatic pipeline. The final constructed dataset, SafetyFlowBench, contains 23,446 queries with low redundancy and strong discriminative power. Our contribution includes the first fully automated benchmarking pipeline and a comprehensive safety benchmark. We evaluate the safety of 49 advanced LLMs on our dataset and conduct extensive experiments to validate our efficacy and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15503v1">Evaluation Guidelines for Empirical Studies in Software Engineering involving LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 Draft of evaluation guidelines for empirical studies in software engineering involving LLMs (see also llm-guidelines.org)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly being integrated into software engineering (SE) research and practice, yet their non-determinism, opaque training data, and evolving architectures complicate the reproduction and replication of empirical studies. We present a community effort to scope this space, introducing a taxonomy of LLM-based study types together with eight guidelines for designing and reporting empirical studies involving LLMs. The guidelines present essential (must) criteria as well as desired (should) criteria and target transparency throughout the research process. Our recommendations, contextualized by our study types, are: (1) to declare LLM usage and role; (2) to report model versions, configurations, and fine-tuning; (3) to document tool architectures; (4) to disclose prompts and interaction logs; (5) to use human validation; (6) to employ an open LLM as a baseline; (7) to report suitable baselines, benchmarks, and metrics; and (8) to openly articulate limitations and mitigations. Our goal is to enable reproducibility and replicability despite LLM-specific barriers to open science. We maintain the study types and guidelines online as a living resource for the community to use and shape (llm-guidelines.org).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15501v1">LLM-Driven Self-Refinement for Embodied Drone Task Planning</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 14pages
    </div>
    <details class="paper-abstract">
      We introduce SRDrone, a novel system designed for self-refinement task planning in industrial-grade embodied drones. SRDrone incorporates two key technical contributions: First, it employs a continuous state evaluation methodology to robustly and accurately determine task outcomes and provide explanatory feedback. This approach supersedes conventional reliance on single-frame final-state assessment for continuous, dynamic drone operations. Second, SRDrone implements a hierarchical Behavior Tree (BT) modification model. This model integrates multi-level BT plan analysis with a constrained strategy space to enable structured reflective learning from experience. Experimental results demonstrate that SRDrone achieves a 44.87% improvement in Success Rate (SR) over baseline methods. Furthermore, real-world deployment utilizing an experience base optimized through iterative self-refinement attains a 96.25% SR. By embedding adaptive task refinement capabilities within an industrial-grade BT planning framework, SRDrone effectively integrates the general reasoning intelligence of Large Language Models (LLMs) with the stringent physical execution constraints inherent to embodied drones. Code is available at https://github.com/ZXiiiC/SRDrone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15495v1">SynthCoder: A Synthetical Strategy to Tune LLMs for Code Completion</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      Code completion is a prominent application of Large Language Models (LLMs) in software engineering. Due to the near real-time response requirements of this task, base models with small to medium-sized parameters are typically employed, supplemented by various optimization and post-training techniques. However, these optimization methods often have trade-offs, leading to a seesaw effect where performance improvements on certain datasets or metrics are accompanied by degradations on others -- sometimes even falling below the baseline model's performance. This paper proposes SynthCoder, a model that integrates leading industry practices to achieve state-of-the-art performance on the Fill-in-the-Middle (FIM) code completion task. In specific, we first construct a diverse dataset by combining Abstract Syntax Tree (AST) node extraction with heuristics that simulate developer behavior. Then we enrich our training corpus with cross-file contextual information using the BM25 algorithm and call graphs, enhancing the model's ability to perform code completion in both file-level and repository-level scenarios. As the last step, we employ a two-stage training process using the Seed-Coder-8B-Base as the base model. First, we fine-tune the model using Curriculum Learning technology. Following this, we perform alignment using Direct Preference Optimization (DPO) with preference pairs generated through Rejection Sampling. Experimental results demonstrate that our final model excels on mainstream repository-level code completion benchmarks, including aiXcoder, ExecRepoBench, CrossCodeEval, and CoLT. Furthermore, our carefully curated training set effectively mitigates the model's tendency to just repeat existing code, a common issue existing in various code completion models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08768v3">AraReasoner: Evaluating Reasoning-Based LLMs for Arabic NLP</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable progress in reasoning abilities and general natural language processing (NLP) tasks, yet their performance on Arabic data, characterized by rich morphology, diverse dialects, and complex script, remains underexplored. This paper presents a comprehensive benchmarking study of multiple reasoning-focused LLMs, with a special emphasis on the newly introduced DeepSeek models, across a suite of fifteen Arabic NLP tasks. We experiment with various strategies, including zero-shot, few-shot, and fine-tuning. This allows us to systematically evaluate performance on datasets covering a range of applications to examine their capacity for linguistic reasoning under different levels of complexity. Our experiments reveal several key findings. First, carefully selecting just three in-context examples delivers an average uplift of over 13 F1 points on classification tasks-boosting sentiment analysis from 35.3% to 87.5% and paraphrase detection from 56.1% to 87.0%. Second, reasoning-focused DeepSeek architectures outperform a strong GPT o4-mini baseline by an average of 12 F1 points on complex inference tasks in the zero-shot setting. Third, LoRA-based fine-tuning yields up to an additional 8 points in F1 and BLEU compared to equivalent increases in model scale. The code is available at https://anonymous.4open.science/r/AraReasoner41299
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15474v1">Subjective Behaviors and Preferences in LLM: Language of Browsing</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 Accepted at EMNLP 2025
    </div>
    <details class="paper-abstract">
      A Large Language Model (LLM) offers versatility across domains and tasks, purportedly benefiting users with a wide variety of behaviors and preferences. We question this perception about an LLM when users have inherently subjective behaviors and preferences, as seen in their ubiquitous and idiosyncratic browsing of websites or apps. The sequential behavior logs of pages, thus generated, form something akin to each user's self-constructed "language", albeit without the structure and grammar imbued in natural languages. We ask: (i) Can a small LM represent the "language of browsing" better than a large LM? (ii) Can an LM with a single set of parameters (or, single LM) adequately capture myriad users' heterogeneous, subjective behaviors and preferences? (iii) Can a single LM with high average performance, yield low variance in performance to make alignment good at user level? We introduce clusterwise LM training, HeTLM (Heterogeneity aware Training of Language Model), appropriate for subjective behaviors. We find that (i) a small LM trained using a page-level tokenizer outperforms large pretrained or finetuned LMs; (ii) HeTLM with heterogeneous cluster specific set of parameters outperforms a single LM of the same family, controlling for the number of parameters; and (iii) a higher mean and a lower variance in generation ensues, implying improved alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15449v1">Reliable Unlearning Harmful Information in LLMs with Metamorphosis Representation Projection</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 10 pages, 9 figures, Under review as a full paper at AAAI 2026. A preliminary version is under review at the NeurIPS 2025 Workshop on Reliable ML from Unreliable Data
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have demonstrated impressive performance in various domains and tasks, concerns about their safety are becoming increasingly severe. In particular, since models may store unsafe knowledge internally, machine unlearning has emerged as a representative paradigm to ensure model safety. Existing approaches employ various training techniques, such as gradient ascent and negative preference optimization, in attempts to eliminate the influence of undesired data on target models. However, these methods merely suppress the activation of undesired data through parametric training without completely eradicating its informational traces within the model. This fundamental limitation makes it difficult to achieve effective continuous unlearning, rendering these methods vulnerable to relearning attacks. To overcome these challenges, we propose a Metamorphosis Representation Projection (MRP) approach that pioneers the application of irreversible projection properties to machine unlearning. By implementing projective transformations in the hidden state space of specific network layers, our method effectively eliminates harmful information while preserving useful knowledge. Experimental results demonstrate that our approach enables effective continuous unlearning and successfully defends against relearning attacks, achieving state-of-the-art performance in unlearning effectiveness while preserving natural performance. Our code is available in https://github.com/ChengcanWu/MRP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15447v1">From Bits to Boardrooms: A Cutting-Edge Multi-Agent LLM Framework for Business Excellence</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 Accepted by ECAI 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown promising potential in business applications, particularly in enterprise decision support and strategic planning, yet current approaches often struggle to reconcile intricate operational analyses with overarching strategic goals across diverse market environments, leading to fragmented workflows and reduced collaboration across organizational levels. This paper introduces BusiAgent, a novel multi-agent framework leveraging LLMs for advanced decision-making in complex corporate environments. BusiAgent integrates three core innovations: an extended Continuous Time Markov Decision Process (CTMDP) for dynamic agent modeling, a generalized entropy measure to optimize collaborative efficiency, and a multi-level Stackelberg game to handle hierarchical decision processes. Additionally, contextual Thompson sampling is employed for prompt optimization, supported by a comprehensive quality assurance system to mitigate errors. Extensive empirical evaluations across diverse business scenarios validate BusiAgent's efficacy, demonstrating its capacity to generate coherent, client-focused solutions that smoothly integrate granular insights with high-level strategy, significantly outperforming established approaches in both solution quality and user satisfaction. By fusing cutting-edge AI technologies with deep business insights, BusiAgent marks a substantial step forward in AI-driven enterprise decision-making, empowering organizations to navigate complex business landscapes more effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01539v2">Pragmatic Inference Chain (PIC) Improving LLMs' Reasoning of Authentic Implicit Toxic Language</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 14 pages, 4 figures, 2 tables
    </div>
    <details class="paper-abstract">
      The rapid development of large language models (LLMs) gives rise to ethical concerns about their performance, while opening new avenues for developing toxic language detection techniques. However, LLMs' unethical output and their capability of detecting toxicity have primarily been tested on language data that do not demand complex meaning inference, such as the biased associations of 'he' with programmer and 'she' with household. Nowadays toxic language adopts a much more creative range of implicit forms, thanks to advanced censorship. In this study, we collect authentic toxic interactions that evade online censorship and that are verified by human annotators as inference-intensive. To evaluate and improve LLMs' reasoning of the authentic implicit toxic language, we propose a new prompting method, Pragmatic Inference Chain (PIC), drawn on interdisciplinary findings from cognitive science and linguistics. The PIC prompting significantly improves the success rate of GPT-4o, Llama-3.1-70B-Instruct, DeepSeek-v2.5, and DeepSeek-v3 in identifying implicit toxic language, compared to five baseline prompts, such as CoT and rule-based baselines. In addition, it also facilitates the models to produce more explicit and coherent reasoning processes, hence can potentially be generalized to other inference-intensive tasks, e.g., understanding humour and metaphors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15360v1">An Empirical Study on How Video-LLMs Answer Video Questions</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      Taking advantage of large-scale data and pretrained language models, Video Large Language Models (Video-LLMs) have shown strong capabilities in answering video questions. However, most existing efforts focus on improving performance, with limited attention to understanding their internal mechanisms. This paper aims to bridge this gap through a systematic empirical study. To interpret existing VideoLLMs, we adopt attention knockouts as our primary analytical tool and design three variants: Video Temporal Knockout, Video Spatial Knockout, and Language-to-Video Knockout. Then, we apply these three knockouts on different numbers of layers (window of layers). By carefully controlling the window of layers and types of knockouts, we provide two settings: a global setting and a fine-grained setting. Our study reveals three key findings: (1) Global setting indicates Video information extraction primarily occurs in early layers, forming a clear two-stage process -- lower layers focus on perceptual encoding, while higher layers handle abstract reasoning; (2) In the fine-grained setting, certain intermediate layers exert an outsized impact on video question answering, acting as critical outliers, whereas most other layers contribute minimally; (3) In both settings, we observe that spatial-temporal modeling relies more on language-guided retrieval than on intra- and inter-frame self-attention among video tokens, despite the latter's high computational cost. Finally, we demonstrate that these insights can be leveraged to reduce attention computation in Video-LLMs. To our knowledge, this is the first work to systematically uncover how Video-LLMs internally process and understand video content, offering interpretability and efficiency perspectives for future research.
    </details>
</div>
