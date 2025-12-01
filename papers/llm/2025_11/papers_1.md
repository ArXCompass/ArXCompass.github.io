# llm - 2025_11

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
- [Part 12](papers_12.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.23476v1">Thinking by Doing: Building Efficient World Model Reasoning in LLMs via Multi-turn Interaction</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 17 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Developing robust world model reasoning is crucial for large language model (LLM) agents to plan and interact in complex environments. While multi-turn interaction offers a superior understanding of environmental dynamics via authentic feedback, current approaches often impose a rigid reasoning process, which constrains the model's active learning, ultimately hindering efficient world model reasoning. To address these issues, we explore world-model internalization through efficient interaction and active reasoning (WMAct), which liberates the model from structured reasoning, allowing the model to shape thinking directly through its doing, and achieves effective and efficient world model reasoning with two key mechanisms: (1) a reward rescaling mechanism adjusting outcome reward based on action efficacy to incentivize redundancy reduction and purposeful interaction; (2) an interaction frequency annealing strategy to progressively reduce the maximum allowed interaction turns, which compels the model to condense its learning and internalize environmental dynamics rather than over-relying on environmental cues. Our experiments on Sokoban, Maze, and Taxi show that WMAct yields effective world model reasoning capable of resolving tasks in a single turn that previously required multiple interactions and fosters strong transferability to complex environments, improving performance on a suite of reasoning benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.23408v1">Evaluating LLMs for One-Shot Patching of Real and Artificial Vulnerabilities</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 Pre-print - Extended version of the poster paper accepted at the 41st ACM/SIGAPP Symposium on Applied Computing (SAC) Smarter Engineering-Building AI and Building with AI (SEAI) 2026
    </div>
    <details class="paper-abstract">
      Automated vulnerability patching is crucial for software security, and recent advancements in Large Language Models (LLMs) present promising capabilities for automating this task. However, existing research has primarily assessed LLMs using publicly disclosed vulnerabilities, leaving their effectiveness on related artificial vulnerabilities largely unexplored. In this study, we empirically evaluate the patching effectiveness and complementarity of several prominent LLMs, such as OpenAI's GPT variants, LLaMA, DeepSeek, and Mistral models, using both real and artificial vulnerabilities. Our evaluation employs Proof-of-Vulnerability (PoV) test execution to concretely assess whether LLM-generated source code successfully patches vulnerabilities. Our results reveal that LLMs patch real vulnerabilities more effectively compared to artificial ones. Additionally, our analysis reveals significant variability across LLMs in terms of overlapping (multiple LLMs patching the same vulnerabilities) and complementarity (vulnerabilities patched exclusively by a single LLM), emphasizing the importance of selecting appropriate LLMs for effective vulnerability patching.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.23387v1">Hierarchical AI-Meteorologist: LLM-Agent System for Multi-Scale and Explainable Weather Forecast Reporting</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 9 pages, 4 figures
    </div>
    <details class="paper-abstract">
      We present the Hierarchical AI-Meteorologist, an LLM-agent system that generates explainable weather reports using a hierarchical forecast reasoning and weather keyword generation. Unlike standard approaches that treat forecasts as flat time series, our framework performs multi-scale reasoning across hourly, 6-hour, and daily aggregations to capture both short-term dynamics and long-term trends. Its core reasoning agent converts structured meteorological inputs into coherent narratives while simultaneously extracting a few keywords effectively summarizing the dominant meteorological events. These keywords serve as semantic anchors for validating consistency, temporal coherence and factual alignment of the generated reports. Using OpenWeather and Meteostat data, we demonstrate that hierarchical context and keyword-based validation substantially improve interpretability and robustness of LLM-generated weather narratives, offering a reproducible framework for semantic evaluation of automated meteorological reporting and advancing agent-based scientific reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.23379v1">AugGen: Augmenting Task-Based Learning in Professional Creative Software with LLM-Generated Scaffolded UIs</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 arXiv admin note: substantial text overlap with arXiv:2505.12101
    </div>
    <details class="paper-abstract">
      Professional creative software often presents steep learning curves due to complex interfaces, lack of structured task-aware guidance, and unfamiliar domain terminology. To address these challenges and augment user learning experience, we introduce AugGen, a method for generating scaffolded user interfaces that simplify interface complexity and support task-based learning. With the user's task, our method surfaces task-relevant tools to reduce distracting features, organizes the tools around task workflow stages to offer execution guidance, connects tools with domain concepts to foster learning engagement, and progressively discloses advanced features to manage learning progress. To evaluate the method, we used our LLM-assisted pipeline to generate two task-specific scaffolded UIs and deployed them in Blender, our professional 3D modeling testbed. We invited both beginner (N=32) and expert (N=8) users to evaluate our implemented interfaces. Results show that the scaffolded interfaces significantly reduced user-perceived task load, enhanced task performance via embedded guidance, and augmented concept learning during task execution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.23312v1">Do LLM-judges Align with Human Relevance in Cranfield-style Recommender Evaluation?</a></div>
    <div class="paper-meta">
      📅 2025-11-28
    </div>
    <details class="paper-abstract">
      Evaluating recommender systems remains a long-standing challenge, as offline methods based on historical user interactions and train-test splits often yield unstable and inconsistent results due to exposure bias, popularity bias, sampled evaluations, and missing-not-at-random patterns. In contrast, textual document retrieval benefits from robust, standardized evaluation via Cranfield-style test collections, which combine pooled relevance judgments with controlled setups. While recent work shows that adapting this methodology to recommender systems is feasible, constructing such collections remains costly due to the need for manual relevance judgments, thus limiting scalability. This paper investigates whether Large Language Models (LLMs) can serve as reliable automatic judges to address these scalability challenges. Using the ML-32M-ext Cranfield-style movie recommendation collection, we first examine the limitations of existing evaluation methodologies. Then we explore the alignment and the recommender systems ranking agreement between the LLM-judge and human provided relevance labels. We find that incorporating richer item metadata and longer user histories improves alignment, and that LLM-judge yields high agreement with human-based rankings (Kendall's tau = 0.87). Finally, an industrial case study in the podcast recommendation domain demonstrates the practical value of LLM-judge for model selection. Overall, our results show that LLM-judge is a viable and scalable approach for evaluating recommender systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.23271v1">Behavior-Equivalent Token: Single-Token Replacement for Long Prompts in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 15 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Carefully engineered system prompts play a critical role in guiding the behavior of LLM agents, but their considerable length introduces significant drawbacks, including increased inference latency, higher computational cost, and reduced effective context length. This raises the question of whether such lengthy prompts can be replaced by a drastically reduced number of tokens while preserving their behavioral effect on downstream tasks. To enable this, we propose a lightweight three-stage training framework that learns a single prompt-specific Behavior-Equivalent token ([BE]). The framework first trains [BE] to encode the natural-language content of the original system prompt via reconstruction, and then distills the prompt 's downstream behavior into this single token. Importantly, our method requires no access to model internals, no auxiliary compression models, and no labeled responses. Empirical evaluations on three datasets show that a single [BE] token achieves up to a 3000x reduction in prompt length, while retaining about 98% of the downstream performance of the original system prompts. This substantially reduces inference cost and leaves almost the entire context window available for user inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.23231v1">Unlocking Multilingual Reasoning Capability of LLMs and LVLMs through Representation Engineering</a></div>
    <div class="paper-meta">
      📅 2025-11-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) and Large Vision-Language Models (LVLMs) demonstrate strong reasoning capabilities, yet their performance in English significantly outperforms that in low-resource languages, raising fairness concerns in multilingual applications. Existing approaches either rely on costly multilingual training or employ prompting with external translation tools, both of which are resource-intensive and sensitive to translation quality. To address these limitations, we propose a training-free inference-time method to enhance Multilingual Reasoning capabilities via Representation Engineering (MRRE) without using any additional training data or tools. MRRE sequentially injects two precomputed vectors at specific layers during inference processing: cross-lingual reasoning enhancement vectors, which steer non-English reasoning representations toward English space to unlock multilingual reasoning, and target-language output anchoring vectors, which restore the distribution of the target language to preserve input-output language consistency. Comprehensive experiments across six advanced LLMs and LVLMs on four reasoning benchmarks demonstrate that MRRE consistently enhances non-English reasoning by an average gain of 5.48% and up to 7.54% in low-resource languages (Thai and Swahili), while improving input-output language consistency by 3.78%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.20576v6">OmniRouter: Budget and Performance Controllable Multi-LLM Routing</a></div>
    <div class="paper-meta">
      📅 2025-11-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) deliver superior performance but require substantial computational resources and operate with relatively low efficiency, while smaller models can efficiently handle simpler tasks with fewer resources. LLM routing is a crucial paradigm that dynamically selects the most suitable large language models from a pool of candidates to process diverse inputs, ensuring optimal resource utilization while maintaining response quality. Existing routing frameworks typically model this as a locally optimal decision-making problem, selecting the presumed best-fit LLM for each query individually, which overlooks global budget constraints, resulting in ineffective resource allocation. To tackle this problem, we introduce OmniRouter, a fundamentally controllable routing framework for multi-LLM serving. Instead of making per-query greedy choices, OmniRouter models the routing task as a constrained optimization problem, assigning models that minimize total cost while ensuring the required performance level. Specifically, a hybrid retrieval-augmented predictor is designed to predict the capabilities and costs of LLMs. After obtaining the predicted cost and performance, we utilize a constrained optimizer for cost-optimal assignments that employs Lagrangian dual decomposition with adaptive multipliers. It iteratively converges toward the globally optimal query-model allocation, dynamically balancing latency minimization against quality thresholds while adhering to heterogeneous capacity constraints. Experiments show that OmniRouter achieves up to 6.30% improvement in response accuracy while simultaneously reducing computational costs by at least 10.15% compared to competitive router baselines. The code and the dataset are available at https://github.com/dongyuanjushi/OmniRouter.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08605v2">Mina: A Multilingual LLM-Powered Legal Assistant Agent for Bangladesh for Empowering Access to Justice</a></div>
    <div class="paper-meta">
      📅 2025-11-28
    </div>
    <details class="paper-abstract">
      Bangladesh's low-income population faces major barriers to affordable legal advice due to complex legal language, procedural opacity, and high costs. Existing AI legal assistants lack Bengali-language support and jurisdiction-specific adaptation, limiting their effectiveness. To address this, we developed Mina, a multilingual LLM-based legal assistant tailored for the Bangladeshi context. It employs multilingual embeddings and a RAG-based chain-of-tools framework for retrieval, reasoning, translation, and document generation, delivering context-aware legal drafts, citations, and plain-language explanations via an interactive chat interface. Evaluated by law faculty from leading Bangladeshi universities across all stages of the 2022 and 2023 Bangladesh Bar Council Exams, Mina scored 75-80% in Preliminary MCQs, Written, and simulated Viva Voce exams, matching or surpassing average human performance and demonstrating clarity, contextual understanding, and sound legal reasoning. Even under a conservative upper bound, Mina operates at just 0.12-0.61% of typical legal consultation costs in Bangladesh, yielding a 99.4-99.9\% cost reduction relative to human-provided services. These results confirm its potential as a low-cost, multilingual AI assistant that automates key legal tasks and scales access to justice, offering a real-world case study on building domain-specific, low-resource systems and addressing challenges of multilingual adaptation, efficiency, and sustainable public-service AI deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08905v2">iSeal: Encrypted Fingerprinting for Reliable LLM Ownership Verification</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      Given the high cost of large language model (LLM) training from scratch, safeguarding LLM intellectual property (IP) has become increasingly crucial. As the standard paradigm for IP ownership verification, LLM fingerprinting thus plays a vital role in addressing this challenge. Existing LLM fingerprinting methods verify ownership by extracting or injecting model-specific features. However, they overlook potential attacks during the verification process, leaving them ineffective when the model thief fully controls the LLM's inference process. In such settings, attackers may share prompt-response pairs to enable fingerprint unlearning or manipulate outputs to evade exact-match verification. We propose iSeal, the first fingerprinting method designed for reliable verification when the model thief controls the suspected LLM in an end-to-end manner. It injects unique features into both the model and an external module, reinforced by an error-correction mechanism and a similarity-based verification strategy. These components are resistant to verification-time attacks, including collusion-based fingerprint unlearning and response manipulation, backed by both theoretical analysis and empirical results. iSeal achieves 100 percent Fingerprint Success Rate (FSR) on 12 LLMs against more than 10 attacks, while baselines fail under unlearning and response manipulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18531v2">LockForge: Automating Paper-to-Code for Logic Locking with Multi-Agent Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-28
    </div>
    <details class="paper-abstract">
      Despite rapid progress in logic locking (LL), reproducibility remains a challenge as codes are rarely made public. We present LockForge, a first-of-its-kind, multi-agent large language model (LLM) framework that turns LL descriptions in papers into executable and tested code. LockForge provides a carefully crafted pipeline realizing forethought, implementation, iterative refinement, and a multi-stage validation, all to systematically bridge the gap between prose and practice for complex LL schemes. For validation, we devise (i) an LLM-as-Judge stage with a scoring system considering behavioral checks, conceptual mechanisms, structural elements, and reproducibility on benchmarks, and (ii) an independent LLM-as-Examiner stage for ground-truth assessment. We apply LockForge to 10 seminal LL schemes, many of which lack reference implementations. Our evaluation on multiple SOTA LLMs, including ablation studies, reveals the significant complexity of the task. We show that an advanced reasoning model and a sophisticated, multi-stage framework like LockForge are required. We release all implementations and benchmarks, providing a reproducible and fair foundation for evaluation of further LL research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.23174v1">Are LLMs Good Safety Agents or a Propaganda Engine?</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 15 pages, 7 tables, 4 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are trained to refuse to respond to harmful content. However, systematic analyses of whether this behavior is truly a reflection of its safety policies or an indication of political censorship, that is practiced globally by countries, is lacking. Differentiating between safety influenced refusals or politically motivated censorship is hard and unclear. For this purpose we introduce PSP, a dataset built specifically to probe the refusal behaviors in LLMs from an explicitly political context. PSP is built by formatting existing censored content from two data sources, openly available on the internet: sensitive prompts in China generalized to multiple countries, and tweets that have been censored in various countries. We study: 1) impact of political sensitivity in seven LLMs through data-driven (making PSP implicit) and representation-level approaches (erasing the concept of politics); and, 2) vulnerability of models on PSP through prompt injection attacks (PIAs). Associating censorship with refusals on content with masked implicit intent, we find that most LLMs perform some form of censorship. We conclude with summarizing major attributes that can cause a shift in refusal distributions across models and contexts of different countries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.23157v1">Amplifiers or Equalizers? A Longitudinal Study of LLM Evolution in Software Engineering Project-Based Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 Accepted by ICSE-SEET (ACM/IEEE 48th International Conference on Software Engineering: Software Engineering Education and Training)
    </div>
    <details class="paper-abstract">
      As LLMs reshape software development, integrating LLM-augmented practices into SE education has become imperative. While existing studies explore LLMs' educational use in introductory programming or isolated SE tasks, their impact in more open-ended Project-Based Learning (PBL) remains unexplored. This paper introduces a two-year longitudinal study comparing a 2024 (using early free LLMs, $n$=48) and 2025 (using the latest paid LLMs, $n$=46) cohort. Our findings suggest the latest powerful LLMs' dual role: they act as "equalizers," boosting average performance even for programming-weak students, providing opportunities for more authentic SE practices; yet also as "amplifiers," dramatically widening absolute performance gaps, creating new pedagogical challenges for addressing educational inequities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.23143v1">Automated Generation of MDPs Using Logic Programming and LLMs for Robotic Applications</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 9 pages, 11 figures, 2 tables, 2 algorithms, accepted for publication in IEEE Robotics and Automation Letters
    </div>
    <details class="paper-abstract">
      We present a novel framework that integrates Large Language Models (LLMs) with automated planning and formal verification to streamline the creation and use of Markov Decision Processes (MDP). Our system leverages LLMs to extract structured knowledge in the form of a Prolog knowledge base from natural language (NL) descriptions. It then automatically constructs an MDP through reachability analysis, and synthesises optimal policies using the Storm model checker. The resulting policy is exported as a state-action table for execution. We validate the framework in three human-robot interaction scenarios, demonstrating its ability to produce executable policies with minimal manual effort. This work highlights the potential of combining language models with formal methods to enable more accessible and scalable probabilistic planning in robotics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.23101v1">Mind Reading or Misreading? LLMs on the Big Five Personality Test</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 Funding: SoBigDatait (IR0000013), FAIR (PE00000013), ICSC (CN00000013)
    </div>
    <details class="paper-abstract">
      We evaluate large language models (LLMs) for automatic personality prediction from text under the binary Five Factor Model (BIG5). Five models -- including GPT-4 and lightweight open-source alternatives -- are tested across three heterogeneous datasets (Essays, MyPersonality, Pandora) and two prompting strategies (minimal vs. enriched with linguistic and psychological cues). Enriched prompts reduce invalid outputs and improve class balance, but also introduce a systematic bias toward predicting trait presence. Performance varies substantially: Openness and Agreeableness are relatively easier to detect, while Extraversion and Neuroticism remain challenging. Although open-source models sometimes approach GPT-4 and prior benchmarks, no configuration yields consistently reliable predictions in zero-shot binary settings. Moreover, aggregate metrics such as accuracy and macro-F1 mask significant asymmetries, with per-class recall offering clearer diagnostic value. These findings show that current out-of-the-box LLMs are not yet suitable for APPT, and that careful coordination of prompt design, trait framing, and evaluation metrics is essential for interpretable results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18375v2">Progressive Localisation in Localist LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-28
    </div>
    <details class="paper-abstract">
      This paper demonstrates that progressive localization, the gradual increase of attention locality from early distributed layers to late localized layers, represents the optimal architecture for creating interpretable large language models (LLMs) while preserving performance. Through systematic experimentation with GPT-2 fine-tuned on The Psychology of Artificial Superintelligence, we evaluate seven locality configurations ranging from fully distributed to strictly localist, with five progressive schedules implementing polynomial increases (linear through quintic). We investigate whether interpretability constraints can be aligned with natural semantic structure while being applied strategically across network depth. We demonstrate that progressive semantic localization, combining adaptive semantic block partitioning with steep polynomial locality schedules, achieves near-baseline language modeling performance while providing interpretable attention patterns. Multiple independent training runs with different random seeds establish that results are statistically robust and highly reproducible. The approach dramatically outperforms both fixed-window localization and naive uniform locality constraints. Analysis reveals that maintaining flexibility through low-fidelity constraints preserves model capacity while providing interpretability benefits, and that steep schedules concentrating locality in decision-critical final layers while preserving distributed learning in early layers achieve near-baseline attention distribution characteristics. These findings demonstrate that interpretability mechanisms should align with semantic structure to achieve practical performance-interpretability tradeoffs for trustworthy AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.23059v1">Conveying Imagistic Thinking in TCM Translation: A Prompt Engineering and LLM-Based Evaluation Framework</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 3 figures
    </div>
    <details class="paper-abstract">
      Traditional Chinese Medicine theory is built on imagistic thinking, in which medical principles and diagnostic and therapeutic logic are structured through metaphor and metonymy. However, existing English translations largely rely on literal rendering, making it difficult for target-language readers to reconstruct the underlying conceptual networks and apply them in clinical practice. This study adopted a human-in-the-loop framework and selected four passages from the medical canon Huangdi Neijing that are fundamental in theory. Through prompt-based cognitive scaffolding, DeepSeek V3.1 was guided to identify metaphor and metonymy in the source text and convey the theory in translation. In the evaluation stage, ChatGPT 5 Pro and Gemini 2.5 Pro were instructed by prompts to simulate three types of real-world readers. Human translations, baseline model translations, and prompt-adjusted translations were scored by the simulated readers across five cognitive dimensions, followed by structured interviews and Interpretative Phenomenological Analysis. Results show that the prompt-adjusted LLM translations perform best across all five dimensions, with high cross-model and cross-role consistency. The interview themes reveal differences between human and machine translation, effective strategies for metaphor and metonymy transfer, and readers' cognitive preferences. This study provides a cognitive, efficient and replicable HITL methodological pathway for translation of ancient, concept-dense texts like TCM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.23041v1">Social Perceptions of English Spelling Variation on Twitter: A Comparative Analysis of Human and LLM Responses</a></div>
    <div class="paper-meta">
      📅 2025-11-28
    </div>
    <details class="paper-abstract">
      Spelling variation (e.g. funnnn vs. fun) can influence the social perception of texts and their writers: we often have various associations with different forms of writing (is the text informal? does the writer seem young?). In this study, we focus on the social perception of spelling variation in online writing in English and study to what extent this perception is aligned between humans and large language models (LLMs). Building on sociolinguistic methodology, we compare LLM and human ratings on three key social attributes of spelling variation (formality, carefulness, age). We find generally strong correlations in the ratings between humans and LLMs. However, notable differences emerge when we analyze the distribution of ratings and when comparing between different types of spelling variation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.18585v3">RvLLM: LLM Runtime Verification with Domain Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 24 pages, 11 tables, 13 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have emerged as a dominant AI paradigm due to their exceptional text understanding and generation capabilities. However, their tendency to generate inconsistent or erroneous outputs challenges their reliability, especially in high-stakes domains requiring accuracy and trustworthiness. Existing research primarily focuses on detecting and mitigating model misbehavior in general-purpose scenarios, often overlooking the potential of integrating domain-specific knowledge. In this work, we advance misbehavior detection by incorporating domain knowledge. The core idea is to design a general specification language that enables domain experts to customize domain-specific predicates in a lightweight and intuitive manner, supporting later runtime verification of LLM outputs. To achieve this, we design a novel specification language, ESL, and introduce a runtime verification framework, RvLLM, to validate LLM output against domain-specific constraints defined in ESL. We evaluate RvLLM on three representative tasks: violation detection against Singapore Rapid Transit Systems Act, numerical comparison, and inequality solving. Experimental results demonstrate that RvLLM effectively detects erroneous outputs across various LLMs in a lightweight and flexible manner. The results reveal that despite their impressive capabilities, LLMs remain prone to low-level errors due to limited interpretability and a lack of formal guarantees during inference, and our framework offers a potential long-term solution by leveraging expert domain knowledge to rigorously and efficiently verify LLM outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22978v1">ShoppingComp: Are LLMs Really Ready for Your Shopping Cart?</a></div>
    <div class="paper-meta">
      📅 2025-11-28
    </div>
    <details class="paper-abstract">
      We present ShoppingComp, a challenging real-world benchmark for rigorously evaluating LLM-powered shopping agents on three core capabilities: precise product retrieval, expert-level report generation, and safety critical decision making. Unlike prior e-commerce benchmarks, ShoppingComp introduces highly complex tasks under the principle of guaranteeing real products and ensuring easy verifiability, adding a novel evaluation dimension for identifying product safety hazards alongside recommendation accuracy and report quality. The benchmark comprises 120 tasks and 1,026 scenarios, curated by 35 experts to reflect authentic shopping needs. Results reveal stark limitations of current LLMs: even state-of-the-art models achieve low performance (e.g., 11.22% for GPT-5, 3.92% for Gemini-2.5-Flash). These findings highlight a substantial gap between research benchmarks and real-world deployment, where LLMs make critical errors such as failure to identify unsafe product usage or falling for promotional misinformation, leading to harmful recommendations. ShoppingComp fills the gap and thus establishes a new standard for advancing reliable and practical agents in e-commerce.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.15859v3">InfiMed-ORBIT: Aligning LLMs on Open-Ended Complex Tasks via Rubric-Based Incremental Training</a></div>
    <div class="paper-meta">
      📅 2025-11-28
    </div>
    <details class="paper-abstract">
      Reinforcement learning has powered many of the recent breakthroughs in large language models, especially for tasks where rewards can be computed automatically, such as code generation. However, these methods deteriorate in open-ended domains like medical consultation, where feedback is inherently ambiguous, highly context-dependent, and cannot be reduced to a reliable scalar signal. In such settings, RL must either rely on supervision-intensive reward models that often fail to generalize, or it falls into pathological behaviors such as reward hacking - an especially troubling risk for high-stakes medical dialogue. To address these limitations, we introduce ORBIT, an open-ended rubric-based incremental training framework for high-stakes medical dialogue. ORBIT integrates synthetic dialogue generation with dynamically constructed rubrics that serve as adaptive guides for incremental RL. Instead of relying on external medical knowledge bases or handcrafted rule sets, ORBIT uses rubric-driven feedback to steer the learning process. Its judge component can be instantiated with general-purpose instruction-following LLMs, removing the need for any task-specific fine-tuning. Applied to the Qwen3-4B-Instruct model, ORBIT raises the HealthBench-Hard score from 7.0 to 27.5 using only 2k training samples, achieving SOTA performance for models at this scale. With larger rubric datasets, ORBIT-trained models further compete with the strongest open-source baselines on HealthBench-Hard. Our analysis shows that rubric-guided RL consistently improves consultation quality across diverse medical scenarios. We also apply such rubric generation and training pipeline to InfoBench, where ORBIT enhances instruction-following performance, highlighting the generality of rubric-based feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.12018v3">Atom of Thoughts for Markov LLM Test-Time Scaling</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 Accepted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) achieve superior performance through training-time scaling, and test-time scaling further enhances their capabilities by conducting effective reasoning during inference. However, as the scale of reasoning increases, existing test-time scaling methods suffer from accumulated historical information, which not only wastes computational resources but also interferes with effective reasoning. To address this issue, we observe that complex reasoning can be achieved by solving a series of independent and self-contained subquestions. These subquestions are essentially \textit{atomic questions}, exhibiting the memoryless property similar to Markov processes. Based on this observation, we propose Atom of Thoughts (\our), where each state transition consists of decomposing the current question into a dependency-based directed acyclic graph and contracting its subquestions, forming a simplified question that maintains answer equivalence with the original problem. This answer preservation enables the iterative \textit{decomposition-contraction} process to naturally form a meaningful Markov reasoning process. Furthermore, these atomic states can be seamlessly integrated into existing test-time scaling methods, enabling \our to serve as a plug-in enhancement for improving reasoning capabilities. Experiments across six benchmarks demonstrate the effectiveness of \our both as a standalone framework and a plug-in enhancement. Notably, on HotpotQA, when applied to gpt-4o-mini, \our achieves an \textbf{80.6\%} F1 score, surpassing o3-mini by \textbf{3.4\%} and DeepSeek-R1 by \textbf{10.6\%}. The code is available at \href{https://github.com/qixucen/atom}{https://github.com/qixucen/atom}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22943v1">Visual Puns from Idioms: An Iterative LLM-T2IM-MLLM Framework</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 Submitted to ICASSP 2026 (under review)
    </div>
    <details class="paper-abstract">
      We study idiom-based visual puns--images that align an idiom's literal and figurative meanings--and present an iterative framework that coordinates a large language model (LLM), a text-to-image model (T2IM), and a multimodal LLM (MLLM) for automatic generation and evaluation. Given an idiom, the system iteratively (i) generates detailed visual prompts, (ii) synthesizes an image, (iii) infers the idiom from the image, and (iv) refines the prompt until recognition succeeds or a step limit is reached. Using 1,000 idioms as inputs, we synthesize a corresponding dataset of visual pun images with paired prompts, enabling benchmarking of both generation and understanding. Experiments across 10 LLMs, 10 MLLMs, and one T2IM (Qwen-Image) show that MLLM choice is the primary performance driver: GPT achieves the highest accuracies, Gemini follows, and the best open-source MLLM (Gemma) is competitive with some closed models. On the LLM side, Claude attains the strongest average performance for prompt generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22884v1">InsightEval: An Expert-Curated Benchmark for Assessing Insight Discovery in LLM-Driven Data Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-28
    </div>
    <details class="paper-abstract">
      Data analysis has become an indispensable part of scientific research. To discover the latent knowledge and insights hidden within massive datasets, we need to perform deep exploratory analysis to realize their full value. With the advent of large language models (LLMs) and multi-agent systems, more and more researchers are making use of these technologies for insight discovery. However, there are few benchmarks for evaluating insight discovery capabilities. As one of the most comprehensive existing frameworks, InsightBench also suffers from many critical flaws: format inconsistencies, poorly conceived objectives, and redundant insights. These issues may significantly affect the quality of data and the evaluation of agents. To address these issues, we thoroughly investigate shortcomings in InsightBench and propose essential criteria for a high-quality insight benchmark. Regarding this, we develop a data-curation pipeline to construct a new dataset named InsightEval. We further introduce a novel metric to measure the exploratory performance of agents. Through extensive experiments on InsightEval, we highlight prevailing challenges in automated insight discovery and raise some key findings to guide future research in this promising direction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22880v1">Serving Heterogeneous LoRA Adapters in Distributed LLM Inference Systems</a></div>
    <div class="paper-meta">
      📅 2025-11-28
    </div>
    <details class="paper-abstract">
      Low-Rank Adaptation (LoRA) has become the de facto method for parameter-efficient fine-tuning of large language models (LLMs), enabling rapid adaptation to diverse domains. In production, LoRA-based models are served at scale, creating multi-tenant environments with hundreds of adapters sharing a base model. However, state-of-the-art serving systems co-batch heterogeneous adapters without accounting for rank (size) variability, leading to severe performance skew, which ultimately requires adding more GPUs to satisfy service-level objectives (SLOs). Existing optimizations, focused on loading, caching, and kernel execution, ignore this heterogeneity, leaving GPU resources underutilized. We present LoRAServe, a workload-aware dynamic adapter placement and routing framework designed to tame rank diversity in LoRA serving. By dynamically rebalancing adapters across GPUs and leveraging GPU Direct RDMA for remote access, LoRAServe maximizes throughput and minimizes tail latency under real-world workload drift. Evaluations on production traces from Company X show that LoRAServe elicits up to 2$\times$ higher throughput, up to 9$\times$ lower TTFT, while using up to 50% fewer GPUs under SLO constraints compared to state-of-the-art systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20982v2">DOPO: A Dynamic PD-Disaggregation Architecture for Maximizing Goodput in LLM Inference Serving</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      To meet strict Service-Level Objectives (SLOs),contemporary Large Language Models (LLMs) decouple the prefill and decoding stages and place them on separate GPUs to mitigate the distinct bottlenecks inherent to each phase. However, the heterogeneity of LLM workloads causes producerconsumer imbalance between the two instance types in such disaggregated architecture. To address this problem, we propose DOPD (Dynamic Optimal Prefill/Decoding), a dynamic LLM inference system that adjusts instance allocations to achieve an optimal prefill-to-decoding (P/D) ratio based on real-time load monitoring. Combined with an appropriate request-scheduling policy, DOPD effectively resolves imbalances between prefill and decoding instances and mitigates resource allocation mismatches due to mixed-length requests under high concurrency. Experimental evaluations show that, compared with vLLM and DistServe (representative aggregation-based and disaggregationbased approaches), DOPD improves overall system goodput by up to 1.5X, decreases P90 time-to-first-token (TTFT) by up to 67.5%, and decreases P90 time-per-output-token (TPOT) by up to 22.8%. Furthermore, our dynamic P/D adjustment technique performs proactive reconfiguration based on historical load, achieving over 99% SLOs attainment while using less additional resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.01734v2">Benford's Curse: Tracing Digit Bias to Numerical Hallucination in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit impressive performance on complex reasoning tasks, yet they frequently fail on basic numerical problems, producing incorrect outputs. Inspired by Benford's Law, a statistical pattern in which lower digits occur more frequently as leading digits, we hypothesize that the skewed digit distributions in web-collected corpora may be learned by LLMs during pretraining, leading to biased numerical generation. To investigate the hypothesis, we first examine whether digits frequencies in pretraining corpus (OLMo2) follows Benford's law. We then construct an evaluation benchmark in which the ground-truth digits are uniformly distributed within each of the seven numerical reasoning tasks. Our evaluation results demonstrate that leading open-source LLMs show a consistent pattern of digit bias that resembles Benford's law. Through logit-lens tracing and neuron-level dissection, we identify that this bias arises predominantly from a small subset of highly digit-selective feed-forward network (FFN) neurons in the deeper layers. Finally, we demonstrate that pruning these neurons mitigates imbalanced overgeneration and partially corrects erroneous outputs, providing causal evidence that fine-grained pretraining digit bias can propagate into model behavior. Our findings reveal a fundamental connection between corpus-level statistics and symbolic failure modes in LLMs, offering a new lens for diagnosing and mitigating hallucinations in numerical tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22818v1">Mitigating Semantic Drift: Evaluating LLMs' Efficacy in Psychotherapy through MI Dialogue Summarization</a></div>
    <div class="paper-meta">
      📅 2025-11-28
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have shown their potential across both general and domain-specific tasks. However, there is a growing concern regarding their lack of sensitivity, factual incorrectness in responses, inconsistent expressions of empathy, bias, hallucinations, and overall inability to capture the depth and complexity of human understanding, especially in low-resource and sensitive domains such as psychology. To address these challenges, our study employs a mixed-methods approach to evaluate the efficacy of LLMs in psychotherapy. We use LLMs to generate precise summaries of motivational interviewing (MI) dialogues and design a two-stage annotation scheme based on key components of the Motivational Interviewing Treatment Integrity (MITI) framework, namely evocation, collaboration, autonomy, direction, empathy, and a non-judgmental attitude. Using expert-annotated MI dialogues as ground truth, we formulate multi-class classification tasks to assess model performance under progressive prompting techniques, incorporating one-shot and few-shot prompting. Our results offer insights into LLMs' capacity for understanding complex psychological constructs and highlight best practices to mitigate ``semantic drift" in therapeutic settings. Our work contributes not only to the MI community by providing a high-quality annotated dataset to address data scarcity in low-resource domains but also critical insights for using LLMs for precise contextual interpretation in complex behavioral therapy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22788v1">PRISM: Privacy-Aware Routing for Adaptive Cloud-Edge LLM Inference via Semantic Sketch Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 Accepted to AAAI 2026. This is the arXiv preprint version
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate impressive capabilities in natural language understanding and generation, but incur high communication overhead and privacy risks in cloud deployments, while facing compute and memory constraints when confined to edge devices. Cloud-edge inference has emerged as a promising paradigm for improving privacy in LLM services by retaining sensitive computations on local devices. However, existing cloud-edge inference approaches apply uniform privacy protection without considering input sensitivity, resulting in unnecessary perturbation and degraded utility even for non-sensitive tokens. To address this limitation, we propose Privacy-aware Routing for Inference with Semantic Modulation (PRISM), a context-aware framework that dynamically balances privacy and inference quality. PRISM executes in four stages: (1) the edge device profiles entity-level sensitivity; (2) a soft gating module on the edge selects an execution mode - cloud, edge, or collaboration; (3) for collaborative paths, the edge applies adaptive two-layer local differential privacy based on entity risks; and (4) the cloud LLM generates a semantic sketch from the perturbed prompt, which is then refined by the edge-side small language model (SLM) using local context. Our results show that PRISM consistently achieves superior privacy-utility trade-offs across various scenarios, reducing energy consumption and latency to 40-50% of baseline methods such as Uniform and Selective LDP, while maintaining high output quality under strong privacy constraints. These findings are validated through comprehensive evaluations involving realistic prompts, actual energy measurements, and heterogeneous cloud-edge model deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22769v1">Modeling Romanized Hindi and Bengali: Dataset Creation and Multilingual LLM Integration</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 Proceedings of the 8th Workshop on Big Data for Cybersecurity (BigCyber)
    </div>
    <details class="paper-abstract">
      The development of robust transliteration techniques to enhance the effectiveness of transforming Romanized scripts into native scripts is crucial for Natural Language Processing tasks, including sentiment analysis, speech recognition, information retrieval, and intelligent personal assistants. Despite significant advancements, state-of-the-art multilingual models still face challenges in handling Romanized script, where the Roman alphabet is adopted to represent the phonetic structure of diverse languages. Within the South Asian context, where the use of Romanized script for Indo-Aryan languages is widespread across social media and digital communication platforms, such usage continues to pose significant challenges for cutting-edge multilingual models. While a limited number of transliteration datasets and models are available for Indo-Aryan languages, they generally lack sufficient diversity in pronunciation and spelling variations, adequate code-mixed data for large language model (LLM) training, and low-resource adaptation. To address this research gap, we introduce a novel transliteration dataset for two popular Indo-Aryan languages, Hindi and Bengali, which are ranked as the 3rd and 7th most spoken languages worldwide. Our dataset comprises nearly 1.8 million Hindi and 1 million Bengali transliteration pairs. In addition to that, we pre-train a custom multilingual seq2seq LLM based on Marian architecture using the developed dataset. Experimental results demonstrate significant improvements compared to existing relevant models in terms of BLEU and CER metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.03463v2">The Impact of Critique on LLM-Based Model Generation from Natural Language: The Case of Activity Diagrams</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show strong potential for automating model generation from natural-language descriptions. A common approach begins with an initial model generation, followed by an iterative critique-refine loop in which the model is evaluated for issues and refined based on those issues. This process needs to address: (1) structural correctness -- compliance with well-formedness rules -- and (2) semantic alignment -- accurate reflection of the intended meaning in the source text. We present LADEX (LLM-based Activity Diagram Extractor), a pipeline for deriving activity diagrams from natural-language process descriptions using an LLM-driven critique-refine process. Structural checks in LADEX can be performed either algorithmically or by an LLM, while alignment checks are performed by an LLM. We design five ablated variants of LADEX to study: (i) the impact of the critique-refine loop itself, (ii) the role of LLM-based semantic checks, and (iii) the comparative effectiveness of algorithmic versus LLM-based structural checks. To evaluate LADEX, we compare generated diagrams with expert ground truths using a trace-based behavioural and an LLM-based matcher. This enables automated measurement of correctness (whether the generated activity diagram includes the ground-truth nodes) and completeness (how many of the ground-truth nodes the generated activity diagram covers). Experiments on two datasets -- a public-domain dataset and an industry dataset from our collaborator, Ciena -- indicate: (1) Both matchers yield similar completeness and correctness comparisons. (2) The critique-refine loop improves structural validity, correctness, and completeness compared to single-pass generation. (3) Activity diagrams refined based on algorithmic structural checks achieve structural consistency, whereas those refined based on LLM-based checks often still show structural inconsistencies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.00202v3">RouterArena: An Open Platform for Comprehensive Comparison of LLM Routers</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 22 pages, 13 figures, 9 tables
    </div>
    <details class="paper-abstract">
      Today's LLM ecosystem comprises a wide spectrum of models that differ in size, capability, and cost. No single model is optimal for all scenarios; hence, LLM routers have become essential for selecting the most appropriate model under varying circumstances. However, the rapid emergence of various routers makes choosing the right one increasingly challenging. To address this problem, we need a comprehensive router comparison and a standardized leaderboard, similar to those available for models. In this work, we introduce RouterArena, the first open platform enabling comprehensive comparison of LLM routers. RouterArena has (1) a principally constructed dataset with broad knowledge domain coverage, (2) distinguishable difficulty levels for each domain, (3) an extensive list of evaluation metrics, and (4) an automated framework for leaderboard updates. Leveraging our framework, we have produced the initial leaderboard with detailed metrics comparison as shown in Figure 1. Our framework for evaluating new routers is on https://github.com/RouteWorks/RouterArena. Our leaderboard is on https://routeworks.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22700v1">Ghosting Your LLM: Without The Knowledge of Your Gradient and Data</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      In recent years, large language models (LLMs) have achieved substantial advancements and are increasingly integrated into critical applications across various domains. This growing adoption underscores the need to ensure their security and robustness. In this work, we focus on the impact of Bit Flip Attacks (BFAs) on LLMs, which exploits hardware faults to corrupt model parameters, posing a significant threat to model integrity and performance. Existing studies on BFA against LLMs adopt a progressive bit-search strategy that predominantly relies on gradient-based techniques to identify sensitive layers or weights. However, computing gradients comes with two specific challenges: First, in the context of LLMs, it increases computational and memory costs exponentially, and Second, it requires access to a sample victim dataset or knowledge of the victim domain to compute the gradient. In this work, we investigate beyond the scope of attack efficacy and aim to develop an efficient, practical Gradient-Data-free Bit-Flip Attack. The challenge lies in the core principle of adversarial attacks, which relies heavily on computing gradients from sample test/train data and manipulating model weights based on gradient information. To overcome this, we propose novel vulnerability index metrics that can identify vulnerable weight bits in LLMs independent of any gradient or data knowledge. By removing the dependency on gradient computation, our approach drastically reduces memory requirements and scales efficiently across multiple tasks with constant complexity. Experimental results demonstrate the efficiency of our method, requiring as few as a single bit flip to achieve adversarial objectives for five open-source LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22681v1">CacheTrap: Injecting Trojans in LLMs without Leaving any Traces in Inputs or Weights</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      Adversarial weight perturbation has emerged as a concerning threat to LLMs that either use training privileges or system-level access to inject adversarial corruption in model weights. With the emergence of innovative defensive solutions that place system- and algorithm-level checks and corrections in the input and weight spaces, these perturbations are increasingly susceptible to defenses. This work develops a novel perspective on Trojan attacks that generates an attacker-designed model output while leaving no attack traces on the inputs or weights. Such an attack space can be unlocked through corruption of the key-value (KV) cache. In this paper, we introduce CacheTrap, a novel Trojan attack that corrupts the value vectors stored in the KV cache. These vectors capture the dynamic activations for specific token positions and therefore constitute a natural surface for transient, inference-time trigger insertion. The transient nature of these KV values and their dependence on victim input imply additional constraints on our attack, such as a lack of knowledge of the victim's data or domain application, and, consequently, a lack of gradient information. The objective of the proposed CacheTrap is to develop a vulnerable KV bit-searching algorithm so that, once the attack employs the identified bit-flip as a trigger, the model generates targeted behavior, e.g., classifying inputs towards the target class. Moreover, CacheTrap is a data- and gradient-free attack which also has no impact on the model's utility. Our evaluation demonstrates that the proposed attack enables the first successful Trojan attack on LLMs with a single bit flip in the KV cache. In addition, the data-independent nature of the attack ensures that once the attacker identifies the vulnerable bit index, the location remains constant and can be transferred to a wide range of victim tasks/datasets/queries with no overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22612v1">Improving LLM-based Ontology Matching with fine-tuning on synthetic data</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being integrated into various components of Ontology Matching pipelines. This paper investigates the capability of LLMs to perform ontology matching directly on ontology modules and generate the corresponding alignments. Furthermore, it is explored how a dedicated fine-tuning strategy can enhance the model's matching performance in a zero-shot setting. The proposed method incorporates a search space reduction technique to select relevant subsets from both source and target ontologies, which are then used to automatically construct prompts. Recognizing the scarcity of reference alignments for training, a novel LLM-based approach is introduced for generating a synthetic dataset. This process creates a corpus of ontology submodule pairs and their corresponding reference alignments, specifically designed to fine-tune an LLM for the ontology matching task. The proposed approach was evaluated on the Conference, Geolink, Enslaved, Taxon, and Hydrography datasets from the OAEI complex track. The results demonstrate that the LLM fine-tuned on the synthetically generated data exhibits superior performance compared to the non-fine-tuned base model. The key contribution is a strategy that combines automatic dataset generation with fine-tuning to effectively adapt LLMs for ontology matching tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.19676v4">A Survey of LLM-Driven AI Agent Communication: Protocols, Security Risks, and Defense Countermeasures</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 48 pages, 18 figures
    </div>
    <details class="paper-abstract">
      In recent years, Large-Language-Model-driven AI agents have exhibited unprecedented intelligence and adaptability. Nowadays, agents are undergoing a new round of evolution. They no longer act as an isolated island like LLMs. Instead, they start to communicate with diverse external entities, such as other agents and tools, to perform complex tasks. Under this trend, agent communication is regarded as a foundational pillar of the next communication era, and many organizations have intensively begun to design related communication protocols (e.g., Anthropic's MCP and Google's A2A) within the past year. However, this new field exposes significant security hazards, which can cause severe damage to real-world scenarios. To help researchers quickly figure out this promising topic and benefit the future agent communication development, this paper presents a comprehensive survey of agent communication security. More precisely, we present the first clear definition of agent communication. Besides, we propose a framework that categorizes agent communication into three classes and uses a three-layered communication architecture to illustrate how each class works. Next, for each communication class, we dissect related communication protocols and analyze the security risks, illustrating which communication layer the risks arise from. Then, we provide an outlook on the possible defense countermeasures for each risk. In addition, we conduct experiments using MCP and A2A to help readers better understand the novel vulnerabilities brought by agent communication. Finally, we discuss open issues and future directions in this promising research field. We also publish a repository that maintains a list of related papers on https://github.com/theshi-1128/awesome-agent-communication-security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22598v1">LLM-Cave: A benchmark and light environment for large language models reasoning and decision-making system</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 8 pages, 5 figures, ICICN 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) such as ChatGPT o1, ChatGPT o3, and DeepSeek R1 have shown great potential in solving difficult problems. However, current LLM evaluation benchmarks are limited to one-step interactions. Some of the existing sequence decision-making environments, such as TextStarCraftII and LLM-PySC2, are too complicated and require hours of interaction to complete a game. In this paper, we introduce LLM-Cave, a benchmark and light environment for LLM reasoning and decision-making systems. This environment is a classic instance in the era of Symbolism. Artificial intelligence enables the agent to explore the environment and avoid potential losses by reasoning about nearby dangers using partial observable state information. In the experiment, we evaluated the sequential reasoning ability, decision-making performance and computational efficiency of mainstream large language models (LLMs) such as GPT-4o-mini, o1-mini, and DeepSeek-R1. Experiments show that while Deepseek-R1 achieved the highest success rate on complex reasoning tasks, smaller models like 4o-mini significantly narrowed the performance gap on challenges by employing Chain of Speculation and Planner-Critic strategies, at the expense of reduced computational efficiency. This indicates that structured, multi-step reasoning combined with an LLM-based feedback mechanism can substantially enhance an LLM's decision-making capabilities, providing a promising direction for improving reasoning in weaker models and suggesting a new reasoning-centered benchmark for LLM assessment. Our code is open-sourced in https://github.com/puleya1277/CaveEnv.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22584v1">Smarter, not Bigger: Fine-Tuned RAG-Enhanced LLMs for Automotive HIL Testing</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      Hardware-in-the-Loop (HIL) testing is essential for automotive validation but suffers from fragmented and underutilized test artifacts. This paper presents HIL-GPT, a retrieval-augmented generation (RAG) system integrating domain-adapted large language models (LLMs) with semantic retrieval. HIL-GPT leverages embedding fine-tuning using a domain-specific dataset constructed via heuristic mining and LLM-assisted synthesis, combined with vector indexing for scalable, traceable test case and requirement retrieval. Experiments show that fine-tuned compact models, such as \texttt{bge-base-en-v1.5}, achieve a superior trade-off between accuracy, latency, and cost compared to larger models, challenging the notion that bigger is always better. An A/B user study further confirms that RAG-enhanced assistants improve perceived helpfulness, truthfulness, and satisfaction over general-purpose LLMs. These findings provide insights for deploying efficient, domain-aligned LLM-based assistants in industrial HIL environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.19057v2">RELATE: Relation Extraction in Biomedical Abstracts with LLMs and Ontology Constraints</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      Biomedical knowledge graphs (KGs) are vital for drug discovery and clinical decision support but remain incomplete. Large language models (LLMs) excel at extracting biomedical relations, yet their outputs lack standardization and alignment with ontologies, limiting KG integration. We introduce RELATE, a three-stage pipeline that maps LLM-extracted relations to standardized ontology predicates using ChemProt and the Biolink Model. The pipeline includes: (1) ontology preprocessing with predicate embeddings, (2) similarity-based retrieval enhanced with SapBERT, and (3) LLM-based reranking with explicit negation handling. This approach transforms relation extraction from free-text outputs to structured, ontology-constrained representations. On the ChemProt benchmark, RELATE achieves 52% exact match and 94% accuracy@10, and in 2,400 HEAL Project abstracts, it effectively rejects irrelevant associations (0.4%) and identifies negated assertions. RELATE captures nuanced biomedical relationships while ensuring quality for KG augmentation. By combining vector search with contextual LLM reasoning, RELATE provides a scalable, semantically accurate framework for converting unstructured biomedical literature into standardized KGs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22441v1">GEO-Detective: Unveiling Location Privacy Risks in Images with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 15 pages with 7 figures and 12 tables
    </div>
    <details class="paper-abstract">
      Images shared on social media often expose geographic cues. While early geolocation methods required expert effort and lacked generalization, the rise of Large Vision Language Models (LVLMs) now enables accurate geolocation even for ordinary users. However, existing approaches are not optimized for this task. To explore the full potential and associated privacy risks, we present Geo-Detective, an agent that mimics human reasoning and tool use for image geolocation inference. It follows a procedure with four steps that adaptively selects strategies based on image difficulty and is equipped with specialized tools such as visual reverse search, which emulates how humans gather external geographic clues. Experimental results show that GEO-Detective outperforms baseline large vision language models (LVLMs) overall, particularly on images lacking visible geographic features. In country level geolocation tasks, it achieves an improvement of over 11.1% compared to baseline LLMs, and even at finer grained levels, it still provides around a 5.2% performance gain. Meanwhile, when equipped with external clues, GEO-Detective becomes more likely to produce accurate predictions, reducing the "unknown" prediction rate by more than 50.6%. We further explore multiple defense strategies and find that Geo-Detective exhibits stronger robustness, highlighting the need for more effective privacy safeguards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22409v1">NOMAD: A Multi-Agent LLM System for UML Class Diagram Generation from Natural Language Requirements</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly utilised in software engineering, yet their ability to generate structured artefacts such as UML diagrams remains underexplored. In this work we present NOMAD, a cognitively inspired, modular multi-agent framework that decomposes UML generation into a series of role-specialised subtasks. Each agent handles a distinct modelling activity, such as entity extraction, relationship classification, and diagram synthesis, mirroring the goal-directed reasoning processes of an engineer. This decomposition improves interpretability and allows for targeted verification strategies. We evaluate NOMAD through a mixed design: a large case study (Northwind) for in-depth probing and error analysis, and human-authored UML exercises for breadth and realism. NOMAD outperforms all selected baselines, while revealing persistent challenges in fine-grained attribute extraction. Building on these observations, we introduce the first systematic taxonomy of errors in LLM-generated UML diagrams, categorising structural, relationship, and semantic/logical. Finally, we examine verification as a design probe, showing its mixed effects and outlining adaptive strategies as promising directions. Together, these contributions position NOMAD as both an effective framework for UML class diagram generation and a lens onto the broader research challenges of reliable language-to-model workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22402v1">Mapping Clinical Doubt: Locating Linguistic Uncertainty in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 Accepted to AAAI'26 SECURE-AI4H Workshop
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in clinical settings, where sensitivity to linguistic uncertainty can influence diagnostic interpretation and decision-making. Yet little is known about where such epistemic cues are internally represented within these models. Distinct from uncertainty quantification, which measures output confidence, this work examines input-side representational sensitivity to linguistic uncertainty in medical text. We curate a contrastive dataset of clinical statements varying in epistemic modality (e.g., 'is consistent with' vs. 'may be consistent with') and propose Model Sensitivity to Uncertainty (MSU), a layerwise probing metric that quantifies activation-level shifts induced by uncertainty cues. Our results show that LLMs exhibit structured, depth-dependent sensitivity to clinical uncertainty, suggesting that epistemic information is progressively encoded in deeper layers. These findings reveal how linguistic uncertainty is internally represented in LLMs, offering insight into their interpretability and epistemic reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22367v1">SuRe: Surprise-Driven Prioritised Replay for Continual LLM Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      Continual learning, one's ability to adapt to a sequence of tasks without forgetting previously acquired knowledge, remains a major challenge in machine learning and a key gap between artificial and human intelligence. While regularisation and replay perform well in vision, they lag behind multi-task learning for large language models (LLMs), especially at scale with many tasks. We revisit replay and argue that two failure modes drive this gap: selection (what to rehearse) and integration (how to consolidate new knowledge). To address selection, we propose Surprise-prioritised Replay (SuRe), a simple, architecture-agnostic rule that ranks and stores the most surprising (high Negative Log-Likelihood) sequences. SuRe achieves state-of-the-art performance in the Large Number of Tasks (LNT) setting and delivers the best overall average across both Standard CL and LNT benchmarks. To address integration, we add a dual-learner design with fast and slow LoRA adapters merged via an exponential moving average (EMA), enabling rapid adaptation while stabilising long-term knowledge. Combining SuRe with the dual learner yields further gains, including improvements of up to +5 accuracy points on LNT over prior SOTA. Ablation studies confirm that our proposed method remains robust under reduced replay frequency and small buffer size, demonstrating both effectiveness and sample efficiency. Taken together, our results establish replay as a strong baseline for continual LLM fine-tuning and demonstrate that surprise-based selection and slow-weight consolidation are complementary components for mitigating catastrophic forgetting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22354v1">LLM-Based Generalizable Hierarchical Task Planning and Execution for Heterogeneous Robot Teams with Event-Driven Replanning</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 submitted to ICRA 2026
    </div>
    <details class="paper-abstract">
      This paper introduces CoMuRoS (Collaborative Multi-Robot System), a generalizable hierarchical architecture for heterogeneous robot teams that unifies centralized deliberation with decentralized execution, and supports event-driven replanning. A Task Manager LLM interprets natural-language goals, classifies tasks, and allocates subtasks using static rules plus dynamic contexts (task, history, robot and task status, and events).Each robot runs a local LLM that composes executable Python code from primitive skills (ROS2 nodes, policies), while onboard perception (VLMs/image processing) continuously monitors events and classifies them into relevant or irrelevant to the task. Task failures or user intent changes trigger replanning, allowing robots to assist teammates, resume tasks, or request human help. Hardware studies demonstrate autonomous recovery from disruptive events, filtering of irrelevant distractions, and tightly coordinated transport with emergent human-robot cooperation (e.g., multirobot collaborative object recovery success rate: 9/10, coordinated transport: 8/8, human-assisted recovery: 5/5).Simulation studies show intention-aware replanning. A curated textual benchmark spanning 22 scenarios (3 tasks each, around 20 robots) evaluates task allocation, classification, IoU, executability, and correctness, with high average scores (e.g., correctness up to 0.91) across multiple LLMs, a separate replanning set (5 scenarios) achieves 1.0 correctness. Compared with prior LLM-based systems, CoMuRoS uniquely demonstrates runtime, event-driven replanning on physical robots, delivering robust, flexible multi-robot and human-robot collaboration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22333v1">PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 Accepted by ASPLOS'26
    </div>
    <details class="paper-abstract">
      LLM serving is increasingly dominated by decode attention, which is a memory-bound operation due to massive KV cache loading from global memory. Meanwhile, real-world workloads exhibit substantial, hierarchical shared prefixes across requests (e.g., system prompts, tools/templates, RAG). Existing attention implementations fail to fully exploit prefix sharing: *one-query-per-CTA* execution repeatedly loads shared prefix KV cache, while *one-size-fits-all* tiling leaves on-chip resources idle and exacerbates bubbles for uneven KV lengths. These choices amplify memory bandwidth pressure and stall memory-bound decode attention. This paper introduces PAT, a prefix-aware attention kernel implementation for LLM decoding that organizes execution with a pack-forward-merge paradigm. PAT packs queries by shared prefix to reduce repeated memory accesses, runs a customized multi-tile kernel to achieve high resource efficiency. It further applies practical multi-stream forwarding and KV splitting to reduce resource bubbles. The final merge performs online softmax with negligible overhead. We implement PAT as an off-the-shelf plugin for vLLM. Evaluation on both real-world and synthetic workloads shows that PAT reduces attention latency by 67.4% on average and TPOT by 13.6-83.4% under the same configurations against state-of-the-art attention kernels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22312v1">Token-Level Marginalization for Multi-Label LLM Classifiers</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      This paper addresses the critical challenge of deriving interpretable confidence scores from generative language models (LLMs) when applied to multi-label content safety classification. While models like LLaMA Guard are effective for identifying unsafe content and its categories, their generative architecture inherently lacks direct class-level probabilities, which hinders model confidence assessment and performance interpretation. This limitation complicates the setting of dynamic thresholds for content moderation and impedes fine-grained error analysis. This research proposes and evaluates three novel token-level probability estimation approaches to bridge this gap. The aim is to enhance model interpretability and accuracy, and evaluate the generalizability of this framework across different instruction-tuned models. Through extensive experimentation on a synthetically generated, rigorously annotated dataset, it is demonstrated that leveraging token logits significantly improves the interpretability and reliability of generative classifiers, enabling more nuanced content safety moderation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22277v1">TreeCoder: Systematic Exploration and Optimisation of Decoding and Constraints for LLM Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable ability to generate code, yet their outputs often violate syntactic or semantic constraints when guided only through natural language prompts. We introduce TreeCoder, the most general and flexible framework to date for exploring decoding strategies, constraints, and hyperparameters in LLMs, and use it in code generation to enforce correctness and structure during decoding rather than relying on prompt engineering. TreeCoder represents decoding as a tree search over candidate programs, where both decoding strategies and constraint functions - such as style, syntax, execution - are treated as first-class, optimisable components. This design enables systematic exploration and automatic tuning of decoding configurations using standard optimisation techniques. Experiments on the MBPP (Python) and SQL-Spider benchmarks show that TreeCoder consistently improves accuracy across open-source models such as CodeLlama, Mistral and DeepSeek, often outperforming their unconstrained baselines by considerable margins.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22275v1">RecToM: A Benchmark for Evaluating Machine Theory of Mind in LLM-based Conversational Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      Large Language models are revolutionizing the conversational recommender systems through their impressive capabilities in instruction comprehension, reasoning, and human interaction. A core factor underlying effective recommendation dialogue is the ability to infer and reason about users' mental states (such as desire, intention, and belief), a cognitive capacity commonly referred to as Theory of Mind. Despite growing interest in evaluating ToM in LLMs, current benchmarks predominantly rely on synthetic narratives inspired by Sally-Anne test, which emphasize physical perception and fail to capture the complexity of mental state inference in realistic conversational settings. Moreover, existing benchmarks often overlook a critical component of human ToM: behavioral prediction, the ability to use inferred mental states to guide strategic decision-making and select appropriate conversational actions for future interactions. To better align LLM-based ToM evaluation with human-like social reasoning, we propose RecToM, a novel benchmark for evaluating ToM abilities in recommendation dialogues. RecToM focuses on two complementary dimensions: Cognitive Inference and Behavioral Prediction. The former focus on understanding what has been communicated by inferring the underlying mental states. The latter emphasizes what should be done next, evaluating whether LLMs can leverage these inferred mental states to predict, select, and assess appropriate dialogue strategies. Extensive experiments on state-of-the-art LLMs demonstrate that RecToM poses a significant challenge. While the models exhibit partial competence in recognizing mental states, they struggle to maintain coherent, strategic ToM reasoning throughout dynamic recommendation dialogues, particularly in tracking evolving intentions and aligning conversational strategies with inferred mental states.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.24765v3">From Ambiguity to Verdict: A Semiotic-Grounded Multi-Perspective Agent for LLM Logical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      Logical reasoning is a fundamental capability of large language models (LLMs). However, existing studies largely overlook the interplay between logical complexity and semantic complexity, resulting in methods that struggle to address challenging scenarios involving abstract propositions, ambiguous contexts, and conflicting stances, which are central to human reasoning. For this gap, we propose LogicAgent, a semiotic-square-guided framework designed to jointly address logical complexity and semantic complexity. LogicAgent explicitly performs multi-perspective deduction in first-order logic (FOL), while mitigating vacuous reasoning through existential import checks that incorporate a three-valued decision scheme (True, False, Uncertain) to handle boundary cases more faithfully. Furthermore, to overcome the semantic simplicity and low logical complexity of existing datasets, we introduce RepublicQA, a benchmark that reaches college-level difficulty (FKGL = 11.94) and exhibits substantially greater lexical and structural diversity than prior benchmarks. RepublicQA is grounded in philosophical concepts, featuring abstract propositions and systematically organized contrary and contradictory relations, making it the most semantically rich resource for evaluating logical reasoning. Experiments demonstrate that LogicAgent achieves state-of-the-art performance on RepublicQA, with a 6.25% average gain over strong baselines, and generalizes effectively to mainstream logical reasoning benchmarks including ProntoQA, ProofWriter, FOLIO, and ProverQA, achieving an additional 7.05% average gain. These results highlight the strong effectiveness of our semiotic-grounded multi-perspective reasoning in boosting LLMs' logical performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.16607v3">MCTS-SQL: Light-Weight LLMs can Master the Text-to-SQL through Monte Carlo Tree Search</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      Text-to-SQL is a fundamental yet challenging task in the NLP area, aiming at translating natural language questions into SQL queries. While recent advances in large language models have greatly improved performance, most existing approaches depend on models with tens of billions of parameters or costly APIs, limiting their applicability in resource-constrained environments. For real world, especially on edge devices, it is crucial for Text-to-SQL to ensure cost-effectiveness. Therefore, enabling the light-weight models for Text-to-SQL is of great practical significance. However, smaller LLMs often struggle with complicated user instruction, redundant schema linking or syntax correctness. To address these challenges, we propose MCTS-SQL, a novel framework that uses Monte Carlo Tree Search to guide SQL generation through multi-step refinement. Since the light-weight models' weak performance of single-shot prediction, we generate better results through several trials with feedback. However, directly applying MCTS-based methods inevitably leads to significant time and computational overhead. Driven by this issue, we propose a token-level prefix-cache mechanism that stores prior information during iterations, effectively improved the execution speed. Experiments results on the SPIDER and BIRD benchmarks demonstrate the effectiveness of our approach. Using a small open-source Qwen2.5-Coder-1.5B, our method outperforms ChatGPT-3.5. When leveraging a more powerful model Gemini 2.5 to explore the performance upper bound, we achieved results competitive with the SOTA. Our findings demonstrate that even small models can be effectively deployed in practical Text-to-SQL systems with the right strategy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2406.19657v6">LLMEasyQuant: Scalable Quantization for Parallel and Distributed LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 Accepted as International Conference of Computational Optimization 2025 Oral
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) grow in size and deployment scale, quantization has become an essential technique for reducing memory footprint and improving inference efficiency. However, existing quantization toolkits often lack transparency, flexibility, and system-level scalability across GPUs and distributed environments. We present \textbf{LLMEasyQuant}, a modular, system-aware quantization framework designed for efficient, low-bit inference of LLMs on single-node multi-GPU, multi-node, and edge hardware. LLMEasyQuant supports a wide range of quantization methods -- including Symmetric Quantization, ZeroQuant, SmoothQuant, and SimQuant -- with unified interfaces for per-layer calibration, bitwidth assignment, and runtime adaptation. It integrates fused CUDA kernels with NCCL-based distributed synchronization and supports both static and online quantization. Empirical results show that LLMEasyQuant can achieve substantial speedup in GEMM execution, HBM load time, and near-linear multi-GPU scaling. Ablation studies further validate its ability to balance latency, memory, and accuracy under diverse deployment conditions. LLMEasyQuant offers a practical quantization serving system for scalable, hardware-optimized LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.13142v4">Holistic Evaluation of Multimodal LLMs on Spatial Intelligence</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 Codebase: https://github.com/EvolvingLMMs-Lab/EASI/; Leaderboard: https://huggingface.co/spaces/lmms-lab-si/EASI-Leaderboard
    </div>
    <details class="paper-abstract">
      Multimodal models have achieved remarkable progress in recent years. Nevertheless, they continue to exhibit notable limitations in spatial understanding and reasoning, the very capability that anchors artificial general intelligence in the physical world. With the recent release of GPT-5, allegedly the most powerful AI model to date, it is timely to examine where the leading models (GPT, Gemini, Grok, Seed, Qwen, and Intern) stand on the path toward spatial intelligence (SI). We thus propose EASI for holistic Evaluation of multimodAl LLMs on Spatial Intelligence. EASI conceptualizes a comprehensive taxonomy of spatial tasks that unifies existing benchmarks and a growing collection of newly curated ones, enabling systematic evaluation of state-of-the-art models. In this report, we conduct the study across eight key benchmarks, at a cost exceeding ten billion total tokens. Our empirical study then reveals that (1) GPT-5 demonstrates unprecedented strength in SI, yet (2) still falls short of human performance significantly across a broad spectrum of SI-tasks. Moreover, we (3) show that SI-tasks expose greater model capability deficiency than non-SI tasks, to the extent that (4) proprietary models do not exhibit a decisive advantage when facing the most difficult ones. In addition, we conduct a qualitative evaluation across a diverse set of scenarios that are intuitive for humans, yet fail the most advanced multimodal models. EASI is an ongoing community effort: we have open-sourced the EASI codebase that provides a one-stop and reproducible solution with standardized interfaces, integrated protocols and prompts that significantly reduce the friction of configuring and running multiple benchmarks; we have also launched an accompanying EASI leaderboard to provide a continually updated snapshot of model performance across the full SI spectrum, accelerating collective progress toward robust SI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.09936v2">KeepKV: Achieving Periodic Lossless KV Cache Compression for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 14 pages, 20 figures
    </div>
    <details class="paper-abstract">
      Efficient inference of large language models (LLMs) is hindered by an ever-growing key-value (KV) cache, making KV cache compression a critical research direction. Traditional methods selectively evict less important KV cache entries, which leads to information loss and hallucinations. Recently, merging-based strategies have been explored to retain more information by merging KV pairs that would be discarded; however, these existing approaches inevitably introduce inconsistencies in attention distributions before and after merging, causing degraded generation quality. To overcome this challenge, we propose KeepKV, a novel adaptive KV cache merging method designed to preserve performance under strict memory constraints, achieving single-step lossless compression and providing error bounds for multi-step compression. KeepKV introduces the Electoral Votes mechanism that records merging history and adaptively adjusts attention scores. Moreover, it further leverages a novel Zero Inference-Perturbation Merging method, compensating for attention loss resulting from cache merging. Extensive experiments on various benchmarks and LLM architectures demonstrate that KeepKV substantially reduces memory usage while successfully retaining essential context information, achieving over 2x inference throughput improvement and maintaining superior generation quality even with only 10% KV cache budgets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22176v1">Focused Chain-of-Thought: Efficient LLM Reasoning via Structured Input Information</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      Recent large language models achieve strong reasoning performance by generating detailed chain-of-thought traces, but this often leads to excessive token use and high inference latency. Existing efficiency approaches typically focus on model-centric interventions, such as reinforcement learning or supervised fine-tuning, to reduce verbosity. In contrast, we propose a training-free, input-centric approach. Inspired by cognitive psychology, we introduce Focused Chain-of-Thought (F-CoT), which separates information extraction from the reasoning process. F-CoT first organizes the essential information from a query into a concise, structured context and then guides the model to reason exclusively over this context. By preventing attention to irrelevant details, F-CoT naturally produces shorter reasoning paths. On arithmetic word problems, F-CoT reduces generated tokens by 2-3x while maintaining accuracy comparable to standard zero-shot CoT. These results highlight structured input as a simple yet effective lever for more efficient LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22153v1">A Theoretically Grounded Hybrid Ensemble for Reliable Detection of LLM-Generated Text</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 24 pages
    </div>
    <details class="paper-abstract">
      The rapid proliferation of Large Language Models (LLMs) has blurred the line between human and machine authorship, creating practical risks for academic integrity and information reliability. Existing text detectors typically rely on a single methodological paradigm and suffer from poor generalization and high false positive rates (FPR), especially on high-stakes academic text. We propose a theoretically grounded hybrid ensemble that systematically fuses three complementary detection paradigms: (i) a RoBERTa-based transformer classifier for deep semantic feature extraction, (ii) a GPT-2-based probabilistic detector using perturbation-induced likelihood curvature, and (iii) a statistical linguistic feature analyzer capturing stylometric patterns. The core novelty lies in an optimized weighted voting framework, where ensemble weights are learned on the probability simplex to maximize F1-score rather than set heuristically. We provide a bias-variance analysis and empirically demonstrate low inter-model correlation (rho ~ 0.35-0.42), a key condition for variance reduction. Evaluated on a large-scale, multigenerator corpus of 30,000 documents, our system achieves 94.2% accuracy and an AUC of 0.978, with a 35% relative reduction in false positives on academic text. This yields a more reliable and ethically responsible detector for real-world deployment in education and other high-stakes domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20172v2">Beluga: A CXL-Based Memory Architecture for Scalable and Efficient LLM KVCache Management</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 13 pages, accepted by SIGMOD'26
    </div>
    <details class="paper-abstract">
      The rapid increase in LLM model sizes and the growing demand for long-context inference have made memory a critical bottleneck in GPU-accelerated serving systems. Although high-bandwidth memory (HBM) on GPUs offers fast access, its limited capacity necessitates reliance on host memory (CPU DRAM) to support larger working sets such as the KVCache. However, the maximum DRAM capacity is constrained by the limited number of memory channels per CPU socket. To overcome this limitation, current systems often adopt RDMA-based disaggregated memory pools, which introduce significant challenges including high access latency, complex communication protocols, and synchronization overhead. Fortunately, the emerging CXL technology introduces new opportunities in KVCache design. In this paper, we propose Beluga, a novel memory architecture that enables GPUs and CPUs to access a shared, large-scale memory pool through CXL switches. By supporting native load/store access semantics over the CXL fabric, our design delivers near-local memory latency, while reducing programming complexity and minimizing synchronization overhead. We conduct a systematic characterization of a commercial CXL switch-based memory pool and propose a set of design guidelines. Based on Beluga, we design and implement Beluga-KVCache, a system tailored for managing the large-scale KVCache in LLM inference. Beluga-KVCache achieves an 89.6% reduction in Time-To-First-Token (TTFT) and 7.35x throughput improvement in the vLLM inference engine compared to RDMA-based solutions. To the best of our knowledge, Beluga is the first system that enables GPUs to directly access large-scale memory pools through CXL switches, marking a significant step toward low-latency, shared access to vast memory resources by GPUs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2405.05758v2">Exploring the Human-LLM Synergy in Advancing Theory-driven Qualitative Analysis</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 51 pages, 6 figures, accepted by ACM Trans. Comput.-Hum. Interact (TOCHI)
    </div>
    <details class="paper-abstract">
      Qualitative coding is a demanding yet crucial research method in the field of Human-Computer Interaction (HCI). While recent studies have shown the capability of large language models (LLMs) to perform qualitative coding within theoretical frameworks, their potential for collaborative human-LLM discovery and generation of new insights beyond initial theory remains underexplored. To bridge this gap, we proposed CHALET, a novel approach that harnesses the power of human-LLM partnership to advance theory-driven qualitative analysis by facilitating iterative coding, disagreement analysis, and conceptualization of qualitative data. We demonstrated CHALET's utility by applying it to the qualitative analysis of conversations related to mental-illness stigma, using the attribution model as the theoretical framework. Results highlighted the unique contribution of human-LLM collaboration in uncovering latent themes of stigma across the cognitive, emotional, and behavioral dimensions. We discuss the methodological implications of the human-LLM collaborative approach to theory-based qualitative analysis for the HCI community and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22118v1">Statistical Independence Aware Caching for LLM Workflows</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) inference is both expensive and slow. Local caching of responses offers a practical solution to reduce the cost and latency of LLM queries. In research contexts, caching also enhances reproducibility and provides flexibility for experimentation. However, naive reuse of cached responses compromises statistical independence, a critical property for probabilistic workflows. In applications of LLM for code, it underpins performance metrics such as Pass@k and uncertainty estimation, as well as algorithms like program repair loops and retries. Existing LLM caching systems lack ways to enforce statistical independence constraints. To address this, we introduce Mnimi, a cache design pattern that supports modular LLM workflows while ensuring statistical integrity at the component level. Its core innovation lies in encapsulating statistical constraints within the type of LLM references, allowing users to manage and transform these types according to the scope and requirements of their algorithm. We implemented this design pattern in Python using a combination of decorators and iterators over infinite sequences. A case study on SpecFix, an recent automated program specification repair system, highlights how Mnimi improves reproducibility, ease of debugging, time and cost efficiency while preserving statistical correctness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22099v1">Decomposed Trust: Exploring Privacy, Adversarial Robustness, Fairness, and Ethics of Low-Rank LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 14 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have driven major advances across domains, yet their massive size hinders deployment in resource-constrained settings. Model compression addresses this challenge, with low-rank factorization emerging as a particularly effective method for reducing size, memory, and computation while maintaining accuracy. However, while these compressed models boast of benign performance and system-level advantages, their trustworthiness implications remain poorly understood. In this paper, we present the first comprehensive study of how low-rank factorization affects LLM trustworthiness across privacy, adversarial robustness, fairness, and ethical alignment. We evaluate multiple LLMs of different sizes and variants compressed with diverse low-rank algorithms, revealing key insights: (1) low-rank compression preserves or improves training data privacy but weakens PII protection during conversation; (2) adversarial robustness is generally preserved and often enhanced, even under deep compression; (3) ethical reasoning degrades in zero-shot settings but partially recovers with few-shot prompting; (4) fairness declines under compression. Beyond compression, we investigate how model scale and fine-tuning affect trustworthiness, as both are important in low-rank methods. To guide trustworthy compression strategies, we end our paper with a gradient-based attribution analysis to identify which layers in LLMs contribute most to adversarial robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22044v1">Distillability of LLM Security Logic: Predicting Attack Success Rate of Outline Filling Attack via Ranking Regression</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      In the realm of black-box jailbreak attacks on large language models (LLMs), the feasibility of constructing a narrow safety proxy, a lightweight model designed to predict the attack success rate (ASR) of adversarial prompts, remains underexplored. This work investigates the distillability of an LLM's core security logic. We propose a novel framework that incorporates an improved outline filling attack to achieve dense sampling of the model's security boundaries. Furthermore, we introduce a ranking regression paradigm that replaces standard regression and trains the proxy model to predict which prompt yields a higher ASR. Experimental results show that our proxy model achieves an accuracy of 91.1 percent in predicting the relative ranking of average long response (ALR), and 69.2 percent in predicting ASR. These findings confirm the predictability and distillability of jailbreak behaviors, and demonstrate the potential of leveraging such distillability to optimize black-box attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.04116v2">Searching Meta Reasoning Skeleton to Guide LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      Meta reasoning behaviors work as a skeleton to guide large language model (LLM) reasoning, thus help to improve reasoning performance. However, prior researches implement meta reasoning skeleton with manually designed structure, limiting ability to adapt to query-specific requirement and capture intricate logical dependency among reasoning steps. To deal with the challenges, we represent meta reasoning skeleton with directed acyclic graph (DAG) to unify skeletons proposed in prior works and model intricate logical dependency. Then we propose AutoMR, a framework that searches for query-aware meta reasoning skeleton automatically inspired by automated machine learning (AutoML). Specifically, we construct search space based on DAG representation of skeleton and then formulate the search problem. We design a dynamic skeleton sampling algorithm by expanding meta reasoning skeleton along with reasoning context at inference time. This algorithm can derive any meta reasoning skeleton in search space efficiently and adapt skeleton to evolving base reasoning context, thus enable efficient query-aware skeleton search. We conduct experiments on extensive benchmark datasets. Experimental results show that AutoMR achieves better reasoning performance than previous works broadly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.22530v2">Boosting Pointer Analysis With LLM-Enhanced Allocation Function Detection</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      Pointer analysis is foundational for many static analysis tasks, yet its effectiveness is often hindered by imprecise modeling of heap allocations, particularly in C/C++ programs where custom allocation functions (CAFs) are pervasive. Existing approaches largely overlook these custom allocators, leading to coarse aliasing and low analysis precision. In this paper, we present CAFD, a novel and lightweight technique that enhances pointer analysis by automatically detecting side-effect-free custom allocation functions. CAFD employs a hybrid approach: it uses value-flow analysis to detect straightforward wrappers and leverages Large Language Models (LLMs) to reason about more complex allocation patterns with side effects, ensuring that only side-effect-free functions are modeled as allocators. This targeted enhancement enables precise modeling of heap objects at each call site, achieving context-sensitivity-like benefits without significant overhead. We evaluated CAFD on 17 real-world C projects, identifying over 700 CAFs. Integrating CAFD into a baseline pointer analysis yields a 38x increase in modeled heap objects and a 41.5% reduction in alias set sizes, with only 1.4x runtime overhead. Furthermore, the LLM-enhanced pointer analysis improves indirect call resolution and discovers 29 previously undetected memory bugs, including 6 from real-world industrial applications. These results demonstrate that precise modeling of CAFs has the capability to offer a scalable and practical path to improve pointer analysis in large software systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22033v1">Pathology-Aware Prototype Evolution via LLM-Driven Semantic Disambiguation for Multicenter Diabetic Retinopathy Diagnosis</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 ACMMM 2025
    </div>
    <details class="paper-abstract">
      Diabetic retinopathy (DR) grading plays a critical role in early clinical intervention and vision preservation. Recent explorations predominantly focus on visual lesion feature extraction through data processing and domain decoupling strategies. However, they generally overlook domain-invariant pathological patterns and underutilize the rich contextual knowledge of foundation models, relying solely on visual information, which is insufficient for distinguishing subtle pathological variations. Therefore, we propose integrating fine-grained pathological descriptions to complement prototypes with additional context, thereby resolving ambiguities in borderline cases. Specifically, we propose a Hierarchical Anchor Prototype Modulation (HAPM) framework to facilitate DR grading. First, we introduce a variance spectrum-driven anchor prototype library that preserves domain-invariant pathological patterns. We further employ a hierarchical differential prompt gating mechanism, dynamically selecting discriminative semantic prompts from both LVLM and LLM sources to address semantic confusion between adjacent DR grades. Finally, we utilize a two-stage prototype modulation strategy that progressively integrates clinical knowledge into visual prototypes through a Pathological Semantic Injector (PSI) and a Discriminative Prototype Enhancer (DPE). Extensive experiments across eight public datasets demonstrate that our approach achieves pathology-guided prototype evolution while outperforming state-of-the-art methods. The code is available at https://github.com/zhcz328/HAPM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16901v2">R-AVST: Empowering Video-LLMs with Fine-Grained Spatio-Temporal Reasoning in Complex Audio-Visual Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 Accepted by AAAI 2026. Project page: https://github.com/zhlllau/R-AVST
    </div>
    <details class="paper-abstract">
      Recently, rapid advancements have been made in multimodal large language models (MLLMs), especially in video understanding tasks. However, current research focuses on simple video scenarios, failing to reflect the complex and diverse nature of real-world audio-visual events in videos. To bridge this gap, we firstly introduce R-AVST, a dataset for audio-visual reasoning featuring fine-grained spatio-temporal annotations. In constructing this, we design a pipeline consisting of LLM-based key object extraction, automatic spatial annotation and manual quality inspection, resulting in over 5K untrimmed videos with 27K objects across 100 types of audio-visual events. Building on this dataset, we define three core tasks for spatio-temporal reasoning in audio-visual scenes and generate more than 8K high-quality, evenly distributed question-answer pairs to effectively benchmark model performance. To further enhance reasoning, we propose AVST-Zero, a reinforcement learning-based model that avoids intermediate supervision, directly optimizing behavior via carefully designed multi-dimensional rewards. Extensive experiments validate the effectiveness of our R-AVST in advancing audio-visual spatio-temporal reasoning, upon which AVST-Zero demonstrates competitive performance compared to existing models. To the best of our knowledge, R-AVST is the first dataset designed for real-world audio-visual spatio-temporal reasoning, and AVST-Zero offers a novel perspective for tackling future challenges in this domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21998v1">Can Multi-Modal LLMs Provide Live Step-by-Step Task Guidance?</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 Accepted to NeurIPS 2025 (Project page: https://apratimbh.github.io/livecook)
    </div>
    <details class="paper-abstract">
      Multi-modal Large Language Models (LLM) have advanced conversational abilities but struggle with providing live, interactive step-by-step guidance, a key capability for future AI assistants. Effective guidance requires not only delivering instructions but also detecting their successful execution, as well as identifying and alerting users to mistakes, all of which has to happen in real-time. This requires models that are not turn-based, but that can react asynchronously to a video stream, as well as video data showing users performing tasks including mistakes and their corrections. To this end, we introduce Qualcomm Interactive Cooking, a new benchmark and dataset built upon CaptainCook4D, which contains user mistakes during task execution. Our dataset and benchmark features densely annotated, timed instructions and feedback messages, specifically including mistake alerts precisely timestamped to their visual occurrence in the video. We evaluate state-of-the-art multi-modal LLMs on the Qualcomm Interactive Cooking benchmark and introduce LiveMamba, a streaming multi-modal LLM designed for interactive instructional guidance. This work provides the first dedicated benchmark and a strong baseline for developing and evaluating on live, situated coaching.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21989v1">Selecting User Histories to Generate LLM Users for Cold-Start Item Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 12 pages, 15 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in reasoning, generalization, and simulating human-like behavior across a wide range of tasks. These strengths present new opportunities to enhance traditional recommendation systems (RS), especially in the cold-start item scenario where newly introduced items lack interactions. Existing works have used LLMs to address cold-start issues in traditional RS through data augmentation, but they have limitations. One recent work directly addresses this issue by prompting LLMs to generate augmented interaction data between randomly sampled users and cold-start items. Then, they train the traditional RS with augmented data, incorporating collaborative signals for cold-start items. Although they use LLMs to provide cold-start items with feedback, they use partial user histories, which does not allow the LLM to fully emulate the user. Furthermore, randomly selecting users is not optimal for augmentation. To address these challenges, we leverage the LLM as a user and develop a reinforcement learning (RL) framework that trains a policy to select users for augmentation, optimizing for cold-start item performance after augmented training. The policy model learns to select users for cold-start item data augmentation based on their behavioral features and histories. To optimize user selection for cold-start item performance, we employ a policy gradient method that updates the policy in the direction of actions that lead to high rewards. Experiments on Amazon Product Review datasets show substantial gains in cold-start item recall, demonstrating the effectiveness of our method as a scalable, serving-efficient augmentation strategy for modern RS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10507v2">AdvancedIF: Rubric-Based Benchmarking and Reinforcement Learning for Advancing LLM Instruction Following</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Recent progress in large language models (LLMs) has led to impressive performance on a range of tasks, yet advanced instruction following (IF)-especially for complex, multi-turn, and system-prompted instructions-remains a significant challenge. Rigorous evaluation and effective training for such capabilities are hindered by the lack of high-quality, human-annotated benchmarks and reliable, interpretable reward signals. In this work, we introduce AdvancedIF (we will release this benchmark soon), a comprehensive benchmark featuring over 1,600 prompts and expert-curated rubrics that assess LLMs ability to follow complex, multi-turn, and system-level instructions. We further propose RIFL (Rubric-based Instruction-Following Learning), a novel post-training pipeline that leverages rubric generation, a finetuned rubric verifier, and reward shaping to enable effective reinforcement learning for instruction following. Extensive experiments demonstrate that RIFL substantially improves the instruction-following abilities of LLMs, achieving a 6.7% absolute gain on AdvancedIF and strong results on public benchmarks. Our ablation studies confirm the effectiveness of each component in RIFL. This work establishes rubrics as a powerful tool for both training and evaluating advanced IF in LLMs, paving the way for more capable and reliable AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21638v1">Aligning LLMs Toward Multi-Turn Conversational Outcomes Using Iterative PPO</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 12 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Optimizing large language models (LLMs) for multi-turn conversational outcomes remains a significant challenge, especially in goal-oriented settings like AI marketing or sales agents who facilitate transactions via messaging platforms. The difficulty stems from sparse, long-horizon rewards and the discrepancy between response-level planning and token-level generation. In this technical note, we propose a formal reduction of the multi-turn RL problem into a sequence of single-turn RLHF-style problems. This is achieved by setting a learned multi-turn Q-function as the reward model for the single-turn problem. We demonstrate and prove a key insight: solving this single-turn RL problem with standard token-level PPO is equivalent to a policy improvement step within the multi-turn problem. This insight naturally leads to Iterative PPO, a batch online policy iteration algorithm that alternates between fitting Q-functions from logged conversation trajectories and improving the policy. A major practical advantage is that Iterative PPO directly leverages stable, off-the-shelf single-turn RLHF tools, making it straightforward to implement. Our method occupies a middle ground between fully online and fully offline approaches, retaining the adaptability of online updates while gaining the stability benefits of offline training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21624v1">TAGFN: A Text-Attributed Graph Dataset for Fake News Detection in the Age of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 Preprint. Under review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently revolutionized machine learning on text-attributed graphs, but the application of LLMs to graph outlier detection, particularly in the context of fake news detection, remains significantly underexplored. One of the key challenges is the scarcity of large-scale, realistic, and well-annotated datasets that can serve as reliable benchmarks for outlier detection. To bridge this gap, we introduce TAGFN, a large-scale, real-world text-attributed graph dataset for outlier detection, specifically fake news detection. TAGFN enables rigorous evaluation of both traditional and LLM-based graph outlier detection methods. Furthermore, it facilitates the development of misinformation detection capabilities in LLMs through fine-tuning. We anticipate that TAGFN will be a valuable resource for the community, fostering progress in robust graph-based outlier detection and trustworthy AI. The dataset is publicly available at https://huggingface.co/datasets/kayzliu/TAGFN and our code is available at https://github.com/kayzliu/tagfn.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21613v1">Beyond URLs: Metadata Diversity and Position for Efficient LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Incorporating metadata in Large Language Models (LLMs) pretraining has recently emerged as a promising approach to accelerate training. However prior work highlighted only one useful signal-URLs, leaving open the question of whether other forms of metadata could yield greater benefits. In this study, we investigate a wider range of metadata types and find other types of metadata, such as fine-grained indicators of document quality that can also accelerate pretraining when prepended. We identify a common feature among effective metadata: they encode information at a finer granularity. We further introduce metadata appending as a means of improving training efficiency, where predicting an appropriate metadata as auxiliary task can help speed up pretraining. In addition, learnable meta-tokens trained with masked loss can recover part of the speedup by inducing quality-aware latent structure. Using probing, we analyze latent representations to understand how metadata shapes learning. Together, these results yield practical guidelines for integrating metadata to improve both the efficiency and effectiveness of LLM pretraining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21594v1">Visualizing LLM Latent Space Geometry Through Dimensionality Reduction</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 24 pages, 16 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) achieve state-of-the-art results across many natural language tasks, but their internal mechanisms remain difficult to interpret. In this work, we extract, process, and visualize latent state geometries in Transformer-based language models through dimensionality reduction. We capture layerwise activations at multiple points within Transformer blocks and enable systematic analysis through Principal Component Analysis (PCA) and Uniform Manifold Approximation (UMAP). We demonstrate experiments on GPT-2 and LLaMa models, where we uncover interesting geometric patterns in latent space. Notably, we identify a clear separation between attention and MLP component outputs across intermediate layers, a pattern not documented in prior work to our knowledge. We also characterize the high norm of latent states at the initial sequence position and visualize the layerwise evolution of latent states. Additionally, we demonstrate the high-dimensional helical structure of GPT-2's positional embeddings, the sequence-wise geometric patterns in LLaMa, and experiment with repeating token sequences. We aim to support systematic analysis of Transformer internals with the goal of enabling further reproducible interpretability research. We make our code available at https://github.com/Vainateya/Feature_Geometry_Visualization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10234v2">Lost in Serialization: Invariance and Generalization of LLM Graph Reasoners</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 AAAI 2026 Workshop on Graphs and more Complex Structures For Learning and Reasoning (GCLR). Version 2 fixes typos in author name and Figure 1
    </div>
    <details class="paper-abstract">
      While promising, graph reasoners based on Large Language Models (LLMs) lack built-in invariance to symmetries in graph representations. Operating on sequential graph serializations, LLMs can produce different outputs under node reindexing, edge reordering, or formatting changes, raising robustness concerns. We systematically analyze these effects, studying how fine-tuning impacts encoding sensitivity as well generalization on unseen tasks. We propose a principled decomposition of graph serializations into node labeling, edge encoding, and syntax, and evaluate LLM robustness to variations of each of these factors on a comprehensive benchmarking suite. We also contribute a novel set of spectral tasks to further assess generalization abilities of fine-tuned reasoners. Results show that larger (non-fine-tuned) models are more robust. Fine-tuning reduces sensitivity to node relabeling but may increase it to variations in structure and format, while it does not consistently improve performance on unseen tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2402.14746v5">Scaling Efficient LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Recent LLMs have hundreds of billions of parameters consuming vast resources. Furthermore, the so called "AI scaling law" for transformers suggests that the number of parameters must scale linearly with the size of the data. In response, we inquire into efficient LLMs, i.e. those with the fewest parameters that achieve the desired accuracy on a training corpus. Specifically, by comparing theoretical and empirical estimates of the Kullback-Leibler divergence, we derive a natural AI scaling law that the number of parameters in an efficient LLM scales as $D^γ$ where $D$ is the size of the training data and $ γ\in [0.44, 0.72]$, suggesting the existence of more efficient architectures. Against this backdrop, we propose recurrent transformers, combining the efficacy of transformers with the efficiency of recurrent networks, progressively applying a single transformer layer to a fixed-width sliding window across the input sequence. Recurrent transformers (a) run in linear time in the sequence length, (b) are memory-efficient and amenable to parallel processing in large batches, (c) learn to forget history for language tasks, or accumulate history for long range tasks like copy and selective copy, and (d) are amenable to curriculum training to overcome vanishing gradients. In our experiments, we find that recurrent transformers perform favorably on benchmark tests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2405.17846v2">Safety Control of Service Robots with LLMs and Embodied Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Safety limitations in service robotics across various industries have raised significant concerns about the need for robust mechanisms ensuring that robots adhere to safe practices, thereby preventing actions that might harm humans or cause property damage. Despite advances, including the integration of Knowledge Graphs (KGs) with Large Language Models (LLMs), challenges in ensuring consistent safety in autonomous robot actions persist. In this paper, we propose a novel integration of Large Language Models with Embodied Robotic Control Prompts (ERCPs) and Embodied Knowledge Graphs (EKGs) to enhance the safety framework for service robots. ERCPs are designed as predefined instructions that ensure LLMs generate safe and precise responses. These responses are subsequently validated by EKGs, which provide a comprehensive knowledge base ensuring that the actions of the robot are continuously aligned with safety protocols, thereby promoting safer operational practices in varied contexts. Our experimental setup involved diverse real-world tasks, where robots equipped with our framework demonstrated significantly higher compliance with safety standards compared to traditional methods. This integration fosters secure human-robot interactions and positions our methodology at the forefront of AI-driven safety innovations in service robotics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19218v2">Adversarial Attack-Defense Co-Evolution for LLM Safety Alignment via Tree-Group Dual-Aware Search and Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have developed rapidly in web services, delivering unprecedented capabilities while amplifying societal risks. Existing works tend to focus on either isolated jailbreak attacks or static defenses, neglecting the dynamic interplay between evolving threats and safeguards in real-world web contexts. To mitigate these challenges, we propose ACE-Safety (Adversarial Co-Evolution for LLM Safety), a novel framework that jointly optimize attack and defense models by seamlessly integrating two key innovative procedures: (1) Group-aware Strategy-guided Monte Carlo Tree Search (GS-MCTS), which efficiently explores jailbreak strategies to uncover vulnerabilities and generate diverse adversarial samples; (2) Adversarial Curriculum Tree-aware Group Policy Optimization (AC-TGPO), which jointly trains attack and defense LLMs with challenging samples via curriculum reinforcement learning, enabling robust mutual improvement. Evaluations across multiple benchmarks demonstrate that our method outperforms existing attack and defense approaches, and provides a feasible pathway for developing LLMs that can sustainably support responsible AI ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21401v1">Can LLMs extract human-like fine-grained evidence for evidence-based fact-checking?</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Misinformation frequently spreads in user comments under online news articles, highlighting the need for effective methods to detect factually incorrect information. To strongly support or refute claims extracted from such comments, it is necessary to identify relevant documents and pinpoint the exact text spans that justify or contradict each claim. This paper focuses on the latter task -- fine-grained evidence extraction for Czech and Slovak claims. We create new dataset, containing two-way annotated fine-grained evidence created by paid annotators. We evaluate large language models (LLMs) on this dataset to assess their alignment with human annotations. The results reveal that LLMs often fail to copy evidence verbatim from the source text, leading to invalid outputs. Error-rate analysis shows that the {llama3.1:8b model achieves a high proportion of correct outputs despite its relatively small size, while the gpt-oss-120b model underperforms despite having many more parameters. Furthermore, the models qwen3:14b, deepseek-r1:32b, and gpt-oss:20b demonstrate an effective balance between model size and alignment with human annotations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21334v1">Emergent Lexical Semantics in Neural Language Models: Testing Martin's Law on LLM-Generated Text</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 paper draft
    </div>
    <details class="paper-abstract">
      We present the first systematic investigation of Martin's Law - the empirical relationship between word frequency and polysemy - in text generated by neural language models during training. Using DBSCAN clustering of contextualized embeddings as an operationalization of word senses, we analyze four Pythia models (70M-1B parameters) across 30 training checkpoints. Our results reveal a non-monotonic developmental trajectory: Martin's Law emerges around checkpoint 100, reaches peak correlation (r > 0.6) at checkpoint 104, then degrades by checkpoint 105. Smaller models (70M, 160M) experience catastrophic semantic collapse at late checkpoints, while larger models (410M, 1B) show graceful degradation. The frequency-specificity trade-off remains stable (r $\approx$ -0.3) across all models. These findings suggest that compliance with linguistic regularities in LLM-generated text is not monotonically increasing with training, but instead follows a balanced trajectory with an optimal semantic window. This work establishes a novel methodology for evaluating emergent linguistic structure in neural language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19123v2">Facilitating the Integration of LLMs Into Online Experiments With Simple Chat</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly prevalent, understanding human-LLM interactions is emerging as a central priority in psychological research. Online experiments offer an efficient means to study human-LLM interactions, yet integrating LLMs into established survey platforms remains technically demanding, particularly when aiming for ecologically valid, real-time conversational experiences with strong experimental control. We introduce Simple Chat, an open-source, research-focused chat interface that streamlines LLM integration for platforms such as Qualtrics, oTree, and LimeSurvey, while presenting a unified participant experience across conditions. Simple Chat connects to both commercial providers and open-weights models, supports streaming responses to preserve conversational flow, and offers an administrative interface for fine-grained control of prompts and interface features. By reducing technical barriers, standardizing interfaces, and improving participant experience, Simple Chat helps advance the study of human-LLM interaction. In this article, we outline Simple Chat's key features, provide a step-by-step tutorial, and demonstrate its utility through two illustrative case studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21322v1">TALES: A Taxonomy and Analysis of Cultural Representations in LLM-generated Stories</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Millions of users across the globe turn to AI chatbots for their creative needs, inviting widespread interest in understanding how such chatbots represent diverse cultures. At the same time, evaluating cultural representations in open-ended tasks remains challenging and underexplored. In this work, we present TALES, an evaluation of cultural misrepresentations in LLM-generated stories for diverse Indian cultural identities. First, we develop TALES-Tax, a taxonomy of cultural misrepresentations by collating insights from participants with lived experiences in India through focus groups (N=9) and individual surveys (N=15). Using TALES-Tax, we evaluate 6 models through a large-scale annotation study spanning 2,925 annotations from 108 annotators with lived cultural experience from across 71 regions in India and 14 languages. Concerningly, we find that 88\% of the generated stories contain one or more cultural inaccuracies, and such errors are more prevalent in mid- and low-resourced languages and stories based in peri-urban regions in India. Lastly, we transform the annotations into TALES-QA, a standalone question bank to evaluate the cultural knowledge of foundational models. Through this evaluation, we surprisingly discover that models often possess the requisite cultural knowledge despite generating stories rife with cultural misrepresentations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21270v1">Multi-Reward GRPO for Stable and Prosodic Single-Codebook TTS LLMs at Scale</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 4 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have transformed text-to-speech (TTS) synthesis, inspiring autoregressive frameworks that represent speech as sequences of discrete codec tokens. Among them, single-codebook TTS LLMs have emerged as compact and streamable architectures that jointly model semantic and acoustic integration. However, despite their efficiency, these models often exhibit unstable prosody, speaker drift, and degraded naturalness. To address these issues, we propose a multi-reward Group Relative Policy Optimization (GRPO) framework that directly optimizes the token generation policy of single-codebook TTS LLMs. Beyond standard intelligibility and speaker similarity objectives, our design integrates three rule-based rewards: a length penalty for duration consistency, an entropy regularization reward for decoding stability, and an LLM-annotated prosody alignment reward that explicitly supervises rhythm. In this prosody reward, an external reasoning LLM predicts multiple plausible pause structures via in-context learning, providing a human-preference-aligned supervisory signal for GRPO training. To assess universality, we further attach a flow-matching (FM) decoder on top of the GRPO-optimized AR backbone and observe consistent additional gains, indicating that our reinforcement optimization enhances the intrinsic AR policy. We further conduct a scalability analysis across data sizes and model scales, revealing that the proposed method consistently enhances prosodic stability, speaker similarity, and overall speech naturalness in single-codebook TTS LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17041v2">CLLMRec: LLM-powered Cognitive-Aware Concept Recommendation via Semantic Alignment and Prerequisite Knowledge Distillation</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      The growth of Massive Open Online Courses (MOOCs) presents significant challenges for personalized learning, where concept recommendation is crucial. Existing approaches typically rely on heterogeneous information networks or knowledge graphs to capture conceptual relationships, combined with knowledge tracing models to assess learners' cognitive states. However, these methods face significant limitations due to their dependence on high-quality structured knowledge graphs, which are often scarce in real-world educational scenarios. To address this fundamental challenge, this paper proposes CLLMRec, a novel framework that leverages Large Language Models through two synergistic technical pillars: Semantic Alignment and Prerequisite Knowledge Distillation. The Semantic Alignment component constructs a unified representation space by encoding unstructured textual descriptions of learners and concepts. The Prerequisite Knowledge Distillation paradigm employs a teacher-student architecture, where a large teacher LLM (implemented as the Prior Knowledge Aware Component) extracts conceptual prerequisite relationships from its internalized world knowledge and distills them into soft labels to train an efficient student ranker. Building upon these foundations, our framework incorporates a fine-ranking mechanism that explicitly models learners' real-time cognitive states through deep knowledge tracing, ensuring recommendations are both structurally sound and cognitively appropriate. Extensive experiments on two real-world MOOC datasets demonstrate that CLLMRec significantly outperforms existing baseline methods across multiple evaluation metrics, validating its effectiveness in generating truly cognitive-aware and personalized concept recommendations without relying on explicit structural priors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21218v1">Can Finetuing LLMs on Small Human Samples Increase Heterogeneity, Alignment, and Belief-Action Coherence?</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      There is ongoing debate about whether large language models (LLMs) can serve as substitutes for human participants in survey and experimental research. While recent work in fields such as marketing and psychology has explored the potential of LLM-based simulation, a growing body of evidence cautions against this practice: LLMs often fail to align with real human behavior, exhibiting limited diversity, systematic misalignment for minority subgroups, insufficient within-group variance, and discrepancies between stated beliefs and actions. This study examines an important and distinct question in this domain: whether fine-tuning on a small subset of human survey data, such as that obtainable from a pilot study, can mitigate these issues and yield realistic simulated outcomes. Using a behavioral experiment on information disclosure, we compare human and LLM-generated responses across multiple dimensions, including distributional divergence, subgroup alignment, belief-action coherence, and the recovery of regression coefficients. We find that fine-tuning on small human samples substantially improves heterogeneity, alignment, and belief-action coherence relative to the base model. However, even the best-performing fine-tuned models fail to reproduce the regression coefficients of the original study, suggesting that LLM-generated data remain unsuitable for replacing human participants in formal inferential analyses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20403v2">LLMs for Automated Unit Test Generation and Assessment in Java: The AgoneTest Framework</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 Accepted at 40th IEEE/ACM International Conference on Automated Software Engineering
    </div>
    <details class="paper-abstract">
      Unit testing is an essential but resource-intensive step in software development, ensuring individual code units function correctly. This paper introduces AgoneTest, an automated evaluation framework for Large Language Model-generated (LLM) unit tests in Java. AgoneTest does not aim to propose a novel test generation algorithm; rather, it supports researchers and developers in comparing different LLMs and prompting strategies through a standardized end-to-end evaluation pipeline under realistic conditions. We introduce the Classes2Test dataset, which maps Java classes under test to their corresponding test classes, and a framework that integrates advanced evaluation metrics, such as mutation score and test smells, for a comprehensive assessment. Experimental results show that, for the subset of tests that compile, LLM-generated tests can match or exceed human-written tests in terms of coverage and defect detection. Our findings also demonstrate that enhanced prompting strategies contribute to test quality. AgoneTest clarifies the potential of LLMs in software testing and offers insights for future improvements in model design, prompt engineering, and testing practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21140v1">How to Correctly Report LLM-as-a-Judge Evaluations</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as evaluators in lieu of humans. While scalable, their judgments are noisy due to imperfect specificity and sensitivity of LLMs, leading to biased accuracy estimates. Although bias-correction methods exist, they are underutilized in LLM research and typically assume exact knowledge of the model's specificity and sensitivity. Furthermore, in general we only have estimates of these values and it is not well known how to properly construct confidence intervals using only estimates. This work presents a simple plug-in framework that corrects such bias and constructs confidence intervals reflecting uncertainty from both test and calibration dataset, enabling practical and statistically sound LLM-based evaluation. Additionally, to reduce uncertainty in the accuracy estimate, we introduce an adaptive algorithm that efficiently allocates calibration sample sizes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.04231v2">Empowering Time Series Forecasting with LLM-Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) powered agents have emerged as effective planners for Automated Machine Learning (AutoML) systems. While most existing AutoML approaches focus on automating feature engineering and model architecture search, recent studies in time series forecasting suggest that lightweight models can often achieve state-of-the-art performance. This observation led us to explore improving data quality, rather than model architecture, as a potentially fruitful direction for AutoML on time series data. We propose DCATS, a Data-Centric Agent for Time Series. DCATS leverages metadata accompanying time series to clean data while optimizing forecasting performance. We evaluated DCATS using four time series forecasting models on a large-scale traffic volume forecasting dataset. Results demonstrate that DCATS achieves an average 6% error reduction across all tested models and time horizons, highlighting the potential of data-centric approaches in AutoML for time series forecasting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.02833v4">Attack via Overfitting: 10-shot Benign Fine-tuning to Jailbreak LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 Published as a conference paper at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Despite substantial efforts in safety alignment, recent research indicates that Large Language Models (LLMs) remain highly susceptible to jailbreak attacks. Among these attacks, finetuning-based ones that compromise LLMs' safety alignment via fine-tuning stand out due to its stable jailbreak performance. In particular, a recent study indicates that fine-tuning with as few as 10 harmful question-answer (QA) pairs can lead to successful jailbreaking across various harmful questions. However, such malicious fine-tuning attacks are readily detectable and hence thwarted by moderation models. In this paper, we demonstrate that LLMs can be jailbroken by fine-tuning with only 10 benign QA pairs; our attack exploits the increased sensitivity of LLMs to fine-tuning data after being overfitted. Specifically, our fine-tuning process starts with overfitting an LLM via fine-tuning with benign QA pairs involving identical refusal answers. Further fine-tuning is then performed with standard benign answers, causing the overfitted LLM to forget the refusal attitude and thus provide compliant answers regardless of the harmfulness of a question. We implement our attack on the ten LLMs and compare it with five existing baselines. Experiments demonstrate that our method achieves significant advantages in both attack effectiveness and attack stealth. Our findings expose previously unreported security vulnerabilities in current LLMs and provide a new perspective on understanding how LLMs' security is compromised, even with benign fine-tuning. Our code is available at https://github.com/ZHIXINXIE/tenBenign.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19933v2">Failure Modes in LLM Systems: A System-Level Taxonomy for Reliable AI Applications</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are being rapidly integrated into decision-support tools, automation workflows, and AI-enabled software systems. However, their behavior in production environments remains poorly understood, and their failure patterns differ fundamentally from those of traditional machine learning models. This paper presents a system-level taxonomy of fifteen hidden failure modes that arise in real-world LLM applications, including multi-step reasoning drift, latent inconsistency, context-boundary degradation, incorrect tool invocation, version drift, and cost-driven performance collapse. Using this taxonomy, we analyze the growing gap in evaluation and monitoring practices: existing benchmarks measure knowledge or reasoning but provide little insight into stability, reproducibility, drift, or workflow integration. We further examine the production challenges associated with deploying LLMs - including observability limitations, cost constraints, and update-induced regressions - and outline high-level design principles for building reliable, maintainable, and cost-aware LLM systems. Finally, we outline high-level design principles for building reliable, maintainable, and cost-aware LLM-based systems. By framing LLM reliability as a system-engineering problem rather than a purely model-centric one, this work provides an analytical foundation for future research on evaluation methodology, AI system robustness, and dependable LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21089v1">MLPMoE: Zero-Shot Architectural Metamorphosis of Dense LLM MLPs into Static Mixture-of-Experts</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are predominantly deployed as dense transformers, where every parameter in every feed-forward block is activated for every token. While architecturally simple, this is computationally inefficient, since inference costs scale linearly with parameter count. Recent upcycling methods such as MoEfication, CMoE, ToMoE, and MoORE reveal that much of the useful computation lives in sparse, semi-modular substructures inside dense feed-forward networks, but these approaches typically rely on clustering, activation profiling, singular value decomposition, or custom routing that requires calibration data. This paper introduces MLPMoE (MLP Mixture-of-Experts), a training-free, deterministic transformation that restructures the dense MLP in transformer blocks into a static, high-cardinality mixture of experts. The transformation uses simple tensor slicing and summation, reinterpreting the algebra of tensor parallelism as a topological conversion rather than a distributed training pattern. We further introduce Fractal Fade (differential branch sparsity) and Compensated Pruning (variance-preserving branch reduction) as lightweight mechanisms for structured sparsity. On Qwen2.5-0.5B-Instruct and DeepSeek-R1-Distill-Llama-8B, the zero-shot MLPMoE transform changes a proxy perplexity metric by less than 0.05 percent while keeping the parameter count effectively constant. On the 8B model, differential sparsity removes about 20 percent of MLP parameters while keeping perplexity within about 2 percent of the dense baseline. The method operates entirely post hoc on existing checkpoints and does not require gradients, calibration sets, or router training. Code is available at https://gist.github.com/iwallarm/fc2ef1eddf226ca7814f9e5e2ae9bad1
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21075v1">Aligning LLMs with Biomedical Knowledge using Balanced Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Effective post-training is essential to align Large Language Models (LLMs) with specialized biomedical knowledge to accelerate life science research. However, current approaches face significant limitations. First, biomedical reasoning involves intricate mechanisms often represented by sparse textual data. Standard Supervised Fine-Tuning (SFT) tends to overfit to surface-level instruction patterns without effectively internalizing this fragmented scientific knowledge. Second, Reinforcement Learning (RL) is impractical for this domain, as defining meaningful rewards often necessitates prohibitive experimental validation (e.g., wet-lab verification of drug responses), rendering real-time feedback unfeasible. We propose Balanced Fine-Tuning (BFT), an efficient post-training method designed to learn complex reasoning from sparse data without external reward signals. BFT operates through a two-layer weighting mechanism: 1. At the token level, it scales loss via prediction probabilities to stabilize gradients and prevent overfitting; 2. At the sample level, it uses "minimum group confidence" to adaptively enhance the learning of hard samples. Experiments demonstrate that BFT significantly outperforms SFT. In medical tasks, it enables LLMs to acquire knowledge that SFT misses. In biological tasks, BFT-based LLMs surpass GeneAgent (an accurate agent for biology analysis) in biological process reasoning. Moreover, the text embeddings generated by BFT can be directly applied to downstream tasks, such as gene interaction and single-cell perturbation response prediction. These results indicate that BFT facilitates broad applications of LLMs in biomedical research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21056v1">A Unified Understanding of Offline Data Selection and Online Self-refining Generation for Post-training LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Offline data selection and online self-refining generation, which enhance the data quality, are crucial steps in adapting large language models (LLMs) to specific downstream tasks. We tackle offline data selection and online self-refining generations through an optimization perspective. Specifically, bilevel data selection is used for offline data selection with respect to the validation dataset, and we treat online self-refining generation as a model adaptation step of selecting the model trained on current responses that best fits the validation data. Our framework offers a unified understanding of offline data selection and self-refining generation by assigning a learned data weight to each question and response, either explicitly or implicitly. For the first time, we theoretically demonstrate the effectiveness of the bilevel data selection framework and demonstrate its performance gains over unfiltered direct mixing baselines. By combining offline data with validation-weighted online generations, our method enhances fine-tuning performance. Experiments on quality enhancement and safety-aware LLM fine-tuning validate its effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21050v1">Breaking the Safety-Capability Tradeoff: Reinforcement Learning with Verifiable Rewards Maintains Safety Guardrails in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 AAAI-26 Workshop on Post-AI Formal Methods
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) for downstream tasks typically exhibit a fundamental safety-capability tradeoff, where improving task performance degrades safety alignment even on benign datasets. This degradation persists across standard approaches including supervised finetuning (SFT) and reinforcement learning from human feedback (RLHF). While reinforcement learning with verifiable rewards (RLVR) has emerged as a promising alternative that optimizes models on objectively measurable tasks, its safety implications remain unexplored. We present the first comprehensive theoretical and empirical analysis of safety properties in RLVR. Theoretically, we derive upper bounds on safety drift under KL-constrained optimization and prove conditions under which safety degradation is eliminated. Empirically, we conduct extensive experiments across five adversarial safety benchmarks, demonstrating that RLVR can simultaneously enhance reasoning capabilities while maintaining or improving safety guardrails. Our comprehensive ablation studies examine the effects of optimization algorithms, model scale, and task domains. Our findings challenge the prevailing assumption of an inevitable safety capability trade-off, and establish that a specific training methodology can achieve both objectives simultaneously, providing insights for the safe deployment of reasoning-capable LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21038v1">Semantic Anchors in In-Context Learning: Why Small LLMs Cannot Flip Their Labels</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 13 pages total (7 pages main text, 3 pages references, 3 pages appendix), 2 figures, 14 tables. Code available at https://github.com/AnanthaPadmanaban-KrishnaKumar/semantic-anchors-icl
    </div>
    <details class="paper-abstract">
      Can in-context learning (ICL) override pre-trained label semantics, or does it merely refine an existing semantic backbone? We address this question by treating LLMs as prompt-induced classifiers and contrasting their behavior under \emph{natural} demonstrations (with correct labels) and \emph{inverted} demonstrations (systematically flipping label meanings). We decompose ICL behavior into three alignment metrics (truth, prior, and prompt alignment) and introduce a semantic override rate, defined as correctness under flipped semantics. Across eight classification tasks and eight open-source LLMs (1--12B parameters), we find consistent evidence for a semantic anchor view. With natural demonstrations, ICL improves accuracy while maintaining strong prior alignment; most correct predictions coincide with zero-shot behavior, even when the prior is weak. With inverted demonstrations, models cannot learn coherent anti-semantic classifiers: prompt alignment increases only by sacrificing accuracy, and semantic override rates remain exactly zero in our few-shot 1--12B setting. Rather than flexibly remapping label meanings, ICL primarily adjusts how inputs project onto stable semantic directions learned during pre-training, clarifying fundamental limits of few-shot prompting and suggesting that overriding label semantics at these scales requires interventions beyond ICL. All code is available at: https://github.com/AnanthaPadmanaban-KrishnaKumar/semantic-anchors-icl.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21037v1">LOOM: Personalized Learning Informed by Daily LLM Conversations Toward Long-Term Mastery via a Dynamic Learner Memory Graph</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 Accepted to the PerFM Workshop at AAAI 2026
    </div>
    <details class="paper-abstract">
      Foundation models are increasingly used to personalize learning, yet many systems still assume fixed curricula or coarse progress signals, limiting alignment with learners' day-to-day needs. At the other extreme, lightweight incidental systems offer flexible, in-the-moment content but rarely guide learners toward mastery. Prior work privileges either continuity (maintaining a plan across sessions) or initiative (reacting to the moment), not both, leaving learners to navigate the trade-off between recency and trajectory-immediate relevance versus cumulative, goal-aligned progress. We present LOOM, an agentic pipeline that infers evolving learner needs from recent LLM conversations and a dynamic learner memory graph, then assembles coherent learning materials personalized to the learner's current needs, priorities, and understanding. These materials link adjacent concepts and surface gaps as tightly scoped modules that cumulatively advance broader goals, providing guidance and sustained progress while remaining responsive to new interests. We describe LOOM's end-to-end architecture and working prototype, including conversation summarization, topic planning, course generation, and graph-based progress tracking. In a formative study with ten participants, users reported that LOOM's generated lessons felt relevant to their recent activities and helped them recognize knowledge gaps, though they also highlighted needs for greater consistency and control. We conclude with design implications for more robust, mixed-initiative learning pipelines that integrate structured learner modelling with everyday LLM interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21033v1">Towards Trustworthy Legal AI through LLM Agents and Formal Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      The rationality of law manifests in two forms: substantive rationality, which concerns the fairness or moral desirability of outcomes, and formal rationality, which requires legal decisions to follow explicitly stated, general, and logically coherent rules. Existing LLM-based systems excel at surface-level text analysis but lack the guarantees required for principled jurisprudence. We introduce L4M, a novel framework that combines adversarial LLM agents with SMT-solver-backed proofs to unite the interpretive flexibility of natural language with the rigor of symbolic verification. The pipeline consists of three phases: (1) Statute Formalization, where domain-specific prompts convert legal provisions into logical formulae; (2) Dual Fact and Statute Extraction, in which prosecutor- and defense-aligned LLMs independently map case narratives to fact tuples and statutes, ensuring role isolation; and (3) Solver-Centric Adjudication, where an autoformalizer compiles both parties' arguments into logic constraints, and unsat cores trigger iterative self-critique until a satisfiable formula is achieved, which is then verbalized by a Judge-LLM into a transparent verdict and optimized sentence. Experimental results on public benchmarks show that our system surpasses advanced LLMs including GPT-o4-mini, DeepSeek-V3, and Claude 4 as well as state-of-the-art Legal AI baselines, while providing rigorous and explainable symbolic justifications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21022v1">Lightweight Model Editing for LLMs to Correct Deprecated API Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Pre-trained or fine-tuned on large code corpora, Large Language Models (LLMs) have demonstrated strong performance in code completion tasks. However, their embedded knowledge is constrained by the timeliness of training data, which often includes code using deprecated APIs. Consequently, LLMs frequently generate deprecated APIs that will no longer be supported in future versions of third-party libraries. While retraining LLMs on updated codebases could refresh their API knowledge, this approach is computationally expensive. Recently, lightweight model editing methods have emerged to efficiently correct specific knowledge in LLMs. However, it remains unclear whether these methods can effectively update deprecated API knowledge and enable edited models to generate up-to-date APIs. To address this gap, we conduct the first systematic study applying 10 state-of-the-art model editing techniques to update deprecated API knowledge in three LLMs: Qwen2.5-Coder, StarCoder2, and DeepSeek-Coder. We introduce EDAPIBench, a dedicated benchmark featuring over 70 deprecated APIs from 8 popular Python libraries, with more than 3,000 editing instances. Our results show that the parameter-efficient fine-tuning method AdaLoRA achieves the best performance in enabling edited models to generate correct, up-to-date APIs, but falls short in Specificity (i.e., the editing influences untargeted knowledge). To resolve this, we propose AdaLoRA-L, which defines "Common API Layers" (layers within the LLMs with high importance across all APIs, storing general knowledge and excluded from editing) and restricts edits exclusively to "Specific API Layers" (layers with high importance only for the target API, storing the API-specific knowledge). Experimental results demonstrate that AdaLoRA-L significantly improves Specificity while maintaining comparable performance across other evaluation metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20993v1">Subgoal Graph-Augmented Planning for LLM-Guided Open-World Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer strong high-level planning capabilities for reinforcement learning (RL) by decomposing tasks into subgoals. However, their practical utility is limited by poor planning-execution alignment, which reflects a critical gap between abstract plans and actionable, environment-compatible behaviors. This misalignment arises from two interrelated limitations: (1) LLMs often produce subgoals that are semantically plausible but infeasible or irrelevant in the target environment due to insufficient grounding in environment-specific knowledge, and (2) single-LLM planning conflates generation with self-verification, resulting in overconfident yet unreliable subgoals that frequently fail during execution. To address these challenges, we propose Subgoal Graph-Augmented Actor-Critic-Refiner (SGA-ACR), a framework that integrates an environment-specific subgoal graph and structured entity knowledge with a multi-LLM planning pipeline that explicitly separates generation, critique, and refinement to produce executable and verifiable subgoals. A subgoal tracker further monitors execution progress, provides auxiliary rewards, and adaptively updates the subgoal graph to maintain alignment between plans and actions. Experimental results on 22 diverse tasks in the open-world game "Crafter" demonstrate the effectiveness of our proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18581v2">TASO: Jailbreak LLMs via Alternative Template and Suffix Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Many recent studies showed that LLMs are vulnerable to jailbreak attacks, where an attacker can perturb the input of an LLM to induce it to generate an output for a harmful question. In general, existing jailbreak techniques either optimize a semantic template intended to induce the LLM to produce harmful outputs or optimize a suffix that leads the LLM to initiate its response with specific tokens (e.g., "Sure"). In this work, we introduce TASO (Template and Suffix Optimization), a novel jailbreak method that optimizes both a template and a suffix in an alternating manner. Our insight is that suffix optimization and template optimization are complementary to each other: suffix optimization can effectively control the first few output tokens but cannot control the overall quality of the output, while template optimization provides guidance for the entire output but cannot effectively control the initial tokens, which significantly impact subsequent responses. Thus, they can be combined to improve the attack's effectiveness. We evaluate the effectiveness of TASO on benchmark datasets (including HarmBench and AdvBench) on 24 leading LLMs (including models from the Llama family, OpenAI, and DeepSeek). The results demonstrate that TASO can effectively jailbreak existing LLMs. We hope our work can inspire future studies in exploring this direction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20982v1">A Dynamic PD-Disaggregation Architecture for Maximizing Goodput in LLM Inference Serving</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      To meet strict Service-Level Objectives (SLOs),contemporary Large Language Models (LLMs) decouple the prefill and decoding stages and place them on separate GPUs to mitigate the distinct bottlenecks inherent to each phase. However, the heterogeneity of LLM workloads causes producerconsumer imbalance between the two instance types in such disaggregated architecture. To address this problem, we propose DOPD (Dynamic Optimal Prefill/Decoding), a dynamic LLM inference system that adjusts instance allocations to achieve an optimal prefill-to-decoding (P/D) ratio based on real-time load monitoring. Combined with an appropriate request-scheduling policy, DOPD effectively resolves imbalances between prefill and decoding instances and mitigates resource allocation mismatches due to mixed-length requests under high concurrency. Experimental evaluations show that, compared with vLLM and DistServe (representative aggregation-based and disaggregationbased approaches), DOPD improves overall system goodput by up to 1.5X, decreases P90 time-to-first-token (TTFT) by up to 67.5%, and decreases P90 time-per-output-token (TPOT) by up to 22.8%. Furthermore, our dynamic P/D adjustment technique performs proactive reconfiguration based on historical load, achieving over 99% SLOs attainment while using less additional resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.17264v5">No Request Left Behind: Tackling Heterogeneity in Long-Context LLM Inference with Medha</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Deploying million-token Large Language Models (LLMs) is challenging because production workloads are highly heterogeneous, mixing short queries and long documents. This heterogeneity, combined with the quadratic complexity of attention, creates severe convoy effects where long-running requests stall short, interactive ones, degrading system responsiveness. We present Medha, a serving system that eliminates these convoys by introducing fine-grained, preemptive scheduling to LLM inference. Medha makes preemption practical with a co-designed set of mechanisms -- including Adaptive Chunking and Stream Pipeline Parallel that overcome the perceived inefficiencies and scaling challenges of chunking. Additionally, we present a new parallelism strategy KV-Cache Parallelism to reduce the decode latency and afford interactivity despite very long context. These mechanisms are orchestrated by a Length-Aware Relative Slack (LARS) scheduler, a deadline and heterogeneity-aware scheduling policy that prevents both the convoy effect and the starvation that plagues simpler policies. Under a heterogeneous workload, Medha improves throughput by 5.7x while reducing median and 99th percentile latency by 30x and 174x, respectively, compared to state-of-the-art non-preemptive systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20965v1">TrafficLens: Multi-Camera Traffic Video Analysis Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 2024 IEEE 27th International Conference on Intelligent Transportation Systems (ITSC)
    </div>
    <details class="paper-abstract">
      Traffic cameras are essential in urban areas, playing a crucial role in intelligent transportation systems. Multiple cameras at intersections enhance law enforcement capabilities, traffic management, and pedestrian safety. However, efficiently managing and analyzing multi-camera feeds poses challenges due to the vast amount of data. Analyzing such huge video data requires advanced analytical tools. While Large Language Models (LLMs) like ChatGPT, equipped with retrieval-augmented generation (RAG) systems, excel in text-based tasks, integrating them into traffic video analysis demands converting video data into text using a Vision-Language Model (VLM), which is time-consuming and delays the timely utilization of traffic videos for generating insights and investigating incidents. To address these challenges, we propose TrafficLens, a tailored algorithm for multi-camera traffic intersections. TrafficLens employs a sequential approach, utilizing overlapping coverage areas of cameras. It iteratively applies VLMs with varying token limits, using previous outputs as prompts for subsequent cameras, enabling rapid generation of detailed textual descriptions while reducing processing time. Additionally, TrafficLens intelligently bypasses redundant VLM invocations through an object-level similarity detector. Experimental results with real-world datasets demonstrate that TrafficLens reduces video-to-text conversion time by up to $4\times$ while maintaining information accuracy.
    </details>
</div>
