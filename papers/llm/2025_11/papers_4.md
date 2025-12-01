# llm - 2025_11

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- Part 4
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
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16901v1">R-AVST: Empowering Video-LLMs with Fine-Grained Spatio-Temporal Reasoning in Complex Audio-Visual Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 Accepted by AAAI 2026. Project page: https://github.com/zhlllau/R-AVST
    </div>
    <details class="paper-abstract">
      Recently, rapid advancements have been made in multimodal large language models (MLLMs), especially in video understanding tasks. However, current research focuses on simple video scenarios, failing to reflect the complex and diverse nature of real-world audio-visual events in videos. To bridge this gap, we firstly introduce R-AVST, a dataset for audio-visual reasoning featuring fine-grained spatio-temporal annotations. In constructing this, we design a pipeline consisting of LLM-based key object extraction, automatic spatial annotation and manual quality inspection, resulting in over 5K untrimmed videos with 27K objects across 100 types of audio-visual events. Building on this dataset, we define three core tasks for spatio-temporal reasoning in audio-visual scenes and generate more than 8K high-quality, evenly distributed question-answer pairs to effectively benchmark model performance. To further enhance reasoning, we propose AVST-Zero, a reinforcement learning-based model that avoids intermediate supervision, directly optimizing behavior via carefully designed multi-dimensional rewards. Extensive experiments validate the effectiveness of our R-AVST in advancing audio-visual spatio-temporal reasoning, upon which AVST-Zero demonstrates competitive performance compared to existing models. To the best of our knowledge, R-AVST is the first dataset designed for real-world audio-visual spatio-temporal reasoning, and AVST-Zero offers a novel perspective for tackling future challenges in this domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16885v1">Improving Latent Reasoning in LLMs via Soft Concept Mixing</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 7 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Unlike human reasoning in abstract conceptual spaces, large language models (LLMs) typically reason by generating discrete tokens, which potentially limit their expressive power. The recent work Soft Thinking has shown that LLMs' latent reasoning via soft concepts is a promising direction, but LLMs are trained on discrete tokens. To reduce this gap between the soft concepts in reasoning and the discrete tokens in training, we propose Soft Concept Mixing (SCM), a soft concept aware training scheme that directly exposes the model to soft representations during training. Specifically, SCM constructs a soft concept vector by forming a probability-weighted average of embeddings. Then, this vector is mixed into the model's hidden states, which embody rich contextual information. Finally, the entire latent reasoning process is optimized with Reinforcement Learning (RL). Experiments on five reasoning benchmarks demonstrate that SCM improves the reasoning performance of LLMs, and simultaneously maintains a stable training dynamic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16883v1">PersonalizedRouter: Personalized LLM Routing via Graph-based User Preference Modeling</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      The growing number of Large Language Models (LLMs) with diverse capabilities and response styles provides users with a wider range of choices, which presents challenges in selecting appropriate LLMs, as user preferences vary in terms of performance, cost, and response style. Current LLM selection methods typically optimize for a single fixed objective, such as performance, cost, or a trade-off between them, and fail to learn individual user preferences from interaction data. To address these limitations, we propose PersonalizedRouter, a graph-based framework that models diverse user profiles and performs personalized LLM selection by leveraging interaction data that includes task context, queries, candidate LLMs, and user decisions. To capture contextual information between user queries and optimal LLMs, PersonalizedRouter converts the interaction data into a heterogeneous graph, where the relationships between different types of nodes are represented by edges. To evaluate adaptability across users, we design two strategies: the multi-cost-efficiency simulation strategy and the LLM-as-a-Judge strategy. In addition, we construct PersonaRoute-Bench, a large-scale benchmark with 1,000 simulated users and 10 LLMs. Experimental results show that PersonalizedRouter significantly outperforms existing LLM selection methods and surpasses the strongest methods by a large margin of 15.38% and 9.83% under two simulation strategies. On the PersonaRoute-Bench with 1,000 users, it further surpasses the best methods by 16.19% and 59.69% while maintaining higher efficiency. Moreover, PersonalizedRouter demonstrates strong few-shot generalization, achieving 64.81% and 85.80% of the fully trained model's performance when adapting to new users and new LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.12188v3">LLM-DSE: Searching Accelerator Parameters with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      Even though high-level synthesis (HLS) tools mitigate the challenges of programming domain-specific accelerators (DSAs) by raising the abstraction level, optimizing hardware directive parameters remains a significant hurdle. Existing heuristic and learning-based methods struggle with adaptability and sample efficiency. We present LLM-DSE, a multi-agent framework designed specifically for optimizing HLS directives. Combining LLM with design space exploration (DSE), our explorer coordinates four agents: Router, Specialists, Arbitrator, and Critic. These multi-agent components interact with various tools to accelerate the optimization process. LLM-DSE leverages essential domain knowledge to identify efficient parameter combinations while maintaining adaptability through verbal learning from online interactions. Evaluations on the HLSyn dataset demonstrate that LLM-DSE achieves substantial $2.55\times$ performance gains over state-of-the-art methods, uncovering novel designs while reducing runtime. Ablation studies validate the effectiveness and necessity of the proposed agent interactions. Our code is open-sourced here: https://github.com/Nozidoali/LLM-DSE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.05080v2">An Architectural Advantage of The Instruction-Tuned LLM in Containing The Readability-Accuracy Tension in Text Simplification</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      The increasing health-seeking behavior and digital consumption of biomedical information by the general public necessitate scalable solutions for automatically adapting complex scientific and technical documents into plain language. Automatic text simplification solutions, including advanced large language models (LLMs), however, continue to face challenges in reliably arbitrating the tension between optimizing readability performance and ensuring preservation of discourse fidelity. This report empirically assesses two major classes of general-purpose LLMs, demonstrating how they navigate the readability-accuracy tension compared to a human benchmark. Using a comparative analysis of the instruction-tuned Mistral-Small 3 24B and the reasoning-augmented QWen2.5 32B, we identify an architectural advantage in the instruction-tuned LLM. Mistral exhibits a tempered lexical simplification strategy that enhances readability across a suite of metrics while preserving human-level discourse with a BERTScore of 0.91. QWen also attains enhanced readability performance and a reasonable BERTScore of 0.89, but its operational strategy shows a disconnect in balancing between readability and accuracy. Additionally, a comprehensive correlation analysis of a suite of 21 metrics spanning readability, discourse fidelity, content safety, and underlying distributional measures for mechanistic insights, confirms strong functional redundancies, and informs metric selection and domain adaptation for text simplification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.08250v2">Hybrid LLM Routing for Efficient App Feedback Classification</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      The emergence of large language models (LLMs), pre-trained on massive datasets, has demonstrated strong performance across a wide range of natural language processing (NLP) tasks, including text classification. While prior studies have examined the use of LLMs for predicting the intent of user feedback and reported encouraging results, these investigations remain limited in scope. Furthermore, the vast volume of feedback posted daily, particularly for popular applications, combined with the computational and financial overhead of commercial LLMs, renders large-scale deployment impractical. In contrast, smaller models provide greater efficiency and lower cost but generally at the expense of reduced accuracy. In this paper, we aim to balance accuracy and efficiency in feedback classification. We first present a comprehensive study of zero-shot classification using four widely adopted LLMs, GPT-3.5-Turbo, GPT-4o, Flan-T5, and Llama3-70B, on diverse feedback datasets collected from multiple platforms, including app stores, forums, and X, which are categorized under different schemes. This analysis reveals how classification scheme design and platform characteristics influence the predictive performance of LLMs. Building on these insights, we propose a two-tier routing strategy for scalable app store feedback classification. In this approach, low-complexity instances are processed by lightweight fine-tuned models, while ambiguous cases are routed to high-capacity LLMs for more reliable decisions. Experimental results show that this strategy retains 98.4% to 100.4% of zero-shot LLM accuracy while reducing request and token costs by 67.8% and 66.3%, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17833v1">Learning to Debug: LLM-Organized Knowledge Trees for Solving RTL Assertion Failures</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      Debugging is the dominant cost in modern hardware verification, where assertion failures are among the most frequent and expensive to resolve. While Large Language Models (LLMs) show promise, they often fail to capture the precise, reusable expertise that engineers apply, leading to inaccurate responses. We propose GROVE, a hierarchical knowledge management framework that learns and organizes reusable debugging expertise into an LLM-organized knowledge tree for solving assertion failures. GROVE distills debugging knowledge from prior cases and organizes it into a vertical tree of configurable depth, with each node encoding a concise knowledge item and explicit applicability conditions. During training, GROVE uses a parallel, gradient-free loop where an LLM proposes tree modifications as structured JSON edits by learning from the cases. At test time, a budget-aware iterative zoom is performed to navigate the tree, retrieving a small set of applicable knowledge items that guide a base LLM's hypothesis generation and fix proposals. Evaluated on a suite of assertion-failure cases, GROVE delivers consistent gains in pass@1 and pass@5, demonstrating the value of structured knowledge evolution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17818v1">APRIL: Annotations for Policy evaluation with Reliable Inference from LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      Off-policy evaluation (OPE) estimates the value of a contextual bandit policy prior to deployment. As such, OPE plays a critical role in ensuring safety in high-stakes domains such as healthcare. However, standard OPE approaches are limited by the size and coverage of the behavior dataset. While previous work has explored using expert-labeled counterfactual annotations to enhance dataset coverage, obtaining such annotations is expensive, limiting the scalability of prior approaches. We propose leveraging large language models (LLMs) to generate counterfactual annotations for OPE in medical domains. Our method uses domain knowledge to guide LLMs in predicting how key clinical features evolve under alternate treatments. These predicted features can then be transformed using known reward functions to create counterfactual annotations. We first evaluate the ability of several LLMs to predict clinical features across two patient subsets in MIMIC-IV, finding that state-of-the-art LLMs achieve comparable performance. Building on this capacity to predict clinical features, we generate LLM-based counterfactual annotations and incorporate them into an OPE estimator. Our empirical results analyze the benefits of counterfactual annotations under varying degrees of shift between the behavior and target policies. We find that in most cases, the LLM-based counterfactual annotations significantly improve OPE estimates up to a point. We provide an entropy-based metric to identify when additional annotations cease to be useful. Our results demonstrate that LLM-based counterfactual annotations offer a scalable approach for addressing coverage limitations in healthcare datasets, enabling safer deployment of decision-making policies in clinical settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17813v1">Point of Order: Action-Aware LLM Persona Modeling for Realistic Civic Simulation</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 8 pages (29 pages including appendix), 18 figures. Code and datasets are available at https://github.com/smerrillunc/action-aware-llms. Submitted to ACL 2026
    </div>
    <details class="paper-abstract">
      Large language models offer opportunities to simulate multi-party deliberation, but realistic modeling remains limited by a lack of speaker-attributed data. Transcripts produced via automatic speech recognition (ASR) assign anonymous speaker labels (e.g., Speaker_1), preventing models from capturing consistent human behavior. This work introduces a reproducible pipeline to transform public Zoom recordings into speaker-attributed transcripts with metadata like persona profiles and pragmatic action tags (e.g., [propose_motion]). We release three local government deliberation datasets: Appellate Court hearings, School Board meetings, and Municipal Council sessions. Fine-tuning LLMs to model specific participants using this "action-aware" data produces a 67% reduction in perplexity and nearly doubles classifier-based performance metrics for speaker fidelity and realism. Turing-style human evaluations show our simulations are often indistinguishable from real deliberations, providing a practical and scalable method for complex realistic civic simulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2401.11641v3">Revolutionizing Finance with LLMs: An Overview of Applications and Insights</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) like ChatGPT have seen considerable advancements and have been applied in diverse fields. Built on the Transformer architecture, these models are trained on extensive datasets, enabling them to understand and generate human language effectively. In the financial domain, the deployment of LLMs is gaining momentum. These models are being utilized for automating financial report generation, forecasting market trends, analyzing investor sentiment, and offering personalized financial advice. Leveraging their natural language processing capabilities, LLMs can distill key insights from vast financial data, aiding institutions in making informed investment choices and enhancing both operational efficiency and customer satisfaction. In this study, we provide a comprehensive overview of the emerging integration of LLMs into various financial tasks. Additionally, we conducted holistic tests on multiple financial tasks through the combination of natural language instructions. Our findings show that GPT-4 effectively follow prompt instructions across various financial tasks. This survey and evaluation of LLMs in the financial domain aim to deepen the understanding of LLMs' current role in finance for both financial practitioners and LLM researchers, identify new research and application prospects, and highlight how these technologies can be leveraged to solve practical challenges in the finance industry.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.23799v4">Estimating LLM Consistency: A User Baseline vs Surrogate Metrics</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 Published as a main conference paper at EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are prone to hallucinations and sensitive to prompt perturbations, often resulting in inconsistent or unreliable generated text. Different methods have been proposed to mitigate such hallucinations and fragility, one of which is to measure the consistency of LLM responses -- the model's confidence in the response or likelihood of generating a similar response when resampled. In previous work, measuring LLM response consistency often relied on calculating the probability of a response appearing within a pool of resampled responses, analyzing internal states, or evaluating logits of responses. However, it was not clear how well these approaches approximated users' perceptions of consistency of LLM responses. To find out, we performed a user study ($n=2,976$) demonstrating that current methods for measuring LLM response consistency typically do not align well with humans' perceptions of LLM consistency. We propose a logit-based ensemble method for estimating LLM consistency and show that our method matches the performance of the best-performing existing metric in estimating human ratings of LLM consistency. Our results suggest that methods for estimating LLM consistency without human evaluation are sufficiently imperfect to warrant broader use of evaluation with human input; this would avoid misjudging the adequacy of models because of the imperfections of automated consistency metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.19659v2">Bias in the Picture: Benchmarking VLMs with Social-Cue News Images and LLM-as-Judge Assessment</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 Accepted to NeurIPS 2025 Workshop (Evaluating the Evolving LLM Lifecycle)
    </div>
    <details class="paper-abstract">
      Large vision-language models (VLMs) can jointly interpret images and text, but they are also prone to absorbing and reproducing harmful social stereotypes when visual cues such as age, gender, race, clothing, or occupation are present. To investigate these risks, we introduce a news-image benchmark consisting of 1,343 image-question pairs drawn from diverse outlets, which we annotated with ground-truth answers and demographic attributes (age, gender, race, occupation, and sports). We evaluate a range of state-of-the-art VLMs and employ a large language model (LLM) as judge, with human verification. Our findings show that: (i) visual context systematically shifts model outputs in open-ended settings; (ii) bias prevalence varies across attributes and models, with particularly high risk for gender and occupation; and (iii) higher faithfulness does not necessarily correspond to lower bias. We release the benchmark prompts, evaluation rubric, and code to support reproducible and fairness-aware multimodal assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14774v2">LiveCLKTBench: Towards Reliable Evaluation of Cross-Lingual Knowledge Transfer in Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      Evaluating cross-lingual knowledge transfer in large language models is challenging, as correct answers in a target language may arise either from genuine transfer or from prior exposure during pre-training. We present LiveCLKTBench, an automated generation pipeline specifically designed to isolate and measure cross-lingual knowledge transfer. Our pipeline identifies self-contained, time-sensitive knowledge entities from real-world domains, filters them based on temporal occurrence, and verifies them against the model's knowledge. The documents of these valid entities are then used to generate factual questions, which are translated into multiple languages to evaluate transferability across linguistic boundaries. Using LiveCLKTBench, we evaluate several LLMs across five languages and observe that cross-lingual transfer is strongly influenced by linguistic distance and often asymmetric across language directions. While larger models improve transfer, the gains diminish with scale and vary across domains. These findings provide new insights into multilingual transfer and demonstrate the value of LiveCLKTBench as a reliable benchmark for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17746v1">Computational frame analysis revisited: On LLMs for studying news coverage</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      Computational approaches have previously shown various promises and pitfalls when it comes to the reliable identification of media frames. Generative LLMs like GPT and Claude are increasingly being used as content analytical tools, but how effective are they for frame analysis? We address this question by systematically evaluating them against their computational predecessors: bag-of-words models and encoder-only transformers; and traditional manual coding procedures. Our analysis rests on a novel gold standard dataset that we inductively and iteratively developed through the study, investigating six months of news coverage of the US Mpox epidemic of 2022. While we discover some potential applications for generative LLMs, we demonstrate that they were consistently outperformed by manual coders, and in some instances, by smaller language models. Some form of human validation was always necessary to determine appropriate model choice. Additionally, by examining how the suitability of various approaches depended on the nature of different tasks that were part of our frame analytical workflow, we provide insights as to how researchers may leverage the complementarity of these approaches to use them in tandem. We conclude by endorsing a methodologically pluralistic approach and put forth a roadmap for computational frame analysis for researchers going forward.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2407.20240v3">Social and Ethical Risks Posed by General-Purpose LLMs for Settling Newcomers in Canada</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 26 pages, 8 figures
    </div>
    <details class="paper-abstract">
      The non-profit settlement sector in Canada supports newcomers in achieving successful integration. This sector faces increasing operational pressures amidst rising immigration targets, which highlights a need for enhanced efficiency and innovation, potentially through reliable AI solutions. The ad-hoc use of general-purpose generative AI, such as ChatGPT, might become a common practice among newcomers and service providers to address this need. However, these tools are not tailored for the settlement domain and can have detrimental implications for immigrants and refugees. We explore the risks that these tools might pose on newcomers to first, warn against the unguarded use of generative AI, and second, to incentivize further research and development in creating AI literacy programs as well as customized LLMs that are aligned with the preferences of the impacted communities. Crucially, such technologies should be designed to integrate seamlessly into the existing workflow of the settlement sector, ensuring human oversight, trustworthiness, and accountability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.07777v2">Drift No More? Context Equilibria in Multi-Turn LLM Interactions</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at single-turn tasks such as instruction following and summarization, yet real-world deployments require sustained multi-turn interactions where user goals and conversational context persist and evolve. A recurring challenge in this setting is context drift: the gradual divergence of a model's outputs from goal-consistent behavior across turns. Unlike single-turn errors, drift unfolds temporally and is poorly captured by static evaluation metrics. In this work, we present a study of context drift in multi-turn interactions and propose a simple dynamical framework to interpret its behavior. We formalize drift as the turn-wise KL divergence between the token-level predictive distributions of the test model and a goal-consistent reference model, and propose a recurrence model that interprets its evolution as a bounded stochastic process with restoring forces and controllable interventions. We instantiate this framework in both synthetic long-horizon rewriting tasks and realistic user-agent simulations such as in $τ$-Bench, measuring drift for several open-weight LLMs that are used as user simulators. Our experiments consistently reveal stable, noise-limited equilibria rather than runaway degradation, and demonstrate that simple reminder interventions reliably reduce divergence in line with theoretical predictions. Together, these results suggest that multi-turn drift can be understood as a controllable equilibrium phenomenon rather than as inevitable decay, providing a foundation for studying and mitigating context drift in extended interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15998v2">Hiding in the AI Traffic: Abusing MCP for LLM-Powered Agentic Red Teaming</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 23 pages, 9 figures, 3 tables. Submitted as a full paper for review
    </div>
    <details class="paper-abstract">
      Generative AI is reshaping offensive cybersecurity by enabling autonomous red team agents that can plan, execute, and adapt during penetration tests. However, existing approaches face trade-offs between generality and specialization, and practical deployments reveal challenges such as hallucinations, context limitations, and ethical concerns. In this work, we introduce a novel command & control (C2) architecture leveraging the Model Context Protocol (MCP) to coordinate distributed, adaptive reconnaissance agents covertly across networks. Notably, we find that our architecture not only improves goal-directed behavior of the system as whole, but also eliminates key host and network artifacts that can be used to detect and prevent command & control behavior altogether. We begin with a comprehensive review of state-of-the-art generative red teaming methods, from fine-tuned specialist models to modular or agentic frameworks, analyzing their automation capabilities against task-specific accuracy. We then detail how our MCP-based C2 can overcome current limitations by enabling asynchronous, parallel operations and real-time intelligence sharing without periodic beaconing. We furthermore explore advanced adversarial capabilities of this architecture, its detection-evasion techniques, and address dual-use ethical implications, proposing defensive measures and controlled evaluation in lab settings. Experimental comparisons with traditional C2 show drastic reductions in manual effort and detection footprint. We conclude with future directions for integrating autonomous exploitation, defensive LLM agents, predictive evasive maneuvers, and multi-agent swarms. The proposed MCP-enabled C2 framework demonstrates a significant step toward realistic, AI-driven red team operations that can simulate advanced persistent threats while informing the development of next-generation defensive systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17683v1">Datacenters in the Desert: Feasibility and Sustainability of LLM Inference in the Middle East</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 3 pages, 1 figure
    </div>
    <details class="paper-abstract">
      As the Middle East emerges as a strategic hub for artificial intelligence (AI) infrastructure, the feasibility of deploying sustainable datacenters in desert environments has become a topic of growing relevance. This paper presents an empirical study analyzing the energy consumption and carbon footprint of large language model (LLM) inference across four countries: the United Arab Emirates, Iceland, Germany, and the United States of America using DeepSeek Coder 1.3B and the HumanEval dataset on the task of code generation. We use the CodeCarbon library to track energy and carbon emissions andcompare geographical trade-offs for climate-aware AI deployment. Our findings highlight both the challenges and potential of datacenters in desert regions and provide a balanced outlook on their role in global AI expansion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17676v1">LLM and Agent-Driven Data Analysis: A Systematic Approach for Enterprise Applications and System-level Deployment</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      The rapid progress in Generative AI and Agent technologies is profoundly transforming enterprise data management and analytics. Traditional database applications and system deployment are fundamentally impacted by AI-driven tools, such as Retrieval-Augmented Generation (RAG) and vector database technologies, which provide new pathways for semantic querying over enterprise knowledge bases. In the meantime, data security and compliance are top priorities for organizations adopting AI technologies. For enterprise data analysis, SQL generations powered by large language models (LLMs) and AI agents, has emerged as a key bridge connecting natural language with structured data, effectively lowering the barrier to enterprise data access and improving analytical efficiency. This paper focuses on enterprise data analysis applications and system deployment, covering a range of innovative frameworks, enabling complex query understanding, multi-agent collaboration, security verification, and computational efficiency. Through representative use cases, key challenges related to distributed deployment, data security, and inherent difficulties in SQL generation tasks are discussed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17673v1">Bridging Symbolic Control and Neural Reasoning in LLM Agents: The Structured Cognitive Loop</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 27 pages
    </div>
    <details class="paper-abstract">
      Large language model agents suffer from fundamental architectural problems: entangled reasoning and execution, memory volatility, and uncontrolled action sequences. We introduce Structured Cognitive Loop (SCL), a modular architecture that explicitly separates agent cognition into five phases: Retrieval, Cognition, Control, Action, and Memory (R-CCAM). At the core of SCL is Soft Symbolic Control, an adaptive governance mechanism that applies symbolic constraints to probabilistic inference, preserving neural flexibility while restoring the explainability and controllability of classical symbolic systems. Through empirical validation on multi-step conditional reasoning tasks, we demonstrate that SCL achieves zero policy violations, eliminates redundant tool calls, and maintains complete decision traceability. These results address critical gaps in existing frameworks such as ReAct, AutoGPT, and memory-augmented approaches. Our contributions are threefold: (1) we situate SCL within the taxonomy of hybrid intelligence, differentiating it from prompt-centric and memory-only approaches; (2) we formally define Soft Symbolic Control and contrast it with neuro-symbolic AI; and (3) we derive three design principles for trustworthy agents: modular decomposition, adaptive symbolic governance, and transparent state management. We provide a complete open-source implementation demonstrating the R-CCAM loop architecture, alongside a live GPT-4o-powered travel planning agent. By connecting expert system principles with modern LLM capabilities, this work offers a practical and theoretically grounded path toward reliable, explainable, and governable AI agents. Code: https://github.com/enkiluv/scl-core-experiment Demo: https://scl-travel-planner.streamlit.app/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16664v1">Nemotron Elastic: Towards Efficient Many-in-One Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      Training a family of large language models targeting multiple scales and deployment objectives is prohibitively expensive, requiring separate training runs for each different size. Recent work on model compression through pruning and knowledge distillation has reduced this cost; however, this process still incurs hundreds of billions of tokens worth of training cost per compressed model. In this paper, we present Nemotron Elastic, a framework for building reasoning-oriented LLMs, including hybrid Mamba-Attention architectures, that embed multiple nested submodels within a single parent model, each optimized for different deployment configurations and budgets. Each of these submodels shares weights with the parent model and can be extracted zero-shot during deployment without additional training or fine-tuning. We enable this functionality through an end-to-end trained router, tightly coupled to a two-stage training curriculum designed specifically for reasoning models. We additionally introduce group-aware SSM elastification that preserves Mamba's structural constraints, heterogeneous MLP elastification, normalized MSE-based layer importance for improved depth selection, and knowledge distillation enabling simultaneous multi-budget optimization. We apply Nemotron Elastic to the Nemotron Nano V2 12B model, simultaneously producing a 9B and a 6B model using only 110B training tokens; this results in over 360x cost reduction compared to training model families from scratch, and around 7x compared to SoTA compression techniques. Each of the nested models performs on par or better than the SoTA in accuracy. Moreover, unlike other compression methods, the nested capability of our approach allows having a many-in-one reasoning model that has constant deployment memory against the number of models in the family.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16660v1">Cognitive Foundations for Reasoning and Their Manifestation in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 40 pages, 4 tables, 6 figures
    </div>
    <details class="paper-abstract">
      Large language models solve complex problems yet fail on simpler variants, suggesting they achieve correct outputs through mechanisms fundamentally different from human reasoning. We synthesize cognitive science research into a taxonomy of 28 cognitive elements spanning computational constraints, meta-cognitive controls, knowledge representations, and transformation operations, then analyze their behavioral manifestations in reasoning traces. We propose a fine-grained cognitive evaluation framework and conduct the first large-scale analysis of 170K traces from 17 models across text, vision, and audio modalities, alongside 54 human think-aloud traces, which we make publicly available. Our analysis reveals systematic structural differences: humans employ hierarchical nesting and meta-cognitive monitoring while models rely on shallow forward chaining, with divergence most pronounced on ill-structured problems. Meta-analysis of 1,598 LLM reasoning papers reveals the research community concentrates on easily quantifiable behaviors (sequential organization: 55%, decomposition: 60%) while neglecting meta-cognitive controls (self-awareness: 16%, evaluation: 8%) that correlate with success. Models possess behavioral repertoires associated with success but fail to deploy them spontaneously. Leveraging these patterns, we develop test-time reasoning guidance that automatically scaffold successful structures, improving performance by up to 60% on complex problems. By bridging cognitive science and LLM research, we establish a foundation for developing models that reason through principled cognitive mechanisms rather than brittle spurious reasoning shortcuts or memorization, opening new directions for both improving model capabilities and testing theories of human cognition at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2407.01082v8">Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 Oral presentation at ICLR 2025. Camera-ready version available at https://iclr.cc/virtual/2025/poster/30358
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) generate text by sampling the next token from a probability distribution over the vocabulary at each decoding step. Popular sampling methods like top-p (nucleus sampling) often struggle to balance quality and diversity, especially at higher temperatures which lead to incoherent or repetitive outputs. We propose min-p sampling, a dynamic truncation method that adjusts the sampling threshold based on the model's confidence by using the top token's probability as a scaling factor. Our experiments on benchmarks including GPQA, GSM8K, and AlpacaEval Creative Writing show that min-p sampling improves both the quality and diversity of generated text across different model families (Mistral and Llama 3) and model sizes (1B to 123B parameters), especially at higher temperatures. Human evaluations further show a clear preference for min-p sampling, in both text quality and creativity. Min-p sampling has been adopted by popular open-source LLM frameworks, including Hugging Face Transformers, VLLM, and many others, highlighting its considerable impact on improving text generation quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.06017v2">AgentSwift: Efficient LLM Agent Design via Value-guided Hierarchical Search</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 AAAI-2026
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have demonstrated strong capabilities across diverse domains, yet automated agent design remains a significant challenge. Current automated agent design approaches are often constrained by limited search spaces that primarily optimize workflows but fail to integrate crucial human-designed components like memory, planning, and tool use. Furthermore, these methods are hampered by high evaluation costs, as evaluating even a single new agent on a benchmark can require tens of dollars. The difficulty of this exploration is further exacerbated by inefficient search strategies that struggle to navigate the large design space effectively, making the discovery of novel agents a slow and resource-intensive process. To address these challenges, we propose AgentSwift, a novel framework for automated agent design. We formalize a hierarchical search space that jointly models agentic workflow and composable functional components. This structure moves beyond optimizing workflows alone by co-optimizing functional components, which enables the discovery of more complex and effective agent architectures. To make exploration within this expansive space feasible, we mitigate high evaluation costs by training a value model on a high-quality dataset, generated via a novel strategy combining combinatorial coverage and balanced Bayesian sampling for low-cost evaluation. Guiding the entire process is a hierarchical MCTS strategy, which is informed by uncertainty to efficiently navigate the search space. Evaluated across a comprehensive set of seven benchmarks spanning embodied, math, web, tool, and game domains, AgentSwift discovers agents that achieve an average performance gain of 8.34\% over both existing automated agent search methods and manually designed agents. Our framework serves as a launchpad for researchers to rapidly discover powerful agent architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.04420v5">KVTuner: Sensitivity-Aware Layer-Wise Mixed-Precision KV Cache Quantization for Efficient and Nearly Lossless LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 Accepted by ICML25. Code: https://github.com/cmd2001/KVTuner
    </div>
    <details class="paper-abstract">
      KV cache quantization can improve Large Language Models (LLMs) inference throughput and latency in long contexts and large batch-size scenarios while preserving LLMs effectiveness. However, current methods have three unsolved issues: overlooking layer-wise sensitivity to KV cache quantization, high overhead of online fine-grained decision-making, and low flexibility to different LLMs and constraints. Therefore, we theoretically analyze the inherent correlation of layer-wise transformer attention patterns to KV cache quantization errors and study why key cache is generally more important than value cache for quantization error reduction. We further propose a simple yet effective framework KVTuner to adaptively search for the optimal hardware-friendly layer-wise KV quantization precision pairs for coarse-grained KV cache with multi-objective optimization and directly utilize the offline searched configurations during online inference. To reduce the computational cost of offline calibration, we utilize the intra-layer KV precision pair pruning and inter-layer clustering to reduce the search space. Experimental results show that we can achieve nearly lossless 3.25-bit mixed precision KV cache quantization for LLMs like Llama-3.1-8B-Instruct and 4.0-bit for sensitive models like Qwen2.5-7B-Instruct on mathematical reasoning tasks. The maximum inference throughput can be improved by 21.25\% compared with KIVI-KV8 quantization over various context lengths. Our code and searched configurations are available at https://github.com/cmd2001/KVTuner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16450v1">Optimizing Federated Learning in the Era of LLMs: Message Quantization and Streaming</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 FLLM 2025
    </div>
    <details class="paper-abstract">
      Federated Learning (FL) offers a promising solution for training machine learning models across distributed data sources while preserving data privacy. However, FL faces critical challenges related to communication overhead and local resource constraints, especially in the era of Large Language Models (LLMs) with billions of parameters. The sheer size of these models exacerbates both memory and communication constraints, making efficient transmission and processing essential for practical deployment. NVIDIA FLARE, an open-source SDK for federated learning, addresses these challenges by introducing advanced communication capabilities. Building upon existing solutions for large object streaming, we enhance FL workflows for LLMs through two key techniques: message quantization and container/file streaming. Quantization reduces message size, while streaming enables efficient memory management, improving scalability and integration with existing workflows. These advancements significantly enhance the robustness and efficiency of FL with LLMs, ensuring better performance in real-world federated learning scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16414v1">An Efficient LLM-based Evolutional Recommendation with Locate-Forget-Update Paradigm</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      Nowadays, Large Language Models (LLMs) have shown exceptional performance in sequential recommendations, and the adoption of LLM-based recommender systems (LLMRec) is becoming increasingly widespread in existing e-commerce platforms. Despite the impressive performance, the constant high volume of new user-item interactions makes it difficult to adapt to the evolution of user preference over time, especially for LLM-based recommender systems. The challenge arises from the large number of parameters in LLMs, which makes traditional evolution methods (i.e., Re-training or Fine-tuning) impractical. Specifically, Re-training with all interactions results in prohibitively high computational costs. On the other hand, fine-tuning with only new interactions leads to preference forgetting among inactive users, ultimately compromising overall performance. To tackle this problem, we propose EvoRec, an efficient Locate-Forget-Update framework designed for LLM-based recommender systems to model the evolution of user preferences. EvoRec identifies a small set of parameters associated with preference changes and updates them precisely, thereby saving computational resources while maintaining strong recommendation performance. Notably, the modified parameters account for only 30\% of LoRA adapter parameters, with no additional parameters introduced. Extensive experiments on two real-world datasets demonstrate that, compared to existing methods, EvoRec not only efficiently evolves LLMRec to adapt to the preferences of active users, but also preserves the interests of inactive users from being disturbed during evolution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16395v1">CorrectHDL: Agentic HDL Design with LLMs Leveraging High-Level Synthesis as Reference</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 7 pages, 15 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable potential in hardware front-end design using hardware description languages (HDLs). However, their inherent tendency toward hallucination often introduces functional errors into the generated HDL designs. To address this issue, we propose the framework CorrectHDL that leverages high-level synthesis (HLS) results as functional references to correct potential errors in LLM-generated HDL designs.The input to the proposed framework is a C/C++ program that specifies the target circuit's functionality. The program is provided to an LLM to directly generate an HDL design, whose syntax errors are repaired using a Retrieval-Augmented Generation (RAG) mechanism. The functional correctness of the LLM-generated circuit is iteratively improved by comparing its simulated behavior with an HLS reference design produced by conventional HLS tools, which ensures the functional correctness of the result but can lead to suboptimal area and power efficiency. Experimental results demonstrate that circuits generated by the proposed framework achieve significantly better area and power efficiency than conventional HLS designs and approach the quality of human-engineered circuits. Meanwhile, the correctness of the resulting HDL implementation is maintained, highlighting the effectiveness and potential of agentic HDL design leveraging the generative capabilities of LLMs and the rigor of traditional correctness-driven IC design flows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16324v1">SDA: Steering-Driven Distribution Alignment for Open LLMs without Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      With the rapid advancement of large language models (LLMs), their deployment in real-world applications has become increasingly widespread. LLMs are expected to deliver robust performance across diverse tasks, user preferences, and practical scenarios. However, as demands grow, ensuring that LLMs produce responses aligned with human intent remains a foundational challenge. In particular, aligning model behavior effectively and efficiently during inference, without costly retraining or extensive supervision, is both a critical requirement and a non-trivial technical endeavor. To address the challenge, we propose SDA (Steering-Driven Distribution Alignment), a training-free and model-agnostic alignment framework designed for open-source LLMs. SDA dynamically redistributes model output probabilities based on user-defined alignment instructions, enhancing alignment between model behavior and human intents without fine-tuning. The method is lightweight, resource-efficient, and compatible with a wide range of open-source LLMs. It can function independently during inference or be integrated with training-based alignment strategies. Moreover, SDA supports personalized preference alignment, enabling flexible control over the model response behavior. Empirical results demonstrate that SDA consistently improves alignment performance across 8 open-source LLMs with varying scales and diverse origins, evaluated on three key alignment dimensions, helpfulness, harmlessness, and honesty (3H). Specifically, SDA achieves average gains of 64.4% in helpfulness, 30% in honesty and 11.5% in harmlessness across the tested models, indicating its effectiveness and generalization across diverse models and application scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.25939v3">SoK: Honeypots & LLMs, More Than the Sum of Their Parts?</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 Systemization of Knowledge
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) promised to resolve the long-standing paradox in honeypot design, achieving high-fidelity deception with low operational risk. Through a flurry of research since late 2022, steady progress from ideation to prototype implementation is exhibited. Since late 2022, a flurry of research has demonstrated steady progress from ideation to prototype implementation. While promising, evaluations show only incremental progress in real-world deployments, and the field still lacks a cohesive understanding of the emerging architectural patterns, core challenges, and evaluation paradigms. To fill this gap, this Systematization of Knowledge (SoK) paper provides the first comprehensive overview and analysis of this new domain. We survey and systematize the field by focusing on three critical, intersecting research areas: first, we provide a taxonomy of honeypot detection vectors, structuring the core problems that LLM-based realism must solve; second, we synthesize the emerging literature on LLM-powered honeypots, identifying a canonical architecture and key evaluation trends; and third, we chart the evolutionary path of honeypot log analysis, from simple data reduction to automated intelligence generation. We synthesize these findings into a forward-looking research roadmap, arguing that the true potential of this technology lies in creating autonomous, self-improving deception systems to counter the emerging threat of intelligent, automated attackers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16278v1">"To Survive, I Must Defect": Jailbreaking LLMs via the Game-Theory Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      As LLMs become more common, non-expert users can pose risks, prompting extensive research into jailbreak attacks. However, most existing black-box jailbreak attacks rely on hand-crafted heuristics or narrow search spaces, which limit scalability. Compared with prior attacks, we propose Game-Theory Attack (GTA), an scalable black-box jailbreak framework. Concretely, we formalize the attacker's interaction against safety-aligned LLMs as a finite-horizon, early-stoppable sequential stochastic game, and reparameterize the LLM's randomized outputs via quantal response. Building on this, we introduce a behavioral conjecture "template-over-safety flip": by reshaping the LLM's effective objective through game-theoretic scenarios, the originally safety preference may become maximizing scenario payoffs within the template, which weakens safety constraints in specific contexts. We validate this mechanism with classical game such as the disclosure variant of the Prisoner's Dilemma, and we further introduce an Attacker Agent that adaptively escalates pressure to increase the ASR. Experiments across multiple protocols and datasets show that GTA achieves over 95% ASR on LLMs such as Deepseek-R1, while maintaining efficiency. Ablations over components, decoding, multilingual settings, and the Agent's core model confirm effectiveness and generalization. Moreover, scenario scaling studies further establish scalability. GTA also attains high ASR on other game-theoretic scenarios, and one-shot LLM-generated variants that keep the model mechanism fixed while varying background achieve comparable ASR. Paired with a Harmful-Words Detection Agent that performs word-level insertions, GTA maintains high ASR while lowering detection under prompt-guard models. Beyond benchmarks, GTA jailbreaks real-world LLM applications and reports a longitudinal safety monitoring of popular HuggingFace LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16275v1">SeSE: A Structural Information-Guided Uncertainty Quantification Framework for Hallucination Detection in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 14 pages of main text and 10 pages of appendices
    </div>
    <details class="paper-abstract">
      Reliable uncertainty quantification (UQ) is essential for deploying large language models (LLMs) in safety-critical scenarios, as it enables them to abstain from responding when uncertain, thereby avoiding hallucinating falsehoods. However, state-of-the-art UQ methods primarily rely on semantic probability distributions or pairwise distances, overlooking latent semantic structural information that could enable more precise uncertainty estimates. This paper presents Semantic Structural Entropy (SeSE), a principled UQ framework that quantifies the inherent semantic uncertainty of LLMs from a structural information perspective for hallucination detection. Specifically, to effectively model semantic spaces, we first develop an adaptively sparsified directed semantic graph construction algorithm that captures directional semantic dependencies while automatically pruning unnecessary connections that introduce negative interference. We then exploit latent semantic structural information through hierarchical abstraction: SeSE is defined as the structural entropy of the optimal semantic encoding tree, formalizing intrinsic uncertainty within semantic spaces after optimal compression. A higher SeSE value corresponds to greater uncertainty, indicating that LLMs are highly likely to generate hallucinations. In addition, to enhance fine-grained UQ in long-form generation -- where existing methods often rely on heuristic sample-and-count techniques -- we extend SeSE to quantify the uncertainty of individual claims by modeling their random semantic interactions, providing theoretically explicable hallucination detection. Extensive experiments across 29 model-dataset combinations show that SeSE significantly outperforms advanced UQ baselines, including strong supervised methods and the recently proposed KLE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08916v3">HalluClean: A Unified Framework to Combat Hallucinations in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved impressive performance across a wide range of natural language processing tasks, yet they often produce hallucinated content that undermines factual reliability. To address this challenge, we introduce HalluClean, a lightweight and task-agnostic framework for detecting and correcting hallucinations in LLM-generated text. HalluClean adopts a reasoning-enhanced paradigm, explicitly decomposing the process into planning, execution, and revision stages to identify and refine unsupported claims. It employs minimal task-routing prompts to enable zero-shot generalization across diverse domains, without relying on external knowledge sources or supervised detectors. We conduct extensive evaluations on five representative tasks-question answering, dialogue, summarization, math word problems, and contradiction detection. Experimental results show that HalluClean significantly improves factual consistency and outperforms competitive baselines, demonstrating its potential to enhance the trustworthiness of LLM outputs in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16224v1">Beyond Code Similarity: Benchmarking the Plausibility, Efficiency, and Complexity of LLM-Generated Smart Contracts</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      Smart Contracts are critical components of blockchain ecosystems, with Solidity as the dominant programming language. While LLMs excel at general-purpose code generation, the unique constraints of Smart Contracts, such as gas consumption, security, and determinism, raise open questions about the reliability of LLM-generated Solidity code. Existing studies lack a comprehensive evaluation of these critical functional and non-functional properties. We benchmark four state-of-the-art models under zero-shot and retrieval-augmented generation settings across 500 real-world functions. Our multi-faceted assessment employs code similarity metrics, semantic embeddings, automated test execution, gas profiling, and cognitive and cyclomatic complexity analysis. Results show that while LLMs produce code with high semantic similarity to real contracts, their functional correctness is low: only 20% to 26% of zero-shot generations behave identically to ground-truth implementations under testing. The generated code is consistently simpler, with significantly lower complexity and gas consumption, often due to omitted validation logic. Retrieval-Augmented Generation markedly improves performance, boosting functional correctness by up to 45% and yielding more concise and efficient code. Our findings reveal a significant gap between semantic similarity and functional plausibility in LLM-generated Smart Contracts. We conclude that while RAG is a powerful enhancer, achieving robust, production-ready code generation remains a substantial challenge, necessitating careful expert validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16209v1">PSM: Prompt Sensitivity Minimization via LLM-Guided Black-Box Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      System prompts are critical for guiding the behavior of Large Language Models (LLMs), yet they often contain proprietary logic or sensitive information, making them a prime target for extraction attacks. Adversarial queries can successfully elicit these hidden instructions, posing significant security and privacy risks. Existing defense mechanisms frequently rely on heuristics, incur substantial computational overhead, or are inapplicable to models accessed via black-box APIs. This paper introduces a novel framework for hardening system prompts through shield appending, a lightweight approach that adds a protective textual layer to the original prompt. Our core contribution is the formalization of prompt hardening as a utility-constrained optimization problem. We leverage an LLM-as-optimizer to search the space of possible SHIELDs, seeking to minimize a leakage metric derived from a suite of adversarial attacks, while simultaneously preserving task utility above a specified threshold, measured by semantic fidelity to baseline outputs. This black-box, optimization-driven methodology is lightweight and practical, requiring only API access to the target and optimizer LLMs. We demonstrate empirically that our optimized SHIELDs significantly reduce prompt leakage against a comprehensive set of extraction attacks, outperforming established baseline defenses without compromising the model's intended functionality. Our work presents a paradigm for developing robust, utility-aware defenses in the escalating landscape of LLM security. The code is made public on the following link: https://github.com/psm-defense/psm
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.16267v3">From Confidence to Collapse in LLM Factual Robustness</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      Ensuring the robustness of factual knowledge in LLMs is critical for reliable applications in tasks such as question answering and reasoning. However, existing evaluation methods predominantly focus on performance-based metrics, often investigating from the perspective of prompt perturbations, which captures only the externally triggered side of knowledge robustness. To bridge this gap, we introduce a principled approach to measure factual robustness from the perspective of the generation process by analyzing token distribution entropy in combination with temperature scaling sensitivity. These two factors build the Factual Robustness Score (FRS), a novel metric which quantifies the stability of a fact against perturbations in decoding conditions, given its initial uncertainty. To validate our approach, we conduct extensive experiments on 5 LLMs across 3 closed-book QA datasets (SQuAD, TriviaQA, and HotpotQA). We show that factual robustness varies significantly -- smaller models report an FRS of $0.76$, larger ones $0.93$ -- with accuracy degrading by ~$60\%$ under increased uncertainty. These insights demonstrate how entropy and temperature scaling impact factual accuracy, and lay a foundation for developing more robust knowledge retention and retrieval in future models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.05310v3">Oracular Programming: A Modular Foundation for Building LLM-Enabled Software</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      Large Language Models can solve a wide range of tasks from just a few examples, but they remain difficult to steer and lack a capability essential for building reliable software at scale: the modular composition of computations under enforceable contracts. As a result, they are typically embedded in larger software pipelines that use domain-specific knowledge to decompose tasks and improve reliability through validation and search. Yet the complexity of writing, tuning, and maintaining such pipelines has so far limited their sophistication. We propose oracular programming: a foundational paradigm for integrating traditional, explicit computations with inductive oracles such as LLMs. It rests on two directing principles: the full separation of core and search logic, and the treatment of few-shot examples as grounded and evolvable program components. Within this paradigm, experts express high-level problem-solving strategies as programs with unresolved choice points. These choice points are resolved at runtime by LLMs, which generalize from user-provided examples of correct and incorrect decisions. An oracular program is composed of three orthogonal components: a strategy that consists in a nondeterministic program with choice points that can be reified into a search tree, a policy that specifies how to navigate this tree with the help of LLM oracles, and a set of demonstrations that describe successful and unsuccessful tree navigation scenarios across diverse problem instances. Each component is expressed in a dedicated programming language and can be independently improved or substituted. We address the key programming language design challenges of modularly composing oracular programs and enforcing consistency between their components as they evolve.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09837v2">MoFa: A Unified Performance Modeling Framework for LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      The exponential growth in LLM scales, with parameters soaring from billions to trillions, has necessitated distributed pretraining across large clusters comprising thousands to tens of thousands of devices. While hybrid parallelization strategies enable such pretraining, the vast combinatorial strategy space introduces significant optimization challenges. Traditional manual tuning methods incur prohibitive trial-and-error costs, and existing performance modeling approaches exhibit critical limitations: they fail to comprehensively account for prevalent optimization features and ignore the substantial overhead imposed by essential fault tolerance mechanisms like checkpoint recovery in long-duration pretraining. To address these gaps, we propose MoFa, a novel pretraining performance modeling framework that unifies multi-dimensional optimization features and fault tolerance. MoFa incorporates an enhanced cost model to accurately capture the effects of key optimizations and integrates a fault tolerance model based on historical cluster reliability data. Besides, a MoFa-based tuning system is developed to explore optimal pretraining performance and potential bottlenecks in various scenarios. Extensive modeling evaluations demonstrate that MoFa can achieve high prediction accuracy across various scenarios. In addition, through comprehensive tuning experiments, our framework systematically reveals the key factors influencing pretraining performance under different configurations, which provides solid a priori guidance for LLM pretraining system design and deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.05919v2">Injecting Falsehoods: Adversarial Man-in-the-Middle Attacks Undermining Factual Recall in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      LLMs are now an integral part of information retrieval. As such, their role as question answering chatbots raises significant concerns due to their shown vulnerability to adversarial man-in-the-middle (MitM) attacks. Here, we propose the first principled attack evaluation on LLM factual memory under prompt injection via Xmera, our novel, theory-grounded MitM framework. By perturbing the input given to "victim" LLMs in three closed-book and fact-based QA settings, we undermine the correctness of the responses and assess the uncertainty of their generation process. Surprisingly, trivial instruction-based attacks report the highest success rate (up to ~85.3%) while simultaneously having a high uncertainty for incorrectly answered questions. To provide a simple defense mechanism against Xmera, we train Random Forest classifiers on the response uncertainty levels to distinguish between attacked and unattacked queries (average AUC of up to ~96%). We believe that signaling users to be cautious about the answers they receive from black-box and potentially corrupt LLMs is a first checkpoint toward user cyberspace safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15192v2">As If We've Met Before: LLMs Exhibit Certainty in Recognizing Seen Files</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      The remarkable language ability of Large Language Models (LLMs) stems from extensive training on vast datasets, often including copyrighted material, which raises serious concerns about unauthorized use. While Membership Inference Attacks (MIAs) offer potential solutions for detecting such violations, existing approaches face critical limitations and challenges due to LLMs' inherent overconfidence, limited access to ground truth training data, and reliance on empirically determined thresholds. We present COPYCHECK, a novel framework that leverages uncertainty signals to detect whether copyrighted content was used in LLM training sets. Our method turns LLM overconfidence from a limitation into an asset by capturing uncertainty patterns that reliably distinguish between ``seen" (training data) and ``unseen" (non-training data) content. COPYCHECK further implements a two-fold strategy: (1) strategic segmentation of files into smaller snippets to reduce dependence on large-scale training data, and (2) uncertainty-guided unsupervised clustering to eliminate the need for empirically tuned thresholds. Experiment results show that COPYCHECK achieves an average balanced accuracy of 90.1% on LLaMA 7b and 91.6% on LLaMA2 7b in detecting seen files. Compared to the SOTA baseline, COPYCHECK achieves over 90% relative improvement, reaching up to 93.8\% balanced accuracy. It further exhibits strong generalizability across architectures, maintaining high performance on GPT-J 6B. This work presents the first application of uncertainty for copyright detection in LLMs, offering practical tools for training data transparency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16193v1">Fast LLM Post-training via Decoupled and Best-of-N Speculation</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      Rollout dominates the training time in large language model (LLM) post-training, where the trained model is used to generate tokens given a batch of prompts. SpecActor achieves fast rollout with speculative decoding that deploys a fast path (e.g., a smaller model) to accelerate the unparallelizable generation, while the correctness is guaranteed by fast parallel verification of the outputs with the original model. SpecActor addresses two foundational challenges in speculative rollout by (1) a \emph{dynamic decoupled speculation} execution method that maximizes the GPU computational efficiency to realize speedup for large-batch execution -- a configuration common in training but unfriendly to speculative execution and (2) a \emph{dynamic Best-of-N speculation} method that selects and combines different drafting methods according to the rollout progress. It substantially improves the speculation accuracy even when the best drafting method is unknown a priori, meanwhile without requiring adding extra computation resources. {\sys} is {1.3--1.7}\,$\times$ faster than common post-training baselines, and is {1.3--1.5}\,$\times$ faster compared to naively adopting speculative decoding for rollout.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.03628v5">LLMDistill4Ads: Using Cross-Encoders to Distill from LLM Signals for Advertiser Keyphrase Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      E-commerce sellers are advised to bid on keyphrases to boost their advertising campaigns. These keyphrases must be relevant to prevent irrelevant items from cluttering search systems and to maintain positive seller perception. It is vital that keyphrase suggestions align with seller, search and buyer judgments. Given the challenges in collecting negative feedback in these systems, LLMs have been used as a scalable proxy to human judgments. This paper presents an empirical study on a major ecommerce platform of a distillation framework involving an LLM teacher, a cross-encoder assistant and a bi-encoder Embedding Based Retrieval (EBR) student model, aimed at mitigating click-induced biases in keyphrase recommendations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10712v2">Do Not Merge My Model! Safeguarding Open-Source LLMs Against Unauthorized Model Merging</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 Accepted by AAAI 2026 Conference
    </div>
    <details class="paper-abstract">
      Model merging has emerged as an efficient technique for expanding large language models (LLMs) by integrating specialized expert models. However, it also introduces a new threat: model merging stealing, where free-riders exploit models through unauthorized model merging. Unfortunately, existing defense mechanisms fail to provide effective protection. Specifically, we identify three critical protection properties that existing methods fail to simultaneously satisfy: (1) proactively preventing unauthorized merging; (2) ensuring compatibility with general open-source settings; (3) achieving high security with negligible performance loss. To address the above issues, we propose MergeBarrier, a plug-and-play defense that proactively prevents unauthorized merging. The core design of MergeBarrier is to disrupt the Linear Mode Connectivity (LMC) between the protected model and its homologous counterparts, thereby eliminating the low-loss path required for effective model merging. Extensive experiments show that MergeBarrier effectively prevents model merging stealing with negligible accuracy loss.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.13246v3">Atomic Calibration of LLMs in Long-Form Generations</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 ACL 2025 KnowFM Oral / AACL-IJCNLP 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often suffer from hallucinations, posing significant challenges for real-world applications. Confidence calibration, as an effective indicator of hallucination, is thus essential to enhance the trustworthiness of LLMs. Prior work mainly focuses on short-form tasks using a single response-level score (macro calibration), which is insufficient for long-form outputs that may contain both accurate and inaccurate claims. In this work, we systematically study atomic calibration, which evaluates factuality calibration at a fine-grained level by decomposing long responses into atomic claims. We further categorize existing confidence elicitation methods into discriminative and generative types, and propose two new confidence fusion strategies to improve calibration. Our experiments demonstrate that LLMs exhibit poorer calibration at the atomic level during long-form generation. More importantly, atomic calibration uncovers insightful patterns regarding the alignment of confidence methods and the changes of confidence throughout generation. This sheds light on future research directions for confidence estimation in long-form generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.04250v2">How many patients could we save with LLM priors?</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 9 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Imagine a world where clinical trials need far fewer patients to achieve the same statistical power, thanks to the knowledge encoded in large language models (LLMs). We present a novel framework for hierarchical Bayesian modeling of adverse events in multi-center clinical trials, leveraging LLM-informed prior distributions. Unlike data augmentation approaches that generate synthetic data points, our methodology directly obtains parametric priors from the model. Our approach systematically elicits informative priors for hyperparameters in hierarchical Bayesian models using a pre-trained LLM, enabling the incorporation of external clinical expertise directly into Bayesian safety modeling. Through comprehensive temperature sensitivity analysis and rigorous cross-validation on real-world clinical trial data, we demonstrate that LLM-derived priors consistently improve predictive performance compared to traditional meta-analytical approaches. This methodology paves the way for more efficient and expert-informed clinical trial design, enabling substantial reductions in the number of patients required to achieve robust safety assessment and with the potential to transform drug safety monitoring and regulatory decision making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15168v2">Finetuning LLMs for Automatic Form Interaction on Web-Browser in Selenium Testing Framework</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 Published in the Proceedings of KSE 2025
    </div>
    <details class="paper-abstract">
      Automated web application testing is a critical component of modern software development, with frameworks like Selenium widely adopted for validating functionality through browser automation. Among the essential aspects of such testing is the ability to interact with and validate web forms, a task that requires syntactically correct, executable scripts with high coverage of input fields. Despite its importance, this task remains underexplored in the context of large language models (LLMs), and no public benchmark or dataset exists to evaluate LLMs on form interaction generation systematically. This paper introduces a novel method for training LLMs to generate high-quality test cases in Selenium, specifically targeting form interaction testing. We curate both synthetic and human-annotated datasets for training and evaluation, covering diverse real-world forms and testing scenarios. We define clear metrics for syntax correctness, script executability, and input field coverage. Our empirical study demonstrates that our approach significantly outperforms strong baselines, including GPT-4o and other popular LLMs, across all evaluation metrics. Our work lays the groundwork for future research on LLM-based web testing and provides resources to support ongoing progress in this area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.15258v2">Multi-dimensional Data Analysis and Applications Basing on LLM Agents and Knowledge Graph Interactions</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 14 pages, 7 figures, 40 references
    </div>
    <details class="paper-abstract">
      In the current era of big data, extracting deep insights from massive, heterogeneous, and complexly associated multi-dimensional data has become a significant challenge. Large Language Models (LLMs) perform well in natural language understanding and generation, but still suffer from "hallucination" issues when processing structured knowledge and are difficult to update in real-time. Although Knowledge Graphs (KGs) can explicitly store structured knowledge, their static nature limits dynamic interaction and analytical capabilities. Therefore, this paper proposes a multi-dimensional data analysis method based on the interactions between LLM agents and KGs, constructing a dynamic, collaborative analytical ecosystem. This method utilizes LLM agents to automatically extract product data from unstructured data, constructs and visualizes the KG in real-time, and supports users in deep exploration and analysis of graph nodes through an interactive platform. Experimental results show that this method has significant advantages in product ecosystem analysis, relationship mining, and user-driven exploratory analysis, providing new ideas and tools for multi-dimensional data analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.14765v2">PepThink-R1: LLM for Interpretable Cyclic Peptide Optimization with CoT SFT and Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      Designing therapeutic peptides with tailored properties is hindered by the vastness of sequence space, limited experimental data, and poor interpretability of current generative models. To address these challenges, we introduce PepThink-R1, a generative framework that integrates large language models (LLMs) with chain-of-thought (CoT) supervised fine-tuning and reinforcement learning (RL). Unlike prior approaches, PepThink-R1 explicitly reasons about monomer-level modifications during sequence generation, enabling interpretable design choices while optimizing for multiple pharmacological properties. Guided by a tailored reward function balancing chemical validity and property improvements, the model autonomously explores diverse sequence variants. We demonstrate that PepThink-R1 generates cyclic peptides with significantly enhanced lipophilicity, stability, and exposure, outperforming existing general LLMs (e.g., GPT-5) and domain-specific baseline in both optimization success and interpretability. To our knowledge, this is the first LLM-based peptide design framework that combines explicit reasoning with RL-driven property control, marking a step toward reliable and transparent peptide optimization for therapeutic discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.21830v2">GAPO: Robust Advantage Estimation for Real-World Code LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) is widely used for post-training large language models (LLMs) in code editing, where group-relative methods like GRPO are popular for their critic-free, normalized advantage estimation. However, in real-world code-editing scenarios, reward distributions are often skewed with unpredictable outliers, leading to distorted advantage computation and increased noise. To address this issue, we propose Group Adaptive Policy Optimization (GAPO), which adaptively finds an outlier-free highest-density interval (HDI) per prompt and then uses the median of that interval as an adaptive Q to replace the group mean in advantage calculation. This adaptive Q robustly handles skewed distributions while remaining plug-and-play and efficient. We validate GAPO on nine instruction-tuned LLMs (3B-14B) using a large internal dataset of 51,844 real-world, history-aware code-editing tasks across 10 languages, demonstrating consistent improvements in exact match accuracy over GRPO and its variant DAPO. Code is publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.04559v2">One Model for All: Unified Try-On and Try-Off in Any Pose via LLM-Inspired Bidirectional Tweedie Diffusion</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      Recent diffusion-based approaches have made significant advances in image-based virtual try-on, enabling more realistic and end-to-end garment synthesis. However, most existing methods remain constrained by their reliance on exhibition garments and segmentation masks, as well as their limited ability to handle flexible pose variations. These limitations reduce their practicality in real-world scenarios - for instance, users cannot easily transfer garments worn by one person onto another, and the generated try-on results are typically restricted to the same pose as the reference image. In this paper, we introduce OMFA (One Model For All), a unified diffusion framework for both virtual try-on and try-off that operates without the need for exhibition garments and supports arbitrary poses. OMFA is inspired by language modeling, where generation is guided by conditioning prompts. However, our framework differs fundamentally from LLMs in two key aspects. First, it employs a bidirectional modeling paradigm that symmetrically allows prompting either from the garment to generate try-on results or from the dressed person to recover the try-off garment. Second, it strictly adheres to Tweedie's formula, enabling faithful estimation of the underlying data distribution during the denoising process. Instead of imposing lower body constraints, OMFA is an entirely mask-free framework that requires only a single portrait and a target garment as input, and is designed to support flexible outfit combinations and cross-person garment transfer, making it better aligned with practical usage scenarios. Additionally, by leveraging SMPL-X-based pose conditioning, OMFA supports multi-view and arbitrary-pose try-on from just one image. Extensive experiments demonstrate that OMFA achieves state-of-the-art results on both try-on and try-off tasks, providing a practical solution for virtual garment synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16037v1">LLMs-based Augmentation for Domain Adaptation in Long-tailed Food Datasets</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      Training a model for food recognition is challenging because the training samples, which are typically crawled from the Internet, are visually different from the pictures captured by users in the free-living environment. In addition to this domain-shift problem, the real-world food datasets tend to be long-tailed distributed and some dishes of different categories exhibit subtle variations that are difficult to distinguish visually. In this paper, we present a framework empowered with large language models (LLMs) to address these challenges in food recognition. We first leverage LLMs to parse food images to generate food titles and ingredients. Then, we project the generated texts and food images from different domains to a shared embedding space to maximize the pair similarities. Finally, we take the aligned features of both modalities for recognition. With this simple framework, we show that our proposed approach can outperform the existing approaches tailored for long-tailed data distribution, domain adaptation, and fine-grained classification, respectively, on two food datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2406.13803v3">LLMs as Models for Analogical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 The title has been changed from Semantic Structure-Mapping in LLM and Human Analogical Reasoning to LLMs as Models for Analogical Reasoning to improve clarity and accuracy
    </div>
    <details class="paper-abstract">
      Analogical reasoning -- the capacity to identify and map structural relationships between different domains -- is fundamental to human cognition and learning. Recent studies have shown that large language models (LLMs) can sometimes match humans in analogical reasoning tasks, opening the possibility that analogical reasoning might emerge from domain-general processes. However, it is still debated whether these emergent capacities are largely superficial and limited to simple relations seen during training or whether they encompass the flexible representational and mapping capabilities which are the focus of leading cognitive models of analogy. In this study, we introduce novel analogical reasoning tasks that require participants to map between semantically contentful words and sequences of letters and other abstract characters. This task necessitates the ability to flexibly re-represent rich semantic information -- an ability which is known to be central to human analogy but which is thus far not well captured by existing cognitive theories and models. We assess the performance of both human participants and LLMs on tasks focusing on reasoning from semantic structure and semantic content, introducing variations that test the robustness of their analogical inferences. Advanced LLMs match human performance across several conditions, though humans and LLMs respond differently to certain task variations and semantic distractors. Our results thus provide new evidence that LLMs might offer a how-possibly explanation of human analogical reasoning in contexts that are not yet well modeled by existing theories, but that even today's best models are unlikely to yield how-actually explanations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16016v1">CARE: Turning LLMs Into Causal Reasoning Expert</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently demonstrated impressive capabilities across a range of reasoning and generation tasks. However, research studies have shown that LLMs lack the ability to identify causal relationships, a fundamental cornerstone of human intelligence. We first conduct an exploratory investigation of LLMs' behavior when asked to perform a causal-discovery task and find that they mostly rely on the semantic meaning of variable names, ignoring the observation data. This is unsurprising, given that LLMs were never trained to process structural datasets. To first tackle this challenge, we prompt the LLMs with the outputs of established causal discovery algorithms designed for observational datasets. These algorithm outputs effectively serve as the sufficient statistics of the observation data. However, quite surprisingly, we find that prompting the LLMs with these sufficient statistics decreases the LLMs' performance in causal discovery. To address this current limitation, we propose CARE, a framework that enhances LLMs' causal-reasoning ability by teaching them to effectively utilize the outputs of established causal-discovery algorithms through supervised fine-tuning. Experimental results show that a finetuned Qwen2.5-1.5B model produced by CARE significantly outperforms both traditional causal-discovery algorithms and state-of-the-art LLMs with over a thousand times more parameters, demonstrating effective utilization of its own knowledge and the external algorithmic clues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15998v1">Hiding in the AI Traffic: Abusing MCP for LLM-Powered Agentic Red Teaming</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 23 pages, 9 figures, 3 tables. Submitted as a full paper for review
    </div>
    <details class="paper-abstract">
      Generative AI is reshaping offensive cybersecurity by enabling autonomous red team agents that can plan, execute, and adapt during penetration tests. However, existing approaches face trade-offs between generality and specialization, and practical deployments reveal challenges such as hallucinations, context limitations, and ethical concerns. In this work, we introduce a novel command & control (C2) architecture leveraging the Model Context Protocol (MCP) to coordinate distributed, adaptive reconnaissance agents covertly across networks. Notably, we find that our architecture not only improves goal-directed behavior of the system as whole, but also eliminates key host and network artifacts that can be used to detect and prevent command & control behavior altogether. We begin with a comprehensive review of state-of-the-art generative red teaming methods, from fine-tuned specialist models to modular or agentic frameworks, analyzing their automation capabilities against task-specific accuracy. We then detail how our MCP-based C2 can overcome current limitations by enabling asynchronous, parallel operations and real-time intelligence sharing without periodic beaconing. We furthermore explore advanced adversarial capabilities of this architecture, its detection-evasion techniques, and address dual-use ethical implications, proposing defensive measures and controlled evaluation in lab settings. Experimental comparisons with traditional C2 show drastic reductions in manual effort and detection footprint. We conclude with future directions for integrating autonomous exploitation, defensive LLM agents, predictive evasive maneuvers, and multi-agent swarms. The proposed MCP-enabled C2 framework demonstrates a significant step toward realistic, AI-driven red team operations that can simulate advanced persistent threats while informing the development of next-generation defensive systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15996v1">QueryGym: A Toolkit for Reproducible LLM-Based Query Reformulation</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 4 pages
    </div>
    <details class="paper-abstract">
      We present QueryGym, a lightweight, extensible Python toolkit that supports large language model (LLM)-based query reformulation. This is an important tool development since recent work on llm-based query reformulation has shown notable increase in retrieval effectiveness. However, while different authors have sporadically shared the implementation of their methods, there is no unified toolkit that provides a consistent implementation of such methods, which hinders fair comparison, rapid experimentation, consistent benchmarking and reliable deployment. QueryGym addresses this gap by providing a unified framework for implementing, executing, and comparing llm-based reformulation methods. The toolkit offers: (1) a Python API for applying diverse LLM-based methods, (2) a retrieval-agnostic interface supporting integration with backends such as Pyserini and PyTerrier, (3) a centralized prompt management system with versioning and metadata tracking, (4) built-in support for benchmarks like BEIR and MS MARCO, and (5) a completely open-source extensible implementation available to all researchers. QueryGym is publicly available at https://github.com/radinhamidi/QueryGym.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.16785v3">Interpreting the Effects of Quantization on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 Accepted to AACL 2025 Main
    </div>
    <details class="paper-abstract">
      Quantization offers a practical solution to deploy LLMs in resource-constraint environments. However, its impact on internal representations remains understudied, raising questions about the reliability of quantized models. In this study, we employ a range of interpretability techniques to investigate how quantization affects model and neuron behavior. We analyze multiple LLMs under 4-bit and 8-bit quantization. Our findings reveal that the impact of quantization on model calibration is generally minor. Analysis of neuron activations indicates that the number of dead neurons, i.e., those with activation values close to 0 across the dataset, remains consistent regardless of quantization. In terms of neuron contribution to predictions, we observe that smaller full precision models exhibit fewer salient neurons, whereas larger models tend to have more, with the exception of Llama-2-7B. The effect of quantization on neuron redundancy varies across models. Overall, our findings suggest that effect of quantization may vary by model and tasks, however, we did not observe any drastic change which may discourage the use of quantization as a reliable model compression technique.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15974v1">KRAL: Knowledge and Reasoning Augmented Learning for LLM-assisted Clinical Antimicrobial Therapy</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      Clinical antimicrobial therapy requires the dynamic integration of pathogen profiles, host factors, pharmacological properties of antimicrobials, and the severity of infection.This complexity imposes fundamental limitations on the applicability of Large Language Models (LLMs) in high-stakes clinical decision-making including knowledge gaps, data privacy concerns, high deployment costs, and limited reasoning capabilities. To address these challenges, we propose KRAL (Knowledge and Reasoning Augmented Learning), a low-cost, scalable, privacy-preserving paradigm that leverages teacher-model reasoning to automatically distill knowledge and reasoning trajectories via answer-to-question reverse generation, employs heuristic learning for semi-supervised data augmentation (reducing manual annotation requirements by approximately 80%), and utilizes agentic reinforcement learning to jointly enhance medical knowledge and reasoning while optimizing computational and memory efficiency. A hierarchical evaluation employing diverse teacher-model proxies reduces assessment costs, while modular interface design facilitates seamless system updates. Experimental results demonstrate that KRAL significantly outperforms traditional Retrieval-Augmented Generation (RAG) and Supervised Fine-Tuning (SFT) methods. It improves knowledge question-answering capability (Accuracy@1 on the external open-source benchmark MEDQA increased by 1.8% vs. SFT and 3.6% vs. RAG) and reasoning capability (Pass@1 on the external benchmark PUMCH Antimicrobial increased by 27% vs. SFT and 27.2% vs. RAG), achieved at ~20% of SFT's long-term training costs. This establishes KRAL as an effective solution for enhancing local LLMs' clinical diagnostic capabilities, enabling low-cost, high-safety deployment in complex medical decision support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2401.17435v5">Can LLMs Replace Economic Choice Prediction Labs? The Case of Language-based Persuasion Games</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      Human choice prediction in economic contexts is crucial for applications in marketing, finance, public policy, and more. This task, however, is often constrained by the difficulties in acquiring human choice data. With most experimental economics studies focusing on simple choice settings, the AI community has explored whether LLMs can substitute for humans in these predictions and examined more complex experimental economics settings. However, a key question remains: can LLMs generate training data for human choice prediction? We explore this in language-based persuasion games, a complex economic setting involving natural language in strategic interactions. Our experiments show that models trained on LLM-generated data can effectively predict human behavior in these games and even outperform models trained on actual human data. Beyond data generation, we investigate the dual role of LLMs as both data generators and predictors, introducing a comprehensive empirical study on the effectiveness of utilizing LLMs for data generation, human choice prediction, or both. We then utilize our choice prediction framework to analyze how strategic factors shape decision-making, showing that interaction history (rather than linguistic sentiment alone) plays a key role in predicting human decision-making in repeated interactions. Particularly, when LLMs capture history-dependent decision patterns similarly to humans, their predictive success improves substantially. Finally, we demonstrate the robustness of our findings across alternative persuasion-game settings, highlighting the broader potential of using LLM-generated data to model human decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2402.08546v3">Grounding LLMs For Robot Task Planning Using Closed-loop State Feedback</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 Preprint version. Accepted full paper available here: https://advanced.onlinelibrary.wiley.com/doi/10.1002/adrr.202500072
    </div>
    <details class="paper-abstract">
      Planning algorithms decompose complex problems into intermediate steps that can be sequentially executed by robots to complete tasks. Recent works have employed Large Language Models (LLMs) for task planning, using natural language to generate robot policies in both simulation and real-world environments. LLMs like GPT-4 have shown promising results in generalizing to unseen tasks, but their applicability is limited due to hallucinations caused by insufficient grounding in the robot environment. The robustness of LLMs in task planning can be enhanced with environmental state information and feedback. In this paper, we introduce a novel approach to task planning that utilizes two separate LLMs for high-level planning and low-level control, improving task-related success rates and goal condition recall. Our algorithm, \textit{BrainBody-LLM}, draws inspiration from the human neural system, emulating its brain-body architecture by dividing planning across two LLMs in a structured, hierarchical manner. BrainBody-LLM implements a closed-loop feedback mechanism, enabling learning from simulator errors to resolve execution errors in complex settings. We demonstrate the successful application of BrainBody-LLM in the VirtualHome simulation environment, achieving a 29\% improvement in task-oriented success rates over competitive baselines with the GPT-4 backend. Additionally, we evaluate our algorithm on seven complex tasks using a realistic physics simulator and the Franka Research 3 robotic arm, comparing it with various state-of-the-art LLMs. Our results show advancements in the reasoning capabilities of recent LLMs, which enable them to learn from raw simulator/controller errors to correct plans, making them highly effective in robotic task planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16846v1">ConCISE: A Reference-Free Conciseness Evaluation Metric for LLM-Generated Answers</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) frequently generate responses that are lengthy and verbose, filled with redundant or unnecessary details. This diminishes clarity and user satisfaction, and it increases costs for model developers, especially with well-known proprietary models that charge based on the number of output tokens. In this paper, we introduce a novel reference-free metric for evaluating the conciseness of responses generated by LLMs. Our method quantifies non-essential content without relying on gold standard references and calculates the average of three calculations: i) a compression ratio between the original response and an LLM abstractive summary; ii) a compression ratio between the original response and an LLM extractive summary; and iii) wordremoval compression, where an LLM removes as many non-essential words as possible from the response while preserving its meaning, with the number of tokens removed indicating the conciseness score. Experimental results demonstrate that our proposed metric identifies redundancy in LLM outputs, offering a practical tool for automated evaluation of response brevity in conversational AI systems without the need for ground truth human annotations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16837v1">Cognitive BASIC: An In-Model Interpreted Reasoning Language for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 6 pages, Submitted to ESANN 2026
    </div>
    <details class="paper-abstract">
      Cognitive BASIC is a minimal, BASIC-style prompting language and in-model interpreter that structures large language model (LLM) reasoning into explicit, stepwise execution traces. Inspired by the simplicity of retro BASIC, we repurpose numbered lines and simple commands as an interpretable cognitive control layer. Modern LLMs can reliably simulate such short programs, enabling transparent multi-step reasoning inside the model. A natural-language interpreter file specifies command semantics, memory updates, and logging behavior. Our mental-model interpreter extracts declarative and procedural knowledge, detects contradictions, and produces resolutions when necessary. A comparison across three LLMs on a benchmark of knowledge extraction, conflict detection, and reasoning tasks shows that all models can execute Cognitive BASIC programs, with overall strong but not uniform performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.16549v2">ReviewGuard: Enhancing Deficient Peer Review Detection via LLM-Driven Data Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 Accepted as a full paper at the 2025 ACM/IEEE Joint Conference on Digital Libraries (JCDL 2025)
    </div>
    <details class="paper-abstract">
      Peer review serves as the gatekeeper of science, yet the surge in submissions and widespread adoption of large language models (LLMs) in scholarly evaluation present unprecedented challenges. While recent work has focused on using LLMs to improve review efficiency, unchecked deficient reviews from both human experts and AI systems threaten to systematically undermine academic integrity. To address this issue, we introduce ReviewGuard, an automated system for detecting and categorizing deficient reviews through a four-stage LLM-driven framework: data collection from ICLR and NeurIPS on OpenReview, GPT-4.1 annotation with human validation, synthetic data augmentation yielding 6,634 papers with 24,657 real and 46,438 synthetic reviews, and fine-tuning of encoder-based models and open-source LLMs. Feature analysis reveals that deficient reviews exhibit lower rating scores, higher self-reported confidence, reduced structural complexity, and more negative sentiment than sufficient reviews. AI-generated text detection shows dramatic increases in AI-authored reviews since ChatGPT's emergence. Mixed training with synthetic and real data substantially improves detection performance - for example, Qwen 3-8B achieves recall of 0.6653 and F1 of 0.7073, up from 0.5499 and 0.5606 respectively. This study presents the first LLM-driven system for detecting deficient peer reviews, providing evidence to inform AI governance in peer review. Code, prompts, and data are available at https://github.com/haoxuan-unt2024/ReviewGuard
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.24068v2">A Small Math Model: Recasting Strategy Choice Theory in an LLM-Inspired Architecture</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      Strategy Choice Theory (SCT; Siegler and Shrager, 1984; Siegler, 2000) explains important aspects of children's arithmetic learning based upon principles including learning from developmentally naturalistic data, probabilistic representation, confidence-based retrieval, and the phase-like importance of scaffolding strategies, such as finger-counting. Here we recast SCT as a ``Small Math Model'' (SMM), employing a neural-network-based architecture analogous to LLMs. The SMM extends SCT to include counting practice, symbol (number) embedding, and gated attention. Similar to earlier work, the SMM demonstrates constructive and destructive interference between counting and addition, and the ``wave-like'' use of finger-counting as sum recall improves. We plan to extend the SMM to later aspects of the decades-long SCT program, including adaptive strategy choice and eventually strategy discovery, providing a unified platform to investigate the understanding of numerical characteristics and relationships essential for mathematical reasoning -- as it can emerge in LLM-based agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.13290v2">Towards Formal Verification of LLM-Generated Code from Natural Language Prompts</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 28 pages, 10 figures
    </div>
    <details class="paper-abstract">
      In the past few years LLMs have emerged as a tool that can aid programmers by taking natural language descriptions and generating code based on it. However, the reliability of LLM code generation and current validation techniques for it are far from strong enough to be used for mission-critical or safety-critical applications. In this work we explore ways to offer formal guarantees of correctness to LLM generated code; such guarantees could improve the quality of general AI Code Assistants and support their use for critical applications. To address this challenge we propose to incorporate a Formal Query Language that can represent a user's intent in a formally defined but natural language-like manner that a user can confirm matches their intent. We then have a formal specification of the user intent which we can use to verify that LLM-generated code matches the user's intent. We implement these ideas in our system, Astrogator, for the Ansible programming language, widely used for system administration, including for critical systems. The system includes an intuitive formal query language, a calculus for representing the behavior of Ansible programs, and a symbolic interpreter and a unification algorithm which together are used for the verification. A key innovation in Astrogator is the use of a Knowledge Base to capture system-specific implementation dependencies that greatly reduce the need for system knowledge in expressing formal queries. On a benchmark suite of 21 code-generation tasks, our verifier is able to verify correct code in 83% of cases and identify incorrect code in 92%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16767v1">When Structure Doesn't Help: LLMs Do Not Read Text-Attributed Graphs as Effectively as We Expected</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      Graphs provide a unified representation of semantic content and relational structure, making them a natural fit for domains such as molecular modeling, citation networks, and social graphs. Meanwhile, large language models (LLMs) have excelled at understanding natural language and integrating cross-modal signals, sparking interest in their potential for graph reasoning. Recent work has explored this by either designing template-based graph templates or using graph neural networks (GNNs) to encode structural information. In this study, we investigate how different strategies for encoding graph structure affect LLM performance on text-attributed graphs. Surprisingly, our systematic experiments reveal that: (i) LLMs leveraging only node textual descriptions already achieve strong performance across tasks; and (ii) most structural encoding strategies offer marginal or even negative gains. We show that explicit structural priors are often unnecessary and, in some cases, counterproductive when powerful language models are involved. This represents a significant departure from traditional graph learning paradigms and highlights the need to rethink how structure should be represented and utilized in the LLM era. Our study is to systematically challenge the foundational assumption that structure is inherently beneficial for LLM-based graph reasoning, opening the door to new, semantics-driven approaches for graph learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16709v1">AutoBackdoor: Automating Backdoor Attacks via LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 23 pages
    </div>
    <details class="paper-abstract">
      Backdoor attacks pose a serious threat to the secure deployment of large language models (LLMs), enabling adversaries to implant hidden behaviors triggered by specific inputs. However, existing methods often rely on manually crafted triggers and static data pipelines, which are rigid, labor-intensive, and inadequate for systematically evaluating modern defense robustness. As AI agents become increasingly capable, there is a growing need for more rigorous, diverse, and scalable \textit{red-teaming frameworks} that can realistically simulate backdoor threats and assess model resilience under adversarial conditions. In this work, we introduce \textsc{AutoBackdoor}, a general framework for automating backdoor injection, encompassing trigger generation, poisoned data construction, and model fine-tuning via an autonomous agent-driven pipeline. Unlike prior approaches, AutoBackdoor uses a powerful language model agent to generate semantically coherent, context-aware trigger phrases, enabling scalable poisoning across arbitrary topics with minimal human effort. We evaluate AutoBackdoor under three realistic threat scenarios, including \textit{Bias Recommendation}, \textit{Hallucination Injection}, and \textit{Peer Review Manipulation}, to simulate a broad range of attacks. Experiments on both open-source and commercial models, including LLaMA-3, Mistral, Qwen, and GPT-4o, demonstrate that our method achieves over 90\% attack success with only a small number of poisoned samples. More importantly, we find that existing defenses often fail to mitigate these attacks, underscoring the need for more rigorous and adaptive evaluation techniques against agent-driven threats as explored in this work. All code, datasets, and experimental configurations will be merged into our primary repository at https://github.com/bboylyg/BackdoorLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17662v1">Enhancing Breast Cancer Prediction with LLM-Inferred Confounders</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 2 pages, 1 figure, 1 table
    </div>
    <details class="paper-abstract">
      This study enhances breast cancer prediction by using large language models to infer the likelihood of confounding diseases, namely diabetes, obesity, and cardiovascular disease, from routine clinical data. These AI-generated features improved Random Forest model performance, particularly for LLMs like Gemma (3.9%) and Llama (6.4%). The approach shows promise for noninvasive prescreening and clinical integration, supporting improved early detection and shared decision-making in breast cancer diagnosis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.05034v2">Eguard: Defending LLM Embeddings Against Inversion Attacks via Text Mutual Information Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      Embeddings have become a cornerstone in the functionality of large language models (LLMs) due to their ability to transform text data into rich, dense numerical representations that capture semantic and syntactic properties. These embedding vector databases serve as the long-term memory of LLMs, enabling efficient handling of a wide range of natural language processing tasks. However, the surge in popularity of embedding vector databases in LLMs has been accompanied by significant concerns about privacy leakage. Embedding vector databases are particularly vulnerable to embedding inversion attacks, where adversaries can exploit the embeddings to reverse-engineer and extract sensitive information from the original text data. Existing defense mechanisms have shown limitations, often struggling to balance security with the performance of downstream tasks. To address these challenges, we introduce Eguard, a novel defense mechanism designed to mitigate embedding inversion attacks. Eguard employs a transformer-based projection network and text mutual information optimization to safeguard embeddings while preserving the utility of LLMs. Our approach significantly reduces privacy risks, protecting over 95% of tokens from inversion while maintaining high performance across downstream tasks consistent with original embeddings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15921v1">Thinking, Faithful and Stable: Mitigating Hallucinations in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-19
      | 💬 Originally released June 5, 2025
    </div>
    <details class="paper-abstract">
      This project develops a self correcting framework for large language models (LLMs) that detects and mitigates hallucinations during multi-step reasoning. Rather than relying solely on final answer correctness, our approach leverages fine grained uncertainty signals: 1) self-assessed confidence alignment, and 2) token-level entropy spikes to detect unreliable and unfaithful reasoning in real time. We design a composite reward function that penalizes unjustified high confidence and entropy spikes, while encouraging stable and accurate reasoning trajectories. These signals guide a reinforcement learning (RL) policy that makes the model more introspective and shapes the model's generation behavior through confidence-aware reward feedback, improving not just outcome correctness but the coherence and faithfulness of their intermediate reasoning steps. Experiments show that our method improves both final answer accuracy and reasoning calibration, with ablations validating the individual contribution of each signal.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15915v1">AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      We present AccelOpt, a self-improving large language model (LLM) agentic system that autonomously optimizes kernels for emerging AI acclerators, eliminating the need for expert-provided hardware-specific optimization knowledge. AccelOpt explores the kernel optimization space through iterative generation, informed by an optimization memory that curates experiences and insights from previously encountered slow-fast kernel pairs. We build NKIBench, a new benchmark suite of AWS Trainium accelerator kernels with varying complexity extracted from real-world LLM workloads to evaluate the effectiveness of AccelOpt. Our evaluation confirms that AccelOpt's capability improves over time, boosting the average percentage of peak throughput from $49\%$ to $61\%$ on Trainium 1 and from $45\%$ to $59\%$ on Trainium 2 for NKIBench kernels. Moreover, AccelOpt is highly cost-effective: using open-source models, it matches the kernel improvements of Claude Sonnet 4 while being $26\times$ cheaper.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.12339v4">A Closer Look at Adversarial Suffix Learning for Jailbreaking LLMs: Augmented Adversarial Trigger Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-19
      | 💬 the Association for Computational Linguistics: NAACL 2025
    </div>
    <details class="paper-abstract">
      Gradient optimization-based adversarial attack methods automate the learning of adversarial triggers to generate jailbreak prompts or leak system prompts. In this work, we take a closer look at the optimization objective of adversarial trigger learning and propose ATLA: Adversarial Trigger Learning with Augmented objectives. ATLA improves the negative log-likelihood loss used by previous studies into a weighted loss formulation that encourages the learned adversarial triggers to optimize more towards response format tokens. This enables ATLA to learn an adversarial trigger from just one query-response pair and the learned trigger generalizes well to other similar queries. We further design a variation to augment trigger optimization with an auxiliary loss that suppresses evasive responses. We showcase how to use ATLA to learn adversarial suffixes jailbreaking LLMs and to extract hidden system prompts. Empirically we demonstrate that ATLA consistently outperforms current state-of-the-art techniques, achieving nearly 100% success in attacking while requiring 80% fewer queries. ATLA learned jailbreak suffixes demonstrate high generalization to unseen queries and transfer well to new LLMs. We released our code https://github.com/QData/ALTA_Augmented_Adversarial_Trigger_Learning
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15895v1">Decomposing Theory of Mind: How Emotional Processing Mediates ToM Abilities in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-19
      | 💬 Published at ToM4AI workshop@AAAI2026
    </div>
    <details class="paper-abstract">
      Recent work shows activation steering substantially improves language models' Theory of Mind (ToM) (Bortoletto et al. 2024), yet the mechanisms of what changes occur internally that leads to different outputs remains unclear. We propose decomposing ToM in LLMs by comparing steered versus baseline LLMs' activations using linear probes trained on 45 cognitive actions. We applied Contrastive Activation Addition (CAA) steering to Gemma-3-4B and evaluated it on 1,000 BigToM forward belief scenarios (Gandhi et al. 2023), we find improved performance on belief attribution tasks (32.5\% to 46.7\% accuracy) is mediated by activations processing emotional content : emotion perception (+2.23), emotion valuing (+2.20), while suppressing analytical processes: questioning (-0.78), convergent thinking (-1.59). This suggests that successful ToM abilities in LLMs are mediated by emotional understanding, not analytical reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15862v1">The Subtle Art of Defection: Understanding Uncooperative Behaviors in LLM based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      This paper introduces a novel framework for simulating and analyzing how uncooperative behaviors can destabilize or collapse LLM-based multi-agent systems. Our framework includes two key components: (1) a game theory-based taxonomy of uncooperative agent behaviors, addressing a notable gap in the existing literature; and (2) a structured, multi-stage simulation pipeline that dynamically generates and refines uncooperative behaviors as agents' states evolve. We evaluate the framework via a collaborative resource management setting, measuring system stability using metrics such as survival time and resource overuse rate. Empirically, our framework achieves 96.7% accuracy in generating realistic uncooperative behaviors, validated by human evaluations. Our results reveal a striking contrast: cooperative agents maintain perfect system stability (100% survival over 12 rounds with 0% resource overuse), while any uncooperative behavior can trigger rapid system collapse within 1 to 7 rounds. These findings demonstrate that uncooperative agents can significantly degrade collective outcomes, highlighting the need for designing more resilient multi-agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13984v2">Node-Level Uncertainty Estimation in LLM-Generated SQL</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      We present a practical framework for detecting errors in LLM-generated SQL by estimating uncertainty at the level of individual nodes in the query's abstract syntax tree (AST). Our approach proceeds in two stages. First, we introduce a semantically aware labeling algorithm that, given a generated SQL and a gold reference, assigns node-level correctness without over-penalizing structural containers or alias variation. Second, we represent each node with a rich set of schema-aware and lexical features - capturing identifier validity, alias resolution, type compatibility, ambiguity in scope, and typo signals - and train a supervised classifier to predict per-node error probabilities. We interpret these probabilities as calibrated uncertainty, enabling fine-grained diagnostics that pinpoint exactly where a query is likely to be wrong. Across multiple databases and datasets, our method substantially outperforms token log-probabilities: average AUC improves by +27.44% while maintaining robustness under cross-database evaluation. Beyond serving as an accuracy signal, node-level uncertainty supports targeted repair, human-in-the-loop review, and downstream selective execution. Together, these results establish node-centric, semantically grounded uncertainty estimation as a strong and interpretable alternative to aggregate sequence level confidence measures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15817v1">A Causal Perspective on Measuring, Explaining and Mitigating Smells in \llm-Generated Code</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have accelerated their adoption in software engineering contexts. However, concerns persist about the structural quality of the code they produce. In particular, LLMs often replicate poor coding practices, introducing code smells (i.e., patterns that hinder readability, maintainability, or design integrity). Although prior research has examined the detection or repair of smells, we still lack a clear understanding of how and when these issues emerge in generated code. This paper addresses this gap by systematically measuring, explaining and mitigating smell propensity in LLM-generated code. We build on the Propensity Smelly Score (PSC), a probabilistic metric that estimates the likelihood of generating particular smell types, and establish its robustness as a signal of structural quality. Using PSC as an instrument for causal analysis, we identify how generation strategy, model size, model architecture and prompt formulation shape the structural properties of generated code. Our findings show that prompt design and architectural choices play a decisive role in smell propensity and motivate practical mitigation strategies that reduce its occurrence. A user study further demonstrates that PSC helps developers interpret model behavior and assess code quality, providing evidence that smell propensity signals can support human judgement. Taken together, our work lays the groundwork for integrating quality-aware assessments into the evaluation and deployment of LLMs for code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.17061v2">Parallelism Meets Adaptiveness: Scalable Documents Understanding in Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-11-19
      | 💬 Accepted at AAAI 2026 Workshop on WoMAPF
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have shown increasing promise for collaborative task completion. However, existing multi-agent frameworks often rely on static workflows, fixed roles, and limited inter-agent communication, reducing their effectiveness in open-ended, high-complexity domains. This paper proposes a coordination framework that enables adaptiveness through three core mechanisms: dynamic task routing, bidirectional feedback, and parallel agent evaluation. The framework allows agents to reallocate tasks based on confidence and workload, exchange structured critiques to iteratively improve outputs, and crucially compete on high-ambiguity subtasks with evaluator-driven selection of the most suitable result. We instantiate these principles in a modular architecture and demonstrate substantial improvements in factual coverage, coherence, and efficiency over static and partially adaptive baselines. Our findings highlight the benefits of incorporating both adaptiveness and structured competition in multi-agent LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15762v1">A time for monsters: Organizational knowing after LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-19
      | 💬 Forthcoming at Strategic Organization
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are reshaping organizational knowing by unsettling the epistemological foundations of representational and practice-based perspectives. We conceptualize LLMs as Haraway-ian monsters, that is, hybrid, boundary-crossing entities that destabilize established categories while opening new possibilities for inquiry. Focusing on analogizing as a fundamental driver of knowledge, we examine how LLMs generate connections through large-scale statistical inference. Analyzing their operation across the dimensions of surface/deep analogies and near/far domains, we highlight both their capacity to expand organizational knowing and the epistemic risks they introduce. Building on this, we identify three challenges of living with such epistemic monsters: the transformation of inquiry, the growing need for dialogical vetting, and the redistribution of agency. By foregrounding the entangled dynamics of knowing-with-LLMs, the paper extends organizational theory beyond human-centered epistemologies and invites renewed attention to how knowledge is created, validated, and acted upon in the age of intelligent technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15757v1">Rethinking Kernel Program Repair: Benchmarking and Enhancing LLMs with RGym</a></div>
    <div class="paper-meta">
      📅 2025-11-19
      | 💬 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: Evaluating the Evolving LLM Lifecycle: Benchmarks, Emergent Abilities, and Scaling
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized automated program repair (APR) but current benchmarks like SWE-Bench predominantly focus on userspace applications and overlook the complexities of kernel-space debugging and repair. The Linux kernel poses unique challenges due to its monolithic structure, concurrency, and low-level hardware interactions. Prior efforts such as KGym and CrashFixer have highlighted the difficulty of APR in this domain, reporting low success rates or relying on costly and complex pipelines and pricey cloud infrastructure. In this work, we introduce RGym, a lightweight, platform-agnostic APR evaluation framework for the Linux kernel designed to operate on local commodity hardware. Built on RGym, we propose a simple yet effective APR pipeline leveraging specialized localization techniques (e.g., call stacks and blamed commits) to overcome the unrealistic usage of oracles in KGym. We test on a filtered and verified dataset of 143 bugs. Our method achieves up to a 43.36% pass rate with GPT-5 Thinking while maintaining a cost of under $0.20 per bug. We further conduct an ablation study to analyze contributions from our proposed localization strategy, prompt structure, and model choice, and demonstrate that feedback-based retries can significantly enhance success rates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15755v1">Multi-Agent LLM Orchestration Achieves Deterministic, High-Quality Decision Support for Incident Response</a></div>
    <div class="paper-meta">
      📅 2025-11-19
      | 💬 8 pages, 4 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) promise to accelerate incident response in production systems, yet single-agent approaches generate vague, unusable recommendations. We present MyAntFarm.ai, a reproducible containerized framework demonstrating that multi-agent orchestration fundamentally transforms LLM-based incident response quality. Through 348 controlled trials comparing single-agent copilot versus multi-agent systems on identical incident scenarios, we find that multi-agent orchestration achieves 100% actionable recommendation rate versus 1.7% for single-agent approaches, an 80 times improvement in action specificity and 140 times improvement in solution correctness. Critically, multi-agent systems exhibit zero quality variance across all trials, enabling production SLA commitments impossible with inconsistent single-agent outputs. Both architectures achieve similar comprehension latency (approx.40s), establishing that the architectural value lies in deterministic quality, not speed. We introduce Decision Quality (DQ), a novel metric capturing validity, specificity, and correctness properties essential for operational deployment that existing LLM metrics do not address. These findings reframe multi-agent orchestration from a performance optimization to a production-readiness requirement for LLM-based incident response. All code, Docker configurations, and trial data are publicly available for reproduction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15698v1">RescueLens: LLM-Powered Triage and Action on Volunteer Feedback for Food Rescue</a></div>
    <div class="paper-meta">
      📅 2025-11-19
      | 💬 Accepted at IAAI'26
    </div>
    <details class="paper-abstract">
      Food rescue organizations simultaneously tackle food insecurity and waste by working with volunteers to redistribute food from donors who have excess to recipients who need it. Volunteer feedback allows food rescue organizations to identify issues early and ensure volunteer satisfaction. However, food rescue organizations monitor feedback manually, which can be cumbersome and labor-intensive, making it difficult to prioritize which issues are most important. In this work, we investigate how large language models (LLMs) assist food rescue organizers in understanding and taking action based on volunteer experiences. We work with 412 Food Rescue, a large food rescue organization based in Pittsburgh, Pennsylvania, to design RescueLens, an LLM-powered tool that automatically categorizes volunteer feedback, suggests donors and recipients to follow up with, and updates volunteer directions based on feedback. We evaluate the performance of RescueLens on an annotated dataset, and show that it can recover 96% of volunteer issues at 71% precision. Moreover, by ranking donors and recipients according to their rates of volunteer issues, RescueLens allows organizers to focus on 0.5% of donors responsible for more than 30% of volunteer issues. RescueLens is now deployed at 412 Food Rescue and through semi-structured interviews with organizers, we find that RescueLens streamlines the feedback process so organizers better allocate their time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.08640v2">Automating Android Build Repair: Bridging the Reasoning-Execution Gap in LLM Agents with Domain-Specific Tools</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      Android is the largest mobile platform, yet automatically building applications remains a practical challenge. While Large Language Models (LLMs) show promise for code repair, their use for fixing Android build errors remains underexplored. To address this gap, we first introduce AndroidBuildBench, a benchmark of 1,019 build failures curated from the commit histories of 43 open-source Android projects. Each problem is paired with a verified solution from a subsequent commit, ensuring that fixes are feasible. Second, we propose GradleFixer, an LLM agent with domain-specific tools for inspecting and manipulating the Gradle build environment. GradleFixer achieves a resolve rate of 81.4% (pass@1), significantly outperforming a state-of-the-art coding agent that relies on a general-purpose shell. GradleFixer's success suggests that while LLMs possess the high-level knowledge to solve these failures, they struggle to translate this knowledge into effective low-level actions using a general-purpose shell. We demonstrate the effectiveness of a strategy we term Tool Bridging, which replaces general-purpose shell commands with domain-aware abstractions. We hypothesize this approach works through two mechanisms: 1) it provides tools in an API-like format that LLMs use more reliably, and 2) it constrains the action space to relevant operations. This approach bridges the gap between the model's high-level reasoning and effective low-level execution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.03628v4">LLMDistill4Ads: Using Cross-Encoders to Distill from LLM Signals for Advertiser Keyphrase Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      E-commerce sellers are advised to bid on keyphrases to boost their advertising campaigns. These keyphrases must be relevant to prevent irrelevant items from cluttering search systems and to maintain positive seller perception. It is vital that keyphrase suggestions align with seller, search and buyer judgments. Given the challenges in collecting negative feedback in these systems, LLMs have been used as a scalable proxy to human judgments. This paper presents an empirical study on a major ecommerce platform of a distillation framework involving an LLM teacher, a cross-encoder assistant and a bi-encoder Embedding Based Retrieval (EBR) student model, aimed at mitigating click-induced biases in keyphrase recommendations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15676v1">DuoZone: A User-Centric, LLM-Guided Mixed-Initiative XR Window Management System</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      Mixed reality (XR) environments offer vast spatial possibilities, but current window management systems require users to manually place, resize, and organize multiple applications across large 3D spaces. This creates cognitive and interaction burdens that limit productivity. We introduce DuoZone, a mixed-initiative XR window management system that combines user-defined spatial layouts with LLM-guided automation. DuoZone separates window management into two complementary zones. The Recommendation Zone enables fast setup by providing spatial layout templates and automatically recommending relevant applications based on user tasks and high-level goals expressed through voice or text. The Arrangement Zone supports precise refinement through direct manipulation, allowing users to adjust windows using natural spatial actions such as dragging, resizing, and snapping. Through this dual-zone approach, DuoZone promotes efficient organization while reducing user cognitive load. We conducted a user study comparing DuoZone with a baseline manual XR window manager. Results show that DuoZone improves task completion speed, reduces mental effort, and increases sense of control when working with multiple applications in XR. We discuss design implications for future mixed-initiative systems and outline opportunities for integrating adaptive, goal-aware intelligence into spatial computing workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15665v1">Quantum-Guided Test Case Minimization for LLM-Based Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-19
      | 💬 This is a preprint version, full paper has been accepted in IEEE CASCON 2025 and will appear on lEEE Xplore
    </div>
    <details class="paper-abstract">
      Precisely controlling Large Language Models (LLMs) to generate efficient and concise code is a central challenge in software engineering. We introduce a framework based on Test-Driven Development (TDD) that transforms code specification into a combinatorial optimization task. The framework first prompts an LLM to generate a test suite, then formulates the Test Case Minimization (TCM) problem as a Quadratic Unconstrained Binary Optimization (QUBO) model. This QUBO paradigm is compatible with both classical solvers and emerging hardware such as quantum annealers. Experimentally, quantum annealing solves the core TCM task 16 times faster than simulated annealing. This performance underpins our end-to-end framework, which reduces total token consumption by 36.5\% and significantly improves code quality. This work demonstrates a powerful synergy between generative AI and combinatorial optimization in software engineering, highlighting the critical importance of precise model formulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14214v2">Do Large Language Models (LLMs) Understand Chronology?</a></div>
    <div class="paper-meta">
      📅 2025-11-19
      | 💬 Version 2: corrected footnote and added code repository link. Extended version of our work presented at the AAAI-26 AI4TS Workshop (poster) and AAAI-26 Student Abstract Program (oral)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in finance and economics, where prompt-based attempts against look-ahead bias implicitly assume that models understand chronology. We test this fundamental question with a series of chronological ordering tasks with increasing complexities over facts the model already knows from pre-training. Our tasks cover (1) chronological ordering, (2) conditional sorting (filter, then order), and (3) anachronism detection. We evaluate GPT-4.1, Claude-3.7 Sonnet, with and without Extended Thinking (ET), and GPT-5 across multiple reasoning-effort settings. Across models, Exact match rate drops sharply as sequences lengthen even while rank correlations stay high as LLMs largely preserve local order but struggle to maintain a single globally consistent timeline. In conditional sorting, most failures stem from the filtering step rather than the ordering step, but GPT-5 and Claude-3.7 Sonnet with Extended Thinking outshine normal models significantly. Lastly, anachronism detection is found to be the easiest task for the LLMs but performance still declines with increasingly overlapping timelines or entities. Overall, our main contribution is showing that allocating explicit reasoning budget helps with chronological ordering with GPT-5 at medium/high reasoning effort achieving flawless ordering at all lengths and perfect conditional sorting (both self-filtered and given-subset), whereas low/minimal effort degrades with longer lists, mirroring earlier models. Our findings delineate limits of current LLMs on chronological tasks, providing insights into task complexity, and demonstrate scenarios in which reasoning helps. These patterns are important for the real-time application of LLMs in finance. We release all code and evaluation templates to support full reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15504v1">Game Master LLM: Task-Based Role-Playing for Natural Slang Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      Natural and idiomatic expressions are essential for fluent, everyday communication, yet many second-language learners struggle to acquire and spontaneously use casual slang despite strong formal proficiency. To address this gap, we designed and evaluated an LLM-powered, task-based role-playing game in which a GPT-4o-based Game Master guides learners through an immersive, three-phase spoken narrative. After selecting five unfamiliar slang phrases to practice, participants engage in open-ended dialogue with non-player characters; the Game Master naturally incorporates the target phrases in rich semantic contexts (implicit input enhancement) while a dedicated Practice Box provides real-time explicit tracking and encouragement. Post-session, learners receive multi-level formative feedback analyzing the entire interaction. We evaluated the system in a between-subjects study with 14 international graduate students, randomly assigned to either the RPG condition or a control condition consisting of a traditional AI-led virtual classroom. Results from an immediate post-test show that the RPG group achieved greater gains in both comprehension of the target phrases and their accurate, contextual use in sentences. Quantitative analysis of in-activity word-usage frequency, combined with qualitative survey responses, further indicates that the game-based approach provided more practice opportunities and higher perceived engagement, resulting in a more natural learning experience. These findings highlight the potential of narrative-driven LLM interactions in vocabulary acquisition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15456v1">Know Your Intent: An Autonomous Multi-Perspective LLM Agent Framework for DeFi User Transaction Intent Mining</a></div>
    <div class="paper-meta">
      📅 2025-11-19
      | 💬 Written in 2025 Q1
    </div>
    <details class="paper-abstract">
      As Decentralized Finance (DeFi) develops, understanding user intent behind DeFi transactions is crucial yet challenging due to complex smart contract interactions, multifaceted on-/off-chain factors, and opaque hex logs. Existing methods lack deep semantic insight. To address this, we propose the Transaction Intent Mining (TIM) framework. TIM leverages a DeFi intent taxonomy built on grounded theory and a multi-agent Large Language Model (LLM) system to robustly infer user intents. A Meta-Level Planner dynamically coordinates domain experts to decompose multiple perspective-specific intent analyses into solvable subtasks. Question Solvers handle the tasks with multi-modal on/off-chain data. While a Cognitive Evaluator mitigates LLM hallucinations and ensures verifiability. Experiments show that TIM significantly outperforms machine learning models, single LLMs, and single Agent baselines. We also analyze core challenges in intent inference. This work helps provide a more reliable understanding of user motivations in DeFi, offering context-aware explanations for complex blockchain activity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.08343v3">A Data-driven ML Approach for Maximizing Performance in LLM-Adapter Serving</a></div>
    <div class="paper-meta">
      📅 2025-11-19
      | 💬 Accepted in a computer science workshop
    </div>
    <details class="paper-abstract">
      With the rapid adoption of Large Language Models (LLMs), LLM-adapters have become increasingly common, providing lightweight specialization of large-scale models. Serving hundreds or thousands of these adapters on a single GPU allows request aggregation, increasing throughput, but may also cause request starvation if GPU memory limits are exceeded. To address this issue, this study focuses on determining the joint configuration of concurrent and parallel adapters that maximizes GPU throughput without inducing starvation, given heterogeneous adapter and traffic properties. We propose a data-driven ML approach leveraging interpretable models to tackle this caching problem and introduce the first Digital Twin capable of reproducing an LLM-adapter serving system, enabling efficient training data generation. Experiments with the vLLM framework and LoRA adapters show that the Digital Twin reproduces throughput within 5.1% of real results, while the ML approach predicts optimal numbers of concurrent and parallel adapters with an error of at most 7.2% under heterogeneous, real-world workloads. The code is publicly available at https://github.com/FerranAgulloLopez/GPULLMAdapterOptimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15424v1">LLM-MemCluster: Empowering Large Language Models with Dynamic Memory for Text Clustering</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are reshaping unsupervised learning by offering an unprecedented ability to perform text clustering based on their deep semantic understanding. However, their direct application is fundamentally limited by a lack of stateful memory for iterative refinement and the difficulty of managing cluster granularity. As a result, existing methods often rely on complex pipelines with external modules, sacrificing a truly end-to-end approach. We introduce LLM-MemCluster, a novel framework that reconceptualizes clustering as a fully LLM-native task. It leverages a Dynamic Memory to instill state awareness and a Dual-Prompt Strategy to enable the model to reason about and determine the number of clusters. Evaluated on several benchmark datasets, our tuning-free framework significantly and consistently outperforms strong baselines. LLM-MemCluster presents an effective, interpretable, and truly end-to-end paradigm for LLM-based text clustering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15392v1">DEPO: Dual-Efficiency Preference Optimization for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-19
      | 💬 Accepted to AAAI 2026
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have greatly improved their reasoning and decision-making abilities when deployed as agents. Richer reasoning, however, often comes at the cost of longer chain of thought (CoT), hampering interaction efficiency in real-world scenarios. Nevertheless, there still lacks systematic definition of LLM agent efficiency, hindering targeted improvements. To this end, we introduce dual-efficiency, comprising (i) step-level efficiency, which minimizes tokens per step, and (ii) trajectory-level efficiency, which minimizes the number of steps to complete a task. Building on this definition, we propose DEPO, a dual-efficiency preference optimization method that jointly rewards succinct responses and fewer action steps. Experiments on WebShop and BabyAI show that DEPO cuts token usage by up to 60.9% and steps by up to 26.9%, while achieving up to a 29.3% improvement in performance. DEPO also generalizes to three out-of-domain math benchmarks and retains its efficiency gains when trained on only 25% of the data. Our project page is at https://opencausalab.github.io/DEPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15389v1">Unveiling Inference Scaling for Difference-Aware User Modeling in LLM Personalization</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrated into users' daily lives, driving a growing demand for personalized outputs. Prior work has primarily leveraged a user's own history, often overlooking inter-user differences that are critical for effective personalization. While recent methods have attempted to model such differences, their feature extraction processes typically rely on fixed dimensions and quick, intuitive inference (System-1 thinking), limiting both the coverage and granularity of captured user differences. To address these limitations, we propose Difference-aware Reasoning Personalization (DRP), a framework that reconstructs the difference extraction mechanism by leveraging inference scaling to enhance LLM personalization. DRP autonomously identifies relevant difference feature dimensions and generates structured definitions and descriptions, enabling slow, deliberate reasoning (System-2 thinking) over user differences. Experiments on personalized review generation demonstrate that DRP consistently outperforms baseline methods across multiple metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15357v1">Cost-Aware Prediction (CAP): An LLM-Enhanced Machine Learning Pipeline and Decision Support System for Heart Failure Mortality Prediction</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      Objective: Machine learning (ML) predictive models are often developed without considering downstream value trade-offs and clinical interpretability. This paper introduces a cost-aware prediction (CAP) framework that combines cost-benefit analysis assisted by large language model (LLM) agents to communicate the trade-offs involved in applying ML predictions. Materials and Methods: We developed an ML model predicting 1-year mortality in patients with heart failure (N = 30,021, 22% mortality) to identify those eligible for home care. We then introduced clinical impact projection (CIP) curves to visualize important cost dimensions - quality of life and healthcare provider expenses, further divided into treatment and error costs, to assess the clinical consequences of predictions. Finally, we used four LLM agents to generate patient-specific descriptions. The system was evaluated by clinicians for its decision support value. Results: The eXtreme gradient boosting (XGB) model achieved the best performance, with an area under the receiver operating characteristic curve (AUROC) of 0.804 (95% confidence interval (CI) 0.792-0.816), area under the precision-recall curve (AUPRC) of 0.529 (95% CI 0.502-0.558) and a Brier score of 0.135 (95% CI 0.130-0.140). Discussion: The CIP cost curves provided a population-level overview of cost composition across decision thresholds, whereas LLM-generated cost-benefit analysis at individual patient-levels. The system was well received according to the evaluation by clinicians. However, feedback emphasizes the need to strengthen the technical accuracy for speculative tasks. Conclusion: CAP utilizes LLM agents to integrate ML classifier outcomes and cost-benefit analysis for more transparent and interpretable decision support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11729v2">Harli: SLO-Aware Co-location of LLM Inference and PEFT-based Finetuning on Model-as-a-Service Platforms</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed under the Model-as-a-Service (MaaS) paradigm. To meet stringent quality-of-service (QoS) requirements, existing LLM serving systems disaggregate the prefill and decode phases of inference. However, decode instances often experience low GPU utilization due to their memory-bound nature and insufficient batching in dynamic workloads, leaving compute resources underutilized. We introduce Harli, a serving system that improves GPU utilization by co-locating parameter-efficient finetuning (PEFT) tasks with LLM decode instances. PEFT tasks are compute-bound and memory-efficient, making them ideal candidates for safe co-location. Specifically, Harli addresses key challenges--limited memory and unpredictable interference--using three components: a unified memory allocator for runtime memory reuse, a two-stage latency predictor for decode latency modeling, and a QoS-guaranteed throughput-maximizing scheduler for throughput maximization. Experimental results show that Harli improves the finetune throughput by 46.2% on average (up to 92.0%) over state-of-the-art serving systems, while maintaining strict QoS guarantees for inference decode.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14334v2">When Words Change the Model: Sensitivity of LLMs for Constraint Programming Modelling</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      One of the long-standing goals in optimisation and constraint programming is to describe a problem in natural language and automatically obtain an executable, efficient model. Large language models appear to bring this vision closer, showing impressive results in automatically generating models for classical benchmarks. However, much of this apparent success may derive from data contamination rather than genuine reasoning: many standard CP problems are likely included in the training data of these models. To examine this hypothesis, we systematically rephrased and perturbed a set of well-known CSPLib problems to preserve their structure while modifying their context and introducing misleading elements. We then compared the models produced by three representative LLMs across original and modified descriptions. Our qualitative analysis shows that while LLMs can produce syntactically valid and semantically plausible models, their performance drops sharply under contextual and linguistic variation, revealing shallow understanding and sensitivity to wording.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15248v1">EntroPIC: Towards Stable Long-Term Training of LLMs via Entropy Stabilization with Proportional-Integral Control</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      Long-term training of large language models (LLMs) requires maintaining stable exploration to prevent the model from collapsing into sub-optimal behaviors. Entropy is crucial in this context, as it controls exploration and helps avoid premature convergence to sub-optimal solutions. However, existing reinforcement learning methods struggle to maintain an appropriate level of entropy, as the training process involves a mix of positive and negative samples, each affecting entropy in different ways across steps. To address this, we propose Entropy stablilization via Proportional-Integral Control (EntroPIC), a novel method that adaptively adjusts the influence of positive and negative samples by dynamically tuning their loss coefficients. This approach stabilizes entropy throughout training, ensuring efficient exploration and steady progress. We provide a comprehensive theoretical analysis for both on-policy and off-policy learning settings, demonstrating that EntroPIC is effective at controlling entropy in large-scale LLM training. Experimental results show that our method successfully maintains desired entropy levels, enabling stable and optimal RL training for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08916v2">HalluClean: A Unified Framework to Combat Hallucinations in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved impressive performance across a wide range of natural language processing tasks, yet they often produce hallucinated content that undermines factual reliability. To address this challenge, we introduce HalluClean, a lightweight and task-agnostic framework for detecting and correcting hallucinations in LLM-generated text. HalluClean adopts a reasoning-enhanced paradigm, explicitly decomposing the process into planning, execution, and revision stages to identify and refine unsupported claims. It employs minimal task-routing prompts to enable zero-shot generalization across diverse domains, without relying on external knowledge sources or supervised detectors. We conduct extensive evaluations on five representative tasks-question answering, dialogue, summarization, math word problems, and contradiction detection. Experimental results show that HalluClean significantly improves factual consistency and outperforms competitive baselines, demonstrating its potential to enhance the trustworthiness of LLM outputs in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08008v2">Combining LLM Semantic Reasoning with GNN Structural Modeling for Multi-View Multi-Label Feature Selection</a></div>
    <div class="paper-meta">
      📅 2025-11-19
      | 💬 9 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Multi-view multi-label feature selection aims to identify informative features from heterogeneous views, where each sample is associated with multiple interdependent labels. This problem is particularly important in machine learning involving high-dimensional, multimodal data such as social media, bioinformatics or recommendation systems. Existing Multi-View Multi-Label Feature Selection (MVMLFS) methods mainly focus on analyzing statistical information of data, but seldom consider semantic information. In this paper, we aim to use these two types of information jointly and propose a method that combines Large Language Models (LLMs) semantic reasoning with Graph Neural Networks (GNNs) structural modeling for MVMLFS. Specifically, the method consists of three main components. (1) LLM is first used as an evaluation agent to assess the latent semantic relevance among feature, view, and label descriptions. (2) A semantic-aware heterogeneous graph with two levels is designed to represent relations among features, views and labels: one is a semantic graph representing semantic relations, and the other is a statistical graph. (3) A lightweight Graph Attention Network (GAT) is applied to learn node embedding in the heterogeneous graph as feature saliency scores for ranking and selection. Experimental results on multiple benchmark datasets demonstrate the superiority of our method over state-of-the-art baselines, and it is still effective when applied to small-scale datasets, showcasing its robustness, flexibility, and generalization ability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.07252v3">Tasa: Thermal-aware 3D-Stacked Architecture Design with Bandwidth Sharing for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      The autoregressive decoding in LLMs is the major inference bottleneck due to the memory-intensive operations and limited hardware bandwidth. 3D-stacked architecture is a promising solution with significantly improved memory bandwidth, which vertically stacked multi DRAM dies on top of logic die. However, our experiments also show the 3D-stacked architecture faces severer thermal issues compared to 2D architecture, in terms of thermal temperature, gradient and scalability. To better exploit the potential of 3D-stacked architecture, we present Tasa, a heterogeneous architecture with cross-stack thermal optimizations to balance the temperature distribution and maximize the performance under the thermal constraints. High-performance core is designed for compute-intensive operations, while high-efficiency core is used for memory-intensive operators, e.g. attention layers. Furthermore, we propose a bandwidth sharing scheduling to improve the bandwidth utilization in such heterogeneous architecture. Extensive thermal experiments show that our Tasa architecture demonstrates greater scalability compared with the homogeneous 3D-stacked architecture, i.e. up to 5.55 $\tccentigrade$, 9.37 $\tccentigrade$, and 7.91 $\tccentigrade$ peak temperature reduction for 48, 60, and 72 core configurations. Our experimental for Llama-65B and GPT-3 66B inferences also demonstrate 2.85x and 2.21x speedup are obtained over the GPU baselines and state-of-the-art heterogeneous PIM-based LLM accelerator
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15203v1">Taxonomy, Evaluation and Exploitation of IPI-Centric LLM Agent Defense Frameworks</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based agents with function-calling capabilities are increasingly deployed, but remain vulnerable to Indirect Prompt Injection (IPI) attacks that hijack their tool calls. In response, numerous IPI-centric defense frameworks have emerged. However, these defenses are fragmented, lacking a unified taxonomy and comprehensive evaluation. In this Systematization of Knowledge (SoK), we present the first comprehensive analysis of IPI-centric defense frameworks. We introduce a comprehensive taxonomy of these defenses, classifying them along five dimensions. We then thoroughly assess the security and usability of representative defense frameworks. Through analysis of defensive failures in the assessment, we identify six root causes of defense circumvention. Based on these findings, we design three novel adaptive attacks that significantly improve attack success rates targeting specific frameworks, demonstrating the severity of the flaws in these defenses. Our paper provides a foundation and critical insights for the future development of more secure and usable IPI-centric agent defense frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15202v1">SOLID: a Framework of Synergizing Optimization and LLMs for Intelligent Decision-Making</a></div>
    <div class="paper-meta">
      📅 2025-11-19
      | 💬 NeurIPS 2025 WORKSHOP ML*OR Workshop: Mathematical Foundations and Operational Integration of Machine Learning for Uncertainty-Aware Decision-Making
    </div>
    <details class="paper-abstract">
      This paper introduces SOLID (Synergizing Optimization and Large Language Models for Intelligent Decision-Making), a novel framework that integrates mathematical optimization with the contextual capabilities of large language models (LLMs). SOLID facilitates iterative collaboration between optimization and LLMs agents through dual prices and deviation penalties. This interaction improves the quality of the decisions while maintaining modularity and data privacy. The framework retains theoretical convergence guarantees under convexity assumptions, providing insight into the design of LLMs prompt. To evaluate SOLID, we applied it to a stock portfolio investment case with historical prices and financial news as inputs. Empirical results demonstrate convergence under various scenarios and indicate improved annualized returns compared to a baseline optimizer-only method, validating the synergy of the two agents. SOLID offers a promising framework for advancing automated and intelligent decision-making across diverse domains.
    </details>
</div>
