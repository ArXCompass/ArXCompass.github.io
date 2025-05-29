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
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- Part 11
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09901v1">Comparing Exploration-Exploitation Strategies of LLMs and Humans: Insights from Standard Multi-armed Bandit Tasks</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to simulate or automate human behavior in complex sequential decision-making tasks. A natural question is then whether LLMs exhibit similar decision-making behavior to humans, and can achieve comparable (or superior) performance. In this work, we focus on the exploration-exploitation (E&E) tradeoff, a fundamental aspect of dynamic decision-making under uncertainty. We employ canonical multi-armed bandit (MAB) tasks introduced in the cognitive science and psychiatry literature to conduct a comparative study of the E&E strategies of LLMs, humans, and MAB algorithms. We use interpretable choice models to capture the E&E strategies of the agents and investigate how explicit reasoning, through both prompting strategies and reasoning-enhanced models, shapes LLM decision-making. We find that reasoning shifts LLMs toward more human-like behavior, characterized by a mix of random and directed exploration. In simple stationary tasks, reasoning-enabled LLMs exhibit similar levels of random and directed exploration compared to humans. However, in more complex, non-stationary environments, LLMs struggle to match human adaptability, particularly in effective directed exploration, despite achieving similar regret in certain scenarios. Our findings highlight both the promise and limits of LLMs as simulators of human behavior and tools for automated decision-making and point to potential areas of improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.06018v4">TARGET: Automated Scenario Generation from Traffic Rules for Testing Autonomous Vehicles via Validated LLM-Guided Knowledge Extraction</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Recent incidents with autonomous vehicles highlight the need for rigorous testing to ensure safety and robustness. Constructing test scenarios for autonomous driving systems (ADSs), however, is labor-intensive. We propose TARGET, an end-to-end framework that automatically generates test scenarios from traffic rules. To address complexity, we leverage a Large Language Model (LLM) to extract knowledge from traffic rules. To mitigate hallucinations caused by large context during input processing, we introduce a domain-specific language (DSL) designed to be syntactically simple and compositional. This design allows the LLM to learn and generate test scenarios in a modular manner while enabling syntactic and semantic validation for each component. Based on these validated representations, TARGET synthesizes executable scripts to render scenarios in simulation. Evaluated seven ADSs with 284 scenarios derived from 54 traffic rules, TARGET uncovered 610 rule violations, collisions, and other issues. For each violation, TARGET generates scenario recordings and detailed logs, aiding root cause analysis. Two identified issues were confirmed by ADS developers: one linked to an existing bug report and the other to limited ADS functionality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09986v2">From Equations to Insights: Unraveling Symbolic Structures in PDEs with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Motivated by the remarkable success of artificial intelligence (AI) across diverse fields, the application of AI to solve scientific problems, often formulated as partial differential equations (PDEs), has garnered increasing attention. While most existing research concentrates on theoretical properties (such as well-posedness, regularity, and continuity) of the solutions, alongside direct AI-driven methods for solving PDEs, the challenge of uncovering symbolic relationships within these equations remains largely unexplored. In this paper, we propose leveraging large language models (LLMs) to learn such symbolic relationships. Our results demonstrate that LLMs can effectively predict the operators involved in PDE solutions by utilizing the symbolic information in the PDEs both theoretically and numerically. Furthermore, we show that discovering these symbolic relationships can substantially improve both the efficiency and accuracy of symbolic machine learning for finding analytical approximation of PDE solutions, delivering a fully interpretable solution pipeline. This work opens new avenues for understanding the symbolic structure of scientific problems and advancing their solution processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14609v2">DiSCo: LLM Knowledge Distillation for Efficient Sparse Retrieval in Conversational Search</a></div>
    <div class="paper-meta">
      📅 2025-05-15
      | 💬 11 pages, 6 figures. SIGIR '25 Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval July 13--18, 2025 Padua, Italy
    </div>
    <details class="paper-abstract">
      Conversational Search (CS) involves retrieving relevant documents from a corpus while considering the conversational context, integrating retrieval with context modeling. Recent advancements in Large Language Models (LLMs) have significantly enhanced CS by enabling query rewriting based on conversational context. However, employing LLMs during inference poses efficiency challenges. Existing solutions mitigate this issue by distilling embeddings derived from human-rewritten queries, focusing primarily on learning the context modeling task. These methods, however, often separate the contrastive retrieval task from the distillation process, treating it as an independent loss term. To overcome these limitations, we introduce DiSCo (Distillation of Sparse Conversational retrieval), a novel approach that unifies retrieval and context modeling through a relaxed distillation objective. Instead of relying exclusively on representation learning, our method distills similarity scores between conversations and documents, providing more freedom in the representation space and better leveraging the contrastive nature of document relevance. Extensive experiments on Learned Sparse Retrieval (LSR) across five CS datasets demonstrate that DiSCo achieves substantial improvements in both in-domain and out-of-domain retrieval tasks, achieving up to a six-point gain in recall for out-of-domain datasets over state-of-the-art methods. Additionally, DiSCo employs a multi-teacher distillation strategy, using multiple LLMs as teachers, further enhancing performance and surpassing the individual teachers in in-domain settings. Furthermore, analysis of model sparsity reveals that DiSCo allows for more effective control over the sparsity of the trained models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10708v1">SafeTrans: LLM-assisted Transpilation from C to Rust</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Rust is a strong contender for a memory-safe alternative to C as a "systems" programming language, but porting the vast amount of existing C code to Rust is a daunting task. In this paper, we evaluate the potential of large language models (LLMs) to automate the transpilation of C code to idiomatic Rust, while ensuring that the generated code mitigates any memory-related vulnerabilities present in the original code. To that end, we present the design and implementation of SafeTrans, a framework that uses LLMs to i) transpile C code into Rust and ii) iteratively fix any compilation and runtime errors in the resulting code. A key novelty of our approach is the introduction of a few-shot guided repair technique for translation errors, which provides contextual information and example code snippets for specific error types, guiding the LLM toward the correct solution. Another novel aspect of our work is the evaluation of the security implications of the transpilation process, i.e., whether potential vulnerabilities in the original C code have been properly addressed in the translated Rust code. We experimentally evaluated SafeTrans with six leading LLMs and a set of 2,653 C programs accompanied by comprehensive unit tests, which were used for validating the correctness of the translated code. Our results show that our iterative repair strategy improves the rate of successful translations from 54% to 80% for the best-performing LLM (GPT-4o), and that all types of identified vulnerabilities in the original C code are effectively mitigated in the translated Rust code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09598v2">How Hungry is AI? Benchmarking Energy, Water, and Carbon Footprint of LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      This paper introduces a novel infrastructure-aware benchmarking framework for quantifying the environmental footprint of LLM inference across 30 state-of-the-art models as deployed in commercial data centers. Our framework combines public API performance data with region-specific environmental multipliers and statistical inference of hardware configurations. We additionally utilize cross-efficiency Data Envelopment Analysis (DEA) to rank models by performance relative to environmental cost. Our results show that o3 and DeepSeek-R1 emerge as the most energy-intensive models, consuming over 33 Wh per long prompt, more than 70 times the consumption of GPT-4.1 nano, and that Claude-3.7 Sonnet ranks highest in eco-efficiency. While a single short GPT-4o query consumes 0.43 Wh, scaling this to 700 million queries/day results in substantial annual environmental impacts. These include electricity use comparable to 35,000 U.S. homes, freshwater evaporation matching the annual drinking needs of 1.2 million people, and carbon emissions requiring a Chicago-sized forest to offset. These findings illustrate a growing paradox: Although AI is becoming cheaper and faster, its global adoption drives disproportionate resource consumption. Our study provides a standardized, empirically grounded methodology for benchmarking the sustainability of LLM deployments, laying a foundation for future environmental accountability in AI development and sustainability standards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10681v1">Towards an LLM-powered Social Digital Twinning Platform</a></div>
    <div class="paper-meta">
      📅 2025-05-15
      | 💬 13 pages, 3 figures, 23rd International Conference on Practical applications of Agents and Multi-Agent Systems (PAAMS 2025)
    </div>
    <details class="paper-abstract">
      We present Social Digital Twinner, an innovative social simulation tool for exploring plausible effects of what-if scenarios in complex adaptive social systems. The architecture is composed of three seamlessly integrated parts: a data infrastructure featuring real-world data and a multi-dimensionally representative synthetic population of citizens, an LLM-enabled agent-based simulation engine, and a user interface that enable intuitive, natural language interactions with the simulation engine and the artificial agents (i.e. citizens). Social Digital Twinner facilitates real-time engagement and empowers stakeholders to collaboratively design, test, and refine intervention measures. The approach is promoting a data-driven and evidence-based approach to societal problem-solving. We demonstrate the tool's interactive capabilities by addressing the critical issue of youth school dropouts in Kragero, Norway, showcasing its ability to create and execute a dedicated social digital twin using natural language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10670v1">Interpretable Risk Mitigation in LLM Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Autonomous agents powered by large language models (LLMs) enable novel use cases in domains where responsible action is increasingly important. Yet the inherent unpredictability of LLMs raises safety concerns about agent reliability. In this work, we explore agent behaviour in a toy, game-theoretic environment based on a variation of the Iterated Prisoner's Dilemma. We introduce a strategy-modification method-independent of both the game and the prompt-by steering the residual stream with interpretable features extracted from a sparse autoencoder latent space. Steering with the good-faith negotiation feature lowers the average defection probability by 28 percentage points. We also identify feasible steering ranges for several open-source LLM agents. Finally, we hypothesise that game-theoretic evaluation of LLM agents, combined with representation-steering alignment, can generalise to real-world applications on end-user devices and embodied platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10597v1">Two Minds Better Than One: Collaborative Reward Modeling for LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Reward models (RMs) are essential for aligning large language models (LLMs) with human values. However, noisy preferences in human feedback often lead to reward misgeneralization, where RMs overfit to spurious patterns and provide misleading signals during policy optimization. We systematically analyze the training dynamics of preference pairs and identify that noisy examples are harder to fit and introduce instability. Empirical evidence shows that LLMs optimized using reward models trained on full noisy datasets perform worse than those trained on filtered, high-quality preferences. To address this, we propose Collaborative Reward Modeling (CRM), an online framework that enhances robustness by combining peer review and curriculum learning. Two reward models are trained in parallel and assess each other's data selections to filter out potential noise. Curriculum learning structures the preference data from easy to hard, ensuring synchronized training and stable feedback. Extensive experiments demonstrate that CRM improves generalization, with up to 9.94 points of accuracy gain on RewardBench under 40 percent label noise. CRM is also compatible with implicit-reward alignment methods, offering a practical and versatile strategy for robust alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10593v1">LLM-Explorer: Towards Efficient and Affordable LLM-based Exploration for Mobile Apps</a></div>
    <div class="paper-meta">
      📅 2025-05-15
      | 💬 Accepted by MobiCom 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have opened new opportunities for automated mobile app exploration, an important and challenging problem that used to suffer from the difficulty of generating meaningful UI interactions. However, existing LLM-based exploration approaches rely heavily on LLMs to generate actions in almost every step, leading to a huge cost of token fees and computational resources. We argue that such extensive usage of LLMs is neither necessary nor effective, since many actions during exploration do not require, or may even be biased by the abilities of LLMs. Further, based on the insight that a precise and compact knowledge plays the central role for effective exploration, we introduce LLM-Explorer, a new exploration agent designed for efficiency and affordability. LLM-Explorer uses LLMs primarily for maintaining the knowledge instead of generating actions, and knowledge is used to guide action generation in a LLM-less manner. Based on a comparison with 5 strong baselines on 20 typical apps, LLM-Explorer was able to achieve the fastest and highest coverage among all automated app explorers, with over 148x lower cost than the state-of-the-art LLM-based approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10495v1">RouteNator: A Router-Based Multi-Modal Architecture for Generating Synthetic Training Data for Function Calling LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-15
      | 💬 Proceedings of the 4th International Workshop on Knowledge-Augmented Methods for Natural Language Processing
    </div>
    <details class="paper-abstract">
      This paper addresses fine-tuning Large Language Models (LLMs) for function calling tasks when real user interaction data is unavailable. In digital content creation tools, where users express their needs through natural language queries that must be mapped to API calls, the lack of real-world task-specific data and privacy constraints for training on it necessitate synthetic data generation. Existing approaches to synthetic data generation fall short in diversity and complexity, failing to replicate real-world data distributions and leading to suboptimal performance after LLM fine-tuning. We present a novel router-based architecture that leverages domain resources like content metadata and structured knowledge graphs, along with text-to-text and vision-to-text language models to generate high-quality synthetic training data. Our architecture's flexible routing mechanism enables synthetic data generation that matches observed real-world distributions, addressing a fundamental limitation of traditional approaches. Evaluation on a comprehensive set of real user queries demonstrates significant improvements in both function classification accuracy and API parameter selection. Models fine-tuned with our synthetic data consistently outperform traditional approaches, establishing new benchmarks for function calling tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10490v1">Campus AI vs Commercial AI: A Late-Breaking Study on How LLM As-A-Service Customizations Shape Trust and Usage Patterns</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      As the use of Large Language Models (LLMs) by students, lecturers and researchers becomes more prevalent, universities - like other organizations - are pressed to develop coherent AI strategies. LLMs as-a-Service (LLMaaS) offer accessible pre-trained models, customizable to specific (business) needs. While most studies prioritize data, model, or infrastructure adaptations (e.g., model fine-tuning), we focus on user-salient customizations, like interface changes and corporate branding, which we argue influence users' trust and usage patterns. This study serves as a functional prequel to a large-scale field study in which we examine how students and employees at a German university perceive and use their institution's customized LLMaaS compared to ChatGPT. The goals of this prequel are to stimulate discussions on psychological effects of LLMaaS customizations and refine our research approach through feedback. Our forthcoming findings will deepen the understanding of trust dynamics in LLMs, providing practical guidance for organizations considering LLMaaS deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18036v4">Harnessing Multiple Large Language Models: A Survey on LLM Ensemble</a></div>
    <div class="paper-meta">
      📅 2025-05-15
      | 💬 9 pages, 2 figures, codebase: https://github.com/junchenzhi/Awesome-LLM-Ensemble
    </div>
    <details class="paper-abstract">
      LLM Ensemble -- which involves the comprehensive use of multiple large language models (LLMs), each aimed at handling user queries during downstream inference, to benefit from their individual strengths -- has gained substantial attention recently. The widespread availability of LLMs, coupled with their varying strengths and out-of-the-box usability, has profoundly advanced the field of LLM Ensemble. This paper presents the first systematic review of recent developments in LLM Ensemble. First, we introduce our taxonomy of LLM Ensemble and discuss several related research problems. Then, we provide a more in-depth classification of the methods under the broad categories of "ensemble-before-inference, ensemble-during-inference, ensemble-after-inference'', and review all relevant methods. Finally, we introduce related benchmarks and applications, summarize existing studies, and suggest several future research directions. A curated list of papers on LLM Ensemble is available at https://github.com/junchenzhi/Awesome-LLM-Ensemble.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10425v1">Learning to Think: Information-Theoretic Reinforcement Fine-Tuning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at complex tasks thanks to advances in reasoning abilities. However, existing methods overlook the trade-off between reasoning effectiveness and computational efficiency, often encouraging unnecessarily long reasoning chains and wasting tokens. To address this, we propose Learning to Think (L2T), an information-theoretic reinforcement fine-tuning framework for LLMs to make the models achieve optimal reasoning with fewer tokens. Specifically, L2T treats each query-response interaction as a hierarchical session of multiple episodes and proposes a universal dense process reward, i.e., quantifies the episode-wise information gain in parameters, requiring no extra annotations or task-specific evaluators. We propose a method to quickly estimate this reward based on PAC-Bayes bounds and the Fisher information matrix. Theoretical analyses show that it significantly reduces computational complexity with high estimation accuracy. By immediately rewarding each episode's contribution and penalizing excessive updates, L2T optimizes the model via reinforcement learning to maximize the use of each episode and achieve effective updates. Empirical results on various reasoning benchmarks and base models demonstrate the advantage of L2T across different tasks, boosting both reasoning effectiveness and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10409v1">Are LLM-generated plain language summaries truly understandable? A large-scale crowdsourced evaluation</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Plain language summaries (PLSs) are essential for facilitating effective communication between clinicians and patients by making complex medical information easier for laypeople to understand and act upon. Large language models (LLMs) have recently shown promise in automating PLS generation, but their effectiveness in supporting health information comprehension remains unclear. Prior evaluations have generally relied on automated scores that do not measure understandability directly, or subjective Likert-scale ratings from convenience samples with limited generalizability. To address these gaps, we conducted a large-scale crowdsourced evaluation of LLM-generated PLSs using Amazon Mechanical Turk with 150 participants. We assessed PLS quality through subjective Likert-scale ratings focusing on simplicity, informativeness, coherence, and faithfulness; and objective multiple-choice comprehension and recall measures of reader understanding. Additionally, we examined the alignment between 10 automated evaluation metrics and human judgments. Our findings indicate that while LLMs can generate PLSs that appear indistinguishable from human-written ones in subjective evaluations, human-written PLSs lead to significantly better comprehension. Furthermore, automated evaluation metrics fail to reflect human judgment, calling into question their suitability for evaluating PLSs. This is the first study to systematically evaluate LLM-generated PLSs based on both reader preferences and comprehension outcomes. Our findings highlight the need for evaluation frameworks that move beyond surface-level quality and for generation methods that explicitly optimize for layperson comprehension.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10402v1">Rethinking Repetition Problems of LLMs in Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-05-15
      | 💬 Accepted to ACL 2025 (main)
    </div>
    <details class="paper-abstract">
      With the advent of neural language models, the performance of code generation has been significantly boosted. However, the problem of repetitions during the generation process continues to linger. Previous work has primarily focused on content repetition, which is merely a fraction of the broader repetition problem in code generation. A more prevalent and challenging problem is structural repetition. In structural repetition, the repeated code appears in various patterns but possesses a fixed structure, which can be inherently reflected in grammar. In this paper, we formally define structural repetition and propose an efficient decoding approach called RPG, which stands for Repetition Penalization based on Grammar, to alleviate the repetition problems in code generation for LLMs. Specifically, RPG first leverages grammar rules to identify repetition problems during code generation, and then strategically decays the likelihood of critical tokens that contribute to repetitions, thereby mitigating them in code generation. To facilitate this study, we construct a new dataset CodeRepetEval to comprehensively evaluate approaches for mitigating the repetition problems in code generation. Extensive experimental results demonstrate that RPG substantially outperforms the best-performing baselines on CodeRepetEval dataset as well as HumanEval and MBPP benchmarks, effectively reducing repetitions and enhancing the quality of generated code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06046v2">Healthy LLMs? Benchmarking LLM Knowledge of UK Government Public Health Information</a></div>
    <div class="paper-meta">
      📅 2025-05-15
      | 💬 24 pages, 10 pages main text
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become widely accessible, a detailed understanding of their knowledge within specific domains becomes necessary for successful real world use. This is particularly critical in public health, where failure to retrieve relevant, accurate, and current information could significantly impact UK residents. However, currently little is known about LLM knowledge of UK Government public health information. To address this issue, this paper introduces a new benchmark, PubHealthBench, with over 8000 questions for evaluating LLMs' Multiple Choice Question Answering (MCQA) and free form responses to public health queries. To create PubHealthBench we extract free text from 687 current UK government guidance documents and implement an automated pipeline for generating MCQA samples. Assessing 24 LLMs on PubHealthBench we find the latest private LLMs (GPT-4.5, GPT-4.1 and o1) have a high degree of knowledge, achieving >90% accuracy in the MCQA setup, and outperform humans with cursory search engine use. However, in the free form setup we see lower performance with no model scoring >75%. Importantly we find in both setups LLMs have higher accuracy on guidance intended for the general public. Therefore, there are promising signs that state of the art (SOTA) LLMs are an increasingly accurate source of public health information, but additional safeguards or tools may still be needed when providing free form responses on public health topics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10321v1">AutoPentest: Enhancing Vulnerability Management With Autonomous LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-15
      | 💬 24 pages, 1 figure, for implementation, see https://github.com/JuliusHenke/autopentest
    </div>
    <details class="paper-abstract">
      A recent area of increasing research is the use of Large Language Models (LLMs) in penetration testing, which promises to reduce costs and thus allow for higher frequency. We conduct a review of related work, identifying best practices and common evaluation issues. We then present AutoPentest, an application for performing black-box penetration tests with a high degree of autonomy. AutoPentest is based on the LLM GPT-4o from OpenAI and the LLM agent framework LangChain. It can perform complex multi-step tasks, augmented by external tools and knowledge bases. We conduct a study on three capture-the-flag style Hack The Box (HTB) machines, comparing our implementation AutoPentest with the baseline approach of manually using the ChatGPT-4o user interface. Both approaches are able to complete 15-25 % of the subtasks on the HTB machines, with AutoPentest slightly outperforming ChatGPT. We measure a total cost of \$96.20 US when using AutoPentest across all experiments, while a one-month subscription to ChatGPT Plus costs \$20. The results show that further implementation efforts and the use of more powerful LLMs released in the future are likely to make this a viable part of vulnerability management.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10320v1">J1: Incentivizing Thinking in LLM-as-a-Judge via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-15
      | 💬 10 pages, 8 tables, 11 figures
    </div>
    <details class="paper-abstract">
      The progress of AI is bottlenecked by the quality of evaluation, and powerful LLM-as-a-Judge models have proved to be a core solution. Improved judgment ability is enabled by stronger chain-of-thought reasoning, motivating the need to find the best recipes for training such models to think. In this work we introduce J1, a reinforcement learning approach to training such models. Our method converts both verifiable and non-verifiable prompts to judgment tasks with verifiable rewards that incentivize thinking and mitigate judgment bias. In particular, our approach outperforms all other existing 8B or 70B models when trained at those sizes, including models distilled from DeepSeek-R1. J1 also outperforms o1-mini, and even R1 on some benchmarks, despite training a smaller model. We provide analysis and ablations comparing Pairwise-J1 vs Pointwise-J1 models, offline vs online training recipes, reward strategies, seed prompts, and variations in thought length and content. We find that our models make better judgments by learning to outline evaluation criteria, comparing against self-generated reference answers, and re-evaluating the correctness of model responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10260v1">Comparing LLM Text Annotation Skills: A Study on Human Rights Violations in Social Media Data</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      In the era of increasingly sophisticated natural language processing (NLP) systems, large language models (LLMs) have demonstrated remarkable potential for diverse applications, including tasks requiring nuanced textual understanding and contextual reasoning. This study investigates the capabilities of multiple state-of-the-art LLMs - GPT-3.5, GPT-4, LLAMA3, Mistral 7B, and Claude-2 - for zero-shot and few-shot annotation of a complex textual dataset comprising social media posts in Russian and Ukrainian. Specifically, the focus is on the binary classification task of identifying references to human rights violations within the dataset. To evaluate the effectiveness of these models, their annotations are compared against a gold standard set of human double-annotated labels across 1000 samples. The analysis includes assessing annotation performance under different prompting conditions, with prompts provided in both English and Russian. Additionally, the study explores the unique patterns of errors and disagreements exhibited by each model, offering insights into their strengths, limitations, and cross-linguistic adaptability. By juxtaposing LLM outputs with human annotations, this research contributes to understanding the reliability and applicability of LLMs for sensitive, domain-specific tasks in multilingual contexts. It also sheds light on how language models handle inherently subjective and context-dependent judgments, a critical consideration for their deployment in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10259v1">SpecOffload: Unlocking Latent GPU Capacity for LLM Inference on Resource-Constrained Devices</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Efficient LLM inference on resource-constrained devices presents significant challenges in compute and memory utilization. Due to limited GPU memory, existing systems offload model weights to CPU memory, incurring substantial I/O overhead between the CPU and GPU. This leads to two major inefficiencies: (1) GPU cores are underutilized, often remaining idle while waiting for data to be loaded; and (2) GPU memory has low impact on performance, as reducing its capacity has minimal effect on overall throughput.In this paper, we propose SpecOffload, a high-throughput inference engine that embeds speculative decoding into offloading. Our key idea is to unlock latent GPU resources for storing and executing a draft model used for speculative decoding, thus accelerating inference at near-zero additional cost. To support this, we carefully orchestrate the interleaved execution of target and draft models in speculative decoding within the offloading pipeline, and propose a planner to manage tensor placement and select optimal parameters. Compared to the best baseline, SpecOffload improves GPU core utilization by 4.49x and boosts inference throughput by 2.54x. Our code is available at https://github.com/MobiSense/SpecOffload .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.15674v2">TensorLLM: Tensorising Multi-Head Attention for Enhanced Reasoning and Compression in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-15
      | 💬 Accpeted for IEEE International Joint Conference on Neural Networks (IJCNN 2025). The code is available at https://github.com/guyuxuan9/TensorLLM
    </div>
    <details class="paper-abstract">
      The reasoning abilities of Large Language Models (LLMs) can be improved by structurally denoising their weights, yet existing techniques primarily focus on denoising the feed-forward network (FFN) of the transformer block, and can not efficiently utilise the Multi-head Attention (MHA) block, which is the core of transformer architectures. To address this issue, we propose a novel intuitive framework that, at its very core, performs MHA compression through a multi-head tensorisation process and the Tucker decomposition. This enables both higher-dimensional structured denoising and compression of the MHA weights, by enforcing a shared higher-dimensional subspace across the weights of the multiple attention heads. We demonstrate that this approach consistently enhances the reasoning capabilities of LLMs across multiple benchmark datasets, and for both encoder-only and decoder-only architectures, while achieving compression rates of up to $\sim 250$ times in the MHA weights, all without requiring any additional data, training, or fine-tuning. Furthermore, we show that the proposed method can be seamlessly combined with existing FFN-only-based denoising techniques to achieve further improvements in LLM reasoning performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10218v1">RAIDEN-R1: Improving Role-awareness of LLMs via GRPO with Verifiable Reward</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Role-playing conversational agents (RPCAs) face persistent challenges in maintaining role consistency. To address this, we propose RAIDEN-R1, a novel reinforcement learning framework that integrates Verifiable Role-Awareness Reward (VRAR). The method introduces both singular and multi-term mining strategies to generate quantifiable rewards by assessing role-specific keys. Additionally, we construct a high-quality, role-aware Chain-of-Thought dataset through multi-LLM collaboration, and implement experiments to enhance reasoning coherence. Experiments on the RAIDEN benchmark demonstrate RAIDEN-R1's superiority: our 14B-GRPO model achieves 88.04% and 88.65% accuracy on Script-Based Knowledge and Conversation Memory metrics, respectively, outperforming baseline models while maintaining robustness. Case analyses further reveal the model's enhanced ability to resolve conflicting contextual cues and sustain first-person narrative consistency. This work bridges the non-quantifiability gap in RPCA training and provides insights into role-aware reasoning patterns, advancing the development of RPCAs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10213v1">Informed Forecasting: Leveraging Auxiliary Knowledge to Boost LLM Performance on Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      With the widespread adoption of Large Language Models (LLMs), there is a growing need to establish best practices for leveraging their capabilities beyond traditional natural language tasks. In this paper, a novel cross-domain knowledge transfer framework is proposed to enhance the performance of LLMs in time series forecasting -- a task of increasing relevance in fields such as energy systems, finance, and healthcare. The approach systematically infuses LLMs with structured temporal information to improve their forecasting accuracy. This study evaluates the proposed method on a real-world time series dataset and compares it to a naive baseline where the LLM receives no auxiliary information. Results show that knowledge-informed forecasting significantly outperforms the uninformed baseline in terms of predictive accuracy and generalization. These findings highlight the potential of knowledge transfer strategies to bridge the gap between LLMs and domain-specific forecasting tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10212v1">Do LLMs Memorize Recommendation Datasets? A Preliminary Study on MovieLens-1M</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become increasingly central to recommendation scenarios due to their remarkable natural language understanding and generation capabilities. Although significant research has explored the use of LLMs for various recommendation tasks, little effort has been dedicated to verifying whether they have memorized public recommendation dataset as part of their training data. This is undesirable because memorization reduces the generalizability of research findings, as benchmarking on memorized datasets does not guarantee generalization to unseen datasets. Furthermore, memorization can amplify biases, for example, some popular items may be recommended more frequently than others. In this work, we investigate whether LLMs have memorized public recommendation datasets. Specifically, we examine two model families (GPT and Llama) across multiple sizes, focusing on one of the most widely used dataset in recommender systems: MovieLens-1M. First, we define dataset memorization as the extent to which item attributes, user profiles, and user-item interactions can be retrieved by prompting the LLMs. Second, we analyze the impact of memorization on recommendation performance. Lastly, we examine whether memorization varies across model families and model sizes. Our results reveal that all models exhibit some degree of memorization of MovieLens-1M, and that recommendation performance is related to the extent of memorization. We have made all the code publicly available at: https://github.com/sisinflab/LLM-MemoryInspector
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10325v2">Collaborative Speculative Inference for Efficient LLM Inference Serving</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Speculative inference is a promising paradigm employing small speculative models (SSMs) as drafters to generate draft tokens, which are subsequently verified in parallel by the target large language model (LLM). This approach enhances the efficiency of inference serving by reducing LLM inference latency and costs while preserving generation quality. However, existing speculative methods face critical challenges, including inefficient resource utilization and limited draft acceptance, which constrain their scalability and overall effectiveness. To overcome these obstacles, we present CoSine, a novel speculative inference system that decouples sequential speculative decoding from parallel verification, enabling efficient collaboration among multiple nodes. Specifically, CoSine routes inference requests to specialized drafters based on their expertise and incorporates a confidence-based token fusion mechanism to synthesize outputs from cooperating drafters, ensuring high-quality draft generation. Additionally, CoSine dynamically orchestrates the execution of speculative decoding and verification in a pipelined manner, employing batch scheduling to selectively group requests and adaptive speculation control to minimize idle periods. By optimizing parallel workflows through heterogeneous node collaboration, CoSine balances draft generation and verification throughput in real-time, thereby maximizing resource utilization. Experimental results demonstrate that CoSine achieves superior performance compared to state-of-the-art speculative approaches. Notably, with equivalent resource costs, CoSine achieves up to a 23.2% decrease in latency and a 32.5% increase in throughput compared to baseline methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10182v1">Mining Hidden Thoughts from Texts: Evaluating Continual Pretraining with Synthetic Data for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated significant improvements in reasoning capabilities through supervised fine-tuning and reinforcement learning. However, when training reasoning models, these approaches are primarily applicable to specific domains such as mathematics and programming, which imposes fundamental constraints on the breadth and scalability of training data. In contrast, continual pretraining (CPT) offers the advantage of not requiring task-specific signals. Nevertheless, how to effectively synthesize training data for reasoning and how such data affect a wide range of domains remain largely unexplored. This study provides a detailed evaluation of Reasoning CPT, a form of CPT that uses synthetic data to reconstruct the hidden thought processes underlying texts, based on the premise that texts are the result of the author's thinking process. Specifically, we apply Reasoning CPT to Gemma2-9B using synthetic data with hidden thoughts derived from STEM and Law corpora, and compare it to standard CPT on the MMLU benchmark. Our analysis reveals that Reasoning CPT consistently improves performance across all evaluated domains. Notably, reasoning skills acquired in one domain transfer effectively to others; the performance gap with conventional methods widens as problem difficulty increases, with gains of up to 8 points on the most challenging problems. Furthermore, models trained with hidden thoughts learn to adjust the depth of their reasoning according to problem difficulty.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.01797v3">LLM A*: Human in the Loop Large Language Models Enabled A* Search for Robotics</a></div>
    <div class="paper-meta">
      📅 2025-05-15
      | 💬 7 figures, 8 pages
    </div>
    <details class="paper-abstract">
      This research focuses on how Large Language Models (LLMs) can help with (path) planning for mobile embodied agents such as robots, in a human-in-the-loop and interactive manner. A novel framework named LLM A*, aims to leverage the commonsense of LLMs, and the utility-optimal A* is proposed to facilitate few-shot near-optimal path planning. Prompts are used for two main purposes: 1) to provide LLMs with essential information like environments, costs, heuristics, etc.; 2) to communicate human feedback on intermediate planning results to LLMs. This approach takes human feedback on board and renders the entire planning process transparent (akin to a `white box') to humans. Moreover, it facilitates code-free path planning, thereby fostering the accessibility and inclusiveness of artificial intelligence techniques to communities less proficient in coding. Comparative analysis against A* and RL demonstrates that LLM A* exhibits greater efficiency in terms of search space and achieves paths comparable to A* while outperforming RL. The interactive nature of LLM A* also makes it a promising tool for deployment in collaborative human-robot tasks. Codes and Supplemental Materials can be found at GitHub: https://github.com/speedhawk/LLM-A-.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09602v1">Adversarial Suffix Filtering: a Defense Pipeline for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly embedded in autonomous systems and public-facing environments, yet they remain susceptible to jailbreak vulnerabilities that may undermine their security and trustworthiness. Adversarial suffixes are considered to be the current state-of-the-art jailbreak, consistently outperforming simpler methods and frequently succeeding even in black-box settings. Existing defenses rely on access to the internal architecture of models limiting diverse deployment, increase memory and computation footprints dramatically, or can be bypassed with simple prompt engineering methods. We introduce $\textbf{Adversarial Suffix Filtering}$ (ASF), a lightweight novel model-agnostic defensive pipeline designed to protect LLMs against adversarial suffix attacks. ASF functions as an input preprocessor and sanitizer that detects and filters adversarially crafted suffixes in prompts, effectively neutralizing malicious injections. We demonstrate that ASF provides comprehensive defense capabilities across both black-box and white-box attack settings, reducing the attack efficacy of state-of-the-art adversarial suffix generation methods to below 4%, while only minimally affecting the target model's capabilities in non-adversarial scenarios.
    </details>
</div>
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
