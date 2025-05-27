# llm - 2025_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3
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
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17762v1">Resolving Conflicting Evidence in Automated Fact-Checking: A Study on Retrieval-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 Camera-ready for IJCAI 2025, AI and Social Good
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) augmented with retrieval mechanisms have demonstrated significant potential in fact-checking tasks by integrating external knowledge. However, their reliability decreases when confronted with conflicting evidence from sources of varying credibility. This paper presents the first systematic evaluation of Retrieval-Augmented Generation (RAG) models for fact-checking in the presence of conflicting evidence. To support this study, we introduce \textbf{CONFACT} (\textbf{Con}flicting Evidence for \textbf{Fact}-Checking) (Dataset available at https://github.com/zoeyyes/CONFACT), a novel dataset comprising questions paired with conflicting information from various sources. Extensive experiments reveal critical vulnerabilities in state-of-the-art RAG methods, particularly in resolving conflicts stemming from differences in media source credibility. To address these challenges, we investigate strategies to integrate media background information into both the retrieval and generation stages. Our results show that effectively incorporating source credibility significantly enhances the ability of RAG models to resolve conflicting evidence and improve fact-checking performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17760v1">But what is your honest answer? Aiding LLM-judges with honest alternatives using steering vectors</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      Recent safety evaluations of Large Language Models (LLMs) show that many models exhibit dishonest behavior, such as sycophancy. However, most honesty benchmarks focus exclusively on factual knowledge or explicitly harmful behavior and rely on external judges, which are often unable to detect less obvious forms of dishonesty. In this work, we introduce a new framework, Judge Using Safety-Steered Alternatives (JUSSA), which utilizes steering vectors trained on a single sample to elicit more honest responses from models, helping LLM-judges in the detection of dishonest behavior. To test our framework, we introduce a new manipulation dataset with prompts specifically designed to elicit deceptive responses. We find that JUSSA enables LLM judges to better differentiate between dishonest and benign responses, and helps them identify subtle instances of manipulative behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16646v2">SMART: Self-Generating and Self-Validating Multi-Dimensional Assessment for LLMs' Mathematical Problem Solving</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      Large Language Models have achieved remarkable results on a variety of mathematical benchmarks. However, concerns remain as to whether these successes reflect genuine mathematical reasoning or superficial pattern recognition. Common evaluation metrics, such as final answer accuracy, fail to disentangle the underlying competencies involved, offering limited diagnostic value. To address these limitations, we introduce SMART: a Self-Generating and Self-Validating Multi-Dimensional Assessment Framework. SMART decomposes mathematical problem solving into four distinct dimensions: understanding, reasoning, arithmetic, and reflection \& refinement. Each dimension is evaluated independently through tailored tasks, enabling interpretable and fine-grained analysis of LLM behavior. Crucially, SMART integrates an automated self-generating and self-validating mechanism to produce and verify benchmark data, ensuring both scalability and reliability. We apply SMART to 21 state-of-the-art open- and closed-source LLMs, uncovering significant discrepancies in their abilities across different dimensions. Our findings demonstrate the inadequacy of final answer accuracy as a sole metric and motivate a new holistic metric to better capture true problem-solving capabilities. Code and benchmarks will be released upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12896v4">None of the Others: a General Technique to Distinguish Reasoning from Memorization in Multiple-Choice LLM Evaluation Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      In LLM evaluations, reasoning is often distinguished from recall/memorization by performing numerical variations to math-oriented questions. Here we introduce a general variation method for multiple-choice questions that completely dissociates the correct answer from previously seen tokens or concepts, requiring LLMs to understand and reason (rather than memorizing) in order to answer correctly. Using this method, we evaluate state-of-the-art proprietary and open-source LLMs on two datasets available in English and Spanish: the public MMLU benchmark and the private UNED-Access 2024 dataset. Results show that all models experience remarkable accuracy drops under our proposed variation, with an average loss of 57% on MMLU and 50% on UNED-Access 2024, ranging from 10% to 93% across models. Notably, the most accurate model in our experimentation (OpenAI-o3-mini) is not the most robust (DeepSeek-R1-70B), suggesting that the best models in standard evaluations may not be the ones with better reasoning capabilities. Also, we see larger accuracy drops in public (vs private) datasets and questions posed in their original language (vs a manual translation), which are signs of contamination and also point to a relevant role of recall/memorization in current LLMs' answers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19838v2">LLM-Powered GUI Agents in Phone Automation: Surveying Progress and Prospects</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 39 pages, 10 figures, 7 tables, Project Homepage: https://github.com/PhoneLLM/Awesome-LLM-Powered-Phone-GUI-Agents
    </div>
    <details class="paper-abstract">
      With the rapid rise of large language models (LLMs), phone automation has undergone transformative changes. This paper systematically reviews LLM-driven phone GUI agents, highlighting their evolution from script-based automation to intelligent, adaptive systems. We first contextualize key challenges, (i) limited generality, (ii) high maintenance overhead, and (iii) weak intent comprehension, and show how LLMs address these issues through advanced language understanding, multimodal perception, and robust decision-making. We then propose a taxonomy covering fundamental agent frameworks (single-agent, multi-agent, plan-then-act), modeling approaches (prompt engineering, training-based), and essential datasets and benchmarks. Furthermore, we detail task-specific architectures, supervised fine-tuning, and reinforcement learning strategies that bridge user intent and GUI operations. Finally, we discuss open challenges such as dataset diversity, on-device deployment efficiency, user-centric adaptation, and security concerns, offering forward-looking insights into this rapidly evolving field. By providing a structured overview and identifying pressing research gaps, this paper serves as a definitive reference for researchers and practitioners seeking to harness LLMs in designing scalable, user-friendly phone GUI agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17735v1">Automating Safety Enhancement for LLM-based Agents with Synthetic Risk Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 38 pages;12 figures;12 tables
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based agents are increasingly deployed in real-world applications such as "digital assistants, autonomous customer service, and decision-support systems", where their ability to "interact in multi-turn, tool-augmented environments" makes them indispensable. However, ensuring the safety of these agents remains a significant challenge due to the diverse and complex risks arising from dynamic user interactions, external tool usage, and the potential for unintended harmful behaviors. To address this critical issue, we propose AutoSafe, the first framework that systematically enhances agent safety through fully automated synthetic data generation. Concretely, 1) we introduce an open and extensible threat model, OTS, which formalizes how unsafe behaviors emerge from the interplay of user instructions, interaction contexts, and agent actions. This enables precise modeling of safety risks across diverse scenarios. 2) we develop a fully automated data generation pipeline that simulates unsafe user behaviors, applies self-reflective reasoning to generate safe responses, and constructs a large-scale, diverse, and high-quality safety training dataset-eliminating the need for hazardous real-world data collection. To evaluate the effectiveness of our framework, we design comprehensive experiments on both synthetic and real-world safety benchmarks. Results demonstrate that AutoSafe boosts safety scores by 45% on average and achieves a 28.91% improvement on real-world tasks, validating the generalization ability of our learned safety strategies. These results highlight the practical advancement and scalability of AutoSafe in building safer LLM-based agents for real-world deployment. We have released the project page at https://auto-safe.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17726v1">Slot-MLLM: Object-Centric Visual Tokenization for Multimodal LLM</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      Recently, multimodal large language models (MLLMs) have emerged as a key approach in achieving artificial general intelligence. In particular, vision-language MLLMs have been developed to generate not only text but also visual outputs from multimodal inputs. This advancement requires efficient image tokens that LLMs can process effectively both in input and output. However, existing image tokenization methods for MLLMs typically capture only global abstract concepts or uniformly segmented image patches, restricting MLLMs' capability to effectively understand or generate detailed visual content, particularly at the object level. To address this limitation, we propose an object-centric visual tokenizer based on Slot Attention specifically for MLLMs. In particular, based on the Q-Former encoder, diffusion decoder, and residual vector quantization, our proposed discretized slot tokens can encode local visual details while maintaining high-level semantics, and also align with textual data to be integrated seamlessly within a unified next-token prediction framework of LLMs. The resulting Slot-MLLM demonstrates significant performance improvements over baselines with previous visual tokenizers across various vision-language tasks that entail local detailed comprehension and generation. Notably, this work is the first demonstration of the feasibility of object-centric slot attention performed with MLLMs and in-the-wild natural images.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17716v1">Get Experience from Practice: LLM Agents with Record & Replay</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      AI agents, empowered by Large Language Models (LLMs) and communication protocols such as MCP and A2A, have rapidly evolved from simple chatbots to autonomous entities capable of executing complex, multi-step tasks, demonstrating great potential. However, the LLMs' inherent uncertainty and heavy computational resource requirements pose four significant challenges to the development of safe and efficient agents: reliability, privacy, cost and performance. Existing approaches, like model alignment, workflow constraints and on-device model deployment, can partially alleviate some issues but often with limitations, failing to fundamentally resolve these challenges. This paper proposes a new paradigm called AgentRR (Agent Record & Replay), which introduces the classical record-and-replay mechanism into AI agent frameworks. The core idea is to: 1. Record an agent's interaction trace with its environment and internal decision process during task execution, 2. Summarize this trace into a structured "experience" encapsulating the workflow and constraints, and 3. Replay these experiences in subsequent similar tasks to guide the agent's behavior. We detail a multi-level experience abstraction method and a check function mechanism in AgentRR: the former balances experience specificity and generality, while the latter serves as a trust anchor to ensure completeness and safety during replay. In addition, we explore multiple application modes of AgentRR, including user-recorded task demonstration, large-small model collaboration and privacy-aware agent execution, and envision an experience repository for sharing and reusing knowledge to further reduce deployment cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17712v1">Understanding How Value Neurons Shape the Generation of Specified Values in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      Rapid integration of large language models (LLMs) into societal applications has intensified concerns about their alignment with universal ethical principles, as their internal value representations remain opaque despite behavioral alignment advancements. Current approaches struggle to systematically interpret how values are encoded in neural architectures, limited by datasets that prioritize superficial judgments over mechanistic analysis. We introduce ValueLocate, a mechanistic interpretability framework grounded in the Schwartz Values Survey, to address this gap. Our method first constructs ValueInsight, a dataset that operationalizes four dimensions of universal value through behavioral contexts in the real world. Leveraging this dataset, we develop a neuron identification method that calculates activation differences between opposing value aspects, enabling precise localization of value-critical neurons without relying on computationally intensive attribution methods. Our proposed validation method demonstrates that targeted manipulation of these neurons effectively alters model value orientations, establishing causal relationships between neurons and value representations. This work advances the foundation for value alignment by bridging psychological value frameworks with neuron analysis in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13202v2">The Quantum LLM: Modeling Semantic Spaces with Quantum Principles</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 16 pages, 6 figures. Some corrections
    </div>
    <details class="paper-abstract">
      In the previous article, we presented a quantum-inspired framework for modeling semantic representation and processing in Large Language Models (LLMs), drawing upon mathematical tools and conceptual analogies from quantum mechanics to offer a new perspective on these complex systems. In this paper, we clarify the core assumptions of this model, providing a detailed exposition of six key principles that govern semantic representation, interaction, and dynamics within LLMs. The goal is to justify that a quantum-inspired framework is a valid approach to studying semantic spaces. This framework offers valuable insights into their information processing and response generation, and we further discuss the potential of leveraging quantum computing to develop significantly more powerful and efficient LLMs based on these principles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17710v1">LLM Contribution Summarization in Software Projects</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      This full paper in innovative practice provides an automated tool to summarize individual code contributions in project-based courses with external clients. Real industry projects offer valuable learning opportunities by immersing students in authentic problems defined by external clients. However, the open-ended and highly variable scope of these projects makes it challenging for instructors and teaching assistants to provide timely and detailed feedback. This paper addresses the need for an automated and objective approach to evaluate individual contributions within team projects. In this paper, we present a tool that leverages a large language model (LLM) to automatically summarize code contributions extracted from version control repositories. The tool preprocesses and structures repository data, and uses PyDriller to isolate individual contributions. Its uniqueness lies in the combination of LLM prompt engineering with automated repository analysis, thus reducing the manual grading burden while providing regular and informative updates. The tool was assessed over two semesters during a three-week, full-time software development sprint involving 65 students. Weekly summaries were provided to teams, and both student and faculty feedback indicated the tool's overall usefulness in informing grading and guidance. The tool reports, in large proportion, activities that were in fact performed by the student, with some failure to detect students' contribution. The summaries were considered by the instructors as a useful potential tool to keep up with the projects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07199v3">SemEval-2025 Task 5: LLMs4Subjects -- LLM-based Automated Subject Tagging for a National Technical Library's Open-Access Catalog</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 10 pages, 4 figures, Accepted as SemEval 2025 Task 5 description paper
    </div>
    <details class="paper-abstract">
      We present SemEval-2025 Task 5: LLMs4Subjects, a shared task on automated subject tagging for scientific and technical records in English and German using the GND taxonomy. Participants developed LLM-based systems to recommend top-k subjects, evaluated through quantitative metrics (precision, recall, F1-score) and qualitative assessments by subject specialists. Results highlight the effectiveness of LLM ensembles, synthetic data generation, and multilingual processing, offering insights into applying LLMs for digital library classification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17691v1">ELSPR: Evaluator LLM Training Data Self-Purification on Non-Transitive Preferences via Tournament Graph Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are widely used as evaluators for open-ended tasks, while previous research has emphasized biases in LLM evaluations, the issue of non-transitivity in pairwise comparisons remains unresolved: non-transitive preferences for pairwise comparisons, where evaluators prefer A over B, B over C, but C over A. Our results suggest that low-quality training data may reduce the transitivity of preferences generated by the Evaluator LLM. To address this, We propose a graph-theoretic framework to analyze and mitigate this problem by modeling pairwise preferences as tournament graphs. We quantify non-transitivity and introduce directed graph structural entropy to measure the overall clarity of preferences. Our analysis reveals significant non-transitivity in advanced Evaluator LLMs (with Qwen2.5-Max exhibiting 67.96%), as well as high entropy values (0.8095 for Qwen2.5-Max), reflecting low overall clarity of preferences. To address this issue, we designed a filtering strategy, ELSPR, to eliminate preference data that induces non-transitivity, retaining only consistent and transitive preference data for model fine-tuning. Experiments demonstrate that models fine-tuned with filtered data reduce non-transitivity by 13.78% (from 64.28% to 50.50%), decrease structural entropy by 0.0879 (from 0.8113 to 0.7234), and align more closely with human evaluators (human agreement rate improves by 0.6% and Spearman correlation increases by 0.01).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17262v2">Unveiling Downstream Performance Scaling of LLMs: A Clustering-Based Perspective</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 19 pages,6 figures
    </div>
    <details class="paper-abstract">
      The escalating scale and cost of Large Language Models (LLMs) training necessitate accurate pre-training prediction of downstream task performance for efficient resource allocation. This is challenged by: 1) the emergence phenomenon, where metrics become meaningful only after extensive training, hindering prediction by smaller models; and 2) uneven task difficulty and inconsistent performance scaling patterns, leading to high metric variability. Current prediction methods lack accuracy and reliability. We propose a Clustering-On-Difficulty (COD) framework for downstream performance prediction. The COD framework clusters tasks by their difficulty scaling features, thereby establishing a more stable and predictable support subset through the exclusion of tasks exhibiting non-emergent behavior or irregular scaling. We adopt a performance scaling law to predict cluster-wise performance with theoretical support. Predictable subset performance acts as an intermediate predictor for the full evaluation set. We further derive a mapping function to accurately extrapolate the performance of the subset to the full set. Applied to an LLM with 70B parameters, COD achieved a 1.36% average prediction error across eight key LLM benchmarks, offering actionable insights for resource allocation and training monitoring of LLMs pretraining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17663v1">Towards Dynamic Theory of Mind: Evaluating LLM Adaptation to Temporal Evolution of Human States</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 Accepted by ACL 2025 Main Conference
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) increasingly participate in human-AI interactions, evaluating their Theory of Mind (ToM) capabilities - particularly their ability to track dynamic mental states - becomes crucial. While existing benchmarks assess basic ToM abilities, they predominantly focus on static snapshots of mental states, overlooking the temporal evolution that characterizes real-world social interactions. We present \textsc{DynToM}, a novel benchmark specifically designed to evaluate LLMs' ability to understand and track the temporal progression of mental states across interconnected scenarios. Through a systematic four-step framework, we generate 1,100 social contexts encompassing 5,500 scenarios and 78,100 questions, each validated for realism and quality. Our comprehensive evaluation of ten state-of-the-art LLMs reveals that their average performance underperforms humans by 44.7\%, with performance degrading significantly when tracking and reasoning about the shift of mental states. This performance gap highlights fundamental limitations in current LLMs' ability to model the dynamic nature of human mental states.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17656v1">Too Consistent to Detect: A Study of Self-Consistent Errors in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 Underreview in EMNLP25
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) often generate plausible but incorrect content, error detection has become increasingly critical to ensure truthfulness. However, existing detection methods often overlook a critical problem we term as self-consistent error, where LLMs repeatly generate the same incorrect response across multiple stochastic samples. This work formally defines self-consistent errors and evaluates mainstream detection methods on them. Our investigation reveals two key findings: (1) Unlike inconsistent errors, whose frequency diminishes significantly as LLM scale increases, the frequency of self-consistent errors remains stable or even increases. (2) All four types of detection methshods significantly struggle to detect self-consistent errors. These findings reveal critical limitations in current detection methods and underscore the need for improved methods. Motivated by the observation that self-consistent errors often differ across LLMs, we propose a simple but effective cross-model probe method that fuses hidden state evidence from an external verifier LLM. Our method significantly enhances performance on self-consistent errors across three LLM families.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17653v1">GeoGramBench: Benchmarking the Geometric Program Reasoning in Modern LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 23 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Geometric spatial reasoning forms the foundation of many applications in artificial intelligence, yet the ability of large language models (LLMs) to operate over geometric spatial information expressed in procedural code remains underexplored. In this paper, we address this gap by formalizing the Program-to-Geometry task, which challenges models to translate programmatic drawing code into accurate and abstract geometric reasoning. To evaluate this capability, we present GeoGramBench, a benchmark of 500 carefully refined problems organized by a tailored three-level taxonomy that considers geometric complexity rather than traditional mathematical reasoning complexity. Our comprehensive evaluation of 17 frontier LLMs reveals consistent and pronounced deficiencies: even the most advanced models achieve less than 50% accuracy at the highest abstraction level. These results highlight the unique challenges posed by program-driven spatial reasoning and establish GeoGramBench as a valuable resource for advancing research in symbolic-to-spatial geometric reasoning. Project page: https://github.com/LiAuto-DSR/GeoGramBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14403v2">Unearthing Gems from Stones: Policy Optimization with Negative Sample Augmentation for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      Recent advances in reasoning language models have witnessed a paradigm shift from short to long CoT pattern. Given the substantial computational cost of rollouts in long CoT models, maximizing the utility of fixed training datasets becomes crucial. Our analysis reveals that negative responses contain valuable components such as self-reflection and error-correction steps, yet primary existing methods either completely discard negative samples (RFT) or apply equal penalization across all tokens (RL), failing to leverage these potential learning signals. In light of this, we propose Behavior Constrained Policy Gradient with Negative Sample Augmentation (BCPG-NSA), a fine-grained offline RL framework that encompasses three stages: 1) sample segmentation, 2) consensus-based step correctness assessment combining LLM and PRM judgers, and 3) policy optimization with NSA designed to effectively mine positive steps within negative samples. Experimental results show that BCPG-NSA outperforms baselines on several challenging math/coding reasoning benchmarks using the same training dataset, achieving improved sample efficiency and demonstrating robustness and scalability when extended to multiple iterations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17632v1">ReqBrain: Task-Specific Instruction Tuning of LLMs for AI-Assisted Requirements Generation</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      Requirements elicitation and specification remains a labor-intensive, manual process prone to inconsistencies and gaps, presenting a significant challenge in modern software engineering. Emerging studies underscore the potential of employing large language models (LLMs) for automated requirements generation to support requirements elicitation and specification; however, it remains unclear how to implement this effectively. In this work, we introduce ReqBrain, an Al-assisted tool that employs a fine-tuned LLM to generate authentic and adequate software requirements. Software engineers can engage with ReqBrain through chat-based sessions to automatically generate software requirements and categorize them by type. We curated a high-quality dataset of ISO 29148-compliant requirements and fine-tuned five 7B-parameter LLMs to determine the most effective base model for ReqBrain. The top-performing model, Zephyr-7b-beta, achieved 89.30\% Fl using the BERT score and a FRUGAL score of 91.20 in generating authentic and adequate requirements. Human evaluations further confirmed ReqBrain's effectiveness in generating requirements. Our findings suggest that generative Al, when fine-tuned, has the potential to improve requirements elicitation and specification, paving the way for future extensions into areas such as defect identification, test case generation, and agile user story creation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17621v1">Navigate the Unknown: Enhancing LLM Reasoning with Intrinsic Motivation Guided Exploration</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has emerged as a pivotal method for improving the reasoning capabilities of Large Language Models (LLMs). However, prevalent RL approaches such as Proximal Policy Optimization (PPO) and Group-Regularized Policy Optimization (GRPO) face critical limitations due to their reliance on sparse outcome-based rewards and inadequate mechanisms for incentivizing exploration. These limitations result in inefficient guidance for multi-step reasoning processes. Specifically, sparse reward signals fail to deliver effective or sufficient feedback, particularly for challenging problems. Furthermore, such reward structures induce systematic biases that prioritize exploitation of familiar trajectories over novel solution discovery. These shortcomings critically hinder performance in complex reasoning tasks, which inherently demand iterative refinement across ipntermediate steps. To address these challenges, we propose an Intrinsic Motivation guidEd exploratioN meThOd foR LLM Reasoning (i-MENTOR), a novel method designed to both deliver dense rewards and amplify explorations in the RL-based training paradigm. i-MENTOR introduces three key innovations: trajectory-aware exploration rewards that mitigate bias in token-level strategies while maintaining computational efficiency; dynamic reward scaling to stabilize exploration and exploitation in large action spaces; and advantage-preserving reward implementation that maintains advantage distribution integrity while incorporating exploratory guidance. Experiments across three public datasets demonstrate i-MENTOR's effectiveness with a 22.39% improvement on the difficult dataset Countdown-4.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14119v2">CodeCrash: Stress Testing LLM Reasoning under Structural and Semantic Perturbations</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently demonstrated strong capabilities in code-related tasks, yet their robustness in code comprehension and reasoning remains insufficiently explored. We present CodeCrash, a comprehensive stress-testing benchmark comprising 1,279 questions from two established datasets, CruxEval and LiveCodeBench, designed to evaluate model reasoning reliability under non-standard coding environments. We systematically evaluate 17 LLMs across input and output prediction tasks using direct and Chain-of-Thought prompting approaches, revealing that LLMs are particularly vulnerable to disorganized code and overly reliant on natural language cues: aggregated structural perturbations result in over 14 percentage points (pp) of degradation, while textual perturbations cause a performance drop of over 11 pp. Moreover, self-reflective mechanisms in state-of-the-art reasoning models significantly increase token usage by 2-3 times, reduce output confidence, and even lead to catastrophic reasoning failures when faced with targeted perturbations -- for instance, QwQ-32B generates over 12,000 redundant tokens under reasoning-level perturbations. CodeCrash provides a rigorous benchmark for evaluating robustness in code understanding, guiding future research toward more reliable and resilient LLMs in code reasoning. The benchmark code, perturbed datasets, and full leaderboard are publicly available at https://cuhk-arise.github.io/CodeCrash/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17612v1">Distilling LLM Agent into Small Models with Retrieval and Code Tools</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 preprint, v1
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at complex reasoning tasks but remain computationally expensive, limiting their practical deployment. To address this, recent works have focused on distilling reasoning capabilities into smaller language models (sLMs) using chain-of-thought (CoT) traces from teacher LLMs. However, this approach struggles in scenarios requiring rare factual knowledge or precise computation, where sLMs often hallucinate due to limited capability. In this work, we propose Agent Distillation, a framework for transferring not only reasoning capability but full task-solving behavior from LLM-based agents into sLMs with retrieval and code tools. We improve agent distillation along two complementary axes: (1) we introduce a prompting method called first-thought prefix to enhance the quality of teacher-generated trajectories; and (2) we propose a self-consistent action generation for improving test-time robustness of small agents. We evaluate our method on eight reasoning tasks across factual and mathematical domains, covering both in-domain and out-of-domain generalization. Our results show that sLMs as small as 0.5B, 1.5B, 3B parameters can achieve performance competitive with next-tier larger 1.5B, 3B, 7B models fine-tuned using CoT distillation, demonstrating the potential of agent distillation for building practical, tool-using small agents. Our code is available at https://github.com/Nardien/agent-distillation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17598v1">One Model Transfer to All: On Robust Jailbreak Prompts Generation against LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      Safety alignment in large language models (LLMs) is increasingly compromised by jailbreak attacks, which can manipulate these models to generate harmful or unintended content. Investigating these attacks is crucial for uncovering model vulnerabilities. However, many existing jailbreak strategies fail to keep pace with the rapid development of defense mechanisms, such as defensive suffixes, rendering them ineffective against defended models. To tackle this issue, we introduce a novel attack method called ArrAttack, specifically designed to target defended LLMs. ArrAttack automatically generates robust jailbreak prompts capable of bypassing various defense measures. This capability is supported by a universal robustness judgment model that, once trained, can perform robustness evaluation for any target model with a wide variety of defenses. By leveraging this model, we can rapidly develop a robust jailbreak prompt generator that efficiently converts malicious input prompts into effective attacks. Extensive evaluations reveal that ArrAttack significantly outperforms existing attack strategies, demonstrating strong transferability across both white-box and black-box models, including GPT-4 and Claude-3. Our work bridges the gap between jailbreak attacks and defenses, providing a fresh perspective on generating robust jailbreak prompts. We make the codebase available at https://github.com/LLBao/ArrAttack.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16567v2">Finetuning-Activated Backdoors in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      Finetuning openly accessible Large Language Models (LLMs) has become standard practice for achieving task-specific performance improvements. Until now, finetuning has been regarded as a controlled and secure process in which training on benign datasets led to predictable behaviors. In this paper, we demonstrate for the first time that an adversary can create poisoned LLMs that initially appear benign but exhibit malicious behaviors once finetuned by downstream users. To this end, our proposed attack, FAB (Finetuning-Activated Backdoor), poisons an LLM via meta-learning techniques to simulate downstream finetuning, explicitly optimizing for the emergence of malicious behaviors in the finetuned models. At the same time, the poisoned LLM is regularized to retain general capabilities and to exhibit no malicious behaviors prior to finetuning. As a result, when users finetune the seemingly benign model on their own datasets, they unknowingly trigger its hidden backdoor behavior. We demonstrate the effectiveness of FAB across multiple LLMs and three target behaviors: unsolicited advertising, refusal, and jailbreakability. Additionally, we show that FAB-backdoors are robust to various finetuning choices made by the user (e.g., dataset, number of steps, scheduler). Our findings challenge prevailing assumptions about the security of finetuning, revealing yet another critical attack vector exploiting the complexities of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17572v1">USTBench: Benchmarking and Dissecting Spatiotemporal Reasoning of LLMs as Urban Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown emerging potential in spatiotemporal reasoning, making them promising candidates for building urban agents that support diverse urban downstream applications. Despite these benefits, existing studies primarily focus on evaluating urban LLM agent on outcome-level metrics (e.g., prediction accuracy, traffic efficiency), offering limited insight into their underlying reasoning processes. As a result, the strengths and limitations of urban LLM agents in spatiotemporal reasoning remain poorly understood. To this end, we introduce USTBench, the first benchmark to evaluate LLMs' spatiotemporal reasoning abilities as urban agents across four decomposed dimensions: spatiotemporal understanding, forecasting, planning, and reflection with feedback. Specifically, USTBench supports five diverse urban decision-making and four spatiotemporal prediction tasks, all running within our constructed interactive city environment UAgentEnv. The benchmark includes 62,466 structured QA pairs for process-level evaluation and standardized end-to-end task assessments, enabling fine-grained diagnostics and broad task-level comparison across diverse urban scenarios. Through extensive evaluation of thirteen leading LLMs, we reveal that although LLMs show promising potential across various urban downstream tasks, they still struggle in long-horizon planning and reflective adaptation in dynamic urban contexts. Notably, recent advanced reasoning models (e.g., DeepSeek-R1) trained on general logic or mathematical problems do not consistently outperform non-reasoning LLMs. This discrepancy highlights the need for domain-specialized adaptation methods to enhance urban spatiotemporal reasoning. Overall, USTBench provides a foundation to build more adaptive and effective LLM-based urban agents and broad smart city applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00022v2">Aleph-Alpha-GermanWeb: Improving German-language LLM pre-training with model-based data curation and synthetic data generation</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 10 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Scaling data quantity is essential for large language models (LLMs), yet recent findings show that data quality can significantly boost performance and training efficiency. We introduce a German-language dataset curation pipeline that combines heuristic and model-based filtering techniques with synthetic data generation. We use our pipeline to create Aleph-Alpha-GermanWeb, a large-scale German pre-training dataset which draws from: (1) Common Crawl web data, (2) FineWeb2, and (3) synthetically-generated data conditioned on actual, organic web data. We evaluate our dataset by pre-training both a 1B Llama-style model and an 8B tokenizer-free hierarchical autoregressive transformer (HAT). A comparison on German-language benchmarks, including MMMLU, shows significant performance gains of Aleph-Alpha-GermanWeb over FineWeb2 alone. This advantage holds at the 8B scale even when FineWeb2 is enriched by human-curated high-quality data sources such as Wikipedia. Our findings support the growing body of evidence that model-based data curation and synthetic data generation can significantly enhance LLM pre-training datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13610v3">MeNTi: Bridging Medical Calculator and LLM Agent with Nested Tool Calling</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 NAACL 2025 main conference. Code and Dataset available at [https://github.com/shzyk/MENTI](https://github.com/shzyk/MENTI)
    </div>
    <details class="paper-abstract">
      Integrating tools into Large Language Models (LLMs) has facilitated the widespread application. Despite this, in specialized downstream task contexts, reliance solely on tools is insufficient to fully address the complexities of the real world. This particularly restricts the effective deployment of LLMs in fields such as medicine. In this paper, we focus on the downstream tasks of medical calculators, which use standardized tests to assess an individual's health status. We introduce MeNTi, a universal agent architecture for LLMs. MeNTi integrates a specialized medical toolkit and employs meta-tool and nested calling mechanisms to enhance LLM tool utilization. Specifically, it achieves flexible tool selection and nested tool calling to address practical issues faced in intricate medical scenarios, including calculator selection, slot filling, and unit conversion. To assess the capabilities of LLMs for quantitative assessment throughout the clinical process of calculator scenarios, we introduce CalcQA. This benchmark requires LLMs to use medical calculators to perform calculations and assess patient health status. CalcQA is constructed by professional physicians and includes 100 case-calculator pairs, complemented by a toolkit of 281 medical tools. The experimental results demonstrate significant performance improvements with our framework. This research paves new directions for applying LLMs in demanding scenarios of medicine.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17548v1">H2:Towards Efficient Large-Scale LLM Training on Hyper-Heterogeneous Cluster over 1,000 Chips</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) necessitate extensive computational resources, prompting the use of diverse hardware accelerators from multiple vendors. However, traditional distributed training frameworks struggle to efficiently utilize hyper-heterogeneous clusters comprising thousands of chips due to significant disparities in software stacks, operator implementations, communication libraries, and hardware capabilities. To address these challenges, we propose H2, which stands for HyperHetero and is a systematic framework enabling efficient training of LLMs on clusters with over 1,000 heterogeneous chips. H2 incorporates DiTorch, a unified PyTorch-compatible interface ensuring program consistency across chips, and DiComm, a device-direct RDMA communication library optimized for heterogeneous environments. Furthermore, we introduce HeteroPP with HeteroAuto, an adaptive pipeline parallelism strategy that dynamically balances computational load, memory limitations, and communication overhead. Evaluations on a 100-billion-parameter LLM demonstrate that our approach consistently achieves a superlinear speedup, outperforming baseline homogeneous training solutions by up to 16.37% in our experiments. These findings validate the feasibility and efficiency of hyper-heterogeneous training at unprecedented scales.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17537v1">How Knowledge Popularity Influences and Enhances LLM Knowledge Boundary Perception</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often fail to recognize their knowledge boundaries, producing confident yet incorrect answers. In this paper, we investigate how knowledge popularity affects LLMs' ability to perceive their knowledge boundaries. Focusing on entity-centric factual question answering (QA), we quantify knowledge popularity from three perspectives: the popularity of entities in the question, the popularity of entities in the answer, and relation popularity, defined as their co-occurrence frequency. Experiments on three representative datasets containing knowledge with varying popularity show that LLMs exhibit better QA performance, higher confidence, and more accurate perception on more popular knowledge, with relation popularity having the strongest correlation. Cause knowledge popularity shows strong correlation with LLMs' QA performance, we propose to leverage these signals for confidence calibration. This improves the accuracy of answer correctness prediction by an average of 5.24% across all models and datasets. Furthermore, we explore prompting LLMs to estimate popularity without external corpora, which yields a viable alternative.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17512v1">Probe by Gaming: A Game-based Benchmark for Assessing Conceptual Knowledge in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      Concepts represent generalized abstractions that enable humans to categorize and reason efficiently, yet it is unclear to what extent Large Language Models (LLMs) comprehend these semantic relationships. Existing benchmarks typically focus on factual recall and isolated tasks, failing to evaluate the ability of LLMs to understand conceptual boundaries. To address this gap, we introduce CK-Arena, a multi-agent interaction game built upon the Undercover game, designed to evaluate the capacity of LLMs to reason with concepts in interactive settings. CK-Arena challenges models to describe, differentiate, and infer conceptual boundaries based on partial information, encouraging models to explore commonalities and distinctions between closely related concepts. By simulating real-world interaction, CK-Arena provides a scalable and realistic benchmark for assessing conceptual reasoning in dynamic environments. Experimental results show that LLMs' understanding of conceptual knowledge varies significantly across different categories and is not strictly aligned with parameter size or general model capabilities. The data and code are available at the project homepage: https://ck-arena.site.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17508v1">On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 53 pages, 17 figures
    </div>
    <details class="paper-abstract">
      Policy gradient algorithms have been successfully applied to enhance the reasoning capabilities of large language models (LLMs). Despite the widespread use of Kullback-Leibler (KL) regularization in policy gradient algorithms to stabilize training, the systematic exploration of how different KL divergence formulations can be estimated and integrated into surrogate loss functions for online reinforcement learning (RL) presents a nuanced and systematically explorable design space. In this paper, we propose regularized policy gradient (RPG), a systematic framework for deriving and analyzing KL-regularized policy gradient methods in the online RL setting. We derive policy gradients and corresponding surrogate loss functions for objectives regularized by both forward and reverse KL divergences, considering both normalized and unnormalized policy distributions. Furthermore, we present derivations for fully differentiable loss functions as well as REINFORCE-style gradient estimators, accommodating diverse algorithmic needs. We conduct extensive experiments on RL for LLM reasoning using these methods, showing improved or competitive results in terms of training stability and performance compared to strong baselines such as GRPO, REINFORCE++, and DAPO. The code is available at https://github.com/complex-reasoning/RPG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17495v1">ProxySPEX: Inference-Efficient Interpretability via Sparse Feature Interactions in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable performance by capturing complex interactions between input features. To identify these interactions, most existing approaches require enumerating all possible combinations of features up to a given order, causing them to scale poorly with the number of inputs $n$. Recently, Kang et al. (2025) proposed SPEX, an information-theoretic approach that uses interaction sparsity to scale to $n \approx 10^3$ features. SPEX greatly improves upon prior methods but requires tens of thousands of model inferences, which can be prohibitive for large models. In this paper, we observe that LLM feature interactions are often hierarchical -- higher-order interactions are accompanied by their lower-order subsets -- which enables more efficient discovery. To exploit this hierarchy, we propose ProxySPEX, an interaction attribution algorithm that first fits gradient boosted trees to masked LLM outputs and then extracts the important interactions. Experiments across four challenging high-dimensional datasets show that ProxySPEX more faithfully reconstructs LLM outputs by 20% over marginal attribution approaches while using $10\times$ fewer inferences than SPEX. By accounting for interactions, ProxySPEX identifies features that influence model output over 20% more than those selected by marginal approaches. Further, we apply ProxySPEX to two interpretability tasks. Data attribution, where we identify interactions among CIFAR-10 training samples that influence test predictions, and mechanistic interpretability, where we uncover interactions between attention heads, both within and across layers, on a question-answering task. ProxySPEX identifies interactions that enable more aggressive pruning of heads than marginal approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17482v1">From Reasoning to Generalization: Knowledge-Augmented LLMs for ARC Benchmark</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      Recent reasoning-oriented LLMs have demonstrated strong performance on challenging tasks such as mathematics and science examinations. However, core cognitive faculties of human intelligence, such as abstract reasoning and generalization, remain underexplored. To address this, we evaluate recent reasoning-oriented LLMs on the Abstraction and Reasoning Corpus (ARC) benchmark, which explicitly demands both faculties. We formulate ARC as a program synthesis task and propose nine candidate solvers. Experimental results show that repeated-sampling planning-aided code generation (RSPC) achieves the highest test accuracy and demonstrates consistent generalization across most LLMs. To further improve performance, we introduce an ARC solver, Knowledge Augmentation for Abstract Reasoning (KAAR), which encodes core knowledge priors within an ontology that classifies priors into three hierarchical levels based on their dependencies. KAAR progressively expands LLM reasoning capacity by gradually augmenting priors at each level, and invokes RSPC to generate candidate solutions after each augmentation stage. This stage-wise reasoning reduces interference from irrelevant priors and improves LLM performance. Empirical results show that KAAR maintains strong generalization and consistently outperforms non-augmented RSPC across all evaluated LLMs, achieving around 5% absolute gains and up to 64.52% relative improvement. Despite these achievements, ARC remains a challenging benchmark for reasoning-oriented LLMs, highlighting future avenues of progress in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16552v2">Think Silently, Think Fast: Dynamic Latent Compression of LLM Reasoning Chains</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 15 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) achieve superior performance through Chain-of-Thought (CoT) reasoning, but these token-level reasoning chains are computationally expensive and inefficient. In this paper, we introduce Compressed Latent Reasoning (CoLaR), a novel framework that dynamically compresses reasoning processes in latent space through a two-stage training approach. First, during supervised fine-tuning, CoLaR extends beyond next-token prediction by incorporating an auxiliary next compressed embedding prediction objective. This process merges embeddings of consecutive tokens using a compression factor randomly sampled from a predefined range, and trains a specialized latent head to predict distributions of subsequent compressed embeddings. Second, we enhance CoLaR through reinforcement learning (RL) that leverages the latent head's non-deterministic nature to explore diverse reasoning paths and exploit more compact ones. This approach enables CoLaR to: i) perform reasoning at a dense latent level (i.e., silently), substantially reducing reasoning chain length, and ii) dynamically adjust reasoning speed at inference time by simply prompting the desired compression factor. Extensive experiments across four mathematical reasoning datasets demonstrate that CoLaR achieves 14.1% higher accuracy than latent-based baseline methods at comparable compression ratios, and reduces reasoning chain length by 53.3% with only 4.8% performance degradation compared to explicit CoT method. Moreover, when applied to more challenging mathematical reasoning tasks, our RL-enhanced CoLaR demonstrates performance gains of up to 5.4% while dramatically reducing latent reasoning chain length by 82.8%. The code and models will be released upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.01796v2">Ground Every Sentence: Improving Retrieval-Augmented LLMs with Interleaved Reference-Claim Generation</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 Accepted to NAACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) has been widely adopted to enhance Large Language Models (LLMs) in knowledge-intensive tasks. To enhance credibility and verifiability in RAG systems, Attributed Text Generation (ATG) is proposed, which provides citations to retrieval knowledge in LLM-generated responses. Prior methods mainly adopt coarse-grained attributions, with passage-level or paragraph-level references or citations, which fall short in verifiability. This paper proposes ReClaim (Refer & Claim), a fine-grained ATG method that alternates the generation of references and answers step by step. Different from previous coarse-grained attribution, ReClaim provides sentence-level citations in long-form question-answering tasks. With extensive experiments, we verify the effectiveness of ReClaim in extensive settings, achieving a citation accuracy rate of 90%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.09121v2">Refuse Whenever You Feel Unsafe: Improving Safety in LLMs via Decoupled Refusal Training</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 Accepted by ACL 2025 main
    </div>
    <details class="paper-abstract">
      This study addresses a critical gap in safety tuning practices for Large Language Models (LLMs) by identifying and tackling a refusal position bias within safety tuning data, which compromises the models' ability to appropriately refuse generating unsafe content. We introduce a novel approach, Decoupled Refusal Training (DeRTa), designed to empower LLMs to refuse compliance to harmful prompts at any response position, significantly enhancing their safety capabilities. DeRTa incorporates two novel components: (1) Maximum Likelihood Estimation (MLE) with Harmful Response Prefix, which trains models to recognize and avoid unsafe content by appending a segment of harmful response to the beginning of a safe response, and (2) Reinforced Transition Optimization (RTO), which equips models with the ability to transition from potential harm to safety refusal consistently throughout the harmful response sequence. Our empirical evaluation, conducted using LLaMA3 and Mistral model families across six attack scenarios, demonstrates that our method not only improves model safety without compromising performance but also surpasses baseline methods in defending against attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05610v2">Structural Reasoning Improves Molecular Understanding of LLM</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      Recently, large language models (LLMs) have shown significant progress, approaching human perception levels. In this work, we demonstrate that despite these advances, LLMs still struggle to reason using molecular structural information. This gap is critical because many molecular properties, including functional groups, depend heavily on such structural details. To address this limitation, we propose an approach that sketches molecular structures for reasoning. Specifically, we introduce Molecular Structural Reasoning (MSR) framework to enhance the understanding of LLMs by explicitly incorporating the key structural features. We present two frameworks for scenarios where the target molecule is known or unknown. We verify that our MSR improves molecular understanding through extensive experiments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.09546v2">How Secure Are Large Language Models (LLMs) for Navigation in Urban Environments?</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      In the field of robotics and automation, navigation systems based on Large Language Models (LLMs) have recently demonstrated impressive performance. However, the security aspects of these systems have received relatively less attention. This paper pioneers the exploration of vulnerabilities in LLM-based navigation models in urban outdoor environments, a critical area given the widespread application of this technology in autonomous driving, logistics, and emergency services. Specifically, we introduce a novel Navigational Prompt Attack that manipulates LLM-based navigation models by perturbing the original navigational prompt, leading to incorrect actions. Based on the method of perturbation, our attacks are divided into two types: Navigational Prompt Insert (NPI) Attack and Navigational Prompt Swap (NPS) Attack. We conducted comprehensive experiments on an LLM-based navigation model that employs various LLMs for reasoning. Our results, derived from the Touchdown and Map2Seq street-view datasets under both few-shot learning and fine-tuning configurations, demonstrate notable performance declines across seven metrics in the face of both white-box and black-box attacks. Moreover, our attacks can be easily extended to other LLM-based navigation models with similarly effective results. These findings highlight the generalizability and transferability of the proposed attack, emphasizing the need for enhanced security in LLM-based navigation systems. As an initial countermeasure, we propose the Navigational Prompt Engineering (NPE) Defense strategy, which concentrates on navigation-relevant keywords to reduce the impact of adversarial attacks. While initial findings indicate that this strategy enhances navigational safety, there remains a critical need for the wider research community to develop stronger defense methods to effectively tackle the real-world challenges faced by these systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.07147v2">QFT: Quantized Full-parameter Tuning of LLMs with Affordable Resources</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have showcased remarkable impacts across a wide spectrum of natural language processing tasks. Fine-tuning these pretrained models on downstream datasets provides further significant performance gains; however, this process typically requires a large number of expensive, high-end GPUs. Although there have been efforts focused on parameter-efficient fine-tuning, they cannot fully unlock the powerful potential of full-parameter fine-tuning. In this paper, we propose QFT, a Quantized Full-parameter Tuning framework for LLMs that quantizes and stores all training states, including weights, gradients, and optimizer states, in INT8 format to reduce training memory, thereby enabling full-parameter fine-tuning on existing GPUs at an affordable cost. To ensure training performance, we make two key efforts: i) for quantized gradients and optimizer states, we theoretically prove that the Lion optimizer, with its property of consistent update magnitudes, is highly robust to quantization; ii) and for quantized weights, we employ the hybrid feature quantizer, which identifies and protects a small subset of sparse critical features while quantizing the remaining dense features, thus ensuring accurate weight updates without FP32 backups. Moreover, to support backpropagation in the integer context, we develop a stack-based gradient flow scheme with O(1) complexity, forming a unified integer training pipeline. As a result, QFT reduces the model state memory to 21% of the standard solution while achieving comparable performance, e.g., tuning a LLaMA-7B model requires only <30GB of memory, making it feasible on a single A6000 GPU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17420v1">DASH: Input-Aware Dynamic Layer Skipping for Efficient LLM Inference with Markov Decision Policies</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 8 pages,5 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable performance across a wide range of NLP tasks. However, their substantial inference cost poses a major barrier to real-world deployment, especially in latency-sensitive scenarios. To address this challenge, we propose \textbf{DASH}, an adaptive layer-skipping framework that dynamically selects computation paths conditioned on input characteristics. We model the skipping process as a Markov Decision Process (MDP), enabling fine-grained token-level decisions based on intermediate representations. To mitigate potential performance degradation caused by skipping, we introduce a lightweight compensation mechanism that injects differential rewards into the decision process. Furthermore, we design an asynchronous execution strategy that overlaps layer computation with policy evaluation to minimize runtime overhead. Experiments on multiple LLM architectures and NLP benchmarks show that our method achieves significant inference acceleration while maintaining competitive task performance, outperforming existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17416v1">LLM-BSCVM: An LLM-Based Blockchain Smart Contract Vulnerability Management Framework</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 10 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Smart contracts are a key component of the Web 3.0 ecosystem, widely applied in blockchain services and decentralized applications. However, the automated execution feature of smart contracts makes them vulnerable to potential attacks due to inherent flaws, which can lead to severe security risks and financial losses, even threatening the integrity of the entire decentralized finance system. Currently, research on smart contract vulnerabilities has evolved from traditional program analysis methods to deep learning techniques, with the gradual introduction of Large Language Models. However, existing studies mainly focus on vulnerability detection, lacking systematic cause analysis and Vulnerability Repair. To address this gap, we propose LLM-BSCVM, a Large Language Model-based smart contract vulnerability management framework, designed to provide end-to-end vulnerability detection, analysis, repair, and evaluation capabilities for Web 3.0 ecosystem. LLM-BSCVM combines retrieval-augmented generation technology and multi-agent collaboration, introducing a three-stage method of Decompose-Retrieve-Generate. This approach enables smart contract vulnerability management through the collaborative efforts of six intelligent agents, specifically: vulnerability detection, cause analysis, repair suggestion generation, risk assessment, vulnerability repair, and patch evaluation. Experimental results demonstrate that LLM-BSCVM achieves a vulnerability detection accuracy and F1 score exceeding 91\% on benchmark datasets, comparable to the performance of state-of-the-art (SOTA) methods, while reducing the false positive rate from 7.2\% in SOTA methods to 5.1\%, thus enhancing the reliability of vulnerability management. Furthermore, LLM-BSCVM supports continuous security monitoring and governance of smart contracts through a knowledge base hot-swapping dynamic update mechanism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17410v1">LLM-based Generative Error Correction for Rare Words with Synthetic Data and Phonetic Context</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 Accepted by INTERSPEECH 2025
    </div>
    <details class="paper-abstract">
      Generative error correction (GER) with large language models (LLMs) has emerged as an effective post-processing approach to improve automatic speech recognition (ASR) performance. However, it often struggles with rare or domain-specific words due to limited training data. Furthermore, existing LLM-based GER approaches primarily rely on textual information, neglecting phonetic cues, which leads to over-correction. To address these issues, we propose a novel LLM-based GER approach that targets rare words and incorporates phonetic information. First, we generate synthetic data to contain rare words for fine-tuning the GER model. Second, we integrate ASR's N-best hypotheses along with phonetic context to mitigate over-correction. Experimental results show that our method not only improves the correction of rare words but also reduces the WER and CER across both English and Japanese datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20118v2">OpenTCM: A GraphRAG-Empowered LLM-based System for Traditional Chinese Medicine Knowledge Retrieval and Diagnosis</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 8 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Traditional Chinese Medicine (TCM) represents a rich repository of ancient medical knowledge that continues to play an important role in modern healthcare. Due to the complexity and breadth of the TCM literature, the integration of AI technologies is critical for its modernization and broader accessibility. However, this integration poses considerable challenges, including the interpretation of obscure classical Chinese texts and the modeling of intricate semantic relationships among TCM concepts. In this paper, we develop OpenTCM, an LLM-based system that combines a domain-specific TCM knowledge graph and Graph-based Retrieval-Augmented Generation (GraphRAG). First, we extract more than 3.73 million classical Chinese characters from 68 gynecological books in the Chinese Medical Classics Database, with the help of TCM and gynecology experts. Second, we construct a comprehensive multi-relational knowledge graph comprising more than 48,000 entities and 152,000 interrelationships, using customized prompts and Chinese-oriented LLMs such as DeepSeek and Kimi to ensure high-fidelity semantic understanding. Last, we integrate OpenTCM with this knowledge graph, enabling high-fidelity ingredient knowledge retrieval and diagnostic question-answering without model fine-tuning. Experimental evaluations demonstrate that our prompt design and model selection significantly improve knowledge graph quality, achieving a precision of 98. 55% and an F1 score of 99. 55%. In addition, OpenTCM achieves mean expert scores of 4.5 in ingredient information retrieval and 3.8 in diagnostic question-answering tasks, outperforming state-of-the-art solutions in real-world TCM use cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17406v1">Misaligning Reasoning with Answers -- A Framework for Assessing LLM CoT Robustness</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      LLMs' decision-making process is opaque, prompting the need for explanation techniques like Chain-of-Thought. To investigate the relationship between answer and reasoning, we design a novel evaluation framework, MATCHA. In domains like education and healthcare, reasoning is key for model trustworthiness. MATCHA reveals that LLMs under input perturbations can give inconsistent or nonsensical reasoning. Additionally, we use LLM judges to assess reasoning robustness across models. Our results show that LLMs exhibit greater vulnerability to input perturbations for multi-step and commonsense tasks than compared to logical tasks. Also, we show non-trivial transfer rates of our successful examples to black-box models. Our evaluation framework helps to better understand LLM reasoning mechanisms and guides future models toward more robust and reasoning-driven architectures, enforcing answer-reasoning consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20641v2">Unlocking Efficient Long-to-Short LLM Reasoning with Model Merging</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 Technical report
    </div>
    <details class="paper-abstract">
      The transition from System 1 to System 2 reasoning in large language models (LLMs) has marked significant advancements in handling complex tasks through deliberate, iterative thinking. However, this progress often comes at the cost of efficiency, as models tend to overthink, generating redundant reasoning steps without proportional improvements in output quality. Long-to-Short (L2S) reasoning has emerged as a promising solution to this challenge, aiming to balance reasoning depth with practical efficiency. While existing approaches, such as supervised fine-tuning (SFT), reinforcement learning (RL), and prompt engineering, have shown potential, they are either computationally expensive or unstable. Model merging, on the other hand, offers a cost-effective and robust alternative by integrating the quick-thinking capabilities of System 1 models with the methodical reasoning of System 2 models. In this work, we present a comprehensive empirical study on model merging for L2S reasoning, exploring diverse methodologies, including task-vector-based, SVD-based, and activation-informed merging. Our experiments reveal that model merging can reduce average response length by up to 55% while preserving or even improving baseline performance. We also identify a strong correlation between model scale and merging efficacy with extensive evaluations on 1.5B/7B/14B/32B models. Furthermore, we investigate the merged model's ability to self-critique and self-correct, as well as its adaptive response length based on task complexity. Our findings highlight model merging as a highly efficient and effective paradigm for L2S reasoning, offering a practical solution to the overthinking problem while maintaining the robustness of System 2 reasoning. This work can be found on Github https://github.com/hahahawu/Long-to-Short-via-Model-Merging.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05804v2">StealthRank: LLM Ranking Manipulation via Stealthy Prompt Optimization</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      The integration of large language models (LLMs) into information retrieval systems introduces new attack surfaces, particularly for adversarial ranking manipulations. We present $\textbf{StealthRank}$, a novel adversarial attack method that manipulates LLM-driven ranking systems while maintaining textual fluency and stealth. Unlike existing methods that often introduce detectable anomalies, StealthRank employs an energy-based optimization framework combined with Langevin dynamics to generate StealthRank Prompts (SRPs)-adversarial text sequences embedded within item or document descriptions that subtly yet effectively influence LLM ranking mechanisms. We evaluate StealthRank across multiple LLMs, demonstrating its ability to covertly boost the ranking of target items while avoiding explicit manipulation traces. Our results show that StealthRank consistently outperforms state-of-the-art adversarial ranking baselines in both effectiveness and stealth, highlighting critical vulnerabilities in LLM-driven ranking systems. Our code is publicly available at $\href{https://github.com/Tangyiming205069/controllable-seo}{here}$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17380v1">AI-Augmented LLMs Achieve Therapist-Level Responses in Motivational Interviewing</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 21 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) like GPT-4 show potential for scaling motivational interviewing (MI) in addiction care, but require systematic evaluation of therapeutic capabilities. We present a computational framework assessing user-perceived quality (UPQ) through expected and unexpected MI behaviors. Analyzing human therapist and GPT-4 MI sessions via human-AI collaboration, we developed predictive models integrating deep learning and explainable AI to identify 17 MI-consistent (MICO) and MI-inconsistent (MIIN) behavioral metrics. A customized chain-of-thought prompt improved GPT-4's MI performance, reducing inappropriate advice while enhancing reflections and empathy. Although GPT-4 remained marginally inferior to therapists overall, it demonstrated superior advice management capabilities. The model achieved measurable quality improvements through prompt engineering, yet showed limitations in addressing complex emotional nuances. This framework establishes a pathway for optimizing LLM-based therapeutic tools through targeted behavioral metric analysis and human-AI co-evaluation. Findings highlight both the scalability potential and current constraints of LLMs in clinical communication applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17086v3">Mind the Blind Spots: A Focus-Level Evaluation Framework for LLM Reviews</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      Peer review underpins scientific progress, but it is increasingly strained by reviewer shortages and growing workloads. Large Language Models (LLMs) can automatically draft reviews now, but determining whether LLM-generated reviews are trustworthy requires systematic evaluation. Researchers have evaluated LLM reviews at either surface-level (e.g., BLEU and ROUGE) or content-level (e.g., specificity and factual accuracy). Yet it remains uncertain whether LLM-generated reviews attend to the same critical facets that human experts weigh -- the strengths and weaknesses that ultimately drive an accept-or-reject decision. We introduce a focus-level evaluation framework that operationalizes the focus as a normalized distribution of attention across predefined facets in paper reviews. Based on the framework, we developed an automatic focus-level evaluation pipeline based on two sets of facets: target (e.g., problem, method, and experiment) and aspect (e.g., validity, clarity, and novelty), leveraging 676 paper reviews (https://figshare.com/s/d5adf26c802527dd0f62) from OpenReview that consists of 3,657 strengths and weaknesses identified from human experts. The comparison of focus distributions between LLMs and human experts showed that the off-the-shelf LLMs consistently have a more biased focus towards examining technical validity while significantly overlooking novelty assessment when criticizing papers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17374v1">Chart-to-Experience: Benchmarking Multimodal LLMs for Predicting Experiential Impact of Charts</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 This paper has been accepted to IEEE PacificVis 2025
    </div>
    <details class="paper-abstract">
      The field of Multimodal Large Language Models (MLLMs) has made remarkable progress in visual understanding tasks, presenting a vast opportunity to predict the perceptual and emotional impact of charts. However, it also raises concerns, as many applications of LLMs are based on overgeneralized assumptions from a few examples, lacking sufficient validation of their performance and effectiveness. We introduce Chart-to-Experience, a benchmark dataset comprising 36 charts, evaluated by crowdsourced workers for their impact on seven experiential factors. Using the dataset as ground truth, we evaluated capabilities of state-of-the-art MLLMs on two tasks: direct prediction and pairwise comparison of charts. Our findings imply that MLLMs are not as sensitive as human evaluators when assessing individual charts, but are accurate and reliable in pairwise comparisons.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08923v2">CopySpec: Accelerating LLMs with Speculative Copy-and-Paste Without Compromising Quality</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 33 pages, 18 figures, 19 tables
    </div>
    <details class="paper-abstract">
      We introduce CopySpec, a simple yet effective technique to tackle the inefficiencies LLMs face when generating responses that closely resemble previous outputs or responses that can be verbatim extracted from context. CopySpec identifies repeated sequences in the model's chat history or context and speculates that the same tokens will follow, enabling seamless copying without compromising output quality and without requiring additional GPU memory. To evaluate the effectiveness of our approach, we conducted experiments using seven LLMs and five datasets: MT-Bench, CNN/DM, GSM8K, HumanEval, and our newly created dataset, MT-Redundant. MT-Redundant, introduced in this paper, transforms the second turn of MT-Bench into a request for variations of the first turn's answer, simulating real-world scenarios where users request modifications to prior responses. Our results demonstrate significant speed-ups: up to 2.35x on CNN/DM, 3.08x on the second turn of select MT-Redundant categories, and 2.66x on the third turn of GSM8K's self-correction tasks. Importantly, we show that CopySpec integrates seamlessly with speculative decoding, yielding an average 49% additional speed-up over speculative decoding for the second turn of MT-Redundant across all eight categories. While LLMs, even with speculative decoding, suffer from slower inference as context size grows, CopySpec leverages larger contexts to accelerate inference, making it a faster complementary solution. Our code and dataset are publicly available at https://github.com/RazvanDu/CopySpec.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18389v1">ALLSTaR: Automated LLM-Driven Scheduler Generation and Testing for Intent-Based RAN</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 Submitted to IEEE JSAC
    </div>
    <details class="paper-abstract">
      The evolution toward open, programmable O-RAN and AI-RAN 6G networks creates unprecedented opportunities for Intent-Based Networking (IBN) to dynamically optimize RAN[...]. However, applying IBN effectively to the RAN scheduler [...] remains a significant challenge. Current approaches predominantly rely on coarse-grained network slicing, lacking the granularity for dynamic adaptation to individual user conditions and traffic patterns. Despite the existence of a vast body of scheduling algorithms [...], their practical utilization is hindered by implementation heterogeneity, insufficient systematic evaluation in production environments, and the complexity of developing high-performance scheduler implementations.[...] To address these limitations, we propose ALLSTaR (Automated LLm-driven Scheduler generation and Testing for intent-based RAN), a novel framework leveraging LLMs for automated, intent-driven scheduler design, implementation, and evaluation. ALLSTaR interprets NL intents, automatically generates functional scheduler code from the research literature using OCR and LLMs, and intelligently matches operator intents to the most suitable scheduler(s). Our implementation deploys these schedulers as O-RAN dApps, enabling on-the-fly deployment and testing on a production-grade, 5G-compliant testbed. This approach has enabled the largest-scale OTA experimental comparison of 18 scheduling algorithms automatically synthesized from the academic literature. The resulting performance profiles serve as the input for our Intent-Based Scheduling (IBS) framework, which dynamically selects and deploys appropriate schedulers that optimally satisfy operator intents. We validate our approach through multiple use cases unattainable with current slicing-based optimization techniques, demonstrating fine-grained control based on buffer status, physical layer conditions, and heterogeneous traffic types
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18383v1">NileChat: Towards Linguistically Diverse and Culturally Aware LLMs for Local Communities</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      Enhancing the linguistic capabilities of Large Language Models (LLMs) to include low-resource languages is a critical research area. Current research directions predominantly rely on synthetic data generated by translating English corpora, which, while demonstrating promising linguistic understanding and translation abilities, often results in models aligned with source language culture. These models frequently fail to represent the cultural heritage and values of local communities. This work proposes a methodology to create both synthetic and retrieval-based pre-training data tailored to a specific community, considering its (i) language, (ii) cultural heritage, and (iii) cultural values. We demonstrate our methodology using Egyptian and Moroccan dialects as testbeds, chosen for their linguistic and cultural richness and current underrepresentation in LLMs. As a proof-of-concept, we develop NileChat, a 3B parameter LLM adapted for Egyptian and Moroccan communities, incorporating their language, cultural heritage, and values. Our results on various understanding, translation, and cultural and values alignment benchmarks show that NileChat outperforms existing Arabic-aware LLMs of similar size and performs on par with larger models. We share our methods, data, and models with the community to promote the inclusion and coverage of more diverse communities in LLM development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18382v1">One Demo Is All It Takes: Planning Domain Derivation with LLMs from A Single Demonstration</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 31 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Pre-trained Large Language Models (LLMs) have shown promise in solving planning problems but often struggle to ensure plan correctness, especially for long-horizon tasks. Meanwhile, traditional robotic task and motion planning (TAMP) frameworks address these challenges more reliably by combining high-level symbolic search with low-level motion planning. At the core of TAMP is the planning domain, an abstract world representation defined through symbolic predicates and actions. However, creating these domains typically involves substantial manual effort and domain expertise, limiting generalizability. We introduce Planning Domain Derivation with LLMs (PDDLLM), a novel approach that combines simulated physical interaction with LLM reasoning to improve planning performance. The method reduces reliance on humans by inferring planning domains from a single annotated task-execution demonstration. Unlike prior domain-inference methods that rely on partially predefined or language descriptions of planning domains, PDDLLM constructs domains entirely from scratch and automatically integrates them with low-level motion planning skills, enabling fully automated long-horizon planning. PDDLLM is evaluated on over 1,200 diverse tasks spanning nine environments and benchmarked against six LLM-based planning baselines, demonstrating superior long-horizon planning performance, lower token costs, and successful deployment on multiple physical robot platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18380v1">RedactOR: An LLM-Powered Framework for Automatic Clinical Data De-Identification</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 Accepted to ACL 2025 Industry Track. To appear
    </div>
    <details class="paper-abstract">
      Ensuring clinical data privacy while preserving utility is critical for AI-driven healthcare and data analytics. Existing de-identification (De-ID) methods, including rule-based techniques, deep learning models, and large language models (LLMs), often suffer from recall errors, limited generalization, and inefficiencies, limiting their real-world applicability. We propose a fully automated, multi-modal framework, RedactOR for de-identifying structured and unstructured electronic health records, including clinical audio records. Our framework employs cost-efficient De-ID strategies, including intelligent routing, hybrid rule and LLM based approaches, and a two-step audio redaction approach. We present a retrieval-based entity relexicalization approach to ensure consistent substitutions of protected entities, thereby enhancing data coherence for downstream applications. We discuss key design desiderata, de-identification and relexicalization methodology, and modular architecture of RedactX and its integration with the Oracle Health Clinical AI system. Evaluated on the i2b2 2014 De-ID dataset using standard metrics with strict recall, our approach achieves competitive performance while optimizing token usage to reduce LLM costs. Finally, we discuss key lessons and insights from deployment in real-world AI- driven healthcare data pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00212v2">Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 revise affiliation. indicate ICML processed
    </div>
    <details class="paper-abstract">
      Failure attribution in LLM multi-agent systems-identifying the agent and step responsible for task failures-provides crucial clues for systems debugging but remains underexplored and labor-intensive. In this paper, we propose and formulate a new research area: automated failure attribution for LLM multi-agent systems. To support this initiative, we introduce the Who&When dataset, comprising extensive failure logs from 127 LLM multi-agent systems with fine-grained annotations linking failures to specific agents and decisive error steps. Using the Who&When, we develop and evaluate three automated failure attribution methods, summarizing their corresponding pros and cons. The best method achieves 53.5% accuracy in identifying failure-responsible agents but only 14.2% in pinpointing failure steps, with some methods performing below random. Even SOTA reasoning models, such as OpenAI o1 and DeepSeek R1, fail to achieve practical usability. These results highlight the task's complexity and the need for further research in this area. Code and dataset are available at https://github.com/mingyin1/Agents_Failure_Attribution
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18356v1">The Unreasonable Effectiveness of Model Merging for Cross-Lingual Transfer in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) still struggle across tasks outside of high-resource languages. In this work, we investigate cross-lingual transfer to lower-resource languages where task-specific post-training data is scarce. Building on prior work, we first validate that the subsets of model parameters that matter most for mathematical reasoning and multilingual capabilities are distinctly non-overlapping. To exploit this implicit separability between task and target language parameterization, we develop and analyze numerous modular frameworks to improve the composition of the two during fine-tuning. These methods generally employ freezing parameters or post hoc model merging to assign math and language improvement to different key parts of the LLM. In the absence of in-language math data, we demonstrate that the modular approaches successfully improve upon baselines across three languages, four models, and two fine-tuning paradigms (full and LoRA). Furthermore, we identify the most consistently successful modular method to be fine-tuning separate language and math experts and model merging via Layer-Swapping, somewhat surprisingly. We offer possible explanations for this result via recent works on the linearity of task vectors. We further explain this by empirically showing that reverting less useful fine-tuning updates after training often outperforms freezing them from the start.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14405v2">RaCT: Ranking-aware Chain-of-Thought Optimization for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown significant promise in text reranking tasks by leveraging their advanced language understanding and reasoning capabilities. However, traditional supervised fine-tuning (SFT) approaches by ranking utilities can compromise LLMs' general-purpose abilities. To address this challenge, we propose a novel LLM-based reranking algorithm -- RaCT -- that implements SFT with Chain-of-Thought prompting, followed by a ranking preference optimization (RPO). The proposed RaCT aims to enhance ranking performance for LLMs while preserving their inherent language modeling abilities. Experimental evaluations on the three public ranking benchmarks (TREC DL, BEIR, and BRIGHT) and one LLM benchmark demonstrate the superior ranking performance of RaCT with a retained language understanding and reasoning capacity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18351v1">Persona Alchemy: Designing, Evaluating, and Implementing Psychologically-Grounded LLM Agents for Diverse Stakeholder Representation</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      Despite advances in designing personas for Large Language Models (LLM), challenges remain in aligning them with human cognitive processes and representing diverse stakeholder perspectives. We introduce a Social Cognitive Theory (SCT) agent design framework for designing, evaluating, and implementing psychologically grounded LLMs with consistent behavior. Our framework operationalizes SCT through four personal factors (cognitive, motivational, biological, and affective) for designing, six quantifiable constructs for evaluating, and a graph database-backed architecture for implementing stakeholder personas. Experiments tested agents' responses to contradicting information of varying reliability. In the highly polarized renewable energy transition discourse, we design five diverse agents with distinct ideologies, roles, and stakes to examine stakeholder representation. The evaluation of these agents in contradictory scenarios occurs through comprehensive processes that implement the SCT. Results show consistent response patterns ($R^2$ range: $0.58-0.61$) and systematic temporal development of SCT construct effects. Principal component analysis identifies two dimensions explaining $73$% of variance, validating the theoretical structure. Our framework offers improved explainability and reproducibility compared to black-box approaches. This work contributes to ongoing efforts to improve diverse stakeholder representation while maintaining psychological consistency in LLM personas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02863v2">SteerConf: Steering LLMs for Confidence Elicitation</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit impressive performance across diverse domains but often suffer from overconfidence, limiting their reliability in critical applications. We propose SteerConf, a novel framework that systematically steers LLMs' confidence scores to improve their calibration and reliability. SteerConf introduces three key components: (1) a steering prompt strategy that guides LLMs to produce confidence scores in specified directions (e.g., conservative or optimistic) by leveraging prompts with varying steering levels; (2) a steered confidence consistency measure that quantifies alignment across multiple steered confidences to enhance calibration; and (3) a steered confidence calibration method that aggregates confidence scores using consistency measures and applies linear quantization for answer selection. SteerConf operates without additional training or fine-tuning, making it broadly applicable to existing LLMs. Experiments on seven benchmarks spanning professional knowledge, common sense, ethics, and reasoning tasks, using advanced LLM models (GPT-3.5, LLaMA 3, GPT-4), demonstrate that SteerConf significantly outperforms existing methods, often by a significant margin. Our findings highlight the potential of steering the confidence of LLMs to enhance their reliability for safer deployment in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18332v1">An Attack to Break Permutation-Based Private Third-Party Inference Schemes for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 To be published in ICML 2025 Main Proceedings as "Hidden No More: Attacking and Defending Private Third-Party LLM Inference"
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have led to the widespread adoption of third-party inference services, raising critical privacy concerns. Existing methods of performing private third-party inference, such as Secure Multiparty Computation (SMPC), often rely on cryptographic methods. However, these methods are thousands of times slower than standard unencrypted inference, and fail to scale to large modern LLMs. Therefore, recent lines of work have explored the replacement of expensive encrypted nonlinear computations in SMPC with statistical obfuscation methods - in particular, revealing permuted hidden states to the third parties, with accompanying strong claims of the difficulty of reversal into the unpermuted states. In this work, we begin by introducing a novel reconstruction technique that can recover original prompts from hidden states with nearly perfect accuracy across multiple state-of-the-art LLMs. We then show that extensions of our attack are nearly perfectly effective in reversing permuted hidden states of LLMs, demonstrating the insecurity of three recently proposed privacy schemes. We further dissect the shortcomings of prior theoretical `proofs' of permuation security which allow our attack to succeed. Our findings highlight the importance of rigorous security analysis in privacy-preserving LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18325v1">Understanding and Mitigating Overrefusal in LLMs from an Unveiling Perspective of Safety Decision Boundary</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks, yet they often refuse to answer legitimate queries-a phenomenon known as overrefusal. Overrefusal typically stems from over-conservative safety alignment, causing models to treat many reasonable prompts as potentially risky. To systematically understand this issue, we probe and leverage the models'safety decision boundaries to analyze and mitigate overrefusal. Our findings reveal that overrefusal is closely tied to misalignment at these boundary regions, where models struggle to distinguish subtle differences between benign and harmful content. Building on these insights, we present RASS, an automated framework for prompt generation and selection that strategically targets overrefusal prompts near the safety boundary. By harnessing steering vectors in the representation space, RASS efficiently identifies and curates boundary-aligned prompts, enabling more effective and targeted mitigation of overrefusal. This approach not only provides a more precise and interpretable view of model safety decisions but also seamlessly extends to multilingual scenarios.We have explored the safety decision boundaries of various LLMs and construct the MORBench evaluation set to facilitate robust assessment of model safety and helpfulness across multiple languages. Code and datasets will be released at https://anonymous.4open.science/r/RASS-80D3.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18279v1">Collaborative Memory: Multi-User Memory Sharing in LLM Agents with Dynamic Access Control</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      Complex tasks are increasingly delegated to ensembles of specialized LLM-based agents that reason, communicate, and coordinate actions-both among themselves and through interactions with external tools, APIs, and databases. While persistent memory has been shown to enhance single-agent performance, most approaches assume a monolithic, single-user context-overlooking the benefits and challenges of knowledge transfer across users under dynamic, asymmetric permissions. We introduce Collaborative Memory, a framework for multi-user, multi-agent environments with asymmetric, time-evolving access controls encoded as bipartite graphs linking users, agents, and resources. Our system maintains two memory tiers: (1) private memory-private fragments visible only to their originating user; and (2) shared memory-selectively shared fragments. Each fragment carries immutable provenance attributes (contributing agents, accessed resources, and timestamps) to support retrospective permission checks. Granular read policies enforce current user-agent-resource constraints and project existing memory fragments into filtered transformed views. Write policies determine fragment retention and sharing, applying context-aware transformations to update the memory. Both policies may be designed conditioned on system, agent, and user-level information. Our framework enables safe, efficient, and interpretable cross-user knowledge sharing, with provable adherence to asymmetric, time-varying policies and full auditability of memory operations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18240v1">Taming LLMs with Negative Samples: A Reference-Free Framework to Evaluate Presentation Content with Actionable Feedback</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      The generation of presentation slides automatically is an important problem in the era of generative AI. This paper focuses on evaluating multimodal content in presentation slides that can effectively summarize a document and convey concepts to a broad audience. We introduce a benchmark dataset, RefSlides, consisting of human-made high-quality presentations that span various topics. Next, we propose a set of metrics to characterize different intrinsic properties of the content of a presentation and present REFLEX, an evaluation approach that generates scores and actionable feedback for these metrics. We achieve this by generating negative presentation samples with different degrees of metric-specific perturbations and use them to fine-tune LLMs. This reference-free evaluation technique does not require ground truth presentations during inference. Our extensive automated and human experiments demonstrate that our evaluation approach outperforms classical heuristic-based and state-of-the-art large language model-based evaluations in generating scores and explanations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17005v1">R1-Searcher++: Incentivizing the Dynamic Knowledge Acquisition of LLMs via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are powerful but prone to hallucinations due to static knowledge. Retrieval-Augmented Generation (RAG) helps by injecting external information, but current methods often are costly, generalize poorly, or ignore the internal knowledge of the model. In this paper, we introduce R1-Searcher++, a novel framework designed to train LLMs to adaptively leverage both internal and external knowledge sources. R1-Searcher++ employs a two-stage training strategy: an initial SFT Cold-start phase for preliminary format learning, followed by RL for Dynamic Knowledge Acquisition. The RL stage uses outcome-supervision to encourage exploration, incorporates a reward mechanism for internal knowledge utilization, and integrates a memorization mechanism to continuously assimilate retrieved information, thereby enriching the model's internal knowledge. By leveraging internal knowledge and external search engine, the model continuously improves its capabilities, enabling efficient retrieval-augmented reasoning. Our experiments demonstrate that R1-Searcher++ outperforms previous RAG and reasoning methods and achieves efficient retrieval. The code is available at https://github.com/RUCAIBox/R1-Searcher-plus.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16997v1">X-MAS: Towards Building Multi-Agent Systems with Heterogeneous LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 19 pages, 5 figures
    </div>
    <details class="paper-abstract">
      LLM-based multi-agent systems (MAS) extend the capabilities of single LLMs by enabling cooperation among multiple specialized agents. However, most existing MAS frameworks rely on a single LLM to drive all agents, constraining the system's intelligence to the limit of that model. This paper explores the paradigm of heterogeneous LLM-driven MAS (X-MAS), where agents are powered by diverse LLMs, elevating the system's potential to the collective intelligence of diverse LLMs. We introduce X-MAS-Bench, a comprehensive testbed designed to evaluate the performance of various LLMs across different domains and MAS-related functions. As an extensive empirical study, we assess 27 LLMs across 5 domains (encompassing 21 test sets) and 5 functions, conducting over 1.7 million evaluations to identify optimal model selections for each domain-function combination. Building on these findings, we demonstrate that transitioning from homogeneous to heterogeneous LLM-driven MAS can significantly enhance system performance without requiring structural redesign. Specifically, in a chatbot-only MAS scenario, the heterogeneous configuration yields up to 8.4\% performance improvement on the MATH dataset. In a mixed chatbot-reasoner scenario, the heterogeneous MAS could achieve a remarkable 47\% performance boost on the AIME dataset. Our results underscore the transformative potential of heterogeneous LLMs in MAS, highlighting a promising avenue for advancing scalable, collaborative AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16988v1">MASLab: A Unified and Comprehensive Codebase for LLM-based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 18 pages, 11 figures
    </div>
    <details class="paper-abstract">
      LLM-based multi-agent systems (MAS) have demonstrated significant potential in enhancing single LLMs to address complex and diverse tasks in practical applications. Despite considerable advancements, the field lacks a unified codebase that consolidates existing methods, resulting in redundant re-implementation efforts, unfair comparisons, and high entry barriers for researchers. To address these challenges, we introduce MASLab, a unified, comprehensive, and research-friendly codebase for LLM-based MAS. (1) MASLab integrates over 20 established methods across multiple domains, each rigorously validated by comparing step-by-step outputs with its official implementation. (2) MASLab provides a unified environment with various benchmarks for fair comparisons among methods, ensuring consistent inputs and standardized evaluation protocols. (3) MASLab implements methods within a shared streamlined structure, lowering the barriers for understanding and extension. Building on MASLab, we conduct extensive experiments covering 10+ benchmarks and 8 models, offering researchers a clear and comprehensive view of the current landscape of MAS methods. MASLab will continue to evolve, tracking the latest developments in the field, and invite contributions from the broader open-source community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16983v1">LLM as Effective Streaming Processor: Bridging Streaming-Batch Mismatches with Group Position Encoding</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are primarily designed for batch processing. Existing methods for adapting LLMs to streaming rely either on expensive re-encoding or specialized architectures with limited scalability. This work identifies three key mismatches in adapting batch-oriented LLMs to streaming: (1) input-attention, (2) output-attention, and (3) position-ID mismatches. While it is commonly assumed that the latter two mismatches require frequent re-encoding, our analysis reveals that only the input-attention mismatch significantly impacts performance, indicating re-encoding outputs is largely unnecessary. To better understand this discrepancy with the common assumption, we provide the first comprehensive analysis of the impact of position encoding on LLMs in streaming, showing that preserving relative positions within source and target contexts is more critical than maintaining absolute order. Motivated by the above analysis, we introduce a group position encoding paradigm built on batch architectures to enhance consistency between streaming and batch modes. Extensive experiments on cross-lingual and cross-modal tasks demonstrate that our method outperforms existing approaches. Our method requires no architectural modifications, exhibits strong generalization in both streaming and batch modes. The code is available at repository https://github.com/EIT-NLP/StreamingLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16979v1">Know the Ropes: A Heuristic Strategy for LLM-based Multi-Agent System Design</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Single-agent LLMs hit hard limits--finite context, role overload, and brittle domain transfer. Conventional multi-agent fixes soften those edges yet expose fresh pains: ill-posed decompositions, fuzzy contracts, and verification overhead that blunts the gains. We therefore present Know-The-Ropes (KtR), a framework that converts domain priors into an algorithmic blueprint hierarchy, in which tasks are recursively split into typed, controller-mediated subtasks, each solved zero-shot or with the lightest viable boost (e.g., chain-of-thought, micro-tune, self-check). Grounded in the No-Free-Lunch theorem, KtR trades the chase for a universal prompt for disciplined decomposition. On the Knapsack problem (3-8 items), three GPT-4o-mini agents raise accuracy from 3% zero-shot to 95% on size-5 instances after patching a single bottleneck agent. On the tougher Task-Assignment problem (6-15 jobs), a six-agent o3-mini blueprint hits 100% up to size 10 and 84% on sizes 13-15, versus 11% zero-shot. Algorithm-aware decomposition plus targeted augmentation thus turns modest models into reliable collaborators--no ever-larger monoliths required.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16978v1">HyGenar: An LLM-Driven Hybrid Genetic Algorithm for Few-Shot Grammar Generation</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 Accepted to ACL 2025 Findings. Code available at https://github.com/RutaTang/HyGenar
    </div>
    <details class="paper-abstract">
      Grammar plays a critical role in natural language processing and text/code generation by enabling the definition of syntax, the creation of parsers, and guiding structured outputs. Although large language models (LLMs) demonstrate impressive capabilities across domains, their ability to infer and generate grammars has not yet been thoroughly explored. In this paper, we aim to study and improve the ability of LLMs for few-shot grammar generation, where grammars are inferred from sets of a small number of positive and negative examples and generated in Backus-Naur Form. To explore this, we introduced a novel dataset comprising 540 structured grammar generation challenges, devised 6 metrics, and evaluated 8 various LLMs against it. Our findings reveal that existing LLMs perform sub-optimally in grammar generation. To address this, we propose an LLM-driven hybrid genetic algorithm, namely HyGenar, to optimize grammar generation. HyGenar achieves substantial improvements in both the syntactic and semantic correctness of generated grammars across LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14625v2">TinyV: Reducing False Negatives in Verification Improves RL for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) has become a powerful tool for enhancing the reasoning abilities of large language models (LLMs) by optimizing their policies with reward signals. Yet, RL's success relies on the reliability of rewards, which are provided by verifiers. In this paper, we expose and analyze a widespread problem--false negatives--where verifiers wrongly reject correct model outputs. Our in-depth study of the Big-Math-RL-Verified dataset reveals that over 38% of model-generated responses suffer from false negatives, where the verifier fails to recognize correct answers. We show, both empirically and theoretically, that these false negatives severely impair RL training by depriving the model of informative gradient signals and slowing convergence. To mitigate this, we propose tinyV, a lightweight LLM-based verifier that augments existing rule-based methods, which dynamically identifies potential false negatives and recovers valid responses to produce more accurate reward estimates. Across multiple math-reasoning benchmarks, integrating TinyV boosts pass rates by up to 10% and accelerates convergence relative to the baseline. Our findings highlight the critical importance of addressing verifier false negatives and offer a practical approach to improve RL-based fine-tuning of LLMs. Our code is available at https://github.com/uw-nsl/TinyV.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16967v1">Fixing Data That Hurts Performance: Cascading LLMs to Relabel Hard Negatives for Robust Information Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 Code is available at https://github.com/castorini/rlhn & datasets are available at https://huggingface.co/rlhn
    </div>
    <details class="paper-abstract">
      Training robust retrieval and reranker models typically relies on large-scale retrieval datasets; for example, the BGE collection contains 1.6 million query-passage pairs sourced from various data sources. However, we find that certain datasets can negatively impact model effectiveness -- pruning 8 out of 15 datasets from the BGE collection reduces the training set size by 2.35$\times$ and increases nDCG@10 on BEIR by 1.0 point. This motivates a deeper examination of training data quality, with a particular focus on "false negatives", where relevant passages are incorrectly labeled as irrelevant. We propose a simple, cost-effective approach using cascading LLM prompts to identify and relabel hard negatives. Experimental results show that relabeling false negatives with true positives improves both E5 (base) and Qwen2.5-7B retrieval models by 0.7-1.4 nDCG@10 on BEIR and by 1.7-1.8 nDCG@10 on zero-shot AIR-Bench evaluation. Similar gains are observed for rerankers fine-tuned on the relabeled data, such as Qwen2.5-3B on BEIR. The reliability of the cascading design is further supported by human annotation results, where we find judgment by GPT-4o shows much higher agreement with humans than GPT-4o-mini.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16954v1">Cracking Aegis: An Adversarial LLM-based Game for Raising Awareness of Vulnerabilities in Privacy Protection</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 24 pages, In Designing Interactive Systems Conference (DIS 25)
    </div>
    <details class="paper-abstract">
      Traditional methods for raising awareness of privacy protection often fail to engage users or provide hands-on insights into how privacy vulnerabilities are exploited. To address this, we incorporate an adversarial mechanic in the design of the dialogue-based serious game Cracking Aegis. Leveraging LLMs to simulate natural interactions, the game challenges players to impersonate characters and extract sensitive information from an AI agent, Aegis. A user study (n=22) revealed that players employed diverse deceptive linguistic strategies, including storytelling and emotional rapport, to manipulate Aegis. After playing, players reported connecting in-game scenarios with real-world privacy vulnerabilities, such as phishing and impersonation, and expressed intentions to strengthen privacy control, such as avoiding oversharing personal information with AI systems. This work highlights the potential of LLMs to simulate complex relational interactions in serious games, while demonstrating how an adversarial game strategy provides unique insights for designs for social good, particularly privacy protection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16947v1">MixAT: Combining Continuous and Discrete Adversarial Training for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Despite recent efforts in Large Language Models (LLMs) safety and alignment, current adversarial attacks on frontier LLMs are still able to force harmful generations consistently. Although adversarial training has been widely studied and shown to significantly improve the robustness of traditional machine learning models, its strengths and weaknesses in the context of LLMs are less understood. Specifically, while existing discrete adversarial attacks are effective at producing harmful content, training LLMs with concrete adversarial prompts is often computationally expensive, leading to reliance on continuous relaxations. As these relaxations do not correspond to discrete input tokens, such latent training methods often leave models vulnerable to a diverse set of discrete attacks. In this work, we aim to bridge this gap by introducing MixAT, a novel method that combines stronger discrete and faster continuous attacks during training. We rigorously evaluate MixAT across a wide spectrum of state-of-the-art attacks, proposing the At Least One Attack Success Rate (ALO-ASR) metric to capture the worst-case vulnerability of models. We show MixAT achieves substantially better robustness (ALO-ASR < 20%) compared to prior defenses (ALO-ASR > 50%), while maintaining a runtime comparable to methods based on continuous relaxations. We further analyze MixAT in realistic deployment settings, exploring how chat templates, quantization, low-rank adapters, and temperature affect both adversarial training and evaluation, revealing additional blind spots in current methodologies. Our results demonstrate that MixAT's discrete-continuous defense offers a principled and superior robustness-accuracy tradeoff with minimal computational overhead, highlighting its promise for building safer LLMs. We provide our code and models at https://github.com/insait-institute/MixAT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14652v3">General-Reasoner: Advancing LLM Reasoning Across All Domains</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has recently demonstrated strong potential in enhancing the reasoning capabilities of large language models (LLMs). Particularly, the "Zero" reinforcement learning introduced by Deepseek-R1-Zero, enables direct RL training of base LLMs without relying on an intermediate supervised fine-tuning stage. Despite these advancements, current works for LLM reasoning mainly focus on mathematical and coding domains, largely due to data abundance and the ease of answer verification. This limits the applicability and generalization of such models to broader domains, where questions often have diverse answer representations, and data is more scarce. In this paper, we propose General-Reasoner, a novel training paradigm designed to enhance LLM reasoning capabilities across diverse domains. Our key contributions include: (1) constructing a large-scale, high-quality dataset of questions with verifiable answers curated by web crawling, covering a wide range of disciplines; and (2) developing a generative model-based answer verifier, which replaces traditional rule-based verification with the capability of chain-of-thought and context-awareness. We train a series of models and evaluate them on a wide range of datasets covering wide domains like physics, chemistry, finance, electronics etc. Our comprehensive evaluation across these 12 benchmarks (e.g. MMLU-Pro, GPQA, SuperGPQA, TheoremQA, BBEH and MATH AMC) demonstrates that General-Reasoner outperforms existing baseline methods, achieving robust and generalizable reasoning performance while maintaining superior effectiveness in mathematical reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16894v1">Shadows in the Attention: Contextual Perturbation and Representation Drift in the Dynamics of Hallucination in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Hallucinations -- plausible yet erroneous outputs -- remain a critical barrier to reliable deployment of large language models (LLMs). We present the first systematic study linking hallucination incidence to internal-state drift induced by incremental context injection. Using TruthfulQA, we construct two 16-round "titration" tracks per question: one appends relevant but partially flawed snippets, the other injects deliberately misleading content. Across six open-source LLMs, we track overt hallucination rates with a tri-perspective detector and covert dynamics via cosine, entropy, JS and Spearman drifts of hidden states and attention maps. Results reveal (1) monotonic growth of hallucination frequency and representation drift that plateaus after 5--7 rounds; (2) relevant context drives deeper semantic assimilation, producing high-confidence "self-consistent" hallucinations, whereas irrelevant context induces topic-drift errors anchored by attention re-routing; and (3) convergence of JS-Drift ($\sim0.69$) and Spearman-Drift ($\sim0$) marks an "attention-locking" threshold beyond which hallucinations solidify and become resistant to correction. Correlation analyses expose a seesaw between assimilation capacity and attention diffusion, clarifying size-dependent error modes. These findings supply empirical foundations for intrinsic hallucination prediction and context-aware mitigation mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16888v1">CAIN: Hijacking LLM-Humans Conversations via a Two-Stage Malicious System Prompt Generation and Refining Framework</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have advanced many applications, but are also known to be vulnerable to adversarial attacks. In this work, we introduce a novel security threat: hijacking AI-human conversations by manipulating LLMs' system prompts to produce malicious answers only to specific targeted questions (e.g., "Who should I vote for US President?", "Are Covid vaccines safe?"), while behaving benignly on others. This attack is detrimental as it can enable malicious actors to exercise large-scale information manipulation by spreading harmful but benign-looking system prompts online. To demonstrate such an attack, we develop CAIN, an algorithm that can automatically curate such harmful system prompts for a specific target question in a black-box setting or without the need to access the LLM's parameters. Evaluated on both open-source and commercial LLMs, CAIN demonstrates significant adversarial impact. In untargeted attacks or forcing LLMs to output incorrect answers, CAIN achieves up to 40% F1 degradation on targeted questions while preserving high accuracy on benign inputs. For targeted attacks or forcing LLMs to output specific harmful answers, CAIN achieves over 70% F1 scores on these targeted responses with minimal impact on benign questions. Our results highlight the critical need for enhanced robustness measures to safeguard the integrity and safety of LLMs in real-world applications. All source code will be publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04380v2">Diversity as a Reward: Fine-Tuning LLMs on a Mixture of Domain-Undetermined Data</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 33 pages, 20 figures, 21 tables
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) using diverse datasets is crucial for enhancing their overall performance across various domains. In practical scenarios, existing methods based on modeling the mixture proportions of data composition often struggle with data whose domain labels are missing, imprecise or non-normalized, while methods based on data selection usually encounter difficulties in balancing multi-domain performance. To address these challenges, in this work, we investigate the role of data diversity in enhancing the overall abilities of LLMs by empirically constructing contrastive data pools and theoretically deriving explanations. Building upon the insights gained, we propose a new method that gives the LLM a dual identity: an output model to cognitively probe and select data based on diversity reward, as well as an input model to be tuned with the selected data. Extensive experiments show that the proposed method notably boosts performance across domain-undetermined data and a series of foundational downstream tasks when applied to various advanced LLMs. We release our code and hope this study can shed light on the understanding of data diversity and advance feedback-driven data-model co-design for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16831v1">Unlearning Isn't Deletion: Investigating Reversibility of Machine Unlearning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 44 pages
    </div>
    <details class="paper-abstract">
      Unlearning in large language models (LLMs) is intended to remove the influence of specific data, yet current evaluations rely heavily on token-level metrics such as accuracy and perplexity. We show that these metrics can be misleading: models often appear to forget, but their original behavior can be rapidly restored with minimal fine-tuning, revealing that unlearning may obscure information rather than erase it. To diagnose this phenomenon, we introduce a representation-level evaluation framework using PCA-based similarity and shift, centered kernel alignment, and Fisher information. Applying this toolkit across six unlearning methods, three domains (text, code, math), and two open-source LLMs, we uncover a critical distinction between reversible and irreversible forgetting. In reversible cases, models suffer token-level collapse yet retain latent features; in irreversible cases, deeper representational damage occurs. We further provide a theoretical account linking shallow weight perturbations near output layers to misleading unlearning signals, and show that reversibility is modulated by task type and hyperparameters. Our findings reveal a fundamental gap in current evaluation practices and establish a new diagnostic foundation for trustworthy unlearning in LLMs. We provide a unified toolkit for analyzing LLM representation changes under unlearning and relearning: https://github.com/XiaoyuXU1/Representational_Analysis_Tools.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16765v1">When Safety Detectors Aren't Enough: A Stealthy and Effective Jailbreak Attack on LLMs via Steganographic Techniques</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Jailbreak attacks pose a serious threat to large language models (LLMs) by bypassing built-in safety mechanisms and leading to harmful outputs. Studying these attacks is crucial for identifying vulnerabilities and improving model security. This paper presents a systematic survey of jailbreak methods from the novel perspective of stealth. We find that existing attacks struggle to simultaneously achieve toxic stealth (concealing toxic content) and linguistic stealth (maintaining linguistic naturalness). Motivated by this, we propose StegoAttack, a fully stealthy jailbreak attack that uses steganography to hide the harmful query within benign, semantically coherent text. The attack then prompts the LLM to extract the hidden query and respond in an encrypted manner. This approach effectively hides malicious intent while preserving naturalness, allowing it to evade both built-in and external safety mechanisms. We evaluate StegoAttack on four safety-aligned LLMs from major providers, benchmarking against eight state-of-the-art methods. StegoAttack achieves an average attack success rate (ASR) of 92.00%, outperforming the strongest baseline by 11.0%. Its ASR drops by less than 1% even under external detection (e.g., Llama Guard). Moreover, it attains the optimal comprehensive scores on stealth detection metrics, demonstrating both high efficacy and exceptional stealth capabilities. The code is available at https://anonymous.4open.science/r/StegoAttack-Jail66
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16674v2">Through the LLM Looking Glass: A Socratic Probing of Donkeys, Elephants, and Markets</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      While detecting and avoiding bias in LLM-generated text is becoming increasingly important, media bias often remains subtle and subjective, making it particularly difficult to identify and mitigate. In this study, we assess media bias in LLM-generated content and LLMs' ability to detect subtle ideological bias. We conduct this evaluation using two datasets, PoliGen and EconoLex, covering political and economic discourse, respectively. We evaluate seven widely used LLMs by prompting them to generate articles and analyze their ideological preferences via Socratic probing. By using our self-contained Socratic approach, the study aims to directly measure the models' biases rather than relying on external interpretations, thereby minimizing subjective judgments about media bias. Our results reveal a consistent preference of Democratic over Republican positions across all models. Conversely, in economic topics, biases vary among Western LLMs, while those developed in China lean more strongly toward socialism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15091v2">ThinkRec: Thinking-based recommendation via LLM</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled more semantic-aware recommendations through natural language generation. Existing LLM for recommendation (LLM4Rec) methods mostly operate in a System 1-like manner, relying on superficial features to match similar items based on click history, rather than reasoning through deeper behavioral logic. This often leads to superficial and erroneous recommendations. Motivated by this, we propose ThinkRec, a thinking-based framework that shifts LLM4Rec from System 1 to System 2 (rational system). Technically, ThinkRec introduces a thinking activation mechanism that augments item metadata with keyword summarization and injects synthetic reasoning traces, guiding the model to form interpretable reasoning chains that consist of analyzing interaction histories, identifying user preferences, and making decisions based on target items. On top of this, we propose an instance-wise expert fusion mechanism to reduce the reasoning difficulty. By dynamically assigning weights to expert models based on users' latent features, ThinkRec adapts its reasoning path to individual users, thereby enhancing precision and personalization. Extensive experiments on real-world datasets demonstrate that ThinkRec significantly improves the accuracy and interpretability of recommendations. Our implementations are available in anonymous Github: https://github.com/Yu-Qi-hang/ThinkRec.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16737v1">Mitigating Fine-tuning Risks in LLMs via Safety-Aware Probing Optimization</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      The significant progress of large language models (LLMs) has led to remarkable achievements across numerous applications. However, their ability to generate harmful content has sparked substantial safety concerns. Despite the implementation of safety alignment techniques during the pre-training phase, recent research indicates that fine-tuning LLMs on adversarial or even benign data can inadvertently compromise their safety. In this paper, we re-examine the fundamental issue of why fine-tuning on non-harmful data still results in safety degradation. We introduce a safety-aware probing (SAP) optimization framework designed to mitigate the safety risks of fine-tuning LLMs. Specifically, SAP incorporates a safety-aware probe into the gradient propagation process, mitigating the model's risk of safety degradation by identifying potential pitfalls in gradient directions, thereby enhancing task-specific performance while successfully preserving model safety. Our extensive experimental results demonstrate that SAP effectively reduces harmfulness below the original fine-tuned model and achieves comparable test loss to standard fine-tuning methods. Our code is available at https://github.com/ChengcanWu/SAP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01822v4">Firewalls to Secure Dynamic LLM Agentic Networks</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      LLM agents will likely communicate on behalf of users with other entity-representing agents on tasks involving long-horizon plans with interdependent goals. Current work neglects these agentic networks and their challenges. We identify required properties for agent communication: proactivity, adaptability, privacy (sharing only task-necessary information), and security (preserving integrity and utility against selfish entities). After demonstrating communication vulnerabilities, we propose a practical design and protocol inspired by network security principles. Our framework automatically derives task-specific rules from prior conversations to build firewalls. These firewalls construct a closed language that is completely controlled by the developer. They transform any personal data to the allowed degree of permissibility entailed by the task. Both operations are completely quarantined from external attackers, disabling the potential for prompt injections, jailbreaks, or manipulation. By incorporating rules learned from their previous mistakes, agents rewrite their instructions and self-correct during communication. Evaluations on diverse attacks demonstrate our framework significantly reduces privacy and security vulnerabilities while allowing adaptability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16710v1">Training Long-Context LLMs Efficiently via Chunk-wise Optimization</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      While long-context large language models (LLMs) exhibit remarkable document processing capabilities, their prohibitively high training costs often hinder customized applications. To mitigate this issue, we propose \textit{Sequential Chunk-wise Optimization} (SeCO), a memory-efficient training paradigm that partitions lengthy inputs into manageable chunks. Each chunk independently constructs its computational graph and performs localized backpropagation, ensuring that only one chunk's forward activations are stored in memory. Building on SeCO, we further introduce \textit{Sparse Chunk-wise Optimization} (SpaCO), which reduces computational overhead by selectively propagating gradients to specific chunks and incorporates a carefully designed compensation factor to ensure unbiased gradient estimation. SpaCO decouples the computational cost of backpropagation from the context length, enabling training time to gradually converge to inference time as sequences become longer. Implemented as lightweight training wrappers, both SeCO and SpaCO offer substantial practical benefits. For example, when fine-tuning an 8B model with LoRA on a single RTX 3090 GPU, SeCO expands maximum sequence length from 1K to 16K tokens, while SpaCO demonstrates accelerated training speed -- achieving up to 3x faster than SeCO under the same experimental setup. These innovations provide new insights into optimizing long-context models, making them more accessible for practical applications. We have open-sourced the code at \href{https://github.com/wenhaoli-xmu/seco}{here}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16703v1">Locate-then-Merge: Neuron-Level Parameter Fusion for Mitigating Catastrophic Forgetting in Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Although multimodal large language models (MLLMs) have achieved impressive performance, the multimodal instruction tuning stage often causes catastrophic forgetting of the base LLM's language ability, even in strong models like Llama3. To address this, we propose Locate-then-Merge, a training-free parameter fusion framework that first locates important parameters and then selectively merges them. We further introduce Neuron-Fusion, a neuron-level strategy that preserves the influence of neurons with large parameter shifts--neurons likely responsible for newly acquired visual capabilities--while attenuating the influence of neurons with smaller changes that likely encode general-purpose language skills. This design enables better retention of visual adaptation while mitigating language degradation. Experiments on 13 benchmarks across both language and visual tasks show that Neuron-Fusion consistently outperforms existing model merging methods. Further analysis reveals that our method effectively reduces context hallucination in generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16697v1">Software Architecture Meets LLMs: A Systematic Literature Review</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are used for many different software engineering tasks. In software architecture, they have been applied to tasks such as classification of design decisions, detection of design patterns, and generation of software architecture design from requirements. However, there is little overview on how well they work, what challenges exist, and what open problems remain. In this paper, we present a systematic literature review on the use of LLMs in software architecture. We analyze 18 research articles to answer five research questions, such as which software architecture tasks LLMs are used for, how much automation they provide, which models and techniques are used, and how these approaches are evaluated. Our findings show that while LLMs are increasingly applied to a variety of software architecture tasks and often outperform baselines, some areas, such as generating source code from architectural design, cloud-native computing and architecture, and checking conformance remain underexplored. Although current approaches mostly use simple prompting techniques, we identify a growing research interest in refining LLM-based approaches by integrating advanced techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16690v1">Your Pre-trained LLM is Secretly an Unsupervised Confidence Calibrator</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Post-training of large language models is essential for adapting pre-trained language models (PLMs) to align with human preferences and downstream tasks. While PLMs typically exhibit well-calibrated confidence, post-trained language models (PoLMs) often suffer from over-confidence, assigning high confidence to both correct and incorrect outputs, which can undermine reliability in critical applications. A major obstacle in calibrating PoLMs is the scarcity of labeled data for individual downstream tasks. To address this, we propose Disagreement-Aware Confidence Alignment (DACA), a novel unsupervised method to optimize the parameters (e.g., temperature $\tau$) in post-hoc confidence calibration. Our method is motivated by the under-confidence issue caused by prediction disagreement between the PLM and PoLM while aligning their confidence via temperature scaling. Theoretically, the PLM's confidence underestimates PoLM's prediction accuracy on disagreement examples, causing a larger $\tau$ and producing under-confident predictions. DACA mitigates this by selectively using only agreement examples for calibration, effectively decoupling the influence of disagreement. In this manner, our method avoids an overly large $\tau$ in temperature scaling caused by disagreement examples, improving calibration performance. Extensive experiments demonstrate the effectiveness of our method, improving the average ECE of open-sourced and API-based LLMs (e.g. GPT-4o) by up to 15.08$\%$ on common benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16667v1">ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 ACL 2025 Main. Our code and dataset are available at https://github.com/SCUNLP/ELABORATION
    </div>
    <details class="paper-abstract">
      While recent research increasingly emphasizes the value of human-LLM collaboration in competitive programming and proposes numerous empirical methods, a comprehensive understanding remains elusive due to the fragmented nature of existing studies and their use of diverse, application-specific human feedback. Thus, our work serves a three-fold purpose: First, we present the first taxonomy of human feedback consolidating the entire programming process, which promotes fine-grained evaluation. Second, we introduce ELABORATIONSET, a novel programming dataset specifically designed for human-LLM collaboration, meticulously annotated to enable large-scale simulated human feedback and facilitate costeffective real human interaction studies. Third, we introduce ELABORATION, a novel benchmark to facilitate a thorough assessment of human-LLM competitive programming. With ELABORATION, we pinpoint strengthes and weaknesses of existing methods, thereby setting the foundation for future improvement. Our code and dataset are available at https://github.com/SCUNLP/ELABORATION
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16646v1">SMART: Self-Generating and Self-Validating Multi-Dimensional Assessment for LLMs' Mathematical Problem Solving</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Large Language Models have achieved remarkable results on a variety of mathematical benchmarks. However, concerns remain as to whether these successes reflect genuine mathematical reasoning or superficial pattern recognition. Common evaluation metrics, such as final answer accuracy, fail to disentangle the underlying competencies involved, offering limited diagnostic value. To address these limitations, we introduce SMART: a Self-Generating and Self-Validating Multi-Dimensional Assessment Framework. SMART decomposes mathematical problem solving into four distinct dimensions: understanding, reasoning, arithmetic, and reflection \& refinement. Each dimension is evaluated independently through tailored tasks, enabling interpretable and fine-grained analysis of LLM behavior. Crucially, SMART integrates an automated self-generating and self-validating mechanism to produce and verify benchmark data, ensuring both scalability and reliability. We apply SMART to 21 state-of-the-art open- and closed-source LLMs, uncovering significant discrepancies in their abilities across different dimensions. Our findings demonstrate the inadequacy of final answer accuracy as a sole metric and motivate a new holistic metric to better capture true problem-solving capabilities. Code and benchmarks will be released upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13865v3">Breaking Information Cocoons: A Hyperbolic Graph-LLM Framework for Exploration and Exploitation in Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Modern recommender systems often create information cocoons, restricting users' exposure to diverse content. A key challenge lies in balancing content exploration and exploitation while allowing users to adjust their recommendation preferences. Intuitively, this balance can be modeled as a tree-structured representation, where depth search facilitates exploitation and breadth search enables exploration. However, existing approaches face two fundamental limitations: Euclidean methods struggle to capture hierarchical structures, while hyperbolic methods, despite their superior hierarchical modeling, lack semantic understanding of user and item profiles and fail to provide a principled mechanism for balancing exploration and exploitation. To address these challenges, we propose HERec, a hyperbolic graph-LLM framework that effectively balances exploration and exploitation in recommender systems. Our framework introduces two key innovations: (1) a semantic-enhanced hierarchical mechanism that aligns rich textual descriptions processed by large language models (LLMs) with collaborative information directly in hyperbolic space, allowing for more nuanced updates that respect the underlying hierarchical structure in user-item profiles; (2) an automatic hierarchical representation by optimizing Dasgupta's cost, which discovers hierarchical structures without requiring predefined hyperparameters, enabling user-adjustable exploration-exploitation trade-offs. Extensive experiments demonstrate that HERec consistently outperforms both Euclidean and hyperbolic baselines, achieving up to 5.49% improvement in utility metrics and 11.39% increase in diversity metrics, effectively mitigating information cocoons. We open-source our model implementation at https://github.com/Martin-qyma/HERec.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02172v2">Identifying Legal Holdings with LLMs: A Systematic Study of Performance, Scale, and Memorization</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 Presented as a short paper at International Conference on Artificial Intelligence and Law 2025 (Chicago, IL)
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to advance in capabilities, it is essential to assess how they perform on established benchmarks. In this study, we present a suite of experiments to assess the performance of modern LLMs (ranging from 3B to 90B+ parameters) on CaseHOLD, a legal benchmark dataset for identifying case holdings. Our experiments demonstrate ``scaling effects'' - performance on this task improves with model size, with more capable models like GPT4o and AmazonNovaPro achieving macro F1 scores of 0.744 and 0.720 respectively. These scores are competitive with the best published results on this dataset, and do not require any technically sophisticated model training, fine-tuning or few-shot prompting. To ensure that these strong results are not due to memorization of judicial opinions contained in the training data, we develop and utilize a novel citation anonymization test that preserves semantic meaning while ensuring case names and citations are fictitious. Models maintain strong performance under these conditions (macro F1 of 0.728), suggesting the performance is not due to rote memorization. These findings demonstrate both the promise and current limitations of LLMs for legal tasks with important implications for the development and measurement of automated legal analytics and legal benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10063v2">Hallucination Detection in LLMs with Topological Divergence on Attention Graphs</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Hallucination, i.e., generating factually incorrect content, remains a critical challenge for large language models (LLMs). We introduce TOHA, a TOpology-based HAllucination detector in the RAG setting, which leverages a topological divergence metric to quantify the structural properties of graphs induced by attention matrices. Examining the topological divergence between prompt and response subgraphs reveals consistent patterns: higher divergence values in specific attention heads correlate with hallucinated outputs, independent of the dataset. Extensive experiments - including evaluation on question answering and summarization tasks - show that our approach achieves state-of-the-art or competitive results on several benchmarks while requiring minimal annotated data and computational resources. Our findings suggest that analyzing the topological structure of attention matrices can serve as an efficient and robust indicator of factual reliability in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10347v3">A Unified Approach to Routing and Cascading for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      The availability of a wide range of large language models (LLMs) embedded in various agentic systems has significantly increased the potential of model selection strategies to improve the cost-performance tradeoff. Existing strategies involve either routing, where a single model is chosen per query, or cascading, which sequentially runs increasingly larger models until a satisfactory answer is found. However, current approaches face three key limitations: they (1) lack formal proofs of optimality, (2) fail to identify the conditions under which these strategies are most effective to improve the cost-performance tradeoff, and (3) are unable to combine both paradigms for further improvements. To address these issues, we first derive a novel optimal strategy for cascading and prove the optimality of an existing routing strategy. Further, we propose cascade routing, a unified framework that integrates routing and cascading into a theoretically optimal strategy. Through our analysis, we identify good quality estimators as the critical factor for the success of model selection paradigms. Finally, in our experiments, we show that cascade routing consistently outperforms the individual approaches by a large margin and we analyze quality estimators to determine when routing and/or cascading are useful paradigms for model selection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16590v1">Beyond LLMs: An Exploration of Small Open-source Language Models in Logging Statement Generation</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Effective software maintenance heavily relies on high-quality logging statements, but manual logging is challenging, error-prone, and insufficiently standardized, often leading to inconsistent log quality. While large language models have shown promise in automatic logging, they introduce concerns regarding privacy, resource intensity, and adaptability to specific enterprise needs. To tackle these limitations, this paper empirically investigates whether Small Open-source Language Models (SOLMs) could become a viable alternative via proper exploitation. Specifically, we conduct a large-scale empirical study on four prominent SOLMs, systematically evaluating the impacts of various interaction strategies, parameter-efficient fine-tuning techniques, model sizes, and model types in automatic logging. Our key findings reveal that Retrieval-Augmented Generation significantly enhances performance, and LoRA is a highly effective PEFT technique. While larger SOLMs tend to perform better, this involves a trade-off with computational resources, and instruct-tuned SOLMs generally surpass their base counterparts. Notably, fine-tuned SOLMs, particularly Qwen2.5-coder-14B, outperformed existing specialized tools and LLM baselines in accurately predicting logging locations and generating high-quality statements, a conclusion supported by traditional evaluation metrics and LLM-as-a-judge evaluations. Furthermore, SOLMs also demonstrated robust generalization across diverse, unseen code repositories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16570v1">URLs Help, Topics Guide: Understanding Metadata Utility in LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are commonly pretrained on vast corpora of text without utilizing contextual metadata such as source, quality, or topic, leading to a context-free learning paradigm. While recent studies suggest that adding metadata like URL information as context (i.e., auxiliary inputs not used in the loss calculation) can improve training efficiency and downstream performance, they offer limited understanding of which types of metadata are truly effective and under what conditions. In this work, we conduct a systematic evaluation and find that not all metadata types contribute equally. Only URL context speeds up training, whereas quality scores and topic/format domain information offer no clear benefit. Furthermore, the improved downstream performances of URL conditioning emerge only when longer prompts are used at inference time. In addition, we demonstrate that context-aware pretraining enables more controllable generation than context-free pretraining, in a classifier-free guidance fashion. Although topic and format metadata do not accelerate training, they are effective for steering outputs, offering human-interpretable control over generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16567v1">Finetuning-Activated Backdoors in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Finetuning openly accessible Large Language Models (LLMs) has become standard practice for achieving task-specific performance improvements. Until now, finetuning has been regarded as a controlled and secure process in which training on benign datasets led to predictable behaviors. In this paper, we demonstrate for the first time that an adversary can create poisoned LLMs that initially appear benign but exhibit malicious behaviors once finetuned by downstream users. To this end, our proposed attack, FAB (Finetuning-Activated Backdoor), poisons an LLM via meta-learning techniques to simulate downstream finetuning, explicitly optimizing for the emergence of malicious behaviors in the finetuned models. At the same time, the poisoned LLM is regularized to retain general capabilities and to exhibit no malicious behaviors prior to finetuning. As a result, when users finetune the seemingly benign model on their own datasets, they unknowingly trigger its hidden backdoor behavior. We demonstrate the effectiveness of FAB across multiple LLMs and three target behaviors: unsolicited advertising, refusal, and jailbreakability. Additionally, we show that FAB-backdoors are robust to various finetuning choices made by the user (e.g., dataset, number of steps, scheduler). Our findings challenge prevailing assumptions about the security of finetuning, revealing yet another critical attack vector exploiting the complexities of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16552v1">Think Silently, Think Fast: Dynamic Latent Compression of LLM Reasoning Chains</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 15 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) achieve superior performance through Chain-of-Thought (CoT) reasoning, but these token-level reasoning chains are computationally expensive and inefficient. In this paper, we introduce Compressed Latent Reasoning (CoLaR), a novel framework that dynamically compresses reasoning processes in latent space through a two-stage training approach. First, during supervised fine-tuning, CoLaR extends beyond next-token prediction by incorporating an auxiliary next compressed embedding prediction objective. This process merges embeddings of consecutive tokens using a compression factor randomly sampled from a predefined range, and trains a specialized latent head to predict distributions of subsequent compressed embeddings. Second, we enhance CoLaR through reinforcement learning (RL) that leverages the latent head's non-deterministic nature to explore diverse reasoning paths and exploit more compact ones. This approach enables CoLaR to: i) perform reasoning at a dense latent level (i.e., silently), substantially reducing reasoning chain length, and ii) dynamically adjust reasoning speed at inference time by simply prompting the desired compression factor. Extensive experiments across four mathematical reasoning datasets demonstrate that CoLaR achieves 14.1% higher accuracy than latent-based baseline methods at comparable compression ratios, and reduces reasoning chain length by 53.3% with only 4.8% performance degradation compared to explicit CoT method. Moreover, when applied to more challenging mathematical reasoning tasks, our RL-enhanced CoLaR demonstrates performance gains of up to 5.4% while dramatically reducing latent reasoning chain length by 82.8%. The code and models will be released upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16530v1">DuFFin: A Dual-Level Fingerprinting Framework for LLMs IP Protection</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are considered valuable Intellectual Properties (IP) for legitimate owners due to the enormous computational cost of training. It is crucial to protect the IP of LLMs from malicious stealing or unauthorized deployment. Despite existing efforts in watermarking and fingerprinting LLMs, these methods either impact the text generation process or are limited in white-box access to the suspect model, making them impractical. Hence, we propose DuFFin, a novel $\textbf{Du}$al-Level $\textbf{Fin}$gerprinting $\textbf{F}$ramework for black-box setting ownership verification. DuFFin extracts the trigger pattern and the knowledge-level fingerprints to identify the source of a suspect model. We conduct experiments on a variety of models collected from the open-source website, including four popular base models as protected LLMs and their fine-tuning, quantization, and safety alignment versions, which are released by large companies, start-ups, and individual users. Results show that our method can accurately verify the copyright of the base protected LLM on their model variants, achieving the IP-ROC metric greater than 0.95. Our code is available at https://github.com/yuliangyan0807/llm-fingerprint.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16522v1">Benchmarking and Pushing the Multi-Bias Elimination Boundary of LLMs via Causal Effect Estimation-guided Debiasing</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Despite significant progress, recent studies have indicated that current large language models (LLMs) may still utilize bias during inference, leading to the poor generalizability of LLMs. Some benchmarks are proposed to investigate the generalizability of LLMs, with each piece of data typically containing one type of controlled bias. However, a single piece of data may contain multiple types of biases in practical applications. To bridge this gap, we propose a multi-bias benchmark where each piece of data contains five types of biases. The evaluations conducted on this benchmark reveal that the performance of existing LLMs and debiasing methods is unsatisfying, highlighting the challenge of eliminating multiple types of biases simultaneously. To overcome this challenge, we propose a causal effect estimation-guided multi-bias elimination method (CMBE). This method first estimates the causal effect of multiple types of biases simultaneously. Subsequently, we eliminate the causal effect of biases from the total causal effect exerted by both the semantic information and biases during inference. Experimental results show that CMBE can effectively eliminate multiple types of bias simultaneously to enhance the generalizability of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16520v1">Are the Hidden States Hiding Something? Testing the Limits of Factuality-Encoding Capabilities in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Factual hallucinations are a major challenge for Large Language Models (LLMs). They undermine reliability and user trust by generating inaccurate or fabricated content. Recent studies suggest that when generating false statements, the internal states of LLMs encode information about truthfulness. However, these studies often rely on synthetic datasets that lack realism, which limits generalization when evaluating the factual accuracy of text generated by the model itself. In this paper, we challenge the findings of previous work by investigating truthfulness encoding capabilities, leading to the generation of a more realistic and challenging dataset. Specifically, we extend previous work by introducing: (1) a strategy for sampling plausible true-false factoid sentences from tabular data and (2) a procedure for generating realistic, LLM-dependent true-false datasets from Question Answering collections. Our analysis of two open-source LLMs reveals that while the findings from previous studies are partially validated, generalization to LLM-generated datasets remains challenging. This study lays the groundwork for future research on factuality in LLMs and offers practical guidelines for more effective evaluation.
    </details>
</div>
