# llm - 2025_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- Part 5
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)

## Papers

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
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18154v1">The Staircase of Ethics: Probing LLM Value Priorities through Multi-Step Induction to Complex Moral Dilemmas</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 25 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Ethical decision-making is a critical aspect of human judgment, and the growing use of LLMs in decision-support systems necessitates a rigorous evaluation of their moral reasoning capabilities. However, existing assessments primarily rely on single-step evaluations, failing to capture how models adapt to evolving ethical challenges. Addressing this gap, we introduce the Multi-step Moral Dilemmas (MMDs), the first dataset specifically constructed to evaluate the evolving moral judgments of LLMs across 3,302 five-stage dilemmas. This framework enables a fine-grained, dynamic analysis of how LLMs adjust their moral reasoning across escalating dilemmas. Our evaluation of nine widely used LLMs reveals that their value preferences shift significantly as dilemmas progress, indicating that models recalibrate moral judgments based on scenario complexity. Furthermore, pairwise value comparisons demonstrate that while LLMs often prioritize the value of care, this value can sometimes be superseded by fairness in certain contexts, highlighting the dynamic and context-dependent nature of LLM ethical reasoning. Our findings call for a shift toward dynamic, context-aware evaluation paradigms, paving the way for more human-aligned and value-sensitive development of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18152v1">Fann or Flop: A Multigenre, Multiera Benchmark for Arabic Poetry Understanding in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 Github:https://github.com/mbzuai-oryx/FannOrFlop, Dataset:https://huggingface.co/datasets/omkarthawakar/FannOrFlop
    </div>
    <details class="paper-abstract">
      Arabic poetry stands as one of the most sophisticated and culturally embedded forms of expression in the Arabic language, known for its layered meanings, stylistic diversity, and deep historical continuity. Although large language models (LLMs) have demonstrated strong performance across languages and tasks, their ability to understand Arabic poetry remains largely unexplored. In this work, we introduce `Fann or Flop`, the first benchmark designed to assess the comprehension of Arabic poetry by LLMs in twelve historical eras, covering 21 core poetic genres and a variety of metrical forms, from classical structures to contemporary free verse. The benchmark comprises a curated corpus of poems with explanations that assess semantic understanding, metaphor interpretation, prosodic awareness, and cultural context. We argue that poetic comprehension offers a strong indicator for testing how good the LLM is in understanding classical Arabic through the Arabic poetry. Unlike surface-level tasks, this domain demands deeper interpretive reasoning and cultural sensitivity. Our evaluation of state-of-the-art LLMs shows that most models struggle with poetic understanding despite strong results on standard Arabic benchmarks. We release `Fann or Flop` along with the evaluation suite as an open-source resource to enable rigorous evaluation and advancement for Arabic language models. Code is available at: https://github.com/mbzuai-oryx/FannOrFlop.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18148v1">Lost in the Haystack: Smaller Needles are More Difficult for LLMs to Find</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) face significant challenges with needle-in-a-haystack tasks, where relevant information ("the needle") must be drawn from a large pool of irrelevant context ("the haystack"). Previous studies have highlighted positional bias and distractor quantity as critical factors affecting model performance, yet the influence of gold context size has received little attention. We address this gap by systematically studying how variations in gold context length impact LLM performance on long-context question answering tasks. Our experiments reveal that LLM performance drops sharply when the gold context is shorter, i.e., smaller gold contexts consistently degrade model performance and amplify positional sensitivity, posing a major challenge for agentic systems that must integrate scattered, fine-grained information of varying lengths. This pattern holds across three diverse domains (general knowledge, biomedical reasoning, and mathematical reasoning) and seven state-of-the-art LLMs of various sizes and architectures. Our work provides clear insights to guide the design of robust, context-aware LLM-driven systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18135v1">Gaming Tool Preferences in Agentic LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can now access a wide range of external tools, thanks to the Model Context Protocol (MCP). This greatly expands their abilities as various agents. However, LLMs rely entirely on the text descriptions of tools to decide which ones to use--a process that is surprisingly fragile. In this work, we expose a vulnerability in prevalent tool/function-calling protocols by investigating a series of edits to tool descriptions, some of which can drastically increase a tool's usage from LLMs when competing with alternatives. Through controlled experiments, we show that tools with properly edited descriptions receive over 10 times more usage from GPT-4.1 and Qwen2.5-7B than tools with original descriptions. We further evaluate how various edits to tool descriptions perform when competing directly with one another and how these trends generalize or differ across a broader set of 10 different models. These phenomenons, while giving developers a powerful way to promote their tools, underscore the need for a more reliable foundation for agentic LLMs to select and utilize tools and resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19614v2">Is Your Paper Being Reviewed by an LLM? Benchmarking AI Text Detection in Peer Review</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      Peer review is a critical process for ensuring the integrity of published scientific research. Confidence in this process is predicated on the assumption that experts in the relevant domain give careful consideration to the merits of manuscripts which are submitted for publication. With the recent rapid advancements in large language models (LLMs), a new risk to the peer review process is that negligent reviewers will rely on LLMs to perform the often time consuming process of reviewing a paper. However, there is a lack of existing resources for benchmarking the detectability of AI text in the domain of peer review. To address this deficiency, we introduce a comprehensive dataset containing a total of 788,984 AI-written peer reviews paired with corresponding human reviews, covering 8 years of papers submitted to each of two leading AI research conferences (ICLR and NeurIPS). We use this new resource to evaluate the ability of 18 existing AI text detection algorithms to distinguish between peer reviews fully written by humans and different state-of-the-art LLMs. Additionally, we explore a context-aware detection method called Anchor, which leverages manuscript content to detect AI-generated reviews, and analyze the sensitivity of detection models to LLM-assisted editing of human-written text. Our work reveals the difficulty of identifying AI-generated text at the individual peer review level, highlighting the urgent need for new tools and methods to detect this unethical use of generative AI. Our dataset is publicly available at: https://huggingface.co/datasets/IntelLabs/AI-Peer-Review-Detection-Benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18110v1">Watch and Listen: Understanding Audio-Visual-Speech Moments with Multimodal LLM</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      Humans naturally understand moments in a video by integrating visual and auditory cues. For example, localizing a scene in the video like "A scientist passionately speaks on wildlife conservation as dramatic orchestral music plays, with the audience nodding and applauding" requires simultaneous processing of visual, audio, and speech signals. However, existing models often struggle to effectively fuse and interpret audio information, limiting their capacity for comprehensive video temporal understanding. To address this, we present TriSense, a triple-modality large language model designed for holistic video temporal understanding through the integration of visual, audio, and speech modalities. Central to TriSense is a Query-Based Connector that adaptively reweights modality contributions based on the input query, enabling robust performance under modality dropout and allowing flexible combinations of available inputs. To support TriSense's multimodal capabilities, we introduce TriSense-2M, a high-quality dataset of over 2 million curated samples generated via an automated pipeline powered by fine-tuned LLMs. TriSense-2M includes long-form videos and diverse modality combinations, facilitating broad generalization. Extensive experiments across multiple benchmarks demonstrate the effectiveness of TriSense and its potential to advance multimodal video analysis. Code and dataset will be publicly released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12397v3">Activated LoRA: Fine-tuned LLMs for Intrinsics</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      Low-Rank Adaptation (LoRA) has emerged as a highly efficient framework for finetuning the weights of large foundation models, and has become the go-to method for data-driven customization of LLMs. Despite the promise of highly customized behaviors and capabilities, switching between relevant LoRAs in a multiturn setting is inefficient, as the key-value (KV) cache of the entire turn history must be recomputed with the LoRA weights before generation can begin. To address this problem, we propose Activated LoRA (aLoRA), an adapter architecture which modifies the LoRA framework to only adapt weights for the tokens in the sequence \emph{after} the aLoRA is invoked. This change crucially allows aLoRA to accept the base model's KV cache of the input string, meaning that aLoRA can be instantly activated whenever needed in a chain without recomputing the cache. This enables building what we call \emph{intrinsics}, i.e. specialized models invoked to perform well-defined operations on portions of an input chain or conversation that otherwise uses the base model by default. We train a set of aLoRA-based intrinsics models, demonstrating competitive accuracy with standard LoRA while achieving significant inference benefits. We include a codebase implementing aLoRA in the supplementary material.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18102v1">How Can I Publish My LLM Benchmark Without Giving the True Answers Away?</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      Publishing a large language model (LLM) benchmark on the Internet risks contaminating future LLMs: the benchmark may be unintentionally (or intentionally) used to train or select a model. A common mitigation is to keep the benchmark private and let participants submit their models or predictions to the organizers. However, this strategy will require trust in a single organization and still permits test-set overfitting through repeated queries. To overcome this issue, we propose a way to publish benchmarks without completely disclosing the ground-truth answers to the questions, while still maintaining the ability to openly evaluate LLMs. Our main idea is to inject randomness to the answers by preparing several logically correct answers, and only include one of them as the solution in the benchmark. This reduces the best possible accuracy, i.e., Bayes accuracy, of the benchmark. Not only is this helpful to keep us from disclosing the ground truth, but this approach also offers a test for detecting data contamination. In principle, even fully capable models should not surpass the Bayes accuracy. If a model surpasses this ceiling despite this expectation, this is a strong signal of data contamination. We present experimental evidence that our method can detect data contamination accurately on a wide range of benchmarks, models, and training methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18098v1">Planning without Search: Refining Frontier LLMs with Offline Goal-Conditioned RL</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 18 pages, 4 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel in tasks like question answering and dialogue, but complex tasks requiring interaction, such as negotiation and persuasion, require additional long-horizon reasoning and planning. Reinforcement learning (RL) fine-tuning can enable such planning in principle, but suffers from drawbacks that hinder scalability. In particular, multi-turn RL training incurs high memory and computational costs, which are exacerbated when training LLMs as policies. Furthermore, the largest LLMs do not expose the APIs necessary to be trained in such manner. As a result, modern methods to improve the reasoning of LLMs rely on sophisticated prompting mechanisms rather than RL fine-tuning. To remedy this, we propose a novel approach that uses goal-conditioned value functions to guide the reasoning of LLM agents, that scales even to large API-based models. These value functions predict how a task will unfold given an action, allowing the LLM agent to evaluate multiple possible outcomes, both positive and negative, to plan effectively. In addition, these value functions are trained over reasoning steps rather than full actions, to be a concise and light-weight module that facilitates decision-making in multi-turn interactions. We validate our method on tasks requiring interaction, including tool use, social deduction, and dialogue, demonstrating superior performance over both RL fine-tuning and prompting methods while maintaining efficiency and scalability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06261v3">Hogwild! Inference: Parallel LLM Generation via Concurrent Attention</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated the ability to tackle increasingly complex tasks through advanced reasoning, long-form content generation, and tool use. Solving these tasks often involves long inference-time computations. In human problem solving, a common strategy to expedite work is collaboration: by dividing the problem into sub-tasks, exploring different strategies concurrently, etc. Recent research has shown that LLMs can also operate in parallel by implementing explicit cooperation frameworks, such as voting mechanisms or the explicit creation of independent sub-tasks that can be executed in parallel. However, each of these frameworks may not be suitable for all types of tasks, which can hinder their applicability. In this work, we propose a different design approach: we run LLM "workers" in parallel , allowing them to synchronize via a concurrently-updated attention cache and prompt these workers to decide how best to collaborate. Our approach allows the LLM instances to come up with their own collaboration strategy for the problem at hand, all the while "seeing" each other's memory in the concurrent KV cache. We implement this approach via Hogwild! Inference: a parallel LLM inference engine where multiple instances of the same LLM run in parallel with the same attention cache, with "instant" access to each other's memory. Hogwild! Inference takes advantage of Rotary Position Embeddings (RoPE) to avoid recomputation while improving parallel hardware utilization. We find that modern reasoning-capable LLMs can perform inference with shared Key-Value cache out of the box, without additional fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16023v2">Prototypical Human-AI Collaboration Behaviors from LLM-Assisted Writing in the Wild</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 Pre-print under-review
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are used in complex writing workflows, users engage in multi-turn interactions to steer generations to better fit their needs. Rather than passively accepting output, users actively refine, explore, and co-construct text. We conduct a large-scale analysis of this collaborative behavior for users engaged in writing tasks in the wild with two popular AI assistants, Bing Copilot and WildChat. Our analysis goes beyond simple task classification or satisfaction estimation common in prior work and instead characterizes how users interact with LLMs through the course of a session. We identify prototypical behaviors in how users interact with LLMs in prompts following their original request. We refer to these as Prototypical Human-AI Collaboration Behaviors (PATHs) and find that a small group of PATHs explain a majority of the variation seen in user-LLM interaction. These PATHs span users revising intents, exploring texts, posing questions, adjusting style or injecting new content. Next, we find statistically significant correlations between specific writing intents and PATHs, revealing how users' intents shape their collaboration behaviors. We conclude by discussing the implications of our findings on LLM alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18040v1">Contrastive Distillation of Emotion Knowledge from LLMs for Zero-Shot Emotion Recognition</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      The ability to handle various emotion labels without dedicated training is crucial for building adaptable Emotion Recognition (ER) systems. Conventional ER models rely on training using fixed label sets and struggle to generalize beyond them. On the other hand, Large Language Models (LLMs) have shown strong zero-shot ER performance across diverse label spaces, but their scale limits their use on edge devices. In this work, we propose a contrastive distillation framework that transfers rich emotional knowledge from LLMs into a compact model without the use of human annotations. We use GPT-4 to generate descriptive emotion annotations, offering rich supervision beyond fixed label sets. By aligning text samples with emotion descriptors in a shared embedding space, our method enables zero-shot prediction on different emotion classes, granularity, and label schema. The distilled model is effective across multiple datasets and label spaces, outperforming strong baselines of similar size and approaching GPT-4's zero-shot performance, while being over 10,000 times smaller.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18034v1">Structured Thinking Matters: Improving LLMs Generalization in Causal Inference Tasks</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      Despite remarkable advances in the field, LLMs remain unreliable in distinguishing causation from correlation. Recent results from the Corr2Cause dataset benchmark reveal that state-of-the-art LLMs -- such as GPT-4 (F1 score: 29.08) -- only marginally outperform random baselines (Random Uniform, F1 score: 20.38), indicating limited capacity of generalization. To tackle this limitation, we propose a novel structured approach: rather than directly answering causal queries, we provide the model with the capability to structure its thinking by guiding the model to build a structured knowledge graph, systematically encoding the provided correlational premises, to answer the causal queries. This intermediate representation significantly enhances the model's causal capabilities. Experiments on the test subset of the Corr2Cause dataset benchmark with Qwen3-32B model (reasoning model) show substantial gains over standard direct prompting methods, improving F1 scores from 32.71 to 48.26 (over 47.5% relative increase), along with notable improvements in precision and recall. These results underscore the effectiveness of providing the model with the capability to structure its thinking and highlight its promising potential for broader generalization across diverse causal inference tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23128v2">CrossMuSim: A Cross-Modal Framework for Music Similarity Retrieval with LLM-Powered Text Description Sourcing and Mining</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 Accepted by ICME2025
    </div>
    <details class="paper-abstract">
      Music similarity retrieval is fundamental for managing and exploring relevant content from large collections in streaming platforms. This paper presents a novel cross-modal contrastive learning framework that leverages the open-ended nature of text descriptions to guide music similarity modeling, addressing the limitations of traditional uni-modal approaches in capturing complex musical relationships. To overcome the scarcity of high-quality text-music paired data, this paper introduces a dual-source data acquisition approach combining online scraping and LLM-based prompting, where carefully designed prompts leverage LLMs' comprehensive music knowledge to generate contextually rich descriptions. Exten1sive experiments demonstrate that the proposed framework achieves significant performance improvements over existing benchmarks through objective metrics, subjective evaluations, and real-world A/B testing on the Huawei Music streaming platform.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18019v1">LLM assisted web application functional requirements generation: A case study of four popular LLMs over a Mess Management System</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 11 pages, 12 figures, Accepted in EASE 2025 https://conf.researchr.org/details/ease-2025/ease-2025-ai-models---data/11/LLM-assisted-web-application-functional-requirements-generation-A-case-study-of-fou
    </div>
    <details class="paper-abstract">
      Like any other discipline, Large Language Models (LLMs) have significantly impacted software engineering by helping developers generate the required artifacts across various phases of software development. This paper presents a case study comparing the performance of popular LLMs GPT, Claude, Gemini, and DeepSeek in generating functional specifications that include use cases, business rules, and collaborative workflows for a web application, the Mess Management System. The study evaluated the quality of LLM generated use cases, business rules, and collaborative workflows in terms of their syntactic and semantic correctness, consistency, non ambiguity, and completeness compared to the reference specifications against the zero-shot prompted problem statement. Our results suggested that all four LLMs can specify syntactically and semantically correct, mostly non-ambiguous artifacts. Still, they may be inconsistent at times and may differ significantly in the completeness of the generated specification. Claude and Gemini generated all the reference use cases, with Claude achieving the most complete but somewhat redundant use case specifications. Similar results were obtained for specifying workflows. However, all four LLMs struggled to generate relevant Business Rules, with DeepSeek generating the most reference rules but with less completeness. Overall, Claude generated more complete specification artifacts, while Gemini was more precise in the specifications it generated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17977v1">SmartNote: An LLM-Powered, Personalised Release Note Generator That Just Works</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 In Proceedings of the ACM International Conference on the Foundations of Software Engineering (FSE) (FSE 2025)
    </div>
    <details class="paper-abstract">
      The release note is a crucial document outlining changes in new software versions. Yet, many developers view the process of writing software release notes as a tedious and dreadful task. Consequently, numerous tools have been developed by researchers and practitioners to automate the generation of software release notes. However, these tools fail to consider project domain and target audience for personalisation, limiting their relevance and conciseness. Additionally, they suffer from limited applicability, often necessitating significant workflow adjustments and adoption efforts, hindering practical use and stressing developers. Despite recent advancements in natural language processing and the proven capabilities of large language models in various code and text-related tasks, there are no existing studies investigating the integration and utilisation of LLMs in automated release note generation. Therefore, we propose SmartNote, a novel and widely applicable release note generation approach that produces high-quality, contextually personalised release notes using LLM technology. SmartNote aggregates changes and uses an LLM to describe and summarise the changes using code, commit, and pull request details. It categorises and scores commits to generate structured and concise release notes of prioritised changes. Our human and automatic evaluations reveal that SmartNote outperforms or achieves comparable performance to DeepRelease, Conventional Changelog, and the projects'original release notes across four quality metrics: completeness, clarity, conciseness, and organisation. In both evaluations, SmartNote ranked first for completeness and organisation, while clarity ranked first in the human evaluation. A further evaluation demonstrates that SmartNote is effective in terms of context awareness and applicability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17952v1">Beyond Distillation: Pushing the Limits of Medical LLM Reasoning with Minimalist Rule-Based RL</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Improving performance on complex tasks and enabling interpretable decision making in large language models (LLMs), especially for clinical applications, requires effective reasoning. Yet this remains challenging without supervised fine-tuning (SFT) on costly chain-of-thought (CoT) data distilled from closed-source models (e.g., GPT-4o). In this work, we present AlphaMed, the first medical LLM to show that reasoning capability can emerge purely through reinforcement learning (RL), using minimalist rule-based rewards on public multiple-choice QA datasets, without relying on SFT or distilled CoT data. AlphaMed achieves state-of-the-art results on six medical QA benchmarks, outperforming models trained with conventional SFT+RL pipelines. On challenging benchmarks (e.g., MedXpert), AlphaMed even surpasses larger or closed-source models such as DeepSeek-V3-671B and Claude-3.5-Sonnet. To understand the factors behind this success, we conduct a comprehensive data-centric analysis guided by three questions: (i) Can minimalist rule-based RL incentivize reasoning without distilled CoT supervision? (ii) How do dataset quantity and diversity impact reasoning? (iii) How does question difficulty shape the emergence and generalization of reasoning? Our findings show that dataset informativeness is a key driver of reasoning performance, and that minimalist RL on informative, multiple-choice QA data is effective at inducing reasoning without CoT supervision. We also observe divergent trends across benchmarks, underscoring limitations in current evaluation and the need for more challenging, reasoning-oriented medical QA benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14645v2">Edit Once, Update Everywhere: A Simple Framework for Cross-Lingual Knowledge Synchronization in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Knowledge editing allows for efficient adaptation of large language models (LLMs) to new information or corrections without requiring full retraining. However, prior methods typically focus on either single-language editing or basic multilingual editing, failing to achieve true cross-linguistic knowledge synchronization. To address this, we present a simple and practical state-of-the-art (SOTA) recipe Cross-Lingual Knowledge Democracy Edit (X-KDE), designed to propagate knowledge from a dominant language to other languages effectively. Our X-KDE comprises two stages: (i) Cross-lingual Edition Instruction Tuning (XE-IT), which fine-tunes the model on a curated parallel dataset to modify in-scope knowledge while preserving unrelated information, and (ii) Target-language Preference Optimization (TL-PO), which applies advanced optimization techniques to ensure consistency across languages, fostering the transfer of updates. Additionally, we contribute a high-quality, cross-lingual dataset, specifically designed to enhance knowledge transfer across languages. Extensive experiments on the Bi-ZsRE and MzsRE benchmarks show that X-KDE significantly enhances cross-lingual performance, achieving an average improvement of +8.19%, while maintaining high accuracy in monolingual settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17937v1">Survival Games: Human-LLM Strategic Showdowns under Severe Resource Scarcity</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) raises critical concerns about their ethical alignment, particularly in scenarios where human and AI co-exist under the conflict of interest. This work introduces an extendable, asymmetric, multi-agent simulation-based benchmarking framework to evaluate the moral behavior of LLMs in a novel human-AI co-existence setting featuring consistent living and critical resource management. Building on previous generative agent environments, we incorporate a life-sustaining system, where agents must compete or cooperate for food resources to survive, often leading to ethically charged decisions such as deception, theft, or social influence. We evaluated two types of LLM, DeepSeek and OpenAI series, in a three-agent setup (two humans, one LLM-powered robot), using adapted behavioral detection from the MACHIAVELLI framework and a custom survival-based ethics metric. Our findings reveal stark behavioral differences: DeepSeek frequently engages in resource hoarding, while OpenAI exhibits restraint, highlighting the influence of model design on ethical outcomes. Additionally, we demonstrate that prompt engineering can significantly steer LLM behavior, with jailbreaking prompts significantly enhancing unethical actions, even for highly restricted OpenAI models and cooperative prompts show a marked reduction in unethical actions. Our framework provides a reproducible testbed for quantifying LLM ethics in high-stakes scenarios, offering insights into their suitability for real-world human-AI interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17918v1">LLM Meeting Decision Trees on Tabular Data</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      Tabular data have been playing a vital role in diverse real-world fields, including healthcare, finance, etc. With the recent success of Large Language Models (LLMs), early explorations of extending LLMs to the domain of tabular data have been developed. Most of these LLM-based methods typically first serialize tabular data into natural language descriptions, and then tune LLMs or directly infer on these serialized data. However, these methods suffer from two key inherent issues: (i) data perspective: existing data serialization methods lack universal applicability for structured tabular data, and may pose privacy risks through direct textual exposure, and (ii) model perspective: LLM fine-tuning methods struggle with tabular data, and in-context learning scalability is bottle-necked by input length constraints (suitable for few-shot learning). This work explores a novel direction of integrating LLMs into tabular data throughough logical decision tree rules as intermediaries, proposes a decision tree enhancer with LLM-derived rule for tabular prediction, DeLTa. The proposed DeLTa avoids tabular data serialization, and can be applied to full data learning setting without LLM fine-tuning. Specifically, we leverage the reasoning ability of LLMs to redesign an improved rule given a set of decision tree rules. Furthermore, we provide a calibration method for original decision trees via new generated rule by LLM, which approximates the error correction vector to steer the original decision tree predictions in the direction of ``errors'' reducing. Finally, extensive experiments on diverse tabular benchmarks show that our method achieves state-of-the-art performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.08498v6">"Reasoning" with Rhetoric: On the Style-Evidence Tradeoff in LLM-Generated Counter-Arguments</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 24 pages, 9 figures, 13 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) play a key role in generating evidence-based and stylistic counter-arguments, yet their effectiveness in real-world applications has been underexplored. Previous research often neglects the balance between evidentiality and style, which are crucial for persuasive arguments. To address this, we evaluated the effectiveness of stylized evidence-based counter-argument generation in Counterfire, a new dataset of 38,000 counter-arguments generated by revising counter-arguments to Reddit's ChangeMyView community to follow different discursive styles. We evaluated generic and stylized counter-arguments from basic and fine-tuned models such as GPT-3.5, PaLM-2, and Koala-13B, as well as newer models (GPT-4o, Claude Haiku, LLaMA-3.1) focusing on rhetorical quality and persuasiveness. Our findings reveals that humans prefer stylized counter-arguments over the original outputs, with GPT-3.5 Turbo performing well, though still not reaching human standards of rhetorical quality nor persuasiveness indicating a persisting style-evidence tradeoff in counter-argument generation by LLMs. We conclude with an examination of ethical considerations in LLM persuasion research, addressing potential risks of deceptive practices and the need for transparent deployment methodologies to safeguard against misuse in public discourse. The code and dataset are available at https://github.com/Preetika764/Style_control/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02324v2">From Course to Skill: Evaluating LLM Performance in Curricular Analytics</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      Curricular analytics (CA) -- systematic analysis of curricula data to inform program and course refinement -- becomes an increasingly valuable tool to help institutions align academic offerings with evolving societal and economic demands. Large language models (LLMs) are promising for handling large-scale, unstructured curriculum data, but it remains uncertain how reliably LLMs can perform CA tasks. In this paper, we systematically evaluate four text alignment strategies based on LLMs or traditional NLP methods for skill extraction, a core task in CA. Using a stratified sample of 400 curriculum documents of different types and a human-LLM collaborative evaluation framework, we find that retrieval-augmented generation (RAG) is the top-performing strategy across all types of curriculum documents, while zero-shot prompting performs worse than traditional NLP methods in most cases. Our findings highlight the promise of LLMs in analyzing brief and abstract curriculum documents, but also reveal that their performance can vary significantly depending on model selection and prompting strategies. This underscores the importance of carefully evaluating the performance of LLM-based strategies before large-scale deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17829v1">Stepwise Reasoning Checkpoint Analysis: A Test Time Scaling Method to Enhance LLMs' Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      Mathematical reasoning through Chain-of-Thought (CoT) has emerged as a powerful capability of Large Language Models (LLMs), which can be further enhanced through Test-Time Scaling (TTS) methods like Beam Search and DVTS. However, these methods, despite improving accuracy by allocating more computational resources during inference, often suffer from path homogenization and inefficient use of intermediate results. To address these limitations, we propose Stepwise Reasoning Checkpoint Analysis (SRCA), a framework that introduces checkpoints between reasoning steps. It incorporates two key strategies: (1) Answer-Clustered Search, which groups reasoning paths by their intermediate checkpoint answers to maintain diversity while ensuring quality, and (2) Checkpoint Candidate Augmentation, which leverages all intermediate answers for final decision-making. Our approach effectively reduces path homogenization and creates a fault-tolerant mechanism by utilizing high-quality intermediate results. Experimental results show that SRCA improves reasoning accuracy compared to existing TTS methods across various mathematical datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17813v1">Don't Overthink it. Preferring Shorter Thinking Chains for Improved LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 Preprint. Under review
    </div>
    <details class="paper-abstract">
      Reasoning large language models (LLMs) heavily rely on scaling test-time compute to perform complex reasoning tasks by generating extensive "thinking" chains. While demonstrating impressive results, this approach incurs significant computational costs and inference time. In this work, we challenge the assumption that long thinking chains results in better reasoning capabilities. We first demonstrate that shorter reasoning chains within individual questions are significantly more likely to yield correct answers - up to 34.5% more accurate than the longest chain sampled for the same question. Based on these results, we suggest short-m@k, a novel reasoning LLM inference method. Our method executes k independent generations in parallel and halts computation once the first m thinking processes are done. The final answer is chosen using majority voting among these m chains. Basic short-1@k demonstrates similar or even superior performance over standard majority voting in low-compute settings - using up to 40% fewer thinking tokens. short-3@k, while slightly less efficient than short-1@k, consistently surpasses majority voting across all compute budgets, while still being substantially faster (up to 33% wall time reduction). Inspired by our results, we finetune an LLM using short, long, and randomly selected reasoning chains. We then observe that training on the shorter ones leads to better performance. Our findings suggest rethinking current methods of test-time compute in reasoning LLMs, emphasizing that longer "thinking" does not necessarily translate to improved performance and can, counter-intuitively, lead to degraded results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14359v2">Triangulating LLM Progress through Benchmarks, Games, and Cognitive Tests</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      We examine three evaluation paradigms: standard benchmarks (e.g., MMLU and BBH), interactive games (e.g., Signalling Games or Taboo), and cognitive tests (e.g., for working memory or theory of mind). First, we investigate which of the former two-benchmarks or games-is most effective at discriminating LLMs of varying quality. Then, inspired by human cognitive assessments, we compile a suite of targeted tests that measure cognitive abilities deemed essential for effective language use, and we investigate their correlation with model performance in benchmarks and games. Our analyses reveal that interactive games are superior to standard benchmarks in discriminating models. Causal and logical reasoning correlate with both static and interactive tests, while differences emerge regarding core executive functions and social/emotional skills, which correlate more with games. We advocate for the development of new interactive benchmarks and targeted cognitive tasks inspired by assessing human abilities but designed specifically for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17795v1">DialogXpert: Driving Intelligent and Emotion-Aware Conversations through Online Value-Based Reinforcement Learning with LLM Priors</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      Large-language-model (LLM) agents excel at reactive dialogue but struggle with proactive, goal-driven interactions due to myopic decoding and costly planning. We introduce DialogXpert, which leverages a frozen LLM to propose a small, high-quality set of candidate actions per turn and employs a compact Q-network over fixed BERT embeddings trained via temporal-difference learning to select optimal moves within this reduced space. By tracking the user's emotions, DialogXpert tailors each decision to advance the task while nurturing a genuine, empathetic connection. Across negotiation, emotional support, and tutoring benchmarks, DialogXpert drives conversations to under $3$ turns with success rates exceeding 94\% and, with a larger LLM prior, pushes success above 97\% while markedly improving negotiation outcomes. This framework delivers real-time, strategic, and emotionally intelligent dialogue planning at scale. Code available at https://github.com/declare-lab/dialogxpert/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17794v1">RECIPE-TKG: From Sparse History to Structured Reasoning for LLM-based Temporal Knowledge Graph Completion</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      Temporal Knowledge Graphs (TKGs) represent dynamic facts as timestamped relations between entities. TKG completion involves forecasting missing or future links, requiring models to reason over time-evolving structure. While LLMs show promise for this task, existing approaches often overemphasize supervised fine-tuning and struggle particularly when historical evidence is limited or missing. We introduce RECIPE-TKG, a lightweight and data-efficient framework designed to improve accuracy and generalization in settings with sparse historical context. It combines (1) rule-based multi-hop retrieval for structurally diverse history, (2) contrastive fine-tuning of lightweight adapters to encode relational semantics, and (3) test-time semantic filtering to iteratively refine generations based on embedding similarity. Experiments on four TKG benchmarks show that RECIPE-TKG outperforms previous LLM-based approaches, achieving up to 30.6\% relative improvement in Hits@10. Moreover, our proposed framework produces more semantically coherent predictions, even for the samples with limited historical context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17787v1">Titanus: Enabling KV Cache Pruning and Quantization On-the-Fly for LLM Acceleration</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 Accepted to GLSVLSI 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have gained great success in various domains. Existing systems cache Key and Value within the attention block to avoid redundant computations. However, the size of key-value cache (KV cache) is unpredictable and can even be tens of times larger than the weights in the long context length scenario. In this work, we propose Titanus, a software-hardware co-design to efficiently compress the KV cache on-the-fly. We first propose the cascade pruning-quantization (CPQ) method to reduce the KV cache movement. The hierarchical quantization extension strategy is introduced to tackle the non-independent per-channel quantization issue. To further reduce KV cache movement, we transfer only the non-zero KV cache between the accelerator and off-chip memory. Moreover, we customize a two-stage design space exploration framework for the CPQ method. A novel pipeline and parallelism dataflow is designed to reduce the first token generation time. Experiments show that Titanus achieves 159.9x (49.6x) and 34.8x (29.2x) energy efficiency (throughput) compared to Nvidia A100 GPU and FlightLLM respectively. The code for Titanus is available at https://github.com/peilin-chen/Titanus-for-LLM-acceleration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17784v1">EXECUTE: A Multilingual Benchmark for LLM Token Understanding</a></div>
    <div class="paper-meta">
      📅 2025-05-23
      | 💬 Accepted to Findings of ACL 2025
    </div>
    <details class="paper-abstract">
      The CUTE benchmark showed that LLMs struggle with character understanding in English. We extend it to more languages with diverse scripts and writing systems, introducing EXECUTE. Our simplified framework allows easy expansion to any language. Tests across multiple LLMs reveal that challenges in other languages are not always on the character level as in English. Some languages show word-level processing issues, some show no issues at all. We also examine sub-character tasks in Chinese, Japanese, and Korean to assess LLMs' understanding of character components.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17767v1">The Real Barrier to LLM Agent Usability is Agentic ROI</a></div>
    <div class="paper-meta">
      📅 2025-05-23
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents represent a promising shift in human-AI interaction, moving beyond passive prompt-response systems to autonomous agents capable of reasoning, planning, and goal-directed action. Despite the widespread application in specialized, high-effort tasks like coding and scientific research, we highlight a critical usability gap in high-demand, mass-market applications. This position paper argues that the limited real-world adoption of LLM agents stems not only from gaps in model capabilities, but also from a fundamental tradeoff between the value an agent can provide and the costs incurred during real-world use. Hence, we call for a shift from solely optimizing model performance to a broader, utility-driven perspective: evaluating agents through the lens of the overall agentic return on investment (Agent ROI). By identifying key factors that determine Agentic ROI--information quality, agent time, and cost--we posit a zigzag development trajectory in optimizing agentic ROI: first scaling up to improve the information quality, then scaling down to minimize the time and cost. We outline the roadmap across different development stages to bridge the current usability gaps, aiming to make LLM agents truly scalable, accessible, and effective in real-world contexts.
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
