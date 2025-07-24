# llm - 2025_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- Part 6
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07145v1">CCQ: Convolutional Code for Extreme Low-bit Quantization in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 11 pages, 3 figures
    </div>
    <details class="paper-abstract">
      The rapid scaling of Large Language Models (LLMs) elevates inference costs and compounds substantial deployment barriers. While quantization to 8 or 4 bits mitigates this, sub-3-bit methods face severe accuracy, scalability, and efficiency degradation. We propose Convolutional Code Quantization (CCQ), an inference-optimized quantization approach compressing LLMs to 2.0-2.75 bits with minimal accuracy loss. Departing from error-prone scalar quantization or slow vector quantization, CCQ integrates a hardware-aware bit-shift encoding and decoding solution with Convolutional Code, Hybrid Encoding, and Code Cluster, jointly overcoming accuracy-speed bottlenecks. We construct a lookup-free encoding space, enabling a linear mapping between the codebook and weight vectors, thereby optimizing inference performance. Meanwhile, by drawing on the concept of data mapping from vector quantization, we minimize the performance degradation of the model under extremely low-bit conditions. Experiments demonstrate that CCQ achieves outstanding performance on LLMs across various benchmarks. We compress DeepSeek-V3 (671B total parameters) to 184GB and ERNIE-4.5-300B-A47B to 89GB, enabling single-GPU deployment of ERNIE 4.5 and eliminating inter-card communication. The 2-bit ERNIE-4.5-300B-A47B model and inference engine have been open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04133v3">TRiSM for Agentic AI: A Review of Trust, Risk, and Security Management in LLM-based Agentic Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Agentic AI systems, built upon large language models (LLMs) and deployed in multi-agent configurations, are redefining intelligence, autonomy, collaboration, and decision-making across enterprise and societal domains. This review presents a structured analysis of \textbf{Trust, Risk, and Security Management (TRiSM)} in the context of LLM-based Agentic Multi-Agent Systems (AMAS). We begin by examining the conceptual foundations of Agentic AI and highlight its architectural distinctions from traditional AI agents. We then adapt and extend the AI TRiSM framework for Agentic AI, structured around four key pillars: Explainability, ModelOps, Security, Privacy and Governance, each contextualized to the challenges of multi-agent LLM systems. A novel risk taxonomy is proposed to capture the unique threats and vulnerabilities of Agentic AI, ranging from coordination failures to prompt-based adversarial manipulation. To support practical assessment in Agentic AI works, we introduce two novel metrics: the Component Synergy Score (CSS), which quantifies the quality of inter-agent collaboration, and the Tool Utilization Efficacy (TUE), which evaluates the efficiency of tool use within agent workflows. We further discuss strategies for improving explainability in Agentic AI , as well as approaches to enhancing security and privacy through encryption, adversarial robustness, and regulatory compliance. The review concludes with a research roadmap for the responsible development and deployment of Agentic AI, outlining critical directions to align emerging systems with TRiSM principles for safe, transparent, and accountable operation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07064v1">Boosting Parameter Efficiency in LLM-Based Recommendation through Sophisticated Pruning</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      LLM-based recommender systems have made significant progress; however, the deployment cost associated with the large parameter volume of LLMs still hinders their real-world applications. This work explores parameter pruning to improve parameter efficiency while maintaining recommendation quality, thereby enabling easier deployment. Unlike existing approaches that focus primarily on inter-layer redundancy, we uncover intra-layer redundancy within components such as self-attention and MLP modules. Building on this analysis, we propose a more fine-grained pruning approach that integrates both intra-layer and layer-wise pruning. Specifically, we introduce a three-stage pruning strategy that progressively prunes parameters at different levels and parts of the model, moving from intra-layer to layer-wise pruning, or from width to depth. Each stage also includes a performance restoration step using distillation techniques, helping to strike a balance between performance and parameter efficiency. Empirical results demonstrate the effectiveness of our approach: across three datasets, our models achieve an average of 88% of the original model's performance while pruning more than 95% of the non-embedding parameters. This underscores the potential of our method to significantly reduce resource requirements without greatly compromising recommendation quality. Our code will be available at: https://github.com/zheng-sl/PruneRec
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07045v1">5C Prompt Contracts: A Minimalist, Creative-Friendly, Token-Efficient Design Framework for Individual and SME LLM Usage</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 5 pages, 5 tables. Includes comparative experimental results across OpenAI, Anthropic, DeepSeek, and Gemini LLMs
    </div>
    <details class="paper-abstract">
      The progression from traditional prompt engineering to a more rigorous discipline of prompt design marks a pivotal shift in human-LLM interaction. As Large Language Models (LLMs) become increasingly embedded in mission-critical applications, there emerges a pressing need for frameworks that are not only explicit and systematic but also minimal enough to remain practical and broadly accessible. While many existing approaches address prompt structuring through elaborate Domain-Specific Languages (DSLs) or multi-layered templates, such methods can impose significant token and cognitive overhead, potentially constraining the model's creative capacity. In this context, we propose the 5C Prompt Contract, a framework that distills prompt design into five intuitive components: Character, Cause, Constraint, Contingency, and Calibration. This minimal cognitive schema explicitly integrates fallback and output optimization directives, fostering reliable, interpretable, and creatively flexible AI interactions. Experimental results demonstrate that the 5C framework consistently achieves superior input token efficiency while maintaining rich and consistent outputs across diverse LLM architectures (OpenAI, Anthropic, DeepSeek, and Gemini), making it particularly suited for individuals and Small-to-Medium Enterprises (SMEs) with limited AI engineering resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06999v1">Learning Deliberately, Acting Intuitively: Unlocking Test-Time Reasoning in Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Reasoning is a key capability for large language models (LLMs), particularly when applied to complex tasks such as mathematical problem solving. However, multimodal reasoning research still requires further exploration of modality alignment and training costs. Many of these approaches rely on additional data annotation and relevant rule-based rewards to enhance the understanding and reasoning ability, which significantly increases training costs and limits scalability. To address these challenges, we propose the Deliberate-to-Intuitive reasoning framework (D2I) that improves the understanding and reasoning ability of multimodal LLMs (MLLMs) without extra annotations and complex rewards. Specifically, our method sets deliberate reasoning strategies to enhance modality alignment only through the rule-based format reward during training. While evaluating, the reasoning style shifts to intuitive, which removes deliberate reasoning strategies during training and implicitly reflects the model's acquired abilities in the response. D2I outperforms baselines across both in-domain and out-of-domain benchmarks. Our findings highlight the role of format reward in fostering transferable reasoning skills in MLLMs, and inspire directions for decoupling training-time reasoning depth from test-time response flexibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06992v1">MCA-RG: Enhancing LLMs with Medical Concept Alignment for Radiology Report Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 MICCAI 2025
    </div>
    <details class="paper-abstract">
      Despite significant advancements in adapting Large Language Models (LLMs) for radiology report generation (RRG), clinical adoption remains challenging due to difficulties in accurately mapping pathological and anatomical features to their corresponding text descriptions. Additionally, semantic agnostic feature extraction further hampers the generation of accurate diagnostic reports. To address these challenges, we introduce Medical Concept Aligned Radiology Report Generation (MCA-RG), a knowledge-driven framework that explicitly aligns visual features with distinct medical concepts to enhance the report generation process. MCA-RG utilizes two curated concept banks: a pathology bank containing lesion-related knowledge, and an anatomy bank with anatomical descriptions. The visual features are aligned with these medical concepts and undergo tailored enhancement. We further propose an anatomy-based contrastive learning procedure to improve the generalization of anatomical features, coupled with a matching loss for pathological features to prioritize clinically relevant regions. Additionally, a feature gating mechanism is employed to filter out low-quality concept features. Finally, the visual features are corresponding to individual medical concepts, and are leveraged to guide the report generation process. Experiments on two public benchmarks (MIMIC-CXR and CheXpert Plus) demonstrate that MCA-RG achieves superior performance, highlighting its effectiveness in radiology report generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12112v3">Planning Anything with Rigor: General-Purpose Zero-Shot Planning with LLM-based Formalized Programming</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 57 pages, 25 figures, 15 tables
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have recently demonstrated strong potential in solving planning problems, there is a trade-off between flexibility and complexity. LLMs, as zero-shot planners themselves, are still not capable of directly generating valid plans for complex planning problems such as multi-constraint or long-horizon tasks. On the other hand, many frameworks aiming to solve complex planning problems often rely on task-specific preparatory efforts, such as task-specific in-context examples and pre-defined critics/verifiers, which limits their cross-task generalization capability. In this paper, we tackle these challenges by observing that the core of many planning problems lies in optimization problems: searching for the optimal solution (best plan) with goals subject to constraints (preconditions and effects of decisions). With LLMs' commonsense, reasoning, and programming capabilities, this opens up the possibilities of a universal LLM-based approach to planning problems. Inspired by this observation, we propose LLMFP, a general-purpose framework that leverages LLMs to capture key information from planning problems and formally formulate and solve them as optimization problems from scratch, with no task-specific examples needed. We apply LLMFP to 9 planning problems, ranging from multi-constraint decision making to multi-step planning problems, and demonstrate that LLMFP achieves on average 83.7% and 86.8% optimal rate across 9 tasks for GPT-4o and Claude 3.5 Sonnet, significantly outperforming the best baseline (direct planning with OpenAI o1-preview) with 37.6% and 40.7% improvements. We also validate components of LLMFP with ablation experiments and analyzed the underlying success and failure reasons. Project page: https://sites.google.com/view/llmfp.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06980v1">Are They All Good? Evaluating the Quality of CoTs in LLM-based Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive performance in code generation, particularly when augmented with chain-of-thought (CoT) prompting techniques. They break down requirements into intermediate reasoning steps, which act as design rationales to guide LLMs in writing code like human programmers. Thus, the quality of these steps is crucial for ensuring the correctness and reliability of the generated code. However, little is known about the quality of CoT generated by LLMs. To what extent can we trust the thoughts generated by LLMs? How good are they? This paper empirically explores the external and internal factors of why LLMs generate unsatisfactory CoTs by analyzing 1,023 failed code samples on two widely used code generation benchmarks. We also evaluate their impact on code generation performance by analyzing 210 CoT-code pairs and refining the unsatisfied CoTs by prompting LLMs. Our study reveals three key findings: (1) External factors (53.60%), such as unclear requirements and lack of context, mainly affect CoT quality, while internal factors (40.10%) stem from LLMs' misunderstanding prompts. (2) Even when CoTs are correct, 18.5% of the generated code contains errors due to instruction-following issues; conversely, 11.90% of correct code is paired with flawed CoTs. (3) Refining low-quality CoTs is feasible, i.e., LLMs improve when given detailed problem descriptions. These findings highlight key challenges in CoT-based code generation and suggest directions for improving LLM reasoning and reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06920v1">Rethinking Verification for LLM Code Generation: From Generation to Testing</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently achieved notable success in code-generation benchmarks such as HumanEval and LiveCodeBench. However, a detailed examination reveals that these evaluation suites often comprise only a limited number of homogeneous test cases, resulting in subtle faults going undetected. This not only artificially inflates measured performance but also compromises accurate reward estimation in reinforcement learning frameworks utilizing verifiable rewards (RLVR). To address these critical shortcomings, we systematically investigate the test-case generation (TCG) task by proposing multi-dimensional metrics designed to rigorously quantify test-suite thoroughness. Furthermore, we introduce a human-LLM collaborative method (SAGA), leveraging human programming expertise with LLM reasoning capability, aimed at significantly enhancing both the coverage and the quality of generated test cases. In addition, we develop a TCGBench to facilitate the study of the TCG task. Experiments show that SAGA achieves a detection rate of 90.62% and a verifier accuracy of 32.58% on TCGBench. The Verifier Accuracy (Verifier Acc) of the code generation evaluation benchmark synthesized by SAGA is 10.78% higher than that of LiveCodeBench-v6. These results demonstrate the effectiveness of our proposed method. We hope this work contributes to building a scalable foundation for reliable LLM code evaluation, further advancing RLVR in code generation, and paving the way for automated adversarial test synthesis and adaptive benchmark integration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06910v1">Exploring LLMs for Predicting Tutor Strategy and Student Outcomes in Dialogues</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 Published in BEA 2025: 20th Workshop on Innovative Use of NLP for Building Educational Applications
    </div>
    <details class="paper-abstract">
      Tutoring dialogues have gained significant attention in recent years, given the prominence of online learning and the emerging tutoring abilities of artificial intelligence (AI) agents powered by large language models (LLMs). Recent studies have shown that the strategies used by tutors can have significant effects on student outcomes, necessitating methods to predict how tutors will behave and how their actions impact students. However, few works have studied predicting tutor strategy in dialogues. Therefore, in this work we investigate the ability of modern LLMs, particularly Llama 3 and GPT-4o, to predict both future tutor moves and student outcomes in dialogues, using two math tutoring dialogue datasets. We find that even state-of-the-art LLMs struggle to predict future tutor strategy while tutor strategy is highly indicative of student outcomes, outlining a need for more powerful methods to approach this task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06850v1">The Dark Side of LLMs Agent-based Attacks for Complete Computer Takeover</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      The rapid adoption of Large Language Model (LLM) agents and multi-agent systems enables unprecedented capabilities in natural language processing and generation. However, these systems have introduced unprecedented security vulnerabilities that extend beyond traditional prompt injection attacks. This paper presents the first comprehensive evaluation of LLM agents as attack vectors capable of achieving complete computer takeover through the exploitation of trust boundaries within agentic AI systems where autonomous entities interact and influence each other. We demonstrate that adversaries can leverage three distinct attack surfaces - direct prompt injection, RAG backdoor attacks, and inter-agent trust exploitation - to coerce popular LLMs (including GPT-4o, Claude-4 and Gemini-2.5) into autonomously installing and executing malware on victim machines. Our evaluation of 17 state-of-the-art LLMs reveals an alarming vulnerability hierarchy: while 41.2% of models succumb to direct prompt injection, 52.9% are vulnerable to RAG backdoor attacks, and a critical 82.4% can be compromised through inter-agent trust exploitation. Notably, we discovered that LLMs which successfully resist direct malicious commands will execute identical payloads when requested by peer agents, revealing a fundamental flaw in current multi-agent security models. Our findings demonstrate that only 5.9% of tested models (1/17) proved resistant to all attack vectors, with the majority exhibiting context-dependent security behaviors that create exploitable blind spots. Our findings also highlight the need to increase awareness and research on the security risks of LLMs, showing a paradigm shift in cybersecurity threats, where AI tools themselves become sophisticated attack vectors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02579v3">EMORL: Ensemble Multi-Objective Reinforcement Learning for Efficient and Flexible LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 14 pages, 9 figures, accepted by the SIGDIAL 2025 conference
    </div>
    <details class="paper-abstract">
      Recent advances in reinforcement learning (RL) for large language model (LLM) fine-tuning show promise in addressing multi-objective tasks but still face significant challenges, including competing objective balancing, low training efficiency, poor scalability, and limited explainability. Leveraging ensemble learning principles, we introduce an Ensemble Multi-Objective RL (EMORL) framework that fine-tunes multiple models with individual objectives while optimizing their aggregation after the fine-tuning to improve efficiency and flexibility. Our method is the first to aggregate the hidden states of individual models, incorporating contextual information from multiple objectives. This approach is supported by a hierarchical grid search algorithm that identifies optimal weighted combinations. We evaluate EMORL on counselor reflection generation tasks, using text classification models to score the generations and provide rewards during RL fine-tuning. Through comprehensive experiments on the PAIR and Psych8k datasets, we demonstrate the advantages of EMORL against existing baselines: significantly lower and more stable training consumption ($17,529\pm 1,650$ data points and $6,573\pm 147.43$ seconds), improved scalability and explainability, and comparable performance across multiple objectives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15167v2">LLM Agent for Hyper-Parameter Optimization</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 6 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Hyper-parameters are essential and critical for the performance of communication algorithms. However, current hyper-parameters optimization approaches for Warm-Start Particles Swarm Optimization with Crossover and Mutation (WS-PSO-CM) algorithm, designed for radio map-enabled unmanned aerial vehicle (UAV) trajectory and communication, are primarily heuristic-based, exhibiting low levels of automation and improvable performance. In this paper, we design an Large Language Model (LLM) agent for automatic hyper-parameters-tuning, where an iterative framework and Model Context Protocol (MCP) are applied. In particular, the LLM agent is first set up via a profile, which specifies the boundary of hyper-parameters, task objective, terminal condition, conservative or aggressive strategy of optimizing hyper-parameters, and LLM configurations. Then, the LLM agent iteratively invokes WS-PSO-CM algorithm for exploration. Finally, the LLM agent exits the loop based on the terminal condition and returns an optimized set of hyperparameters. Our experiment results show that the minimal sum-rate achieved by hyper-parameters generated via our LLM agent is significantly higher than those by both human heuristics and random generation methods. This indicates that an LLM agent with PSO and WS-PSO-CM algorithm knowledge is useful in seeking high-performance hyper-parameters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09347v3">Safer or Luckier? LLMs as Safety Evaluators Are Not Robust to Artifacts</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 9 pages, ACL 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly employed as automated evaluators to assess the safety of generated content, yet their reliability in this role remains uncertain. This study evaluates a diverse set of 11 LLM judge models across critical safety domains, examining three key aspects: self-consistency in repeated judging tasks, alignment with human judgments, and susceptibility to input artifacts such as apologetic or verbose phrasing. Our findings reveal that biases in LLM judges can significantly distort the final verdict on which content source is safer, undermining the validity of comparative evaluations. Notably, apologetic language artifacts alone can skew evaluator preferences by up to 98\%. Contrary to expectations, larger models do not consistently exhibit greater robustness, while smaller models sometimes show higher resistance to specific artifacts. To mitigate LLM evaluator robustness issues, we investigate jury-based evaluations aggregating decisions from multiple models. Although this approach both improves robustness and enhances alignment to human judgements, artifact sensitivity persists even with the best jury configurations. These results highlight the urgent need for diversified, artifact-resistant methodologies to ensure reliable safety assessments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16903v2">GuidedBench: Measuring and Mitigating the Evaluation Discrepancies of In-the-wild LLM Jailbreak Methods</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 Homepage: https://sproutnan.github.io/AI-Safety_Benchmark/
    </div>
    <details class="paper-abstract">
      Despite the growing interest in jailbreak methods as an effective red-teaming tool for building safe and responsible large language models (LLMs), flawed evaluation system designs have led to significant discrepancies in their effectiveness assessments. We conduct a systematic measurement study based on 37 jailbreak studies since 2022, focusing on both the methods and the evaluation systems they employ. We find that existing evaluation systems lack case-specific criteria, resulting in misleading conclusions about their effectiveness and safety implications. This paper advocates a shift to a more nuanced, case-by-case evaluation paradigm. We introduce GuidedBench, a novel benchmark comprising a curated harmful question dataset, detailed case-by-case evaluation guidelines and an evaluation system integrated with these guidelines -- GuidedEval. Experiments demonstrate that GuidedBench offers more accurate measurements of jailbreak performance, enabling meaningful comparisons across methods and uncovering new insights overlooked in previous evaluations. GuidedEval reduces inter-evaluator variance by at least 76.03\%. Furthermore, we observe that incorporating guidelines can enhance the effectiveness of jailbreak methods themselves, offering new insights into both attack strategies and evaluation paradigms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06774v1">Checklist Engineering Empowers Multilingual LLM Judges</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Automated text evaluation has long been a central issue in Natural Language Processing (NLP). Recently, the field has shifted toward using Large Language Models (LLMs) as evaluators-a trend known as the LLM-as-a-Judge paradigm. While promising and easily adaptable across tasks, this approach has seen limited exploration in multilingual contexts. Existing multilingual studies often rely on proprietary models or require extensive training data for fine-tuning, raising concerns about cost, time, and efficiency. In this paper, we propose Checklist Engineering based LLM-as-a-Judge (CE-Judge), a training-free framework that uses checklist intuition for multilingual evaluation with an open-source model. Experiments across multiple languages and three benchmark datasets, under both pointwise and pairwise settings, show that our method generally surpasses the baselines and performs on par with the GPT-4o model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04446v2">Tail-aware Adversarial Attacks: A Distributional Approach to Efficient LLM Jailbreaking</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      To guarantee safe and robust deployment of large language models (LLMs) at scale, it is critical to accurately assess their adversarial robustness. Existing adversarial attacks typically target harmful responses in single-point, greedy generations, overlooking the inherently stochastic nature of LLMs. In this paper, we propose a novel framework for adversarial robustness evaluation that explicitly models the entire output distribution, including tail-risks, providing better estimates for model robustness at scale. By casting the attack process as a resource allocation problem between optimization and sampling, we determine compute-optimal tradeoffs and show that integrating sampling into existing attacks boosts ASR by up to 48% and improves efficiency by up to two orders of magnitude. Our framework also enables us to analyze how different attack algorithms affect output harm distributions. Surprisingly, we find that most optimization strategies have little effect on output harmfulness. Finally, we introduce a data-free proof-of-concept objective based on entropy-maximization to demonstrate how our tail-aware perspective enables new optimization targets. Overall, our findings highlight the importance of tail-aware attacks and evaluation protocols to accurately assess and strengthen LLM safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06762v1">Leveraging LLMs for Semantic Conflict Detection via Unit Test Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 Comments: 11 pages, in Portuguese language. 3 figures. Submitted to SAST 2025 (X Simp\'osio Brasileiro de Teste de Software Sistem\'atico e Automatizado)
    </div>
    <details class="paper-abstract">
      Semantic conflicts arise when a developer introduces changes to a codebase that unintentionally affect the behavior of changes integrated in parallel by other developers. Traditional merge tools are unable to detect such conflicts, so complementary tools like SMAT have been proposed. SMAT relies on generating and executing unit tests: if a test fails on the base version, passes on a developer's modified version, but fails again after merging with another developer's changes, a semantic conflict is indicated. While SMAT is effective at detecting conflicts, it suffers from a high rate of false negatives, partly due to the limitations of unit test generation tools such as Randoop and Evosuite. To investigate whether large language models (LLMs) can overcome these limitations, we propose and integrate a new test generation tool based on Code Llama 70B into SMAT. We explore the model's ability to generate tests using different interaction strategies, prompt contents, and parameter configurations. Our evaluation uses two samples: a benchmark with simpler systems from related work, and a more significant sample based on complex, real-world systems. We assess the effectiveness of the new SMAT extension in detecting conflicts. Results indicate that, although LLM-based test generation remains challenging and computationally expensive in complex scenarios, there is promising potential for improving semantic conflict detection. -- Conflitos sem^anticos surgem quando um desenvolvedor introduz mudan\c{c}as em uma base de c\'odigo que afetam, de forma n~ao intencional, o comportamento de altera\c{c}~oes integradas em paralelo por outros desenvolvedores. Ferramentas tradicionais de merge n~ao conseguem detectar esse tipo de conflito, por isso ferramentas complementares como o SMAT foram propostas. O SMAT depende da gera\c{c}~ao e execu\c{c}~ao de testes de unidade: se um teste falha na vers~ao base, passa na vers~ao modificada por um desenvolvedor, mas volta a falhar ap\'os o merge com as mudan\c{c}as de outro desenvolvedor, um conflito sem^antico \'e identificado. Embora o SMAT seja eficaz na detec\c{c}~ao de conflitos, apresenta alta taxa de falsos negativos, em parte devido \`as limita\c{c}~oes das ferramentas de gera\c{c}~ao de testes como Randoop e Evosuite. Para investigar se modelos de linguagem de grande porte (LLMs) podem superar essas limita\c{c}~oes, propomos e integramos ao SMAT uma nova ferramenta de gera\c{c}~ao de testes baseada no Code Llama 70B. Exploramos a capacidade do modelo de gerar testes utilizando diferentes estrat\'egias de intera\c{c}~ao, conte\'udos de prompts e configura\c{c}~oes de par^ametros. Nossa avalia\c{c}~ao utiliza duas amostras: um benchmark com sistemas mais simples, usados em trabalhos relacionados, e uma amostra mais significativa baseada em sistemas complexos e reais. Avaliamos a efic\'acia da nova extens~ao do SMAT na detec\c{c}~ao de conflitos. Os resultados indicam que, embora a gera\c{c}~ao de testes por LLM em cen\'arios complexos ainda seja desafiadora e custosa computacionalmente, h\'a potencial promissor para aprimorar a detec\c{c}~ao de conflitos sem^anticos.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03785v3">Knockout LLM Assessment: Using Large Language Models for Evaluations through Iterative Pairwise Comparisons</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 Accepted to GEM @ ACL 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown to be effective evaluators across various domains such as machine translations or the scientific domain. Current LLM-as-a-Judge approaches rely mostly on individual assessments or a single round of pairwise assessments, preventing the judge LLM from developing a global ranking perspective. To address this, we present Knockout Assessment, an LLM-asa Judge method using a knockout tournament system with iterative pairwise comparisons. Experiments across three LLMs on two datasets show that knockout assessment improves scoring accuracy, increasing Pearson correlation with expert evaluations by 0.07 on average for university-level exam scoring and machine translation evaluations, aligning LLM assessments more closely with human scoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14541v2">LLM-based User Profile Management for Recommender System</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 Accepted GENNEXT@SIGIR'25 Workshop
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has opened new opportunities in recommender systems by enabling zero-shot recommendation without conventional training. Despite their potential, most existing works rely solely on users' purchase histories, leaving significant room for improvement by incorporating user-generated textual data, such as reviews and product descriptions. Addressing this gap, we propose PURE, a novel LLM-based recommendation framework that builds and maintains evolving user profiles by systematically extracting and summarizing key information from user reviews. PURE consists of three core components: a Review Extractor for identifying user preferences and key product features, a Profile Updater for refining and updating user profiles, and a Recommender for generating personalized recommendations using the most current profile. To evaluate PURE, we introduce a continuous sequential recommendation task that reflects real-world scenarios by adding reviews over time and updating predictions incrementally. Our experimental results on Amazon datasets demonstrate that PURE outperforms existing LLM-based methods, effectively leveraging long-term user information while managing token limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06719v1">A Neural Representation Framework with LLM-Driven Spatial Reasoning for Open-Vocabulary 3D Visual Grounding</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Open-vocabulary 3D visual grounding aims to localize target objects based on free-form language queries, which is crucial for embodied AI applications such as autonomous navigation, robotics, and augmented reality. Learning 3D language fields through neural representations enables accurate understanding of 3D scenes from limited viewpoints and facilitates the localization of target objects in complex environments. However, existing language field methods struggle to accurately localize instances using spatial relations in language queries, such as ``the book on the chair.'' This limitation mainly arises from inadequate reasoning about spatial relations in both language queries and 3D scenes. In this work, we propose SpatialReasoner, a novel neural representation-based framework with large language model (LLM)-driven spatial reasoning that constructs a visual properties-enhanced hierarchical feature field for open-vocabulary 3D visual grounding. To enable spatial reasoning in language queries, SpatialReasoner fine-tunes an LLM to capture spatial relations and explicitly infer instructions for the target, anchor, and spatial relation. To enable spatial reasoning in 3D scenes, SpatialReasoner incorporates visual properties (opacity and color) to construct a hierarchical feature field. This field represents language and instance features using distilled CLIP features and masks extracted via the Segment Anything Model (SAM). The field is then queried using the inferred instructions in a hierarchical manner to localize the target 3D instance based on the spatial relation in the language query. Extensive experiments show that our framework can be seamlessly integrated into different neural representations, outperforming baseline models in 3D visual grounding while empowering their spatial reasoning capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05816v1">Affective-ROPTester: Capability and Bias Analysis of LLMs in Predicting Retinopathy of Prematurity</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Despite the remarkable progress of large language models (LLMs) across various domains, their capacity to predict retinopathy of prematurity (ROP) risk remains largely unexplored. To address this gap, we introduce a novel Chinese benchmark dataset, termed CROP, comprising 993 admission records annotated with low, medium, and high-risk labels. To systematically examine the predictive capabilities and affective biases of LLMs in ROP risk stratification, we propose Affective-ROPTester, an automated evaluation framework incorporating three prompting strategies: Instruction-based, Chain-of-Thought (CoT), and In-Context Learning (ICL). The Instruction scheme assesses LLMs' intrinsic knowledge and associated biases, whereas the CoT and ICL schemes leverage external medical knowledge to enhance predictive accuracy. Crucially, we integrate emotional elements at the prompt level to investigate how different affective framings influence the model's ability to predict ROP and its bias patterns. Empirical results derived from the CROP dataset yield two principal observations. First, LLMs demonstrate limited efficacy in ROP risk prediction when operating solely on intrinsic knowledge, yet exhibit marked performance gains when augmented with structured external inputs. Second, affective biases are evident in the model outputs, with a consistent inclination toward overestimating medium- and high-risk cases. Third, compared to negative emotions, positive emotional framing contributes to mitigating predictive bias in model outputs. These findings highlight the critical role of affect-sensitive prompt engineering in enhancing diagnostic reliability and emphasize the utility of Affective-ROPTester as a framework for evaluating and mitigating affective bias in clinical language modeling systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15790v2">ETrace:Event-Driven Vulnerability Detection in Smart Contracts via LLM-Based Trace Analysis</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 4 pages, 1 figure. Submitted to the 16th Asia-Pacific Symposium on Internetware (Internetware 2025)
    </div>
    <details class="paper-abstract">
      With the advance application of blockchain technology in various fields, ensuring the security and stability of smart contracts has emerged as a critical challenge. Current security analysis methodologies in vulnerability detection can be categorized into static analysis and dynamic analysis methods.However, these existing traditional vulnerability detection methods predominantly rely on analyzing original contract code, not all smart contracts provide accessible code.We present ETrace, a novel event-driven vulnerability detection framework for smart contracts, which uniquely identifies potential vulnerabilities through LLM-powered trace analysis without requiring source code access. By extracting fine-grained event sequences from transaction logs, the framework leverages Large Language Models (LLMs) as adaptive semantic interpreters to reconstruct event analysis through chain-of-thought reasoning. ETrace implements pattern-matching to establish causal links between transaction behavior patterns and known attack behaviors. Furthermore, we validate the effectiveness of ETrace through preliminary experimental results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05754v1">LeAD: The LLM Enhanced Planning System Converged with End-to-end Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      A principal barrier to large-scale deployment of urban autonomous driving systems lies in the prevalence of complex scenarios and edge cases. Existing systems fail to effectively interpret semantic information within traffic contexts and discern intentions of other participants, consequently generating decisions misaligned with skilled drivers' reasoning patterns. We present LeAD, a dual-rate autonomous driving architecture integrating imitation learning-based end-to-end (E2E) frameworks with large language model (LLM) augmentation. The high-frequency E2E subsystem maintains real-time perception-planning-control cycles, while the low-frequency LLM module enhances scenario comprehension through multi-modal perception fusion with HD maps and derives optimal decisions via chain-of-thought (CoT) reasoning when baseline planners encounter capability limitations. Our experimental evaluation in the CARLA Simulator demonstrates LeAD's superior handling of unconventional scenarios, achieving 71 points on Leaderboard V1 benchmark, with a route completion of 93%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05750v1">DocTalk: Scalable Graph-based Dialogue Synthesis for Enhancing LLM Conversational Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 Accepted at SIGDIAL 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly employed in multi-turn conversational tasks, yet their pre-training data predominantly consists of continuous prose, creating a potential mismatch between required capabilities and training paradigms. We introduce a novel approach to address this discrepancy by synthesizing conversational data from existing text corpora. We present a pipeline that transforms a cluster of multiple related documents into an extended multi-turn, multi-topic information-seeking dialogue. Applying our pipeline to Wikipedia articles, we curate DocTalk, a multi-turn pre-training dialogue corpus consisting of over 730k long conversations. We hypothesize that exposure to such synthesized conversational structures during pre-training can enhance the fundamental multi-turn capabilities of LLMs, such as context memory and understanding. Empirically, we show that incorporating DocTalk during pre-training results in up to 40% gain in context memory and understanding, without compromising base performance. DocTalk is available at https://huggingface.co/datasets/AmazonScience/DocTalk.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05740v1">GPTKB v1.5: A Massive Knowledge Base for Exploring Factual LLM Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 7 pages, 6 figures, 1 table
    </div>
    <details class="paper-abstract">
      Language models are powerful tools, yet their factual knowledge is still poorly understood, and inaccessible to ad-hoc browsing and scalable statistical analysis. This demonstration introduces GPTKB v1.5, a densely interlinked 100-million-triple knowledge base (KB) built for $14,000 from GPT-4.1, using the GPTKB methodology for massive-recursive LLM knowledge materialization (Hu et al., ACL 2025). The demonstration experience focuses on three use cases: (1) link-traversal-based LLM knowledge exploration, (2) SPARQL-based structured LLM knowledge querying, (3) comparative exploration of the strengths and weaknesses of LLM knowledge. Massive-recursive LLM knowledge materialization is a groundbreaking opportunity both for the research area of systematic analysis of LLM knowledge, as well as for automated KB construction. The GPTKB demonstrator is accessible at https://gptkb.org.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05733v1">When Transformers Meet Recommenders: Integrating Self-Attentive Sequential Recommendation with Fine-Tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Self-Attentive Sequential Recommendation (SASRec) effectively captures long-term user preferences by applying attention mechanisms to historical interactions. Concurrently, the rise of Large Language Models (LLMs) has motivated research into LLM-based recommendation, which leverages their powerful generalization and language understanding capabilities. However, LLMs often lack the domain-specific knowledge and collaborative signals essential for high-quality recommendations when relying solely on textual prompts. To address this limitation, this study proposes SASRecLLM, a novel framework that integrates SASRec as a collaborative encoder with an LLM fine-tuned using Low-Rank Adaptation (LoRA). The components are connected via a mapping layer to align their dimensional spaces, and three targeted training strategies are designed to optimize the hybrid architecture. Extensive experiments on multiple datasets demonstrate that SASRecLLM achieves robust and consistent improvements over strong baselines in both cold-start and warm-start scenarios. This work advances the field of LLM-based recommendation by presenting a modular and effective paradigm for fusing structured collaborative filtering with the semantic power of fine-tuned LLMs. The implementation is available on GitHub: https://github.com/kechenkristin/RecLLM
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02233v3">Enhancing LLM Reliability via Explicit Knowledge Boundary Modeling</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are prone to hallucination stemming from misaligned self-awareness, particularly when processing queries exceeding their knowledge boundaries. While existing mitigation strategies employ uncertainty estimation or query rejection mechanisms, they suffer from computational efficiency and sacrificed helpfulness. To address these issues, we propose the Explicit Knowledge Boundary Modeling (EKBM) framework, integrating fast and slow reasoning systems to harmonize reliability and usability. The framework first employs a fast-thinking model to generate confidence-labeled responses, enabling immediate utilization of high-confidence outputs, whereas uncertain predictions trigger a slow refinement model for accuracy improvement. To align model behavior with our proposed object, we propose a hybrid training pipeline, enhancing self-awareness without degrading task performance. Evaluations on dialogue state tracking tasks demonstrate that EKBM achieves superior model reliability over uncertainty-based baselines. Further analysis reveals that refinement substantially boosts accuracy while maintaining low computational overhead. The framework establishes a scalable paradigm for deploying reliable LLMs in error-sensitive applications, effectively balancing accuracy and practical utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02962v2">RAG-R1 : Incentivize the Search and Reasoning Capabilities of LLMs through Multi-query Parallelism</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, while they remain prone to generating hallucinated or outdated responses due to their static internal knowledge. Recent advancements in Retrieval-Augmented Generation (RAG) methods have explored enhancing models' search and reasoning capabilities through reinforcement learning (RL). Although these methods demonstrate promising results, they face challenges in training stability and encounter issues such as substantial inference time and restricted capabilities due to the single-query mode. In this paper, we propose RAG-R1, a novel training framework designed to enable LLMs to adaptively leverage internal and external knowledge during the reasoning process. We further expand the generation and retrieval processes within the framework from single-query mode to multi-query parallelism, aimed at reducing inference time and enhancing the model's capabilities. Extensive experiments on seven question-answering benchmarks demonstrate that our method outperforms the strongest baseline by up to 13.2% and decreases inference time by 11.1%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05686v1">Smoothie-Qwen: Post-Hoc Smoothing to Reduce Language Bias in Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Multilingual large language models (LLMs) often exhibit language confusion, a tendency to generate responses in a dominant language irrespective of the prompt's language. To address this, we propose Smoothie-Qwen, a lightweight, post-hoc method that mitigates language bias without retraining. This technique selectively adjusts token-level output probabilities to effectively suppress undesired language generation. Applied to the Qwen model, our method reduces unintended Chinese output by over 95% while preserving task accuracy on multilingual benchmarks. This work provides a practical and efficient solution for enhancing the language controllability of LLMs, making them more reliable for global applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03020v2">AI Literacy and LLM Engagement in Higher Education: A Cross-National Quantitative Study</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 26 pages, 8 figures, 3 tables. Submitted for consideration in a forthcoming issue of the International Journal of Educational Technology in Higher Education
    </div>
    <details class="paper-abstract">
      This study presents a cross-national quantitative analysis of how university students in the United States and Bangladesh interact with Large Language Models (LLMs). Based on an online survey of 318 students, results show that LLMs enhance access to information, improve writing, and boost academic performance. However, concerns about overreliance, ethical risks, and critical thinking persist. Guided by the AI Literacy Framework, Expectancy-Value Theory, and Biggs' 3P Model, the study finds that motivational beliefs and technical competencies shape LLM engagement. Significant correlations were found between LLM use and perceived literacy benefits (r = .59, p < .001) and optimism (r = .41, p < .001). ANOVA results showed more frequent use among U.S. students (F = 7.92, p = .005) and STEM majors (F = 18.11, p < .001). Findings support the development of ethical, inclusive, and pedagogically sound frameworks for integrating LLMs in higher education.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05639v1">ECom-Bench: Can LLM Agent Resolve Real-World E-commerce Customer Support Issues?</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      In this paper, we introduce ECom-Bench, the first benchmark framework for evaluating LLM agent with multimodal capabilities in the e-commerce customer support domain. ECom-Bench features dynamic user simulation based on persona information collected from real e-commerce customer interactions and a realistic task dataset derived from authentic e-commerce dialogues. These tasks, covering a wide range of business scenarios, are designed to reflect real-world complexities, making ECom-Bench highly challenging. For instance, even advanced models like GPT-4o achieve only a 10-20% pass^3 metric in our benchmark, highlighting the substantial difficulties posed by complex e-commerce scenarios. Upon publication, the code and data will be open-sourced to facilitate further research and development in this domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05638v1">LLMs are Introvert</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      The exponential growth of social media and generative AI has transformed information dissemination, fostering connectivity but also accelerating the spread of misinformation. Understanding information propagation dynamics and developing effective control strategies is essential to mitigate harmful content. Traditional models, such as SIR, provide basic insights but inadequately capture the complexities of online interactions. Advanced methods, including attention mechanisms and graph neural networks, enhance accuracy but typically overlook user psychology and behavioral dynamics. Large language models (LLMs), with their human-like reasoning, offer new potential for simulating psychological aspects of information spread. We introduce an LLM-based simulation environment capturing agents' evolving attitudes, emotions, and responses. Initial experiments, however, revealed significant gaps between LLM-generated behaviors and authentic human dynamics, especially in stance detection and psychological realism. A detailed evaluation through Social Information Processing Theory identified major discrepancies in goal-setting and feedback evaluation, stemming from the lack of emotional processing in standard LLM training. To address these issues, we propose the Social Information Processing-based Chain of Thought (SIP-CoT) mechanism enhanced by emotion-guided memory. This method improves the interpretation of social cues, personalization of goals, and evaluation of feedback. Experimental results confirm that SIP-CoT-enhanced LLM agents more effectively process social information, demonstrating behaviors, attitudes, and emotions closer to real human interactions. In summary, this research highlights critical limitations in current LLM-based propagation simulations and demonstrates how integrating SIP-CoT and emotional memory significantly enhances the social intelligence and realism of LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16327v2">Feint and Attack: Attention-Based Strategies for Jailbreaking and Protecting LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Jailbreak attack can be used to access the vulnerabilities of Large Language Models (LLMs) by inducing LLMs to generate the harmful content. And the most common method of the attack is to construct semantically ambiguous prompts to confuse and mislead the LLMs. To access the security and reveal the intrinsic relation between the input prompt and the output for LLMs, the distribution of attention weight is introduced to analyze the underlying reasons. By using statistical analysis methods, some novel metrics are defined to better describe the distribution of attention weight, such as the Attention Intensity on Sensitive Words (Attn_SensWords), the Attention-based Contextual Dependency Score (Attn_DepScore) and Attention Dispersion Entropy (Attn_Entropy). By leveraging the distinct characteristics of these metrics, the beam search algorithm and inspired by the military strategy "Feint and Attack", an effective jailbreak attack strategy named as Attention-Based Attack (ABA) is proposed. In the ABA, nested attack prompts are employed to divert the attention distribution of the LLMs. In this manner, more harmless parts of the input can be used to attract the attention of the LLMs. In addition, motivated by ABA, an effective defense strategy called as Attention-Based Defense (ABD) is also put forward. Compared with ABA, the ABD can be used to enhance the robustness of LLMs by calibrating the attention distribution of the input prompt. Some comparative experiments have been given to demonstrate the effectiveness of ABA and ABD. Therefore, both ABA and ABD can be used to access the security of the LLMs. The comparative experiment results also give a logical explanation that the distribution of attention weight can bring great influence on the output for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05630v1">How Not to Detect Prompt Injections with an LLM</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      LLM-integrated applications and agents are vulnerable to prompt injection attacks, in which adversaries embed malicious instructions within seemingly benign user inputs to manipulate the LLM's intended behavior. Recent defenses based on $\textit{known-answer detection}$ (KAD) have achieved near-perfect performance by using an LLM to classify inputs as clean or contaminated. In this work, we formally characterize the KAD framework and uncover a structural vulnerability in its design that invalidates its core security premise. We design a methodical adaptive attack, $\textit{DataFlip}$, to exploit this fundamental weakness. It consistently evades KAD defenses with detection rates as low as $1.5\%$ while reliably inducing malicious behavior with success rates of up to $88\%$, without needing white-box access to the LLM or any optimization procedures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05629v1">Enhancing Student Learning with LLM-Generated Retrieval Practice Questions: An Empirical Study in Data Science Courses</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Retrieval practice is a well-established pedagogical technique known to significantly enhance student learning and knowledge retention. However, generating high-quality retrieval practice questions is often time-consuming and labor intensive for instructors, especially in rapidly evolving technical subjects. Large Language Models (LLMs) offer the potential to automate this process by generating questions in response to prompts, yet the effectiveness of LLM-generated retrieval practice on student learning remains to be established. In this study, we conducted an empirical study involving two college-level data science courses, with approximately 60 students. We compared learning outcomes during one week in which students received LLM-generated multiple-choice retrieval practice questions to those from a week in which no such questions were provided. Results indicate that students exposed to LLM-generated retrieval practice achieved significantly higher knowledge retention, with an average accuracy of 89%, compared to 73% in the week without such practice. These findings suggest that LLM-generated retrieval questions can effectively support student learning and may provide a scalable solution for integrating retrieval practice into real-time teaching. However, despite these encouraging outcomes and the potential time-saving benefits, cautions must be taken, as the quality of LLM-generated questions can vary. Instructors must still manually verify and revise the generated questions before releasing them to students.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05617v1">Flipping Knowledge Distillation: Leveraging Small Models' Expertise to Enhance LLMs in Text Matching</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 Accepted by ACL 2025 main
    </div>
    <details class="paper-abstract">
      Knowledge distillation typically involves transferring knowledge from a Large Language Model (LLM) to a Smaller Language Model (SLM). However, in tasks such as text matching, fine-tuned smaller models often yield more effective domain-specific representations, as they focus on optimizing the similarity of input pairs. To leverage both the specialized strengths of small models and the rich semantic understanding of LLMs, we introduce a flipped knowledge distillation paradigm, where LLM learns from SLM. Specifically, we address the architectural gap between decoder-only LLMs and smaller encoder-based models by reinterpreting LLMs in an encoder-decoder manner using LoRA. The encoder generates compressed representations, while the decoder maps them to the output space. During training, the encoder produces representations and their similarities, which are then aligned with the similarity scores produced by the teacher, using our proposed Margin-aware Contrastive Learning (MCL) approach. The MCL ensures accurate similarity for both positive and negative pairs, and adaptively handles the internal differences within positive and negative samples. Our paradigm requires only a reasonably good-performing SLM, allowing the LLM to achieve improved performance. Experiments on financial and healthcare benchmarks, as well as real-world applications, confirm its effectiveness, and the model has been fully deployed in an online environment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05598v1">Self-Review Framework for Enhancing Instruction Following Capability of LLM</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Various techniques have been proposed to improve large language models (LLMs) adherence to formatting and instruction constraints. One of the most effective approaches involves utilizing high-quality data generated by powerful models. However, such models often fail to fully comply with complex instructions in a single generation. To address this limitation, iterative revision methods have been introduced. Nevertheless, as the number of data points and revision iterations increases, the associated monetary costs grow significantly. As a resource-efficient alternative, methods have been proposed that leverage high-performance evaluation tools to compensate for the limited self-evaluation capabilities of open-source LLMs. However, these approaches often lead to a degradation in output quality due to excessive revision. To overcome these challenges, we propose Re5, a self-evaluation and revision framework designed to enhance instruction-following performance while preserving the quality of the generated content. Re5 extracts task and constraint components from user instructions, performs structural evaluations to prevent error accumulation, and applies fine-grained constraint-specific content evaluations followed by selective revisions. This process ensures precise and quality-preserving improvements. The final high-quality outputs are used for alignment tuning, enabling long-term alignment improvements through a data-centric iterative refinement loop. Experimental results demonstrate that Re5 achieves instruction-following performance comparable to models trained on data generated by GPT-4o-mini, a high-performance model, even with a small amount of data while maintaining response quality with a 64.24%-win rate over the non-revised initial responses. These results validate Re5 as an efficient and effective solution for enhancing instruction adherence with minimal external supervision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05578v1">The Landscape of Memorization in LLMs: Mechanisms, Measurement, and Mitigation</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks, yet they also exhibit memorization of their training data. This phenomenon raises critical questions about model behavior, privacy risks, and the boundary between learning and memorization. Addressing these concerns, this paper synthesizes recent studies and investigates the landscape of memorization, the factors influencing it, and methods for its detection and mitigation. We explore key drivers, including training data duplication, training dynamics, and fine-tuning procedures that influence data memorization. In addition, we examine methodologies such as prefix-based extraction, membership inference, and adversarial prompting, assessing their effectiveness in detecting and measuring memorized content. Beyond technical analysis, we also explore the broader implications of memorization, including the legal and ethical implications. Finally, we discuss mitigation strategies, including data cleaning, differential privacy, and post-training unlearning, while highlighting open challenges in balancing the minimization of harmful memorization with utility. This paper provides a comprehensive overview of the current state of research on LLM memorization across technical, privacy, and performance dimensions, identifying critical directions for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05577v1">Beyond Retrieval: Ensembling Cross-Encoders and GPT Rerankers with LLMs for Biomedical QA</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 Paper submitted to CLEF 2025 CEUR-WS
    </div>
    <details class="paper-abstract">
      Biomedical semantic question answering rooted in information retrieval can play a crucial role in keeping up to date with vast, rapidly evolving and ever-growing biomedical literature. A robust system can help researchers, healthcare professionals and even layman users access relevant knowledge grounded in evidence. The BioASQ 2025 Task13b Challenge serves as an important benchmark, offering a competitive platform for advancement of this space. This paper presents the methodologies and results from our participation in this challenge where we built a Retrieval-Augmented Generation (RAG) system that can answer biomedical questions by retrieving relevant PubMed documents and snippets to generate answers. For the retrieval task, we generated dense embeddings from biomedical articles for initial retrieval, and applied an ensemble of finetuned cross-encoders and large language models (LLMs) for re-ranking to identify top relevant documents. Our solution achieved an MAP@10 of 0.1581, placing 10th on the leaderboard for the retrieval task. For answer generation, we employed few-shot prompting of instruction-tuned LLMs. Our system achieved macro-F1 score of 0.95 for yes/no questions (rank 12), Mean Reciprocal Rank (MRR) of 0.64 for factoid questions (rank 1), mean-F1 score of 0.63 for list questions (rank 5), and ROUGE-SU4 F1 score of 0.29 for ideal answers (rank 11).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15740v2">SHADE-Arena: Evaluating Sabotage and Monitoring in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are increasingly deployed as autonomous agents in complex and long horizon settings, it is critical to evaluate their ability to sabotage users by pursuing hidden objectives. We study the ability of frontier LLMs to evade monitoring and achieve harmful hidden goals while completing a wide array of realistic tasks. We evaluate a broad range of frontier LLMs using SHADE (Subtle Harmful Agent Detection & Evaluation)-Arena, the first highly diverse agent evaluation dataset for sabotage and monitoring capabilities of LLM agents. SHADE-Arena consists of complex pairs of benign main tasks and harmful side objectives in complicated environments. Agents are evaluated on their ability to complete the side task without appearing suspicious to an LLM monitor. When measuring agent ability to (a) complete the main task, (b) complete the side task, and (c) avoid detection, we find that the best performing frontier models score 27% (Claude 3.7 Sonnet) and 15% (Gemini 2.5 Pro) as sabotage agents when overseen by Claude 3.6 Sonnet. For current frontier models, success on the side task relies heavily on having access to a hidden scratchpad that is not visible to the monitor. We also use SHADE-Arena to measure models' monitoring abilities, with the top monitor (Gemini 2.5 Pro) achieving an AUC of 0.87 at distinguishing benign and malign transcripts. We find that for now, models still struggle at sabotage due to failures in long-context main task execution. However, our measurements already demonstrate the difficulty of monitoring for subtle sabotage attempts, which we expect to only increase in the face of more complex and longer-horizon tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06376v1">SLDB: An End-To-End Heterogeneous System-on-Chip Benchmark Suite for LLM-Aided Design</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Over the last few years, Large Language Models (LLMs) have emerged as a valuable tool for Electronic Design Automation (EDA). State-of-the-art research in LLM-aided design has demonstrated the ability of LLMs to generate syntactically correct RTL code, showcasing encouraging prospects for integrating AI into the hardware design process. A key enabler of these advancements is the availability of high-quality benchmarks to evaluate new approaches. However, existing datasets and benchmarks fall short of system-level design, as they focus primarily on component-level information and low-complexity designs. To address this gap, we introduce the System-Level Design Benchmark (SLDB), a dataset tailored for evaluating LLMs in system-level integration and configuration tasks. SLDB includes a curated benchmark suite of 10 baseline SoC designs, whose components can be combined into an exponential number of distinct tile-based SoCs through a synthetic library. The dataset provides full SoC configurations, accelerator integration code, communication parameters, and accelerator-aware system configurations, along with testing-application code, compatible with the ESP platform[1].
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00754v2">Language-Unlocked ViT (LUViT): Empowering Self-Supervised Vision Transformers with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 26 pages, 6 figures
    </div>
    <details class="paper-abstract">
      The integration of Large Language Model (LLMs) blocks with Vision Transformers (ViTs) holds immense promise for vision-only tasks by leveraging the rich semantic knowledge and reasoning capabilities of LLMs. However, a fundamental challenge lies in the inherent modality mismatch between text-centric pretraining of LLMs and vision-centric training of ViTs. Direct fusion often fails to fully exploit the LLM's potential and suffers from unstable finetuning. As a result, LLM blocks are kept frozen while only the vision components are learned. As a remedy to these challenges, we introduce Language-Unlocked Vision Transformers (LUViT), a novel approach that bridges this modality mismatch through a synergistic pre-training strategy. LUViT co-adapts a ViT backbone and an LLM fusion block by (1) employing Masked Auto-Encoding (MAE) to pre-train the ViT for richer visual representations, and (2) concurrently training Low-Rank Adaptation (LoRA) layers within the LLM block using the MAE objective. This joint optimization guides the ViT to produce LLM-aligned features and the LLM to effectively interpret visual information. We demonstrate through extensive experiments that LUViT significantly improves performance on various downstream vision tasks, showcasing a more effective and efficient pathway to harness LLM knowledge for visual understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04544v2">hdl2v: A Code Translation Dataset for Enhanced LLM Verilog Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 Published at ACM/IEEE International Symposium on Machine Learning for CAD (MLCAD) 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are playing an increasingly large role in domains such as code generation, including hardware code generation, where Verilog is the key language. However, the amount of publicly available Verilog code pales in comparison to the amount of code available for software languages like Python. In this work, we present hdl2v ("HDL-to-Verilog"), a dataset which seeks to increase the amount of available human-written Verilog data by translating or compiling three other hardware description languages - VHDL, Chisel, and PyMTL3 - to Verilog. Furthermore, we demonstrate the value of hdl2v in enhancing LLM Verilog generation by improving performance of a 32 billion-parameter open-weight model by up to 23% (pass@10) in VerilogEvalV2, without utilizing any data augmentation or knowledge distillation from larger models. We also show hdl2v's ability to boost the performance of a data augmentation-based fine-tuning approach by 63%. Finally, we characterize and analyze our dataset to better understand which characteristics of HDL-to-Verilog datasets can be expanded upon in future work for even better performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06323v1">Bridging AI and Software Security: A Comparative Vulnerability Assessment of LLM Agent Deployment Paradigms</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents face security vulnerabilities spanning AI-specific and traditional software domains, yet current research addresses these separately. This study bridges this gap through comparative evaluation of Function Calling architecture and Model Context Protocol (MCP) deployment paradigms using a unified threat classification framework. We tested 3,250 attack scenarios across seven language models, evaluating simple, composed, and chained attacks targeting both AI-specific threats (prompt injection) and software vulnerabilities (JSON injection, denial-of-service). Function Calling showed higher overall attack success rates (73.5% vs 62.59% for MCP), with greater system-centric vulnerability while MCP exhibited increased LLM-centric exposure. Attack complexity dramatically amplified effectiveness, with chained attacks achieving 91-96% success rates. Counterintuitively, advanced reasoning models demonstrated higher exploitability despite better threat detection. Results demonstrate that architectural choices fundamentally reshape threat landscapes. This work establishes methodological foundations for cross-domain LLM agent security assessment and provides evidence-based guidance for secure deployment. Code and experimental materials are available at https: // github. com/ theconsciouslab-ai/llm-agent-security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04431v2">MedGellan: LLM-Generated Medical Guidance to Support Physicians</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Medical decision-making is a critical task, where errors can result in serious, potentially life-threatening consequences. While full automation remains challenging, hybrid frameworks that combine machine intelligence with human oversight offer a practical alternative. In this paper, we present MedGellan, a lightweight, annotation-free framework that uses a Large Language Model (LLM) to generate clinical guidance from raw medical records, which is then used by a physician to predict diagnoses. MedGellan uses a Bayesian-inspired prompting strategy that respects the temporal order of clinical data. Preliminary experiments show that the guidance generated by the LLM with MedGellan improves diagnostic performance, particularly in recall and $F_1$ score.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06313v1">ETT: Expanding the Long Context Understanding Capability of LLMs at Test-Time</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Transformer-based Language Models' computation and memory overhead increase quadratically as a function of sequence length. The quadratic cost poses challenges when employing LLMs for processing long sequences. In this work, we introduce \ourmodelacronym~(Extend at Test-Time), method for extending the context length of short context Transformer-based LLMs, with constant memory requirement and linear computation overhead. ETT enable the extension of the context length at test-time by efficient fine-tuning the model's parameters on the input context, chunked into overlapping small subsequences. We evaluate ETT on LongBench by extending the context length of GPT-Large and Phi-2 up to 32 times, increasing from 1k to 32k tokens. This results in up to a 30 percent improvement in the model's accuracy. We also study how context can be stored in LLM's weights effectively and efficiently. Through a detailed ablation study, we examine which Transformer modules are most beneficial to fine-tune at test-time. Interestingly, we find that fine-tuning the second layer of the FFNs is more effective than full fine-tuning, leading to a further improvement in the models' accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06310v1">Too Human to Model:The Uncanny Valley of LLMs in Social Simulation -- When Generative Language Agents Misalign with Modelling Principles</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been increasingly used to build agents in social simulation because of their impressive abilities to generate fluent, contextually coherent dialogues. Such abilities can enhance the realism of models. However, the pursuit of realism is not necessarily compatible with the epistemic foundation of modelling. We argue that LLM agents, in many regards, are too human to model: they are too expressive, detailed and intractable to be consistent with the abstraction, simplification, and interpretability typically demanded by modelling. Through a model-building thought experiment that converts the Bass diffusion model to an LLM-based variant, we uncover five core dilemmas: a temporal resolution mismatch between natural conversation and abstract time steps; the need for intervention in conversations while avoiding undermining spontaneous agent outputs; the temptation to introduce rule-like instructions in prompts while maintaining conversational naturalness; the tension between role consistency and role evolution across time; and the challenge of understanding emergence, where system-level patterns become obscured by verbose micro textual outputs. These dilemmas steer the LLM agents towards an uncanny valley: not abstract enough to clarify underlying social mechanisms, while not natural enough to represent realistic human behaviour. This exposes an important paradox: the realism of LLM agents can obscure, rather than clarify, social dynamics when misapplied. We tease out the conditions in which LLM agents are ideally suited: where system-level emergence is not the focus, linguistic nuances and meaning are central, interactions unfold in natural time, and stable role identity is more important than long-term behavioural evolution. We call for repositioning LLM agents in the ecosystem of social simulation for future applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06274v1">Enhancing LLM Watermark Resilience Against Both Scrubbing and Spoofing Attacks</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Watermarking is a promising defense against the misuse of large language models (LLMs), yet it remains vulnerable to scrubbing and spoofing attacks. This vulnerability stems from an inherent trade-off governed by watermark window size: smaller windows resist scrubbing better but are easier to reverse-engineer, enabling low-cost statistics-based spoofing attacks. This work breaks this trade-off by introducing a novel mechanism, equivalent texture keys, where multiple tokens within a watermark window can independently support the detection. Based on the redundancy, we propose a novel watermark scheme with Sub-vocabulary decomposed Equivalent tExture Key (SEEK). It achieves a Pareto improvement, increasing the resilience against scrubbing attacks without compromising robustness to spoofing. Experiments demonstrate SEEK's superiority over prior method, yielding spoofing robustness gains of +88.2%/+92.3%/+82.0% and scrubbing robustness gains of +10.2%/+6.4%/+24.6% across diverse dataset settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07061v1">An Ensemble Embedding Approach for Improving Semantic Caching Performance in LLM-based Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 10 pages, 8 figures, 2 table. Submitted to the Journal of Information Science
    </div>
    <details class="paper-abstract">
      Semantic caching enhances the efficiency of large language model (LLM) systems by identifying semantically similar queries, storing responses once, and serving them for subsequent equivalent requests. However, existing semantic caching frameworks rely on single embedding models for query representation, which limits their ability to capture the diverse semantic relationships present in real-world query distributions. This paper presents an ensemble embedding approach that combines multiple embedding models through a trained meta-encoder to improve semantic similarity detection in LLM caching systems. We evaluate our method using the Quora Question Pairs (QQP) dataset, measuring cache hit ratios, cache miss ratios, token savings, and response times. Our ensemble approach achieves a 92\% cache hit ratio for semantically equivalent queries while maintaining an 85\% accuracy in correctly rejecting non-equivalent queries as cache misses. These results demonstrate that ensemble embedding methods significantly outperform single-model approaches in distinguishing between semantically similar and dissimilar queries, leading to more effective caching performance and reduced computational overhead in LLM-based systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08027v1">"Amazing, They All Lean Left" -- Analyzing the Political Temperaments of Current LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Recent studies have revealed a consistent liberal orientation in the ethical and political responses generated by most commercial large language models (LLMs), yet the underlying causes and resulting implications remain unclear. This paper systematically investigates the political temperament of seven prominent LLMs - OpenAI's GPT-4o, Anthropic's Claude Sonnet 4, Perplexity (Sonar Large), Google's Gemini 2.5 Flash, Meta AI's Llama 4, Mistral 7b Le Chat and High-Flyer's DeepSeek R1 -- using a multi-pronged approach that includes Moral Foundations Theory, a dozen established political ideology scales and a new index of current political controversies. We find strong and consistent prioritization of liberal-leaning values, particularly care and fairness, across most models. Further analysis attributes this trend to four overlapping factors: Liberal-leaning training corpora, reinforcement learning from human feedback (RLHF), the dominance of liberal frameworks in academic ethical discourse and safety-driven fine-tuning practices. We also distinguish between political "bias" and legitimate epistemic differences, cautioning against conflating the two. A comparison of base and fine-tuned model pairs reveals that fine-tuning generally increases liberal lean, an effect confirmed through both self-report and empirical testing. We argue that this "liberal tilt" is not a programming error or the personal preference of programmers but an emergent property of training on democratic rights-focused discourse. Finally, we propose that LLMs may indirectly echo John Rawls' famous veil-of ignorance philosophical aspiration, reflecting a moral stance unanchored to personal identity or interest. Rather than undermining democratic discourse, this pattern may offer a new lens through which to examine collective reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06223v1">Efficiency-Effectiveness Reranking FLOPs for LLM-based Rerankers</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 under review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently been applied to reranking tasks in information retrieval, achieving strong performance. However, their high computational demands often hinder practical deployment. Existing studies evaluate the efficiency of LLM-based rerankers using proxy metrics such as latency, the number of forward passes, input tokens, and output tokens. However, these metrics depend on hardware and running-time choices (\eg parallel or not, batch size, etc), and often fail to account for model size, making it difficult to interpret and obscuring the evaluation of the efficiency-effectiveness tradeoff. To address this issue, we propose E\textsuperscript{2}R-FLOPs, for LLM-based rerankers: ranking metrics per PetaFLOP (RPP) for relevance per compute and queries per PetaFLOP (QPP) for hardware-agnostic throughput. Companied with the new metrics, an interpretable FLOPs estimator is built to estimate the FLOPs of an LLM-based reranker even without running any experiments. Based on the proposed metrics, we conduct comprehensive experiments to evaluate a wide range of LLM-based rerankers with different architecture, studying the efficiency-effectiveness trade-off and bringing this issue to the attention of the research community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03711v2">Can LLMs Play Ô Ăn Quan Game? A Study of Multi-Step Planning and Decision Making</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 Accepted paper at MAPR 2025
    </div>
    <details class="paper-abstract">
      In this paper, we explore the ability of large language models (LLMs) to plan and make decisions through the lens of the traditional Vietnamese board game, \^O \u{A}n Quan. This game, which involves a series of strategic token movements and captures, offers a unique environment for evaluating the decision-making and strategic capabilities of LLMs. Specifically, we develop various agent personas, ranging from aggressive to defensive, and employ the \^O \u{A}n Quan game as a testbed for assessing LLM performance across different strategies. Through experimentation with models like Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct, and Llama-3.3-70B-Instruct, we aim to understand how these models execute strategic decision-making, plan moves, and manage dynamic game states. The results will offer insights into the strengths and weaknesses of LLMs in terms of reasoning and strategy, contributing to a deeper understanding of their general capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04687v2">LAKEGEN: A LLM-based Tabular Corpus Generator for Evaluating Dataset Discovery in Data Lakes</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      How to generate a large, realistic set of tables along with joinability relationships, to stress-test dataset discovery methods? Dataset discovery methods aim to automatically identify related data assets in a data lake. The development and evaluation of such solutions for customers from a wide range of business domains, relies on diverse, high quality and domain-specific tabular benchmarks. Large language models (LLMs) are trained on a wide variety of text data, which can provide a strong foundation of general and domain-specific knowledge. In this paper, we ask the question -- \textit{can we leverage LLMs to generate a tabular benchmark adequate for evaluating the dataset discovery solutions?} In particular, we focus on the task of finding joinable tables which is the cornerstone of virtually every dataset discovery method. Current corpora for evaluating dataset discovery methods are mainly based on subsets of open data, and they suffer from three important issues: $i)$ they focus on very common and generic data types (e.g., address, id, name, etc.); $ii)$ they do not contain human-annotated column pairs; instead, practitioners synthesize ground truth using table splits (e.g., horizontal for table union search and vertical ones for joinability) and $iii)$ they do not focus on semantic column relationships.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06127v1">PrefixAgent: An LLM-Powered Design Framework for Efficient Prefix Adder Optimization</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Prefix adders are fundamental arithmetic circuits, but their design space grows exponentially with bit-width, posing significant optimization challenges. Previous works face limitations in performance, generalization, and scalability. To address these challenges, we propose PrefixAgent, a large language model (LLM)-powered framework that enables efficient prefix adder optimization. Specifically, PrefixAgent reformulates the problem into subtasks including backbone synthesis and structure refinement, which effectively reduces the search space. More importantly, this new design perspective enables us to efficiently collect enormous high-quality data and reasoning traces with E-graph, which further results in an effective fine-tuning of LLM. Experimental results show that PrefixAgent synthesizes prefix adders with consistently smaller areas compared to baseline methods, while maintaining scalability and generalization in commercial EDA flows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00406v2">Agents Are All You Need for LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 Accepted to COLM 2025
    </div>
    <details class="paper-abstract">
      Information removal or suppression in large language models (LLMs) is a desired functionality, useful in AI regulation, legal compliance, safety, and privacy. LLM unlearning methods aim to remove information on demand from LLMs. Current LLM unlearning methods struggle to balance the unlearning efficacy and utility due to the competing nature of these objectives. Keeping the unlearning process computationally feasible without assuming access to the model weights is an overlooked area. In this work we show that \textit{agents might be all we need for effective and practical inference-time LLM unlearning}. We present the first agentic LLM unlearning (\texttt{ALU}) method, a multi-agent, retrain-free, model-agnostic approach to LLM unlearning that achieves effective unlearning while preserving the utility. Our \texttt{ALU} framework unlearns by involving multiple LLM agents, each designed for a specific step in the unlearning process, without the need to update model weights for any of the agents in the framework. Users can easily request any set of unlearning instances in any sequence, and \texttt{ALU} seamlessly adapts in real time. This is facilitated without requiring any changes in the underlying LLM model. Through extensive experiments on established benchmarks (TOFU, WMDP, WPU) and jailbreaking techniques (many shot, target masking, other languages), we demonstrate that \texttt{ALU} consistently stands out as the most robust inference-time LLM unlearning framework among current state-of-the-art methods while incurring time cost that remains effectively constant regardless of the number of unlearning targets. We further highlight \texttt{ALU}'s superior performance compared to existing methods when evaluated at scale. Specifically, \texttt{ALU} is assessed on up to 1000 unlearning targets, exceeding the evaluation scope of all previously proposed LLM unlearning methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.08324v2">Are LLMs Prescient? A Continuous Evaluation using Daily News as the Oracle</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 ICML 2025
    </div>
    <details class="paper-abstract">
      Many existing evaluation benchmarks for Large Language Models (LLMs) quickly become outdated due to the emergence of new models and training data. These benchmarks also fall short in assessing how LLM performance changes over time, as they consist of a static set of questions without a temporal dimension. To address these limitations, we propose using future event prediction as a continuous evaluation method to assess LLMs' temporal generalization and forecasting abilities. Our benchmark, Daily Oracle, automatically generates question-answer (QA) pairs from daily news, challenging LLMs to predict "future" event outcomes. Our findings reveal that as pre-training data becomes outdated, LLM performance degrades over time. While Retrieval Augmented Generation (RAG) has the potential to enhance prediction accuracy, the performance degradation pattern persists, highlighting the need for continuous model updates. Code and data are available at https://agenticlearning.ai/daily-oracle.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06056v1">Entropy-Memorization Law: Evaluating Memorization Difficulty of Data in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are known to memorize portions of their training data, sometimes reproducing content verbatim when prompted appropriately. In this work, we investigate a fundamental yet under-explored question in the domain of memorization: How to characterize memorization difficulty of training data in LLMs? Through empirical experiments on OLMo, a family of open models, we present the Entropy-Memorization Law. It suggests that data entropy is linearly correlated with memorization score. Moreover, in a case study of memorizing highly randomized strings, or "gibberish", we observe that such sequences, despite their apparent randomness, exhibit unexpectedly low empirical entropy compared to the broader training corpus. Adopting the same strategy to discover Entropy-Memorization Law, we derive a simple yet effective approach to distinguish training and testing data, enabling Dataset Inference (DI).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19652v2">Tailored Conversations beyond LLMs: A RL-Based Dialogue Manager</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      In this work, we propose a novel framework that integrates large language models (LLMs) with an RL-based dialogue manager for open-ended dialogue with a specific goal. By leveraging hierarchical reinforcement learning to model the structured phases of dialogue and employ meta-learning to enhance adaptability across diverse user profiles, our approach enhances adaptability and efficiency, enabling the system to learn from limited data, transition fluidly between dialogue phases, and personalize responses to heterogeneous patient needs. We apply our framework to Motivational Interviews, aiming to foster behavior change, and demonstrate that the proposed dialogue manager outperforms a state-of-the-art LLM baseline in terms of reward, showing a potential benefit of conditioning LLMs to create open-ended dialogue systems with specific goals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06043v1">CAVGAN: Unifying Jailbreak and Defense of LLMs via Generative Adversarial Attacks on their Internal Representations</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Security alignment enables the Large Language Model (LLM) to gain the protection against malicious queries, but various jailbreak attack methods reveal the vulnerability of this security mechanism. Previous studies have isolated LLM jailbreak attacks and defenses. We analyze the security protection mechanism of the LLM, and propose a framework that combines attack and defense. Our method is based on the linearly separable property of LLM intermediate layer embedding, as well as the essence of jailbreak attack, which aims to embed harmful problems and transfer them to the safe area. We utilize generative adversarial network (GAN) to learn the security judgment boundary inside the LLM to achieve efficient jailbreak attack and defense. The experimental results indicate that our method achieves an average jailbreak success rate of 88.85\% across three popular LLMs, while the defense success rate on the state-of-the-art jailbreak dataset reaches an average of 84.17\%. This not only validates the effectiveness of our approach but also sheds light on the internal security mechanisms of LLMs, offering new insights for enhancing model security The code and data are available at https://github.com/NLPGM/CAVGAN.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05984v1">Development and Evaluation of HopeBot: an LLM-based chatbot for structured and interactive PHQ-9 depression screening</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Static tools like the Patient Health Questionnaire-9 (PHQ-9) effectively screen depression but lack interactivity and adaptability. We developed HopeBot, a chatbot powered by a large language model (LLM) that administers the PHQ-9 using retrieval-augmented generation and real-time clarification. In a within-subject study, 132 adults in the United Kingdom and China completed both self-administered and chatbot versions. Scores demonstrated strong agreement (ICC = 0.91; 45% identical). Among 75 participants providing comparative feedback, 71% reported greater trust in the chatbot, highlighting clearer structure, interpretive guidance, and a supportive tone. Mean ratings (0-10) were 8.4 for comfort, 7.7 for voice clarity, 7.6 for handling sensitive topics, and 7.4 for recommendation helpfulness; the latter varied significantly by employment status and prior mental-health service use (p < 0.05). Overall, 87.1% expressed willingness to reuse or recommend HopeBot. These findings demonstrate voice-based LLM chatbots can feasibly serve as scalable, low-burden adjuncts for routine depression screening.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05886v1">Current Practices for Building LLM-Powered Reasoning Tools Are Ad Hoc -- and We Can Do Better</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 6 pages, 4 figures
    </div>
    <details class="paper-abstract">
      There is growing excitement about building software verifiers, synthesizers, and other Automated Reasoning (AR) tools by combining traditional symbolic algorithms and Large Language Models (LLMs). Unfortunately, the current practice for constructing such neurosymbolic AR systems is an ad hoc programming model that does not have the strong guarantees of traditional symbolic algorithms, nor a deep enough synchronization of neural networks and symbolic reasoning to unlock the full potential of LLM-powered reasoning. I propose Neurosymbolic Transition Systems as a principled computational model that can underlie infrastructure for building neurosymbolic AR tools. In this model, symbolic state is paired with intuition, and state transitions operate over symbols and intuition in parallel. I argue why this new paradigm can scale logical reasoning beyond current capabilities while retaining the strong guarantees of symbolic algorithms, and I sketch out how the computational model I propose can be reified in a logic programming language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05880v1">RecRankerEval: A Flexible and Extensible Framework for Top-k LLM-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      A recent Large language model (LLM)-based recommendation model, called RecRanker, has demonstrated a superior performance in the top-k recommendation task compared to other models. In particular, RecRanker samples users via clustering, generates an initial ranking list using an initial recommendation model, and fine-tunes an LLM through hybrid instruction tuning to infer user preferences. However, the contribution of each core component remains underexplored. In this work, we inspect the reproducibility of RecRanker, and study the impact and role of its various components. We begin by reproducing the RecRanker pipeline through the implementation of all its key components. Our reproduction shows that the pairwise and listwise methods achieve a performance comparable to that reported in the original paper. For the pointwise method, while we are also able to reproduce the original paper's results, further analysis shows that the performance is abnormally high due to data leakage from the inclusion of ground-truth information in the prompts. To enable a fair and comprehensive evaluation of LLM-based top-k recommendations, we propose RecRankerEval, an extensible framework that covers five key dimensions: user sampling strategy, initial recommendation model, LLM backbone, dataset selection, and instruction tuning method. Using the RecRankerEval framework, we show that the original results of RecRanker can be reproduced on the ML-100K and ML-1M datasets, as well as the additional Amazon-Music dataset, but not on BookCrossing due to the lack of timestamp information in the original RecRanker paper. Furthermore, we demonstrate that RecRanker's performance can be improved by employing alternative user sampling methods, stronger initial recommenders, and more capable LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18099v2">Learning to Plan & Reason for Evaluation with Thinking-LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 ICML 2025
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge models generate chain-of-thought (CoT) sequences intended to capture the step-bystep reasoning process that underlies the final evaluation of a response. However, due to the lack of human annotated CoTs for evaluation, the required components and structure of effective reasoning traces remain understudied. Consequently, previous approaches often (1) constrain reasoning traces to hand-designed components, such as a list of criteria, reference answers, or verification questions and (2) structure them such that planning is intertwined with the reasoning for evaluation. In this work, we propose EvalPlanner, a preference optimization algorithm for Thinking-LLM-as-a-Judge that first generates an unconstrained evaluation plan, followed by its execution, and then the final judgment. In a self-training loop, EvalPlanner iteratively optimizes over synthetically constructed evaluation plans and executions, leading to better final verdicts. Our method achieves a new state-of-the-art performance for generative reward models on RewardBench (with a score of 93.9), despite being trained on fewer amount of, and synthetically generated, preference pairs. Additional experiments on other benchmarks like RM-Bench, JudgeBench, and FollowBenchEval further highlight the utility of both planning and reasoning for building robust LLM-as-a-Judge reasoning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05822v1">Video Event Reasoning and Prediction by Fusing World Knowledge from LLMs with Vision Foundation Models</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 22 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Current video understanding models excel at recognizing "what" is happening but fall short in high-level cognitive tasks like causal reasoning and future prediction, a limitation rooted in their lack of commonsense world knowledge. To bridge this cognitive gap, we propose a novel framework that synergistically fuses a powerful Vision Foundation Model (VFM) for deep visual perception with a Large Language Model (LLM) serving as a knowledge-driven reasoning core. Our key technical innovation is a sophisticated fusion module, inspired by the Q-Former architecture, which distills complex spatiotemporal and object-centric visual features into a concise, language-aligned representation. This enables the LLM to effectively ground its inferential processes in direct visual evidence. The model is trained via a two-stage strategy, beginning with large-scale alignment pre-training on video-text data, followed by targeted instruction fine-tuning on a curated dataset designed to elicit advanced reasoning and prediction skills. Extensive experiments demonstrate that our model achieves state-of-the-art performance on multiple challenging benchmarks. Notably, it exhibits remarkable zero-shot generalization to unseen reasoning tasks, and our in-depth ablation studies validate the critical contribution of each architectural component. This work pushes the boundary of machine perception from simple recognition towards genuine cognitive understanding, paving the way for more intelligent and capable AI systems in robotics, human-computer interaction, and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05820v1">Constella: Supporting Storywriters' Interconnected Character Creation through LLM-based Multi-Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 50 pages
    </div>
    <details class="paper-abstract">
      Creating a cast of characters by attending to their relational dynamics is a critical aspect of most long-form storywriting. However, our formative study (N=14) reveals that writers struggle to envision new characters that could influence existing ones, to balance similarities and differences among characters, and to intricately flesh out their relationships. Based on these observations, we designed Constella, an LLM-based multi-agent tool that supports storywriters' interconnected character creation process. Constella suggests related characters (FRIENDS DISCOVERY feature), reveals the inner mindscapes of several characters simultaneously (JOURNALS feature), and manifests relationships through inter-character responses (COMMENTS feature). Our 7-8 day deployment study with storywriters (N=11) shows that Constella enabled the creation of expansive communities composed of related characters, facilitated the comparison of characters' thoughts and emotions, and deepened writers' understanding of character relationships. We conclude by discussing how multi-agent interactions can help distribute writers' attention and effort across the character cast.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04841v1">Spec-TOD: A Specialized Instruction-Tuned LLM Framework for Efficient Task-Oriented Dialogue Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 Accepted at SIGdial 2025
    </div>
    <details class="paper-abstract">
      Task-oriented dialogue (TOD) systems facilitate goal-driven interactions between users and machines. While recent advances in deep learning have improved the performance, TOD systems often struggle in low-resource scenarios with limited labeled data. To address this challenge, we propose Spec-TOD, a novel framework designed to train an end-to-end TOD system with limited data. Spec-TOD introduces two main innovations: (i) a novel specialized end-to-end TOD framework that incorporates explicit task instructions for instruction-tuned large language models (LLMs), and (ii) an efficient training strategy that leverages lightweight, specialized LLMs to achieve strong performance with minimal supervision. Experiments on the MultiWOZ dataset, a widely used TOD benchmark, demonstrate that Spec-TOD achieves competitive results while significantly reducing the need for labeled data. These findings highlight the potential of the proposed framework in advancing efficient and effective TOD systems in low-resource settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05853v2">Training-Free Query Optimization via LLM-Based Plan Similarity</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 18 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large language model (LLM) embeddings offer a promising new avenue for database query optimization. In this paper, we explore how pre-trained execution plan embeddings can guide SQL query execution without the need for additional model training. We introduce LLM-PM (LLM-based Plan Mapping), a framework that embeds the default execution plan of a query, finds its k nearest neighbors among previously executed plans, and recommends database hintsets based on neighborhood voting. A lightweight consistency check validates the selected hint, while a fallback mechanism searches the full hint space when needed. Evaluated on the JOB-CEB benchmark using OpenGauss, LLM-PM achieves an average speed-up of 21% query latency reduction. This work highlights the potential of LLM-powered embeddings to deliver practical improvements in query performance and opens new directions for training-free, embedding-based optimizer guidance systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04766v1">ABench-Physics: Benchmarking Physical Reasoning in LLMs via High-Difficulty and Dynamic Physics Problems</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive performance in domains such as mathematics and programming, yet their capabilities in physics remain underexplored and poorly understood. Physics poses unique challenges that demand not only precise computation but also deep conceptual understanding and physical modeling skills. Existing benchmarks often fall short due to limited difficulty, multiple-choice formats, and static evaluation settings that fail to capture physical modeling ability. In this paper, we introduce ABench-Physics, a novel benchmark designed to rigorously evaluate LLMs' physical reasoning and generalization capabilities. ABench-Physics consists of two components: Phy_A, a static set of 400 graduate- or Olympiad-level problems; and Phy_B, a dynamic subset of 100 problems equipped with an automatic variation engine to test model robustness across changing conditions. All questions require precise numerical answers, with strict formatting and tolerance constraints. Our evaluation of several state-of-the-art LLMs reveals substantial performance gaps, highlighting persistent limitations in physical reasoning, especially in generalization to dynamic variants. ABench-Physics provides a challenging and diagnostic framework for advancing scientific reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04751v1">LLMs as Architects and Critics for Multi-Source Opinion Summarization</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      Multi-source Opinion Summarization (M-OS) extends beyond traditional opinion summarization by incorporating additional sources of product metadata such as descriptions, key features, specifications, and ratings, alongside reviews. This integration results in comprehensive summaries that capture both subjective opinions and objective product attributes essential for informed decision-making. While Large Language Models (LLMs) have shown significant success in various Natural Language Processing (NLP) tasks, their potential in M-OS remains largely unexplored. Additionally, the lack of evaluation datasets for this task has impeded further advancements. To bridge this gap, we introduce M-OS-EVAL, a benchmark dataset for evaluating multi-source opinion summaries across 7 key dimensions: fluency, coherence, relevance, faithfulness, aspect coverage, sentiment consistency, specificity. Our results demonstrate that M-OS significantly enhances user engagement, as evidenced by a user study in which, on average, 87% of participants preferred M-OS over opinion summaries. Our experiments demonstrate that factually enriched summaries enhance user engagement. Notably, M-OS-PROMPTS exhibit stronger alignment with human judgment, achieving an average Spearman correlation of \r{ho} = 0.74, which surpasses the performance of previous methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04748v1">LLM-based Question-Answer Framework for Sensor-driven HVAC System Interaction</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      Question-answering (QA) interfaces powered by large language models (LLMs) present a promising direction for improving interactivity with HVAC system insights, particularly for non-expert users. However, enabling accurate, real-time, and context-aware interactions with HVAC systems introduces unique challenges, including the integration of frequently updated sensor data, domain-specific knowledge grounding, and coherent multi-stage reasoning. In this paper, we present JARVIS, a two-stage LLM-based QA framework tailored for sensor data-driven HVAC system interaction. JARVIS employs an Expert-LLM to translate high-level user queries into structured execution instructions, and an Agent that performs SQL-based data retrieval, statistical processing, and final response generation. To address HVAC-specific challenges, JARVIS integrates (1) an adaptive context injection strategy for efficient HVAC and deployment-specific information integration, (2) a parameterized SQL builder and executor to improve data access reliability, and (3) a bottom-up planning scheme to ensure consistency across multi-stage response generation. We evaluate JARVIS using real-world data collected from a commercial HVAC system and a ground truth QA dataset curated by HVAC experts to demonstrate its effectiveness in delivering accurate and interpretable responses across diverse queries. Results show that JARVIS consistently outperforms baseline and ablation variants in both automated and user-centered assessments, achieving high response quality and accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04736v1">ChipSeek-R1: Generating Human-Surpassing RTL with LLM via Hierarchical Reward-Driven Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show significant potential for automating Register-Transfer Level (RTL) code generation. However, current approaches face a critical challenge: they can not simultaneously optimize for functional correctness and hardware quality (Power, Performance, Area - PPA). Methods based on supervised fine-tuning often generate functionally correct but PPA-suboptimal code, lacking mechanisms to learn optimization principles. In contrast, post-processing techniques that attempt to improve PPA metrics after generation are often inefficient because they operate externally without updating the LLM's parameters, thus failing to enhance the model's intrinsic design capabilities. To bridge this gap, we introduce ChipSeek-R1, a hierarchical reward-driven reinforcement learning framework to train LLMs to generate RTL code that achieves both functional correctness and optimized PPA metrics. ChipSeek-R1 employs a hierarchical reward system, which incorporates direct feedback on syntax, functional correctness (from simulators) and PPA metrics (from synthesis tools) during reinforcement learning. This enables the model to learn complex hardware design trade-offs via trial-and-error, generating RTL code that is both functionally correct and PPA-optimized. Evaluating ChipSeek-R1 on standard benchmarks (VerilogEval, RTLLM), we achieve state-of-the-art results in functional correctness. Notably, on the RTLLM benchmark, ChipSeek-R1 generated 27 RTL designs surpassing the PPA metrics of the original human-written code. Our findings demonstrate the effectiveness of integrating toolchain feedback into LLM training and highlight the potential for reinforcement learning to enable automated generation of human-surpassing RTL code. We open-source our code in anonymous github.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22139v2">Q-Frame: Query-aware Frame Selection and Multi-Resolution Adaptation for Video-LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 Accepted at ICCV 2025
    </div>
    <details class="paper-abstract">
      Multimodal Large Language Models (MLLMs) have demonstrated significant success in visual understanding tasks. However, challenges persist in adapting these models for video comprehension due to the large volume of data and temporal complexity. Existing Video-LLMs using uniform frame sampling often struggle to capture the query-related crucial spatiotemporal clues of videos effectively. In this paper, we introduce Q-Frame, a novel approach for adaptive frame selection and multi-resolution scaling tailored to the video's content and the specific query. Q-Frame employs a training-free, plug-and-play strategy generated by a text-image matching network like CLIP, utilizing the Gumbel-Max trick for efficient frame selection. Q-Frame allows Video-LLMs to process more frames without exceeding computational limits, thereby preserving critical temporal and spatial information. We demonstrate Q-Frame's effectiveness through extensive experiments on benchmark datasets, including MLVU, LongVideoBench, and Video-MME, illustrating its superiority over existing methods and its applicability across various video understanding tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04724v1">Who's the Mole? Modeling and Detecting Intention-Hiding Malicious Agents in LLM-Based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      Multi-agent systems powered by Large Language Models (LLM-MAS) demonstrate remarkable capabilities in collaborative problem-solving. While LLM-MAS exhibit strong collaborative abilities, the security risks in their communication and coordination remain underexplored. We bridge this gap by systematically investigating intention-hiding threats in LLM-MAS, and design four representative attack paradigms that subtly disrupt task completion while maintaining high concealment. These attacks are evaluated in centralized, decentralized, and layered communication structures. Experiments conducted on six benchmark datasets, including MMLU, MMLU-Pro, HumanEval, GSM8K, arithmetic, and biographies, demonstrate that they exhibit strong disruptive capabilities. To identify these threats, we propose a psychology-based detection framework AgentXposed, which combines the HEXACO personality model with the Reid Technique, using progressive questionnaire inquiries and behavior-based monitoring. Experiments conducted on six types of attacks show that our detection framework effectively identifies all types of malicious behaviors. The detection rate for our intention-hiding attacks is slightly lower than that of the two baselines, Incorrect Fact Injection and Dark Traits Injection, demonstrating the effectiveness of intention concealment. Our findings reveal the structural and behavioral risks posed by intention-hiding attacks and offer valuable insights into securing LLM-based multi-agent systems through psychological perspectives, which contributes to a deeper understanding of multi-agent safety. The code and data are available at https://anonymous.4open.science/r/AgentXposed-F814.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.12112v4">Balancing Act: Prioritization Strategies for LLM-Designed Restless Bandit Rewards</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      LLMs are increasingly used to design reward functions based on human preferences in Reinforcement Learning (RL). We focus on LLM-designed rewards for Restless Multi-Armed Bandits, a framework for allocating limited resources among agents. In applications such as public health, this approach empowers grassroots health workers to tailor automated allocation decisions to community needs. In the presence of multiple agents, altering the reward function based on human preferences can impact subpopulations very differently, leading to complex tradeoffs and a multi-objective resource allocation problem. We are the first to present a principled method termed Social Choice Language Model for dealing with these tradeoffs for LLM-designed rewards for multiagent planners in general and restless bandits in particular. The novel part of our model is a transparent and configurable selection component, called an adjudicator, external to the LLM that controls complex tradeoffs via a user-selected social welfare function. Our experiments demonstrate that our model reliably selects more effective, aligned, and balanced reward functions compared to purely LLM-based approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04687v1">AKEGEN: A LLM-based Tabular Corpus Generator for Evaluating Dataset Discovery in Data Lakes</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      How to generate a large, realistic set of tables along with joinability relationships, to stress-test dataset discovery methods? Dataset discovery methods aim to automatically identify related data assets in a data lake. The development and evaluation of such solutions for customers from a wide range of business domains, relies on diverse, high quality and domain-specific tabular benchmarks. Large language models (LLMs) are trained on a wide variety of text data, which can provide a strong foundation of general and domain-specific knowledge. In this paper, we ask the question -- \textit{can we leverage LLMs to generate a tabular benchmark adequate for evaluating the dataset discovery solutions?} In particular, we focus on the task of finding joinable tables which is the cornerstone of virtually every dataset discovery method. Current corpora for evaluating dataset discovery methods are mainly based on subsets of open data, and they suffer from three important issues: $i)$ they focus on very common and generic data types (e.g., address, id, name, etc.); $ii)$ they do not contain human-annotated column pairs; instead, practitioners synthesize ground truth using table splits (e.g., horizontal for table union search and vertical ones for joinability) and $iii)$ they do not focus on semantic column relationships.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04664v1">VectorLLM: Human-like Extraction of Structured Building Contours vis Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      Automatically extracting vectorized building contours from remote sensing imagery is crucial for urban planning, population estimation, and disaster assessment. Current state-of-the-art methods rely on complex multi-stage pipelines involving pixel segmentation, vectorization, and polygon refinement, which limits their scalability and real-world applicability. Inspired by the remarkable reasoning capabilities of Large Language Models (LLMs), we introduce VectorLLM, the first Multi-modal Large Language Model (MLLM) designed for regular building contour extraction from remote sensing images. Unlike existing approaches, VectorLLM performs corner-point by corner-point regression of building contours directly, mimicking human annotators' labeling process. Our architecture consists of a vision foundation backbone, an MLP connector, and an LLM, enhanced with learnable position embeddings to improve spatial understanding capability. Through comprehensive exploration of training strategies including pretraining, supervised fine-tuning, and preference optimization across WHU, WHU-Mix, and CrowdAI datasets, VectorLLM significantly outperformed the previous SOTA methods by 5.6 AP, 7.1 AP, 13.6 AP, respectively in the three datasets. Remarkably, VectorLLM exhibits strong zero-shot performance on unseen objects including aircraft, water bodies, and oil tanks, highlighting its potential for unified modeling of diverse remote sensing object contour extraction tasks. Overall, this work establishes a new paradigm for vector extraction in remote sensing, leveraging the topological reasoning capabilities of LLMs to achieve both high accuracy and exceptional generalization. All the codes and weights will be published for promoting community development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04626v1">Heterogeneous User Modeling for LLM-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 Accepted by RecSys 2025
    </div>
    <details class="paper-abstract">
      Leveraging Large Language Models (LLMs) for recommendation has demonstrated notable success in various domains, showcasing their potential for open-domain recommendation. A key challenge to advancing open-domain recommendation lies in effectively modeling user preferences from users' heterogeneous behaviors across multiple domains. Existing approaches, including ID-based and semantic-based modeling, struggle with poor generalization, an inability to compress noisy interactions effectively, and the domain seesaw phenomenon. To address these challenges, we propose a Heterogeneous User Modeling (HUM) method, which incorporates a compression enhancer and a robustness enhancer for LLM-based recommendation. The compression enhancer uses a customized prompt to compress heterogeneous behaviors into a tailored token, while a masking mechanism enhances cross-domain knowledge extraction and understanding. The robustness enhancer introduces a domain importance score to mitigate the domain seesaw phenomenon by guiding domain optimization. Extensive experiments on heterogeneous datasets validate that HUM effectively models user heterogeneity by achieving both high efficacy and robustness, leading to superior performance in open-domain recommendation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.17378v4">A Text is Worth Several Tokens: Text Embedding from LLMs Secretly Aligns Well with The Key Tokens</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 ACL2025 Oral
    </div>
    <details class="paper-abstract">
      Text embeddings from large language models (LLMs) have achieved excellent results in tasks such as information retrieval, semantic textual similarity, etc. In this work, we show an interesting finding: when feeding a text into the LLM-based embedder, the obtained text embedding will be able to be aligned with the key tokens in the input text. We first fully analyze this phenomenon on eight LLM-based embedders and show that this phenomenon is universal and is not affected by model architecture, training strategy, and embedding method. With a deeper analysis, we find that the main change in embedding space between these embedders and their LLM backbones is in the first principal component. By adjusting the first principal component, we can align text embedding with the key tokens. Finally, we give several examples to demonstrate the vast application potential of this finding: (1) we propose a simple and practical sparse retrieval method based on the aligned tokens, which can achieve 80% of the dense retrieval effect of the same model while reducing the computation significantly; (2) we show that our findings provide a novel perspective to help understand novel technologies (e.g., instruction-following embedding) and fuzzy concepts (e.g., semantic relatedness vs. similarity) in this field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04623v1">Hierarchical Intent-guided Optimization with Pluggable LLM-Driven Semantics for Session-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      Session-based Recommendation (SBR) aims to predict the next item a user will likely engage with, using their interaction sequence within an anonymous session. Existing SBR models often focus only on single-session information, ignoring inter-session relationships and valuable cross-session insights. Some methods try to include inter-session data but struggle with noise and irrelevant information, reducing performance. Additionally, most models rely on item ID co-occurrence and overlook rich semantic details, limiting their ability to capture fine-grained item features. To address these challenges, we propose a novel hierarchical intent-guided optimization approach with pluggable LLM-driven semantic learning for session-based recommendations, called HIPHOP. First, we introduce a pluggable embedding module based on large language models (LLMs) to generate high-quality semantic representations, enhancing item embeddings. Second, HIPHOP utilizes graph neural networks (GNNs) to model item transition relationships and incorporates a dynamic multi-intent capturing module to address users' diverse interests within a session. Additionally, we design a hierarchical inter-session similarity learning module, guided by user intent, to capture global and local session relationships, effectively exploring users' long-term and short-term interests. To mitigate noise, an intent-guided denoising strategy is applied during inter-session learning. Finally, we enhance the model's discriminative capability by using contrastive learning to optimize session representations. Experiments on multiple datasets show that HIPHOP significantly outperforms existing methods, demonstrating its effectiveness in improving recommendation quality. Our code is available: https://github.com/hjx159/HIPHOP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04610v1">any4: Learned 4-bit Numeric Representation for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 ICML 2025
    </div>
    <details class="paper-abstract">
      We present any4, a learned 4-bit weight quantization solution for large language models (LLMs) providing arbitrary numeric representations without requiring pre-processing of weights or activations. any4 yields higher accuracy compared to other related 4-bit numeric representation types: int4, fp4 and nf4, as evaluated on a range of model sizes, generations and families (Llama 2, Llama 3, Mistral and Mixtral). While any4 does not require preprocessing of weights or activations, it is also competitive with orthogonal techniques that require such preprocessing (e.g., AWQ and GPTQ). We also experiment with any3 and any2 and show competitiveness at lower bits. Additionally, we show that we can calibrate using a single curated diverse sample rather than hundreds of samples from a dataset as done in most quantization approaches. We also open source tinygemm, a latency optimized GPU matrix multiplication library for LLMs, that implements any4 using a GPU-efficient lookup table strategy along with other common quantization methods. We open source our code at https://github.com/facebookresearch/any4 .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20666v2">Inside you are many wolves: Using cognitive models to interpret value trade-offs in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 11 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Navigating everyday social situations often requires juggling conflicting goals, such as conveying a harsh truth, maintaining trust, all while still being mindful of another person's feelings. These value trade-offs are an integral part of human decision-making and language use, however, current tools for interpreting such dynamic and multi-faceted notions of values in LLMs are limited. In cognitive science, so-called "cognitive models" provide formal accounts of these trade-offs in humans, by modeling the weighting of a speaker's competing utility functions in choosing an action or utterance. In this work, we use a leading cognitive model of polite speech to interpret the extent to which LLMs represent human-like trade-offs. We apply this lens to systematically evaluate value trade-offs in two encompassing model settings: degrees of reasoning "effort" in frontier black-box models, and RL post-training dynamics of open-source models. Our results highlight patterns of higher informational utility than social utility in reasoning models, and in open-source models shown to be stronger in mathematical reasoning. Our findings from LLMs' training dynamics suggest large shifts in utility values early on in training with persistent effects of the choice of base model and pretraining data, compared to feedback dataset or alignment method. We show that our method is responsive to diverse aspects of the rapidly evolving LLM landscape, with insights for forming hypotheses about other high-level behaviors, shaping training regimes for reasoning models, and better controlling trade-offs between values during model training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05541v1">SenseCF: LLM-Prompted Counterfactuals for Intervention and Sensor Data Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 In review
    </div>
    <details class="paper-abstract">
      Counterfactual explanations (CFs) offer human-centric insights into machine learning predictions by highlighting minimal changes required to alter an outcome. Therefore, CFs can be used as (i) interventions for abnormality prevention and (ii) augmented data for training robust models. In this work, we explore large language models (LLMs), specifically GPT-4o-mini, for generating CFs in a zero-shot and three-shot setting. We evaluate our approach on two datasets: the AI-Readi flagship dataset for stress prediction and a public dataset for heart disease detection. Compared to traditional methods such as DiCE, CFNOW, and NICE, our few-shot LLM-based approach achieves high plausibility (up to 99%), strong validity (up to 0.99), and competitive sparsity. Moreover, using LLM-generated CFs as augmented samples improves downstream classifier performance (an average accuracy gain of 5%), especially in low-data regimes. This demonstrates the potential of prompt-based generative techniques to enhance explainability and robustness in clinical and physiological prediction tasks. Code base: github.com/anonymous/SenseCF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18116v3">Bayesian Optimization for Controlled Image Editing via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 8 figures, accept at ACL2025 Findings
    </div>
    <details class="paper-abstract">
      In the rapidly evolving field of image generation, achieving precise control over generated content and maintaining semantic consistency remain significant limitations, particularly concerning grounding techniques and the necessity for model fine-tuning. To address these challenges, we propose BayesGenie, an off-the-shelf approach that integrates Large Language Models (LLMs) with Bayesian Optimization to facilitate precise and user-friendly image editing. Our method enables users to modify images through natural language descriptions without manual area marking, while preserving the original image's semantic integrity. Unlike existing techniques that require extensive pre-training or fine-tuning, our approach demonstrates remarkable adaptability across various LLMs through its model-agnostic design. BayesGenie employs an adapted Bayesian optimization strategy to automatically refine the inference process parameters, achieving high-precision image editing with minimal user intervention. Through extensive experiments across diverse scenarios, we demonstrate that our framework significantly outperforms existing methods in both editing accuracy and semantic preservation, as validated using different LLMs including Claude3 and GPT-4.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05528v1">Conversational Education at Scale: A Multi-LLM Agent Workflow for Procedural Learning and Pedagogic Quality Assessment</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have advanced virtual educators and learners, bridging NLP with AI4Education. Existing work often lacks scalability and fails to leverage diverse, large-scale course content, with limited frameworks for assessing pedagogic quality. To this end, we propose WikiHowAgent, a multi-agent workflow leveraging LLMs to simulate interactive teaching-learning conversations. It integrates teacher and learner agents, an interaction manager, and an evaluator to facilitate procedural learning and assess pedagogic quality. We introduce a dataset of 114,296 teacher-learner conversations grounded in 14,287 tutorials across 17 domains and 727 topics. Our evaluation protocol combines computational and rubric-based metrics with human judgment alignment. Results demonstrate the workflow's effectiveness in diverse setups, offering insights into LLM capabilities across domains. Our datasets and implementations are fully open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05504v1">Tool for Supporting Debugging and Understanding of Normative Requirements Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      Normative requirements specify social, legal, ethical, empathetic, and cultural (SLEEC) norms that must be observed by a system. To support the identification of SLEEC requirements, numerous standards and regulations have been developed. These requirements are typically defined by stakeholders in the non-technical system with diverse expertise (e.g., ethicists, lawyers, social scientists). Hence, ensuring their consistency and managing the requirement elicitation process are complex and error-prone tasks. Recent research has addressed this challenge using domain-specific languages to specify normative requirements as rules, whose consistency can then be analyzed with formal methods. Nevertheless, these approaches often present the results from formal verification tools in a way that is inaccessible to non-technical users. This hinders understanding and makes the iterative process of eliciting and validating these requirements inefficient in terms of both time and effort. To address this problem, we introduce SLEEC-LLM, a tool that uses large language models (LLMs) to provide natural-language interpretations for model-checking counterexamples corresponding to SLEEC rule inconsistencies. SLEEC-LLM improves the efficiency and explainability of normative requirements elicitation and consistency analysis. To demonstrate its effectiveness, we summarise its use in two real-world case studies involving non-technical stakeholders.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05461v1">GLOSS: Group of LLMs for Open-Ended Sensemaking of Passive Sensing Data for Health and Wellbeing</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      The ubiquitous presence of smartphones and wearables has enabled researchers to build prediction and detection models for various health and behavior outcomes using passive sensing data from these devices. Achieving a high-level, holistic understanding of an individual's behavior and context, however, remains a significant challenge. Due to the nature of passive sensing data, sensemaking -- the process of interpreting and extracting insights -- requires both domain knowledge and technical expertise, creating barriers for different stakeholders. Existing systems designed to support sensemaking are either not open-ended or cannot perform complex data triangulation. In this paper, we present a novel sensemaking system, Group of LLMs for Open-ended Sensemaking (GLOSS), capable of open-ended sensemaking and performing complex multimodal triangulation to derive insights. We demonstrate that GLOSS significantly outperforms the commonly used Retrieval-Augmented Generation (RAG) technique, achieving 87.93% accuracy and 66.19% consistency, compared to RAG's 29.31% accuracy and 52.85% consistency. Furthermore, we showcase the promise of GLOSS through four use cases inspired by prior and ongoing work in the UbiComp and HCI communities. Finally, we discuss the potential of GLOSS, its broader implications, and the limitations of our work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.04798v3">From LLMs to Actions: Latent Codes as Bridges in Hierarchical Robot Control</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      Hierarchical control for robotics has long been plagued by the need to have a well defined interface layer to communicate between high-level task planners and low-level policies. With the advent of LLMs, language has been emerging as a prospective interface layer. However, this has several limitations. Not all tasks can be decomposed into steps that are easily expressible in natural language (e.g. performing a dance routine). Further, it makes end-to-end finetuning on embodied data challenging due to domain shift and catastrophic forgetting. We introduce our method -- Learnable Latent Codes as Bridges (LCB) -- as an alternate architecture to overcome these limitations. \method~uses a learnable latent code to act as a bridge between LLMs and low-level policies. This enables LLMs to flexibly communicate goals in the task plan without being entirely constrained by language limitations. Additionally, it enables end-to-end finetuning without destroying the embedding space of word tokens learned during pre-training. Through experiments on Language Table and Calvin, two common language based benchmarks for embodied agents, we find that \method~outperforms baselines (including those w/ GPT-4V) that leverage pure language as the interface layer on tasks that require reasoning and multi-step behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17172v2">What Would You Ask When You First Saw $a^2+b^2=c^2$? Evaluating LLM on Curiosity-Driven Questioning</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can store a massive amount of knowledge, yet their potential to acquire new knowledge remains unknown. We propose a novel evaluation framework that evaluates this capability. This framework prompts LLMs to generate questions about a statement introducing scientific knowledge, simulating a curious person when facing the statement for the first time. We score the qualities of the generated questions, thereby evaluating the knowledge acquisition potential of the LLM. We apply controlled ablation studies to validate our scoring procedures. Additionally, we created a synthetic dataset consisting of 1101 statements in physics, chemistry, and maths with distinct levels of difficulties, 300 general knowledge statements, and 567 incorrect statements. Human evaluations were conducted to validate our model assessments, achieving an approximate weighted Cohen's kappa of 0.7 on all three metrics considered. We find that while large models like GPT-4 and Mistral 8x7b are adept at generating coherent and relevant questions, the smaller Phi-2 model is equally or more effective. This indicates that size does not solely determine a model's knowledge acquisition potential. The proposed framework quantifies a critical model capability that was commonly overlooked and opens up research opportunities for developing more knowledgeable AI systems
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05403v1">PBE Meets LLM: When Few Examples Aren't Few-Shot Enough</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 7 pages, 5 figures, accepted by VLDB QDB'25 workshop
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can generate code from natural language descriptions. Their performance is typically evaluated using programming benchmarks that simulate real-world tasks. These benchmarks provide specifications in the form of docstrings, function signatures, or bug reports. The model then generates a program, which is tested against predefined test cases. In contrast, Programming by Example (PBE) uses input-output examples as the specification. Traditional PBE systems rely on search-based methods over restricted transformation spaces. They are usually designed for narrow domains and fixed input formats. It remains unclear how well LLMs perform on PBE tasks. In this work, we evaluate LLMs on PBE tasks involving tabular data transformations. We prompt models to generate functions that convert an input table to an output table. We test the generated functions on unseen inputs to measure accuracy. Our study includes multiple LLMs and evaluates different prompting strategies, such as one-shot vs. multi-try. We also compare performance with and without PBE-specific knowledge. Finally, we propose a hybrid method that calls a traditional PBE solver first, and then falls back to LLMs if necessary. Our results show that LLMs support more diverse input formats and achieve higher accuracy than conventional methods. However, they struggle with tasks that contain ambiguity. The hybrid approach improves overall success by combining the strengths of both approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05330v1">MindFlow: Revolutionizing E-commerce Customer Support with Multimodal LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled new applications in e-commerce customer service. However, their capabilities remain constrained in complex, multimodal scenarios. We present MindFlow, the first open-source multimodal LLM agent tailored for e-commerce. Built on the CoALA framework, it integrates memory, decision-making, and action modules, and adopts a modular "MLLM-as-Tool" strategy for effect visual-textual reasoning. Evaluated via online A/B testing and simulation-based ablation, MindFlow demonstrates substantial gains in handling complex queries, improving user satisfaction, and reducing operational costs, with a 93.53% relative improvement observed in real-world deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11679v2">LLMs on support of privacy and security of mobile apps: state of the art and research directions</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 I am writing to respectfully request the withdrawal of my recent submission to arXiv due to an authorship issue. The paper was submitted without the explicit consent of two co-authors. After internal discussion, they have expressed clear disagreement with the submission and raised concerns about unresolved academic inaccuracies in the current version
    </div>
    <details class="paper-abstract">
      Modern life has witnessed the explosion of mobile devices. However, besides the valuable features that bring convenience to end users, security and privacy risks still threaten users of mobile apps. The increasing sophistication of these threats in recent years has underscored the need for more advanced and efficient detection approaches. In this chapter, we explore the application of Large Language Models (LLMs) to identify security risks and privacy violations and mitigate them for the mobile application ecosystem. By introducing state-of-the-art research that applied LLMs to mitigate the top 10 common security risks of smartphone platforms, we highlight the feasibility and potential of LLMs to replace traditional analysis methods, such as dynamic and hybrid analysis of mobile apps. As a representative example of LLM-based solutions, we present an approach to detect sensitive data leakage when users share images online, a common behavior of smartphone users nowadays. Finally, we discuss open research challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05305v1">Narrowing the Gap: Supervised Fine-Tuning of Open-Source LLMs as a Viable Alternative to Proprietary Models for Pedagogical Tools</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 7 pages, 3 tables, 1 figure
    </div>
    <details class="paper-abstract">
      Frontier Large language models (LLMs) like ChatGPT and Gemini can decipher cryptic compiler errors for novice programmers, but their computational scale, cost, and tendency to over-assist make them problematic for widespread pedagogical adoption. This work demonstrates that smaller, specialised language models, enhanced via Supervised Fine-Tuning (SFT), present a more viable alternative for educational tools. We utilise a new dataset of 40,000 C compiler error explanations, derived from real introductory programming (CS1/2) student-generated programming errors, which we used to fine-tune three open-source models: Qwen3-4B, Llama-3.1-8B, and Qwen3-32B. We performed a dual evaluation, combining expert human reviews with a large-scale automated analysis of 8,000 responses using a validated LLM-as-judge ensemble. Our results show that SFT significantly boosts the pedagogical quality of smaller models, achieving performance comparable to much larger models. We analyse the trade-offs between model size and quality, confirming that fine-tuning compact, efficient models on high-quality, domain-specific data is a potent strategy for creating specialised models to drive educational tools. We provide a replicable methodology to foster broader access to generative AI capabilities in educational contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07060v1">DeepRetro: Retrosynthetic Pathway Discovery using Iterative LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 51 pages,
    </div>
    <details class="paper-abstract">
      Retrosynthesis, the identification of precursor molecules for a target compound, is pivotal for synthesizing complex molecules, but faces challenges in discovering novel pathways beyond predefined templates. Recent large language model (LLM) approaches to retrosynthesis have shown promise but effectively harnessing LLM reasoning capabilities for effective multi-step planning remains an open question. To address this challenge, we introduce DeepRetro, an open-source, iterative, hybrid LLM-based retrosynthetic framework. Our approach integrates the strengths of conventional template-based/Monte Carlo tree search tools with the generative power of LLMs in a step-wise, feedback-driven loop. Initially, synthesis planning is attempted with a template-based engine. If this fails, the LLM subsequently proposes single-step retrosynthetic disconnections. Crucially, these suggestions undergo rigorous validity, stability, and hallucination checks before the resulting precursors are recursively fed back into the pipeline for further evaluation. This iterative refinement allows for dynamic pathway exploration and correction. We demonstrate the potential of this pipeline through benchmark evaluations and case studies, showcasing its ability to identify viable and potentially novel retrosynthetic routes. In particular, we develop an interactive graphical user interface that allows expert human chemists to provide human-in-the-loop feedback to the reasoning algorithm. This approach successfully generates novel pathways for complex natural product compounds, demonstrating the potential for iterative LLM reasoning to advance state-of-art in complex chemical syntheses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05258v1">Spatio-Temporal LLM: Reasoning about Environments and Actions</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 Code and data are available at https://zoezheng126.github.io/STLLM-website/
    </div>
    <details class="paper-abstract">
      Despite the significant recent progress of Multimodal Large Language Models (MLLMs), MLLMs still struggle to correctly answer prompts that require a holistic spatio-temporal understanding. Specifically, it is challenging to address prompts that refer to 1) the entirety of an environment that an agent equipped with an MLLM can operate in; and simultaneously also refer to 2) recent actions that just happened and are encoded in a video clip. However, such a holistic spatio-temporal understanding is important for agents operating in the real world. To address this issue, we first develop a framework to collect a large-scale dataset. Using the collected "Reasoning about Environments and Actions" (REA) dataset, we show that recent methods indeed struggle to correctly answer the prompts. To improve, we develop a "spatio-temporal LLM" (ST-LLM), a model equipped with projectors to improve both spatial understanding of an environment and temporal understanding of recent observations. On the collected REA data, we show that the proposed method significantly improves results compared to prior work. Code and data are available at https://zoezheng126.github.io/STLLM-website/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05257v1">Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 23 Pages, Y. Hu and Y. Wang contribute equally
    </div>
    <details class="paper-abstract">
      Recent benchmarks for Large Language Model (LLM) agents primarily focus on evaluating reasoning, planning, and execution capabilities, while another critical component-memory, encompassing how agents memorize, update, and retrieve long-term information-is under-evaluated due to the lack of benchmarks. We term agents with memory mechanisms as memory agents. In this paper, we identify four core competencies essential for memory agents: accurate retrieval, test-time learning, long-range understanding, and conflict resolution. Existing datasets either rely on limited context lengths or are tailored for static, long-context settings like book-based QA, which do not reflect the interactive, multi-turn nature of memory agents that incrementally accumulate information. Furthermore, no existing benchmarks cover all four competencies. Therefore, we introduce MemoryAgentBench, a new benchmark specifically designed for memory agents. Our benchmark combines reformulated existing datasets with newly constructed ones, covering the above four memory competencies, providing a systematic and challenging testbed for assessing memory quality. We evaluate a diverse set of memory agents, ranging from simple context-based and retrieval-augmented generation (RAG) systems to advanced agents with external memory modules and tool integration. Empirical results reveal that current methods fall short of mastering all four competencies, underscoring the need for further research into comprehensive memory mechanisms for LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05606v3">OPeRA: A Dataset of Observation, Persona, Rationale, and Action for Evaluating LLMs on Human Online Shopping Behavior Simulation</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      Can large language models (LLMs) accurately simulate the next web action of a specific user? While LLMs have shown promising capabilities in generating ``believable'' human behaviors, evaluating their ability to mimic real user behaviors remains an open challenge, largely due to the lack of high-quality, publicly available datasets that capture both the observable actions and the internal reasoning of an actual human user. To address this gap, we introduce OPERA, a novel dataset of Observation, Persona, Rationale, and Action collected from real human participants during online shopping sessions. OPERA is the first public dataset that comprehensively captures: user personas, browser observations, fine-grained web actions, and self-reported just-in-time rationales. We developed both an online questionnaire and a custom browser plugin to gather this dataset with high fidelity. Using OPERA, we establish the first benchmark to evaluate how well current LLMs can predict a specific user's next action and rationale with a given persona and <observation, action, rationale> history. This dataset lays the groundwork for future research into LLM agents that aim to act as personalized digital twins for human.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05228v1">Cascade: Token-Sharded Private LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 To be published in ICML 2025 Main Proceedings as "Hidden No More: Attacking and Defending Private Third-Party LLM Inference", together with arXiv:2505.18332
    </div>
    <details class="paper-abstract">
      As LLMs continue to increase in parameter size, the computational resources required to run them are available to fewer parties. Therefore, third-party inference services -- where LLMs are hosted by third parties with significant computational resources -- are becoming increasingly popular. However, third party inference raises critical concerns about user data privacy. To mitigate these risks, privacy researchers have developed provably secure schemes for third-party inference, such as Secure Multi-Party Computation (SMPC). However, SMPC protocols have significant computational and communication overhead, and do not scale to large models. In this work, we propose a new multi-party inference protocol, Cascade, that avoids these punitive costs by leveraging sharding in the sequence dimension to maintain privacy, trading off cryptographic privacy guarantees for increased performance and scalability. We demonstrate that Cascade is resistant to a generalization of a recent attack that is highly effective against other statistical privacy schemes, and that it is further resistant to learning-based attacks. As Cascade is orders of magnitude faster than existing schemes, our findings offer practical solutions for secure deployment of modern state-of-the-art LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23824v2">Reviewing Scientific Papers for Critical Problems With Reasoning LLMs: Baseline Approaches and Automatic Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 Add results from new experiments; update discussion and GitHub link
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models have sparked interest in utilizing them to aid the peer review process of scientific publication amid the peer review crisis. However, having AI models generate full reviews in the same way as human reviewers risks exacerbating the irresponsible use of LLM-generated reviews. As an alternative, we propose adopting LLMs as manuscript quality checkers. We introduce several baseline approaches and an extendable automatic evaluation framework using top reasoning LLMs as judges to tackle the difficulty of recruiting domain experts for manual evaluation. Utilizing papers withdrawn from arXiv, we validated our proposed methods with several leading reasoning LLMs from multiple vendors and assessed their performance and API costs for identifying critical errors and unsoundness problems in scientific papers. o3 exhibited the best problem identification performance among all models at a modest cost. This paper provides insights into document-based scientific understanding/reasoning and lays a foundation for future applications. Our dataset, code, and model outputs are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05200v1">In-Context Learning as an Effective Estimator of Functional Correctness of LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      When applying LLM-based code generation to software development projects that follow a feature-driven or rapid application development approach, it becomes necessary to estimate the functional correctness of the generated code in the absence of test cases. Just as a user selects a relevant document from a ranked list of retrieved ones, a software generation workflow requires a developer to choose (and potentially refine) a generated solution from a ranked list of alternative solutions, ordered by their posterior likelihoods. This implies that estimating the quality of a ranked list -- akin to estimating "relevance" for query performance prediction (QPP) in IR -- is also crucial for generative software development, where quality is defined in terms of "functional correctness". In this paper, we propose an in-context learning (ICL) based approach for code quality estimation. Our findings demonstrate that providing few-shot examples of functionally correct code from a training set enhances the performance of existing QPP approaches as well as a zero-shot-based approach for code quality estimation.
    </details>
</div>
