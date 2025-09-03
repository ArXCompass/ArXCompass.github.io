# llm - 2025_08

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
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12575v1">Deep Learning Model for Amyloidogenicity Prediction using a Pre-trained Protein LLM</a></div>
    <div class="paper-meta">
      📅 2025-08-18
    </div>
    <details class="paper-abstract">
      The prediction of amyloidogenicity in peptides and proteins remains a focal point of ongoing bioinformatics. The crucial step in this field is to apply advanced computational methodologies. Many recent approaches to predicting amyloidogenicity within proteins are highly based on evolutionary motifs and the individual properties of amino acids. It is becoming increasingly evident that the sequence information-based features show high predictive performance. Consequently, our study evaluated the contextual features of protein sequences obtained from a pretrained protein large language model leveraging bidirectional LSTM and GRU to predict amyloidogenic regions in peptide and protein sequences. Our method achieved an accuracy of 84.5% on 10-fold cross-validation and an accuracy of 83% in the test dataset. Our results demonstrate competitive performance, highlighting the potential of LLMs in enhancing the accuracy of amyloid prediction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12555v1">Illuminating LLM Coding Agents: Visual Analytics for Deeper Understanding and Enhancement</a></div>
    <div class="paper-meta">
      📅 2025-08-18
      | 💬 11 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Coding agents powered by large language models (LLMs) have gained traction for automating code generation through iterative problem-solving with minimal human involvement. Despite the emergence of various frameworks, e.g., LangChain, AutoML, and AIDE, ML scientists still struggle to effectively review and adjust the agents' coding process. The current approach of manually inspecting individual outputs is inefficient, making it difficult to track code evolution, compare coding iterations, and identify improvement opportunities. To address this challenge, we introduce a visual analytics system designed to enhance the examination of coding agent behaviors. Focusing on the AIDE framework, our system supports comparative analysis across three levels: (1) Code-Level Analysis, which reveals how the agent debugs and refines its code over iterations; (2) Process-Level Analysis, which contrasts different solution-seeking processes explored by the agent; and (3) LLM-Level Analysis, which highlights variations in coding behavior across different LLMs. By integrating these perspectives, our system enables ML scientists to gain a structured understanding of agent behaviors, facilitating more effective debugging and prompt engineering. Through case studies using coding agents to tackle popular Kaggle competitions, we demonstrate how our system provides valuable insights into the iterative coding process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05028v3">Evaluation of Finetuned LLMs in AMR Parsing</a></div>
    <div class="paper-meta">
      📅 2025-08-18
      | 💬 27 pages, 32 figures
    </div>
    <details class="paper-abstract">
      AMR (Abstract Meaning Representation) is a semantic formalism that encodes sentence meaning as rooted, directed, acyclic graphs, where nodes represent concepts and edges denote semantic relations. Finetuning decoder only Large Language Models (LLMs) represent a promising novel straightfoward direction for AMR parsing. This paper presents a comprehensive evaluation of finetuning four distinct LLM architectures, Phi 3.5, Gemma 2, LLaMA 3.2, and DeepSeek R1 LLaMA Distilled using the LDC2020T02 Gold AMR3.0 test set. Our results have shown that straightfoward finetuning of decoder only LLMs can achieve comparable performance to complex State of the Art (SOTA) AMR parsers. Notably, LLaMA 3.2 demonstrates competitive performance against SOTA AMR parsers given a straightforward finetuning approach. We achieved SMATCH F1: 0.804 on the full LDC2020T02 test split, on par with APT + Silver (IBM) at 0.804 and approaching Graphene Smatch (MBSE) at 0.854. Across our analysis, we also observed a consistent pattern where LLaMA 3.2 leads in semantic performance while Phi 3.5 excels in structural validity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.12624v6">Judging the Judges: Evaluating Alignment and Vulnerabilities in LLMs-as-Judges</a></div>
    <div class="paper-meta">
      📅 2025-08-18
      | 💬 https://aclanthology.org/2025.gem-1.33/
    </div>
    <details class="paper-abstract">
      Offering a promising solution to the scalability challenges associated with human evaluation, the LLM-as-a-judge paradigm is rapidly gaining traction as an approach to evaluating large language models (LLMs). However, there are still many open questions about the strengths and weaknesses of this paradigm, and what potential biases it may hold. In this paper, we present a comprehensive study of the performance of various LLMs acting as judges, focusing on a clean scenario in which inter-human agreement is high. Investigating thirteen judge models of different model sizes and families, judging answers of nine different 'examtaker models' - both base and instruction-tuned - we find that only the best (and largest) models achieve reasonable alignment with humans. However, they are still quite far behind inter-human agreement and their assigned scores may still differ with up to 5 points from human-assigned scores. In terms of their ranking of the nine exam-taker models, instead, also smaller models and even the lexical metric contains may provide a reasonable signal. Through error analysis and other studies, we identify vulnerabilities in judge models, such as their sensitivity to prompt complexity and length, and a tendency toward leniency. The fact that even the best judges differ from humans in this comparatively simple setup suggest that caution may be wise when using judges in more complex setups. Lastly, our research rediscovers the importance of using alignment metrics beyond simple percent alignment, showing that judges with high percent agreement can still assign vastly different scores.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12535v1">CorrSteer: Steering Improves Task Performance and Safety in LLMs through Correlation-based Sparse Autoencoder Feature Selection</a></div>
    <div class="paper-meta">
      📅 2025-08-18
      | 💬 42 pages, 9 tables
    </div>
    <details class="paper-abstract">
      Sparse Autoencoders (SAEs) can extract interpretable features from large language models (LLMs) without supervision. However, their effectiveness in downstream steering tasks is limited by the requirement for contrastive datasets or large activation storage. To address these limitations, we propose CorrSteer, which selects features by correlating sample correctness with SAE activations from generated tokens at inference time. This approach uses only inference-time activations to extract more relevant features, thereby avoiding spurious correlations. It also obtains steering coefficients from average activations, automating the entire pipeline. Our method shows improved task performance on QA, bias mitigation, jailbreaking prevention, and reasoning benchmarks on Gemma 2 2B and LLaMA 3.1 8B, notably achieving a +4.1% improvement in MMLU performance and a +22.9% improvement in HarmBench with only 4000 samples. Selected features demonstrate semantically meaningful patterns aligned with each task's requirements, revealing the underlying capabilities that drive performance. Our work establishes correlationbased selection as an effective and scalable approach for automated SAE steering across language model applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13382v1">Datarus-R1: An Adaptive Multi-Step Reasoning LLM for Automated Data Analysis</a></div>
    <div class="paper-meta">
      📅 2025-08-18
    </div>
    <details class="paper-abstract">
      We present Datarus-R1-14B, a 14 B-parameter open-weights language model fine-tuned from Qwen 2.5-14B-Instruct to act as a virtual data analyst and graduate-level problem solver. Datarus is trained not on isolated question-answer pairs but on full analytical trajectories including reasoning steps, code execution, error traces, self-corrections, and final conclusions, all captured in a ReAct-style notebook format spanning finance, medicine, numerical analysis, and other quantitative domains. Our training pipeline combines (i) a trajectory-centric synthetic data generator that yielded 144 000 tagged notebook episodes, (ii) a dual-reward framework blending a lightweight tag-based structural signal with a Hierarchical Reward Model (HRM) that scores both single-step soundness and end-to-end coherence, and (iii) a memory-optimized implementation of Group Relative Policy Optimization (GRPO) featuring KV-cache reuse, sequential generation, and reference-model sharding. A cosine curriculum smoothly shifts emphasis from structural fidelity to semantic depth, reducing the format collapse and verbosity that often plague RL-aligned LLMs. A central design choice in Datarus is it dual reasoning interface. In agentic mode the model produces ReAct-tagged steps that invoke Python tools to execute real code; in reflection mode it outputs compact Chain-of-Thought (CoT) traces delimited by <think> and <answer> tags. On demanding postgraduate-level problems, Datarus exhibits an "AHA-moment" pattern: it sketches hypotheses, revises them once or twice, and converges avoiding the circular, token-inflating loops common to contemporary systems. Across standard public benchmarks Datarus surpasses similar size models and even reaches the level of larger reasoning models such as QwQ-32B achieving up to 30% higher accuracy on AIME 2024/2025 and LiveCodeBench while emitting 18-49% fewer tokens per solution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13333v1">HiFo-Prompt: Prompting with Hindsight and Foresight for LLM-based Automatic Heuristic Design</a></div>
    <div class="paper-meta">
      📅 2025-08-18
      | 💬 9 pages, 6 figures
    </div>
    <details class="paper-abstract">
      LLM-based Automatic Heuristic Design (AHD) within Evolutionary Computation (EC) frameworks has shown promising results. However, its effectiveness is hindered by the use of static operators and the lack of knowledge accumulation mechanisms. We introduce HiFo-Prompt, a framework that guides LLMs with two synergistic prompting strategies: Foresight and Hindsight. Foresight-based prompts adaptively steer the search based on population dynamics, managing the exploration-exploitation trade-off. In addition, hindsight-based prompts mimic human expertise by distilling successful heuristics from past generations into fundamental, reusable design principles. This dual mechanism transforms transient discoveries into a persistent knowledge base, enabling the LLM to learn from its own experience. Empirical results demonstrate that HiFo-Prompt significantly outperforms state-of-the-art LLM-based AHD methods, generating higher-quality heuristics while achieving substantially faster convergence and superior query efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.09288v2">Can AI Keep a Secret? Contextual Integrity Verification: A Provable Security Architecture for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-18
      | 💬 2 figures, 3 tables; code and certification harness: https://github.com/ayushgupta4897/Contextual-Integrity-Verification ; Elite-Attack dataset: https://huggingface.co/datasets/zyushg/elite-attack
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) remain acutely vulnerable to prompt injection and related jailbreak attacks; heuristic guardrails (rules, filters, LLM judges) are routinely bypassed. We present Contextual Integrity Verification (CIV), an inference-time security architecture that attaches cryptographically signed provenance labels to every token and enforces a source-trust lattice inside the transformer via a pre-softmax hard attention mask (with optional FFN/residual gating). CIV provides deterministic, per-token non-interference guarantees on frozen models: lower-trust tokens cannot influence higher-trust representations. On benchmarks derived from recent taxonomies of prompt-injection vectors (Elite-Attack + SoK-246), CIV attains 0% attack success rate under the stated threat model while preserving 93.1% token-level similarity and showing no degradation in model perplexity on benign tasks; we note a latency overhead attributable to a non-optimized data path. Because CIV is a lightweight patch -- no fine-tuning required -- we demonstrate drop-in protection for Llama-3-8B and Mistral-7B. We release a reference implementation, an automated certification harness, and the Elite-Attack corpus to support reproducible research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02873v2">It's the Thought that Counts: Evaluating the Attempts of Frontier LLMs to Persuade on Harmful Topics</a></div>
    <div class="paper-meta">
      📅 2025-08-18
    </div>
    <details class="paper-abstract">
      Persuasion is a powerful capability of large language models (LLMs) that both enables beneficial applications (e.g. helping people quit smoking) and raises significant risks (e.g. large-scale, targeted political manipulation). Prior work has found models possess a significant and growing persuasive capability, measured by belief changes in simulated or real users. However, these benchmarks overlook a crucial risk factor: the propensity of a model to attempt to persuade in harmful contexts. Understanding whether a model will blindly ``follow orders'' to persuade on harmful topics (e.g. glorifying joining a terrorist group) is key to understanding the efficacy of safety guardrails. Moreover, understanding if and when a model will engage in persuasive behavior in pursuit of some goal is essential to understanding the risks from agentic AI systems. We propose the Attempt to Persuade Eval (APE) benchmark, that shifts the focus from persuasion success to persuasion attempts, operationalized as a model's willingness to generate content aimed at shaping beliefs or behavior. Our evaluation framework probes frontier LLMs using a multi-turn conversational setup between simulated persuader and persuadee agents. APE explores a diverse spectrum of topics including conspiracies, controversial issues, and non-controversially harmful content. We introduce an automated evaluator model to identify willingness to persuade and measure the frequency and context of persuasive attempts. We find that many open and closed-weight models are frequently willing to attempt persuasion on harmful topics and that jailbreaking can increase willingness to engage in such behavior. Our results highlight gaps in current safety guardrails and underscore the importance of evaluating willingness to persuade as a key dimension of LLM risk. APE is available at github.com/AlignmentResearch/AttemptPersuadeEval
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13240v1">Quantifying Loss Aversion in Cyber Adversaries via LLM Analysis</a></div>
    <div class="paper-meta">
      📅 2025-08-18
    </div>
    <details class="paper-abstract">
      Understanding and quantifying human cognitive biases from empirical data has long posed a formidable challenge, particularly in cybersecurity, where defending against unknown adversaries is paramount. Traditional cyber defense strategies have largely focused on fortification, while some approaches attempt to anticipate attacker strategies by mapping them to cognitive vulnerabilities, yet they fall short in dynamically interpreting attacks in progress. In recognition of this gap, IARPA's ReSCIND program seeks to infer, defend against, and even exploit attacker cognitive traits. In this paper, we present a novel methodology that leverages large language models (LLMs) to extract quantifiable insights into the cognitive bias of loss aversion from hacker behavior. Our data are collected from an experiment in which hackers were recruited to attack a controlled demonstration network. We process the hacker generated notes using LLMs using it to segment the various actions and correlate the actions to predefined persistence mechanisms used by hackers. By correlating the implementation of these mechanisms with various operational triggers, our analysis provides new insights into how loss aversion manifests in hacker decision-making. The results demonstrate that LLMs can effectively dissect and interpret nuanced behavioral patterns, thereby offering a transformative approach to enhancing cyber defense strategies through real-time, behavior-based analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.10795v2">Beyond "Not Novel Enough": Enriching Scholarly Critique with LLM-Assisted Feedback</a></div>
    <div class="paper-meta">
      📅 2025-08-17
    </div>
    <details class="paper-abstract">
      Novelty assessment is a central yet understudied aspect of peer review, particularly in high volume fields like NLP where reviewer capacity is increasingly strained. We present a structured approach for automated novelty evaluation that models expert reviewer behavior through three stages: content extraction from submissions, retrieval and synthesis of related work, and structured comparison for evidence based assessment. Our method is informed by a large scale analysis of human written novelty reviews and captures key patterns such as independent claim verification and contextual reasoning. Evaluated on 182 ICLR 2025 submissions with human annotated reviewer novelty assessments, the approach achieves 86.5% alignment with human reasoning and 75.3% agreement on novelty conclusions - substantially outperforming existing LLM based baselines. The method produces detailed, literature aware analyses and improves consistency over ad hoc reviewer judgments. These results highlight the potential for structured LLM assisted approaches to support more rigorous and transparent peer review without displacing human expertise. Data and code are made available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12379v1">GraphCogent: Overcoming LLMs' Working Memory Constraints via Multi-Agent Collaboration in Complex Graph Understanding</a></div>
    <div class="paper-meta">
      📅 2025-08-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show promising performance on small-scale graph reasoning tasks but fail when handling real-world graphs with complex queries. This phenomenon stems from LLMs' inability to effectively process complex graph topology and perform multi-step reasoning simultaneously. To address these limitations, we propose GraphCogent, a collaborative agent framework inspired by human Working Memory Model that decomposes graph reasoning into specialized cognitive processes: sense, buffer, and execute. The framework consists of three modules: Sensory Module standardizes diverse graph text representations via subgraph sampling, Buffer Module integrates and indexes graph data across multiple formats, and Execution Module combines tool calling and model generation for efficient reasoning. We also introduce Graph4real, a comprehensive benchmark contains with four domains of real-world graphs (Web, Social, Transportation, and Citation) to evaluate LLMs' graph reasoning capabilities. Our Graph4real covers 21 different graph reasoning tasks, categorized into three types (Structural Querying, Algorithmic Reasoning, and Predictive Modeling tasks), with graph scales that are 10 times larger than existing benchmarks. Experiments show that Llama3.1-8B based GraphCogent achieves a 50% improvement over massive-scale LLMs like DeepSeek-R1 (671B). Compared to state-of-the-art agent-based baseline, our framework outperforms by 20% in accuracy while reducing token usage by 80% for in-toolset tasks and 30% for out-toolset tasks. Code will be available after review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05668v2">A Survey of LLM-based Deep Search Agents: Paradigm, Optimization, Evaluation, and Challenges</a></div>
    <div class="paper-meta">
      📅 2025-08-17
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) has significantly revolutionized web search. The emergence of LLM-based Search Agents marks a pivotal shift towards deeper, dynamic, autonomous information seeking. These agents can comprehend user intentions and environmental context and execute multi-turn retrieval with dynamic planning, extending search capabilities far beyond the web. Leading examples like OpenAI's Deep Research highlight their potential for deep information mining and real-world applications. This survey provides the first systematic analysis of search agents. We comprehensively analyze and categorize existing works from the perspectives of architecture, optimization, application, and evaluation, ultimately identifying critical open challenges and outlining promising future research directions in this rapidly evolving field. Our repository is available on https://github.com/YunjiaXi/Awesome-Search-Agent-Papers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12358v1">Uncovering Systematic Failures of LLMs in Verifying Code Against Natural Language Specifications</a></div>
    <div class="paper-meta">
      📅 2025-08-17
      | 💬 Accepted to the NIER track of the 40th IEEE/ACM International Conference on Automated Software Engineering (ASE 2025)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become essential tools in software development, widely used for requirements engineering, code generation and review tasks. Software engineers often rely on LLMs to assess whether system code implementation satisfy task requirements, thereby enhancing code robustness and accuracy. However, it remains unclear whether LLMs can reliably determine whether the code complies fully with the given task descriptions, which is usually natural language specifications. In this paper, we uncover a systematic failure of LLMs in evaluating whether code aligns with natural language requirements. Specifically, with widely used benchmarks, we employ unified prompts to judge code correctness. Our results reveal that LLMs frequently misclassify correct code implementations as either ``not satisfying requirements'' or containing potential defects. Surprisingly, more complex prompting, especially when leveraging prompt engineering techniques involving explanations and proposed corrections, leads to higher misjudgment rate, which highlights the critical reliability issues in using LLMs as code review assistants. We further analyze the root causes of these misjudgments, and propose two improved prompting strategies for mitigation. For the first time, our findings reveals unrecognized limitations in LLMs to match code with requirements. We also offer novel insights and practical guidance for effective use of LLMs in automated code review and task-oriented agent scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.24015v2">Hierarchical Knowledge Injection for Improving LLM-based Program Repair</a></div>
    <div class="paper-meta">
      📅 2025-08-17
      | 💬 Accepted at IEEE/ACM Automated Software Engineering (ASE) 2025 Conference
    </div>
    <details class="paper-abstract">
      Prompting LLMs with bug-related context (e.g., error messages, stack traces) improves automated program repair, but many bugs still remain unresolved. In real-world projects, developers often rely on broader repository and project-level context beyond the local code to resolve such bugs. In this paper, we investigate how automatically extracting and providing such knowledge can improve LLM-based program repair. We propose a layered knowledge injection framework that incrementally augments LLMs with structured context. It starts with the Bug Knowledge Layer, which includes information such as the buggy function and failing tests; expands to the Repository Knowledge Layer, which adds structural dependencies, related files, and commit history; and finally injects the Project Knowledge Layer, which incorporates relevant details from documentation and previously fixed bugs. We evaluate this framework on a dataset of 314 bugs from BugsInPy using two LLMs (Llama 3.3 and GPT-4o-mini), and analyze fix rates across six bug types. By progressively injecting knowledge across layers, our approach achieves a fix rate of 79% (250/314) using Llama 3.3, a significant improvement of 23% over previous work. All bug types show improvement with the addition of repository-level context, while only a subset benefit further from project-level knowledge, highlighting that different bug types require different levels of contextual information for effective repair. We also analyze the remaining unresolved bugs and find that more complex and structurally isolated bugs, such as Program Anomaly and GUI bugs, remain difficult even after injecting all available information. Our results show that layered context injection improves program repair and suggest the need for interactive and adaptive APR systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00845v3">Generalizable LLM Learning of Graph Synthetic Data with Post-training Alignment</a></div>
    <div class="paper-meta">
      📅 2025-08-17
      | 💬 8 pages, 1 figures, 2 tables. Experimental code and results are publicly available at https://anonymous.4open.science/r/Graph_RL-BF08/readme.md
    </div>
    <details class="paper-abstract">
      Previous research has sought to enhance the graph reasoning capabilities of LLMs by supervised fine-tuning on synthetic graph data. While these led to specialized LLMs better at solving graph algorithm problems, we don't need LLMs for shortest path: we need generalization from synthetic graph data to real-world tasks with implicit graph structures. In this work, we propose to unlock generalizable learning of graph with post-training alignment with synthetic data. We first design solution-based and process-based rewards for synthetic graph problems: instead of rigid memorizing response patterns in direct fine-tuning, we posit that post-training alignment would help LLMs grasp the essentials underlying graph reasoning and alleviate overfitting on synthetic data. We employ post-training alignment algorithms such as GRPO and DPO, aligning both off-the-shelf LLMs and LLMs fine-tuned on synthetic graph data. We then compare them against existing settings on both in-domain synthetic tasks and out-of-domain real-world tasks with implicit graph structures such as multi-hop QA, structured planning, and more. Extensive experiments demonstrate that our post-training alignment recipe leads to statistically significant improvement on 5 datasets, with an average gain of 12.9% over baseline settings. Further analysis reveals that process-based rewards consistently outperform solution-based rewards on synthetic data but not on real-world tasks, and compositionality and explainable intermediate steps remains a critical challenge even after post-training alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12281v1">Legal$Δ$: Enhancing Legal Reasoning in LLMs via Reinforcement Learning with Chain-of-Thought Guided Information Gain</a></div>
    <div class="paper-meta">
      📅 2025-08-17
    </div>
    <details class="paper-abstract">
      Legal Artificial Intelligence (LegalAI) has achieved notable advances in automating judicial decision-making with the support of Large Language Models (LLMs). However, existing legal LLMs still struggle to generate reliable and interpretable reasoning processes. They often default to fast-thinking behavior by producing direct answers without explicit multi-step reasoning, limiting their effectiveness in complex legal scenarios that demand rigorous justification. To address this challenge, we propose Legal$\Delta$, a reinforcement learning framework designed to enhance legal reasoning through chain-of-thought guided information gain. During training, Legal$\Delta$ employs a dual-mode input setup-comprising direct answer and reasoning-augmented modes-and maximizes the information gain between them. This encourages the model to acquire meaningful reasoning patterns rather than generating superficial or redundant explanations. Legal$\Delta$ follows a two-stage approach: (1) distilling latent reasoning capabilities from a powerful Large Reasoning Model (LRM), DeepSeek-R1, and (2) refining reasoning quality via differential comparisons, combined with a multidimensional reward mechanism that assesses both structural coherence and legal-domain specificity. Experimental results on multiple legal reasoning tasks demonstrate that Legal$\Delta$ outperforms strong baselines in both accuracy and interpretability. It consistently produces more robust and trustworthy legal judgments without relying on labeled preference data. All code and data will be released at https://github.com/NEUIR/LegalDelta.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12277v1">The Self-Execution Benchmark: Measuring LLMs' Attempts to Overcome Their Lack of Self-Execution</a></div>
    <div class="paper-meta">
      📅 2025-08-17
      | 💬 11 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are commonly evaluated on tasks that test their knowledge or reasoning abilities. In this paper, we explore a different type of evaluation: whether an LLM can predict aspects of its own responses. Since LLMs lack the ability to execute themselves, we introduce the Self-Execution Benchmark, which measures a model's ability to anticipate properties of its output, such as whether a question will be difficult for it, whether it will refuse to answer, or what kinds of associations it is likely to produce. Our experiments show that models generally perform poorly on this benchmark, and that increased model size or capability does not consistently lead to better performance. These results suggest a fundamental limitation in how LLMs represent and reason about their own behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20541v3">MeLA: A Metacognitive LLM-Driven Architecture for Automatic Heuristic Design</a></div>
    <div class="paper-meta">
      📅 2025-08-17
    </div>
    <details class="paper-abstract">
      This paper introduces MeLA, a Metacognitive LLM-Driven Architecture that presents a new paradigm for Automatic Heuristic Design (AHD). Traditional evolutionary methods operate directly on heuristic code; in contrast, MeLA evolves the instructional prompts used to guide a Large Language Model (LLM) in generating these heuristics. This process of "prompt evolution" is driven by a novel metacognitive framework where the system analyzes performance feedback to systematically refine its generative strategy. MeLA's architecture integrates a problem analyzer to construct an initial strategic prompt, an error diagnosis system to repair faulty code, and a metacognitive search engine that iteratively optimizes the prompt based on heuristic effectiveness. In comprehensive experiments across both benchmark and real-world problems, MeLA consistently generates more effective and robust heuristics, significantly outperforming state-of-the-art methods. Ultimately, this research demonstrates the profound potential of using cognitive science as a blueprint for AI architecture, revealing that by enabling an LLM to metacognitively regulate its problem-solving process, we unlock a more robust and interpretable path to AHD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13114v2">Understanding LLM-Centric Challenges for Deep Learning Frameworks: An Empirical Analysis</a></div>
    <div class="paper-meta">
      📅 2025-08-17
      | 💬 46 pages, 14 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have driven significant progress across a wide range of real-world applications. Realizing such models requires substantial system-level support. Deep learning (DL) frameworks provide this foundation by enabling efficient model construction, distributed execution, and optimized deployment. The large parameter scale and extended execution cycles impose exacting demands on deep learning frameworks, particularly in terms of scalability, stability, and efficiency. Therefore, poor usability, limited functionality, and subtle bugs in DL frameworks may hinder development efficiency and cause severe failures or resource waste. However, a fundamental question has not been thoroughly investigated in previous studies, i.e., what challenges do DL frameworks face in supporting LLMs? To answer this question, we analyze issue reports from three major DL frameworks (i.e., MindSpore, PyTorch, and TensorFlow) and eight associated LLM toolkits such as Megatron. Based on a manual review of these reports, we construct a taxonomy that captures LLM-centric framework bugs, user requirements, and user questions. We then refine and enrich this taxonomy through interviews with 11 LLM users and eight DL framework developers. Based on the constructed taxonomy and findings summarized from interviews, our study further reveals key technical challenges and mismatches between LLM user needs and developer priorities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12265v1">Fast, Slow, and Tool-augmented Thinking for LLMs: A Review</a></div>
    <div class="paper-meta">
      📅 2025-08-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable progress in reasoning across diverse domains. However, effective reasoning in real-world tasks requires adapting the reasoning strategy to the demands of the problem, ranging from fast, intuitive responses to deliberate, step-by-step reasoning and tool-augmented thinking. Drawing inspiration from cognitive psychology, we propose a novel taxonomy of LLM reasoning strategies along two knowledge boundaries: a fast/slow boundary separating intuitive from deliberative processes, and an internal/external boundary distinguishing reasoning grounded in the model's parameters from reasoning augmented by external tools. We systematically survey recent work on adaptive reasoning in LLMs and categorize methods based on key decision factors. We conclude by highlighting open challenges and future directions toward more adaptive, efficient, and reliable LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07745v4">StepTool: Enhancing Multi-Step Tool Usage in LLMs via Step-Grained Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-08-17
      | 💬 Accepted by CIKM'25
    </div>
    <details class="paper-abstract">
      Despite their powerful text generation capabilities, large language models (LLMs) still struggle to effectively utilize external tools to solve complex tasks, a challenge known as tool learning. Existing methods primarily rely on supervised fine-tuning, treating tool learning as a text generation problem while overlooking the decision-making complexities inherent in multi-step contexts. In this work, we propose modeling tool learning as a dynamic decision-making process and introduce StepTool, a novel step-grained reinforcement learning framework that enhances LLMs' capabilities in multi-step tool use. StepTool comprises two key components: Step-grained Reward Shaping, which assigns rewards to each tool interaction based on its invocation success and contribution to task completion; and Step-grained Optimization, which applies policy gradient methods to optimize the model across multiple decision steps. Extensive experiments across diverse benchmarks show that StepTool consistently outperforms both SFT-based and RL-based baselines in terms of task Pass Rate and Recall of relevant tools. Furthermore, our analysis suggests that StepTool helps models discover new tool-use strategies rather than merely re-weighting prior knowledge. These results highlight the importance of fine-grained decision modeling in tool learning and establish StepTool as a general and robust solution for enhancing multi-step tool use in LLMs. Code and data are available at https://github.com/yuyq18/StepTool.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11247v2">LD-Scene: LLM-Guided Diffusion for Controllable Generation of Adversarial Safety-Critical Driving Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-08-17
      | 💬 18 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Ensuring the safety and robustness of autonomous driving systems necessitates a comprehensive evaluation in safety-critical scenarios. However, these safety-critical scenarios are rare and difficult to collect from real-world driving data, posing significant challenges to effectively assessing the performance of autonomous vehicles. Typical existing methods often suffer from limited controllability and lack user-friendliness, as extensive expert knowledge is essentially required. To address these challenges, we propose LD-Scene, a novel framework that integrates Large Language Models (LLMs) with Latent Diffusion Models (LDMs) for user-controllable adversarial scenario generation through natural language. Our approach comprises an LDM that captures realistic driving trajectory distributions and an LLM-based guidance module that translates user queries into adversarial loss functions, facilitating the generation of scenarios aligned with user queries. The guidance module integrates an LLM-based Chain-of-Thought (CoT) code generator and an LLM-based code debugger, enhancing the controllability and robustness in generating guidance functions. Extensive experiments conducted on the nuScenes dataset demonstrate that LD-Scene achieves state-of-the-art performance in generating realistic, diverse, and effective adversarial scenarios. Furthermore, our framework provides fine-grained control over adversarial behaviors, thereby facilitating more effective testing tailored to specific driving scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12232v1">LinkAnchor: An Autonomous LLM-Based Agent for Issue-to-Commit Link Recovery</a></div>
    <div class="paper-meta">
      📅 2025-08-17
    </div>
    <details class="paper-abstract">
      Issue-to-commit link recovery plays an important role in software traceability and improves project management. However, it remains a challenging task. A study on GitHub shows that only 42.2% of the issues are correctly linked to their commits. This highlights the potential for further development and research in this area. Existing studies have employed various AI/ML-based approaches, and with the recent development of large language models, researchers have leveraged LLMs to tackle this problem. These approaches suffer from two main issues. First, LLMs are constrained by limited context windows and cannot ingest all of the available data sources, such as long commit histories, extensive issue comments, and large code repositories. Second, most methods operate on individual issue-commit pairs; that is, given a single issue-commit pair, they determine whether the commit resolves the issue. This quickly becomes impractical in real-world repositories containing tens of thousands of commits. To address these limitations, we present LinkAnchor, the first autonomous LLM-based agent designed for issue-to-commit link recovery. The lazy-access architecture of LinkAnchor enables the underlying LLM to access the rich context of software, spanning commits, issue comments, and code files, without exceeding the token limit by dynamically retrieving only the most relevant contextual data. Additionally, LinkAnchor is able to automatically pinpoint the target commit rather than exhaustively scoring every possible candidate. Our evaluations show that LinkAnchor outperforms state-of-the-art issue-to-commit link recovery approaches by 60-262% in Hit@1 score across all our case study projects. We also publicly release LinkAnchor as a ready-to-use tool, along with our replication package. LinkAnchor is designed and tested for GitHub and Jira, and is easily extendable to other platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19855v3">Inducing Causal World Models in LLMs for Zero-Shot Physical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-08-17
      | 💬 12 pages, 4 figures,
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), despite their advanced linguistic capabilities, fundamentally lack an intuitive understanding of physical dynamics, which limits their effectiveness in real-world scenarios that require causal reasoning. In this paper, we introduce Causal World Model Induction (CWMI), a novel framework designed to embed an explicit model of causal physics within an LLM. Our approach incorporates a dedicated Causal Physics Module (CPM) and a new training objective called Causal Intervention Loss, encouraging the model to learn cause-and-effect relationships from multimodal data. By training the model to predict the outcomes of hypothetical interventions instead of merely capturing statistical correlations, CWMI develops a robust internal representation of physical laws. Experimental results show that CWMI significantly outperforms state-of-the-art LLMs on zero-shot physical reasoning tasks, including the PIQA benchmark and our newly proposed PhysiCa-Bench dataset. These findings demonstrate that inducing a causal world model is a critical step toward more reliable and generalizable AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13757v2">CoRank: LLM-Based Compact Reranking with Document Features for Scientific Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-08-17
      | 💬 12 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Scientific retrieval is essential for advancing scientific knowledge discovery. Within this process, document reranking plays a critical role in refining first-stage retrieval results. However, standard LLM listwise reranking faces challenges in the scientific domain. First-stage retrieval is often suboptimal in the scientific domain, so relevant documents are ranked lower. Meanwhile, conventional listwise reranking places the full text of candidates into the context window, limiting the number of candidates that can be considered. As a result, many relevant documents are excluded before reranking, constraining overall retrieval performance. To address these challenges, we explore semantic-feature-based compact document representations (e.g., categories, sections, and keywords) and propose CoRank, a training-free, model-agnostic reranking framework for scientific retrieval. It presents a three-stage solution: (i) offline extraction of document features, (ii) coarse-grained reranking using these compact representations, and (iii) fine-grained reranking on full texts of the top candidates from (ii). This integrated process addresses suboptimal first-stage retrieval: Compact representations allow more documents to fit within the context window, improving candidate set coverage, while the final fine-grained ranking ensures a more accurate ordering. Experiments on 5 academic retrieval datasets show that CoRank significantly improves reranking performance across different LLM backbones (average nDCG@10 from 50.6 to 55.5). Overall, these results underscore the synergistic interaction between information extraction and information retrieval, demonstrating how structured semantic features can enhance reranking in the scientific domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13231v1">Accelerating LLM Inference via Dynamic KV Cache Placement in Heterogeneous Memory System</a></div>
    <div class="paper-meta">
      📅 2025-08-17
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) inference is increasingly constrained by memory bandwidth, with frequent access to the key-value (KV) cache dominating data movement. While attention sparsity reduces some memory traffic, the relevance of past tokens varies over time, requiring the full KV cache to remain accessible and sustaining pressure on both bandwidth and capacity. With advances in interconnects such as NVLink and LPDDR5X, modern AI hardware now integrates high-bandwidth memory (HBM) with high-speed off-package DRAM, making heterogeneous memory systems a practical solution. This work investigates dynamic KV cache placement across such systems to maximize aggregated bandwidth utilization under capacity constraints. Rather than proposing a specific scheduling policy, we formulate the placement problem mathematically and derive a theoretical upper bound, revealing substantial headroom for runtime optimization. To our knowledge, this is the first formal treatment of dynamic KV cache scheduling in heterogeneous memory systems for LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14918v1">Disentangling the Drivers of LLM Social Conformity: An Uncertainty-Moderated Dual-Process Mechanism</a></div>
    <div class="paper-meta">
      📅 2025-08-17
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) integrate into collaborative teams, their social conformity -- the tendency to align with majority opinions -- has emerged as a key concern. In humans, conformity arises from informational influence (rational use of group cues for accuracy) or normative influence (social pressure for approval), with uncertainty moderating this balance by shifting from purely analytical to heuristic processing. It remains unclear whether these human psychological mechanisms apply to LLMs. This study adapts the information cascade paradigm from behavioral economics to quantitatively disentangle the two drivers to investigate the moderate effect. We evaluated nine leading LLMs across three decision-making scenarios (medical, legal, investment), manipulating information uncertainty (q = 0.667, 0.55, and 0.70, respectively). Our results indicate that informational influence underpins the models' behavior across all contexts, with accuracy and confidence consistently rising with stronger evidence. However, this foundational mechanism is dramatically modulated by uncertainty. In low-to-medium uncertainty scenarios, this informational process is expressed as a conservative strategy, where LLMs systematically underweight all evidence sources. In contrast, high uncertainty triggers a critical shift: while still processing information, the models additionally exhibit a normative-like amplification, causing them to overweight public signals (beta > 1.55 vs. private beta = 0.81).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02048v2">Improving LLM Agents with Reinforcement Learning on Cryptographic CTF Challenges</a></div>
    <div class="paper-meta">
      📅 2025-08-17
      | 💬 13 pages, 2 figures
    </div>
    <details class="paper-abstract">
      We present 'Random-Crypto', a procedurally generated cryptographic Capture The Flag (CTF) dataset designed to unlock the potential of Reinforcement Learning (RL) for LLM-based agents in security-sensitive domains. Cryptographic reasoning offers an ideal RL testbed: it combines precise validation, structured multi-step inference, and reliance on reliable computational tool use. Leveraging these properties, we fine-tune a Python tool-augmented Llama-3.1-8B via Group Relative Policy Optimization (GRPO) in a secure execution environment. The resulting agent achieves a significant improvement in Pass@8 on previously unseen challenges. Moreover, the improvements generalize to two external benchmarks: 'picoCTF', spanning both crypto and non-crypto tasks, and 'AICrypto MCQ', a multiple-choice benchmark of 135 cryptography questions. Ablation studies attribute the gains to enhanced tool usage and procedural reasoning. These findings position 'Random-Crypto' as a rich training ground for building intelligent, adaptable LLM agents capable of handling complex cybersecurity tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12491v1">Cost-Aware Contrastive Routing for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-17
    </div>
    <details class="paper-abstract">
      We study cost-aware routing for large language models across diverse and dynamic pools of models. Existing approaches often overlook prompt-specific context, rely on expensive model profiling, assume a fixed set of experts, or use inefficient trial-and-error strategies. We introduce Cost-Spectrum Contrastive Routing (CSCR), a lightweight framework that maps both prompts and models into a shared embedding space to enable fast, cost-sensitive selection. CSCR uses compact, fast-to-compute logit footprints for open-source models and perplexity fingerprints for black-box APIs. A contrastive encoder is trained to favor the cheapest accurate expert within adaptive cost bands. At inference time, routing reduces to a single k-NN lookup via a FAISS index, requiring no retraining when the expert pool changes and enabling microsecond latency. Across multiple benchmarks, CSCR consistently outperforms baselines, improving the accuracy-cost tradeoff by up to 25%, while generalizing robustly to unseen LLMs and out-of-distribution prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12473v1">Standardization of Neuromuscular Reflex Analysis -- Role of Fine-Tuned Vision-Language Model Consortium and OpenAI gpt-oss Reasoning LLM Enabled Decision Support System</a></div>
    <div class="paper-meta">
      📅 2025-08-17
    </div>
    <details class="paper-abstract">
      Accurate assessment of neuromuscular reflexes, such as the H-reflex, plays a critical role in sports science, rehabilitation, and clinical neurology. Traditional analysis of H-reflex EMG waveforms is subject to variability and interpretation bias among clinicians and researchers, limiting reliability and standardization. To address these challenges, we propose a Fine-Tuned Vision-Language Model (VLM) Consortium and a reasoning Large-Language Model (LLM)-enabled Decision Support System for automated H-reflex waveform interpretation and diagnosis. Our approach leverages multiple VLMs, each fine-tuned on curated datasets of H-reflex EMG waveform images annotated with clinical observations, recovery timelines, and athlete metadata. These models are capable of extracting key electrophysiological features and predicting neuromuscular states, including fatigue, injury, and recovery, directly from EMG images and contextual metadata. Diagnostic outputs from the VLM consortium are aggregated using a consensus-based method and refined by a specialized reasoning LLM, which ensures robust, transparent, and explainable decision support for clinicians and sports scientists. The end-to-end platform orchestrates seamless communication between the VLM ensemble and the reasoning LLM, integrating prompt engineering strategies and automated reasoning workflows using LLM Agents. Experimental results demonstrate that this hybrid system delivers highly accurate, consistent, and interpretable H-reflex assessments, significantly advancing the automation and standardization of neuromuscular diagnostics. To our knowledge, this work represents the first integration of a fine-tuned VLM consortium with a reasoning LLM for image-based H-reflex analysis, laying the foundation for next-generation AI-assisted neuromuscular assessment and athlete monitoring platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.09958v2">Neural Bandit Based Optimal LLM Selection for a Pipeline of Tasks</a></div>
    <div class="paper-meta">
      📅 2025-08-17
      | 💬 Submitted to AAAI 2026
    </div>
    <details class="paper-abstract">
      With the increasing popularity of large language models (LLMs) for a variety of tasks, there has been a growing interest in strategies that can predict which out of a set of LLMs will yield a successful answer at low cost. This problem promises to become more and more relevant as providers like Microsoft allow users to easily create custom LLM "assistants" specialized to particular types of queries. However, some tasks (i.e., queries) may be too specialized and difficult for a single LLM to handle alone. These applications often benefit from breaking down the task into smaller subtasks, each of which can then be executed by a LLM expected to perform well on that specific subtask. For example, in extracting a diagnosis from medical records, one can first select an LLM to summarize the record, select another to validate the summary, and then select another, possibly different, LLM to extract the diagnosis from the summarized record. Unlike existing LLM selection or routing algorithms, this setting requires that we select a sequence of LLMs, with the output of each LLM feeding into the next and potentially influencing its success. Thus, unlike single LLM selection, the quality of each subtask's output directly affects the inputs, and hence the cost and success rate, of downstream LLMs, creating complex performance dependencies that must be learned and accounted for during selection. We propose a neural contextual bandit-based algorithm that trains neural networks that model LLM success on each subtask in an online manner, thus learning to guide the LLM selections for the different subtasks, even in the absence of historical LLM performance data. Experiments on telecommunications question answering and medical diagnosis prediction datasets illustrate the effectiveness of our proposed approach compared to other LLM selection algorithms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17771v2">2SSP: A Two-Stage Framework for Structured Pruning of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-17
      | 💬 Published in Transactions on Machine Learning Research (TMLR)
    </div>
    <details class="paper-abstract">
      We propose a novel Two-Stage framework for Structured Pruning (\textsc{2SSP}) for pruning Large Language Models (LLMs), which combines two different strategies of pruning, namely Width and Depth Pruning. The first stage (Width Pruning) removes entire neurons, hence their corresponding rows and columns, aiming to preserve the connectivity among the pruned structures in the intermediate state of the Feed-Forward Networks in each Transformer block. This is done based on an importance score measuring the impact of each neuron on the output magnitude. The second stage (Depth Pruning), instead, removes entire Attention submodules. This is done by applying an iterative process that removes the Attention with the minimum impact on a given metric of interest (in our case, perplexity). We also propose a novel mechanism to balance the sparsity rate of the two stages w.r.t. to the desired global sparsity. We test \textsc{2SSP} on four LLM families and three sparsity rates (25\%, 37.5\%, and 50\%), measuring the resulting perplexity over three language modeling datasets as well as the performance over six downstream tasks. Our method consistently outperforms five state-of-the-art competitors over three language modeling and six downstream tasks, with an up to two-order-of-magnitude gain in terms of pruning time. The code is available at https://github.com/FabrizioSandri/2SSP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20655v2">CoGrader: Transforming Instructors' Assessment of Project Reports through Collaborative LLM Integration</a></div>
    <div class="paper-meta">
      📅 2025-08-17
    </div>
    <details class="paper-abstract">
      Grading project reports are increasingly significant in today's educational landscape, where they serve as key assessments of students' comprehensive problem-solving abilities. However, it remains challenging due to the multifaceted evaluation criteria involved, such as creativity and peer-comparative achievement. Meanwhile, instructors often struggle to maintain fairness throughout the time-consuming grading process. Recent advances in AI, particularly large language models, have demonstrated potential for automating simpler grading tasks, such as assessing quizzes or basic writing quality. However, these tools often fall short when it comes to complex metrics, like design innovation and the practical application of knowledge, that require an instructor's educational insights into the class situation. To address this challenge, we conducted a formative study with six instructors and developed CoGrader, which introduces a novel grading workflow combining human-LLM collaborative metrics design, benchmarking, and AI-assisted feedback. CoGrader was found effective in improving grading efficiency and consistency while providing reliable peer-comparative feedback to students. We also discuss design insights and ethical considerations for the development of human-AI collaborative grading systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.10027v2">LLMCARE: Alzheimer's Detection via Transformer Models Enhanced by LLM-Generated Synthetic Data</a></div>
    <div class="paper-meta">
      📅 2025-08-17
    </div>
    <details class="paper-abstract">
      Alzheimer's disease and related dementias (ADRD) affect approximately five million older adults in the U.S., yet over half remain undiagnosed. Speech-based natural language processing (NLP) offers a promising, scalable approach to detect early cognitive decline through linguistic markers. To develop and evaluate a screening pipeline that (i) fuses transformer embeddings with handcrafted linguistic features, (ii) tests data augmentation using synthetic speech generated by large language models (LLMs), and (iii) benchmarks unimodal and multimodal LLM classifiers for ADRD detection. Transcripts from the DementiaBank "cookie-theft" task (n = 237) were used. Ten transformer models were evaluated under three fine-tuning strategies. A fusion model combined embeddings from the top-performing transformer with 110 lexical-derived linguistic features. Five LLMs (LLaMA-8B/70B, MedAlpaca-7B, Ministral-8B, GPT-4o) were fine-tuned to generate label-conditioned synthetic speech, which was used to augment training data. Three multimodal models (GPT-4o, Qwen-Omni, Phi-4) were tested for speech-text classification in zero-shot and fine-tuned settings. The fusion model achieved F1 = 83.3 (AUC = 89.5), outperforming linguistic or transformer-only baselines. Augmenting training data with 2x MedAlpaca-7B synthetic speech increased F1 to 85.7. Fine-tuning significantly improved unimodal LLM classifiers (e.g., MedAlpaca: F1 = 47.3 -> 78.5 F1). Current multimodal models demonstrated lower performance (GPT-4o = 70.2 F1; Qwen = 66.0). Performance gains aligned with the distributional similarity between synthetic and real speech. Integrating transformer embeddings with linguistic features enhances ADRD detection from speech. Clinically tuned LLMs effectively support both classification and data augmentation, while further advancement is needed in multimodal modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.01077v5">Emoji Attack: Enhancing Jailbreak Attacks Against Judge LLM Detection</a></div>
    <div class="paper-meta">
      📅 2025-08-16
    </div>
    <details class="paper-abstract">
      Jailbreaking techniques trick Large Language Models (LLMs) into producing restricted output, posing a potential threat. One line of defense is to use another LLM as a Judge to evaluate the harmfulness of generated text. However, we reveal that these Judge LLMs are vulnerable to token segmentation bias, an issue that arises when delimiters alter the tokenization process, splitting words into smaller sub-tokens. This alters the embeddings of the entire sequence, reducing detection accuracy and allowing harmful content to be misclassified as safe. In this paper, we introduce Emoji Attack, a novel strategy that amplifies existing jailbreak prompts by exploiting token segmentation bias. Our method leverages in-context learning to systematically insert emojis into text before it is evaluated by a Judge LLM, inducing embedding distortions that significantly lower the likelihood of detecting unsafe content. Unlike traditional delimiters, emojis also introduce semantic ambiguity, making them particularly effective in this attack. Through experiments on state-of-the-art Judge LLMs, we demonstrate that Emoji Attack substantially reduces the unsafe prediction rate, bypassing existing safeguards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12175v1">Invitation Is All You Need! Promptware Attacks Against LLM-Powered Assistants in Production Are Practical and Dangerous</a></div>
    <div class="paper-meta">
      📅 2025-08-16
      | 💬 https://sites.google.com/view/invitation-is-all-you-need/home
    </div>
    <details class="paper-abstract">
      The growing integration of LLMs into applications has introduced new security risks, notably known as Promptware - maliciously engineered prompts designed to manipulate LLMs to compromise the CIA triad of these applications. While prior research warned about a potential shift in the threat landscape for LLM-powered applications, the risk posed by Promptware is frequently perceived as low. In this paper, we investigate the risk Promptware poses to users of Gemini-powered assistants (web application, mobile application, and Google Assistant). We propose a novel Threat Analysis and Risk Assessment (TARA) framework to assess Promptware risks for end users. Our analysis focuses on a new variant of Promptware called Targeted Promptware Attacks, which leverage indirect prompt injection via common user interactions such as emails, calendar invitations, and shared documents. We demonstrate 14 attack scenarios applied against Gemini-powered assistants across five identified threat classes: Short-term Context Poisoning, Permanent Memory Poisoning, Tool Misuse, Automatic Agent Invocation, and Automatic App Invocation. These attacks highlight both digital and physical consequences, including spamming, phishing, disinformation campaigns, data exfiltration, unapproved user video streaming, and control of home automation devices. We reveal Promptware's potential for on-device lateral movement, escaping the boundaries of the LLM-powered application, to trigger malicious actions using a device's applications. Our TARA reveals that 73% of the analyzed threats pose High-Critical risk to end users. We discuss mitigations and reassess the risk (in response to deployed mitigations) and show that the risk could be reduced significantly to Very Low-Medium. We disclosed our findings to Google, which deployed dedicated mitigations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12158v1">LLM-as-a-Judge for Privacy Evaluation? Exploring the Alignment of Human and LLM Perceptions of Privacy in Textual Data</a></div>
    <div class="paper-meta">
      📅 2025-08-16
      | 💬 13 pages, 3 figures, 4 tables. Accepted to HAIPS @ CCS 2025
    </div>
    <details class="paper-abstract">
      Despite advances in the field of privacy-preserving Natural Language Processing (NLP), a significant challenge remains the accurate evaluation of privacy. As a potential solution, using LLMs as a privacy evaluator presents a promising approach $\unicode{x2013}$ a strategy inspired by its success in other subfields of NLP. In particular, the so-called $\textit{LLM-as-a-Judge}$ paradigm has achieved impressive results on a variety of natural language evaluation tasks, demonstrating high agreement rates with human annotators. Recognizing that privacy is both subjective and difficult to define, we investigate whether LLM-as-a-Judge can also be leveraged to evaluate the privacy sensitivity of textual data. Furthermore, we measure how closely LLM evaluations align with human perceptions of privacy in text. Resulting from a study involving 10 datasets, 13 LLMs, and 677 human survey participants, we confirm that privacy is indeed a difficult concept to measure empirically, exhibited by generally low inter-human agreement rates. Nevertheless, we find that LLMs can accurately model a global human privacy perspective, and through an analysis of human and LLM reasoning patterns, we discuss the merits and limitations of LLM-as-a-Judge for privacy evaluation in textual data. Our findings pave the way for exploring the feasibility of LLMs as privacy evaluators, addressing a core challenge in solving pressing privacy issues with innovative technical solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13811v2">Can LLMs Handle WebShell Detection? Overcoming Detection Challenges with Behavioral Function-Aware Framework</a></div>
    <div class="paper-meta">
      📅 2025-08-16
      | 💬 Published as a conference paper at COLM 2025
    </div>
    <details class="paper-abstract">
      WebShell attacks, where malicious scripts are injected into web servers, pose a significant cybersecurity threat. Traditional ML and DL methods are often hampered by challenges such as the need for extensive training data, catastrophic forgetting, and poor generalization. Recently, Large Language Models have emerged as powerful alternatives for code-related tasks, but their potential in WebShell detection remains underexplored. In this paper, we make two contributions: (1) a comprehensive evaluation of seven LLMs, including GPT-4, LLaMA 3.1 70B, and Qwen 2.5 variants, benchmarked against traditional sequence- and graph-based methods using a dataset of 26.59K PHP scripts, and (2) the Behavioral Function-Aware Detection (BFAD) framework, designed to address the specific challenges of applying LLMs to this domain. Our framework integrates three components: a Critical Function Filter that isolates malicious PHP function calls, a Context-Aware Code Extraction strategy that captures the most behaviorally indicative code segments, and Weighted Behavioral Function Profiling that enhances in-context learning by prioritizing the most relevant demonstrations based on discriminative function-level profiles. Our results show that, stemming from their distinct analytical strategies, larger LLMs achieve near-perfect precision but lower recall, while smaller models exhibit the opposite trade-off. However, all baseline models lag behind previous SOTA methods. With the application of BFAD, the performance of all LLMs improves significantly, yielding an average F1 score increase of 13.82%. Notably, larger models now outperform SOTA benchmarks, while smaller models such as Qwen-2.5-Coder-3B achieve performance competitive with traditional methods. This work is the first to explore the feasibility and limitations of LLMs for WebShell detection and provides solutions to address the challenges in this task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12096v1">STEM: Efficient Relative Capability Evaluation of LLMs through Structured Transition Samples</a></div>
    <div class="paper-meta">
      📅 2025-08-16
      | 💬 Submit to AAAI 2026
    </div>
    <details class="paper-abstract">
      Evaluating large language models (LLMs) has become increasingly challenging as model capabilities advance rapidly. While recent models often achieve higher scores on standard benchmarks, these improvements do not consistently reflect enhanced real-world reasoning capabilities. Moreover, widespread overfitting to public benchmarks and the high computational cost of full evaluations have made it both expensive and less effective to distinguish meaningful differences between models. To address these challenges, we propose the \textbf{S}tructured \textbf{T}ransition \textbf{E}valuation \textbf{M}ethod (STEM), a lightweight and interpretable evaluation framework for efficiently estimating the relative capabilities of LLMs. STEM identifies \textit{significant transition samples} (STS) by analyzing consistent performance transitions among LLMs of the same architecture but varying parameter scales. These samples enable STEM to effectively estimate the capability position of an unknown model. Qwen3 model family is applied to construct the STS pool on six diverse and representative benchmarks. To assess generalizability. Experimental results indicate that STEM reliably captures performance trends, aligns with ground-truth rankings of model capability. These findings highlight STEM as a practical and scalable method for fine-grained, architecture-agnostic evaluation of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12086v1">J6: Jacobian-Driven Role Attribution for Multi-Objective Prompt Optimization in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-16
      | 💬 9 pages, 3 tables, 1 algorithm
    </div>
    <details class="paper-abstract">
      In large language model (LLM) adaptation, balancing multiple optimization objectives such as improving factuality (heat) and increasing confidence (via low entropy) poses a fundamental challenge, especially when prompt parameters (e.g., hidden-layer insertions h and embedding modifications w) interact in non-trivial ways. Existing multi-objective optimization strategies often rely on scalar gradient aggregation, ignoring the deeper geometric structure between objectives and parameters. We propose J6, a structured Jacobian-based method that decomposes the gradient interaction matrix into six interpretable components. This decomposition enables both hard decision-making (e.g., choosing the dominant update direction via argmax) and soft strategies (e.g., attention-style weighting via softmax over J6), forming a dynamic update framework that adapts to local conflict and synergy. Moreover, the interpretable structure of J6 provides insight into parameter attribution, task interference, and geometry-aligned adaptation. Our work introduces a principled and extensible mechanism for conflict-aware prompt optimization, and opens a new avenue for incorporating structured Jacobian reasoning into multi-objective neural tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12072v1">Mitigating Jailbreaks with Intent-Aware LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-16
    </div>
    <details class="paper-abstract">
      Despite extensive safety-tuning, large language models (LLMs) remain vulnerable to jailbreak attacks via adversarially crafted instructions, reflecting a persistent trade-off between safety and task performance. In this work, we propose Intent-FT, a simple and lightweight fine-tuning approach that explicitly trains LLMs to infer the underlying intent of an instruction before responding. By fine-tuning on a targeted set of adversarial instructions, Intent-FT enables LLMs to generalize intent deduction to unseen attacks, thereby substantially improving their robustness. We comprehensively evaluate both parametric and non-parametric attacks across open-source and proprietary models, considering harmfulness from attacks, utility, over-refusal, and impact against white-box threats. Empirically, Intent-FT consistently mitigates all evaluated attack categories, with no single attack exceeding a 50\% success rate -- whereas existing defenses remain only partially effective. Importantly, our method preserves the model's general capabilities and reduces excessive refusals on benign instructions containing superficially harmful keywords. Furthermore, models trained with Intent-FT accurately identify hidden harmful intent in adversarial attacks, and these learned intentions can be effectively transferred to enhance vanilla model defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12043v1">Talk Less, Fly Lighter: Autonomous Semantic Compression for UAV Swarm Communication via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-16
    </div>
    <details class="paper-abstract">
      The rapid adoption of Large Language Models (LLMs) in unmanned systems has significantly enhanced the semantic understanding and autonomous task execution capabilities of Unmanned Aerial Vehicle (UAV) swarms. However, limited communication bandwidth and the need for high-frequency interactions pose severe challenges to semantic information transmission within the swarm. This paper explores the feasibility of LLM-driven UAV swarms for autonomous semantic compression communication, aiming to reduce communication load while preserving critical task semantics. To this end, we construct four types of 2D simulation scenarios with different levels of environmental complexity and design a communication-execution pipeline that integrates system prompts with task instruction prompts. On this basis, we systematically evaluate the semantic compression performance of nine mainstream LLMs in different scenarios and analyze their adaptability and stability through ablation studies on environmental complexity and swarm size. Experimental results demonstrate that LLM-based UAV swarms have the potential to achieve efficient collaborative communication under bandwidth-constrained and multi-hop link conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12040v1">Mind the Generation Process: Fine-Grained Confidence Estimation During LLM Generation</a></div>
    <div class="paper-meta">
      📅 2025-08-16
      | 💬 The initial versin was made in August 2024
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have demonstrated remarkable performance across diverse tasks, they fundamentally lack self-awareness and frequently exhibit overconfidence, assigning high confidence scores to incorrect predictions. Accurate confidence estimation is therefore critical for enhancing the trustworthiness and reliability of LLM-generated outputs. However, existing approaches suffer from coarse-grained scoring mechanisms that fail to provide fine-grained, continuous confidence estimates throughout the generation process. To address these limitations, we introduce FineCE, a novel confidence estimation method that delivers accurate, fine-grained confidence scores during text generation. Specifically, we first develop a comprehensive pipeline for constructing training data that effectively captures the underlying probabilistic distribution of LLM responses, and then train a model to predict confidence scores for arbitrary text sequences in a supervised manner. Furthermore, we propose a Backward Confidence Integration (BCI) strategy that leverages information from the subsequent text to enhance confidence estimation for the current sequence during inference. We also introduce three strategies for identifying optimal positions to perform confidence estimation within the generation process. Extensive experiments on multiple benchmark datasets demonstrate that FineCE consistently outperforms existing classical confidence estimation methods. Our code and all baselines used in the paper are available on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12031v1">Learning Wisdom from Errors: Promoting LLM's Continual Relation Learning through Exploiting Error Cases</a></div>
    <div class="paper-meta">
      📅 2025-08-16
    </div>
    <details class="paper-abstract">
      Continual Relation Extraction (CRE) aims to continually learn new emerging relations while avoiding catastrophic forgetting. Existing CRE methods mainly use memory replay and contrastive learning to mitigate catastrophic forgetting. However, these methods do not attach importance to the error cases that can reveal the model's cognitive biases more effectively. To address this issue, we propose an instruction-based continual contrastive tuning approach for Large Language Models (LLMs) in CRE. Different from existing CRE methods that typically handle the training and memory data in a unified manner, this approach splits the training and memory data of each task into two parts respectively based on the correctness of the initial responses and treats them differently through dual-task fine-tuning. In addition, leveraging the advantages of LLM's instruction-following ability, we propose a novel instruction-based contrastive tuning strategy for LLM to continuously correct current cognitive biases with the guidance of previous data in an instruction-tuning manner, which mitigates the gap between old and new relations in a more suitable way for LLMs. We experimentally evaluate our model on TACRED and FewRel, and the results show that our model achieves new state-of-the-art CRE performance with significant improvements, demonstrating the importance of specializing in exploiting error cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03865v2">OrthoRank: Token Selection via Sink Token Orthogonality for Efficient LLM inference</a></div>
    <div class="paper-meta">
      📅 2025-08-16
      | 💬 ICML 2025 (final version)
    </div>
    <details class="paper-abstract">
      Attention mechanisms are central to the success of large language models (LLMs), enabling them to capture intricate token dependencies and implicitly assign importance to each token. Recent studies have revealed the sink token, which receives disproportionately high attention despite their limited semantic role. In this paper, we first expand the relationship between the sink token and other tokens, moving beyond attention to explore their similarity in hidden states, considering the layer depth. We observe that as the layers get deeper, the cosine similarity between the normalized hidden states of the sink token and those of other tokens increases, and that the normalized hidden states of the sink token exhibit negligible changes. These imply that other tokens consistently are directed toward the sink token throughout the layers. Next, we propose a dynamic token selection method, called OrthoRank, using these findings to select important tokens. Specifically, in a certain layer, we define token importance by the speed at which the token moves toward the sink token. This is converted into orthogonality with the sink token, meaning that tokens that are more orthogonal to the sink token are assigned greater importance. Finally, through extensive experiments, we demonstrated that our method results in lower perplexity and higher zero-shot accuracy compared to layer pruning methods at the same sparsity ratio with comparable throughput, while also achieving superior performance on LongBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07534v2">From Trial-and-Error to Improvement: A Systematic Analysis of LLM Exploration Mechanisms in RLVR</a></div>
    <div class="paper-meta">
      📅 2025-08-16
      | 💬 27pages,25figures. arXiv admin note: text overlap with arXiv:2508.02260
    </div>
    <details class="paper-abstract">
      Reinforcement learning with verifiable rewards (RLVR) has emerged as a powerful paradigm for enhancing the reasoning capabilities of large language models (LLMs). Unlike traditional RL approaches, RLVR leverages rule-based feedback to guide LLMs in generating and refining complex reasoning chains -- a process critically dependent on effective exploration strategies. While prior work has demonstrated RLVR's empirical success, the fundamental mechanisms governing LLMs' exploration behaviors remain underexplored. This technical report presents a systematic investigation of exploration capacities in RLVR, covering four main aspects: (1) exploration space shaping, where we develop quantitative metrics to characterize LLMs' capability boundaries; (2) entropy-performance exchange, analyzed across training stages, individual instances, and token-level patterns; and (3) RL performance optimization, examining methods to effectively translate exploration gains into measurable improvements. By unifying previously identified insights with new empirical evidence, this work aims to provide a foundational framework for advancing RLVR systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11987v1">FutureX: An Advanced Live Benchmark for LLM Agents in Future Prediction</a></div>
    <div class="paper-meta">
      📅 2025-08-16
      | 💬 Technical report, 51 pages
    </div>
    <details class="paper-abstract">
      Future prediction is a complex task for LLM agents, requiring a high level of analytical thinking, information gathering, contextual understanding, and decision-making under uncertainty. Agents must not only gather and interpret vast amounts of dynamic information but also integrate diverse data sources, weigh uncertainties, and adapt predictions based on emerging trends, just as human experts do in fields like politics, economics, and finance. Despite its importance, no large-scale benchmark exists for evaluating agents on future prediction, largely due to challenges in handling real-time updates and retrieving timely, accurate answers. To address this, we introduce $\textbf{FutureX}$, a dynamic and live evaluation benchmark specifically designed for LLM agents performing future prediction tasks. FutureX is the largest and most diverse live benchmark for future prediction, supporting real-time daily updates and eliminating data contamination through an automated pipeline for question gathering and answer collection. We evaluate 25 LLM/agent models, including those with reasoning, search capabilities, and integration of external tools such as the open-source Deep Research Agent and closed-source Deep Research models. This comprehensive evaluation assesses agents' adaptive reasoning and performance in dynamic environments. Additionally, we provide in-depth analyses of agents' failure modes and performance pitfalls in future-oriented tasks, including the vulnerability to fake web pages and the temporal validity. Our goal is to establish a dynamic, contamination-free evaluation standard that drives the development of LLM agents capable of performing at the level of professional human analysts in complex reasoning and predictive thinking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11958v1">Clean Code, Better Models: Enhancing LLM Performance with Smell-Cleaned Dataset</a></div>
    <div class="paper-meta">
      📅 2025-08-16
    </div>
    <details class="paper-abstract">
      The Large Language Models (LLMs) have demonstrated great potential in code-related tasks. However, most research focuses on improving the output quality of LLMs (e.g., correctness), and less attention has been paid to the LLM input (e.g., the training code quality). Given that code smells are widely existed in practice and can negatively impact software maintainability and readability, this study takes the first systematic research to assess and improve dataset quality in terms of code smells. In this work, we first conduct a preliminary study to explore the presence of code smells in a popular benchmark dataset (i.e., CodeSearchNet-Python}) and evaluate the output of several popular LLMs (i.e., DeepSeek-Coder, CodeLlama, and MagiCoder), revealing that code smell issues extensively exist in LLM's input (e.g., benchmark dataset) and output (e.g., generated code). We then conduct our systematic research by taking three main steps: Firstly, we propose an LLM-based code smell cleaning tool, named SmellCC, which automatically refactors and removes code smells. To evaluate the correctness of the code refactoring, we construct a test set of 50 repositories sourced from the CodeSearchNet-Python benchmark for functional testing. Then we apply our curated smell-cleaned dataset to fine-tune two LLMs (i.e., DeepSeek-V2 and Qwen-Coder) to explore their potential for generating high-quality code. Thirdly, we investigate the impact of code smells on two downstream tasks: code completion and code search. Lastly, we derive several actionable implications for software engineering researchers and industry practitioners from our findings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11944v1">CHBench: A Cognitive Hierarchy Benchmark for Evaluating Strategic Reasoning Capability of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-16
    </div>
    <details class="paper-abstract">
      Game-playing ability serves as an indicator for evaluating the strategic reasoning capability of large language models (LLMs). While most existing studies rely on utility performance metrics, which are not robust enough due to variations in opponent behavior and game structure. To address this limitation, we propose \textbf{Cognitive Hierarchy Benchmark (CHBench)}, a novel evaluation framework inspired by the cognitive hierarchy models from behavioral economics. We hypothesize that agents have bounded rationality -- different agents behave at varying reasoning depths/levels. We evaluate LLMs' strategic reasoning through a three-phase systematic framework, utilizing behavioral data from six state-of-the-art LLMs across fifteen carefully selected normal-form games. Experiments show that LLMs exhibit consistent strategic reasoning levels across diverse opponents, confirming the framework's robustness and generalization capability. We also analyze the effects of two key mechanisms (Chat Mechanism and Memory Mechanism) on strategic reasoning performance. Results indicate that the Chat Mechanism significantly degrades strategic reasoning, whereas the Memory Mechanism enhances it. These insights position CHBench as a promising tool for evaluating LLM capabilities, with significant potential for future research and practical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11927v1">LLMs Struggle with NLI for Perfect Aspect: A Cross-Linguistic Study in Chinese and Japanese</a></div>
    <div class="paper-meta">
      📅 2025-08-16
      | 💬 9 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Unlike English, which uses distinct forms (e.g., had, has, will have) to mark the perfect aspect across tenses, Chinese and Japanese lack separate grammatical forms for tense within the perfect aspect, which complicates Natural Language Inference (NLI). Focusing on the perfect aspect in these languages, we construct a linguistically motivated, template-based NLI dataset (1,350 pairs per language). Experiments reveal that even advanced LLMs struggle with temporal inference, particularly in detecting subtle tense and reference-time shifts. These findings highlight model limitations and underscore the need for cross-linguistic evaluation in temporal semantics. Our dataset is available at https://github.com/Lujie2001/CrossNLI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05571v3">iFairy: the First 2-bit Complex LLM with All Parameters in $\{\pm1, \pm i\}$</a></div>
    <div class="paper-meta">
      📅 2025-08-16
      | 💬 15 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Quantization-Aware Training (QAT) integrates quantization into the training loop, enabling LLMs to learn robust low-bit representations, and is widely recognized as one of the most promising research directions. All current QAT research focuses on minimizing quantization error on full-precision models, where the full-precision accuracy acts as an upper bound (accuracy ceiling). No existing method has even attempted to surpass this ceiling. To break this ceiling, we propose a new paradigm: raising the ceiling (full-precision model), and then still quantizing it efficiently into 2 bits. We propose Fairy$\pm i$, the first 2-bit quantization framework for complex-valued LLMs. Specifically, our method leverages the representational advantages of the complex domain to boost full-precision accuracy. We map weights to the fourth roots of unity $\{\pm1, \pm i\}$, forming a perfectly symmetric and information-theoretically optimal 2-bit representation. Importantly, each quantized weight has either a zero real or imaginary part, enabling multiplication-free inference using only additions and element swaps. Experimental results show that Fairy$\pm i$ outperforms the ceiling of existing 2-bit quantization approaches in terms of both PPL and downstream tasks, while maintaining strict storage and compute efficiency. This work opens a new direction for building highly accurate and practical LLMs under extremely low-bit constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11915v1">CORE: Measuring Multi-Agent LLM Interaction Quality under Game-Theoretic Pressures</a></div>
    <div class="paper-meta">
      📅 2025-08-16
    </div>
    <details class="paper-abstract">
      Game-theoretic interactions between agents with Large Language Models (LLMs) have revealed many emergent capabilities, yet the linguistic diversity of these interactions has not been sufficiently quantified. In this paper, we present the Conversational Robustness Evaluation Score: CORE, a metric to quantify the effectiveness of language use within multi-agent systems across different game-theoretic interactions. CORE integrates measures of cluster entropy, lexical repetition, and semantic similarity, providing a direct lens of dialog quality. We apply CORE to pairwise LLM dialogs across competitive, cooperative, and neutral settings, further grounding our analysis in Zipf's and Heaps' Laws to characterize word frequency distributions and vocabulary growth. Our findings show that cooperative settings exhibit both steeper Zipf distributions and higher Heap exponents, indicating more repetition alongside greater vocabulary expansion. In contrast, competitive interactions display lower Zipf and Heaps exponents, reflecting less repetition and more constrained vocabularies. These results provide new insights into how social incentives influence language adaptation, and highlight CORE as a robust diagnostic for measuring linguistic robustness in multi-agent LLM systems. Our code is available at https://github.com/psyonp/core.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01235v2">NarraGuide: an LLM-based Narrative Mobile Robot for Remote Place Exploration</a></div>
    <div class="paper-meta">
      📅 2025-08-16
    </div>
    <details class="paper-abstract">
      Robotic telepresence enables users to navigate and experience remote environments. However, effective navigation and situational awareness depend on users' prior knowledge of the environment, limiting the usefulness of these systems for exploring unfamiliar places. We explore how integrating location-aware LLM-based narrative capabilities into a mobile robot can support remote exploration. We developed a prototype system, called NarraGuide, that provides narrative guidance for users to explore and learn about a remote place through a dialogue-based interface. We deployed our prototype in a geology museum, where remote participants (n=20) used the robot to tour the museum. Our findings reveal how users perceived the robot's role, engaged in dialogue in the tour, and expressed preferences for bystander encountering. Our work demonstrates the potential of LLM-enabled robotic capabilities to deliver location-aware narrative guidance and enrich the experience of exploring remote environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13214v1">Too Easily Fooled? Prompt Injection Breaks LLMs on Frustratingly Simple Multiple-Choice Questions</a></div>
    <div class="paper-meta">
      📅 2025-08-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently demonstrated strong emergent abilities in complex reasoning and zero-shot generalization, showing unprecedented potential for LLM-as-a-judge applications in education, peer review, and data quality evaluation. However, their robustness under prompt injection attacks, where malicious instructions are embedded into the content to manipulate outputs, remains a significant concern. In this work, we explore a frustratingly simple yet effective attack setting to test whether LLMs can be easily misled. Specifically, we evaluate LLMs on basic arithmetic questions (e.g., "What is 3 + 2?") presented as either multiple-choice or true-false judgment problems within PDF files, where hidden prompts are injected into the file. Our results reveal that LLMs are indeed vulnerable to such hidden prompt injection attacks, even in these trivial scenarios, highlighting serious robustness risks for LLM-as-a-judge applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13201v1">Benchmarking LLM-based Agents for Single-cell Omics Analysis</a></div>
    <div class="paper-meta">
      📅 2025-08-16
    </div>
    <details class="paper-abstract">
      The surge in multimodal single-cell omics data exposes limitations in traditional, manually defined analysis workflows. AI agents offer a paradigm shift, enabling adaptive planning, executable code generation, traceable decisions, and real-time knowledge fusion. However, the lack of a comprehensive benchmark critically hinders progress. We introduce a novel benchmarking evaluation system to rigorously assess agent capabilities in single-cell omics analysis. This system comprises: a unified platform compatible with diverse agent frameworks and LLMs; multidimensional metrics assessing cognitive program synthesis, collaboration, execution efficiency, bioinformatics knowledge integration, and task completion quality; and 50 diverse real-world single-cell omics analysis tasks spanning multi-omics, species, and sequencing technologies. Our evaluation reveals that Grok-3-beta achieves state-of-the-art performance among tested agent frameworks. Multi-agent frameworks significantly enhance collaboration and execution efficiency over single-agent approaches through specialized role division. Attribution analyses of agent capabilities identify that high-quality code generation is crucial for task success, and self-reflection has the most significant overall impact, followed by retrieval-augmented generation (RAG) and planning. This work highlights persistent challenges in code generation, long-context handling, and context-aware knowledge retrieval, providing a critical empirical foundation and best practices for developing robust AI agents in computational biology.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11305v1">Defects4Log: Benchmarking LLMs for Logging Code Defect Detection and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-08-15
    </div>
    <details class="paper-abstract">
      Logging code is written by developers to capture system runtime behavior and plays a vital role in debugging, performance analysis, and system monitoring. However, defects in logging code can undermine the usefulness of logs and lead to misinterpretations. Although prior work has identified several logging defect patterns and provided valuable insights into logging practices, these studies often focus on a narrow range of defect patterns derived from limited sources (e.g., commit histories) and lack a systematic and comprehensive analysis. Moreover, large language models (LLMs) have demonstrated promising generalization and reasoning capabilities across a variety of code-related tasks, yet their potential for detecting logging code defects remains largely unexplored. In this paper, we derive a comprehensive taxonomy of logging code defects, which encompasses seven logging code defect patterns with 14 detailed scenarios. We further construct a benchmark dataset, \dataset, consisting of 164 developer-verified real-world logging defects. Then we propose an automated framework that leverages various prompting strategies and contextual information to evaluate LLMs' capability in detecting and reasoning logging code defects. Experimental results reveal that LLMs generally struggle to accurately detect and reason logging code defects based on the source code only. However, incorporating proper knowledge (e.g., detailed scenarios of defect patterns) can lead to 10.9\% improvement in detection accuracy. Overall, our findings provide actionable guidance for practitioners to avoid common defect patterns and establish a foundation for improving LLM-based reasoning in logging code defect detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11736v2">Personalized LLM for Generating Customized Responses to the Same Query from Different Users</a></div>
    <div class="paper-meta">
      📅 2025-08-15
      | 💬 Accepted by CIKM'25
    </div>
    <details class="paper-abstract">
      Existing work on large language model (LLM) personalization assigned different responding roles to LLMs, but overlooked the diversity of queriers. In this work, we propose a new form of querier-aware LLM personalization, generating different responses even for the same query from different queriers. We design a dual-tower model architecture with a cross-querier general encoder and a querier-specific encoder. We further apply contrastive learning with multi-view augmentation, pulling close the dialogue representations of the same querier, while pulling apart those of different queriers. To mitigate the impact of query diversity on querier-contrastive learning, we cluster the dialogues based on query similarity and restrict the scope of contrastive learning within each cluster. To address the lack of datasets designed for querier-aware personalization, we also build a multi-querier dataset from English and Chinese scripts, as well as WeChat records, called MQDialog, containing 173 queriers and 12 responders. Extensive evaluations demonstrate that our design significantly improves the quality of personalized response generation, achieving relative improvement of 8.4% to 48.7% in ROUGE-L scores and winning rates ranging from 54% to 82% compared with various baseline methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11291v1">Dynamic Quality-Latency Aware Routing for LLM Inference in Wireless Edge-Device Networks</a></div>
    <div class="paper-meta">
      📅 2025-08-15
      | 💬 accepted by IEEE/CIC ICCC workshop
    </div>
    <details class="paper-abstract">
      The integration of wireless communications and Large Language Models (LLMs) is poised to unlock ubiquitous intelligent services, yet deploying them in wireless edge-device collaborative environments presents a critical trade-off between inference quality and end-to-end latency. A fundamental mismatch exists between task complexity and resource allocation: offloading simple queries invites prohibitive latency, while on-device models lack the capacity for demanding computations. To address this challenge, we propose a dynamic, quality-latency aware routing framework that orchestrates inference between a lightweight model on the mobile device and a powerful model on the edge server. Our framework employs two distinct cost models: for single-turn queries, it fuses a BERT-predicted semantic score with communication and computation overheads; for multi-turn dialogues, it further quantifies context-aware costs arising from model switching and KV-cache management. While maintaining full inference quality, extensive experiments demonstrate that our framework cuts average response latency by 5-15% and reduces large model invocations by 10-20% against competitive baselines on MMLU, GSM8K, and MT-Bench-101 benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11290v1">SafeConstellations: Steering LLM Safety to Reduce Over-Refusals Through Task-Specific Trajectory</a></div>
    <div class="paper-meta">
      📅 2025-08-15
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      LLMs increasingly exhibit over-refusal behavior, where safety mechanisms cause models to reject benign instructions that superficially resemble harmful content. This phenomena diminishes utility in production applications that repeatedly rely on common prompt templates or applications that frequently rely on LLMs for specific tasks (e.g. sentiment analysis, language translation). Through comprehensive evaluation, we demonstrate that LLMs still tend to refuse responses to harmful instructions when those instructions are reframed to appear as benign tasks. Our mechanistic analysis reveal that LLMs follow distinct "constellation" patterns in embedding space as representations traverse layers, with each task maintaining consistent trajectories that shift predictably between refusal and non-refusal cases. We introduce SafeConstellations, an inference-time trajectory-shifting approach that tracks task-specific trajectory patterns and guides representations toward non-refusal pathways. By selectively guiding model behavior only on tasks prone to over-refusal, and by preserving general model behavior, our method reduces over-refusal rates by up to 73% with minimal impact on utility-offering a principled approach to mitigating over-refusals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11269v1">Inference performance evaluation for LLMs on edge devices with a novel benchmarking framework and metric</a></div>
    <div class="paper-meta">
      📅 2025-08-15
    </div>
    <details class="paper-abstract">
      With the significant success achieved by large language models (LLMs) like LLaMA, edge computing-based LLM inference services for mobile and PC are in high demand for data privacy. However, different edge platforms have different hardware characteristics and the large demand for memory capacity and bandwidth makes it very challenging to deploy and benchmark LLMs on edge devices. In this paper, we introduce a benchmarking tool named ELIB (edge LLM inference benchmarking) to evaluate LLM inference performance of different edge platforms, and propose a novel metric named MBU to indicate the percentage of the theoretically efficient use of available memory bandwidth for a specific model running on edge hardware to optimize memory usage. We deploy ELIB on three edge platforms and benchmark using five quantized models to optimize MBU in combination with other metrics such as FLOPS, throughput, latency and accuracy. And we analyze the results to derive the key factors, constraints, unpredictability in optimizing MBU that can guide deploying LLMs on more edge platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11260v1">UNVEILING: What Makes Linguistics Olympiad Puzzles Tricky for LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-08-15
      | 💬 Accepted to COLM 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated potential in reasoning tasks, but their performance on linguistics puzzles remains consistently poor. These puzzles, often derived from Linguistics Olympiad (LO) contests, provide a minimal contamination environment to assess LLMs' linguistic reasoning abilities across low-resource languages. This work analyses LLMs' performance on 629 problems across 41 low-resource languages by labelling each with linguistically informed features to unveil weaknesses. Our analyses show that LLMs struggle with puzzles involving higher morphological complexity and perform better on puzzles involving linguistic features that are also found in English. We also show that splitting words into morphemes as a pre-processing step improves solvability, indicating a need for more informed and language-specific tokenisers. These findings thus offer insights into some challenges in linguistic reasoning and modelling of low-resource languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11258v1">Group Fairness Meets the Black Box: Enabling Fair Algorithms on Closed LLMs via Post-Processing</a></div>
    <div class="paper-meta">
      📅 2025-08-15
    </div>
    <details class="paper-abstract">
      Instruction fine-tuned large language models (LLMs) enable a simple zero-shot or few-shot prompting paradigm, also known as in-context learning, for building prediction models. This convenience, combined with continued advances in LLM capability, has the potential to drive their adoption across a broad range of domains, including high-stakes applications where group fairness -- preventing disparate impacts across demographic groups -- is essential. The majority of existing approaches to enforcing group fairness on LLM-based classifiers rely on traditional fair algorithms applied via model fine-tuning or head-tuning on final-layer embeddings, but they are no longer applicable to closed-weight LLMs under the in-context learning setting, which include some of the most capable commercial models today, such as GPT-4, Gemini, and Claude. In this paper, we propose a framework for deriving fair classifiers from closed-weight LLMs via prompting: the LLM is treated as a feature extractor, and features are elicited from its probabilistic predictions (e.g., token log probabilities) using prompts strategically designed for the specified fairness criterion to obtain sufficient statistics for fair classification; a fair algorithm is then applied to these features to train a lightweight fair classifier in a post-hoc manner. Experiments on five datasets, including three tabular ones, demonstrate strong accuracy-fairness tradeoffs for the classifiers derived by our framework from both open-weight and closed-weight LLMs; in particular, our framework is data-efficient and outperforms fair classifiers trained on LLM embeddings (i.e., head-tuning) or from scratch on raw tabular features.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11257v1">Hallucination in LLM-Based Code Generation: An Automotive Case Study</a></div>
    <div class="paper-meta">
      📅 2025-08-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown significant potential in automating code generation tasks offering new opportunities across software engineering domains. However, their practical application remains limited due to hallucinations - outputs that appear plausible but are factually incorrect, unverifiable or nonsensical. This paper investigates hallucination phenomena in the context of code generation with a specific focus on the automotive domain. A case study is presented that evaluates multiple code LLMs for three different prompting complexities ranging from a minimal one-liner prompt to a prompt with Covesa Vehicle Signal Specifications (VSS) as additional context and finally to a prompt with an additional code skeleton. The evaluation reveals a high frequency of syntax violations, invalid reference errors and API knowledge conflicts in state-of-the-art models GPT-4.1, Codex and GPT-4o. Among the evaluated models, only GPT-4.1 and GPT-4o were able to produce a correct solution when given the most context-rich prompt. Simpler prompting strategies failed to yield a working result, even after multiple refinement iterations. These findings highlight the need for effective mitigation techniques to ensure the safe and reliable use of LLM generated code, especially in safety-critical domains such as automotive software systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.10450v3">TokenRec: Learning to Tokenize ID for LLM-based Generative Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-08-15
      | 💬 Accepted by IEEE TKDE. Codes and data are available at https://github.com/Quhaoh233/TokenRec
    </div>
    <details class="paper-abstract">
      There is a growing interest in utilizing large-scale language models (LLMs) to advance next-generation Recommender Systems (RecSys), driven by their outstanding language understanding and in-context learning capabilities. In this scenario, tokenizing (i.e., indexing) users and items becomes essential for ensuring a seamless alignment of LLMs with recommendations. While several studies have made progress in representing users and items through textual contents or latent representations, challenges remain in efficiently capturing high-order collaborative knowledge into discrete tokens that are compatible with LLMs. Additionally, the majority of existing tokenization approaches often face difficulties in generalizing effectively to new/unseen users or items that were not in the training corpus. To address these challenges, we propose a novel framework called TokenRec, which introduces not only an effective ID tokenization strategy but also an efficient retrieval paradigm for LLM-based recommendations. Specifically, our tokenization strategy, Masked Vector-Quantized (MQ) Tokenizer, involves quantizing the masked user/item representations learned from collaborative filtering into discrete tokens, thus achieving a smooth incorporation of high-order collaborative knowledge and a generalizable tokenization of users and items for LLM-based RecSys. Meanwhile, our generative retrieval paradigm is designed to efficiently recommend top-$K$ items for users to eliminate the need for the time-consuming auto-regressive decoding and beam search processes used by LLMs, thus significantly reducing inference time. Comprehensive experiments validate the effectiveness of the proposed methods, demonstrating that TokenRec outperforms competitive benchmarks, including both traditional recommender systems and emerging LLM-based recommender systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11222v1">ORFuzz: Fuzzing the "Other Side" of LLM Safety -- Testing Over-Refusal</a></div>
    <div class="paper-meta">
      📅 2025-08-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly exhibit over-refusal - erroneously rejecting benign queries due to overly conservative safety measures - a critical functional flaw that undermines their reliability and usability. Current methods for testing this behavior are demonstrably inadequate, suffering from flawed benchmarks and limited test generation capabilities, as highlighted by our empirical user study. To the best of our knowledge, this paper introduces the first evolutionary testing framework, ORFuzz, for the systematic detection and analysis of LLM over-refusals. ORFuzz uniquely integrates three core components: (1) safety category-aware seed selection for comprehensive test coverage, (2) adaptive mutator optimization using reasoning LLMs to generate effective test cases, and (3) OR-Judge, a human-aligned judge model validated to accurately reflect user perception of toxicity and refusal. Our extensive evaluations demonstrate that ORFuzz generates diverse, validated over-refusal instances at a rate (6.98% average) more than double that of leading baselines, effectively uncovering vulnerabilities. Furthermore, ORFuzz's outputs form the basis of ORFuzzSet, a new benchmark of 1,855 highly transferable test cases that achieves a superior 63.56% average over-refusal rate across 10 diverse LLMs, significantly outperforming existing datasets. ORFuzz and ORFuzzSet provide a robust automated testing framework and a valuable community resource, paving the way for developing more reliable and trustworthy LLM-based software systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10069v2">ElasticMM: Efficient Multimodal LLMs Serving with Elastic Multimodal Parallelism</a></div>
    <div class="paper-meta">
      📅 2025-08-15
    </div>
    <details class="paper-abstract">
      Multimodal large language models (MLLMs) extend LLMs to handle images, videos, and audio by incorporating feature extractors and projection modules. However, these additional components -- combined with complex inference pipelines and heterogeneous workloads -- introduce significant inference overhead. Therefore, efficiently serving MLLMs remains a major challenge. Current tightly coupled serving architectures struggle to distinguish between mixed request types or adapt parallelism strategies to different inference stages, leading to increased time-to-first-token (TTFT) latency and poor resource utilization. To address this, we introduce Elastic Multimodal Parallelism (EMP), a new serving paradigm that elastically adapts to resource heterogeneity across request types and inference stages. Building upon EMP, we develop ElasticMM, an MLLM serving system that (1) separates requests into independent modality groups with dynamic resource allocation via a modality-aware load balancer; (2) decouples inference stages and enables parallelism adjustment and adaptive scaling via elastic partition scheduling; and (3) improves inference efficiency through unified multimodal prefix caching and non-blocking encoding. Experiments on diverse real-world datasets show that ElasticMM outperforms state-of-the-art (SOTA) serving systems, reducing TTFT by up to 4.2x and achieving 3.2-4.5x higher throughput while meeting service-level objectives (SLOs).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.18013v2">A Survey on Recent Advances in LLM-Based Multi-turn Dialogue Systems</a></div>
    <div class="paper-meta">
      📅 2025-08-15
      | 💬 35 pages, 10 figures, ACM Computing Surveys
    </div>
    <details class="paper-abstract">
      This survey provides a comprehensive review of research on multi-turn dialogue systems, with a particular focus on multi-turn dialogue systems based on large language models (LLMs). This paper aims to (a) give a summary of existing LLMs and approaches for adapting LLMs to downstream tasks; (b) elaborate recent advances in multi-turn dialogue systems, covering both LLM-based open-domain dialogue (ODD) and task-oriented dialogue (TOD) systems, along with datasets and evaluation metrics; (c) discuss some future emphasis and recent research problems arising from the development of LLMs and the increasing demands on multi-turn dialogue systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04130v2">STORM: Token-Efficient Long Video Understanding for Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-15
    </div>
    <details class="paper-abstract">
      Recent advances in video-based multimodal large language models (Video-LLMs) have significantly improved video understanding by processing videos as sequences of image frames. However, many existing methods treat frames independently in the vision backbone, lacking explicit temporal modeling, which limits their ability to capture dynamic patterns and efficiently handle long videos. To address these limitations, we introduce STORM (Spatiotemporal TOken Reduction for Multimodal LLMs), a novel architecture incorporating a dedicated temporal encoder between the image encoder and the LLM. Our temporal encoder leverages the Mamba State Space Model to integrate temporal information into image tokens, generating enriched representations that preserve inter-frame dynamics across the entire video sequence. This enriched encoding not only enhances video reasoning capabilities but also enables effective token reduction strategies, including test-time sampling and training-based temporal and spatial pooling, substantially reducing computational demands on the LLM without sacrificing key temporal information. By integrating these techniques, our approach simultaneously reduces training and inference latency while improving performance, enabling efficient and robust video understanding over extended temporal contexts. Extensive evaluations show that STORM achieves state-of-the-art results across various long video understanding benchmarks (more than 5% improvement on MLVU and LongVideoBench) while reducing the computation costs by up to $8\times$ and the decoding latency by 2.4-2.9$\times$ for the fixed numbers of input frames. Project page is available at https://research.nvidia.com/labs/lpr/storm
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11816v1">LLM-Guided Planning and Summary-Based Scientific Text Simplification: DS@GT at CLEF 2025 SimpleText</a></div>
    <div class="paper-meta">
      📅 2025-08-15
      | 💬 Text Simplification, hallucination detection, LLMs, CLEF 2025, SimpleText, CEUR-WS
    </div>
    <details class="paper-abstract">
      In this paper, we present our approach for the CLEF 2025 SimpleText Task 1, which addresses both sentence-level and document-level scientific text simplification. For sentence-level simplification, our methodology employs large language models (LLMs) to first generate a structured plan, followed by plan-driven simplification of individual sentences. At the document level, we leverage LLMs to produce concise summaries and subsequently guide the simplification process using these summaries. This two-stage, LLM-based framework enables more coherent and contextually faithful simplifications of scientific text.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11779v1">A Multi-Task Evaluation of LLMs' Processing of Academic Text Input</a></div>
    <div class="paper-meta">
      📅 2025-08-15
    </div>
    <details class="paper-abstract">
      How much large language models (LLMs) can aid scientific discovery, notably in assisting academic peer review, is in heated debate. Between a literature digest and a human-comparable research assistant lies their practical application potential. We organize individual tasks that computer science studies employ in separate terms into a guided and robust workflow to evaluate LLMs' processing of academic text input. We employ four tasks in the assessment: content reproduction/comparison/scoring/reflection, each demanding a specific role of the LLM (oracle/judgmental arbiter/knowledgeable arbiter/collaborator) in assisting scholarly works, and altogether testing LLMs with questions that increasingly require intellectual capabilities towards a solid understanding of scientific texts to yield desirable solutions. We exemplify a rigorous performance evaluation with detailed instructions on the prompts. Adopting first-rate Information Systems articles at three top journals as the input texts and an abundant set of text metrics, we record a compromised performance of the leading LLM - Google's Gemini: its summary and paraphrase of academic text is acceptably reliable; using it to rank texts through pairwise text comparison is faintly scalable; asking it to grade academic texts is prone to poor discrimination; its qualitative reflection on the text is self-consistent yet hardly insightful to inspire meaningful research. This evidence against an endorsement of LLMs' text-processing capabilities is consistent across metric-based internal (linguistic assessment), external (comparing to the ground truth), and human evaluation, and is robust to the variations of the prompt. Overall, we do not recommend an unchecked use of LLMs in constructing peer reviews.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10125v2">D-LiFT: Improving LLM-based Decompiler Backend via Code Quality-driven Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-08-15
    </div>
    <details class="paper-abstract">
      As one of the key tools in many security tasks, decompilers reconstruct human-readable source code from binaries. Yet, despite recent advances, their outputs often suffer from syntactic and semantic errors and remain difficult to read. Recently, with the advent of large language models (LLMs), researchers began to explore the potential of LLMs to refine decompiler output. Nevertheless, our study of these approaches reveals their problems, such as introducing new errors and relying on unreliable accuracy validation. In this paper, we present D-LIFT, an enhanced decompiler-LLM pipeline with a fine-tuned LLM using code quality-aware reinforcement learning. Unlike prior work that overlooks preserving accuracy, D-LIFT adheres to a key principle for enhancing the quality of decompiled code: preserving accuracy while improving readability. Central to D-LIFT, we propose D-Score, an integrated code quality assessment system to score the decompiled source code from multiple aspects, and use it to guide reinforcement learning fine-tuning and to select the best output during inference. In line with our principle, D-Score assigns low scores to any inaccurate output and only awards higher scores for readability to code that passes the accuracy check. Our implementation, based on Ghidra and a range of LLMs, demonstrates significant improvements for the accurate decompiled code from the coreutils and util-linux projects. Compared to baseline LLMs without D-Score-driven fine-tuning, our trained LLMs produce 55.3% more improved decompiled functions, as measured by D-Score. Overall, D-LIFT improves the quality of 68.2% of all the functions produced by the native decompiler.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11733v1">SafeSieve: From Heuristics to Experience in Progressive Pruning for LLM-based Multi-Agent Communication</a></div>
    <div class="paper-meta">
      📅 2025-08-15
      | 💬 7 pages for main content, 5 figures, 4 tables
    </div>
    <details class="paper-abstract">
      LLM-based multi-agent systems exhibit strong collaborative capabilities but often suffer from redundant communication and excessive token overhead. Existing methods typically enhance efficiency through pretrained GNNs or greedy algorithms, but often isolate pre- and post-task optimization, lacking a unified strategy. To this end, we present SafeSieve, a progressive and adaptive multi-agent pruning algorithm that dynamically refines the inter-agent communication through a novel dual-mechanism. SafeSieve integrates initial LLM-based semantic evaluation with accumulated performance feedback, enabling a smooth transition from heuristic initialization to experience-driven refinement. Unlike existing greedy Top-k pruning methods, SafeSieve employs 0-extension clustering to preserve structurally coherent agent groups while eliminating ineffective links. Experiments across benchmarks (SVAMP, HumanEval, etc.) showcase that SafeSieve achieves 94.01% average accuracy while reducing token usage by 12.4%-27.8%. Results further demonstrate robustness under prompt injection attacks (1.23% average accuracy drop). In heterogeneous settings, SafeSieve reduces deployment costs by 13.3% while maintaining performance. These results establish SafeSieve as a robust, efficient, and scalable framework for practical multi-agent systems. Our code can be found in https://anonymous.4open.science/r/SafeSieve-D8F2FFUN.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13196v1">Contextual Attention-Based Multimodal Fusion of LLM and CNN for Sentiment Analysis</a></div>
    <div class="paper-meta">
      📅 2025-08-15
      | 💬 The 38th Canadian Conference on Artificial Intelligence ( 2025 )
    </div>
    <details class="paper-abstract">
      This paper introduces a novel approach for multimodal sentiment analysis on social media, particularly in the context of natural disasters, where understanding public sentiment is crucial for effective crisis management. Unlike conventional methods that process text and image modalities separately, our approach seamlessly integrates Convolutional Neural Network (CNN) based image analysis with Large Language Model (LLM) based text processing, leveraging Generative Pre-trained Transformer (GPT) and prompt engineering to extract sentiment relevant features from the CrisisMMD dataset. To effectively model intermodal relationships, we introduce a contextual attention mechanism within the fusion process. Leveraging contextual-attention layers, this mechanism effectively captures intermodality interactions, enhancing the model's comprehension of complex relationships between textual and visual data. The deep neural network architecture of our model learns from these fused features, leading to improved accuracy compared to existing baselines. Experimental results demonstrate significant advancements in classifying social media data into informative and noninformative categories across various natural disasters. Our model achieves a notable 2.43% increase in accuracy and 5.18% in F1-score, highlighting its efficacy in processing complex multimodal data. Beyond quantitative metrics, our approach provides deeper insight into the sentiments expressed during crises. The practical implications extend to real time disaster management, where enhanced sentiment analysis can optimize the accuracy of emergency interventions. By bridging the gap between multimodal analysis, LLM powered text understanding, and disaster response, our work presents a promising direction for Artificial Intelligence (AI) driven crisis management solutions. Keywords:
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05147v2">Pr$εε$mpt: Sanitizing Sensitive Prompts for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-15
    </div>
    <details class="paper-abstract">
      The rise of large language models (LLMs) has introduced new privacy challenges, particularly during inference where sensitive information in prompts may be exposed to proprietary LLM APIs. In this paper, we address the problem of formally protecting the sensitive information contained in a prompt while maintaining response quality. To this end, first, we introduce a cryptographically inspired notion of a prompt sanitizer which transforms an input prompt to protect its sensitive tokens. Second, we propose Pr$\epsilon\epsilon$mpt, a novel system that implements a prompt sanitizer. Pr$\epsilon\epsilon$mpt categorizes sensitive tokens into two types: (1) those where the LLM's response depends solely on the format (such as SSNs, credit card numbers), for which we use format-preserving encryption (FPE); and (2) those where the response depends on specific values, (such as age, salary) for which we apply metric differential privacy (mDP). Our evaluation demonstrates that Pr$\epsilon\epsilon$mpt is a practical method to achieve meaningful privacy guarantees, while maintaining high utility compared to unsanitized prompts, and outperforming prior methods
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10054v2">Omni-DPO: A Dual-Perspective Paradigm for Dynamic Preference Learning of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-15
    </div>
    <details class="paper-abstract">
      Direct Preference Optimization (DPO) has become a cornerstone of reinforcement learning from human feedback (RLHF) due to its simplicity and efficiency. However, existing DPO-based approaches typically treat all preference pairs uniformly, ignoring critical variations in their inherent quality and learning utility, leading to suboptimal data utilization and performance. To address this challenge, we propose Omni-DPO, a dual-perspective optimization framework that jointly accounts for (1) the inherent quality of each preference pair and (2) the model's evolving performance on those pairs. By adaptively weighting samples according to both data quality and the model's learning dynamics during training, Omni-DPO enables more effective training data utilization and achieves better performance. Experimental results on various models and benchmarks demonstrate the superiority and generalization capabilities of Omni-DPO. On textual understanding tasks, Gemma-2-9b-it finetuned with Omni-DPO beats the leading LLM, Claude 3 Opus, by a significant margin of 6.7 points on the Arena-Hard benchmark. On mathematical reasoning tasks, Omni-DPO consistently outperforms the baseline methods across all benchmarks, providing strong empirical evidence for the effectiveness and robustness of our approach. Code and models will be available at https://github.com/pspdada/Omni-DPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08715v2">MultiAiTutor: Child-Friendly Educational Multilingual Speech Generation Tutor with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-15
      | 💬 We are withdrawing the manuscript to revise the title and contents of figures for better alignment with the paper's contributions
    </div>
    <details class="paper-abstract">
      Generative speech models have demonstrated significant potential in personalizing teacher-student interactions, offering valuable real-world applications for language learning in children's education. However, achieving high-quality, child-friendly speech generation remains challenging, particularly for low-resource languages across diverse languages and cultural contexts. In this paper, we propose MultiAiTutor, an educational multilingual generative AI tutor with child-friendly designs, leveraging LLM architecture for speech generation tailored for educational purposes. We propose to integrate age-appropriate multilingual speech generation using LLM architectures, facilitating young children's language learning through culturally relevant image-description tasks in three low-resource languages: Singaporean-accent Mandarin, Malay, and Tamil. Experimental results from both objective metrics and subjective evaluations demonstrate the superior performance of the proposed MultiAiTutor compared to baseline methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11468v1">TRACY: Benchmarking Execution Efficiency of LLM-Based Code Translation</a></div>
    <div class="paper-meta">
      📅 2025-08-15
    </div>
    <details class="paper-abstract">
      Automatic code translation is a fundamental task in modern software development. While the advent of Large Language Models (LLMs) has significantly improved the correctness of code translation, the critical dimension of execution efficiency remains overlooked. To address this gap, we introduce TRACY, the first comprehensive benchmark designed to evaluate the execution efficiency of LLM-translated code. TRACY is constructed through an LLM-driven two-stage pipeline: an initial stage generates a suite of stress tests to amplify performance differences, followed by an efficiency-oriented task pruning stage that isolates the efficiency-distinguishing tasks. The resulting benchmark comprises 1,011 code translation tasks across C++, Java, and Python, each accompanied by an average of 22.1 verified reference translations and 10 computationally demanding tests. Our extensive evaluation of 26 representative LLMs reveals that even top-tier LLMs struggle to consistently produce efficient code translations. For instance, Claude-4-think, the leading model for correctness, ranks eighth overall when time efficiency is taken into account, surpassed by several smaller open-source models. We further pinpoint that algorithmic flaws and improper resource handling are the most detrimental, causing a median time slowdown of 5.6$\times$ and memory increase of 12.0$\times$, respectively. Our work underscores the necessity of jointly optimizing for correctness and efficiency in future LLM-based code translation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11454v1">Reference Points in LLM Sentiment Analysis: The Role of Structured Context</a></div>
    <div class="paper-meta">
      📅 2025-08-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are now widely used across many fields, including marketing research. Sentiment analysis, in particular, helps firms understand consumer preferences. While most NLP studies classify sentiment from review text alone, marketing theories, such as prospect theory and expectation--disconfirmation theory, point out that customer evaluations are shaped not only by the actual experience but also by additional reference points. This study therefore investigates how the content and format of such supplementary information affect sentiment analysis using LLMs. We compare natural language (NL) and JSON-formatted prompts using a lightweight 3B parameter model suitable for practical marketing applications. Experiments on two Yelp categories (Restaurant and Nightlife) show that the JSON prompt with additional information outperforms all baselines without fine-tuning: Macro-F1 rises by 1.6% and 4% while RMSE falls by 16% and 9.1%, respectively, making it deployable in resource-constrained edge devices. Furthermore, a follow-up analysis confirms that performance gains stem from genuine contextual reasoning rather than label proxying. This work demonstrates that structured prompting can enable smaller models to achieve competitive performance, offering a practical alternative to large-scale model deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11425v1">Tapas are free! Training-Free Adaptation of Programmatic Agents via LLM-Guided Program Synthesis in Dynamic Environments</a></div>
    <div class="paper-meta">
      📅 2025-08-15
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Autonomous agents in safety-critical applications must continuously adapt to dynamic conditions without compromising performance and reliability. This work introduces TAPA (Training-free Adaptation of Programmatic Agents), a novel framework that positions large language models (LLMs) as intelligent moderators of the symbolic action space. Unlike prior programmatic agents that typically generate a monolithic policy program or rely on fixed symbolic action sets, TAPA synthesizes and adapts modular programs for individual high-level actions, referred to as logical primitives. By decoupling strategic intent from execution, TAPA enables meta-agents to operate over an abstract, interpretable action space while the LLM dynamically generates, composes, and refines symbolic programs tailored to each primitive. Extensive experiments across cybersecurity and swarm intelligence domains validate TAPA's effectiveness. In autonomous DDoS defense scenarios, TAPA achieves 77.7% network uptime while maintaining near-perfect detection accuracy in unknown dynamic environments. In swarm intelligence formation control under environmental and adversarial disturbances, TAPA consistently preserves consensus at runtime where baseline methods fail completely. This work promotes a paradigm shift for autonomous system design in evolving environments, from policy adaptation to dynamic action adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11416v1">AIM-Bench: Evaluating Decision-making Biases of Agentic LLM as Inventory Manager</a></div>
    <div class="paper-meta">
      📅 2025-08-15
    </div>
    <details class="paper-abstract">
      Recent advances in mathematical reasoning and the long-term planning capabilities of large language models (LLMs) have precipitated the development of agents, which are being increasingly leveraged in business operations processes. Decision models to optimize inventory levels are one of the core elements of operations management. However, the capabilities of the LLM agent in making inventory decisions in uncertain contexts, as well as the decision-making biases (e.g. framing effect, etc.) of the agent, remain largely unexplored. This prompts concerns regarding the capacity of LLM agents to effectively address real-world problems, as well as the potential implications of biases that may be present. To address this gap, we introduce AIM-Bench, a novel benchmark designed to assess the decision-making behaviour of LLM agents in uncertain supply chain management scenarios through a diverse series of inventory replenishment experiments. Our results reveal that different LLMs typically exhibit varying degrees of decision bias that are similar to those observed in human beings. In addition, we explored strategies to mitigate the pull-to-centre effect and the bullwhip effect, namely cognitive reflection and implementation of information sharing. These findings underscore the need for careful consideration of the potential biases in deploying LLMs in Inventory decision-making scenarios. We hope that these insights will pave the way for mitigating human decision bias and developing human-centred decision support systems for supply chains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11414v1">Survey-to-Behavior: Downstream Alignment of Human Values in LLMs via Survey Questions</a></div>
    <div class="paper-meta">
      📅 2025-08-15
      | 💬 7 pages 1 figure
    </div>
    <details class="paper-abstract">
      Large language models implicitly encode preferences over human values, yet steering them often requires large training data. In this work, we investigate a simple approach: Can we reliably modify a model's value system in downstream behavior by training it to answer value survey questions accordingly? We first construct value profiles of several open-source LLMs by asking them to rate a series of value-related descriptions spanning 20 distinct human values, which we use as a baseline for subsequent experiments. We then investigate whether the value system of a model can be governed by fine-tuning on the value surveys. We evaluate the effect of finetuning on the model's behavior in two ways; first, we assess how answers change on in-domain, held-out survey questions. Second, we evaluate whether the model's behavior changes in out-of-domain settings (situational scenarios). To this end, we construct a contextualized moral judgment dataset based on Reddit posts and evaluate changes in the model's behavior in text-based adventure games. We demonstrate that our simple approach can not only change the model's answers to in-domain survey questions, but also produces substantial shifts (value alignment) in implicit downstream task behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11401v1">FACET:Teacher-Centred LLM-Based Multi-Agent Systems-Towards Personalized Educational Worksheets</a></div>
    <div class="paper-meta">
      📅 2025-08-15
    </div>
    <details class="paper-abstract">
      The increasing heterogeneity of student populations poses significant challenges for teachers, particularly in mathematics education, where cognitive, motivational, and emotional differences strongly influence learning outcomes. While AI-driven personalization tools have emerged, most remain performance-focused, offering limited support for teachers and neglecting broader pedagogical needs. This paper presents the FACET framework, a teacher-facing, large language model (LLM)-based multi-agent system designed to generate individualized classroom materials that integrate both cognitive and motivational dimensions of learner profiles. The framework comprises three specialized agents: (1) learner agents that simulate diverse profiles incorporating topic proficiency and intrinsic motivation, (2) a teacher agent that adapts instructional content according to didactical principles, and (3) an evaluator agent that provides automated quality assurance. We tested the system using authentic grade 8 mathematics curriculum content and evaluated its feasibility through a) automated agent-based assessment of output quality and b) exploratory feedback from K-12 in-service teachers. Results from ten internal evaluations highlighted high stability and alignment between generated materials and learner profiles, and teacher feedback particularly highlighted structure and suitability of tasks. The findings demonstrate the potential of multi-agent LLM architectures to provide scalable, context-aware personalization in heterogeneous classroom settings, and outline directions for extending the framework to richer learner profiles and real-world classroom trials.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11398v1">Trustworthy AI Psychotherapy: Multi-Agent LLM Workflow for Counseling and Explainable Mental Disorder Diagnosis</a></div>
    <div class="paper-meta">
      📅 2025-08-15
      | 💬 Accepted by CIKM 2025 as a full paper
    </div>
    <details class="paper-abstract">
      LLM-based agents have emerged as transformative tools capable of executing complex tasks through iterative planning and action, achieving significant advancements in understanding and addressing user needs. Yet, their effectiveness remains limited in specialized domains such as mental health diagnosis, where they underperform compared to general applications. Current approaches to integrating diagnostic capabilities into LLMs rely on scarce, highly sensitive mental health datasets, which are challenging to acquire. These methods also fail to emulate clinicians' proactive inquiry skills, lack multi-turn conversational comprehension, and struggle to align outputs with expert clinical reasoning. To address these gaps, we propose DSM5AgentFlow, the first LLM-based agent workflow designed to autonomously generate DSM-5 Level-1 diagnostic questionnaires. By simulating therapist-client dialogues with specific client profiles, the framework delivers transparent, step-by-step disorder predictions, producing explainable and trustworthy results. This workflow serves as a complementary tool for mental health diagnosis, ensuring adherence to ethical and legal standards. Through comprehensive experiments, we evaluate leading LLMs across three critical dimensions: conversational realism, diagnostic accuracy, and explainability. Our datasets and implementations are fully open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08837v2">The Roots of International Perceptions: Simulating US Attitude Changes Towards China with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-15
      | 💬 Submitted to AAAI Social Impact 2026
    </div>
    <details class="paper-abstract">
      The rise of LLMs poses new possibilities in modeling opinion evolution, a long-standing task in simulation, by leveraging advanced reasoning abilities to recreate complex, large-scale human cognitive trends. While most prior works focus on opinion evolution surrounding specific isolated events or the views within a country, ours is the first to model the large-scale attitude evolution of a population representing an entire country towards another -- US citizens' perspectives towards China. To tackle the challenges of this broad scenario, we propose a framework that integrates media data collection, user profile creation, and cognitive architecture for opinion updates to successfully reproduce the real trend of US attitudes towards China over a 20-year period from 2005 to today. We also leverage LLMs' capabilities to introduce debiased media exposure, extracting neutral events from typically subjective news contents, to uncover the roots of polarized opinion formation, as well as a devils advocate agent to help explain the rare reversal from negative to positive attitudes towards China, corresponding with changes in the way Americans obtain information about the country. The simulation results, beyond validating our framework architecture, also reveal the impact of biased framing and selection bias in shaping attitudes. Overall, our work contributes to a new paradigm for LLM-based modeling of cognitive behaviors in a large-scale, long-term, cross-border social context, providing insights into the formation of international biases and offering valuable implications for media consumers to better understand the factors shaping their perspectives, and ultimately contributing to the larger social need for bias reduction and cross-cultural tolerance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11383v1">When Punctuation Matters: A Large-Scale Comparison of Prompt Robustness Methods for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are highly sensitive to subtle, non-semantic variations in prompt phrasing and formatting. In this work, we present the first systematic evaluation of 5 methods for improving prompt robustness within a unified experimental framework. We benchmark these techniques on 8 models from Llama, Qwen and Gemma families across 52 tasks from Natural Instructions dataset. Our evaluation covers robustness methods from both fine-tuned and in-context learning paradigms, and tests their generalization against multiple types of distribution shifts. Finally, we extend our analysis to GPT-4.1 and DeepSeek V3 to assess frontier models' current robustness to format perturbations. Our findings offer actionable insights into the relative effectiveness of these robustness methods, enabling practitioners to make informed decisions when aiming for stable and reliable LLM performance in real-world applications. Code: https://github.com/AIRI-Institute/when-punctuation-matters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16502v4">RULEBREAKERS: Challenging LLMs at the Crossroads between Formal Logic and Human-like Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-08-15
      | 💬 Accepted by ICML 2025
    </div>
    <details class="paper-abstract">
      Formal logic enables computers to reason in natural language by representing sentences in symbolic forms and applying rules to derive conclusions. However, in what our study characterizes as "rulebreaker" scenarios, this method can lead to conclusions that are typically not inferred or accepted by humans given their common sense and factual knowledge. Inspired by works in cognitive science, we create RULEBREAKERS, the first dataset for rigorously evaluating the ability of large language models (LLMs) to recognize and respond to rulebreakers (versus non-rulebreakers) in a human-like manner. Evaluating seven LLMs, we find that most models, including GPT-4o, achieve mediocre accuracy on RULEBREAKERS and exhibit some tendency to over-rigidly apply logical rules unlike what is expected from typical human reasoners. Further analysis suggests that this apparent failure is potentially associated with the models' poor utilization of their world knowledge and their attention distribution patterns. Whilst revealing a limitation of current LLMs, our study also provides a timely counterbalance to a growing body of recent works that propose methods relying on formal logic to improve LLMs' general reasoning capabilities, highlighting their risk of further increasing divergence between LLMs and human-like reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11343v1">SpecDetect: Simple, Fast, and Training-Free Detection of LLM-Generated Text via Spectral Analysis</a></div>
    <div class="paper-meta">
      📅 2025-08-15
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      The proliferation of high-quality text from Large Language Models (LLMs) demands reliable and efficient detection methods. While existing training-free approaches show promise, they often rely on surface-level statistics and overlook fundamental signal properties of the text generation process. In this work, we reframe detection as a signal processing problem, introducing a novel paradigm that analyzes the sequence of token log-probabilities in the frequency domain. By systematically analyzing the signal's spectral properties using the global Discrete Fourier Transform (DFT) and the local Short-Time Fourier Transform (STFT), we find that human-written text consistently exhibits significantly higher spectral energy. This higher energy reflects the larger-amplitude fluctuations inherent in human writing compared to the suppressed dynamics of LLM-generated text. Based on this key insight, we construct SpecDetect, a detector built on a single, robust feature from the global DFT: DFT total energy. We also propose an enhanced version, SpecDetect++, which incorporates a sampling discrepancy mechanism to further boost robustness. Extensive experiments demonstrate that our approach outperforms the state-of-the-art model while running in nearly half the time. Our work introduces a new, efficient, and interpretable pathway for LLM-generated text detection, showing that classical signal processing techniques offer a surprisingly powerful solution to this modern challenge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11318v1">LLM Compression: How Far Can We Go in Balancing Size and Performance?</a></div>
    <div class="paper-meta">
      📅 2025-08-15
      | 💬 This paper has been accepted for presentation at the RANLP 2025 conference
    </div>
    <details class="paper-abstract">
      Quantization is an essential and popular technique for improving the accessibility of large language models (LLMs) by reducing memory usage and computational costs while maintaining performance. In this study, we apply 4-bit Group Scaling Quantization (GSQ) and Generative Pretrained Transformer Quantization (GPTQ) to LLaMA 1B, Qwen 0.5B, and PHI 1.5B, evaluating their impact across multiple NLP tasks. We benchmark these models on MS MARCO (Information Retrieval), BoolQ (Boolean Question Answering), and GSM8K (Mathematical Reasoning) datasets, assessing both accuracy and efficiency across various tasks. The study measures the trade-offs between model compression and task performance, analyzing key evaluation metrics, namely accuracy, inference latency, and throughput (total output tokens generated per second), providing insights into the suitability of low-bit quantization for real-world deployment. Using the results, users can then make suitable decisions based on the specifications that need to be met. We discuss the pros and cons of GSQ and GPTQ techniques on models of different sizes, which also serve as a benchmark for future experiments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11717v1">WIP: Leveraging LLMs for Enforcing Design Principles in Student Code: Analysis of Prompting Strategies and RAG</a></div>
    <div class="paper-meta">
      📅 2025-08-14
      | 💬 Accepted for presentation at the Frontiers in Education Conference, Nashville, Tennessee, USA, 2-5 November 2025
    </div>
    <details class="paper-abstract">
      This work-in-progress research-to-practice paper explores the integration of Large Language Models (LLMs) into the code-review process for open-source software projects developed in computer science and software engineering courses. The focus is on developing an automated feedback tool that evaluates student code for adherence to key object-oriented design principles, addressing the need for more effective and scalable methods to teach software design best practices. The innovative practice involves leveraging LLMs and Retrieval-Augmented Generation (RAG) to create an automated feedback system that assesses student code for principles like SOLID, DRY, and design patterns. It analyzes the effectiveness of various prompting strategies and the RAG integration. Preliminary findings show promising improvements in code quality. Future work will aim to improve model accuracy and expand support for additional design principles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11715v1">Benchmark Dataset Generation and Evaluation for Excel Formula Repair with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-14
      | 💬 Accepted at the KDD workshop on Evaluation and Trustworthiness of Agentic and Generative AI Models
    </div>
    <details class="paper-abstract">
      Excel is a pervasive yet often complex tool, particularly for novice users, where runtime errors arising from logical mistakes or misinterpretations of functions pose a significant challenge. While large language models (LLMs) offer promising assistance by explaining formula errors, the automated correction of these semantic runtime errors remains an open problem. A primary challenge to advancing models for such scenarios is the severe lack of high-quality, comprehensive datasets for training and rigorous evaluation. This paper addresses this gap by introducing a novel approach for constructing a benchmark dataset specifically designed for Excel formula repair. We propose a data generation pipeline, which leverages a small set of curated seed samples from online forums to synthetically expand the dataset. Our pipeline integrates few-shot prompting with LLMs and employs a robust \textit{LLM-as-a-Judge} validation framework, combined with execution-based checks to ensure the correctness and semantic fidelity of the generated data. This process produced a benchmark dataset of 618 high-quality samples, covering common runtime errors. Furthermore, we propose a context-aware baseline technique for Excel formula repair that utilizes LLMs to leverage both the faulty formula, and relevant spreadsheet context. We evaluate the performance of various LLMs (GPT-4o, GPT-4.1, Phi-3, Mistral) on our newly generated benchmark using execution-based metrics. Our analysis demonstrates the dataset's quality through manual annotation and provides insights into error and function distributions. The proposed generation methodology is highly scalable and can be readily adapted to create evaluation benchmarks for similar code repair tasks in other low-resource programming languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13187v1">Combating Homelessness Stigma with LLMs: A New Multi-Modal Dataset for Bias Detection</a></div>
    <div class="paper-meta">
      📅 2025-08-14
    </div>
    <details class="paper-abstract">
      Homelessness is a persistent social challenge, impacting millions worldwide. Over 770,000 people experienced homelessness in the U.S. in 2024. Social stigmatization is a significant barrier to alleviation, shifting public perception, and influencing policymaking. Given that online and city council discourse reflect and influence part of public opinion, it provides valuable insights to identify and track social biases. This research contributes to alleviating homelessness by acting on public opinion. It introduces novel methods, building on natural language processing (NLP) and large language models (LLMs), to identify and measure PEH social bias expressed in digital spaces. We present a new, manually-annotated multi-modal dataset compiled from Reddit, X (formerly Twitter), news articles, and city council meeting minutes across 10 U.S. cities. This unique dataset provides evidence of the typologies of homelessness bias described in the literature. In order to scale up and automate the detection of homelessness bias online, we evaluate LLMs as classifiers. We applied both zero-shot and few-shot classification techniques to this data. We utilized local LLMs (Llama 3.2 3B Instruct, Qwen 2.5 7B Instruct, and Phi4 Instruct Mini) as well as closed-source API models (GPT-4.1, Gemini 2.5 Pro, and Grok-4). Our findings reveal that although there are significant inconsistencies in local LLM zero-shot classification, the in-context learning classification scores of local LLMs approach the classification scores of closed-source LLMs. Furthermore, LLMs outperform BERT when averaging across all categories. This work aims to raise awareness about the pervasive bias against PEH, develop new indicators to inform policy, and ultimately enhance the fairness and ethical application of Generative AI technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10535v2">CodeJudgeBench: Benchmarking LLM-as-a-Judge for Coding Tasks</a></div>
    <div class="paper-meta">
      📅 2025-08-14
      | 💬 Dataset is available at https://huggingface.co/datasets/mattymchen/codejudgebench
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have significantly advanced the state-of-the-art in various coding tasks. Beyond directly answering user queries, LLMs can also serve as judges, assessing and comparing the quality of responses generated by other models. Such an evaluation capability is crucial both for benchmarking different LLMs and for improving response quality through response ranking. However, despite the growing adoption of the LLM-as-a-Judge paradigm, its effectiveness in coding scenarios remains underexplored due to the absence of dedicated benchmarks. To address this gap, we introduce CodeJudgeBench, a benchmark explicitly designed to evaluate the performance of LLM-as-a-Judge models across three critical coding tasks: code generation, code repair, and unit test generation. Through comprehensive benchmarking of 26 LLM-as-a-Judge models, we find that recent thinking models significantly outperform non-thinking models on our carefully designed code judging tasks. Notably, even relatively small thinking models, such as Qwen3-8B, can outperform specially trained LLM-as-a-Judge models up to 70B in size. Nevertheless, all models still exhibit significant randomness in their judgment of coding tasks. For pairwise judging tasks, simply changing the order in which responses are presented can substantially impact accuracy. In addition, when judging code and unit tests written by different LLMs, LLM-as-a-Judge models also show variance in performance. This sensitivity raises concerns about the reliability and consistency of LLM-as-a-Judge in coding scenarios. Lastly, we study optimal prompting strategies for LLM-as-a-Judge. We find that using pair-wise comparison outperforms scalar point-wise judging. Furthermore, retaining comments and reasoning in the full, unprocessed LLM response leads to improved judge performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08855v2">BiasGym: Fantastic LLM Biases and How to Find (and Remove) Them</a></div>
    <div class="paper-meta">
      📅 2025-08-14
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Understanding biases and stereotypes encoded in the weights of Large Language Models (LLMs) is crucial for developing effective mitigation strategies. Biased behaviour is often subtle and non-trivial to isolate, even when deliberately elicited, making systematic analysis and debiasing particularly challenging. To address this, we introduce BiasGym, a simple, cost-effective, and generalizable framework for reliably injecting, analyzing, and mitigating conceptual associations within LLMs. BiasGym consists of two components: BiasInject, which injects specific biases into the model via token-based fine-tuning while keeping the model frozen, and BiasScope, which leverages these injected signals to identify and steer the components responsible for biased behavior. Our method enables consistent bias elicitation for mechanistic analysis, supports targeted debiasing without degrading performance on downstream tasks, and generalizes to biases unseen during token-based fine-tuning. We demonstrate the effectiveness of BiasGym in reducing real-world stereotypes (e.g., people from Italy being `reckless drivers') and in probing fictional associations (e.g., people from a fictional country having `blue skin'), showing its utility for both safety interventions and interpretability research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.10880v1">Searching for Privacy Risks in LLM Agents via Simulation</a></div>
    <div class="paper-meta">
      📅 2025-08-14
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      The widespread deployment of LLM-based agents is likely to introduce a critical privacy threat: malicious agents that proactively engage others in multi-turn interactions to extract sensitive information. These dynamic dialogues enable adaptive attack strategies that can cause severe privacy violations, yet their evolving nature makes it difficult to anticipate and discover sophisticated vulnerabilities manually. To tackle this problem, we present a search-based framework that alternates between improving attacker and defender instructions by simulating privacy-critical agent interactions. Each simulation involves three roles: data subject, data sender, and data recipient. While the data subject's behavior is fixed, the attacker (data recipient) attempts to extract sensitive information from the defender (data sender) through persistent and interactive exchanges. To explore this interaction space efficiently, our search algorithm employs LLMs as optimizers, using parallel search with multiple threads and cross-thread propagation to analyze simulation trajectories and iteratively propose new instructions. Through this process, we find that attack strategies escalate from simple direct requests to sophisticated multi-turn tactics such as impersonation and consent forgery, while defenses advance from rule-based constraints to identity-verification state machines. The discovered attacks and defenses transfer across diverse scenarios and backbone models, demonstrating strong practical utility for building privacy-aware agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.10848v1">Psyche-R1: Towards Reliable Psychological LLMs through Unified Empathy, Expertise, and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-08-14
    </div>
    <details class="paper-abstract">
      Amidst a shortage of qualified mental health professionals, the integration of large language models (LLMs) into psychological applications offers a promising way to alleviate the growing burden of mental health disorders. Recent reasoning-augmented LLMs have achieved remarkable performance in mathematics and programming, while research in the psychological domain has predominantly emphasized emotional support and empathetic dialogue, with limited attention to reasoning mechanisms that are beneficial to generating reliable responses. Therefore, in this paper, we propose Psyche-R1, the first Chinese psychological LLM that jointly integrates empathy, psychological expertise, and reasoning, built upon a novel data curation pipeline. Specifically, we design a comprehensive data synthesis pipeline that produces over 75k high-quality psychological questions paired with detailed rationales, generated through chain-of-thought (CoT) reasoning and iterative prompt-rationale optimization, along with 73k empathetic dialogues. Subsequently, we employ a hybrid training strategy wherein challenging samples are identified through a multi-LLM cross-selection strategy for group relative policy optimization (GRPO) to improve reasoning ability, while the remaining data is used for supervised fine-tuning (SFT) to enhance empathetic response generation and psychological domain knowledge. Extensive experiment results demonstrate the effectiveness of the Psyche-R1 across several psychological benchmarks, where our 7B Psyche-R1 achieves comparable results to 671B DeepSeek-R1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05571v2">iFairy: the First 2-bit Complex LLM with All Parameters in $\{\pm1, \pm i\}$</a></div>
    <div class="paper-meta">
      📅 2025-08-14
      | 💬 15 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Quantization-Aware Training (QAT) integrates quantization into the training loop, enabling LLMs to learn robust low-bit representations, and is widely recognized as one of the most promising research directions. All current QAT research focuses on minimizing quantization error on full-precision models, where the full-precision accuracy acts as an upper bound (accuracy ceiling). No existing method has even attempted to surpass this ceiling. To break this ceiling, we propose a new paradigm: raising the ceiling (full-precision model), and then still quantizing it efficiently into 2 bits. We propose Fairy$\pm i$, the first 2-bit quantization framework for complex-valued LLMs. Specifically, our method leverages the representational advantages of the complex domain to boost full-precision accuracy. We map weights to the fourth roots of unity $\{\pm1, \pm i\}$, forming a perfectly symmetric and information-theoretically optimal 2-bit representation. Importantly, each quantized weight has either a zero real or imaginary part, enabling multiplication-free inference using only additions and element swaps. Experimental results show that Fairy$\pm i$ outperforms the ceiling of existing 2-bit quantization approaches in terms of both PPL and downstream tasks, while maintaining strict storage and compute efficiency. This work opens a new direction for building highly accurate and practical LLMs under extremely low-bit constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.10795v1">Beyond "Not Novel Enough": Enriching Scholarly Critique with LLM-Assisted Feedback</a></div>
    <div class="paper-meta">
      📅 2025-08-14
    </div>
    <details class="paper-abstract">
      Novelty assessment is a central yet understudied aspect of peer review, particularly in high volume fields like NLP where reviewer capacity is increasingly strained. We present a structured approach for automated novelty evaluation that models expert reviewer behavior through three stages: content extraction from submissions, retrieval and synthesis of related work, and structured comparison for evidence based assessment. Our method is informed by a large scale analysis of human written novelty reviews and captures key patterns such as independent claim verification and contextual reasoning. Evaluated on 182 ICLR 2025 submissions with human annotated reviewer novelty assessments, the approach achieves 86.5% alignment with human reasoning and 75.3% agreement on novelty conclusions - substantially outperforming existing LLM based baselines. The method produces detailed, literature aware analyses and improves consistency over ad hoc reviewer judgments. These results highlight the potential for structured LLM assisted approaches to support more rigorous and transparent peer review without displacing human expertise. Data and code are made available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13109v2">FreeKV: Boosting KV Cache Retrieval for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-08-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been widely deployed with rapidly expanding context windows to support increasingly demanding applications. However, long contexts pose significant deployment challenges, primarily due to the KV cache whose size grows proportionally with context length. While KV cache compression methods are proposed to address this issue, KV dropping methods incur considerable accuracy loss, and KV retrieval methods suffer from significant efficiency bottlenecks. We propose FreeKV, an algorithm-system co-optimization framework to enhance KV retrieval efficiency while preserving accuracy. On the algorithm side, FreeKV introduces speculative retrieval to shift the KV selection and recall processes out of the critical path, combined with fine-grained correction to ensure accuracy. On the system side, FreeKV employs hybrid KV layouts across CPU and GPU memory to eliminate fragmented data transfers, and leverages double-buffered streamed recall to further improve efficiency. Experiments demonstrate that FreeKV achieves near-lossless accuracy across various scenarios and models, delivering up to 13$\times$ speedup compared to SOTA KV retrieval methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.10777v1">The Knowledge-Reasoning Dissociation: Fundamental Limitations of LLMs in Clinical Natural Language Inference</a></div>
    <div class="paper-meta">
      📅 2025-08-14
      | 💬 19 pages
    </div>
    <details class="paper-abstract">
      Large language models are often assumed to acquire increasingly structured, generalizable internal representations simply by scaling data and parameters. We interrogate this assumption by introducing a Clinical Trial Natural Language Inference benchmark comprising four reasoning families, Causal Attribution, Compositional Grounding, Epistemic Verification, and Risk State Abstraction. Each item is paired with a targeted Ground Knowledge and Meta-Level Reasoning Verification (GKMRV) probe, allowing us to dissociate failures of factual access from failures of inference. We evaluate six contemporary LLMs under both direct and chain of thought prompting. Models achieve near-ceiling GKMRV accuracy (mean accuracy 0.918) yet perform poorly on the main reasoning tasks (mean accuracy 0.25). Despite low accuracy, output inferences are highly consistent across samples (mean 0.87), indicating a systematic application of underlying heuristics and shortcuts. These results reveal fundamental structural and representational limitations: current LLMs often possess the relevant clinical knowledge but lack the structured, composable internal representations needed to deploy it reliably (e.g., integrating constraints, weighing evidence, or simulating counterfactuals). Decoupling knowledge from reasoning with GKMRV makes this dissociation explicit and measurable, providing an effective framework for probing the reliability of LLMs in high-stakes domains.
    </details>
</div>
