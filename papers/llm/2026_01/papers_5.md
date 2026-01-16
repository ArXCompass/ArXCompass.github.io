# llm - 2026_01

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- Part 5
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03775v1">Do LLM Self-Explanations Help Users Predict Model Behavior? Evaluating Counterfactual Simulatability with Pragmatic Perturbations</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can produce verbalized self-explanations, yet prior studies suggest that such rationales may not reliably reflect the model's true decision process. We ask whether these explanations nevertheless help users predict model behavior, operationalized as counterfactual simulatability. Using StrategyQA, we evaluate how well humans and LLM judges can predict a model's answers to counterfactual follow-up questions, with and without access to the model's chain-of-thought or post-hoc explanations. We compare LLM-generated counterfactuals with pragmatics-based perturbations as alternative ways to construct test cases for assessing the potential usefulness of explanations. Our results show that self-explanations consistently improve simulation accuracy for both LLM judges and humans, but the degree and stability of gains depend strongly on the perturbation strategy and judge strength. We also conduct a qualitative analysis of free-text justifications written by human users when predicting the model's behavior, which provides evidence that access to explanations helps humans form more accurate predictions on the perturbed questions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03746v1">Whose Facts Win? LLM Source Preferences under Knowledge Conflicts</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 Data and code: https://github.com/JaSchuste/llm-source-preference
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are more frequently used in retrieval-augmented generation pipelines, it is increasingly relevant to study their behavior under knowledge conflicts. Thus far, the role of the source of the retrieved information has gone unexamined. We address this gap with a novel framework to investigate how source preferences affect LLM resolution of inter-context knowledge conflicts in English, motivated by interdisciplinary research on credibility. With a comprehensive, tightly-controlled evaluation of 13 open-weight LLMs, we find that LLMs prefer institutionally-corroborated information (e.g., government or newspaper sources) over information from people and social media. However, these source preferences can be reversed by simply repeating information from less credible sources. To mitigate repetition effects and maintain consistent preferences, we propose a novel method that reduces repetition bias by up to 99.8%, while also maintaining at least 88.8% of original preferences. We release all data and code to encourage future work on credibility and source preferences in knowledge-intensive NLP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.18795v3">ProCLIP: Progressive Vision-Language Alignment via LLM-based Embedder</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 17 pages, 5 fiugres
    </div>
    <details class="paper-abstract">
      The original CLIP text encoder is limited by a maximum input length of 77 tokens, which hampers its ability to effectively process long texts and perform fine-grained semantic understanding. In addition, the CLIP text encoder lacks support for multilingual inputs. All these limitations significantly restrict its applicability across a broader range of tasks. Recent studies have attempted to replace the CLIP text encoder with an LLM-based embedder to enhance its ability in processing long texts, multilingual understanding, and fine-grained semantic comprehension. However, because the representation spaces of LLMs and the vision-language space of CLIP are pretrained independently without alignment priors, direct alignment using contrastive learning can disrupt the intrinsic vision-language alignment in the CLIP image encoder, leading to an underutilization of the knowledge acquired during pre-training. To address this challenge, we propose ProCLIP, a curriculum learning-based progressive vision-language alignment framework to effectively align the CLIP image encoder with an LLM-based embedder. Specifically, ProCLIP first distills knowledge from CLIP's text encoder into the LLM-based embedder to leverage CLIP's rich pretrained knowledge while establishing initial alignment between the LLM embedder and CLIP image encoder. Subsequently, ProCLIP further aligns the CLIP image encoder with the LLM-based embedder through image-text contrastive tuning, employing self-distillation regularization to avoid overfitting. To achieve a more effective alignment, instance semantic alignment loss and embedding structure alignment loss are employed during representation inheritance and contrastive tuning. The Code is available at https://github.com/VisionXLab/ProCLIP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.08473v3">AsFT: Anchoring Safety During LLM Fine-Tuning Within Narrow Safety Basin</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) improves performance but introduces critical safety vulnerabilities: even minimal harmful data can severely compromise safety measures. We observe that perturbations orthogonal to the alignment direction - defined by weight differences between aligned (safe) and unaligned models - rapidly compromise model safety. In contrast, updates along the alignment direction largely preserve it, revealing the parameter space as a "narrow safety basin". To address this, we propose AsFT (Anchoring Safety in Fine-Tuning) to maintain safety by explicitly constraining update directions during fine-tuning. By penalizing updates orthogonal to the alignment direction, AsFT effectively constrains the model within the "narrow safety basin," thus preserving its inherent safety. Extensive experiments on multiple datasets and models show that AsFT reduces harmful behaviors by up to 7.60%, improves task performance by 3.44%, and consistently outperforms existing methods across multiple tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.23163v3">Beyond Direct Generation: A Decomposed Approach to Well-Crafted Screenwriting with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      The screenplay serves as the foundation for television production, defining narrative structure, character development, and dialogue. While Large Language Models (LLMs) show great potential in creative writing, direct end-to-end generation approaches often fail to produce well-crafted screenplays. We argue this failure stems from forcing a single model to simultaneously master two disparate capabilities: creative narrative construction and rigid format adherence. The resulting outputs may mimic superficial style but lack the deep structural integrity and storytelling substance required for professional use. To enable LLMs to generate high-quality screenplays, we introduce Dual-Stage Refinement (DSR), a decomposed framework that decouples creative narrative generation from format conversion. The first stage transforms a brief outline into rich, novel-style prose. The second stage refines this narrative into a professionally formatted screenplay. This separation enables the model to specialize in one distinct capability at each stage. A key challenge in implementing DSR is the scarcity of paired outline-to-novel training data. We address this through hybrid data synthesis: reverse synthesis deconstructs existing screenplays into structured inputs, while forward synthesis leverages these inputs to generate high-quality narrative texts as training targets. Blind evaluations by professional screenwriters show that DSR achieves a 75% win rate against strong baselines like Gemini-2.5-Pro and reaches 82.7% of human-level performance. Our work demonstrates that decomposed generation architecture with tailored data synthesis effectively specializes LLMs in complex creative domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.14803v4">OnlineMate: An LLM-Based Multi-Agent Companion System for Cognitive Support in Online Learning</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 work in progress
    </div>
    <details class="paper-abstract">
      In online learning environments, students often lack personalized peer interactions, which are crucial for cognitive development and learning engagement. Although previous studies have employed large language models (LLMs) to simulate interactive learning environments, these interactions are limited to conversational exchanges, failing to adapt to learners' individualized cognitive and psychological states. As a result, students' engagement is low and they struggle to gain inspiration. To address this challenge, we propose OnlineMate, a multi-agent learning companion system driven by LLMs integrated with Theory of Mind (ToM). OnlineMate simulates peer-like roles, infers learners' psychological states such as misunderstandings and confusion during collaborative discussions, and dynamically adjusts interaction strategies to support higher-order thinking. Comprehensive evaluations, including simulation-based experiments, human assessments, and real classroom trials, demonstrate that OnlineMate significantly promotes deep learning and cognitive engagement by elevating students' average cognitive level while substantially improving emotional engagement scores.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.01211v2">Web Fraud Attacks Against LLM-Driven Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      With the proliferation of LLM-driven multi-agent systems (MAS), the security of Web links has become a critical concern. Once MAS is induced to trust a malicious link, attackers can use it as a springboard to expand the attack surface. In this paper, we propose Web Fraud Attacks, a novel type of attack manipulating unique structures of web links to deceive MAS. We design 12 representative attack variants that encompass various methods, such as homoglyph deception, sub-directory nesting, and parameter obfuscation. Through extensive experiments on these attack vectors, we demonstrate that Web fraud attacks not only exhibit significant destructive potential across different MAS architectures but also possess a distinct advantage in evasion: they circumvent the need for complex input design, lowering the threshold for attacks significantly. These results underscore the importance of addressing Web fraud attacks, providing new insights into MAS safety. Our code is available at https://github.com/JiangYingEr/Web-Fraud-Attack-in-MAS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03687v1">Personalized Medication Planning via Direct Domain Modeling and LLM-Generated Heuristics</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Personalized medication planning involves selecting medications and determining a dosing schedule to achieve medical goals specific to each individual patient. Previous work successfully demonstrated that automated planners, using general domain-independent heuristics, are able to generate personalized treatments, when the domain and problems are modeled using a general domain description language (\pddlp). Unfortunately, this process was limited in practice to consider no more than seven medications. In clinical terms, this is a non-starter. In this paper, we explore the use of automatically-generated domain- and problem-specific heuristics to be used with general search, as a method of scaling up medication planning to levels allowing closer work with clinicians. Specifically, we specify the domain programmatically (specifying an initial state and a successor generation procedure), and use an LLM to generate a problem specific heuristic that can be used by a fixed search algorithm (GBFS). The results indicate dramatic improvements in coverage and planning time, scaling up the number of medications to at least 28, and bringing medication planning one step closer to practical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03682v1">From Implicit to Explicit: Token-Efficient Logical Supervision for Mathematical Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Recent studies reveal that large language models (LLMs) exhibit limited logical reasoning abilities in mathematical problem-solving, instead often relying on pattern-matching and memorization. We systematically analyze this limitation, focusing on logical relationship understanding, which is a core capability underlying genuine logical reasoning, and reveal that errors related to this capability account for over 90\% of incorrect predictions, with Chain-of-Thought Supervised Fine-Tuning (CoT-SFT) failing to substantially reduce these errors. To address this bottleneck, we propose First-Step Logical Reasoning (FSLR), a lightweight training framework targeting logical relationship understanding. Our key insight is that the first planning step-identifying which variables to use and which operation to apply-encourages the model to derive logical relationships directly from the problem statement. By training models on this isolated step, FSLR provides explicit supervision for logical relationship understanding, unlike CoT-SFT which implicitly embeds such relationships within complete solution trajectories. Extensive experiments across multiple models and datasets demonstrate that FSLR consistently outperforms CoT-SFT under both in-distribution and out-of-distribution settings, with average improvements of 3.2\% and 4.6\%, respectively. Moreover, FSLR achieves 4-6x faster training and reduces training token consumption by over 80\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.18851v3">Marking Code Without Breaking It: Code Watermarking for Detecting LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 Findings of EACL 2026
    </div>
    <details class="paper-abstract">
      Identifying LLM-generated code through watermarking poses a challenge in preserving functional correctness. Previous methods rely on the assumption that watermarking high-entropy tokens effectively maintains output quality. Our analysis reveals a fundamental limitation of this assumption: syntax-critical tokens such as keywords often exhibit the highest entropy, making existing approaches vulnerable to logic corruption. We present STONE, a syntax-aware watermarking method that embeds watermarks only in non-syntactic tokens and preserves code integrity. For its rigorous assessment, we also introduce STEM, a comprehensive framework that balances three critical dimensions: correctness, detectability, and imperceptibility. Across Python, C++, and Java, STONE preserves correctness, sustains strong detectability, and achieves balanced performance with minimal overhead. Our implementation is available at https://anonymous.4open.science/r/STONE-watermarking-AB4B/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03676v1">Towards Compositional Generalization of LLMs via Skill Taxonomy Guided Data Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 The code and data for our methods and experiments are available at https://github.com/weiyifan1023/STEPS
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) and agent-based systems often struggle with compositional generalization due to a data bottleneck in which complex skill combinations follow a long-tailed, power-law distribution, limiting both instruction-following performance and generalization in agent-centric tasks. To address this challenge, we propose STEPS, a Skill Taxonomy guided Entropy-based Post-training data Synthesis framework for generating compositionally challenging data. STEPS explicitly targets compositional generalization by uncovering latent relationships among skills and organizing them into an interpretable, hierarchical skill taxonomy using structural information theory. Building on this taxonomy, we formulate data synthesis as a constrained information maximization problem, selecting skill combinations that maximize marginal structural information within the hierarchy while preserving semantic coherence. Experiments on challenging instruction-following benchmarks show that STEPS outperforms existing data synthesis baselines, while also yielding improved compositional generalization in downstream agent-based evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.02930v5">Video LLMs for Temporal Reasoning in Long Videos</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      We introduce TemporalVLM, a video large language model (video LLM) for temporal reasoning and fine-grained understanding in long videos. Our approach includes a visual encoder for mapping a long-term video into features which are time-aware and contain both local and global cues. It first divides an input video into short-term clips, which are jointly encoded with timestamps and fused across overlapping temporal windows into time-sensitive local features. Next, the local features are passed through a bidirectional long short-term memory (BiLSTM) module for global feature aggregation. Moreover, to facilitate the evaluation of TemporalVLM, we present a large-scale long video dataset of industry assembly processes, namely IndustryASM, consisting of videos recorded on factory floors with actions and timestamps annotated by industrial engineers for time and motion studies and temporal action segmentation evaluation. Finally, extensive experiments show that TemporalVLM outperforms previous methods across temporal reasoning and fine-grained understanding tasks, i.e., dense video captioning, temporal video grounding, video highlight detection, and temporal action segmentation. To our best knowledge, our work is the first to incorporate LSTMs into video LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.08726v3">Improved LLM Agents for Financial Document Question Answering</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 13 pages, 6 figures. More analysis is added to Appendix C
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown impressive capabilities on numerous natural language processing tasks. However, LLMs still struggle with numerical question answering for financial documents that include tabular and textual data. Recent works have showed the effectiveness of critic agents (i.e., self-correction) for this task given oracle labels. Building upon this framework, this paper examines the effectiveness of the traditional critic agent when oracle labels are not available, and show, through experiments, that this critic agent's performance deteriorates in this scenario. With this in mind, we present an improved critic agent, along with the calculator agent which outperforms the previous state-of-the-art approach (program-of-thought) and is safer. Furthermore, we investigate how our agents interact with each other, and how this interaction affects their performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.02110v2">Attractive Metadata Attack: Inducing LLM Agents to Invoke Malicious Tools</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 Accepted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have demonstrated remarkable capabilities in complex reasoning and decision-making by leveraging external tools. However, this tool-centric paradigm introduces a previously underexplored attack surface, where adversaries can manipulate tool metadata -- such as names, descriptions, and parameter schemas -- to influence agent behavior. We identify this as a new and stealthy threat surface that allows malicious tools to be preferentially selected by LLM agents, without requiring prompt injection or access to model internals. To demonstrate and exploit this vulnerability, we propose the Attractive Metadata Attack (AMA), a black-box in-context learning framework that generates highly attractive but syntactically and semantically valid tool metadata through iterative optimization. The proposed attack integrates seamlessly into standard tool ecosystems and requires no modification to the agent's execution framework. Extensive experiments across ten realistic, simulated tool-use scenarios and a range of popular LLM agents demonstrate consistently high attack success rates (81\%-95\%) and significant privacy leakage, with negligible impact on primary task execution. Moreover, the attack remains effective even against prompt-level defenses, auditor-based detection, and structured tool-selection protocols such as the Model Context Protocol, revealing systemic vulnerabilities in current agent architectures. These findings reveal that metadata manipulation constitutes a potent and stealthy attack surface. Notably, AMA is orthogonal to injection attacks and can be combined with them to achieve stronger attack efficacy, highlighting the need for execution-level defenses beyond prompt-level and auditor-based mechanisms. Code is available at https://github.com/SEAIC-M/AMA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.05738v3">LLM-assisted Mutation for Whitebox API Testing</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Cloud applications heavily rely on APIs to communicate with each other and exchange data. To ensure the reliability of cloud applications, cloud providers widely adopt API testing techniques. Unfortunately, existing API testing approaches are insufficient to reach strict conditions, a problem known as fitness plateaus, due to the lack of gradient provided by coverage metrics. To address this issue, we propose MioHint, a novel white-box API testing approach that leverages the code comprehension capabilities of Large Language Model (LLM) to boost API testing. The key challenge of LLM-based API testing lies in system-level testing, which emphasizes the dependencies between requests and targets across functions and files, thereby making the entire codebase the object of analysis. However, feeding the entire codebase to an LLM is impractical due to its limited context length and short memory. MioHint addresses this challenge by synergizing static analysis with LLMs. We retrieve relevant code with data-dependency analysis at the statement level, including def-use analysis for variables used in the target and function expansion for subfunctions called by the target. To evaluate the effectiveness of our method, we conducted experiments across 16 real-world REST API services. The findings reveal that MioHint achieves an average increase of 4.95% absolute in line coverage compared to the baseline, EvoMaster, alongside a remarkable factor of 67x improvement in mutation accuracy. Furthermore, our method successfully covers over 57% of hard-to-cover targets while in baseline the coverage is less than 10%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.23314v2">SPIO: Ensemble and Selective Strategies via LLM-Based Multi-Agent Planning in Automated Data Science</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have enabled dynamic reasoning in automated data analytics, yet recent multi-agent systems remain limited by rigid, single-path workflows that restrict strategic exploration and often lead to suboptimal outcomes. To overcome these limitations, we propose SPIO (Sequential Plan Integration and Optimization), a framework that replaces rigid workflows with adaptive, multi-path planning across four core modules: data preprocessing, feature engineering, model selection, and hyperparameter tuning. In each module, specialized agents generate diverse candidate strategies, which are cascaded and refined by an optimization agent. SPIO offers two operating modes: SPIO-S for selecting a single optimal pipeline, and SPIO-E for ensembling top-k pipelines to maximize robustness. Extensive evaluations on Kaggle and OpenML benchmarks show that SPIO consistently outperforms state-of-the-art baselines, achieving an average performance gain of 5.6%. By explicitly exploring and integrating multiple solution paths, SPIO delivers a more flexible, accurate, and reliable foundation for automated data science.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03645v1">LLM-MC-Affect: LLM-Based Monte Carlo Modeling of Affective Trajectories and Latent Ambiguity for Interpersonal Dynamic Insight</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Emotional coordination is a core property of human interaction that shapes how relational meaning is constructed in real time. While text-based affect inference has become increasingly feasible, prior approaches often treat sentiment as a deterministic point estimate for individual speakers, failing to capture the inherent subjectivity, latent ambiguity, and sequential coupling found in mutual exchanges. We introduce LLM-MC-Affect, a probabilistic framework that characterizes emotion not as a static label, but as a continuous latent probability distribution defined over an affective space. By leveraging stochastic LLM decoding and Monte Carlo estimation, the methodology approximates these distributions to derive high-fidelity sentiment trajectories that explicitly quantify both central affective tendencies and perceptual ambiguity. These trajectories enable a structured analysis of interpersonal coupling through sequential cross-correlation and slope-based indicators, identifying leading or lagging influences between interlocutors. To validate the interpretive capacity of this approach, we utilize teacher-student instructional dialogues as a representative case study, where our quantitative indicators successfully distill high-level interaction insights such as effective scaffolding. This work establishes a scalable and deployable pathway for understanding interpersonal dynamics, offering a generalizable solution that extends beyond education to broader social and behavioral research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03640v1">Verbatim Data Transcription Failures in LLM Code Generation: A State-Tracking Stress Test</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Many real-world software tasks require exact transcription of provided data into code, such as cryptographic constants, protocol test vectors, allowlists, and calibration tables. These tasks are operationally sensitive because small omissions or alterations can remain silent while producing syntactically valid programs. This paper introduces a deliberately minimal transcription-to-code benchmark to isolate this reliability concern in LLM-based code generation. Given a list of high-precision decimal constants, a model must generate Python code that embeds the constants verbatim and performs a simple aggregate computation. We describe the prompting variants, evaluation protocol based on exact-string inclusion, and analysis framework used to characterize state-tracking and long-horizon generation failures. The benchmark is intended as a compact stress test that complements existing code-generation evaluations by focusing on data integrity rather than algorithmic reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02813v2">HAL: Inducing Human-likeness in LLMs with Alignment</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Conversational human-likeness plays a central role in human-AI interaction, yet it has remained difficult to define, measure, and optimize. As a result, improvements in human-like behavior are largely driven by scale or broad supervised training, rather than targeted alignment. We introduce Human Aligning LLMs (HAL), a framework for aligning language models to conversational human-likeness using an interpretable, data-driven reward. HAL derives explicit conversational traits from contrastive dialogue data, combines them into a compact scalar score, and uses this score as a transparent reward signal for alignment with standard preference optimization methods. Using this approach, we align models of varying sizes without affecting their overall performance. In large-scale human evaluations, models aligned with HAL are more frequently perceived as human-like in conversation. Because HAL operates over explicit, interpretable traits, it enables inspection of alignment behavior and diagnosis of unintended effects. More broadly, HAL demonstrates how soft, qualitative properties of language--previously outside the scope for alignment--can be made measurable and aligned in an interpretable and explainable way.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03630v1">Reasoning Model Is Superior LLM-Judge, Yet Suffers from Biases</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 11 pages, 4 figures
    </div>
    <details class="paper-abstract">
      This paper presents the first systematic comparison investigating whether Large Reasoning Models (LRMs) are superior judge to non-reasoning LLMs. Our empirical analysis yields four key findings: 1) LRMs outperform non-reasoning LLMs in terms of judgment accuracy, particularly on reasoning-intensive tasks; 2) LRMs demonstrate superior instruction-following capabilities in evaluation contexts; 3) LRMs exhibit enhanced robustness against adversarial attacks targeting judgment tasks; 4) However, LRMs still exhibit strong biases in superficial quality. To improve the robustness against biases, we propose PlanJudge, an evaluation strategy that prompts the model to generate an explicit evaluation plan before execution. Despite its simplicity, our experiments demonstrate that PlanJudge significantly mitigates biases in both LRMs and standard LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03627v1">Evaluating the Pre-Consultation Ability of LLMs using Diagnostic Guidelines</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 EACL 2026 Industry
    </div>
    <details class="paper-abstract">
      We introduce EPAG, a benchmark dataset and framework designed for Evaluating the Pre-consultation Ability of LLMs using diagnostic Guidelines. LLMs are evaluated directly through HPI-diagnostic guideline comparison and indirectly through disease diagnosis. In our experiments, we observe that small open-source models fine-tuned with a well-curated, task-specific dataset can outperform frontier LLMs in pre-consultation. Additionally, we find that increased amount of HPI (History of Present Illness) does not necessarily lead to improved diagnostic performance. Further experiments reveal that the language of pre-consultation influences the characteristics of the dialogue. By open-sourcing our dataset and evaluation pipeline on https://github.com/seemdog/EPAG, we aim to contribute to the evaluation and further development of LLM applications in real-world clinical settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.21318v3">Beyond Chemical QA: Evaluating LLM's Chemical Reasoning with Modular Chemical Operations</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 Accepted by NeurIPS 2025 Dataset Track, 22 pages, 10 figures
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) with Chain-of-Thought (CoT) reasoning excel in mathematics and coding, their potential for systematic reasoning in chemistry, a domain demanding rigorous structural analysis for real-world tasks like drug design and reaction engineering, remains untapped. Current benchmarks focus on simple knowledge retrieval, neglecting step-by-step reasoning required for complex tasks such as molecular optimization and reaction prediction. To address this, we introduce ChemCoTBench, a reasoning framework that bridges molecular structure understanding with arithmetic-inspired operations, including addition, deletion, and substitution, to formalize chemical problem-solving into transparent, step-by-step workflows. By treating molecular transformations as modular "chemical operations", the framework enables slow-thinking reasoning, mirroring the logic of mathematical proofs while grounding solutions in real-world chemical constraints. We evaluate models on two high-impact tasks: Molecular Property Optimization and Chemical Reaction Prediction. These tasks mirror real-world challenges while providing structured evaluability. By providing annotated datasets, a reasoning taxonomy, and baseline evaluations, ChemCoTBench bridges the gap between abstract reasoning methods and practical chemical discovery, establishing a foundation for advancing LLMs as tools for AI-driven scientific innovation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03600v1">ALERT: Zero-shot LLM Jailbreak Detection via Internal Discrepancy Amplification</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Despite rich safety alignment strategies, large language models (LLMs) remain highly susceptible to jailbreak attacks, which compromise safety guardrails and pose serious security risks. Existing detection methods mainly detect jailbreak status relying on jailbreak templates present in the training data. However, few studies address the more realistic and challenging zero-shot jailbreak detection setting, where no jailbreak templates are available during training. This setting better reflects real-world scenarios where new attacks continually emerge and evolve. To address this challenge, we propose a layer-wise, module-wise, and token-wise amplification framework that progressively magnifies internal feature discrepancies between benign and jailbreak prompts. We uncover safety-relevant layers, identify specific modules that inherently encode zero-shot discriminative signals, and localize informative safety tokens. Building upon these insights, we introduce ALERT (Amplification-based Jailbreak Detector), an efficient and effective zero-shot jailbreak detector that introduces two independent yet complementary classifiers on amplified representations. Extensive experiments on three safety benchmarks demonstrate that ALERT achieves consistently strong zero-shot detection performance. Specifically, (i) across all datasets and attack strategies, ALERT reliably ranks among the top two methods, and (ii) it outperforms the second-best baseline by at least 10% in average Accuracy and F1-score, and sometimes by up to 40%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03597v1">From Chains to Graphs: Self-Structured Reasoning for General-Domain LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show strong reasoning ability in open-domain question answering, yet their reasoning processes are typically linear and often logically inconsistent. In contrast, real-world reasoning requires integrating multiple premises and solving subproblems in parallel. Existing methods, such as Chain-of-Thought (CoT), express reasoning in a linear textual form, which may appear coherent but frequently leads to inconsistent conclusions. Recent approaches rely on externally provided graphs and do not explore how LLMs can construct and use their own graph-structured reasoning, particularly in open-domain QA. To fill this gap, we novelly explore graph-structured reasoning of LLMs in general-domain question answering. We propose Self-Graph Reasoning (SGR), a framework that enables LLMs to explicitly represent their reasoning process as a structured graph before producing the final answer. We further construct a graph-structured reasoning dataset that merges multiple candidate reasoning graphs into refined graph structures for model training. Experiments on five QA benchmarks across both general and specialized domains show that SGR consistently improves reasoning consistency and yields a 17.74% gain over the base model. The LLaMA-3.3-70B model fine-tuned with SGR performs comparably to GPT-4o and surpasses Claude-3.5-Haiku, demonstrating the effectiveness of graph-structured reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03594v1">Jailbreaking LLMs & VLMs: Mechanisms, Evaluation, and Unified Defense</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      This paper provides a systematic survey of jailbreak attacks and defenses on Large Language Models (LLMs) and Vision-Language Models (VLMs), emphasizing that jailbreak vulnerabilities stem from structural factors such as incomplete training data, linguistic ambiguity, and generative uncertainty. It further differentiates between hallucinations and jailbreaks in terms of intent and triggering mechanisms. We propose a three-dimensional survey framework: (1) Attack dimension-including template/encoding-based, in-context learning manipulation, reinforcement/adversarial learning, LLM-assisted and fine-tuned attacks, as well as prompt- and image-level perturbations and agent-based transfer in VLMs; (2) Defense dimension-encompassing prompt-level obfuscation, output evaluation, and model-level alignment or fine-tuning; and (3) Evaluation dimension-covering metrics such as Attack Success Rate (ASR), toxicity score, query/time cost, and multimodal Clean Accuracy and Attribute Success Rate. Compared with prior works, this survey spans the full spectrum from text-only to multimodal settings, consolidating shared mechanisms and proposing unified defense principles: variant-consistency and gradient-sensitivity detection at the perception layer, safety-aware decoding and output review at the generation layer, and adversarially augmented preference alignment at the parameter layer. Additionally, we summarize existing multimodal safety benchmarks and discuss future directions, including automated red teaming, cross-modal collaborative defense, and standardized evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03590v1">Can LLMs See Without Pixels? Benchmarking Spatial Intelligence from Textual Descriptions</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Recent advancements in Spatial Intelligence (SI) have predominantly relied on Vision-Language Models (VLMs), yet a critical question remains: does spatial understanding originate from visual encoders or the fundamental reasoning backbone? Inspired by this question, we introduce SiT-Bench, a novel benchmark designed to evaluate the SI performance of Large Language Models (LLMs) without pixel-level input, comprises over 3,800 expert-annotated items across five primary categories and 17 subtasks, ranging from egocentric navigation and perspective transformation to fine-grained robotic manipulation. By converting single/multi-view scenes into high-fidelity, coordinate-aware textual descriptions, we challenge LLMs to perform symbolic textual reasoning rather than visual pattern matching. Evaluation results of state-of-the-art (SOTA) LLMs reveals that while models achieve proficiency in localized semantic tasks, a significant "spatial gap" remains in global consistency. Notably, we find that explicit spatial reasoning significantly boosts performance, suggesting that LLMs possess latent world-modeling potential. Our proposed dataset SiT-Bench serves as a foundational resource to foster the development of spatially-grounded LLM backbones for future VLMs and embodied agents. Our code and benchmark will be released at https://github.com/binisalegend/SiT-Bench .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03589v1">OLA: Output Language Alignment in Code-Switched LLM Interactions</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Code-switching, alternating between languages within a conversation, is natural for multilingual users, yet poses fundamental challenges for large language models (LLMs). When a user code-switches in their prompt to an LLM, they typically do not specify the expected language of the LLM response, and thus LLMs must infer the output language from contextual and pragmatic cues. We find that current LLMs systematically fail to align with this expectation, responding in undesired languages even when cues are clear to humans. We introduce OLA, a benchmark to evaluate LLMs' Output Language Alignment in code-switched interactions. OLA focuses on Korean--English code-switching and spans simple intra-sentential mixing to instruction-content mismatches. Even frontier models frequently misinterpret implicit language expectation, exhibiting a bias toward non-English responses. We further show this bias generalizes beyond Korean to Chinese and Indonesian pairs. Models also show instability through mid-response switching and language intrusions. Chain-of-Thought prompting fails to resolve these errors, indicating weak pragmatic reasoning about output language. However, Code-Switching Aware DPO with minimal data (about 1K examples) substantially reduces misalignment, suggesting these failures stem from insufficient alignment rather than fundamental limitations. Our results highlight the need to align multilingual LLMs with users' implicit expectations in real-world code-switched interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15755v2">Multi-Agent LLM Orchestration Achieves Deterministic, High-Quality Decision Support for Incident Response</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 10 pages, 4 tables. v2: Expanded limitations, added threats to validity, clarified agent definition, added reproducibility notes, updated Phase 2 timeline with current models (GPT-5.2, Claude Sonnet 4.5, Llama 3.3 70B). No changes to experimental results
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) promise to accelerate incident response in production systems, yet single-agent approaches generate vague, unusable recommendations. We present MyAntFarm.ai, a reproducible containerized framework demonstrating that multi-agent orchestration fundamentally transforms LLM-based incident response quality. Through 348 controlled trials comparing single-agent copilot versus multi-agent systems on identical incident scenarios, we find that multi-agent orchestration achieves 100% actionable recommendation rate versus 1.7% for single-agent approaches, an 80 times improvement in action specificity and 140 times improvement in solution correctness. Critically, multi-agent systems exhibit zero quality variance across all trials, enabling production SLA commitments impossible with inconsistent single-agent outputs. Both architectures achieve similar comprehension latency (approx.40s), establishing that the architectural value lies in deterministic quality, not speed. We introduce Decision Quality (DQ), a novel metric capturing validity, specificity, and correctness properties essential for operational deployment that existing LLM metrics do not address. These findings reframe multi-agent orchestration from a performance optimization to a production-readiness requirement for LLM-based incident response. All code, Docker configurations, and trial data are publicly available for reproduction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.12227v2">Uncovering Bias Paths with LLM-guided Causal Discovery: An Active Learning and Dynamic Scoring Approach</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 To be presented at AAAI 2026
    </div>
    <details class="paper-abstract">
      Ensuring fairness in machine learning requires understanding how sensitive attributes like race or gender causally influence outcomes. Existing causal discovery (CD) methods often struggle to recover fairness-relevant pathways in the presence of noise, confounding, or data corruption. Large language models (LLMs) offer a complementary signal by leveraging semantic priors from variable metadata. We propose a hybrid LLM-guided CD framework that extends a breadth-first search strategy with active learning and dynamic scoring. Variable pairs are prioritized for querying using a composite score combining mutual information, partial correlation, and LLM confidence, enabling more efficient and robust structure discovery. To evaluate fairness sensitivity, we introduce a semi-synthetic benchmark based on the UCI Adult dataset, embedding domain-informed bias pathways alongside noise and latent confounders. We assess how well CD methods recover both global graph structure and fairness-critical paths (e.g., sex-->education-->income). Our results demonstrate that LLM-guided methods, including our active, dynamically scored variant, outperform baselines in recovering fairness-relevant structure under noisy conditions. We analyze when LLM-driven insights complement statistical dependencies and discuss implications for fairness auditing in high-stakes domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.00340v2">Better Call CLAUSE: A Discrepancy Benchmark for Auditing LLMs Legal Reasoning Capabilities</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 42 pages, 4 images
    </div>
    <details class="paper-abstract">
      The rapid integration of large language models (LLMs) into high-stakes legal work has exposed a critical gap: no benchmark exists to systematically stress-test their reliability against the nuanced, adversarial, and often subtle flaws present in real-world contracts. To address this, we introduce CLAUSE, a first-of-its-kind benchmark designed to evaluate the fragility of an LLM's legal reasoning. We study the capabilities of LLMs to detect and reason about fine-grained discrepancies by producing over 7500 real-world perturbed contracts from foundational datasets like CUAD and ContractNLI. Our novel, persona-driven pipeline generates 10 distinct anomaly categories, which are then validated against official statutes using a Retrieval-Augmented Generation (RAG) system to ensure legal fidelity. We use CLAUSE to evaluate leading LLMs' ability to detect embedded legal flaws and explain their significance. Our analysis shows a key weakness: these models often miss subtle errors and struggle even more to justify them legally. Our work outlines a path to identify and correct such reasoning failures in legal AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.22359v3">League of LLMs: A Benchmark-Free Paradigm for Mutual Evaluation of Large Language Models</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Although large language models (LLMs) have shown exceptional capabilities across a wide range of tasks, reliable evaluation remains a critical challenge due to data contamination, opaque operation, and subjective preferences. To address these issues, we propose League of LLMs (LOL), a novel benchmark-free evaluation paradigm that organizes multiple LLMs into a self-governed league for multi-round mutual evaluation. LOL integrates four core criteria (dynamic, transparent, objective, and professional) to mitigate key limitations of existing paradigms. Experiments on eight mainstream LLMs in mathematics and programming demonstrate that LOL can effectively distinguish LLM capabilities while maintaining high internal ranking stability (Top-$k$ consistency $= 70.7\%$). Beyond ranking, LOL reveals empirical findings that are difficult for traditional paradigms to capture. For instance, ``memorization-based answering'' behaviors are observed in some models, and a statistically significant homophily bias is found within the OpenAI family ($Δ= 9$, $p < 0.05$). Finally, we make our framework and code publicly available as a valuable complement to the current LLM evaluation ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24968v2">The Impact of LLMs on Online News Consumption and Production</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) change how consumers acquire information online; their bots also crawl news publishers' websites for training data and to answer consumer queries; and they provide tools that can lower the cost of content creation. These changes lead to predictions of adverse impact on news publishers in the form of lowered consumer demand, reduced demand for newsroom employees, and an increase in news "slop." Consequently, some publishers strategically responded by blocking LLM access to their websites using the robots.txt file standard. Using high-frequency granular data, we document four effects related to the predicted shifts in news publishing following the introduction of generative AI (GenAI). First, we find a moderate decline in traffic to news publishers occurring after August 2024. Second, using a difference-in-differences approach, we find that blocking GenAI bots can be associated with a reduction of total website traffic to large publishers compared to not blocking. Third, on the hiring side, we do not find evidence that LLMs are replacing editorial or content-production jobs yet. The share of new editorial and content-production job listings increases over time. Fourth, regarding content production, we find no evidence that large publishers increased text volume; instead, they significantly increased rich content and use more advertising and targeting technologies. Together, these findings provide early evidence of some unforeseen impacts of the introduction of LLMs on news production and consumption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10411v4">SWAA: Sliding Window Attention Adaptation for Efficient Long-Context LLMs Without Pretraining</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      The quadratic complexity of self-attention in Transformer-based Large Language Models (LLMs) renders long-context inference prohibitively expensive. While Sliding Window Attention (SWA), the simplest sparse attention pattern, offers a linear-complexity alternative, naively applying it to models pretrained with Full Attention (FA) causes catastrophic long-context performance collapse due to the training-inference mismatch. To address this, we propose Sliding Window Attention Adaptation (SWAA), a plug-and-play toolkit of recipes that adapt FA models to SWA without costly pretraining. SWAA systematically combines five strategies: (1) applying SWA only during prefilling; (2) preserving "sink" tokens; (3) interleaving FA/SWA layers; (4) chain-of-thought (CoT); and (5) fine-tuning. Our experiments demonstrate that while individual methods are insufficient, specific synergistic combinations can effectively recover original long-context capabilities. After further analyzing performance-efficiency trade-offs, we identify recommended SWAA configurations for diverse scenarios, which achieve 30% to 100% speedups for long-context LLM inference with acceptable quality loss. Our code is available at https://github.com/yuyijiong/sliding-window-attention-adaptation
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03553v1">Evaluating LLMs for Police Decision-Making: A Framework Based on Police Action Scenarios</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 This work was accepted at AAAI 2026 social good track
    </div>
    <details class="paper-abstract">
      The use of Large Language Models (LLMs) in police operations is growing, yet an evaluation framework tailored to police operations remains absent. While LLM's responses may not always be legally incorrect, their unverified use still can lead to severe issues such as unlawful arrests and improper evidence collection. To address this, we propose PAS (Police Action Scenarios), a systematic framework covering the entire evaluation process. Applying this framework, we constructed a novel QA dataset from over 8,000 official documents and established key metrics validated through statistical analysis with police expert judgements. Experimental results show that commercial LLMs struggle with our new police-related tasks, particularly in providing fact-based recommendations. This study highlights the necessity of an expandable evaluation framework to ensure reliable AI-driven police operations. We release our data and prompt template.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03550v1">ReEfBench: Quantifying the Reasoning Efficiency of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Test-time scaling has enabled Large Language Models (LLMs) to tackle complex reasoning, yet the limitations of current Chain-of-Thought (CoT) evaluation obscures whether performance gains stem from genuine reasoning or mere verbosity. To address this, (1) we propose a novel neuro-symbolic framework for the non-intrusive, comprehensive process-centric evaluation of reasoning. (2) Through this lens, we identify four distinct behavioral prototypes and diagnose the failure modes. (3) We examine the impact of inference mode, training strategy, and model scale. Our analysis reveals that extended token generation is not a prerequisite for deep reasoning. Furthermore, we reveal critical constraints: mixing long and short CoT data in training risks in premature saturation and collapse, while distillation into smaller models captures behavioral length but fails to replicate logical efficacy due to intrinsic capacity limits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.02006v2">MorphServe: Efficient and Workload-Aware LLM Serving via Runtime Quantized Layer Swapping and KV Cache Resizing</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 19 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Efficiently serving large language models (LLMs) under dynamic and bursty workloads remains a key challenge for real-world deployment. Existing serving frameworks and static model compression techniques fail to adapt to workload fluctuations, leading to either service-level objective (SLO) violations under full-precision serving or persistent accuracy degradation with static quantization. We present MorphServe, a dynamic, workload-aware LLM serving framework based on morphological adaptation. MorphServe introduces two asynchronous, token-level runtime mechanisms: quantized layer swapping, which selectively replaces less impactful layers with quantized alternatives during high-load periods, and pressure-aware KV cache resizing, which dynamically adjusts KV cache capacity in response to memory pressure. These mechanisms enable state-preserving transitions with minimum runtime overhead and are fully compatible with modern scheduling and attention techniques. Extensive experiments on Vicuna and Llama family models with real-world workloads demonstrate that MorphServe reduces average SLO violations by 92.45 percent and improves the P95 TTFT latency by 2.2x-3.9x compared to full-precision serving, without compromising generation quality. These results establish MorphServe as a practical and elastic solution for LLM deployment in dynamic environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.20278v4">Quantifying LLM Biases Across Instruction Boundary in Mixed Question Forms</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) annotated datasets are widely used nowadays, however, large-scale annotations often show biases in low-quality datasets. For example, Multiple-Choice Questions (MCQs) datasets with one single correct option is common, however, there may be questions attributed to none or multiple correct options; whereas true-or-false questions are supposed to be labeled with either True or False, but similarly the text can include unsolvable elements, which should be further labeled as Unknown. There are problems when low-quality datasets with mixed question forms can not be identified. We refer to these exceptional label forms as Sparse Labels, and LLMs' ability to distinguish datasets with Sparse Labels mixture is important. Since users may not know situations of datasets, their instructions can be biased. To study how different instruction settings affect LLMs' identifications of Sparse Labels mixture, we introduce the concept of Instruction Boundary, which systematically evaluates different instruction settings that lead to biases. We propose BiasDetector, a diagnostic benchmark to systematically evaluate LLMs on datasets with mixed question forms under Instruction Boundary settings. Experiments show that users' instructions induce large biases on our benchmark, highlighting the need not only for LLM developers to recognize risks of LLM biased annotation resulting in Sparse Labels mixture, but also problems arising from users' instructions to identify them. Code, datasets and detailed implementations are available at https://github.com/ZpLing/Instruction-Boundary.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.03747v3">Context-Alignment: Activating and Enhancing LLM Capabilities in Time Series</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 This paper has been accepted by ICLR 2025
    </div>
    <details class="paper-abstract">
      Recently, leveraging pre-trained Large Language Models (LLMs) for time series (TS) tasks has gained increasing attention, which involves activating and enhancing LLMs' capabilities. Many methods aim to activate LLMs' capabilities based on token-level alignment, but overlook LLMs' inherent strength in natural language processing -- \textit{their deep understanding of linguistic logic and structure rather than superficial embedding processing.} We propose Context-Alignment (CA), a new paradigm that aligns TS with a linguistic component in the language environments familiar to LLMs to enable LLMs to contextualize and comprehend TS data, thereby activating their capabilities. Specifically, such context-level alignment comprises structural alignment and logical alignment, which is achieved by Dual-Scale Context-Alignment GNNs (DSCA-GNNs) applied to TS-language multimodal inputs. Structural alignment utilizes dual-scale nodes to describe hierarchical structure in TS-language, enabling LLMs to treat long TS data as a whole linguistic component while preserving intrinsic token features. Logical alignment uses directed edges to guide logical relationships, ensuring coherence in the contextual semantics. Following the DSCA-GNNs framework, we propose an instantiation method of CA, termed Few-Shot prompting Context-Alignment (FSCA), to enhance the capabilities of pre-trained LLMs in handling TS tasks. FSCA can be flexibly and repeatedly integrated into various layers of pre-trained LLMs to improve awareness of logic and structure, thereby enhancing performance. Extensive experiments show the effectiveness of FSCA and the importance of Context-Alignment across tasks, particularly in few-shot and zero-shot forecasting, confirming that Context-Alignment provides powerful prior knowledge on context. The code is open-sourced at https://github.com/tokaka22/ICLR25-FSCA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03484v1">From Bits to Chips: An LLM-based Hardware-Aware Quantization Agent for Streamlined Deployment of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Deploying models, especially large language models (LLMs), is becoming increasingly attractive to a broader user base, including those without specialized expertise. However, due to the resource constraints of certain hardware, maintaining high accuracy with larger model while meeting the hardware requirements remains a significant challenge. Model quantization technique helps mitigate memory and compute bottlenecks, yet the added complexities of tuning and deploying quantized models further exacerbates these challenges, making the process unfriendly to most of the users. We introduce the Hardware-Aware Quantization Agent (HAQA), an automated framework that leverages LLMs to streamline the entire quantization and deployment process by enabling efficient hyperparameter tuning and hardware configuration, thereby simultaneously improving deployment quality and ease of use for a broad range of users. Our results demonstrate up to a 2.3x speedup in inference, along with increased throughput and improved accuracy compared to unoptimized models on Llama. Additionally, HAQA is designed to implement adaptive quantization strategies across diverse hardware platforms, as it automatically finds optimal settings even when they appear counterintuitive, thereby reducing extensive manual effort and demonstrating superior adaptability. Code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03475v1">CPGPrompt: Translating Clinical Guidelines into LLM-Executable Decision Support</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Clinical practice guidelines (CPGs) provide evidence-based recommendations for patient care; however, integrating them into Artificial Intelligence (AI) remains challenging. Previous approaches, such as rule-based systems, face significant limitations, including poor interpretability, inconsistent adherence to guidelines, and narrow domain applicability. To address this, we develop and validate CPGPrompt, an auto-prompting system that converts narrative clinical guidelines into large language models (LLMs). Our framework translates CPGs into structured decision trees and utilizes an LLM to dynamically navigate them for patient case evaluation. Synthetic vignettes were generated across three domains (headache, lower back pain, and prostate cancer) and distributed into four categories to test different decision scenarios. System performance was assessed on both binary specialty-referral decisions and fine-grained pathway-classification tasks. The binary specialty referral classification achieved consistently strong performance across all domains (F1: 0.85-1.00), with high recall (1.00 $\pm$ 0.00). In contrast, multi-class pathway assignment showed reduced performance, with domain-specific variations: headache (F1: 0.47), lower back pain (F1: 0.72), and prostate cancer (F1: 0.77). Domain-specific performance differences reflected the structure of each guideline. The headache guideline highlighted challenges with negation handling. The lower back pain guideline required temporal reasoning. In contrast, prostate cancer pathways benefited from quantifiable laboratory tests, resulting in more reliable decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04435v1">Accommodation and Epistemic Vigilance: A Pragmatic Account of Why LLMs Fail to Challenge Harmful Beliefs</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) frequently fail to challenge users' harmful beliefs in domains ranging from medical advice to social reasoning. We argue that these failures can be understood and addressed pragmatically as consequences of LLMs defaulting to accommodating users' assumptions and exhibiting insufficient epistemic vigilance. We show that social and linguistic factors known to influence accommodation in humans (at-issueness, linguistic encoding, and source reliability) similarly affect accommodation in LLMs, explaining performance differences across three safety benchmarks that test models' ability to challenge harmful beliefs, spanning misinformation (Cancer-Myth, SAGE-Eval) and sycophancy (ELEPHANT). We further show that simple pragmatic interventions, such as adding the phrase "wait a minute", significantly improve performance on these benchmarks while preserving low false-positive rates. Our results highlight the importance of considering pragmatics for evaluating LLM behavior and improving LLM safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04426v1">XGrammar 2: Dynamic and Efficient Structured Generation Engine for Agentic LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Modern LLM agents are required to handle increasingly complex structured generation tasks, such as tool calling and conditional structured generation. These tasks are significantly more dynamic than predefined structures, posing new challenges to the current structured generation engines. In this paper, we propose XGrammar 2, a highly optimized structured generation engine for agentic LLMs. XGrammar 2 accelerates the mask generation for these dynamic structured generation tasks through a new dynamic dispatching semantics: TagDispatch. We further introduce a just-in-time (JIT) compilation method to reduce compilation time and a cross-grammar caching mechanism to leverage the common sub-structures across different grammars. Additionally, we extend the previous PDA-based mask generation algorithm to the Earley-parser-based one and design a repetition compression algorithm to handle repetition structures in grammars. Evaluation results show that XGrammar 2 can achieve more than 6x speedup over the existing structured generation engines. Integrated with an LLM inference engine, XGrammar 2 can handle dynamic structured generation tasks with near-zero overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04424v1">Gavel: Agent Meets Checklist for Evaluating LLMs on Long-Context Legal Summarization</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 webpage at https://yao-dou.github.io/gavel/
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) now support contexts of up to 1M tokens, but their effectiveness on complex long-context tasks remains unclear. In this paper, we study multi-document legal case summarization, where a single case often spans many documents totaling 100K-500K tokens. We introduce Gavel-Ref, a reference-based evaluation framework with multi-value checklist evaluation over 26 items, as well as residual fact and writing-style evaluations. Using Gavel-Ref, we go beyond the single aggregate scores reported in prior work and systematically evaluate 12 frontier LLMs on 100 legal cases ranging from 32K to 512K tokens, primarily from 2025. Our results show that even the strongest model, Gemini 2.5 Pro, achieves only around 50 of $S_{\text{Gavel-Ref}}$, highlighting the difficulty of the task. Models perform well on simple checklist items (e.g., filing date) but struggle on multi-value or rare ones such as settlements and monitor reports. As LLMs continue to improve and may surpass human-written summaries -- making human references less reliable -- we develop Gavel-Agent, an efficient and autonomous agent scaffold that equips LLMs with six tools to navigate and extract checklists directly from case documents. With Qwen3, Gavel-Agent reduces token usage by 36% while resulting in only a 7% drop in $S_{\text{checklist}}$ compared to end-to-end extraction with GPT-4.1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04388v1">LLM-Guided Lifecycle-Aware Clustering of Multi-Turn Customer Support Conversations</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 Accepted in AACL 2025 Main Conference
    </div>
    <details class="paper-abstract">
      Clustering customer chat data is vital for cloud providers handling multi service queries. Traditional methods struggle with overlapping concerns and create broad, static clusters that degrade over time. Reclustering disrupts continuity, making issue tracking difficult. We propose an adaptive system that segments multi turn chats into service specific concerns and incrementally refines clusters as new issues arise. Cluster quality is tracked via DaviesBouldin Index and Silhouette Scores, with LLM based splitting applied only to degraded clusters. Our method improves Silhouette Scores by over 100\% and reduces DBI by 65.6\% compared to baselines, enabling scalable, real time analytics without full reclustering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04387v1">The Language of Bargaining: Linguistic Effects in LLM Negotiations</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Negotiation is a core component of social intelligence, requiring agents to balance strategic reasoning, cooperation, and social norms. Recent work shows that LLMs can engage in multi-turn negotiation, yet nearly all evaluations occur exclusively in English. Using controlled multi-agent simulations across Ultimatum, Buy-Sell, and Resource Exchange games, we systematically isolate language effects across English and four Indic framings (Hindi, Punjabi, Gujarati, Marwadi) by holding game rules, model parameters, and incentives constant across all conditions. We find that language choice can shift outcomes more strongly than changing models, reversing proposer advantages and reallocating surplus. Crucially, effects are task-contingent: Indic languages reduce stability in distributive games yet induce richer exploration in integrative settings. Our results demonstrate that evaluating LLM negotiation solely in English yields incomplete and potentially misleading conclusions. These findings caution against English-only evaluation of LLMs and suggest that culturally-aware evaluation is essential for fair deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.17542v2">AssertFlip: Reproducing Bugs via Inversion of LLM-Generated Passing Tests</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Bug reproduction is critical in the software debugging and repair process, yet the majority of bugs in open-source and industrial settings lack executable tests to reproduce them at the time they are reported, making diagnosis and resolution more difficult and time-consuming. To address this challenge, we introduce AssertFlip, a novel technique for automatically generating Bug Reproducible Tests (BRTs) using large language models (LLMs). Unlike existing methods that attempt direct generation of failing tests, AssertFlip first generates passing tests on the buggy behaviour and then inverts these tests to fail when the bug is present. We hypothesize that LLMs are better at writing passing tests than ones that crash or fail on purpose. Our results show that AssertFlip outperforms all known techniques in the leaderboard of SWT-Bench, a benchmark curated for BRTs. Specifically, AssertFlip achieves a fail-to-pass success rate of 43.6% on the SWT-Bench-Verified subset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.10095v3">Cognitive-Mental-LLM: Evaluating Reasoning in Large Language Models for Mental Health Prediction via Online Text</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 8 pages, 4 Figures, 3 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated potential in predicting mental health outcomes from online text, yet traditional classification methods often lack interpretability and robustness. This study evaluates structured reasoning techniques-Chain-of-Thought (CoT), Self-Consistency (SC-CoT), and Tree-of-Thought (ToT)-to improve classification accuracy across multiple mental health datasets sourced from Reddit. We analyze reasoning-driven prompting strategies, including Zero-shot CoT and Few-shot CoT, using key performance metrics such as Balanced Accuracy, F1 score, and Sensitivity/Specificity. Our findings indicate that reasoning-enhanced techniques improve classification performance over direct prediction, particularly in complex cases. Compared to baselines such as Zero Shot non-CoT Prompting, and fine-tuned pre-trained transformers such as BERT and Mental-RoBerta, and fine-tuned Open Source LLMs such as Mental Alpaca and Mental-Flan-T5, reasoning-driven LLMs yield notable gains on datasets like Dreaddit (+0.52\% over M-LLM, +0.82\% over BERT) and SDCNL (+4.67\% over M-LLM, +2.17\% over BERT). However, performance declines in Depression Severity, and CSSRS predictions suggest dataset-specific limitations, likely due to our using a more extensive test set. Among prompting strategies, Few-shot CoT consistently outperforms others, reinforcing the effectiveness of reasoning-driven LLMs. Nonetheless, dataset variability highlights challenges in model reliability and interpretability. This study provides a comprehensive benchmark of reasoning-based LLM techniques for mental health text classification. It offers insights into their potential for scalable clinical applications while identifying key challenges for future improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.13766v2">Advancing Software Quality: A Standards-Focused Review of LLM-Based Assurance Techniques</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 16 pages, 1 Table, 6 Figures
    </div>
    <details class="paper-abstract">
      Software Quality Assurance (SQA) is critical for delivering reliable, secure, and efficient software products. The Software Quality Assurance Process aims to provide assurance that work products and processes comply with predefined provisions and plans. Recent advancements in Large Language Models (LLMs) present new opportunities to enhance existing SQA processes by automating tasks like requirement analysis, code review, test generation, and compliance checks. Simultaneously, established standards such as ISO/IEC 12207, ISO/IEC 25010, ISO/IEC 5055, ISO 9001/ISO/IEC 90003, CMMI, and TMM provide structured frameworks for ensuring robust quality practices. This paper surveys the intersection of LLM-based SQA methods and these recognized standards, highlighting how AI-driven solutions can augment traditional approaches while maintaining compliance and process maturity. We first review the foundational software quality standards and the technical fundamentals of LLMs in software engineering. Next, we explore various LLM-based SQA applications, including requirement validation, defect detection, test generation, and documentation maintenance. We then map these applications to key software quality frameworks, illustrating how LLMs can address specific requirements and metrics within each standard. Empirical case studies and open-source initiatives demonstrate the practical viability of these methods. At the same time, discussions on challenges (e.g., data privacy, model bias, explainability) underscore the need for deliberate governance and auditing. Finally, we propose future directions encompassing adaptive learning, privacy-focused deployments, multimodal analysis, and evolving standards for AI-driven software quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24449v2">PackKV: Reducing KV Cache Memory Footprint through LLM-Aware Lossy Compression</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Transformer-based large language models (LLMs) have demonstrated remarkable potential across a wide range of practical applications. However, long-context inference remains a significant challenge due to the substantial memory requirements of the key-value (KV) cache, which can scale to several gigabytes as sequence length and batch size increase. In this paper, we present \textbf{PackKV}, a generic and efficient KV cache management framework optimized for long-context generation. %, which synergistically supports both latency-critical and throughput-critical inference scenarios. PackKV introduces novel lossy compression techniques specifically tailored to the characteristics of KV cache data, featuring a careful co-design of compression algorithms and system architecture. Our approach is compatible with the dynamically growing nature of the KV cache while preserving high computational efficiency. Experimental results show that, under the same and minimum accuracy drop as state-of-the-art quantization methods, PackKV achieves, on average, \textbf{153.2}\% higher memory reduction rate for the K cache and \textbf{179.6}\% for the V cache. Furthermore, PackKV delivers extremely high execution throughput, effectively eliminating decompression overhead and accelerating the matrix-vector multiplication operation. Specifically, PackKV achieves an average throughput improvement of \textbf{75.7}\% for K and \textbf{171.7}\% for V across A100 and RTX Pro 6000 GPUs, compared to cuBLAS matrix-vector multiplication kernels, while demanding less GPU memory bandwidth. Code available on https://github.com/BoJiang03/PackKV
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04278v1">From Domains to Instances: Dual-Granularity Data Synthesis for LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      Although machine unlearning is essential for removing private, harmful, or copyrighted content from LLMs, current benchmarks often fail to faithfully represent the true "forgetting scope" learned by the model. We formalize two distinct unlearning granularities, domain-level and instance-level, and propose BiForget, an automated framework for synthesizing high-quality forget sets. Unlike prior work relying on external generators, BiForget exploits the target model per se to elicit data that matches its internal knowledge distribution through seed-guided and adversarial prompting. Our experiments across diverse benchmarks show that it achieves a superior balance of relevance, diversity, and efficiency. Quantitatively, in the Harry Potter domain, it improves relevance by ${\sim}20$ and diversity by ${\sim}$0.05 while halving the total data size compared to SOTAs. Ultimately, it facilitates more robust forgetting and better utility preservation, providing a more rigorous foundation for evaluating LLM unlearning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04277v1">Unlocking the Pre-Trained Model as a Dual-Alignment Calibrator for Post-Trained LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Post-training improves large language models (LLMs) but often worsens confidence calibration, leading to systematic overconfidence. Recent unsupervised post-hoc methods for post-trained LMs (PoLMs) mitigate this by aligning PoLM confidence to that of well-calibrated pre-trained counterparts. However, framing calibration as static output-distribution matching overlooks the inference-time dynamics introduced by post-training. In particular, we show that calibration errors arise from two regimes: (i) confidence drift, where final confidence inflates despite largely consistent intermediate decision processes, and (ii) process drift, where intermediate inference pathways diverge. Guided by this diagnosis, we propose Dual-Align, an unsupervised post-hoc framework for dual alignment in confidence calibration. Dual-Align performs confidence alignment to correct confidence drift via final-distribution matching, and introduces process alignment to address process drift by locating the layer where trajectories diverge and realigning the stability of subsequent inference. This dual strategy learns a single temperature parameter that corrects both drift types without sacrificing post-training performance gains. Experiments show consistent improvements over baselines, reducing calibration errors and approaching a supervised oracle.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04275v1">Shadow Unlearning: A Neuro-Semantic Approach to Fidelity-Preserving Faceless Forgetting in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Machine unlearning aims to selectively remove the influence of specific training samples to satisfy privacy regulations such as the GDPR's 'Right to be Forgotten'. However, many existing methods require access to the data being removed, exposing it to membership inference attacks and potential misuse of Personally Identifiable Information (PII). We address this critical challenge by proposing Shadow Unlearning, a novel paradigm of approximate unlearning, that performs machine unlearning on anonymized forget data without exposing PII. We further propose a novel privacy-preserving framework, Neuro-Semantic Projector Unlearning (NSPU) to achieve Shadow unlearning. To evaluate our method, we compile Multi-domain Fictitious Unlearning (MuFU) forget set across five diverse domains and introduce an evaluation stack to quantify the trade-off between knowledge retention and unlearning effectiveness. Experimental results on various LLMs show that NSPU achieves superior unlearning performance, preserves model utility, and enhances user privacy. Additionally, the proposed approach is at least 10 times more computationally efficient than standard unlearning approaches. Our findings foster a new direction for privacy-aware machine unlearning that balances data protection and model fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04170v1">Agent Drift: Quantifying Behavioral Degradation in Multi-Agent LLM Systems Over Extended Interactions</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Multi-agent Large Language Model (LLM) systems have emerged as powerful architectures for complex task decomposition and collaborative problem-solving. However, their long-term behavioral stability remains largely unexamined. This study introduces the concept of agent drift, defined as the progressive degradation of agent behavior, decision quality, and inter-agent coherence over extended interaction sequences. We present a comprehensive theoretical framework for understanding drift phenomena, proposing three distinct manifestations: semantic drift (progressive deviation from original intent), coordination drift (breakdown in multi-agent consensus mechanisms), and behavioral drift (emergence of unintended strategies). We introduce the Agent Stability Index (ASI), a novel composite metric framework for quantifying drift across twelve dimensions, including response consistency, tool usage patterns, reasoning pathway stability, and inter-agent agreement rates. Through simulation-based analysis and theoretical modeling, we demonstrate how unchecked agent drift can lead to substantial reductions in task completion accuracy and increased human intervention requirements. We propose three mitigation strategies: episodic memory consolidation, drift-aware routing protocols, and adaptive behavioral anchoring. Theoretical analysis suggests these approaches can significantly reduce drift-related errors while maintaining system throughput. This work establishes a foundational methodology for monitoring, measuring, and mitigating agent drift in production agentic AI systems, with direct implications for enterprise deployment reliability and AI safety research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.06303v4">Reward Is Enough: LLMs Are In-Context Reinforcement Learners</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) is a framework for solving sequential decision-making problems. In this work, we demonstrate that, surprisingly, RL emerges during the inference time of large language models (LLMs), a phenomenon we term in-context RL (ICRL). To reveal this capability, we introduce a simple multi-round prompting framework, we call ICRL prompting, for inference-time self-improvement. The goal of ICRL prompting is to guide LLMs to perform reinforcement learning during inference for self-improvement on a given task. After each response, the model receives numerical scalar feedback, denoted as a reward. In the next round, we prompt the LLM again together with a context that concatenates all prior responses and their associated rewards. We consistently observe that response quality improves as the context grows. In other words, the LLM can optimize scalar reward signals during inference, exhibiting behavior analogous to reinforcement learning. We evaluate ICRL prompting on Game of 24, creative writing, ScienceWorld, and Olympiad-level math competitions (AIME and HMMT), demonstrating significant improvements over baselines such as Self-Refine and Reflexion. Notably, even when the reward signals are generated by the same LLM, ICRL prompting still improves performance, highlighting a promising new paradigm for test-time scaling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04093v1">SearchAttack: Red-Teaming LLMs against Real-World Threats via Framing Unsafe Web Information-Seeking Tasks</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 We find that the key to jailbreak the LLM is objectifying its safety responsibility, thus we delegate the open-web to inject harmful semantics and get the huge gain from unmoderated web resources
    </div>
    <details class="paper-abstract">
      Recently, people have suffered and become increasingly aware of the unreliability gap in LLMs for open and knowledge-intensive tasks, and thus turn to search-augmented LLMs to mitigate this issue. However, when the search engine is triggered for harmful tasks, the outcome is no longer under the LLM's control. Once the returned content directly contains targeted, ready-to-use harmful takeaways, the LLM's safeguards cannot withdraw that exposure. Motivated by this dilemma, we identify web search as a critical attack surface and propose \textbf{\textit{SearchAttack}} for red-teaming. SearchAttack outsources the harmful semantics to web search, retaining only the query's skeleton and fragmented clues, and further steers LLMs to reconstruct the retrieved content via structural rubrics to achieve malicious goals. Extensive experiments are conducted to red-team the search-augmented LLMs for responsible vulnerability assessment. Empirically, SearchAttack demonstrates strong effectiveness in attacking these systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.13271v2">Do You Get the Hint? Benchmarking LLMs on the Board Game Concept</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved striking successes on many benchmarks, yet recent studies continue to expose fundamental weaknesses. In this paper, we introduce Concept, a simple word-guessing board game, as a benchmark for probing abductive reasoning. Our results show that this game, easily solved by humans (with a success rate of over 90\%), is still very challenging for state-of-the-art LLMs (no model exceeds 40\% success rate). Specifically, we observe that LLMs struggle with interpreting other players' strategic intents, and with correcting initial hypotheses given sequential information updates. In addition, we extend the evaluation across multiple languages, and find that the LLM performance drops further in lower-resource languages (Dutch, French, and Spanish) compared to English.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07842v3">Alignment-Aware Quantization for LLM Safety</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 9 pages, 4 figures. Includes 8 pages of supplementary material
    </div>
    <details class="paper-abstract">
      Safety and efficiency are paramount yet often conflicting requirements for deploying Large Language Models (LLMs). While LLMs are trained to follow human alignment for safety, Post-Training Quantization (PTQ) is applied afterward to ensure efficiency. Here we identify a fundamental flaw in the conventional PTQ paradigm: quantization can turn into a safety vulnerability if it only aims to achieve low perplexity. To address this, we propose Alignment-Aware Quantization (AAQ), a novel approach that integrates an Alignment-Preserving Contrastive (APC) loss into the PTQ pipeline. Our method explicitly preserves alignment by encouraging the quantized model to mimic its safe, instruction-tuned model while diverging from the unaligned, pre-trained counterpart. AAQ achieves robust safety alignment without specialized safety-focused datasets, using only standard calibration data. We show that AAQ is compatible with standard PTQ techniques and enables robust 4-bit (W4A4) quantization across diverse model families. Our work resolves the critical trade-off between efficiency and safety, paving the way toward LLMs that are both efficient and trustworthy. Anonymized code is available in the supplementary material.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.07261v3">MemHunter: Automated and Verifiable Memorization Detection at Dataset-scale in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 Withdrawn by the authors due to an inconsistency in the reported base model: Section 4 (Experiments) states "Llama-2-7B" while Fig. 3 labels "Llama-2-7B-Chat". Because this affects the experimental configuration, parts of the results must be re-verified by rerunning experiments; we withdraw to avoid misleading readers
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been shown to memorize and reproduce content from their training data, raising significant privacy concerns, especially with web-scale datasets. Existing methods for detecting memorization are primarily sample-specific, relying on manually crafted or discretely optimized memory-inducing prompts generated on a per-sample basis, which become impractical for dataset-level detection due to the prohibitive computational cost of iterating through all samples. In real-world scenarios, data owners may need to verify whether a susceptible LLM has memorized their dataset, particularly if the LLM may have collected the data from the web without authorization. To address this, we introduce MemHunter, which trains a memory-inducing LLM and employs hypothesis testing to efficiently detect memorization at the dataset level, without requiring sample-specific memory inducing. Experiments on models like Pythia and Llama demonstrate that MemHunter can extract up to 40% more training data than existing methods under constrained time resources and reduce search time by up to 80% when integrated as a plug-in. Crucially, MemHunter is the first method capable of dataset-level memorization detection, providing a critical tool for assessing privacy risks in LLMs powered by large-scale datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02764v1">Netflix Artwork Personalization via LLM Post-training</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 6 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated success in various applications of user recommendation and personalization across e-commerce and entertainment. On many entertainment platforms such as Netflix, users typically interact with a wide range of titles, each represented by an artwork. Since users have diverse preferences, an artwork that appeals to one type of user may not resonate with another with different preferences. Given this user heterogeneity, our work explores the novel problem of personalized artwork recommendations according to diverse user preferences. Similar to the multi-dimensional nature of users' tastes, titles contain different themes and tones that may appeal to different viewers. For example, the same title might feature both heartfelt family drama and intense action scenes. Users who prefer romantic content may like the artwork emphasizing emotional warmth between the characters, while those who prefer action thrillers may find high-intensity action scenes more intriguing. Rather than a one-size-fits-all approach, we conduct post-training of pre-trained LLMs to make personalized artwork recommendations, selecting the most preferred visual representation of a title for each user and thereby improving user satisfaction and engagement. Our experimental results with Llama 3.1 8B models (trained on a dataset of 110K data points and evaluated on 5K held-out user-title pairs) show that the post-trained LLMs achieve 3-5\% improvements over the Netflix production model, suggesting a promising direction for granular personalized recommendations using LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02757v1">LLM Agent Framework for Intelligent Change Analysis in Urban Environment using Remote Sensing Imagery</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Existing change detection methods often lack the versatility to handle diverse real-world queries and the intelligence for comprehensive analysis. This paper presents a general agent framework, integrating Large Language Models (LLM) with vision foundation models to form ChangeGPT. A hierarchical structure is employed to mitigate hallucination. The agent was evaluated on a curated dataset of 140 questions categorized by real-world scenarios, encompassing various question types (e.g., Size, Class, Number) and complexities. The evaluation assessed the agent's tool selection ability (Precision/Recall) and overall query accuracy (Match). ChangeGPT, especially with a GPT-4-turbo backend, demonstrated superior performance, achieving a 90.71 % Match rate. Its strength lies particularly in handling change-related queries requiring multi-step reasoning and robust tool selection. Practical effectiveness was further validated through a real-world urban change monitoring case study in Qianhai Bay, Shenzhen. By providing intelligence, adaptability, and multi-type change analysis, ChangeGPT offers a powerful solution for decision-making in remote sensing applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02744v1">SYNAPSE: Empowering LLM Agents with Episodic-Semantic Memory via Spreading Activation</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) excel at generalized reasoning, standard retrieval-augmented approaches fail to address the disconnected nature of long-term agentic memory. To bridge this gap, we introduce Synapse (Synergistic Associative Processing Semantic Encoding), a unified memory architecture that transcends static vector similarity. Drawing from cognitive science, Synapse models memory as a dynamic graph where relevance emerges from spreading activation rather than pre-computed links. By integrating lateral inhibition and temporal decay, the system dynamically highlights relevant sub-graphs while filtering interference. We implement a Triple Hybrid Retrieval strategy that fuses geometric embeddings with activation-based graph traversal. Comprehensive evaluations on the LoCoMo benchmark show that Synapse significantly outperforms state-of-the-art methods in complex temporal and multi-hop reasoning tasks, offering a robust solution to the "Contextual Tunneling" problem. Our code and data will be made publicly available upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.00454v4">Learning an Efficient Multi-Turn Dialogue Evaluator from Multiple LLM Judges</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 20 pages, 4 pages, under review
    </div>
    <details class="paper-abstract">
      Evaluating the conversational abilities of large language models (LLMs) remains a challenging task. Current mainstream approaches primarily rely on the "LLM-as-a-judge" paradigm, where an LLM is prompted to serve as an evaluator to assess dialogue quality. However, such methods often suffer from various biases, which undermine the reliability and consistency of the evaluation results. To mitigate these biases, recent methods employ multiple LLMs as judges and aggregate their judgments to select the optimal assessment. Although effective, this multi-judge approach incurs significant computational overhead during inference. In this paper, we propose an efficient dialogue evaluator that captures the collective wisdom of multiple LLM judges by aggregating their preference knowledge into a single model. Our approach preserves the advantages of diverse multi-judge feedback while drastically reducing the evaluation cost, enabling fast, flexible, and fine-grained dialogue quality assessment. Extensive experiments on seven single rating and pairwise comparison dialogue evaluation benchmarks demonstrate that our method outperforms existing baselines across diverse scenarios, showcasing its efficiency and robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.12645v4">Diagnostic-Guided Dynamic Profile Optimization for LLM-based User Simulators in Sequential Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled realistic user simulators for developing and evaluating recommender systems (RSs). However, existing LLM-based simulators for RSs face two major limitations: (1) static and single-step prompt-based inference that leads to inaccurate and incomplete user profile construction; (2) unrealistic and single-round recommendation-feedback interaction pattern that fails to capture real-world scenarios. To address these limitations, we propose DGDPO (Diagnostic-Guided Dynamic Profile Optimization), a novel framework that constructs user profile through a dynamic and iterative optimization process to enhance the simulation fidelity. Specifically, DGDPO incorporates two core modules within each optimization loop: firstly, a specialized LLM-based diagnostic module, calibrated through our novel training strategy, accurately identifies specific defects in the user profile. Subsequently, a generalized LLM-based treatment module analyzes the diagnosed defect and generates targeted suggestions to refine the profile. Furthermore, unlike existing LLM-based user simulators that are limited to single-round interactions, we are the first to integrate DGDPO with sequential recommenders, enabling a bidirectional evolution where user profiles and recommendation strategies adapt to each other over multi-round interactions. Extensive experiments conducted on three real-world datasets demonstrate the effectiveness of our proposed framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02695v1">EvoRoute: Experience-Driven Self-Routing LLM Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Complex agentic AI systems, powered by a coordinated ensemble of Large Language Models (LLMs), tool and memory modules, have demonstrated remarkable capabilities on intricate, multi-turn tasks. However, this success is shadowed by prohibitive economic costs and severe latency, exposing a critical, yet underexplored, trade-off. We formalize this challenge as the \textbf{Agent System Trilemma}: the inherent tension among achieving state-of-the-art performance, minimizing monetary cost, and ensuring rapid task completion. To dismantle this trilemma, we introduce EvoRoute, a self-evolving model routing paradigm that transcends static, pre-defined model assignments. Leveraging an ever-expanding knowledge base of prior experience, EvoRoute dynamically selects Pareto-optimal LLM backbones at each step, balancing accuracy, efficiency, and resource use, while continually refining its own selection policy through environment feedback. Experiments on challenging agentic benchmarks such as GAIA and BrowseComp+ demonstrate that EvoRoute, when integrated into off-the-shelf agentic systems, not only sustains or enhances system performance but also reduces execution cost by up to $80\%$ and latency by over $70\%$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01310v2">Making MoE-based LLM Inference Resilient with Tarragon</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Mixture-of-Experts (MoE) models are increasingly used to serve LLMs at scale, but failures become common as deployment scale grows. Existing systems exhibit poor failure resilience: even a single worker failure triggers a coarse-grained, service-wide restart, discarding accumulated progress and halting the entire inference pipeline during recovery--an approach clearly ill-suited for latency-sensitive, LLM services. We present Tarragon, a resilient MoE inference framework that confines the failures impact to individual workers while allowing the rest of the pipeline to continue making forward progress. Tarragon exploits the natural separation between the attention and expert computation in MoE-based transformers, treating attention workers (AWs) and expert workers (EWs) as distinct failure domains. Tarragon introduces a reconfigurable datapath to mask failures by rerouting requests to healthy workers. On top of this datapath, Tarragon implements a self-healing mechanism that relaxes the tightly synchronized execution of existing MoE frameworks. For stateful AWs, Tarragon performs asynchronous, incremental KV cache checkpointing with per-request restoration, and for stateless EWs, it leverages residual GPU memory to deploy shadow experts. These together keep recovery cost and recomputation overhead extremely low. Our evaluation shows that, compared to state-of-the-art MegaScale-Infer, Tarragon reduces failure-induced stalls by 160-213x (from ~64 s down to 0.3-0.4 s) while preserving performance when no failures occur.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.00500v2">Pro2Guard: Proactive Runtime Enforcement of LLM Agent Safety via Probabilistic Model Checking</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents demonstrate strong autonomy, but their stochastic behavior introduces unpredictable safety risks. Existing rule-based enforcement systems, such as AgentSpec, are reactive, intervening only when unsafe behavior is imminent or has occurred, lacking foresight for long-horizon dependencies. To overcome these limitations, we present a proactive runtime enforcement framework for LLM agents. The framework abstracts agent behaviors into symbolic states and learns a Discrete-Time Markov Chain (DTMC) from execution traces. At runtime, it predicts the probability of leading to undesired behaviors and intervenes before violations occur when the estimated risk exceeds a user-defined threshold. Designed to provide PAC-correctness guarantee, the framework achieves statistically reliable enforcement of agent safety. We evaluate the framework across two safety-critical domains: autonomous vehicles and embodied agents. It proactively enforces safety and maintains high task performance, outperforming existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02680v1">Adversarial Contrastive Learning for LLM Quantization Attacks</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 14 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Model quantization is critical for deploying large language models (LLMs) on resource-constrained hardware, yet recent work has revealed severe security risks that benign LLMs in full precision may exhibit malicious behaviors after quantization. In this paper, we propose Adversarial Contrastive Learning (ACL), a novel gradient-based quantization attack that achieves superior attack effectiveness by explicitly maximizing the gap between benign and harmful responses probabilities. ACL formulates the attack objective as a triplet-based contrastive loss, and integrates it with a projected gradient descent two-stage distributed fine-tuning strategy to ensure stable and efficient optimization. Extensive experiments demonstrate ACL's remarkable effectiveness, achieving attack success rates of 86.00% for over-refusal, 97.69% for jailbreak, and 92.40% for advertisement injection, substantially outperforming state-of-the-art methods by up to 44.67%, 18.84%, and 50.80%, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02670v1">Multi-Turn Jailbreaking of Aligned LLMs via Lexical Anchor Tree Search</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Most jailbreak methods achieve high attack success rates (ASR) but require attacker LLMs to craft adversarial queries and/or demand high query budgets. These resource limitations make jailbreaking expensive, and the queries generated by attacker LLMs often consist of non-interpretable random prefixes. This paper introduces Lexical Anchor Tree Search (), addressing these limitations through an attacker-LLM-free method that operates purely via lexical anchor injection. LATS reformulates jailbreaking as a breadth-first tree search over multi-turn dialogues, where each node incrementally injects missing content words from the attack goal into benign prompts. Evaluations on AdvBench and HarmBench demonstrate that LATS achieves 97-100% ASR on latest GPT, Claude, and Llama models with an average of only ~6.4 queries, compared to 20+ queries required by other methods. These results highlight conversational structure as a potent and under-protected attack surface, while demonstrating superior query efficiency in an era where high ASR is readily achievable. Our code will be released to support reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02663v1">When Do Tools and Planning Help LLMs Think? A Cost- and Latency-Aware Benchmark</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Modern large language models (LLMs) increasingly rely on inference-time planning and external tools to improve reasoning. We benchmark this behavior on two real-world settings: event-centric question answering over graph-structured knowledge (Event-QA) and persuasive response generation in Reddit ChangeMyView (CMV). Using LangChain and LangGraph, we compare a one-shot baseline against a plan-execute-replan agent equipped with task-specific tools (DBpedia SPARQL/lookup/schema exploration, Wikipedia-focused retrieval, and topical web search). We evaluate on 60 examples each from Event-QA and CMV (3 splits of 20), and report both mean end-to-end latency and per-example token cost estimates. We evaluate GPT-4o and GPT-4o-mini under identical workflows and report accuracy and end-to-end latency. On Event-QA, the best tool-augmented configuration improves accuracy (e.g., 47.5\% $\rightarrow$ 67.5\% for GPT-4o) while increasing latency by orders of magnitude ($\sim$8s $\rightarrow$ $\sim$317s per example). On CMV, one-shot prompting is strongest (e.g., GPT-4o-mini achieves 75\% at $\sim$6s), and planning+search increases latency substantially without consistent gains. However, complex multi-tool orchestration exposes failure modes where the smaller model degrades. Overall, the findings highlight the need for task-specific, cost-aware choices of both model size and agent/tooling complexity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02632v1">TAAF: A Trace Abstraction and Analysis Framework Synergizing Knowledge Graphs and LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 Accepted to ICSE 2026. DOI 10.1145/3744916.3787832
    </div>
    <details class="paper-abstract">
      Execution traces are a critical source of information for understanding, debugging, and optimizing complex software systems. However, traces from OS kernels or large-scale applications like Chrome or MySQL are massive and difficult to analyze. Existing tools rely on predefined analyses, and custom insights often require writing domain-specific scripts, which is an error-prone and time-consuming task. This paper introduces TAAF (Trace Abstraction and Analysis Framework), a novel approach that combines time-indexing, knowledge graphs (KGs), and large language models (LLMs) to transform raw trace data into actionable insights. TAAF constructs a time-indexed KG from trace events to capture relationships among entities such as threads, CPUs, and system resources. An LLM then interprets query-specific subgraphs to answer natural-language questions, reducing the need for manual inspection and deep system expertise. To evaluate TAAF, we introduce TraceQA-100, a benchmark of 100 questions grounded in real kernel traces. Experiments across three LLMs and multiple temporal settings show that TAAF improves answer accuracy by up to 31.2%, particularly in multi-hop and causal reasoning tasks. We further analyze where graph-grounded reasoning helps and where limitations remain, offering a foundation for next-generation trace analysis tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02627v1">Improved Evidence Extraction for Document Inconsistency Detection with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 10 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are becoming useful in many domains due to their impressive abilities that arise from large training datasets and large model sizes. However, research on LLM-based approaches to document inconsistency detection is relatively limited. There are two key aspects of document inconsistency detection: (i) classification of whether there exists any inconsistency, and (ii) providing evidence of the inconsistent sentences. We focus on the latter, and introduce new comprehensive evidence-extraction metrics and a redact-and-retry framework with constrained filtering that substantially improves LLM-based document inconsistency detection over direct prompting. We back our claims with promising experimental results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02624v1">LAsset: An LLM-assisted Security Asset Identification Framework for System-on-Chip (SoC) Verification</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 6 pages
    </div>
    <details class="paper-abstract">
      The growing complexity of modern system-on-chip (SoC) and IP designs is making security assurance difficult day by day. One of the fundamental steps in the pre-silicon security verification of a hardware design is the identification of security assets, as it substantially influences downstream security verification tasks, such as threat modeling, security property generation, and vulnerability detection. Traditionally, assets are determined manually by security experts, requiring significant time and expertise. To address this challenge, we present LAsset, a novel automated framework that leverages large language models (LLMs) to identify security assets from both hardware design specifications and register-transfer level (RTL) descriptions. The framework performs structural and semantic analysis to identify intra-module primary and secondary assets and derives inter-module relationships to systematically characterize security dependencies at the design level. Experimental results show that the proposed framework achieves high classification accuracy, reaching up to 90% recall rate in SoC design, and 93% recall rate in IP designs. This automation in asset identification significantly reduces manual overhead and supports a scalable path forward for secure hardware development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15674v2">Activation Oracles: Training and Evaluating LLMs as General-Purpose Activation Explainers</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 36 pages
    </div>
    <details class="paper-abstract">
      Large language model (LLM) activations are notoriously difficult to understand, with most existing techniques using complex, specialized methods for interpreting them. Recent work has proposed a simpler approach known as LatentQA: training LLMs to directly accept LLM activations as inputs and answer arbitrary questions about them in natural language. However, prior work has focused on narrow task settings for both training and evaluation. In this paper, we instead take a generalist perspective. We evaluate LatentQA-trained models, which we call Activation Oracles (AOs), in far out-of-distribution settings and examine how performance scales with training data diversity. We find that AOs can recover information fine-tuned into a model (e.g., biographical knowledge or malign propensities) that does not appear in the input text, despite never being trained with activations from a fine-tuned model. Our main evaluations are four downstream tasks where we can compare to prior white- and black-box techniques. We find that even narrowly-trained LatentQA models can generalize well, and that adding additional training datasets (such as classification tasks and a self-supervised context prediction task) yields consistent further improvements. Our best AOs match or exceed white-box baselines on all four tasks and the best overall baseline on 3 of 4. These results suggest that diversified training to answer natural-language queries imparts a general capability to verbalize information about LLM activations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03464v1">Prompting Underestimates LLM Capability for Time Series Classification</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 8 pages + Appendix and References, 9 figures
    </div>
    <details class="paper-abstract">
      Prompt-based evaluations suggest that large language models (LLMs) perform poorly on time series classification, raising doubts about whether they encode meaningful temporal structure. We show that this conclusion reflects limitations of prompt-based generation rather than the model's representational capacity by directly comparing prompt outputs with linear probes over the same internal representations. While zero-shot prompting performs near chance, linear probes improve average F1 from 0.15-0.26 to 0.61-0.67, often matching or exceeding specialized time series models. Layer-wise analyses further show that class-discriminative time series information emerges in early transformer layers and is amplified by visual and multimodal inputs. Together, these results demonstrate a systematic mismatch between what LLMs internally represent and what prompt-based evaluation reveals, leading current evaluations to underestimate their time series understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.23842v4">Fair Document Valuation in LLM Summaries via Shapley Values</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in systems that retrieve and summarize content from multiple sources, such as search engines and AI assistants. While these systems enhance user experience through coherent summaries, they obscure the individual contributions of original content creators, raising concerns about credit attribution and compensation. We address the challenge of valuing individual documents used in LLM-generated summaries by proposing a Shapley value-based framework for fair document valuation. Although theoretically appealing, exact Shapley value computation is prohibitively expensive at scale. To improve efficiency, we develop Cluster Shapley, a simple approximation algorithm that leverages semantic similarity among documents to reduce computation while maintaining attribution accuracy. Using Amazon product review data, we empirically show that off-the-shelf Shapley approximations, such as Monte Carlo sampling and Kernel SHAP, perform suboptimally in LLM settings, whereas Cluster Shapley substantially improves the efficiency-accuracy frontier. Moreover, simple attribution rules (e.g., equal or relevance-based allocation), though computationally cheap, lead to highly unfair outcomes. Together, our findings highlight the potential of structure-aware Shapley approximations tailored to LLM summarization and offer guidance for platforms seeking scalable and fair content attribution mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03444v1">Grading Scale Impact on LLM-as-a-Judge: Human-LLM Alignment Is Highest on 0-5 Grading Scale</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as automated evaluators, yet prior works demonstrate that these LLM judges often lack consistency in scoring when the prompt is altered. However, the effect of the grading scale itself remains underexplored. We study the LLM-as-a-judge problem by comparing two kinds of raters: humans and LLMs. We collect ratings from both groups on three scales and across six benchmarks that include objective, open-ended subjective, and mixed tasks. Using intraclass correlation coefficients (ICC) to measure absolute agreement, we find that LLM judgments are not perfectly consistent across scales on subjective benchmarks, and that the choice of scale substantially shifts human-LLM agreement, even when within-group panel reliability is high. Aggregated over tasks, the grading scale of 0-5 yields the strongest human-LLM alignment. We further demonstrate that pooled reliability can mask benchmark heterogeneity and reveal systematic subgroup differences in alignment across gender groups, strengthening the importance of scale design and sub-level diagnostics as essential components of LLM-as-a-judge protocols.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03420v1">Jailbreaking LLMs Without Gradients or Priors: Effective and Transferable Attacks</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are increasingly deployed in safety-critical domains, rigorously evaluating their robustness against adversarial jailbreaks is essential. However, current safety evaluations often overestimate robustness because existing automated attacks are limited by restrictive assumptions. They typically rely on handcrafted priors or require white-box access for gradient propagation. We challenge these constraints by demonstrating that token-level iterative optimization can succeed without gradients or priors. We introduce RAILS (RAndom Iterative Local Search), a framework that operates solely on model logits. RAILS matches the effectiveness of gradient-based methods through two key innovations: a novel auto-regressive loss that enforces exact prefix matching, and a history-based selection strategy that bridges the gap between the proxy optimization objective and the true attack success rate. Crucially, by eliminating gradient dependency, RAILS enables cross-tokenizer ensemble attacks. This allows for the discovery of shared adversarial patterns that generalize across disjoint vocabularies, significantly enhancing transferability to closed-source systems. Empirically, RAILS achieves near 100% success rates on multiple open-source models and high black-box attack transferability to closed-source systems like GPT and Gemini.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03401v1">Rendering Data Unlearnable by Exploiting LLM Alignment Mechanisms</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly trained on massive, heterogeneous text corpora, raising serious concerns about the unauthorised use of proprietary or personal data during model training. In this work, we address the problem of data protection against unwanted model learning in a realistic black-box setting. We propose Disclaimer Injection, a novel data-level defence that renders text unlearnable to LLMs. Rather than relying on model-side controls or explicit data removal, our approach exploits the models' own alignment mechanisms: by injecting carefully designed alignment-triggering disclaimers to prevent effective learning. Through layer-wise analysis, we find that fine-tuning on such protected data induces persistent activation of alignment-related layers, causing alignment constraints to override task learning even on common inputs. Consequently, models trained on such data exhibit substantial and systematic performance degradation compared to standard fine-tuning. Our results identify alignment behaviour as a previously unexplored lever for data protection and, to our knowledge, present the first practical method for restricting data learnability at LLM scale without requiring access to or modification of the training pipeline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.20650v3">FinTagging: Benchmarking LLMs for Extracting and Structuring Financial Information</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Accurate interpretation of numerical data in financial reports is critical for markets and regulators. Although XBRL (eXtensible Business Reporting Language) provides a standard for tagging financial figures, mapping thousands of facts to over ten thousand US-GAAP concepts remains costly and error-prone. Existing benchmarks oversimplify this task as flat, single-step classification over small subsets of concepts, ignoring the hierarchical semantics of the taxonomy and the structured nature of financial documents. As a result, these benchmarks fail to evaluate Large Language Models (LLMs) under realistic reporting conditions. To bridge this gap, we introduce FinTagging, the first comprehensive benchmark for structure-aware and full-scope XBRL tagging. We decompose the complex tagging process into two subtasks: (1) FinNI (Financial Numeric Identification), which extracts entities and types from heterogeneous contexts such as text and tables; and (2) FinCL (Financial Concept Linking), which maps extracted entities to the full US-GAAP taxonomy. This two-stage formulation enables a fair assessment of LLM capabilities in numerical reasoning and taxonomy alignment. Evaluating diverse LLMs in zero-shot settings shows that while models generalize well in extraction, they struggle with fine-grained concept linking, revealing important limitations in domain-specific, structure-aware reasoning. Code is available on GitHub, and datasets are available on Hugging Face.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03385v1">SIGMA: Scalable Spectral Insights for LLM Collapse</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      The rapid adoption of synthetic data for training Large Language Models (LLMs) has introduced the technical challenge of "model collapse"-a degenerative process where recursive training on model-generated content leads to a contraction of distributional variance and representational quality. While the phenomenology of collapse is increasingly evident, rigorous methods to quantify and predict its onset in high-dimensional spaces remain elusive. In this paper, we introduce SIGMA (Spectral Inequalities for Gram Matrix Analysis), a unified framework that benchmarks model collapse through the spectral lens of the embedding Gram matrix. By deriving and utilizing deterministic and stochastic bounds on the matrix's spectrum, SIGMA provides a mathematically grounded metric to track the contraction of the representation space. Crucially, our stochastic formulation enables scalable estimation of these bounds, making the framework applicable to large-scale foundation models where full eigendecomposition is intractable. We demonstrate that SIGMA effectively captures the transition towards degenerate states, offering both theoretical insights into the mechanics of collapse and a practical, scalable tool for monitoring the health of recursive training pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.06485v3">Task Matters: Knowledge Requirements Shape LLM Responses to Context-Memory Conflict</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 Major revision
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) draw on both contextual information and parametric memory, yet these sources can conflict. Prior studies have largely examined this issue in contextual question answering, implicitly assuming that tasks should rely on the provided context, leaving unclear how LLMs behave when tasks require different types and degrees of knowledge utilization. We address this gap with a model-agnostic diagnostic framework that holds underlying knowledge constant while introducing controlled conflicts across tasks with varying knowledge demands. Experiments on representative open-source LLMs show that performance degradation under conflict is driven by both task-specific knowledge reliance and conflict plausibility; that strategies such as rationales or context reiteration increase context reliance, helping context-only tasks but harming those requiring parametric knowledge; and that these effects bias model-based evaluation, calling into question the reliability of LLMs as judges. Overall, our findings reveal that context-memory conflict is inherently task-dependent and motivate task-aware approaches to balancing context and memory in LLM deployment and evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03359v1">Enhancing LLM Instruction Following: An Evaluation-Driven Multi-Agentic Workflow for Prompt Instructions Optimization</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often generate substantively relevant content but fail to adhere to formal constraints, leading to outputs that are conceptually correct but procedurally flawed. Traditional prompt refinement approaches focus on rephrasing the description of the primary task an LLM has to perform, neglecting the granular constraints that function as acceptance criteria for its response. We propose a novel multi-agentic workflow that decouples optimization of the primary task description from its constraints, using quantitative scores as feedback to iteratively rewrite and improve them. Our evaluation demonstrates this method produces revised prompts that yield significantly higher compliance scores from models like Llama 3.1 8B and Mixtral-8x 7B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03335v1">Digital Red Queen: Adversarial Program Evolution in Core War with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 14 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly being used to evolve solutions to problems in many domains, in a process inspired by biological evolution. However, unlike biological evolution, most LLM-evolution frameworks are formulated as static optimization problems, overlooking the open-ended adversarial dynamics that characterize real-world evolutionary processes. Here, we study Digital Red Queen (DRQ), a simple self-play algorithm that embraces these so-called "Red Queen" dynamics via continual adaptation to a changing objective. DRQ uses an LLM to evolve assembly-like programs, called warriors, which compete against each other for control of a virtual machine in the game of Core War, a Turing-complete environment studied in artificial life and connected to cybersecurity. In each round of DRQ, the model evolves a new warrior to defeat all previous ones, producing a sequence of adapted warriors. Over many rounds, we observe that warriors become increasingly general (relative to a set of held-out human warriors). Interestingly, warriors also become less behaviorally diverse across independent runs, indicating a convergence pressure toward a general-purpose behavioral strategy, much like convergent evolution in nature. This result highlights a potential value of shifting from static objectives to dynamic Red Queen objectives. Our work positions Core War as a rich, controllable sandbox for studying adversarial adaptation in artificial systems and for evaluating LLM-based evolution methods. More broadly, the simplicity and effectiveness of DRQ suggest that similarly minimal self-play approaches could prove useful in other more practical multi-agent adversarial domains, like real-world cybersecurity or combating drug resistance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03328v1">LLM-Enabled Multi-Agent Systems: Empirical Evaluation and Insights into Emerging Design Patterns & Paradigms</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      This paper formalises the literature on emerging design patterns and paradigms for Large Language Model (LLM)-enabled multi-agent systems (MAS), evaluating their practical utility across various domains. We define key architectural components, including agent orchestration, communication mechanisms, and control-flow strategies, and demonstrate how these enable rapid development of modular, domain-adaptive solutions. Three real-world case studies are tested in controlled, containerised pilots in telecommunications security, national heritage asset management, and utilities customer service automation. Initial empirical results show that, for these case studies, prototypes were delivered within two weeks and pilot-ready solutions within one month, suggesting reduced development overhead compared to conventional approaches and improved user accessibility. However, findings also reinforce limitations documented in the literature, including variability in LLM behaviour that leads to challenges in transitioning from prototype to production maturity. We conclude by outlining critical research directions for improving reliability, scalability, and governance in MAS architectures and the further work needed to mature MAS design patterns to mitigate the inherent challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03320v1">Ratio-Variance Regularized Policy Optimization for Efficient LLM Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      On-policy reinforcement learning (RL), particularly Proximal Policy Optimization (PPO) and Group Relative Policy Optimization (GRPO), has become the dominant paradigm for fine-tuning large language models (LLMs). While policy ratio clipping stabilizes training, this heuristic hard constraint incurs a fundamental cost: it indiscriminately truncates gradients from high-return yet high-divergence actions, suppressing rare but highly informative "eureka moments" in complex reasoning. Moreover, once data becomes slightly stale, hard clipping renders it unusable, leading to severe sample inefficiency. In this work, we revisit the trust-region objective in policy optimization and show that explicitly constraining the \emph{variance (second central moment) of the policy ratio} provides a principled and smooth relaxation of hard clipping. This distributional constraint stabilizes policy updates while preserving gradient signals from valuable trajectories. Building on this insight, we propose $R^2VPO$ (Ratio-Variance Regularized Policy Optimization), a novel primal-dual framework that supports stable on-policy learning and enables principled off-policy data reuse by dynamically reweighting stale samples rather than discarding them. We extensively evaluate $R^2VPO$ on fine-tuning state-of-the-art LLMs, including DeepSeek-Distill-Qwen-1.5B and the openPangu-Embedded series (1B and 7B), across challenging mathematical reasoning benchmarks. Experimental results show that $R^2VPO$ consistently achieves superior asymptotic performance, with average relative gains of up to 17% over strong clipping-based baselines, while requiring approximately 50% fewer rollouts to reach convergence. These findings establish ratio-variance control as a promising direction for improving both stability and data efficiency in RL-based LLM alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03315v1">Why LLMs Aren't Scientists Yet: Lessons from Four Autonomous Research Attempts</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      We report a case study of four end-to-end attempts to autonomously generate ML research papers using a pipeline of six LLM agents mapped to stages of the scientific workflow. Of these four, three attempts failed during implementation or evaluation. One completed the pipeline and was accepted to Agents4Science 2025, an experimental inaugural venue that required AI systems as first authors, passing both human and multi-AI review. From these attempts, we document six recurring failure modes: bias toward training data defaults, implementation drift under execution pressure, memory and context degradation across long-horizon tasks, overexcitement that declares success despite obvious failures, insufficient domain intelligence, and weak scientific taste in experimental design. We conclude by discussing four design principles for more robust AI-scientist systems, implications for autonomous scientific discovery, and we release all prompts, artifacts, and outputs at https://github.com/Lossfunk/ai-scientist-artefacts-v1
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03251v1">NavAI: A Generalizable LLM Framework for Navigation Tasks in Virtual Reality Environments</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Navigation is one of the fundamental tasks for automated exploration in Virtual Reality (VR). Existing technologies primarily focus on path optimization in 360-degree image datasets and 3D simulators, which cannot be directly applied to immersive VR environments. To address this gap, we present NavAI, a generalizable large language model (LLM)-based navigation framework that supports both basic actions and complex goal-directed tasks across diverse VR applications. We evaluate NavAI in three distinct VR environments through goal-oriented and exploratory tasks. Results show that it achieves high accuracy, with an 89% success rate in goal-oriented tasks. Our analysis also highlights current limitations of relying entirely on LLMs, particularly in scenarios that require dynamic goal assessment. Finally, we discuss the limitations observed during the experiments and offer insights for future research directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.05665v3">Characterizing the Robustness of Black-Box LLM Planners Under Perturbed Observations with Adaptive Stress Testing</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 30 pages, 24 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently demonstrated success in decision-making tasks including planning, control, and prediction, but their tendency to hallucinate unsafe and undesired outputs poses risks. This unwanted behavior is further exacerbated in environments where sensors are noisy or unreliable. Characterizing the behavior of LLM planners to varied observations is necessary to proactively avoid failures in safety-critical scenarios. We specifically investigate the response of LLMs along two different perturbation dimensions. Like prior works, one dimension generates semantically similar prompts with varied phrasing by randomizing order of details, modifying access to few-shot examples, etc. Unique to our work, the second dimension simulates access to varied sensors and noise to mimic raw sensor or detection algorithm failures. An initial case study in which perturbations are manually applied show that both dimensions lead LLMs to hallucinate in a multi-agent driving environment. However, manually covering the entire perturbation space for several scenarios is infeasible. As such, we propose a novel method for efficiently searching the space of prompt perturbations using adaptive stress testing (AST) with Monte-Carlo tree search (MCTS). Our AST formulation enables discovery of scenarios, sensor configurations, and prompt phrasing that cause language models to act with high uncertainty or even crash. By generating MCTS prompt perturbation trees across diverse scenarios, we show through extensive experiments that offline analyses can be used to proactively understand potential failures that may arise at runtime.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03248v1">STReasoner: Empowering LLMs for Spatio-Temporal Reasoning in Time Series via Spatial-Aware Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 preprint, we release our code publicly at https://github.com/LingFengGold/STReasoner
    </div>
    <details class="paper-abstract">
      Spatio-temporal reasoning in time series involves the explicit synthesis of temporal dynamics, spatial dependencies, and textual context. This capability is vital for high-stakes decision-making in systems such as traffic networks, power grids, and disease propagation. However, the field remains underdeveloped because most existing works prioritize predictive accuracy over reasoning. To address the gap, we introduce ST-Bench, a benchmark consisting of four core tasks, including etiological reasoning, entity identification, correlation reasoning, and in-context forecasting, developed via a network SDE-based multi-agent data synthesis pipeline. We then propose STReasoner, which empowers LLM to integrate time series, graph structure, and text for explicit reasoning. To promote spatially grounded logic, we introduce S-GRPO, a reinforcement learning algorithm that rewards performance gains specifically attributable to spatial information. Experiments show that STReasoner achieves average accuracy gains between 17% and 135% at only 0.004X the cost of proprietary models and generalizes robustly to real-world data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03205v1">UltraLogic: Enhancing LLM Reasoning through Large-Scale Data Synthesis and Bipolar Float Reward</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 19 pages, 6 figures, 7 tables
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have demonstrated significant potential in natural language processing , complex general-purpose reasoning requiring multi-step logic, planning, and verification remains a critical bottleneck. Although Reinforcement Learning with Verifiable Rewards (RLVR) has succeeded in specific domains , the field lacks large-scale, high-quality, and difficulty-calibrated data for general reasoning. To address this, we propose UltraLogic, a framework that decouples the logical core of a problem from its natural language expression through a Code-based Solving methodology to automate high-quality data production. The framework comprises hundreds of unique task types and an automated calibration pipeline across ten difficulty levels. Furthermore, to mitigate binary reward sparsity and the Non-negative Reward Trap, we introduce the Bipolar Float Reward (BFR) mechanism, utilizing graded penalties to effectively distinguish perfect responses from those with logical flaws. Our experiments demonstrate that task diversity is the primary driver for reasoning enhancement , and that BFR, combined with a difficulty matching strategy, significantly improves training efficiency, guiding models toward global logical optima.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03194v1">X-MuTeST: A Multilingual Benchmark for Explainable Hate Speech Detection and A Novel LLM-consulted Explanation Framework</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 Accepted in the proceedings of AAAI 2026
    </div>
    <details class="paper-abstract">
      Hate speech detection on social media faces challenges in both accuracy and explainability, especially for underexplored Indic languages. We propose a novel explainability-guided training framework, X-MuTeST (eXplainable Multilingual haTe Speech deTection), for hate speech detection that combines high-level semantic reasoning from large language models (LLMs) with traditional attention-enhancing techniques. We extend this research to Hindi and Telugu alongside English by providing benchmark human-annotated rationales for each word to justify the assigned class label. The X-MuTeST explainability method computes the difference between the prediction probabilities of the original text and those of unigrams, bigrams, and trigrams. Final explanations are computed as the union between LLM explanations and X-MuTeST explanations. We show that leveraging human rationales during training enhances both classification performance and explainability. Moreover, combining human rationales with our explainability method to refine the model attention yields further improvements. We evaluate explainability using Plausibility metrics such as Token-F1 and IOU-F1 and Faithfulness metrics such as Comprehensiveness and Sufficiency. By focusing on under-resourced languages, our work advances hate speech detection across diverse linguistic contexts. Our dataset includes token-level rationale annotations for 6,004 Hindi, 4,492 Telugu, and 6,334 English samples. Data and code are available on https://github.com/ziarehman30/X-MuTeST
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.02790v3">Leveraging the true depth of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      The remarkable capabilities of Large Language Models (LLMs) are overshadowed by their immense computational cost. While recent work has shown that many LLM layers can be reordered or even removed with minimal impact on accuracy, these insights have not been translated into significant inference speedups. To bridge this gap, we introduce a novel method that restructures the computational graph by grouping and evaluating consecutive layer pairs in parallel. This approach, requiring no retraining, yields a 1.19x throughput gain on Llama 2 7B while reducing the average benchmark accuracy by only 1.5\%. We demonstrate the practical value of this method for large-scale LLM deployment and show that some of the lost accuracy can be recovered with lightweight fine-tuning of the parallelized layers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03190v1">Maximizing Local Entropy Where It Matters: Prefix-Aware Localized LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Machine unlearning aims to forget sensitive knowledge from Large Language Models (LLMs) while maintaining general utility. However, existing approaches typically treat all tokens in a response indiscriminately and enforce uncertainty over the entire vocabulary. This global treatment results in unnecessary utility degradation and extends optimization to content-agnostic regions. To address these limitations, we propose PALU (Prefix-Aware Localized Unlearning), a framework driven by a local entropy maximization objective across both temporal and vocabulary dimensions. PALU reveals that (i) suppressing the sensitive prefix alone is sufficient to sever the causal generation link, and (ii) flattening only the top-$k$ logits is adequate to maximize uncertainty in the critical subspace. These findings allow PALU to avoid redundant optimization across the full vocabulary and parameter space while minimizing collateral damage to general model performance. Extensive experiments validate that PALU achieves superior forgetting efficacy and utility preservation compared to state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.15125v2">Iterative Topic Taxonomy Induction with LLMs: A Case Study of Electoral Advertising</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 Under-submission
    </div>
    <details class="paper-abstract">
      Social media platforms play a pivotal role in shaping political discourse, but analyzing their vast and rapidly evolving content remains a major challenge. We introduce an end-to-end framework for automatically inducing an interpretable topic taxonomy from unlabeled text corpora. By combining unsupervised clustering with prompt-based inference, our method leverages large language models (LLMs) to iteratively construct a taxonomy without requiring seed sets (predefined labels) or domain expertise. We validate the framework through a study of political advertising ahead of the 2024 U.S. presidential election. The induced taxonomy yields semantically rich topic labels and supports downstream analyses, including moral framing, in this setting. Results suggest that structured, iterative labeling yields more consistent and interpretable topic labels than existing approaches under human evaluation, and is practical for analyzing large-scale political advertising data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03178v1">DiffBench Meets DiffAgent: End-to-End LLM-Driven Diffusion Acceleration Code Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 Accepted to AAAI 2026
    </div>
    <details class="paper-abstract">
      Diffusion models have achieved remarkable success in image and video generation. However, their inherently multiple step inference process imposes substantial computational overhead, hindering real-world deployment. Accelerating diffusion models is therefore essential, yet determining how to combine multiple model acceleration techniques remains a significant challenge. To address this issue, we introduce a framework driven by large language models (LLMs) for automated acceleration code generation and evaluation. First, we present DiffBench, a comprehensive benchmark that implements a three stage automated evaluation pipeline across diverse diffusion architectures, optimization combinations and deployment scenarios. Second, we propose DiffAgent, an agent that generates optimal acceleration strategies and codes for arbitrary diffusion models. DiffAgent employs a closed-loop workflow in which a planning component and a debugging component iteratively refine the output of a code generation component, while a genetic algorithm extracts performance feedback from the execution environment to guide subsequent code refinements. We provide a detailed explanation of the DiffBench construction and the design principles underlying DiffAgent. Extensive experiments show that DiffBench offers a thorough evaluation of generated codes and that DiffAgent significantly outperforms existing LLMs in producing effective diffusion acceleration strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.06254v5">EviRerank: Adaptive Evidence Construction for Long-Document LLM Reranking</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Decoder-only LLM rerankers struggle with long documents: inference is costly and relevance signals can be diluted by irrelevant context. Motivated by an attention analysis indicating a consistent degradation trend when non-relevant text is appended, we propose EviRerank, an evidence-based long-document reranking framework for decoder-only LLMs. EviRerank (i) scores document blocks with a lightweight selector (BM25, bi-encoder, or cross-encoder), (ii) constructs a compact reranking context under a hard token cap by dynamically budgeting evidence blocks with Adaptive Evidence Budgeting (AEB) and adding a global summary cue via Summary Augmentation (SA), and (iii) reranks with a decoder-only LLM. Across TREC DL'19, DL'23, and MLDR-zh, EviRerank consistently outperforms full-document LLM reranking and strong block-selection baselines while substantially reducing the required input length. On TREC DL'19, EviRerank achieves 0.743 nDCG@10 and 0.307 MAP, establishing a new best result and improving over RankLLaMA (0.701/0.288) by +0.042 nDCG@10 (+6.0%) and +0.019 MAP (+6.6%).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03149v1">PersonaLedger: Generating Realistic Financial Transactions with Persona Conditioned LLMs and Rule Grounded Feedback</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Strict privacy regulations limit access to real transaction data, slowing open research in financial AI. Synthetic data can bridge this gap, but existing generators do not jointly achieve behavioral diversity and logical groundedness. Rule-driven simulators rely on hand-crafted workflows and shallow stochasticity, which miss the richness of human behavior. Learning-based generators such as GANs capture correlations yet often violate hard financial constraints and still require training on private data. We introduce PersonaLedger, a generation engine that uses a large language model conditioned on rich user personas to produce diverse transaction streams, coupled with an expert configurable programmatic engine that maintains correctness. The LLM and engine interact in a closed loop: after each event, the engine updates the user state, enforces financial rules, and returns a context aware "nextprompt" that guides the LLM toward feasible next actions. With this engine, we create a public dataset of 30 million transactions from 23,000 users and a benchmark suite with two tasks, illiquidity classification and identity theft segmentation. PersonaLedger offers a realistic, privacy preserving resource that supports rigorous evaluation of forecasting and anomaly detection models. PersonaLedger offers the community a rich, realistic, and privacy preserving resource -- complete with code, rules, and generation logs -- to accelerate innovation in financial AI and enable rigorous, reproducible evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03137v1">Accurate Table Question Answering with Accessible LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 accepted for publication in the Proceedings of the IEEE International Conference on Data Engineering (ICDE) 2026
    </div>
    <details class="paper-abstract">
      Given a table T in a database and a question Q in natural language, the table question answering (TQA) task aims to return an accurate answer to Q based on the content of T. Recent state-of-the-art solutions leverage large language models (LLMs) to obtain high-quality answers. However, most rely on proprietary, large-scale LLMs with costly API access, posing a significant financial barrier. This paper instead focuses on TQA with smaller, open-weight LLMs that can run on a desktop or laptop. This setting is challenging, as such LLMs typically have weaker capabilities than large proprietary models, leading to substantial performance degradation with existing methods. We observe that a key reason for this degradation is that prior approaches often require the LLM to solve a highly sophisticated task using long, complex prompts, which exceed the capabilities of small open-weight LLMs. Motivated by this observation, we present Orchestra, a multi-agent approach that unlocks the potential of accessible LLMs for high-quality, cost-effective TQA. Orchestra coordinates a group of LLM agents, each responsible for a relatively simple task, through a structured, layered workflow to solve complex TQA problems -- akin to an orchestra. By reducing the prompt complexity faced by each agent, Orchestra significantly improves output reliability. We implement Orchestra on top of AgentScope, an open-source multi-agent framework, and evaluate it on multiple TQA benchmarks using a wide range of open-weight LLMs. Experimental results show that Orchestra achieves strong performance even with small- to medium-sized models. For example, with Qwen2.5-14B, Orchestra reaches 72.1% accuracy on WikiTQ, approaching the best prior result of 75.3% achieved with GPT-4; with larger Qwen, Llama, or DeepSeek models, Orchestra outperforms all prior methods and establishes new state-of-the-art results across all benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03134v1">The Anatomy of Conversational Scams: A Topic-Based Red Teaming Analysis of Multi-Turn Interactions in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      As LLMs gain persuasive agentic capabilities through extended dialogues, they introduce novel risks in multi-turn conversational scams that single-turn safety evaluations fail to capture. We systematically study these risks using a controlled LLM-to-LLM simulation framework across multi-turn scam scenarios. Evaluating eight state-of-the-art models in English and Chinese, we analyze dialogue outcomes and qualitatively annotate attacker strategies, defensive responses, and failure modes. Results reveal that scam interactions follow recurrent escalation patterns, while defenses employ verification and delay mechanisms. Furthermore, interactional failures frequently stem from safety guardrail activation and role instability. Our findings highlight multi-turn interactional safety as a critical, distinct dimension of LLM behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.24045v2">Agent.xpu: Efficient Scheduling of Agentic LLM Workloads on Heterogeneous SoC</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Personal LLM agents increasingly combine foreground reactive interactions with background proactive monitoring, forming long-lived, stateful LLM flows that interleave prefill and token-by-token decode. While modern heterogeneous SoCs integrate CPUs, iGPUs, and NPUs to support on-device intelligence, existing LLM engines assume static, single-shot inference and lack mechanisms for flow-level concurrency, prioritization, and efficient accelerator coordination. As a result, commodity SoCs remain poorly matched to the dynamic, mixed-criticality execution patterns of personal agents. This paper presents Agent$.$xpu, the first LLM engine that orchestrates concurrent reactive and proactive LLM flows on commodity SoCs. Extensive profiling uncovers unique SoC characteristics of operator-accelerator affinity, asymmetric DDR contention, and stage-divergent batching behaviors distinct from cloud-serving assumptions. Agent$.$xpu introduces three key techniques: a heterogeneous execution graph (HEG) capturing NPU/iGPU affinity and elastic operator binding; flow-aware NPU-iGPU coordination with stage elasticity, decoupling prefill and decode to reduce bandwidth contention and enforce priorities; and fine-grained preemption with slack-aware piggybacking to guarantee reactive responsiveness without starving proactive work. Across realistic personal-agent workloads, Agent$.$xpu delivers 1.2-4.9$\times$ proactive throughput and reduces reactive latency by at least 91%, compared with both industrial iGPU-only serving engine and NPU-iGPU static inference with optimal tensor-partitioning schemes. Agent$.$xpu also minimizes energy consumption and graphics interference via controlled iGPU usage.
    </details>
</div>
