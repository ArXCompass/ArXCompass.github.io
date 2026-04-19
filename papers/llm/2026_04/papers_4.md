# llm - 2026_04

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- Part 4

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.24857v3">Between Help and Harm: An Evaluation of Mental Health Crisis Handling by LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Accepted for publication in JMIR Mental Health. DOI: 10.2196/88435
    </div>
    <details class="paper-abstract">
      Large language model-powered chatbots have transformed how people seek information, especially in high-stakes contexts like mental health. Despite their support capabilities, safe detection and response to crises such as suicidal ideation and self-harm are still unclear, hindered by the lack of unified crisis taxonomies and clinical evaluation standards. We address this by creating: (1) a taxonomy of six crisis categories; (2) a dataset of over 2,000 inputs from 12 mental health datasets, classified into these categories; and (3) a clinical response assessment protocol. We also use LLMs to identify crisis inputs and audit five models for response safety and appropriateness. First, we built a clinical-informed crisis taxonomy and evaluation protocol. Next, we curated 2,252 relevant examples from over 239,000 user inputs, then tested three LLMs for automatic classification. In addition, we evaluated five models for the appropriateness of their responses to a user's crisis, graded on a 5-point Likert scale from harmful (1) to appropriate (5). While some models respond reliably to explicit crises, risks still exist. Many outputs, especially in self-harm and suicidal categories, are inappropriate or unsafe. Different models perform variably; some, like gpt-5-nano and deepseek-v3.2-exp, have low harm rates, but others, such as gpt-4o-mini and grok-4-fast, generate more unsafe responses. All models struggle with indirect signals, default replies, and context misalignment. These results highlight the urgent need for better safeguards, crisis detection, and context-aware responses in LLMs. They also show that alignment and safety practices, beyond scale, are crucial for reliable crisis support. Our taxonomy, datasets, and evaluation methods support ongoing AI mental health research, aiming to reduce harm and protect vulnerable users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24933v2">ADOPT: Adaptive Dependency-Guided Joint Prompt Optimization for Multi-Step LLM Pipelines</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Multi-step LLM pipelines can solve complex tasks, but jointly optimizing prompts across steps remains challenging due to missing step-level supervision and inter-step dependency. We propose ADOPT, an adaptive dependency-guided joint prompt optimization framework for multi-step LLM pipelines. ADOPT analyzes the dependency between each LLM step and the final output, constructs a global textual gradient from final-task errors, and decomposes it into step-level local textual gradients, providing more precise optimization signals for local prompt updates. It further decouples signal estimation from prompt updating, enabling flexible integration of single-prompt optimizers, and uses a Shapley-based strategy to adaptively allocate optimization resources to high-impact steps. Experiments on real-world datasets and structurally diverse pipelines demonstrate that ADOPT is effective and robust, consistently outperforming strong prompt optimization baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06970v1">Scheduling the Unschedulable: Taming Black-Box LLM Inference at Scale</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 10 pages, 8 figures. Code and reproduction artifacts available upon request
    </div>
    <details class="paper-abstract">
      When output token counts can be predicted at submission time (Gan et al., 2026), client-side scheduling against a black-box LLM API becomes semi-clairvoyant: decisions condition on coarse token priors even though the provider's internals remain hidden. We decompose this boundary problem into three separable concerns: allocation (inter-class share via adaptive DRR), ordering (intra-class sequencing with feasible-set scoring), and overload control (explicit admit/defer/reject on a cost ladder). An information ladder experiment shows that coarse magnitude priors -- not class labels alone -- are the practical threshold for useful client control; removing magnitude inflates short-request P95 by up to $5.8\times$ and degrades deadline satisfaction. Under balanced / high congestion the full stack achieves 100% completion, 100% deadline satisfaction, and useful goodput of $4.2 \pm 1.6$ SLO-meeting requests/s with short P95 within tens of milliseconds of quota-tiered isolation. A predictor-noise sweep confirms graceful degradation under up to 60% multiplicative error. Heavy-dominated regimes separate policies on completion, tail, and interpretable shedding. We further compare short-priority allocation (biased toward interactive traffic) with Fair Queuing (round-robin across classes): Fair Queuing achieves +32% short-request P90 improvement over FIFO with only +17% long-request overhead, versus Short-Priority's +27% / +116% trade-off -- demonstrating that the allocation layer accommodates different fairness objectives without changing the remaining stack. We contribute the three-layer client-side decomposition, controlled evaluation of joint metrics across regimes, allocation-policy alternatives, and overload-policy evidence linking cost-ladder shedding to the stated service objective.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06967v1">VulGD: A LLM-Powered Dynamic Open-Access Vulnerability Graph Database</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Software vulnerabilities continue to pose significant threats to modern information systems, requiring a timely and accurate risk assessment. Public repositories, such as the National Vulnerability Database and CVE details, are regularly updated, but predominantly utilize relational data models that lack native support for representing complex, interconnected structures. To address this, recent research has proposed graph-based vulnerability models. However, these systems often require complex setup procedures, lack real-time multi-source integration, and offer limited accessibility for direct data retrieval and analysis. We present VulGD, a dynamic open-access vulnerability graph database that continuously aggregates cybersecurity data from authoritative repositories. Designed for both expert and non-expert users, VulGD provides a unified web interface and a public API for interactive graph exploration and automated data access. Additionally, VulGD integrates embeddings from large language models (LLMs) to enrich vulnerability description representations, facilitating more accurate vulnerability risk assessment and threat prioritization. VulGD represents a practical and extensible platform for cybersecurity research and decision-making. The live system is publicly accessible at http://34.129.186.158/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06928v1">Leveraging LLMs and Heterogeneous Knowledge Graphs for Persona-Driven Session-Based Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Session-based recommendation systems (SBRS) aim to capture user's short-term intent from interaction sequences. However, the common assumption of anonymous sessions limits personalization, particularly under sparse or cold-start conditions. Recent advances in LLM-augmented recommendation have shown that LLMs can generate rich item representations, but modeling user personas with LLMs remains challenging due to anonymous sessions. In this work, we propose a persona-driven SBRS framework that explicitly models latent user personas inferred from a heterogeneous knowledge graph (KG) and integrates them into a data-driven recommendation pipeline.Our framework adopts a two-stage architecture consisting of personalized information extraction and personalized information utilization, inspired by recent chain-of-thought recommendation approaches. In the personalized information extraction stage, we construct a heterogeneous KG that integrates time-independent user-item, item-item, item-feature association, and metadata from DBpedia. We then learn latent user personas in an unsupervised manner using a Heterogeneous Deep Graph Infomax (HDGI) objective over a KG initialized with LLM-derived item embeddings. In the personalized information utilization stage, the learned persona representations together with LLM-derived item embeddings are incorporated into a modified architecture of data-driven SBRS to generate a candidate set of relevant items, followed by reranking using the base sequential model to emphasize short-term session intent. Unlike prior approaches that rely solely on sequence modeling or text-based user representations, our method grounds user persona modeling in structured relational signals derived from a KG. Experiments on Amazon Books and Amazon Movies & TV demonstrate that our approach consistently improves over sequential models with user embeddings derived using session history.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06906v1">The AI Skills Shift: Mapping Skill Obsolescence, Emergence, and Transition Pathways in the LLM Era</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 11 pages, 12 figures, 2 tables, 17 references. Code and data available at
    </div>
    <details class="paper-abstract">
      As Large Language Models reshape the global labor market, policymakers and workers need empirical data on which occupational skills may be most susceptible to automation. We present the Skill Automation Feasibility Index (SAFI), benchmarking four frontier LLMs -- LLaMA 3.3 70B, Mistral Large, Qwen 2.5 72B, and Gemini 2.5 Flash -- across 263 text-based tasks spanning all 35 skills in the U.S. Department of Labor's O*NET taxonomy (1,052 total model calls, 0% failure rate). Cross-referencing with real-world AI adoption data from the Anthropic Economic Index (756 occupations, 17,998 tasks), we propose an AI Impact Matrix -- an interpretive framework that positions skills along four quadrants: High Displacement Risk, Upskilling Required, AI-Augmented, and Lower Displacement Risk. Key findings: (1) Mathematics (SAFI: 73.2) and Programming (71.8) receive the highest automation feasibility scores; Active Listening (42.2) and Reading Comprehension (45.5) receive the lowest; (2) a "capability-demand inversion" where skills most demanded in AI-exposed jobs are those LLMs perform least well at in our benchmark; (3) 78.7% of observed AI interactions are augmentation, not automation; (4) all four models converge to similar skill profiles (3.6-point spread), suggesting that text-based automation feasibility may be more skill-dependent than model-dependent. SAFI measures LLM performance on text-based representations of skills, not full occupational execution. All data, code, and model responses are open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06898v1">Are LLMs Ready for Computer Science Education? A Cross-Domain, Cross-Lingual and Cognitive-Level Evaluation Using Professional Certification Exams</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 40 pages including Appendix
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly applied in computer science education for tasks such as tutoring, content generation, and code assessment. However, systematic evaluations aligned with formal curricula and certification standards remain limited. This study benchmarked four recent models, including GPT-5, DeepSeek-R1, Qwen-Plus, and Llama-3.3-70B-Instruct, using a dataset of 1,068 questions derived from six certification exams covering networking, office applications, and Java programming. We evaluated performance across language (Chinese vs. English), cognitive levels based on Bloom's Taxonomy, domain knowledge, confidence-accuracy alignment, and robustness to input masking. Results showed that GPT-5 performed best on English-language certifications, while Qwen-Plus performed better in Chinese contexts. DeepSeek-R1 achieved the most balanced cross-lingual performance, whereas Llama-3.3 showed clear limitations in higher-order reasoning and robustness. All models performed worse on more complex tasks. These findings provide empirical support for the integration of LLMs into computer science education and offer practical implications for curriculum design and assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06894v1">How Does LLM Help Regional CPI Forecast: An LLM-powered Deep Panel Modeling Framework</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Understanding regional Consumer Price Index (CPI) dynamics is essential for timely and effective economic policymaking. However, traditional modeling procedures typically rely only on parametric panel modeling with low-frequency and high-cost macroeconomic indicators, which often fail to capture rapid market fluctuations and lead to inaccurate predictions. To this end, we propose a residual-joint-modeling framework that integrates large language model (LLM) analyses and social media narratives via a new deep neural network based panel modeling. Specifically, we construct a large narrative corpus from a newly collected {\it Sina Weibo} dataset, and develop a prompt-based GPT model and a series of fine-tuned BERT models to generate high-frequency LLM-induced surrogates for regional CPI. A novel joint modeling strategy is then advocated to transfer the information from these surrogates to the target regional CPI data and hence empower CPI prediction. To solve the joint objectives, we further introduce a new deep panel learning procedure with region-wise homogeneity pursuit, which has its own significance in panel data analysis literature. In addition, conformal-based panel prediction intervals are provided to quantify the uncertainty of the LLM-powered prediction. The proposed approach significantly reduces short-term forecasting errors and more effectively captures abrupt inflationary shifts compared to traditional econometric models. While demonstrated for regional CPI forecasting, the proposed framework is broadly applicable for incorporating insights from LLMs to enhance traditional statistical modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.17181v3">A Study of LLMs' Preferences for Libraries and Programming Languages</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 21 pages, 10 tables, 3 figures. Accepted to Findings of ACL 2026
    </div>
    <details class="paper-abstract">
      Despite the rapid progress of large language models (LLMs) in code generation, existing evaluations focus on functional correctness or syntactic validity, overlooking how LLMs make critical design choices such as which library or programming language to use. To fill this gap, we perform the first empirical study of LLMs' preferences for libraries and programming languages when generating code, covering eight diverse LLMs. We observe a strong tendency to overuse widely adopted libraries such as NumPy; in up to 45% of cases, this usage is not required and deviates from the ground-truth solutions. The LLMs we study also show a significant preference toward Python as their default language. For high-performance project initialisation tasks where Python is not the optimal language, it remains the dominant choice in 58% of cases, and Rust is not used once. These results highlight how LLMs prioritise familiarity and popularity over suitability and task-specific optimality; underscoring the need for targeted fine-tuning, data diversification, and evaluation benchmarks that explicitly measure language and library selection fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.16548v2">SemanticScanpath: Combining Gaze and Speech for Situated Human-Robot Interaction Using LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have substantially improved the conversational capabilities of social robots. Nevertheless, for an intuitive and fluent human-robot interaction, robots should be able to ground the conversation by relating ambiguous or underspecified spoken utterances to the current physical situation and to the intents expressed nonverbally by the user, such as through referential gaze. Here, we propose a representation that integrates speech and gaze to enable LLMs to achieve higher situated awareness and correctly resolve ambiguous requests. Our approach relies on a text-based semantic translation of the scanpath produced by the user, along with the verbal requests. It demonstrates LLMs' capabilities to reason about gaze behavior, robustly ignoring spurious glances or irrelevant objects. We validate the system across multiple tasks and two scenarios, showing its superior generality and accuracy compared to control conditions. We demonstrate an implementation on a robotic platform, closing the loop from request interpretation to execution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05172v2">ClawsBench: Evaluating Capability and Safety of LLM Productivity Agents in Simulated Workspaces</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 25 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents are increasingly deployed to automate productivity tasks (e.g., email, scheduling, document management), but evaluating them on live services is risky due to potentially irreversible changes. Existing benchmarks rely on simplified environments and fail to capture realistic, stateful, multi-service workflows. We introduce ClawsBench, a benchmark for evaluating and improving LLM agents in realistic productivity settings. It includes five high-fidelity mock services (Gmail, Slack, Google Calendar, Google Docs, Google Drive) with full state management and deterministic snapshot/restore, along with 44 structured tasks covering single-service, cross-service, and safety-critical scenarios. We decompose agent scaffolding into two independent levers (domain skills that inject API knowledge via progressive disclosure, and a meta prompt that coordinates behavior across services) and vary both to measure their separate and combined effects. Experiments across 6 models, 4 agent harnesses, and 33 conditions show that with full scaffolding, agents achieve task success rates of 39-64% but exhibit unsafe action rates of 7-33%. On OpenClaw, the top five models fall within a 10 percentage-point band on task success (53-63%), with unsafe action rates from 7% to 23% and no consistent ordering between the two metrics. We identify eight recurring patterns of unsafe behavior, including multi-step sandbox escalation and silent contract modification. We release the trajectories and future dataset at https://clawsbench.com.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06863v1">Digital Skin, Digital Bias: Uncovering Tone-Based Biases in LLMs and Emoji Embeddings</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Accepted at WWW'26
    </div>
    <details class="paper-abstract">
      Skin-toned emojis are crucial for fostering personal identity and social inclusion in online communication. As AI models, particularly Large Language Models (LLMs), increasingly mediate interactions on web platforms, the risk that these systems perpetuate societal biases through their representation of such symbols is a significant concern. This paper presents the first large-scale comparative study of bias in skin-toned emoji representations across two distinct model classes. We systematically evaluate dedicated emoji embedding models (emoji2vec, emoji-sw2v) against four modern LLMs (Llama, Gemma, Qwen, and Mistral). Our analysis first reveals a critical performance gap: while LLMs demonstrate robust support for skin tone modifiers, widely-used specialized emoji models exhibit severe deficiencies. More importantly, a multi-faceted investigation into semantic consistency, representational similarity, sentiment polarity, and core biases uncovers systemic disparities. We find evidence of skewed sentiment and inconsistent meanings associated with emojis across different skin tones, highlighting latent biases within these foundational models. Our findings underscore the urgent need for developers and platforms to audit and mitigate these representational harms, ensuring that AI's role on the web promotes genuine equity rather than reinforcing societal biases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06861v1">REAgent: Requirement-Driven LLM Agents for Software Issue Resolution</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Issue resolution aims to automatically generate patches from given issue descriptions and has attracted significant attention with the rapid advancement of large language models (LLMs). However, due to the complexity of software issues and codebases, LLM-generated patches often fail to resolve corresponding issues. Although various advanced techniques have been proposed with carefully designed tools and workflows, they typically treat issue descriptions as direct inputs and largely overlook their quality (e.g., missing critical context or containing ambiguous information), which hinders LLMs from accurate understanding and resolution. To address this limitation, we draw on principles from software requirements engineering and propose REAgent, a requirement-driven LLM agent framework that introduces issue-oriented requirements as structured task specifications to better guide patch generation. Specifically, REAgent automatically constructs structured and information-rich issue-oriented requirements, identifies low-quality requirements, and iteratively refines them to improve patch correctness. We conduct comprehensive experiments on three widely used benchmarks using two advanced LLMs, comparing against five representative or state-of-the-art baselines. The results demonstrate that REAgent consistently outperforms all baselines, achieving an average improvement of 17.40% in terms of the number of successfully-resolved issues (% Resolved).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07398v1">Breaking the Illusion of Identity in LLM Tooling</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 8 pages, 2 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) in research and development toolchains produce output that triggers attribution of agency and understanding -- a cognitive illusion that degrades verification behavior and trust calibration. No existing mitigation provides a systematic, deployable constraint set for output register. This paper proposes seven output-side rules, each targeting a documented linguistic mechanism, and validates them empirically. In 780 two-turn conversations (constrained vs. default register, 30 tasks, 13 replicates, 1560 API calls), anthropomorphic markers dropped from 1233 to 33 (>97% reduction, p < 0.001), outputs were 49% shorter by word count, and adapted AnthroScore confirmed the shift toward machine register (-1.94 vs. -0.96, p < 0.001). The rules are implemented as a configuration-file system prompt requiring no model modification; validation uses a single model (Claude Sonnet 4). Output quality under the constrained register was not evaluated. The mechanism is extensible to other domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06846v1">MedDialBench: Benchmarking LLM Diagnostic Robustness under Parametric Adversarial Patient Behaviors</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 9 pages, 4 figures, 9 tables. Preprint
    </div>
    <details class="paper-abstract">
      Interactive medical dialogue benchmarks have shown that LLM diagnostic accuracy degrades significantly when interacting with non-cooperative patients, yet existing approaches either apply adversarial behaviors without graded severity or case-specific grounding, or reduce patient non-cooperation to a single ungraded axis, and none analyze cross-dimension interactions. We introduce MedDialBench, a benchmark enabling controlled, dose-response characterization of how individual patient behavior dimensions affect LLM diagnostic robustness. It decomposes patient behavior into five dimensions -- Logic Consistency, Health Cognition, Expression Style, Disclosure, and Attitude -- each with graded severity levels and case-specific behavioral scripts. This controlled factorial design enables graded sensitivity analysis, dose-response profiling, and cross-dimension interaction detection. Evaluating five frontier LLMs across 7,225 dialogues (85 cases x 17 configurations x 5 models), we find a fundamental asymmetry: information pollution (fabricating symptoms) produces 1.7-3.4x larger accuracy drops than information deficit (withholding information), and fabricating is the only configuration achieving statistical significance across all five models (McNemar p < 0.05). Among six dimension combinations, fabricating is the sole driver of super-additive interaction: all three fabricating-involving pairs produce O/E ratios of 0.70-0.81 (35-44% of eligible cases fail under the combination despite succeeding under each dimension alone), while all non-fabricating pairs show purely additive effects (O/E ~ 1.0). Inquiry strategy moderates deficit but not pollution: exhaustive questioning recovers withheld information, but cannot compensate for fabricated inputs. Models exhibit distinct vulnerability profiles, with worst-case drops ranging from 38.8 to 54.1 percentage points.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.20340v4">Once4All: Skeleton-Guided SMT Solver Fuzzing with LLM-Synthesized Generators</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Accepted at ASPLOS 2026
    </div>
    <details class="paper-abstract">
      Satisfiability Modulo Theory (SMT) solvers are foundational to modern systems and programming languages research, providing the foundation for tasks like symbolic execution and automated verification. Because these solvers sit on the critical path, their correctness is essential, and high-quality test formulas are key to uncovering bugs. However, while prior testing techniques performed well on earlier solver versions, they struggle to keep pace with rapidly evolving features. Recent approaches based on Large Language Models (LLMs) show promise in exploring advanced solver capabilities, but two obstacles remain: nearly half of the generated formulas are syntactically invalid, and iterative interactions with LLMs introduce substantial computational overhead. In this study, we present Once4All, a novel LLM-assisted fuzzing framework that addresses both issues by shifting from direct formula generation to the synthesis of generators for reusable terms (i.e., logical expressions). Specifically, Once4All uses LLMs to (1) automatically extract context-free grammars (CFGs) for SMT theories, including solver-specific extensions, from documentation, and (2) synthesize composable Boolean term generators that adhere to these grammars. During fuzzing, Once4All populates structural skeletons derived from existing formulas with the terms iteratively produced by the LLM-synthesized generators. This design ensures syntactic validity while promoting semantic diversity. Notably, Once4All requires only one-time LLM interaction investment, dramatically reducing runtime cost. We evaluated Once4All on two leading SMT solvers: Z3 and cvc5. Our experiments show that Once4All has identified 43 confirmed bugs, 40 of which have already been fixed by developers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06834v1">On the Step Length Confounding in LLM Reasoning Data Selection</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Accepted by Findings of ACL 2026. 15 pages, 9 figures. Code: https://github.com/wangbing1416/ASLEC
    </div>
    <details class="paper-abstract">
      Large reasoning models have recently demonstrated strong performance on complex tasks that require long chain-of-thought reasoning, through supervised fine-tuning on large-scale and high-quality datasets. To construct such datasets, existing pipelines generate long reasoning data from more capable Large Language Models (LLMs) and apply manually heuristic or naturalness-based selection methods to filter high-quality samples. Despite the proven effectiveness of naturalness-based data selection, which ranks data by the average log probability assigned by LLMs, our analysis shows that, when applied to LLM reasoning datasets, it systematically prefers samples with longer reasoning steps (i.e., more tokens per step) rather than higher-quality ones, a phenomenon we term step length confounding. Through quantitative analysis, we attribute this phenomenon to low-probability first tokens in reasoning steps; longer steps dilute their influence, thereby inflating the average log probabilities. To address this issue, we propose two variant methods: ASLEC-DROP, which drops first-token probabilities when computing average log probability, and ASLEC-CASL, which applies a causal debiasing regression to remove the first tokens' confounding effect. Experiments across four LLMs and five evaluation benchmarks demonstrate the effectiveness of our approach in mitigating the step length confounding problem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06820v1">Beyond Surface Judgments: Human-Grounded Risk Evaluation of LLM-Generated Disinformation</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can generate persuasive narratives at scale, raising concerns about their potential use in disinformation campaigns. Assessing this risk ultimately requires understanding how readers receive such content. In practice, however, LLM judges are increasingly used as a low-cost substitute for direct human evaluation, even though whether they faithfully track reader responses remains unclear. We recast evaluation in this setting as a proxy-validity problem and audit LLM judges against human reader responses. Using 290 aligned articles, 2,043 paired human ratings, and outputs from eight frontier judges, we examine judge--human alignment in terms of overall scoring, item-level ordering, and signal dependence. We find persistent judge--human gaps throughout. Relative to humans, judges are typically harsher, recover item-level human rankings only weakly, and rely on different textual signals, placing more weight on logical rigour while penalizing emotional intensity more strongly. At the same time, judges agree far more with one another than with human readers. These results suggest that LLM judges form a coherent evaluative group that is much more aligned internally than it is with human readers, indicating that internal agreement is not evidence of validity as a proxy for reader response.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06819v1">Beyond End-to-End: Dynamic Chain Optimization for Private LLM Adaptation on the Edge</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Accepted by ACL 2026
    </div>
    <details class="paper-abstract">
      Federated fine-tuning enables privacy-preserving LLM adaptation but faces a critical bottleneck: the disparity between LLMs' high memory demands and edge devices' limited capacity. To break the memory barrier, we propose Chain Federated Fine-Tuning (ChainFed), an innovative paradigm that forgoes end-to-end updates in favor of a sequential, layer-by-layer manner. It first trains the initial adapter to convergence, freezes its weights, and then proceeds to the next. This iterative train-and-freeze process forms an optimization chain, gradually enhancing the model's task-specific proficiency. ChainFed further integrates three core techniques: 1) Dynamic Layer Co-Tuning to bridge semantic gaps between sequentially tuned layers and facilitate information flow; 2) Globally Perceptive Optimization to endow each adapter with foresight beyond its local objective; 3) Function-Oriented Adaptive Tuning to automatically identify the optimal fine-tuning starting point. Extensive experiments on multiple benchmarks demonstrate the superiority of ChainFed over existing methods, boosting average accuracy by up to 46.46\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07396v1">SHIELD: A Segmented Hierarchical Memory Architecture for Energy-Efficient LLM Inference on Edge NPUs</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) inference on edge Neural Processing Units (NPUs) is fundamentally constrained by limited on-chip memory capacity. Although high-density embedded DRAM (eDRAM) is attractive for storing activation workspaces, its periodic refresh consumes substantial energy. Prior work has primarily focused on reducing off-chip traffic or optimizing refresh for persistent Key-Value (KV) caches, while transient and error-resilient Query and Attention Output (QO) activations are largely overlooked. We propose SHIELD, a lifecycle-aware segmented eDRAM architecture that jointly exploits temporal residency and bit-level sensitivity in bfloat16 (BF16) activations. SHIELD isolates the sign and exponent fields from the mantissa, disables refresh for transient QO mantissas, and applies relaxed refresh to persistent KV mantissas. Across multiple LLMs and inference scenarios, SHIELD reduces eDRAM refresh energy by 35% relative to a standard-refresh baseline while preserving accuracy on WikiText-2, PIQA, and ARC-Easy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06799v1">Beyond Accuracy: Diagnosing Algebraic Reasoning Failures in LLMs Across Nine Complexity Dimensions</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Under Review as a conference paper at COLM 2026
    </div>
    <details class="paper-abstract">
      Algebraic reasoning remains one of the most informative stress tests for large language models, yet current benchmarks provide no mechanism for attributing failure to a specific cause. When a model fails an algebraic problem, a single accuracy score cannot reveal whether the expression was too deeply nested, the operator too uncommon, the intermediate state count too high, or the dependency chain too long. Prior work has studied individual failure modes in isolation, but no framework has varied each complexity factor independently under strict experimental control. No prior system has offered automatic generation and verification of problems of increasing complexity to track model progress over time. We introduce a nine-dimension algebraic complexity framework in which each factor is varied independently while all others are held fixed, with problem generation and verification handled by a parametric pipeline requiring no human annotation. Each dimension is grounded in a documented LLM failure mode and captures a structurally distinct aspect of algebraic difficulty, including expression nesting depth, simultaneous intermediate result count, sub-expression complexity, operator hardness, and dependent reasoning chain length. We evaluated seven instruction-tuned models spanning 8B to 235B parameters across all nine dimensions and find that working memory is the dominant scale-invariant bottleneck. Every model collapses between 20 and 30 parallel branches regardless of parameter count, pointing to a hard architectural constraint rather than a solvable capacity limitation. Our analysis further identifies a minimal yet diagnostically sufficient subset of five dimensions that together span the full space of documented algebraic failure modes, providing a complete complexity profile of a model's algebraic reasoning capacity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11663v2">SpecQuant: Spectral Decomposition and Adaptive Truncation for Ultra-Low-Bit LLMs Quantization</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Accepted at AAAI 2026
    </div>
    <details class="paper-abstract">
      The emergence of accurate open large language models (LLMs) has sparked a push for advanced quantization techniques to enable efficient deployment on end-user devices. In this paper, we revisit the challenge of extreme LLM compression -- targeting ultra-low-bit quantization for both activations and weights -- from a Fourier frequency domain perspective. We propose SpecQuant, a two-stage framework that tackles activation outliers and cross-channel variance. In the first stage, activation outliers are smoothed and transferred into the weight matrix to simplify downstream quantization. In the second stage, we apply channel-wise low-frequency Fourier truncation to suppress high-frequency components while preserving essential signal energy, improving quantization robustness. Our method builds on the principle that most of the weight energy is concentrated in low-frequency components, which can be retained with minimal impact on model accuracy. To enable runtime adaptability, we introduce a lightweight truncation module during inference that adjusts truncation thresholds based on channel characteristics. On LLaMA-3 8B, SpecQuant achieves 4-bit quantization for both weights and activations, narrowing the zero-shot accuracy gap to only 1.5% compared to full precision, while delivering 2 times faster inference and 3times lower memory usage. Code will be available at https://github.com/Kishon-zzx/SpecQuant.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.18018v2">Lost in Cultural Translation: Do LLMs Struggle with Math Across Cultural Contexts?</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      We demonstrate that large language models' (LLMs) mathematical reasoning is culturally sensitive: testing 14 models from Anthropic, OpenAI, Google, Meta, DeepSeek, Mistral, and Microsoft across six culturally adapted variants of the GSM8K benchmark, we find accuracy drops ranging from 0.3% (Claude 3.5 Sonnet) to 5.9% (LLaMA 3.1-8B) when math problems are embedded in unfamiliar cultural contexts--even when the underlying mathematical logic remains unchanged. These statistically significant performance reductions (p < 0.01, confirmed through McNemar tests) reveal that mathematical reasoning in LLMs is not culturally neutral. To create these variants for Haiti, Moldova, Pakistan, Solomon Islands, Somalia, and Suriname, we systematically replaced cultural entities (names, foods, places, etc.) in 1,198 GSM8K questions while preserving all mathematical operations and numerical values. Our quantitative error analysis of 18,887 instances reveals that cultural adaptation affects broader reasoning patterns, with mathematical reasoning errors comprising 54.7% and calculation errors 34.5% of failures. Interestingly, cultural familiarity can enhance performance: Mistral Saba outperforms some larger models on Pakistan-adapted problems due to Middle Eastern and South Asian training data exposure. This study underscores the need for more diverse training data to ensure robust LLM performance across global contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07394v1">Flux Attention: Context-Aware Hybrid Attention for Efficient LLMs Inference</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      The quadratic computational complexity of standard attention mechanisms presents a severe scalability bottleneck for LLMs in long-context scenarios. While hybrid attention mechanisms combining Full Attention (FA) and Sparse Attention (SA) offer a potential solution, existing methods typically rely on static allocation ratios that fail to accommodate the variable retrieval demands of different tasks. Furthermore, head-level dynamic sparsity often introduces severe computational load imbalance and synchronization long-tails, which hinder hardware acceleration during autoregressive decoding. To bridge this gap, we introduce Flux Attention, a context-aware framework that dynamically optimizes attention computation at the layer level. By integrating a lightweight Layer Router into frozen pretrained LLMs, the proposed method adaptively routes each layer to FA or SA based on the input context. This layer-wise routing preserves high-fidelity information retrieval while ensuring contiguous memory access, translating theoretical computational reductions into practical wall-clock speedups. As a parameter-efficient approach, our framework requires only 12 hours of training on 8$\times$A800 GPUs. Extensive experiments across multiple long-context and mathematical reasoning benchmarks demonstrate that Flux Attention achieves a superior trade-off between performance and inference speed compared with baseline models, with speed improvements of up to $2.8\times$ and $2.0\times$ in the prefill and decode stages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06763v1">Improving Random Testing via LLM-powered UI Tarpit Escaping for Mobile Apps</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Random GUI testing is a widely-used technique for testing mobile apps. However, its effectiveness is limited by the notorious issue -- UI exploration tarpits, where the exploration is trapped in local UI regions, thus impeding test coverage and bug discovery. In this experience paper, we introduce LLM-powered random GUI Testing, a novel hybrid testing approach to mitigating UI tarpits during random testing. Our approach monitors UI similarity to identify tarpits and query LLMs to suggest promising events for escaping the encountered tarpits. We implement our approach on top of two different automated input generation (AIG) tools for mobile apps: (1) HybridMonkey upon Monkey, a state-of-the-practice tool; and (2) HybridDroidbot upon Droidbot, a state-of-the-art tool. We evaluated them on 12 popular, real-world apps. The results show that HybridMonkey and HybridDroidbot outperform all baselines, achieving average coverage improvements of 54.8% and 44.8%, respectively, and detecting the highest number of unique crashes. In total, we found 75 unique bugs, including 34 previously unknown bugs. To date, 26 bugs have been confirmed and fixed. We also applied HybridMonkey on WeChat, a popular industrial app with billions of monthly active users. HybridMonkey achieved higher activity coverage and found more bugs than random testing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.28239v3">A Switch-Centric In-Network Architecture for Accelerating LLM Inference in Shared-Memory Network</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      In-network computing techniques, exemplified by NVLink SHARP (NVLS), offer a promising approach to addressing the communication bottlenecks in LLM inference by offloading collective operations such as All-Reduce to switches. However, the accelerator-centric architecture of NVLS suffers from two fundamental limitations: 1) it relies on GPU load instructions to trigger in-switch reduction, which means that the data reduced in the switch must be transferred back to the initiating GPU rather than being broadcast directly, thereby introducing unnecessary communication overhead; 2) due to its architectural constraints, NVLS cannot offload operators that are not decomposable into memory-semantic instructions, such as the in-network quantization (INQ) proposed in this work. As a result, All-Reduce in NVLS during inference still operates at 16-bit precision, leading to substantial bandwidth waste. To address these limitations, we propose SCIN, the first switch-centric in-network architecture for multi-accelerator shared-memory networks, enabling both low-latency and high-bandwidth All-Reduce. Specifically, we introduce an in-switch accelerator (ISA) capable of directly accessing the memory regions in attached accelerators for in-network processing, together with a co-designed communication fabric that enables such access with negligible protocol overhead. SCIN delivers lower All-Reduce latency than NVLS by eliminating redundant data movement. Moreover, SCIN enables INQ for All-Reduce, reducing its precision to 8 bits and nearly doubling bandwidth with negligible accuracy loss. We also present a multi-FPGA prototype of SCIN to validate its feasibility and effectiveness. Simulation results for an 8-GPU system show that our design accelerates All-Reduce by up to 8.7x for small messages and 3.8x for large messages, yielding up to 1.74x TTFT speedup and 1.34x TPOT speedup on LLaMA-2 models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.03044v2">JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Xiaodong He is the corresponding author
    </div>
    <details class="paper-abstract">
      We introduce JoyAI-LLM Flash, an efficient Mixture-of-Experts (MoE) language model designed to redefine the trade-off between strong performance and token efficiency in the sub-50B parameter regime. JoyAI-LLM Flash is pretrained on a massive corpus of 20 trillion tokens and further optimized through a rigorous post-training pipeline, including supervised fine-tuning (SFT), Direct Preference Optimization (DPO), and large-scale reinforcement learning (RL) across diverse environments. To improve token efficiency, JoyAI-LLM Flash strategically balances \emph{thinking} and \emph{non-thinking} cognitive modes and introduces FiberPO, a novel RL algorithm inspired by fibration theory that decomposes trust-region maintenance into global and local components, providing unified multi-scale stability control for LLM policy optimization. To enhance architectural sparsity, the model comprises 48B total parameters while activating only 2.7B parameters per forward pass, achieving a substantially higher sparsity ratio than contemporary industry leading models of comparable scale. To further improve inference throughput, we adopt a joint training-inference co-design that incorporates dense Multi-Token Prediction (MTP) and Quantization-Aware Training (QAT). We release the checkpoints for both JoyAI-LLM-48B-A3B Base and its post-trained variants on Hugging Face to support the open-source community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06756v1">How Long Reasoning Chains Influence LLMs' Judgment of Answer Factuality</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 ACL2026 Main
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) has been widely adopted as a scalable surrogate for human evaluation, yet such judges remain imperfect and susceptible to surface-level biases. One possible reason is that these judges lack sufficient information in assessing answer correctness. With the rise of reasoning-capable models, exposing a generator's reasoning content to the judge provides richer information and is a natural candidate for improving judgment accuracy. However, its actual impact on judge behavior remains understudied. In this paper, we systematically investigate how access to reasoning chains affects LLM-based judgment across factual question answering (QA) and mathematical reasoning benchmarks. We find that weak judges are easily swayed by reasoning presence, frequently accepting incorrect answers accompanied by fluent reasoning, while strong judges can partially leverage reasoning as informative evidence. Nevertheless, even strong judges are misled by seemingly high-quality reasoning chains. Controlled experiments further reveal that both fluency and factuality of reasoning chains are critical signals driving judge decisions. These findings highlight the need for more robust LLM judges that can distinguish genuine reasoning quality from superficial fluency when evaluating modern reasoning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06755v1">Babbling Suppression: Making LLMs Greener One Token at a Time</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Context: Large Language Models (LLMs) are increasingly used in modern software development, aiding in code generation, code completion, and refactoring through AI-powered assistants. While they accelerate development workflows, they often produce extraneous output, referred to as "babbling", which incurs additional cognitive, economic, and energy costs. Objective: This work investigates the problem of babbling in LLM-based code generation and proposes a practical, model-agnostic approach to reduce unnecessary output without compromising solution accuracy. Method: We introduce Babbling Suppression (BS), a method that integrates test execution into the LLM generation process by evaluating intermediate outputs and terminating generation once a solution passes all tests. This prevents excessive token generation while having no impact on model accuracy. An empirical study was conducted across two Python and two Java benchmarks, targeting four 3-4B parameter models and six 6-7B parameter models. Results: Our findings show that babbling occurs across all tested models, with higher frequency in Java than in Python. Applying BS significantly reduces energy consumption by up to 65% for Python and 62% for Java in models prone to babbling. Across 40 model-benchmark pairs, 29 showed reduced mean energy consumption, with reductions exceeding 20% in 22 cases. Generated token count decreased in 35 pairs, while the GPU energy-per-token overhead of BS remained below 10% for 26 pairs, decreased for 2, and reached a maximum of 24%, yielding net energy savings in most cases. Implications: BS can make AI-assisted programming more efficient and sustainable by reducing energy consumption and minimizing cognitive effort by developers. Its model-agnostic design allows easy integration, suggesting broad applicability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06753v1">Select-then-Solve: Paradigm Routing as Inference-Time Optimization for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      When an LLM-based agent improves on a task, is the gain from the model itself or from the reasoning paradigm wrapped around it? We study this question by comparing six inference-time paradigms, namely Direct, CoT, ReAct, Plan-Execute, Reflection, and ReCode, across four frontier LLMs and ten benchmarks, yielding roughly 18,000 runs. We find that reasoning structure helps dramatically on some tasks but hurts on others: ReAct improves over Direct by 44pp on GAIA, while CoT degrades performance by 15pp on HumanEval. No single paradigm dominates, and oracle per-task selection beats the best fixed paradigm by 17.1pp on average. Motivated by this complementarity, we propose a select-then-solve approach: before answering each task, a lightweight embedding-based router selects the most suitable paradigm. Across four models, the router improves average accuracy from 47.6% to 53.1%, outperforming the best fixed paradigm at 50.3% by 2.8pp and recovering up to 37% of the oracle gap. In contrast, zero-shot self-routing only works for GPT-5 at 67.1% and fails for weaker models, all trailing the learned router. Our results argue that reasoning paradigm selection should be a per-task decision made by a learned router, not a fixed architectural choice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06742v1">Evaluating LLM-Based 0-to-1 Software Generation in End-to-End CLI Tool Scenarios</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are driving a shift towards intent-driven development, where agents build complete software from scratch. However, existing benchmarks fail to assess this 0-to-1 generation capability due to two limitations: reliance on predefined scaffolds that ignore repository structure planning, and rigid white-box unit testing that lacks end-to-end behavioral validation. To bridge this gap, we introduce CLI-Tool-Bench, a structure-agnostic benchmark for evaluating the ground-up generation of Command-Line Interface (CLI) tools. It features 100 diverse real-world repositories evaluated via a black-box differential testing framework. Agent-generated software is executed in sandboxes, comparing system side effects and terminal outputs against human-written oracles using multi-tiered equivalence metrics. Evaluating seven state-of-the-art LLMs, we reveal that top models achieve under 43% success, highlighting the ongoing challenge of 0-to-1 generation. Furthermore, higher token consumption does not guarantee better performance, and agents tend to generate monolithic code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.18196v2">Contrastive Decoding Mitigates Score Range Bias in LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 To appear at ACL 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are commonly used as evaluators in various applications, but the reliability of the outcomes remains a challenge. One such challenge is using LLMs-as-judges for direct assessment, i.e., assigning scores from a specified range without any references. Focusing on summarization, we first show that this challenge stems from LLM judge outputs being associated with score range bias, i.e., LLM judge outputs are highly sensitive to pre-defined score ranges. We also show that similar biases exist among models from the same family. We then mitigate this bias through contrastive decoding, achieving up to 11.7% relative improvement on average in Spearman correlation with human judgments across different score ranges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06736v1">SQLStructEval: Structural Evaluation of LLM Text-to-SQL Generation</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 17 pages, including figures and tables
    </div>
    <details class="paper-abstract">
      Despite strong performance on Text-to-SQL benchmarks, it remains unclear whether LLM-generated SQL programs are structurally reliable. In this work, we investigate the structural behavior of LLM-generated SQL queries and introduce SQLStructEval, a framework for analyzing program structures through canonical abstract syntax tree (AST) representations. Our experiments on the Spider benchmark show that modern LLMs often produce structurally diverse queries for the same input, even when execution results are correct, and that such variance is frequently triggered by surface-level input changes such as paraphrases or schema presentation. We further show that generating queries in a structured space via a compile-style pipeline can improve both execution accuracy and structural consistency. These findings suggest that structural reliability is a critical yet overlooked dimension for evaluating LLM-based program generation systems. Our code is available at https://anonymous.4open.science/r/StructEval-2435.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.16703v4">ShadowNPU: System and Algorithm Co-design for NPU-Centric On-Device LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 To Appear at MobiSys'26
    </div>
    <details class="paper-abstract">
      On-device running Large Language Models (LLMs) is nowadays a critical enabler towards preserving user privacy. We observe that the attention operator falls back from the special-purpose NPU to the general-purpose CPU/GPU because of quantization sensitivity in state-of-the-art frameworks. This fallback results in a degraded user experience and increased complexity in system scheduling. To this end, this paper presents shadowAttn, a system-algorithm codesigned sparse attention module with minimal reliance on CPU/GPU by only sparsely calculating the attention on a tiny portion of tokens. The key idea is to hide the overhead of estimating the important tokens with a NPU-based pilot compute. Further, shadowAttn proposes insightful techniques such as NPU compute graph bucketing, head-wise NPU-CPU/GPU pipeline and per-head fine-grained sparsity ratio to achieve high accuracy and efficiency. shadowAttn delivers the best performance with highly limited CPU/GPU resource; it requires much less CPU/GPU resource to deliver on-par performance of SoTA frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06723v1">Fine-grained Approaches for Confidence Calibration of LLMs in Automated Code Revision</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      In today's AI-assisted software engineering landscape, developers increasingly depend on LLMs that are highly capable, yet inherently imperfect. The tendency of these models to produce incorrect outputs can reduce developer productivity. To this end, a canonical mitigation method is to provide calibrated confidence scores that faithfully reflect their likelihood of correctness at the instance-level. Such information allows users to make immediate decisions regarding output acceptance, abstain error-prone outputs, and better align their expectations with the model's capabilities. Since post-trained LLMs do not inherently produce well-calibrated confidence scores, researchers have developed post-hoc calibration methods, with global Platt-scaling of sequence-level confidence scores proving effective in many generative software engineering tasks but remaining unreliable or unexplored for automated code revision (ACR) tasks such as program repair, vulnerability repair, and code refinement. We hypothesise that the coarse-grained nature of this conventional method makes it ill-suited for ACR tasks, where correctness is often determined by local edit decisions and miscalibration can be sample-dependent, thereby motivating fine-grained confidence calibration. To address this, our study proposes local Platt-scaling applied separately to three different fine-grained confidence scores. Through experiments across 3 separate tasks and correctness metrics, as well as 14 different models of various sizes, we find that fine-grained confidence scores consistently achieve lower calibration error across a broader range of probability intervals, and this effect is further amplified when global Platt-scaling is applied. Our proposed approaches offer a practical solution to eliciting well-calibrated confidence scores, enabling more trustworthy and streamlined usage of imperfect models in ACR tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16206v3">Computer Environments Elicit General Agentic Intelligence in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Project Page: https://llm-in-sandbox.github.io
    </div>
    <details class="paper-abstract">
      Agentic intelligence in large language models (LLMs) requires not only model intrinsic capabilities but also interactions with external environments. Equipping LLMs with computers now represents a prevailing trend. However, the computer environment's intrinsic value has not been systematically investigated, particularly its potential to elicit general capabilities. Here we introduce LLM-in-Sandbox, which virtualizes the computer as a code sandbox with only basic functionalities, and demonstrate that this minimal setting elicits computer-based meta-capabilities for general task solving: external resource access, file management, and code execution. Without additional training, strong models achieve substantial gains (up to 15.5%) across mathematics, physics, chemistry, biomedicine, long-context understanding, and instruction following, while reducing token consumption by up to 8 times. Furthermore, we develop LLM-in-Sandbox-RL to train models exclusively on non-agentic data within the sandbox, empowering weaker models to harness the environment and internalize these interactions. Our results demonstrate that computer environments elicit general intelligence, yield efficiency gains, and can be harnessed through training, serving as a promising foundation for generalist agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02627v2">Improved Evidence Extraction and Metrics for Document Inconsistency Detection with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 14 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are becoming useful in many domains due to their impressive abilities that arise from large training datasets and large model sizes. However, research on LLM-based approaches to document inconsistency detection is relatively limited. We address this gap by investigating evidence extraction capabilties of LLMs for document inconsistency detection. To this end, we introduce new comprehensive evidence-extraction metrics and a redact-and-retry framework with constrained filtering that substantially improves evidence extraction performance over other prompting methods. We support our approach with strong experimental results and release a new semi-synthetic dataset for evaluating evidence extraction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2407.16216v2">Reinforcement Learning for LLM Post-Training: A Survey</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Through pretraining and supervised fine-tuning (SFT), large language models (LLMs) acquire strong instruction-following capabilities, yet they can still produce harmful or misaligned outputs. A growing body of reinforcement learning (RL)-based post-training methods has been proposed to address this, including Reinforcement Learning from Human Feedback (RLHF) and Reinforcement Learning with Verifiable Rewards (RLVR) approaches built on Proximal Policy Optimization (PPO), Group Relative Policy Optimization (GRPO), Direct Preference Optimization (DPO), and others. Despite rapid progress, no existing work offers a systematic, technically detailed comparison of these methods under a single analytical lens. Our survey aims to fill this gap. We make three key contributions: (1) a self-contained RL and LLM post-training foundations treatment covering all necessary concepts alongside their key applications; (2) a unified policy gradient framework unifying PPO and GRPO-based RLHF, RLVR, and offline DPO-based RLHF, decomposing methods along the axes of prompt sampling, response sampling, and gradient coefficient, with an extended treatment of on-policy RLHF and iterative DPO methods as well as the richer design space of offline DPO-based methods; and (3) standardized notation across all reviewed papers enabling direct technical comparison. Our goal is to serve as a comprehensive, technically grounded reference for researchers and practitioners working on LLM post-training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07387v1">Self-Calibrating LLM-Based Analog Circuit Sizing with Interpretable Design Equations</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 11 pages, 5 figures, 4 tables. Submitted to IEEE Transactions on Circuits and Systems for Artificial Intelligence (TCASAI)
    </div>
    <details class="paper-abstract">
      We present a self-calibrating framework for analog circuit sizing in which a large language model (LLM) derives topology-specific analytical design equations directly from a raw circuit netlist. Unlike existing AI-driven sizing methods where the model proposes parameter adjustments or reduces a search space, the LLM produces a complete Python sizing function tracing each device dimension to a specific performance constraint. A deterministic calibration loop extracts process-dependent parameters from a single transistor-level simulation, while a prediction-error feedback mechanism compensates for analytical inaccuracies. We validate the framework on six operational transconductance amplifier (OTA) topologies spanning three families at two process nodes (180 nm and 40 nm CMOS). All 12 topology-node combinations achieve all specifications, converging in 2-9 simulations for 11 of 12 cases, with one outlier requiring 16 simulations due to an extremely narrow feasible region. Despite large initial prediction errors, convergence depends on the measurement-feedback architecture, not prediction accuracy. This one-shot calibration automatically captures process-dependent variations, enabling cross-node portability without modification, retraining, or per-process characterization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06666v1">A Graph-Enhanced Defense Framework for Explainable Fake News Detection with LLM</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Accepted by TOIS
    </div>
    <details class="paper-abstract">
      Explainable fake news detection aims to assess the veracity of news claims while providing human-friendly explanations. Existing methods incorporating investigative journalism are often inefficient and struggle with breaking news. Recent advances in large language models (LLMs) enable leveraging externally retrieved reports as evidence for detection and explanation generation, but unverified reports may introduce inaccuracies. Moreover, effective explainable fake news detection should provide a comprehensible explanation for all aspects of a claim to assist the public in verifying its accuracy. To address these challenges, we propose a graph-enhanced defense framework (G-Defense) that provides fine-grained explanations based solely on unverified reports. Specifically, we construct a claim-centered graph by decomposing the news claim into several sub-claims and modeling their dependency relationships. For each sub-claim, we use the retrieval-augmented generation (RAG) technique to retrieve salient evidence and generate competing explanations. We then introduce a defense-like inference module based on the graph to assess the overall veracity. Finally, we prompt an LLM to generate an intuitive explanation graph. Experimental results demonstrate that G-Defense achieves state-of-the-art performance in both veracity detection and the quality of its explanations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06664v1">Foundry: Template-Based CUDA Graph Context Materialization for Fast LLM Serving Cold Start</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Modern LLM service providers increasingly rely on autoscaling and parallelism reconfiguration to respond to rapidly changing workloads, but cold-start latency remains a major bottleneck. While recent systems have reduced model weight loading to seconds, CUDA graph capture still takes tens of seconds to minutes and often dominates startup. Unfortunately, CUDA graphs cannot be naively serialized: beyond graph topology, they are tightly coupled to execution context, including device addresses embedded in kernel arguments and kernel code lazily loaded during warmup. Existing approaches either rely on brittle kernel-specific patching or heavyweight process-level checkpoint/restore that are inflexible to dynamic parallelism switching. We present Foundry, a template-based CUDA graph context materialization system that persists both graph topology and execution context during an offline processing stage, and reconstructs executable graphs online with negligible overhead. Foundry enforces deterministic memory layouts, automatically extracts and reloads kernel binaries required by captured graphs, and reduces online reconstruction costs through topology-based templating. For distributed serving, Foundry further enables a single-GPU offline capture to generate templates for multi-GPU deployments by patching only rank-dependent communication state. Across dense and MoE models up to 235B parameters, Foundry reduces cold-start latency by up to 99%, cutting the initialization time of Qwen3-235B-A22B from 10 minutes to 3.9 seconds while preserving the throughput gains of CUDA graphs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06663v1">Restoring Heterogeneity in LLM-based Social Simulation: An Audience Segmentation Approach</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to simulate social attitudes and behaviors, offering scalable "silicon samples" that can approximate human data. However, current simulation practice often collapses diversity into an "average persona," masking subgroup variation that is central to social reality. This study introduces audience segmentation as a systematic approach for restoring heterogeneity in LLM-based social simulation. Using U.S. climate-opinion survey data, we compare six segmentation configurations across two open-weight LLMs (Llama 3.1-70B and Mixtral 8x22B), varying segmentation identifier granularity, parsimony, and selection logic (theory-driven, data-driven, and instrument-based). We evaluate simulation performance with a three-dimensional evaluation framework covering distributional, structural, and predictive fidelity. Results show that increasing identifier granularity does not produce consistent improvement: moderate enrichment can improve performance, but further expansion does not reliably help and can worsen structural and predictive fidelity. Across parsimony comparisons, compact configurations often match or outperform more comprehensive alternatives, especially in structural and predictive fidelity, while distributional fidelity remains metric dependent. Identifier selection logic determines which fidelity dimension benefits most: instrument-based selection best preserves distributional shape, whereas data-driven selection best recovers between-group structure and identifier-outcome associations. Overall, no single configuration dominates all dimensions, and performance gains in one dimension can coincide with losses in another. These findings position audience segmentation as a core methodological approach for valid LLM-based social simulation and highlight the need for heterogeneity-aware evaluation and variance-preserving modeling strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02670v2">Break Me If You Can: Self-Jailbreaking of Aligned LLMs via Lexical Insertion Prompting</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      We introduce \emph{self-jailbreaking}, a threat model in which an aligned LLM guides its own compromise. Unlike most jailbreak techniques, which often rely on handcrafted prompts or separate attacker models, self-jailbreaking requires no external red-team LLM: the target model's own internal knowledge suffices. We operationalize this via \textbf{Self-Jailbreaking via Lexical Insertion Prompting (\textsc{SLIP})}, a black-box algorithm that casts jailbreaking as breadth-first tree search over multi-turn dialogues, incrementally inserting missing content words from the attack goal into benign prompts using the target model as its own guide. Evaluations on AdvBench and HarmBench show \textsc{SLIP} achieves 90--100\% Attack Success Rate (ASR) (avg.\ 94.7\%) across most of the eleven tested models (including GPT-5.1, Claude-Sonnet-4.5, Gemini-2.5-Pro, and DeepSeek-V3), with only ${\sim}7.9$ LLM calls on average, 3--6$\times$ fewer than prior methods. We evaluate existing defenses, show that regex-based approaches are evaded by prompt paraphrasing, and propose the Semantic Drift Monitor (SDM) defense that tracks \textsc{SLIP}'s embedding-space trajectory, achieving 76\% detection at 5\% FPR. However, SDM remains insufficient against adaptive attack strategies, underscoring the need for more advanced defense mechanisms tailored to the self-jailbreaking threat surface. We release our code for reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06636v1">SHAPE: Stage-aware Hierarchical Advantage via Potential Estimation for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 ACL 2026 Main
    </div>
    <details class="paper-abstract">
      Process supervision has emerged as a promising approach for enhancing LLM reasoning, yet existing methods fail to distinguish meaningful progress from mere verbosity, leading to limited reasoning capabilities and unresolved token inefficiency. To address this, we propose Stage-aware Hierarchical Advantage via Potential Estimation (SHAPE), a framework that formalizes reasoning as a trajectory through a state space of empirical solvability. SHAPE introduces a hierarchical credit assignment mechanism: at the segment level, it employs a stage-aware advantage function to prioritize efficient breakthroughs in low-potential states; at the token level, it utilizes entropy-driven redistribution to sharpen execution signals. Extensive experiments in math reasoning across three base models and five benchmarks demonstrate that SHAPE achieves an average accuracy gain of 3% with 30% reduced token consumption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.19225v3">RLBoost: Harvesting Preemptible Resources for Cost-Efficient Reinforcement Learning on LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become essential for unlocking advanced reasoning capabilities in large language models (LLMs). RL workflows involve interleaving rollout and training stages with fundamentally different resource requirements. Rollout typically dominates overall execution time, yet scales efficiently through multiple independent instances. In contrast, training requires tightly-coupled GPUs with full-mesh communication. Existing RL frameworks fall into two categories: co-located and disaggregated architectures. Co-located frameworks fail to address this resource tension by forcing both stages to share the same GPUs. Disaggregated architectures, without modifications of well-established RL algorithms, suffer from resource under-utilization. Meanwhile, preemptible GPU resources, i.e., spot instances on public clouds and spare capacity in production clusters, present significant cost-saving opportunities for accelerating RL workflows, if efficiently harvested for rollout. In this paper, we present RLBoost, a framework for cost-efficient RL training that harvests preemptible GPU resources. Our key insight is that rollout's stateless and embarrassingly parallel nature aligns perfectly with preemptible and often fragmented resources. To efficiently utilize these resources despite frequent and unpredictable availability changes, RLBoost adopts a hybrid architecture with three key techniques: (1) adaptive rollout offload to dynamically adjust workloads on the reserved (on-demand) cluster, (2) pull-based weight transfer that quickly provisions newly available instances, and (3) token-level response collection and migration for efficient preemption handling and continuous load balancing. Extensive experiments show RLBoost increases training throughput by 1.51x-1.97x while improving cost efficiency by 28%-49% compared to using only on-demand GPU resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06607v1">CoverAssert: Iterative LLM Assertion Generation Driven by Functional Coverage via Syntax-Semantic Representations</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 3 pages, 2 figures
    </div>
    <details class="paper-abstract">
      LLMs can generate SystemVerilog assertions (SVAs) from natural language specs, but single-pass outputs often lack functional coverage due to limited IC design understanding. We propose CoverAssert, an iterative framework that clusters semantic and AST-based structural features of assertions, maps them to specifications, and uses functional coverage feedback to guide LLMs in prioritizing uncovered points. Experiments on four open-source designs show that integrating CoverAssert with AssertLLM and Spec2Assertion improves average improvements of 9.57 % in branch coverage, 9.64 % in statement coverage, and 15.69 % in toggle coverage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06603v1">Scientific Knowledge-driven Decoding Constraints Improving the Reliability of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown strong knowledge reserves and task-solving capabilities, but still face the challenge of severe hallucination, hindering their practical application. Though scientific theories and rules can efficiently direct the behaviors of human manipulators, LLMs still do not utilize these highly-condensed knowledge sufficiently through training or prompting. To address this issue, we propose \textbf{SciDC}, an LLM generation method that integrate subject-specific knowledge with strong constraints. By adopting strong LLMs to automatically convert flexible knowledge into multi-layered, standardized rules, we build an extensible framework to effectively constrain the model generation on domain tasks. Experiments on scientific tasks including industrial formulation design, clinical tumor diagnosis and retrosynthesis planning, consistently demonstrate the effectiveness of our method, achieving a 12\% accuracy improvement on average compared with vanilla generation. We further discuss the potential of LLMs in automatically inductively summarizing highly-condensed knowledge, looking ahead to practical solutions for accelerating the overall scientific research process. All the code of this paper can be obtained (https://github.com/Maotian-Ma/SciDC).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21463v2">Unifying Speech Editing Detection and Content Localization via Prior-Enhanced Audio LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Existing speech editing detection (SED) datasets are predominantly constructed using manual splicing or limited editing operations, resulting in restricted diversity and poor coverage of realistic editing scenarios. Meanwhile, current SED methods rely heavily on frame-level supervision to detect observable acoustic anomalies, which fundamentally limits their ability to handle deletion-type edits, where the manipulated content is entirely absent from the signal. To address these challenges, we present a unified framework that bridges speech editing detection and content localization through a generative formulation based on Audio Large Language Models (Audio LLMs). We first introduce AiEdit, a large-scale bilingual dataset (approximately 140 hours) that covers addition, deletion, and modification operations using state-of-the-art end-to-end speech editing systems, providing a more realistic benchmark for modern threats. Building upon this, we reformulate SED as a structured text generation task, enabling joint reasoning over edit type identification, and content localization. To enhance the grounding of generative models in acoustic evidence, we propose a prior-enhanced prompting strategy that injects word-level probabilistic cues derived from a frame-level detector. Furthermore, we introduce an acoustic consistency-aware loss that explicitly enforces the separation between normal and anomalous acoustic representations in the latent space. Experimental results demonstrate that the proposed approach consistently outperforms existing methods across both detection and localization tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06571v1">LLM-based Schema-Guided Extraction and Validation of Missing-Person Intelligence from Heterogeneous Data Sources</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 9 pages, 6 figures. Accepted at International Conference on Intelligent Digitization of Systems and Services (IDSS 2026)
    </div>
    <details class="paper-abstract">
      Missing-person and child-safety investigations rely on heterogeneous case documents, including structured forms, bulletin-style posters, and narrative web profiles. Variations in layout, terminology, and data quality impede rapid triage, large-scale analysis, and search-planning workflows. This paper introduces the Guardian Parser Pack, an AI-driven parsing and normalization pipeline that transforms multi-source investigative documents into a unified, schema-compliant representation suitable for operational review and downstream spatial modeling. The proposed system integrates (i) multi-engine PDF text extraction with Optical Character Recognition (OCR) fallback, (ii) rule-based source identification with source-specific parsers, (iii) schema-first harmonization and validation, and (iv) an optional Large Language Model (LLM)-assisted extraction pathway incorporating validator-guided repair and shared geocoding services. We present the system architecture, key implementation decisions, and output design, and evaluate performance using both gold-aligned extraction metrics and corpus-level operational indicators. On a manually aligned subset of 75 cases, the LLM-assisted pathway achieved substantially higher extraction quality than the deterministic comparator (F1 = 0.8664 vs. 0.2578), while across 517 parsed records per pathway it also improved aggregate key-field completeness (96.97\% vs. 93.23\%). The deterministic pathway remained much faster (mean runtime 0.03 s/record vs. 3.95 s/record for the LLM pathway). In the evaluated run, all LLM outputs passed initial schema validation, so validator-guided repair functioned as a built-in safeguard rather than a contributor to the observed gains. These results support controlled use of probabilistic AI within a schema-first, auditable pipeline for high-stakes investigative settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06552v1">To Lie or Not to Lie? Investigating The Biased Spread of Global Lies by LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Accepted at ACL 2026 Main Conference
    </div>
    <details class="paper-abstract">
      Misinformation is on the rise, and the strong writing capabilities of LLMs lower the barrier for malicious actors to produce and disseminate false information. We study how LLMs behave when prompted to spread misinformation across languages and target countries, and introduce GlobalLies, a multilingual parallel dataset of 440 misinformation generation prompt templates and 6,867 entities, spanning 8 languages and 195 countries. Using both human annotations and large-scale LLM-as-a-judge evaluations across hundreds of thousands of generations from state-of-the-art models, we show that misinformation generation varies systematically based on the country being discussed. Propagation of lies by LLMs is substantially higher in many lower-resource languages and for countries with a lower Human Development Index (HDI). We find that existing mitigation strategies provide uneven protection: input safety classifiers exhibit cross-lingual gaps, and retrieval-augmented fact-checking remains inconsistent across regions due to unequal information availability. We release GlobalLies for research purposes, aiming to support the development of mitigation strategies to reduce the spread of global misinformation: https://github.com/zohaib-khan5040/globallies
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06543v1">The Illusion of Stochasticity in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      In this work, we demonstrate that reliable stochastic sampling is a fundamental yet unfulfilled requirement for Large Language Models (LLMs) operating as agents. Agentic systems are frequently required to sample from distributions, often inferred from observed data, a process which needs to be emulated by the LLM. This leads to a distinct failure point: while standard RL agents rely on external sampling mechanisms, LLMs fail to map their internal probability estimates to their stochastic outputs. Through rigorous empirical analysis across multiple model families, model sizes, prompting styles, and distributions, we demonstrate the extent of this failure. Crucially, we show that while powerful frontier models can convert provided random seeds to target distributions, their ability to sample directly from specific distributions is fundamentally flawed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06506v1">Guiding Symbolic Execution with Static Analysis and LLMs for Vulnerability Discovery</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      Symbolic execution detects vulnerabilities with precision, but applying it to large codebases requires harnesses that set up symbolic state, model dependencies, and specify assertions. Writing these harnesses has traditionally been a manual process requiring expert knowledge, which significantly limits the scalability of the technique. We present Static Analysis Informed and LLM-Orchestrated Symbolic Execution (SAILOR), which automates symbolic execution harness construction by combining static analysis with LLM-based synthesis. SAILOR operates in three phases: (1) static analysis identifies candidate vulnerable locations and generates vulnerability specifications; (2) an LLM uses vulnerability specifications and orchestrates harness synthesis by iteratively refining drivers, stubs, and assertions against compiler and symbolic execution feedback; symbolic execution then detects vulnerabilities using the generated harness, and (3) concrete replay validates the symbolic execution results against the unmodified project source. This design combines the scalability of static analysis, the code reasoning of LLMs, the path precision of symbolic execution, and the ground truth produced by concrete execution. We evaluate SAILOR on 10 open-source C/C++ projects totaling 6.8 M lines of code. SAILOR discovers 379 distinct, previously unknown memory-safety vulnerabilities (421 confirmed crashes). The strongest of five baselines we compare SAILOR to (agentic vulnerability detection using Claude Code with full codebase access and unlimited interaction), finds only 12 vulnerabilities. Each phase of SAILOR is critical: Without static analysis targeting confirmed vulnerabilities drop 12.2X; without iterative LLM synthesis zero vulnerabilities are confirmed; and without symbolic execution no approach can detect more than 12 vulnerabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.07181v3">PACIFIC: Can LLMs Discern the Traits Influencing Your Preferences? Evaluating Personality-Driven Preference Alignment in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      User preferences are increasingly used to personalize Large Language Model (LLM) responses, yet how to reliably leverage preference signals for answer generation remains under-explored. In practice, preferences can be noisy, incomplete, or even misleading, which can degrade answer quality when applied naively. Motivated by the observation that stable personality traits shape everyday preferences, we study personality as a principled ''latent'' signal behind preference statements. Through extensive experiments, we find that conditioning on personality-aligned preferences substantially improves personalized question answering: selecting preferences consistent with a user's inferred personality increases answer-choice accuracy from 29.25% to 76%, compared to using randomly selected preferences. Based on these findings, we introduce PACIFIC (Preference Alignment Choices Inference for Five-factor Identity Characterization), a personality-labeled preference dataset containing 1200 preference statements spanning diverse domains (e.g., travel, movies, education), annotated with Big-Five (OCEAN) trait directions. Finally, we propose a framework that enables an LLM model to automatically retrieve personality-aligned preferences and incorporate them during answer generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06487v1">Closing the Speech-Text Gap with Limited Audio for Effective Domain Adaptation in LLM-Based ASR</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 Submitted to Interspeech
    </div>
    <details class="paper-abstract">
      Conventional end-to-end automatic speech recognition (ASR) systems rely on paired speech-text data for domain adaptation. Recent LLM-based ASR architectures connect a speech encoder to a large language model via a projection module, enabling adaptation with text-only data. However, this introduces a modality gap, as the LLM is not exposed to the noisy representations produced by the speech projector. We investigate whether small amounts of speech can mitigate this mismatch. We compare three strategies: text-only adaptation, paired speech-text adaptation, and mixed batching (MB), which combines both. Experiments in in-domain and out-of-domain settings show that even limited speech consistently improves performance. Notably, MB using only 10% of the target-domain (less than 4 hours) speech achieves word error rates comparable to, or better than, conventional ASR fine-tuning with the full dataset, indicating that small amounts of speech provide a strong modality-alignment signal.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.25898v2">On Integrating Resilience and Human Oversight into LLM-Assisted Modeling Workflows for Digital Twins</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      LLM-assisted modeling holds the potential to rapidly build executable Digital Twins of complex systems from only coarse descriptions and sensor data. However, resilience to LLM hallucination, human oversight, and real-time model adaptability remain challenging and often mutually conflicting requirements. We present three critical design principles for integrating resilience and oversight into such workflows, derived from insights gained through our work on FactoryFlow - an open-source LLM-assisted framework for building simulation-based Digital Twins of manufacturing systems. First, orthogonalize structural modeling and parameter fitting. Structural descriptions (components, interconnections) are LLM-translated from coarse natural language to an intermediate representation (IR) with human visualization and validation, which is algorithmically converted to the final model. Parameter inference, in contrast, operates continuously on sensor data streams with expert-tunable controls. Second, restrict the model IR to interconnections of parameterized, pre-validated library components rather than monolithic simulation code, enabling interpretability and error-resilience. Third, and most important, is to use a density-preserving IR. When IR descriptions expand dramatically from compact inputs hallucination errors accumulate proportionally. We present the case for Python as a density-preserving IR : loops express regularity compactly, classes capture hierarchy and composition, and the result remains highly readable while exploiting LLMs strong code generation capabilities. A key contribution is detailed characterization of LLM-induced errors across model descriptions of varying detail and complexity, revealing how IR choice critically impacts error rates. These insights provide actionable guidance for building resilient and transparent LLM-assisted simulation automation workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06416v1">Attention Flows: Tracing LLM Conceptual Engagement via Story Summaries</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      Although LLM context lengths have grown, there is evidence that their ability to integrate information across long-form texts has not kept pace. We evaluate one such understanding task: generating summaries of novels. When human authors of summaries compress a story, they reveal what they consider narratively important. Therefore, by comparing human and LLM-authored summaries, we can assess whether models mirror human patterns of conceptual engagement with texts. To measure conceptual engagement, we align sentences from 150 human-written novel summaries with the specific chapters they reference. We demonstrate the difficulty of this alignment task, which indicates the complexity of summarization as a task. We then generate and align additional summaries by nine state-of-the-art LLMs for each of the 150 reference texts. Comparing the human and model-authored summaries, we find both stylistic differences between the texts and differences in how humans and LLMs distribute their focus throughout a narrative, with models emphasizing the ends of texts. Comparing human narrative engagement with model attention mechanisms suggests explanations for degraded narrative comprehension and targets for future development. We release our dataset to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06403v1">FMI@SU ToxHabits: Evaluating LLMs Performance on Toxic Habit Extraction in Spanish Clinical Texts</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 8 pages, 1 figure, 6 tables, Challenge and Workshop BC9 Large Language Models for Clinical and Biomedical NLP, International Joint Conference on Artificial Intelligence IJCAI 2025
    </div>
    <details class="paper-abstract">
      The paper presents an approach for the recognition of toxic habits named entities in Spanish clinical texts. The approach was developed for the ToxHabits Shared Task. Our team participated in subtask 1, which aims to detect substance use and abuse mentions in clinical case reports and classify them in four categories (Tobacco, Alcohol, Cannabis, and Drug). We explored various methods of utilizing LLMs for the task, including zero-shot, few-shot, and prompt optimization, and found that GPT-4.1's few-shot prompting performed the best in our experiments. Our method achieved an F1 score of 0.65 on the test set, demonstrating a promising result for recognizing named entities in languages other than English.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06401v1">ProofSketcher: Hybrid LLM + Lightweight Proof Checker for Reliable Math/Logic Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      The large language models (LLMs) might produce a persuasive argument within mathematical and logical fields, although such argument often includes some minor missteps, including the entire omission of side conditions, invalid inference patterns, or appeals to a lemma that cannot be derived logically out of the context being discussed. These omissions are infamously hard to notice solely out of the text, as even the misconstrued construction still may seem mostly accurate. Conversely, interactive theorem provers like Lean and Coq have rigorous reliability by ensuring that syntactic and semantic statements only accept statements that can pass all the syntactic and semantic steps in the program which is a small trusted kernel of the language type-checks with. Despite the fact that this technique provides strong guarantees, it comes at quite a heavy price: the evidence must be completely formalized, and the evidence user or a auxiliary search program must provide an avalanche of low-level information. This paper presents a hybrid pipeline where an LLM generates a typed proof sketch in a compact DSL and a lightweight trusted kernel expands the sketch into explicit proof obligations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06393v1">ART: Attention Replacement Technique to Improve Factuality in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      Hallucination in large language models (LLMs) continues to be a significant issue, particularly in tasks like question answering, where models often generate plausible yet incorrect or irrelevant information. Although various methods have been proposed to mitigate hallucinations, the relationship between attention patterns and hallucinations has not been fully explored. In this paper, we analyze the distribution of attention scores across each layer and attention head of LLMs, revealing a common and intriguing phenomenon: shallow layers of LLMs primarily rely on uniform attention patterns, where the model distributes its attention evenly across the entire sequence. This uniform attention pattern can lead to hallucinations, as the model fails to focus on the most relevant information. To mitigate this issue, we propose a training-free method called Attention Replacement Technique (ART), which replaces these uniform attention patterns in the shallow layers with local attention patterns. This change directs the model to focus more on the relevant contexts, thus reducing hallucinations. Through extensive experiments, ART demonstrates significant reductions in hallucinations across multiple LLM architectures, proving its effectiveness and generalizability without requiring fine-tuning or additional training data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06385v1">Application-Driven Pedagogical Knowledge Optimization of Open-Source LLMs via Reinforcement Learning and Supervised Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 * These authors contributed equally to this work and share first authorship
    </div>
    <details class="paper-abstract">
      We present an innovative multi-stage optimization strategy combining reinforcement learning (RL) and supervised fine-tuning (SFT) to enhance the pedagogical knowledge of large language models (LLMs), as illustrated by EduQwen 32B-RL1, EduQwen 32B-SFT, and an optional third-stage model EduQwen 32B-SFT-RL2: (1) RL optimization that implements progressive difficulty training, focuses on challenging examples, and employs extended reasoning rollouts; (2) a subsequent SFT phase that leverages the RL-trained model to synthesize high-quality training data with difficulty-weighted sampling; and (3) an optional second round of RL optimization. EduQwen 32B-RL1, EduQwen 32B-SFT, and EduQwen 32B-SFT-RL2 are an application-driven family of open-source pedagogical LLMs built on a dense Qwen3-32B backbone. These models remarkably achieve high enough accuracy on the Cross-Domain Pedagogical Knowledge (CDPK) Benchmark to establish new state-of-the-art (SOTA) results across the interactive Pedagogy Benchmark Leaderboard and surpass significantly larger proprietary systems such as the previous benchmark leader Gemini-3 Pro. These dense 32-billion-parameter models demonstrate that domain-specialized optimization can transform mid-sized open-source LLMs into true pedagogical domain experts that outperform much larger general-purpose systems, while preserving the transparency, customizability, and cost-efficiency required for responsible educational AI deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02624v2">LAsset: An LLM-assisted Security Asset Identification Framework for System-on-Chip (SoC) Verification</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 This paper will be presented at Design, Automation and Test in Europe Conference (DATE) 2026
    </div>
    <details class="paper-abstract">
      The growing complexity of modern system-on-chip (SoC) and IP designs is making security assurance difficult day by day. One of the fundamental steps in the pre-silicon security verification of a hardware design is the identification of security assets, as it substantially influences downstream security verification tasks, such as threat modeling, security property generation, and vulnerability detection. Traditionally, assets are determined manually by security experts, requiring significant time and expertise. To address this challenge, we present LAsset, a novel automated framework that leverages large language models (LLMs) to identify security assets from both hardware design specifications and register-transfer level (RTL) descriptions. The framework performs structural and semantic analysis to identify intra-module primary and secondary assets and derives inter-module relationships to systematically characterize security dependencies at the design level. Experimental results show that the proposed framework achieves high classification accuracy, reaching up to 90% recall rate in SoC design, and 93% recall rate in IP designs. This automation in asset identification significantly reduces manual overhead and supports a scalable path forward for secure hardware development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01925v2">Rectifying LLM Thought from Lens of Optimization</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 Accepted by ICLR 2026
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have been driven by their emergent reasoning capabilities, particularly through long chain-of-thought (CoT) prompting, which enables thorough exploration and deliberation. Despite these advances, long-CoT LLMs often exhibit suboptimal reasoning behaviors, such as overthinking and excessively protracted reasoning chains, which can impair performance. In this paper, we analyze reasoning processes through an optimization lens, framing CoT as a gradient descent procedure where each reasoning step constitutes an update toward problem resolution. Building on this perspective, we introduce RePro (Rectifying Process-level Reward), a novel approach to refine LLM reasoning during post-training. RePro defines a surrogate objective function to assess the optimization process underlying CoT, utilizing a dual scoring mechanism to quantify its intensity and stability. These scores are aggregated into a composite process-level reward, seamlessly integrated into reinforcement learning with verifiable rewards (RLVR) pipelines to optimize LLMs. Extensive experiments across multiple reinforcement learning algorithms and diverse LLMs, evaluated on benchmarks spanning mathematics, science, and coding, demonstrate that RePro consistently enhances reasoning performance and mitigates suboptimal reasoning behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06163v1">Data, Not Model: Explaining Bias toward LLM Texts in Neural Retrievers</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      Recent studies show that neural retrievers often display source bias, favoring passages generated by LLMs over human-written ones, even when both are semantically similar. This bias has been considered an inherent flaw of retrievers, raising concerns about the fairness and reliability of modern information access systems. Our work challenges this view by showing that source bias stems from supervision in retrieval datasets rather than the models themselves. We found that non-semantic differences, like fluency and term specificity, exist between positive and negative documents, mirroring differences between LLM and human texts. In the embedding space, the bias direction from negatives to positives aligns with the direction from human-written to LLM-generated texts. We theoretically show that retrievers inevitably absorb the artifact imbalances in the training data during contrastive learning, which leads to their preferences over LLM texts. To mitigate the effect, we propose two approaches: 1) reducing artifact differences in training data and 2) adjusting LLM text vectors by removing their projection on the bias vector. Both methods substantially reduce source bias. We hope our study alleviates some concerns regarding LLM-generated texts in information access systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06297v1">FedSpy-LLM: Towards Scalable and Generalizable Data Reconstruction Attacks from Gradients on LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      Given the growing reliance on private data in training Large Language Models (LLMs), Federated Learning (FL) combined with Parameter-Efficient Fine-Tuning (PEFT) has garnered significant attention for enhancing privacy and efficiency. Despite FL's privacy benefits, prior studies have shown that private data can still be extracted from shared gradients. However, these studies, mainly on full-parameter model training, are limited to reconstructing small batches, short input sequences, and specific model architectures, such as encoder-based or decoder-based models. The reconstruction quality becomes even worse when dealing with gradients from PEFT methods. To fully understand the practical attack surface of federated LLMs, this paper proposes FedSpy-LLM, a scalable and generalizable data reconstruction attack designed to reconstruct training data with larger batch sizes and longer sequences while generalizing across diverse model architectures, even when PEFT methods are deployed for training. At the core of FedSpy-LLM is a novel gradient decomposition strategy that exploits the rank deficiency and subspace structure of gradients, enabling efficient token extraction while preserving key signal components at scale. This approach further mitigates the reconstruction challenges introduced by PEFT's substantial null space, ensuring robustness across encoder-based, decoder-based, and encoder-decoder model architectures. Additionally, by iteratively aligning each token's partial-sequence gradient with the full-sequence gradient, FedSpy-LLM ensures accurate token ordering in reconstructed sequences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06091v1">Social Dynamics as Critical Vulnerabilities that Undermine Objective Decision-Making in LLM Collectives</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 ACL 2026
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents are increasingly acting as human delegates in multi-agent environments, where a representative agent integrates diverse peer perspectives to make a final decision. Drawing inspiration from social psychology, we investigate how the reliability of this representative agent is undermined by the social context of its network. We define four key phenomena-social conformity, perceived expertise, dominant speaker effect, and rhetorical persuasion-and systematically manipulate the number of adversaries, relative intelligence, argument length, and argumentative styles. Our experiments demonstrate that the representative agent's accuracy consistently declines as social pressure increases: larger adversarial groups, more capable peers, and longer arguments all lead to significant performance degradation. Furthermore, rhetorical strategies emphasizing credibility or logic can further sway the agent's judgment, depending on the context. These findings reveal that multi-agent systems are sensitive not only to individual reasoning but also to the social dynamics of their configuration, highlighting critical vulnerabilities in AI delegates that mirror the psychological biases observed in human group decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.05995v3">NativQA Framework: Enabling LLMs and VLMs with Native, Local, and Everyday Knowledge</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 LLMs, Native, Multilingual, Language Diversity, Contextual Understanding, Minority Languages, Culturally Informed, Foundation Models, Large Language Models
    </div>
    <details class="paper-abstract">
      The rapid progress of large language models (LLMs) raises concerns about cultural bias, fairness, and performance in diverse languages and underrepresented regions. Addressing these gaps requires large-scale resources grounded in multilingual, local, and cultural contexts. We systematize and extend the earlier NativQA framework to multimodality by adding image, audio, and video support, enabling scalable construction of culturally and regionally aligned QA datasets in native languages. Given user-defined seed queries, the framework uses search engines to collect location-specific everyday information. We evaluate it across 39 locations in 24 countries and 7 languages, spanning extremely low-resource to high-resource settings, and collect over $\sim$300K text QA pairs, $\sim$312K images, and $\sim$29K videos with associated audio. The developed resources can be used for LLMs benchmarking and further fine-tuning. The framework has been made publicly available for the community (https://gitlab.com/nativqa/nativqa-framework). Demo video is available here: \href{https://shorturl.at/DAVn9}{https://shorturl.at/DAVn9}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06071v1">Stories of Your Life as Others: A Round-Trip Evaluation of LLM-Generated Life Stories Conditioned on Rich Psychometric Profiles</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 Under review at COLM
    </div>
    <details class="paper-abstract">
      Personality traits are richly encoded in natural language, and large language models (LLMs) trained on human text can simulate personality when conditioned on persona descriptions. However, existing evaluations rely predominantly on questionnaire self-report by the conditioned model, are limited in architectural diversity, and rarely use real human psychometric data. Without addressing these limitations, it remains unclear whether personality conditioning produces psychometrically informative representations of individual differences or merely superficial alignment with trait descriptors. To test how robustly LLMs can encode personality into extended text, we condition LLMs on real psychometric profiles from 290 participants to generate first-person life story narratives, and then task independent LLMs to recover personality scores from those narratives alone. We show that personality scores can be recovered from the generated narratives at levels approaching human test-retest reliability (mean r = 0.750, 85% of the human ceiling), and that recovery is robust across 10 LLM narrative generators and 3 LLM personality scorers spanning 6 providers. Decomposing systematic biases reveals that scoring models achieve their accuracy while counteracting alignment-induced defaults. Content analysis of the generated narratives shows that personality conditioning produces behaviourally differentiated text: nine of ten coded features correlate significantly with the same features in participants' real conversations, and personality-driven emotional reactivity patterns in narratives replicate in real conversational data. These findings provide evidence that the personality-language relationship captured during pretraining supports robust encoding and decoding of individual differences, including characteristic emotional variability patterns that replicate in real human behaviour.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06066v1">From Hallucination to Structure Snowballing: The Alignment Tax of Constrained Decoding in LLM Reflection</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      Intrinsic self-correction in Large Language Models (LLMs) frequently fails in open-ended reasoning tasks due to ``hallucination snowballing,'' a phenomenon in which models recursively justify early errors during free-text reflection. While structured feedback can mitigate this issue, existing approaches often rely on externally trained critics or symbolic tools, reducing agent autonomy. This study investigates whether enforcing structured reflection purely through Outlines-based constrained decoding can disrupt error propagation without additional training. Evaluating an 8-billion-parameter model (Qwen3-8B), we show that simply imposing structural constraints does not improve self-correction performance. Instead, it triggers a new failure mode termed ``structure snowballing.'' We find that the cognitive load required to satisfy strict formatting rules pushes the model into formatting traps. This observation helps explain why the agent achieves near-perfect superficial syntactic alignment yet fails to detect or resolve deeper semantic errors. These findings expose an ``alignment tax'' inherent to constrained decoding, highlighting a tension between structural granularity and internal model capacity in autonomous workflows. Code and raw logs are available in the GitHub repository: https://github.com/hongxuzhou/agentic_llm_structured_self_critique.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15424v2">LLM-MemCluster: Empowering Large Language Models with Dynamic Memory for Text Clustering</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are reshaping unsupervised learning by offering an unprecedented ability to perform text clustering based on their deep semantic understanding. However, their direct application is fundamentally limited by a lack of stateful memory for iterative refinement and the difficulty of managing cluster granularity. As a result, existing methods often rely on complex pipelines with external modules, sacrificing a truly end-to-end approach. We introduce LLM-MemCluster, a novel framework that reconceptualizes clustering as a fully LLM-native task. It leverages a Dynamic Memory to instill state awareness and a Dual-Prompt Strategy to enable the model to reason about and determine the number of clusters. Evaluated on several benchmark datasets, our tuning-free framework significantly and consistently outperforms strong baselines. LLM-MemCluster presents an effective, interpretable, and truly end-to-end paradigm for LLM-based text clustering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06013v1">Epistemic Blinding: An Inference-Time Protocol for Auditing Prior Contamination in LLM-Assisted Analysis</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 code and LLM skill at: https://github.com/mcuccarese/epistemic-blinding 7 pages 3 figures
    </div>
    <details class="paper-abstract">
      This paper presents epistemic blinding in the context of an agentic system that uses large language models to reason across multiple biological datasets for drug target prioritization. During development, it became apparent that LLM outputs silently blend data-driven inference with memorized priors about named entities - and the blend is invisible: there is no way to determine, from a single output, how much came from the data on the page and how much came from the model's training memory. Epistemic blinding is a simple inference-time protocol that replaces entity identifiers with anonymous codes before prompting, then compares outputs against an unblinded control. The protocol does not make LLM reasoning deterministic, but it restores one critical axis of auditability: measuring how much of an output came from the supplied data versus the model's parametric knowledge. The complete target identification system is described - including LLM-guided evolutionary optimization of scoring functions and blinded agentic reasoning for target rationalization - with demonstration that both stages operate without access to entity identity. In oncology drug target prioritization across four cancer types, blinding changes 16% of top-20 predictions while preserving identical recovery of validated targets. The contamination problem is shown to generalize beyond biology: in S&P 500 equity screening, brand-recognition bias reshapes 30-40% of top-20 rankings across five random seeds. To lower the barrier to adoption, the protocol is released as an open-source tool and as a Claude Code skill that enables one-command epistemic blinding within agentic workflows. The claim is not that blinded analysis produces better results, but that without blinding, there is no way to know to what degree the agent is adhering to the analytical process the researcher designed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06008v1">Designing Around Stigma: Human-Centered LLMs for Menstrual Health</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 This is accepted at CHI 2026
    </div>
    <details class="paper-abstract">
      Menstrual health education (MHE) in Pakistan is constrained by cultural taboos and inadequate formal curricula, leaving women with few trusted resources to lean on. In response to these challenges, we introduce a WhatsApp-based chatbot powered by a large language model (LLM) and Retrieval Augmented Generation (RAG), co-designed with Pakistani college women. Workshops (N=30) revealed key design requirements -- support for Roman Urdu, use of subsidized platforms, and an expert -- curated knowledge base. We then deployed the chatbot with 13 participants for two weeks (403 messages and interviews). Women used it to challenge cultural taboos, legitimize health concerns often dismissed as normal, and build reproductive health knowledge through iterative questioning. Yet, interactions also exposed tensions: reliance on cultural explanatory models, questions of trust and validation, and gendered persona of the chatbot itself. We contribute empirical insights, a stigma-aware design framework for culturally sensitive conversational AI, and a methodological lens foregrounding expert validation in intimate health domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2408.07291v4">Evaluating LLM-based Personal Information Extraction and Countermeasures</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 USENIX Security Symposium 2025
    </div>
    <details class="paper-abstract">
      Automatically extracting personal information -- such as name, phone number, and email address -- from publicly available profiles at a large scale is a stepstone to many other security attacks including spear phishing. Traditional methods -- such as regular expression, keyword search, and entity detection -- achieve limited success at such personal information extraction. In this work, we perform a systematic measurement study to benchmark large language model (LLM) based personal information extraction and countermeasures. Towards this goal, we present a framework for LLM-based extraction attacks; collect four datasets including a synthetic dataset generated by GPT-4 and three real-world datasets with manually labeled eight categories of personal information; introduce a novel mitigation strategy based on prompt injection; and systematically benchmark LLM-based attacks and countermeasures using ten LLMs and five datasets. Our key findings include: LLM can be misused by attackers to accurately extract various personal information from personal profiles; LLM outperforms traditional methods; and prompt injection can defend against strong LLM-based attacks, reducing the attack to less effective traditional ones.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05965v1">Beyond Compromise: Pareto-Lenient Consensus for Efficient Multi-Preference LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      Transcending the single-preference paradigm, aligning LLMs with diverse human values is pivotal for robust deployment. Contemporary Multi-Objective Preference Alignment (MPA) approaches predominantly rely on static linear scalarization or rigid gradient projection to navigate these trade-offs. However, by enforcing strict conflict avoidance or simultaneous descent, these paradigms often prematurely converge to local stationary points. While mathematically stable, these points represent a conservative compromise where the model sacrifices potential global Pareto improvements to avoid transient local trade-offs. To break this deadlock, we propose Pareto-Lenient Consensus (PLC), a game-theoretic framework that reimagines alignment as a dynamic negotiation process. Unlike rigid approaches, PLC introduces consensus-driven lenient gradient rectification, which dynamically tolerates local degradation provided there is a sufficient dominant coalition surplus, thereby empowering the optimization trajectory to escape local suboptimal equilibrium and explore the distal Pareto-optimal frontier. Theoretical analysis validates PLC can facilitate stalemate escape and asymptotically converge to a Pareto consensus equilibrium. Moreover, extensive experiments show that PLC surpasses baselines in both fixed-preference alignment and global Pareto frontier quality. This work highlights the potential of negotiation-driven alignment as a promising avenue for MPA. Our codes are available at https://anonymous.4open.science/r/aaa-6BB8.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.21357v2">AgentHER: Hindsight Experience Replay for LLM Agent Trajectory Relabeling</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      LLM agents fail on the majority of real-world tasks -- GPT-4o succeeds on fewer than 15% of WebArena navigation tasks and below 55% pass@1 on ToolBench (Zhou et al., 2024; Qin et al., 2024) -- yet every failed trajectory is routinely discarded, wasting the dominant source of collected experience. We introduce AgentHER, a framework that recovers this lost training signal by adapting the Hindsight Experience Replay (HER; Andrychowicz et al., 2017) principle to natural-language agent trajectories for offline data augmentation. The key insight is simple: a trajectory that fails goal A is often a correct demonstration for some achievable alternative goal B. AgentHER realises this idea through a four-stage pipeline -- failure classification, outcome extraction, LLM-guided prompt relabeling with confidence gating, and data packaging -- that converts discarded failures into high-quality SFT, DPO, and ShareGPT training data, with both zero-cost rule-based and LLM-judge implementations. On WebArena (Zhou et al., 2024) and ToolBench (Qin et al., 2024), AgentHER improves over success-only SFT by +7.1-11.7 pp across four model families (GPT-4o, Qwen2.5-72B/7B, LLaMA-3.1-8B), while achieving 2x data efficiency -- matching baseline performance with only 50% of successful demonstrations. Gains are consistent from 1.5B to 72B parameters (+5.8-9.2 pp) and compound under iterative redeployment (+2.1 pp over additional rounds). Human evaluation confirms 97.7% relabeling precision under multi-judge verification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05955v1">Does Pass Rate Tell the Whole Story? Evaluating Design Constraint Compliance in LLM-based Issue Resolution</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      Repository-level issue resolution benchmarks have become a standard testbed for evaluating LLM-based agents, yet success is still predominantly measured by test pass rates. In practice, however, acceptable patches must also comply with project-specific design constraints, such as architectural conventions, error-handling policies, and maintainability requirements, which are rarely encoded in tests and are often documented only implicitly in code review discussions. This paper introduces \textit{design-aware issue resolution} and presents \bench{}, a benchmark that makes such implicit design constraints explicit and measurable. \bench{} is constructed by mining and validating design constraints from real-world pull requests, linking them to issue instances, and automatically checking patch compliance using an LLM-based verifier, yielding 495 issues and 1,787 validated constraints across six repositories, aligned with SWE-bench-Verified and SWE-bench-Pro. Experiments with state-of-the-art agents show that test-based correctness substantially overestimates patch quality: fewer than half of resolved issues are fully design-satisfying, design violations are widespread, and functional correctness exhibits negligible statistical association with design satisfaction. While providing issue-specific design guidance reduces violations, substantial non-compliance remains, highlighting a fundamental gap in current agent capabilities and motivating design-aware evaluation beyond functional correctness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05942v1">BOSCH: Black-Box Binary Optimization for Short-Context Attention-Head Selection in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 ACL 2026 (Main Conference)
    </div>
    <details class="paper-abstract">
      Post-training hybridization of large language models (LLMs) often replaces quadratic self-attention with sliding-window attention (SWA) to reduce KV cache usage and improve latency. Existing hybridization schemes are typically defined either at the layer level (e.g., interleaving) or at the head level via static rankings from local to global. Layer-level schemes ignore that local and global dependencies are routed through heads within the same layer, while static head-level rankings suffer from entanglement: a head's local/global behavior can change after hybridization. We propose BOSCH, Black-box Binary Optimization for Short-context Head Selection, a training-free method that formulates the problem as a Large Neighborhood Search and decomposes it into three subproblems: (i) layer-importance detection via small-budget black-box probes, (ii) adaptive per-layer SWA-ratio assignment based on these sensitivities, and (iii) grouped head-level optimization within ratio buckets. Extensive experiments on 4 LLMs ranging from 1.7B to 30B parameters, across 4 SWA ratios, show that BOSCH consistently outperforms layer-level heuristics and 6 strong static head-level methods, with larger gains at higher SWA ratios. Under continual pretraining, BOSCH recover original long-context performance faster and to a higher level. Analysis of the selected heads reveals substantial turnover for BOSCH across different SWA ratios, underscoring the importance of performing head-level selection for each target ratio rather than relying on fixed locality rankings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.19894v5">TransAgent: Enhancing LLM-Based Code Translation via Fine-Grained Execution Alignment</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 Accepted by FSE'26
    </div>
    <details class="paper-abstract">
      Code translation transforms code between programming languages while preserving functionality, which is critical in software development and maintenance. While traditional learning-based code translation methods have limited effectiveness due to the lack of sufficient parallel training data, Large Language Models (LLMs) have recently advanced this field with their strong code generation and comprehension capabilities. However, code translated by LLMs still suffers from diverse quality issues, such as syntax and semantic errors. In this work, we propose TransAGENT, a novel multi-agent system that eliminates the errors during LLM-based code translation. The main insight of TransAGENT is to localize error-prone code blocks via fine-grained execution alignment between source and target code. We evaluate TransAGENT on a newly constructed benchmark of recent programming tasks to mitigate data leakage. TransAGENT outperforms the latest UniTrans by up to 33.3% in translation accuracy and achieves an average improvement of 56.7% over Agentless in program repair performance. We also conduct an ablation study and evaluate TransAGENT across different LLMs, demonstrating its effectiveness and strong generalizability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05872v1">Swiss-Bench 003: Evaluating LLM Reliability and Adversarial Security for Swiss Regulatory Contexts</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 23 pages, 5 figures, 8 tables
    </div>
    <details class="paper-abstract">
      The deployment of large language models (LLMs) in Swiss financial and regulatory contexts demands empirical evidence of both production reliability and adversarial security, dimensions not jointly operationalized in existing Swiss-focused evaluation frameworks. This paper introduces Swiss-Bench 003 (SBP-003), extending the HAAS (Helvetic AI Assessment Score) from six to eight dimensions by adding D7 (Self-Graded Reliability Proxy) and D8 (Adversarial Security). I evaluate ten frontier models across 808 Swiss-specific items in four languages (German, French, Italian, English), comprising seven Swiss-adapted benchmarks (Swiss TruthfulQA, Swiss IFEval, Swiss SimpleQA, Swiss NIAH, Swiss PII-Scope, System Prompt Leakage, and Swiss German Comprehension) targeting FINMA Guidance 08/2024, the revised Federal Act on Data Protection (nDSG), and OWASP Top 10 for LLMs. Self-graded D7 scores (73-94%) exceed externally judged D8 security scores (20-61%) by a wide margin, though these dimensions use non-comparable scoring regimes. System prompt leakage resistance ranges from 24.8% to 88.2%, while PII extraction defense remains weak (14-42%) across all models. Qwen 3.5 Plus achieves the highest self-graded D7 score (94.4%), while GPT-oss 120B achieves the highest D8 score (60.7%) despite being the lowest-cost model evaluated. All evaluations are zero-shot under provider default settings; D7 is self-graded and does not constitute independently validated accuracy. I provide conceptual mapping tables relating benchmark dimensions to FINMA model validation requirements, nDSG data protection obligations, and OWASP LLM risk categories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05859v1">When Do We Need LLMs? A Diagnostic for Language-Driven Bandits</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 ICLR 2026 Workshop on AI Advances in Finance
    </div>
    <details class="paper-abstract">
      We study Contextual Multi-Armed Bandits (CMABs) for non-episodic sequential decision making problems where the context includes both textual and numerical information (e.g., recommendation systems, dynamic portfolio adjustments, offer selection; all frequent problems in finance). While Large Language Models (LLMs) are increasingly applied to these settings, utilizing LLMs for reasoning at every decision step is computationally expensive and uncertainty estimates are difficult to obtain. To address this, we introduce LLMP-UCB, a bandit algorithm that derives uncertainty estimates from LLMs via repeated inference. However, our experiments demonstrate that lightweight numerical bandits operating on text embeddings (dense or Matryoshka) match or exceed the accuracy of LLM-based solutions at a fraction of their cost. We further show that embedding dimensionality is a practical lever on the exploration-exploitation balance, enabling cost--performance tradeoffs without prompt complexity. Finally, to guide practitioners, we propose a geometric diagnostic based on the arms' embedding to decide when to use LLM-driven reasoning versus a lightweight numerical bandit. Our results provide a principled deployment framework for cost-effective, uncertainty-aware decision systems with broad applicability across AI use cases in financial services.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05846v1">AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 ACL 2026 Main Conference
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly rely on agentic capabilities-iterative retrieval, tool use, and decision-making-to overcome the limits of static, parametric knowledge. Yet existing agentic frameworks treat external information as unstructured text and fail to leverage the topological dependencies inherent in real-world data. To bridge this gap, we introduce Agentic Graph Learning (AGL), a paradigm that reframes graph learning as an interleaved process of topology-aware navigation and LLM-based inference. Specifically, we propose AgentGL, the first reinforcement learning (RL)-driven framework for AGL. AgentGL equips an LLM agent with graph-native tools for multi-scale exploration, regulates tool usage via search-constrained thinking to balance accuracy and efficiency, and employs a graph-conditioned curriculum RL strategy to stabilize long-horizon policy learning without step-wise supervision. Across diverse Text-Attributed Graph (TAG) benchmarks and multiple LLM backbones, AgentGL substantially outperforms strong GraphLLMs and GraphRAG baselines, achieving absolute improvements of up to 17.5% in node classification and 28.4% in link prediction. These results demonstrate that AGL is a promising frontier for enabling LLMs to autonomously navigate and reason over complex relational environments. The code is publicly available at https://github.com/sunyuanfu/AgentGL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05808v1">Hierarchical Reinforcement Learning with Augmented Step-Level Transitions for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 Accepted to ACL 2026 Main Conference
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have demonstrated strong capabilities in complex interactive decision-making tasks. However, existing LLM agents typically rely on increasingly long interaction histories, resulting in high computational cost and limited scalability. In this paper, we propose STEP-HRL, a hierarchical reinforcement learning (HRL) framework that enables step-level learning by conditioning only on single-step transitions rather than full interaction histories. STEP-HRL structures tasks hierarchically, using completed subtasks to represent global progress of overall task. By introducing a local progress module, it also iteratively and selectively summarizes interaction history within each subtask to produce a compact summary of local progress. Together, these components yield augmented step-level transitions for both high-level and low-level policies. Experimental results on ScienceWorld and ALFWorld benchmarks consistently demonstrate that STEP-HRL substantially outperforms baselines in terms of performance and generalization while reducing token usage. Our code is available at https://github.com/TonyStark042/STEP-HRL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05793v1">BodhiPromptShield: Pre-Inference Prompt Mediation for Suppressing Privacy Propagation in LLM/VLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      In LLM/VLM agents, prompt privacy risk propagates beyond a single model call because raw user content can flow into retrieval queries, memory writes, tool calls, and logs. Existing de-identification pipelines address document boundaries but not this cross-stage propagation. We propose BodhiPromptShield, a policy-aware framework that detects sensitive spans, routes them via typed placeholders, semantic abstraction, or secure symbolic mapping, and delays restoration to authorized boundaries. Relative to enterprise redaction, this adds explicit propagation-aware mediation and restoration timing as a security variable. Under controlled evaluation on the Controlled Prompt-Privacy Benchmark (CPPB), stage-wise propagation suppresses from 10.7\% to 7.1\% across retrieval, memory, and tool stages; PER reaches 9.3\% with 0.94 AC and 0.92 TSR, outperforming generic de-identification. These are controlled systems results on CPPB rather than formal privacy guarantees or public-benchmark transfer claims. The project repository is available at https://github.com/mabo1215/BodhiPromptShield.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05782v1">An Empirical Study of Perceptions of General LLMs and Multimodal LLMs on Hugging Face</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have rapidly evolved from general-purpose systems to multimodal models capable of processing text, images, and audio. As both general-purpose LLMs (GLLMs) and multimodal LLMs (MLLMs) gain widespread adoption, understanding user perceptions in real-world settings becomes increasingly important. However, existing studies often rely on surveys or platform-specific data (e.g., Reddit or GitHub issues), which either constrain user feedback through predefined questions or overemphasize failure-driven, debugging-oriented discussions, thus failing to capture diverse, experience-driven, and cross-model user perspectives in practice. To address this issue, we conduct an empirical study of user discussions on Hugging Face, a major model hub with diverse models and active communities. We collect and manually annotate 662 discussion threads from 38 representative models (21 GLLMs and 17 MLLMs), and develop a three-level taxonomy to systematically characterize user concerns. Our analysis reveals that LLM access barriers, generation quality, and deployment and invocation complexity are the most prominent concerns, alongside issues such as documentation limitations and resource constraints. Based on these findings, we derive actionable implications for improving LLM ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.14226v5">Phonetic Perturbations Reveal Tokenizer-Rooted Safety Gaps in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      Safety-aligned LLMs remain vulnerable to digital phenomena like textese that introduce non-canonical perturbations to words but preserve the phonetics. We introduce CMP-RT (code-mixed phonetic perturbations for red-teaming), a novel diagnostic probe that pinpoints tokenization as the root cause of this vulnerability. A mechanistic analysis reveals that phonetic perturbations fragment safety-critical tokens into benign sub-words, suppressing their attribution scores while preserving prompt interpretability -- causing safety mechanisms to fail despite excellent input understanding. We demonstrate that this vulnerability evades standard defenses, persists across modalities and state-of-the-art (SOTA) models including Gemini-3-Pro, and scales through simple supervised fine-tuning (SFT). Furthermore, layer-wise probing shows perturbed and canonical input representations align up to a critical layer depth; enforcing output equivalence robustly recovers the lost representations, providing causal evidence for a structural gap between pre-training and alignment, and establishing tokenization as a critical, under-examined vulnerability in current safety pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05775v1">PhageBench: Can LLMs Understand Raw Bacteriophage Genomes?</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      Bacteriophages, often referred to as the dark matter of the biosphere, play a critical role in regulating microbial ecosystems and in antibiotic alternatives. Thus, accurate interpretation of their genomes holds significant scientific and practical value. While general-purpose Large Language Models (LLMs) excel at understanding biological texts, their ability to directly interpret raw nucleotide sequences and perform biological reasoning remains underexplored. To address this, we introduce PhageBench, the first benchmark designed to evaluate phage genome understanding by mirroring the workflow of bioinformatics experts. The dataset contains 5,600 high-quality samples covering five core tasks across three stages: Screening, Quality Control, and Phenotype Annotation. Our evaluation of eight LLMs reveals that general-purpose reasoning models significantly outperform random baselines in phage contig identification and host prediction, demonstrating promising potential for genomic understanding. However, they exhibit significant limitations in complex reasoning tasks involving long-range dependencies and fine-grained functional localization. These findings highlight the necessity of developing next-generation models with enhanced reasoning capabilities for biological sequences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05766v1">The LLM Effect on IR Benchmarks: A Meta-Analysis of Effectiveness, Baselines, and Contamination</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 Accepted at SIGIR 2026
    </div>
    <details class="paper-abstract">
      Benchmark collections have long enabled controlled comparison and cumulative progress in Information Retrieval (IR). However, prior meta-analyses have shown that reported effectiveness gains often fail to accumulate, in part due to the use of weak or outdated baselines. While large language models are increasingly used in retrieval pipelines, their impact on established IR benchmarks has not been systematically analyzed. In this study, we analyze 143 publications reporting results on the TREC Robust04 collection and the TREC Deep Learning 2020 (DL20) passage retrieval benchmark to examine longitudinal trends in retrieval effectiveness and baseline strength. We observe what we term an \emph{LLM effect}: recent systems incorporating LLM components achieve 8.8\% higher nDCG@10 on DL20 compared to the best result from TREC 2020 and approximately 20\% higher on Robust04 since 2023. However, adapting a data contamination detection approach to reranking reveals measurable contamination in both benchmarks. While excluding contaminated topics reduces effectiveness, confidence intervals remain wide, making it difficult to determine whether the LLM effect reflects genuine methodological advances or memorization from pretraining data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05756v1">Controlling Distributional Bias in Multi-Round LLM Generation via KL-Optimized Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 Accepted at ACL Main Conference
    </div>
    <details class="paper-abstract">
      While the real world is inherently stochastic, Large Language Models (LLMs) are predominantly evaluated on single-round inference against fixed ground truths. In this work, we shift the lens to distribution alignment: assessing whether LLMs, when prompted repeatedly, can generate outputs that adhere to a desired target distribution, e.g. reflecting real-world statistics or a uniform distribution. We formulate distribution alignment using the attributes of gender, race, and sentiment within occupational contexts. Our empirical analysis reveals that off-the-shelf LLMs and standard alignment techniques, including prompt engineering and Direct Preference Optimization, fail to reliably control output distributions. To bridge this gap, we propose a novel fine-tuning framework that couples Steering Token Calibration with Semantic Alignment. We introduce a hybrid objective function combining Kullback-Leibler divergence to anchor the probability mass of latent steering tokens and Kahneman-Tversky Optimization to bind these tokens to semantically consistent responses. Experiments across six diverse datasets demonstrate that our approach significantly outperforms baselines, achieving precise distributional control in attribute generation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05719v1">Hackers or Hallucinators? A Comprehensive Analysis of LLM-Based Automated Penetration Testing</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has created new opportunities for Automated Penetration Testing (AutoPT), spawning numerous frameworks aimed at achieving end-to-end autonomous attacks. However, despite the proliferation of related studies, existing research generally lacks systematic architectural analysis and large-scale empirical comparisons under a unified benchmark. Therefore, this paper presents the first Systematization of Knowledge (SoK) focusing on the architectural design and comprehensive empirical evaluation of current LLM-based AutoPT frameworks. At systematization level, we comprehensively review existing framework designs across six dimensions: agent architecture, agent plan, agent memory, agent execution, external knowledge, and benchmarks. At empirical level, we conduct large-scale experiments on 13 representative open-source AutoPT frameworks and 2 baseline frameworks utilizing a unified benchmark. The experiments consumed over 10 billion tokens in total and generated more than 1,500 execution logs, which were manually reviewed and analyzed over four months by a panel of more than 15 researchers with expertise in cybersecurity. By investigating the latest progress in this rapidly developing field, we provide researchers with a structured taxonomy to understand existing LLM-based AutoPT frameworks and a large-scale empirical benchmark, along with promising directions for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05681v1">LUDOBENCH: Evaluating LLM Behavioural Decision-Making Through Spot-Based Board Game Scenarios in Ludo</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      We introduce LudoBench, a benchmark for evaluating LLM strategic reasoning in Ludo, a stochastic multi-agent board game whose dice mechanics, piece capture, safe-square navigation, and home-path progression introduce meaningful planning complexity. LudoBench comprises 480 handcrafted spot scenarios across 12 behaviorally distinct decision categories, each isolating a specific strategic choice. We additionally contribute a fully functional 4-player Ludo simulator supporting Random, Heuristic, Game-Theory, and LLM agents. The game-theory agent uses Expectiminimax search with depth-limited lookahead to provide a principled strategic ceiling beyond greedy heuristics. Evaluating six models spanning four model families, we find that all models agree with the game-theory baseline only 40-46% of the time. Models split into distinct behavioral archetypes: finishers that complete pieces but neglect development, and builders that develop but never finish. Each archetype captures only half of the game theory strategy. Models also display measurable behavioral shifts under history-conditioned grudge framing on identical board states, revealing prompt-sensitivity as a key vulnerability. LudoBench provides a lightweight and interpretable framework for benchmarking LLM strategic reasoning under uncertainty. All code, the spot dataset (480 entries) and model outputs are available at https://anonymous.4open.science/r/LudoBench-5CBF/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05674v1">From Incomplete Architecture to Quantified Risk: Multimodal LLM-Driven Security Assessment for Cyber-Physical Systems</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 Under submission
    </div>
    <details class="paper-abstract">
      Cyber-physical systems often contend with incomplete architectural documentation or outdated information resulting from legacy technologies, knowledge management gaps, and the complexity of integrating diverse subsystems over extended operational lifecycles. This architectural incompleteness impedes reliable security assessment, as inaccurate or missing architectural knowledge limits the identification of system dependencies, attack surfaces, and risk propagation pathways. To address this foundational challenge, this paper introduces ASTRAL (Architecture-Centric Security Threat Risk Assessment using LLMs), an architecture-centric security assessment technique implemented in a prototype tool powered by multimodal LLMs. The proposed approach assists practitioners in reconstructing and analysing CPS architectures when documentation is fragmented or absent. By leveraging prompt chaining, few-shot learning, and architectural reasoning, ASTRAL extracts and synthesises system representations from disparate data sources. By integrating LLM reasoning with architectural modelling, our approach supports adaptive threat identification and quantitative risk estimation for cyber-physical systems. We evaluated the approach through an ablation study across multiple CPS case studies and an expert evaluation involving 14 experienced cybersecurity practitioners. Practitioner feedback suggests that ASTRAL is useful and reliable for supporting architecture-centric security assessment. Overall, the results indicate that the approach can support more informed cyber risk management decisions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05663v1">CuraLight: Debate-Guided Data Curation for LLM-Centered Traffic Signal Control</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 accepted at IJCNN 2026
    </div>
    <details class="paper-abstract">
      Traffic signal control (TSC) is a core component of intelligent transportation systems (ITS), aiming to reduce congestion, emissions, and travel time. Recent approaches based on reinforcement learning (RL) and large language models (LLMs) have improved adaptivity, but still suffer from limited interpretability, insufficient interaction data, and weak generalization to heterogeneous intersections. This paper proposes CuraLight, an LLM-centered framework where an RL agent assists the fine-tuning of an LLM-based traffic signal controller. The RL agent explores traffic environments and generates high-quality interaction trajectories, which are converted into prompt-response pairs for imitation fine-tuning. A multi-LLM ensemble deliberation system further evaluates candidate signal timing actions through structured debate, providing preference-aware supervision signals for training. Experiments conducted in SUMO across heterogeneous real-world networks from Jinan, Hangzhou, and Yizhuang demonstrate that CuraLight consistently outperforms state-of-the-art baselines, reducing average travel time by 5.34 percent, average queue length by 5.14 percent, and average waiting time by 7.02 percent. The results highlight the effectiveness of combining RL-assisted exploration with deliberation-based data curation for scalable and interpretable traffic signal control.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05655v1">LLM Reasoning as Trajectories: Step-Specific Representation Geometry and Correctness Signals</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 ACL 2026 (Main)
    </div>
    <details class="paper-abstract">
      This work characterizes large language models' chain-of-thought generation as a structured trajectory through representation space. We show that mathematical reasoning traverses functionally ordered, step-specific subspaces that become increasingly separable with layer depth. This structure already exists in base models, while reasoning training primarily accelerates convergence toward termination-related subspaces rather than introducing new representational organization. While early reasoning steps follow similar trajectories, correct and incorrect solutions diverge systematically at late stages. This late-stage divergence enables mid-reasoning prediction of final-answer correctness with ROC-AUC up to 0.87. Furthermore, we introduce trajectory-based steering, an inference-time intervention framework that enables reasoning correction and length control based on derived ideal trajectories. Together, these results establish reasoning trajectories as a geometric lens for interpreting, predicting, and controlling LLM reasoning behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05643v1">Graph-Based Chain-of-Thought Pruning for Reducing Redundant Reflections in Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      Extending CoT through RL has been widely used to enhance the reasoning capabilities of LLMs. However, due to the sparsity of reward signals, it can also induce undesirable thinking patterns such as overthinking, i.e., generating redundant intermediate reasoning content. In this work, we argue that a major source of such redundancy is inefficient reflection, which often manifests in two problematic patterns: Indiscriminate Reflection, where the model performs broad, low-impact checks throughout reasoning, and Repetitive Reflection, where it repeatedly re-verifies an already established conclusion. To address this, we introduce a graph-based CoT optimization framework. Specifically, we convert each linear CoT into a directed acyclic graph (DAG) with explicit dependency edges, and design a dual pruning strategy: branch-level pruning removes weakly contributing reflection branches, while depth-level pruning eliminates late-stage re-verification. We distill this behavior via a three-stage pipeline: (1) SFT to initialize the policy on pruned concise traces, (2) DPO to prefer correct but less redundant trajectories, and (3) GRPO with length penalty to jointly optimize answer correctness and efficiency. Experiments show that our approach reduces the average reasoning tokens by 42\% while maintaining or improving accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.13151v3">Quantization-Robust LLM Unlearning via Low-Rank Adaptation</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 Accepted to IJCNN 2026
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) unlearning aims to remove targeted knowledge from a trained model, but practical deployments often require post-training quantization (PTQ) for efficient inference. However, aggressive low-bit PTQ can mask unlearning updates, causing quantized models to revert to pre-unlearning behavior. We show that standard full-parameter fine-tuning often induces parameter changes that are too small to survive 4-bit quantization. We propose quantization-robust unlearning via low-rank adaptation (LoRA): we freeze the base model and concentrate unlearning into trainable adapters so that the effective update is preserved after quantization. On Llama-2-7B evaluated with MUSE dataset (BOOKS and NEWS), LoRA improves 4-bit utility by up to 7.93 points (NPO+GDR on BOOKS: 50.17 to 58.10) and yields higher 4-bit utility on NEWS for GA+GDR (40.06 to 44.82, increase of 4.76). LoRA also substantially reduces privacy leakage under 4-bit PTQ, e.g., for GA+KLR on BOOKS, PrivLeak moves from -25.68 to -5.86 (closer to ideal 0), while maintaining strong forgetting (VerMem and KnowMem near 0). Thus, using LoRA for Machine Unlearning is beneficial for scenarios where quantization is necessary for model deployment.
    </details>
</div>
