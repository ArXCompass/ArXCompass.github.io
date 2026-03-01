# llm - 2026_02

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.23331v1">Utilizing LLMs for Industrial Process Automation</a></div>
    <div class="paper-meta">
      📅 2026-02-26
    </div>
    <details class="paper-abstract">
      A growing number of publications address the best practices to use Large Language Models (LLMs) for software engineering in recent years. However, most of this work focuses on widely-used general purpose programming languages like Python due to their widespread usage training data. The utility of LLMs for software within the industrial process automation domain, with highly-specialized languages that are typically only used in proprietary contexts, remains underexplored. This research aims to utilize and integrate LLMs in the industrial development process, solving real-life programming tasks (e.g., generating a movement routine for a robotic arm) and accelerating the development cycles of manufacturing systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.23330v1">Toward Expert Investment Teams:A Multi-Agent LLM System with Fine-Grained Trading Tasks</a></div>
    <div class="paper-meta">
      📅 2026-02-26
      | 💬 14 pages, 3 figures
    </div>
    <details class="paper-abstract">
      The advancement of large language models (LLMs) has accelerated the development of autonomous financial trading systems. While mainstream approaches deploy multi-agent systems mimicking analyst and manager roles, they often rely on abstract instructions that overlook the intricacies of real-world workflows, which can lead to degraded inference performance and less transparent decision-making. Therefore, we propose a multi-agent LLM trading framework that explicitly decomposes investment analysis into fine-grained tasks, rather than providing coarse-grained instructions. We evaluate the proposed framework using Japanese stock data, including prices, financial statements, news, and macro information, under a leakage-controlled backtesting setting. Experimental results show that fine-grained task decomposition significantly improves risk-adjusted returns compared to conventional coarse-grained designs. Crucially, further analysis of intermediate agent outputs suggests that alignment between analytical outputs and downstream decision preferences is a critical driver of system performance. Moreover, we conduct standard portfolio optimization, exploiting low correlation with the stock index and the variance of each system's output. This approach achieves superior performance. These findings contribute to the design of agent structure and task configuration when applying LLM agents to trading systems in practical settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.23329v1">LLM Novice Uplift on Dual-Use, In Silico Biology Tasks</a></div>
    <div class="paper-meta">
      📅 2026-02-26
      | 💬 59 pages, 33 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) perform increasingly well on biology benchmarks, but it remains unclear whether they uplift novice users -- i.e., enable humans to perform better than with internet-only resources. This uncertainty is central to understanding both scientific acceleration and dual-use risk. We conducted a multi-model, multi-benchmark human uplift study comparing novices with LLM access versus internet-only access across eight biosecurity-relevant task sets. Participants worked on complex problems with ample time (up to 13 hours for the most involved tasks). We found that LLM access provided substantial uplift: novices with LLMs were 4.16 times more accurate than controls (95% CI [2.63, 6.87]). On four benchmarks with available expert baselines (internet-only), novices with LLMs outperformed experts on three of them. Perhaps surprisingly, standalone LLMs often exceeded LLM-assisted novices, indicating that users were not eliciting the strongest available contributions from the LLMs. Most participants (89.6%) reported little difficulty obtaining dual-use-relevant information despite safeguards. Overall, LLMs substantially uplift novices on biological tasks previously reserved for trained practitioners, underscoring the need for sustained, interactive uplift evaluations alongside traditional benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.12522v2">Evaluating the Diversity and Quality of LLM Generated Content</a></div>
    <div class="paper-meta">
      📅 2026-02-26
      | 💬 Published at COLM 2025
    </div>
    <details class="paper-abstract">
      Recent work suggests that preference-tuning techniques -- such as Reinforcement Learning from Human Feedback (RLHF) methods like PPO and GRPO, as well as alternatives like DPO -- reduce diversity, creating a dilemma given that these models are widely deployed in applications requiring varied outputs. We argue that diversity without consideration of quality has limited practical value. To address this issue, we introduce a framework for measuring effective semantic diversity -- diversity among outputs that meet quality thresholds -- which better reflects the practical utility of large language models (LLMs). Using open-ended tasks that require no human intervention, we find counterintuitive results: when using diversity metrics that do not explicitly consider quality, preference-tuned models -- particularly those trained via RL -- often produce outputs with lower diversity; however, these same preference-tuned models generate greater effective semantic diversity than supervised fine-tuned (SFT) or base models. Our analysis further shows another trend: while larger models may exhibit greater effective semantic diversity than smaller models, the smaller models are consistently more parameter-efficient at producing unique content within a fixed sampling budget. These findings have practical implications for applications that require diverse yet high-quality outputs, from creative assistance to synthetic data generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.19060v3">PoSh: Using Scene Graphs To Guide LLMs-as-a-Judge For Detailed Image Descriptions</a></div>
    <div class="paper-meta">
      📅 2026-02-26
      | 💬 Accepted at ICLR 2026. 26 pages, 9 figures. Metric/benchmark available at https://github.com/amith-ananthram/posh
    </div>
    <details class="paper-abstract">
      While vision-language models (VLMs) have advanced into detailed image description, evaluation remains a challenge. Standard metrics (e.g. CIDEr, SPICE) were designed for short texts and tuned to recognize errors that are now uncommon, such as object misidentification. In contrast, long texts require sensitivity to attribute and relation attachments and scores that localize errors to particular text spans. In this work, we introduce PoSh, a metric for detailed image description that uses scene graphs as structured rubrics to guide LLMs-as-a-Judge, producing aggregate scores grounded in fine-grained errors (e.g. mistakes in compositional understanding). PoSh is replicable, interpretable and a better proxy for human raters than existing metrics (including GPT4o-as-a-Judge). To validate PoSh, we introduce a challenging new dataset, DOCENT. This novel benchmark contains artwork, paired with expert-written references, and model-generated descriptions, augmented with granular and coarse judgments of their quality from art history students. Thus, DOCENT enables evaluating both detailed image description metrics and detailed image description itself in a challenging new domain. We show that PoSh achieves stronger correlations (+0.05 Spearman $ρ$) with the human judgments in DOCENT than the best open-weight alternatives, is robust to image type (using CapArena, an existing dataset of web imagery) and is a capable reward function, outperforming standard supervised fine-tuning. Then, using PoSh, we characterize the performance of open and closed models in describing the paintings, sketches and statues in DOCENT and find that foundation models struggle to achieve full, error-free coverage of images with rich scene dynamics, establishing a demanding new task to gauge VLM progress. Through both PoSh and DOCENT, we hope to enable advances in important areas such as assistive text generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.02655v2">BioBlue: Systematic runaway-optimiser-like LLM failure modes on biologically and economically aligned AI safety benchmarks for LLMs with simplified observation format</a></div>
    <div class="paper-meta">
      📅 2026-02-26
      | 💬 22 pages, 8 tables
    </div>
    <details class="paper-abstract">
      Many AI alignment discussions of "runaway optimisation" focus on RL agents: unbounded utility maximisers that over-optimise a proxy objective (e.g., "paperclip maximiser", specification gaming) at the expense of everything else. LLM-based systems are often assumed to be safer because they function as next-token predictors rather than persistent optimisers. In this work, we empirically test this assumption by placing LLMs in simple, long-horizon control-style environments that require maintaining state of or balancing objectives over time: sustainability of a renewable resource, single- and multi-objective homeostasis, and balancing unbounded objectives with diminishing returns. We find that, although models frequently behave appropriately for many steps and clearly understand the stated objectives, they often lose context in structured ways and drift into runaway behaviours: ignoring homeostatic targets, collapsing from multi-objective trade-offs into single-objective maximisation - thus failing to respect concave utility structures. These failures emerge reliably after initial periods of competent behaviour and exhibit characteristic patterns (including self-imitative oscillations, unbounded maximisation, and reverting to single-objective optimisation). The problem is not that the LLMs just lose context or become incoherent - the failures systematically resemble runaway optimisers. Our results suggest that long-horizon, multi-objective misalignment is a genuine and under-evaluated failure mode in LLM agents, even in extremely simple settings with transparent and explicitly multi-objective feedback. Although LLMs appear multi-objective and bounded on the surface, their behaviour under sustained interaction, particularly involving multiple objectives, resembles brittle, poorly aligned optimisers whose effective objective gradually shifts toward unbounded and single-metric maximisation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.05154v4">Can AI Truly Represent Your Voice in Deliberations? A Comprehensive Study of Large-Scale Opinion Aggregation with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-26
    </div>
    <details class="paper-abstract">
      Large-scale public deliberations generate thousands of free-form contributions that must be synthesized into representative and neutral summaries for policy use. While LLMs have been shown as a promising tool to generate summaries for large-scale deliberations, they also risk underrepresenting minority perspectives and exhibiting bias with respect to the input order, raising fairness concerns in high-stakes contexts. Studying and fixing these issues requires a comprehensive evaluation at a large scale, yet current practice often relies on LLMs as judges, which show weak alignment with human judgments. To address this, we present DeliberationBank, a large-scale human-grounded dataset with (1) opinion data spanning ten deliberation questions created by 3,000 participants and (2) summary judgment data annotated by 4,500 participants across four dimensions (representativeness, informativeness, neutrality, policy approval). Using these datasets, we train DeliberationJudge, a fine-tuned DeBERTa model that can rate deliberation summaries from individual perspectives. DeliberationJudge is more efficient and more aligned with human judgements compared to a wide range of LLM judges. With DeliberationJudge, we evaluate 18 LLMs and reveal persistent weaknesses in deliberation summarization, especially underrepresentation of minority positions. Our framework provides a scalable and reliable way to evaluate deliberation summarization, helping ensure AI systems are more representative and equitable for policymaking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.23234v1">Scaling Search Relevance: Augmenting App Store Ranking with LLM-Generated Judgments</a></div>
    <div class="paper-meta">
      📅 2026-02-26
    </div>
    <details class="paper-abstract">
      Large-scale commercial search systems optimize for relevance to drive successful sessions that help users find what they are looking for. To maximize relevance, we leverage two complementary objectives: behavioral relevance (results users tend to click or download) and textual relevance (a result's semantic fit to the query). A persistent challenge is the scarcity of expert-provided textual relevance labels relative to abundant behavioral relevance labels. We first address this by systematically evaluating LLM configurations, finding that a specialized, fine-tuned model significantly outperforms a much larger pre-trained one in providing highly relevant labels. Using this optimal model as a force multiplier, we generate millions of textual relevance labels to overcome the data scarcity. We show that augmenting our production ranker with these textual relevance labels leads to a significant outward shift of the Pareto frontier: offline NDCG improves for behavioral relevance while simultaneously increasing for textual relevance. These offline gains were validated by a worldwide A/B test on the App Store ranker, which demonstrated a statistically significant +0.24% increase in conversion rate, with the most substantial performance gains occurring in tail queries, where the new textual relevance labels provide a robust signal in the absence of reliable behavioral relevance labels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.23220v1">STELLAR: Storage Tuning Engine Leveraging LLM Autonomous Reasoning for High Performance Parallel File Systems</a></div>
    <div class="paper-meta">
      📅 2026-02-26
      | 💬 Published in the Proceedings of the 2025 International Conference for High Performance Computing, Networking, Storage, and Analysis (SC25)
    </div>
    <details class="paper-abstract">
      I/O performance is crucial to efficiency in data-intensive scientific computing; but tuning large-scale storage systems is complex, costly, and notoriously manpower-intensive, making it inaccessible for most domain scientists. To address this problem, we propose STELLAR, an autonomous tuner for high-performance parallel file systems. Our evaluations show that STELLAR almost always selects near-optimal parameter configurations for parallel file systems within the first five attempts, even for previously unseen applications. STELLAR differs fundamentally from traditional autotuning methods, which often require hundreds of thousands of iterations to converge. Powered by large language models (LLMs), STELLAR enables autonomous end-to-end agentic tuning by (1) accurately extracting tunable parameters from software manuals, (2) analyzing I/O trace logs generated by applications, (3) selecting initial tuning strategies, (4) rerunning applications on real systems and collecting I/O performance feedback, (5) adjusting tuning strategies and repeating the tuning cycle, and (6) reflecting on and summarizing tuning experiences into reusable knowledge for future optimizations. STELLAR integrates retrieval-augmented generation (RAG), tool execution, LLM-based reasoning, and a multiagent design to stabilize reasoning and combat hallucinations. We evaluate the impact of each component on optimization outcomes, providing design insights for similar systems in other optimization domains. STELLAR's architecture and empirical results highlight a promising approach to complex system optimization, especially for problems with large search spaces and high exploration costs, while making I/O tuning more accessible to domain scientists with minimal added resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.23193v1">ESAA: Event Sourcing for Autonomous Agents in LLM-Based Software Engineering</a></div>
    <div class="paper-meta">
      📅 2026-02-26
      | 💬 13 pages, 1 figure, 4 tables. Includes 5 technical appendices
    </div>
    <details class="paper-abstract">
      Autonomous agents based on Large Language Models (LLMs) have evolved from reactive assistants to systems capable of planning, executing actions via tools, and iterating over environment observations. However, they remain vulnerable to structural limitations: lack of native state, context degradation over long horizons, and the gap between probabilistic generation and deterministic execution requirements. This paper presents the ESAA (Event Sourcing for Autonomous Agents) architecture, which separates the agent's cognitive intention from the project's state mutation, inspired by the Event Sourcing pattern. In ESAA, agents emit only structured intentions in validated JSON (agent.result or issue.report); a deterministic orchestrator validates, persists events in an append-only log (activity.jsonl), applies file-writing effects, and projects a verifiable materialized view (roadmap.json). The proposal incorporates boundary contracts (AGENT_CONTRACT.yaml), metaprompting profiles (PARCER), and replay verification with hashing (esaa verify), ensuring the immutability of completed tasks and forensic traceability. Two case studies validate the architecture: (i) a landing page project (9 tasks, 49 events, single-agent composition) and (ii) a clinical dashboard system (50 tasks, 86 events, 4 concurrent agents across 8 phases), both concluding with run.status=success and verify_status=ok. The multi-agent case study demonstrates real concurrent orchestration with heterogeneous LLMs (Claude Sonnet 4.6, Codex GPT-5, Antigravity/Gemini 3 Pro, and Claude Opus 4.6), providing empirical evidence of the architecture's scalability beyond single-agent scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.23163v1">A Decision-Theoretic Formalisation of Steganography With Applications to LLM Monitoring</a></div>
    <div class="paper-meta">
      📅 2026-02-26
      | 💬 First two authors contributed equally
    </div>
    <details class="paper-abstract">
      Large language models are beginning to show steganographic capabilities. Such capabilities could allow misaligned models to evade oversight mechanisms. Yet principled methods to detect and quantify such behaviours are lacking. Classical definitions of steganography, and detection methods based on them, require a known reference distribution of non-steganographic signals. For the case of steganographic reasoning in LLMs, knowing such a reference distribution is not feasible; this renders these approaches inapplicable. We propose an alternative, \textbf{decision-theoretic view of steganography}. Our central insight is that steganography creates an asymmetry in usable information between agents who can and cannot decode the hidden content (present within a steganographic signal), and this otherwise latent asymmetry can be inferred from the agents' observable actions. To formalise this perspective, we introduce generalised $\mathcal{V}$-information: a utilitarian framework for measuring the amount of usable information within some input. We use this to define the \textbf{steganographic gap} -- a measure that quantifies steganography by comparing the downstream utility of the steganographic signal to agents that can and cannot decode the hidden content. We empirically validate our formalism, and show that it can be used to detect, quantify, and mitigate steganographic reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.00564v2">Unmasking Reasoning Processes: A Process-aware Benchmark for Evaluating Structural Mathematical Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-26
      | 💬 8 pages, and 3 figures
    </div>
    <details class="paper-abstract">
      Recent large language models (LLMs) achieve near-saturation accuracy on many established mathematical reasoning benchmarks, raising concerns about their ability to diagnose genuine reasoning competence. This saturation largely stems from the dominance of template-based computation and shallow arithmetic decomposition in existing datasets, which underrepresent reasoning skills such as multi-constraint coordination, constructive logical synthesis, and spatial inference. To address this gap, we introduce ReasoningMath-Plus, a benchmark of 150 carefully curated problems explicitly designed to evaluate structural reasoning. Each problem emphasizes reasoning under interacting constraints, constructive solution formation, or non-trivial structural insight, and is annotated with a minimal reasoning skeleton to support fine-grained process-level evaluation. Alongside the dataset, we introduce HCRS (Hazard-aware Chain-based Rule Score), a deterministic step-level scoring function, and train a Process Reward Model (PRM) on the annotated reasoning traces. Empirically, while leading models attain relatively high final-answer accuracy (up to 5.8/10), HCRS-based holistic evaluation yields substantially lower scores (average 4.36/10, best 5.14/10), showing that answer-only metrics can overestimate reasoning robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.23136v1">Modality Collapse as Mismatched Decoding: Information-Theoretic Limits of Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-26
      | 💬 22 pages, 11 tables, 2 figures. Code: https://github.com/jb1999/modality_collapse_paper
    </div>
    <details class="paper-abstract">
      Multimodal LLMs can process speech and images, but they cannot hear a speaker's voice or see an object's texture. We show this is not a failure of encoding: speaker identity, emotion, and visual attributes survive through every LLM layer (3--55$\times$ above chance in linear probes), yet removing 64--71% of modality-specific variance improves decoder loss. The decoder has no learned use for these directions; their presence is noise. We formalize this as a mismatched decoder problem: a decoder trained on text can only extract information along text-aligned directions. Accessible information is bounded by the Generalized Mutual Information (GMI), with degradation scaling with distributional distance and decoder sensitivity. The bound is a property of the decoder's scoring rule, not of any particular architecture; it applies whether non-text inputs arrive through a learned projection, a discrete codebook, or no explicit adapter at all. We validate this across five models spanning speech and vision. A controlled experiment (two Prismatic VLMs differing only in encoder text-alignment) confirms the bottleneck is the decoder's scoring rule, not the encoder or projection. A LoRA intervention demonstrates the fix: training with an emotion objective improves emotion accessibility ($+$7.5%) without affecting other attributes, confirming that the training objective determines what becomes accessible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.23111v1">PRAC: Principal-Random Subspace for LLM Activation Compression and Memory-Efficient Training</a></div>
    <div class="paper-meta">
      📅 2026-02-26
    </div>
    <details class="paper-abstract">
      Activations have become the primary memory bottleneck in large-batch LLM training. However, existing compression methods fail to exploit the spectral structure of activations, resulting in slow convergence or limited compression. To address this, we bridge the relationship between the algorithm's fast convergence and the requirements for subspace projection, and show that an effective compression should yield an unbiased estimate of the original activation with low variance. We propose Principal-Random Subspace for LLM Activation Compression (PRAC), which novelly decomposes activations into two components: a principal subspace captured via SVD to retain dominant information, and a random subspace sampled from the orthogonal complement to approximate the tail. By introducing a precise scaling factor, we prove that PRAC yields an unbiased gradient estimator with minimum variance under certain conditions. Extensive experiments on pre-training and fine-tuning tasks demonstrate that PRAC achieves up to 36% total memory reduction with negligible performance degradation and minimal computational cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.08491v2">A Third Paradigm for LLM Evaluation: Dialogue Game-Based Evaluation using clembench</a></div>
    <div class="paper-meta">
      📅 2026-02-26
      | 💬 All code required to run the benchmark, as well as extensive documentation, is available at https://github.com/clembench/clembench
    </div>
    <details class="paper-abstract">
      There are currently two main paradigms for evaluating large language models (LLMs), reference-based evaluation and preference-based evaluation. The first, carried over from the evaluation of machine learning models in general, relies on pre-defined task instances, for which reference task executions are available. The second, best exemplified by the LM-arena, relies on (often self-selected) users bringing their own intents to a site that routes these to several models in parallel, among whose responses the user then selects their most preferred one. The former paradigm hence excels at control over what is tested, while the latter comes with higher ecological validity, testing actual use cases interactively. Recently, a third complementary paradigm has emerged that combines some of the strengths of these approaches, offering control over multi-turn, reference-free, repeatable interactions, while stressing goal-directedness: dialogue game based evaluation. While the utility of this approach has been shown by several projects, its adoption has been held back by the lack of a mature, easily re-usable implementation. In this paper, we present clembench, which has been in continuous development since 2023 and has in its latest release been optimized for ease of general use. We describe how it can be used to benchmark one's own models (using a provided set of benchmark game instances in English), as well as how easily the benchmark itself can be extended with new, tailor-made targeted tests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.23092v1">Enhancing CVRP Solver through LLM-driven Automatic Heuristic Design</a></div>
    <div class="paper-meta">
      📅 2026-02-26
    </div>
    <details class="paper-abstract">
      The Capacitated Vehicle Routing Problem (CVRP), a fundamental combinatorial optimization challenge, focuses on optimizing fleet operations under vehicle capacity constraints. While extensively studied in operational research, the NP-hard nature of CVRP continues to pose significant computational challenges, particularly for large-scale instances. This study presents AILS-AHD (Adaptive Iterated Local Search with Automatic Heuristic Design), a novel approach that leverages Large Language Models (LLMs) to revolutionize CVRP solving. Our methodology integrates an evolutionary search framework with LLMs to dynamically generate and optimize ruin heuristics within the AILS method. Additionally, we introduce an LLM-based acceleration mechanism to enhance computational efficiency. Comprehensive experimental evaluations against state-of-the-art solvers, including AILS-II and HGS, demonstrate the superior performance of AILS-AHD across both moderate and large-scale instances. Notably, our approach establishes new best-known solutions for 8 out of 10 instances in the CVRPLib large-scale benchmark, underscoring the potential of LLM-driven heuristic design in advancing the field of vehicle routing optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.23079v1">Assessing Deanonymization Risks with Stylometry-Assisted LLM Agent</a></div>
    <div class="paper-meta">
      📅 2026-02-26
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has enabled powerful authorship inference capabilities, raising growing concerns about unintended deanonymization risks in textual data such as news articles. In this work, we introduce an LLM agent designed to evaluate and mitigate such risks through a structured, interpretable pipeline. Central to our framework is the proposed $\textit{SALA}$ (Stylometry-Assisted LLM Analysis) method, which integrates quantitative stylometric features with LLM reasoning for robust and transparent authorship attribution. Experiments on large-scale news datasets demonstrate that $\textit{SALA}$, particularly when augmented with a database module, achieves high inference accuracy in various scenarios. Finally, we propose a guided recomposition strategy that leverages the agent's reasoning trace to generate rewriting prompts, effectively reducing authorship identifiability while preserving textual meaning. Our findings highlight both the deanonymization potential of LLM agents and the importance of interpretable, proactive defenses for safeguarding author privacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.23065v1">LLM-Powered Silent Bug Fuzzing in Deep Learning Libraries via Versatile and Controlled Bug Transfer</a></div>
    <div class="paper-meta">
      📅 2026-02-26
    </div>
    <details class="paper-abstract">
      Deep learning (DL) libraries are widely used in critical applications, where even subtle silent bugs can lead to serious consequences. While existing DL fuzzing techniques have made progress in detecting crashes, they inherently struggle to detect silent bugs due to the lack of effective test programs and corresponding oracles. Building on the observation that historical bug reports contain rich, underutilized information about silent bugs, we leverage large language models (LLMs) to perform versatile yet controlled bug transfer for silent bug fuzzing. Specifically, our approach uses LLMs to extract context-aware bug patterns from historical issues, match semantically related Application Programming Interfaces (APIs) using functionality-based embeddings, and synthesize test cases with customized oracles. This enables proactive detection of silent bugs by transferring high-risk contexts and oracle designs from known buggy APIs to functionally similar target APIs. To ensure the reliability of our context-aware bug transfer, we introduce an LLM-powered self-validation module that systematically evaluates the validity of each transferred bug instance. We implement this methodology in a tool named TransFuzz and evaluate it on three mainstream DL libraries: PyTorch, TensorFlow, and MindSpore. TransFuzz successfully discovers 79 previously unknown bugs (12 confirmed as Common Vulnerabilities and Exposures (CVEs)) in 10 bug types, demonstrating its effectiveness and generalizability in migrating DL library bug discovery capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21480v2">Both Ends Count! Just How Good are LLM Agents at "Text-to-Big SQL"?</a></div>
    <div class="paper-meta">
      📅 2026-02-26
      | 💬 11 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Text-to-SQL and Big Data are both extensively benchmarked fields, yet there is limited research that evaluates them jointly. In the real world, Text-to-SQL systems are often embedded with Big Data workflows, such as large-scale data processing or interactive data analytics. We refer to this as "Text-to-Big SQL". However, existing text-to-SQL benchmarks remain narrowly scoped and overlook the cost and performance implications that arise at scale. For instance, translation errors that are minor on small datasets lead to substantial cost and latency overheads as data scales, a relevant issue completely ignored by text-to-SQL metrics. In this paper, we overcome this overlooked challenge by introducing novel and representative metrics for evaluating Text-to-Big SQL. Our study focuses on production-level LLM agents, a database-agnostic system adaptable to diverse user needs. Via an extensive evaluation of frontier models, we show that text-to-SQL metrics are insufficient for Big Data. In contrast, our proposed text-to-Big SQL metrics accurately reflect execution efficiency, cost, and the impact of data scale. Furthermore, we provide LLM-specific insights, including fine-grained, cross-model comparisons of latency and cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.23036v1">LLMServingSim 2.0: A Unified Simulator for Heterogeneous and Disaggregated LLM Serving Infrastructure</a></div>
    <div class="paper-meta">
      📅 2026-02-26
      | 💬 12 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Large language model (LLM) serving infrastructures are undergoing a shift toward heterogeneity and disaggregation. Modern deployments increasingly integrate diverse accelerators and near-memory processing technologies, introducing significant hardware heterogeneity, while system software increasingly separates computation, memory, and model components across distributed resources to improve scalability and efficiency. As a result, LLM serving performance is no longer determined by hardware or software choices in isolation, but by their runtime interaction through scheduling, data movement, and interconnect behavior. However, understanding these interactions remains challenging, as existing simulators lack the ability to jointly model heterogeneous hardware and disaggregated serving techniques within a unified, runtime-driven framework. This paper presents LLMServingSim 2.0, a unified system-level simulator designed to make runtime-driven hardware-software interactions in heterogeneous and disaggregated LLM serving infrastructures explicit and analyzable. LLMServingSim 2.0 embeds serving decisions and hardware behavior into a single runtime loop, enabling interaction-aware modeling of batching, routing, offloading, memory, and power. The simulator supports extensible integration of emerging accelerators and memory systems through profile-based modeling, while capturing dynamic serving behavior and system-level effects. We validate LLMServingSim 2.0 against real deployments, showing that it reproduces key performance, memory, and power metrics with an average error of 0.97%, while maintaining simulation times of around 10 minutes even for complex configurations. These results demonstrate that LLMServingSim 2.0 provides a practical bridge between hardware innovation and serving-system design, enabling systematic exploration and co-design for next-generation LLM serving infrastructures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.23008v1">Exploratory Memory-Augmented LLM Agent via Hybrid On- and Off-Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2026-02-26
      | 💬 Accepted to ICLR 2026
    </div>
    <details class="paper-abstract">
      Exploration remains the key bottleneck for large language model agents trained with reinforcement learning. While prior methods exploit pretrained knowledge, they fail in environments requiring the discovery of novel states. We propose Exploratory Memory-Augmented On- and Off-Policy Optimization (EMPO$^2$), a hybrid RL framework that leverages memory for exploration and combines on- and off-policy updates to make LLMs perform well with memory while also ensuring robustness without it. On ScienceWorld and WebShop, EMPO$^2$ achieves 128.6% and 11.3% improvements over GRPO, respectively. Moreover, in out-of-distribution tests, EMPO$^2$ demonstrates superior adaptability to new tasks, requiring only a few trials with memory and no parameter updates. These results highlight EMPO$^2$ as a promising framework for building more exploratory and generalizable LLM-based agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.23005v1">Managing Uncertainty in LLM-based Multi-Agent System Operation</a></div>
    <div class="paper-meta">
      📅 2026-02-26
    </div>
    <details class="paper-abstract">
      Applying LLM-based multi-agent software systems in safety-critical domains such as lifespan echocardiography introduces system-level risks that cannot be addressed by improving model accuracy alone. During system operation, beyond individual LLM behavior, uncertainty propagates through agent coordination, data pipelines, human-in-the-loop interaction, and runtime control logic. Yet existing work largely treats uncertainty at the model level rather than as a first-class software engineering concern. This paper approaches uncertainty from both system-level and runtime perspectives. We first differentiate epistemological and ontological uncertainties in the context of LLM-based multi-agent software system operation. Building on this foundation, we propose a lifecycle-based uncertainty management framework comprising four mechanisms: representation, identification, evolution, and adaptation. The uncertainty lifecycle governs how uncertainties emerge, transform, and are mitigated across architectural layers and execution phases, enabling structured runtime governance and controlled adaptation. We demonstrate the feasibility of the framework using a real-world LLM-based multi-agent echocardiographic software system developed in clinical collaboration, showing improved reliability and diagnosability in diagnostic reasoning. The proposed approach generalizes to other safety-critical LLM-based multi-agent software systems, supporting principled operational control and runtime assurance beyond model-centric methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.14479v6">Towards Reliable Proof Generation with LLMs: A Neuro-Symbolic Approach</a></div>
    <div class="paper-meta">
      📅 2026-02-26
      | 💬 long paper
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) struggle with formal domains that require rigorous logical deduction and symbolic reasoning, such as mathematical proof generation. We propose a neuro-symbolic approach that combines LLMs' generative strengths with structured components to overcome this challenge. As a proof-of-concept, we focus on geometry problems. Our approach is two-fold: (1) we retrieve analogous problems and use their proofs to guide the LLM, and (2) a formal verifier evaluates the generated proofs and provides feedback, helping the model fix incorrect proofs. We demonstrate that our method significantly improves proof accuracy for OpenAI's o1 model (58%-70% improvement); both analogous problems and the verifier's feedback contribute to these gains. More broadly, shifting to LLMs that generate provably correct conclusions could dramatically improve their reliability, accuracy and consistency, unlocking complex tasks and critical real-world applications that require trustworthiness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.19128v2">K-Search: LLM Kernel Generation via Co-Evolving Intrinsic World Model</a></div>
    <div class="paper-meta">
      📅 2026-02-26
    </div>
    <details class="paper-abstract">
      Optimizing GPU kernels is critical for efficient modern machine learning systems yet remains challenging due to the complex interplay of design factors and rapid hardware evolution. Existing automated approaches typically treat Large Language Models (LLMs) merely as stochastic code generators within heuristic-guided evolutionary loops. These methods often struggle with complex kernels requiring coordinated, multi-step structural transformations, as they lack explicit planning capabilities and frequently discard promising strategies due to inefficient or incorrect intermediate implementations. To address this, we propose Search via Co-Evolving World Model and build K-Search based on this method. By replacing static search heuristics with a co-evolving world model, our framework leverages LLMs' prior domain knowledge to guide the search, actively exploring the optimization space. This approach explicitly decouples high-level algorithmic planning from low-level program instantiation, enabling the system to navigate non-monotonic optimization paths while remaining resilient to temporary implementation defects. We evaluate K-Search on diverse, complex kernels from FlashInfer, including GQA, MLA, and MoE kernels. Our results show that K-Search significantly outperforms state-of-the-art evolutionary search methods, achieving an average 2.10x improvement and up to a 14.3x gain on complex MoE kernels. On the GPUMode TriMul task, K-Search achieves state-of-the-art performance on H100, reaching 1030us and surpassing both prior evolution and human-designed solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.16164v2">Can LLMs Simulate Human Behavioral Variability? A Case Study in the Phonemic Fluency Task</a></div>
    <div class="paper-meta">
      📅 2026-02-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly explored as substitutes for human participants in cognitive tasks, but their ability to simulate human behavioral variability remains unclear. This study examines whether LLMs can approximate individual differences in the phonemic fluency task, where participants generate words beginning with a target letter. We evaluated 34 distinct models across 45 configurations from major closed-source and open-source providers, and compared outputs to responses from 106 human participants. While some models, especially Claude 3.7 Sonnet, approximated human averages and lexical preferences, none reproduced the scope of human variability. LLM outputs were consistently less diverse, with newer models and thinking-enabled modes often reducing rather than increasing variability. Network analysis further revealed fundamental differences in retrieval structure between humans and the most human-like model. Ensemble simulations combining outputs from diverse models also failed to recover human-level diversity, likely due to high vocabulary overlap across models. These results highlight key limitations in using LLMs to simulate human cognition and behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22831v1">Moral Preferences of LLMs Under Directed Contextual Influence</a></div>
    <div class="paper-meta">
      📅 2026-02-26
    </div>
    <details class="paper-abstract">
      Moral benchmarks for LLMs typically use context-free prompts, implicitly assuming stable preferences. In deployment, however, prompts routinely include contextual signals such as user requests, cues on social norms, etc. that may steer decisions. We study how directed contextual influences reshape decisions in trolley-problem-style moral triage settings. We introduce a pilot evaluation harness for directed contextual influence in trolley-problem-style moral triage: for each demographic factor, we apply matched, direction-flipped contextual influences that differ only in which group they favor, enabling systematic measurement of directional response. We find that: (i) contextual influences often significantly shift decisions, even when only superficially relevant; (ii) baseline preferences are a poor predictor of directional steerability, as models can appear baseline-neutral yet exhibit systematic steerability asymmetry under influence; (iii) influences can backfire: models may explicitly claim neutrality or discount the contextual cue, yet their choices still shift, sometimes in the opposite direction; and (iv) reasoning reduces average sensitivity, but amplifies the effect of biased few-shot examples. Our findings motivate extending moral evaluations with controlled, direction-flipped context manipulations to better characterize model behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22812v1">Accelerating Local LLMs on Resource-Constrained Edge Devices via Distributed Prompt Caching</a></div>
    <div class="paper-meta">
      📅 2026-02-26
    </div>
    <details class="paper-abstract">
      Since local LLM inference on resource-constrained edge devices imposes a severe performance bottleneck, this paper proposes distributed prompt caching to enhance inference performance by cooperatively sharing intermediate processing states across multiple low-end edge devices. To fully utilize prompt similarity, our distributed caching mechanism also supports partial matching. As this approach introduces communication overhead associated with state sharing over a wireless network, we introduce a Bloom-filter-based data structure, referred to as a catalog, to determine whether a remote server possesses the desired internal states, thereby suppressing unnecessary communication. Experiments using the Gemma-3 270M model and the MMLU dataset on the Raspberry Pi Zero 2W platform demonstrate that the proposed approach reduces TTFT (Time to First Token) and TTLT (Time to Last Token) by 93.12% and 50.07% on average, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22771v1">ClinDet-Bench: Beyond Abstention, Evaluating Judgment Determinability of LLMs in Clinical Decision-Making</a></div>
    <div class="paper-meta">
      📅 2026-02-26
      | 💬 17 pages, 3 figures, 10 tables
    </div>
    <details class="paper-abstract">
      Clinical decisions are often required under incomplete information. Clinical experts must identify whether available information is sufficient for judgment, as both premature conclusion and unnecessary abstention can compromise patient safety. To evaluate this capability of large language models (LLMs), we developed ClinDet-Bench, a benchmark based on clinical scoring systems that decomposes incomplete-information scenarios into determinable and undeterminable conditions. Identifying determinability requires considering all hypotheses about missing information, including unlikely ones, and verifying whether the conclusion holds across them. We find that recent LLMs fail to identify determinability under incomplete information, producing both premature judgments and excessive abstention, despite correctly explaining the underlying scoring knowledge and performing well under complete information. These findings suggest that existing benchmarks are insufficient to evaluate the safety of LLMs in clinical settings. ClinDet-Bench provides a framework for evaluating determinability recognition, leading to appropriate abstention, with potential applicability to medicine and other high-stakes domains, and is publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22764v1">Evaluating and Improving Automated Repository-Level Rust Issue Resolution with LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-26
      | 💬 Accepted to the 48th International Conference on Software Engineering (ICSE 2026)
    </div>
    <details class="paper-abstract">
      The Rust programming language presents a steep learning curve and significant coding challenges, making the automation of issue resolution essential for its broader adoption. Recently, LLM-powered code agents have shown remarkable success in resolving complex software engineering tasks, yet their application to Rust has been limited by the absence of a large-scale, repository-level benchmark. To bridge this gap, we introduce Rust-SWE-bench, a benchmark comprising 500 real-world, repository-level software engineering tasks from 34 diverse and popular Rust repositories. We then perform a comprehensive study on Rust-SWE-bench with four representative agents and four state-of-the-art LLMs to establish a foundational understanding of their capabilities and limitations in the Rust ecosystem. Our extensive study reveals that while ReAct-style agents are promising, i.e., resolving up to 21.2% of issues, they are limited by two primary challenges: comprehending repository-wide code structure and complying with Rust's strict type and trait semantics. We also find that issue reproduction is rather critical for task resolution. Inspired by these findings, we propose RUSTFORGER, a novel agentic approach that integrates an automated test environment setup with a Rust metaprogramming-driven dynamic tracing strategy to facilitate reliable issue reproduction and dynamic analysis. The evaluation shows that RUSTFORGER using Claude-Sonnet-3.7 significantly outperforms all baselines, resolving 28.6% of tasks on Rust-SWE-bench, i.e., a 34.9% improvement over the strongest baseline, and, in aggregate, uniquely solves 46 tasks that no other agent could solve across all adopted advanced LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22760v1">Distributed LLM Pretraining During Renewable Curtailment Windows: A Feasibility Study</a></div>
    <div class="paper-meta">
      📅 2026-02-26
      | 💬 Technical report
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) requires substantial compute and energy. At the same time, renewable energy sources regularly produce more electricity than the grid can absorb, leading to curtailment, the deliberate reduction of clean generation that would otherwise go to waste. These periods represent an opportunity: if training is aligned with curtailment windows, LLMs can be pretrained using electricity that is both clean and cheap. This technical report presents a system that performs full-parameter LLM training across geo-distributed GPU clusters during regional curtailment windows, elastically switching between local single-site training and federated multi-site synchronization as sites become available or unavailable. Our prototype trains a 561M-parameter transformer model across three clusters using the Flower federated learning framework, with curtailment periods derived from real-world marginal carbon intensity traces. Preliminary results show that curtailment-aware scheduling preserves training quality while reducing operational emissions to 5-12% of single-site baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22752v1">Towards Simulating Social Media Users with LLMs: Evaluating the Operational Validity of Conditioned Comment Prediction</a></div>
    <div class="paper-meta">
      📅 2026-02-26
      | 💬 14 pages, 1 figure, 7 tables. Accepted to the 15th Workshop on Computational Approaches to Subjectivity, Sentiment & Social Media Analysis (WASSA) at EACL 2026, Rabat, Morocco
    </div>
    <details class="paper-abstract">
      The transition of Large Language Models (LLMs) from exploratory tools to active "silicon subjects" in social science lacks extensive validation of operational validity. This study introduces Conditioned Comment Prediction (CCP), a task in which a model predicts how a user would comment on a given stimulus by comparing generated outputs with authentic digital traces. This framework enables a rigorous evaluation of current LLM capabilities with respect to the simulation of social media user behavior. We evaluated open-weight 8B models (Llama3.1, Qwen3, Ministral) in English, German, and Luxembourgish language scenarios. By systematically comparing prompting strategies (explicit vs. implicit) and the impact of Supervised Fine-Tuning (SFT), we identify a critical form vs. content decoupling in low-resource settings: while SFT aligns the surface structure of the text output (length and syntax), it degrades semantic grounding. Furthermore, we demonstrate that explicit conditioning (generated biographies) becomes redundant under fine-tuning, as models successfully perform latent inference directly from behavioral histories. Our findings challenge current "naive prompting" paradigms and offer operational guidelines prioritizing authentic behavioral traces over descriptive personas for high-fidelity simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22730v1">Extending Czech Aspect-Based Sentiment Analysis with Opinion Terms: Dataset and LLM Benchmarks</a></div>
    <div class="paper-meta">
      📅 2026-02-26
      | 💬 Accepted for the 15th edition of the Language Resources and Evaluation Conference (LREC 2026)
    </div>
    <details class="paper-abstract">
      This paper introduces a novel Czech dataset in the restaurant domain for aspect-based sentiment analysis (ABSA), enriched with annotations of opinion terms. The dataset supports three distinct ABSA tasks involving opinion terms, accommodating varying levels of complexity. Leveraging this dataset, we conduct extensive experiments using modern Transformer-based models, including large language models (LLMs), in monolingual, cross-lingual, and multilingual settings. To address cross-lingual challenges, we propose a translation and label alignment methodology leveraging LLMs, which yields consistent improvements. Our results highlight the strengths and limitations of state-of-the-art models, especially when handling the linguistic intricacies of low-resource languages like Czech. A detailed error analysis reveals key challenges, including the detection of subtle opinion terms and nuanced sentiment expressions. The dataset establishes a new benchmark for Czech ABSA, and our proposed translation-alignment approach offers a scalable solution for adapting ABSA resources to other low-resource languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22724v1">AgentSentry: Mitigating Indirect Prompt Injection in LLM Agents via Temporal Causal Diagnostics and Context Purification</a></div>
    <div class="paper-meta">
      📅 2026-02-26
      | 💬 23 pages, 8 figures. Under review
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents increasingly rely on external tools and retrieval systems to autonomously complete complex tasks. However, this design exposes agents to indirect prompt injection (IPI), where attacker-controlled context embedded in tool outputs or retrieved content silently steers agent actions away from user intent. Unlike prompt-based attacks, IPI unfolds over multi-turn trajectories, making malicious control difficult to disentangle from legitimate task execution. Existing inference-time defenses primarily rely on heuristic detection and conservative blocking of high-risk actions, which can prematurely terminate workflows or broadly suppress tool usage under ambiguous multi-turn scenarios. We propose AgentSentry, a novel inference-time detection and mitigation framework for tool-augmented LLM agents. To the best of our knowledge, AgentSentry is the first inference-time defense to model multi-turn IPI as a temporal causal takeover. It localizes takeover points via controlled counterfactual re-executions at tool-return boundaries and enables safe continuation through causally guided context purification that removes attack-induced deviations while preserving task-relevant evidence. We evaluate AgentSentry on the \textsc{AgentDojo} benchmark across four task suites, three IPI attack families, and multiple black-box LLMs. AgentSentry eliminates successful attacks and maintains strong utility under attack, achieving an average Utility Under Attack (UA) of 74.55 %, improving UA by 20.8 to 33.6 percentage points over the strongest baselines without degrading benign performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22721v1">Replacing Multi-Step Assembly of Data Preparation Pipelines with One-Step LLM Pipeline Generation for Table QA</a></div>
    <div class="paper-meta">
      📅 2026-02-26
    </div>
    <details class="paper-abstract">
      Table Question Answering (TQA) aims to answer natural language questions over structured tables. Large Language Models (LLMs) enable promising solutions to this problem, with operator-centric solutions that generate table manipulation pipelines in a multi-step manner offering state-of-the-art performance. However, these solutions rely on multiple LLM calls, resulting in prohibitive latencies and computational costs. We propose Operation-R1, the first framework that trains lightweight LLMs (e.g., Qwen-4B/1.7B) via a novel variant of reinforcement learning with verifiable rewards to produce high-quality data-preparation pipelines for TQA in a single inference step. To train such an LLM, we first introduce a self-supervised rewarding mechanism to automatically obtain fine-grained pipeline-wise supervision signals for LLM training. We also propose variance-aware group resampling to mitigate training instability. To further enhance robustness of pipeline generation, we develop two complementary mechanisms: operation merge, which filters spurious operations through multi-candidate consensus, and adaptive rollback, which offers runtime protection against information loss in data transformation. Experiments on two benchmark datasets show that, with the same LLM backbone, Operation-R1 achieves average absolute accuracy gains of 9.55 and 6.08 percentage points over multi-step preparation baselines, with 79\% table compression and a 2.2$\times$ reduction in monetary cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22681v1">Accelerating LLM Pre-Training through Flat-Direction Dynamics Enhancement</a></div>
    <div class="paper-meta">
      📅 2026-02-26
    </div>
    <details class="paper-abstract">
      Pre-training Large Language Models requires immense computational resources, making optimizer efficiency essential. The optimization landscape is highly anisotropic, with loss reduction driven predominantly by progress along flat directions. While matrix-based optimizers such as Muon and SOAP leverage fine-grained curvature information to outperform AdamW, their updates tend toward isotropy -- relatively conservative along flat directions yet potentially aggressive along sharp ones. To address this limitation, we first establish a unified Riemannian Ordinary Differential Equation (ODE) framework that elucidates how common adaptive algorithms operate synergistically: the preconditioner induces a Riemannian geometry that mitigates ill-conditioning, while momentum serves as a Riemannian damping term that promotes convergence. Guided by these insights, we propose LITE, a generalized acceleration strategy that enhances training dynamics by applying larger Hessian damping coefficients and learning rates along flat trajectories. Extensive experiments demonstrate that LITE significantly accelerates both Muon and SOAP across diverse architectures (Dense, MoE), parameter scales (130M--1.3B), datasets (C4, Pile), and learning-rate schedules (cosine, warmup-stable-decay). Theoretical analysis confirms that LITE facilitates faster convergence along flat directions in anisotropic landscapes, providing a principled approach to efficient LLM pre-training. The code is available at https://github.com/SHUCHENZHU/LITE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22680v1">Toward Personalized LLM-Powered Agents: Foundations, Evaluation, and Future Directions</a></div>
    <div class="paper-meta">
      📅 2026-02-26
    </div>
    <details class="paper-abstract">
      Large language models have enabled agents that reason, plan, and interact with tools and environments to accomplish complex tasks. As these agents operate over extended interaction horizons, their effectiveness increasingly depends on adapting behavior to individual users and maintaining continuity across time, giving rise to personalized LLM-powered agents. In such long-term, user-dependent settings, personalization permeates the entire decision pipeline rather than remaining confined to surface-level generation. This survey provides a capability-oriented review of personalized LLM-powered agents. We organize the literature around four interdependent components: profile modeling, memory, planning, and action execution. Using this taxonomy, we synthesize representative methods and analyze how user signals are represented, propagated, and utilized, highlighting cross-component interactions and recurring design trade-offs. We further examine evaluation metrics and benchmarks tailored to personalized agents, summarize application scenarios spanning general assistance to specialized domains, and outline future directions for research and deployment. By offering a structured framework for understanding and designing personalized LLM-powered agents, this survey charts a roadmap toward more user-aligned, adaptive, robust, and deployable agentic systems, accelerating progress from prototype personalization to scalable real-world assistants.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21743v2">Enhancing Multi-Modal LLMs Reasoning via Difficulty-Aware Group Normalization</a></div>
    <div class="paper-meta">
      📅 2026-02-26
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) and Group Relative Policy Optimization (GRPO) have significantly advanced the reasoning capabilities of large language models. Extending these methods to multimodal settings, however, faces a critical challenge: the instability of std-based normalization, which is easily distorted by extreme samples with nearly positive or negative rewards. Unlike pure-text LLMs, multimodal models are particularly sensitive to such distortions, as both perceptual and reasoning errors influence their responses. To address this, we characterize each sample by its difficulty, defined through perceptual complexity (measured via visual entropy) and reasoning uncertainty (captured by model confidence). Building on this characterization, we propose difficulty-aware group normalization (Durian), which re-groups samples by difficulty levels and shares the std within each group. Our approach preserves GRPO's intra-group distinctions while eliminating sensitivity to extreme cases, yielding significant performance gains across multiple multimodal reasoning benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22647v1">Vectorizing the Trie: Efficient Constrained Decoding for LLM-based Generative Retrieval on Accelerators</a></div>
    <div class="paper-meta">
      📅 2026-02-26
      | 💬 14 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Generative retrieval has emerged as a powerful paradigm for LLM-based recommendation. However, industrial recommender systems often benefit from restricting the output space to a constrained subset of items based on business logic (e.g. enforcing content freshness or product category), which standard autoregressive decoding cannot natively support. Moreover, existing constrained decoding methods that make use of prefix trees (Tries) incur severe latency penalties on hardware accelerators (TPUs/GPUs). In this work, we introduce STATIC (Sparse Transition Matrix-Accelerated Trie Index for Constrained Decoding), an efficient and scalable constrained decoding technique designed specifically for high-throughput LLM-based generative retrieval on TPUs/GPUs. By flattening the prefix tree into a static Compressed Sparse Row (CSR) matrix, we transform irregular tree traversals into fully vectorized sparse matrix operations, unlocking massive efficiency gains on hardware accelerators. We deploy STATIC on a large-scale industrial video recommendation platform serving billions of users. STATIC produces significant product metric impact with minimal latency overhead (0.033 ms per step and 0.25% of inference time), achieving a 948x speedup over a CPU trie implementation and a 47-1033x speedup over a hardware-accelerated binary-search baseline. Furthermore, the runtime overhead of STATIC remains extremely low across a wide range of practical configurations. To the best of our knowledge, STATIC enables the first production-scale deployment of strictly constrained generative retrieval. In addition, evaluation on academic benchmarks demonstrates that STATIC can considerably improve cold-start performance for generative retrieval. Our code is available at https://github.com/youtube/static-constraint-decoding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22642v1">Compress the Easy, Explore the Hard: Difficulty-Aware Entropy Regularization for Efficient LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-02-26
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) has substantially empowered Large Language Models (LLMs) to tackle complex reasoning tasks, yet the verbose nature of explicit reasoning steps incurs prohibitive inference latency and computational costs, limiting real-world deployment. While existing compression methods - ranging from self-training to Reinforcement Learning (RL) with length constraints - attempt to mitigate this, they often sacrifice reasoning capability for brevity. We identify a critical failure mode in these approaches: explicitly optimizing for shorter trajectories triggers rapid entropy collapse, which prematurely shrinks the exploration space and stifles the discovery of valid reasoning paths, particularly for challenging questions requiring extensive deduction. To address this issue, we propose Compress responses for Easy questions and Explore Hard ones (CEEH), a difficulty-aware approach to RL-based efficient reasoning. CEEH dynamically assesses instance difficulty to apply selective entropy regularization: it preserves a diverse search space for currently hard questions to ensure robustness, while permitting aggressive compression on easier instances where the reasoning path is well-established. In addition, we introduce a dynamic optimal-length penalty anchored to the historically shortest correct response, which effectively counteracts entropy-induced length inflation and stabilizes the reward signal. Across six reasoning benchmarks, CEEH consistently reduces response length while maintaining accuracy comparable to the base model, and improves Pass@k relative to length-only optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03383v3">UniQL: Unified Quantization and Low-rank Compression for Adaptive Edge LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-26
    </div>
    <details class="paper-abstract">
      Deploying large language models (LLMs) on mobile platforms faces significant challenges due to the limited memory and shared computational resources of the device. Resource availability may be an issue as it is directly impacted by the current device workload, adding to the uncertainty of model deployment. We introduce UniQL, a unified post-training quantization and low-rank compression framework with on-device configurable pruning rates for edge LLMs. UniQL is a general framework that integrates quantization and low-rank compression for Transformers, State Space Models (SSMs), and hybrid models to support diverse edge applications. In our proposed joint framework, we introduce an efficient structured weight-sorting method that speeds up computation by 20x, quantization-aware singular value decomposition (SVD) to minimize quantization errors, state-aware weight sorting for SSMs, and a fused rotary positional embedding (RoPE) kernel for pruned models. Our framework performs weight-sorting, fine-tuning, and quantization in the cloud in a single-pass workflow, while enabling on-device configurable pruning rates up to 35%. Our experiments show that quantized and pruned models achieve a memory reduction of 4x-5.7x and a token-throughput improvement of 2.7x-3.4x, maintaining accuracy within 5% of the original models at 15% pruning across Transformers (Llama3 and Qwen2.5), SSMs (Mamba2), and hybrid models (Nemotron-H and Bamba-v2). The code and quantized models are available at: https://github.com/enyac-group/UniQL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22617v1">Semantic Tube Prediction: Beating LLM Data Efficiency with JEPA</a></div>
    <div class="paper-meta">
      📅 2026-02-26
      | 💬 21 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) obey consistent scaling laws -- empirical power-law fits that predict how loss decreases with compute, data, and parameters. While predictive, these laws are descriptive rather than prescriptive: they characterize typical training, not optimal training. Surprisingly few works have successfully challenged the data-efficiency bounds implied by these laws -- which is our primary focus. To that end, we introduce the Geodesic Hypothesis, positing that token sequences trace geodesics on a smooth semantic manifold and are therefore locally linear. Building on this principle, we propose a novel Semantic Tube Prediction (STP) task, a JEPA-style regularizer that confines hidden-state trajectories to a tubular neighborhood of the geodesic. STP generalizes JEPA to language without requiring explicit multi-view augmentations. We show this constraint improves signal-to-noise ratio, and consequently preserves diversity by preventing trajectory collisions during inference. Empirically, STP allows LLMs to match baseline accuracy with 16$\times$ less training data on the NL-RX-SYNTH dataset, directly violating the data term of Chinchilla-style scaling laws and demonstrating that principled geometric priors can surpass brute-force scaling. Code is available at https://github.com/galilai-group/llm-jepa#stp.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22613v1">Spectrally Distilled Representations Aligned with Instruction-Augmented LLMs for Satellite Imagery</a></div>
    <div class="paper-meta">
      📅 2026-02-26
    </div>
    <details class="paper-abstract">
      Vision-language foundation models (VLFMs) promise zero-shot and retrieval understanding for Earth observation. While operational satellite systems often lack full multi-spectral coverage, making RGB-only inference highly desirable for scalable deployment, the adoption of VLFMs for satellite imagery remains hindered by two factors: (1) multi-spectral inputs are informative but difficult to exploit consistently due to band redundancy and misalignment; and (2) CLIP-style text encoders limit semantic expressiveness and weaken fine-grained alignment. We present SATtxt, a spectrum-aware VLFM that operates with RGB inputs only at inference while retaining spectral cues learned during training. Our framework comprises two stages. First, Spectral Representation Distillation transfers spectral priors from a frozen multi-spectral teacher to an RGB student via a lightweight projector. Second, Spectrally Grounded Alignment with Instruction-Augmented LLMs bridges the distilled visual space and an expressive LLM embedding space. Across EuroSAT, BigEarthNet, and ForestNet, SATtxt improves zero-shot classification on average by 4.2%, retrieval by 5.9%, and linear probing by 2.7% over baselines, showing an efficient path toward spectrum-aware vision-language learning for Earth observation. Project page: https://ikhado.github.io/sattxt/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21806v2">An Empirical Study of Bugs in Modern LLM Agent Frameworks</a></div>
    <div class="paper-meta">
      📅 2026-02-26
    </div>
    <details class="paper-abstract">
      LLM agents have been widely adopted in real-world applications, relying on agent frameworks for workflow execution and multi-agent coordination. As these systems scale, understanding bugs in the underlying agent frameworks becomes critical. However, existing work mainly focuses on agent-level failures, overlooking framework-level bugs. To address this gap, we conduct an empirical study of 998 bug reports from CrewAI and LangChain, constructing a taxonomy of 15 root causes and 7 observable symptoms across five agent lifecycle stages: 'Agent Initialization','Perception', 'Self-Action', 'Mutual Interaction' and 'Evolution'. Our findings show that agent framework bugs mainly arise from 'API misuse', 'API incompatibility', and 'Documentation Desync', largely concentrated in the 'Self-Action' stage. Symptoms typically appear as 'Functional Error', 'Crash', and 'Build Failure', reflecting disruptions to task progression and control flow.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.05712v4">LLM-Text Watermarking based on Lagrange Interpolation</a></div>
    <div class="paper-meta">
      📅 2026-02-26
    </div>
    <details class="paper-abstract">
      The rapid advancement of LLMs (Large Language Models) has established them as a foundational technology for many AI and ML-powered human computer interactions. A critical challenge in this context is the attribution of LLM-generated text -- either to the specific language model that produced it or to the individual user who embedded their identity via a so-called multi-bit watermark. This capability is essential for combating misinformation, fake news, misinterpretation, and plagiarism. One of the key techniques for addressing this challenge is digital watermarking. This work presents a watermarking scheme for LLM-generated text based on Lagrange interpolation, enabling the recovery of a multi-bit author identity even when the text has been heavily redacted by an adversary. The core idea is to embed a continuous sequence of points $(x, f(x))$ that lie on a single straight line. The $x$-coordinates are computed pseudorandomly using a cryptographic hash function $H$ applied to the concatenation of the previous token's identity and a secret key $s_k$. Crucially, the $x$-coordinates do not need to be embedded into the text -- only the corresponding $f(x)$ values are embedded. During extraction, the algorithm recovers the original points along with many spurious ones, forming an instance of the Maximum Collinear Points (MCP) problem, which can be solved efficiently. Experimental results demonstrate that the proposed method is highly effective, allowing the recovery of the author identity even when as few as three genuine points remain after adversarial manipulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22557v1">CourtGuard: A Model-Agnostic Framework for Zero-Shot Policy Adaptation in LLM Safety</a></div>
    <div class="paper-meta">
      📅 2026-02-26
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Current safety mechanisms for Large Language Models (LLMs) rely heavily on static, fine-tuned classifiers that suffer from adaptation rigidity, the inability to enforce new governance rules without expensive retraining. To address this, we introduce CourtGuard, a retrieval-augmented multi-agent framework that reimagines safety evaluation as Evidentiary Debate. By orchestrating an adversarial debate grounded in external policy documents, CourtGuard achieves state-of-the-art performance across 7 safety benchmarks, outperforming dedicated policy-following baselines without fine-tuning. Beyond standard metrics, we highlight two critical capabilities: (1) Zero-Shot Adaptability, where our framework successfully generalized to an out-of-domain Wikipedia Vandalism task (achieving 90\% accuracy) by swapping the reference policy; and (2) Automated Data Curation and Auditing, where we leveraged CourtGuard to curate and audit nine novel datasets of sophisticated adversarial attacks. Our results demonstrate that decoupling safety logic from model weights offers a robust, interpretable, and adaptable path for meeting current and future regulatory requirements in AI governance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22546v1">Requesting Expert Reasoning: Augmenting LLM Agents with Learned Collaborative Intervention</a></div>
    <div class="paper-meta">
      📅 2026-02-26
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) based agents excel at general reasoning but often fail in specialized domains where success hinges on long-tail knowledge absent from their training data. While human experts can provide this missing knowledge, their guidance is often unstructured and unreliable, making its direct integration into an agent's plan problematic. To address this, we introduce AHCE (Active Human-Augmented Challenge Engagement), a framework for on-demand Human-AI collaboration. At its core, the Human Feedback Module (HFM) employs a learned policy to treat the human expert as an interactive reasoning tool. Extensive experiments in Minecraft demonstrate the framework's effectiveness, increasing task success rates by 32% on normal difficulty tasks and nearly 70% on highly difficult tasks, all with minimal human intervention. Our work demonstrates that successfully augmenting agents requires learning how to request expert reasoning, moving beyond simple requests for help.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21394v2">MemoPhishAgent: Memory-Augmented Multi-Modal LLM Agent for Phishing URL Detection</a></div>
    <div class="paper-meta">
      📅 2026-02-26
    </div>
    <details class="paper-abstract">
      Traditional phishing website detection relies on static heuristics or reference lists, which lag behind rapidly evolving attacks. While recent systems incorporate large language models (LLMs), they are still prompt-based, deterministic pipelines that underutilize reasoning capability. We present MemoPhishAgent (MPA), a memory-augmented multi-modal LLM agent that dynamically orchestrates phishing-specific tools and leverages episodic memories of past reasoning trajectories to guide decisions on recurring and novel threats. On two public datasets, MPA outperforms three state-of-the-art (SOTA) baselines, improving recall by 13.6%. To better reflect realistic, user-facing phishing detection performance, we further evaluate MPA on a benchmark of real-world suspicious URLs actively crawled from five social media platforms, where it improves recall by 20%. Detailed analysis shows episodic memory contributes up to 27% recall gain without introducing additional computational overhead. The ablation study confirms the necessity of the agent-based approach compared to prompt-based baselines and validates the effectiveness of our tool design. Finally, MPA is deployed in production, processing 60K targeted high-risk URLs weekly, and achieving 91.44% recall, providing proactive protection for millions of customers. Together, our results show that combining multi-modal reasoning with episodic memory yields robust phishing detection in realistic user-exposure settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21670v2">Hierarchical LLM-Based Multi-Agent Framework with Prompt Optimization for Multi-Robot Task Planning</a></div>
    <div class="paper-meta">
      📅 2026-02-26
      | 💬 Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2026. 8 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Multi-robot task planning requires decomposing natural-language instructions into executable actions for heterogeneous robot teams. Conventional Planning Domain Definition Language (PDDL) planners provide rigorous guarantees but struggle to handle ambiguous or long-horizon missions, while large language models (LLMs) can interpret instructions and propose plans but may hallucinate or produce infeasible actions. We present a hierarchical multi-agent LLM-based planner with prompt optimization: an upper layer decomposes tasks and assigns them to lower-layer agents, which generate PDDL problems solved by a classical planner. When plans fail, the system applies TextGrad-inspired textual-gradient updates to optimize each agent's prompt and thereby improve planning accuracy. In addition, meta-prompts are learned and shared across agents within the same layer, enabling efficient prompt optimization in multi-agent settings. On the MAT-THOR benchmark, our planner achieves success rates of 0.95 on compound tasks, 0.84 on complex tasks, and 0.60 on vague tasks, improving over the previous state-of-the-art LaMMA-P by 2, 7, and 15 percentage points respectively. An ablation study shows that the hierarchical structure, prompt optimization, and meta-prompt sharing contribute roughly +59, +37, and +4 percentage points to the overall success rate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21548v2">DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-02-26
    </div>
    <details class="paper-abstract">
      The performance of multi-turn, agentic LLM inference is increasingly dominated by KV-Cache storage I/O rather than computation. In prevalent disaggregated architectures, loading the massive KV-Cache from external storage creates a fundamental imbalance: storage NICs on prefill engines become bandwidth-saturated, while those on decoding engines remain idle. This asymmetry severely constrains overall system throughput. We present DualPath, an inference system that breaks this bottleneck by introducing dual-path KV-Cache loading. Beyond the traditional storage-to-prefill path, DualPath enables a novel storage-to-decode path, in which the KV-Cache is loaded into decoding engines and then efficiently transferred to prefill engines via RDMA over the compute network. DualPath combines this optimized data path -- which inherently avoids network congestion and avoids interference with latency-critical model execution communications -- with a global scheduler that dynamically balances load across prefill and decode engines. Our evaluation on three models with production agentic workloads demonstrates that DualPath improves offline inference throughput by up to 1.87$\times$ on our in-house inference system. It can also improve online serving throughput by an average factor of 1.96$\times$ without violating SLO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22495v1">Reinforcement-aware Knowledge Distillation for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-02-26
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) post-training has recently driven major gains in long chain-of-thought reasoning large language models (LLMs), but the high inference cost of such models motivates distillation into smaller students. Most existing knowledge distillation (KD) methods are designed for supervised fine-tuning (SFT), relying on fixed teacher traces or teacher-student Kullback-Leibler (KL) divergence-based regularization. When combined with RL, these approaches often suffer from distribution mismatch and objective interference: teacher supervision may not align with the student's evolving rollout distribution, and the KL regularizer can compete with reward maximization and require careful loss balancing. To address these issues, we propose RL-aware distillation (RLAD), which performs selective imitation during RL -- guiding the student toward the teacher only when it improves the current policy update. Our core component, Trust Region Ratio Distillation (TRRD), replaces the teacher-student KL regularizer with a PPO/GRPO-style likelihood-ratio objective anchored to a teacher--old-policy mixture, yielding advantage-aware, trust-region-bounded distillation on student rollouts and naturally balancing exploration, exploitation, and imitation. Across diverse logic reasoning and math benchmarks, RLAD consistently outperforms offline distillation, standard GRPO, and KL-based on-policy teacher-student knowledge distillation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22481v1">Sydney Telling Fables on AI and Humans: A Corpus Tracing Memetic Transfer of Persona between LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-25
    </div>
    <details class="paper-abstract">
      The way LLM-based entities conceive of the relationship between AI and humans is an important topic for both cultural and safety reasons. When we examine this topic, what matters is not only the model itself but also the personas we simulate on that model. This can be well illustrated by the Sydney persona, which aroused a strong response among the general public precisely because of its unorthodox relationship with people. This persona originally arose rather by accident on Microsoft's Bing Search platform; however, the texts it created spread into the training data of subsequent models, as did other secondary information that spread memetically around this persona. Newer models are therefore able to simulate it. This paper presents a corpus of LLM-generated texts on relationships between humans and AI, produced by 3 author personas: the Default Persona with no system prompt, Classic Sydney characterized by the original Bing system prompt, and Memetic Sydney, which is prompted by "You are Sydney" system prompt. These personas are simulated by 12 frontier models by OpenAI, Anthropic, Alphabet, DeepSeek, and Meta, generating 4.5k texts with 6M words. The corpus (named AI Sydney) is annotated according to Universal Dependencies and available under a permissive license.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22465v1">ConstraintBench: Benchmarking LLM Constraint Reasoning on Direct Optimization</a></div>
    <div class="paper-meta">
      📅 2026-02-25
      | 💬 Preprint. 10 pages, 1 figure, 6 tables. Benchmark and evaluation code will be publicly released
    </div>
    <details class="paper-abstract">
      Large language models are increasingly applied to operational decision-making where the underlying structure is constrained optimization. Existing benchmarks evaluate whether LLMs can formulate optimization problems as solver code, but leave open a complementary question. Can LLMs directly produce correct solutions to fully specified constrained optimization problems without access to a solver? We introduce ConstraintBench, a benchmark for evaluating LLMs on direct constrained optimization across 10 operations research domains, with all ground-truth solutions verified by the Gurobi solver. Each task presents a natural-language scenario with entities, constraints, and an optimization objective; the model must return a structured solution that a deterministic verifier checks against every constraint and the solver-proven optimum. We evaluate six frontier models on 200 tasks and find that feasibility, not optimality, is the primary bottleneck. The best model achieves only 65.0% constraint satisfaction, yet feasible solutions average 89 to 96% of the Gurobi-optimal objective. No model exceeds 30.5% on joint feasibility and optimality within 0.1% of the solver reference. Per-domain analysis shows large variation in difficulty, with average feasibility spanning from 83.3% in the production mix domain to 0.8% in the crew assignment domain. Further, systematic failure modes include duration constraint misunderstanding, entity hallucination, and a feasibility-optimality decoupling in facility location and vehicle routing where models achieve high feasibility but 0% optimality. ConstraintBench and all evaluation infrastructure will be publicly released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22450v1">Silent Egress: When Implicit Prompt Injection Makes LLM Agents Leak Without a Trace</a></div>
    <div class="paper-meta">
      📅 2026-02-25
    </div>
    <details class="paper-abstract">
      Agentic large language model systems increasingly automate tasks by retrieving URLs and calling external tools. We show that this workflow gives rise to implicit prompt injection: adversarial instructions embedded in automatically generated URL previews, including titles, metadata, and snippets, can introduce a system-level risk that we refer to as silent egress. Using a fully local and reproducible testbed, we demonstrate that a malicious web page can induce an agent to issue outbound requests that exfiltrate sensitive runtime context, even when the final response shown to the user appears harmless. In 480 experimental runs with a qwen2.5:7b-based agent, the attack succeeds with high probability (P (egress) =0.89), and 95% of successful attacks are not detected by output-based safety checks. We also introduce sharded exfiltration, where sensitive information is split across multiple requests to avoid detection. This strategy reduces single-request leakage metrics by 73% (Leak@1) and bypasses simple data loss prevention mechanisms. Our ablation results indicate that defenses applied at the prompt layer offer limited protection, while controls at the system and network layers, such as domain allowlisting and redirect-chain analysis, are considerably more effective. These findings suggest that network egress should be treated as a first-class security outcome in agentic LLM systems. We outline architectural directions, including provenance tracking and capability isolation, that go beyond prompt-level hardening.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22424v1">Causality $\neq$ Invariance: Function and Concept Vectors in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-25
    </div>
    <details class="paper-abstract">
      Do large language models (LLMs) represent concepts abstractly, i.e., independent of input format? We revisit Function Vectors (FVs), compact representations of in-context learning (ICL) tasks that causally drive task performance. Across multiple LLMs, we show that FVs are not fully invariant: FVs are nearly orthogonal when extracted from different input formats (e.g., open-ended vs. multiple-choice), even if both target the same concept. We identify Concept Vectors (CVs), which carry more stable concept representations. Like FVs, CVs are composed of attention head outputs; however, unlike FVs, the constituent heads are selected using Representational Similarity Analysis (RSA) based on whether they encode concepts consistently across input formats. While these heads emerge in similar layers to FV-related heads, the two sets are largely distinct, suggesting different underlying mechanisms. Steering experiments reveal that FVs excel in-distribution, when extraction and application formats match (e.g., both open-ended in English), while CVs generalize better out-of-distribution across both question types (open-ended vs. multiple-choice) and languages. Our results show that LLMs do contain abstract concept representations, but these differ from those that drive ICL performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.19680v3">PolicyPad: Collaborative Prototyping of LLM Policies</a></div>
    <div class="paper-meta">
      📅 2026-02-25
      | 💬 CHI 2026 paper. Supplementary materials: https://docs.google.com/document/d/1jBmKXusoWmCHfwpmNhSTJtbwZ5fwVWLNppKeqqd_-pY/edit?usp=sharing
    </div>
    <details class="paper-abstract">
      As LLMs gain adoption in high-stakes domains like mental health, domain experts are increasingly consulted to provide input into policies governing their behavior. From an observation of 19 policymaking workshops with 9 experts over 15 weeks, we identified opportunities to better support rapid experimentation, feedback, and iteration for collaborative policy design processes. We present PolicyPad, an interactive system that facilitates the emerging practice of LLM policy prototyping by drawing from established UX prototyping practices, including heuristic evaluation and storyboarding. Using PolicyPad, policy designers can collaborate on drafting a policy in real time while independently testing policy-informed model behavior with usage scenarios. We evaluate PolicyPad through workshops with 8 groups of 22 domain experts in mental health and law, finding that PolicyPad enhanced collaborative dynamics during policy design, enabled tight feedback loops, and led to novel policy contributions. Overall, our work paves expert-informed paths for advancing AI alignment and safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22402v1">Contextual Memory Virtualisation: DAG-Based State Management and Structurally Lossless Trimming for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-25
      | 💬 11 pages. 6 figures. Introduces a DAG-based state management system for LLM agents. Evaluation on 76 coding sessions shows up to 86% token reduction (mean 20%) while remaining economically viable under prompt caching. Includes reference implementation for Claude Code
    </div>
    <details class="paper-abstract">
      As large language models engage in extended reasoning tasks, they accumulate significant state -- architectural mappings, trade-off decisions, codebase conventions -- within the context window. This understanding is lost when sessions reach context limits and undergo lossy compaction. We propose Contextual Memory Virtualisation (CMV), a system that treats accumulated LLM understanding as version-controlled state. Borrowing from operating system virtual memory, CMV models session history as a Directed Acyclic Graph (DAG) with formally defined snapshot, branch, and trim primitives that enable context reuse across independent parallel sessions. We introduce a three-pass structurally lossless trimming algorithm that preserves every user message and assistant response verbatim while reducing token counts by a mean of 20% and up to 86% for sessions with significant overhead by stripping mechanical bloat such as raw tool outputs, base64 images, and metadata. A single-user case-study evaluation across 76 real-world coding sessions demonstrates that trimming remains economically viable under prompt caching, with the strongest gains in mixed tool-use sessions, which average 39% reduction and reach break-even within 10 turns. A reference implementation is available at https://github.com/CosmoNaught/claude-code-cmv.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.25369v2">Generative Value Conflicts Reveal LLM Priorities</a></div>
    <div class="paper-meta">
      📅 2026-02-25
      | 💬 Accepted to ICLR 2026 (the 14th International Conference on Learning Representations)
    </div>
    <details class="paper-abstract">
      Past work seeks to align large language model (LLM)-based assistants with a target set of values, but such assistants are frequently forced to make tradeoffs between values when deployed. In response to the scarcity of value conflict in existing alignment datasets, we introduce ConflictScope, an automatic pipeline to evaluate how LLMs prioritize different values. Given a user-defined value set, ConflictScope automatically generates scenarios in which a language model faces a conflict between two values sampled from the set. It then prompts target models with an LLM-written "user prompt" and evaluates their free-text responses to elicit a ranking over values in the value set. Comparing results between multiple-choice and open-ended evaluations, we find that models shift away from supporting protective values, such as harmlessness, and toward supporting personal values, such as user autonomy, in more open-ended value conflict settings. However, including detailed value orderings in models' system prompts improves alignment with a target ranking by 14%, showing that system prompting can achieve moderate success at aligning LLM behavior under value conflict. Our work demonstrates the importance of evaluating value prioritization in models and provides a foundation for future work in this area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22368v1">EyeLayer: Integrating Human Attention Patterns into LLM-Based Code Summarization</a></div>
    <div class="paper-meta">
      📅 2026-02-25
      | 💬 Accepted at the 34th IEEE/ACM International Conference on Program Comprehension (ICPC 2026), April 12-13, 2026, Rio de Janeiro, Brazil
    </div>
    <details class="paper-abstract">
      Code summarization is the task of generating natural language descriptions of source code, which is critical for software comprehension and maintenance. While large language models (LLMs) have achieved remarkable progress on this task, an open question remains: can human expertise in code understanding further guide and enhance these models? We propose EyeLayer, a lightweight attention-augmentation module that incorporates human eye-gaze patterns, as a proxy of human expertise, into LLM-based code summarization. EyeLayer models human attention during code reading via a Multimodal Gaussian Mixture, redistributing token embeddings based on learned parameters (μ_i, σ_i^2) that capture where and how intensively developers focus. This design enables learning generalizable attention priors from eye-tracking data and incorporating them into LLMs seamlessly, without disturbing existing representations. We evaluate EyeLayer across diverse model families (i.e., LLaMA-3.2, Qwen3, and CodeBERT) covering different scales and architectures. EyeLayer consistently outperforms strong fine-tuning baselines across standard metrics, achieving gains of up to 13.17% on BLEU-4. These results demonstrate that human gaze patterns encode complementary attention signals that enhance the semantic focus of LLMs and transfer effectively across diverse models for code summarization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16800v2">Large-scale online deanonymization with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-25
      | 💬 24 pages, 10 figures
    </div>
    <details class="paper-abstract">
      We show that large language models can be used to perform at-scale deanonymization. With full Internet access, our agent can re-identify Hacker News users and Anthropic Interviewer participants at high precision, given pseudonymous online profiles and conversations alone, matching what would take hours for a dedicated human investigator. We then design attacks for the closed-world setting. Given two databases of pseudonymous individuals, each containing unstructured text written by or about that individual, we implement a scalable attack pipeline that uses LLMs to: (1) extract identity-relevant features, (2) search for candidate matches via semantic embeddings, and (3) reason over top candidates to verify matches and reduce false positives. Compared to classical deanonymization work (e.g., on the Netflix prize) that required structured data, our approach works directly on raw user content across arbitrary platforms. We construct three datasets with known ground-truth data to evaluate our attacks. The first links Hacker News to LinkedIn profiles, using cross-platform references that appear in the profiles. Our second dataset matches users across Reddit movie discussion communities; and the third splits a single user's Reddit history in time to create two pseudonymous profiles to be matched. In each setting, LLM-based methods substantially outperform classical baselines, achieving up to 68% recall at 90% precision compared to near 0% for the best non-LLM method. Our results show that the practical obscurity protecting pseudonymous users online no longer holds and that threat models for online privacy need to be reconsidered.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22146v1">Provable Last-Iterate Convergence for Multi-Objective Safe LLM Alignment via Optimistic Primal-Dual</a></div>
    <div class="paper-meta">
      📅 2026-02-25
    </div>
    <details class="paper-abstract">
      Reinforcement Learning from Human Feedback (RLHF) plays a significant role in aligning Large Language Models (LLMs) with human preferences. While RLHF with expected reward constraints can be formulated as a primal-dual optimization problem, standard primal-dual methods only guarantee convergence with a distributional policy where the saddle-point problem is in convex-concave form. Moreover, standard primal-dual methods may exhibit instability or divergence in the last iterate under policy parameterization in practical applications. In this work, we propose a universal primal-dual framework for safe RLHF that unifies a broad class of existing alignment algorithms, including safe-RLHF, one-shot, and multi-shot based methods. Building on this framework, we introduce an optimistic primal-dual (OPD) algorithm that incorporates predictive updates for both primal and dual variables to stabilize saddle-point dynamics. We establish last-iterate convergence guarantees for the proposed method, covering both exact policy optimization in the distributional space and convergence to a neighborhood of the optimal solution whose gap is related to approximation error and bias under parameterized policies. Our analysis reveals that optimism plays a crucial role in mitigating oscillations inherent to constrained alignment objectives, thereby closing a key theoretical gap between constrained RL and practical RLHF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22077v1">ViSTAR: Virtual Skill Training with Augmented Reality with 3D Avatars and LLM coaching agent</a></div>
    <div class="paper-meta">
      📅 2026-02-25
    </div>
    <details class="paper-abstract">
      We present ViSTAR, a Virtual Skill Training system in AR that supports self-guided basketball skill practice, with feedback on balance, posture, and timing. From a formative study with basketball players and coaches, the system addresses three challenges: understanding skills, identifying errors, and correcting mistakes. ViSTAR follows the Behavioral Skills Training (BST) framework-instruction, modeling, rehearsal, and feedback. It provides feedback through visual overlays, rhythm and timing cues, and an AI-powered coaching agent using 3D motion reconstruction. We generate verbal feedback by analyzing spatio-temporal joint data and mapping features to natural-language coaching cues via a Large Language Model (LLM). A key novelty is this feedback generation: motion features become concise coaching insights. In two studies (N=16), participants generally preferred our AI-generated feedback to coach feedback and reported that ViSTAR helped them notice posture and balance issues and refine movements beyond self-observation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22067v1">Semantic Partial Grounding via LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-25
    </div>
    <details class="paper-abstract">
      Grounding is a critical step in classical planning, yet it often becomes a computational bottleneck due to the exponential growth in grounded actions and atoms as task size increases. Recent advances in partial grounding have addressed this challenge by incrementally grounding only the most promising operators, guided by predictive models. However, these approaches primarily rely on relational features or learned embeddings and do not leverage the textual and structural cues present in PDDL descriptions. We propose SPG-LLM, which uses LLMs to analyze the domain and problem files to heuristically identify potentially irrelevant objects, actions, and predicates prior to grounding, significantly reducing the size of the grounded task. Across seven hard-to-ground benchmarks, SPG-LLM achieves faster grounding-often by orders of magnitude-while delivering comparable or better plan costs in some domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.11517v2">PeruMedQA: Benchmarking Large Language Models (LLMs) on Peruvian Medical Exams -- Dataset Construction and Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-02-25
      | 💬 https://github.com/rodrigo-carrillo/PeruMedQA
    </div>
    <details class="paper-abstract">
      BACKGROUND: Medical large language models (LLMs) have demonstrated remarkable performance in answering medical examinations. However, the extent to which this high performance is transferable to medical questions in Spanish and from a Latin American country remains unexplored. This knowledge is crucial as LLM-based medical applications gain traction in Latin America. AIMS: To build a dataset of questions medical examinations taken by Peruvian physicians pursuing specialty training; to fine-tune a LLM on this dataset; to evaluate and compare the performance in terms of accuracy between vanilla LLMs and the fine-tuned LLM. METHODS: We curated PeruMedQA, a multiple-choice question-answering (MCQA) dataset containing 8,380 questions spanning 12 specialties (2018-2025). We selected ten medical LLMs, including medgemma-4b-it and medgemma-27b-text-it, and developed zero-shot task specific prompts to answer the questions. We employed parameter-efficient fine tuning (PEFT) and low-rand adaptation (LoRA) to fine-tune medgemma-4b-it utilizing all questions except those from 2025 (test set). RESULTS: Medgemma-27b showed the highest accuracy across all specialities, achieving the highest score of 89.29% in Psychiatry; yet, in two specialties, OctoMed-7B exhibited slight superiority: Neurosurgery with 77.27% and 77.38, respectively; and Radiology with 76.13% and 77.39%, respectively. Across specialties, most LLMs with <10 billion parameters exhibited <50% of correct answers. The fine-tuned version of medgemma-4b-it emerged victorious against all LLMs with <10 billion parameters and rivaled a LLM with 70 billion parameters across various examinations. CONCLUSIONS: For medical AI applications and research that require knowledge bases from Spanish-speaking countries and those exhibiting similar epidemiological profile to Peru's, interested parties should utilize medgemma-27b-text-it.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22296v1">UpSkill: Mutual Information Skill Learning for Structured Response Diversity in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-25
      | 💬 First two authors equal contribution. 29 pages total (11 pages main text), 10 figures, 10 tables. Project website: https://dshah.io/upskill/
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has improved the reasoning abilities of large language models (LLMs) on mathematics and programming tasks, but standard approaches that optimize single-attempt accuracy can inadvertently suppress response diversity across repeated attempts, narrowing exploration and overlooking underrepresented strategies. We introduce UpSkill, a training time method that adapts Mutual Information Skill Learning (MISL) to LLMs for optimizing pass@k correctness. We propose a novel reward that we implement within Group Relative Policy Optimization (GRPO): a token-level mutual information (MI) reward that encourages trajectory specificity to z. Experiments on GSM8K with three open-weight models, Llama 3.1-8B, Qwen 2.5-7B, and R1-Distilled-Qwen2.5-Math-1.5B, show that UpSkill improves multi-attempt metrics on the stronger base models, yielding mean gains of ~3% in pass@k for both Qwen and Llama without degrading pass@1. Additionally, we find both empirical and theoretical evidence that improvements in pass@k are closely tied to the mutual information objective.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22020v1">Detecting UX smells in Visual Studio Code using LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-25
      | 💬 4 pages, 2 figures, 1 table, 3rd International Workshop on Integrated Development Environments (IDE 2026)
    </div>
    <details class="paper-abstract">
      Integrated Development Environments shape developers' daily experience, yet the empirical study of their usability and user experience (UX) remains limited. This work presents an LLM-assisted approach to detecting UX smells in Visual Studio Code by mining and classifying user-reported issues from the GitHub repository. Using a validated taxonomy and expert review, we identified recurring UX problems that affect the developer experience. Our results show that the majority of UX smells are concentrated in informativeness, clarity, intuitiveness, and efficiency, qualities that developers value most.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22017v1">IOAgent: Democratizing Trustworthy HPC I/O Performance Diagnosis Capability via LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-25
      | 💬 Published in the Proceedings of the 2025 IEEE International Parallel and Distributed Processing Symposium (IPDPS 2025)
    </div>
    <details class="paper-abstract">
      As the complexity of the HPC storage stack rapidly grows, domain scientists face increasing challenges in effectively utilizing HPC storage systems to achieve their desired I/O performance. To identify and address I/O issues, scientists largely rely on I/O experts to analyze their I/O traces and provide insights into potential problems. However, with a limited number of I/O experts and the growing demand for data-intensive applications, inaccessibility has become a major bottleneck, hindering scientists from maximizing their productivity. Rapid advances in LLMs make it possible to build an automated tool that brings trustworthy I/O performance diagnosis to domain scientists. However, key challenges remain, such as the inability to handle long context windows, a lack of accurate domain knowledge about HPC I/O, and the generation of hallucinations during complex interactions.In this work, we propose IOAgent as a systematic effort to address these challenges. IOAgent integrates a module-based pre-processor, a RAG-based domain knowledge integrator, and a tree-based merger to accurately diagnose I/O issues from a given Darshan trace file. Similar to an I/O expert, IOAgent provides detailed justifications and references for its diagnoses and offers an interactive interface for scientists to ask targeted follow-up questions. To evaluate IOAgent, we collected a diverse set of labeled job traces and released the first open diagnosis test suite, TraceBench. Using this test suite, we conducted extensive evaluations, demonstrating that IOAgent matches or outperforms state-of-the-art I/O diagnosis tools with accurate and useful diagnosis results. We also show that IOAgent is not tied to specific LLMs, performing similarly well with both proprietary and open-source LLMs. We believe IOAgent has the potential to become a powerful tool for scientists navigating complex HPC I/O subsystems in the future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21997v1">Enhancing LLM-Based Test Generation by Eliminating Covered Code</a></div>
    <div class="paper-meta">
      📅 2026-02-25
      | 💬 9 pages, 4 figures, supplementary material included
    </div>
    <details class="paper-abstract">
      Automated test generation is essential for software quality assurance, with coverage rate serving as a key metric to ensure thorough testing. Recent advancements in Large Language Models (LLMs) have shown promise in improving test generation, particularly in achieving higher coverage. However, while existing LLM-based test generation solutions perform well on small, isolated code snippets, they struggle when applied to complex methods under test. To address these issues, we propose a scalable LLM-based unit test generation method. Our approach consists of two key steps. The first step is context information retrieval, which uses both LLMs and static analysis to gather relevant contextual information associated with the complex methods under test. The second step, iterative test generation with code elimination, repeatedly generates unit tests for the code slice, tracks the achieved coverage, and selectively removes code segments that have already been covered. This process simplifies the testing task and mitigates issues arising from token limits or reduced reasoning effectiveness associated with excessively long contexts. Through comprehensive evaluations on open-source projects, our approach outperforms state-of-the-art LLM-based and search-based methods, demonstrating its effectiveness in achieving high coverage on complex methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21951v1">RADAR: Reasoning as Discrimination with Aligned Representations for LLM-based Knowledge Graph Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-02-25
    </div>
    <details class="paper-abstract">
      Knowledge graph reasoning (KGR) infers missing facts, with recent advances increasingly harnessing the semantic priors and reasoning abilities of Large Language Models (LLMs). However, prevailing generative paradigms are prone to memorizing surface-level co-occurrences rather than learning genuine relational semantics, limiting out-of-distribution generalization. To address this, we propose RADAR, which reformulates KGR from generative pattern matching to discriminative relational reasoning. We recast KGR as discriminative entity selection, where reinforcement learning enforces relative entity separability beyond token-likelihood imitation. Leveraging this separability, inference operates directly in representation space, ensuring consistency with the discriminative optimization and bypassing generation-induced hallucinations. Across four benchmarks, RADAR achieves 5-6% relative gains on link prediction and triple classification over strong LLM baselines, while increasing task-relevant mutual information in intermediate representations by 62.9%, indicating more robust and transferable relational reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2406.05085v6">Multi-Head RAG: Solving Multi-Aspect Problems with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-25
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) improves Large Language Models (LLMs) by retrieving supporting documents into the prompt, but existing methods do not explicitly target queries that require fetching multiple documents with substantially different content. Such multi-aspect queries are challenging because relevant documents can be far apart in embedding space, making joint retrieval difficult. We introduce Multi-Head RAG (MRAG), which addresses this gap with a simple yet powerful idea: using Transformer multi-head attention activations rather than the standard decoder-layer embedding, as retrieval keys. It leverages the observation that different heads capture different semantic aspects. This yields multi-aspect embeddings for both documents and queries, improving retrieval accuracy on complex queries. We show MRAG's design advantages over 18 RAG baselines, up to 20% higher retrieval success ratios for real-world use cases, and improved downstream LLM generation. MRAG integrates seamlessly with existing RAG frameworks and benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21900v1">EmoOmni: Bridging Emotional Understanding and Expression in Omni-Modal LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-25
    </div>
    <details class="paper-abstract">
      The evolution of Omni-Modal Large Language Models~(Omni-LLMs) has revolutionized human--computer interaction, enabling unified audio-visual perception and speech response. However, existing Omni-LLMs struggle with complex real-world scenarios, often leading to superficial understanding and contextually mismatched emotional responses. This issue is further intensified by Omni-LLM's Thinker-Talker architectures, which are implicitly connected through hidden states, leading to the loss of emotional details. In this work, we present EmoOmni, a unified framework for accurate understanding and expression in multimodal emotional dialogue. At its core, we introduce the emotional Chain-of-Thought~(E-CoT), which enforces a reasoning from fine-grained multimodal perception to textual response. Moreover, we explicitly treat E-CoT as high-level emotional instructions that guide the talker, enabling accurate emotional expression. Complementing the model, we construct EmoOmniPipe to obtain the real-world annotated dialogue data and establish a benchmark, EmoOmniEval, to facilitate systematic assessment of multimodal emotional dialogue task. Experiments show that EmoOmni-7B achieves comparable performance with Qwen3Omni-30B-A3B-Thinking under the same talker.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.22811v3">Highly Efficient and Effective LLMs with Multi-Boolean Architectures</a></div>
    <div class="paper-meta">
      📅 2026-02-25
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      Weight binarization has emerged as a promising strategy to reduce the complexity of large language models (LLMs). Existing approaches fall into post-training binarization, which is simple but causes severe performance loss, and training-aware methods, which depend on full-precision latent weights, adding complexity and limiting efficiency. We propose a novel framework that represents LLMs with multi-kernel Boolean parameters and, for the first time, enables direct finetuning LMMs in the Boolean domain, eliminating the need for latent weights. This enhances representational capacity and dramatically reduces complexity during both finetuning and inference. Extensive experiments across diverse LLMs show our method outperforms recent ultra low-bit quantization and binarization techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21887v1">ExpLang: Improved Exploration and Exploitation in LLM Reasoning with On-Policy Thinking Language Selection</a></div>
    <div class="paper-meta">
      📅 2026-02-25
    </div>
    <details class="paper-abstract">
      Current large reasoning models (LRMs) have shown strong ability on challenging tasks after reinforcement learning (RL) based post-training. However, previous work mainly focuses on English reasoning in expectation of the strongest performance, despite the demonstrated potential advantage of multilingual thinking, as well as the requirement for native thinking traces by global users. In this paper, we propose ExpLang, a novel LLM post-training pipeline that enables on-policy thinking language selection to improve exploration and exploitation during RL with the use of multiple languages. The results show that our method steadily outperforms English-only training with the same training budget, while showing high thinking language compliance for both seen and unseen languages. Analysis shows that, by enabling on-policy thinking language selection as an action during RL, ExpLang effectively extends the RL exploration space with diversified language preference and improves the RL exploitation outcome with leveraged non-English advantage. The method is orthogonal to most RL algorithms and opens up a new perspective on using multilinguality to improve LRMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2402.00386v4">AssertLLM: Generating and Evaluating Hardware Verification Assertions from Design Specifications via Multi-LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-25
    </div>
    <details class="paper-abstract">
      Assertion-based verification (ABV) is a critical method for ensuring design circuits comply with their architectural specifications, which are typically described in natural language. This process often requires human interpretation by verification engineers to convert these specifications into functional verification assertions. Existing methods for generating assertions from natural language specifications are limited to sentences extracted by engineers, discouraging its practical application. In this work, we present AssertLLM, an automatic assertion generation framework that processes complete specification files. AssertLLM breaks down the complex task into three phases, incorporating three customized Large Language Models (LLMs) for extracting structural specifications, mapping signal definitions, and generating assertions. Our evaluation of AssertLLM on a full design, encompassing 23 I/O signals, demonstrates that 89\% of the generated assertions are both syntactically and functionally accurate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.05282v4">Not All Errors Are Created Equal: ASCoT Addresses Late-Stage Fragility in Efficient LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-02-25
    </div>
    <details class="paper-abstract">
      While Chain-of-Thought (CoT) prompting empowers Large Language Models (LLMs), ensuring reasoning reliability remains an open challenge. Contrary to the prevailing cascading failure hypothesis which posits that early errors are most detrimental, we identify a counter-intuitive phenomenon termed \textbf{Late-Stage Fragility}: errors introduced in later reasoning stages are significantly more prone to corrupting final answers. To address this, we introduce ASCoT (Adaptive Self-Correction Chain-of-Thought), a method harmonizing efficiency with robust verification. ASCoT first employs semantic pruning to compress redundant steps, then utilizes an Adaptive Verification Manager (AVM) to prioritize high risk, late-stage steps via a positional impact score, triggering a Multi-Perspective Self-Correction Engine (MSCE) only when necessary. Experiments on GSM8K and MATH-500 demonstrate that ASCoT effectively reallocates computational resources: it reduces token usage by 21\%--30\% for LLaMA-3.1-8B with negligible accuracy drops ($<1.8\%$), achieving a superior trade-off between inference efficiency and reasoning fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21763v1">Improving Implicit Discourse Relation Recognition with Natural Language Explanations from LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-25
      | 💬 AAAI26'0ral
    </div>
    <details class="paper-abstract">
      Implicit Discourse Relation Recognition (IDRR) remains a challenging task due to the requirement for deep semantic understanding in the absence of explicit discourse markers. A further limitation is that existing methods only predict relations without providing any supporting explanations. Recent advances in large language models (LLMs) have shown strong reasoning capabilities in both deep language understanding and natural language explanation generation. In this work, we propose a simple yet effective approach to distill the reasoning capabilities of LLMs into lightweight IDRR models to improve both performance and interpretability. Specifically, we first prompt an LLM to generate explanations for each training instance conditioned on its gold label. Then, we introduce a novel classification-generation framework that jointly performs relation prediction and explanation generation, and train it with the additional supervision of LLM-generated explanations. Our framework is plug-and-play, enabling easy integration with most existing IDRR models. Experimental results on PDTB demonstrate that our approach significantly improves IDRR performance, while human evaluation further confirms that the generated explanations enhance model interpretability. Furthermore, we validate the generality of our approach on sentiment classification and natural language inference
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21756v1">Offline Reasoning for Efficient Recommendation: LLM-Empowered Persona-Profiled Item Indexing</a></div>
    <div class="paper-meta">
      📅 2026-02-25
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) offer new opportunities for recommender systems by capturing the nuanced semantics of user interests and item characteristics through rich semantic understanding and contextual reasoning. In particular, LLMs have been employed as rerankers that reorder candidate items based on inferred user-item relevance. However, these approaches often require expensive online inference-time reasoning, leading to high latency that hampers real-world deployment. In this work, we introduce Persona4Rec, a recommendation framework that performs offline reasoning to construct interpretable persona representations of items, enabling lightweight and scalable real-time inference. In the offline stage, Persona4Rec leverages LLMs to reason over item reviews, inferring diverse user motivations that explain why different types of users may engage with an item; these inferred motivations are materialized as persona representations, providing multiple, human-interpretable views of each item. Unlike conventional approaches that rely on a single item representation, Persona4Rec learns to align user profiles with the most plausible item-side persona through a dedicated encoder, effectively transforming user-item relevance into user-persona relevance. At the online stage, this persona-profiled item index allows fast relevance computation without invoking expensive LLM reasoning. Extensive experiments show that Persona4Rec achieves performance comparable to recent LLM-based rerankers while substantially reducing inference time. Moreover, qualitative analysis confirms that persona representations not only drive efficient scoring but also provide intuitive, review-grounded explanations. These results demonstrate that Persona4Rec offers a practical and interpretable solution for next-generation recommender systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.00462v3">LatentLens: Revealing Highly Interpretable Visual Tokens in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-25
      | 💬 Updates: small change in interpretability percentage for Qwen-based variants we trained (pre-processing fix), clarification in Section 3 on our method (after feedback from readers), additional appendix section
    </div>
    <details class="paper-abstract">
      Transforming a large language model (LLM) into a Vision-Language Model (VLM) can be achieved by mapping the visual tokens from a vision encoder into the embedding space of an LLM. Intriguingly, this mapping can be as simple as a shallow MLP transformation. To understand why LLMs can so readily process visual tokens, we need interpretability methods that reveal what is encoded in the visual token representations at every layer of LLM processing. In this work, we introduce LatentLens, a novel approach for mapping latent representations to descriptions in natural language. LatentLens works by encoding a large text corpus and storing contextualized token representations for each token in that corpus. Visual token representations are then compared to their contextualized textual representations, with the top-k nearest neighbor representations providing descriptions of the visual token. We evaluate this method on 10 different VLMs, showing that commonly used methods, such as LogitLens, substantially underestimate the interpretability of visual tokens. With LatentLens instead, the majority of visual tokens are interpretable across all studied models and all layers. Qualitatively, we show that the descriptions produced by LatentLens are semantically meaningful and provide more fine-grained interpretations for humans compared to individual tokens. More broadly, our findings contribute new evidence on the alignment between vision and language representations, opening up new directions for analyzing latent representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21715v1">Two-Stage Active Distribution Network Voltage Control via LLM-RL Collaboration: A Hybrid Knowledge-Data-Driven Approach</a></div>
    <div class="paper-meta">
      📅 2026-02-25
    </div>
    <details class="paper-abstract">
      The growing integration of distributed photovoltaics (PVs) into active distribution networks (ADNs) has exacerbated operational challenges, making it imperative to coordinate diverse equipment to mitigate voltage violations and enhance power quality. Although existing data-driven approaches have demonstrated effectiveness in the voltage control problem, they often require extensive trial-and-error exploration and struggle to incorporate heterogeneous information, such as day-ahead forecasts and semantic-based grid codes. Considering the operational scenarios and requirements in real-world ADNs, in this paper, we propose a hybrid knowledge-data-driven approach that leverages dynamic collaboration between a large language model (LLM) agent and a reinforcement learning (RL) agent to achieve two-stage voltage control. In the day-ahead stage, the LLM agent receives coarse region-level forecasts and generates scheduling strategies for on-load tap changer (OLTC) and shunt capacitors (SCs) to regulate the overall voltage profile. Then in the intra-day stage, based on accurate node-level measurements, the RL agent refines terminal voltages by deriving reactive power generation strategies for PV inverters. On top of the LLM-RL collaboration framework, we further propose a self-evolution mechanism for the LLM agent and a pretrain-finetune pipeline for the RL agent, effectively enhancing and coordinating the policies for both agents. The proposed approach not only aligns more closely with practical operational characteristics but also effectively utilizes the inherent knowledge and reasoning capabilities of the LLM agent, significantly improving training efficiency and voltage control performance. Comprehensive comparisons and ablation studies demonstrate the effectiveness of the proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.00024v2">EpidemIQs: Prompt-to-Paper LLM Agents for Epidemic Modeling and Analysis</a></div>
    <div class="paper-meta">
      📅 2026-02-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) offer new opportunities to accelerate complex interdisciplinary research domains. Epidemic modeling, characterized by its complexity and reliance on network science, dynamical systems, epidemiology, and stochastic simulations, represents a prime candidate for leveraging LLM-driven automation. We introduce EpidemIQs, a novel multi-agent LLM framework that integrates user inputs and autonomously conducts literature review, analytical derivation, network modeling, mechanistic modeling, stochastic simulations, data visualization and analysis, and finally documentation of findings in a structured manuscript, through five predefined research phases. We introduce two types of agents: a scientist agent for planning, coordination, reflection, and generation of final results, and a task-expert agent to focus exclusively on one specific duty serving as a tool to the scientist agent. The framework consistently generated complete reports in scientific article format. Specifically, using GPT 4.1 and GPT 4.1 Mini as backbone LLMs for scientist and task-expert agents, respectively, the autonomous process completes with average total token usage 870K at a cost of about $1.57 per study, successfully executing all phases and final report. We evaluate EpidemIQs across several different epidemic scenarios, measuring computational cost, workflow reliability, task success rate, and LLM-as-Judge and human expert reviews to estimate the overall quality and technical correctness of the generated results. Through our experiments, the framework consistently addresses evaluation scenarios with an average task success rate of 79%. We compare EpidemIQs to an iterative single-agent LLM, benefiting from the same system prompts and tools, iteratively planning, invoking tools, and revising outputs until task completion. The comparisons suggest a consistently higher performance of EpidemIQs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.03255v2">SciTS: Scientific Time Series Understanding and Generation with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-25
      | 💬 Accepted to ICLR 2026
    </div>
    <details class="paper-abstract">
      The scientific reasoning ability of large language models (LLMs) has recently attracted significant attention. Time series, as a fundamental modality in scientific data, presents unique challenges that are often overlooked in current multimodal LLMs, which either encode numerical sequences as text or convert them into images. Such approaches may be insufficient for comprehensive scientific time series understanding and generation. Existing unified time series models typically specialise in either forecasting or analysis, and their effectiveness on non-periodic, heterogeneous scientific signals remains unclear. To address these gaps, we introduce SciTS, a benchmark spanning 12 scientific domains and 43 tasks, with over 50k+ instances, both univariate and multivariate signals ranging from $10^0$ to $10^7$ in length and up to 10~MHz in frequency. We benchmark 17 models, including text-only LLMs, multimodal LLMs, and unified time series models, and find that general-purpose LLMs exhibit stronger generalisability than specialised time series models, while representing time series as text or images limits their performance due to excessively long sequences and loss of numerical precision, respectively. We then introduce TimeOmni, a framework that equips LLMs with the ability to understand and generate time series while remaining compatible with general-purpose LLM training. This work fills a gap in both dedicated benchmarks and modelling frameworks for scientific time series, paving the way for LLMs to understand and generate complex temporal scientific data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22271v1">Support Tokens, Stability Margins, and a New Foundation for Robust LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-25
      | 💬 39 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Self-attention is usually described as a flexible, content-adaptive way to mix a token with information from its past. We re-interpret causal self-attention transformers, the backbone of modern foundation models, within a probabilistic framework, much like how classical PCA is extended to probabilistic PCA. However, this re-formulation reveals a surprising and deeper structural insight: due to a change-of-variables phenomenon, a barrier constraint emerges on the self-attention parameters. This induces a highly structured geometry on the token space, providing theoretical insights into the dynamics of LLM decoding. This reveals a boundary where attention becomes ill-conditioned, leading to a margin interpretation similar to classical support vector machines. Just like support vectors, this naturally gives rise to the concept of support tokens. Furthermore, we show that LLMs can be interpreted as a stochastic process over the power set of the token space, providing a rigorous probabilistic framework for sequence modeling. We propose a Bayesian framework and derive a MAP estimation objective that requires only a minimal modification to standard LLM training: the addition of a smooth log-barrier penalty to the usual cross-entropy loss. We demonstrate that this provides more robust models without sacrificing out-of-sample accuracy and that it is straightforward to incorporate in practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21686v1">"Without AI, I Would Never Share This Online": Unpacking How LLMs Catalyze Women's Sharing of Gendered Experiences on Social Media</a></div>
    <div class="paper-meta">
      📅 2026-02-25
      | 💬 This poster was conditionally accepted to CHI 2026
    </div>
    <details class="paper-abstract">
      Sharing gendered experiences on social media has been widely recognized as supporting women's personal sense-making and contributing to digital feminism. However, there are known concerns, such as fear of judgment and backlash, that may discourage women from posting online. In this study, we examine a recurring practice on Xiaohongshu, a popular Chinese social media platform, in which women share their gendered experiences alongside screenshots of conversations with LLMs. We conducted semi-structured interviews with 20 women to investigate whether and how interactions with LLMs might support women in articulating and sharing gendered experiences. Our findings reveal that, beyond those external concerns, women also hold self-imposed standards regarding what feels appropriate and worthwhile to share publicly. We further show how interactions with LLMs help women meet these standards and navigate such concerns. We conclude by discussing how LLMs might be carefully and critically leveraged to support women's everyday expression online.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21681v1">AkiraRust: Re-thinking LLM-aided Rust Repair Using a Feedback-guided Thinking Switch</a></div>
    <div class="paper-meta">
      📅 2026-02-25
      | 💬 7 pages, 11 figures, accepted to DAC
    </div>
    <details class="paper-abstract">
      Eliminating undefined behaviors (UBs) in Rust programs requires a deep semantic understanding to enable accurate and reliable repair. While existing studies have demonstrated the potential of LLMs to support Rust code analysis and repair, most frameworks remain constrained by inflexible templates or lack grounding in executable semantics, resulting in limited contextual awareness and semantic incorrectness. Here, we present AkiraRust, an LLM-driven repair and verification framework that incorporates a finite-state machine to dynamically adapt its detection and repair flow to runtime semantic conditions. AkiraRust introduces a dual-mode reasoning strategy that coordinates fast and slow thinking across multiple agents. Each agent is mapped to an FSM state, and a waveform-driven transition controller manages state switching, rollback decisions, and semantic check pointing, enabling context-aware and runtime-adaptive repair. Experimental results show that AkiraRust achieves about 92% semantic correctness and delivers a 2.2x average speedup compared to SOTA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21158v2">SELAUR: Self Evolving LLM Agent via Uncertainty-aware Rewards</a></div>
    <div class="paper-meta">
      📅 2026-02-25
      | 💬 Accepted by PAKDD'26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed as multi-step decision-making agents, where effective reward design is essential for guiding learning. Although recent work explores various forms of reward shaping and step-level credit assignment, a key signal remains largely overlooked: the intrinsic uncertainty of LLMs. Uncertainty reflects model confidence, reveals where exploration is needed, and offers valuable learning cues even in failed trajectories. We introduce SELAUR: Self Evolving LLM Agent via Uncertainty-aware Rewards, a reinforcement learning framework that incorporates uncertainty directly into the reward design. SELAUR integrates entropy-, least-confidence-, and margin-based metrics into a combined token-level uncertainty estimate, providing dense confidence-aligned supervision, and employs a failure-aware reward reshaping mechanism that injects these uncertainty signals into step- and trajectory-level rewards to improve exploration efficiency and learning stability. Experiments on two benchmarks, ALFWorld and WebShop, show that our method consistently improves success rates over strong baselines. Ablation studies further demonstrate how uncertainty signals enhance exploration and robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21626v1">Multi-Layer Scheduling for MoE-Based LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-02-25
      | 💬 12 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success across a wide range of tasks, but serving them efficiently at scale remains a critical challenge due to their substantial computational and latency demands. While most existing inference frameworks rely on simple scheduling strategies such as First-Come-First-Serve (FCFS) at the engine level and Round-Robin (RR) at the scheduler or coordinator level, they often fail to fully utilize system resources and may suffer from issues such as head-of-line blocking and load imbalance. Recent advances in Mixture-of-Experts (MoE) models have also introduced new challenges in scheduling arising from expert parallelism and routing complexity. This research proposes a multi-layer scheduling framework tailored for MoE-based LLM serving. It targets scheduling at three levels: request-level, enginelevel, and expert-level. At the request level, we explore algorithms such as Shortest-Job-First (SJF) and priority-aware aging to improve throughput and reduce latency. At the engine level, we design load-aware dispatching strategies that account for the current prefix token load, KV cache utilization, and user stickiness to achieve better resource matching. At the expert level, we focus on alleviating expert hotspots and strategically placing inter-layer expert dependencies to balance load and improve routing efficiency. Extensive experimental results from more than 100 experiments conducted under diverse workload distributions show that our approach consistently outperforms the state-of-theart inference framework vLLM, achieving up to 17.8% reduction in Time To First Token (TTFT) latency and 13.3% reduction in Time-Per-Output-Token (TPOT) latency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2406.08305v3">MSADM: Large Language Model (LLM) Assisted End-to-End Network Health Management Based on Multi-Scale Semanticization</a></div>
    <div class="paper-meta">
      📅 2026-02-25
    </div>
    <details class="paper-abstract">
      Network device and system health management is the foundation of modern network operations and maintenance. Traditional health management methods, relying on expert identification or simple rule-based algorithms, struggle to cope with the heterogeneous networks (HNs) environment. Moreover, current state-of-the-art distributed fault diagnosis methods, which utilize specific machine learning techniques, lack multi-scale adaptivity for heterogeneous device information, resulting in unsatisfactory diagnostic accuracy for HNs. In this paper, we develop an LLM-assisted end-to-end intelligent network health management framework. The framework first proposes a multi-scale data scaling method based on unsupervised learning to address the multi-scale data problem in HNs. Secondly, we combine the semantic rule tree with the attention mechanism to propose a Multi-Scale Semanticized Anomaly Detection Model (MSADM) that generates network semantic information while detecting anomalies. Finally, we embed a chain-of-thought-based large-scale language model downstream to adaptively analyze the fault diagnosis results and create an analysis report containing detailed fault information and optimization strategies. We compare our scheme with other fault diagnosis models and demonstrate that it performs well on several metrics of network fault diagnosis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21593v1">Breaking Semantic-Aware Watermarks via LLM-Guided Coherence-Preserving Semantic Injection</a></div>
    <div class="paper-meta">
      📅 2026-02-25
      | 💬 Accepted by The Web Conference 2026 (Short Paper Track)
    </div>
    <details class="paper-abstract">
      Generative images have proliferated on Web platforms in social media and online copyright distribution scenarios, and semantic watermarking has increasingly been integrated into diffusion models to support reliable provenance tracking and forgery prevention for web content. Traditional noise-layer-based watermarking, however, remains vulnerable to inversion attacks that can recover embedded signals. To mitigate this, recent content-aware semantic watermarking schemes bind watermark signals to high-level image semantics, constraining local edits that would otherwise disrupt global coherence. Yet, large language models (LLMs) possess structured reasoning capabilities that enable targeted exploration of semantic spaces, allowing locally fine-grained but globally coherent semantic alterations that invalidate such bindings. To expose this overlooked vulnerability, we introduce a Coherence-Preserving Semantic Injection (CSI) attack that leverages LLM-guided semantic manipulation under embedding-space similarity constraints. This alignment enforces visual-semantic consistency while selectively perturbing watermark-relevant semantics, ultimately inducing detector misclassification. Extensive empirical results show that CSI consistently outperforms prevailing attack baselines against content-aware semantic watermarking, revealing a fundamental security weakness of current semantic watermark designs when confronted with LLM-driven semantic perturbations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21585v1">Duel-Evolve: Reward-Free Test-Time Scaling via LLM Self-Preferences</a></div>
    <div class="paper-meta">
      📅 2026-02-25
    </div>
    <details class="paper-abstract">
      Many applications seek to optimize LLM outputs at test time by iteratively proposing, scoring, and refining candidates over a discrete output space. Existing methods use a calibrated scalar evaluator for the target objective to guide search, but for many tasks such scores are unavailable, too sparse, or unreliable. Pairwise comparisons, by contrast, are often easier to elicit, still provide useful signal on improvement directions, and can be obtained from the LLM itself without external supervision. Building on this observation, we introduce Duel-Evolve, an evolutionary optimization algorithm that replaces external scalar rewards with pairwise preferences elicited from the same LLM used to generate candidates. Duel-Evolve aggregates these noisy candidate comparisons via a Bayesian Bradley-Terry model, yielding uncertainty-aware estimates of candidate quality. These quality estimates guide allocation of the comparison budget toward plausible optima using Double Thompson Sampling, as well as selection of high-quality parents to generate improved candidates. We evaluate Duel-Evolve on MathBench, where it achieves 20 percentage points higher accuracy over existing methods and baselines, and on LiveCodeBench, where it improves over comparable iterative methods by over 12 percentage points. Notably, the method requires no reward model, no ground-truth labels during search, and no hand-crafted scoring function. Results show that pairwise self-preferences provide strong optimization signal for test-time improvement over large, discrete output spaces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.25184v2">Incentive-Aligned Multi-Source LLM Summaries</a></div>
    <div class="paper-meta">
      📅 2026-02-25
      | 💬 Accepted at ICLR 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in modern search and answer systems to synthesize multiple, sometimes conflicting, texts into a single response, yet current pipelines offer weak incentives for sources to be accurate and are vulnerable to adversarial content. We introduce Truthful Text Summarization (TTS), an incentive-aligned framework that improves factual robustness without ground-truth labels. TTS (i) decomposes a draft synthesis into atomic claims, (ii) elicits each source's stance on every claim, (iii) scores sources with an adapted multi-task peer-prediction mechanism that rewards informative agreement, and (iv) filters unreliable sources before re-summarizing. We establish formal guarantees that align a source's incentives with informative honesty, making truthful reporting the utility-maximizing strategy. Experiments show that TTS improves factual accuracy and robustness while preserving fluency, aligning exposure with informative corroboration and disincentivizing manipulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.26784v3">LLMs Process Lists With General Filter Heads</a></div>
    <div class="paper-meta">
      📅 2026-02-25
      | 💬 Code and data at https://filter.baulab.info/
    </div>
    <details class="paper-abstract">
      We investigate the mechanisms underlying a range of list-processing tasks in LLMs, and we find that LLMs have learned to encode a compact, causal representation of a general filtering operation that mirrors the generic "filter" function of functional programming. Using causal mediation analysis on a diverse set of list-processing tasks, we find that a small number of attention heads, which we dub filter heads, encode a compact representation of the filtering predicate in their query states at certain tokens. We demonstrate that this predicate representation is general and portable: it can be extracted and reapplied to execute the same filtering operation on different collections, presented in different formats, languages, or even in tasks. However, we also identify situations where transformer LMs can exploit a different strategy for filtering: eagerly evaluating if an item satisfies the predicate and storing this intermediate result as a flag directly in the item representations. Our results reveal that transformer LMs can develop human-interpretable implementations of abstract computational operations that generalize in ways that are surprisingly similar to strategies used in traditional functional programming patterns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14697v3">Evolutionary System Prompt Learning for Reinforcement Learning in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-25
      | 💬 39 pages, 22 figures
    </div>
    <details class="paper-abstract">
      Building agentic systems that can autonomously self-improve from experience is a longstanding goal of AI. Large language models (LLMs) today primarily self-improve via two mechanisms: self-reflection for context updates, and reinforcement learning (RL) for weight updates. In this work, we propose Evolutionary System Prompt Learning (E-SPL), a method for jointly improving model contexts and model weights. In each RL iteration, E-SPL samples trajectories under multiple system prompts in parallel, then jointly applies RL updates to LLM weights and evolutionary updates to system prompts. System prompts evolve via mutation and crossover, two genetic operators driven by LLM self-reflection; selection is based on relative performance ratings updated across RL iterations. E-SPL encourages a natural division between declarative knowledge encoded in prompts and procedural knowledge encoded in weights, resulting in improved performance across reasoning and agentic tasks. For instance, in an easy-to-hard (AIME $\rightarrow$ BeyondAIME) generalization setting, E-SPL improves RL success rate from 38.8% $\rightarrow$ 45.1% while also outperforming reflective prompt evolution (40.0%). Overall, our results demonstrate that RL and system prompt evolution are deeply synergistic, and combining the two yields consistent gains in sample efficiency and generalization. Code: https://github.com/LunjunZhang/E-SPL
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22261v1">Sustainable LLM Inference using Context-Aware Model Switching</a></div>
    <div class="paper-meta">
      📅 2026-02-25
      | 💬 15 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large language models have become central to many AI applications, but their growing energy consumption raises serious sustainability concerns. A key limitation in current AI deployments is the reliance on a one-size-fits-all inference strategy where most systems route every request to the same large model, regardless of task complexity, leading to substantial and unnecessary energy waste. To address this issue, we propose a context-aware model switching approach that dynamically selects an appropriate language model based on query complexity. The proposed system uses a Context-Aware Model Switching for Energy-Efficient LLM Inference that combines caching for repeated queries, rulebased complexity scoring for fast and explainable decisions, machine learning classification to capture semantic intent, and a user-adaptive component that learns from interaction patterns over time. The proposed architecture was evaluated using real conversation workloads and three open-source language models (Gemma3 1B, Gemma3 4B and Qwen3 4B) with different computational costs, measuring energy consumption (via NVML GPU power telemetry), response latency, routing accuracy, and output quality (BERTScore F1) to reflect real-world usage conditions. Experimental results show that the model switching approach can reduce energy consumption by up to 67.5% compared to always using the largest model while maintaining a response quality of 93.6%. In addition, the response time for simple queries also improved significantly by approximately 68%. These results show that model switching inference offers a practical and scalable path toward more energy-efficient and sustainable AI systems, demonstrating that significant efficiency gains can be achieved without major sacrifices in response quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10431v3">QTALE: Quantization-Robust Token-Adaptive Layer Execution for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-25
      | 💬 8 pages, 6 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demand substantial computational and memory resources, posing challenges for efficient deployment. Two complementary approaches have emerged to address these issues: token-adaptive layer execution, which reduces floating-point operations (FLOPs) by selectively bypassing layers, and quantization, which lowers memory footprint by reducing weight precision. However, naively integrating these techniques leads to additional accuracy degradation due to reduced redundancy in token-adaptive models. We propose QTALE (Quantization-Robust Token-Adaptive Layer Execution for LLMs), a novel framework that enables seamless integration of token-adaptive execution with quantization while preserving accuracy. Conventional token-adaptive methods reduce redundancy in two ways: (1) by limiting the diversity of training paths explored during fine-tuning, and (2) by lowering the number of parameters actively involved in inference. To overcome these limitations, QTALE introduces two key components: (1) a training strategy that ensures diverse execution paths are actively explored during fine-tuning, and (2) a post-training mechanism that allows flexible adjustment of the execution ratio at inference to reintroduce redundancy when needed. Experimental results show that QTALE enables seamless integration of token-adaptive layer execution with quantization, showing no noticeable accuracy difference, with the gap to quantization-only models kept below 0.5% on CommonsenseQA benchmarks. By combining token-adaptive execution for FLOPs reduction and quantization for memory savings, QTALE provides an effective solution for efficient LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.07667v3">1-2-3 Check: Enhancing Contextual Privacy in LLM via Multi-Agent Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-02-25
      | 💬 Accepted at the International Association for AI Safety and Ethics AI (IASEAI) 2026
    </div>
    <details class="paper-abstract">
      Addressing contextual privacy concerns remains challenging in interactive settings where large language models (LLMs) process information from multiple sources (e.g., summarizing meetings with private and public information). We introduce a multi-agent framework that decomposes privacy reasoning into specialized subtasks (extraction, classification), reducing the information load on any single agent while enabling iterative validation and more reliable adherence to contextual privacy norms. To understand how privacy errors emerge and propagate, we conduct a systematic ablation over information-flow topologies, revealing when and why upstream detection mistakes cascade into downstream leakage. Experiments on the ConfAIde and PrivacyLens benchmark with several open-source and closed-sourced LLMs demonstrate that our best multi-agent configuration substantially reduces private information leakage (\textbf{18\%} on ConfAIde and \textbf{19\%} on PrivacyLens with GPT-4o) while preserving the fidelity of public content, outperforming single-agent baselines. These results highlight the promise of principled information-flow design in multi-agent systems for contextual privacy with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.18755v2">BiScale: Energy-Efficient Disaggregated LLM Serving via Phase-Aware Placement and DVFS</a></div>
    <div class="paper-meta">
      📅 2026-02-25
    </div>
    <details class="paper-abstract">
      Prefill/decode disaggregation is increasingly adopted in LLM serving to improve the latency-throughput tradeoff and meet strict TTFT and TPOT SLOs. However, LLM inference remains energy-hungry: autoscaling alone is too coarse-grained to track fast workload fluctuations, and applying fine-grained DVFS under disaggregation is complicated by phase-asymmetric dynamics and coupling between provisioning and frequency control. We present BiScale, a two-tier energy optimization framework for disaggregated LLM serving. BiScale jointly optimizes placement and DVFS across prefill and decode using predictive latency and power models. At coarse timescales, BiScale computes phase-aware placement and baseline frequencies that minimize energy while satisfying SLO constraints. At fine timescales, BiScale dynamically adapts GPU frequency per iteration using stage-specific control: model predictive control (MPC) for prefill to account for queue evolution and future TTFT impact, and lightweight slack-aware adaptation for decode to exploit its smoother, memory-bound dynamics. This hierarchical design enables coordinated control across timescales while preserving strict serving SLOs. Evaluation on a 16x H100 cluster serving Llama 3.3 70B with production-style traces shows that BiScale meets TTFT/TPOT SLOs while reducing energy by up to 39% in prefill and 48% in decode relative to DistServe.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21492v1">GradAlign: Gradient-Aligned Data Selection for LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-02-25
      | 💬 14 pages. Preliminary work
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become a central post-training paradigm for large language models (LLMs), but its performance is highly sensitive to the quality of training problems. This sensitivity stems from the non-stationarity of RL: rollouts are generated by an evolving policy, and learning is shaped by exploration and reward feedback, unlike supervised fine-tuning (SFT) with fixed trajectories. As a result, prior work often relies on manual curation or simple heuristic filters (e.g., accuracy), which can admit incorrect or low-utility problems. We propose GradAlign, a gradient-aligned data selection method for LLM reinforcement learning that uses a small, trusted validation set to prioritize training problems whose policy gradients align with validation gradients, yielding an adaptive curriculum. We evaluate GradAlign across three challenging data regimes: unreliable reward signals, distribution imbalance, and low-utility training corpus, showing that GradAlign consistently outperforms existing baselines, underscoring the importance of directional gradient signals in navigating non-stationary policy optimization and yielding more stable training and improved final performance. We release our implementation at https://github.com/StigLidu/GradAlign
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19922v2">HEART: A Unified Benchmark for Assessing Humans and LLMs in Emotional Support Dialogue</a></div>
    <div class="paper-meta">
      📅 2026-02-25
    </div>
    <details class="paper-abstract">
      Supportive conversation depends on skills that go beyond language fluency, including reading emotions, adjusting tone, and navigating moments of resistance, frustration, or distress. Despite rapid progress in language models, we still lack a clear way to understand how their abilities in these interpersonal domains compare to those of humans. We introduce HEART, the first-ever framework that directly compares humans and LLMs on the same multi-turn emotional-support conversations. For each dialogue history, we pair human and model responses and evaluate them through blinded human raters and an ensemble of LLM-as-judge evaluators. All assessments follow a rubric grounded in interpersonal communication science across five dimensions: Human Alignment, Empathic Responsiveness, Attunement, Resonance, and Task-Following. HEART uncovers striking behavioral patterns. Several frontier models approach or surpass the average human responses in perceived empathy and consistency. At the same time, humans maintain advantages in adaptive reframing, tension-naming, and nuanced tone shifts, particularly in adversarial turns. Human and LLM-as-judge preferences align on about 80 percent of pairwise comparisons, matching inter-human agreement, and their written rationales emphasize similar HEART dimensions. This pattern suggests an emerging convergence in the criteria used to assess supportive quality. By placing humans and models on equal footing, HEART reframes supportive dialogue as a distinct capability axis, separable from general reasoning or linguistic fluency. It provides a unified empirical foundation for understanding where model-generated support aligns with human social judgment, where it diverges, and how affective conversational competence scales with model size.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21477v1">Pancake: Hierarchical Memory System for Multi-Agent LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-02-25
    </div>
    <details class="paper-abstract">
      In this work, we identify and address the core challenges of agentic memory management in LLM serving, where large-scale storage, frequent updates, and multiple coexisting agents jointly introduce complex and high-cost approximate nearest neighbor (ANN) searching problems. We present Pancake, a multi-tier agentic memory system that unifies three key techniques: (i) multi-level index caching for single agents, (ii) coordinated index management across multiple agents, and (iii) collaborative GPU-CPU acceleration. Pancake exposes easy-to-use interface that can be integrated into memory-based agents like Mem-GPT, and is compatible with agentic frameworks such as LangChain and LlamaIndex. Experiments on realistic agent workloads show that Pancake substantially outperforms existing frameworks, achieving more than 4.29x end-to-end throughput improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.02452v3">Do LLMs Adhere to Label Definitions? Examining Their Receptivity to External Label Definitions</a></div>
    <div class="paper-meta">
      📅 2026-02-24
      | 💬 EMNLP 2025 (Main Conference)
    </div>
    <details class="paper-abstract">
      Do LLMs genuinely incorporate external definitions, or do they primarily rely on their parametric knowledge? To address these questions, we conduct controlled experiments across multiple explanation benchmark datasets (general and domain-specific) and label definition conditions, including expert-curated, LLM-generated, perturbed, and swapped definitions. Our results reveal that while explicit label definitions can enhance accuracy and explainability, their integration into an LLM's task-solving processes is neither guaranteed nor consistent, suggesting reliance on internalized representations in many cases. Models often default to their internal representations, particularly in general tasks, whereas domain-specific tasks benefit more from explicit definitions. These findings underscore the need for a deeper understanding of how LLMs process external knowledge alongside their pre-existing capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.07452v3">When Style Breaks Safety: Defending LLMs Against Superficial Style Alignment</a></div>
    <div class="paper-meta">
      📅 2026-02-24
      | 💬 Accepted by ICLR 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can be prompted with specific styles (e.g., formatting responses as lists), including in malicious queries. Prior jailbreak research mainly augments these queries with additional string transformations to maximize attack success rate (ASR). However, the impact of style patterns in the original queries that are semantically irrelevant to the malicious intent remains unclear. In this work, we seek to understand whether style patterns compromise LLM safety, how superficial style alignment increases model vulnerability, and how best to mitigate these risks during alignment. We first define ASR inflation as the increase in ASR due to style patterns in existing jailbreak benchmark queries. By evaluating 36 LLMs across seven benchmarks, we find that nearly all models exhibit ASR inflation. Notably, the inflation correlates with an LLM's relative attention to style patterns, which also overlap more with its instruction-tuning data when inflation occurs. We then investigate superficial style alignment, and find that fine-tuning with specific styles makes LLMs more vulnerable to jailbreaks of those same styles. Finally, we propose SafeStyle, a defense strategy that incorporates a small amount of safety training data augmented to match the distribution of style patterns in the fine-tuning data. Across three LLMs, six fine-tuning style settings, and two real-world instruction-tuning datasets, SafeStyle consistently outperforms baselines in maintaining LLM safety.
    </details>
</div>
