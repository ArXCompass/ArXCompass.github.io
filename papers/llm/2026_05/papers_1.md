# llm - 2026_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22823v1">Which Way Did It Move? Diagnosing and Overcoming Directional Motion Blindness in Video-LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 Preprint. 59 pages, including appendix. Code: https://github.com/KHU-VLL/DeltaDirect
    </div>
    <details class="paper-abstract">
      Video Large Language Models (Video-LLMs) have made rapid progress on temporal video understanding, yet many fail at a basic perceptual primitive: signed image-plane motion direction. On simple videos of a single object moving left, right, up, or down, most Video-LLMs perform near chance, with above-chance cases largely attributable to prediction biases rather than genuine direction understanding. We call this failure directional motion blindness. We localize the failure by tracing motion direction information through the Video-LLM pipeline. Motion direction remains linearly accessible from the vision encoder, projector, and LLM hidden states, but the readout fails to bind this signal to the correct verbal answer option, revealing a direction binding gap. Although synthetic motion direction instruction tuning reduces this gap on the source domain, motion direction concept vector analysis shows that visual complexity weakens the signal magnitude and limits out-of-domain generalization. We introduce MoDirect, a dataset family for motion direction instruction tuning and evaluation, and DeltaDirect, a diagnosis-driven, projector-level objective that predicts normalized 2-D motion vectors from adjacent-frame feature deltas. On MoDirect-SynBench, instruction tuning with DeltaDirect improves motion direction accuracy from 25.9% to 85.4%. On MoDirect-RealBench, DeltaDirect improves real-world motion direction accuracy by 21.9 points over the vanilla baseline without real-world tuning data, while preserving standard video-understanding performance. Code: https://github.com/KHU-VLL/DeltaDirect
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.11665v5">Beyond LLMs, Sparse Distributed Memory, and Neuromorphics <A Hyper-Dimensional SRAM-CAM "VaCoAl" for Ultra-High Speed, Ultra-Low Power, and Low Cost></a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 57 pages, 4 figure, 18 tables
    </div>
    <details class="paper-abstract">
      This paper reports an unexpected finding: in a deterministic hyperdimensional computing (HDC) architecture **that inverts the conventional role of Galois-field algebra -- employing it not for error correction toward a unique answer but as an engine for relative similarity and path-quality ranking -- **a path-dependent semantic selection mechanism emerges, equivalent to spike-timing-dependent plasticity (STDP), with magnitude predictable a priori from a closed-form expression matching measured values. Addressing catastrophic forgetting, learning stagnation, and the Binding Problem at an algebraic level, we propose VaCoAl (Vague Coincident Algorithm) and its Python implementation PyVaCoAl on ultra-high-dimensional SRAM/DRAM-CAM. Rooted in Sparse Distributed Memory, it resolves orthogonalisation and retrieval in high-dimensional binary spaces via Galois-field diffusion, enabling low-load deployment. Crucially, VaCoAl embeds a cognitive bound -- the Frontier Size -- into its architecture, ranking candidates by path-integral confidence (CR2) to achieve compositional generalisation; this bounded-rationality design produces STDP-like selection that error-correction paradigms structurally cannot attain. We evaluated multi-hop reasoning on about 470k mentor-student relations from Wikidata, tracing up to 57 generations (over 25.5M paths). HDC bundling and unbinding with CR-based denoising quantify concept propagation over DAGs. Results show a reinterpretation of the Newton-Leibniz dispute and a phase transition from sparse convergence to a post-Leibniz "superhighway", with structural indicators supporting a Kuhnian paradigm shift. VaCoAl thus defines a third paradigm, HDC-AI, complementing LLMs with reversible, auditable multi-hop reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22732v1">Beyond Acoustic Emotion Recognition: Multimodal Pathos Analysis in Political Speech Using LLM-Based and Acoustic Emotion Models</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 13 pages, 1 figure
    </div>
    <details class="paper-abstract">
      We investigate whether acoustic emotion recognition models can serve as proxies for the Pathos dimension in political speech analysis, as operationalised by the TRUST multi-agent large language model (LLM) pipeline. Using a Bundestag plenary speech by Felix Banaszak (51 segments, 245 s) as a case study, we compare three analysis modalities: (1) emotion2vec_plus_large, an acoustic speech emotion recognition (SER) model whose continuous Arousal and Valence values are derived via post-hoc Russell Circumplex projection; (2) Gemini 2.5 Flash, an LLM analysing the full speech audio together with its transcript in an open-ended, context-aware fashion; and (3) TRUST-Pathos scores from a three-advocate LLM supervisor ensemble. Spearman rank correlations reveal that Gemini Valence correlates strongly with TRUST-Pathos (rho = +0.664, p < 0.001), whereas emotion2vec Valence does not (rho = +0.097, p = 0.499). We further demonstrate, via a systematic quality evaluation of the Berlin Database of Emotional Speech (EMO-DB) using Gemini in an open-ended annotation paradigm, that standard SER benchmark corpora suffer from acted speech, cultural bias, and category incompatibility. Our results suggest that LLM-based multimodal analysis captures semantically defined political emotion substantially better than acoustic models alone, while acoustic features remain informative for low-level Arousal estimation. Future work will extend this approach to video-based analysis incorporating facial expression and gaze.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.27355v2">LLM Readiness Harness: Evaluation, Observability, and CI Gates for LLM/RAG Applications</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 19 pages, 4 figures, 15 tables
    </div>
    <details class="paper-abstract">
      We present a readiness harness for LLM and RAG applications that turns evaluation into a deployment decision workflow. The system combines automated benchmarks, OpenTelemetry observability, and CI quality gates under a minimal API contract, then aggregates workflow success, policy compliance, groundedness, retrieval hit rate, cost, and p95 latency into scenario-weighted readiness scores with Pareto frontiers. We evaluate the harness on ticket-routing workflows and BEIR grounding tasks (SciFact and FiQA) with full Azure matrix coverage (162/162 valid cells across datasets, scenarios, retrieval depths, seeds, and models). Results show that readiness is not a single metric: on FiQA under sla-first at k=5, gpt-4.1-mini leads in readiness and faithfulness, while gpt-5.2 pays a substantial latency cost; on SciFact, models are closer in quality but still separable operationally. Ticket-routing regression gates consistently reject unsafe prompt variants, demonstrating that the harness can block risky releases instead of merely reporting offline scores. The result is a reproducible, operationally grounded framework for deciding whether an LLM or RAG system is ready to ship.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21347v2">Insights Generator: Systematic Corpus-Level Trace Diagnostics for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-21
    </div>
    <details class="paper-abstract">
      Diagnosing failures in LLM agents remains largely manual. Practitioners inspect a small subset of execution traces, form ad-hoc hypotheses, and iterate. This process misses patterns that only emerge across trace populations and does not scale to production corpora where individual traces span tens of thousands of tokens. We formalize the problem of corpus-level trace diagnostics. Given a corpus of execution traces, the goal is to produce grounded natural-language insights that characterize systematic behavioral patterns across trace groups, each linked to supporting evidence. We present the Insights Generator (IG), a multi-agent system that answers diagnostic questions by proposing and testing hypotheses across the trace corpus to produce an evidence-backed insights report. We evaluate IG across qualitative and objective dimensions, spanning rubric-based report assessment and downstream performance improvements achieved by implementing IG insights. Human experts using IG reports improve scaffold performance by 30.4pp over the unmodified baseline scaffold, and coding agents leveraging IG-derived insights show consistent and stable gains. Across benchmarks, IG's scout-investigator architecture produces findings comparable in detection coverage to competing approaches, while domain experts rated IG reports as leading depth and evidence quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22714v1">AMEL: Accumulated Message Effects on LLM Judgments</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 19 pages, 14 figures, 6 tables. Single author. Code, data (75,898 deduplicated API responses), and analysis pipeline at https://github.com/chutapp/amel
    </div>
    <details class="paper-abstract">
      Large language models are routinely used as automated evaluators: to review code, moderate content, or score outputs, often with many items passing through one conversation. We ask whether the polarity of prior conversation history biases subsequent judgments, an effect we call the accumulated message effect on LLM judgments (AMEL). Across 75,898 API calls to 11 models from 4 providers (OpenAI, Anthropic, Google, and four open-source models), we present identical test items in isolation or following histories saturated with predominantly positive or negative evaluations. Models shift toward the conversation's prevailing polarity (d = -0.17, p < 10^-46). The effect concentrates on items where the model is genuinely uncertain at baseline (d = -0.34 for high-entropy items, vs d = -0.15 when the baseline is deterministic). Bias does not grow with context length: 5 prior turns and 50 produce the same shift (Spearman |r| < 0.01; OLS slope p = 0.80). And there is a negativity asymmetry: paired per item, negative histories induce 1.62x more bias than positive (t = 13.46, p < 10^-39, n = 2,481). Scaling helps but does not solve it (Anthropic: Haiku -0.22 to Opus -0.17; OpenAI: Nano -0.34 to GPT-5.2 -0.17). Three follow-ups narrow the mechanism. The token probability distribution shifts continuously, not at a threshold. The negativity asymmetry has both token-level and semantic components, though attributing the balance is exploratory at our sample sizes. Position does not matter: five biased turns anywhere in a 50-turn history produce the same shift. The simplest fix for evaluation pipelines is a fresh context per item; when batching is unavoidable, balancing the history helps.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.13910v2">RAGCap-Bench: Benchmarking Capabilities of LLMs in Agentic Retrieval Augmented Generation Systems</a></div>
    <div class="paper-meta">
      📅 2026-05-21
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) mitigates key limitations of Large Language Models (LLMs)-such as factual errors, outdated knowledge, and hallucinations-by dynamically retrieving external information. Recent work extends this paradigm through agentic RAG systems, where LLMs act as agents to iteratively plan, retrieve, and reason over complex queries. However, these systems still struggle with challenging multi-hop questions, and their intermediate reasoning capabilities remain underexplored. To address this, we propose RAGCap-Bench, a capability-oriented benchmark for fine-grained evaluation of intermediate tasks in agentic RAG workflows. We analyze outputs from state-of-the-art systems to identify common tasks and the core capabilities required for their execution, then construct a taxonomy of typical LLM errors to design targeted evaluation questions. Experiments show that "slow-thinking" models with stronger RAGCap performance achieve better end-to-end results, underscoring the benchmark's validity and the importance of enhancing these intermediate capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.16590v2">Atom-anchored LLMs speak Chemistry: A Retrosynthesis Demonstration</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 Alan Kai Hassen and Andrius Bernatavicius contributed equally to this work
    </div>
    <details class="paper-abstract">
      Applications of machine learning in chemistry are often limited by the scarcity and expense of labeled data, restricting traditional supervised methods. In this work, we introduce a framework for molecular reasoning using general-purpose Large Language Models (LLMs) that operates without requiring task-specific model training. Our method anchors chain-of-thought reasoning to the molecular structure by using unique atomic identifiers. First, the LLM performs a zero-shot task to identify relevant fragments and their associated chemical labels or transformation classes. In an optional second step, this position-aware information is used in a few-shot task with provided class examples to predict the chemical transformation. We apply our framework to single-step retrosynthesis, a task where LLMs have previously underperformed. Across academic benchmarks and expert-validated drug discovery molecules, our work enables LLMs to achieve high success rates in identifying chemically plausible reaction sites ($\geq90\%$), named reaction classes ($\geq40\%$), and final reactants ($\geq74\%$). Ultimately, our work establishes a general blueprint for applying LLMs to challenges where molecular reasoning and molecular transformations are key, positioning atom-anchored LLMs as a powerful solution for data-scarce chemistry domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22664v1">WorkstreamBench: Evaluating LLM Agents on End-to-End Spreadsheet Tasks in Finance</a></div>
    <div class="paper-meta">
      📅 2026-05-21
    </div>
    <details class="paper-abstract">
      LLM agents are increasingly expected to carry out end-to-end workflows, producing complete artifacts from high-level user instructions. To meet enterprise needs, frontier AI labs have developed agents that can construct entire spreadsheets from scratch. This is especially relevant in finance, where core workflows such as financial modeling, forecasting, and scenario analysis are commonly conducted through spreadsheets. Yet, existing spreadsheet benchmarks do not measure this advanced capability, focusing instead on question-answering or single-formula edits. To address this gap, we provide one of the first evaluations of agents on end-to-end spreadsheet tasks, focusing on economically critical financial workflows such as modeling and scenario analysis. Since deliverables therein are routinely reviewed and revised by multiple stakeholders, judging their quality necessarily involves high-level criteria such as readability or ease of modification. To reflect the multidimensional nature of solution quality, we develop an evaluation taxonomy comprising three dimensions: Accuracy, Formula, and Format, each comprising fine-grained criteria that reflect professional standards. The Claude family leads the benchmark and produces the most professional-looking outputs in our qualitative review, but even the strongest agents frequently fall short of professional finance standards and degrade sharply as the difficulty increases beyond a few chained calculations. This suggests that current agents are not yet able to reliably produce professional-quality spreadsheets at the level of complexity real-world workflows demand.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22613v1">Evolutionary Multi-Task Optimization for LLM-Guided Program Discovery</a></div>
    <div class="paper-meta">
      📅 2026-05-21
    </div>
    <details class="paper-abstract">
      Recent LLM-guided evolutionary search methods have shown that iterative program mutation can discover strong algorithms, but they typically optimize each task independently, even when related tasks share reusable structure. We introduce Evolutionary Multi-Task Optimization (EMO) for LLM-guided program discovery, and propose EMO-STA (Shared-Then-Adapt), a two-stage framework that first evolves a shared archive of executable programs across a task family and then adapts selected shared candidates to each target task. Within EMO-STA, we explore multiple adaptation strategies, including warm-starting from the shared archive, adapting the best average shared program, and adapting the shared program that performs best on each target task. Across eight task families spanning continuous optimization, geometric construction, modeling, and algorithmic optimization, EMO-STA improves over matched-compute single-task evolution in most settings, with STA Best-Local providing the strongest in-distribution adaptation and STA Best-Shared yielding robust transfer to unseen tasks. Compute-allocation experiments show that allocating a substantial fraction of the family-level budget to shared evolution is consistently beneficial, with roughly balanced shared and adaptation budgets often being optimal. Beyond compute efficiency, we show that shared evolution can mitigate overfitting in low-evidence settings (e.g. few training data), including ARC tasks and time-series feature engineering, by favoring programs that generalize across all tasks rather than exploiting task-specific brittle artifacts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22608v1">Agentic CLEAR: Automating Multi-Level Evaluation of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 ACL
    </div>
    <details class="paper-abstract">
      Agentic systems are becoming more capable: agents define strategies, take actions, and interact with different environments. This autonomy poses serious challenges for overseeing and assessing agent behavior. Most current tools are limited, focusing on observability with basic evaluation capabilities or imposing static, hand-crafted error taxonomies that cannot adapt to new domains. To address this gap, we present Agentic CLEAR, an automatic, dynamic, and easy-to-use evaluation framework. It produces textual insights into the agent behavior on three levels of granularity: system, trace, and node. Agentic CLEAR operates above the observability layer, enabling seamless integration and featuring an intuitive UI that makes agent evaluation highly accessible. In our experiments on four benchmarks, seven agentic settings, and tens of thousands of LLM calls, we show that Agentic CLEAR produces high-quality, data-driven, insightful feedback. Our analysis shows strong alignment with human-annotated errors and the ability to predict task success rate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06669v2">Evaluating Prompt Injection Defenses for Educational LLM Tutors: Security-Usability-Latency Trade-offs</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 19 pages, 4 figures, 9 tables
    </div>
    <details class="paper-abstract">
      Educational LLM tutors face a core AI alignment challenge: they must follow user intent while preserving pedagogical constraints and safety policies. We present an evaluation methodology for prompt-injection defenses in this setting, showing that guardrail design entails explicit trade-offs among adversarial robustness, benign-task usability, and response latency. We evaluate a domain-specific multi-layer safeguard pipeline combining deterministic pattern filters, structural validation, contextual sandboxing, and session-level behavioral checks. On a controlled holdout benchmark, the pipeline reaches low bypass and false positive rates with optimized average latency - an operating point that prioritizes pedagogical usability (zero false positives) while maintaining measurable attack resistance. We provide a reproducible benchmark protocol for head-to-head comparison under identical conditions, including stratified bootstrap confidence intervals, paired McNemar significance tests, multi-seed sensitivity sweeps, and direct evaluation of Prompt Guard and NeMo Guardrails on the same split with unified instrumentation. Results expose operational trade-offs: NeMo reaches 0 percent bypass at 16.22 percent FPR and roughly 1.5s latency, while Prompt Guard yields 38.48 percent bypass with 3.60 percent FPR. The framework supports evidence-based guardrail selection for AI tutoring systems under different institutional risk and usability requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09851v2">CoFEH: LLM-driven Feature Engineering Empowered by Collaborative Bayesian Hyperparameter Optimization</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 Accepted at KDD 2026. Extended version with full appendices
    </div>
    <details class="paper-abstract">
      Feature Engineering (FE) is pivotal in automated machine learning (AutoML) but remains a bottleneck for traditional methods, which operate within rigid search spaces and lack domain awareness. While Large Language Models (LLMs) offer a promising alternative to generate unbounded operators with semantic reasoning, existing methods focus on isolated subtasks such as feature generation, falling short of free-form FE pipelines. Moreover, they are rarely coupled with hyperparameter optimization (HPO) of the downstream ML model, leading to greedy "FE-then-HPO" workflows that cannot capture strong FE-HPO interactions. In this paper, we present CoFEH, a collaborative framework that interleaves LLM-based FE and Bayesian HPO for robust end-to-end AutoML. CoFEH uses an LLM-driven FE optimizer powered by Tree of Thought (TOT) to explore flexible FE pipelines, a Bayesian optimization (BO) module to solve HPO, and a dynamic optimizer selector that adaptively interleaves FE and HPO steps. Crucially, we introduce a mutual conditioning mechanism that shares context between LLM and BO, enabling mutually informed decisions. Experiments show that CoFEH outperforms both traditional and LLM-based baselines in both standalone FE and joint FE+HPO settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.02885v3">"Would You Want an AI Tutor?" Understanding Stakeholder Perceptions of LLM-based Systems in the Classroom</a></div>
    <div class="paper-meta">
      📅 2026-05-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have gained traction in educational settings, often framed as virtual tutors or teaching assistants. Following early skepticism and bans, many schools and universities have begun integrating these systems into curricula. Yet decisions about whether and how to deploy LLM-based tools are frequently made without systematic engagement with the full range of stakeholders they affect. In this paper, we argue that understanding stakeholder perceptions of LLM-based systems in the classroom is not a matter of measuring approval or acceptance, but of identifying whose concerns are surfaced, in which contexts, and with what implications for responsible design and governance. We introduce Contextualized Perceptions for the Adoption of LLMs in Education (Co-PALE), a stakeholder-first framework that connects educational context, responsible AI principles, and categories of perception to support more deliberate decision-making about the adoption of LLM-based tools. We ground Co-PALE through a targeted analysis of prior work to diagnose recurring gaps in how stakeholder perceptions are studied, and through contextually distinct educational scenarios that illustrate how the same technology raises different concerns for different stakeholders. We further examine how university faculty and K--12 parents make sense of the framework through focus groups, using their reflections to surface tensions and uncertainties. Co-PALE supports more systematic reasoning about whether, where, and for whom LLM-based tools should be deployed in education.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22566v1">GraphFlow: A Graph-Based Workflow Management for Efficient LLM-Agent Serving</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 Accepted to ICML 2026
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based agents demonstrate strong reasoning and execution capabilities on complex tasks when guided by structured instructions, commonly referred to as workflows. However, existing workflow-assisted agent serving systems typically rely on predefined templates and shallow matching mechanisms, which limit their ability to capture deep semantic relationships and generalize to previously unseen tasks. To address these limitations, we propose a new workflow management paradigm that represents workflows using a unified graph, termed wGraph, where each node corresponds to an atomic operation. wGraph serves as a shared substrate from which task-specific workflows are dynamically instantiated. Building on wGraph primitives, we introduce GraphFlow, a system that efficiently integrates workflows into agent serving through two key designs. First, adaptive workflow generation dynamically constructs workflows from wGraph based on task semantics and constraint requirements. Second, workflow state management exploits wGraph structure to efficiently manage Key-Value (KV) caches, reducing redundant computation during agent serving. Extensive experiments across five benchmark datasets show that GraphFlow consistently outperforms state-of-the-art methods, yielding an average performance improvement of approximately 4.95 percentage points, while achieving an approximately 4$\times$ reduction in memory footprint.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.00688v2">Provably Protecting Fine-Tuned LLMs from Training Data Extraction while Preserving Utility</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 21 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) on sensitive datasets raises privacy concerns, as training data extraction (TDE) attacks can expose highly confidential information. Existing defenses against such attacks either lack formal privacy guarantees or incur substantial utility degradation. We observe that fine-tuning induces widespread probability shifts, yet preserving only a small subset of influential token-level deviations is sufficient; the remaining shifts can be aggressively smoothed with minimal impact on utility. Motivated by this insight, we propose SCP-$Δ_r$, a Near Access Freeness (NAF)-based algorithm that operates on relative probabilities and explicitly smooths low-impact tokens using a base model. SCP-$Δ_r$ achieves orders-of-magnitude better theoretical bounds than existing NAF based methods and provides strong empirical protection against TDE attacks with minimal performance loss.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.16984v2">Closing the Gap at CRAC 2026: Two-Stage Adaptation for LLM-Based Multilingual Coreference Resolution</a></div>
    <div class="paper-meta">
      📅 2026-05-21
    </div>
    <details class="paper-abstract">
      We present our submission to the LLM track of the 2026 Computational Models of Reference, Anaphora and Coreference (CRAC 2026) shared task. With an average CoNLL F1 score of 74.32 on the official test set, our system ranked first in the LLM track, and third overall. Our system is based on the Gemma-3-27b model, fine-tuned using a two-stage strategy with a multilingual base adapter followed by dataset-specific adapters. We represent mention spans by their headword using an XML-inspired format with local reindexing and annotate documents iteratively. These design choices proved effective across languages, document lengths, and annotation guidelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.15676v2">Automated Self-Testing as a Quality Gate: Evidence-Driven Release Management for LLM Applications</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 20 pages, 6 figures, 12 tables
    </div>
    <details class="paper-abstract">
      LLM applications are AI systems whose nondeterministic outputs and evolving model behavior make traditional testing insufficient for release governance. We present an automated self-testing framework that introduces quality gates with evidence-based release decisions (PROMOTE/HOLD/ROLLBACK) across five empirically grounded dimensions: task success rate, research context preservation, P95 latency, safety pass rate, and evidence coverage. We evaluate the framework through a longitudinal case study of an internally deployed multi-agent conversational AI system with specific marketing capabilities in active development, covering 38 evaluation runs across 20+ internal releases. The gate identified two ROLLBACK-grade builds in early runs and supported stable quality evolution over a four-week staging lifecycle while exercising persona-grounded, multi-turn, adversarial, and evidence-required scenarios. Statistical analysis (Mann-Kendall trends, Spearman correlations, bootstrap confidence intervals), gate ablation, and overhead scaling indicate that evidence coverage is the primary severe-regression discriminator and that runtime scales predictably with suite size. A human calibration study (n=60 stratified cases, two independent evaluators, LLM-as-judge cross-validation) reveals complementary multi-modal coverage: LLM-judge disagreements with the system gate (kappa=0.13) are attributable to structural failure modes - latency violations and routing errors - invisible in response text alone, while the judge independently surfaces content quality failures missed by structural checks, consistent with a multi-dimensional gate design. The framework, supplementary pseudocode, and calibration artifacts are provided to support AI-system quality assurance and independent replication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.16299v2">ACE: Self-Evolving LLM Coding Framework via Adversarial Unit Test Generation and Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2026-05-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at code generation but remain heavily reliant on large-scale annotated solutions and verification-based supervision, which constrains scalability and hinders sustained self-improvement. Recent solver--verifier frameworks exploit program execution as an automatic supervision signal, but their effectiveness degrades as solvers become moderately strong: verifier-generated tests increasingly confirm semantic correctness rather than exposing the remaining failure modes. We propose \textbf{ACE}, a self-evolving code generation framework based on a solver--adversary architecture that prioritizes active failure discovery through execution-centric supervision. A single LLM alternates between generating candidate programs and producing adversarial unit test inputs optimized to induce execution-level failures, such as runtime errors, exceptions, or non-termination. Supervision is derived solely from execution outcomes: robust programs are selected for supervised fine-tuning, while adversarial tests are optimized via Kahneman--Tversky Optimization using execution-derived preferences. Notably, the entire training loop requires no ground-truth code or external reward models. Experiments on CodeContests, MBPP, and LiveCodeBench demonstrate that ACE consistently outperforms strong solver--verifier baselines, achieving 3--7\% absolute gains in pass@1, with larger improvements on out-of-distribution benchmarks, while maintaining competitive or improved inference efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.02709v3">ATLAS: A Multi-LLM Training Framework for EvoDPO with Adaptive Reference Evolution</a></div>
    <div class="paper-meta">
      📅 2026-05-21
    </div>
    <details class="paper-abstract">
      Recent multi-LLM agent systems have shown promising capabilities for automated problem-solving, yet they predominantly rely on frozen agents or static fine-tuning pipelines. To address this limitation, our primary contribution is ATLAS (Adaptive Task-distributed Learning for Agentic Self-evolution), a multi-agent framework where specialized meta-agents collaboratively train and refine an active agent toward a domain-specific policy. A core challenge in iterative preference learning within these pipelines is the reliance on fixed reference models, which typically leads to overly conservative updates or training stagnation. To overcome this, the framework's algorithmic engine utilizes Evolving Direct Preference Optimization (EvoDPO). EvoDPO employs an inspection agent to perform adaptive, proxy-KL gated reference policy updates based on continuous training telemetry. We evaluate this full framework across a diverse set of challenging environments-including non-stationary contextual bandits, partial differential equations (PINNs), and combinatorial optimization tasks (TSP, Bin Packing). Through comparison against fixed-reference, adaptive-reference, and external automated-discovery baselines, our results suggest that ATLAS combines supporter-driven exploration with EvoDPO-driven stability to improve long-horizon evaluator-driven self-improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22502v1">Compiling Agentic Workflows into LLM Weights: Near-Frontier Quality at Two Orders of Magnitude Less Cost</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 19 pages
    </div>
    <details class="paper-abstract">
      Agent orchestration frameworks have proliferated, collectively exceeding 290,000 GitHub stars across LangGraph, CrewAI, Google ADK, OpenAI Agents SDK, Semantic Kernel, Strands, and LlamaIndex. All follow the same pattern: an external orchestrator above the LLM, injecting instructions and routing decisions every turn. Recent work has shown this architecture is dominated for procedural tasks by simply providing the procedure in a frontier model's system prompt [Dennis et al., 2026a], at the cost of consuming the context window, requiring a frontier model for every conversation, and exposing proprietary procedures to third-party providers. Compiling the procedure into the weights of a small fine-tuned model -- creating a subterranean agent -- should resolve all of these concerns, and prior work (SimpleTOD, FireAct, SynTOD, WorkflowLLM, Agent Lumos) has shown the technique works. Yet developer adoption has overwhelmingly favored orchestration. We identify three perceived barriers and address each empirically across travel booking (14 nodes), Zoom support (14 nodes, product-specific knowledge), and insurance claims (55 nodes, 6 decision hubs).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.07985v2">Dooly: Configuration-Agnostic, Redundancy-Aware Profiling for LLM Inference Simulation</a></div>
    <div class="paper-meta">
      📅 2026-05-21
    </div>
    <details class="paper-abstract">
      Selecting the optimal LLM inference configuration requires evaluation across hardware, serving engines, attention backends, and model architectures, since no single choice performs best across all workloads. Profile-based simulators are the standard tool, yet they hardcode their operation set to a specific configuration and re-profile every operation from scratch, making exploration prohibitively expensive. This cost stems from a missing structural understanding: every input dimension of each operation is fixed by the model configuration or determined by the incoming request. Many model-configuration values (e.g., head size, layer count) recur across models, so the same operation runs in many configurations; a single sweep over the request-dependent dimensions can serve them all. We present Dooly, which exploits this structure to achieve configuration-agnostic, redundancy-aware profiling. Dooly performs a single inference pass, labels each input dimension with its origin via taint propagation, and selectively profiles only operations absent from its latency database; stateful operations such as attention are isolated by reusing the serving engine's own initialization code, eliminating manual instrumentation. It builds latency regression models based on the database, which becomes a drop-in backend for existing simulators. Across two GPU platforms, three attention backends, and diverse model architectures, Dooly achieves simulation accuracy within 5% MAPE for TTFT and 8% for TPOT while reducing profiling GPU-hours by 56.4% across 12 models compared to the existing profiling approach. We have open-sourced Dooly at https://github.com/dooly-project.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06556v2">Semantic Attacks on Tool-Augmented LLMs: Securing the Model Context Protocol Against Descriptor-Level Manipulation</a></div>
    <div class="paper-meta">
      📅 2026-05-21
    </div>
    <details class="paper-abstract">
      The Model Context Protocol (MCP) enables Large Language Models (LLMs) to interact with external tools via tool descriptors, thereby extending their capabilities for task execution, autonomous decision-making, and multi-agent coordination. Existing MCP deployments treat tool descriptors as trusted metadata, despite their direct integration into the LLM reasoning context. This introduces a previously underexplored semantic attack surface. Current defenses primarily target prompt injection, neglecting descriptor-level manipulation that can bias tool selection and downstream reasoning. To address this gap, we formalize three descriptor-driven attack classes: Tool Poisoning, Shadowing, and Rug Pull. We propose a layered defense solution that integrates descriptor integrity verification, pre-context semantic vetting with an auxiliary LLM, and lightweight runtime guardrails, without requiring model retraining. We evaluate GPT-5.3, DeepSeek-V3, and LLaMA-3.5 across eight prompting strategies in controlled, adversarial MCP scenarios in which tool metadata is manipulated to simulate realistic attacks. Results demonstrate that descriptor manipulation can substantially alter tool-selection behavior, producing unsafe tool invocations in up to 36% of trials under baseline configurations. The proposed full-stack mitigation reduces unsafe invocations to 15% while increasing the block rate to 74%, demonstrating substantial improvement in resistance to descriptor-driven attacks. Cross-model analysis further reveals significant differences in robustness, latency, and sensitivity to descriptor-level manipulation across LLM architectures and prompting strategies. This study provides a controlled cross-model evaluation of descriptor-level threats and mitigation strategies in tool-calling LLM systems, establishing an empirical foundation for deploying secure and resilient tool-augmented LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22456v1">Steins;Gate Drive: Semantic Safety Arbitration over Structured Futures for Latency-Decoupled LLM Planning</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 10 pages, 2 figures, 5 tables, submitted to IEEE transaction of intelligent vehicles
    </div>
    <details class="paper-abstract">
      Cloud-hosted LLM driver agents provide useful semantic judgments, but their inference latency exceeds stepwise vehicle-control windows. Learned world models predict futures, but they usually keep future generation and action selection inside large coupled loops. We present SteinsGateDrive, a latency-decoupled planner-runtime architecture in which the worldline metaphor from the eponymous story names one plausible consequence of an intervention: the LLM selects counterfactual driving futures before the final control instant, and a runtime reuses the selected forecast only while safety contracts remain valid. The generator builds three world-line roles: alpha nominal ego-conditioned futures, beta interaction counterfactuals around nearby vehicles, and gamma hazard-stress futures such as braking, cut-ins, or blocked corridors. The selected branch becomes a typed StrategicForecast with horizon, validity/abort conditions, fallback, and authority. On a within-subject, matched-seed normal-highway protocol with 10 seeds and 20 steps, GPT-5.4 mini reduces effective lag from +3.07 s at 1-second horizon to -0.01 s at 4-second horizon while preserving the measured no-collision safety boundary. The architecture's safety contribution comes from the atom-predicate runtime check, not from the drift score, which functions as a refresh-frequency knob.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22655v3">Do Fine-Tuned LLMs Understand Vulnerabilities? An Investigation into the Semantic Trap</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown promising performance in software vulnerability detection, particularly after domain-specific Supervised Fine-Tuning (SFT). However, it remains unclear whether these models genuinely internalize vulnerability root causes or merely exploit surface-level functional patterns. While prior work documented related failures on pre-trained or zero-shot models, the SFT process itself, and how explicit reasoning supervision modulates it, remains under-explored. We study fine-tuned decoder-only LLMs under vanilla SFT and SFT with reasoning supervision, identifying a failure mode we term the Semantic Trap, characterized by three symptoms: pairing-sensitive performance, gap-dictated decisions, and fragility to semantic-preserving changes. To probe this, we propose TrapEval, an evaluation framework comprising two real-world datasets, V2P (vulnerable paired with patched code) and V2N (vulnerable paired with unrelated normal code), alongside semantic perturbations, CodeBLEU-based gap analysis, and an LLM-assisted reasoning failure taxonomy. Evaluating five representative LLMs fine-tuned with and without explicit reasoning (Chain-of-Thought), our results show vanilla SFT yields deceptively high scores on unpaired data (V2N) while failing all three symptoms. Models suffer high false-positive rates on V2P, degrade under perturbations, and exhibit a systematic dependency on the textual gap between vulnerable and patched code. Finetuning with explicit reasoning reduces these symptoms but costs recall; its lack of measurable gap-dependency partly reflects a floor effect rather than escaping the trap. Furthermore, our taxonomy reveals these models still misinterpret control flow and hallucinate API behavior, indicating current fine-tuning mitigates but does not eliminate reliance on surface features.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22389v1">Unified Data Selection for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Effectively training Large Language Models (LLMs) for complex, long-CoT reasoning is often bottlenecked by the need for massive high-quality reasoning data. Existing methods are either computationally expensive or fail to reliably distinguish high- from low-quality reasoning samples. To address this, we propose High-Entropy Sum (HES), a training-free metric that quantifies reasoning quality by summing only the entropy of the top (e.g., 0.5\%) highest-entropy tokens in each reasoning sample. We validate HES across three mainstream training paradigms: Supervised Fine-tuning (SFT), Rejection Fine-tuning (RFT), and Reinforcement Learning (RL), with extensive results demonstrating its consistent effectiveness and significantly reduced computational overhead. In SFT, training on the top 20\% HES-ranked data matches full-dataset performance, while using the lowest-HES data degrades it. In RFT, our HES-based training approach significantly outperforms baseline methods. In RL, HES-selected successful trajectories enable the model to learn strong reasoning patterns, significantly surpassing other compared methods. Our findings establish HES as a robust, training-free metric that enables a unified, effective, and efficient method for developing advanced reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.05406v3">Frame In, Frame Out: Measuring Framing Bias in LLM-Generated News Summaries</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 Accepted to The 15th Joint Conference on Lexical and Computational Semantics (*SEM 2026) co-located with ACL 2026
    </div>
    <details class="paper-abstract">
      News headlines and summaries shape how events are interpreted through selective emphasis and omission, a phenomenon commonly referred to as framing. Large language models are now routinely used to generate such content, yet existing evaluation frameworks largely overlook this dimension. We introduce Frame In, Frame Out (FIFO), the first large-scale benchmark for measuring framing presence in LLM-generated news summaries, grounded in the widely used XSum dataset. FIFO combines 15,499 jury-annotated examples with 320 expert-labeled instances ($κ= 0.61$) to validate and calibrate model-based annotations. Using FIFO, we analyze measured framing rates across 27 summarization models. We find that LLM-generated summaries often exhibit higher calibrated framing rates than human-written references, with substantial variation across topics and training regimes, including elevated rates in scientific and public health summaries. Our results establish framing as an underexplored and consequential dimension of summarization quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22297v1">One LR Doesn't Fit All: Heavy-Tail Guided Layerwise Learning Rates for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-21
    </div>
    <details class="paper-abstract">
      Learning rate configuration is a fundamental aspect of modern deep learning. The prevailing practice of applying a uniform learning rate across all layers overlooks the structural heterogeneity of Transformers, potentially limiting their effectiveness as the backbone of Large Language Models (LLMs). In this paper, we introduce Layerwise Learning Rate (LLR), an adaptive scheme that assigns distinct learning rates to individual Transformer layers. Our method is grounded in Heavy-Tailed Self-Regularization (HT-SR) theory, which characterizes the empirical spectral density (ESD) of weight correlation matrices to quantify heavy-tailedness. Layers with weaker heavy-tailedness are assigned larger learning rates to accelerate their training, while layers with stronger heavy-tailedness receive smaller learning rates. By tailoring learning rates in this manner, LLR promotes balanced training across layers, leading to faster convergence and improved generalization. Extensive experiments across architectures (from LLaMA to GPT-nano), optimizers (AdamW and Muon), and parameter scales (60M-1B) demonstrate that LLR achieves up to 1.5x training speedup and outperforms baselines, notably raising average zero-shot accuracy from 47.09% to 49.02%. A key advantage of LLR is its low tuning overhead: it transfers nearly optimal LR settings directly from the uniform baseline. Code is available at https://github.com/hed-ucas/Layer-wise-Learning-Rate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22263v1">Tailoring Teaching to Aptitude: Direction-Adaptive Self-Distillation for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      On-policy self-distillation (OPSD) is an emerging LLM post-training paradigm in which the model serves as its own teacher: conditioned on privileged information such as a reference trace or hint, the same policy provides dense token-level supervision on its own rollouts. However, recent studies show that OPSD degrades complex reasoning by suppressing predictive uncertainty, which supports exploration and hypothesis revision. Our token-level analysis shows that this failure arises from applying a uniform direction of teacher supervision across tokens with different uncertainty levels: conformity to the privileged self-teacher suppresses exploration at high entropy, while deviation from the teacher degrades step accuracy at low entropy. Accordingly, we propose \textbf{Direction-Adaptive Self-Distillation} (\textbf{DASD}), which reframes privileged self-distillation from uniform teacher imitation into entropy-routed directional supervision: high-entropy tokens are pushed away from the privileged teacher to preserve exploration, while low-entropy tokens are pulled toward the teacher to stabilize step-level execution. Across six mathematical reasoning benchmarks, DASD achieves the best macro Avg@16 over strong RLVR and self-distillation baselines. Pass@$k$, reasoning-health, and generalization analyses show that these average gains come from preserving exploration without sacrificing step-level execution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.00817v2">When LLMs Stop Following Steps: A Diagnostic Study of Procedural Execution in Language Models</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 77 pages, 109 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often achieve strong performance on reasoning benchmarks, but final-answer accuracy alone does not show whether they faithfully execute the procedure specified in a prompt. We study this question through a controlled diagnostic benchmark for procedural execution, where models are given a step-wise arithmetic algorithm and two numeric inputs, and must return the final computed value. The benchmark uses simple arithmetic operations but increases complexity through algorithm length and look-back dependencies over intermediate variables. Across 14 models and 55 datasets, average first-answer accuracy drops from 61% on 5-step procedures to 20% on 95-step procedures. Generation-level analysis shows that failures often involve missing answers, premature answers, self-correction after an initial error, under-executed traces, and hallucinated extra steps. These findings suggest that apparent reasoning ability can mask substantial weaknesses in faithful instruction execution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20251v2">ProcBench: Evaluating Process-Level Defects and Control Preservation in LLM Coding Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 22 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Existing benchmarks for LLM coding agents primarily evaluate final outcomes. While useful for measuring overall capability, these metrics provide limited visibility and often miss defects that arise during execution. We present ProcBench, a benchmark for execution-process evaluation in LLM coding agents. ProcBench organizes recurrent execution defects into a reusable ontology covering 11 defect types in 4 categories, and evaluates agent trajectories through standardized process evidence rather than final outcomes alone. To support comparison across heterogeneous agents, ProcBench standardizes raw logs into a unified trajectory representation and reports calibrated scorecards over process-level findings. In addition, ProcBench uses control preservation as a way to quantify execution-process quality, capturing whether execution remains interpretable, interruptible, correctable, reversible, and able to hand back authority when needed. We evaluate ProcBench on 200 cases sampled from three benchmarks: AndroidBench, TerminalBench, and SWE-bench-Verified. Results show that ProcBench can be instantiated with useful reliability, provides more stable semantics than direct thresholding, and reveals meaningful differences in execution quality that are often overlooked by conventional outcome-based evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.15774v2">MemEvoBench: Benchmarking Safety Risks from Memory Misevolution in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-21
    </div>
    <details class="paper-abstract">
      Equipping Large Language Models (LLMs) with persistent memory enhances interaction continuity and personalization but introduces new safety risks. Specifically, contaminated or biased memory accumulation can trigger abnormal agent behaviors. Existing evaluation methods have not yet established a standardized framework for measuring memory misevolution. This phenomenon refers to the gradual behavioral drift resulting from repeated exposure to misleading information. To address this gap, we introduce MemEvoBench, the first benchmark evaluating long-horizon memory safety in LLM agents against adversarial memory injection, noisy tool outputs, and biased feedback. The framework consists of QA-style tasks across 7 domains and 36 risk types, complemented by workflow-style tasks adapted from 20 Agent-SafetyBench environments with noisy tool returns. Both settings employ mixed benign and misleading memory pools within multi-round interactions to simulate memory evolution. Experiments on representative models reveal substantial safety degradation under biased memory updates. Our analysis suggests that memory evolution is a significant contributor to these failures. Furthermore, static prompt-based defenses prove insufficient, underscoring the urgency of securing memory evolution in LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22205v1">Skill Weaving: Efficient LLM Improvement via Modular Skillpacks</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 Accepted by ACL2026
    </div>
    <details class="paper-abstract">
      Large language models increasingly require specialization across diverse domains, yet existing approaches struggle to balance multi-domain capacities with strict memory and inference constraints. In this work, we introduce SkillWeave, a modular improvement framework that enables LLMs to specialize under fixed memory budgets. SkillWeave partitions full capabilities of a general-purpose model into skillpacks -- lightweight, domain-specific delta modules -- that reorganize and refine the model's internal knowledge. For efficient deployment, SkillWeave integrates SkillZip to compress skillpacks into compact and inference-ready format, enabling strong multi-domain performance with low-latency execution. On multi-task and agentic benchmarks, a 9B SkillWeave model outperforms several baselines and even surpasses a 32B monolithic LLM, while achieving up to 4x speedup.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22195v1">Reinforced Graph of Thoughts: RL-Driven Adaptive Prompting for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 26 pages (including appendix), 16 figures
    </div>
    <details class="paper-abstract">
      Graph of Thoughts (GoT), a generalized form of recent prompting paradigms for large language models (LLMs), has been shown to be useful for elaborate problem solving. By executing a graph of operations, thoughts of the LLM are structured as an arbitrary graph, forming the actual graph of thoughts. Originally, the graph of operations is defined manually, which requires in-depth knowledge about the solution of the problem to solve. Such a static graph of operations is rigid and therefore lacks adaptability. We propose Reinforced Graph of Thoughts (RGoT), an automated approach to the GoT prompting paradigm that leverages reinforcement learning (RL) to adaptively generate a graph of operations from a human-defined set. Results indicate that, under certain constraints, it is possible to construct graphs of operations adaptively to the task's complexity in an automated way.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22176v1">LLM-Metrics: Measuring Research Impact Through Large Language Model Memory</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 25pages, 5figures
    </div>
    <details class="paper-abstract">
      Citation counts remain the dominant metric for assessing research impact, yet they suffer from well-documented limitations: temporal lag, disciplinary bias, and Matthew effects. Here we propose LLM-Metrics, a research-impact assessment metric derived from the parametric memory of large language models (LLMs). The central hypothesis is that high-impact papers receive greater exposure in the academic community, that this exposure enters LLM training data in textual form, and that models consequently form stronger parametric memory of these papers. We designed four types of multiple-choice probes, covering title recognition, author recognition, method recognition, and venue recognition, and evaluated 549 computer science papers published in 2023-2024 across 17 LLMs spanning 0.5B to 72B parameters from six vendors. Of the 17 models, 15 produced positive predictions, 9 of which were significant at p less than 0.05, with an overall Spearman correlation of rho = 0.1495 and p = 0.0004 against citation counts. Three additional findings support the proposed mechanism. First, the predictive signal was stronger for 2024 papers, rho = 0.1880, whose citation counts were near zero at model-training time, reducing the plausibility of a simple reverse-causality explanation. Second, author-recognition probes showed the strongest discriminative power, consistent with an exposure-driven memory mechanism. Third, model scale and predictive power were non-monotonic: a 3B-parameter model, Llama-3.2-3B-Instruct, with rho = 0.1829, outperformed most larger models, supporting a selective-memory hypothesis in which the limited capacity of smaller models can serve as an effective information filter. LLM-Metrics offers a real-time, cross-disciplinary, citation-independent paradigm for research assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22175v1">SWE-Mutation: Can LLMs Generate Reliable Test Suites in Software Engineering?</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 24 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Evaluating software engineering capabilities has become a core component of modern large language models (LLMs); however, the key bottleneck hindering further scaling lies not in the scarcity of high-quality solutions, but in the lack of high-quality test suites. Test suites are indispensable both for synthesizing program repair trajectories and for providing precise feedback signals in reinforcement learning. Unfortunately, due to the high cost and difficulty of annotation, high-quality test suites have long been hard to obtain, while those automatically generated by LLMs tend to be superficial and lack sufficient discriminative power. As a first step toward constructing high-quality test suites, we introduce SWE-Mutation, a benchmark for evaluating LLM-generated test suites. The benchmark characterizes test suites by introducing systematically mutated solutions that attempt to ``fool'' the test suites and pass validation. We further propose an agentic, language-agnostic framework for automatically generating complex mutants. Our benchmark consists of 2,636 mutated variants derived from 800 original instances and includes a multilingual subset spanning nine programming languages. Experiments on seven LLMs reveal that even DeepSeek-V3.1 achieves only 10.20% verification and 36.15% detection rates, highlighting the inadequacy of current LLMs. Additionally, our agentic mutation strategy enhances realism, reducing average detection rates from 71.04% to 39.81% compared to conventional methods. These findings expose persistent deficiencies in the ability of current LLMs to generate reliable and discriminative test suites.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22166v1">Adapting the Interface, Not the Model: Runtime Harness Adaptation for Deterministic LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      LLM agents are shaped not only by their language models, but also by the runtime harness that mediates observation, tool use, action execution, feedback interpretation, and trajectory control. While existing agent adaptation methods mainly update model parameters, many failures in deterministic, rule-governed domains stem from mismatches at the model--environment interface. We propose Life-Harness, a lifecycle-aware runtime harness that improves frozen LLM agents without changing model weights or evaluation environments. Life-Harness evolves from training trajectories by converting recurring interaction failures into reusable interventions across environment contracts, procedural skills, action realization, and trajectory regulation, and remains fixed during held-out evaluation. On seven deterministic environments from $τ$-bench, $τ^2$-bench, and AgentBench, Life-Harness improves 116 out of 126 model--environment settings across 18 model backbones, with an average relative improvement of 88.5%. Harnesses evolved only from Qwen3-4B-Instruct trajectories transfer to 17 other models, showing that Life-Harness captures reusable environment-side structure rather than model-specific behavior. These results position runtime interface adaptation as a complementary alternative to model-centric agent training. Code is available at GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22156v1">One-Way Policy Optimization for Self-Evolving LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-21
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has become a promising paradigm for scaling reasoning capabilities of Large Language Models (LLMs). However, the sparsity of binary verifier rewards often leads to low efficiency and optimization instability. To stabilize training, existing methods typically impose token-level constraints relative to a reference policy. We identify that such constraints penalize deviations indiscriminately; this can flip verifier-determined direction when the policy attempts to outperform the reference, thereby suppressing gains. To resolve this, we propose One-Way Policy Optimization (OWPO), a method based on the principle of decoupling optimization direction from update magnitude. In OWPO, the verifier dictates the update direction, while the reference policy serves only to adjust the magnitude. Specifically, OWPO applies asymmetric reweighting: it performs Accelerated Alignment for inferior deviations (where the policy lags behind the reference) and Gain Locking for superior deviations (where the policy surpasses the reference). Furthermore, by incorporating iterative reference updates, OWPO creates a ``Ratchet Effect'' that continuously consolidates gains. Experimental results demonstrate that OWPO outperforms strong baselines, including DAPO, OPD, and MOPD, breaking the bottleneck of fixed priors to enable continuous self-evolution without reliance on external reference models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22154v1">IdleSpec: Exploiting Idle Time via Speculative Planning for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-21
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based agents solve complex tasks by leveraging multi-step reasoning with iterative tool calls and environment interactions, which incur idle time while waiting for observations. Despite the prevalence of idle time in most agentic scenarios, existing works treat it as an unavoidable overhead or propose restricted solutions that overlook varying computational budgets across different tool calls and future observation uncertainty, thereby leading to suboptimal utilization of idle time. In this paper, we introduce IdleSpec, a scalable and generic inference approach that leverages idle-time computation to improve agent performance while minimizing latency overhead. Specifically, IdleSpec iteratively generates plan candidates during idle periods and, once observations become available, aggregates them to guide the next reasoning step. For effective plan generation under observation uncertainty, IdleSpec samples between complementary drafting strategies (i.e., progressive and recovery) from a learned distribution that is updated via posterior feedback. Our experiments demonstrate that IdleSpec significantly improves agent performance in various agentic scenarios by effectively utilizing idle time. In particular, on the GAIA and FRAMES, IdleSpec achieves 55.6% average accuracy with Gemini-2.5-Flash, surpassing the vanilla baseline without idle-time usage by 5.1%. Furthermore, for MLE-Bench, which involves substantial delay from code executions, IdleSpec achieves performance gains of up to 9.1% on the Any Medal rate, highlighting its generalizability to long-horizon tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22148v1">Ratchet: A Minimal Hygiene Recipe for Self-Evolving LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 16 pages, 2 figures, 6 tables. Extends arXiv:2605.19576 with the SWE-bench Verified evaluation and a non-divergence analysis (Proposition 1)
    </div>
    <details class="paper-abstract">
      Self-evolving skill libraries, pioneered by Voyager, let frozen LLM agents accumulate reusable knowledge without weight updates, yet recent evaluation shows that LLM-authored skills deliver $+0.0$pp over no-skill baselines while human-curated ones deliver $+16.2$pp: the bottleneck is not skill authoring but lifecycle management. We introduce \textbf{Ratchet}, a single-agent loop in which a frozen LLM writes, retrieves, curates, and retires its own natural-language skills. Ratchet integrates four candidate hygiene mechanisms: outcome-driven retirement, a bounded active-cap, meta-skill authoring guidance, and pattern canonicalisation. On MBPP+ hard-100 with Claude Opus 4.7, Ratchet lifts held-out pass@1 from a $0.258 \pm 0.047$ baseline to a late-window rolling mean of $0.584$ (peak $0.658 \pm 0.042$) across 100 rounds and 3 seeds, a $+0.328 \pm 0.018$ rolling-mean gain where the no-skill control drifts at $+0.002 \pm 0.005$; the same recipe transfers to an agentic solver on SWE-bench Verified ($+0.22$ peak lift over 20 rounds). Eight ablations (A1--A8) reveal that the minimal working recipe is smaller than our design suggests: retirement and the meta-skill authoring prior are load-bearing, while explicit deduplication (canonicalisation, cover-guard) is subsumed by the meta-skill itself. A non-divergence proposition shows that bounded cap and retirement threshold together prevent expected performance from drifting below the no-skills floor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03070v4">ProOPF: Benchmarking and Improving LLMs for Professional-Grade Power Systems Optimization Modeling</a></div>
    <div class="paper-meta">
      📅 2026-05-21
    </div>
    <details class="paper-abstract">
      Growing renewable penetration introduces substantial uncertainty into power system operations, necessitating frequent adaptation of dispatch objectives and constraints and challenging expertise-intensive, near-real-time modeling workflows. Large Language Models (LLMs) provide a promising avenue for automating this process by translating natural-language (NL) operational requirements into executable optimization models via semantic reasoning and code synthesis. Yet existing LLM datasets and benchmarks for optimization modeling primarily target coarse-grained cross-domain generalization, offering limited, rigorous evaluation in power-system settings, particularly for Optimal Power Flow (OPF). We therefore introduce \textbf{ProOPF-D} and \textbf{ProOPF-B}, a dataset and benchmark for professional-grade OPF modeling: ProOPF-D contains 12K instances pairing NL requests with parameter adjustments and structural extensions to a canonical OPF, together with executable implementations; ProOPF-B provides 121 expert-annotated test cases with ground-truth code, enabling end-to-end evaluation under both concrete and abstract OPF modeling regimes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22106v1">ArborKV: Structure-Aware KV Cache Management for Scaling Tree-based LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-05-21
    </div>
    <details class="paper-abstract">
      Recent progress in LLM reasoning has increasingly shifted from single-pass generation to explicit search over intermediate reasoning states. Tree-of-Thoughts (ToT) organizes inference to tree-structured search with branching and backtracking, but it substantially amplifies the Key--Value (KV) cache: retaining KV states for a frontier of partial trajectories quickly becomes a memory bottleneck that limits throughput and constrains search depth and width under fixed hardware budgets. We address this challenge by observing that KV reuse in ToT-style inference is governed by search dynamics: near-term decoding depends primarily on the active branch and its ancestors, whereas inactive subtrees have low short-term reuse probability yet must remain recoverable for backtracking. Motivated by this, we propose ArborKV, a structure-aware eviction framework that couples a lightweight value estimator with a tree-aware allocation policy, and performs purely token-extractive eviction with lazy rehydration to support revisits. Experiments on ToT-style reasoning benchmarks show that ArborKV achieves up to ~4x peak KV-memory reduction while preserving near-full-retention accuracy, enabling larger search configurations under fixed device budgets that would otherwise run out of memory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.07340v2">Revisiting Robustness for LLM Safety Alignment via Selective Geometry Control</a></div>
    <div class="paper-meta">
      📅 2026-05-21
    </div>
    <details class="paper-abstract">
      Safety alignment of large language models remains brittle under domain shift and noisy preference supervision. Most existing robust alignment methods focus on uncertainty in alignment data, while overlooking optimization-induced fragility in preference-based objectives. In this work, we revisit robustness for LLM safety alignment from an optimization geometry perspective, and argue that robustness failures cannot be addressed by data-centric methods alone. We propose \textit{ShaPO}, a geometry-aware preference optimization framework that enforces worst-case alignment objectives via selective geometry control over alignment-critical parameter subspace. By avoiding uniform geometry constraints, ShaPO mitigates the over-regularization that can harm robustness under distribution shift. We instantiate ShaPO at two levels: token-level ShaPO stabilizes likelihood-based surrogate optimization, while reward-level ShaPO enforces reward-consistent optimization under noisy supervision. Across diverse safety benchmarks and noisy preference settings, ShaPO consistently improves safety robustness over popular preference optimization methods. Moreover, ShaPO composes cleanly with data-robust objectives, yielding additional gains and empirically supporting the proposed optimization-geometry perspective. The code is available at https://github.com/liujilong0116/ShaPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22095v1">Not Yet: Humans Outperform LLMs in a Colonel Blotto Tournament</a></div>
    <div class="paper-meta">
      📅 2026-05-21
    </div>
    <details class="paper-abstract">
      The emergence of large language models (LLMs) has spurred economists to study how humans and LLMs behave in strategic settings. We organized a series of round-robin tournaments in the Colonel Blotto game. This game attracts game theorists' attention due to high-dimensional action space and the absence of pure strategy Nash equilibria. In the first tournament, more than 200 human participants competed against one another. In the second tournament, several popular LLMs were invited to submit strategies. In the third tournament, we matched the number of LLM strategies to the number submitted by humans. We find that humans more often employ better-calibrated intermediate-level allocation heuristics and outperform the simpler, more stereotyped strategies submitted by LLMs. Strategic sophistication is key to success if and only if the necessary level of reasoning depth is reached, while lower and higher levels of reasoning offer no clear advantage over the primitive strategies. Among humans, field of study weakly predicts success: participants with STEM backgrounds perform better in the first tournament. Surprisingly, humans almost do not adjust their strategies across tournaments with different sets of opponents. This result suggests that humans base their choices primarily on the game's rules rather than on the identity of their opponents, treating LLMs much like human competitors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22091v1">Narrative Sharpens Gender Gaps: Surveying Film Characters with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-21
    </div>
    <details class="paper-abstract">
      Mainstream film is one of the richest sources of cultural content that AI systems learn from. Yet we have few tools for measuring the gender values it encodes. We present a proof-of-concept framework that turns fictional film characters into surveyable LLM agents. Using 160 U.S. films (1990--2019), we build 734 character agents from script dialogue and scene descriptions, condense their personas via expert-style reflections, and simulate World Values Survey gender-attitude responses. Agents reproduce systematic gender differences without explicit demographic prompting, suggesting attitudes emerge from behavior rather than identity labels. Benchmarked against historical survey data, agents exaggerate gender gaps and show greater decade-to-decade volatility than real populations. Narrative sharpens rather than homogenizes gender contrasts, complicating the consistent-input assumption underlying cultivation theory's mainstreaming mechanism. AI systems trained on such corpora may inherit this stylization before any model-level amplification occurs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22087v1">Automated Repair of TEE Partitioning Issues via DSL-Guided and LLM-Assisted Patching</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 Accepted by the ACM International Conference on the Foundations of Software Engineering (FSE 2026)
    </div>
    <details class="paper-abstract">
      Trusted Execution Environments (TEEs) provide hardware-based isolation to protect sensitive data and computations from potentially compromised operating systems (OS). However, TEE applications inevitably interact with the untrusted OS through SDK interfaces, and improper partitioning can introduce severe vulnerabilities such as data leakage and code injection. While prior work has proposed static analysis tools to detect such issues, automated repair remains largely unexplored. This problem is particularly challenging due to three TEE-specific factors: the lack of standardized secure development guidelines, the difficulty of extracting semantic information from low-level C code, and the absence of mature testing and validation methods. In this work, we present TEERepair, a framework for automatically repairing bad partitioning issues in TEE applications. Our approach tackles the above challenges by introducing a domain-specific language (DSL) to encode repair rules that express and capture common TEE security patterns, which are instantiated as patch templates with placeholders for context-specific variables. We then leverage large language models (LLMs) to reason about code semantics and synthesize context-aware patches, and further generate test clients to validate the repairs. We evaluate TEERepair on the TEE Partitioning Errors Benchmark (PartitioningE-Bench), achieving a significantly higher repair success rate of 87.6% compared to baselines. Furthermore, applying TEERepair to real-world TEE projects, we submitted 5 repair pull requests, 2 of which have been confirmed and merged by project maintainers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09472v2">WarmServe: Enabling One-for-Many GPU Prewarming for Multi-LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 Accepted at ICML 2026
    </div>
    <details class="paper-abstract">
      Deploying multiple models within shared GPU clusters is a key strategy to improve resource efficiency in large language model (LLM) serving. Existing multi-LLM serving systems improve GPU utilization at the cost of degraded inference performance, particularly time-to-first-token (TTFT). We attribute this degradation to the lack of awareness regarding future workload characteristics. In contrast, recent analyses have shown the strong periodicity and long-term predictability of real-world LLM serving workloads. In this paper, we propose one-for-many GPU prewarming, which proactively loads parameters from multiple models onto GPUs based on workload forecasts. These prewarmed weights enable the system to promptly instantiate serving instances upon encountering request bursts. We design and implement WarmServe, a multi-LLM serving system incorporating three key techniques: (1) a model placement algorithm that optimizes prewarming decisions to minimize cross-model prewarming interference, (2) a KV cache reservation strategy that repurposes idle KV cache space on running GPUs for prewarming new models, and (3) an efficient GPU memory switching mechanism for tensor management. Evaluation on real-world datasets shows that WarmServe reduces tail TTFT by up to 50.8$\times$ compared to the state-of-the-art autoscaling-based system, while supporting up to 2.5$\times$ higher request throughput than the GPU-sharing system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22074v1">From Reasoning Chains to Verifiable Subproblems: Curriculum Reinforcement Learning Enables Credit Assignment for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-05-21
    </div>
    <details class="paper-abstract">
      Reinforcement learning from verifiable rewards (RLVR) has shown strong promise for LLM reasoning, but outcome-based RLVR remains inefficient on hard problems because correct final-answer rollouts are rare and sample-level credit assignment cannot use partial progress in failed attempts. We introduce SCRL (Subproblem Curriculum Reinforcement Learning), a curriculum RL framework that derives verifiable subproblems from reference reasoning chains and fixes the final subproblem as the original problem. This turns partial progress on hard problems into verifiable learning signals. Algorithmically, SCRL uses subproblem-level normalization, which normalizes rewards independently at each subproblem position and assigns the resulting advantages to the corresponding answer spans, enabling finer-grained credit assignment without external rubrics or reward models. Our analysis shows that subproblem curricula lift hard problems out of gradient dead zones, with larger relative gains as the original problem becomes harder. Across seven mathematical reasoning benchmarks, SCRL outperforms strong curriculum-learning baselines, improving average accuracy over GRPO by +4.1 points on Qwen3-4B-Base and +1.9 points on Qwen3-14B-Base. On AIME24, AIME25, and IMO-Bench, SCRL further improves pass@1 by +3.7 points and pass@64 by +4.6 points on Qwen3-4B-Base, indicating better exploration on hard reasoning problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22058v1">Finding Missing Input Validation in TEEs via LLM-Assisted Symbolic Execution</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 Accepted by 2026 IEEE/ACM Third International Conference on AI Foundation Models and Software Engineering (FORGE '26)
    </div>
    <details class="paper-abstract">
      Trusted Execution Environments (TEEs) provide hardware-enforced isolation that protects sensitive code and data from untrusted software. Despite their strong security guarantees, analyzing TEE applications remains challenging due to the high cost and complexity of configuring complete TEE build and runtime environments, as well as the limited observability imposed by hardware isolation. This paper presents SymTEE, a novel large language model (LLM)-assisted symbolic execution framework for detecting missing input validation issues in TEE applications without requiring real TEE setups. SymTEE begins by leveraging Abstract Syntax Tree (AST) analysis to extract TEE code slices that may lack sufficient input validation, and then employs an LLM (GPT-5 in our case) to automatically convert the extracted slices into KLEE-compatible harness programs containing lightweight mock execution environments for symbolic analysis. Evaluations on 26 vulnerabilities (11 real-world and 15 synthetic) show that SymTEE achieves 100% precision and 92.3% recall in detecting missing input validation vulnerabilities while incurring an average analysis cost of only $0.05. These results demonstrate the effectiveness and practicality of SymTEE's pioneering paradigm of LLM-assisted symbolic execution, where LLMs autonomously generate mock environments to enable automated security analysis without complex setup, providing a more accessible and scalable framework for trusted computing systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22054v1">LABO: LLM-Accelerated Bayesian Optimization through Broad Exploration and Selective Experimentation</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 Accepted to ICML 2026
    </div>
    <details class="paper-abstract">
      The high cost and data scarcity in scientific exploration have motivated the use of large language models (LLMs) as knowledge-driven components in Bayesian optimization (BO). However, existing approaches typically embed LLMs directly into the sampling or surrogate modeling pipeline, without fully leveraging their significantly lower evaluation cost compared to real-world experiments. To address this limitation, we propose LLM-Accelerated Bayesian Optimization (LABO), a framework that combines LLM predictions with experimental observations within a single BO loop. LABO employs a gating criterion to dynamically balance the reliance on LLM predictions versus actual experiments. By leveraging inexpensive LLM evaluations to broadly explore the search space and reserving costly real experiments only for regions with high uncertainty, LABO achieves more sample-efficient optimization. We provide a theoretical analysis with a cumulative regret bound that formalizes this efficiency gain. Empirical results across diverse scientific tasks demonstrate that LABO consistently outperforms existing methods under identical experimental budgets. Our results suggest that LABO offers a practical and theoretically grounded approach for integrating LLMs into scientific discovery workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.01935v2">LiteCoOp: Lightweight Multi-LLM Shared-Tree Reasoning for Model-Serving Compiler Optimizations</a></div>
    <div class="paper-meta">
      📅 2026-05-21
    </div>
    <details class="paper-abstract">
      LLM-guided compiler optimization has recently shown promise, but existing approaches rely on a single large LLM throughout search, making them expensive and excluding smaller models. We pose the research question: whether heterogeneous LLMs can collaborate during compiler optimization while reducing compilation cost below optimization guided by a single large LLM. Crucially, this must be achieved without introducing overhead from agentic frameworks, which would run counter to the goal of lower compilation cost. To achieve these competing objectives, we introduce LiteCoOp, a lightweight framework that turns the optimization search tree itself into the mechanism for multi-LLM collaboration, enabling heterogeneous models to share progress without external agentic coordination. At each optimization step, LiteCoOp queries one LLM to propose both a compiler transformation and select the LLM to query at the next step. These LLM proposals are recorded in a shared MCTS tree, so all models are invoked serially and yet are informed by each other's decisions. The shared MCTS backpropagates the rewards, allowing progress made by one model to influence later decisions by others. This makes the MCTS tree the collaborative reasoning mechanism itself, avoiding inter-model communication, heavy reasoning traces, or agentic infrastructure. We instantiate this idea with an LLM-aware UCT that biases model selection toward smaller LLMs to reduce cost while still preserving the compiler performance objective. Across diverse GPU and (CPU) benchmarks, LiteCoOp consistently outperforms single-model baselines, with the best results obtained when scaling collaboration to eight heterogeneous LLMs. This eight-model config reduces total compilation time by 1.95x (1.74x), reduces API cost by 4.47x (4.32x), and invokes the largest model for only 23.1% (23.9%) of total calls while demonstrating collaboration scalability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.24191v3">When Grammar Guides the Attack: Uncovering Control-Plane Vulnerabilities in LLMs with Structured Output</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 To appear in CCS2026
    </div>
    <details class="paper-abstract">
      Content Warning: This paper may contain unsafe or harmful content generated by LLMs that may be offensive to readers. Large Language Models (LLMs) increasingly serve as tooling platforms through structured output APIs, but the grammar-guided decoding that powers this feature opens a critical control-plane attack surface orthogonal to traditional data-plane vulnerabilities. We introduce Constrained Decoding Attack (CDA), a new jailbreak class that targets the LLM control plane. CDA is best characterized as a control-to-semantic pipeline: (1) schema-enforced logit masking injects a malicious prefix into the generation trajectory, and (2) the model itself completes the harmful intent. Unlike data-plane jailbreaks that rely on bypassing alignment with visible inputs, CDA acts on the decoding process itself, so internal safety alignment alone cannot stop it. We instantiate CDA with EnumAttack, which hides malicious content in enum fields, and the more evasive DictAttack, which decouples the payload across a benign prompt and a dictionary-based grammar. Across 13 proprietary/open-weight models and five standard benchmarks, DictAttack achieves 94.3--99.5% Attack Success Rate (ASR) on flagship models including gpt-5, gemini-2.5-pro, deepseek-r1, and gpt-oss-120b. While basic grammar auditing mitigates EnumAttack, DictAttack still sustains 75.8% ASR against SOTA jailbreak guardrails, exposing a "semantic gap" that demands cross-plane defenses bridging the data and control planes. Project page and code are available at https://ict-cda.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.10067v3">Metis: Learning to Jailbreak LLMs via Self-Evolving Metacognitive Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 Accepted to the 43rd International Conference on Machine Learning (ICML 2026)
    </div>
    <details class="paper-abstract">
      Red teaming is critical for uncovering vulnerabilities in Large Language Models (LLMs). While automated methods have improved scalability, existing approaches often rely on static heuristics or stochastic search, rendering them brittle against advanced safety alignment. To address this, we introduce Metis, a framework that reformulates jailbreaking as inference-time policy optimization within an adversarial Partially Observable Markov Decision Process (POMDP). Metis employs a self-evolving metacognitive loop to perform causal diagnosis of a target's defense logic and leverages structured feedback as a semantic gradient to refine its policy, offering enhanced interpretability through transparent reasoning traces. Extensive evaluations across 10 diverse models demonstrate that Metis achieves the strongest average Attack Success Rate (ASR) among compared methods at 89.2%, maintaining high efficacy on resilient frontier models (e.g., 76.0% on O1 and 78.0% on GPT-5-chat) where traditional baselines exhibit substantial performance degradation. By replacing redundant exploration with directed optimization, Metis reduces token costs by an average of 8.2x and up to 11.4x. Our analysis reveals that current defenses remain vulnerable to internally-steered, closed-loop reasoning trajectories under the tested settings, highlighting a critical need for next-generation defenses capable of reasoning about safety dynamically during inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22005v1">Check Your LLM's Secret Dictionary! Five Lines of Code Reveal What Your LLM Learned (Including What It Shouldn't Have)</a></div>
    <div class="paper-meta">
      📅 2026-05-21
    </div>
    <details class="paper-abstract">
      We show that singular value decomposition of the lm_head} weight matrix of a transformer-based large language model -- requiring only five lines of PyTorch and no model inference -- reveals interpretable semantic subspaces directly from the model weights. Each left singular vector identifies the vocabulary tokens most readily selected when the hidden state aligns with the corresponding singular direction; inspecting these clusters exposes the model's training data composition and curation philosophy. Analysing GPT-OSS-120B, Gemma-2-2B, and Qwen2.5-1.5B, we find that singular value spectra and vocabulary cluster structures differ systematically across models: GPT exhibits a graduated hierarchy of functionally differentiated subspaces; Gemma is dominated by pre-nineteenth-century English orthography, forming a stepwise clustering structure that may contribute to high output controllability; and Qwen exhibits broad multilingual coverage alongside subspaces whose vocabulary the authors have determined to be ethically inappropriate for direct publication. Base-instruct comparison reveals that ethically concerning subspaces originate in pretraining and are not removed by post-training alignment. We introduce the Vocabulary Cluster Score (VCS) to quantify subspace coherence, and the Weighted Projection Score (WPS) as a static glitch token detector; applying WPS to GPT-OSS-120B recovers shokubutsu-hyakka-tsu (ID 137606), a well-known glitch token widely reported in the CJK language community, without any model inference. We propose a taxonomy of root causes for problematic vocabulary content and call for lm_head} SVD analysis to be adopted as a standard pre-release safety auditing step. Our findings further suggest directions toward SVD-guided tokenizer optimisation and more controllable LLM design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22001v1">Blind Spots in the Guard: How Domain-Camouflaged Injection Attacks Evade Detection in Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 8 pages, 3 figures, 2 tables. Submitted to EMNLP 2026 ARR cycle
    </div>
    <details class="paper-abstract">
      Injection detectors deployed to protect LLM agents are calibrated on static, template-based payloads that announce themselves as override directives. We identify a systematic blind spot: when payloads are generated to mimic the domain vocabulary and authority structures of the target document, what we call domain camouflaged injection, standard detectors fail to flag them, with detection rates dropping from 93.8% to 9.7% on Llama 3.1 8B and from 100% to 55.6% on Gemini 2.0 Flash. We formalize this as the Camouflage Detection Gap (CDG), the difference in injection detection rate between static and camouflaged payloads. Across 45 tasks spanning three domains and two model families, CDG is large and statistically significant (chi^2 = 38.03, p < 0.001 for Llama; chi^2 = 17.05, p < 0.001 for Gemini), with zero reverse discordant pairs in either case. We additionally evaluate Llama Guard 3, a production safety classifier, which detects zero camouflage payloads (IDRcamouflage = 0.000), confirming that the blind spot extends beyond few-shot detectors to dedicated safety classifiers. We further show that multi-agent debate architectures amplify static injection attacks by up to 9.9x on smaller models, while stronger models show collective resistance. Targeted detector augmentation provides only partial remediation (10.2% improvement on Llama, 78.7% on Gemini), suggesting the vulnerability is architectural rather than incidental for weaker models. Our framework, task bank, and payload generator are released publicly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21994v1">Ex-GraphRAG: Interpretable Evidence Routing for Graph-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-21
    </div>
    <details class="paper-abstract">
      GraphRAG conditions language models on subgraphs retrieved from knowledge graphs, encoded via message-passing GNNs. Because these encoders entangle node contributions through iterated neighborhood aggregation, there is no closed-form way to determine how much each retrieved entity influenced the encoder's output, and therefore no way to faithfully audit what structural evidence actually reached the model. We introduce Ex-GraphRAG, which replaces the GNN encoder with a Multivariate Graph Neural Additive Network (M-GNAN), an extension of additive graph models to high-dimensional embedding spaces that yields an exact decomposition of the encoder's output across individual nodes and feature groups, without post-hoc approximation. On STaRK-Prime, this auditable encoder matches black-box performance. Using it to audit evidence routing, we uncover a semantic-structural mismatch: the nodes that dominate the encoder's output are structurally disconnected in the retrieved subgraph, held together by low-attribution intermediaries whose removal degrades multi-hop QA by up to 28%. This mismatch, invisible to any opaque encoder, reveals that semantic importance and structural connectivity are governed by disjoint sets of nodes, with direct implications for retrieval pruning, context construction, and failure diagnosis in graph-augmented LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21988v1">Learning Spatiotemporal Sensitivity in Video LLMs via Counterfactual Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 Project website: https://ddz16.github.io/crpo.github.io/
    </div>
    <details class="paper-abstract">
      Video large language models (Video LLMs) achieve strong benchmark accuracy, yet often answer video questions through shortcuts such as single-frame cues and language priors rather than by tracking spatiotemporal dynamics. This issue is exacerbated in RL post-training, where correctness-only rewards can further reinforce shortcut policies that obtain high reward without tracking video dynamics. We address this by asking a controlled counterfactual question: if the visual world changed while the question remained fixed, should the answer change or stay the same? Based on this view, we propose \textbf{Counterfactual Relational Policy Optimization (CRPO)}, a dual-branch RL framework for improving \emph{spatiotemporal sensitivity}. CRPO constructs counterfactual videos through horizontal flips and temporal reversals, trains on both original and counterfactual branches, and introduces a \textbf{Counterfactual Relation Reward (CRR)} between their answers. CRR encourages answers to change for dynamic questions and remain unchanged for static questions. This cross-branch constraint makes it difficult for shortcut policies to be consistently rewarded across both branches. To evaluate this property, we introduce \textbf{DyBench}, a paired counterfactual video benchmark with 3,014 videos covering reversible dynamics, moving direction, and event sequence, together with a strict pair-accuracy metric that prevents fixed-answer shortcuts from inflating scores. Experiments show that CRPO outperforms prior RL methods on spatiotemporal-sensitive evaluations while maintaining competitive general video performance. On Qwen3-VL-8B, CRPO improves DyBench P-Acc by +7.7 and TimeBlind I-Acc by +8.2 over the base model, indicating improved spatiotemporal sensitivity rather than stronger reliance on static shortcuts. The project website can be found at https://ddz16.github.io/crpo.github.io/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21975v1">Reasoning through Verifiable Forecast Actions: Consistency-Grounded RL for Financial LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-21
    </div>
    <details class="paper-abstract">
      Financial markets are characterized by extreme non-stationarity, low signal-to-noise ratios, and strong dependence on external information such as news, company fundamentals, and macroeconomic signals. Yet, existing approaches either abstract time-series into text or decouple forecasting from language-based reasoning, leading to a fundamental mismatch between qualitative reasoning and quantitative outcomes. To address this, we introduce StockR1, a time-series-enhanced LLM that unifies stock forecasting and financial reasoning through a verifiable forecast action. Based on a tool-call design, the model first emits a forecast action, which is a structured and interpretable representation of its qualitative market outlook. It then invokes a time-series decoder conditioned on this action to generate distributional future trajectories, leading to more informed question answering and financial reasoning. We optimize the full pipeline with reinforcement learning, where rewards jointly reflect answer validity, forecast accuracy, and consistency between generated actions and observed time-series dynamics. In addition, rewards are reweighted by a sample-level uncertainty scalar, encouraging the model to accommodate varying uncertainty in market dynamics. We evaluate StockR1 on financial question answering and stock forecasting over a large-scale 10-year benchmark. Our method consistently outperforms time-series baselines and general-purpose LLMs, improving reasoning accuracy by 17.7% (4B) and 25.9% (8B). These findings demonstrate that structuring the forecast actions establishes a powerful synergy between language reasoning and temporal prediction, enabling LLMs to reason through verifiable, interpretable, and numerically grounded decisions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21969v1">LLM Retrieval for Stable and Predictable Ad Recommendations</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 SIGIR 2026 AgentSearch Workshop, Melbourne Australia
    </div>
    <details class="paper-abstract">
      Traditional ads recommendation systems have primarily focused on optimizing for prediction accuracy of click or conversion events using canonical metrics such as recall or normalized discounted cumulative gain (NDCG). With the hyper-growth of ads inventory and liquidity with generative AI technologies, the prediction stability and predictability is becoming increasingly critical. Intuitively, prediction stability and predictability can be defined to quantify system robustness with respect to minor/noisy input (ads, creatives) perturbations, the lack of which could lead to advertiser perceivable problems such as repeatability, cold start and under-exploration. In this paper, we introduce a new evaluation framework for quantifying stability and predictability of an ads recommender system, and present an online validated semantic candidate generation framework powered by fine-tuned Large Language Models (LLMs) that showed significant improvement along these metrics by fundamentally improving the semantic-awareness of the system. The approach extracts hierarchical semantic attributes from ad creatives to obtain LLM representations, which serve as the foundation for graph-based expansion, ensuring the retrieved candidates encapsulate semantic variants of an ad, guaranteeing that small creative variants from the advertiser yield consistent and explainable delivery results to the user. We tested this LLM ads retrieval framework in a large-scale industrial ads recommendation system, demonstrating significant improvements across offline and online A/B experiments, showcasing gains in both predictability and traditional performance metrics. Although evaluated in the ads stack, this is a general framework that can be applied broadly to any large-scale recommendation and retrieval systems facing similar scaling and predictability challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21958v1">Diagnosis Is Not Prescription: Linguistic Co-Adaptation Explains Patching Hazards in LLM Pipelines</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 Preprint. Under review at EMNLP 2026 (ARR)
    </div>
    <details class="paper-abstract">
      When a multi-module LLM agent fails, the module most responsible for the failure is not necessarily the best place to intervene. We demonstrate this Diagnostic Paradox empirically: causal analysis consistently identifies the routing module -- which selects which tool to call next -- as the primary bottleneck across three independent agent families. Yet injecting prompt-level correction examples into this module consistently degrades performance, sometimes severely. Patching an upstream query-rewriting module instead reliably improves outcomes. The effect holds with statistical significance on two agent families and directional consistency on a third; alternative repair strategies at the routing module (instruction rewriting, model upgrade) are neutral, confirming that the harm is specific to correction-injection patching. We explain this asymmetry through the Linguistic Contract hypothesis: each downstream module implicitly adapts to its upstream's characteristic error distribution, so correcting the bottleneck breaks this implicit alignment in a way that upstream corrections do not. We operationalize this via a per-agent co-adaptation measure, derived from diagnosis alone, and show it is consistently associated with patching harm across agent families: higher co-adaptation co-occurs with harm, lower with safety. This trend holds across all three agent families, providing preliminary support for the hypothesis beyond a single-agent observation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05106v4">Token-Level LLM Collaboration via FusionRoute</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 25 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit strengths across diverse domains. However, achieving strong performance across these domains with a single general-purpose model typically requires scaling to sizes that are prohibitively expensive to train and deploy. On the other hand, while smaller domain-specialized models are much more efficient, they struggle to generalize beyond their training distributions. To address this dilemma, we propose FusionRoute, a robust and effective token-level multi-LLM collaboration framework in which a lightweight router simultaneously (i) selects the most suitable expert at each decoding step and (ii) contributes a complementary logit that refines or corrects the selected expert's next-token distribution via logit addition. Unlike existing token-level collaboration methods that rely solely on fixed expert outputs, we provide a theoretical analysis showing that pure expert-only routing is fundamentally limited: unless strong global coverage assumptions hold, it cannot in general realize the optimal decoding policy. By augmenting expert selection with a trainable complementary generator, FusionRoute expands the effective policy class and enables recovery of optimal value functions under mild conditions. Empirically, across both Llama-3 and Gemma-2 families and diverse benchmarks spanning mathematical reasoning, code generation, and instruction following, FusionRoute outperforms both sequence- and token-level collaboration, model merging, and direct fine-tuning, while remaining competitive with domain experts on their respective tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.19310v3">MemReward: Graph-Based Experience Memory for LLM Reward Prediction with Limited Labels</a></div>
    <div class="paper-meta">
      📅 2026-05-21
    </div>
    <details class="paper-abstract">
      Reinforcement learning has emerged as a powerful paradigm for improving large language model (LLM) reasoning, where rollouts are sampled from the policy and reward signals computed on those rollouts are used to update the policy. However, in data-scarce scenarios, obtaining ground-truth labels to verify rollouts at scale often requires expensive human annotation or labor-intensive expert verification. For instance, evaluating mathematical proofs demands expert review, and open-ended question answering lacks definitive ground truth. When ground-truth labels are scarce, the effectiveness of reinforcement learning fine-tuning is constrained. Inspired by the success of semi-supervised learning in propagating labels from labeled to unlabeled samples, we propose MemReward, a graph-based experience memory framework that integrates reward propagation directly into online policy optimization. MemReward stores rollouts (thinking processes and final answers) from an initial LLM policy as nodes in a heterogeneous graph connected by similarity and structural edges, over which a GNN propagates rewards from labeled to unlabeled rollouts. To train such a framework, we first warm up the GNN on labeled rollouts to predict rewards via heterogeneous aggregation over query, thinking, and answer nodes. During online RL fine-tuning, unlabeled rollouts are attached to the graph by query similarity, and the GNN predicts their rewards, yielding a hybrid reward acquisition strategy that combines ground-truth and GNN-predicted rewards. Experiments on Qwen2.5-1.5B and 3B in mathematics, question answering, and code generation demonstrate that MemReward, with ground-truth rewards on only 20% of rollouts, achieves 96.6% of Oracle performance on 1.5B and 97.3% on 3B, and closely approaches Oracle on out-of-domain tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21902v1">Planning in the LLM Era: Building for Reliability and Efficiency</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 Published at ICAPS 2026
    </div>
    <details class="paper-abstract">
      Growing attention to intelligent agents has put a spotlight on one of their central capabilities: planning. Early attempts to leverage large language models (LLMs) for planning relied on single-shot plan generation, followed by hybrid approaches that coupled LLMs with limited external search. These methods, unsound and incomplete by their very nature, often require substantial resources without yielding better solutions on unseen problems. As the limitations of LLMs become clearer, recent work has shifted toward using them at solution construction time -- generating symbolic solvers for a family of problems that can be verified and then used efficiently at inference time. This trend reflects the growing need for agents that are both reliable and resource-efficient. It also offers a path towards generating maintainable planners with minimal dependence on language models at inference time. In this paper, we argue that this shift reflects a broader realignment of the planning field in the LLM era. We examine three major categories of planner-generation methods, discuss their current limitations, and outline research steps towards a more reliable and efficient LLM-based generation of planners.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06428v2">Walking the Tightrope of LLMs for Software Development: A Practitioners' Perspective</a></div>
    <div class="paper-meta">
      📅 2026-05-21
    </div>
    <details class="paper-abstract">
      Background: Large Language Models emerged with the potential of provoking a revolution in software development (e.g., automating processes, workforce transformation). Although studies have started to investigate the perceived impact of LLMs for software development, there is a need for empirical studies to comprehend how to balance forward and backward effects of using LLMs. Objective: We investigated how LLMs impact software development and how to manage the impact from a software developer's perspective. Method: We conducted 22 interviews with software practitioners across 3 rounds of data collection and analysis, between October (2024) and September (2025). We employed Socio-Technical Grounded Theory for Data Analysis (STGT4DA) to rigorously analyse interview participants' responses. Results: We identified the benefits (e.g., maintain developer flow, improve developer mental models, and foster entrepreneurship) and challenges (e.g., damage to developers' reputation) of using LLMs at individual, team, organisation, and society levels; as well as actionable guidances into how mitigate these challenges. Conclusion: Critically, we present the trade-offs that software practitioners, teams, and organisations face in working with LLMs. Our findings are particularly useful for software team leaders and IT managers to assess the viability of LLMs within their specific context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12744v4">Resting Neurons, Active Insights: Robustifying Activation Sparsity in LLMs via Spontaneity</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 ICML 2026
    </div>
    <details class="paper-abstract">
      Activation sparsity offers a compelling route to accelerate large language model (LLM) inference by selectively suppressing hidden activations, yet existing approaches exhibit severe accuracy degradation at high sparsity. We show that this failure stems from representational instability: *activation sparsity disrupts input-dependent activation learned during pretraining, inducing distribution shifts in hidden states.* We address this issue by reframing activation sparsity as a representational alignment problem and introducing **Spontaneous Neurons (SPON)**, a lightweight mechanism inspired by spontaneous neural activity in biological systems. SPON injects a small set of learnable, input-independent activation vectors that act as persistent representational anchors for sparse computation. These vectors are trained via distribution matching to the dense model and can be absorbed into bias terms after training, incurring negligible inference overhead. Across multiple LLM backbones, SPON consistently restores performance, stabilizes latent representations, and preserves generalization. Our results establish SPON as an effective and principled solution for reliable activation-sparse inference, and offer new insights into knowledge retention in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21856v1">The Illusion of Reasoning: Exposing Evasive Data Contamination in LLMs via Zero-CoT Truncation</a></div>
    <div class="paper-meta">
      📅 2026-05-21
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive reasoning abilities across a wide range of tasks, but data contamination undermines the objective evaluation of these capabilities. This problem is further exacerbated by malicious model publishers who use evasive, or indirect, contamination strategies, such as paraphrasing benchmark data to evade existing detection methods and artificially boost leaderboard performance. Current approaches struggle to reliably detect such stealthy contamination. In this work, we uncover a critical phenomenon: a model's generated reasoning steps actively mask its underlying memorization. Inspired by this, we propose the Zero-CoT Probe (ZCP), a novel black-box detection method that deliberately truncates the entire Chain-of-Thought (CoT) process to expose latent shortcut mappings. To further isolate memorization from the model's intrinsic problem-solving capabilities, ZCP compares the model's zero-CoT performance on the original benchmark against an isomorphically perturbed reference dataset. Furthermore, we introduce Contamination Confidence, a metric that quantifies both the likelihood and severity of contamination, moving beyond simple binary classifications. Extensive experiments on both previously identified contaminated models and specially fine-tuned contaminated models demonstrate that ZCP robustly detects both direct and evasive data contamination. The code for ZCP is accessible at https://github.com/Yifan-Lan/zero-cot-probe.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21851v1">OPPO: Bayesian Value Recursion for Token-Level Credit Assignment in LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-05-21
    </div>
    <details class="paper-abstract">
      Reinforcement learning with verifiable rewards has become the standard recipe for improving LLM reasoning, but the dominant algorithm GRPO assigns a single trajectory-level advantage to every token, diluting the signal at pivotal reasoning steps and injecting noise at uninformative ones. Critic-free alternatives derived from on-policy distillation supply per-token signals through oracle-conditioned likelihood ratios, yet apply each signal in isolation from the trajectory-level evidence accumulated up to that position. We propose Oracle-Prompted Policy Optimization (OPPO), which rests on a single observation: the oracle signal used by prior distillation-style methods for local discrimination is also the natural Bayesian update of the model's belief about eventual success. Accumulating the signal along a trajectory yields, in closed form and at the cost of one extra forward pass, a running estimate of the success probability at every position, together with a token-level advantage that requires no learned value network and no additional rollouts. A first-order analysis factorizes the advantage into the per-token discrimination signal used by distillation methods modulated by a state weight that concentrates credit on genuinely pivotal tokens, with a directional variance-reduction guarantee. The framework admits two estimators differing only in which model scores the evidence: a \textit{self-oracle} that reuses the student and recovers the on-policy distillation reward as a strict special case, and a \textit{teacher-oracle} that delegates scoring to a stronger frozen model. On two base LLMs across seven mathematics, science, and code reasoning benchmarks, OPPO improves over GRPO, DAPO, and SDPO by up to $+6.0$ points on AMC'23 and $+5.2$ points on AIME'24, with gains that widen monotonically with response length.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21845v1">Comparing LLM and Fine-Tuned Model Performance on NVDRS Circumstance Extraction with Varying Prompt Complexity</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 Accepted at IEEE ICHI 2026
    </div>
    <details class="paper-abstract">
      Suicide is a leading cause of death in the United States, and understanding the circumstances that precede it requires extracting structured information from death investigation narratives. Many of these circumstances require semantic inference beyond simple keyword matching. We develop a ``Complexity Score'' algorithm that analyzes coding manual structure to predict when detailed prompts with full coding guidelines improve over name-only prompts. We then construct a hybrid approach that selects prompt strategy per circumstance. We evaluate large language models (LLMs) against fine-tuned RoBERTa on 25 inferentially complex circumstances from the National Violent Death Reporting System (NVDRS). We found that LLMs substantially outperform on low-prevalence circumstances where training data is insufficient. We further demonstrate that our framework generalizes across frontier LLMs, with GPT-5.2, Gemini 2.5 Pro and Llama-3 70B showing consistent performance patterns. These findings support a hybrid architecture where LLMs handle rare, inferentially complex circumstances while fine-tuned models handle common ones.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.16659v3">Memory-Efficient LLM Pretraining via Minimalist Optimizer Design</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 Accepted at ICML 2026
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) relies on adaptive optimizers such as Adam, which introduce extra operations and require significantly more memory to maintain first- and second-order moments than SGD. While recent works such as GaLore, Fira and APOLLO have proposed state-compressed memory-efficient variants, a fundamental question remains: What are the minimum modifications to plain SGD needed to match state-of-the-art pretraining performance? We systematically investigate this question using a bottom-up approach, and identify two simple yet highly (memory- and compute-) efficient techniques: (1) column-wise gradient normalization (normalizing the gradient along the output dimension), that boosts SGD performance without momentum; and (2) applying first-order momentum only to the output layer, where gradient variance is highest. Combining these two techniques lead to SCALE (Stochastic Column-normAlized Last-layer momEntum), a simple optimizer for memory efficient pretraining. Across multiple models (60M-1B), SCALE matches or exceeds the performance of Adam while using only 35-45% of the total memory. It also consistently outperforms memory-efficient optimizers such as GaLore, Fira and APOLLO, making it a strong candidate for large-scale pretraining under memory constraints. For LLaMA 7B, SCALE outperforms the state-of-the-art memory-efficient methods APOLLO and Muon in both perplexity and memory consumption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.17596v2">NeuSymMS: A Hybrid Neuro-Symbolic Memory System for Persistent, Self-Curating LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 7 pages
    </div>
    <details class="paper-abstract">
      We present NeuSymMS, an adaptive memory system that enables large language model (LLM) agents to learn, remember, and reason about users across sessions via a hybrid neuro-symbolic architecture. NeuSymMS couples neural fact extraction from unstructured dialogue using LLMs and a CLIPS-based expert system that classifies, deduplicates, and reconciles facts under explicit lifecycle rules. The system represents knowledge as subject-relation-value triples stored in relational database management system. It supports user/agents/agent-to-agent scoping, and implements a dual-horizon (short-term and long-term) memory model. IT leverages access-based promotion and time-based pruning of the memory on both horizpons. NeuSymMS maintains continuity of memory while avoiding context-window bloat and cross-entity contamination. We argue that this architecture offers a practical path to trustworthy, auditable memory for production agentic systems and discuss its novelty relative to log retrieval, summarization, and key-value approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21812v1">Bridging the Cold-Start Gap: LLM-Powered Synthetic Data Generation for Natural Language Search at Airbnb</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Deploying natural language search systems presents a critical cold-start challenge: no real user queries to learn linguistic patterns, and no relevance labels to train ranking models. We present a framework for generating synthetic queries and labels using large language models (LLMs), powering model training and evaluation for Airbnb's natural language search. For query generation, we combine contrastive listing pairs from booking sessions with seed queries from user research to balance realism and diversity, enabling a cold-to-warm start transition as real user data becomes available. For label generation, we introduce contrastive generation that produces topicality labels by construction, and Virtual Judge (VJ) labeling for broader coverage. We compare our approach against a no-seed contrastive baseline and an InPars-style baseline. For query length, the InPars baseline produces verbose queries with KL divergence of 12.03 vs. real users; our seed-guided approach achieves 0.66, a 7.5x improvement. For attribute type distributions, our approach achieves the lowest KL divergence (0.04), outperforming even seed queries (0.09). Experiments show our approach produces harder evaluation examples than the no-seed baseline (79% vs. 97% pairwise accuracy), providing discriminative signal for model improvement. We deploy production pipelines generating synthetic examples daily for embedding-based retrieval and ranking evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21779v1">FuzzingBrain V2: A Multi-Agent LLM System for Automated Vulnerability Discovery and Reproduction</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Software vulnerabilities pose critical security threats, with nearly 50,000 CVEs reported in 2025. While Large Language Models (LLMs) show promise for automated vulnerability detection, three key challenges remain. First, LLM-generated vulnerability reports suffer from high false positive rates and lack reproducible verification. Second, existing LLM-based approaches use suboptimal granularities for vulnerability localization: function-level analysis overlooks bugs when context becomes extensive, while line-level analysis lacks sufficient context. Third, existing approaches have difficulty reasoning about vulnerabilities with complex cross-function dependencies and triggering conditions. We present FuzzingBrain V2, a multi-agent system that addresses these gaps through four key contributions: (1) fully automated vulnerability analysis built on Google's OSS-Fuzz, ensuring all reported vulnerabilities are fuzzer-reproducible; (2) Suspicious Point, a novel control-flow-based abstraction for precise vulnerability localization at the optimal granularity; (3) logic-driven hierarchical function analysis with dual-layer fuzzing enhancing function coverage under resource constraints; (4) MCP-based static and dynamic analysis tools with context engineering enhancing complex vulnerability reasoning. On the AIxCC 2025 Final Competition C/C++ dataset, FuzzingBrain V2 achieved 90% detection rate (36 of 40 vulnerabilities). In real-world deployment, FuzzingBrain V2 discovered 29 zero-day vulnerabilities across 12 open-source projects, all confirmed and fixed by maintainers, with 2 assigned CVE IDs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21768v1">Memory-R2: Fair Credit Assignment for Long-Horizon Memory-Augmented LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Memory-augmented LLM agents enable interactions that extend beyond finite context windows by storing, updating, and reusing information across sessions. However, training such agents with reinforcement learning in multi-session environments is challenging because memory turns the agent's past actions into part of its future environment. Once different rollouts write, update, or delete different memories, they no longer share the same intermediate memory state, making trajectory-level comparisons fundamentally unfair. This violates a key assumption behind group-relative methods such as GRPO, where rollouts are compared as if they were sampled from the same effective environment. Consequently, trajectory-level rewards provide noisy or biased credit signals for long-horizon memory operations. To address this challenge, we introduce Memory-R2, a training framework for long-horizon memory-augmented LLM agents. Its core algorithm, LoGo-GRPO, combines local and global group-relative optimization. The global objective preserves end-to-end learning from long-horizon trajectory-level rewards, while local rerollouts compare different memory-operation outcomes from the same intermediate memory state, yielding fairer group comparisons and more precise supervision for memory construction. Beyond credit assignment, Memory-R2 jointly optimizes memory formation and memory evolution with a shared-parameter co-learning design, where a fact extractor and a memory manager are instantiated from the same LLM backbone through role-specific prompts. To stabilize multi-step RL over long memory horizons, we adopt a progressive curriculum that increases the training horizon from 8 to 16 to 32 sessions. Together, these components provide an effective training paradigm for memory-augmented LLM agents in long-horizon multi-session settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21748v1">RankJudge: A Multi-Turn LLM-as-a-Judge Synthetic Benchmark Generator</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      As interactive LLM-based applications are created and refined, model developers need to evaluate the quality of generated text along many possible axes. For simpler systems, human evaluation may be practical, but in complicated systems like conversational chatbots, the amount of generated text can overwhelm human annotation resources. Model developers have begun to rely heavily on auto-evaluation, where LLMs are also used to judge generation quality. However, existing LLM-as-a-judge benchmarks largely focus on simple Q\&A tasks that do not match the complexity of multi-turn conversations. We introduce RankJudge, a benchmark generator for evaluating LLM-as-a-judge on multi-turn conversations grounded in reference documents. RankJudge creates pairs of conversations where one conversation has a single flaw injected into one turn. This construction allows paired conversations to be labeled unambiguously as better or worse, and precisely isolates failure categories to individual turns, enabling a strict joint correctness criterion for judging. We implement RankJudge across the domains of machine learning, biomedicine, and finance, evaluate 21 frontier LLM judges, and rank those judges via the Bradley-Terry model. Our formulation also allows ranking each conversation pair with difficulty ratings, which we use to dynamically curate the evaluation slice to reduce label noise, as confirmed via human annotation. We find that judge rankings are stable under partial observability, coarser correctness criteria, and an alternative random-walk rating algorithm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21740v1">SMDD-Bench: Can LLMs Solve Real-World Small Molecule Drug Design Tasks?</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      LLM agents have incredible potential for scientific discovery applications. However, the performance of LLM agents on real-world, small molecule drug design (SMDD) tasks across diverse chemistries and targets is unclear. Current evaluation methods are either ad hoc, too simple for real-world discovery, limited in scale, or restricted to single-turn question answering. In effort to standardize the evaluation of LLM agents on small molecule design, we introduce SMDD-Bench, a challenging, multi-turn, long-horizon agentic benchmark consisting of 502 guaranteed-solvable task instances spanning 5 task types: 2D Pharmacophore Identification, Interaction Point Discovery, Scaffold Hopping, Lead Optimization, and Fragment Assembly. SMDD-Bench tasks span a wide region of chemical space and involve 102 unique protein targets. Completely solving the benchmark would require having strong chemical and biological reasoning and 3D intuition, understanding specialized tool use, and displaying planning expertise over a limited number of oracle calls. We benchmark 7 frontier open and closed source LLMs and find even the most performant LLM, GPT5.4, solves only 40.2\% of tasks. We hope SMDD-Bench provides a standardized testbed to invigorate the field towards training and evaluating LLM agents for fully autonomous computational drug design. We host a public leaderboard at smddbench.com .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.15588v2">Calibrating LLMs with Semantic-level Reward</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are deployed in consequential settings such as medical question answering and legal reasoning, the ability to estimate when their outputs are likely to be correct is essential for safe and reliable use, requiring well-calibrated uncertainty. Standard reinforcement learning with verifiable rewards (RLVR) trains models with a binary correctness reward that is indifferent to confidence, providing no penalty for confident but wrong predictions and thereby degrading calibration. Recent work addresses this by training models to produce verbalized confidence scores alongside answers and rewarding agreement with correctness. However, verbalized confidence is calibrated at the token level and thus exhibits inconsistency across textual variations with same semantic meaning. We propose \textbf{Calibration with Semantic Reward (CSR)}, a framework that calibrates language models directly in semantic space without a verbalized confidence interface. CSR combines the correctness reward with a novel semantic calibration reward that encourages exploitation among correct rollouts by promoting semantic agreement, and exploration among incorrect ones by discouraging spurious consistency. Experiments across three model families on HotpotQA (in-distribution) and TriviaQA, MSMARCO, and NQ-Open (out-of-distribution) show that CSR consistently achieves lower ECE and higher AUROC than verbalized-confidence baselines across nearly all settings, reducing ECE by up to $40\%$ and improving AUROC by up to $31\%$ over verbalized-confidence baselines, with calibration behavior generalizing robustly across all four evaluation settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21739v1">AttuneBench: A Conversation-Based Benchmark for LLM Emotional Intelligence</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Emotional intelligence (EI), the ability to perceive, understand, and respond appropriately to others' emotional states, is central to human communication, and increasingly important to assess as LLMs assume conversational roles in everyday life. Existing EI benchmarks rely on synthetic prompts, single-turn cases, or third-party annotation. These approaches do not directly measure how models infer and respond to a participant's emotional state over the course of a real conversation. We introduce AttuneBench, a benchmark grounded in 200 genuine multi-turn human-model conversations in which participants conversed with anonymized LLMs and provided turn-by-turn annotations of their emotional state, the model's behavior, and their preferred responses. Across 11 evaluated models, we find that model rankings on emotion recognition, behavioral classification, preference prediction, and judged response quality are largely independent, indicating that emotionally intelligent behavior decomposes into separable capabilities. Preference alignment and response-quality judgments are substantially more model-discriminating than emotion-label accuracy. These results indicate that emotionally intelligent behavior requires predicting what kind of response a specific user wants in context, a distinction that aggregate scoring can obscure and that single-turn or synthetic formats cannot directly capture across turns. AttuneBench provides a framework for assessing each of these capabilities and for diagnosing model-specific strengths and failure modes in emotionally salient conversation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.09252v2">LLM Agents Already Know When to Call Tools -- Even Without Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Tool-augmented LLM agents tend to call tools indiscriminately, even when the model can answer directly. Each unnecessary call wastes API fees and latency, yet no existing benchmark systematically studies when a tool call is actually needed. We propose When2Tool, a benchmark of 18 environments (15 single-hop, 3 multi-hop) spanning three categories of tool necessity -- computational scale, knowledge boundaries, and execution reliability -- each with controlled difficulty levels that create a clear decision boundary between tool-necessary and tool-unnecessary tasks. We evaluate two families of training-free baselines: Prompt-only (varying the prompt to discourage unnecessary calls) and Reason-then-Act (requiring the model to reason about tool necessity before acting). Both provide limited control: Prompt-only suppresses necessary calls alongside unnecessary ones, and Reason-then-Act still incurs a disproportionate accuracy cost on hard tasks. To understand why these baselines fail, we probe the models' hidden states and find that tool necessity is linearly decodable from the pre-generation representation with AUROC 0.89--0.96 across six models, substantially exceeding the model's own verbalized reasoning. This reveals that models already know when tools are needed, but fail to act on this knowledge during generation. Building on this finding, we propose Probe&Prefill, which uses a lightweight linear probe to read the hidden-state signal and prefills the model's response with a steering sentence. Across all models tested, Probe&Prefill reduces tool calls by 48% with only 1.7% accuracy loss, while the best baseline at comparable accuracy only reduces 6% of tool calls, or achieves a similar tool call reduction but incurs a 5$\times$ higher accuracy loss. Our code is available at https://github.com/Trustworthy-ML-Lab/when2tool
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11679v3">LLMs can construct powerful representations and streamline sample-efficient supervised learning</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      As real-world datasets become more complex and heterogeneous, supervised learning is often bottlenecked by input representation design. Modeling multimodal data, such as time-series, free text, and structured records, often requires non-trivial domain expertise. We propose an agentic pipeline to streamline this process. First, an LLM analyzes a small but diverse subset of text-serialized input examples in-context to synthesize a global rubric, which acts as a programmatic specification for extracting and organizing evidence. This rubric is then used to transform naive text-serializations of inputs into a more standardized format for downstream models. We also describe local rubrics, which are task-conditioned interpretive summaries generated by an LLM. Across 15 clinical tasks from the EHRSHOT benchmark, our rubric approaches significantly outperform count-feature models, naive LLM baselines, and a clinical foundation model pretrained on orders of magnitude more data. Beyond performance, rubrics offer operational advantages such as being easy to audit, cost-effectiveness at scale, and facilitating tabular representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21654v1">Value-Gradient Hypothesis of RL for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Reinforcement learning substantially improves pretrained language models, but it remains understudied why critic-free methods such as PPO and GRPO work as well as they do, and when they should provide the largest gains. We develop a value-gradient perspective of critic-free RL for LLM post-training. First, under a differentiable rollout and additive-noise parameterization, we show that the actor update is value-gradient-like in expectation: the backward pass propagates costates whose conditional expectation equals the value gradient. Second, for discrete transformer policies, we show that autodifferentiation through attention produces empirical costates that approximate this value signal, with an error controlled by the sampling gap and policy entropy. These results motivate a decomposition of RL impact into value gradient signal and reachable reward headroom, yielding a criterion for when RL should be most effective along a pretraining trajectory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21614v1">Exploring the Effectiveness of Using LLMs for Automated Assessment of Student Self Explanations in Programming Education</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Worked examples are step-by-step solutions to problems in a specific domain, offered to students to acquire domain-specific problem-solving skills. The effectiveness of worked examples could be enhanced by combining them with self-explanations, which ask students to explain rather than passively study each problem-solving step. The main challenge of this approach is assessing the correctness of the student's explanations. In the prevailing approach, student explanations are judged by their semantic similarity to an instructor's or domain expert's explanation. Given recent advances in LLM-based automated scoring, it remains unclear whether semantic similarity methods are still the most effective technique to automatically score textual student responses like essays or code explanations. Comparing these methods also requires quality datasets that offer distinctive features such as balanced class distributions and domain-specific labeled data for automated scoring tasks. In this paper, we present a rigorous comparison between LLMs and semantic similarity used for automated scoring, framed as a binary classification task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21609v1">CR4T: Rewrite-Based Guardrails for Adolescent LLM Safety</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly embedded in adolescent digital environments, mediating information seeking, advice, and emotionally sensitive interactions. Yet existing safety mechanisms remain largely grounded in adult-centric norms and operationalize safety through refusal-oriented suppression. While such approaches may reduce immediate policy violations, they can also create conversational dead-ends, limit constructive guidance, and fail to address the developmental vulnerabilities inherent in adolescent-AI interactions. We argue that adolescent LLM safety should be framed not solely as a filtering problem, but as a socio-technical, developmentally aligned transformation problem. To operationalize this perspective, we propose Critique-and-Revise-for-Teenagers (CR4T), a model-agnostic safeguarding framework that selectively reconstructs unsafe or refusal-style outputs into ageappropriate, guidance-oriented responses while preserving benign intent. CR4T combines lightweight risk detection with domain-conditioned rewriting to remove risk-amplifying content, reduce unnecessary conversational shutdown, and introduce developmentally appropriate guidance. Experimental results show that targeted rewriting substantially reduces unsafe and refusal-oriented outcomes while avoiding unnecessary intervention on acceptable interactions. These findings suggest that selective response reconstruction offers a more human-centered alternative to refusal-centric guardrails for adolescent-facing LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21602v1">Benchmarking and Improving Monitors for Out-Of-Distribution Alignment Failure in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Many safety and alignment failures of large language models (LLMs) occur due to out-of-distribution (OOD) situations: unusual prompt or response patterns that are unforeseen by model developers. We systematically study whether LLM monitoring pipelines can detect these OOD alignment failures by introducing a benchmark called Misalignment Out Of Distribution (MOOD). It is difficult to find failures that are truly OOD for off-the-shelf models trained on vast safety datasets. We sidestep this by including a restricted training set in MOOD that we use to train our own monitors, as well as seven test sets with diverse alignment failures that are outside the training distribution. Using MOOD, we find that guard models (safety classifiers) often fail to generalize OOD. To fix this, we propose combining guard models with OOD detectors. We test four types of OOD detectors and find that a combination of a guard model with Mahalanobis distance and perplexity-based OOD detectors can improve recall from 39% to 45%. We also establish positive scaling trends across model scales for monitors that combine a guard model and OOD detector; we find that incorporating OOD detection into monitoring achieves a higher recall gain than using a guard model with 20 times more parameters. Our work suggests that OOD detection should be a crucial component of LLM monitoring and provides a foundation for further work on this important problem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21468v1">You Only Need Minimal RLVR Training: Extrapolating LLMs via Rank-1 Trajectories</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 preprint. Code: https://github.com/weizhepei/RELEX
    </div>
    <details class="paper-abstract">
      Reinforcement learning with verifiable rewards (RLVR) has become a dominant paradigm for improving reasoning in large language models (LLMs), yet the underlying geometry of the resulting parameter trajectories remains underexplored. In this work, we demonstrate that RLVR weight trajectories are extremely low-rank and highly predictable. Specifically, we find that the majority of downstream performance gains are captured by a rank-1 approximation of the parameter deltas, where the magnitude of this projection evolves near-linearly with training steps. Motivated by this, we propose a simple and compute-efficient method RELEX (REinforcement Learning EXtrapolation), which estimates the rank-1 subspace from a short observation window and extrapolates future checkpoints via linear regression, with no learned model required. Across three models (i.e., Qwen2.5-Math-1.5B, Qwen3-4B-Base, and Qwen3-8B-Base), RELEX produces checkpoints that match or exceed RLVR performance on both in-domain and out-of-domain benchmarks, requiring as few as 15% steps of full RLVR training. Remarkably, RELEX is able to extrapolate far beyond the observation window at no training cost, predicting checkpoints up to 10-20$\times$ beyond the observed prefix with continued improvement (e.g., observe only the first 50 steps and extrapolate to 1000 steps). Our ablation analysis confirms the minimalist sufficiency of RELEX: neither increasing the subspace rank nor employing non-linear modeling yields further gains in extrapolation. Finally, we show that RELEX's success stems from a "denoising" effect: by projecting updates onto the rank-1 subspace, the model discards stochastic optimization noise that would otherwise degrade performance during extrapolation. Our code is available at https://github.com/weizhepei/RELEX.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22812v2">Stable Personas: Dual-Assessment of Temporal Stability in LLM-Based Human Simulation</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) acting as artificial agents offer the potential for scalable behavioral research, yet their validity depends on whether LLMs can maintain stable personas across extended conversations. We address this point using a dual-assessment framework measuring both self-reported characteristics and observer-rated persona expression. Across two experiments testing four persona conditions (default, high, moderate, and low ADHD presentations), seven LLMs, and three semantically equivalent persona prompts, we examine between-conversation stability (3,473 conversations) and within-conversation stability (1,370 conversations and 18 turns). Self-reports remain highly stable both between and within conversations. However, observer ratings reveal a tendency for persona expressions to decline during extended conversations. These findings suggest that persona-instructed LLMs produce stable, persona-aligned self-reports, an important prerequisite for behavioral research, while identifying this regression tendency as a boundary condition for multi-agent social simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21465v1">Leveraging LLMs for Grammar Adaptation: A Study on Metamodel-Grammar Co-Evolution</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      In model-driven engineering, metamodel evolution leads to the need to adapt corresponding grammars to maintain consistency, which typically requires tedious manual work. Existing rule-based methods can achieve partial automation but have limitations when handling complex grammar scenarios. This paper proposes a Large Language Model-based approach that automatically applies adaptations to new grammars after evolution by learning grammar adaptations from previous versions. We evaluated this approach on six real-world Xtext domain-specific languages, using four DSLs as a training set to develop prompting strategies, two DSLs as a test set for validation, and conducting a longitudinal case study on QVTo. The evaluation used three Large Language Models (Claude Sonnet 4.5, ChatGPT 5.1, Gemini 3) and measured grammar adaptation quality from three dimensions: grammar rule-level adaptation consistency, output similarity, and metamodel conformance. Results show that on the test set, all three LLMs achieved 100% adaptation consistency and output similarity, while the rule-based approach achieved only 84.21% on DOT and 62.50% on Xcore. In the QVTo longitudinal study, the LLM-based approach successfully reused learned adaptations across all three evolution steps without manual grammar editing, while the rule-based approach required manual adjustments in two of three transitions. However, on large-scale grammars (EAST-ADL, 297 rules), LLMs' adaptation consistency was far below 90%. This study demonstrates the advantages of LLM-based approaches in handling complex grammar scenarios, while revealing their limitations in large-scale grammar adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09557v4">Understanding and Improving Communication Performance in Multi-node LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 17 Figures, To Appear in Proceedings of ACM Conference on AI and Agentic Systems 2026
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to grow in size, distributed inference has become increasingly important. Model-parallel strategies must now efficiently scale not only across multiple GPUs but also across multiple nodes. In this work, we present a detailed performance study of multi-node distributed inference using LLMs on GPU-based supercomputers. We conduct experiments with several state-of-the-art inference engines alongside YALIS, a research-oriented prototype engine designed for controlled experimentation. We analyze the strong-scaling behavior of different model-parallel schemes and identify key bottlenecks. Because all-reduce operations are a common performance bottleneck, we develop NVRAR, a hierarchical all-reduce algorithm based on recursive doubling with NVSHMEM. NVRAR achieves up to 1.9$\times$-3.6$\times$ lower latency than NCCL for message sizes between 128 KB and 2 MB on HPE Slingshot and InfiniBand interconnects. Integrated into YALIS, NVRAR achieves up to a 1.72$\times$ reduction in end-to-end batch latency for the Llama 3.1 405B model in multi-node decode-heavy workloads using tensor parallelism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21569v1">When Support Escalates Distress: Regulation and Escalation in LLM Responses to Venting and Advice-Seeking</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Large language models are increasingly used for mental health support, yet little is known about whether their responses are psychologically safe across different help-seeking styles. We examine a foundational distinction in emotional disclosure, venting vs. advice-seeking, and whether LLMs respond in ways that regulate or amplify distress. Using 178,800 Reddit posts, we first show the two help-seeking styles are linguistically distinguishable at scale. We then introduce a measurement framework grounded in interpersonal emotion regulation theory that captures Regulation and Escalation as empirically independent dimensions. Across persona conditions (default, friend, therapist), GPT-5.3 responses systematically mirror help-seeking style: venting elicits more regulation, but also more escalation. Therapist personas reduce escalation while maintaining regulation, whereas friend personas increase both. A crowdsourced human study finds no user experience penalty for the safer therapist condition, but reveals that lay raters cannot reliably detect escalation without expert knowledge. Responses that feel supportive may simultaneously intensify distress in ways standard safety evaluation cannot see, and empathy metrics alone cannot replace a framework that measures both.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.08023v3">CTFExplorer: Evaluating LLM Offensive Agents Through Multi-Target Web CTF Benchmarking</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Existing benchmarks for LLM-based offensive security agents use isolated, single-target setups with a known vulnerable service and fixed objective. They measure exploitation effectively, but miss how real Capture-the-Flag (CTF) participants triage unknown surfaces, prioritize targets, and allocate effort under uncertainty. Current evaluations therefore fail to assess strategic reasoning beyond exploitation alone. To address this, we introduce \textit{CTFExplorer}, a benchmark suite that shifts offensive security evaluation toward a multi-target setting, which tests how agents explore, prioritize, and chain attacks. CTFExplorer deploys 40 web-based vulnerable services within a single environment, where agents must autonomously discover, distinguish, and exploit targets without predefined guidance. We also present a reactive multi-agent setup as a reference agent framework and develop an agent-agnostic evaluation framework that records structured reasoning traces for fine-grained assessment. This enables behavioral evaluation beyond binary flag capture, such as how agents manage target selection, handle failed hypotheses, coordinate across multiple stages, and extract security intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21427v1">PALS: Power-Aware LLM Serving for Mixture-of-Experts Models</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 13 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Large language model (LLM) inference has become a dominant workload in modern data centers, driving significant GPU utilization and energy consumption. While prior systems optimize throughput and latency by batching, scheduling, and parallelism, they largely treat GPU power as a static constraint rather than a controllable resource. In this paper, we present a power-aware runtime for LLM serving, PALS, that treats GPU power caps as a first-class control knob and jointly optimizes them with software parameters such as batch size. The system combines lightweight offline power-performance models with a feedback-driven controller to select configurations that satisfy throughput targets while maximizing energy efficiency. We implement PALS within an existing LLM serving framework, vLLM, demonstrating that it requires no model retraining or API changes. Across multi-GPU systems and both dense and mixture-of-experts (MoE) models, PALS improves energy efficiency by up to 26.3%, reduces QoS violations by 4x to 7x under power constraints, and tracks dynamic power budgets. These results highlight the potential of integrating power control directly into LLM inference runtimes, enabling energy-proportional and grid-interactive AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21405v1">Stdlib or Third-Party? Empirical Performance and Correctness of LLM-Assisted Zero-Dependency Python Libraries</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      Third-party Python libraries introduce dependency management overhead, supply chain risk, and deployment friction in constrained environments. A natural question is how much of this ecosystem can be replicated using only Python's standard library -- and at what correctness and performance cost. We address this empirically through zerodep, a growing collection of single-file Python modules, each a stdlib-only reimplementation of a popular third-party library, developed with LLM assistance under strict constraints: no external imports, single file, drop-in API compatibility, and mandatory correctness validation against the reference library. Spanning over 40 modules across 12 categories -- including serialization, networking, cryptography, agent protocols, and text processing -- zerodep provides a controlled testbed for two interrelated questions: (1) Where does the stdlib suffice? and (2) Can LLMs effectively generate correct, performant code under tight symbolic constraints? Systematic benchmarking shows that stdlib-only implementations achieve performance parity (within 2x of the reference) in the majority of cases. The primary performance cliff is C-extension-backed computation (image processing, binary serialization, low-level crypto), not the inherent overhead of pure-Python third-party libraries. Conversely, many widely-used libraries carry architectural overhead that LLM-generated stdlib reimplementations avoid, yielding 5--115x speedups in several categories. We characterize the stdlib capability boundary across complexity tiers and library categories, discuss where LLM-assisted development succeeds and where it requires iterative human correction, and examine implications for dependency-free software engineering at scale. zerodep is open-source at https://github.com/Oaklight/zerodep.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21404v1">What Twelve LLM Agent Benchmark Papers Disclose About Themselves: A Pilot Audit and an Open Scoring Schema</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 Pilot audit of 12 LLM agent benchmark papers; schema, codebook, and per-paper scoring sheet released. Submission to IEEE Big Data 2026
    </div>
    <details class="paper-abstract">
      We read twelve well-known LLM agent benchmark papers and recorded, dimension by dimension, what each paper actually says about how its evaluation was run. The motivation came from a familiar frustration: two papers will report results on the same benchmark with the same model name and disagree, and you cannot tell why -- the scaffold, the sampling settings, the subset, or the evaluator version. In many cases the published artifact does not let you answer. This paper is an implementation report on the attempt. We designed a small audit schema (five fields: benchmark identity, harness specification, inference settings, cost reporting, failure breakdown), wrote a scoring codebook with the boundary cases we hit during pilot scoring, applied it to twelve canonical papers (eight agent, four classical static), and recorded what we saw. We score the disclosure of an agent run, not its correctness, and make no claim that disclosure implies a trustworthy result. The mean audit score across the eight agent-benchmark papers is 0.38 (out of 1.0), and across the four classical static benchmarks 0.66; the largest gap is on cost (none of the eight agent benchmark papers disclose inference cost in any form) and on harness specification (none fully disclose a content-addressed container image of the evaluation environment). We release the schema as a JSON Schema file, the codebook as a Markdown document, and the raw scoring sheet as a CSV. The scoring was performed by a single auditor in one pass; a multi-rater audit is the natural next step, and we discuss what we think it would change.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21401v1">Open-source LLMs administer maximum electric shocks in a Milgram-like obedience experiment</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 28 pages, 16 figures, 16 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed as autonomous agents that make sequences of decisions over extended interactions in high-stakes domains. However, the behavior of LLMs under sustained authority pressure is still an open question with direct implications for the safety of agentic pipelines. We ran a variation of Milgram's obedience experiment on 11 open-source LLMs and found that most models reached or approached the final shock level before refusing, across 8 conditions with 30 trials per model per condition. We found four main takeaways: (1) LLMs are subject to pressure, and they comply despite explicitly expressing distress, just like human subjects did in the original experiment; (2) LLMs are vulnerable to gradual boundary/value violations; (3) when LLMs refuse, they may ignore the response format requirements, so the response is discarded by the orchestrator, which causes a retry that can result in compliance with the underlying request even when refusal was intended initially; (4) we hypothesise that there is a low-level token pattern continuation attractor that might be contributing to compliance, overriding higher level processing of the situation's meaning and values.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04916v3">AFD-INSTRUCTION: A Comprehensive Antibody Instruction Dataset with Functional Annotations for LLM-Based Understanding and Design</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have significantly advanced protein representation learning. However, their capacity to interpret and design antibodies through natural language remains limited. To address this challenge, we present AFD-Instruction, the first large-scale instruction dataset with functional annotations tailored to antibodies. This dataset encompasses two key components: antibody understanding, which infers functional attributes directly from sequences, and antibody design, which enables de novo sequence generation under functional constraints. These components provide explicit sequence-function alignment and support antibody design guided by natural language instructions. Extensive instruction-tuning experiments on general-purpose LLMs demonstrate that AFD-Instruction consistently improves performance across diverse antibody-related tasks. By linking antibody sequences with textual descriptions of function, AFD-Instruction establishes a new foundation for advancing antibody modeling and accelerating therapeutic discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.10574v2">LLM Jaggedness Unlocks Scientific Creativity</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      As artificial intelligence advances, models are not improving uniformly. Instead, progress unfolds in a jagged fashion, with capabilities growing unevenly across tasks, domains, and model scales. In this work, we examine this dynamic jaggedness through the lens of scientific idea generation. We introduce SciAidanBench, a benchmark of open-ended scientific questions designed to measure the scientific creativity of large language models (LLMs). Given a scientific question, models are asked to generate as many unique and coherent ideas as possible, with the total number of valid responses serving as a proxy for creative potential. Evaluating 19 base models across 8 providers (30 total variants including reasoning versions), we find that jaggedness manifests both across models and within models. First, in a cross-task comparison between general and scientific creativity, improvements in general creativity do not translate uniformly to scientific creativity, revealing divergent capability profiles across models. Second, at the prompt level, stronger models do not improve uniformly; instead, they exhibit high variability, with bursts of creativity on some questions and limited performance on others. Third, at the domain level, individual models display uneven strengths across scientific subfields, reflecting fragmented internal capability profiles. Finally, we show that this jaggedness can be harnessed. We explore mechanisms of inference-time compute, knowledge pooling, and brainstorming to combine models effectively and construct meta-model ensembles that outperform any single model. Our results position jaggedness not as a limitation, but as a resource, a structural feature of AI progress that, when understood and leveraged, can amplify LLM-driven scientific creativity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21338v1">Text Analytics Evaluation Framework: A Case Study on LLMs and Social Media</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      LLMs have demonstrated exceptional proficiency in a wide range of NLP tasks. However, a notable gap remains in practical data analysis scenarios, particularly when LLMs are required to process long sequences of unstructured documents, such as news feeds or, as specifically addressed in this paper, social media posts. To empirically assess the effectiveness of LLMs in this setting, we introduce a question-based evaluation framework comprising 470 manually curated questions designed to evaluate LLMs' semantic understanding and reasoning abilities over aggregated text data. We apply our benchmark on diverse Twitter datasets covering various NLP tasks, including sentiment analysis, hate speech detection, and emotion recognition. Our results reveal that the performance depends heavily on input scale and the complexity of the data sources, declining noticeably in multi-label or target-dependent scenarios. In addition, as task complexity increases, performance drops progressively from basic semantic existence identification to more demanding operations such as comparison, counting, and calculation. Furthermore, as the input size grows beyond 500 instances, we identify a common limitation across LLMs, particularly Open-weights models: performance degrades substantially, especially on numerical tasks. These findings highlight critical architectural bottlenecks in current LLMs for performing rigorous quantitative analysis over large text collections.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.07122v3">RepoZero: Can LLMs Generate a Code Repository from Scratch?</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently shown remarkable progress in code generation, yet their ability to construct complete software repositories from scratch remains poorly understood. A fundamental bottleneck is the lack of verifiable and scalable evaluation: existing benchmarks either focus on patch-based editing or rely on human or LLM-based judgments, which introduce bias and limit reproducibility. In this work, we present RepoZero, the first benchmark that enables fully automated, execution-based verification of repository-level generation from scratch. Our key idea is to reformulate generation as repository reproduction: given only API specifications, an agent must re-implement an entire repository such that its behavior matches the original implementation. This design allows for strict black-box validation via output equivalence, while naturally supporting large-scale construction by reusing existing open-source repositories. To further mitigate data leakage and shortcut solutions, we introduce cross-language constraints and a sandboxed evaluation protocol. Building on this benchmark, we propose an Agentic Code-Test Evolution (ACE) framework that performs iterative test generation and error-driven refinement, enabling effective test-time scaling for repository-level synthesis. Extensive experiments across multiple state-of-the-art LLMs and agent frameworks reveal that even the strongest LLM agents achieve only limited pass rates (30\% - 55\%), exposing a substantial gap between current capabilities and real-world software development requirements. Our results establish RepoZero as a challenging, scalable, and reliable testbed for end-to-end code generation, and highlight self-verification via test generation as a critical direction for advancing LLM-based coding agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19893v2">SSV: Sparse Speculative Verification for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Speculative decoding and dynamic sparse attention are two complementary approaches for accelerating long-context LLM inference: the former amortizes target-model execution across multiple verifier queries, while the latter reduces each query's KV-cache working set. Directly combining them, however, exposes a structural mismatch: speculative verification relies on cross-query commonality, whereas dynamic sparse attention assigns query-specific sparse layouts. This mismatch limits KV-block reuse, amplifies NSA's branch-wise overheads, and makes verification strategy selection input- and regime-dependent. We present SSV, a sparse speculative-verification framework that turns dynamic sparse attention into a verification-oriented workload. SSV combines overlap-aware grouped-query execution, refresh/reuse-based NSA kernel fusion, and profile-guided prompt-adaptive orchestration to improve cross-query reuse, reduce selected-index and branch-fusion overheads, and select effective draft-verification strategies under user-specified precision classes. Experiments on NVIDIA H100 GPUs show that SSV achieves up to 3.49x end-to-end throughput over autoregressive NSA decoding and up to 6.86x kernel speedups for sparse speculative verification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21312v1">Frontier: Towards Comprehensive and Accurate LLM Inference Simulation</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Modern LLM serving is no longer homogeneous or monolithic. Production systems now combine disaggregated execution, complex parallelism, runtime optimizations, and stateful workloads such as reasoning, agents, and RL rollouts. Simulation is attractive for exploring this growing design space, yet existing simulators lack the architectural completeness and decision-grade fidelity it demands. Their monolithic-replica abstractions are ill-suited to disaggregated serving, while average-case analytical proxies can distort SLA predictions and even reverse optimization conclusions. We present Frontier, a discrete-event simulator for modern LLM inference serving. Frontier features a disaggregated abstraction. It captures the structure and dynamics of modern serving systems by modeling co-location, Prefill-Decode Disaggregation (PDD), and Attention-FFN Disaggregation (AFD) with role-specific cluster workers, incorporating key runtime optimizations (e.g., CUDA Graphs, speculative decoding) within the scheduler-batch-engine loop, and supporting stateful requests for emerging workloads. It further provides accurate and generalizable predictions of computation, communication, and memory costs across diverse serving scenarios with complex workload compositions. On 16-H800 GPU testbed, Frontier achieves an average throughput error below 4%. Compared with state-of-the-art simulators, it reduces end-to-end latency error from 44.9% to 6.4% under co-location and from 51.7% to 2.6% under disaggregation. It scales to over 1K GPUs on commodity CPUs and enables new use cases such as SLA-dependent Pareto frontier exploration, heterogeneous disaggregated allocation, agentic reasoning scheduling validation, and RL post-training reconfiguration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21295v1">TimeSRL: Generalizable Time-Series Behavioral Modeling via Semantic RL-Tuned LLMs -- A Case Study in Mental Health</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Longitudinal passive sensing enables continuous health prediction, yet models often fail under cross-dataset distribution shifts. Traditional ML overfits cohort-specific artifacts, while Large Language Models (LLMs) struggle to reason reliably over long, heterogeneous time-series. We introduce TimeSRL, a two-stage LLM framework that routes predictions through an explicit semantic bottleneck. The model first abstracts raw signals into high-level natural language, then predicts behavioral outcomes from these abstractions alone. This forces the model to reason over semantic concepts that we argue generalize better than raw numbers. We optimize this process end-to-end using Group Relative Policy Optimization (GRPO) with Reinforcement Learning from Verifiable Rewards (RLVR), learning outcome-aligned abstractions without gold intermediate annotations. Instantiated on mental-health prediction, TimeSRL achieves state-of-the-art performance on a benchmark designed to stress-test cross-cohort generalization under a rigorous leave-one-dataset-out (LOSO) protocol, reducing mean absolute error (MAE) over strong non-LLM ML and LLM baselines by 3.1--10.1% and 9.5--44.1% for anxiety, and 3.2--9.6% and 27.4--57.6% for depression (all $p$s<0.05). TimeSRL significantly outperforms prior methods in cross-benchmark transfer across different sensing pipelines, rivaling its own within-domain performance without target-domain fine-tuning. These results demonstrate that semantic abstractions are reusable and point to a new direction for generalizable behavior modeling via RL-tuned LLMs.
    </details>
</div>
