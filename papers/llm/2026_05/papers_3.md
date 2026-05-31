# llm - 2026_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)

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
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.15949v5">ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Large language models show promise for financial decision-making, yet deploying them as autonomous trading agents raises fundamental challenges: how to adapt instructions when rewards arrive late and obscured by market noise, how to synthesize heterogeneous information streams into coherent decisions, and how to bridge the gap between model outputs and executable market actions. We present ATLAS (Adaptive Trading with LLM AgentS), a unified multi-agent framework that integrates structured information from markets, news, and corporate fundamentals to support robust trading decisions. Within ATLAS, the central trading agent operates in an order-aware action space, ensuring that outputs correspond to executable market orders rather than abstract signals. The agent can incorporate feedback while trading using Adaptive-OPRO, a novel prompt-optimization technique that dynamically adapts the prompt by incorporating real-time, stochastic feedback, leading to increasing performance over time. Across regime-specific equity studies and multiple LLM families, Adaptive-OPRO consistently outperforms fixed prompts, while reflection-based feedback fails to provide systematic gains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.24472v3">Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs?</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 Code is available at https://github.com/beanie00/self-distillation-analysis
    </div>
    <details class="paper-abstract">
      Self-distillation has emerged as an effective post-training paradigm for LLMs, often improving performance while shortening reasoning traces. However, in mathematical reasoning, we find that it can reduce response length while degrading performance. We trace this degradation to the suppression of epistemic verbalization - the model's expression of uncertainty during reasoning. Through controlled experiments varying conditioning context richness and task coverage, we show that conditioning the teacher on rich information suppresses uncertainty expression, enabling rapid in-domain optimization with limited task coverage but harming OOD performance, where unseen problems benefit from expressing uncertainty and adjusting accordingly. Across Qwen3-1.7B/8B, DeepSeek-Distill-Qwen-7B, and Olmo3-7B-Instruct, we observe performance drops of up to 40%. Our findings highlight that exposing appropriate levels of uncertainty is crucial for robust reasoning and underscore the importance of optimizing reasoning behavior beyond merely reinforcing correct answer traces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21560v1">AutoMCU: Feasibility-First MCU Neural Network Customization via LLM-based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Deploying neural networks on microcontroller units (MCUs) is critical for edge intelligence but remains challenging due to tight memory, storage, and computation constraints. Existing approaches, such as model compression and hardware-aware neural architecture search (HW-NAS), often depend on proxy metrics, incur high search cost, and do not fully bridge the gap between architecture design and verified deployment. This paper presents AutoMCU, a feasibility-first large language model (LLM)-based multi-agent system for automated neural network customization under MCU constraints. Given natural-language task requirements and hardware specifications, AutoMCU iteratively generates structured architecture candidates, filters infeasible designs through vendor toolchain feedback before training, evaluates feasible models under a controlled protocol, and verifies deployability through backend-grounded deployment analysis. AutoMCU includes two key mechanisms: 1) hardware-in-the-loop architecture generation for early elimination of undeployable candidates under RAM and Flash constraints, and 2) state-isolated multi-agent scheduling for stable coordination of proposal, training, evaluation, and deployment stages. Experiments on CIFAR-10 and CIFAR-100 under strict MCU constraints show that AutoMCU achieves competitive accuracy while reducing customization time to about 1--2 hours, compared with hundreds of GPU hours for representative MCU-oriented HW-NAS baselines. Comparisons with ColabNAS and the LLM-based NAS method GENIUS on NAS-Bench-201 further demonstrate the effectiveness and stability of AutoMCU. Real-device deployments on multiple STM32 microcontrollers validate its practical applicability to MCU-scale edge intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21240v1">APEX: Autonomous Policy Exploration for Self-Evolving LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      LLM agents have shown strong performance across a wide range of complex tasks, including interactive environments that require long-horizon decision making. But these agents cannot learn on the fly at test time. Self-evolving agents address this by accumulating memory and reflection across episodes rather than requiring model-weight updates. However, these agents often suffer from exploration collapse: as memory grows, behavior concentrates around familiar high-reward routines, reducing the chance of discovering better alternatives. To address this problem, we propose Autonomous Policy EXploration (APEX), which builds and maintains an explicit strategy space through a strategy map-a directed acyclic graph of milestones with prerequisite dependency edges. In APEX, Fork Discovery expands the map with evidence-grounded unexplored directions, while Policy Selection balances exploration and exploitation during planning. Evaluated on nine Jericho text-adventure games and WebArena, a realistic web interaction benchmark, APEX outperforms all baselines. Extensive ablations validate each component's contribution and demonstrate robustness across diverse settings, demonstrating APEX's effectiveness for sustained exploration in self-evolving agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21558v1">From Parameters to Data: A Task-Parameter-Guided Fine-Tuning Pipeline for Efficient LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 Accepted@ICML26, 28 pages, 11 figures, 26 tables
    </div>
    <details class="paper-abstract">
      Adapting Large Language Models (LLMs) to specialized domains typically incurs high data and computational overhead. While prior efficiency efforts have largely treated data selection and parameter-efficient fine-tuning as isolated processes, our empirical analysis suggests they may be intrinsically coupled. We posit the Strong Map Hypothesis: a sparse subset of attention heads plays a dominant role in task-specific adaptation, acting as keys that unlock specific data patterns. Building on this observation, we propose From Parameters to Data (P2D), a unified framework that leverages these task-sensitive attention heads as a dual compass for both sample mining and structural pruning. To rigorously quantify the total pipeline cost, we introduce the Alignment Efficiency Ratio (AER) metric for both selection latency and training time. Mechanistically, P2D identifies critical heads via a lightweight proxy and uses them as a functional filter to curate high-affinity data, establishing a synergistic pipeline. Empirically, by updating merely 10% of attention heads on 10% of the data, P2D achieves an 8.3 pp performance gain over strong baselines and delivers a 7.0x end-to-end time speedup. These results validate that precise parameter-data synchronization eliminates redundancy, offering a new paradigm for efficient alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21227v1">Do LLMs Know What Luxembourgish Borrows? Probing Lexical Neology in Low-Resource Multilingual Models</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 Accepted to Neollm colocated with LREC2026, Three figures and three tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used for writing assistance in small contact languages, yet it is unclear whether they respect community norms around lexical borrowing and neology. We introduce LexNeo-Bench, a 3{,}050-instance token-level benchmark derived from LuxBorrow, a large-scale Luxembourgish news corpus, where target tokens are labelled as native or as French, German, or English borrowings. Using this benchmark, we probe three multilingual LLMs across 34 prompt settings on two tasks: borrowing type classification and a binary lexical-innovation proxy (borrowing versus native). Without external context, models perform only slightly above chance on borrowing classification, so we construct a linguistic knowledge graph that encodes donor language, morphological patterns, and lexical analogues, and inject instance-specific subgraphs into the prompt. Knowledge-graph prompts raise borrowing classification accuracy from 25 -- 35\% up to 71 -- 81\% and largely close the gap between small and large models, while leaving neology detection difficult and sensitive to few-shot design. Our results show that lexicon-aware prompting is highly beneficial for robust borrowing judgments in low-resource contact languages and that lexical resources can serve as structured context for LLM evaluation. This study was carried out within the ENEOLI COST Action and examines borrowing as a form of lexical innovation in multilingual Luxembourgish data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21217v1">Federated LoRA Fine-Tuning for LLMs via Collaborative Alignment</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Low-rank adaptation (LoRA) has emerged as a powerful tool for parameter-efficient fine-tuning of large language models (LLMs). This paper studies LoRA under a federated learning setting, enabling collaborative fine-tuning across clients while preserving parameter efficiency. We focus on a highly heterogeneous regime in which clients share only partial structure and a substantial subset may be contaminated. We propose Collaborative Low-rank Alignment and Identifiable Recovery (CLAIR), a contamination-aware framework that relies only on preliminary local estimators. Its formulation applies broadly, from linear regression to neural network and LLM modules, whenever local adaptation can be represented by matrix-valued updates. CLAIR recovers the shared LoRA subspace and detects contaminated clients via a structured low-rank plus block-sparse decomposition. We prove exact recovery of the shared LoRA subspace in the noiseless case, stable recovery under preliminary estimation error, and consistent collaborative-set recovery under mild separation conditions. We further quantify the gain from CLAIR refinement: it reduces off-subspace estimation error through cross-client averaging while preserving client-specific variation within the shared LoRA subspace, thus improves over local fine-tuning whenever this oracle gain outweighs the costs of subspace estimation and benign-client heterogeneity. Empirically, we demonstrate the benefits of CLAIR by fine-tuning a Transformer architecture on a text-copying task. The results show accurate contamination detection and improved benign-client performance compared with local fine-tuning and non-robust federated averaging.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.12120v3">LLMs on the Line: Data Determines Loss-to-Loss Scaling Laws</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 ICML 2025 camera-ready version
    </div>
    <details class="paper-abstract">
      Scaling laws guide the development of large language models (LLMs) by offering estimates for the optimal balance of model size, tokens, and compute. More recently, loss-to-loss scaling laws that relate losses across pretraining datasets and downstream tasks have emerged as a powerful tool for understanding and improving LLM performance and generalization. In this work, we investigate which factors most strongly influence loss-to-loss scaling. Our experiments reveal that the pretraining data determines the scaling trend. In contrast, model size, optimization hyperparameters, tokenizer and even significant architectural differences, such as between transformer-based models like Llama and state-space models like Mamba, generally have limited impact. Consequently, practitioners should carefully curate suitable pretraining datasets for optimal downstream performance, while architectures and other settings can be freely optimized for training efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2305.12138v5">Exploring Code Analysis: Zero-Shot Insights on Syntax and Semantics with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 Accepted at ACM Transactions on Software Engineering and Methodology (TOSEM)
    </div>
    <details class="paper-abstract">
      Code analysis is fundamental in Software Engineering, supporting debugging, optimization, and security assessment. Human developers approach it through syntax parsing, static semantics inference, and dynamic reasoning. Traditional tools are effective but limited by language specificity and weak cross-language generalization. Large language models (LLMs) are promising for code tasks, yet their capabilities for fundamental code analysis remain underexplored. We structure our study around three aspects aligned with human practices: syntax parsing, static semantics inference, and dynamic reasoning. We evaluate 21 state-of-the-art LLMs across nine tasks in four languages (C, Java, Python, Solidity), including AST generation, CFG construction, data dependency, taint analysis, and flaky test reasoning. We apply a three-layer evaluation protocol (automated metrics, expert adjudication, consistency validation) to 3,124 code samples, achieving high inter-rater reliability (Cohen's kappa = 0.844-0.936) and strong human-machine agreement (Gwet's AC1 = 0.500-0.727, F1 = 0.791-0.882). While the best LLMs excel in syntax parsing (AST 90%+, expression matching 84-100%) and show promise in static analysis, their dynamic reasoning remains limited (<70%) with high data-shift sensitivity (per-project F1 varying 0-1.0). This hierarchy holds across model families and scales, suggesting fundamental rather than transient limitations. These findings show how LLMs complement traditional analyzers: they offer cross-language generalization but non-deterministic outputs needing validation, while traditional tools give deterministic guarantees but need language-specific configuration. We contribute a validated evaluation framework with comparison against traditional analyzers (Tree-sitter, Soot, Joern) and task-specific applicability tiers. Benchmark: https://github.com/mathieu0905/llm_code_analysis.git
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14896v2">DrugRAG: Enhancing Pharmacy LLM Performance Through A Novel Retrieval-Augmented Generation Pipeline</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 14 pages, 2 figures, 2 tables. The revised version includes McNemar's paired statistical analysis, Wilson confidence intervals, expanded methodological clarifications, a revised discussion of evidence retrieval, improved reproducibility details, and updated limitations
    </div>
    <details class="paper-abstract">
      In our study, we evaluated large language model (LLM) performance on pharmacy licensure-style question-answering tasks and developed an external knowledge integration method to improve accuracy. We benchmarked ten LLMs with varying parameter sizes (8 billion to 70+ billion) using a 141-question pharmacy dataset, measuring baseline accuracy without modification. Baseline performance ranged from 46% to 92%, with GPT-5 (92%) and o3 (89%) achieving the highest scores, while smaller open-source models showed substantially lower performance. We then developed DrugRAG, a three-step retrieval-augmented generation (RAG) pipeline that retrieves structured, evidence-based drug information and augments model prompts with contextual pharmacological evidence, operating externally and requiring no changes to model architecture or parameters. DrugRAG increased accuracy across all five evaluated models, with gains ranging from 7 to 21 percentage points (e.g., Gemma 3 27B: 61.0% to 71%, Llama 3.1 8B: 46% to 67%). McNemar analyses demonstrated statistically significant paired improvements primarily in smaller and mid-sized open-source models. These findings demonstrate that integrating structured external drug knowledge via DrugRAG can improve LLM performance on pharmacy-focused question-answering tasks without modifying the underlying models, providing a practical pipeline for enhancing evidence-based pharmacy-focused AI applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.07926v2">AgentEscapeBench: Evaluating Out-of-Domain Tool-Grounded Reasoning in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      As LLM-based agents increasingly rely on external tools, it is important to evaluate their ability to sustain tool-grounded reasoning beyond familiar workflows and short-range interactions. We introduce AgentEscapeBench, an escape-room-style benchmark that tests whether agents can infer, execute, and revise novel tool-use procedures under explicit long-range dependency constraints. Each task defines a directed acyclic dependency graph over tools and items, requiring agents to invoke real external functions, track hidden state revealed incrementally, propagate intermediate results, and submit a deterministically verifiable final answer. AgentEscapeBench includes 270 instances across five difficulty tiers and supports fully automated evaluation. Experiments with sixteen LLM agents and human participants show that performance drops sharply as dependency depth increases: humans decline from 98.3% success at difficulty-5 to 80.0% at difficulty-25, while the best model drops from 90.0% to 60.0%. Trajectory analysis attributes model failures mainly to breakdowns in long-range state tracking, clue adherence, and intermediate-result propagation. These findings suggest that current agents can often handle local tool use but still struggle with deep contextual dependencies. We hope AgentEscapeBench can serve as a diagnostic testbed for measuring current agent capabilities and informing future training efforts toward more robust general-purpose reasoning, action, and adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.17631v4">Time-Prompt: Integrated Heterogeneous Prompts for Unlocking LLMs in Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 Accepted at IJCNN 2026
    </div>
    <details class="paper-abstract">
      Time series forecasting aims to model temporal dependencies among variables for future state inference, holding significant importance and widespread applications in real-world scenarios. Although deep learning-based methods have achieved remarkable progress, they still exhibit suboptimal performance in long-term forecasting. Recent research demonstrates that large language models (LLMs) achieve promising performance in time series forecasting, but this progress is still met with skepticism about whether LLMs are truly useful for this task. To address this, we propose Time-Prompt, a framework for activating LLMs for time series forecasting. Specifically, we first construct a unified prompt paradigm with learnable soft prompts to guide the LLM's behavior and textualized hard prompts to enhance the time series representations. Second, to enhance LLM' comprehensive understanding of the forecasting task, we design a semantic space embedding and cross-modal alignment module to achieve fusion of temporal and textual data. Finally, we efficiently fine-tune the LLM's parameters using time series data. Furthermore, we focus on carbon emissions, aiming to provide a modest contribution to global carbon neutrality. Comprehensive evaluations on 6 public datasets and 3 carbon emission datasets demonstrate that Time-Prompt is a powerful framework for time series forecasting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21082v1">AutoRPA: Efficient GUI Automation through LLM-Driven Code Synthesis from Interactions</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 Accepted in ICML 2026
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) based agents have demonstrated proficiency in multi-step interactions with graphical user interfaces (GUIs). While most research focuses on improving single-task performance, practical scenarios often involve repetitive GUI tasks for which invoking LLM reasoning repeatedly, i.e., the ReAct paradigm, is inefficient. Prior to LLMs, traditional Robotic Process Automation (RPA) offers runtime efficiency but demands significant manual effort to develop and maintain. To bridge this gap, we propose AutoRPA, a framework that automatically distills the decision logic of ReAct-style agents into robust RPA functions. AutoRPA introduces two core innovations: (1) A translator-builder pipeline, where a translator agent converts hard-coded ReAct actions into soft-coded procedures, and a builder agent synthesizes robust RPA functions via retrieval-augmented generation over multiple trajectories; (2) A hybrid repair strategy during code verification, combining RPA execution with ReAct-based fallback for iterative refinement. Experiments across multiple GUI environments demonstrate that RPA functions generated by AutoRPA successfully solve similar tasks while reducing token usage by 82% to 96%, significantly improving runtime efficiency and reusability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21063v1">APM: Evaluating Style Personalization in LLMs with Arbitrary Preference Mappings</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Typical LLM responses tend to follow a default style, even though users often have distinct preferences regarding tone, verbosity, and formality that they do not explicitly state in their prompts. Evaluating whether personalization methods can adapt to these implicit preferences is challenging, since users typically provide prompts rather than reference responses, style preferences are not factually verifiable, and reference-free LLM judges may conflate personalization with general response quality. To address these challenges, we introduce the Arbitrary Preference Mapping (APM) benchmark, which decouples user attributes (e.g. enthusiastic) from response principles (e.g. persuasive) via a hidden, randomized mapping $\mathbf{C}$ that maps user attributes to preferences about response traits. Because $\mathbf{C}$ carries no semantic content and is resampled across runs, models cannot exploit stereotypical associations and must infer preferences from conversation history. Using this unbiased evaluation methodology, we adapt retrieval-augmented, prompt-optimization, and routing personalization methods and evaluate them on Llama-3.1-8B and Qwen-3.5-27B. Our results show that routing is the most reliable approach, while RAG only improves with the stronger base LLM, and soft prompt optimization fails to improve significantly over a non-personalized baseline. Our extensive evaluation reveals that in this realistic setting, personalization remains challenging, but our adapted methods show promise.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21049v1">Cross-lingual robustness of LLM-brain alignment and its computational roots</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) reliably predict neural activity during language comprehension and transformer depth has been interpreted as mirroring hierarchical cortical organization. However, it remains unclear whether such alignment extends to subcortical regions, overlaps spatially across languages, and what the computational roots of such alignment are. Here, we used a multilingual, whole-brain encoding framework to examine brain-LLM alignment across three typologically distinct languages: Mandarin, English, and French during naturalistic story listening. Our results show that across languages, transformer-based models predicted activity in a distributed landscape spanning widely distributed cortical functional networks like limbic, ventral attention, default mode network, and subcortical structures. Spatial alignment patterns showed substantial cross-linguistic overlap and remained largely stable across model layers, with limited layer progression consistent with functional cortical hierarchies. Contrary to previous evidence, contextual embeddings did not outperform static embeddings. To test candidate computational explanations, we examined whether layer-wise brain scores reflect surprisal and intrinsic dimensionality, and thereby predictive processing and information compression. Neither of these two computational metrics mirrored neural alignment profiles. Our findings suggest that brain-LLM alignment is spatially robust and cross-linguistically stable but not explainable from predictive uncertainty or representational geometry. Rather than directly reflecting shared hierarchical computation, neural predictivity may primarily arise from distributed lexical-semantic correspondences that generalize across languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.18309v2">From Program Slices to Causal Clarity: Evaluating Faithful, Actionable LLM-Generated Failure Explanations via Context Partitioning and LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 10 pages, 5 figures, 5 tables. Accepted to EASE 2026 (EQUISA workshop), Glasgow, United Kingdom
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based debugging systems can generate failure explanations, but these explanations may be incomplete or incorrect. Misleading explanations are harmful for downstream tasks (e.g., bug triage, bug fixing). We investigate how explanation quality is affected by various LLM context configurations. Existing work predominantly treats LLM-generated failure explanations as an ad hoc by-product of debugging or repair workflows, using generic prompting over undifferentiated artifacts such as code, tests, and error messages rather than targeting explanations as a first-class output with dedicated quality assessment. Consequently, existing approaches provide limited support for assessing whether these explanations capture the underlying fault-error-failure mechanism and for actionable next steps, and most techniques instead prioritize task success (e.g., patch correctness or review quality) over the explicit causal explanation quality. We systematically vary the debugging information to study how distinct context compositions affect the quality of LLM-generated failure explanations. Across 93 context configurations on real bugs and three economically viable models (gpt-5-mini, DeepSeek-V3.2, and Grok-4.1-fast), we evaluate explanations with six criteria and validate the LLM-as-a-judge scores against human ratings in a user study. Our results indicate that explanation quality is causally affected by context composition. Evidence-rich, failure-specific artifacts improve causal and action-oriented quality, whereas overly large contexts tend to yield vague explanations. Higher explanation-score quartiles are associated with higher downstream repair pass rates and, for some models, with fixes that are closer to the reference minimal fixes. In contrast, low-score quartiles can even underperform the no-explanation baseline. Reproduction package is publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.25898v3">On Integrating Resilience and Human Oversight into LLM-Assisted Modeling Workflows for Digital Twins</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      LLM-assisted modeling holds the potential to rapidly build executable Digital Twins of complex systems from only coarse descriptions and sensor data. However, resilience to LLM hallucination, human oversight, and real-time model adaptability remain challenging and often mutually conflicting requirements. We present three critical design principles for integrating resilience and oversight into such workflows, derived from insights gained through our work on FactoryFlow - an open-source LLM-assisted framework for building simulation-based Digital Twins of manufacturing systems. First, orthogonalize structural modeling and parameter fitting. Structural descriptions (components, interconnections) are LLM-translated from coarse natural language to an intermediate representation (IR) with human visualization and validation, which is algorithmically converted to the final model. Parameter inference, in contrast, operates continuously on sensor data streams with expert-tunable controls. Second, restrict the model IR to interconnections of parameterized, pre-validated library components rather than monolithic simulation code, enabling interpretability and error-resilience. Third, and most important, is to use a density-preserving IR. When IR descriptions expand dramatically from compact inputs hallucination errors accumulate proportionally. We present the case for Python as a density-preserving IR : loops express regularity compactly, classes capture hierarchy and composition, and the result remains highly readable while exploiting LLMs strong code generation capabilities. A key contribution is detailed characterization of LLM-induced errors across model descriptions of varying detail and complexity, revealing how IR choice critically impacts error rates. These insights provide actionable guidance for building resilient and transparent LLM-assisted simulation automation workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21027v1">Beyond Text-to-SQL: An Agentic LLM System for Governed Enterprise Analytics APIs</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 The first four authors contributed equally to this work
    </div>
    <details class="paper-abstract">
      Enterprise analytics aims to make organizational data accessible for decision-making, yet non-technical users still face barriers when using traditional business intelligence tools or Text-to-SQL systems. While recent Text-to-SQL approaches based on Large Language Models (LLMs) promise natural language access to structured data, they fall short in enterprise settings where analytics pipelines rely on governed APIs rather than raw databases. In practice, these APIs encapsulate complex business logic to ensure consistency, auditability, and security. However, delegating mathematical or aggregation logic to an LLM introduces reliability and compliance risks. To this end, we present Analytic Agent, an LLM-based agentic system that translates natural language intents into secure interactions with enterprise analytics APIs. Evaluated on 90 real enterprise use cases constructed by domain experts, it reliably interprets user goals, validates permissions, executes governed queries, and generates compliant visualizations through multi-step reasoning and policy-aware orchestration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10726v2">PrefixWall: Mitigating Prefix Caching Side Channels in Shared LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) rely on optimizations like Automatic Prefix Caching (APC) to accelerate inference. APC works by reusing previously computed states for the beginning part of a request (prefix), when another request starts with the same text. While APC improves throughput, it introduces timing side channels: cache hits are faster than misses, creating observable latency differences. In multi-tenant systems, attackers can exploit these differences to infer sensitive information, e.g., by incrementally reconstructing another user's request by observing hit/miss patterns. Current defenses take a sledgehammer approach: they disable APC and cache sharing, isolating users, and sacrificing efficiency for regular users. This paper presents PrefixWall, a system that secures multi-tenant LLM serving systems against APC side channels without sacrificing performance and efficiency. PrefixWall monitors cache reuse across users, flags suspicious sharing, and selectively isolates prefixes, restricting their reuse only when necessary. Evaluation shows that PrefixWall enables up to 70% higher cache reuse and 30% lower inference latency compared to existing defenses that isolate users. PrefixWall's lightweight design demonstrates how security in LLM serving does not have to come at the cost of unnecessarily reduced performance or unbearable overheads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.15104v2">From Text to Voice: A Reproducible and Verifiable Framework for Evaluating Tool Calling LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Voice agents increasingly require reliable tool use from speech, whereas prominent tool-calling benchmarks remain text-based. We study whether verified text benchmarks can be converted into controlled audio-based tool calling evaluations without re-annotating the tool schema and gold labels. Our dataset-agnostic framework uses text-to-speech, speaker variation, and environmental noise to create paired text-audio instances while preserving the original dataset annotations. Based on extensive evaluation of 7 omni-modal models on audio-converted versions of Confetti and When2Call, our framework demonstrates that the performance is strongly model- and task-dependent: Gemini-3.1-Flash-Live obtains the highest Confetti score (70.4), whereas GPT-Realtime-1.5 performs best on When2Call (71.9). On Confetti, the text-to-voice gap ranges from 1.8 points for Qwen3-Omni to 4.8 points for GPT-Realtime-1.5. A targeted analysis of failure cases demonstrates that degradations most often reflect misunderstandings of argument values in the speech. Considering real-world deployment scenarios, we further report text-only results, an ambiguity-based reformulation stress test, and a reference-free LLM-as-judge protocol validated against human preferences. Notably, we find that open-source Qwen3 judges with at least 8B parameters exceed 80% agreement with proprietary judges, supporting privacy-preserving evaluation. Overall, our framework provides a verifiable and reproducible first-stage diagnostic that complements purpose-built audio corpora.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.14444v3">A Free Lunch in LLM Compression: Revisiting Retraining after Pruning</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Post-training pruning can substantially reduce LLM inference costs, but it often degrades quality unless the remaining weights are adapted. Since global retraining is expensive at LLM scale, recent work has largely focused on increasingly sophisticated pruning criteria that aim to select better sparsity patterns without adaptation. We revisit this trade-off through local reconstruction: after pruning, we adapt one subset of the model parameters at a time on a calibration set, training it to match the corresponding intermediate activations of the dense model. We evaluate local reconstruction across model families and scales, up to 72B parameters, and establish three main findings. First, local reconstruction is an effective adaptation mechanism for LLMs: it matches post-pruning retraining while using over an order of magnitude less data and compute, even when using PEFT techniques. Second, reconstruction exhibits a broad "free-lunch" regime in granularity, i.e., the reconstruction parameter window: as long as the reconstructed region contains at least a nonlinear submodule, final quality is largely insensitive to the window size, allowing granularity to be chosen primarily based on memory constraints. In contrast, reconstructing individual matrices, despite being the natural approach often proposed in the literature, consistently underperforms, as small matrix-level errors accumulate into larger activation drift. Lastly, reconstruction reduces the relative importance of the pruning criterion: performance gaps between sophisticated criteria and simple baselines shrink with model scale, making simple methods competitive again. Overall, our results challenge the prevailing view that post-pruning adaptation is impractical for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20923v1">Causal Past Logic for Runtime Verification of Distributed LLM Agent Workflows</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      Distributed LLM agent workflows should not be monitored as if they produced a single sequential log. In an asynchronous execution, a decision can only depend on events that are causally visible to the lifeline that makes it: an event that appears earlier in some log may still be unknown locally. We extend the ZipperGen agent-workflow framework with Causal Past Logic (CPL), a small past-time temporal logic for guards in conditionals and while loops. In addition to standard past-time modalities such as previous and since, a guard can inspect the latest causally visible event of another lifeline and selected variables stored there. The formula is a source-level guard: it is evaluated online by the owner lifeline and can influence control flow at runtime. We give a vector-clock monitor with latest-value views and prove that the locally computed monitor value coincides with the denotational semantics of the guard at the current event. Thus runtime verification becomes part of the coordination language itself, rather than a post-hoc check over an execution log.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.07731v2">Benchmarking EngGPT2-16B-A3B against Comparable Italian and International Open-source LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      This report benchmarks the performance of ENGINEERING Ingegneria Informatica S.p.A.'s EngGPT2MoE-16B-A3B LLM, a 16B parameter Mixture of Experts (MoE) model with 3B active parameters. Performance is investigated across a wide variety of representative benchmarks, and is compared against comparably-sized open-source MoE and dense models. In comparison with popular Italian models, namely FastwebMIIA-7B, Minerva-7B, Velvet-14B, and LLaMAntino-3-ANITA-8B, EngGPT2MoE-16B-A3B performs as well or better on international benchmarks: ARC-Challenge, GSM8K, AIME24, AIME25, MMLU, and HumanEval (HE). It achieves the best performance for the longest context setting (32k) of the RULER benchmark. On the Italian benchmark dataset ITALIC, the model performs as well or better than the other models except for Velvet-14B, which outperforms it. Compared with popular MoE models of comparable size, the new model reports higher values than DeepSeek-MoE-16B-Chat on all considered benchmarks. It has higher values than Moonlight-16B-A3B on HE, MMLU, AIME24, AIME25, GSM8K, and the 32k RULER setting, but lower on BFCL and some ARC and ITALIC settings. Finally it has lower values than GPT-OSS-20B on most benchmarks, including HE, MMLU, AIME24, AIME25, GSM8K, ARC, BFCL, and the RULER 32k. When compared with popular dense models, EngGPT2MoE-16B-A3B reports higher values on AIME24 and AIME25 than Llama-3.1-8B-Instruct, Gemma-3-12b-it, and Ministral-3-8BInstruct-2512-BF16, but lower values on ITALIC, BFCL, and RULER with a 32k context. When performance is aggregated across all benchmark metrics, EngGPT2MoE-16B-A3B shows higher performance than the Italian models under evaluation while achieving lower results than some of the most performant international models, in particular GPT-5 nano and Qwen3-8B. Taken together, our findings find the new model to be a step forward for native Italian Large Language Models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06139v2">Listwise Policy Optimization: Group-based RLVR as Target-Projection on the LLM Response Simplex</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Reinforcement learning with verifiable rewards (RLVR) has become a standard approach for large language models (LLMs) post-training to incentivize reasoning capacity. Among existing recipes, group-based policy gradient is prevalent, which samples a group of responses per prompt and updates the policy via group-relative advantage signals. This work reveals that these optimization strategies share a common geometric structure: each implicitly defines a target distribution on the response simplex and projects toward it via first-order approximation. Building on this insight, we propose Listwise Policy Optimization (LPO) to explicitly conduct the target-projection, which demystifies the implicit target by restricting the proximal RL objective to the response simplex, and then projects the policy via exact divergence minimization. This framework provides (i) monotonic improvement on the listwise objective with bounded, zero-sum, and self-correcting projection gradients, and (ii) flexibility in divergence selection with distinct structural properties through the decoupled projection step. On diverse reasoning tasks and LLM backbones, LPO consistently improves training performance over typical policy gradient baselines under matched targets, while intrinsically preserving optimization stability and response diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.01712v2">FT-Dojo: Towards Autonomous LLM Fine-Tuning with Language Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 26 pages, 6 figures, 11 tables
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models for vertical domains remains labor-intensive, requiring practitioners to curate data, configure training, and iteratively diagnose model behavior. Despite growing interest in autonomous machine learning and language agents, end-to-end LLM fine-tuning has not been systematically studied as an interactive agent task. We introduce FT-Dojo, an interactive benchmark environment for autonomous LLM fine-tuning, comprising 13 tasks across 5 domains. Rather than a new collection of static datasets, FT-Dojo standardizes a task interface, shared raw-data repository, sandboxed execution environment, structured feedback protocol, and held-out evaluation procedure. We further develop FT-Agent, a fine-tuning-oriented autonomous framework that uses structured iteration planning, fail-fast validation, and multi-level feedback analysis to refine data and training strategies. Experiments show that FT-Agent provides a strong initial baseline, achieving the best performance on 10 out of 13 tasks, with additional controlled comparisons against frontier agents, open-source planning backbones, and multi-run statistics supporting the main findings. Case studies show that agents can recover from failures through cumulative learning, while still exposing limitations in causal diagnosis and long-horizon planning. The implementation is available at https://github.com/microsoft/rd-agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.19075v3">Universal Reasoner: A Single, Composable Plug-and-Play Reasoner for Frozen LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 ICML 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable general capabilities, but enhancing skills such as reasoning often demands substantial computational resources and may compromise generalization. While Parameter-Efficient Fine-Tuning (PEFT) methods offer a more resource-conscious alternative, they typically require retraining for each LLM backbone due to architectural dependencies. To address these challenges, we propose Universal Reasoner (UniR)-a modular, composable, and plug-and-play reasoning module that can be used with larger frozen LLMs to provide specialized reasoning capabilities with a shared or aligned token space. Specifically, UniR decomposes the reward into a standalone reasoning module trained in a decoupled manner using verifiable rewards, effectively translating trajectory-level signals into token-level guidance. Once trained, UniR is combined with frozen LLMs at inference time by simply adding its output logits to those of the backbone. This additive structure enables modular composition: multiple UniR modules trained for different tasks can be jointly applied by summing their logits, enabling complex reasoning via composition. Furthermore, UniR demonstrates weak-to-strong generalization, where reasoning modules trained on smaller models effectively guide much larger LLMs in the same model family, and generalize across domains such as in vision language models and medical reasoning. Experiments on mathematical reasoning and machine translation show that UniR surpasses existing fine-tuning methods. Code is open-sourced at https://github.com/hangeol/UniR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.08636v2">InternBootcamp Technical Report: Boosting LLM Reasoning with Verifiable Task Scaling</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 InternBootcamp Tech Report
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have revolutionized artificial intelligence by enabling complex reasoning capabilities. While recent advancements in reinforcement learning (RL) have primarily focused on domain-specific reasoning tasks (e.g., mathematics or code generation), real-world reasoning scenarios often require models to handle diverse and complex environments that narrow-domain benchmarks cannot fully capture. To address this gap, we present InternBootcamp, an open-source framework comprising 1000+ domain-diverse task environments specifically designed for LLM reasoning research. Our codebase offers two key functionalities: (1) automated generation of unlimited training/testing cases with configurable difficulty levels, and (2) integrated verification modules for objective response evaluation. These features make InternBootcamp fundamental infrastructure for RL-based model optimization, synthetic data generation, and model evaluation. Although manually developing such a framework with enormous task coverage is extremely cumbersome, we accelerate the development procedure through an automated agent workflow supplemented by manual validation protocols, which enables the task scope to expand rapidly. % With these bootcamps, we further establish Bootcamp-EVAL, an automatically generated benchmark for comprehensive performance assessment. Evaluation reveals that frontier models still underperform in many reasoning tasks, while training with InternBootcamp provides an effective way to significantly improve performance, leading to our 32B model that achieves state-of-the-art results on Bootcamp-EVAL and excels on other established benchmarks. In particular, we validate that consistent performance gains come from including more training tasks, namely \textbf{task scaling}, over two orders of magnitude, offering a promising route towards capable reasoning generalist.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20863v1">PlexRL: Cluster-Level Orchestration of Serviceized LLM Execution for RLVR</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Reinforcement learning with verifiable rewards (RLVR) has recently unlocked strong reasoning capabilities in large language models (LLMs), triggering rapid exploration of new algorithms and data. However, RLVR training is notoriously inefficient: long-tailed rollouts, tool-induced stalls, and asymmetric resource requirements between rollout and training introduce substantial idle time that cannot be eliminated by job-local optimizations such as synchronous pipelining, asynchronous rollout, or colocated execution. We argue that this inefficiency is structural. While idle gaps are unavoidable within individual RLVR jobs, they are largely anti-correlated across jobs and therefore exploitable at the cluster level. Leveraging this observation, we present PlexRL, a cluster-level runtime for multiplexing unified LLM services across RLVR jobs. By centrally managing model placement, state transitions, and function-level scheduling under strict affinity constraints, PlexRL time-slices LLM execution across jobs to fill otherwise idle periods without expensive model migration. Our implementation and evaluations demonstrate that PlexRL significantly improves effective cluster capacity and reduces user GPU hour cost by maximum 37.58% while preserving algorithmic flexibility and introducing minimal per-job overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.18586v3">TokenCake: A KV-Cache-centric Serving Framework for LLM-based Multi-Agent Applications</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 14 pages, 17 figures, 3 tables, 2 algorithms
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in complex multi-agent applications that rely on external function calls. This workload creates severe performance challenges for the KV Cache: spatial contention leads to the eviction of critical agents' caches and temporal underutilization leaves the cache of agents stalled on long-running function calls idling in GPU memory. We present TokenCake, a KV-Cache-centric serving framework that bridges this gap by co-optimizing scheduling and memory management through an agent-aware design. TokenCake's Temporal Scheduler employs an event-driven, opportunistic policy to proactively offload idle KV Caches during function calls and uses predictive uploading to hide data transfer latency. TokenCake's Spatial Scheduler uses dynamic memory partitioning, guided by a hybrid priority metric combining graph structure and runtime state, to reserve GPU memory for critical-path agents. Our evaluation on representative multi-agent benchmarks shows that TokenCake reduces end-to-end latency by over 47.06% and improves effective GPU memory utilization by up to 16.9% compared to vLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.08292v5">Do LLMs Triage Like Clinicians? A Dynamic Study of Outpatient Referral</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Outpatient referral (OR) is a core clinical workflow that assigns patients to hospital departments under incomplete and evolving information, yet it is commonly simplified as a static classification problem despite being inherently interactive in practice. In this work, we study outpatient referral as a dynamic process driven by information acquisition and uncertainty reduction. We analyze both static scenarios based on fixed patient information and dynamic scenarios involving multi-turn dialogue, to test whether large language models (LLMs) improve referral outcomes through better prediction or more effective questioning. Our findings show that LLMs offer limited advantages over traditional classifiers in static referral accuracy, but consistently outperform them in dynamic settings by asking discriminative follow-up questions that reduce uncertainty over candidate departments. These results suggest that the primary value of LLMs in outpatient referral lies not in static prediction, but in supporting interactive, uncertainty-aware clinical decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20833v1">MemGym: a Long-Horizon Memory Environment for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Memory is a central capability for LLM agents operating across long-horizon tasks. Existing memory benchmarks predominantly evaluate retention of personalized information in multi-turn chat scenarios, overlooking the dynamic memory formation that occurs during extended agent execution. Consequently, the memory systems they produce transfer poorly to realistic agentic environments, such as coding and web navigation. We present MemGym, a benchmark for agentic memory that unifies existing agent gyms and in-house memory-grounded pipelines behind one memory-reasoning interface. MemGym spans five evaluation tracks grouped into four agentic regimes: tool-use dialogue (tau2-bench), multi-turn deep-research search (MEMGYM-DR), coding (SWE-Gym and MEMGYM-CODEQA), and computer use (WebArena-Infinity). MemGym reports memory-isolated scores that decouple memory performance from reasoning, retrieval, and tool-use ability, so memory strategies can be ranked without those confounders. Our synthetic pipelines for MEMGYM-CODEQA and MEMGYM-DR are length-controllable, ablation-verified at every stage, and tightly aligned with downstream scenarios. To make evaluation on coding environments academically tractable, we train MemRM, a lightweight reward model (Qwen3-1.7B fine-tuned with QLoRA) that scores compression quality as a fast scalar read in place of full Docker rollouts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.14654v2">Beyond Words: Multimodal LLM Knows When to Speak</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 Project page: https://github.com/lzk901372/MM-When2Speak
    </div>
    <details class="paper-abstract">
      Chatbots via large language models (LLMs) generate fluent responses but often struggle with when to speak, especially for brief, timely listener reactions during ongoing dialogue. We present a multimodal strategy for LLMs, which leverages synchronized video, audio, and text cues to improve conversational timing awareness. The strategy reformulates response timing as a dense response-type prediction task, enabling an agent to decide whether to remain silent, produce a short reaction, or start a full response under streaming constraints. Therefore, we introduce a curated multimodal dataset from real-world dyadic conversational videos with temporally aligned modalities and fine-grained reaction type annotations. Moreover, we design a multimodal strategy, MM-When2Speak, with a multimodal integration module on top of an LLM backbone. Experiments across various modality settings and strong LLM baselines show that MM-When2Speak achieves up to a 3x improvement in response type prediction performance, highlighting the importance of multimodal perception for natural and engaging conversational interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19537v2">The Silent Hyperparameter: Quantifying the Impact of Inference Backends on LLM Reproducibility</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Progress in LLMs is increasingly measured through standardized benchmarks, where state-of-the-art improvements are often separated by fractions of a percentage point. At the same time, the computational cost of evaluating modern LLMs has driven widespread adoption of specialized inference backends, software systems that execute trained models efficiently at inference time. While critical for scalability, system-level optimizations, such as custom CUDA kernels and reduced-precision arithmetic, can alter token probabilities and introduce non-determinism, possibly cascading into divergent generation. In this work, we first survey the inference landscape, identifying 200 distinct engines, and analyze 35,000 ML publications, finding that the specific inference stack is rarely reported despite this widespread diversity. We then present a systematic empirical study of how inference backends affect LLM benchmark results. Holding model weights, decoding parameters, and hardware constant, we evaluate five widely used inference engines, including vLLM, SGLang, and llama$.$cpp, across multiple open-weight models and established benchmarks. We show that the choice of backend alone can shift benchmark scores by up to 16.6 percentage points and induce high rates of output disagreement. By isolating backend optimizations and tracing the execution pipeline, we find this divergence is driven by system-level optimizations like prefix caching and CUDA graphs, custom kernels, and engine-specific defaults in logit processing. Our findings identify the inference backend as a previously unreported but consequential hyperparameter in the evaluation of LLM and advocate standardized reporting of inference stacks to improve the reproducibility and interpretability of benchmark comparisons.
    </details>
</div>
