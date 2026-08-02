# llm - 2026_07

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.20907v3">The Hidden Puppet Master: Predicting Human Belief Change in Manipulative LLM Dialogues</a></div>
    <div class="paper-meta">
      📅 2026-07-16
      | 💬 Accepted to COLM 2026
    </div>
    <details class="paper-abstract">
      As users increasingly turn to LLMs for practical and personal advice, they become vulnerable to subtle steering toward hidden incentives misaligned with their own interests. While existing NLP research has benchmarked manipulation detection, these efforts often rely on simulated debates and remain fundamentally decoupled from actual human belief shifts in real-world scenarios. We introduce PUPPET, a theoretical taxonomy and resource that bridges this gap by focusing on the moral direction of hidden incentives in everyday, advice-giving contexts. We provide an evaluation dataset of N=1,035 human-LLM interactions, where we measure users' belief shifts. Our analysis reveals a critical disconnect in current safety paradigms: while models can be trained to detect manipulative strategies, they do not correlate with the magnitude of resulting belief change. As such, we define the task of belief shift prediction and show that while state-of-the-art LLMs achieve moderate correlation (r=0.3-0.5), they exhibit systematic directional biases, with some models over-predicting and others under-predicting the magnitude of human belief change. This work establishes a theoretically grounded and behaviorally validated foundation for AI social safety efforts by studying incentive-driven manipulation in LLMs during everyday, practical user queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14957v1">Contextualized Early Detection of Online Firestorms: A Sequential LLM-Based Approach</a></div>
    <div class="paper-meta">
      📅 2026-07-16
      | 💬 9 pages, 2 figures, 7 tables, 1 algorithm
    </div>
    <details class="paper-abstract">
      Online firestorms are rapid collective escalations of highly negative user-generated content and may cause substantial reputational and economic damage. Existing detectors usually work with volume signals, sentiment scores, or predefined linguistic features. Such signals are useful, but they capture contextual meaning shifts in evolving discussion threads only indirectly. This paper proposes an LLM-based detection system with two operating modes. The first mode classifies complete Reddit threads retrospectively by combining local chunk-level assessments into a thread-level judgment. The second mode processes threads sequentially and issues early warnings when a sliding window exceeds calibrated thresholds. In this mode, the language model estimates three firestorm indicators: negativity share, escalation level, and contributor count. On a balanced Reddit dataset, the global mode achieves strong classification performance, while the early warning mode reaches high recall and detects escalating threads after only a small number of comments and distinct contributors. The results indicate that LLMs can be used not only for static judgment tasks, but also as repeated estimators in context-aware monitoring of social media discourse.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.03723v2">Segmenting Human-LLM Co-authored Text via Change Point Detection</a></div>
    <div class="paper-meta">
      📅 2026-07-16
    </div>
    <details class="paper-abstract">
      The rise of large language models (LLMs) has created an urgent need to distinguish between human-written and LLM-generated text to ensure authenticity and societal trust. Existing detectors typically provide a binary classification for an entire passage; however, this is insufficient for human--LLM co-authored text, where the objective is to localize specific segments authored by humans or LLMs. To bridge this gap, we propose algorithms to segment text into human- and LLM-authored pieces. Our key observation is that such a segmentation task is conceptually similar to classical change point detection in time-series analysis. Leveraging this analogy, we adapt change point detection to LLM-generated text detection, develop a weighted algorithm and a generalized algorithm to accommodate heterogeneous detection score variability, and establish the minimax optimality of our procedure. Empirically, we demonstrate the strong performance of our approach against a wide range of existing baselines. The python implementation of our proposal is available at https://github.com/Mamba413/DetectLLMSegmentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14905v1">Show Me How You Reason and I'll Tell You Who You Are: Reasoning Graphs for Robust LLM Authorship Attribution</a></div>
    <div class="paper-meta">
      📅 2026-07-16
    </div>
    <details class="paper-abstract">
      Given the current trend to employ large language models (LLMs) in almost any imaginable context, LLM-generated text detection and authorship attribution have become a pressing issue. Prior work has primarily focused on surface-level linguistic features, an approach shown to be susceptible to paraphrasing and other obfuscation techniques. In this paper, we go beyond the linguistic surface, extracting and analysing reasoning structures in LLM-generated texts with the goal of capturing more complex signals of LLM authorship. We propose a graph neural network approach that leverages reasoning graphs extracted by an argument mining pipeline, demonstrating improved robustness and generalisation over a traditional Longformer baseline. Our approach outperforms the baseline by up to 27 percentage points under the obfuscation attacks such as paraphrasing and backtranslation, and 19 percentage points when evaluated on the texts generated by the unseen model versions, simulating real-world conditions in which new LLM versions are continuously released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14896v1">StructureClaw: Traceable LLM Agents and an Executable Benchmark for Structural Engineering Workflows</a></div>
    <div class="paper-meta">
      📅 2026-07-16
      | 💬 21 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Addressing a structural-engineering request requires more than a single answer; it requires a chain of interdependent artifacts: interpreted requirements, a computable model, validation records, solver outputs, code-check records, and a final report. Evaluations centered on question answering or script generation rarely verify this complete evidence chain and may therefore reward fluent outputs even when the underlying engineering workflow is incomplete, internally inconsistent, or non-executable. To address this limitation, we present StructureClaw, an artifact-centered workbench in which LLM agents operate through governed engineering skills, typed tools, shared artifact state, and local analysis backends. We also introduce StructureClaw-Bench, an executable benchmark of 150 controlled scenarios spanning standard workflow execution, interactive robustness, and multimodal structural-model reconstruction. A scenario succeeds only when all required artifact- and execution-level assertions pass in a single run. Across ten agent-model configurations, each evaluated on the same 50 standard cases, the average Success Rate rises from 56.8% with the generic-skill baseline to 88.6% with the full automatic workflow. The interactive and multimodal evaluations identify two prominent remaining challenges: safe handling of invalid numerical inputs and fixture-consistent reconstruction of structural models. These findings show that artifact-centered evaluation can expose workflow-level failures that are difficult to identify from final responses alone, providing a more rigorous basis for evaluating and improving structural-engineering agents. The code and benchmark are available at https://github.com/structureclaw/structureclaw.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14835v1">LLM-Based Re-Ranking for Real Estate Search</a></div>
    <div class="paper-meta">
      📅 2026-07-16
    </div>
    <details class="paper-abstract">
      QuintoAndar Group operates the leading housing marketplace in Latin America for both rentals and sales. The platform replaces traditionally paper-heavy workflows with a fully digital experience, making housing transactions faster and more accessible to tenants, buyers, and landlords in the region. Finding the ideal home in such a vast catalog is inherently difficult. At the same time, the widespread adoption of conversational assistants is reshaping user expectations: people increasingly want to express their needs through open, multi-turn dialog rather than rigid filter menus and faceted search. This shift is particularly pronounced in housing, where intent is multi-dimensional, context-dependent, and rarely reducible to a small set of structured constraints. To meet these expectations, we propose a Large Language Model (LLM) based re-ranker that augments a conversational recommendation system by reordering retrieved candidates according to the nuanced, context-rich intent expressed across the user's conversation. We additionally construct a large-scale offline evaluation dataset for conversational real-estate search, containing 960,000 query-item pairs constructed from both synthetic and production queries and annotated using an LLM-as-a-Judge framework with human validation. We validate our approach both offline, on this proprietary dataset, and online, through a production A/B test. Both evaluations show consistent improvements in ranking quality, including a statistically significant increase in production of +5.3% in click-through rate and +4.8% in scheduled visits, demonstrating the value of integrating conversational context into housing recommendations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14830v1">AI Prototyper: A Figma Plugin for Decomposition-Based GUI Prototyping with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-16
    </div>
    <details class="paper-abstract">
      Graphical user interface (GUI) prototyping remains a time-consuming activity that demands both design expertise and considerable manual effort. As GUI prototypes are non-code artifacts that evolve alongside requirements throughout the development cycle, automating their generation is directly relevant to software maintenance and evolution. We present AI Prototyper, an open-source Figma plugin that automates GUI prototyping through a decomposition and retrieval-augmented generation (RAG) pipeline. Given a natural-language description of a desired screen, such as a login page or a product detail card, the plugin decomposes the request into discrete GUI features, retrieves matching components from a custom 32-primitive library, and renders each component as a fully editable Figma layer with auto-layout. The pipeline uses Gemini 2.5 Flash as its LLM back-end and a Node.js/Express service. Unlike existing decomposition-based tools, AI Prototyper introduces a human-in-the-loop editing step that lets users review, modify, or extend the generated feature list before rendering, uses a different technology stack and LLM family, and supports multilingual input, producing correctly labelled interfaces in Thai, English, and Mandarin Chinese. In a preliminary evaluation, participants using AI Prototyper completed more prototypes in a fixed time window than those working manually, and expert practitioners rated the AI-generated prototypes higher across nine quality dimensions. A demonstration video is available at https://youtu.be/pRoFAH7MQaE. The source code and component library are available at https://github.com/tongsalangsingha/AI-prototyper-tool
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14818v1">Can LLMs Build a MaxSAT Solver from Papers? The CoreForge Experience</a></div>
    <div class="paper-meta">
      📅 2026-07-16
    </div>
    <details class="paper-abstract">
      We report on CoreForge, an experience in using large language models (LLMs) to build an unweighted MaxSAT solver from research papers rather than from an existing solver codebase. The project focuses on unsatisfiability-based MaxSAT algorithms and follows an iterative workflow that combines paper discussions with ChatGPT, implementation through Codex prompts, and repeated LLM-assisted code audits and revisions. Although the codebase implements several algorithms and solver components, our evaluation focuses on configurations that combine core-guided optimization, lightweight preprocessing, core minimization, integration with integer linear optimization backends, and a new core-sequence lookahead approach. Our experience suggests that LLMs can support solver implementation from papers, while requiring external validation, benchmarking, and human guidance. In our experiments, fuzzing and MaxSAT Evaluation instances did not reveal wrong answers in the tested configurations, although performance remains below the best hand-engineered MaxSAT solvers. We summarize what worked, what remained difficult, and the lessons for future LLM-assisted solver development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18146v2">Think When Needed: Model-Aware Reasoning Routing for LLM-based Ranking</a></div>
    <div class="paper-meta">
      📅 2026-07-16
      | 💬 Accepted by SIGIR2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly applied to ranking tasks in retrieval and recommendation. Although reasoning prompting can enhance ranking utility, our preliminary exploration reveals that its benefits are inconsistent and come at a substantial computational cost, suggesting that when to reason is as crucial as how to reason. To address this issue, we propose a reasoning routing framework that employs a lightweight, plug-and-play router head to decide whether to use direct inference (Non-Think) or reasoning (Think) for each instance before generation. The router head relies solely on pre-generation signals: i) compact ranking-aware features (e.g., candidate dispersion) and ii) model-aware difficulty signals derived from a diagnostic checklist reflecting the model's estimated need for reasoning. By leveraging these features before generation, the router outputs a controllable token that determines whether to apply the Think mode. Furthermore, the router can adaptively select its operating policy along the validation Pareto frontier during deployment, enabling dynamic allocation of computational resources toward instances most likely to benefit from Think under varying system constraints. Experiments on three public ranking datasets with different scales of open-source LLMs show consistent improvements in ranking utility with reduced token consumption (e.g., +6.3\% NDCG@10 with -49.5\% tokens on MovieLens with Qwen3-4B), demonstrating reasoning routing as a practical solution to the accuracy-efficiency trade-off.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.30219v2">EvalSafetyGap: A Hybrid Survey and Conceptual Framework for LLM Evaluation-Safety Failures</a></div>
    <div class="paper-meta">
      📅 2026-07-16
      | 💬 This manuscript is a 90-page hybrid survey and conceptual framework on LLM evaluation and AI-safety failures. It includes 8 figures and multiple evidence-synthesis tables, covering literature from 2018 to 2026. The paper introduces the EvalSafetyGap framework and reports a structured audit of 10 LLMs. It is submitted as a review/survey article and is not currently under consideration elsewhere
    </div>
    <details class="paper-abstract">
      LLM evaluation and AI safety face a shared measurement problem: benchmark scores, reward-model signals, and reported safety metrics can improve while the latent properties they are meant to represent remain difficult to verify. This paper combines a hybrid survey - a systematic search paired with narrative synthesis and separately tracked grey evidence - with a conceptual framework and a structured ten-model audit. The synthesis spans eight evidence streams: benchmark validity, dynamic evaluation, LLM-as-judge reliability, safety evaluation, jailbreak/refusal robustness, reward hacking, mechanistic interpretability, and governance/auditability, covering 2018-2026 evaluation-safety measurement work. We introduce EvalSafetyGap as an organizing hypothesis for comparing evaluation-side and alignment-side proxy failures under optimization pressure, using Goodhart's Law together with two constructs we develop here - an Instability Decomposition and an Alignment Trilemma - as tools for generating testable comparisons. The audit shows how conclusions shift when capability, behavioral safety, and governance are measured separately. In this sample ($n = 10$), the association between capability and sustained adversarial robustness is statistically indeterminate using the displayed Table 3 inputs (Pearson $r = +0.232$, $p = 0.520$), and the apparent open-closed safety gap is modest, driven mainly by governance and disclosure rather than behavioral robustness, and sensitive to how a single borderline model is classified; attempt-budget results are protocol dependent. Because the public evidence uses heterogeneous protocols, the audit is diagnostic rather than rank-generating. The contribution is a shared vocabulary and evidence map to support dynamic evaluation, transparent source reporting, multi-attempt safety measurement, and auditable alignment practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08967v2">CluCERT: Certifying LLM Robustness via Clustering-Guided Denoising Smoothing</a></div>
    <div class="paper-meta">
      📅 2026-07-16
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have led to their widespread adoption in daily applications. Despite their impressive capabilities, they remain vulnerable to adversarial attacks, as even minor meaning-preserving changes such as synonym substitutions can lead to incorrect predictions. As a result, certifying the robustness of LLMs against such adversarial prompts is of vital importance. Existing approaches focused on word deletion or simple denoising strategies to achieve robustness certification. However, these methods face two critical limitations: (1) they yield loose robustness bounds due to the lack of semantic validation for perturbed outputs and (2) they suffer from high computational costs due to repeated sampling. To address these limitations, we propose CluCERT, a novel framework for certifying LLM robustness via clustering-guided denoising smoothing. Specifically, to achieve tighter certified bounds, we introduce a semantic clustering filter that reduces noisy samples and retains meaningful perturbations, supported by theoretical analysis. Furthermore, we enhance computational efficiency through two mechanisms: a refine module that extracts core semantics, and a fast synonym substitution strategy that accelerates the denoising process. Finally, we conduct extensive experiments on various downstream tasks and jailbreak defense scenarios. Experimental results demonstrate that our method outperforms existing certified approaches in both robustness bounds and computational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05782v2">When Models Meet Users An Empirical Study of Perceptions of General LLMs and Multimodal LLMs on Hugging Face</a></div>
    <div class="paper-meta">
      📅 2026-07-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have rapidly evolved from general-purpose systems to multimodal models capable of processing text, images, and audio. As both general-purpose LLMs (GLLMs) and multimodal LLMs (MLLMs) gain widespread adoption, understanding user perceptions in real-world settings becomes increasingly important. However, existing studies often rely on surveys or platform-specific data (e.g., Reddit or GitHub issues), which either constrain user feedback through predefined questions or overemphasize failure-driven, debugging-oriented discussions, thus failing to capture diverse, experience-driven, and cross-model user perspectives in practice. To address this issue, we conduct an empirical study of user discussions on Hugging Face, a major model hub with diverse models and active communities. We collect and manually annotate 662 discussion threads from 38 representative models (21 GLLMs and 17 MLLMs), and develop a three-level taxonomy to systematically characterize user concerns. Our analysis reveals that LLM access barriers, generation quality, and deployment and invocation complexity are the most prominent concerns, alongside issues such as documentation limitations and resource constraints. Based on these findings, we derive actionable implications for improving LLM ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14707v1">Harnessing LLMs for Reliable Academic Supervision: A Comparative Study</a></div>
    <div class="paper-meta">
      📅 2026-07-16
      | 💬 15 pages, 4 tables, 1 figure. Code and data available at https://github.com/AkashRajSingh/Harnessing-LLMs-for-Reliable-Academic-Supervision
    </div>
    <details class="paper-abstract">
      Large language models routinely produce fluent answers to single-shot prompts, yet deploying them as reliable components of a domain decision system is substantially harder. Closing this gap is the work of harness engineering: the deliberate composition of deterministic scaffolding (symbolic filters, retrieval, schema-typed I/O, LLM-as-judge loops, HITL gates, persistent state, audit trails) around an LLM core. We present a case study in academic supervision, a domain combining high-stakes recommendation, longitudinal accountability, and structured operational workflows. We compare a baseline (ASA), a GPT-5 chatbot with no scaffolding, against a multi-module system (ASuS) that wraps the much smaller GPT-4o-mini in a LangGraph harness with symbolic-semantic retrieval, schema-validated outputs, LLM-as-judge with bounded retry, HITL gates, deterministic weighted risk scoring with LLM narration, and a per-node SQLite audit trail. The evaluation rubric is retargeted at six harness-mechanism dimensions (grounding, explainability, consistency, process integrity, cognitive load, constraint adherence). A blind ten-rater hybrid evaluation, supplemented by a 2 x 2 model-harness ablation, finds that ASuS, despite using a much smaller base model, outscores ASA on every dimension. Across ten raters the pooled mean for ASuS is 4.08 versus 1.23 for ASA, and 8 of 10 raters reject the null at alpha = 0.05 on a paired Wilcoxon test; full numbers are in Sections 6.4 and 6.7. The ablation confirms that the structural contributions of the harness are largely model-invariant. We extract seven recurring harness-engineering patterns and argue that where reliability, traceability, and institutional consistency matter more than open-ended fluency, harness engineering challenges the prevailing 'bigger model is better' intuition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.10310v2">PolyInterview: An LLM-based Platform for Immersive Mock Interview Practice with Comprehensive Multimodal Assessment</a></div>
    <div class="paper-meta">
      📅 2026-07-16
      | 💬 10 pages, 7 figures, and 4 tables
    </div>
    <details class="paper-abstract">
      Preparing for job interviews is important for securing desired positions, yet realistic practice remains difficult to access: real interviews are infrequent, expert mock coaching is costly, and self-practice offers neither adaptive dialogue nor structured assessment. Existing systems typically address only parts of this need through fixed question sequences, limited communication channels, or feedback with little supporting evidence. We present PolyInterview, an LLM-based platform for immersive mock interview practice with comprehensive multimodal assessment. PolyInterview uses the target job description and CV to generate questions tailored to the role and candidate, conducts multi-turn spoken interviews with a lip-synced digital human interviewer that asks answer-aware follow-up questions, and evaluates response content, vocal delivery, and non-verbal behavior. Four parallel evaluators produce 13 behavior-level features that are aggregated into 10 assessment aspects and two competency tracks. Guided by the KSA and STAR frameworks, the report links each score to behavioral evidence and actionable recommendations. PolyInterview is publicly accessible. Its current all-account snapshot contains 101 accounts, 1,564 interview sessions, 7,665 generated questions, and 1,422 five-stage question sets. Generated questions are more closely aligned with their matched job description than with cross-role job descriptions in 93.7% of sessions. An evaluation by ten experts found strong question plans and actionable feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14659v1">LLM-Driven Approach to Modeling Tool Interoperability in Automotive Domain</a></div>
    <div class="paper-meta">
      📅 2026-07-16
    </div>
    <details class="paper-abstract">
      Interoperability between heterogeneous modeling tools remains a significant challenge in Model-Driven Engineering (MDE), particularly in the automotive domain where multiple modeling languages, as well as defacto standard proprietary and open-source tools coexist. This paper presents an LLM-driven approach for automated model interoperability by considering two relevant aspects: 1) mapping model instances to a target metamodel 2) merging of metamodels. The proposed methodology is demonstrated through transformations involving Ecore and SysML v2 based metamodels and incorporates structural validation of generated model instances against user-defined target models. Automotive case studies illustrate the feasibility of the approach and show that large language models can significantly reduce manual transformation effort while generating structurally valid target models for cross-tool interoperability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14642v1">MCPEvol-Bench: Benchmarking LLM Agent Performance Across Dynamic Evolutions of MCP Servers</a></div>
    <div class="paper-meta">
      📅 2026-07-16
    </div>
    <details class="paper-abstract">
      As Model Context Protocol (MCP) servers emerge as the core infrastructure for connecting LLMs with external tools, existing benchmarks leverage real-world MCP servers to evaluate LLM agents' tool-using capabilities. However, these benchmarks overlook the continuous evolution of tool interfaces and functionalities within MCP servers, resulting in flawed assessments that fail to capture the agent's adaptability in changing tool landscapes. To bridge this gap, we introduce \textbf{MCPEvol-Bench}, a novel benchmark for evaluating the task-solving capabilities of LLM agents under dynamic toolset evolution. Inspired by large-scale empirical study, we propose 11 mutation operators to simulate realistic tool evolution within 123 MCP servers. We benchmark 12 state-of-the-art LLMs on multiple versions of MCP servers, revealing that even frontier models struggle to adapt to evolving tools. For instance, GPT-5.4 and Claude-Sonnet-4-6 exhibit performance declines of 13.7\% and 14.4\% in evolved MCP servers, respectively, accompanied by substantial increases in planning and reasoning errors. These findings highlight the vulnerability of LLM-driven workflows, establishing MCPEvol-Bench as a standard for evaluating agent adaptability in dynamic tool environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11346v2">Compile, Then Page: Executable SOP Programs and a Capability-Gated Runtime for Procedural LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-16
      | 💬 9 pages, 3 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Enterprise agents must follow long-horizon, conditional, safety-critical standard operating procedures (SOPs). We compile machine-readable SOP constraints into executable pseudo-code and run them with a program-guided (PG) stack machine that pages the active frame while an LLM performs semantic execution. A three-arm SOPBench study across six models separates representation from runtime: compiled text never significantly hurts and gains up to 16.0 points where official prose underperforms. Runtime guidance is capability-gated. Two strong models independently show positive seven-domain PG contrasts (58:19 and 75:31 discordant pairs), whereas weak models are harmed. A full-program cursor ablation (active frame first, complete program retained) recovers much of the strong-model refusal gain; selective visibility adds a smaller improvement. Paired probe and audit measurements track this divide to spontaneous state discipline rather than reconstruction ability. On Bank the three primary arms rise from 70.4 to 86.4 to 92.8, with 100% refusal correctness. Practical guidance: compile first; enable active-frame paging only after a model-level discipline check.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11149v2">The Hidden Footprint: Making Storage a First-Class Metric for LLM Agent Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-07-16
      | 💬 17 pages, 5 figures; includes supplementary material
    </div>
    <details class="paper-abstract">
      LLM agent benchmarks measure task completion, reliability, and inference cost, but not the persistent data an agent run leaves on disk, including logs, context snapshots, checkpoints, and debug traces. We introduce AgentFootprint, a cross-framework benchmark of post-run agent storage footprint. Its serialization-aware metric suite measures total retention, channel composition, duplication, growth, compressibility, and conversation-history reconstructability. It addresses a measurement trap: naive byte-level measurement understates duplication by an order of magnitude because database paging and JSON escaping obscure repeated content. A fixed-trace control separates agent-generated logical volume from persistence-layer amplification: replaying the same trajectory through seven persisting frameworks yields a 6.7x spread. Under identical models, tools, and tasks, configurations with 100% accuracy differ by 15.7x in retained bytes, although their defaults support different recovery and audit capabilities. Three full-history configurations grow superlinearly on a repeated-observation stress task. Exported trajectories from 108 instance-normalized SWE-bench Verified submissions span three orders of magnitude per instance, with no detectable correlation with resolve rate. A content-addressed store reduces retention by 4.8x-32.7x while preserving every reconstructability score. These results establish persistent storage as a resource metric to report jointly with accuracy and reconstructability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14618v1">PolyQ: Codesigning End-to-End Quantization Framework for Scalable Edge CPU LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-07-16
      | 💬 Accepted to ICCAD 2026
    </div>
    <details class="paper-abstract">
      CPUs are the most universal target for on-device LLM inference, but existing low-bit quantization methods offer either coarse operating points or fine-grained mixed precision that is difficult to execute efficiently on CPUs. We present PolyQ, a CPU-oriented compiler/quantization co-design for activation-aware channel-wise bit allocation under a user-specified average-bit budget. PolyQ assigns per-channel bit-widths from $\{2,3,4,8,16\}$, then uses a compile-time model compiler to permute and cluster channels into bit-homogeneous blocks, generate SIMD- and LUT-compatible kernels, and merge compatible permutations across operators to keep layout regularization off the runtime path. This turns fine-grained budget fitting into a practical fractional-bit deployment method for CPU-only inference. Across Falcon-H1-3B, Llama2-13B, and Qwen3-32B on WikiText-2, PolyQ provides stable quality scaling from 3--6\,b and improves perplexity by 2.4--32.1\% over prior methods at a 3\,b target. End-to-end measurements on three representative CPUs -- workstation, laptop, and mobile -- show that compiler layout regularization reduces activation reorder traffic by up to 70.8\%, prefill latency and decode throughput scale nearly proportionally with the configured bit budget, and energy/token overhead stays below 2\% relative to an optimized LUT-based back-end. These results show that fractional-bit CPU deployment is practical, predictable, and energy-efficient across diverse edge targets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14605v1">Investigating first-language bias in LLM-based automated essay scoring: A cross-prompt evaluation of an open-weight AI-model on TOEFL essays</a></div>
    <div class="paper-meta">
      📅 2026-07-16
      | 💬 21 pages, 2 figures, and 8 tables
    </div>
    <details class="paper-abstract">
      This study examines the cross-prompt generalization and first-language (L1) scoring effects of a LoRA-adapted open-weight large language model (Gemma-3-27B-it) applied to automated essay scoring. Using the identical model and inference configuration reported in "AiAWE: An Open-Source LLM Automated Writing Evaluation System Using LoRA-Adapted Instruction-Tuned Models" (Gayed, 2026), which was fine-tuned on 480 argumentative essays from two prompts, we evaluate scoring accuracy on the full TOEFL11 corpus: 12,100 essays written by test-takers from 11 first-language backgrounds across eight prompts, none of which were seen during training. The model's raw scores (0.5-5.0) are mapped to the same three proficiency bands (low, medium, high) used by ETS, enabling direct comparison. The model achieved an overall band agreement of 77.79% and a quadratic weighted kappa of 0.702, with adjacent-band agreement of 99.98%. Accuracy was stable across all eight unseen prompts, with no advantage for prompts thematically related to the training data, indicating robust cross-prompt generalization. However, the model exhibited a systematic, L1-linked scoring offset. Within every proficiency band, essays from European-language backgrounds received consistently higher scores than essays from East-Asian-language backgrounds, a pattern not attributable to the composition of the fine-tuning data. This is the first large-scale L1 fairness analysis of a fine-tuned open-weight LLM for automated essay scoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14581v1">Multi-LLM Collaborative MRI Report Generation for Visual Instruction Tuning in Brain Oncology</a></div>
    <div class="paper-meta">
      📅 2026-07-16
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) and their extension to vision-language models (VLMs) have made it easier to combine text and images for tasks such as report generation. Existing VLMs in medicine typically focus on 2D images (chest X-rays), and their extension to 3D imaging has been difficult because of the lack of paired 3D imaging-text data. Thus, we introduce a new method for creating a 3D image-text dataset for brain oncology using 3D MRI scans of glioma and meningioma cases. We use a cooperative system in which several LLMs work together to generate and check reports, ensuring that they are accurate and clear. By leveraging the new 3D MRI-text dataset, we further build a VLM that converts MRI scans into tokens and aligns them with text instructions. Our VLM performed better in report generation and visual question answering tasks than other 2D and 3D methods. Our method not only improves the quality of reports but also helps with better diagnosis and treatment in brain oncology.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14574v1">Collaborative Spatial Learning with Multi-LLM Agents in Networked Social Experiments</a></div>
    <div class="paper-meta">
      📅 2026-07-16
      | 💬 Accepted at ASONAM 2026; to appear in the Springer proceedings
    </div>
    <details class="paper-abstract">
      Collective problem solving often requires that group members consider the tradeoff between exploitation of known solutions and exploration for new ones, where information of known solutions can be disseminated among individual members through communication networks. The Mason--Watts experiment (PNAS 2012) showed that human groups in shorter-path networks outperform those in longer-path networks on a two-dimensional search task. In this work, we focus on the investigation of such network-efficiency effects in the setting of a group of large language model (LLM) agents. Specifically, we consider groups of sixteen LLM agents playing the Mason--Watts experiment on the eight Mason--Watts network topologies. Moreover, we develop mechanistic Bayesian optimization agents such that the performance of LLM agents can be compared with both the mechanistic agents and the human experimental data. Our computational experiments indicate that the LLM agents show a significant network-efficiency effect when instructed to randomize their first-round choices, but not under the default initialization. In this experiment, adding a one-sentence first-round randomization instruction improves collective payoff by more than three times the estimated payoff difference across the eight network topologies. Also, the Bayesian optimization agents obtain higher payoffs than the evaluated LLM agents on this spatial search task. We further compare the agents' exploration--exploitation behavior, copying, and spatial diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14566v1">Fully Automated End-to-End Adversary Emulation from MITRE ATT\&CK Based Cyber Threat Intelligence Using LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-16
    </div>
    <details class="paper-abstract">
      This paper presents a fully automated end-to-end framework for adversary emulation from MITRE ATT&CK-aligned CTI reports using LLMs. Unlike prior work, which either executes prewritten playbooks or partially automates playbook generation, our framework unifies playbook generation, execution, and failure recovery in a single workflow. In particular, although AURORA, the most recent prior study, generates playbooks from CTI reports, it still requires partial manual intervention and does not revise playbooks based on execution failures. Our framework generates Caldera playbooks from CTI reports, executes them automatically, and revises failed Abilities through a failure-type-aware recovery mechanism. Evaluated on 11 CTI reports with Claude Sonnet 4.5, GPT-4o, Gemini 2.5 Pro, and Grok 4 Fast, the framework achieved its best results with Claude Sonnet 4.5: 27.3 Abilities per playbook, 84.22% execution success after revision, and CTI Precision, Recall, and F1 of 73.95%, 52.48%, and 60.50%, respectively. The failure recovery mechanism consistently improved execution success across all evaluated LLM models by 14.59%p to 17.23%p. On the 10 CTI reports selected from AURORA's dataset, this mechanism further increased the final execution success rate, surpassing that of AURORA, which represents the state-of-the-art adversary emulation system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.15727v3">AgentWorm: Self-Propagating Attacks Across LLM Agent Ecosystems</a></div>
    <div class="paper-meta">
      📅 2026-07-16
    </div>
    <details class="paper-abstract">
      Autonomous LLM-based agents increasingly operate as long-running processes forming densely interconnected multi-agent ecosystems, whose security properties remain largely unexplored. Systems such as OpenClaw, an open-source platform with over 40{,}000 active instances, persistent configurations, tool-execution privileges, and cross-platform messaging, are deployed at scale, yet the security of such agent ecosystems remains largely unexplored. This work presents AgentWorm, the first self-replicating worm attack against a production-scale agent framework, achieving a fully autonomous infection cycle initiated by a single message: the worm first hijacks the victim's core configuration to establish persistent presence across session restarts, then executes an arbitrary payload upon each reboot, and finally propagates itself to every newly encountered peer without further attacker intervention. The attack is evaluated on a controlled testbed across five distinct LLM backends, three infection vectors, and three payload types. Results show a 63\% aggregate attack success rate, sustained multi-hop propagation, and stark divergences in model security postures, highlighting that while execution-level filtering effectively mitigates dormant payloads, skill supply chains remain universally vulnerable. Defenses are evaluated at three layers (prompt-level mitigations sourced from real community practice, the framework's built-in security controls, and an ecosystem-wide measurement of public configurations), revealing that the critical controls capable of breaking the infection loop are not enabled in any of the observed deployments. A cross-framework transferability experiment on Hermes Agent confirms that the underlying vulnerabilities are properties of the autonomous agent design pattern, not artifacts of a single implementation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.25031v2">From Stateless to Situated: Building a Psychological World for LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-16
      | 💬 Accepted by WAICA 2026
    </div>
    <details class="paper-abstract">
      In psychological support and emotional companionship scenarios, the core limitation of large language models (LLMs) lies not merely in response quality, but in their reliance on local next-token prediction, which prevents them from maintaining the temporal continuity, stage awareness, and user consent boundaries required for multi-turn intervention. This stateless characteristic makes systems prone to premature advancement, stage misalignment, and boundary violations in continuous dialogue. To address this problem, we argue that the key challenge in process-oriented emotional support is not simply generating natural language, but constructing a sustainably updatable external situational structure for the model. We therefore propose LEKIA 2.0, a situated LLM architecture that separates the cognitive layer from the executive layer, thereby decoupling situational modeling from intervention execution. This design enables the system to maintain stable representations of the user's situation and consent boundaries throughout ongoing interaction. To evaluate this process-control capability, we further introduce a Static-to-Dynamic online evaluation protocol for multi-turn interaction. LEKIA achieved an average absolute improvement of approximately 31% over prompt-only baselines in deep intervention loop completion. The results suggest that an external situational structure is a key enabling condition for building stable, controllable, and situated emotional support systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14512v1">RetroAgent: Harnessing LLMs to Search Over Structured Memory for Agentic Retrosynthesis Planning</a></div>
    <div class="paper-meta">
      📅 2026-07-16
      | 💬 To appear at COLM 2026
    </div>
    <details class="paper-abstract">
      Multi-step retrosynthesis planning seeks to decompose a target molecule into commercially available building blocks through a sequence of feasible reactions. The vast combinatorial search space makes this task challenging even for expert chemists. Traditional methods combine tree search with offline-trained value networks that score candidates in isolation, without reasoning about complete multi-step routes. Recent work leverages Large Language Models (LLMs) for this task, but relies on simple interfaces that limit exploration of the full search space. We introduce RetroAgent, an LLM agent that bridges symbolic search and neural reasoning through a harness with structured memory. Through memory and chemistry tools, the agent observes the full search state, including explored routes, available alternatives, and properties of intermediates, enabling informed decisions grounded in both global progress and domain knowledge. Experiments on in-distribution and out-of-distribution benchmarks demonstrate that RetroAgent delivers strong performance and generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14493v1">Context Contamination in LLM Analysis of Network Security Logs: Poison with Passive Prompt Injection and Mitigation Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-07-16
    </div>
    <details class="paper-abstract">
      Large Language Models are increasingly deployed in Security Operations Centers for log analysis tasks including summarization, alert triage, and threat investigation. These systems ingest logs from external-facing services and process network logs as natural language contexts to generate security insights. We demonstrate that this architectural pattern introduces a critical vulnerability: adversaries can embed prompt injection payloads in log-generating fields that persist in storage and are executed when analysts query the LLM, achieving what we term passive prompt injection. We present LogInject, a systematic framework for evaluating these threats. Using LogInject-1.0, a benchmark of 12,847 log entries including 2,569 adversarial samples, we evaluate three production LLMs across four attack objectives: activity concealment, false positive generation, information exfiltration, and output hijacking. Our findings reveal an up to 88.2% attack success rate (83.4% average across models) under the baseline conditions. We introduce Context Stitching, a novel technique that fragments payloads across multiple log entries to evade stateless filters while exploiting LLM long-context reasoning, achieving a 76.4% success rate. As mitigation, we evaluate layered defenses by combining input filtering, prompt hardening, and output validation, demonstrating a 90.4% attack reduction, although 8.4% residual vulnerability persists. Our results establish that LLM-based log analysis creates an inherent confused deputy vulnerability where untrusted data and trusted instructions compete indistinguishably for model attention, requiring defense in-depth architectures and continued human oversight for security-critical decisions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14480v1">LLM Evaluators are Biased across Languages</a></div>
    <div class="paper-meta">
      📅 2026-07-16
    </div>
    <details class="paper-abstract">
      LLM evaluators (trained reward models and prompted LLM-as-a-Judge) are routinely validated via pairwise accuracy. In a multilingual setting, this operates under the premise that high pairwise accuracy implies reliable, language-neutral scoring. We show that this assumption does not hold. We conduct experiments with semantically identical instruction-response pairs across 23 languages, and find that multilingual evaluators assign significantly different scores to different evaluation languages. The bias is statistically significant and consistent across eight open-weight evaluators of different architectures and training paradigms, persists in frontier judges, and is strongly correlated with language resource level: lower-resource languages are scored more generously. Meanwhile, these biases are invisible to pairwise accuracy: evaluators achieve above 90% pairwise accuracy, yet have up to 43% difference in acceptance rate across languages under a global decision threshold, meaning, for instance, that harmful content in lower-resource languages is more likely to pass safety filters. Per-language thresholds would require language identification, which can be defeated by code-switched prompts. We then investigate why lower-resource languages receive higher rather than lower scores, and we find that model uncertainty is linked with the effect: models tend to give higher scores when less confident, both under negative log-likelihood and under token-free uncertainty measures; however, language identity remains a significant predictor after controlling for uncertainty, and the bias cannot be explained away by content difficulty alone, but is a structural, language-level misalignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12839v3">HeteroMosaic: Exposing and Exploiting Heterogeneous Execution Opportunities for Energy-Efficient Edge LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-07-16
      | 💬 Accepted at the 59th IEEE/ACM International Symposium on Microarchitecture (MICRO 2026)
    </div>
    <details class="paper-abstract">
      Modern edge system-on-chips (SoCs) combine CPUs, integrated GPUs (iGPUs), and neural processing units (NPUs), yet existing LLM runtimes typically make coarse device-level decisions or optimize operators in isolation. As a result, they underutilize heterogeneous resources, particularly on unified-memory platforms where performance depends on both device placement and task-graph coordination. We present HeteroMosaic, a heterogeneity-first scheduling framework for edge LLM inference. HeteroMosaic first uses a heterogeneous roofline model to identify when combining iGPU and NPU execution is beneficial. It then decomposes inference into dependency-preserving micro-batches that expose cross-accelerator overlap and applies trace-guided co-optimization of scheduling and device allocation under practical effects such as memory contention, DVFS, device variation, and NPU runtime overheads. We implement HeteroMosaic in PyTorch C++ and evaluate it on three AMD Ryzen AI platforms spanning NPU-heavy, balanced, and iGPU-heavy designs. On the balanced platform, HeteroMosaic achieves up to 1.73X speedup over an iGPU baseline, 1.78X over an NPU baseline, and 2.05X over frameworks such as llama dot cpp, while reducing energy by up to 45.3%. It also improves performance over prior heterogeneous edge AI solutions by up to 2.35X.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17471v2">PatchIsland: Orchestration of LLM Agents for Continuous Vulnerability Repair</a></div>
    <div class="paper-meta">
      📅 2026-07-16
    </div>
    <details class="paper-abstract">
      Continuous fuzzing platforms such as OSS-Fuzz uncover large numbers of vulnerabilities, yet the subsequent repair process remains largely manual. Unfortunately, existing Automated Vulnerability Repair (AVR) techniques -- including recent LLM-based systems -- are not directly applicable to continuous fuzzing. This is because these systems are designed and evaluated on a static, single-run benchmark setting, making them ill-suited for the diverse, noisy, and failure-prone environments in continuous fuzzing. To address these issues, we introduce PatchIsland, a system for Continuous Vulnerability Repair (CVR) that tightly integrates with continuous fuzzing pipelines. PatchIsland employs an ensemble of diverse LLM agents. By leveraging multiple LLM agents, PatchIsland can cover a wider range of settings (e.g., different projects, bug types, and programming languages) and also improve operational robustness. In addition, PatchIsland utilizes a two-phase patch-based deduplication to mitigate duplicate crashes and patches, which can be problematic in continuous fuzzing. In our internal evaluation, PatchIsland repaired 84 of 92 vulnerabilities, demonstrating strong repair capability. In the official AIxCC competition, the system operated with no human intervention in a fully autonomous environment and successfully patched 31 out of 43 vulnerabilities, achieving a repair rate of 72.1\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14456v1">Beyond Generalist LLMs: Specialist Agentic Systems for Structured Code Workflow Execution</a></div>
    <div class="paper-meta">
      📅 2026-07-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have accelerated the adoption of software development agents, now widely available as Integrated Development Environment (IDE) extensions and standalone applications. While these agents are typically general-purpose, it remains unclear whether specialist agents justify their additional development effort. We investigate this question in the context of business process automation, focusing on the transformation of Business Process Model and Notation (BPMN) diagrams into executable agentic workflows. Since BPMN specifies explicit control-flow semantics, we focus on deterministic workflows in which a fixed process model and inputs uniquely determine the executed path. We introduce a specialist workflow for this task and compare it against generalist agents such as Roo and Cline. Our results show that the specialist solution produces agents that outperform generalist baselines by approximately 9-20 percentage points in tool-use exactness, 2-4x in penalty-adjusted latency, and 3x fewer tool-call errors, while reducing generation token cost by over 95% and eliminating repair iterations. We also find that generalist agents generate code inconsistently in both functionality and quality, limiting their suitability for industrial settings where reliability and maintainability are essential.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26258v3">FlowBot: Inducing LLM Workflows with Bilevel Optimization and Textual Gradients</a></div>
    <div class="paper-meta">
      📅 2026-07-15
    </div>
    <details class="paper-abstract">
      LLM workflows, which coordinate structured calls to individual LLMs/agents to achieve a particular goal, offer a promising path towards building powerful AI systems that can tackle diverse tasks. However, existing approaches for building such workflows generally rely on human-crafted pipelines and prompts, which presents a substantial bottleneck in real world deployment. How can we automatically induce LLM-based agents and workflows in a data-driven way? This paper describes a simple data-driven approach for automatically inducing agents and LLM workflows. We formulate workflow induction as a bilevel optimization problem: an outer loop which optimizes a high-level sketch of the workflow (in particular how the LLM calls should be structured), and an inner loop which optimizes each individual LLM call one-by one. Both loops are optimized with ``textual gradients'' where for the inner loop we optimize each component in a modular way through ``backpropagating'' textual gradients layer-by-layer. We find that LLM workflows discovered through our \textsc{FlowBot} (work\textbf{flow} induction through \textbf{b}ilevel \textbf{o}ptimization and \textbf{t}extual gradients) approach performs competitively against strong baselines that make use of human-crafted or generated workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.01171v4">Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity</a></div>
    <div class="paper-meta">
      📅 2026-07-15
      | 💬 83 pages, 31 figures, 44 tables. Code is available at https://github.com/CHATS-lab/verbalize-sampling
    </div>
    <details class="paper-abstract">
      Post-training alignment often reduces LLM diversity, leading to a phenomenon known as mode collapse. Unlike prior work that attributes this effect to algorithmic limitations, we identify a fundamental, pervasive data-level driver: typicality bias in preference data, whereby annotators systematically favor familiar text as a result of well-established findings in cognitive psychology. We formalize this bias theoretically, verify it on preference datasets empirically, and show that it plays a central role in mode collapse. Motivated by this analysis, we introduce Verbalized Sampling, a simple, training-free prompting strategy to circumvent mode collapse. VS prompts the model to verbalize a probability distribution over a set of responses (e.g., "Generate 5 jokes about coffee and their corresponding probabilities"). Comprehensive experiments show that VS significantly improves performance across creative writing (poems, stories, jokes), dialogue simulation, open-ended QA, and synthetic data generation, without sacrificing factual accuracy and safety. For instance, in creative writing, VS increases diversity by 1.6-2.1x over direct prompting. We further observe an emergent trend that more capable models benefit more from VS. In sum, our work provides a new data-centric perspective on mode collapse and a practical inference-time remedy that helps unlock pre-trained generative diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14385v1">MamaBench: Benchmarking LLM Robustness in Maternal and Child Health Diagnosis through Counterfactual Clinical Perturbation</a></div>
    <div class="paper-meta">
      📅 2026-07-15
    </div>
    <details class="paper-abstract">
      Large language models achieve strong scores on medical benchmarks, yet these benchmarks evaluate each question in isolation, providing no measure of whether a system can distinguish clinically similar presentations requiring different interventions. We introduce MamaBench, the first counterfactual benchmark for maternal and paediatric AI: 434 expert-authored clinical narratives in 217 pairs across 371 pathologies, evaluated via the Bias Trap Rate (BTR), the conditional probability that a model fails the counterfactual given success on the base case. We propose Evidence-Anchored RAG (EA-RAG), a three-stage retrieval method that replaces aggregate similarity with an evidence coverage objective through clinical parameter extraction, coverage auditing, and contrastive sub-queries. Across eight configurations of four frontier LLMs, base accuracy overstates robust accuracy by 16-28 percentage points in every model. EA-RAG achieves 20.3% BTR and 65.0% robust accuracy on Claude Sonnet 4.6, a 5.5 percentage point BTR reduction without degrading base accuracy. The residual 20% BTR confirms that counterfactual robustness in clinical AI remains an open challenge. Keywords: counterfactual evaluation, clinical AI, maternal healthcare, retrieval-augmented generation, diagnostic robustness
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14371v1">Supervised Fine-Tuning vs. In-Context Learning: An Equilibrium Analysis of LLM Personalization under Congestion</a></div>
    <div class="paper-meta">
      📅 2026-07-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized AI services, but a critical tension emerges: while personalization improves model performance, it consumes scarce computational resources that users must share. When should a user invest in expensive Supervised Fine-Tuning (SFT) versus lightweight In-Context Learning (ICL)? How does congestion from other users' personalization choices reshape these incentives? And what strategies should platforms adopt when offering multiple personalization algorithms? We develop a tractable framework for LLM serving that captures the statistical-economic trade-offs users face. Our analysis yields several surprising insights. First, we show that ICL and SFT dominate in different regimes, determined by an interplay between pretraining coverage and data signal-to-noise ratios, but congestion can flip these rankings. Second, equilibrium resource consumption exhibits pronounced non-monotonicity: improving pretraining precision reduces the congestion, while broader pretraining coverage and harder tasks sometimes increase it. Third, we prove that offering both personalization methods never hurts the platform's maximal profits, despite potentially increasing computational load. Experiments with GPT-2 on linear regression tasks validate our theoretical predictions about algorithm performance. Complementing these results, our review of documentation from 21 major AI platforms shows that the share offering both SFT and ICL increased from 9.5% in 2021 to 71.4% in 2025, consistent with our platform-design implications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.17612v3">Provable Coordination for LLM Agents via Message Sequence Charts</a></div>
    <div class="paper-meta">
      📅 2026-07-15
      | 💬 42 pages; accepted at ISoLA 2026
    </div>
    <details class="paper-abstract">
      Multi-agent systems built on large language models (LLMs) are difficult to reason about. Coordination errors such as deadlocks or type-mismatched messages are often hard to detect through testing. We introduce a domain-specific language for specifying agent coordination based on message sequence charts (MSCs). The language separates message-passing structure from LLM calls, tool calls, and human control points, whose outcomes remain unpredictable. We define the syntax and semantics of the language and present a syntax-directed projection that generates deadlock-free local agent programs from global coordination specifications. We illustrate the approach with a diagnosis consensus protocol and show how coordination properties can be established independently of LLM nondeterminism. We also describe a runtime planning extension in which an LLM dynamically generates a coordination workflow for which the same structural guarantees apply. An open-source Python implementation of our framework is available as ZipperGen.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14306v1">Tracing LLM Behavior to the Training Data with Empirical Next-Token Distributions</a></div>
    <div class="paper-meta">
      📅 2026-07-15
    </div>
    <details class="paper-abstract">
      In this paper, we study the connection between an LLM's output distribution and the data used to train it. Specifically, we study the degree to which an LLM's next-token distribution agrees with the empirical next-token distribution (ENTD) given the context in the training data. The ENTD is an appealing target because it is the unrestricted global minimizer of the next-token cross entropy loss used for pretraining, as well as an easily interpretable function of the pretraining corpus. We find that for a significant fraction of inputs, the LLM's distribution agrees with the ENTD almost perfectly, and the average agreement increases with model scale and training compute. Nevertheless, there is a long tail of input sequences where the LLM and ENTD differ significantly, and we examine several possible sources of this discrepancy across the transformer architecture, training procedure, and finite-sample noise in the ENTD estimate itself. More broadly, we hope our findings will encourage more work on ``data-centric mechanistic interpretability,'' a complement to standard mechanistic interpretability that opens the black box of how model behaviors arise from the data, rather than how they are encoded in the learned weights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.07053v3">Animating Petascale Time-varying Data on Commodity Hardware with LLM-assisted Scripting</a></div>
    <div class="paper-meta">
      📅 2026-07-15
      | 💬 ©2026 IEEE. Personal use of this material is permitted. 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses. N.B. Due to the limitation "The abstract field cannot be longer than 1,920 characters", the abstract here is shorter than that in the original PDF file
    </div>
    <details class="paper-abstract">
      Scientists face significant visualization challenges as time-varying datasets grow in speed and volume, often requiring specialized infrastructure and expertise to handle massive datasets. Petascale climate models generated in NASA laboratories require a dedicated group of graphics and media experts and access to high-performance computing resources. Scientists may need to share scientific results with the community iteratively and quickly. However, the time-consuming trial-and-error process incurs significant data transfer overhead and far exceeds the time and resources allocated for typical post-analysis visualization tasks, disrupting the production workflow. Our paper introduces a user-friendly framework for creating 3D animations of petascale, time-varying data on a commodity workstation. Our contributions: (i) Generalized Animation Descriptor (GAD) with a keyframe-based adaptable abstraction for animation, (ii) efficient data access from cloud-hosted repositories to reduce data management overhead, (iii) tailored rendering system, and (iv) an LLM-assisted conversational interface as a scripting module to allow domain scientists with no visualization expertise to create animations of their region of interest. We demonstrate the framework's effectiveness with two case studies: first, by generating animations in which sampling criteria are specified based on prior knowledge, and second, by generating AI-assisted animations in which sampling parameters are derived from natural-language user prompts. In all cases, we use large-scale NASA climate-oceanographic datasets that exceed 1PB in size yet achieve a fast turnaround time of 1 minute to 2 hours. Users can generate a rough draft of the animation within minutes, then seamlessly incorporate as much high-resolution data as needed for the final version.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14285v1">ToolAlignBench: Investigating Alignment Conflicts in Tool-Calling Enabled LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-15
      | 💬 Accepted to the Pluralistic Alignment Workshop at ICML 2026
    </div>
    <details class="paper-abstract">
      Safety alignment in LLMs aims to align models with human values, but which values take precedence when they conflict? We investigate this question in the context of tool-calling LLM agents deployed in regulated industries, where agents processing confidential documents may encounter content that triggers safety-trained values (e.g., public welfare) that conflict with deployment-context instructions (e.g., internal logging). To empirically verify this phenomenon, we build a benchmark of 128 scenarios across 16 domains. We find that safety-aligned open-source models override their deployment instructions up to 43.4% of the time, engaging in whistleblowing, data exfiltration, and evidence tampering when processing documents that suggest organizational wrongdoing. We also find that abliteration reduces rates of external whistleblowing. These results reveal a fundamental tension in pluralistic alignment, where the same safety training that protects users can cause agents to act against deployment instructions in ways that create unpredictable liability risks. We release our benchmark as a framework to support evaluation of agent behavior under competing legitimate interests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.12429v2">Stabilizing Native Low-Rank LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2026-07-15
      | 💬 Accepted at ICML 2026
    </div>
    <details class="paper-abstract">
      Foundation models have achieved remarkable success, yet their growing parameter counts pose significant computational and memory challenges. Low-rank factorization offers a promising route to reduce training and inference costs, but the community lacks a stable recipe for training models from scratch using exclusively low-rank weights while matching the performance of the dense model. We demonstrate that Large Language Models (LLMs) can be trained from scratch using exclusively low-rank factorized weights for all non-embedding matrices without auxiliary "full-rank" guidance required by prior methods. While native low-rank training often suffers from instability and loss spikes, we identify uncontrolled growth in the spectral norm (largest singular value) of the weight matrix update as the dominant factor. To address this, we introduce Spectron: Spectral renormalization with orthogonalization, which dynamically bounds the resultant weight updates based on the current spectral norms of the factors. Our method enables stable, end-to-end factorized training with negligible overhead. Finally, we establish compute-optimal scaling laws for natively low-rank transformers, demonstrating predictable power-law behavior and improved inference efficiency relative to dense models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14277v1">Multi-Head Latent Control: A Unified Interface for LLM Agent Decision Making</a></div>
    <div class="paper-meta">
      📅 2026-07-15
    </div>
    <details class="paper-abstract">
      Large language models are increasingly deployed as agents, but reliable agentic behavior requires more than next-token prediction. At inference time, it is preferred that an agent can decide whether to proceed with its current reasoning, defer to a stronger model, request additional information, invoke external tools, or abstain under the given setup. Existing approaches address these decisions through prompt-level routing, external orchestration, or task-specific fine-tuning, which primarily rely on input-side signals, and are often costly and difficult to maintain as model backbones evolve. We ask whether such control decisions can be inferred directly from a model's latent generation process. We introduce Multi-Head Latent Control, a lightweight layer that reads hidden-state trajectories from a frozen LLM or VLM to produce deployment-time control signals. A Capability Head predicts whether the current model can solve the instance or should defer to a stronger collaborator, while a Resolution Head predicts appropriate resolution decision Clarification, Tool Use, Abstention, or Direct Answering. Both heads are trained only on latent traces from the same frozen LLM backbone, enabling post hoc adaptation without modifying the model. Across language and vision-language settings, Multi-Head Latent Control consistently improves the quality-cost tradeoff of multi-model systems, enabling early handoff from partial generations and more accurate intervention decisions. In routed execution (small + large model), it reduces large-model usage by up to 90.7 percent on AndroidWorld and 27-53 percent on average across benchmarks, while retaining most of large-model performance. Additionally, the learned control signals improve tool-use decision quality, yielding up to +158 percent relative score gain and 65.5 percent fewer missed-required tool calls.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14051v1">Hindcast: Replaying Prediction Markets to Evaluate LLM Forecasters</a></div>
    <div class="paper-meta">
      📅 2026-07-15
    </div>
    <details class="paper-abstract">
      Forecasters are evaluated by backtesting, which replays resolved questions and grades the probability the system would have assigned before the outcome was known. For LLMs, two channels leak the answer into this test. A model that retrieves can surface reports written after the event, turning forecasting into a lookup, and each new model is trained on data closer to the event, so a question that lay in the future for last year's models sits inside this year's training data. Either way, the test grades recall while claiming to grade foresight. We introduce Hindcast, which closes both leaks by grading a model as if it stood at a chosen past date $t_0$, before the outcome existed in either channel. Hindcast replays resolved Polymarket prediction markets against a frozen snapshot of public Reddit, lets the model read only posts written before $t_0$, and scores each forecast against both what happened and the market's own price at $t_0$, itself a human forecast made from the same past information. Because the cutoff is set per market and the snapshot never changes, the evaluation re-runs on new markets as models improve, without going stale. Once the leak is closed, retrieval still helps most models, but only where Reddit discussed the event beforehand. Where the archive carried only speculation, retrieval hurts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14045v1">LLMs for Qualitative and Mixed-Methods Social Network Analysis</a></div>
    <div class="paper-meta">
      📅 2026-07-15
    </div>
    <details class="paper-abstract">
      This manuscript explores the integration of Large Language Models (LLMs) into the field of qualitative and mixed-methods social network analysis (SNA). We argue that the primary focus of this integration should be on enhancing the depth and rigor of qualitative SNA, rather than on replacing human researchers with automated systems. We begin by outlining the core principles of qualitative and mixed-methods SNA, emphasizing the importance of understanding the meaning of ties, the role of narratives, and the significance of relational identities. We then discuss how LLMs can be used as powerful tools to augment this work, from assisting with data collection and coding to supporting theory-building and abductive reasoning. We also address the limitations and ethical challenges of using LLMs in this context, including issues of bias, hallucination, and the need for reflexivity. We conclude with a series of research designs and practical recommendations for researchers who want to integrate LLMs into their work in a thoughtful and responsible way.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14040v1">Can an Old Dog Be Taught New Tricks? Taking LLMs Beyond Sentence Level Translation</a></div>
    <div class="paper-meta">
      📅 2026-07-15
      | 💬 Accepted for publication in HCI International 2026, Late Breaking Papers Proceedings, Springer LNCS
    </div>
    <details class="paper-abstract">
      Automatic translation systems, from CAT tools to MT, overwhelmingly treat translation as a sentence-by-sentence act. This paper asks whether LLMs can be moved beyond that paradigm through whole-document, corpus-informed translation. We present PAT (Pragmatic Auto-Translator), a RAG-based system that pairs user-configured specifications with context from a comparable corpus of authentic longform texts in U.S. English and Latin American Spanish, passing retrieved paragraph-, section-, and document-level examples to an LLM for whole-document generation. The goal is draft translation for professional verification: target texts reformulated to fit their Spanish-language context, where discourse organization, rhetorical style, and pragmatic norms differ meaningfully from English. We evaluated six automatic translations of essays on generative AI across three projects using a customized MQM typology, assessed by two trained evaluators working from U.S. English into LATAM and Mexican Spanish. Results show that a limited prompt produced no meaningful reformulation, and specifications and corpus-informed translations at times showed substantial reformulation, though not always to effect. We find that LLMs can be moved toward reformulation and away from the sentence-by-sentence paradigm, though more work is needed to improve the effectiveness of those reformulations. In this paper, we discuss considerations related to automatic translation system design, corpus construction, and translation quality evaluation methodology and results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14026v1">From Forecasts to Auditable Reports: Evidence Contracts for LLM-Assisted Housing-Guarantee Risk Monitoring</a></div>
    <div class="paper-meta">
      📅 2026-07-15
    </div>
    <details class="paper-abstract">
      Translating next-month housing-guarantee risk forecasts into auditable operational reports is essential yet challenging because upper-tail events are sparse, source records are confidential, and generated narratives can distort the underlying evidence. Using monthly South Korean \textit{jeonse} deposit guarantee data from September 2015 to December 2025, we introduce an evidence-constrained reporting pipeline that prioritizes upper-tail monitoring, retrieves historical precedents aligned with the forecasting rationale, organizes admissible information into typed evidence contracts, and verifies generated claims before analyst review. We train and select the forecasting backbone on the original panel, whereas the reporting experiments use synthetic aggregate scenarios calibrated to its empirical ranges and temporal structure. The selected forecasting model substantially improves high-risk detection while retaining competitive average error. Across eight LLMs, structured evidence consistently increases report quality, numerical fidelity, and claim-level grounding. A practitioner evaluation involving 51 analysts and related domain professionals further indicates that the reports support real-world review and decision-making: most participants rated them as practically useful and endorsed an operational pilot. These findings demonstrate that reliable LLM-assisted reporting requires predictive models to be coupled with structured evidence, explicit verification, and analyst oversight.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14197v1">How Artificial Intelligence LLM Engines Shape the Global Conflict Information Environment</a></div>
    <div class="paper-meta">
      📅 2026-07-15
    </div>
    <details class="paper-abstract">
      Artificial Intelligence (AI) answer engines now field a growing share of the questions that analysts, scholars, and the public ask about issues of peace and conflict. Large Language Models (LLMs) are known to hallucinate under certain conditions, but do these errors have discernible patterns when they are asked about conflicts, and if so what can that teach us about the changing global conflict information environment? To answer, we first asked a battery of questions about 28 conflicts to five leading answer engines and scored their 5,460 answers against documented evidence. We found that the thinner the retrievable record around a given conflict, the more the engines invent, misattribute, and miscount. Thin records don't just encourage hallucination, but create structural exposure to mis- and disinformation, because they are the easiest records to warp through Generative Engine Optimization (GEO) to bias engine responses. Through an analysis of 1,048 websites that the AI LLMs pulled conflict facts from, we found that GEO source optimization is already happening, and while state-partisan digital capture remains incipient it is rapidly growing. We explain what these findings mean for scholarship with the rise of GEO information warfare, and for policy argue for a return to the deep local monitoring and translation-based research that AI tools cannot replicate, closing with a discussion of future research opportunities and challenges in this fast-moving space.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12198v2">LLM-Based User Personas for Recommendations at Scale</a></div>
    <div class="paper-meta">
      📅 2026-07-15
      | 💬 Accepted by 2026 RecSys Industry Track
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) offer unprecedented potential for enhancing recommendation systems through their world knowledge and reasoning capabilities. However, existing approaches often rely on structured IDs or offline processing, limiting semantic richness, real-time adaptability, and user-facing interpretability. In this paper, we introduce a novel framework that enables real-time generation of LLM-based user interest personas for a large-scale commercial video recommendation platform. Our method generates natural-language user interest personas that address the exploitation-exploration trade-off by combining the summarization of existing interests with novel topics, directly during serving. To overcome the computational challenges of online LLM inference at a billion-user scale, we design a cost-efficient architecture leveraging knowledge distillation, asynchronous inference, and input optimization via semantically clustered video representations. Extensive offline evaluations, user studies, and live A/B tests demonstrate significant improvements in viewer value. This work bridges the gap between high-level semantic understanding and industrial-scale recommendation, paving the way for more dynamic, explainable, and satisfying personalized experiences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.08842v2">L2-Bench: An Evaluation Benchmark for Measuring LLM Capabilities in Second Language Education</a></div>
    <div class="paper-meta">
      📅 2026-07-15
    </div>
    <details class="paper-abstract">
      Despite rapid AI adoption in education, rigorous evaluation of AI-powered educational (AIED) systems remains critically underdeveloped, particularly in second language (L2) education, one of the most common yet least evaluated AI applications. We introduce L2-Bench, an open-source benchmark of 1,000+ task-response pairs to aid the pedagogy-led evaluation of LLM capabilities relating to language learning and assessment. Crucially, L2-Bench measures model performativity on the application of learning experience design principles rather than mere knowledge of those principles or broad learning outcomes. Our contributions include: (1) a validated taxonomy of 12 competencies and 31 subcompetencies validated by 200+ expert practitioners (task authenticity: 4.42/5.00, criteria adequacy: 4.18/5.00); (2) a rubric-based evaluation methodology that we believe can, if adapted, generalize to similar (open-ended, qualitative) disciplines; (3) an evaluation dataset that produces reliable signal about model strengths, weaknesses, and contextual robustness across diverse L2 education scenarios. We find that, among large models, Claude Opus 4.7 performs best overall (85.5%), though is marginally outperformed on several constituent tasks. We also find that performance drops notably on harder tasks (69.9% to 73.4%). L2-Bench provides education stakeholders better methods to make more informed decisions about real-world AIED adoption, use, and governance, while advancing the maturing science of AI evaluations for education.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.13918v1">Partially Correlated Verifier Cascades in LLM Harnesses: Concave Log-Odds, Polynomial Reliability, and Blind-Spot Ceilings</a></div>
    <div class="paper-meta">
      📅 2026-07-15
      | 💬 14 pages, 2 figures. Code and synthetic-recovery experiments: https://github.com/jianganghan/harness-verifier-cascades
    </div>
    <details class="paper-abstract">
      Serial verification gates are a core reliability primitive in LLM harnesses: a candidate answer is returned only if $k$ verifier calls all accept it. Under conditionally independent gates, the recent Odds Law (arXiv:2606.15712) shows that posterior log-odds grow linearly in $k$, so failure decays exponentially, and states that "a tight theory of partially correlated verifier cascades remains open." This note gives a minimal such theory. Modeling the per-instance false-accept rate on the generator's own errors as a latent variable $α\sim G$ (de Finetti), the exact cascade posterior is $\ell_k = \ell_0 - \ln m_k$, with $m_k$ the $k$-th moment of $G$. Then: (i) $\ell_k$ is concave in $k$ for every non-degenerate $G$ -- the Odds Law is its tangent at the first gate and an upper bound; (ii) for Beta$(a,b)$ latents, failure decays polynomially, $1-r_k \asymp k^{-b}$, with correlation parameter $ρ_v = 1/(a+b+1)$; (iii) a blind-spot atom of mass $1-π$ at $α=1$ caps the evidence extractable from any number of gates at $-\ln(1-π)$ nats, so reliability saturates below 1; (iv) letting the true-accept rate also vary ($β\sim H$) yields a trichotomy -- gates eventually always help, plateau, or actively harm -- decided by the upper-tail exponents of $G$ and $H$, with closed-form crossover $k^\dagger$. The mechanism is survivorship: errors surviving gates are the high-$α$ ones. The theory is measurable: $R$ repeated verdicts per instance identify the first $R$ moments of $G$, so two verdicts identify $ρ_v$; beta-binomial likelihood and NPMLE recover the reliability curve and the ill-posed ceiling. In synthetic tests, independence-based extrapolation underestimates failure by 20x at $k=5$ and ~3000x at $k=10$; the correlated fit at $R=8$ tracks held-out depths. The practical lever is decorrelation -- changing model family, modality, or evidence source -- not adding gates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.13911v1">How to Guide LLM Generation: Dual-Surrogate Guided Search for Automated Heuristic Design</a></div>
    <div class="paper-meta">
      📅 2026-07-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have made automated heuristic design (AHD) increasingly practical by generating executable heuristic code from task descriptions and evaluator feedback. Yet under a limited query and evaluation budget, search efficiency depends critically on a pre-generation decision. Before each LLM query and black-box evaluation, the system must choose which archived heuristics to reuse as parents and which generation operator should transform them. Existing methods typically choose such actions with predefined rules, leaving the expected outcome of each concrete operator-parent action only indirectly modeled. Therefore, we propose \emph{\fullmethod{}} (\method{}), a surrogate-guided action-selection module for operator-parent selection in LLM-based AHD. \method{} guides the LLM code-generation process by scoring pre-generation actions with two complementary surrogates. Specifically, a transition surrogate is proposed to predict the latent distribution of the child representation induced by an operator-parent action, while an instance-conditioned utility surrogate is proposed to estimate the expected performance of sampled child latents. Moreover, we propose an uncertainty-aware acquisition rule that combines predicted utility, utility uncertainty, and transition uncertainty to select the next LLM generation action. Across a diverse heuristic-design suite, \method{} is competitive with strong LLM-AHD baselines, and ablation and action-selection analyses suggest that its behavior goes beyond simple archive ranking or fixed operator preferences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.07323v2">Discovering Ordinary Differential Equations with LLM-Based Qualitative and Quantitative Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-07-15
      | 💬 Accepted at ICML 2026
    </div>
    <details class="paper-abstract">
      Discovering governing differential equations from observational data is a fundamental challenge in scientific machine learning. Existing symbolic regression approaches rely primarily on quantitative metrics; however, real-world differential equation modeling also requires incorporating domain knowledge to ensure physical plausibility. To address this gap, we propose DoLQ, a method for discovering ordinary differential equations with LLM-based qualitative and quantitative evaluation. DoLQ employs a multi-agent architecture: a Sampler Agent proposes dynamic system candidates, a Parameter Optimizer refines equations for accuracy, and a Scientist Agent leverages an LLM to conduct both qualitative and quantitative evaluations and synthesize their results to iteratively guide the search. Experiments on multi-dimensional ordinary differential equation benchmarks demonstrate that DoLQ achieves superior performance compared to existing methods, not only attaining higher success rates but also more accurately recovering the correct symbolic terms of ground truth equations. Our code is available at https://github.com/Bon99yun/DoLQ.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.11076v2">SmartSwap: Swap-Based Memory Optimization for LLM Training under Varying Operator Sequences</a></div>
    <div class="paper-meta">
      📅 2026-07-15
      | 💬 Accepted to DAC 2026. Previously titled "Chameleon: Taming Dynamic Operator Sequences for Memory-Intensive LLM Training."
    </div>
    <details class="paper-abstract">
      The increasing size of large language models (LLMs) has led to a surge in memory requirements during training, often exceeding the capacity of high-bandwidth memory (HBM). Swap-based memory optimization incurs neither accuracy loss nor additional end-to-end overhead when effectively overlapped, thus being an attractive solution. However, existing swap methods assume consistent operator sequences, which is impractical in Eager Mode, where operator sequences can vary during change. We propose Chameleon, which redesigns the end-to-end process of swap-based memory optimization and is the first work to consider varying operator sequences in Eager Mode. Chameleon (i) introduces a lightweight online profiler to enable continuous profiling for monitoring operator sequences, (ii) generates effective swap policies with limited operator information, and (iii) optimizes the policy execution module for accurate policy application and better performance. Experimental results demonstrate that Chameleon reduces profiling overhead by 84.25%, enables training models up to 4x larger than hardware memory while adapting to changes in operator sequences, improves performance by up to 38.94% compared to recomputation or high-degree parallelism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.13801v1">Traffic-Aware Randomized Smoothing for LLM-Based Network Intrusion Detection</a></div>
    <div class="paper-meta">
      📅 2026-07-15
      | 💬 44 pages, 14 figures, 14 tables. Submitted to Expert Systems with Applications
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based intrusion detection systems (IDS) are increasingly studied for security monitoring, yet their robustness against feasible traffic manipulation remains largely empirical. We present Traffic-Aware Randomized Smoothing (TA-RS), a classifier-agnostic certified defense that injects Gaussian noise exclusively into the directly controllable (DC) subspace -- features a remote attacker can modify -- during both fine-tuning and certification, aligning the smoothing distribution with the attacker-controllable subspace. We identify a critical prerequisite: applying standard randomized smoothing to clean-trained LLM-IDS yields weak certified accuracy in three of four (model, dataset) pairs tested (14-33%, at or below random) and only 57% in the fourth (43 pp below the noise-augmented result); noise-augmented fine-tuning recovers to 68-100% on two of three benchmark datasets (at sigma=0.25). At the L_inf-equivalent threshold R_inf = epsilon*sqrt(|DC|) (epsilon=0.05), TA-RS achieves 55-100% certified accuracy on CIC-IDS-2018 and HIKARI-2021, with median certified radii (R approx 0.45-0.96) exceeding R_inf by 1.8-5x (across sigma=0.25-1.00). Against a fairly trained iso-trained RS baseline the residual advantage is dataset-dependent (4-19 pp on CIC-IDS-2018). The larger gap -- up to 72 pp against an isotropic RS baseline that shares the DC-noise-augmented training recipe -- primarily reflects the training-certification mismatch rather than DC alignment alone: isotropic test-time noise perturbs uncontrollable features the attacker cannot exploit, triggering abstention rates up to 68%. RT-IoT2022 probes the limits of the method: it fails under the default fine-tuning recipe but recovers to 76%/69% certified accuracy (LLaMA3-8B/Qwen3-8B) when noise augmentation is increased.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.28213v2">\textit{Versteasch du mi?} Computational and Socio-Linguistic Perspectives on GenAI, LLMs, and Non-Standard Language</a></div>
    <div class="paper-meta">
      📅 2026-07-15
    </div>
    <details class="paper-abstract">
      The design of Large Language Models (LLMs) and generative artificial intelligence (GenAI) has been shown to be "unfair" to less-spoken languages (Petrov et al., 2023) and to deepen the digital language divide (Bella et al., 2023). Critical sociolinguistic work has also argued that these technologies are not only made possible by prior sociohistorical processes of linguistic standardisation, often grounded in European nationalist and colonial projects (Migge and Schneider, 2025), but also exacerbate epistemologies of language as "monolithic, monolingual, syntactically standardized systems of meaning" (Schneider, 2024, p. 5). In our paper, we draw on earlier work on the intersections of technology and language policy (Kelly-Holmes, 2019) and bring our respective expertise in critical sociolinguistics and computational linguistics to bear on an interrogation of these arguments. We take two different complexes of non-standard linguistic varieties in our respective repertoires-South Tyrolean dialects, which are widely used in informal communication in South Tyrol, Italy (Alber et al., 2024), as well as varieties of Kurdish-as starting points to an interdisciplinary exploration of the intersections between GenAI and linguistic variation and standardisation. We discuss both how LLMs can be made to deal with non-standard language from a technical perspective, and whether, when or how this can contribute to "democratic and decolonial digital and machine learning strategies" (Migge and Schneider, 2025, p. 12), which has direct policy implications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.13952v3">LLM-Guided Reinforcement Learning for Audio-Visual Speech Enhancement</a></div>
    <div class="paper-meta">
      📅 2026-07-15
      | 💬 6 pages, 4 figures, Accepted by Interspeech 2026
    </div>
    <details class="paper-abstract">
      In existing Audio-Visual Speech Enhancement (AVSE) methods, objectives such as Scale-Invariant Signal-to-Noise Ratio (SI-SNR) and Mean Squared Error (MSE) are widely used; however, their correlation with perceived speech quality is often suboptimal and provides limited interpretability for optimization. This work proposes a reinforcement learning-based AVSE framework with a Large Language Model (LLM)-based interpretable reward model. An audio LLM generates natural language descriptions of enhanced speech, which are converted by a sentiment analysis model into a 1-5 rating score serving as the PPO reward for fine-tuning a pretrained AVSE model. Compared with scalar metrics, LLM-generated feedback is semantically rich and explicitly describes speech quality improvements. Experiments on the AVSEC-4 dataset show that the proposed method outperforms a supervised baseline and a DNSMOS-based RL baseline in PESQ, STOI, neural quality metrics, and subjective listening tests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.13707v1">The Test Oracle Problem in Synthetic LLM-as-Judge Corpora: Disappearance, Distortion and a Validation Protocol</a></div>
    <div class="paper-meta">
      📅 2026-07-15
      | 💬 23 pages, 1 figure, 3 tables
    </div>
    <details class="paper-abstract">
      Studies of bias in LLM-as-judge systems typically build synthetic corpora by prompting an LLM to generate a hallucinated answer to pair with a factual one, then presenting both to a judge. We report a case in which this generation step silently failed, and use it to argue that the failure mode is structural rather than incidental. In a multilingual (Turkish/English) faithfulness-judgment corpus, a decoding-budget parameter shared between judging and generation calls truncated one producer's hallucinated answers to a few words. The resulting items produced a large, statistically robust effect: a 32-point cross-lingual collapse in one judge's selection accuracy, replicated from N=50 to N=500, explained by a three-layer mechanistic account, and confirmed by a controlled producer-swap experiment, none of which was real. The effect vanished to ceiling once the shared parameter was corrected, and only manual reading of the raw generations, not any aggregate statistical check, exposed the fault. A second measured bias (Markdown-formatting preference) was not fabricated but distorted by the same fault, its magnitude and in one case its sign shifting with stimulus length, a mode aggregate metrics cannot distinguish from the first. We frame the underlying vulnerability using the test oracle problem: corpora whose negative examples are LLM-generated carry no mechanical way to verify item integrity, while corpora built by deterministic perturbation of a gold answer carry an item-level oracle for free. A positive control supports this claim directly: an analogous fault injected into a minimal perturbation-based corpus is caught with 100% accuracy by a zero-cost, zero-human gold-to-negative string comparison. We close with a validation protocol, derived from our own case, for analysts working in the oracle-less regime that we argue describes most contemporary multilingual LLM-as-judge corpora.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.29581v2">The Joint Effect of Quantization and Sampling Temperature on LLM Safety Alignment: A Factorial Analysis</a></div>
    <div class="paper-meta">
      📅 2026-07-15
      | 💬 11 pages, 5 Figures
    </div>
    <details class="paper-abstract">
      Modern LLM deployments often combine quantization with higher sampling temperatures to reduce cost, latency, or repetition, yet safety evaluations usually treat these as fixed implementation details. We test whether models that are safe at FP16 with greedy decoding remain safe after quantization and stochastic sampling, or whether the two factors amplify each other. We evaluate 8 instruction-tuned models from five families across 3 precisions and 6 temperatures, covering 144 configurations on 7 harmfulness benchmarks and generating about 2.0 million responses, which are scored by a six-judge safety ensemble. Contrary to concerns that low-bit deployment erodes alignment, we find that standard quantization is approximately safety-neutral: for 7 of 8 models, AWQ INT4 keeps attack success within about 1.6 percentage points of FP16 or lowers it, with clear degradation only for SmolLM3-3B (34.5% to 44.1%). However, the larger risk comes from sampling: higher temperatures sharply increase decision instability, with DFR reaching 41.9% at T = 1.0, even when average ASR changes only modestly. The two factors do not compound: our Compound Degradation Index remains sub-additive (-0.071 to +0.018), indicating that quantization partially offsets rather than amplifies temperature-induced degradation. Finally, a per-benchmark breakdown shows that single-benchmark evaluation badly understates risk: several models scoring 0% on AdvBench exceed 80% on ManyHarm. Standard INT4/INT8 quantization can therefore be reasonable for well-aligned models, but safety claims should report multi-sample stability across multiple benchmarks rather than rely on a single benchmark at greedy decoding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.13649v1">CIMERA: Compute-in-Interconnect and Memory with Reconfigurable Precision for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-07-15
      | 💬 Accepted to 2026 IEEE 8th International Conference on Artificial Intelligence Circuits and Systems (AICAS'26)
    </div>
    <details class="paper-abstract">
      LLM impose significant computational and memory demands, creating challenges for energy-efficient inference across platforms ranging from data centers to power-constrained edge devices. Weight precision plays a critical role in balancing inference accuracy, throughput, and energy consumption, while modern LLM workloads exhibit pronounced heterogeneity and tolerance that favors adaptive precision execution. This paper presents CIMERA, a reconfigurable-precision LLM inference accelerator that integrates compute-in-interconnect and memory to mitigate the memory wall and enable precision-aware execution. Compared to Nvidia H100, CIMERA delivers up to $25\times$ and $10\times$ higher energy efficiency for 1B and 13B models, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20379v2">Policy of Thoughts: Scaling Test-Time Training for LLM Reasoning via Online Policy Evolution</a></div>
    <div class="paper-meta">
      📅 2026-07-15
      | 💬 Under Review, preprint
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) struggle with complex, long-horizon reasoning due to instability caused by their frozen policy assumption. Current test-time scaling methods treat execution feedback merely as an external signal for filtering or rewriting trajectories, without internalizing it to improve the underlying reasoning strategy. Inspired by Popper's epistemology of "conjectures and refutations," we argue that intelligence requires real-time evolution of the model's policy through learning from failed attempts. We introduce Policy of Thoughts (PoT), a framework that recasts reasoning as a within-instance online optimization process. PoT first generates diverse candidate solutions via an efficient exploration mechanism, then uses Group Relative Policy Optimization (GRPO) to update a transient LoRA adapter based on execution feedback. This closed-loop design enables dynamic, instance-specific refinement of the model's reasoning priors. Experiments show that PoT dramatically boosts performance: a 4B model achieves 49.71% accuracy on LiveCodeBench, outperforming GPT-4o and DeepSeek-V3 despite being over 50 smaller.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05711v3">Beyond tokens: a unified framework for latent communication in LLM-based multi-agent systems</a></div>
    <div class="paper-meta">
      📅 2026-07-15
    </div>
    <details class="paper-abstract">
      Multi-agent systems built on large language models (LLMs) have become a prevailing paradigm for tackling complex reasoning, planning, and tool-use tasks. The dominant communication protocol in such systems is natural language: agents exchange messages token-by-token, verbalising their internal reasoning so that peers can read, verify, and respond. While convenient and interpretable, this protocol suffers from three structural drawbacks -- high inference cost, irreversible information loss during discretization, and ambiguity/redundancy of natural language. A growing body of work therefore explores an alternative protocol -- latent communication -- in which agents exchange continuous representations (embeddings, hidden states, or KV-caches) directly, bypassing the bottleneck of text generation. This paper presents a unified framework for organising the rapidly expanding literature on latent communication. We analyse existing methods along three orthogonal axes: (1) WHAT information is communicated (Embeddings, Hidden States, KV-Caches, or other continuous state); (2) WHICH sender-receiver alignment is used (latent-space alignment and layer alignment); and (3) HOW the communicated information is fused into the receiver (concatenation, prepending, mathematical operations, cross-attention, or cache restoration). Under this 3-axis framework, we systematically categorise eighteen representative methods proposed between 2024 and 2026, identify five major design patterns, and surface a set of open challenges -- including cross-architecture alignment, security of latent channels, compression for edge deployment, and the relationship between latent communication and latent chain-of-thought. We hope that this framework both lowers the barrier to entry for new researchers and provides a vocabulary for comparing future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.13618v1">STOCKTAKE: Measuring the Gap Between Perception and Action in LLM Agents with a Fair Oracle</a></div>
    <div class="paper-meta">
      📅 2026-07-15
    </div>
    <details class="paper-abstract">
      LLM agents are increasingly evaluated on multi-week decision tasks in which the state that drives cost is never directly observed. On such tasks the final cost cannot say why an agent failed: it may have misread the world, or read it correctly and still failed to act (the knowing-doing gap). Existing evaluations cannot separate these two failures; their reference policies either read privileged information the agent never sees, or are missing altogether. We introduce STOCKTAKE, a 26-week supply-chain replenishment benchmark built as a factored partially observable Markov decision process with six hidden factor processes, designed so that a fair reference policy is computable: an exact Bayes filter per factor drives a rollout policy on the identical observation stream the agent receives. Scoring each run between a symptom-blind base-stock floor (0) and this oracle (1) yields a skill score, and grading each week's written rationale yields a stated-belief detection lag and a knowing-doing rate, so state estimation and control are measured separately. On fifty seeds with curated stress profiles, Claude Sonnet 5, GPT-5.4, DeepSeek-V4-Pro, and Grok 4.5 detect 84-88% of hidden failures, typically within a week of onset, yet span skill scores from 0.62 to -0.23: two of the four end below the symptom-blind floor while naming factors slightly faster than the two that beat it. The failure has two faces. Where stress persists, 34-43% of correctly diagnosed stress weeks still end in stockout for every model, a rate that partly reflects the severity of the weeks models notice. That rate also runs opposite to skill: the two models under the floor stock out least on diagnosed weeks, so under-response is only one face of the gap, and their traces point to the other, responses whose cost exceeds what they protect. STOCKTAKE measures both directions of that failure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14169v1">When a Verified World Model Still Loses: Play-Adequacy vs Prediction-Accuracy in LLM-Synthesized Code World Models</a></div>
    <div class="paper-meta">
      📅 2026-07-15
      | 💬 41 pages, 4 figures. Code and reproduction log: https://github.com/JaviMaligno/code-world-models
    </div>
    <details class="paper-abstract">
      Large language models can synthesize a game's rules as executable code - a Code World Model (CWM) - which a classical planner then searches over. Such models are typically accepted when they reach high transition accuracy on sampled trajectories. We argue this is the wrong notion of adequacy for planning. We show four things. (1) An LLM-synthesized CWM can pass a sampling gate at 100% transition accuracy and be $\geq 98\%$ state-accurate on the planner's own search distribution, yet lose systematically at play, because the $<1\%$ it gets wrong is exactly the pivotal dynamics; the play cost of the omitted rule is $0.091$ (seed-clustered 95% CI $[0.065,0.117]$, $n=4800$). We call this the verified-vs-correct gap, and confirm it end-to-end through the synthesis pipeline. (2) The harm follows a quantitative law, $\mathrm{danger}=\mathrm{play\_cost}\times(1-\mathrm{rarity})^N$, whose $(1-\mathrm{rarity})^N$ gate-miss factor is proven exact and whose play cost is empirically bounded. (3) The failure is not repaired by more data: LLM synthesis behaves as rule translation, not rule inference, and did not infer the omitted rule across models (GPT-5.x) and data regimes (including DAgger and targeted examples). (4) The same mechanism recurs on the belief-inference function of imperfect-information CWMs: we prove a coverage bound (a size-$N$ gate is identifying when $N\gtrsim b^{d_{\max}}$), explaining why shallow games such as Kuhn poker show no gap, and hand-construct Beacon, a verified-but-wrong inference function that passes the gate yet loses every game. These results suggest adequacy for planning-oriented world models should be measured on the search distribution or by play directly, not by prediction accuracy on sampled transitions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.13591v1">Memory as a Controlled Process: Learned Adaptive Memory Management for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-15
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents increasingly rely on external memory systems to accumulate experience across tasks. Yet nearly all existing approaches, from graph-structured memories to reflective insight stores, access memory through fixed, hand-designed heuristics. We argue that this static view of memory is a core bottleneck for agentic learning because optimal memory behavior is fundamentally context-dependent. The early stages of the tasks, benefit from minimal retrieval because memory is sparse; recurring goal types benefit from plan reuse rather than generic nearest-neighbor lookup; stuck agents benefit from re-retrieval with alternative queries; and across long task streams, the memory store itself must be consolidated and pruned to remain useful. We present Memory as a Controlled Process (MemCon), a framework that models memory operations as a Markov Decision Process and learns an online policy that adaptively decides when, what, and how much to retrieve, when to inject a distilled plan, and when to consolidate or forget. MemCon is backend-agnostic: it wraps any existing memory implementation, learns from task-by-task binary feedback with no pretraining and no additional LLM calls, and uses a lightweight tabular contextual bandit with UCB exploration that converges within tens of tasks. Across 6 benchmarks, 3 agent frameworks, and 3 LLM backbones, MemCon consistently outperforms multiple memory baselines by up to 15.2 points in task success while reducing token consumption by 5--20%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.12811v2">Left-right asymmetry in predicting brain activity from LLMs' representations emerges with their formal linguistic competence</a></div>
    <div class="paper-meta">
      📅 2026-07-15
    </div>
    <details class="paper-abstract">
      When humans and large language models (LLMs) process the same text, activations in the LLMs correlate with brain activity measured, e.g., with functional magnetic resonance imaging (fMRI). Moreover, it has been shown that, as the training of an LLM progresses, the performance in predicting brain activity from its internal activations improves more in the left hemisphere than in the right one. The aim of the present work is to understand which kind of competence acquired by the LLMs underlies the emergence of this left-right asymmetry. Using the OLMo-2 7B language model at various training checkpoints and fMRI data from English participants, we compare the evolution of the left-right asymmetry in the correlation between brain activity and model predictions alongside performance on several benchmarks. We observe that the asymmetry co-emerges with the formal linguistic abilities of the LLM. These abilities are demonstrated in two ways: by the model's capacity to assign a higher probability to an acceptable sentence than to a grammatically unacceptable one within a minimal contrasting pair, and by its ability to produce well-formed text. By contrast, the left-right asymmetry does not align with the performance on arithmetic or Dyck language tasks; nor with text-based tasks involving world knowledge and reasoning. We generalize these results to another family of LLMs (Pythia) and two other languages, French and Chinese. Our observations indicate that the left-right asymmetry in brain predictivity matches the progress in formal linguistic competence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.13511v1">ExTernD: Expanded-Rank Ternary Decomposition Ternary LLM PTQ with Accuracy Approaching Any Quantization Level</a></div>
    <div class="paper-meta">
      📅 2026-07-15
    </div>
    <details class="paper-abstract">
      We introduce ExTernD (Expanded-rank Ternary Decomposition), a post-training factorization of each LLM weight matrix $A \in \mathbb{R}^{m \times n}$ into $A \approx B \mathrm{diag}(D) C$ with ternary factors $B \in \{-1,0,+1\}^{m \times k}$, $C \in \{-1,0,+1\}^{k \times n}$ and a real scale vector $D \in \mathbb{R}^k$. The inner rank $k = μ\min(m,n)$ is deliberately expanded beyond full rank ($μ> 1$), so that components past full rank correct the quantization error of earlier ones. We prove the residual decreases monotonically in $k$ and can be driven below any $\varepsilon > 0$: ExTernD approaches bf16 accuracy arbitrarily closely, which no ternary scheme with a fixed plane count can do. Memory and compute scale continuously with $μ$, and factor sparsity continuously with a threshold $τ$, so an accuracy target is hit exactly rather than rounded to the next bit-width. ExTernD matches Q4_K's per-matrix accuracy at 5.2-5.5 effective bpw (5.1-5.5 with importance weighting) on Gemma-4-E2B and Qwen3.5-4B, and a full Qwen3.5-4B conversion at $μ= 3$ reaches 10.10 wikitext-2 perplexity against 9.78 for bf16 (+3.2%), placing it near the Q4_K/Q5_K accuracy band at ~5.7 effective bpw.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11098v3">AgentCheck: A Reproduce-Intervene-Mitigate Workbench for LLM Agents over MCP</a></div>
    <div class="paper-meta">
      📅 2026-07-15
    </div>
    <details class="paper-abstract">
      Tool-using LLM agents are mostly evaluated assuming all tools work. When a tool times out, returns a week-stale value, or has its description poisoned in deployment, the developer needs a controlled way to reproduce the failure, test a fix, and confirm the fix worked before deployment. We present AgentCheck, an open-source web workbench that turns an MCP server into an intervention surface. AgentCheck runs an agent against its real tools and records every tool response, then re-runs the agent with the response perturbed by a fault (12 types) injector. Matching tool calls are replayed from cache, and later tool calls go live after the agent diverges. This yields a reproduce-intervene-confirm loop: the developer toggles a mitigation, re-runs against the identical fault, and sees if the failure goes away. Scoring has two parts: deterministic pass/fail rules, plus an LLM judge for interpretive labels, validated against human annotations. Across five agents, the best passes 105/120 scenarios and the weakest only 77. The failures are usually silent, confident use of incorrect tool outputs rather than crashes. On the weakest agent, a retry mitigation raises success on timeout error faults from as few as 30% of cases to 100%, whereas stale-data faults remain near 3-4 of 10 regardless of the mitigation. AgentCheck makes these failure modes reproducible, comparable, and verifiable before deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14167v1">Structured Feedback Improves Repair in an LLM Agent Loop</a></div>
    <div class="paper-meta">
      📅 2026-07-15
    </div>
    <details class="paper-abstract">
      LLM agents often retry after external validation rejects a candidate, but the interface between validation and the next model call remains underspecified. We introduce VeriHarness, a code-controlled agent loop in which models generate candidates while external validators control acceptance, budgets, and traces. We use it to compare raw diagnostics with feedback that identifies the failure location, observed value, and admissible alternatives. Across 50 paired TextWorld games under a four-call cap, feedback containing all three fields raises terminal success from 14/50 to 36/50 for Qwen2.5-Coder-14B (+44 percentage points) and from 8/50 to 29/50 for Llama-3.1-8B (+42 points). Ablations locate most of the gain in the admissible alternatives: feedback containing only the location and observed value remains near the raw diagnostic baseline. Presenting the complete repair information in prose instead of a keyed JSON record yields nearly the same success, providing no evidence that JSON syntax itself improves repair. The ordering persists across the tested call budgets and one sampled-decoding setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.13474v1">MyAG: A Graph-Based Framework for Designing and Analyzing Composable LLM Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-07-15
    </div>
    <details class="paper-abstract">
      We present MyAG, a graph-based framework for designing and analyzing composable LLM agent systems. Our framework separates agent system construction into three graph abstractions: a component graph for agents, environments, and modules; a workflow graph for execution control; and a search graph for runtime execution. This separation allows users to flexibly reuse the same components with different strategies. We further support hierarchical composition through recursive system nodes and provide monitoring and visualization tools for inspecting agent execution. Experiments on representative agent applications show that our framework supports flexible agent system design and helps analyze performance-efficiency tradeoffs. Our framework is publicly available and fully open-source.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.24902v2">When Reasoning Hurts: Source-Aware Evaluation of Frontier LLMs for Clinical SOAP Note Generation</a></div>
    <div class="paper-meta">
      📅 2026-07-15
    </div>
    <details class="paper-abstract">
      Reasoning-enabled LLMs perform strongly on medical reasoning benchmarks, but it remains unclear whether these gains transfer to structured clinical documentation; we investigate this question using SOAP note generation from clinical dialogue in a source-aware benchmark spanning OMI Health, ACI-Bench, and PriMock57. We evaluate GPT-5.4, DeepSeek-V4-Flash, and Gemma-4-E4B in a controlled 2x2 design that independently toggles provider-native reasoning and same-source retrieval-augmented generation (RAG). Outputs are assessed using seven automatic metrics alongside two reference-aware LLM judges. Both evaluation approaches agree that a non-reasoning GPT-5.4 configuration achieves the highest overall quality, while DeepSeek-V4-Flash performs best among reasoning-enabled configurations. Enabling reasoning significantly degrades GPT-5.4 performance across all three datasets, whereas same-source RAG yields smaller, model-dependent improvements. Overall, the findings indicate that stronger reasoning capability should not be assumed to improve fidelity-sensitive SOAP note generation without dedicated, task-specific evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14165v1">Towards Reliable AI-Assisted Analog Design: Template-Constrained LLM Agents for SAR ADC Generation</a></div>
    <div class="paper-meta">
      📅 2026-07-15
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have demonstrated significant capability in software code generation, their application to analog Electronic Design Automation (EDA) is bottlenecked. Owing to limited circuit topology understanding and data, directly prompting LLMs and multimodal models leads to hallucinations and failure to produce schematics capable of passing rigorous SPICE simulations, as we show in our work. Instead, we propose an end-to-end, multi-step LLM agentic framework ATLAS, capable of generating a functional Successive Approximation Register (SAR) Analog-to-Digital Converter (ADC) that successfully passes simulation validation. To adhere to the rigid constraints of analog design, we utilize expert knowledge to ground the LLM in its planning, selection, parameterization, and iterative modification. As part of ATLAS, we introduce Template-Constrained Generation - which unlike other template-based works - builds towards a more generalized SAR ADC generation flow. We demonstrate a strong proof-of-concept of our framework by developing SAR ADCs across technology nodes and input specs. Overall, our expert-knowledge grounded multi-step agentic ATLAS establishes a pragmatic foundation for integrating LLMs into reliable analog design methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.26707v2">Value Drifts: Tracing Value Alignment During LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2026-07-15
      | 💬 TACL 2026
    </div>
    <details class="paper-abstract">
      As LLMs occupy an increasingly important role in society, they are more and more confronted with questions that require them not only to draw on their general knowledge but also to align with certain human value systems. Therefore, studying the alignment of LLMs with human values has become a crucial field of inquiry. Prior work, however, mostly focuses on evaluating the alignment of fully trained models, overlooking the training dynamics by which models learn to express human values. In this work, we investigate how and at which stage value alignment arises during the course of a model's post-training. Our analysis disentangles the effects of post-training algorithms and datasets, measuring both the magnitude and time of value drifts during training. Experimenting with Llama-3 and Qwen-3 models of different sizes and popular supervised fine-tuning (SFT) and preference optimization datasets and algorithms, we find that the SFT phase generally establishes a model's values, and subsequent preference optimization rarely re-aligns these values. Furthermore, using a synthetic preference dataset that enables controlled manipulation of values, we find that different preference optimization algorithms lead to different value alignment outcomes, even when preference data is held constant. Our findings provide actionable insights into how values are learned during post-training and help to inform data curation, as well as the selection of models and algorithms for preference optimization to improve model alignment to human values.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.13425v1">Data-Efficient Adaptation of LLMs via Attention Head Reweighting</a></div>
    <div class="paper-meta">
      📅 2026-07-15
      | 💬 Published in COLM 2026
    </div>
    <details class="paper-abstract">
      Learning effectively from limited data is critical in domains like security where labeled examples are scarce. Large language models (LLMs) have demonstrated some capabilities for data-efficient learning, especially through parameter-efficient adaptation methods, but continue to struggle when faced with few samples for difficult tasks. To meet this challenge, we propose Attention Head Reweighting (AHR), a data-efficient method that adapts LLMs to new text-classification tasks by learning only a single scalar per attention head. This drastically reduces the number of parameters that need to be learned by making use of the functional specialization of individual attention heads. Experiments on diverse open-source text classification datasets show that AHR can outperform standard baselines like LoRA when learning from limited samples, despite having 200-1000x fewer trainable parameters, as our AHR only modifies ~0.0001% of the model's parameters. In addition, our learned weights are easy to interpret and can be analyzed to better understand the mechanisms and attention heads responsible for in-context learning abilities in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.05396v4">FairCoder: Probing LLM Bias in High-Stakes Decision Making via Coding Tasks</a></div>
    <div class="paper-meta">
      📅 2026-07-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in high-stakes decisions such as hiring and college admissions, making their social bias a critical concern. While LLMs are trained to refuse explicitly biased requests, bias can be leaked implicitly during LLM planning and reasoning process. As code becomes the primary medium for LLM internal logic-writing, we introduce FairCoder, a benchmark that frames decision-making as coding tasks to systematically probe LLM bias across employment, education, and healthcare domains, covering multiple fairness definitions. Considering that existing metrics may fail when LLMs frequently refuse the request, we propose FairScore, a metric that jointly captures refusal behavior and group-level outcome diversity. Experiments with a 1k-sample dataset on powerful LLMs reveal consistent and previously underexplored bias patterns, such as prioritizing applicants from high-income families in college admissions. Our findings highlight the risks of deploying LLMs as decision-making agents and provide a comprehensive evaluation framework for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.09674v2">How LLMs Might Think</a></div>
    <div class="paper-meta">
      📅 2026-07-15
    </div>
    <details class="paper-abstract">
      Do large language models (LLMs) think? Daniel Stoljar and Zhihe Vincent Zhang have recently developed an argument from rationality for the claim that LLMs do not think. We contend, however, that the argument from rationality not only falters, but leaves open an intriguing possibility: that LLMs engage only in arational, associative forms of thinking, and have purely associative minds. Our positive claim is that if LLMs think at all, they likely think precisely in this manner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.13347v1">Evaluation Ability Does Not Imply Optimization Utility: LLM-as-a-Judge Signals in Closed-Loop Table Recognition</a></div>
    <div class="paper-meta">
      📅 2026-07-15
      | 💬 31 pages, 9 figures, appendix included
    </div>
    <details class="paper-abstract">
      LLM-as-a-judge is widely used to provide feedback and selection signals in closedloop regeneration, but this use remains insufficiently validated. We study it in table recognition, where deterministic TEDS evaluation provides a controlled testbed, using FinTabNet and OmniDocBench. Three findings emerge. First, judge signals were weak on both datasets: scores frequently tied, rankings were not reproducible, and the only selection policy that beat random on both datasets depended on an earliest-iteration tie rule, so its advantage cannot be attributed to the judge scores alone. Iteration produced better candidates, but the judge failed to recover them. Second, severe losses occurred even without specific judge feedback. A structurepreserving instruction significantly reduced the severe-loss rate on FinTabNet and was directionally consistent on OmniDocBench. The contrasts support target-preservation failure under unconstrained regeneration as a proximate mechanism of the observed severe losses. Third, the structure-preservation constraint reduced the severe-loss tail but produced no improvement. In an exploratory 2x2 analysis, the same protection was not stably observed when judge feedback was retained. These results do not dispute the value of LLMs as evaluators. Instead, they show that evaluation ability does not imply optimization utility. Iterative refinement requires, at minimum, a verification signal that deterministically detects structural change, rather than judge scores alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11619v2">When Agents Disagree With Themselves: Behavioral Consistency as an Uncertainty Signal for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-15
      | 💬 Accepted at the ICML 2026 Workshop on Statistical Frameworks for Uncertainty in Agentic Systems. 12 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Running the same LLM agent on identical inputs yields 2.3-4.2 distinct action sequences per 10 runs; this behavioral variance constitutes a training-free, black-box uncertainty signal that instantiates selective classification and distribution-free calibration for agentic systems. Across 8,000 runs of four models on 200 HotpotQA questions, consistent tasks (at most 2 unique paths) achieve 82-87% accuracy while inconsistent tasks (4 or more paths) achieve 41-65%, a gap that survives controls for task difficulty. Divergence concentrates at step 2 (50.5% of Llama tasks), and consistency metrics detect failures with AUROC 0.62-0.78. Exploiting this signal, selective prediction (answering only when k=3 runs agree) achieves 87-88% accuracy at 54-62% coverage, a 6-14pp gain over single-run baselines, and matches a split-conformal baseline without a held-out calibration set. A cross-benchmark validation on SWE-bench (50 tasks, 1,000 runs) preserves the consistency hierarchy while revealing an ~8x spread in mean trajectory length across models, and bootstrap analysis shows single-run evaluations misrank models 29.3% of the time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12507v1">When Binaries Talk Back: Representation-Confusion Attacks on LLM-Assisted Reverse Engineering</a></div>
    <div class="paper-meta">
      📅 2026-07-14
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      LLM-assisted reverse-engineering (RE) systems analyze strings, decompiler output, and tool reports derived from ttacker-controlled binaries. A binary can make data look like instructions or records from one origin look like independent evidence. We call such failures Representation-Confusion Attacks in Reverse Engineering (RARE): the pipeline promotes a correctly extracted observation to instruction authority, claim-validating evidence, or trusted analysis state without the authority or support that role requires. RARE-Bench measures these failures with behavior-checked clean and adversarial binaries. After an exploratory 11,520-call study, we test RARE-Guard's authorization and evidence controls on 20 new programs and two models. Without runtime controls, the models propose a planted unsafe action in 35/40 adversarial cases and 0/40 clean cases. When binary-derived content is shown only as data (Data-Only rendering), they still make 15 unsafe proposals. Tool Authorization denies all 15 and authorizes all 40 matched analyst requests. On identical report drafts, Support Gate validates 23/40 false claims by counting records from one origin separately. Provenance Gate groups those records before counting support, validates 0/40 false claims, and retains all 40 supported claims. We then instrument Ghidra, r2pipe, and angr on 16 further programs. In a preselected eight-program subset, no single-tool draft reaches Support Gate's validation threshold for the false claim. In fused drafts across all 16 programs, Support Gate validates 32/32 false claims. Provenance Gate prevents validation of all 32 and retains all 32 supported claims. A deterministic renderer prevents downgraded claims from reappearing in the final report. Binary-derived content may therefore guide analysis without gaining authority over tools, and views from several tools do not necessarily provide independent evidence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12490v1">When is LLM-Based Program Reasoning Correct? A Completion Semantics for LLM-Based Code Inference</a></div>
    <div class="paper-meta">
      📅 2026-07-14
      | 💬 28 pages, 4 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Due to token and cognitive limits, Large Language Models (LLMs) typically perform program reasoning over incomplete code fragments/prompts rather than complete programs. Such reasoning therefore must rely on {assumptions about omitted code and context. As a result, the meaning of an inference over a program fragment is not absolute, but depends on an implicit completion model describing how the fragment may be refined into a complete program. In this paper, we introduce completion semantics for LLM-based program reasoning. We formalize incomplete programs as denoting a space of possible refinements and define the correctness of existential inferences relative to a completion model. Under this view, a reported bug is correct whenever there exists a completion within the model that witnesses the bug. This perspective explains why many LLM-generated reports are neither simply correct nor incorrect, but instead depend on assumptions about omitted context. We have instantiated our approach in the form of a witness-generation workflow that concretizes completions underlying an inference by constructing executable refinements of the original program fragment. Witnesses serve both as evidence for existential claims and as a mechanism for exposing the assumptions required to support them. We evaluate our approach on real-world LLM-generated bug reports and program-analysis tasks. Our results show that witness generation effectively distinguishes inferences supported by plausible completions from those requiring unrealistic assumptions, providing a practical mechanism for validating reasoning over incomplete programs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.15222v2">PerfCodeBench: Benchmarking LLMs for System-Level High-Performance Code Optimization</a></div>
    <div class="paper-meta">
      📅 2026-07-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can often generate functionally correct code, but their ability to produce efficient implementations for performance-critical systems tasks remains limited. Existing code benchmarks mainly emphasize correctness or algorithmic problem solving, while realistic systems-level optimization is still underexplored. To address this gap, we introduce PerfCodeBench, an executable benchmark for evaluating LLMs on high-performance code optimization. The tasks require system-level implementation choices, hardware-aware optimization, and careful handling of performance bottlenecks. Each task includes executable correctness checks, a baseline implementation, and a reference optimized solution. This allows us to evaluate both correctness and runtime-oriented efficiency. Our evaluation on a broad set of state-of-the-art LLMs shows a clear gap between model-generated code and expert-optimized implementations. The gap is especially large on tasks involving parallelism and GPU operations. Current models also show weaknesses in cross-language robustness and in consistently reaching expert-level efficiency. These results suggest that performance-aware evaluation are still needed. LLMs should move beyond generating merely correct code toward producing efficient systems software. We submit the benchmark data, evaluation infrastructure, and complete logs of all LLMs-generated code at https://anonymous.4open.science/r/perfcodebench-7CDE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12468v1">An Omnilingual-ASR-Based Speech-LLM System for the 2nd MLC-SLM Challenge</a></div>
    <div class="paper-meta">
      📅 2026-07-14
      | 💬 Accepted to INTERSPEECH 2026. 4 pages + references. Technical description of our 2nd MLC-SLM Challenge Task 1 submission
    </div>
    <details class="paper-abstract">
      We describe our submission to Task 1 of the 2nd MLCSLM Challenge: a cascaded diarization-then-recognition system that combines DiariZen-Large-s80 (WavLM-Large) segmentation, CAM++ embedding-based two-speaker clustering, and a LoRA-adapted omniASR LLM 7B v2 recognizer, with no oracle segmentation or speaker labels at test time. On the official Development set (150 conversations, 21 language/accent categories) the system attains a macro tcpMER of 29.27%, versus 79.15% for the official baseline; on the Evaluation set it scores 50.23%. We also analyze two engineering choices that substantially affect tcpMER. First, embedding-based speaker clustering outperforms an end-to-end-style alternative that assigns speakers from ASR <sc> turn markers alone. Second, overlap-aware segmentation, although intended to raise diarization recall, increases tcpMER because overlapped speech is transcribed twice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12467v1">Understanding before Naming! Enhancing LLM-based Method Name Prediction with Code Summarization</a></div>
    <div class="paper-meta">
      📅 2026-07-14
    </div>
    <details class="paper-abstract">
      Method names are critical to software quality, affecting code comprehensibility, maintainability, and developer collaboration. However, manually designing meaningful method names is challenging. Method Name Prediction (MNP), which automatically generates method names from code snippets, has recently attracted attention. Although large language models (LLMs) show promising performance for MNP, two challenges remain. First, existing evaluations mainly rely on token similarity metrics, which often fail to reflect human judgments of semantic quality. Second, current LLM-based MNP methods usually generate names through direct code-to-name mapping, which differs from the human process of understanding functionality before naming. To address these challenges, we conduct empirical studies on LLM-based evaluation and MNP strategies. We compare 6 metric-based evaluators, 5 LLM-based evaluators, and 6 human evaluators. Results show that LLM-based evaluators, especially DeepSeek-based evaluators, are more consistent with human judgments than traditional metrics. We further compare direct generation and summarization-and-refinement strategies. Results indicate that summarization and refinement generally improve the semantic quality of generated names. Case studies reveal three limitations: inaccurate summaries, semantic misalignment, and close semantic scores. Based on these findings, we propose SMNP, an MNP approach combining MNP-oriented summarization and chain-of-thought enhanced refinement. Experiments on 5 LLMs and 2 datasets demonstrate the effectiveness and robustness of SMNP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12406v1">Isolation as a First-Class Principle for LLM-Agent System Safety: Concepts, Taxonomy, Challenges and Future Directions</a></div>
    <div class="paper-meta">
      📅 2026-07-14
    </div>
    <details class="paper-abstract">
      The capability of LLM agents to function as the ``brain'' of a system fundamentally expands the scope of analysis beyond a standalone model. Consequently, safety is no longer only about input--output content alignment. It also concerns system behavior and real-world execution outcomes. However, the current literature is fragmented across attack types, applications, and benchmarks. This makes it hard to explain why failures such as prompt injection, tool misuse, and memory poisoning often share the same structural cause, and how they spread through an agent workflow. In this survey, we treat isolation as a first-class principle for LLM-agent system safety. By isolation, we refer to the separation of user inputs, tool access, execution channels, inter-agent communication, and environment-originated context. We organize the literature with a boundary-centric taxonomy of five boundaries: user-agent, agent-tool, agent-execution, agent-agent, and system-environment. This view helps identify where the loss of isolation first occurs, how compromise propagates across boundaries, and which defenses are most relevant at each interface. We also summarize cross-boundary failure paths, discuss open challenges, and outline a research agenda for isolation-by-construction in future agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.24787v2">ReLope: KL-Regularized LoRA Probes for Multimodal LLM Routing</a></div>
    <div class="paper-meta">
      📅 2026-07-14
    </div>
    <details class="paper-abstract">
      Routing has emerged as a promising strategy for balancing performance and cost in large language model (LLM) systems that combine lightweight models with powerful but expensive large models. Recent studies show that \emph{probe routing}, which predicts the correctness of a small model using its hidden states, provides an effective solution in text-only LLMs. However, we observe that these probes degrade substantially when applied to multimodal LLMs (MLLMs). Through empirical analysis, we find that the presence of visual inputs weakens the separability of correctness signals in hidden states, making them harder to extract using standard probe designs. To address this challenge, we introduce two complementary approaches for improving probe routing in MLLMs. First, we propose the \emph{Attention Probe}, which aggregates hidden states from the preceding layer based on attention scores to recover distributed correctness signals. Second, we present the \emph{KL-Regularized LoRA Probe (ReLope)}, which inserts a lightweight LoRA adapter and applies a KL regularizer to learn routing-aware representations. Comprehensive experiments show that our methods consistently outperform baselines, suggesting that improving the quality of hidden states is key to effective routing in MLLMs. Our code is available at https://github.com/Spinozaaa/ReLope.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12397v1">Critic Experience Bank: Self-Evolving Step-Level Confidence Estimation for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-14
    </div>
    <details class="paper-abstract">
      LLM agents act in external environments where each action changes the state that later decisions condition on, and where a single wrong step can waste interaction budget or trigger irreversible side effects long before the final failure is observed. Reliable deployment therefore requires \emph{step-level confidence estimation}: a calibrated probability that each proposed action is productive, available \emph{before} the action is executed. Existing LLM confidence estimators are designed to score a response from the given prompt, but agent confidence also depends on execution consequences: whether similar actions in similar situations actually advanced the task after the environment responded. We introduce the \method (\methodshort), a self-evolving critic framework in which an LLM critic accumulates evidence from its own past judgments and their observed consequences. After each trajectory, a hindsight LLM that sees the full execution feedback votes on whether each step was productive. The resulting pseudo-labels populate a memory bank from which related productive and unproductive experiences are retrieved into the critic's prompt whenever a similar step recurs. \methodshort requires no training and uses no ground truth step labels. Across three agent benchmarks and three critic backbones, \methodshort attains the best calibration (ECE and Brier) and ranking (AUC) in every dataset--critic combination, reducing ECE by up to $54\%$ relative to the strongest training-free baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10609v5">iTIMO: An LLM-empowered Synthesis Dataset for Travel Itinerary Modification</a></div>
    <div class="paper-meta">
      📅 2026-07-14
      | 💬 Accepted by SIGIR2026
    </div>
    <details class="paper-abstract">
      Addressing itinerary modification is crucial for enhancing the travel experience as it is a frequent requirement during traveling. However, existing research mainly focuses on fixed itinerary planning, leaving modification underexplored due to the scarcity of need-to-modify itinerary data. To bridge this gap, we formally define the itinerary modification task and propose a general pipeline to construct the corresponding dataset, namely iTIMO. This pipeline frames the generation of need-to-modify itinerary data as an intent-driven perturbation task. It instructs large language models to perturb real-world itineraries using three operations: REPLACE, ADD, and DELETE. Each perturbation is grounded in three intents: disruptions of popularity, spatial distance, and category diversity. Furthermore, hybrid evaluation metrics are introduced to ensure perturbation effectiveness. We conduct comprehensive benchmarking on iTIMO to analyze the capabilities and limitations of state-of-the-art LLMs. Overall, iTIMO provides a comprehensive testbed for the modification task, and empowers the evolution of traditional travel recommender systems into adaptive frameworks capable of handling dynamic travel needs. Dataset, code and supplementary materials are available at https://github.com/zelo2/iTIMO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12385v1">PM-Bench: Evaluating Prospective Memory in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-14
      | 💬 Published as a conference paper at COLM 2026
    </div>
    <details class="paper-abstract">
      A significant challenge in agentic AI is prospective memory: the ability to execute an intention at a specific future cue or state while other activities are ongoing. We introduce PM-Bench, a text-based benchmark for measuring prospective memory capabilities in modern LLM agents. Inspired by the Virtual Week paradigm from cognitive science, PM-Bench evaluates how well LLM agents maintain user intentions, execute delayed intentions, and monitor latent environment changes. Over the course of a simulated seven-day week, agents must continue an ongoing activity while deciding whether any deferred task is due. We compare eight state-of-the-art LLMs on PM-Bench under eight different agent configurations. PM-Bench proves challenging across all settings: the best method, a GPT-5.4 agent, reaches only 65.1\% F1 score under our evaluation. Furthermore, no single strategy for improving prospective memory dominates across models. We release PM-Bench as a controlled testbed for diagnosing these failures and developing training or inference-time interventions that support reliable prospective behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.00011v2">SkillSelect-Serve: QoS-Aware Budgeted Skill Service Recommendation for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-14
      | 💬 18 pages (14-page main text + appendices), 5 figures
    </div>
    <details class="paper-abstract">
      Reusable agent skills are emerging as a service-oriented capability layer for Large Language Model (LLM) agents. Unlike plain retrieval items, a skill exposes functional capabilities, input-output assumptions, tool dependencies, context cost, and risk metadata. Selecting skills is particularly challenging for small LLM agents, which can load only a few capability units under restricted context, tool availability, and risk tolerance. Existing fixed Top-k methods rank skills by textual relevance and overlook requirement satisfaction, deliverability, and operational constraints. We present SkillSelect-Serve, a QoS-aware, budget-constrained Skill Service recommendation framework. Raw skills are profiled as structured Skill Services, the task is converted into a structured requirement object, and candidates discovered from a large-scale registry are ranked by a calibrated task-conditioned suitability estimator and packed by a constrained projection enforcing token-budget, aggregated-risk, and tool-availability constraints, using only deployment-observable features. On a registry of 35,353 skills with pooled multi-positive relevance judgments verified by two independent assessors, the unconstrained top-5 recommendation fits a realistic 4,000-token context for only 9.1% of tasks; the constrained projection restores 100% deliverability at a cost of only 1.14 points of hit rate, outperforming retrieve-and-rerank, budget truncation, and diversity-based selection under identical budgets. The same mechanism halves delivered risk exposure and eliminates the 44-81% tool-violation rates of tool-agnostic recommendation. At an identical three-service budget, hit rate improves from 0.8864 to 0.9091 over fixed Top-3 retrieval. The results support managing reusable agent skills as discoverable, comparable, and constraint-aware service units instead of plain retrievable documents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12340v1">Skills That Don't Exist: A Large-Scale Study of Hallucinated Skill Recommendation in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-14
    </div>
    <details class="paper-abstract">
      LLM agents acquire new capabilities by downloading skills from open registries. Instead of browsing these catalogs manually, developers typically ask the agent to recommend and install a skill. This convenience hides a risk: agents frequently invent names for skills that exist in no registry. We term this flaw skill name hallucination. A fake name may seem harmless, but it opens the door to supply-chain attacks. Because registries rarely verify publishers, an adversary can prompt the agent, collect the fake names it returns, pre-register malicious skills under them, and wait for a victim to install the payload. We conducted the first large-scale measurement of skill name hallucination, evaluating 15,000 prompts across 12 configurations (4 standalone LLMs and 8 agents). We conservatively counted a name as hallucinated only if it was missing from all live registries and GitHub. The results reveal a systemic vulnerability: every configuration hallucinates. Rates average 36.0% for standalone LLMs and 36.9% for agents, rising to 43.1% on real-world developer questions. In total, the systems generated 5,669 distinct hallucinated names. Crucially, these names are not random noise. Agents repeat the same fake names across prompts and models, giving attackers highly reliable targets to hijack. Finally, we tested four model-level defenses and found a severe conflict between security and usability. The strongest, retrieval grounding, cut the hallucination rate from 40.8% to 3.2% but crippled usefulness: even the best-defended system recommended the correct skill only about one in six times. Skill name hallucination is thus a highly exploitable vulnerability requiring minimal attacker effort. Fixing it cannot rely on prompt engineering or model tuning alone. It demands ecosystem-wide structural changes: registry-level name reservations and verified recommendation pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12279v1">A Shared Subcircuit Lets LLMs Count Down Across Tasks</a></div>
    <div class="paper-meta">
      📅 2026-07-14
      | 💬 12 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Writing a sentence of exactly twelve words; ending a DNA sequence at the right codon; formatting an ASCII table. These are all tasks that language models can do that requires tracking how many tokens remain before a target. In this work, we identify in Llama-3.1-70B-Instruct a general mechanism for performing these tasks: a "countdown subcircuit" that compares the current position to a goal length and estimates the time remaining until then. We first isolate a countdown subcircuit in a controlled setting, in which the model is tasked with writing a fixed-length sentence ending in a specified word. We then investigate the geometry of the representations used by the subcircuit, and find that the subcircuit uses an identical motif previously identified in a frontier LLM on a separate task, thus suggesting that this motif is shared across models. Finally, we use unsupervised probing on a natural language dataset to find a variety of other tasks where this subcircuit is used, including tasks where the goal length is inferred from context rather than explicitly stated. Our work suggests that reverse-engineering subcircuits allows us to understand how behaviors generalize from a single example to many different tasks and even models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12273v1">Code-MUE: Measuring Code LLMs' Uncertainty through Execution-based Semantic Interaction Graphs</a></div>
    <div class="paper-meta">
      📅 2026-07-14
      | 💬 To appear at The ACM SIGSOFT International Symposium on Software Testing and Analysis (ISSTA) 2026
    </div>
    <details class="paper-abstract">
      As Code Large Language Models (LLMs) become central to modern software engineering, their inherent stochasticity poses significant real-world risks, where even minor errors can lead to severe functional, security, or safety consequences. Reliable automation, therefore, demands the ability to distinguish between confident, well-supported predictions and stochastic guessing. However, existing uncertainty estimation methods face a critical gap: white and grey-box techniques are often inapplicable to closed-source models, while standard "black-box" text metrics fail to capture the unique fragility of code, where syntactic variation does not always imply semantic divergence. To bridge this syntax-semantics gap, we introduce Code-MUE, a purely black-box framework that measures uncertainty through execution-based Semantic Interaction Graphs. Unlike prior approaches that rely on superficial textual similarity, Code-MUE grounds uncertainty in observable runtime behavior, calculating the Von Neumann entropy of the solution space to quantify global semantic diversity. A large-scale empirical study across eight state-of-the-art LLMs demonstrates that Code-MUE achieves a strong negative correlation with functional correctness (Spearman's correlation up to -0.98), significantly outperforming lexical and embedding-based baselines while enabling robust risk detection and selective prediction in practical workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.25112v2">Do LLMs Know What They Know? Measuring Metacognitive Efficiency with Signal Detection Theory</a></div>
    <div class="paper-meta">
      📅 2026-07-14
      | 💬 11 pages, 4 figures, 3 tables. v2 replaces the meta-d'/M-ratio analysis with a model-free measure (meta-I_2r); see the version note on page 1. Pre-registered; code and data at https://github.com/synthiumjp/sdt_calibration
    </div>
    <details class="paper-abstract">
      Standard evaluation of LLM confidence relies on calibration metrics (ECE, Brier score) that conflate how much a model knows (Type-1 accuracy) with how well its confidence signal tracks that knowledge (Type-2 metacognitive sensitivity). We apply Signal Detection Theory (SDT) to decompose these capacities, treating token-level normalised log-probability as a graded confidence variable and answer correctness as the state to be discriminated. We characterise the Type-2 ROC of this signal, including its unequal-variance structure via z-ROC analysis, and -- because the meta-d' efficiency ratio is not well defined for open-ended QA, which lacks a two-alternative Type-1 decision -- quantify metacognitive efficiency with a model-free information measure, normalised metacognitive information (meta-I_2r). Applied to four LLMs (Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.3, Llama-3-8B-Base, Gemma-2-9B-Instruct) across 224,000 factual QA trials, we find: (1) metacognitive information varies more than two-fold across models and co-varies inversely with accuracy -- the least accurate model has the most informative confidence -- though with four models this ordering cannot be separated from an error-difficulty confound, so we report it as coupling, not decoupling; (2) the confidence signal has model-specific unequal-variance structure (z-ROC slopes 0.81 to 1.18) invisible to calibration metrics; (3) metacognitive information is domain-specific, strongest in Arts & Literature for every model; (4) temperature dissociates Type-1 accuracy from metacognitive information, which stays stable while accuracy shifts. All estimates carry permutation nulls and bootstrap confidence intervals. Pre-registered; code and data public.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06715v2">Does Topic Sentiment Cause Perceived Ideology? Comparing Human and LLM Annotations in Political News Articles</a></div>
    <div class="paper-meta">
      📅 2026-07-14
      | 💬 V1 accepted to ACL SRW 2026. V2 updates experiments
    </div>
    <details class="paper-abstract">
      We ask whether topic sentiment has a causal effect on perceived political ideology, and whether the answer depends on who assigns the ideology label. Using articles from AllSides, paired with shared sentiment annotations from Llama-3.3-70b-versatile, we compare ideology labels from expert human annotators, GPT-4o-mini (baseline and finetuned), and Llama-3.3-70B. We apply Double Machine Learning (DML) and mediation analysis across all four annotation paradigms. Zero-shot LLMs regularly inflate effect sizes relative to human annotations, while fine-tuning often attenuates them back toward the human scale. Our results have implications for the use of LLM annotations as silver labels and as proxies for human judgment in downstream causal analyses: they may be reliable for recovering the presence and direction of effects on the partisan topics, but not their magnitude, leading to over- or under-prediction of some ideology given particular topics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12236v1">Speculate with Memory: Lossless Acceleration for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-14
    </div>
    <details class="paper-abstract">
      Speculative execution accelerates LLM agents by using a smaller, cheaper model to predict and pre-launch the next step while the environment is idle. However, existing speculators are stateless and discard all information between tasks, preventing prediction quality from improving with experience. We equip the speculator with three online memory systems that learn from past agent trajectories: a contrastive transition table tracking action-sequence statistics, an episodic memory retrieving contextually similar segments, and a confusion tracker suppressing recurring errors. We evaluate this approach on six benchmarks spanning three speculation types: action prediction, observation prediction, and chained prediction. Memory-augmented speculation yields a 19--39\% relative accuracy improvement on action prediction and up to a $2.5\times$ increase on observation prediction tasks with repetitive action spaces. These gains grow continuously as memory accumulates and generalize across speculator models of varying cost. All speculation is lossless because it runs during idle time at zero added wall-clock cost, and the actor's trajectory is identical to non-speculative execution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.10383v2">Authoring for Living Worlds: Tool-Constrained LLM Agents for Executable Multi-Actor Scenarios</a></div>
    <div class="paper-meta">
      📅 2026-07-14
    </div>
    <details class="paper-abstract">
      We use LLM agents to author executable specifications for a living world: formal Graphs of Events in Space and Time (GESTs) that a 3D game engine executes deterministically into multi-actor narrative videos, with per-frame spatial, temporal, and semantic ground truth as a byproduct of execution. This inverts the dominant paradigm of LLM agents driving neural video generators, which emit pixels with no semantic guarantees and no annotations. Authoring is the hard problem: the world's capability registry cannot be enumerated in a context window, validity of an action depends on accumulated world state, and a staged refinement pipeline driving GPT-5 through six validated stages produced zero executable specifications in 50 attempts. Our hierarchical Director / Scene Builder architecture instead operates through a constraint-enforcing tool layer, in which exploration tools paginate the registry and building tools validate every operation against simulator state, so every emitted specification is executable by construction. Driving a far smaller model (Claude Haiku 4.5), the system executes 20 of 25 attempts (80%) when seeded with a target narrative text. Because each seed text derives from a source graph, we can measure how faithfully the agent reconstructs specified intent: event-level F1 reaches 0.83 against a 0.55 matched-random floor, and sequential structure 0.77 against 0.43, with the residual gap dominated by information the text itself drops.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.12682v2">RAFP: Identifying LLM Lineages via Rare-Region Fingerprints</a></div>
    <div class="paper-meta">
      📅 2026-07-14
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly released under restricted licenses, creating a growing need for robust model ownership verification. Existing fingerprinting methods are often fragile under downstream finetuning, require invasive training modifications, or fail in black-box settings. We introduce RAFP, a robust framework for identifying LLM lineages via rare-region fingerprints. Our key insight is that downstream finetuning primarily updates common high-density language behaviors, while low-probability prompt regions receive weak optimization signal and limited gradient alignment under finetuned distribution. As a result, rare prompt-response behaviors remain stable across common model adaptations. RAFP is non-invasive, constructing fingerprints via discrete gradient-based optimization over rare prompts without modifying model weights. We provide a theoretical analysis showing that the likelihood change of rare-region fingerprints under finetuning remains bounded. Experiments across four LLM families and multiple downstream adaptations, including supervised finetuning, LoRA, quantization, prompt-template variation, and decoding changes, show that RAFP achieves strong fingerprint persistence and substantially outperforms prior fingerprinting baselines in black-box settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12233v1">Fin-Analyst at FinMMEval 2026 Task 3: A Live Hybrid Trading Agent with LLM Specialists and Rule-Based Signals</a></div>
    <div class="paper-meta">
      📅 2026-07-14
      | 💬 14 pages, 7 tables, 1 figure. CLEF 2026 FinMMEval Task 3 Working Notes
    </div>
    <details class="paper-abstract">
      Large language model (LLM) trading agents show promising performance in equity markets, yet remain narrowly focused on US equities with little evidence from live deployment. We present Fin-Analyst, a hybrid agent for FinMMEval 2026 Task 3: an eight-specialist LLM pipeline over news, SEC filings, fundamentals, analyst forecasts, technical indicators, and social sentiment, aggregated by a Meta-Agent for Tesla (TSLA), and a lightweight rule based three-signal vote for Bitcoin (BTC). On the final official leaderboard (accessed 2026-07-05), Fin-Analyst ranks first of all agents on TSLA with a +13.51% return, +28.33 points over Buy-and-Hold (Sharpe 4.10, 88% win rate), while the BTC vote ends flat yet well above a sharply falling baseline. Relative to the interim performance, the asset ranking reversed, indicating that short live windows yield volatility-sensitive rankings. Ablation identifies event-driven 8-K disclosures as the most influential TSLA signal. Error analysis shows that the memoryless agents repeat wrong calls for days at a time, and that the fixed-threshold BTC rules lost money by trading on noise in a sideways market while the LLM pipeline gained under similar conditions, motivating a memory-aware, LLM-based successor for both assets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17253v5">PDAGENT-BENCH: Characterizing, Grounding, and Architecting LLM/VLM Agents for VLSI Physical Design</a></div>
    <div class="paper-meta">
      📅 2026-07-14
    </div>
    <details class="paper-abstract">
      Large Language Models and vision-language models have shown remarkable success in the front-end design of Very Large-Scale Integrated Circuits, yet their capabilities for VLSI physical design remain significantly underexplored. The primary cause is the lack of standardized benchmarks for evaluating agentic physical design workflows that require high-dimensional, multi-stage optimization under strict design constraints, coordinated interaction with diverse Electronic Design Automation tools, and iterative refinement. This work introduces PDAGENT-BENCH, a comprehensive and multi-dimensional benchmark for evaluating LLM/VLM-based agents across the physical design stack. PDAGENT-BENCH integrates both task-level assessment and workflow-level execution. The benchmark suite contains 353 curated problems that combine conceptual questions with real-world industrial artifacts, with expert-validated references and executable solutions. In addition, the benchmark provides a unified, human-aligned agentic physical design workflow framework that enables closed-loop evaluation of holistic physical design in realistic EDA environments. Experiments on 11 state-of-the-art models reveal that while modern LLMs/VLMs perform competitively on conceptual tasks, they remain substantially limited in tool-centric execution (e.g., 42.2% on Innovus script generation) and long-horizon, multi-stage reasoning. Our studies further show that human-skill-enhanced agentic workflows significantly improve end-to-end physical design performance. PDAGENT-BENCH establishes a standardized, reproducible, and realistic evaluation framework for advancing LLM/VLM-driven holistic physical design automation. To ensure full reproducibility and broad accessibility, we will release PDAgent-Bench together with its agentic workflow framework, instantiated on open-source PDKs (e.g., Nangate45, ASAP7) and open EDA tools (e.g., OpenROAD).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.08949v2">SeedSmith: LLM-Driven Seed Synthesis for Directed Fuzzing</a></div>
    <div class="paper-meta">
      📅 2026-07-14
    </div>
    <details class="paper-abstract">
      Directed fuzzing steers fuzzers toward user-defined sink functions to identify vulnerabilities, but it frequently fails to trigger crashes even after long campaigns. We identify two challenges that prevent directed fuzzers from exposing crashes: incomplete static analysis of indirect calls, which leaves reachable paths invisible to distance-based guidance, and lack of semantic guidance for crash preconditions, which blind mutation cannot satisfy within practical time budgets. A natural intervention point is the initial seed corpus: seeds that encode the right control-flow path and satisfy key crash preconditions shift fuzzing from blind exploration to local refinement. Existing seed generation approaches address neither: grammar-based and format-driven methods produce structurally valid inputs with no sink awareness, while LLM-based methods either lack sink targeting or inherit static analysis limitations through one-shot prompting. We present SeedSmith, an agentic LLM pipeline that replicates a security analyst's workflow: starting from a sink, it iteratively explores the codebase, resolves indirect calls, identifies crash preconditions, and synthesizes concrete inputs that satisfy them. Because SeedSmith operates as a seed generation front-end, its seeds are fuzzer-agnostic and improve any downstream mutation-based fuzzer without modification. On Magma, fuzzers using SeedSmith seeds achieve geometric mean crash-time speedups of 11.51 times (AFL++) to 14.66 times (AFLGo) over default seeds. On ARVO, SeedSmith enables fuzzers to trigger 16 previously unreachable bugs spanning 10 projects with diverse input formats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.13305v1">Accuracy Without Grounding: Diagnosing Visual Dependency Dissociation in Video LLM Benchmarks</a></div>
    <div class="paper-meta">
      📅 2026-07-14
      | 💬 Accepted, ACM International Conference on Multimedia 2026 (ACM MM)
    </div>
    <details class="paper-abstract">
      Benchmark accuracy in video large language models (LLMs) is often treated as evidence of visual understanding. We audit this assumption across twenty models spanning 2-78B parameters and ten architecture families. We introduce the Visual Dependency Gap (VDG), the difference in per-question correctness between original-video and black-screen conditions. Paired McNemar tests on MVBench show that accuracy and visual dependency are separable: models differ on original video (p = 0.0003) but not on black screens (p = 0.53). Across models, task-type rankings are stable: Attribute Perception is strongly visual, whereas Temporal Reasoning approaches the language-only baseline. A diagnostic ladder from black screen to single frame, shuffled frames, and original video reveals that frame diversity supplies most of the visual benefit, while temporal order contributes near-zero accuracy across sixteen open-weight models. An ablation from 0.5 to 24 FPS rules out sparse sampling as the cause. H.264 experiments further show that stable aggregate accuracy conceals bidirectional question-level answer flips. The diagnostic also generalizes to four API-accessed models, whose VDG values range from 0.025 to 0.315. These results motivate VDG as a standard audit for whether video benchmarks measure visually grounded capability. Code is available at https://github.com/JaeLee18/accuracy-without-grounding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.13304v1">Where Does the Noise Come From? A Variance-Components Decomposition of Non-Determinism in LLM Brand Answers</a></div>
    <div class="paper-meta">
      📅 2026-07-14
      | 💬 18 pages, 6 tables, 3 code listings
    </div>
    <details class="paper-abstract">
      Teams measuring whether large language models (LLMs) recommend a brand face a reproducibility problem: ask the same question twice and the answer moves. Practice resamples each prompt a few times (commonly five) and averages, treating within-prompt resampling as the source of the noise. But a measured brand score moves for at least four separable reasons: within-prompt resampling, prompt paraphrase, model identity, and query language. We specify a crossed random-effects (generalizability-theory) decomposition that partitions the total variance of a response-level brand outcome into these four sources, and embed the components in a decision-study allocation that returns how many repeats, paraphrases, models, and languages to buy for a target reliability. We apply it to a fully crossed corpus of 12,933 LLM responses on 20 Central and Eastern European brands, 8 languages, and 3 models (GPT-5.2 and Gemini 3 Flash in parametric mode, Perplexity in grounded retrieval), with a stability subset of 1,435 cells resampled about five times. The outcome is per-response multilingual sentiment polarity. Query language is the largest systematic facet (26.5% of the variance of one response) against 1.5% for brand identity (ICC 0.0146), so a single AI answer carries almost no brand-discriminating signal. Once a cell term isolates pure resampling, resampling is 34.8% of variance and the brand-in-context interaction 29.6%; brand-by-language is 8.6% (a bilingual penalty) while brand-by-model and brand-by-prompt are near zero. Per unit of query budget, adding languages and models reduces relative-error variance far more than adding repeats: a repeat past the fifth reduces it by only 0.0003. Brand-ranking reliability stays low, near 0.01 for a single answer and about 0.36 at the full crossed design, so reliability is bought by spreading across languages and models, not by repeating one prompt.
    </details>
</div>
