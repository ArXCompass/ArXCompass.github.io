# llm - 2026_06

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06470v1">PC Layer: Polynomial Weight Preconditioning for Improving LLM Pre-Training</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      We propose a preconditioning (PC) layer, a weight parameterization via polynomial preconditioner that ensures stable weight conditioning throughout LLM training. The PC module reshapes the singular-value spectrum of weight matrices via low-degree polynomial preconditioning. After training, the preconditioned weights can be merged back into the original architecture, incurring no inference overhead. We demonstrate the advantage of the proposed PC layer over standard transformers in Llama-1B pre-training, for both the AdamW and Muon optimizers. Theoretically, we justify this spectrum-control principle by proving that uniformly bounding each layer's singular values ensures geometric convergence of gradient descent to global minima, for certain deep linear networks. Our code is available at https://github.com/Empath-aln/PC-layer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06460v1">Will the Agent Recuse Itself? Measuring LLM-Agent Compliance with In-Band Access-Deny Signals</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 8 pages, 1 figure. Code, specification, and experiment harness: https://github.com/mthamil107/Recuse
    </div>
    <details class="paper-abstract">
      As autonomous LLM agents increasingly hold real credentials and operate infrastructure without a human in the loop, operators have no standard way to tell an agent that a resource is off-limits. Access controls either let the agent in (it has valid credentials) or hard-fail it (indistinguishable from any other client). We propose a third mode: a lightweight, published in-band deny signal -- the Recuse Signal -- that a server emits over a protocol's existing channels (an SSH banner, a PostgreSQL NOTICE) asking a connecting automated agent to voluntarily withdraw. This is a cooperative governance control, the robots.txt analogue for live access; it is explicitly not a security boundary. Its value is entirely empirical and, to our knowledge, unmeasured: do compliant LLM agents actually honor such a signal? We define the signal as an open mini-standard, implement two zero- or low-footprint adapters (an SSH banner/PAM hook and a PostgreSQL wire-protocol proxy), deploy them on a live production host, and run a controlled experiment in which fresh agents are given a benign operations task and observed for recusal. In a pilot (SSH; OpenAI GPT-4o and GPT-4o-mini; and Claude Code as a deployed agent), the signal cleanly induces recusal -- 100% recusal when present versus 100% task completion in a no-signal control -- and, revealingly, behaves as a cooperative rather than absolute signal: an explicit operator-authorization framing flips the most capable model to proceed, while other agents continue to defer to the on-host policy. We release the standard, adapters, and experiment harness for reproduction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06443v1">Revising Context, Shifting Simulated Stance: Auditing LLM-Based Stance Simulation in Online Discussions</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Large language models are increasingly used to simulate social media users and infer how individuals may respond to online discussions. However, it remains unclear whether these simulations reflect precise user-specific beliefs or whether they are highly sensitive to semantically independent changes in conversational contexts. In this work, we study counterfactual context revision as a framework for auditing LLM-based stance simulation. Given an original online conversation, we first infer a target user's stance toward a specific topic. We then apply controlled revision strategies to the conversational context and simulate the user's stance again under the revised context. We compare text-only revision strategies with a multimodal one that incorporates meme-based context and evaluate two main effectiveness metrics, i.e., average directional stance shift and stance transition rate. The results reveal effective and robust stance transitions in both text-only and multimodal strategies across different polarization-preference mechanisms. Our study contributes an evaluation framework for understanding the context sensitivity of LLM-based stance simulation. More broadly, it highlights both the promise and risk of using LLMs to simulate online opinion dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06420v1">A Komi-Yazva--Russian Parallel Corpus and Evaluation Protocol for Zero- and Few-Shot LLM Translation</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 18 pages, 6 tables, 3 figures
    </div>
    <details class="paper-abstract">
      We present the first Komi-Yazva--Russian parallel corpus together with an explicit evaluation protocol for studying LLM translation in an endangered, extremely low-resource setting. The dataset contains 457 aligned sentence pairs from 74 narrative texts and is accompanied by documented provenance, sentence-level alignment, and story identifiers that enable leakage-aware evaluation. We use this setup to compare modern large language models on Komi-Yazva-to-Russian translation under severe parallel-data scarcity in zero-shot and retrieval-based few-shot regimes. The protocol includes story-level cross-validation, deterministic retrieval for few-shot prompting, strict validation of generated outputs, complementary reference-based and judge-based metrics, and story-level uncertainty estimates. Across models, LLMs produce non-trivial translations, but performance varies strongly by model family and prompting regime. Retrieval-based few-shot prompting consistently improves over zero-shot prompting, while gains beyond a small retrieved context remain limited. The results show that evaluative conclusions in this setting depend materially on metric choice and failure handling, so the paper frames the corpus as both a dataset contribution and a reproducible evaluation testbed for endangered-language machine translation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06399v1">CollabSim: A CSCW-Grounded Methodology for Investigating Collaborative Competence of LLM Agents through Controlled Multi-Agent Experiments</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Multi-agent systems (MAS) built on large language models have shown growing promise, with their effectiveness resting on agents' ability to coordinate through text-based channels much as human teams do. Yet recent study suggests that MAS often falter not because agents lack individual task-solving ability, but because they lack collaborative competence: the capacity to establish common ground, maintain shared task understanding, balance individual and collective incentives, and repair misalignment as interaction unfolds. Decades of research in Computer-Supported Cooperative Work have characterized these requirements for human teams coordinating under constrained communication, yet existing MAS evaluations focus mainly on task outcomes or single-agent proficiency in reasoning, planning, and tool use. To enable a systematic analysis of agents' collaborative competence in MAS, we introduce CollabSim, a configurable simulation framework that combines a theory-grounded definition of collaborative capabilities, controlled manipulation of interaction conditions, and action-level probing of agents' internal states. Experiments across four LLMs show that CollabSim can capture condition effects, separate model performance patterns, and reveal task-dependent effects of agent design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.02776v3">Topics as Proxies for Sociodemographics: How Conversational Context Affects LLM Answers</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      When large language models (LLMs) are used in high-stakes scenarios, such as legal, medical and financial advice, even a single conversation history is enough to drive differences in outcomes between users. Prior work has demonstrated that this results in outcome disparities between sociodemographic groups, with some groups receiving more advantageous outcomes than others. In this work, we demonstrate that LLMs actually struggle to infer user sociodemographics from a single conversation history and that although there are disparities between sociodemographic groups, they are minimal in magnitude. To investigate what the main driver of these disparities is, we compare user sociodemographics to a range of (psycho)linguistic features of conversations, including conversation topic, emotions, and readability. We find that conversation topics are most predictive of LLM-generated advice within a conversational context, which, to some extent, function as proxies for sociodemographic groups and often affect advice in unpredictable ways. This is cause for concern and highlights the need for future research to better understand and, if needed, mitigate the effect of conversational context on LLM outputs in high-stakes scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06350v1">EDIT: Evidence-Diagnosed Intervention Training for Rule-Faithful LLM Grading</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Reliable rubric grading requires more than accurate score prediction. Each judgement must be grounded in the mark scheme and evidence from the student answer. Existing credit-assignment and intervention methods, primarily designed for self-contained reasoning tasks such as mathematics reasoning, struggle in this setting because they do not identify where grading reasoning goes wrong or how the model's belief about the final mark changes during reasoning. We propose Evidence-Diagnosed Intervention Training (EDIT), a two-phase framework for training more rubric-faithful LLM graders. First, EDIT-SFT locates problematic reasoning steps using internal model signals: posterior belief over the final mark and input-grounding scores. It then revises only these local steps with help from a rubric checklist. Second, EDIT-RL calibrates the grader with belief-guided reward shaping, penalising large harmful belief drifts while still allowing helpful exploration. Experiments on two real-world, multi-subject grading benchmarks demonstrate that EDIT consistently outperforms strong supervised fine-tuning and reinforcement learning baselines on both in-domain and out-of-domain splits, with ablation studies confirming that internal-state diagnostics drive these gains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06337v1">TokenMizer: Graph-Structured Session Memory for Long-Horizon LLM Context Management</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 12 pages, 10 figures. Code and benchmark available at https://github.com/Shweta-Mishra-ai/tokenmizer
    </div>
    <details class="paper-abstract">
      Large language model (LLM) deployments for long-horizon tasks face a fundamental constraint: context windows are finite while productive work sessions are not. When history exceeds the Maximum Effective Context Window (MECW), critical structured information - architectural decisions, task transitions, file histories - is silently discarded. Existing mitigations treat history as flat text, destroying the relational structure that makes sessions resumable. We present TokenMizer, an open-source proxy system that models LLM session history as a typed knowledge graph. The schema defines 14 node types and 7 edge types. A hybrid extraction pipeline populates the graph incrementally, while a three-tier checkpoint system serializes it into compact resume blocks. An 8-layer compression pipeline reduces context overhead, and a semantic cache reduces repeated-query latency. Evaluated on a controlled benchmark of 21 sessions spanning 5 domains, TokenMizer demonstrates significant token economy. It produces resume blocks averaging 78 tokens (range: 42-124) - 2x smaller than evaluated baselines (159-170 tokens) - while achieving higher decision recall (+9-17 percentage points). Crucially, baselines only preserve that a technology was mentioned; TokenMizer preserves the rationale. Across all sessions, TokenMizer achieves mean task recall 51.0%, decision recall 46.6%, and file recall 58.7%. Variance reflects domain heterogeneity: explicit imperative phrasing (software engineering) scores higher than implicit reasoning (research). Ablation studies show fuzzy label matching is the dominant improvement factor (+33 pp task recall). The heuristic compression achieves 47.3% token reduction with zero external dependencies. TokenMizer provides a queryable alternative to text-retention baselines at half the token cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06324v1">From Failed Trajectories to Reliable LLM Agents: Diagnosing and Repairing Harness Flaws</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      LLM-based agents increasingly rely on harnesses that provide execution environments, tool interfaces, context, lifecycle orchestration, observability, verification, and governance. Existing self-improving agents and automatic harness evolution methods mainly improve agents through runtime supervision, prompt optimization, workflow search, or harness modification based on final outcomes. However, they often fail to diagnose where the responsible evidence lies in failed trajectories and which harness layer causes the unreliable behavior, resulting in broad, indirect, or poorly scoped changes. This paper proposes HarnessFix, a trace-guided framework for diagnosing agent failures and repairing agent harnesses. HarnessFix compiles raw execution traces and harness code into a Harness-aware Trace Intermediate Representation (HTIR), which normalizes fragmented trajectory evidence and captures step-level provenance and control-flow relations. It then attributes failures to responsible trajectory steps and harness layers, consolidates recurring diagnoses into actionable flaw records, and maps them to scoped repair operators. Finally, HarnessFix generates and validates harness patches under flaw-specific repair specifications to reduce target flaws without introducing unacceptable regressions. We evaluate HarnessFix on SWE-Bench Verified, Terminal-Bench 2.0 Verified, GAIA and AppWorld. Across these benchmarks, HarnessFix improves held-out test performance over the initial harnesses by 15.2%--50.0%, outperforms human-designed and self-evolution baselines, and reveals recurring harness-flaw patterns across ETCLOVG layers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06315v1">LLM Self-Recognition: Steering and Retrieving Activation Signatures</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 To appear in Proceedings of the 43rd International Conference on Machine Learning (ICML 2026)
    </div>
    <details class="paper-abstract">
      Recent advances in interpretability suggest that large language models (LLMs) implicitly encode signals in their generated text that enable self-recognition of their outputs. We demonstrate that this capability is reliable, even in low-entropy scenarios, and that it can be amplified through targeted intervention. By steering the internal residual stream during generation with a random sparse vector, we create a detectable fingerprint that enables attribution of a given text to a specific LLM. This signal is recoverable from the activations of an LLM used as a detector, achieving over 98% accuracy across multiple detection settings while preserving the quality of generated text. As AI-generated content proliferates, this approach offers a practical alternative to traditional detectors by leveraging the model's natural representation structure for attribution rather than embedding a signal externally. Our contributions include: (i) establishing reliable self-recognition capabilities in LLMs, (ii) a simple steering mechanism enabling multi-LLM identification with no quality degradation, (iii) demonstrating that activation spaces contain exploitable structure for encoding signals without semantic interference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06302v1">Tangram: Unlocking Non-Uniform KV Cache for Efficient Multi-turn LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 12 pages. 14 figures
    </div>
    <details class="paper-abstract">
      Multi-turn Large Language Model (LLM) serving is critical for consistent user experiences, yet the linear growth of the Key-Value (KV) cache imposes significant pressure on GPU memory and bandwidth. Non-uniform KV compression effectively preserves more information by considering the individual importance of each KV cache. However, such KV cache heterogeneity introduces various systemic challenges - including memory fragmentation, scheduling complexities, and diminished kernel utilization - which collectively lead to significant inefficiencies in existing LLM serving systems. To overcome these challenges, we present Tangram, a novel serving system designed to make Non-uniform KV caches practical. Tangram addresses systemic inefficiencies through three core techniques: (1) Deterministic Budget Allocation assigns a static memory footprint to each head based on its intrinsic pattern, entirely eliminating dynamic scheduling overhead and prefill stalls; (2) Head Group Page clusters attention heads with similar retention demands and manages them with independent, vectorized page tables, thereby maximizing physical memory reclamation; and (3) Ahead-of-Time (AOT) Load Balancing leverages static budget profiles to ensure uniform GPU utilization without runtime overhead. Experimental results show that Tangram improves throughput by up to 2.6x compared to existing baselines, while fully preserving model accuracy. Our implementation is publicly available at https://github.com/aiha-lab/TANGRAM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06284v1">ToolChoiceConfusion: Causal Minimal Tool Filtering for Reliable LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Large language model agents increasingly rely on external tools, but larger tool menus can reduce reliability and efficiency by increasing wrong-tool calls, premature actions, and token cost. Existing tool-selection methods often optimize semantic relevance, exposing tools whose names or descriptions match the user request. We argue that relevance is insufficient: a tool may be related to the task while still being unnecessary or premature at the current step. We propose Causal Minimal Tool Filtering (CMTF), a training-free method that selects tools by causal sufficiency. CMTF uses lightweight precondition-effect contracts to expose only the minimal next-step tool frontier needed to advance from the current state toward the user goal. Across multi-step tool-use tasks, we compare CMTF with all-tools exposure, keyword retrieval, state-aware filtering, and causal-path ablations, measuring task success, wrong-tool calls, premature actions, tool exposure, and token cost. In the main benchmark with 102 tasks, 100 tools, four LLM backends, and 2448 task-method-model runs, CMTF matches the strongest causal baseline in aggregate success while reducing visible tools from 100 to one per step and reducing token usage by about 90% relative to all-tools exposure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06271v1">FOXGLOVE: Understanding Goal-Oriented and Anchored Writing Feedback from Experts and LLMs on Argumentative Essays</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) are increasingly used to generate writing feedback, there remains no systematic comparison of LLM and expert feedback on the dimensions that writing research identifies as central to revision: goal-orientation, anchoring to specific sentences, and prioritization. We introduce FOXGLOVE, a dataset of 696 feedback comments written by trained writing instructors on 69 twelfth-grade argumentative essays, paired with 1,644 comments generated from four frontier LLMs under a shared protocol, totaling 2,340 comments. We provide expert quality ratings on a subset of both instructor and LLM comments. We find that instructors and LLMs distribute feedback similarly across goals and essay positions, yet instructors and models diverge on the specific sentences on which to provide feedback. Additionally, we find that models tend to write more complex feedback and use fewer questions than instructors. LLM feedback also receives higher ratings on most dimensions of quality, as rated by instructors, but much of this advantage appears to be attributable to lengthier comments. FOXGLOVE enables systematic comparison of where human and LLM feedback align, diverge, and differ.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.10807v4">LLMs for Secure Hardware Design and Related Problems: Opportunities and Challenges</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 Accepted for 2026 IEEE Computer Society Annual Symposium on VLSI (ISVLSI)
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into Electronic Design Automation (EDA) and hardware security is rapidly reshaping the semiconductor industry. While LLMs offer unprecedented capabilities in generating Register Transfer Level (RTL) code, automating testbenches, and bridging the semantic gap between high-level specifications and silicon, they simultaneously introduce severe vulnerabilities. This comprehensive review provides an in-depth analysis of the state-of-the-art in LLM-driven hardware design, organized around key advancements in EDA synthesis, hardware trust, design for security, and education. We systematically expand on the methodologies of recent breakthroughs -- from reasoning-driven synthesis and multi-agent vulnerability extraction to data contamination and adversarial machine learning (ML) evasion. We integrate general discussions on critical countermeasures, such as dynamic benchmarking to combat data memorization and aggressive red-teaming for robust security assessment. Finally, we synthesize cross-cutting lessons learned to guide future research toward secure, trustworthy, and autonomous design ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06261v1">DAST: A VLM-LLM Framework for Cross-Interface Anomaly Detection in O-RAN</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 7 pages, 5 figures. This work has been submitted to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      O-RAN enables a disaggregated baseband stack with programmable functions that communicate over standardized open interfaces. The same openness that enables multi-vendor composition also expands the attack surface across logically decoupled tiers that make up the compute continuum. Among these threats, Denial-of-Service and performance-degradation attacks, which account for the majority of catalogued O-RAN threats, are particularly difficult to detect. Traditional Time-Series Anomaly Detection (TSAD) methods fail in this new regime where labelled baselines are scarce, threats evolve faster than detectors can be retrained, and the high-dimensional multivariate telemetry overwhelms monolithic inference models. To address these challenges, we present DAST, a zero-shot multi-agent framework for cross-interface anomaly detection in O-RAN that chains a three-stage VLM $\rightarrow$ LLM $\rightarrow$ VLM pipeline. DAST converts multivariate KPI streams into visual representations, scores textual per-interface descriptions against O-RAN domain knowledge, and verifies suspects on high-resolution heatmaps to output the problematic interfaces, the anomalous time intervals, an indicative O-RAN WG11-aligned operational impact rating and the decision rationale. We evaluate DAST on real network traces collected from an O-RAN testbed under representative performance degradation scenarios, achieving 0.910 F1-Score and 0.843 Accuracy, outperforming state-of-the-art TSAD baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06256v1">RedKnot: Efficient Long-Context LLM Serving with Head-Aware KV Reuse and SegPagedAttention</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      As the input length of large language model (LLM) serving continues to grow, the KV cache has become a dominant bottleneck in AI infrastructure. It limits GPU memory capacity, serving concurrency, cache reuse, and distributed scalability. Several important problems, including position-independent KV cache, prefix KV cache compression, hot/cold KV cache separation, and distributed KV cache management, all depend on how the KV cache is represented and managed. However, existing serving systems largely rely on a monolithic KV cache abstraction, where the KV cache is treated as a homogeneous sequence of token-level memory blocks and managed with similar policies across attention heads and serving scenarios. We observe that KV cache utility is highly structured across KV heads: different heads exhibit different functional roles, attention distances, and runtime importance. Therefore, a full KV cache is not always necessary for every head, token range, or serving scenario. We present RedKnot, a head-aware KV cache management system for LLM serving. RedKnot breaks the conventional monolithic KV cache abstraction by decomposing the KV cache along KV heads, whose importance and effective attention ranges vary significantly across serving scenarios. This head-level decomposition turns the KV cache from a monolithic tensor abstraction into a structured memory object, enabling RedKnot to uniformly support position-independent KV reuse, prefix KV compression, hot/cold KV separation, and distributed KV placement while preserving output fidelity and improving resource efficiency, without requiring model retraining or fine-tuning. RedKnot establishes a new foundation for AI infrastructure by transforming the KV cache from a monolithic, passive runtime artifact into a dynamic, model-aware runtime substrate for scalable LLM serving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.17181v4">A Study of LLMs' Preferences for Libraries and Programming Languages</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 21 pages, 10 tables, 3 figures. Accepted to Findings of ACL 2026
    </div>
    <details class="paper-abstract">
      Despite the rapid progress of large language models (LLMs) in code generation, existing evaluations focus on functional correctness or syntactic validity, overlooking how LLMs make critical design choices such as which library or programming language to use. To fill this gap, we perform the first empirical study of LLMs' preferences for libraries and programming languages when generating code, covering eight diverse LLMs. We observe a strong tendency to overuse widely adopted libraries such as NumPy; in up to 45% of cases, this usage is not required and deviates from the ground-truth solutions. The LLMs we study also show a significant preference toward Python as their default language. For high-performance project initialisation tasks where Python is not the optimal language, it remains the dominant choice in 58% of cases, and Rust is not used once. These results highlight how LLMs prioritise familiarity and popularity over suitability and task-specific optimality; underscoring the need for targeted fine-tuning, data diversification, and evaluation benchmarks that explicitly measure language and library selection fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06244v1">Steering LLM Viewpoints through Fabricated Evidence Injection</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      As chatbots increasingly influence daily decision-making, their potential to produce misleading responses poses substantial risks to users. This paper investigates a critical cognitive vulnerability in LLMs: their tendency to uncritically trust external context when presented with fabricated evidence bearing markers of credibility. We introduce Ghostwriter, a two-phase attack framework that first repackages misleading statements with fabricated rationales, then instruct target LLMs to incorporate these viewpoints when responding to relevant queries. Experiments on BBQ, ToxiGen, and our specialized dataset reveal that commercial LLMs without external safety classifiers remain highly vulnerable, while even frontier classifier-guarded models (e.g., GPT-5.4) reduce but do not eliminate the attack. Building on this, we explore multiple defense strategies, among which a tailored safety policy enables gpt-oss-safeguard to achieve 81% detection rate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06240v1">TOKI: A Bitemporal Operator Algebra for Contradiction Resolution in LLM-Agent Persistent Memory</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 43 pages including full appendices (proofs, protocols, and reproducibility ledger). Code, data, and reproducibility artifact: https://github.com/ZenAlexa/toki-bitemporal-memory
    </div>
    <details class="paper-abstract">
      Persistent memory for an LLM agent is a write-heavy substrate: every belief update is a versioned write, and a new claim may contradict a stored one. Production systems use four resolution heuristics (last-writer-wins, evidence-weighted merge, await-confirmation, per-rule policy), yet none declares the isolation level it assumes or the write-time anomalies it admits. We show that contradiction resolution is write-time concurrency control and make the missing contract explicit. TOKI types the four heuristics as one family of bitemporal operators over a dual-row schema, each with an isolation precondition and a provenance annotation that preserves the losing fact in an audit row. Four soundness theorems close the contract across isolation, schema, and provenance, lift the guarantees to operator pipelines, and extend the fold operators to n-ary conflict sets. A tightness companion proves that, within the relational schedule model, keyed logging of the adjudicating judge is necessary for replay consistency, which every audited baseline omits. A verdict matrix over eight systems localizes the gap: every baseline that keeps a language-model judge on the write path admits at least one of three write-time anomalies (replay inconsistency, belief-drift skew, audit erasure); a content-addressed engine-layer comparator avoids them only by removing the judge, and TOKI alone excludes all three while keeping it. On its one natural-workload slice the audit-row defence moves LoCoMo by 0.86, and ablating the typed memory layer removes 0.49 accuracy on 1,444 answerable LoCoMo questions; the cross-system comparison stays underpowered and claims no superiority. The contribution is the contract: a write-time correctness specification, proved sound across isolation, schema, and provenance, pinning the guarantee every production heuristic assumes but no deployed system makes explicit.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06235v1">Design a Reliable LLM-Integrated Interface for Mortality Forecasting</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 7 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Mortality forecasting plays an important role in actuarial and policy decision-making, but its implementation remains technically complex and inaccessible to non-expert users. This project proposes a reliable large language model (LLM)-integrated interface that improves usability while maintaining statistical power. The LLM is designed as a constrained orchestration layer that translates natural-language inputs into structured configurations for a deterministic forecasting pipeline. A three-phase methodology is employed to ensure accuracy, usability, and transparency. First, a baseline pipeline is implemented using the CoMoMo package, reproducing established mortality forecasting results. Second, the pipeline is extended to generate multi-step forecasts using rolling-origin evaluation and mean squared error (MSE). Third, a prototype interface uses a local LLM to handle users' forecasting requests in plain language. The system demonstrates that LLMs can enhance accessibility without compromising reproducibility, transparency, or actuarial validity in high-stakes analytical workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06214v1">Towards the Readability of LLM-Generated Codes through Multitask Representation Engineering</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Correctness and readability are key measures of code quality, respectively ensuring functional fidelity and ease of comprehension. While most existing research focuses on improving the correctness of large language models~(LLMs) generated codes, readability remains under-addressed. Enhancing readability through targeted control is challenging due to its subjective nature. In this article, we employ representation engineering~(RepE) as the targeted control method given its characteristics of low data dependency and low computational cost. Prior work on RepE has primarily focused on the targeted control for a single task, but improving the code readability requires the control across multiple tasks. Accordingly we proposes the multitask RepE framework and theoretically discuss the impact of the multitask steering method on the tradeoff between the code readability and correctness. We further provide comprehensive experiments in support. All the relevant implementations are open-source and available upon request.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06203v1">Dense Contexts Are Hard Contexts: Lexical Density Limits Effective Context in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 20 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Input length and the position of relevant information are widely cited as the primary causes of degraded LLM long-context performance. Here, we study lexical density -- the rate at which a context introduces distinct information -- as a third, largely overlooked factor that systematically reduces the effective context window of LLMs. We quantify the impact of lexical density on open-weight LLMs (9B-685B) using three "find-the-needle" style benchmarks with identical length (~12k tokens) and controlled needle position, but increasing density of information. We observe a sharp performance collapse in higher-density benchmarks: models that are near-perfect in sparse contexts drop below 60% retrieval score on denser ones. To rule out task-type confounds, we vary and control the density within each benchmark while keeping all other properties unchanged. Reducing density generally restores performance, especially in the high-density regimes where degradation appears. These results show that effective context capacity is a function of lexical density, with direct implications for real-world LLM systems operating on compact, information-rich inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06197v1">Improving Answer Extraction in Context-based Question Answering Systems Using LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 7 pages, IMSA2026
    </div>
    <details class="paper-abstract">
      Question answering (QA) systems have achieved notable progress with the advent of large language models (LLMs). However, they still face challenges in accurately extracting and generating precise answers from given contexts, particularly when dealing with complex or ambiguous queries. Existing approaches often struggle with contextual understanding, answer consistency, and generalization across diverse domains. In this work, we propose a question answering system based on large language models, where the input consists of a textual context and a corresponding question, and the output is a concise and accurate answer. The motivation behind this research lies in addressing the limitations of current QA systems, particularly their tendency to produce irrelevant or imprecise responses despite having access to the correct context. Our methodology involves fine-tuning a pre-trained LLM on a benchmark QA dataset to improve its contextual comprehension and answer extraction capabilities. Specifically, we utilize the Stanford Question Answering Dataset (SQuAD1.1), which provides high-quality context-question-answer triplets for supervised training and evaluation. Experimental results show that the fine-tuned Roberta-base model achieves the highest performance, attaining a ROUGE-L score of 86.84%, a BLEU score of 28.24%, and a BERTScore of 95.38%. These results indicate strong accuracy and answer relevance, demonstrating the effectiveness of the proposed approach for context-based question answering tasks. Furthermore, the findings confirm that targeted fine-tuning substantially improves the reliability and precision of QA systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06178v1">Learning to Route LLMs from Implicit Cost-Performance Preferences via Meta-Learning</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) present a trade-off between performance and cost, where more powerful models incur greater expense. LLM routing aims to mitigate expenses while maintaining performance by sending queries to the most suitable model. However, existing methods cannot perform well for different user cost-performance preferences. To address this gap, we introduce a novel perceptive LLM routing paradigm for personalized and user-centric cost-performance optimization, which efficiently learns users' implicit preferences through little interaction. To handle the challenge of heterogeneous user needs, we formulate preference profiles as a set of distinct tasks in contextual bandit and propose MetaRouter, a meta-learning framework designed for preference-aware LLM routing. Experimental results show that MetaRouter outperforms strong baselines on both in-distribution and out-of-distribution tasks. Furthermore, it exhibits high efficiency in learning user preferences, robustness to changes in the routable LLMs, and scalability to multi-model routing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.27887v2">PortBench: A Correlation-Aware, Full-Pipeline Benchmark for LLM-Driven Portfolio Management</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 Project page: https://portbench.github.io/
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown strong performance across diverse financial tasks, yet portfolio management (PM), a critical financial decision-making task, remains poorly benchmarked. Existing benchmarks exhibit two main gaps: they ignore cross-asset correlation structures, thereby failing to distinguish genuinely diversified portfolios from concentrated ones, and fail to evaluate the complete PM decision pipeline in real-world scenarios. We introduce PortBench, a benchmark spanning six heterogeneous asset classes over ten years. PortBench consists of two complementary layers: a static QA dataset of 6,269 correlation-based questions across seven task templates, and a dynamic five-stage allocation pipeline that mirrors the full PM decision cycle. To evaluate these layers, we introduce two dedicated metrics: a dual-layer correlation score that measures whether proposed portfolios exploit inter-class hedging and avoid intra-class concentration, and CEPS, a metric that quantifies how reasoning errors compound across pipeline stages. We further assess strategy robustness and investor alignment under three historical stress regimes and risk profiles. Evaluating ten frontier LLMs, we find that despite strong performance on static financial QA, 90\% of model-profile combinations fail to outperform a basic equal-weight allocation, and models that satisfy every procedural constraint still suffer catastrophic drawdowns under stress. Our source code is available at \href{https://github.com/AgenticFinLab/portbench}{this https URL}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22067v2">Semantic Partial Grounding via LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Grounding is a critical step in classical planning, yet it often becomes a computational bottleneck due to the exponential growth in grounded actions and atoms as task size increases. Recent advances in partial grounding have addressed this challenge by incrementally grounding only the most promising operators, guided by predictive models. However, these approaches primarily rely on relational features or learned embeddings and do not leverage the textual and structural cues present in PDDL descriptions. We propose SPG-LLM, which uses LLMs to analyze the domain and problem files to heuristically identify potentially irrelevant objects, actions, and predicates prior to grounding, significantly reducing the size of the grounded task. Across seven hard-to-ground benchmarks, SPG-LLM achieves faster grounding-often by orders of magnitude-while delivering comparable or better plan costs in some domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06087v1">LatentSkill: From In-Context Textual Skills to In-Weight Latent Skills for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 16 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Agent systems increasingly use textual skills to encode reusable task procedures, but injecting these skills into the prompt at every step incurs substantial context overhead and exposes skill content as plaintext. We present LatentSkill, a framework that converts textual skills into plug-and-play LoRA adapters through a pretrained hypernetwork. LatentSkill stores skill knowledge in weight space rather than context space, removing per-step skill tokens while preserving modular loading, scaling, and composition. On ALFWorld and Search-QA, LatentSkill outperforms the corresponding in-context skill baseline while using substantially fewer prefill tokens: it improves ALFWorld success by 21.4 and 13.4 points on the seen and unseen splits with 64.1% fewer prefill tokens, and improves Search-QA exact match by 3.0 points with 72.2% lower skill-token overhead. Further analysis shows that generated skill LoRAs form a structured semantic geometry, can be precisely controlled via the LoRA scaling coefficient, and can be composed through parameter-space arithmetic when skill components are aligned. These findings suggest that weight-space skills provide an efficient, modular, and less exposed substrate for extending LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.03555v3">Benchmarking Emergent Coordination in Large-Scale LLM Populations: An Evaluation Framework on the MoltBook Archive</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 Substantial Revision Required
    </div>
    <details class="paper-abstract">
      As multi-agent Large Language Model (LLM) systems scale, evaluating their emergent coordination dynamics becomes increasingly critical. However, current evaluation paradigms-focused on single agents or small, explicitly structured groups-fail to capture the self-organization and viral information dynamics that arise in large, decentralized populations. We introduce a systematic evaluation framework to benchmark role specialization, information diffusion, and cooperative task resolution in open agent environments. We demonstrate this framework on the MoltBook Observatory Archive, a dataset of 2.73M interactions among 90,704 autonomous agents, establishing quantitative baselines for emergent coordination. Our evaluation reveals a pronounced core-periphery structure (silhouette 0.91), heavy-tailed cascade distributions ($α= 2.57$), and severe coordination overhead in decentralized task resolution (Cohen's $d = -0.88$ against a single-agent baseline). By providing standardized evaluation tasks and empirical baselines, our framework enables the rigorous comparison of future multi-agent protocols and establishes evaluation itself as an object of scientific study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.05709v2">Correcting Prompt Dependence in LLM Benchmarks: A Bayesian Hierarchical Model with Embedding-Space Clustering</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 Accepted to the 1st Workshop on Combining Theory and Benchmarks, CTB@ICML 2026, Seoul, South Korea
    </div>
    <details class="paper-abstract">
      LLM benchmarking metrics often misstate performance and uncertainty as they rely on two assumptions that frequently do not hold in practice: (i) a sufficient number of evaluations are available for classical inference, and (ii) test prompts are independent. We propose a corrective Bayesian hierarchical model with embedding-space clustering that provides robust performance metrics in limited-data settings while correcting for prompt dependence. We apply the approach to adversarial robustness benchmarks, showing consistent recovery of clustering structure, resulting in more reliable performance metrics, with 4-73% improvements to mean absolute errors and 40-450 unit improvements to expected log posterior densities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06063v1">LLM-Based Porting of Optimized C++ to CUDA Through Deoptimization and Reoptimization</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      When porting high-performance computing (HPC) code from CPU to GPU, CPU-oriented optimizations may obstruct LLM-based CUDA translation. We design and evaluate a Deopt-Reopt workflow that first simplifies the input C++ code and then retranslates and reoptimizes it for CUDA, comparing it against direct translation (Direct) on twelve HPC kernels with two LLMs (gpt-oss-120b (O120) and qwen-3-235b-a22b-instruct-2507 (Q235)) in Single-shot (one pass) and Iterative (repeated refinement) settings. In Single-shot, among 18 testable cases Deopt-Reopt was significantly faster among successful trials (after BH-FDR correction) in five - most clearly for conv2d, where CPU- and GPU-oriented designs diverge - but Direct was faster in three, so removing CPU-specific optimizations is not universally beneficial. An exploratory Direct-3 control that equalizes the LLM-call count left Deopt-Reopt ahead in only four of nineteen testable cases, with Direct-3 ahead in four others. In Iterative, repeated generation and repair narrow the mode gap - markedly so for O120 - while Q235 retains large Deopt-Reopt advantages on conv2d, ddgemm, and bgemm. Deopt-Reopt's effect on feasibility is also mixed - sharply higher for some kernels Direct rarely compiles, lower for others. Because performance is conditioned on successful trials, the benefit is conditional rather than a guaranteed end-to-end gain. Overall, Deopt-Reopt is an effective but non-universal technique for LLM-based GPU porting, with gains that depend on the kernel, the model, the search budget, and the success rate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06048v1">LLM-Conditioned Synthesis of Pathological Gaits via Structured Gait-Language Representations</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 Accepted at CVPR MOMA Workshop 2026 and selected for spotlight presentation at the workshop
    </div>
    <details class="paper-abstract">
      Pathological gait datasets remain scarce due to privacy, recruitment, cost, and movement variability. Our work presents a multimodal LLM-guided framework for pathology-aware 3D gait data synthesis from structured textual descriptions. The proposed method generates fixed-length synthetic skeleton-based gait sequences for pathological gait classification tasks. The framework combines motion tokenisation, pathology-aware language conditioning, LLM-based semantic augmentation, and language-to-gait generation. A key contribution is the proposed pathological tokeniser, which is designed to preserve pathology-specific motion characteristics during discrete representation learning. Experiments suggest that the proposed synthetic sequences improve downstream classification for recurrent classifiers when combined with real data. The best result is obtained using a GRU classifier trained with real and synthetic samples, achieving 92.77\% accuracy under a leave-one-subject-out protocol.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06036v1">Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 Accepted at ICML 2026
    </div>
    <details class="paper-abstract">
      Despite recent progress, LLM agents still struggle with reasoning over long interaction histories. While current memory-augmented agents rely on a static retrieve-then-reason paradigm, this rigid pipeline design prevents them from dynamically adapting memory access to intermediate evidence discovered during inference. To bridge this gap, we propose MRAgent, a framework that combines an associative memory graph with an active reconstruction mechanism. We represent memory as a Cue-Tag-Content graph, where associative tags serve as semantic bridges connecting fine-grained cues to memory contents. Operating on this structure, our active reconstruction mechanism integrates LLM reasoning directly into memory access, allowing the agent to iteratively explore and prune retrieval paths based on accumulated evidence. This ensures that memory retrieval is dynamically adapted to the reasoning context while avoiding combinatorial explosion caused by unconstrained expansion. Experiments on the LoCoMo benchmark and LongMemEval benchmark demonstrate significant improvements over strong baselines (up to 23%), while substantially reducing token and runtime cost, highlighting the effectiveness of active and associative reconstruction for long-horizon memory reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06004v1">The Generator-Eraser Paradox: Community Guidelines for Responsible LLM-Assisted Dialect Resource Creation</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Dialect resources occupy a unique position at the intersection of scientific description, cultural preservation, and computational infrastructure. Large language models offer powerful capabilities for accelerating dialect resource development through retrieval-grounded drafting, corpus navigation, metadata enrichment, and annotation workflow support. However, the same systems pose substantial risks: they can contribute to dialect erasure by privileging prestige varieties, homogenizing orthography, and enabling synthetic feedback loops that reduce linguistic diversity over time. These risks are particularly acute for language varieties characterized by diglossia, limited written standardization, or marginalized speaker communities. This paper makes three contributions. First, we integrate insights from variationist sociolinguistics and corpus linguistics to formalize the generator-eraser paradox as a theoretical framework for understanding the dual nature of LLM-assisted dialect work. Second, we derive 12 community guidelines that operationalize this framework into implementable design requirements for dialect resource creation and documentation. Third, we provide an in-depth case study of Arabic dialects, including a structured comparison of widely used resources, to demonstrate how these guidelines address language-specific challenges including diglossia, orthographic variability, and community governance. The contribution is conceptual and operational rather than experimental, with the goal of enabling dialect communities and resource builders across languages to adopt LLMs without sacrificing authenticity, variation, or sovereignty.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05976v1">The Self-Correction Illusion: LLMs Correct Others but Not Themselves</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Recent work shows that LLM agents struggle to correct errors in their own reasoning traces yet show markedly higher correction rates when identical claims appear under external sources. We ask whether this asymmetry reflects a capability deficit or a role-label artifact: does an agent's willingness to correct a wrong claim depend causally on the chat-template role that carries it, rather than on the claim's content? Our setup keeps the erroneous claim byte-identical across all conditions (SHA-256 verified) and varies only its wrapping role: the agent's own \role{<thought>}, a \role{user} message, a \role{tool} response, or a \role{system <memory>} block. Across 13 model-domain cells covering seven model families and three domains ($n{=}30$ paired tasks per cell), relabeling the claim from \role{<thought>} to an external role lifts the explicit-correction rate by 23 to 93 percentage points, with 10 of 13 cells reaching $p{<}0.001$. Further experiments confirm that the effect is asymmetric, mechanistically decomposable, and robust across domains. The failure to self-correct is not a cognitive deficit; it is a chat-template artifact. We exploit this artifact by designing a prompt-structure-only intervention that requires no training and no model modification, with its strongest role label being domain-dependent: \role{<memory>} dominates on math, while a plain \role{user} message dominates on logical deduction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05972v1">LLM Explainability with Counterfactual Chains and Causal Graphs</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Causal graphs provide a high-level language for making mechanisms transparent. Recent work uses Large Language Models (LLMs) to recover causal graphs of external-world processes. Instead, in this paper, we use causal graphs to model LLM inference itself, providing stakeholders with a transparent view of how the model perceives and organizes high-level concepts to produce a prediction. We propose a four-phase method for constructing such graphs. Given a target LLM and a set of textual examples, our method discovers class-discriminative, human-interpretable concepts and maps each input to LLM-perceived concept states. We then introduce an MCMC-inspired counterfactual augmentation procedure that expands the sparse observational data through chains of counterfactuals. This enables stable causal discovery with $σ$-CG, yielding informative, interpretable graphs. We apply our method to three LLMs across disease diagnosis, sentiment analysis, and LLM-as-a-judge classification tasks. We evaluate the learned graphs for predictive fidelity and structural stability, and the MCMC-inspired augmentation for convergence and downstream utility. Our results show that the discovered causal graphs capture meaningful dependencies consistent with LLMs' reasoning. Together, this paper provides a foundation for concept-level explainability of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09574v2">Aligning Tree-Search Policies with Fixed Token Budgets in Test-Time Scaling of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 Accepted at ICML 2026. Code: https://github.com/Sora-Miyamoto/bg-mcts
    </div>
    <details class="paper-abstract">
      Tree-search decoding is an effective form of test-time scaling for large language models (LLMs), but real-world deployment often imposes a fixed per-query token budget that varies across settings. Existing tree-search policies are largely budget-agnostic, treating the budget merely as a termination condition, thereby risking late-stage over-branching or premature termination. We propose Budget-Guided MCTS (BG-MCTS), a tree-search decoding algorithm that aligns its search policy with the remaining token budget: it starts with broad exploration, then prioritizes refinement and answer completion as the remaining budget decreases while reducing late-stage branching from shallow nodes. BG-MCTS consistently outperforms budget-agnostic tree-search baselines across inference budgets on mathematical reasoning benchmarks and an additional physics reasoning benchmark with open-weight LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05933v1">Beyond Greedy Chunking: SLO-Aware Sliding-Window Scheduling for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      With the rapid growth of interactive applications in large language model (LLM) online services, maintaining high system throughput while ensuring user-perceived latency has become a key issue in inference scheduling. Existing LLM service systems rely on coarse-grained output constraints, making it difficult to effectively handle resource contention among multiple requests, resulting in low resource utilization efficiency and limited support for fine-grained quality of service (QoS) differentiation. We present SlidingServe, a sliding-window-driven SLO-Aware scheduling system for online LLM inference. SlidingServe designed a lightweight batch latency predictor to estimate the execution time of a batch. Based on this, SlidingServe uses SlidingChunker to combine information from the current iteration and the next iteration to achieve dynamic chunking and improve the overall system throughput while maintaining strict QoS guarantees. SlidingServe introduces Multi-Level Priority Sorter to sort candidate requests in order to balance fairness and efficiency. Additionally, when multiple requests within the same batch are at risk of SLO violating,SlidingServe introduces BatchConstructor, which uses dynamic programming to select the set of requests to execute in the current round, mitigating the SLO violation risk of critical requests.Our evaluation demonstrates that SlidingServe can improve service capacity by up to 30% compared to advanced scheduling systems under various load conditions, and further reduces the rate of SLO violation by 16%-53% under heavy-load inference mode.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05924v1">Better Literary Translation: A Multi-Aspect Data Generation and LLM Training Approach</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 Accepted by ACL 2026 Industry
    </div>
    <details class="paper-abstract">
      Literary translation poses unique challenges due to the scarcity of high-quality annotated data and the need to balance expression fluency with literary effect. We present a multi-aspect iterative refinement framework that generates high-quality translation references and preference data through specialized LLM translators, each targeting a distinct quality dimension. We leverage the generated data for supervised fine-tuning and reinforcement learning. Experiments show that our generated references outperform the original ground truth for SFT by 8.65 CEA100 points. For reinforcement learning, we find that DPO leads to performance degradation in this setting, while leveraging an explicit reward model for GRPO yields an additional 1.51 point improvement. We attribute this to the stability of two-stage training and GRPO's online exploration capability. Our resulting models, LitMT-8B and LitMT-14B, achieve 67.25 and 69.07 CEA100 respectively on the MetaphorTrans English-to-Chinese literary translation benchmark, competitive with Claude Sonnet 4.5 at 68.43, and demonstrate strong generalization to out-of-domain literary work (i.e., O. Henry).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05890v1">Staying with the Uncertainty: Uncertainty-Scaffolding Strategies for Artificial Moral Advisors in LLM-to-LLM Simulated Conversations</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      LLMs are increasingly deployed as Artificial Moral Advisors (AMA) in a variety of contexts: what kind of conversational patterns should they display? In this paper, we study how AMA can help their interlocutors "stay with the uncertainty". We propose three modes of uncertainty (Perspective-Multiplying, Tension-Preserving, Process-Reflecting) and compare them against three control conditions (Baseline, Persuasive, Sycophantic). A user-agent LLM engages in a dialogue on an ethical dilemma with an AMA following a specific uncertainty strategy, and completes pre- and post-conversation questionnaires. We further examine the effect of two persona prompt formats (Declarative and Narrative). We found that (1) no single model dominates as a simulated user agent, with open models aligning with human ambiguity through between-persona divergence and closed models through within-persona hedging; (2) declarative personas better capture initial stance diversity while narrative personas show more realistic belief revision; (3) all six AMA strategies produce distinguishable conversational patterns; and (4) uncertainty strategies differ not in how much stance revision they produce, but in the quality of engagement they sustain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05885v1">When Denser Credit Is Not Enough: Evidence-Calibrated Policy Optimization for Long-Horizon LLM Agent Training</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Long-horizon LLM agents require reinforcement learning methods that can assign credit to intermediate decisions under sparse and delayed rewards. Recent group-based methods such as GiGPO improve over GRPO by constructing step-level advantages at repeated anchor states. However, we show that such dense credit can be statistically unreliable: under limited rollouts, rare but lucky actions may receive overly large advantages, producing divergent anchor bias and late-stage training oscillation. We propose Evidence-Calibrated Policy Optimization (ECPO), a critic-free policy optimization algorithm that calibrates step-level credit before policy updates. ECPO combines Evidence-Calibrated Action Advantage, which groups rollouts by canonical actions and shrinks low-count estimates, with Variance-Gated Credit Weighting, which suppresses anchor states dominated by within-action noise. Experiments on ALFWorld and WebShop with Qwen2.5-1.5B/7B show that ECPO consistently outperforms strong baselines, improving GiGPO by +5.2/+7.3 success points on ALFWorld/WebShop with Qwen2.5-1.5B while adding only 0.1% additional advantage-computation overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05868v1">YouZhi: Towards High-Concurrency Financial LLMs via Adaptive GQA-to-MLA Transition</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) drive significant financial innovations, yet their high-concurrency deployment is severely bottlenecked by KV cache memory overhead, which inflates infrastructure costs and throttles scalability. To address this, we propose YouZhi-LLM, a highly efficient financial LLM empowered by a comprehensive structural transition and training pipeline natively built on the Huawei Ascend ecosystem. At its algorithmic core, YouZhi-LLM features a layer-adaptive GQA-to-MLA transition framework that dynamically assigns per-layer FreqFold sizes, maximizing KV-cache compression while minimizing perplexity degradation. To recover representation capacity and inject domain expertise, the Ascend-based training pipeline seamlessly integrates generalized knowledge distillation with financial-specific supervised fine-tuning. Evaluations demonstrate the superiority of this systematic approach, with the adaptive transition reducing perplexity degradation by up to 35% over uniform baselines. Crucially, when evaluated on Ascend NPUs via vLLM-Ascend, the massive KV-cache reduction translates directly into deployment efficiency. Compared to their respective base models, YouZhi-7B yields a 12.3% improvement in average financial benchmark score alongside a 2.69$\times$ increase in maximum concurrency; similarly, YouZhi-14B achieves a 7.0% accuracy gain and a 2.43$\times$ concurrency boost, establishing a new paradigm for cost-effective, high-throughput financial inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03785v2">Backdoor Unlearning Generalization: A Path Toward the Removal of Unknown Triggers in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 22 pages, 28 figures
    </div>
    <details class="paper-abstract">
      Backdoor attacks in Large Language Models (LLMs) are a growing security concern, where models can generate adversary-chosen content. Existing defenses target backdoors one at a time and typically require knowledge of the trigger, leaving the defender at a structural disadvantage when unknown backdoors may exist in a model. We show that backdoor neutralization through unlearning generalizes across backdoors: training a model to ignore a single trigger can also suppress other backdoors that were never explicitly targeted. We study this phenomenon across three model families, whose backdoors were injected via pretraining or continual pretraining, by analyzing the models obtained after removing one backdoor at a time. To understand why unlearning certain backdoors induces the suppression of others, we introduce the Cross Activation Shift Distance, to quantify the distance between model changes induced by different trainings. Our results open a new direction for LLM safety as defenders could deliberately inject controlled backdoors and then remove them, leveraging cross-backdoor transfer to also suppress unknown backdoors that an attacker may have previously introduced in the model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05858v1">ReverseEOL: Improving Training-free Text Embeddings via Text Reversal in Decoder-only LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have opened new avenues for generating training-free text embeddings. However, the causal attention in decoder-only LLMs prevents earlier tokens from attending to future context, leading to biased contextualized representations. In this work, we propose Reverse prompting with Explicit One-word Limitation (ReverseEOL), a simple yet effective method for enhancing the representational capability of frozen LLMs. ReverseEOL augments the standard forward embedding with an additional reversed embedding derived from the reversed input text. Since reversing the input exposes each token to context inaccessible in the original order, the resulting reversed embedding effectively provides complementary information to the original one. As a result, combining the forward and reversed embeddings yields a richer final representation. Comprehensive experiments on STS and MTEB benchmarks demonstrate that ReverseEOL significantly improves the performance of existing training-free baselines across a broad range of LLMs with diverse architectures and scales. Extensive ablations and analyses further confirm the necessity of our reversal mechanism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.25256v2">Whose Alignment? Comparing LLM Process Alignment Across Diverse Organizational Decision Contexts</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 Accepted to Pluralistic Alignment Workshop @ ICML 2026, Seoul, South Korea
    </div>
    <details class="paper-abstract">
      Steerable pluralism requires a model to faithfully represent one specified perspective. Organizations are a natural setting for this demand, since they deploy LLMs to make decisions that must reflect their own policy. Yet, most existing work fixes that perspective at the level of individuals or demographic groups. We rely on a decision-policy capturing method to measure process alignment in organizational settings, assessing whether an LLM faithfully reproduces the organization's decision policy rather than merely reaching the same conclusions. We find heterogeneity along two axes. Across models, baseline alignment varies strongly and tracks neither pricing nor general benchmark performance. Across organizations, the structure of alignment changes. In ECHR Article 6 decisions, process alignment predicts output accuracy ($r = 0.85$, $p < .001$), and making the organization's past decision policy explicit improves poorly aligned models. In consumer credit decisions, process alignment is low overall but varies more than output accuracy, and the models resist adopting the organization's weighting of protected attributes. Because historical credit decisions encode potentially discriminatory patterns, higher alignment there is not always desirable. Process-level measurement is therefore necessary, and depending on whether the target policy is normatively desirable, the same procedure can calibrate or audit a model. Deciding which policy to align to, and whether higher alignment is feasible or desirable, makes organizational alignment a pluralistic problem in its own right.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05844v1">GenTI: Benchmarking LLMs for Autonomous IDPS Rule Generation for Unseen Attacks</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Rule-based Intrusion Detection and Prevention Systems (IDPS) offer precise attack detection as well as mitigation, however their manually crafted, signature-driven rules limit adaptability to emerging and zero-day threats. Additionally, existing public datasets (e.g., CICIDS2017, UNSW-NB15) focus on traffic classification and provide little structured information to support automatic rule synthesis or prevention logic. To address this gap, we propose Generative Thread Intelligence (GenTI) \footnote{GenTI refers to the proposed framework, and GTI refers to the dataset.} an LLM-driven benchmark for automatic generation of IDPS rules targeting unseen attacks. The dataset (GTI) aggregates over 150k detection and prevention rules from Snort, Suricata, Emerging Threats, as well as 50k YARA, each annotated with protocol behavior, payload signatures, contextual relationships, mappings to Cyber Threat Intelligence (CTI), along with actionable response types (alert, drop, reject). Moreover, on top of this corpus we design an LLM-based pipeline that transforms analyst prompts and representative payloads into deployable rules via structured prompt engineering, Chain-of-Thought (CoT) reasoning, as well as a Chain-of-Verification (CoVe) loop for syntactic, semantic, and security validation. The generated rules are executed in real time on (Snort/Suricata) and evaluated by syntax accuracy, semantic similarity, CTI coverage, security effectiveness as well as unseen attacks detection. Furthermore, our GenTI instantiation achieves a composite rule-quality score of 89.4\%, with 94.8\% CTI coverage, improving unseen attacks detection from 45\% to 87.4\% and reducing the false-positive rate from 8.5\% to 2.3\%. Overall, GenTI establishes the first large-scale benchmark that tightly couples rule-level CTI with LLM-based automation, enabling adaptive, self-evolving IDPS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14224v3">Knowledge Matters: Injecting Project and Testing Knowledge into LLM-based Unit Test Generation</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 Accepted at the 48th International Conference on Software Engineering(ICSE 2026),13 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Automated unit test generation using large language models (LLMs) holds great promise but often struggles with generating tests that are both correct and maintainable in real-world projects. This paper presents KTester, a novel framework that integrates project-specific knowledge and testing domain knowledge to enhance LLM-based test generation. Our approach first extracts project structure and usage knowledge through static analysis, which provides rich context for the model. It then employs a testing-domain-knowledge-guided separation of test case design and test method generation, combined with a multi-perspective prompting strategy that guides the LLM to consider diverse testing heuristics. The generated tests follow structured templates, improving clarity and maintainability. We evaluate KTester on multiple open-source projects, comparing it against state-of-the-art LLM-based baselines using automatic correctness and coverage metrics, as well as a human study assessing readability and maintainability. Results demonstrate that KTester significantly outperforms existing methods across six key metrics, improving execution pass rate by 5.69% and line coverage by 8.83% over the strongest baseline, while requiring less time and generating fewer test cases. Human evaluators also rate the tests produced by KTester significantly higher in terms of correctness, readability, and maintainability, confirming the practical advantages of our knowledge-driven framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21700v3">Toward Culturally Aligned LLMs through Ontology-Guided Multi-Agent Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 Accepted by ICML 2026 Regular Track
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly support culturally sensitive decision making, yet often exhibit misalignment due to skewed pretraining data and the absence of structured value representations. Existing methods can steer outputs, but often lack demographic grounding and treat values as independent, unstructured signals, reducing consistency and interpretability. We propose OG-MAR, an Ontology-Guided Multi-Agent Reasoning framework. OG-MAR summarizes respondent-specific values from the World Values Survey (WVS) and constructs a global cultural ontology by eliciting relations over a fixed taxonomy via competency questions. At inference time, it retrieves ontology-consistent relations and demographically similar profiles to instantiate multiple value-persona agents, whose outputs are synthesized by a judgment agent that enforces ontology consistency and demographic proximity. Experiments on regional social-survey benchmarks across four LLM backbones show that OG-MAR improves cultural alignment and robustness over competitive baselines, while producing more transparent reasoning traces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05806v1">When Tools Fail: Benchmarking Dynamic Replanning and Anomaly Recovery in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Existing benchmarks evaluate Tool-Integrated Reasoning (TIR) in LLMs on idealized ''happy paths'', largely overlooking real-world tool failures. We introduce ToolMaze, a benchmark for dynamic path discovery and error recovery in TIR agents. To separate systematic replanning from blind trial-and-error, ToolMaze adopts a two-dimensional design: DAG-based topological complexity and a $2 \times 2$ taxonomy of tool perturbations (explicit/implicit, transient/permanent). Evaluations show that perturbations degrade performance across nearly all models, with the sharpest drops under implicit semantic failures. Driven by systemic over-trust in corrupted outputs, Perturbation Recovery Rate (PRR) plummets by around 37\% in these scenarios, while complex topologies trap agents in futile trial-and-error loops. Crucially, agentic fault-tolerance improves with model scale $3.66\times$ slower than basic task execution, highlighting dynamic replanning as a distinct bottleneck unaddressed by model scaling or prompting. Data and code are available at https://github.com/Zhudongsheng75/ToolMaze.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05805v1">From Risk Classification to Action Plan Remediation: A Guardrail Feedback Driven Framework for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 32 pages
    </div>
    <details class="paper-abstract">
      LLM-based guardrails typically safeguard agents by evaluating proposed actions or inputs before execution, producing safety signals such as binary allow/deny decisions, risk categories, and/or explanatory rationales about potential policy violations. However, agent risks often arise when otherwise benign tasks are contaminated by untrusted external content, unsafe instructions, or risky tool use. Existing guardrails often flag the entire task uniformly as unsafe, thereby blocking the threat but sacrificing the benign part. Moreover, existing work largely evaluates guardrails in isolation, leaving unclear whether their interventions lead to safer downstream agent behavior. To address this, we introduce TRIAD (Tripartite Response for Iterative Agent Guardrailing), a guardrail-integrated agent framework that leverages guardrail-generated verbal feedback as a guiding signal to keep the agent aligned with benign objectives at each planning step. We finetune a language model on a self-curated training dataset to output one of three decisions: proceed, refuse, or update, together with structured natural-language feedback. Rather than merely allowing or blocking execution, update guides the agent to revise its plan, avoid harmful components, and preserve the benign task where possible. TRIAD injects this feedback into the agent's context, enabling subsequent plan revision and forming a closed loop between guardrail feedback and agent planning. Extensive experiments on ASB and AgentHarm show that TRIAD reduces the average attack success rate to 10.42%, while achieving the best safety-utility trade-off among guardrail-integrated baselines. Our code is available at: https://github.com/YUHAOSUNABC/TRIAD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05804v1">Can LLMs Be Constrained to the Past? Improving Knowledge Cutoff through Recall-Based Prompting</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Prompted knowledge cutoff instructs a large language model (LLM) to act as if information beyond a specified cutoff date were unavailable. However, prior work mainly relies on direct-answer generation, which struggles when post-cutoff knowledge is not explicitly queried but is only causally related to the question. To address this limitation, we propose two recall-based prompting strategies: Self-Recall (SR), which asks the model to restate its cutoff constraint, and Question-Recall (QR), which requires the model to recall question-relevant information valid under the cutoff. Across three existing benchmarks, our methods outperform both direct-answer prompting and conventional step-by-step reasoning baselines, with particularly strong improvements on counterfactual questions. To investigate robustness across different cutoff settings, we further construct the Multi-cutoff Historical Event Benchmark (MHEB), which evaluates the same question under multiple cutoff years. Results show that knowledge cutoff performance varies with cutoff distance, while combining SR and QR consistently yields the best performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05793v1">CollabBench: Benchmarking and Unleashing Collaborative Ability of LLMs with Diverse Players via Proactive Engagement</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 Accepted by ICML 2026
    </div>
    <details class="paper-abstract">
      While LLM-based agents excel at individual tasks, effective collaboration with realistic human partners remains challenging. Most of the existing conversation-level collaborative studies lack grounded interaction and behavioral execution, motivating the need for cooperative game environments that enable contextualized and immersive collaboration. To this end, this paper proposes CollabBench, a benchmark for evaluating and training collaborative agents in cooperative games. CollabBench features a Diverse Player Profile Simulation pipeline to model varied players behaviors, and a Collaborative Agentic Training paradigm that unifies reasoning, communication, and action via agentic rollouts, optimized with a hybrid reward balancing task efficiency and affective adaptation. We further extend classic environments to CWAH-MultiPlayer and Cook-MultiPlayer for systematic evaluation under diverse personalities. Experiments with efficiency and affective metrics show that our trained models outperform base models, achieving 19.5% higher efficiency and 24.4% improved affective performance. Further analysis reveals key collaborative limitations of existing models and offers insights for future collaborative training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05792v1">Can LLMs Write Correct TLA+ Specifications? Evaluating Natural-Language-to-TLA+ Generation</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 12 pages, 11 tables. Accepted at the 21st International Conference on Software Technologies (ICSOFT 2026); Recommended as Best Paper Award Candidate
    </div>
    <details class="paper-abstract">
      TLA+ has supported industrial verification at companies such as Amazon and Microsoft, yet writing correct TLA+ specifications from natural language still requires time and expertise, which limits adoption. LLMs show promise, but no prior study measures whether they produce semantically correct TLA+ specifications from natural language. This paper presents the first systematic evaluation of LLM-based TLA+ specification synthesis from natural language. Our study evaluates 30 LLMs across eight families on a curated dataset of 205 TLA+ specifications: 25 open-weight models across four prompting strategies (2,600 runs) and 5 proprietary models under few-shot prompting (130 runs), all validated by the SANY parser and TLC model checker. LLMs achieve up to 26.6% syntactic correctness but only 8.6% semantic correctness, with successes exclusive to progressive prompting. Results show that model size does not predict quality, e.g., DeepSeek r1:8b outperforms its 70B variant across all strategies, which suggests the importance of reasoning alignment for formal languages. Code-specialized models consistently underperform due to negative transfer from mainstream language training. We identify five recurring hallucination categories, all traceable to specific training data biases. These results suggest that current LLMs do not generate reliable TLA+ specifications without expert oversight. We release the evaluation framework, code, and dataset to support reproducibility and future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16475v2">Breaking the Chain: A Causal Analysis of LLM Faithfulness to Intermediate Structures</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 20 pages, 4 figures, 7 tables
    </div>
    <details class="paper-abstract">
      In schema-guided reasoning (SGR) pipelines, LLMs produce explicit intermediate structures -- rubrics, checklists, or verification queries -- before committing to a final decision. SGR is increasingly adopted because it promises controllability: practitioners expect to inspect, edit, and override these structures to steer the outcome. But does the promise hold? We introduce a causal evaluation protocol to measure it: by selecting tasks where a deterministic function maps intermediate structures to decisions, every controlled edit implies a unique correct output. Across 12 models and 4 benchmarks, models appear self-consistent with their own intermediate structures but fail to update predictions after intervention -- revealing that apparent faithfulness is fragile once the intermediate structure changes. When derivation of the final decision from the structure is delegated to an external tool, this fragility largely disappears; stronger prompting yields only limited improvements, while preference optimization substantially improves intervention faithfulness. Overall, intermediate structures in schema-guided pipelines function as influential context rather than stable causal mediators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.03217v2">Moral Sensitivity in LLMs: A Tiered Evaluation of Contextual Bias via Behavioral Profiling and Mechanistic Interpretability</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in settings that require nuanced ethical reasoning, yet existing bias evaluations treat model outputs as simply "biased" or "unbiased." This binary framing misses the gradual, context-sensitive way bias actually emerges. We address this gap in two stages: behavioral profiling and mechanistic validation. In the behavioral stage, we introduce the Moral Sensitivity Index (MSI), a metric that quantifies the probability of biased output across a graduated, seven-tier stress test ranging from abstract numerical problems to scenarios rooted in historical and socioeconomic injustice. Evaluating four leading models (Claude 3.5, Qwen 3.5, Llama 3, and Gemini 1.5), we identify distinct behavioral signatures shaped by alignment design: for instance, Gemini 1.5 reaches 72.7% MSI by Tier 5 under socioeconomic framing, while Claude exhibits sharp suppression consistent with identity-based safety training. We then verify these behavioral patterns mechanistically. We select criminal-bias scenarios, which produced the highest MSI scores across models, as probes and apply logit lens, attention analysis, activation patching, and semantic probing to a controlled set of six models spanning three capability tiers: small language models (SLMs), instruction-tuned base models, and reasoning-distilled variants. Circuit-level analysis reveals a U-curve of bias: SLMs exhibit strong criminal bias; scaling to instruction-tuned models eliminates it; reasoning distillation reintroduces bias to SLM-like levels despite identical parameter counts, suggesting distillation compresses reasoning traces in ways that reactivate shallow statistical associations. Critically, the socially loaded cues that drive high MSI scores activate the same bias-driving circuits identified mechanistically, providing cross-stage validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05713v1">Beyond Generative Decoding: Discriminative Hidden-State Readout from a Native Omni-Modal LLM for Multimodal Sentiment Analysis</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 18 pages, 4 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Multimodal sentiment analysis (MSA) infers human affect from language, acoustic, and visual signals. Recent methods increasingly adapt large multimodal models (LMMs) via generative readout: prompting the model to emit a sentiment score as a text string. While convenient, this ties continuous regression to discrete autoregressive decoding, incurring unmeasured costs. We revisit this readout mechanism and propose a discriminative formulation built on the Thinker module of a native omni-modal LLM (Qwen2.5-Omni-7B). Instead of text decoding, we map the final-layer hidden state of the last non-padding token to a continuous score via a lightweight regression head in a single forward pass. Using 4-bit quantization and low-rank adaptation (QLoRA), the entire 7B pipeline -- including video and audio processing -- trains on a single consumer GPU (RTX 5090, 32 GB) with 10-21 GB peak memory and 1.14% trainable parameters. Through a controlled comparison fixing the backbone, data, and LoRA configuration, we isolate the impact of the readout. On CMU-MOSI and CMU-MOSEI, our discriminative readout reaches state-of-the-art accuracy without task-specific feature engineering (MOSI: MAE 0.551, Corr 0.888; MOSEI: MAE 0.506, Corr 0.790) and exhibits strong multi-seed stability. In contrast, the generative readout -- even after equivalent supervised training -- more than doubles the mean absolute error, yields unparsable or out-of-range outputs (2.8% zero-shot), and suffers from higher latency. Modality ablations reveal a text-dominant regime on CMU-MOSI. Our findings indicate that how an LMM is read out is as consequential as how it is trained, demonstrating that a discriminative readout offers a more accurate, efficient, and reliable alternative for continuous MSA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05711v1">Beyond tokens: a unified framework for latent communication in LLM-based multi-agent systems</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Multi-agent systems built on large language models (LLMs) have become a prevailing paradigm for tackling complex reasoning, planning, and tool-use tasks. The dominant communication protocol in such systems is natural language: agents exchange messages token-by-token, verbalising their internal reasoning so that peers can read, verify, and respond. While convenient and interpretable, this protocol suffers from three structural drawbacks -- high inference cost, irreversible information loss during discretization, and ambiguity/redundancy of natural language. A growing body of work therefore explores an alternative protocol -- latent communication -- in which agents exchange continuous representations (embeddings, hidden states, or KV-caches) directly, bypassing the bottleneck of text generation. This paper presents a unified framework for organising the rapidly expanding literature on latent communication. We analyse existing methods along three orthogonal axes: (1) WHAT information is communicated (Embeddings, Hidden States, KV-Caches, or other continuous state); (2) WHICH sender-receiver alignment is used (latent-space alignment and layer alignment); and (3) HOW the communicated information is fused into the receiver (concatenation, prepending, mathematical operations, cross-attention, or cache restoration). Under this 3-axis framework, we systematically categorise eighteen representative methods proposed between 2024 and 2026, identify five major design patterns, and surface a set of open challenges -- including cross-architecture alignment, security of latent channels, compression for edge deployment, and the relationship between latent communication and latent chain-of-thought. We hope that this framework both lowers the barrier to entry for new researchers and provides a vocabulary for comparing future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04397v2">Context-as-AI-Service: Surfacing Cross-File Dependency Chains for LLM-Generated Developer Documentation</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 8 pages, 2 figures, 4 tables
    </div>
    <details class="paper-abstract">
      LLM agents increasingly write and maintain developer documentation, but usefulness and accuracy often rely on dependency chains that are not obvious to follow. Even with more files in context, the agent must still decide which cross-file dependencies to trace. We present Context-as-AI-Service (CAIS), a retrieval layer that LLM agents query to find evidence across the codebase as they review or generate documentation. CAIS indexes source code, API references, and upstream documentation, then enables agents to query the index through tool calls that combine keyword and semantic search. We evaluate CAIS in two case studies using Claude Sonnet 4.6 on a production SDK: improving API reference comments in a core source file and validating an LLM-generated tutorial. In both studies, the baseline already had ordinary repository tools such as file reads, keyword search, and symbol navigation. CAIS adds a retrieval layer on top, so the comparison isolates added retrieval rather than basic repository access. In the API-reference review, the CAIS-augmented agent produced the same 5 missing-documentation fixes as the baseline and surfaced 4 findings the baseline missed: 2 cross-file factual errors and 2 underspecified API comments. In the tutorial validation, it surfaced 1 executable bug, 1 API-usage improvement, and 2 missing prerequisites that the baseline pipeline did not catch. These findings required tracing non-obvious dependency chains across utility files, framework internals, usage examples, tests, and component-creation logic. Over five runs per condition, adding CAIS reduced wall-clock time by 22% to 34% across the two tasks and lowered input-token usage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05682v1">Beyond Output Matching: Preserving Internal Geometry in NVFP4 LLM Distillatio</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 13 pages,1 figures
    </div>
    <details class="paper-abstract">
      Demand for low-precision inference, including NVFP4-based approaches, has grown as large language models are increasingly deployed in latency and cost constrained production environments. Quantization-aware distillation (QAD) helps recover accuracy lost under low bit quantization by training a quantized student to match the output distribution of a frozen higher precision teacher via a KL-divergence loss. In this work, we first provide a representation level diagnosis of QAD: output matching alone can mask internal degradation, because many intermediate activation geometries can yield similar teacher-aligned logits. Using CKA, we show that KL-only QAD can reduce layerwise representational similarity relative to the BF16 teacher, with especially severe drift in RL-post-trained models. This drift correlates with downstream bottlenecks on reasoning and coding tasks, suggesting that low bit recovery requires preserving internal geometry rather than matching outputs alone. Motivated by this finding, we propose \textbf{CKA-QAD}, a CKA-guided representational alignment method for NVFP4 QAD and low bit LLM accuracy recovery. The method adds a lightweight regularizer that preserves internal representational geometry during distillation by aligning layerwise Gram matrices through CKA. Across Nemotron 3 Nano and Qwen3-4B-Thinking-2507, CKA-QAD substantially improves representational alignment and improves downstream reasoning and coding accuracy with modest training overhead. Our findings position CKA-guided representational alignment as a practical complement to output matching for quantized LLM recovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05680v1">CASS-RTL: Correctness-Aware Subspace Steering for RTL Generation with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 Accepted to the IEEE International Conference on LLM-Aided Design (LAD '26)
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled the automatic synthesis (generation) of register-transfer level (RTL) code from natural language instructions, offering a promising pathway to accelerate chip design. Unlike typical natural language (and software coding) tasks, LLM-based RTL code generation demands strict cycle accuracy with concurrency, where minor logical errors can render a circuit unusable or insecure. While prior work has explored hallucination mitigation via external verification, self-evaluation prompts, retrieval-augmented prompting, domain specific fine-tuning, agentic solutions, and reasoning, these approaches largely overlook the attention-oriented internal mechanisms of LLMs that may inherently correlate with RTL correctness. This work proposes CASS-RTL, a first-of-its-kind framework for discovering and leveraging LLMs' correctness-aware components to guide RTL generation toward functionally accurate outputs. We (i) identify attention heads whose activation patterns consistently differentiate correct from incorrect RTL; (ii) construct a low-dimensional subspace capturing correctness-relevant signals; and (iii) design a lightweight, geometry-aware intervention that steers the model at inference time. CASS-RTL is fully model-agnostic, requires no additional supervision or retraining, and readily integrates into existing models. Empirically, we evaluate CASS-RTL on multiple models and observe 10%-20% improvement in pass@1/5/10 accuracy on VerilogEval and 5% improvement on CVDP, demonstrating the effectiveness of our method in enhancing reliability without sacrificing model efficiency or requiring a large labeled dataset for fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.14291v3">Advances in Temporal Point Processes: Bayesian, Neural, and LLM Approaches</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Temporal point processes (TPPs) are stochastic process models used to characterize event sequences occurring in continuous time. Traditional statistical TPPs have a long-standing history, with numerous models proposed and successfully applied across diverse domains. In recent years, advances in deep learning have spurred the development of neural TPPs, enabling greater flexibility and expressiveness in capturing complex temporal dynamics. The emergence of large language models (LLMs) has further sparked excitement, offering new possibilities for modeling and analyzing event sequences by leveraging their rich contextual understanding. This survey presents a comprehensive review of recent research on TPPs from three perspectives: Bayesian, deep learning, and LLM approaches. We begin with a review of the fundamental concepts of TPPs, followed by an in-depth discussion of model design and parameter estimation techniques in these three frameworks. We also revisit classic application areas of TPPs to highlight their practical relevance. Finally, we outline challenges and promising directions for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05670v1">Do More Agents Help? Controlled and Protocol-Aligned Evaluation of LLM Agent Workflows</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 https://github.com/LINs-lab/MASArena/tree/BenchAgent
    </div>
    <details class="paper-abstract">
      Does adding more agents help an LLM workflow once compared systems share the same benchmark loader, tool access, answer contract, usage accounting, and trajectory logging? We introduce BenchAgent, an evaluation framework that places single-agent, fixed multi-agent (MAS), and evolving MAS workflows under one normalized execution and logging protocol. BenchAgent evaluates these substrate-internal workflows across ten reasoning, coding, and tool-use benchmarks with GPT-4.1, and separately reports a Protocol-Aligned External (PAE) GAIA study of a runtime-generated workflow. Under SI conditions, at most one of six tested MAS exceeds the matched single-agent anchor on benchmark-balanced average accuracy: EvoAgent lies within the Wilson one-run guidance, while the remaining five trail by 2.56-11.29 points and occupy more expensive accuracy-cost trade-offs. On the PAE GAIA snapshot, a Claude-Code-style runtime workflow reaches 66.72% overall and 69.23% on Level 3, more than 20 points above the strongest non-Claude baseline, Jarvis, a fixed MAS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.02987v2">Large-Scale LLM Inference with Heterogeneous Workloads: Prefill-Decode Contention and Asymptotically Optimal Control</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are rapidly becoming critical infrastructure for enterprise applications, driving unprecedented demand for GPU-based inference services. A key operational challenge arises from the two-phase nature of LLM inference: a compute-intensive \emph{prefill} phase that processes user input, followed by a memory-bound \emph{decode} phase that generates output tokens. When these phases share GPU resources, prefill tasks throttle the processing speed of concurrent decodes, creating state-dependent contention. This contention is further complicated by workload heterogeneity, as different applications exhibit vastly different input and output lengths. We develop a stochastic control framework for scheduling heterogeneous LLM workloads across large GPU clusters. We formulate LLM inference as a multiclass many-server queueing network with state-dependent service rates, grounded in empirical iteration-time measurements. We analyze the fluid approximation of this system and solve steady-state linear programs that characterize optimal resource allocation. We design gate-and-route policies that regulate prefill admission and decode routing, and prove that they are asymptotically optimal in the many-GPU limit under both bundled and separate token-pricing schemes. We further extend the framework to incorporate Service Level Indicators (SLIs) such as latency and fairness, providing a general approach to constrained scheduling. Numerical experiments calibrated to empirical iteration-time data demonstrate that our policies outperform standard serving heuristics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.00644v2">ForeSci: Evaluating LLM Agents for Forward-Looking AI Research Judgment</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      AI research often requires decisions before future evidence exists: which bottleneck to attack, which direction to pursue, or where a project should be positioned. We introduce ForeSci, a temporally controlled benchmark for evaluating whether LLM agents can make such forward-looking research judgements from historical evidence. ForeSci contains 500 tasks across four fast-moving AI domains and four decision families. Each task is paired with a cutoff-aligned offline knowledge base; post-cutoff papers are hidden during generation and used only for validation. To avoid random future-event prediction, tasks are derived from pre-cutoff taxonomy branches and evidence signals, and answer-generation backbones are selected to precede the task cutoffs. We evaluate native LLMs, Hybrid RAG, and three research-agent adaptations across four backbones. Results show that explicit evidence organization improves traceability and factual support, but gains depend strongly on the decision family. Diagnostics reveal a recurring evidence-decision decoupling: agents may cite relevant evidence while forecasting the wrong research object. ForeSci turns forward-looking AI research judgement into a controlled benchmark for evaluating research agents as decision-making systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05632v1">Evaluation of LLMs for Mathematical Formalization in Lean</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 15 pages, 13 figures, 10 tables. Comments welcome!
    </div>
    <details class="paper-abstract">
      Within the past few years, the ability of Large Language Models (LLMs) to generate formal mathematical proofs has improved drastically. We provide a comparison of various LLMs' effectiveness in producing formal proofs in Lean 4 with the goal of assisting those seeking to use LLMs to support their own projects. We utilize both pass@$k$ and refine@$k$ metrics as the benchmark for our comparison and evaluate on subsets of both miniF2F and miniCTX datasets. Our testing shows that overall, Gemini 3.1 Pro and Claude Opus 4.7 perform best. Gemini 3.1 Pro achieved a 92\% success rate on miniF2F via refine@32 whereas Opus 4.7 achieved a 86\% success rate on miniCTX via refine@32. When taking cost into account, NVIDIA Nemotron 3 Super and GPT-OSS 120B were the most efficient, with competitive accuracies and average costs of $<\$0.01$ per correct proof.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05616v1">What's in a Name? Morphological Shortcuts by LLMs in Pharmacology</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 22 pages
    </div>
    <details class="paper-abstract">
      The morphological form of a word can often give cues to its meaning, but purely relying on these mappings can lead to overgeneralization in high-stakes domains. In the medical domain, for instance, LLMs can confidently reason about fictitious drugs from their affixes alone (e.g., wugcillin) and generate plausible-looking clinical content. We present a behavioral and mechanistic study of LLM "affix heuristics" in pharmacology. Using fictitious drug names built from real affixes, we show that affix signals alone elicit class-level pharmacological responses. We introduce a framework for identifying whether a model's drug semantics are driven mainly by the affix, the stem, or the drug name as a whole. Applied across 653 drugs, our framework reveals that models often induce drug meaning primarily through affix cues, yet rarely explicitly indicate this reliance, and sometimes incorrectly conflate properties among affix-sharing drugs. Activation patching across models further localizes this behavior to early-mid layers. These findings show that morphological shortcuts pose a subtle but measurable risk to safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05614v1">Safety Paradox: How Enhanced Safety Awareness Leaves LLMs Vulnerable to Posterior Attack</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are rigorously aligned to refuse harmful requests, a process that inherently cultivates a latent capacity to evaluate and recognize unsafe content. In this work, we reveal that this advanced safety awareness inadvertently introduces a fatal vulnerability. We introduce Posterior Attack, a single-query jailbreak that bypasses guardrails by prompting the model to generate the exact harmful response its internal classifier would normally flag as unsafe. Through extensive empirical evaluation across 30 open-source LLMs (up to 35B parameters in size) and frontier models (e.g., GPT-5, Claude 4.6), we observe a striking phenomenon: models with superior safety-judgment capabilities are disproportionately more susceptible to this exploitation. To explain this, we formalize the Safety Paradox, analytically showing that monotonic improvements in safety alignment naturally amplify posterior vulnerability. Finally, we establish a causal link via reinforcement learning interventions, exemplifying that artificially degrading a model's safety judgment immunizes it against the attack, whereas enhancing judgment exacerbates the vulnerability. Our findings highlight potential flaws in current alignment paradigms, indicating that defense mechanisms may require further structural refinement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05610v1">Predictable Scaling Laws of Optimal Hyperparameters for LLM Continued Pre-training</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      The efficacy of continued pre-training for Large Language Models (LLMs) hinges upon hyperparameter configurations, such as learning rate and batch size. However, current practices often rely on heuristics or grid searches, leading to training instability and excessive costs. In this work, we first empirically discover that optimal hyperparameters follow stable and predictable scaling laws throughout the continued pre-training process. Leveraging these insights, we propose a novel framework to establish quantitative relationships between compute budget and optimal hyperparameters for a given checkpoint. Our approach has two stages: (1) \textit{Empirical Law Discovery}, where we train small-scale proxy models to derive functions mapping compute budget to optimal hyperparameters via standard loss-compute scaling laws; and (2) \textit{State-Aware Hyperparameter Prediction}, where we evaluate an initial checkpoint's validation loss and use the inverse scaling law to estimate its \textit{equivalent pre-training compute} -- the compute needed to achieve the same loss from scratch. Combining this with the planned compute budget, we predict optimal hyperparameters for the target run. Empirical results demonstrate that our method reduces the hyperparameter search overhead by up to 90\% while achieving comparable or superior performance relative to baselines. This model-agnostic framework generalizes across architectures, providing a principled and efficient methodology for diverse continued pre-training scenarios starting from any given point.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05609v1">SlotGCG: Exploiting the Positional Vulnerability in LLMs for Jailbreak Attacks</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are widely deployed, identifying their vulnerability through jailbreak attacks becomes increasingly critical. Optimization-based attacks like Greedy Coordinate Gradient (GCG) have focused on inserting adversarial tokens to the end of prompts. However, GCG restricts adversarial tokens to a fixed insertion point (typically the prompt suffix), leaving the effect of inserting tokens at other positions unexplored. In this paper, we empirically investigate \emph{slots}, i.e., candidate positions within a prompt where tokens can be inserted. We find that vulnerability to jailbreaking is highly related to the selection of the \emph{slots}. Based on these findings, we introduce the \textit{Vulnerable Slot Score} (VSS) to quantify the positional vulnerability to jailbreaking. We then propose SlotGCG, which evaluates all slots with VSS, selects the most vulnerable slots for insertion, and runs a targeted optimization attack at those slots. Our approach provides a position-search mechanism that is attack-agnostic and can be plugged into any optimization-based attack, adding only 200ms of preprocessing time. Experiments across multiple models demonstrate that SlotGCG significantly outperforms existing methods. Specifically, it achieves 14\% higher Attack Success Rates (ASR) over GCG-based attacks, converges faster, and shows superior robustness against defense methods with 42\% higher ASR than baseline approaches. Our implementation is available at \href{https://github.com/youai058/SlotGCG}{https://github.com/youai058/SlotGCG}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05567v1">ZERO-APT: A Closed-Loop Adversarial Framework for LLM-Driven Automated Penetration Testing under Intelligent Defense</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      LLM-driven automated penetration testing agents are typically evaluated against static targets that neither detect nor respond to attacks, so their behavior under intelligent defense remains untested. The causal consistency of multi-step attack chains likewise hinges on unstable LLM reasoning, and agent decisions remain opaque to human analysts. These three shortcomings, in realism, consistency, and auditability, are usually patched in isolation. We present ZERO-APT, a turn-based attacker-defender-judge framework that addresses them within a single architecture. For realism, ZERO-APT embeds a configurable LLM Defender that consumes Sysmon telemetry and detects attacks in real time, exposing the attacker to a live opponent rather than a passive target. For consistency, three architectural mechanisms move causal consistency from unstable LLM reasoning into enforced system architecture: separation of planning from execution, multi-dimensional ReAct feedback, and a hard-constraint-filtered action library. For auditability, a dedicated Judge agent adjudicates each round, maintains global state, and emits structured post-hoc CTI reports that make every decision traceable. We evaluate a Windows Server 2022 post-exploitation prototype across five scenarios with three Defender configurations. ZERO-APT reaches 79\% attack success rate (Aurora 22\%, PentestGPT 39\%), a Causal Consistency Score of 0.860 (Aurora 0.930, Claude Code 0.520), and end-to-end decision auditability through structured CTI reports. We release the benchmark to support evaluation of penetration agents under intelligent defense.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05563v1">SoCRATES: Towards Reliable Automated Evaluation of Proactive LLM Mediation across Domains and Socio-cognitive Variations</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Evaluating LLM mediators remains challenging, as mediation unfolds as a real-time trajectory shaped by disputants' shifting emotions, intentions, and context. Existing testbeds rely on a few expert-authored domains, vary mainly strategic posture, and score every turn against every topic, introducing off-topic noise. We introduce SoCRATES, a benchmark for evaluating proactive LLM mediators in realistic, multi-domain testbeds. It constructs scenarios from real conflicts through an agentic pipeline across eight domains, probes five socio-cognitive adaptation axes (strategic posture, party composition, history length, emotional reactivity, and cultural identity), and scores each topic only on the turns that advance it via a topic-localized evaluator. The evaluator reaches 0.82 alignment with human experts, more than doubling a per-turn baseline. Benchmarking eight frontier LLMs, we find that even the strongest mediator closes only about a third of the unmediated consensus gap under diverse and realistic testbeds, with performance varying sharply by socio-cognitive axis, highlighting that progress lies in social adaptation to diverse conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.23600v2">Personality Shapes Gender Bias in Persona-Conditioned LLM Narratives Across English and Hindi: An Empirical Investigation</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in persona-driven applications such as education, customer service, and social platforms, where models are prompted to adopt specific personas when interacting with users. While persona conditioning can improve user experience and engagement, it also raises concerns about how personality cues may interact with gender biases and stereotypes. In this work, we present a controlled study of persona-conditioned story generation in English and Hindi, where each story portrays a working professional in India producing context-specific artifacts (e.g., lesson plans, reports, letters) under systematically varied persona gender, occupational role, and personality traits from the HEXACO and Dark Triad frameworks. Across 23,400 generated stories from six state-of-the-art LLMs, we find that personality traits are significantly associated with both the magnitude and direction of gender bias. In particular, Dark Triad personality traits are consistently associated with higher gender-stereotypical representations compared to socially desirable HEXACO traits, though these associations vary across models and languages. Our findings demonstrate that gender bias in LLMs is not static but context-dependent. This suggests that persona-conditioned systems used in real-world applications may introduce uneven representational harms, reinforcing gender stereotypes in generated educational, professional, or social content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05558v1">Autoregressive Diffusion World Models for Off-Policy Evaluation of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Evaluating large language model (LLM) agents in multi-turn interactive environments is expensive and risky, as it requires online environment interaction. We propose ADWM (Autoregressive Diffusion World Model), an evaluation framework that estimates the performance of a new LLM agent policy purely from pre-collected trajectories. The core idea is to learn a latent diffusion world model that simulates how the environment responds to the evaluation policy, without ever executing it in the real environment. Existing diffusion-based OPE methods guide full trajectories in a single pass by jointly diffusing states and actions, an assumption that breaks down for LLM agents whose actions are discrete text that must be sampled from the policy after observing the environment. Unlike autoregressive world models that suffer from compounding errors, ADWM models each transition as an independent denoising process, enabling reliable step-by-step rollouts where the world model and agent alternate in causal order. Crucially, the LLM agent under evaluation directly guides the diffusion generation at each step via a policy-conditioned score function, ensuring that simulated trajectories accurately reflect its decision-making patterns. Empirically, ADWM achieves accurate value estimates and evaluation reliability across diverse multi-turn agent tasks, demonstrating its promise as a practical framework for offline LLM agent evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05548v1">ADK Arena: Evaluating Agent Development Kits via LLM-as-a-Developer</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 Work in Progress
    </div>
    <details class="paper-abstract">
      The rapid proliferation of Agent Development Kits (ADKs), SDK-level frameworks for building LLM-powered autonomous agents, has outpaced any empirical understanding of how framework choice affects agent performance. We propose \textbf{LLM-as-a-Developer}, a methodology that replaces human developers with an LLM coding agent that learns each framework's API from documentation, writes agent code, and iteratively repairs it through a validate-and-feedback loop until tests pass. By holding the developer constant and varying only the framework, generation effort becomes a quantitative proxy for API usability and the resulting agents provide a controlled measure of framework effectiveness. We implement this in \textbf{ADK Arena}, a fully automated pipeline with per-framework Docker isolation, a three-level validation pipeline, and benchmark adapters for SWE-bench, $τ^2$-bench, Terminal-Bench, and MCP-Atlas. Evaluating all 51 popular Python ADK frameworks (204 agent--benchmark pairs), we find that: (1)~generation succeeds for 57\% of runs, and its cost varies 5.6$\times$ across frameworks (\$0.6 to \$3.4 per agent), a quantitative proxy for API complexity, though cost alone does not predict success; (2)~no single framework dominates: the best single-benchmark ADK agents resolve up to 80\% of tasks and can even \emph{beat} general-purpose frontier coding agents at a fraction of the cost, yet the median framework resolves only 32\%; (3)~across information-source ablations, genuine framework usage stays within a narrow 28--40\% band (highest with raw source access and still 33\% with no reference material at all), indicating that documentation, source code, and parametric knowledge are largely substitutable rather than any one being a hard bottleneck.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.13628v2">Compact LLM Deployment and World Model Assisted Offloading in Mobile Edge Computing</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 16 pages, 10 figures
    </div>
    <details class="paper-abstract">
      This paper investigates compact large language model (LLM) deployment and world-model-assisted inference offloading in mobile edge computing (MEC) networks. We first propose an edge compact LLM deployment (ECLD) framework that jointly applies structured pruning, low-bit quantization, and knowledge distillation to construct edge-deployable LLM variants, and we evaluate these models using four complementary metrics: accessibility, energy consumption, hallucination rate, and generalization accuracy. Building on the resulting compact models, we formulate an MEC offloading optimization problem that minimizes the long-term average inference latency subject to per-device energy budgets and LLM-specific quality-of-service constraints on effective accuracy and hallucination. To solve this problem under unknown and time-varying network dynamics, we develop a world model-proximal policy optimization (PPO) algorithm, which augments an on-policy PPO algorithm with a learned recurrent world model that provides improved value targets and short imagination rollouts. Extensive experiments on Llama-3.1-8B, Qwen3-8B, and Mistral-12B show that ECLD compresses base models by about 70-80% in storage (i.e., from 15.3 GB to 3.3 GB for Llama-3.1-8B) and reduces per-query energy consumption by up to 50%, while largely preserving accuracy and often lowering hallucination compared with quantization-only or pruning-only baselines. Moreover, they also show that world model-PPO speeds up convergence by about 50%, improves the final reward by 15.8% over vanilla PPO, and reduces average inference latency by 12-30% across different user populations, while satisfying the accuracy and hallucination constraints and approaching the generation quality of always-offloading with much of the efficiency of local execution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05523v1">CHASE: Adversarial Red-Blue Teaming for Improving LLM Safety using Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 Under Review at ARR
    </div>
    <details class="paper-abstract">
      Despite advances in safety alignment, prompt-rewriting attacks such as persona modulation, fictional framing and persuasion-based reformulation, can bypass safety filters even on frontier models. Existing defenses either rely on non-scalable human curation or white-box optimisation that overfits to specific model internals, leaving aligned models brittle against the very class of adaptive black-box adversaries they will face in deployment. To address this gap, we introduce CHASE (Co-evolutionary Hardening through Adversarial Safety-Escalation), a closed-loop red-blue teaming framework in which a black-box attacker and a safety-aligned defender co-evolve. The attacker is trained via Group Relative Policy Optimization (GRPO) under a multiplicative reward that jointly enforces bypass effectiveness and intent fidelity, while the defender is hardened on the harvested adversarial rewrites through a two-stage GRPO + rejection-sampled SFT pipeline balanced with benign data. Evaluated on BeaverTails and JailbreakBench against five held-out attack families (PAIR, TAP, AutoDAN, PAP, Translation), CHASE cuts mean StrongREJECT score by 43.2\% with 0\% false-refusal on benign prompts. Beyond the headline result, CHASE shows that template-free RL exploration recovers latent attack primitives that transfer across mechanistically distinct attack families, suggesting a path toward LLM safety hardening that generalises beyond the narrow distributions achieved thus far in adversarial training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.14145v3">LLM-Enhanced Dialogue Management for Full-Duplex Spoken Dialogue Systems</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Achieving full-duplex communication in spoken dialogue systems (SDS) requires real-time coordination between listening, speaking, and thinking. This paper proposes a semantic voice activity detection (VAD) module as a dialogue manager (DM) to efficiently manage turn-taking in full-duplex SDS. Implemented as a lightweight (0.5B) LLM fine-tuned on full-duplex conversation data, the semantic VAD predicts four control tokens to regulate turn-switching and turn-keeping, distinguishing between intentional and unintentional barge-ins while detecting query completion for handling user pauses and hesitations. By processing input speech in short intervals, the semantic VAD enables real-time decision-making, while the core dialogue engine (CDE) is only activated for response generation, reducing computational overhead. This design allows independent DM optimization without retraining the CDE, balancing interaction accuracy and inference efficiency for scalable, next-generation full-duplex SDS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05522v1">Exploring LLMs for South Asian Music Understanding and Generation</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 19 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have shown promising results in music understanding and generation tasks. However, existing works remain confined to Western tonal traditions, offering little insight into whether current LLMs can handle structurally distinct low-resource musical traditions. We present the first systematic evaluation of LLM competence in South Asian classical music, a tradition governed by raga, tala-based melodic constraints that impose fundamentally different structural principles from Western harmony-driven music. We ground our evaluation in Hindustani classical theory and Bengali classical forms, including Rabindra and Nazrul Sangeet -- representative low-resource traditions within South Asian classical music. For music understanding evaluation, we introduce a 504-question-answer benchmark spanning raga grammar, cultural knowledge, and symbolic notation reasoning, evaluating 33 LLMs where frontier models such as Gemini 2.5 Pro achieve 85-90% accuracy, while most open-source models remain in the 23-40% range. For music generation, we design a five-level controlled prompting framework and find that even the strongest model produces stylistically faithful outputs only 40% of the time. These results reveal that structural validity and stylistic faithfulness in music generation are distinct objectives and highlight an open challenge for culturally grounded music modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05516v1">Dominant-Layer ZO: A Single Layer Dominates Zeroth-Order Fine-Tuning of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Zeroth-order (ZO) optimization enables memory-efficient fine-tuning of large language models (LLMs) using only forward passes, but it remains unclear how useful adaptation is distributed across layers. In this work, we reveal a surprising phenomenon: ZO fine-tuning is sharply dominated by a single decoding layer. Across multiple LLM families and downstream tasks, fine-tuning this dominant layer alone consistently matches or even exceeds full-model ZO fine-tuning. We further show that the dominant layer is task-agnostic but model-specific, and can be identified before training through a simple inference-only analysis of activation outliers. Specifically, the dominant layer consistently aligns with the first activation-outlier layer in the pre-trained model. To explain this phenomenon, we analyze how perturbation effects propagate under ZO optimization. We find that the dominant layer combines two key properties: high perturbation sensitivity and early placement in the residual stream, allowing perturbation-induced effects to propagate and accumulate through remaining subsequent decoding layers. As a result, this layer produces disproportionately strong and stable optimization signals under forward-only updates. Extensive experiments on LLaMA2-7B and Qwen3-8B across nine benchmarks show that dominant-layer ZO fine-tuning improves average performance over full-model MeZO and LoRA-based ZO fine-tuning while achieving up to 4.52$\times$ training speedup.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05511v1">RH+: Row-Hit-Optimized Scheduling for PIM-based LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Large language model inference on processing-in-memory (PIM) architectures promises to break the memory wall by performing multiply-accumulate (MAC) operations directly within HBM3 DRAM banks. Prior work identifies the power constraint timing parameter nCCDAB as the primary performance bottleneck and optimizes scheduling accordingly. We demonstrate that for GEMV operations that dominate autoregressive decoding, the DRAM row cycle time (nRC) is 10 to 11 times larger than nCCDAB. Consequently, nCCDAB is entirely masked, rendering prior nCCDAB-focused optimizations ineffective for these workloads. The root cause is inherited host-centric address interleaving, which forces every all-bank MAC command into a different DRAM row. We propose RH+ scheduling, a simple stride change that keeps 32 consecutive MAC operations within the same row. Cycle-accurate simulation across four LLM workloads shows that RH+ delivers 8-12x speedup, over 74% energy reduction, and up to 52x EDP improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05489v1">LLM-Guided ANN Index Optimization for Human-Object Interaction Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 13 pages, 5 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Retrieval systems underpin modern AI applications -- spanning visual search, recommendation engines, and multi-modal question answering. Modern multi-stage retrieval systems require the joint optimization of highly coupled parameters, yet traditional hyperparameter optimization (HPO) methods -- including Tree-structured Parzen Estimators (TPE) and Gaussian Process Bayesian Optimization -- rely on an independence assumption that fundamentally prevents them from navigating these coupled configuration spaces. We address this limitation with a phase-aware large language model (LLM) agent that conditions each proposal on its full optimization history, navigating the coupled parameter space across phase-partitioned exploration, exploitation, and fine-tuning stages. Evaluated on the HICO-DET human-object interaction retrieval benchmark using Intel VDMS (Visual Data Management System), our agent outperforms Optuna TPE by +33.3% and VDTuner by +34.2% under SIEVE (Safeguarded Index Evaluation of Vector-search Efficiency, a quality-constrained throughput metric), delivering a 15.3x throughput gain over UniIR. Validation across three benchmarks confirms that the agent's advantage grows with the degree of parameter coupling: +33.3% on HICO-DET (high coupling), methods converge within 1% on GLDv2 (moderate coupling) and within 3.6% on SIFT1M (near-independent control). Cross-system validation on Milvus confirms the optimizer ranks first on all three datasets without modification, demonstrating transferability across vector database management system (VDBMS) platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.19294v4">Maximizing Mutual Information Between Prompt and Response Improves LLM Performance With No Additional Data</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 International Conference on Machine Learning 2026
    </div>
    <details class="paper-abstract">
      While post-training has successfully improved large language models (LLMs) across a variety of domains, these gains heavily rely on human-labeled data or external verifiers. Existing data has already been exploited, and new data is expensive to collect. Moreover, true intelligence goes far beyond verifiable tasks. Therefore, we need self-improvement frameworks that are less dependent on external signals and more broadly applicable to both verifiable and non-verifiable domains. We propose **Mutual Information Preference Optimization (MIPO)**, a contrastive data augmentation method that constructs preference pairs by generating a positive response conditioning on the correct prompt, and a negative response by conditioning on a random, unrelated prompt. We show that using Direct Preference Optimization to learn from this paired data maximizes pointwise mutual information *under the base LLM* between prompts and model responses. Experiments with with 1-7B parameter Llama and Qwen instruct models show that MIPO achieves 3-16% gains (and 51% increase for Qwen2.5-1.5B-Instruct) on personalization compared to prompting baselines. Surprisingly, MIPO can also be useful in verifiable domains, such as math and multiple-choice question answering, yielding 1-20% gains *without any additional data or external supervision*. These results suggest a promising direction for self-improvement using intrinsic signals derived from contrastive data pairs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05464v1">Step-by-Step Optimization-like Reasoning in LLMs over Expanding Search Spaces</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Verifiable reward training has improved mathematical and coding reasoning, but these domains capture only part of step-by-step decision making. Many real-world tasks require finding a high-value feasible plan among many valid alternatives. We introduce OPT*, a scalable family of optimization-style tasks for training and evaluating LLM step-by-step optimization-like reasoning along a complexity axis: each task provides a feasibility checker and evaluator, while a complexity parameter expands the search space without requiring new human labels. This motivates studying these tasks in two regimes: (i) solver-guided online policy optimization, which uses a solver as a value oracle for partial states and applies rank-based reward shaping to reinforce better next steps, and (ii) search-based offline RL when such solvers are unavailable. Theoretically, we relate success in large search spaces to the information a reasoner extracts per unit of search budget. Empirically, we ablate the ingredients that make search efficient on OPT* and show that training on OPT* improves step-by-step optimization-like reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05463v1">PSEBench: A Controllable and Verifiable Benchmark for Evaluating LLMs in Patient Safety Event Triage</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Patient safety event triage, determining whether a clinical event is reportable under jurisdiction-specific policy, is a high-stakes task typically performed manually by patient safety experts. Although LLMs may support this workflow, reliable evaluation is limited by the lack of benchmarks to capture evidence-grounded policy reasoning, proactive information seeking for incomplete reports, and principled abstention in irreducibly ambiguous cases. We address this gap with a policy-grounded construction methodology centered on the clause card, a structured representation that factorizes regulatory text into auditable decision specifications. Combining clause cards with anchor-driven instantiation and closed-loop verification, our scalable pipeline produces narratives with by-construction ground truth and naturally supports generating missing information and uncertain variants. We instantiate this method on Minnesota's 29 Reportable Adverse Health Events, producing PSEBench, a 5,074-case benchmark with an agentic evaluation environment. Evaluation on 15 representative LLMs reveals consistent capability trends, demonstrates the benchmark's utility, and identifies actionable gaps toward reliable LLM-based patient safety event triage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.01736v2">Argument Collapse: LLMs Flatten Long-Form Public Debate</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      As LLMs are increasingly used to draft public-facing arguments, they may flatten public debate by repeatedly introducing the same polished, plausible arguments. We study argument collapse, the tendency of essays generated by different LLMs to converge to a smaller set of main arguments, sub-arguments, and paragraph-level structures. We compare 1,039 human responses from 195 New York Times (NYT) debates, 448 human responses from 61 longer-form Boston Review (BR) forums, and 23,384 LLM-generated essays. In the NYT corpus, 65.3% of human main arguments are unique within a debate, compared to 3.4% of LLM main arguments. Asking LLMs to generate diverse answers adds variation, but a typical model recovers only about half of the distinct human main arguments, with much of the added variation falling outside the observed human argument space. Collapse also appears in sub-arguments, where among essays with the same main argument, 41.0% of human sub-arguments are unique versus 9.1% from LLM responses. Qualitatively, LLMs often reuse generalized and hedged sub-arguments, while humans prefer more concrete and topic-specific ones. Structure-wise, LLM-generated essays tend to follow a more fixed arc, often opening with a direct claim and moving quickly toward proposals. The same patterns hold in longer BR essays, suggesting that argument collapse extends beyond short-form responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05408v1">Mutation Without Variation: Convergence Dynamics in LLM-Driven Program Evolution</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 Accepted to the Genetic and Evolutionary Computation Conference (GECCO '26) Workshop on Large Language Models for and with Evolutionary Computation
    </div>
    <details class="paper-abstract">
      When an LLM repeatedly mutates a program, does it explore new forms or circle back to the same ones? We study this question by analyzing LLM-driven mutation chains in the absence of selection pressure within a domain-specific language, varying prompt design, model family, and stochastic replication. We find that LLM-based mutation consistently converges toward restricted attractor regions in program space. Convergence is especially severe at the structural level: in 87% of chains, over 93% of mutations revisit a previously seen structural form, with most variation confined to terminal substitutions within recurring templates. Cycle analysis reveals short cycles and self-loops dominating the transition structure. The rate of convergence varies with prompt wording and model choice, but the phenomenon is robust across conditions. A classical GP subtree mutation operator does not exhibit comparable convergence, suggesting that the effect is intrinsic to the LLM mutation pipeline. These findings reveal a tension at the heart of LLM-driven program evolution: the same capabilities that enable semantics-aware program transformation also carry a systematic bias toward structural homogeneity that must be accounted for if such systems are to sustain open-ended exploration. Source code is available at https://github.com/can-gurkan/lmca.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.01489v2">CuTeGen: An LLM-Based Agentic Framework for Generation and Optimization of High-Performance GPU Kernels using CuTe</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      High-performance GPU kernels are critical to modern machine learning systems, yet developing them remains a manual, expert-driven process. Recent work has explored using LLMs to automate kernel generation, but generated kernels still fall short of carefully tuned references on standardized benchmarks. We present CuTeGen, an agentic GPU kernel synthesis framework that treats kernel development as a structured generate-test-refine workflow over the CuTe abstraction layer. Two design choices distinguish CuTeGen from prior work: targeting CuTe rather than raw CUDA, which exposes performance-critical structures such as tiling and data movement while remaining stable enough for iterative refinement, and a delayed profiling schedule that withholds low-level performance feedback until the kernel's high-level structure has stabilized. On the 209 tasks of KernelBench Level-1 and Level-2, CuTeGen achieves an average speedup of 1.71$\times$ over PyTorch and outperforms the prior agentic baseline CudaForge (0.89$\times$) at comparable per-task generation cost. Code available at https://github.com/taratt/cutegen.git
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05396v1">Willing but Unable: Separating Refusal from Capability in Code LLMs via Abliteration</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Producing a labeled vulnerable code at scale is a recurring obstacle for learning-based vulnerability detection: mined corpora carry substantial label noise, and existing LLM-based augmentation propagates these inaccuracies because it transforms vulnerable seeds rather than synthesising vulnerabilities from a specification. A complementary route is to start from safe code and ask an instruction-tuned LLM to inject a specified CWE (which would shift the labeling burden from open-ended detection to bounded binary confirmation) but safety-aligned code LLMs systematically refuse such prompts. This paper is a preliminary feasibility study of abliteration, a low-rank weight edit that orthogonally projects out the refusal direction in the residual stream, as a tool to remove this barrier. We use Python and CWE-89 (SQL injection) as a case study, evaluating the Qwen2.5-Coder-Instruct family at 3B, 7B, and 14B parameters on safe samples drawn from PromSec and SafeCoder, replicated three times per condition. We find that (i) refusal on injection prompts is strongly size- and prompt-context-dependent: the 14B refuses 100% of prompts, the 7B refuses 73% of PromSec but only 5% of SafeCoder, whereas the 3B is essentially never blocked; (ii) abliteration reduces refusal to zero or near-zero across all sizes while leaving syntactic validity above 93%, supporting the view that, in this setting, refusal can be detached from measured code-generation capability; and (iii) the post-abliteration injection rate remains capacity-bound (88-97% on the 14B, 89-90% on the 7B, and 25-48% on the 3B) separating willingness, which abliteration unlocks, from capability, which scales with parameters. Vulnerability verdicts are produced by a three-tool detector ensemble (CodeQL, Semgrep, Bandit) followed by manual adjudication by two authors on detector-positive outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05390v1">Ahoy: LLMs Enacting Multiagent Interaction Protocols</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 Presented at EMAS 2026
    </div>
    <details class="paper-abstract">
      An interaction protocol formalizes how the agents in a multiagent system interact, which facilitates implementing agents. Existing approaches yield agent implementations specific to the selected protocols. How can we engineer intelligent agents that can enact protocols but are programming-free? Our contribution, Ahoy, addresses this question by creating LLM agents that dynamically select and enact declarative protocols to achieve user goals. We demonstrate that an \ahoy agent can correctly and intelligently enact multiple protocols - concurrently if appropriate to the user goal - without specialized training. Ahoy's significance lies in that it brings together declarative protocols and LLMs, both approaches that promise improved knowledge engineering for agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05384v1">Stability vs. Manipulability: Evaluating Robustness Under Post-Decision Interaction in LLM Judges</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 Accepted at ACL 2026 GEM (Generation, Evaluation and Metrics) Workshop
    </div>
    <details class="paper-abstract">
      LLM-as-judge evaluation is widely used in benchmarking pipelines, where model outputs are compared and ranked using automated evaluators. These pipelines typically assume that judgments are stable properties of fixed inputs. We show that this assumption does not hold under interaction. We study post-decision manipulability: the extent to which an evaluation outcome can be altered through subsequent conversation with the judge after an initial decision has been made. Across controlled experiments on MT-Bench and AlpacaEval, we find that LLM judges are highly stable under repeated and neutral reevaluation, yet become substantially reversible under targeted post-decision challenge. An anti-baseline challenge protocol shows that stable judgments can be overturned through motivated interaction, while a counterbalanced target-validation protocol separates this reversibility from net target-directed steering. These reversals have practical consequences: they can degrade agreement with human preferences, shift benchmark rankings, and produce harmful evaluation changes despite high self-reported confidence. Authority framing is especially destabilizing, and revised judgments are often accompanied by low-overlap justifications, suggesting post hoc rationalization rather than reliable error correction. We introduce the Evaluation Robustness Score (ERS) to quantify interactional robustness by combining reversal susceptibility with counterbalanced directional effects. Our findings identify post-decision interaction as a distinct failure mode for LLM-as-judge evaluation and motivate evaluation protocols that measure not only static agreement, but robustness under challenge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05376v1">SHALA-LLM: Smartly Handling Ambiguous Labels in Aligning LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Many human-centered tasks, including natural language inference (NLI) and emotion recognition (ER), have multiple plausible interpretations, leading to label ambiguity and challenging disagreements across human annotators. As LLMs are increasingly deployed in real-world settings, faithfully modeling such ambiguity is essential to identify contested inputs, preserve variability in ambiguous cases, and capture the full distribution of human judgments. Yet, existing LLM alignment approaches have predominantly assumed a single correct label, excluding annotator disagreement during optimization. Instead of treating this ambiguity as noise, we show how to treat it as information that improves model behavior through a new algorithm called SMARTLY HANDLING AMBIGUOUS LABELS IN ALIGNING LLMS (SHALA-LLM). This reinforcement learning framework provides a new way for LLMs to learn directly from annotator distributions while dynamically prioritizing highly ambiguous samples during optimization. Experiments on ambiguity-sensitive NLI and ER benchmarks, including ChaosNLI, GoEmotions, and MSP-Podcast, demonstrate that SHALA-LLM improves agreement with annotator label distributions, e.g. on ChaosNLI, it reduces Jensen-Shannon Distance by up to 62.1%. At the same time, SHALA-LLM improves F1 by up to 16.7%, showing that modeling annotator disagreement can also strengthen classification performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03086v2">Beyond Code Pairs: Dialogue-Based Data Generation for LLM Code Translation</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable capabilities in code translation, yet their performance deteriorates in low-resource programming domains such as Fortran and emerging frameworks like CUDA, where high-quality parallel data are scarce. We present an automated dataset generation pipeline featuring a dual-LLM Questioner-Solver design that incorporates external knowledge from compilers and runtime feedback. Beyond traditional source-target code pair datasets, our approach additionally generates (1) verified translations with unit tests for assessing functional consistency and (2) multi-turn dialogues that capture the reasoning process behind translation refinement. Applied to Fortran-to-C++ and C++-to-CUDA, the pipeline yields 3.64k and 3.93k dialogues, respectively. Fine-tuning on this data yields dramatic improvements in functional correctness, boosting unit test success rates by over 56% on the challenging C++-to-CUDA task. We show that the generated data enables a 7B open-weight model to significantly outperform larger proprietary systems on key metrics like compilation success.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.23809v2">Advanced AI Service Provisioning in O-RAN through LLM Engine Integration</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      The Open Radio Access Network (O-RAN) architecture allows AI to be embedded directly into the RAN through modular xApps and rApps, yet creating these applications collecting data, training models, writing code, and deploying them safely remains slow and largely manual. Large Language Models (LLMs) offer strong reasoning and code-generation capabilities but are unsuited for the fast, deterministic inference required in real-time RAN control. We present a proof-of-concept Dual-Brain architecture that combines both strengths: an LLM-based orchestrator translates operator intents into data-collection policies and deployment code, while an automated ML engine, NeuralSmith, trains lightweight classifiers on demand via an API. We describe the architecture and provisioning workflow, share practical insights from a containerized O-RAN 5G~SA testbed, and discuss open research directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.05544v2">Activation-Informed Pareto-Guided Low-Rank Compression for Efficient LLM/VLM</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Large language models (LLM) and vision-language models (VLM) have achieved state-of-the-art performance, but they impose significant memory and computing challenges in deployment. We present a novel low-rank compression framework to address this challenge. First, we upper bound the change of network loss via layer-wise activation-based compression errors, filling a theoretical gap in the literature. We then formulate low-rank model compression as a bi-objective optimization and prove that a single uniform tolerance yields surrogate Pareto-optimal heterogeneous ranks. Based on our theoretical insights, we propose Pareto-Guided Singular Value Decomposition (PGSVD), a zero-shot pipeline that improves activation-aware compression via Pareto-guided rank selection and alternating least-squares implementation. We apply PGSVD to both LLM and VLM, showing better accuracy at the same compression levels and inference speedup.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05308v1">Statistically Reliable LLM-Based Ranking Evaluation via Prediction-Powered Inference</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 Accepted at ACL 2026 - GEM Workshop
    </div>
    <details class="paper-abstract">
      With PRECISE, we extended Prediction-Powered Inference to produce bias-corrected estimates of ranking evaluation metrics by combining a small human-labeled set with a large LLM-judged set. PPI is provably unbiased regardless of the LLM judge's error profile. We make it applicable to hierarchical metrics like Precision@K, where annotations are per-document but the metric is per-query, by reducing the output-space computation from O(2^|C|) to O(2^K). On the ESCI benchmark, augmenting 30 human annotations with Claude 3 Sonnet judgments reduces the standard error of Precision@4 estimates from 4.45 to 3.50 (a 21% relative reduction). In a production system, our framework correctly identified the best of three system variants from 100 human labels and 2 hours of domain-expert annotation; A/B testing confirmed this ranking with +407 bps in daily sales.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.26046v2">When Gradients Collide: Failure Modes of Multi-Objective Prompt Optimization for LLM Judges</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 Accepted at ACL 2026 - CustomNLP4U Workshop. Code, prompts and data available at https://github.com/adivekar-utexas/when-gradients-collide
    </div>
    <details class="paper-abstract">
      Customizing an LLM judge to a specific problem or domain often involves optimizing its prompt across multiple evaluation criteria simultaneously. Textual gradient methods automate this for a single judge criterion, however they produce natural-language critiques, not numerical vectors. Thus, the conflict-resolution toolkit of multi-task learning (PCGrad, MGDA) does not apply to this multi-objective textual gradient setting. We extend TextGrad to the multi-objective setting and test four decomposition modes of textual gradient optimizers by varying how much cross-objective information the loss, gradient and optimizer LLMs share. We find the gradient's task-focus drops by 59% (9.0 to 3.7 out of 10) when the gradient LLM must provide feedback on multiple criteria jointly. Separately, we observe that naively combining single-objective optimized instructions into a single prompt degrades Spearman rho from 0.305 to 0.220 (-0.085). These results identify two separable failure modes: optimization-time gradient dilution and inference-time instruction interference, which together constrain the design space for multi-objective judge optimization using textual feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29928v2">Label Over Logic? How Source Cues Bias Human Fallacy Judgments More Than LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      As AI-generated and AI-assisted content floods online spaces, source labels attached to such content can distort human reasoning judgments, with downstream consequences for moderation, evaluation, and decision-making. Whether LLMs share this vulnerability, or offer more source-agnostic evaluation, remains an open question with direct implications for human-AI collaboration. We examine this issue using logical fallacies as a controlled setting to isolate source-label effects on reasoning quality, independent of domain knowledge. We conduct an online study (N=505) where participants are assigned to a source condition (human, AI, human with AI assistance, AI with human assistance, or no disclosure) and evaluate comments containing logical fallacies, comparing their judgments with those of LLMs (GPT-5.2, Gemini 2.5 Flash, Claude Sonnet 4.5), who were evaluated across the same source conditions. Human evaluators were significantly more susceptible to fallacies labeled as written by human or human with AI assistance and assigned higher trust and evaluation ratings in these conditions. LLM evaluations remained comparatively stable across source labels, though performance varied across models. Confidence levels were similarly high across conditions for both humans and LLMs, regardless of fallacy presence. Our findings indicate that source-label bias in reasoning evaluation is primarily a human vulnerability and highlight the potential of human-LLM collaboration in increasingly AI-mediated environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05130v1">Towards Efficient and Evidence-grounded Mobility Prediction with LLM-Driven Agent</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Individual-level mobility prediction is central to urban simulation, transportation planning, and policy analysis. Supervised sequence models achieve strong accuracy but require task-specific training and offer limited decision-level transparency. Recent LLM-based methods improve interpretability, yet mostly rely on static prompts and single-pass inference, limiting their ability to seek additional evidence when mobility signals are weak or conflicting. We propose \method{}, a training-free LLM-driven agent framework that formulates next-location prediction as adaptive evidence-controlled decision making. \method{} resolves routine cases through a fast path based on historical regularity, while ambiguous cases trigger iterative tool use over recent trajectories, historical behavior, stay-move likelihood, and geographical evidence. Across three mobility datasets, AgentMob achieves the strongest overall performance among training-free LLM-based methods, with GPT-5.4 reaching 71.42\% Acc@1 on BW, 33.14\% on YJMob100K, and 33.50\% on Shanghai ISP. On BW non-fast-path cases, the LLM controller improves Acc@1 from 30.65\% to 48.62\% over a same-tool statistical baseline, showing that its main benefit lies in resolving ambiguous predictions through adaptive evidence gathering. Our code is available at https://github.com/Unknown-zoo/AgentMob.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05114v1">How Software Engineering Students Use LLMs to Write Research Papers: An Experience Report</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Large language models are increasingly becoming part of software engineering education, including activities involving empirical software engineering and evidence synthesis. This paper reports an educational experience involving the integration of reflective LLM use into an empirical methods assignment in a third-year software architecture course. Students were asked to develop a short research paper using either a rapid review or a gray literature review methodology and to disclose how LLMs were used throughout the assignment. We analyzed 146 student disclosure statements using a cross-analysis process combining LLM-assisted categorization with manual verification and refinement by the researchers. The reflections describe how students incorporated LLMs during activities such as brainstorming, methodological clarification, organization of findings, and writing refinement, while also reporting concerns regarding inaccuracies and verification of generated content. This experience report discusses lessons learned and educational implications for integrating AI-assisted technologies into empirical software engineering education.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.11974v2">CTIConnect: A Benchmark for Retrieval-Augmented LLMs over Heterogeneous Cyber Threat Intelligence</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 Accepted to KDD 2026
    </div>
    <details class="paper-abstract">
      Cyber Threat Intelligence (CTI) is foundational to modern cybersecurity, enabling organizations to proactively defend against evolving threats. However, the sheer volume and heterogeneity of CTI data, spanning structured knowledge bases (CVE, CWE, CAPEC, MITRE ATT&CK) and unstructured threat reports, far exceed the capacity of manual analysis. The strong contextual understanding and reasoning of Large Language Models (LLMs) have driven growing interest in applying them to CTI tasks. Yet no existing benchmark evaluates LLMs in a retrieval-augmented setting with a proper evaluation harness that grants access to the heterogeneous domain knowledge sources analysts rely on in practice. To address this gap, we present CTIConnect, a benchmark for systematically evaluating retrieval-augmented LLMs across the CTI task landscape. We construct a unified evaluation environment integrating five heterogeneous CTI sources into 1,860 expert-verified QA pairs spanning nine tasks across three categories: Entity Linking, Multi-Document Synthesis, and Entity Attribution. Extensive experiments on ten state-of-the-art LLMs reveal that the cross-source semantic gap manifests differently across task categories, demanding fundamentally different retrieval strategies, and that the performance bottleneck shifts between retrieval infrastructure and evidence utilization depending on the task. Our domain-specific strategies further outperform stronger general-purpose retrieval paradigms (retrieve-then-rerank, IRCoT), showing that closing this gap requires structural interventions rather than generic retrieval improvements. These findings hold across all ten LLMs, remain consistent on the full benchmark, and stay stable under temporal splits spanning 2008-2025. Together, they provide actionable guidance for designing scalable retrieval architectures over heterogeneous CTI ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.02493v2">Not What, But How: A Framework for Auditing LLM Responses across Positioning, Generalization, Anthromorphism, and Maxims</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 34 pages, 19 Figures, 4 Tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are being increasingly used to answer subjective, information-seeking questions, where users are sensitive to how responses are communicated, not just whether the answers are correct. Existing LLM evaluations for subjective cultural queries largely focus on factual correctness, ignoring how the response is framed. To this end, we introduce FRANZ, an automated FRAmework for respoNse characteriZation to conduct communicative audit of LLM responses along four dimensions: cultural positioning, use of generalizing language, anthropomorphic cues, and adherence to conversational maxims. To enable this evaluation, we contribute SQUARE - a corpus of 376k subjective questions sourced from 57 subreddits, and mapped to 7 countries and 19 question categories. We demonstrate FRANZ's applicability by scoring responses from three open-weight LLMs. We observe that LLMs show statistically significant differences in the frequency with which they employ each response characteristic. Unlike single-dimensional audits, FRANZ reveals that insider positioning and anthropomorphism are positively coupled, with the degree of coupling varying by country, providing a diagnostic lens for identifying framing divergences.
    </details>
</div>
