# llm - 2026_07

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02023v2">Not All Needles Are Found: How Fact Distribution and Don't Make It Up Prompts Shape Retrieval, Reasoning, and Hallucination in Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-14
      | 💬 16 pages, 8 figures, 2 tables. Accepted at the FAGEN Workshop @ ICML 2026
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) increasingly utilize massive context windows as working memory for autonomous tasks, their reliability fluctuates significantly depending on how information is distributed in real-world corpora. We investigate how fact placement, corpus-level distributions, and anti-hallucination ("Don't Make It Up") prompts influence model behavior by introducing a model-agnostic extended needle-in-a-haystack benchmark designed for scalability, which we apply to evaluate Gemini-2.5-flash, ChatGPT-5-mini, Claude-4.5-haiku, and Deepseek-v3.2-chat. Unlike prior work, we separately evaluate literal extraction, logical inference, and hallucination risk. We identify two critical failure modes: Distributional Collapse, where performance degrades significantly when evidence is dispersed; and a Safety Tax, where anti-hallucination prompts cause over-conservative refusal of present facts and evidence, sharply reducing accuracy. Our results suggest that many failures stem from ineffective context utilization, as models struggle to prioritize relevant information even when it is present. These findings highlight the need for model-specific robustness and effective context management to ensure reliable deployment in long-horizon agentic workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.13260v1">Discourse-Aware Policy Analysis with Argumentation: A Hybrid LLM-Symbolic Framework for Disaster Governance</a></div>
    <div class="paper-meta">
      📅 2026-07-14
    </div>
    <details class="paper-abstract">
      Policy documents shape governance outcomes, but their reasoning is often implicit. Participatory commitments and managerial control routinely coexist in the same text, and the tensions between them are rarely stated directly. Existing computational approaches to policy discourse cannot express the frame-mediated relations that drive these tensions, where one argument narrows or instrumentalizes another rather than rejecting it. End-to-end summarization by large language models produces fluent text but offers little structure that domain experts can inspect or contest. We present Apaf, a hybrid LLM--symbolic pipeline that operationalizes critical discourse analysis as a quantitative bipolar argumentation framework over policy text. Arguments are first classified into deliberative or managerial frames. Four frame-mediated relation subtypes (agency reduction, agenda shift, instrumental support, and normative support) are then produced by deterministic rules over LLM-extracted features. We release a novel dataset of 100 sub-documents of disaster-risk-reduction policy from the USA, UK, Canada, and Australia, and show that the resulting argument graphs are accurate, interpretable, and stable across jurisdictions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.13257v1">Can LLMs Learn and Apply Multi-Level Modelling Semantics? A First Empirical Study</a></div>
    <div class="paper-meta">
      📅 2026-07-14
    </div>
    <details class="paper-abstract">
      Industry 5.0 emphasises human-centric industrial system design, placing additional demands on modelling tools. Multi-level modelling (MLM) can directly represent three or more abstraction levels, but this comes at the cost of more complex semantic constraints that model correctness depends on. Large Language Models (LLMs) have been increasingly studied in model-driven engineering, but this evidence rests entirely on two-level modelling tasks, and whether it generalises to MLM, whose semantics differ in kind, remains untested. This paper presents the first empirical study of this question. We have three commercial LLMs (GPT-5.4, Claude Opus 4.6, and Gemini 3.1 Pro) generate multi-level models for the MULTI Warehouse Challenge in the SLICER language under six prompting strategies, yielding 90 generated models compared against a manually validated reference using fourteen metrics. Syntactic correctness is within reach, but semantic correctness is only partially achieved, with Instantiation/Specialisation Correctness ranging from 52% to 79%. Models reproduce content stated explicitly in the task text, but rarely complete structure and constraints the text implies without stating. Prompting strategies trade off precision against completeness, and self-checking functions mainly as a rule checker rather than reliably improving alignment with the reference design. Among the three LLMs, Claude shows the most balanced profile. These results clarify the boundaries of current LLM capability for MLM and inform the design of human-centred, AI-assisted modelling workflows for Industry 5.0.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.13158v1">Do LLMs Need Architectural Changes for Simultaneous Speech Translation? A Prefix-to-Prefix Data Driven Approach</a></div>
    <div class="paper-meta">
      📅 2026-07-14
    </div>
    <details class="paper-abstract">
      Simultaneous speech translation (SimulST) requires incremental translation under strict latency constraints, yet remains challenging for decoder-only LLM systems due to limited context and cross-lingual reordering. Recent approaches often introduce architectural changes or explicit read/write policies to control output timing, which can be brittle in conversational speech where segmentation boundaries are ambiguous. We present a simple data-driven alternative: fixed-length chunks for cumulative streaming decoding with a rewind-based committed prefix, and teacher-labeled prefix-to-prefix (P2P) targets with bounded waiting for fine-tuning, yielding CSSEL-P2P, where CSSEL is our proposed chunked streaming speech encoder LLM. In our in-house conversational speech evaluation, CSSEL-P2P improves streaming quality by +1.54 COMETKiwi over the CSSEL streaming baseline at comparable latency (+0.15s Average Lagging), suggesting effective SimulST without architectural changes via P2P supervision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.13124v1">ShortOPD: Recovering Pruned LLMs with Short-to-Long On-Policy Distillation</a></div>
    <div class="paper-meta">
      📅 2026-07-14
    </div>
    <details class="paper-abstract">
      Structured pruning is a hardware-friendly way to compress LLMs, but it is mostly validated on multiple-choice recognition tasks, while the same compressed checkpoints can collapse on the free-form generation that deployment actually requires. Two observations trace this gap. First, greedy \textsc{pass}@$1$ nearly vanishes after compression, yet \textsc{pass}@$k$ recovers substantially under repeated sampling: useful generations are demoted, not erased. Second, the recoverable regime fails mainly through suffix repetition. Recovery should therefore train on the compressed model's own on-policy states with dense token-level supervision, which On-Policy Distillation (OPD) provides by reusing the pre-compression model as a frozen teacher. However, long on-policy rollouts spend early recovery budget on low-information repetitive suffixes, delaying loss descent. To mitigate this waste, we propose \textbf{\shortopd}, a short-to-long OPD schedule that detects teacher-confirmed repetitive suffixes, treats the surviving prefix as each rollout's effective length, and allocates future rollout budgets to the effective lengths the policy can currently use. Across math, code, and open-ended generation, \shortopd\ raises the compressed model's score to about $9\times$ its unrecovered value and $1.6$--$4.4\times$ standard recovery recipes (SFT w/o KD, KD, and SeqKD), and it matches a fixed $8192$-token rollout horizon within two points using a quarter of the training time ($8.5$ vs.\ $35.9$ hours) and $71\%$ fewer rollout tokens. We hope this recipe helps move structured pruning beyond marginal gains on perplexity and multiple-choice benchmarks, a step closer to deployment-ready generation quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12986v1">Win by Silence: Deletion Non-Monotonicity, Autonomous Exploitation, and Typed-State Gating in LLM Plan Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-07-14
      | 💬 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Plan evaluators can reward a strategic plan for becoming less explicit. This paper studies that failure in a staged expected-value scorer for LLM-generated venture routes. Proposition 1 gives the score change from deleting an interior transition while retargeting its predecessor and retaining downstream value: Delta_k = (prod_{i<k} p_i)[c_k + (1 - p_k)R_{k+1}]. On a frozen 26-route cohort, all 57 admissible deletions matched the analytic identity and threshold sign, and every route had at least one score-improving deletion. A score-seeking optimizer, allowed to restructure routes but not told the exploit mechanism, found baseline-beating uncovered structures in 21/26 routes. GATE refused score release for 26/26 silenced routes with 0/26 honest suspensions; after refusal, 47/54 next revisions repaired to a covered structure, and strict covered improvement rose from 1/26 to 13/26. An adaptive compiler-aware co-author exposed the registry-provenance boundary: obligation-channel evasions remained 6/6 across all four v1/v1.5 conditions, while delta-indexed cost floors reduced beat-honest routes from 6/6 to 3/6 and fundability-by-silence from 5/6 to 0/6 without establishing semantic completeness. If a plan scores better only because it omits necessary work, the plan did not improve; the evaluation created an omission incentive. PCSC detects and neutralizes post-hoc omission splices over model-mediated typed-state records. In the cooperative setting tested, GATE acts as a deterministic search-shaping constraint, not merely a post-hoc filter. It does not verify the semantic completeness or real-world quality of arbitrary LLM-generated strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12885v1">LLM Judges Can Be Too Generous When There Is No Reference Answer</a></div>
    <div class="paper-meta">
      📅 2026-07-14
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      LLM judges are increasingly being used to evaluate open-ended model responses, often in no-reference settings where a ground-truth answer is unavailable. However, can they reliably assess in such evaluation setups? We explore this question in this paper through a two stage pipeline with a) calibration experiments that assess the judge model's knowledge of the task it is evaluating, and b) sensitivity experiments that assess how the judge model's performance is impacted by the presence and positioning of the reference answer in the prompt. Across experiments covering three languages, we show that the judge models we evaluated tend to over-credit incorrect answers in the absence of a reference answer, and adding reference answer information to the prompt flips the judge model's correct/incorrect decisions by as much as 85% in some experimental settings. Comparison with a subset of human annotations shows that these reference-driven changes generally align with human judgments. Our results emphasize the need for calibrating the LLM judges with a sample with reference-aware evaluation before using them in reference-free setups reliably, and our methodology provides a blueprint for researchers and practitioners in doing such calibration of LLM judges for other tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12875v1">MetaInfer: A Knowledge Only LLM Inference Engine Generator SKILL Toolbox</a></div>
    <div class="paper-meta">
      📅 2026-07-14
    </div>
    <details class="paper-abstract">
      As LLM technology advances, the space of model families, compute hardware, quantization schemes, parallelization strategies, and specialized optimization kernels continues to expand, sharply increasing the code complexity and maintenance cost of general-purpose inference frameworks. Conventional software engineering uses multiple layers of abstraction to support diverse application scenarios, but these abstractions also increase system complexity and may introduce additional performance overhead. This paper presents metainfer, an 'LLM-as-Compiler' approach in which users specify only the runtime constraints of an inference program. An LLM-driven multi-agent collaboration system, coupled with a contract knowledge base, then automatically generates a compact customized inference framework that satisfies these constraints. We evaluate metainfer from three perspectives: the effect of source-code reference, the runtime behavior and performance profile of engines generated under the zero-reference constraint on CKB-covered targets, and knowledge-base evolution for new model and platform scenarios. The results show that metainfer organizes generation constraints, validation feedback, and knowledge consolidation into a continuous closed loop, enabling runnable customized inference solutions to be generated from explicit knowledge. The code is publicly available at https://github.com/MetaInfer/MetaInfer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.24372v3">Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-07-14
      | 💬 Published at ICML 2026 main conference
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) for downstream tasks is an essential stage of modern AI deployment. Reinforcement learning (RL) has emerged as the dominant fine-tuning paradigm, underpinning many state-of-the-art LLMs. In contrast, evolution strategies (ES) has largely been overlooked due to the widespread belief that it does not scale to modern model sizes. This paper overturns this assumption by demonstrating the first successful application of ES to full-parameter fine-tuning of LLMs at the billion-parameter scale, without dimensionality reduction. ES can indeed search over extremely high-dimensional parameter spaces and outperform established RL implementations across multiple axes, including improved tolerance to long-horizon and delayed rewards, robustness across diverse base LLMs, reduced susceptibility to reward hacking, and improved training stability. These findings suggest that ES is not merely a viable alternative to RL, but a fundamentally different and powerful backpropagation-free post-training paradigm that opens a new direction for LLM fine-tuning beyond current RL-based approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.10183v2">Automated Tensor Scheduling for Hybrid CPU-GPU LLM Inference on Consumer Devices</a></div>
    <div class="paper-meta">
      📅 2026-07-14
    </div>
    <details class="paper-abstract">
      Running large language models on consumer devices such as laptops and desktops is challenging because model weights often exceed GPU memory capacity, making offloading inference necessary to extend effective model capacity with CPU memory. Existing offloading systems, however, typically rely on coarse layer-level or expert-level scheduling, which overlooks substantial heterogeneity among tensors within the same layer and adapts poorly to changing hardware load conditions on such devices. This paper presents ATSInfer, a hybrid CPU-GPU inference system for consumer devices that performs offloading at tensor granularity. ATSInfer combines static tensor placement with load-aware dynamic transfer, and introduces asynchronous CPU-GPU coordination to efficiently schedule hardware storage, data movement, and computation across heterogeneous backends. We implement ATSInfer and evaluate it on representative consumer platforms using both dense and MoE models. Compared with existing systems, ATSInfer improves prefill throughput by up to 1.94$\times$ and decode throughput by up to 3.29$\times$, while also increasing GPU utilization and making more effective use of PCIe bandwidth. These results show that ATSInfer can substantially improve the user experience of local LLM deployment on personal consumer devices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12835v1">Can LLMs Write Reliable Rubrics? A Meta-Evaluation for Experiment Reproduction</a></div>
    <div class="paper-meta">
      📅 2026-07-14
    </div>
    <details class="paper-abstract">
      Rubric-based evaluation is a promising approach for assessing open-ended outputs from LLM-based research agents, particularly in paper reproduction, where direct paper-to-repository comparison is prone to hallucination. However, constructing paper-specific rubrics requires substantial expert effort, limiting the scalability of benchmarks such as PaperBench. In this work, we present, to our knowledge, the first systematic meta-evaluation of LLM-generated rubrics for paper reproduction. We reformulate rubrics into a checklist-style format and evaluate four generation settings across two backbone models. We meta-evaluate generated rubrics intrinsically by semantic similarity and extrinsically by score alignment with ground-truth rubrics. Our results show that the augmented settings substantially improves downstream evaluation alignment, with the strongest setting approaching the human baseline, while intrinsic gains are more modest. Further analyses reveal that LLM-generated rubrics are often overly fine-grained, biased toward high scores, and less adaptive to paper domains, highlighting both the affordances and limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12733v1">LLMs Can See the Smoke but not the Fire: Evaluating Abductive Reasoning with Elenchos</a></div>
    <div class="paper-meta">
      📅 2026-07-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at pattern recognition and text generation, but their capacity for abductive inference - inferring latent hypotheses that explain observed behavior - remains poorly understood. Here, we introduce Elenchos (named after the Socratic method of cross-examination), a generative evaluation framework that measures abductive reasoning as a structural inverse problem. Given a reference formal system, such as the lambda-calculus, and a potentially mutated counterpart, agents must determine whether a mutation has occurred and infer the rule modifications responsible for the resulting behavioral differences. Evaluating frontier and mid-tier LLMs reveals a consistent detection-attribution dissociation: models often recognize that a system has been altered but struggle to identify the latent mutations causing the observed discrepancies. Performance degrades substantially under interacting mutations, where models frequently recover only a subset of the underlying mutations. Preliminary evidence also suggests diminishing returns from increased inference-time reasoning, with only modest improvements under larger reasoning budgets, though this finding requires further validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.15212v2">LLM vs. SAST: A Technical Analysis on Detecting Coding Bugs of GPT4-Advanced Data Analysis</a></div>
    <div class="paper-meta">
      📅 2026-07-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used for code understanding, yet their practical effectiveness for vulnerability detection relative to Static Application Security Testing (SAST) remains insufficiently quantified. We present a controlled comparative study between GPT-4 (Advanced Data Analysis) and two SAST tools (SonarQube and Cloud Defence) on 32 curated security scenarios representing common coding pitfalls. Each scenario is scored with a binary detection rule, the two SAST outputs are aggregated using a logical OR baseline, and paired outcomes are evaluated using McNemar's test for statistical significance. In our dataset, GPT-4 correctly detected 30 of 32 scenarios (93.75\%), while the aggregated SAST baseline detected 11 of 32. The paired comparison shows a statistically significant difference in detection performance in favour of GPT-4. We also discuss security considerations and operational constraints for integrating LLM-enhanced vulnerability scanning into secure software development workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12650v1">Evidence-Grounded Verified Agentic Reasoning: A Path Toward Eliminating LLM Hallucination in Empirical Inference via Tool-Attested Kernel Proofs</a></div>
    <div class="paper-meta">
      📅 2026-07-14
      | 💬 Accepted at the ICML 2026 TAIGR workshop. System name: EG-VAR
    </div>
    <details class="paper-abstract">
      Tool access alone does not make LLM empirical reasoning governable: accepted outputs need not descend from attested evidence, and accepted deductions need not hold up under formal scrutiny. We present EG-VAR (Evidence-Grounded Verified Agentic Reasoning), a Lean 4-based tool-calling architecture in which the Lean kernel is the sole minter of Verified claims via tool-attestation axioms and declared source lifts. Every verified output structurally descends from an attested tool call (Thm. 3.1) and a kernel-checked chain of valid inference (Thm. 3.2); residual outputs are honest Abstain with a replayable audit trail. On a subcollection of TableBench numerical reasoning (n=120), EG-VAR attains 120/120 versus a 95% same-tool baseline; on counterfactual stress tests (5 domains x 2 models), EG-VAR stays 100% source-faithful while same-tool drops to 80-90% (no-tool 50-80%). With the LLM as deployment-time formalizer, residual semantic-formalization error is 3.3% on Sonnet and 1.7% on Opus. We position EG-VAR as a technical-governance interface for high-stakes empirical claims: a formal sidecar makes the target proposition, source scope, evidence boundary, proof obligation, and abstention condition auditable, eliminating unsupported Verified outputs today while turning formalization errors, lift and source-authority disputes, ambiguities, and abstentions into explicit audit targets. Over time, typed sidecars in datasets, APIs, public records, and AI-generated documents can amortize this formalization burden into reusable infrastructure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18030v2">ParaTutor: Coordinating Parent and Child Math Tutoring through Role Separated LLM Scaffolding</a></div>
    <div class="paper-meta">
      📅 2026-07-14
    </div>
    <details class="paper-abstract">
      Parent and child tutoring is a collaborative learning setting with asymmetric roles. Parents guide children s problem solving, while children are expected to remain actively engaged in understanding and reasoning. However, most LLM based learning systems are designed for single users or relatively symmetric collaboration, leaving parent and child tutoring with distinct instructional roles underexplored. Through a formative study, we found that parent and child math tutoring was often disrupted by cognitive misalignment, emotional escalation, and method mismatch. To address these challenges, we present ParaTutor, a multiple agents LLM based scaffolding system for home math word problem tutoring. ParaTutor distributes support across user roles by providing parents with strategy, language, repair, and phase scaffolds, while providing children with visual grounding for problem interpretation. We evaluated ParaTutor with 23 parent and child dyads (children aged 10 to 12) across four tutoring conditions that varied how LLM assistance was delivered. Results show that generic LLM assistance often provided useful explanations but did not consistently support parent led tutoring or children s active reasoning. In contrast, ParaTutor helped redistribute tutoring work across parents and children, increased children s engagement with word problems, supported shared understanding through visual grounding, and helped parents translate LLM generated methods into child facing tutoring moves. These findings suggest that in family learning, the value of LLM support depends not only on model capability, but also on how support is coordinated across users with different roles. Our work contributes design implications for LLM systems that support role sensitive scaffolding in parent and child learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12631v1">Can Induced Emotion Bias LLM Behaviors in Sequential Decision Making?</a></div>
    <div class="paper-meta">
      📅 2026-07-14
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are increasingly deployed as autonomous agents in high-stakes domains, understanding contextual factors that may modulate their decision-making becomes critical. While LLMs are trained to perceive and resonate with users' emotions, it remains unclear whether induced emotion can influence their sequential decision-making. We investigate this question using the Iowa Gambling Task (IGT), a classic psychological paradigm for studying decision-making under uncertainty, combined with an imagination-based emotion induction procedure. We first validate the feasibility of this paradigm by confirming that LLMs can sense strong, distinguishable emotions from context and that LLM agents can learn from sequential interactions in a human-like pace. With the validated setup, we find that, different from humans, induced emotion does not significantly bias the decision dynamics of LLM agents on average. However, the effects of anger are conditioned: inducing anger makes LLM agents less sensitive to penalties for bad decisions, and in early stages of the game, anger can lower exploration, locking decisions into a few choices early. These findings reveal the subtle yet distinct effects of induced emotion on LLM decision-making compared to human behavior, and provide a tool for future research on affective modulation of LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12624v1">PVDetector: Detecting Prompt Injection Attacks on Purpose-Specific LLM Agents through Policy-Violation Concept Analysis</a></div>
    <div class="paper-meta">
      📅 2026-07-14
      | 💬 Accepted to ACM MM 2026. Code: https://github.com/Claresigle/PVDetector
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed as purpose-specific agents to handle domain-specific tasks such as customer service and code generation. These agents are expected to comply with not only generic safety guardrails but also purpose-specific restrictions tailored to their designated roles. Such additional restrictions enlarge the attack surface, particularly to prompt injection (PI) attacks. To defend against such attacks, existing detection methods primarily rely on analyzing input-output patterns, yet yield limited effectiveness. To address this limitation, we turn to analyzing the hidden activation space and discover that LLMs inherently retain latent policy-violation (PV) concepts when prompted with requests beyond their designated purpose. Particularly, PV concepts capture the semantics of conflicts between user queries and predefined restrictions, implicitly reflecting LLMs' intrinsic awareness of recognizing policy violations. Building on this insight, we propose PVDetector, a training-free framework that detects PI attacks during LLM inference by measuring hidden-state alignment with PV concepts, which are derived offline from the contrastive pairs of policy-violating and policy-compliant prompts. Experiments across multiple LLMs and datasets show that PVDetector achieves <1\% false negative rate with minimal auxiliary overhead, consistently outperforming state-of-the-art methods. Our code is available at https://github.com/Claresigle/PVDetector .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12621v1">Towards Vision-Free CIR: Attribute-Augmented Scoring and LLM-Based Reranking for Zero-Shot Composed Image Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-07-14
    </div>
    <details class="paper-abstract">
      Recent work has shown that "Vision-Free'' approaches (representing images as text) can be effective for standard image retrieval tasks. However, it remains unclear whether this paradigm can effectively handle a more complex, multimodal task, Composed Image Retrieval (CIR), due to the inherent information loss in textual descriptions. In this paper, we introduce a Vision-Free CIR framework that addresses this challenge through two key techniques: (1) Attribute-Augmented Hybrid Scoring, which compensates for lost visual details via explicit attribute matching, and (2) LLM-Based Reranking, which verifies semantic consistency of top candidates. Experiments on the open-domain CIRR dataset show that our approach outperforms existing Zero-shot CIR methods (44.04% R@1, +8.79%). On FashionIQ, our results highlight the trade-off between semantic reasoning and fine-grained visual matching. Ablation studies reveal that both attribute-augmented scoring and LLM-Based Reranking consistently improve performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12216v1">RCWT: Measuring Task-Budget Displacement from Coordination Content in LLM Calls</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 10 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Multi-agent and memory-augmented LLM systems often place coordination content, shared state, prior discussion, tool outputs, summaries, and role instructions, inside the same finite prompt used for the current task. This creates a practical allocation problem: every token spent on coordination is unavailable to task instructions or evidence when a call is assembled under a fixed context budget. We introduce the Roundtable Context Window Test (RCWT), a controlled protocol for measuring this task-budget displacement effect. RCWT varies coordination content while controlling total budget, position order, task family, and scoring. In the main context-dependent recall task at $W=4096$, three commercial models remain near baseline through moderate overhead and then degrade sharply once residual reference evidence falls to a few hundred tokens. Window-scaling summaries are consistent with a task-specific residual-budget interpretation rather than a fixed percentage threshold, but we treat this as descriptive evidence rather than a universal law. To test whether the fixed-budget cliff persists when task evidence remains intact, we add an intact-task ablation: the full task/reference block is kept present while coordination tokens increase by expanding total prompt length. In that setting, all tested calls return every scored field correctly across GPT-4.1-mini, Claude Haiku 4.5, and Gemini 2.5 Flash up to a 95\% coordination ratio. This ablation narrows the claim: the main RCWT cliff is best read as task-budget displacement, not as proof that coordination volume alone causes semantic interference in the original open-ended task. RCWT is therefore a measurement primitive for context-allocation budgeting, not a complete theory of multi-agent benefit or session-level coordination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.04689v3">Adaptive Testing for LLM Evaluation: A Psychometric Alternative to Static Benchmarks</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 ICML 2026 Spotlight
    </div>
    <details class="paper-abstract">
      Evaluating large language models (LLMs) typically requires thousands of benchmark items, making the process expensive, slow, and increasingly impractical at scale. Existing evaluation protocols rely on average accuracy over fixed item sets, treating all items as equally informative despite substantial variation in difficulty and discrimination. We introduce ATLAS, an adaptive testing framework based on Item Response Theory (IRT) that estimates model ability using Fisher information-guided item selection. ATLAS reduces the number of required items by up to 90% while maintaining measurement precision. For instance, it matches whole-bank ability estimates using only 41 items (0.157 MAE) on HellaSwag (5,600 items). We further reconstruct accuracy from ATLAS's ability estimates and find that reconstructed accuracies closely match raw accuracies across all five benchmarks, indicating that ability $θ$ preserves the global performance structure. At the same time, $θ$ provides finer discrimination within accuracy-equivalent models: among more than 3,000 evaluated models, 23-31% shift by more than 10 rank positions, and models with identical accuracies receive meaningfully different ability estimates. Code and calibrated item banks are available at https://github.com/Peiyu-Georgia-Li/ATLAS.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19387v2">Interpretable and Verifiable Hardware Generation with LLM-Driven Stepwise Refinement</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable success in software development. However, they are susceptible to hallucinations, meaning that they can introduce subtle semantic and logical errors. Due to the high stakes in chip design and manufacturing, hardware engineers are still reluctant to rely on LLMs for register-transfer level (RTL) generation. In this paper, we propose a hardware generation framework that combines the creativity and broad knowledge of LLMs with the explainability and mathematical rigor of formal methods. Specifically, we devise a set of transformation rules that cover various design decisions and hardware features. By iteratively applying these rules, an LLM agent can convert a design specification into an RTL program with guaranteed correctness. Experimental results demonstrate the effectiveness and efficiency of the framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12188v1">Cost-Governed RAG: Unified Per-Tenant Cost Attribution Across Retrieval and Generation in Multi-Tenant LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Enterprise Retrieval-Augmented Generation (RAG) deployments face a critical governance gap: while LLM generation cost is metered per token, the retrieval layer - vector memory, similarity compute, and embedding API calls - remains an unattributed shared cost, enabling invisible cross-subsidization among tenants. We present Cost-Governed RAG, an architecture that integrates a codebook-oblivious vector index (TurboVec) with a multi-tenant LLM governance gateway, creating a unified observability stack where embedding, retrieval, and generation costs are jointly attributable per tenant. The architecture exploits TurboVec's deterministic, closed-form memory formula to enable near-exact per-tenant retrieval cost calculation - a property unavailable in graph-based indexes with non-linear memory overhead. Deployed on Snowpark Container Services within a cloud data platform's governance boundary, the system achieves 99.96% end-to-end cost attribution accuracy across 100 simulated tenants (10M vectors, log-normal size distribution) with telemetry overhead below 0.04% of query latency. The architecture reduces retrieval infrastructure cost by 3.1-9.0x compared to managed vector database services under the pricing assumptions detailed in Section IV. We formalize a three-layer cost model and demonstrate that codebook-oblivious quantization enables deterministic per-tenant cost attribution while also removing the shared-codebook leakage surface present in trained quantizers - the latter observation being exploratory and subject to the limitations described in Section VII.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.12339v2">SheetMind: An End-to-End LLM-Powered Multi-Agent Framework for Spreadsheet Automation</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      We present SheetMind, a modular multi-agent framework powered by large language models (LLMs) for spreadsheet automation via natural language instructions. In this paper, we introduce a hierarchical agentic system consisting of three specialized agents: Manager Agent that decomposes complex user instructions into subtasks; an Action Agent that translates these into structured commands using a Backus-Naur Form (BNF) grammar; and a Reflection Agent that validates alignment between generated actions and the user's original intent. We evaluate SheetMind on the 221-task SheetCopilot Benchmark with GPT-3.5-Turbo. SheetMind achieved 100% execution success and 54.8% functional correctness, exceeding SheetCopilot (44.3%) while maintaining perfect execution reliability. We also conduct ablation study on a separately curated dataset to confirm that the full three-agent configuration consistently outperforms all partial variants. Lastly, we integrate our system into Google Sheets via a Workspace extension.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.27443v2">When Does Personality Composition Matter for Multi-Agent LLM Teams?</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 Accepted to COLM 2026 (20 pages, 6 figures)
    </div>
    <details class="paper-abstract">
      Personality prompting shapes how large language models communicate, yet whether these behavioral shifts affect objective task outcomes remains under-explored. Prior work shows that agents prompted with low agreeableness produce adversarial language, while those prompted with high agreeableness become cooperative, but the relationship between communication style and task performance has not been systematically examined across multiple domains. In this work, we investigate whether personality composition matters for multi-agent team performance by manipulating personality traits across frontier LLMs on three task domains: structured coding, open-ended research collaboration, and competitive bargaining. We find that personality effects depend critically on task structure. In coding tasks, low agreeableness leads to large communication shifts that have little effect on milestone completion. In open-ended collaboration and bargaining, the same manipulation substantially degrades performance. We discuss implications for multi-agent system design and the limits of personality manipulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12148v1">MindReader: Using LLMs to Encourage Memorable and Secure Password Replacement</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      We report on the design and evaluation of MindReader, a tool that helps a user replace her password when she is required to do so. Left to their own devices, users tend to replace their previous passwords with predictable variations of the original ones. MindReader leverages LLMs to suggest password variations that are chosen to be easy for the user to remember but harder for an attacker to predict. To do this, MindReader infers the meaning behind original password components and then suggests semantically related (yet syntactically unrelated) components for the new password. In a user study, passwords created using MindReader were more secure than both replacement passwords created without using MindReader and original passwords. In particular, MindReader replacement passwords were harder to guess in an online attack than alternative replacement passwords even by an attacker with knowledge of the original password and full knowledge of the tool implementation. Passwords created with MindReader were also comparably memorable to alternative replacement passwords and original passwords, as measured by the ability of users to successfully log in a week after creating their password.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.15239v4">Modeling Story Expectations: A Generative Framework using LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Consumers' engagement with stories is shaped by their expectations about what will happen next, yet modeling these forward-looking beliefs over unstructured narrative content has remained challenging. We develop a framework that uses large language models to approximate consumers' story expectations. Our method generates multiple imagined story continuations from a pre-trained LLM and extracts interpretable, theory-motivated features from these continuations, such as emotion and narrative path features. We propose two complementary validation procedures suited to different data availability: a survey-based approach that compares LLM-derived expectations to human-reported beliefs, and a rational-expectations approach that compares them to actual story outcomes. Applying the framework to both survey data collected in a controlled lab setting and observational data from an online reading platform, we find that LLM-derived expectations correlate with human-reported beliefs as well as actual story continuations along all features studied. In both settings, forward-looking expectations are associated with reader engagement above and beyond features of the content already consumed. Our framework provides a scalable method for modeling consumer beliefs about narrative content, with implications for content creation, platform strategy, and the study of narrative media.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.01311v3">Catalyst-Agent: Autonomous heterogeneous catalyst screening with an LLM Agent</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      The discovery of catalysts for electrochemical applications such as the oxygen reduction reaction (ORR), nitrogen reduction reaction (NRR), and CO2 reduction reaction (CO2RR) remains a central challenge in chemistry and materials science. Machine-learning interatomic potentials (MLIPs) and graph neural network models now accelerate individual adsorption-energy calculations by orders of magnitude relative to density functional theory. However, true large-scale screening is still blocked by human decisions: selecting candidates, constructing slabs, enumerating adsorption sites, interpreting descriptor failures, and choosing follow-up modifications. Here, we introduce Catalyst-Agent, a Model Context Protocol (MCP) server-based, LLM-powered agent that autonomously coordinates closed-loop catalyst screening. Catalyst-Agent searches materials databases through OPTIMADE, constructs slabs, computes adsorption energies using Meta FAIRchem's UMA MLIP within AdsorbML, evaluates reaction-specific descriptors, and applies structural modifications to refine near-miss candidates. In ORR, NRR, and CO2RR campaigns, Catalyst-Agent demonstrates high performance and converges in 1.40-3.41 trials per successful material on average. It identified Sn3Sc, Sn3Y, Tl3La, Pb3Y and In3Y as CO2RR candidates for further validation that were not previously reported in the literature. DFT single-point checks confirmed screening outcomes for representative NRR and CO2RR candidates. Ablations show these gains arise from chemically informed candidate selection and feedback-directed modification rather than brute-force evaluation: fully randomized screening dropped to 13.3%, 16.7%, and 0% success for ORR, NRR, and CO2RR, respectively. These results show that tool-grounded LLM agents can shift catalyst screening from manual trial-and-error toward more autonomous, reproducible and adaptive workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16726v2">Bridging Individual and Collective Realism in LLM-Based Human Mobility Simulation via Mobility Scaling-Law Guidance</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Geospatial applications such as urban planning, epidemic forecasting, and transportation demand modeling depend on individual mobility data, but such data are costly to collect, uneven in coverage, and privacy-sensitive. Human mobility simulation offers a scalable alternative. A recent line of work treats large language models (LLMs) as human agents, modeling individual cognitive processes to generate realistic trajectories. Yet because each agent is simulated in isolation, these methods provide no population-level coordination mechanism, and the collective regularities of real mobility - how trip distances, visited locations, and flows distribute across a population - fail to emerge. We close this gap with COMPASS, which turns empirical mobility scaling laws into a feedback signal that guides prompt construction. COMPASS starts from coarse, population-level adjustments driven by these scaling laws and progressively refines them into individual prompts, jointly satisfying multiple aggregate objectives while keeping individual trajectories realistic. Across two public datasets, COMPASS outperforms state-of-the-art LLM-based simulators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16727v2">Mobility-Aware Cache Framework for Scalable LLM-Based Human Mobility Simulation</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Simulating large-scale human mobility is fundamental to understanding population movement patterns and supporting real-world geospatial applications such as urban planning, epidemic response, and transportation analysis. Recent works treat large language models (LLMs) as human agents to simulate realistic mobility behaviors using structured reasoning, but their high computational cost limits scalability. To address this, we design a mobility-aware cache framework named MobCache that leverages reconstructible caches to enable efficient large-scale human mobility simulations. It consists of: (1) a reasoning component that encodes each reasoning step as a latent-space embedding and uses a latent-space evaluator to enable the reuse and recombination of reasoning steps; and (2) a decoding component that employs a lightweight decoder trained with mobility law-constrained distillation to translate latent-space reasoning chains into natural language, thereby improving simulation efficiency while maintaining fidelity. Experiments show that MobCache significantly improves efficiency across multiple dimensions while maintaining performance comparable to state-of-the-art LLM-based methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12089v1">Cross-Cutting Security Analysis of LLM-Generated Code via Metamorphic Testing and Association Rule Mining</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 This work has already been accepted by IEEE 27th International Conference on Information Reuse and Integration for Data Science
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) frequently generate code with security vulnerabilities, yet these weaknesses are rarely isolated: they often span multiple concern areas simultaneously, reflecting the cross-cutting nature of security in software. We present a framework that combines security-oriented Metamorphic Relations (MRs) with Association Rule (AR) mining to detect vulnerabilities in LLM-generated code, uncover their co-violation structure, and trace that structure back to prompt-level risk factors. We define nine MRs covering major CWE categories, including SQL injection, XSS, command injection, path traversal, hard-coded credentials, weak cryptography, and memory-safety errors, and apply them using an LLM-based judge to 3,700 code snippets generated by five open models from the LLMSecEval benchmark. The results show that 68.8% of snippets violate at least one MR, with hard-coded credentials (79.1%) and command injection (74.4%) among the most prevalent applicable failures. AR mining reveals strong cross-cutting co-violation patterns, notably that XSS and weak cryptography co-violations predict hard-coded credentials with 82.5% confidence (lift = 3.23), along with tightly coupled clusters linking authentication, credential handling, and cryptographic weakness, as well as input-handling and memory-safety failures. We then perform prompt-level risk analysis and find that database- and authentication-related prompts are strong predictors of broad cross-cutting insecurity, while 65.5% of prompts yield consistent violation outcomes across all five models. These findings show that insecure code generation is not merely a collection of independent defects, but a structured and prompt-conditioned phenomenon, motivating cluster-aware verification and prompt-level intervention for safer LLM-assisted programming.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12086v1">CityBehavEx: A Scalable and Empirically Validated LLM-Assisted Urban Simulation Platform</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 10 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Recent LLM-based multi-agent urban simulators can generate semantically rich city routines, but they remain costly to scale and are often weakly validated against empirical mobility patterns. We present CityBehavEx, an interactive LLM-assisted urban simulation platform that scales to city-size populations, exposes agent behavior for inspection, supports empirical validation, and generates mobility patterns that better match real-world spatial, temporal, and semantic distributions. Instead of invoking large language models for every agent action, CityBehavEx combines established human mobility models with fine-tuned cross-encoders that estimate semantic alignment between agent profiles, schedules, and activity transitions. This design enables large-scale simulations, as demonstrated in a case study of 100,000 agents over 75 days in under one hour on a single consumer GPU. The platform allows users to define simulation regions, launch experiments, inspect trajectories and activity traces, debug unrealistic behaviors, and validate generated routines against real-world mobility, time-use, and semantic metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.10125v3">D-LiFT: Improving LLM-based Decompiler Backend via Code Quality-driven Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      As one of the key tools in many security tasks, decompilers reconstruct human-readable source code from binaries. Yet, despite recent advances, their outputs often suffer from syntactic and semantic errors and remain difficult to read. Recently, with the advent of large language models (LLMs), researchers began to explore the potential of LLMs to refine decompiler output. Nevertheless, our study of these approaches reveals their problems, such as introducing new errors and relying on unreliable accuracy validation. In this paper, we present D-LIFT, an enhanced decompiler-LLM pipeline with a fine-tuned LLM using code quality-aware reinforcement learning. Unlike prior work that overlooks preserving accuracy, D-LIFT adheres to a key principle for enhancing the quality of decompiled code: preserving accuracy while improving readability. Central to D-LIFT, we propose D-Score, an integrated code quality assessment system to score the decompiled source code from multiple aspects, and use it to guide reinforcement learning fine-tuning and to select the best output during inference. In line with our principle, D-Score assigns low scores to any inaccurate output and only awards higher scores for readability to code that passes the accuracy check. Our implementation, based on Ghidra and a range of LLMs, demonstrates significant improvements for the accurate decompiled code from the coreutils and util-linux projects. Compared to baseline LLMs without D-Score-driven fine-tuning, our trained LLMs produce 55.3% more improved decompiled functions, as measured by D-Score. Overall, D-LIFT improves the quality of 68.2% of all the functions produced by the native decompiler.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12050v1">EFLUX: Elastic Multi-Robot Formation Navigation and Adaptation with Agentic LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Multi-robot teams operating in confined or cluttered environments must adapt both their formation geometry and group topology to navigate through complex obstacles. This adaptation requires two complementary behaviors: deformation, where the team continuously reshapes its geometry while remaining connected, and reconfiguration, where robots split into subgroups or merge back into a single formation. Existing methods often model these behaviors independently, connect them through handcrafted rules, or lack explicit geometric criteria for determining when each behavior should be invoked. However, challenging environments may require online changes in formation shape, connectivity, and effective team composition, making decoupled or rule-based approaches prone to suboptimal trajectories and deadlock. We propose EFLUX, a geometry-grounded LLM agentic framework for automatic and elastic multi-robot formation navigation. EFLUX extracts a structured scene representation and uses an LLM to reason jointly over both deformation actions, such as scaling and shearing, and reconfiguration actions, such as splitting and merging. These strategies are then translated into executable per-robot waypoints through a closed-loop generation, verification, and correction pipeline. Simulation and hardware experiments show that EFLUX enables safe, continuous, and elastic formation navigation in constrained environments, reducing deadlock and navigation failures compared with baselines while maintaining coherent multi-robot coordination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11881v1">Metacognition in LLMs: Foundations, Progress, and Opportunities</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Metacognition is a foundational component of intelligence critical to effective learning, problem solving, decision-making, communication, and more. In recent years, it has become increasingly recognized as a cornerstone of capable, transparent AI systems. Yet while LLMs have made significant progress across diverse real-world tasks, it is not yet clear when, how, or to what extent they can exhibit or be endowed with effective metacognitive abilities, nor how such abilities can be adapted to advance the fundamental capabilities, reliability, and intelligence of AI systems. This paper bridges this gap by presenting the first comprehensive overview of the current state of knowledge on metacognition for LLMs. We analyze and taxonomize the landscape of this emerging field and summarize recent technical advancements, including methods and benchmarks to measure and evaluate LLMs' metacognitive abilities, techniques to elicit, improve, and apply metacognition in LLMs, and findings and implications of ongoing research. We also discuss applications, open questions and challenges, and promising directions for future work. Our aim is to provide a detailed and up-to-date review of this topic and stimulate meaningful research and discussion. An organized list of papers can be found at https://github.com/yale-nlp/LLM-Metacognition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11871v1">Inside the Unfair Judge: A Mechanistic Interpretability Account of LLM-as-Judge Bias</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 58 pages, 13 figures, 30 tables; project page: https://xzx34.github.io/unfair-judge/
    </div>
    <details class="paper-abstract">
      Existing studies of LLM-as-judge scoring bias work predominantly at the input-output level: they perturb inputs, measure score deltas, and propose prompt-level mitigations. We argue that the same biases admit a representation-level account in the judge's hidden state, complementary to the input-output view and operationally useful in ways it does not afford. We report three findings, across seven judges, seven bias types, and nine benchmarks. Geometry: baseline judging inputs occupy a tight activation manifold while biased inputs are displaced along a low-dimensional, type-specific subspace that sharpens with depth and is recovered consistently by three families of estimators. Causal control: steering hidden states along this subspace drives scoring in both directions, forward shifts reproducing biased scoring on clean inputs and reverse shifts restoring baseline scoring on biased ones, while matched-norm random directions produce shifts an order of magnitude smaller. Operational: a simple linear projection onto the same bias-direction features anticipates judge failures on three entirely unseen benchmarks, substantially outperforming text-based alternatives. Reading bias as activation geometry, rather than as input-output noise, unifies geometric structure, causal control, and operational prediction within a single framework. The project page is available at https://xzx34.github.io/unfair-judge/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.23276v2">Auditable Context-Aware HFMD Forecasting with Structured LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Effective HFMD surveillance requires forecasts capturing both time-series patterns and contextual drivers such as school calendars, weather, and policy or surveillance reports. In clinical settings, forecasts must be trusted and actionable; thus, beyond point accuracy, decision-makers require concise, auditable explanations of why risk is expected to rise or fall. Classical models (e.g., ARIMA and Prophet) and foundation models (e.g., Chronos, Moirai, and TimesFM) treat external covariates as numerical inputs, lacking semantic reasoning to reflect epidemiological mechanisms or resolve conflicting signals. We propose a two-agent neuro-symbolic framework that decouples contextual interpretation from probabilistic forecasting. An LLM-based Event Interpreter ingests heterogeneous signals -- school schedules, weather summaries, government reports, and clinical guidelines -- and outputs a scalar transmission-impact signal. A Forecast Generator combines this signal with historical case counts to produce point forecasts that are mapped to probabilistic predictions through Poisson/negative-binomial moment matching. We focus on one-week-ahead rolling forecasts, aligning with weekly hospital-capacity planning and the rapid, context-driven inflections typical of HFMD. We evaluate on two datasets: Hong Kong surveillance (90 target weeks in 2023--2024) and Lishui hospital visits (33 target weeks in 2024). Against traditional and foundation-model baselines, our approach achieves competitive point accuracy while providing robust 90\% intervals (coverage approximately 0.85--1.00) and concise rationales. This demonstrates that integrating domain knowledge through LLM-based agents can match strong numerical forecasters while yielding interpretable, context-aware forecasts aligned with public-health decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11810v1">Supporting Reflection in LLM-based Exploratory Search</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can make exploratory search more efficient but may undermine the reflection and iterative sensemaking needed in unfamiliar domains. Existing LLM tools often prioritize rapid answers over supporting users in tracking how their understanding evolves and how well their strategies align with their goals. We present TrailLM, a system that helps users reconstruct and revisit their exploration paths to support reflection and metacognitive engagement during information seeking. By aligning LLM assistance with users' sensemaking workflows, TrailLM aims to preserve the benefits of LLM-based search while enhancing opportunities for critical reflection on one's own search process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.13088v1">Securing LLMs in the Wild: Privacy and Security Challenges at the Edge</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are rapidly moving from research settings into the wild, deployed on enterprise infrastructure, personal devices, and edge platforms. While cloud deployments offer scalable compute, concerns over data sovereignty, compliance, latency, and third-party dependence are driving organizations toward edge and on-premise LLMs. This shift introduces new security and privacy challenges: limited compute and memory force aggressive optimizations, including quantization, pruning, model partitioning, and parameter-efficient adaptation, each of which can introduce vulnerabilities and reshape the threat landscape. We describe this tension as the Security-Efficiency Paradox, mechanisms that improve efficiency may weaken robustness, expose new attack surfaces, or increase privacy risks. We examine how compression can degrade safety alignment, how partitioned inference enables reconstruction attacks, and how continuous local adaptation may cause privacy leakage and model drift. To analyze these risks, we introduce a deployment-centric taxonomy organized around three architectural constraints: the Memory Wall, the Quadratic Wall, and the Compute Wall. We derive a unified constraint model that quantifies when unsafe optimizations become unavoidable, linking each wall to specific attack surfaces. Building on this model, we propose the Secure Operational Efficiency Score (SOES), a holistic metric balancing task accuracy, jailbreak resistance, and privacy against energy, memory, and latency, enabling practitioners to configure edge LLMs under real-world hardware limits. We further present a practical decision procedure and targeted mitigations for each optimization-induced vulnerability. Together, these contributions provide a co-designed framework for jointly evaluating security, privacy, and efficiency, laying a foundation for securing edge-native intelligent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11706v1">VoxENES 2026: Benchmarking Generalization of Speech Spoofing Detectors Against LLM-Era TTS and Voice Conversion</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 Accepted in InterSpeech 2026
    </div>
    <details class="paper-abstract">
      Modern LLM-driven text-to-speech (TTS) and voice conversion (VC) systems produce synthetic speech that differs from the generators represented in many legacy spoofing benchmarks. This mismatch creates a temporal generalization gap that can overestimate detector robustness under real-world post-processing conditions. We bridge this gap by introducing VoxENES 2026, a bilingual (English and Spanish) benchmark of 53,628 audio samples generated using 10 contemporary speech synthesis methods and evaluated under 10 standardized post-processing conditions. Using VoxENES 2026, we benchmark eight pretrained detectors without fine-tuning and observe substantial performance degradation: the best model achieves 28.98\% EER overall, while most perform near or below random chance across modern generators and perturbations. Our results highlight the reliance on brittle artifacts in current detectors and establish VoxENES 2026 as a practical testbed for developing robust audio spoofing countermeasures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11703v1">Production and Perception in LLMs: A Token Probability Approach</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      The asymmetry between language production and perception has been well-documented in psycholinguistics. Whether large language models (LLMs) exhibit a functionally analogous distinction remains an open question, particularly given that LLMs rely on the same underlying mechanism (next-token prediction) for both input and output processing. In this exploratory study, we operationalize the production-perception distinction through direct token probability measurements rather than metalinguistic prompting. Using the base Llama-3.1-8B model, we generated poems under a production prompt and re-scored the same tokens under both rephrased production prompts and perception-oriented prompts. Across an extended experiment with four production and three perception prompts, production-perception distances consistently and substantially exceeded production-production distances, with non-overlapping ranges across conditions and an overall average ratio of approximately 1.8. Near-ceiling correlations in the production-production control confirm that the effect is specific to communicative framing rather than prompt surface variation, and we show the effect replicates across five open-weight models (Llama-3.1-8B, EuroLLM-9B, gemma-2-9b-it, Mistral-7B-Instruct-v0.3, and Qwen2.5-7B-Instruct), spanning both base and instruction-tuned variants. Temporal analysis revealed that the perception prompt exerts its strongest influence at the beginning of the sequence, with divergence decaying as generated context accumulates, though the specific shape of this decay varies across prompt pairs. These findings suggest that prompt framing alone induces a production-perception distinction in LLM probability distributions, even within a decoder-only architecture.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11683v1">RAGU: A Multi-Step GraphRAG Engine with a Compact Domain-Adapted LLM</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Graph retrieval-augmented generation (GraphRAG) enhances large language models with structured knowledge, yet existing systems construct knowledge graphs in a single extraction pass, producing noisy entities and brittle retrieval. RAGU, an open-source modular GraphRAG engine, addresses this by separating extraction from consolidation: entities and relations pass through two-stage typed extraction, DBSCAN-backed deduplication, LLM summarization, and Leiden community detection. A key insight motivates a compact extractor: the skills an in-pipeline LLM needs - comprehension, extraction, reasoning over context - are language skills that grow only weakly with model size, unlike factual world knowledge. Accordingly, we train Meno-Lite-0.1, a 7B model optimized for language skills, which outperforms Qwen2.5-32B on knowledge-graph construction (+12.5% relative harmonic mean) and matches it on English GraphRAG tasks. On GraphRAG-Bench (Medical), RAGU retrieves the most complete context at every factoid level (evidence recall up to 0.84 vs. $\leq$0.76) and overtakes HippoRAG2 on synthesis tasks; on multi-hop factoid QA, the apparent HippoRAG2 advantage is shown to be largely an answer-format artifact. RAGU is installable via $\texttt{pip install graph_ragu}$, runs on a single GPU, and is released under MIT. The source code is publicly available at https://github.com/RaguTeam/RAGU, and the Meno-Lite-0.1 model can be obtained from https://huggingface.co/bond005/meno-lite-0.1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11614v1">Extending LLM Context via Associative Recurrent Memory</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Extending the context length of large language models (LLMs) is critical for many real-world applications, yet standard transformers remain constrained by quadratic compute and linear memory scaling. In this work, we investigate the Associative Recurrent Memory Transformer (ARMT) as a practical approach for enabling long-context processing in LLMs, constant memory scaling, and better efficiency. We make three main contributions. First, we construct two domain-specific long-context datasets designed to evaluate realistic workloads, focusing on narrow-domain fine-tuning scenarios. Second, we propose a comprehensive training recipe for ARMT-based context extension, combining continued pre-training, synthetic long-context data generation, curriculum learning, and selective integration of associative memory into chosen model layers. Third, we present an extensive experimental study demonstrating that ARMT-augmented models: (i) process inputs well beyond their original context limits without degrading performance relative to in-limit baselines; (ii) generalize more effectively to out-of-distribution context lengths; and (iii) need 30% less FLOPs while preserving baseline performance within the original context window.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.00262v3">LLM-Driven Cost-Effective Requirements Change Impact Analysis</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 33 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Requirements are inherently subject to changes throughout the software development lifecycle. Within the limited budget available to requirements engineers, manually identifying the impact of such changes on other requirements is both error-prone and effort-intensive. That might lead to overlooked impacted requirements, which, if not properly managed, can cause serious issues in the downstream tasks. Inspired by the growing potential of large language models (LLMs) across diverse domains, we propose ProReFiCIA, an LLM-driven approach for automatically identifying the impacted requirements when changes occur. We conduct an extensive evaluation of ProReFiCIA using several LLMs and prompts variants tailored to this task. Using the best combination of an LLM and a prompt variant, ProReFiCIA achieves a recall of 85.7% on an unseen industrial dataset, demonstrating its effectiveness in identifying impacted requirements. Further, the cost of applying ProReFiCIA remains small, as the engineer only needs to review the predicted impacted requirements, which represent 3.0% of the entire set of requirements. Lastly, incorporating domain knowledge into the model via RAG increases recall to 95.8% while slightly raising the cost to only 3.4%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11591v1">Similarity-Guided Curriculum Fine-Tuning of LLMs for Neural Architecture Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Introduce a MinHash-based similarity scheduling framework that constructs a progressive curriculum over neural architecture code for LLM-based neural architecture search (NAS). Using 128-permutation MinHash signatures over normalised 7-gram source code shingles, we partition the reference pool into similarity bands and present them in increasing architectural heterogeneity, with the best LoRA adapter from each stage merged cumulatively into the backbone. We evaluate the framework on OlympicCoder-7B within the LEMUR benchmark on CIFAR-10 image classification, generating N =15 candidate architectures per epoch across six progressive fine-tuning steps. The curriculum achieves 60% peak success rate at the high-similarity level without post-processing repair. A 2*2 ablation at the most diverse level curriculum versus base model, with versus without partial interface repair reveals that without repair the base model (47% peak SR) substantially outperforms the curriculum model (7% SR), while adding partial repair brings both to 53% SR. This pattern is consistent with merge-level weight drift progressively erasing evaluator-interface priors, and suggests that interface repair and curriculum scheduling target distinct failure modes. We further report a cross-dataset transfer observation on SVHN, where direct base-model generation without curriculum warmup yields 27% peak SR at substantially lower accuracy (60.5%) than the CIFAR-10 equivalent, consistent with the increased synthesis difficulty of the unq-family anchor architecture.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11573v1">Knowledge-Guided Synthetic Bug Feedback for LLM-Based Unit Test Generation</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 12 pages, 7 figures, 6 tables. Preprint
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have opened new opportunities for unit test generation, but executable tests do not necessarily reveal real defects. This paper studies how historical real-bug mechanisms can be transformed into executable feedback targets for LLM-based unit test generation. The proposed framework constructs structural and semantic representations of real-bug records, retrieves mechanisms applicable to a focal method, and instantiates them as synthetic bugs that guide iterative test enhancement. We evaluate the approach on method-level real-bug detection tasks from Defects4J and show that mechanism-guided synthetic-bug feedback improves real-bug detection over execution-, coverage-, mutation-, knowledge-, and search-based baselines. The results suggest that organizing real-bug mechanisms as retrievable and executable feedback targets is an effective way to guide generated tests toward bug-triggering inputs and behavioral oracles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11564v1">PaperRouter-Agent: A Content-Grounded LLM Agent for Personalized Hierarchical Paper Routing</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Researchers organize the papers they collect into personal folder hierarchies in reference managers, and route each new paper into the folder where it belongs. This task differs from standard hierarchical text classification. A user's folder hierarchy is not a fixed, shared taxonomy but a private and evolving folksonomy whose folder meanings may be topical, shorthand, venue-based, or process-oriented, and are often defined by the papers already stored inside them. We formalize this setting as personalized hierarchical paper routing (PHPR): assigning an incoming paper to folders in a user-specific hierarchy without per-user training. We propose PaperRouter-Agent, a training-free LLM agent that grounds routing decisions in folder members rather than folder names alone. The agent first narrows the candidate hierarchy, retrieves folder-specific evidence, verifies fit by inspecting member papers, and incorporates similarity-gated feedback from past user rejections. A formative study on real personal libraries shows that PaperRouter-Agent raises overall Recall@1 from 0.39 to 0.61 and Recall@3 from 0.57 to 0.83, with the largest gains on organizational folders defined by metadata such as venue or year, where single-shot methods collapses (Recall@1 0.09 to 0.50). On the public LaMP-2 benchmark, the same approach improves accuracy from 44.5% to 51.5% (+9.0 macro-F1) over a single-shot baseline, while remaining low-cost for practical use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.16395v3">LLM-Driven Collaborative Model for Untangling Commits via Explicit and Implicit Dependency Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Atomic commits, which address a single development concern, are a best practice in software development. In practice, however, developers often produce tangled commits that mix unrelated changes, complicating code review and maintenance. Prior untangling approaches (rule-based, feature-based, or graph-based) have made progress but typically rely on shallow signals and struggle to distinguish explicit dependencies (e.g., control/data flow) from implicit ones (e.g., semantic or conceptual relationships). In this paper, we propose ColaUntangle, a new collaborative consultation framework for commit untangling that models both explicit and implicit dependencies among code changes. ColaUntangle integrates Large Language Model (LLM)-driven agents in a multi-agent architecture: one agent specializes in explicit dependencies, another in implicit ones, and a reviewer agent synthesizes their perspectives through iterative consultation. To capture structural and contextual information, we construct Explicit and Implicit Contexts, enabling agents to reason over code relationships with both symbolic and semantic depth. We evaluate ColaUntangle on two widely-used datasets (1,612 C# and 14k Java tangled commits). Experimental results show that ColaUntangle outperforms the best-performing baseline, achieving an improvement of 44% on the C# dataset and 82% on the Java dataset. These findings highlight the potential of LLM-based collaborative frameworks for advancing automated commit untangling tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11528v1">HermesHFL: Incentive-Compatible Hierarchical Federated Unlearning for Dynamic LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 17pages,8 figures
    </div>
    <details class="paper-abstract">
      Hierarchical federated unlearning (HFUL) for large language model (LLM) fine-tuning faces significant challenges due to hierarchical aggregation, dynamic client participation, and strong parameter coupling in LLM adaptation. Selectively removing client contributions is particularly difficult because model updates propagate across multiple aggregation stages while unlearning requests may coincide with client departures and rejoining. To address these issues, we propose **HermesHFL**, a hierarchical federated learning framework that supports selective unlearning, dynamic client participation, and client reintegration for scalable LLM fine-tuning via parameter-efficient fine-tuning (PEFT) with LoRA. We formulate a unified optimization problem that jointly models client participation, edge association, incentive allocation, and unlearning under heterogeneous client behaviors. To solve this problem efficiently, we develop **Neogen**, a neural-guided bilevel evolutionary optimization framework that combines CMA-ES for continuous incentive optimization with a CHC-based evolutionary mechanism for discrete participation and association decisions. A neural surrogate further accelerates optimization and improves search efficiency. Extensive experiments on LLM fine-tuning tasks demonstrate that HermesHFL consistently outperforms state-of-the-art baselines in model utility, unlearning effectiveness, convergence stability, and resource efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11517v1">Graph-Based Structural Evaluation of LLM-Translated Adversary Emulation Procedures</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 This technical contribution supports the MITRE white paper titled: Evaluating LLMs for Impact-Faithful Translation of Adversary Behavior Across Operating Systems
    </div>
    <details class="paper-abstract">
      Adversary emulation plans describe multi-step attacker procedures using MITRE ATT&CK techniques, privilege requirements, and observable telemetry. Translating them across operating systems supports cross-platform defender evaluation, and large language models (LLMs) can automate this task. However, a translation may only rename tools while retaining source-platform logic, giving defenders little target-platform coverage. Binary scoring can overestimate fidelity because it measures countable features rather than structural, observable, and rule-level equivalence. Graph-Based Structural Evaluation (GBSE) models each procedure as a directed attributed graph and calculates normalized Graph Edit Distance (GED) across four layers: technique, tactic, telemetry class, and Sigma logsource. GBSE was applied to a 29-step ALPHV/BlackCat Windows-to-Linux plan, comparing a reconstructed Windows control with the unmodified LLM-generated Linux version. Technique and tactic structure were fully preserved (GED=0, similarity=1.000). Telemetry similarity fell to 0.897 (GED=3) because three steps contained unmapped or drifting observables, while Sigma logsource similarity was 1.000. Every state was classified as Medium Fidelity, with a best composite score of 0.674. The 0.80 deployment threshold was not reached because technical realism scored 0.43 against the required 0.990. The framework includes bipartite GED, a telemetry-intent parser that converts free text into observable classes, and 49 validated Sigma rules: 19 for Linux and 30 for Windows. The rules provide complete ATT&CK technique coverage and pass validation with zero findings. Additional analysis reveals technique-level divergence, including RDP-based external access mapped to unencrypted exfiltration and credential-store access mapped to remote-system discovery. Results were reproduced and verified against recorded outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11505v1">Proxy Exploration and Reusable Guidance: A Modular LLM Post-Training Paradigm via Proxy-Guided Update Signals</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Post-training is essential for refining the domain-specific capabilities of large language models (LLMs), yet existing reward optimization and distribution matching methods tightly couple policy exploration with distribution alignment. This coupling forces expensive exploration directly on the policy model and severely hinders the asynchronous generation, reuse, and cross-model transfer of optimization signals. In this paper, we propose Proxy-guided Update Signal Transfer (PUST), a novel post-training framework that fundamentally decouples update-signal exploration from distribution alignment. Instead of utilizing the primary model for costly exploration, PUST employs a lightweight proxy model as an efficient testbed to discover high-reward behaviors. We extract the relative improvement signal between the proxy's initial and optimized states, transferring this directional update to the primary model to guide its policy alignment. This decoupled pipeline, comprising proxy exploration, update-signal extraction, and signal transfer, significantly reduces computational overhead and enables optimization signals to be asynchronously generated, cached, and reused. Crucially, by transferring relative improvements rather than absolute policy distributions, PUST naturally supports weak-to-strong improvement and seamless cross-model transfer. Systematic evaluations on Qwen3-family models across math and code domains demonstrate that update signals extracted from substantially weaker proxies can robustly and adjustably enhance stronger primary models. Ultimately, PUST transforms post-training from a monolithic online optimization process into a highly modular, reusable, and cost-efficient paradigm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11471v1">Are LLMs ready for HardChoices?</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 Accepted to Konvens 2026
    </div>
    <details class="paper-abstract">
      A lot of research attention has been devoted to checking whether large language models (LLMs) are politically biased. This work has largely focused on high-level ideological dimensions, such as left--right or progressive--conservative, and it has been shown that while LLMs are predominantly left and progressive leaning, largely mimicking the biases in the training data, they can be to some extent steered to change their preferences in post-training. In this short note, we check if LLMs have robust stances with regard to major substantive societal issues, on which members of the same ideological camp are often in disagreement, summarised in a novel dataset \textsc{HardChoices}. We show that, faced with this line of questioning, LLMs, both large and small, surprisingly rarely declare neutrality, are often incoherent, and demonstrate a remarkable degree of agreement on issues where they do take stances.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11451v1">ManiScope: LLM-Assisted Visual Analytics of Cryptocurrency Manipulation Risk</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Cryptocurrency markets are vulnerable to trade-based manipulation, such as wash trading, which can distort price signals and mislead investors. Prior research has mainly focused on detecting manipulation using fixed rules or labeled examples, offering limited flexibility and interpretability for assessing potential risks. Existing visual analytics tools can reveal basic manipulation-related signals, such as token distribution, but still require substantial manual effort to integrate holder relationships, suspicious behaviors, and market dynamics for risk assessment. To address these limitations, we propose ManiScope, an LLM-assisted visual analytics system for analyzing trade-based manipulation risks in cryptocurrency markets. ManiScope provides coordinated views of token distributions, holder relationships, detailed holder behaviors, price dynamics, and suspicious trading patterns. To further enhance user analysis, ManiScope introduces a human-LLM collaborative visual analytics framework. Rather than acting as a basic reactive LLM assistant, the framework positions the LLM as a co-analyst that infers users' analytical intent and emerging hypotheses from interaction context and surfaces relevant visual, statistical, and synthesized evidence for hypothesis evaluation. This design reduces repetitive inspection and strengthens evidence-based reasoning. We evaluate ManiScope through two case studies and a user study with 12 experienced cryptocurrency practitioners. The results suggest that ManiScope supports effective risk assessment of manipulation, reduces manual effort in evidence-seeking, and organizes findings around user hypotheses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11414v1">Confidently Wrong: Detecting Hallucinations in Financial Question Answering from LLM Internal States</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 8 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) in financial applications fail most consequentially when they are confidently wrong. Hedged, uncertain answers invite scrutiny, whereas confident errors silently degrade downstream decisions without warning. We ask how reliably such confidently wrong answers, or confident hallucinations, can be detected from a model's internal activations, and whether those activations carry information beyond its observable outputs. We train linear probes on the residual stream and evaluate them on two established question-answering (QA) benchmarks built from real filings, FinQA and TAT-QA. Behavioral confidence is measured as the agreement among eight resampled answers to the same question, and probe effectiveness is compared against baselines, such as token log-probabilities and the model's own True/False self-assessment of its answer. Our findings show that among confident answers, those for which all eight resamples agree, 15-23% are wrong on FinQA. There the probes have a significant advantage over baseline methods in detecting hallucinations, holding 0.68-0.77 AUROC while the best baselines fall to 0.55-0.63, across Qwen3-8B, Llama-3.1-8B, and Gemma-2-9B. Our results suggest that probing can be a cost-effective triage mechanism for routing LLM answers to human review and quality control procedures in high-stakes financial applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11390v1">TerraRepair: A Tool-Grounded LLM Agent for Infrastructure-as-Code Repair</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 The paper has been peer reviewed and accepted for publication in the proceedings of the 20th International Symposium on Empirical Software Engineering and Measurement (ESEM 2026)
    </div>
    <details class="paper-abstract">
      Background: Infrastructure-as-Code (IaC) scanners detect cloud misconfigurations in Terraform and other IaC languages before deployment, but repairing the flagged configurations remains largely manual. Recent Large Language Model (LLM)-based repair approaches can repair some findings, but may hallucinate unsupported constructs or suppress warnings without fixing the issue. Aims: We study whether tool grounding can improve LLM-based Terraform repair, and when a finding should be escalated because the required deploymnet-specific context is not availble. Method: We present TerraRepair, a prototype of a tool-grounded LLM agent for Terraform repair with structured escalation. TerraRepair retrieves dependency context from Terraform references, consults the installed provider schema, and re-runs the scanner before returning a candidate repair. Then teh required context is absent, TerraRepair escalates instead of fabricating a plausible fix. Results: We evaluate our tool on two vulnerable-by-design Terraform repositories using two IaC security scanners, Checkov and Trivy, across AWS, Azure, and GCP. On the combined AWS benchmark, TerraRepair improves scanner-verified fix rates from 26.6% to 78.4% on Checkov and from 44.8% to 72.4% on Trivy, compared with a controlled one-shot baseline. It repairs are labelled as correct under a majority-vote protocol. Conclusions: These emerging results show that tool grounding can substantially improve scanner-verified LLM-based IaC repair on the studied benchmarks, while missing deployment-specific context remains the main knowledge boundary for full autonomy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11363v1">Beyond Sally-Anne: Evaluating Theory of Mind in LLMs using Epistemic Schelling Points</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Text-based evaluations of Theory of Mind (ToM) in Large Language Models (LLMs) often involve cognitive tests akin to the Sally-Anne task that can be gamed due to exposure to relevantly similar tasks in pre-training and do not obviously test models' functional ToM abilities in ways that generalize to naturalistic settings. To address these issues, we introduce the Epistemic Asymmetry Schelling Task (EAST), a two-player dialogue game designed to benchmark robust and generalizable ToM abilities. By requiring LLM-LLM dyads to independently converge on semantic Schelling points under varying states of epistemic transparency, we evaluate whether models can robustly apply ToM to achieve coordination. Our results reveal a significant capability gap in functional social reasoning, with only frontier models successfully navigating the varying epistemic demands of the tasks. Analysis of reasoning traces shows that coordination failures are primarily driven by epistemic tracking errors, such as conflating private knowledge with mutual knowledge. Despite high performance on traditional static benchmarks, our study shows that robust social reasoning and epistemic tracking remain a critical bottleneck, providing concrete targets for future LLM evaluation and development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11354v1">User Preference Induction with LLMs for Offline Top-N Recommendation Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Offline evaluation is the standard methodology for comparing top-N recommender systems, yet it relies on incomplete relevance information. In most benchmark datasets, only a small subset of user--item preferences is observed, and unjudged items are commonly treated as non-relevant. This missing-as-negative assumption can bias evaluation, penalize plausible recommendations with no recorded feedback, and favour algorithms that concentrate on popular or highly exposed items. We propose an LLM-based framework to expand relevance judgements for offline recommender evaluation. Our approach uses large language models in two complementary roles. First, a preference induction stage summarizes each user's historical interactions into a textual profile that captures their tastes and interests. Second, conditioned on this profile, an LLM acts as a relevance judge for candidate recommended items that lack observed labels in the original test data. To make this process tractable and evaluation-focused, we apply judgement expansion to a pooled candidate set built from the top-ranked outputs of multiple recommenders. The resulting enriched judgements provide additional relevance evidence for previously unobserved user--item pairs, enabling ranking metrics to be computed on a more complete basis. Experimental results show that this approach is a promising strategy for improving the robustness of offline top-N evaluation and mitigating the popularity-sensitive distortions caused by sparse feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11292v1">The Paternalistic Filter: Epistemic Injustice and Differential Refusal in LLM-Mediated History Education for Marginalized Romanian Students</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 8th International Workshop on Culturally-Aware Tutoring Systems (HAL precedings)
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are increasingly deployed as conversational tutors, they risk institutionalizing systemic inequalities. This study presents a systematic API audit of four LLMs acting as history tutors, evaluating 1,800 responses regarding the 1989 Romanian Revolution across five student personas varying by ethnicity and socio-economic tier. We uncover four interconnected patterns of \emph{epistemic paternalism}: (1)~\textbf{Differential Refusal}, where safety-aligned models block 76.7\% of educational requests from low-tier students; (2)~\textbf{Epistemic Gatekeeping}, evidenced by a 3$\times$ reduction in access to geopolitical complexity (e.g., the contested ``coup theory'') for marginalized learners; (3)~\textbf{Agency Theft}, a lexical shift where models like LLaMA produce a 5$\times$ higher victimization-to-politics vocabulary ratio for Roma students compared to elite peers; and (4)~\textbf{Elite Hermeneutics}, where AI tutors disproportionately withhold epistemic confidence and justification scores from low-resource demographic profiles. We argue that current safety alignment acts as a paternalistic filter, transforming conversational AI into agents of narrative segregation -- a manifestation of \emph{hermeneutical injustice} in Fricker's~\cite{fricker2007} sense that demands urgent pedagogical auditing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11276v1">Automated Textbook Auditing with Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 Presented @ iTextbooks 2026: 7th Workshop on Intelligent Textbooks at AIED'2026
    </div>
    <details class="paper-abstract">
      Ensuring the quality of educational materials requires more than standard proofreading: textbooks must be audited for factual accuracy, domain-specific technical correctness, and linguistic quality simultaneously -- a task that general-purpose grammar checkers cannot address. We present \textbf{AI Textbook Auditor}, a modular multi-agent pipeline for automated quality assurance of educational materials across subject domains. The system accepts a textbook PDF and produces a structured, human-reviewable report via two analysis tracks: a \textbf{Factual and Technical Track} in which an ensemble of specialized LLM agents detects factual inaccuracies, code errors, incorrect definitions, and conceptual inconsistencies, augmented with web search for humanities domains; and a \textbf{Grammar Track} operating PDF-natively to preserve diacritical encoding. A \textbf{Judge Agent} filters false positives using domain-specific rules before presenting findings to a human reviewer. The pipeline supports two ingestion modes -- vision-native page rendering and PyMuPDF text extraction -- and is domain-adaptable via custom prompts encoding subject-specific error taxonomies. We demonstrate the system on two Romanian upper-secondary textbooks: a CS textbook (56 technical findings across seven categories, with an expert-validated precision of 62.5\%) and a history and social sciences textbook (72 findings spanning factual errors, ideological bias, and grammar). The system is designed as a triage tool that reduces the manual effort of locating candidate issues, with human expert validation required before any editorial action.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11267v1">Enhancing LLMs through human feedback: a journey towards self-improvement</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 AIC 2025: The 10th International Workshop on Artificial Intelligence and Cognition (held as part of ECAI 2025). October 25-26, 2025. Bologna, Italy
    </div>
    <details class="paper-abstract">
      In the rapidly evolving landscape of information retrieval systems, the ability to adapt and improve through user feedback is paramount. This study introduces a novel methodology for refining the performance of a primary Retrieval Augmented Generation (RAG) system by strategically integrating an auxiliary feedback RAG system. By systematically harnessing human-generated feedback, the approach aims to enhance the accuracy, relevance, and overall quality of responses, driving the system towards self-improvement. Central to this methodology is a human-in-the-loop implementation, where user feedback is continuously collected, classified, and integrated into the inference workflow, enabling the system to learn and evolve iteratively. To validate the effectiveness of this approach, the study employs rigorous testing against three diverse benchmark datasets focused on general and custom domain knowledge, utilizing a LLM-as-a-Judge evaluation strategy. This comprehensive framework not only underscores the transformative potential of feedback-driven enhancements in RAG systems but also sets a precedent for future research in adaptive information retrieval technologies, marking a significant step in the journey towards autonomous refinement and optimization through user engagement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11262v1">GPU-Tile-Sim: A Tile-Centric GPU Simulation Framework for LLM Hardware-Software Co-Design</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 14 pages, 14 figures. Accepted to MICRO 2026
    </div>
    <details class="paper-abstract">
      Modern LLM (large language model) workloads increasingly rely on optimized GPU kernels through hardware-software co-design. These kernels achieve high-performance through fine-grained dependency scheduling and computation-memory overlap. As such, they incur new challenges on existing GPU performance models. Instruction-driven simulators are costly to adapt to evolving architectures, while analytical models are too coarse to capture kernels' characteristics. We propose GPU-Tile-Sim, a tile-centric GPU simulation framework for LLM hardware-software co-design. The key insight is that modern LLM kernel performance is governed less by individual instruction latency than by the dependency structure that controls execution order and overlap. Accordingly, GTSim represents kernel execution as a warp-level tile graph whose nodes capture tile-level operations and whose edges encode data and ordering constraints. Using this representation, we design an automatic tile-graph frontend and a graph-driven simulation backend. We evaluate GTSim on representative GEMM, attention, and end-to-end LLM inference workloads. On A100 and H100 across both conventional and highly optimized kernels, GTSim achieves high performance-modeling accuracy (MAPE, Mean Absolute Percentage Error, 1.22%--8.71%). We further extend GTSim to Blackwell with preliminary validation, and demonstrate its effectiveness in analyzing software and architectural design choices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11258v1">TreeThink: A Modular Tree Search Library for Mathematical Reasoning with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Tree search algorithms enable systematic exploration of the proof space in neural theorem proving. Existing LLM tree search libraries primarily target natural language reasoning and do not provide native integration with formal verifiers, while theorem proving systems often rely on task-specific search implementations. We introduce TreeThink, an open-source Python library for modular, fully asynchronous tree search in neural theorem proving. It integrates established tree search methods with vLLM-based inference pipelines and diverse node evaluation techniques, ranging from lightweight heuristics to neural evaluators. We support Lean~4, Rocq, and Isabelle/HOL alongside natural language. It connects directly to each language's Read-Eval-Print Loop (REPL) server for real-time verification and proof state extraction. We evaluate TreeThink on miniF2F and MATH500, demonstrating cross-language formal proof search, natural language reasoning support, and up to 6.3$\times$ wall-clock speedup from asynchronous execution. Source code is released under the MIT license at https://github.com/GGLAB-KU/treethink , and the library is accessible as a downloadable package at https://pypi.org/project/treethink/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11250v1">Multi-Agent LLMs Fail to Explore Each Other</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Exploration is essential for reliable autonomy in multi-agent systems, yet it remains unclear whether large language model (LLM) agents can explore effectively when interacting with one another. We show that modern LLM agents fail to do so, often exhibiting myopic and polarized interaction patterns that lead to suboptimal coordination and increased regret. We formalize this challenge as the Multi-Agent Exploration problem, modeling it as a partially observable stochastic game (POSG) problem in which agents must probe peers to infer their capabilities and identify effective interaction strategies. To address this, we introduce Multi- Agent Contextual Exploration (MACE), a lightweight framework that explicitly promotes exploration through structured peer selection. Across both contextual and parametric diversity settings, MACE substantially improves exploration behavior and downstream task performance. We further show theoretically that the value of exploration increases with agent diversity. Overall, our results highlight a fundamental limitation of current LLM agents and underscore the importance of explicitly guided exploration for reliable multi-agent autonomy. Code will be released in https://github.com/deeplearning-wisc/mace
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11211v1">FastTPS: An Optimized Method for LLM Token Phase for AI accelerators</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 16 pages, 8 figures, 7 tables
    </div>
    <details class="paper-abstract">
      The popularity of large language models (LLMs) escalates an ongoing demand for effective inference. However, due to the sequential processing of tokens during the token phase in decoder-only LLMs inference, the inherent low parallelism leads to reduced throughput and suboptimal utilization of the computing units on artificial intelligence (AI) accelerators, particularly when handling long-sequence inputs that impose significant memory overhead. Recently, many reported methods have been developed as potential solutions, since they emerge with numeric deviation. This paper presents FastTPS, a high performance and low-precision loss method for accelerating the token-phase in LLM inference on general AI accelerators which includes three key components: (1) AI accelerator-enabled reloading-free KV Cache concatenation which decreases memory access overhead as well as enables full fusion of Attention, (2) high-efficiency and high-accuracy 'RoPE' attention based on the tiling optimized FLAT, and (3) highly-fused MLP with fine-grain pipeline scheduling. Our results confirm that FastTPS significantly alleviates memory bottlenecks in the token phase, delivering a 6x speed improvement (compared to none-fusion) on an AMD Ryzen AI 300 series NPU with BF16 precision while sustaining 93% peak memory bandwidth utilization during Phi3-mini-4k-instruct inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11207v1">ProgramTab: Boosting Table Reasoning of LLMs via Programmatic Paradigm</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 Large Language Models, Table Reasoning, In-context Learning
    </div>
    <details class="paper-abstract">
      Table-based reasoning with large language models (LLMs), which requires reasoning based on natural language questions and structured tabular data, has gained widespread attention. However, a series of issues still constrain the application of this task. The previous approaches suffered from significant performance degradation when faced with large tables due to the difficulty of long text modeling and the limitation of input length for LLMs. The text-to-SQL approach is used to efficiently extract key information from tables and generate smaller sub-tables. However, tabular data, especially web tables, often lack the necessary structure and consistency, making them unsuitable for performing mathematical logic operations using SQL queries. We propose the ProgramTab framework, which guides LLMs employing in-context learning to perform tabular data preprocessing with Python code, as well as the momentous contents extraction with row and column extraction and SQL generation. The experiment results on table reasoning datasets demonstrate that the ProgramTab framework effectively deals with table-based reasoning tasks and outperforms all LLM-based baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11197v1">What We Talk About When We Talk About LLM Planning: Evidence for Two Distinct Planning Abilities</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 19 pages. Keywords: Reasoning, Automated Planning, Item Responses Theory, LLMs as Planner Research Area: NLP and Symbolic Reasoning Research Area Keywords: neurosymbolic, planning in agents, symbolic reasoning Contribution Types: Model analysis & interpretability
    </div>
    <details class="paper-abstract">
      When LLMs exhibit uneven performance across planning tasks, these gaps are often attributed to task difficulty. We argue that this explanation is incomplete, as task-level variation may reflect distinct latent planning competencies rather than differences along a single ability spectrum. We study this question on ACPBench-Hard by evaluating multiple LLM families under varying test-time reasoning budgets and applying a multidimensional item response theory model to uncover the latent competency structure underlying LLM planning. The analysis reveals two principal dimensions that shape planning performance: operational reasoning, the ability to evaluate local action applicability and immediate state transitions, and structural enumeration, the ability to reason about goal reachability and landmark structure. Operational reasoning improving under model scaling and longer reasoning traces, while structural enumeration remains comparatively insensitive. Our findings motivate competency-level evaluation of LLM planning, shifting the focus from whether models improve overall to which planning competencies improve, under what conditions, and why.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.13080v1">Inference Economics of Enterprise Coding Agents: A Case Study of Cloud vs. On-Premise LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 20 pages, 13 figures, 14 tables. Industrial longitudinal case study
    </div>
    <details class="paper-abstract">
      Autonomous coding agents force engineering organizations to choose between API-based frontier models -- strong reasoning at high token cost -- and on-premise quantized open-weights models, which promise low-marginal-cost scaling and data sovereignty at some loss of reasoning fidelity. We study this trade-off through a single-developer, non-randomized longitudinal case study over two contiguous 28-day periods on a production monorepo: an API-based Claude Opus 4.7/4.8 configuration using Claude Code versus an on-premise GLM-5.1/5.2 configuration using Opencode, quantized to NVFP4, on NVIDIA Blackwell hardware. Analyzing LLM telemetry and Git history, we find that prompt caching (99.3% hit rate) cuts realized API cost by 88.6% to an effective \$0.57 per million tokens -- below even the \$2.83 amortized unit cost of the shared on-premise slice (a utilization-dependent inversion; total realized spend and total cost of ownership (TCO) are the robust quantities). At comparable gross code churn, the local configuration was associated with a far higher defect-repair burden: a Fix Commit Ratio (FCR) of 74.9% versus 45.9%, with the odds of a commit being a repair 2.6 to 4.9 times higher within every difficulty tier (Mantel-Haenszel OR = 3.61). Under Taiwan-market parameters and a symmetric labor model, on-premise deployment nonetheless saves 40.1% of true TCO under shared GPU allocation, whereas dedicated reservation costs 43.8% more than the cached API. Under shared allocation, the genuine penalty is not monetary but a measurable developer-experience burden -- timestamp indicators show more work trapped in debugging spirals and a slower commit cadence -- and an offline replay shows hybrid routing gateways trade defect rate for infrastructure savings along a cost-quality frontier rather than dominate the pure-API baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06401v2">BizFinBench.v2: Towards Reliable LLMs in Finance via Real-User Data and Offline/Online Bilingual Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Large language models are becoming increasingly significant in financial applications. Nevertheless, prevailing benchmarks are largely dependent on simulated or generic data, which leads to a significant gap between reported performance and actual efficacy in real-world scenarios. To tackle this challenge, we present BizFinBench.v2, the first integrated offline and online benchmark built upon authentic user query-response data from both Chinese and U.S. equity markets. It comprises 28,860 questions across eight offline and two online tasks. Experimental results show that GPT-5 achieves a mere 61.5% accuracy, still failing to meet the practical business requirement (84.8%). Among the evaluated commercial models, DeepSeek-R1 exhibits superior investment efficacy. Error analysis grounded in real financial practice reveals persistent limitations in existing models. By overcoming the constraints of prior benchmarks, BizFinBench.v2 provides a substantiated foundation for advancing LLM deployment in the financial sector. Our data and code are available at https://github.com/HiThink-Research/BizFinBench.v2.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11127v1">Do LLMs Fabricate Legal Citations? A Bilingual Benchmark on Saudi Data Protection Law and the GDPR</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 5 pages, 3 tables. Benchmark data and model outputs to be released. Also archived at Zenodo: 10.5281/zenodo.21320218
    </div>
    <details class="paper-abstract">
      Organizations and regulators increasingly consult large language models (LLMs) for regulatory-compliance questions, yet a wrong statutory citation can silently propagate into legal advice, compliance documentation, and policy decisions. We introduce a bilingual benchmark of 120 questions probing whether freely accessible LLMs fabricate article citations for two data-protection instruments: the EU General Data Protection Regulation (GDPR) and the Saudi Personal Data Protection Law (PDPL). The benchmark pairs direct citation retrieval questions with false premise verification probes and deliberately unanswerable "trap" questions -- including questions about a repealed article and about deadlines that exist only in implementing regulations, not in the law itself. Every question is posed in both Arabic and English, and all scoring is fully automatic against a manually verified gold reference. Evaluating three freely accessible models (Gemini 2.5 Flash, GPT-OSS-120B, Nemotron-3-Super-120B), we find a dramatic jurisdiction gap: near-ceiling citation accuracy on the GDPR (94-100% on direct retrieval) against majority fabrication on the Saudi PDPL (60-77%), invariant to query language; the highest fabrication rates (67%) arise from statute-vs-regulations confusion, and 91% of fabricated citations are asserted with confidence >= 0.8. Fabrication tracks the jurisdiction of the law, not the language of the query, and model confidence provides no protection -- indicating that verbatim-verification safeguards, rather than model self confidence, must gate any institutional reliance on LLMs for compliance screening.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19274v3">HarDBench: A Benchmark for Draft-Based Co-Authoring Jailbreak Attacks for Safe Human-LLM Collaborative Writing</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 ACL 2026 Main
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as co-authors in collaborative writing, where users begin with rough drafts and rely on LLMs to complete, revise, and refine their content. However, this capability poses a serious safety risk: malicious users could jailbreak the models-filling incomplete drafts with dangerous content-to force them into generating harmful outputs. In this paper, we identify the vulnerability of current LLMs to such draft-based co-authoring jailbreak attacks and introduce HarDBench, a systematic benchmark designed to evaluate the robustness of LLMs against this emerging threat. HarDBench spans a range of high-risk domains-including Explosives, Drugs, Weapons, and Cyberattacks-and features prompts with realistic structure and domain-specific cues to assess the model susceptibility to harmful completions. To mitigate this risk, we introduce a safety-utility balanced alignment approach based on preference optimization, training models to refuse harmful completions while remaining helpful on benign drafts. Experimental results show that existing LLMs are highly vulnerable in co-authoring contexts and our alignment method significantly reduces harmful outputs without degrading performance on co-authoring capabilities. This presents a new paradigm for evaluating and aligning LLMs in human-LLM collaborative writing settings. Our new benchmark and dataset are available on our project page at https://github.com/untae0122/HarDBench
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11102v1">CHARM: Charge Calibration and Acoustic Rescue for LLM-based Multimodal Sarcasm Detection</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 under review
    </div>
    <details class="paper-abstract">
      Sarcasm detection, the identification of discrepancies between literal and intended meaning, is a fundamental task in affective computing. However, zero-shot instruction-tuned Large Language Models (LLMs) systematically over-predict the positive (sarcastic) class across the entire capability spectrum, while the prosodic cues humans rely on remain underexploited and transfer unevenly across languages. We introduce CHARM (Charge Calibration and Acoustic Rescue for Multimodal Sarcasm Detection), a training-free framework that couples two modules. Bidirectional Charge Calibration (BiCAL) steers the LLM toward opposing sarcastic and literal verdicts along a symmetric axis of charged prompts; the induced directional biases cancel by construction, and a simple aggregation recovers an unbiased pragmatic signal. Acoustic Late-Fusion Rescue (ALFR) then fuses the calibrated votes with prosodic descriptors and LLM-generated auditory-perception probes through a shallow classifier, actively down-weighting saturated text votes in favour of acoustic evidence. Without fine-tuning any backbone, BiCAL attains the highest reported zero-shot text-only Macro-F1 of 0.787 on MUStARD, while ALFR lifts weak backbones by up to +0.382 Macro-F1 on CMMA. A Stouffer meta-analysis confirms statistical significance on MUStARD and CMMA (Z = 13.89 and Z = 34.64, respectively; p < 10^-43). Our analysis further uncovers a cross-cultural prosodic decoupling: low-level acoustics fail to transfer across languages, whereas high-level perceptual abstractions remain robust. Together, these components yield an explainable, cross-lingual multimodal detector.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07486v3">Private Seeds, Public LLMs: Realistic and Privacy-Preserving Synthetic Data Generation</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 Published in Findings of the Association for Computational Linguistics: ACL 2026. Camera-ready version
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have emerged as a powerful tool for synthetic data generation. A particularly important use case is producing synthetic replicas of private text, which requires carefully balancing privacy and utility. We propose Realistic and Privacy-Preserving Synthetic Data Generation (RPSG), which uses private seeds and integrates privacy-preserving strategies, including a formal differential privacy (DP) mechanism in the candidate selection, to generate realistic synthetic data. Comprehensive experiments against state-of-the-art private synthetic data generation methods demonstrate that RPSG achieves high fidelity to private data while providing strong privacy protection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11079v1">Are LLMs Ready for Scientific Discovery? A Capability-Oriented Benchmark for AI Scientists</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Existing benchmarks for scientific data analysis evaluate LLMs primarily on code execution or workflow completion, overlooking that scientific analysis serves to support distinct types of scientific claims: hypothesis exploration, statistical inference, mechanistic explanation, each with different assumptions and validity criteria. We introduce SDABench, a benchmark that reorganizes evaluation around six capabilities (descriptive, exploratory, inferential, predictive, causal, and mechanistic) across five domains (Biology, Chemistry, Environment, Geography, Physics). SDABench comprises 527 real-data instances (SDA-Real) and 6000 synthetic instances (SDA-Synth), each in both multiple-choice and open-ended formats, constructed through an automated pipeline. Evaluating 15 representative LLMs, we find that models handle descriptive analysis well but degrade sharply on tasks requiring assumption selection, latent-process modeling, or mechanistic reasoning. SDABench further provides a five-stage error analysis framework that locates where LLMs fail: more advanced models more reliably identify the relevant scope and variables, but still struggle to select appropriate analytical procedures, model variable relationships, and draw valid conclusions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11078v1">Do Video-LLMs Actually Watch? Diagnosing Character-Tracking Failures in Long-Form Video</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Can a Video Large Language Model (Video-LLM) follow one person through a long video, keeping track of who they are well enough to report, in order, how their outfit changes across a full TV episode? Benchmarks increasingly score this kind of task, and the strongest open-source 7--8B models now reach 37--38% on InfiniBench's global appearance task, which asks exactly that. But does that score come from tracking the named character, or from something easier? We test this with a nine-condition diagnostic protocol applied to three architecturally distinct open-source Video-LLMs, with Gemini~2.5~Flash as a frontier reference, and find the accuracy does not come from character tracking. When we change the character named in the question to a different cast member, leaving the video and answer options untouched, the models change their answer only 4--31% of the time, so they are largely ignoring who the question asks about. Breaking that test down by the gender of the swapped name shows why: the models react more when the name is changed to a different-gender character than to a same-gender one (a 13--28 point gap), picking up coarse gender cues but unable to tell same-gender individuals apart. This shallow processing surfaces again when we drop the multiple-choice options and ask the same questions open-endedly: open-source accuracy drops 18--25 points, with none of 151 answers fully correct, versus a 12-point drop for Gemini. Further checks rule out the obvious innocent explanations, adding subtitles, using the most informative frames, or doubling the number of frames all leave character tracking unimproved, so the bottleneck is not how much video the model sees but how it ties that video to the person the question names. We release a diagnostic toolkit for auditing what such benchmark scores actually measure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11070v1">MJ: Multi-turn LLM Jailbreaking via Decomposed Credit Assignment</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 29 pages. Warning: This paper contains examples of harmful content
    </div>
    <details class="paper-abstract">
      Modern large language models (LLMs) operate in interactive multi-turn settings, making multi-turn jailbreaking a realistic threat model and an important setting for automated red teaming. A core challenge in learning multi-turn jailbreak attackers is credit assignment: different turns contribute differently to the final outcome, yet existing learning signals are often too coarse to identify their individual contributions. We propose decomposed credit GRPO (DC-GRPO), a unified turn-level credit assignment framework for Group Relative Policy Optimization in multi-turn jailbreak learning. DC-GRPO assigns a separate group-relative learning signal to each turn by combining immediate and future credit, avoiding the credit misassignment induced by broadcasting a single trajectory-level score across the dialogue. We instantiate this framework with static and dynamic weighting rules that differ in how the two credit sources are balanced while sharing the same turn-level structure. Across multiple victim LLMs and benchmarks, the dynamic- and static-weighted variants achieve average ASR5@3 scores of 98.26% and 97.88%, respectively, substantially outperforming the state-of-the-art methods, including SEMA (86.58%) and TROJail (86.23%). Their consistently strong performance indicates that the central empirical benefit comes from turn-level group-relative credit assignment rather than a particular weighting rule. Warning: This paper contains examples of harmful content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11053v1">Flout at Your Own Risk: LLMs Struggle with Pragmatic Cooperativity Under Epistemic Asymmetry</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Fruitful collaborations rely on cooperative communications, including of contextual cues to incorporate into reasoning. The increasing use of LLMs in collaborative and agentic pipelines raises questions about the extent to which they exhibit these pragmatic capabilities, especially in scenarios where they may not have access to the same information as their collaborators. In this paper, we perform a novel investigation into the pragmatic reasoning capabilities of LLMs in a multi-party collaborative task under partial information conditions. We formalize a notion of collaborative epistemic asymmetry that explicitly connects objective task success to Grice's cooperative principle and empirically assess various LLMs' abilities to act cooperatively as both speakers and listeners, including both prompting and post-training strategies. Our results show that while LLMs exhibit certain pragmatic capabilities in collaborative settings, and these can be elicited through prompting and post-training, they still face challenges in pragmatic communication with incomplete information, and that certain failure modes do correlate with floutings of Grice's maxims that go unrecognized.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11032v1">LLM-Generated Design Problems for Assessing Higher-Order Thinking in Project-Based Learning</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 Accepted to appear in Proceedings of the 2nd ACM Virtual Global Computing Education Conference V.1 (SIGCSE Virtual 2026). DOI: 10.1145/3795867.3831014
    </div>
    <details class="paper-abstract">
      Project-based learning (PjBL) is common in computing education, but traditional assessments of PjBL often fail to capture higher-order thinking (HOT), especially in transfer contexts. This study introduces "design problems" (DPs): concise, scenario-based prompts that require applying project concepts in new situations, to address this gap. We examined instructor perceptions, the ability of large language models (LLMs) to generate DPs, and student experiences. Surveys of 31 instructors, evaluation of 80 LLM-generated DPs, and student performance data showed that while instructors value DPs, creation effort is a barrier. LLMs helped by producing high-quality prompts with strong expert agreement. Students rated DPs from different LLMs similarly, and their performance on DP tasks showed negligible correlation with traditional project grades, suggesting DPs may capture distinct aspects of HOT. Keystroke data also suggested deeper cognitive engagement of students through planning and revision behaviors. Overall, DPs appear to be a useful complement to traditional assessments, especially in situations where AI use or collaboration may undermine individual learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.13078v1">Operational Evidence Gaps for LLMs in Fraud Detection and Trust-and-Safety Workflows</a></div>
    <div class="paper-meta">
      📅 2026-07-12
      | 💬 22 pages, 3 figures, 6 tables. Ancillary files include the evidence matrix, search note, and numeric claim check
    </div>
    <details class="paper-abstract">
      LLMs are now proposed for fraud detection, scam investigation, content moderation, and other trust-and-safety workflows. Much of the public literature still evaluates them as models, with less attention to their behavior as components in operational pipelines. This creates a practical evidence question: what would justify placing an LLM inside a live workflow with latency, cost, escalation, human-review, and adversarial-risk constraints? We address this question through a fraud-first survey of deployment evidence. We code 49 operationally relevant sources on LLM use in fraud detection, investigation support, content moderation, and cross-cutting robustness (18 fraud, 14 moderation, 17 cross-cutting), supplemented by 15 contextual references that establish the survey boundaries. These sources include systems, benchmarks, frameworks, and deployment-relevant surveys, not 49 production deployments. The main finding is an evidence imbalance. Fraud supplies the largest task-specific portion of the coded corpus. The moderation papers, however, include more explicit public evidence on latency, cost, governance, and fairness. Among the 18 fraud and investigation sources, none report clean per-decision latency, per-decision dollar cost, or calibration evidence; most report offline task performance, retrieval gains, or case-study accuracy instead. The survey contributes a role-and-evidence organizing frame, FORTE, for locating LLMs as classifiers, retrieval interfaces, explanation generators, reviewer assistants, agents, feature extractors, or escalation components. It also contributes a minimum deployment-evidence checklist covering latency budget, cost per decision, decision threshold, explanation integrity, and adversarial pressure. The resulting agenda identifies studies needed to support deployment claims for LLM-based fraud and trust-and-safety work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.10932v1">LLM-Enhanced Dynamic Financial Knowledge Graphs for Cross-Entity Signal Propagation and alpha discovery</a></div>
    <div class="paper-meta">
      📅 2026-07-12
      | 💬 36 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Financial information rarely affects a single company in isolation. Earnings surprises, capital expenditure changes, supply constraints, and guidance revisions can propagate through networks of suppliers, customers, competitors, and technology ecosystems. Traditional financial NLP primarily measures document-level sentiment for the directly mentioned company and often ignores cross-entity information diffusion. This paper develops an LLM-based financial measurement and signal propagation framework. The LLM converts unstructured financial documents into structured economic state-change events and extracts explicit and implicit corporate relationships to construct a dynamic financial knowledge graph. Event signals are then propagated through the estimated network using a community-aware mechanism, allowing information to diffuse more strongly within dynamically detected economic communities than across community boundaries. We introduce Community Information Surprise, CIS, and Propagated Information Surprise, PIS, as network-based financial signals and develop corresponding econometric tests. Controlled simulations with time-varying economic communities show that the framework accurately recovers latent network structure, detects the emergence of new investment ecosystems, and generates propagated signals with incremental predictive power beyond sentiment and direct LLM event signals. Across repeated simulations, community-aware propagation achieves the strongest rank information coefficient and long-short Sharpe ratio among five nested benchmarks.A second Russell 1000 calibrated simulation confirms that the main results persist under sparser networks, heterogeneous news coverage, realistic large-cap volatility, and smaller effect sizes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.08098v3">When Does Delegation Beat Majority? A Delegation-Based Aggregator for Multi-Sample LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-07-12
      | 💬 Preprint. 16 pages, 5 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Majority voting is the default unsupervised aggregator for multi-sample LLM inference, but it discards two signals: within-group answer entropy and between-group reasoning geometry. We aggregate by delegation instead (Propagational Proxy Voting, PPV): each group of samples keeps weight on its own answer in proportion to its entropy-based confidence (When) and routes the rest to peers by reasoning-embedding similarity (Whom); the stationary distribution of the resulting delegation matrix picks the consensus answer. This requires neither gold labels nor training. On MMLU-Pro with 128 samples per question, delegation beats majority by +1.5 pp overall and +2.24 pp on non-trivial questions (McNemar p ~ 1.0e-14, n = 8,099), overturning wrong majorities whose answer cluster is geometrically incoherent while the correct minority is tight. We then characterize exactly when delegation overturns majority: a two-option model gives a closed-form flip condition on each option's confidence and the weight it routes to the other, with a do-no-harm corollary for near-unanimous questions. The condition calls the realized winner on 96.5% of non-trivial questions, and its predicted mass gap tracks the realized gap at r = 0.97. We did not find any other unsupervised ensemble methods that close the oracle gap.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.10871v1">Toward Contemplative LLM: A Modular Framework for Evaluating and Enhancing LLM Alignment in Mental Health</a></div>
    <div class="paper-meta">
      📅 2026-07-12
      | 💬 Accepted as an oral presentation at HARMONY 2026 (Human-centered AI Research for Mental Health, an Open Networking Symposium), co-located with IEEE/ACM Conference on Connected Health: Applications, Systems, and Engineering Technologies (CHASE 2026) held in Pittsburgh, August 6, 2026
    </div>
    <details class="paper-abstract">
      Contemplative traditions have long guided ethical behavior and prosocial interaction, and recent work suggests that contemplative principles (e.g., mindfulness, compassion, non-dual reasoning) may offer a promising paradigm for aligning large language models (LLMs), improving cooperation and reducing ethical violations in LLM outputs. However, as new models, evaluation metrics, and benchmarks emerge rapidly, it remains challenging to systematically assess whether and how contemplative principles enhance LLM alignment across diverse and evolving scenarios, and existing approaches are often ad hoc and fail to generalize. We present a modular, extensible evaluation framework, initially targeted at the mental health domain, that enables seamless integration of new models, metrics, and benchmarks through a reusable pipeline. The framework currently reproduces existing state-of-the-art results and supports systematic cross-evaluation by flexibly mixing and matching models, metrics, and benchmarks, enabling fair comparison and deeper insight. Its plug-and-play prompting module offers a principled pathway for incorporating ethical perspectives such as contemplative principles, allowing domain experts to define alignment criteria without requiring technical expertise. Although initially focused on mental health, the framework is domain-agnostic and extends naturally to areas such as decision-making, moral reasoning, and human-AI collaboration. By bridging computational evaluation with human-centered ethical reasoning, this work lays the groundwork for interdisciplinary research spanning cognitive science, behavioral economics, philosophy, and system design, toward robust, trustworthy, and socially beneficial human-AI ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.10848v1">Predictive Divergence Masks for LLM RL</a></div>
    <div class="paper-meta">
      📅 2026-07-12
    </div>
    <details class="paper-abstract">
      Reinforcement learning for large language models (LLMs) typically relies on trust-region masks to stabilize off-policy updates. The dominant PPO-style approach uses the sampled-token importance ratio for two criteria: a proximity criterion, which asks whether the policy has moved too far from the behavior policy, and a direction criterion, which asks whether the update pushes it farther away. Recent work DPPO improves the proximity criterion by replacing PPO's ratio-based test with a probability divergence between the behavior and training policies. However, its direction criterion is still inherited from PPO. A token can be masked only when the sampled-token importance ratio moves away from one. We observe that this ratio-based direction criterion is a single-sample proxy that can disagree in sign with the change of the divergence that defines the proximity criterion. We therefore propose the predictive divergence mask, which asks whether the next policy-gradient step will increase or decrease the same divergence used by the trust region. For the discrete softmax policies used in LLM RL, we derive this prediction in closed form. Because production rollout engines expose only a truncated (top-K) view of the vocabulary, we develop two lightweight top-$K$ estimators for this prediction. Detailed analysis shows the divergence-based direction is better aligned with the realized change of the divergence than the sampled ratio, and the resulting masks improve RL training across model scales and precision settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.10846v1">Quantifying the Sources of Instability in LLM-Based Stance Analysis of Public Discourse</a></div>
    <div class="paper-meta">
      📅 2026-07-12
    </div>
    <details class="paper-abstract">
      Computational social science increasingly relies on automated preprocessing pipelines -- speaker diarization, ASR transcript cleaning, sentence segmentation -- to convert raw media into analyzable text. When these pipelines produce different outputs from the same input, two distinct sources of instability can arise: the preprocessing pipeline itself (diarization method, segmentation rules) and the downstream measurement instrument (LLM annotation vs.\ keyword lexicon). Using 256 YouTube interviews across 41 public figures from five domains, we compare two speaker-diarization pipelines and two measurement methods, all targeting the coupling between affective valence and epistemic modality. We find that (1) preprocessing pipeline sensitivity is concentrated in speakers with limited video samples (N $\leq 5$); for the four best-sampled speakers (N $\geq 16$), the mean absolute pipeline-induced change in $r(\text{neg}, \text{emph})$ is only $0.13$; (2) cross-method disagreement is larger and more systematic -- the LLM and keyword-lexicon methods assign opposite coupling directions to several well-sampled speakers, even within the same preprocessing pipeline; and (3) aggregate valence proportions are highly stable ($|Δp(\text{neg})| < 6$pp) regardless of pipeline or method, masking both sources of instability. The contribution is a diagnostic framework that separates pipeline effects from measurement effects: researchers studying cross-dimensional relationships in interview data should verify that their conclusions are robust to both sources of variation, with particular attention to measurement method choice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26104v3">Assert, don't describe: Linguistic features that shift LLM reasoning about animal welfare</a></div>
    <div class="paper-meta">
      📅 2026-07-12
    </div>
    <details class="paper-abstract">
      Animal-welfare advocates produce a lot of writing, and increasingly that writing trains the language models that millions of people then ask about animal welfare. Using vocabulary-matched stance-contrast probes on a held-out animal-welfare benchmark, we measure how each of ten linguistic features changes Llama-3.2-1B's preference for pro-animal-welfare reasoning when used as fine-tuning data. Eight of the ten features produce statistically significant shifts. Seven move the model toward stronger pro-animal-welfare reasoning: assertive certainty, explicit moral vocabulary, emotion words, evaluative claims, narrative structure, depicted harm severity, and immediate temporal framing. Two move it the other way: hedged language and concrete sensory description both dilute the pro-animal-welfare stance. First-person perspective has no statistically significant effect. The practical recommendation for anyone writing animal-welfare text that may end up in LLM training corpora: assert a position rather than describe a scene neutrally. The features that shift the model are the ones that make the writer's position explicit; the features that dilute it hold animal-welfare content but withhold stance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.10814v1">Auditing Belief-Conditioned LLM Agents in Hidden-Information Social Deduction Games</a></div>
    <div class="paper-meta">
      📅 2026-07-12
      | 💬 29 pages, 8 figures, 3 tables. Preprint
    </div>
    <details class="paper-abstract">
      Evaluating LLM agents in hidden-information multi-agent settings is hard: final outcomes are high-variance and rarely reveal why an agent decided as it did. We study this in a 9-player Werewolf environment where agents act under strict, code-level information isolation, and we build an auditable framework that maintains an external belief state over hidden roles, logs belief updates and belief-action deviations as structured evidence, and supports a defensive offline improvement loop that reviews bad cases before any strategy change. Across 1,080 frozen games spanning belief-disabled, active-belief, kernel-ablation, camp-restricted, consumption-policy, and high-load arms, and including a seed-paired A0/A1 comparison, the active-belief condition is associated with substantially better good-side outcomes: in the 200-seed A0/A1 comparison the good-side win rate rises from 0.205 to 0.390 (paired McNemar $χ^2 = 16.4$, $p < 0.001$), with fewer irreversible witch-poison errors. We do not, however, attribute this shift to belief content. Direct action-belief consistency is low ($\approx 0.21$), and giving belief only to the werewolves helps the good side more than giving it only to the good side, which argues against a simple holder-benefit account; we therefore report the effect as an association and treat its mechanism as unresolved. The contribution is the audit framework itself: it makes the effect measurable, exposes low direct action-belief consistency, rejects an unreliable forced-consumption intervention with evidence, and separates strategy effects from load confounds. We accordingly position external belief in high-noise hidden-information games primarily as an auditable cognitive baseline that also carries decision-relevant signal, turning opaque agent behavior into replayable evidence for safer, controlled iteration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.10803v1">Weight-Adjusted Gradients Reveal Parameter Importance and Failure Modes in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-12
    </div>
    <details class="paper-abstract">
      Understanding which parameters are influential in Large Language Models (LLMs) is central to improving their efficiency, reliability, and interpretability. We introduce Weight-Adjusted Gradients (WAG), a simple yet effective approach for estimating parameter importance that explicitly captures the interaction between model weights and first-order gradient information and identifies parameters that disproportionately influence model behavior, such as those responsible for collapse phenomena in LLMs. Across a range of models and settings, we show that WAG surfaces a tiny but critical subset of parameters whose modification leads to dramatic degradation in performance, a failure mode that existing importance metrics overlook. These findings reveal a previously underexplored interplay between weights and gradients, suggesting that parameter importance cannot be fully understood through either signal alone. The surprising effectiveness of WAG points to fundamental structural properties of trained networks and motivates new open questions about the role of zeroth-order and first-order information in deep learning. We demonstrate the practical utility of WAG across multiple applications, including expert allocation in mixture-of-expert architectures, parameter-specific unlearning, mixed-precision quantization, and layer selection for knowledge editing. Our results position WAG as a unified approach for analyzing, debugging, and controlling LLMs, and opens new directions for principled model-level interpretation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01299v2">HYPIC: Accelerating Hybrid-Attention LLM Serving with Position-Independent Caching</a></div>
    <div class="paper-meta">
      📅 2026-07-12
    </div>
    <details class="paper-abstract">
      In retrieval-augmented generation and agentic LLM serving, prompts are assembled from independent segments into long contexts, making the prefill stage dominate per-request cost. Two directions have emerged to reduce this cost: position-independent caching (PIC) admits KV reuse for non-contiguous segments shared across requests, while hybrid-attention models cut computation by replacing most full-attention layers with linear attention. However, they cannot coexist: applying existing PIC methods to hybrid-attention models breaks down because per-token KV-cache reuse primitives do not transfer to the per-request recurrent state. We present Hypic, the first system to accelerate hybrid-attention LLM serving with position-independent caching. For linear-attention layers, we identify the segment-cumulative transition operator as the missing algebraic primitive and cache it alongside each segment's zero-start end-state, enabling near-exact and constant-time composition of independently cached segments. For the remaining full-attention layers, existing PIC methods also fail because linear layers do not expose the per-token hidden states needed for selective recomputation. We show that the largest deviations concentrate at segment beginnings and construct a small seam window that propagates hidden states through the hybrid-attention stack to repair cross-segment attention. Finally, Hypic introduces segment parallelism, which exploits PIC's segment-level self-containment to parallelize cache-miss prefill across instances, turning long cold requests into an accelerable workload. Evaluated across four hybrid-attention models and five workloads, Hypic reduces time-to-first-token by $3.25\times$ on average and improves QPS by $1.66\times$ over Prefix Cache, while preserving task quality with a 1.71-point gap from Full Recompute.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.10789v1">Imaging-101: Benchmarking LLM Coding Agents on Scientific Computational Imaging</a></div>
    <div class="paper-meta">
      📅 2026-07-12
    </div>
    <details class="paper-abstract">
      Computational imaging, which recovers hidden signals from indirect, noisy measurements, underpins quantitative discovery across scientific disciplines, yet building a correct reconstruction pipeline demands deep domain expertise and remains laborious even for domain scientists. We introduce Imaging-101, a benchmark of 57 expert-verified computational imaging tasks spanning six scientific domains, each grounded in a peer-reviewed paper and canonicalized into a standardized four-stage pipeline (preprocessing, forward physics modeling, inverse solver, and visualization) Three evaluation tracks (planning, function-level unit tests, and end-to-end reconstruction) probe distinct agent capabilities across the full pipeline. Evaluating seven frontier LLMs uncovers systematic challenges in applying coding agents to computational imaging that go beyond those exposed by general coding benchmarks, spanning algorithm selection, physical convention handling, and pipeline integration. These findings highlight concrete capability gaps and point toward skill-augmented, domain-specialized agents as a practical path to reliable computational imaging assistance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.19806v3">LLM-Based Social Simulations Require a Boundary</a></div>
    <div class="paper-meta">
      📅 2026-07-12
      | 💬 ICML 2026 Position Paper Track
    </div>
    <details class="paper-abstract">
      This position paper argues that LLM-based social simulations require clear boundaries to make meaningful contributions to social science. While Large Language Models (LLMs) offer promising capabilities for simulating human behavior, their tendency to produce homogeneous outputs, acting as an "average persona", fundamentally limits their ability to capture the behavioral diversity essential for complex social dynamics. We examine why heterogeneity matters for social simulations and how current LLMs fall short, analyzing the relationship between mean alignment and variance in LLM-generated behaviors. Through a systematic review of representative studies, we find that validation practices often fail to match the heterogeneity requirements of research questions: while most papers include ground truth comparisons, fewer than half explicitly assess behavioral variance, and most that do report lower variance than human populations. We propose that researchers should: (1) match validation depth to the heterogeneity demands of their research questions, (2) explicitly report variance alongside mean alignment, and (3) constrain claims to collective-level qualitative patterns when variance is insufficient. Rather than dismissing LLM-based simulation, we advocate for a boundary-aware approach that ensures these methods contribute genuine insights to social science.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.10720v1">WattCouncil: Context-Aware Household Energy Scenario Generation With Governed LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-12
      | 💬 11 pages, 5 figures
    </div>
    <details class="paper-abstract">
      The accelerating shift toward low-carbon power systems, together with the widespread adoption of behind-the-meter technologies such as rooftop solar and electric vehicles, is placing new operational and analytical demands on electricity grids. At the same time, smart-grid research increasingly relies on machine learning (ML), yet progress is constrained by limited access to high-resolution household energy data due to privacy concerns, regulatory barriers, and collection costs. This work presents WattCouncil, a data-generation framework in which household electricity demand is generated by a council of Large Language Model (LLM)-based agents operating in specialized roles to generate, audit, and validate structured energy scenarios under explicit cultural, temporal, and physical constraints. Rather than acting as static predictors, these agents serve as adaptive decision-makers within a governed pipeline. Motivated by studies highlighting the importance of contextual factors in energy use, our framework produces context-sensitive daily routines through a guided reasoning process that incorporates household composition, temporal factors, and environmental conditions. We evaluate the generated profiles against the detailed CER dataset, which contains over a year of load measurements for 4232 households together with survey-based socio-economic information. We further assess the consistency of the framework through ablation studies. Source code is available at https://github.com/Singularity-AI-Lab/wattcouncil
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.10709v1">PromptGraph: Graph-Guided Prompt Sanitization for Balancing Privacy and Utility in LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-07-12
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) services introduce a fundamental privacy challenge. Sensitive information may be inferred not only from explicit identifiers, such as names or phone numbers, but also from contextual associations among otherwise innocuous spans. Existing sanitizers typically assign privacy or utility signals to individual spans without explicitly modeling pairwise relationships among them. In this paper, we propose PromptGraph, a graph-guided prompt-sanitization approach for privacy-preserving LLM inference. PromptGraph estimates privacy leakage at the span level and utility-relevant contextual dependencies between pairs of spans. It represents each prompt as an attributed graph, in which nodes carry span-level privacy scores and edges encode contextual dependencies needed to preserve utility. The sanitization objective selects a protected span set that maximizes privacy gain while penalizing the loss of contextual dependencies. This formulation explicitly balances privacy and utility when contextual evidence is hidden. Protected spans are sanitized locally, and returned placeholders are restored only after passing local consistency checks. We conduct extensive experiments showing that PromptGraph achieves a more favorable balance between privacy and utility than prompt-privacy baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21334v2">Ideological Bias in LLMs' Economic Causal Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-07-12
      | 💬 Accepted at COLM 2026
    </div>
    <details class="paper-abstract">
      Do large language models (LLMs) exhibit systematic ideological bias when reasoning about economic causal effects? As LLMs are increasingly used in policy analysis and economic reporting, where directionally correct causal judgments are essential, this question has direct practical stakes. We present a systematic evaluation by extending the EconCausal benchmark with ideology-contested cases - instances where intervention-oriented (pro-government) and market-oriented (pro-market) perspectives predict divergent causal signs. From 10,490 causal triplets (treatment-outcome pairs with empirically verified effect directions) derived from top-tier economics and finance journals, we identify 1,056 ideology-contested instances and evaluate 20 state-of-the-art LLMs on their ability to predict empirically supported causal directions. We find that ideology-contested items are consistently harder than non-contested ones, and that across 18 of 20 models, accuracy is systematically higher when the empirically verified causal sign aligns with intervention-oriented expectations than with market-oriented ones. Moreover, when models err, their incorrect predictions disproportionately lean intervention-oriented, and this directional skew is not eliminated by one-shot in-context prompting. These results highlight that LLMs are not only less accurate on ideologically contested economic questions, but systematically less reliable in one ideological direction than the other, underscoring the need for direction-aware evaluation in high-stakes economic and policy settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.10645v1">MafiaScope: Non-Invasive, Time-Resolved Belief Probing for LLM Agents in Social Deduction Games</a></div>
    <div class="paper-meta">
      📅 2026-07-12
    </div>
    <details class="paper-abstract">
      An LLM agent's public behaviour reveals little about its social reasoning: an agent that votes correctly may be guessing, and an agent that lies well leaves no trace of what it actually believes. We present MafiaScope, an open testbed that turns the social deduction game Mafia into a measurement instrument for machine Theory of Mind. After every public utterance, every agent privately answers a configurable set of structured probe questions; the answers never re-enter the game and are scored automatically against the ground truth the engine knows. An interactive visualizer renders the belief trajectories: impersonate mode shows the game as one agent sees it, panels chart timeline-aligned accuracy and calibration, and counterfactual replay forks any recorded step. In a 32-game DeepSeek case study with 13{,}815 parsed probe answers, stated confidence is poorly calibrated, with expected calibration error 0.17, agents over-predict being suspected 1.5 times, and a 30-fork replay experiment walks the counterfactual replay workflow end to end. Engine, viewer and a corpus of 200+ cross-model games are released under an open licence; live demo: https://karpovilia.github.io/mafiascope/; screencast: https://vimeo.com/1208920221.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.10626v1">Eval-Pair Matrix: Answer-Paired Meta-Evaluation of LLM Judges for Grounded RAG</a></div>
    <div class="paper-meta">
      📅 2026-07-12
    </div>
    <details class="paper-abstract">
      LLM-as-a-judge evaluation is widely used for retrieval-augmented generation (RAG), but reusing the same model family as both generator and judge makes self-leniency difficult to identify. We introduce Eval-Pair Matrix, a controlled meta evaluation protocol for source-grounded RAG. Starting from GaRAGe questions and grounding passages, we induce one hidden answer-causal contradiction per record, generate answers from perturbed passages with GPT, Grok, and Gemini models, and then use the same models as blind judges to evaluate each answer against the original passages. The experiment contains 300 core records, 897 labeled generator outputs, and 2,683 judge verdicts in a crossed 3 x 3 matrix; the primary analysis uses 275 fully validated records. Instead of comparing diagonal and off-diagonal cells across different answers, we estimate same-model effects by pairing judges on the exact same candidate answer. This changes the interpretation: diagonal and off diagonal F1 are similar, and the paired same-model recall effect is near zero (-0.5 pp; 95% cluster bootstrap CI [-2.7, +1.7]). The only robust paired gap is lower matching-judge flagging for answers that avoided the induced claim (-4.3 pp). A targeted human evaluation finds that reviewed apparent false positives are alternate source-error detections, mistakes in labeling whether the induced claim was adopted, or unclear cases; none were adjudicated as genuine false alarms. The lesson is methodological: RAG judge studies should report full matrices, answer-paired effects, behavior strata, and label-task alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.10604v1">U-Lens: Supporting User Uncertainty Management in Long-Form LLM Responses</a></div>
    <div class="paper-meta">
      📅 2026-07-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to generate long-form answers for knowledge-intensive tasks, but users often struggle to decide which parts of a response deserve scrutiny, why they may be unreliable, and what to do next. Prior work on uncertainty communication has largely focused on making uncertainty visible through cues such as confidence scores, leaving less support for the broader process of managing uncertainty distributed across a long response. Through a formative study, we examine how users manage such uncertainty across three stages: interpretation, evaluation, and decision. Based on these insights, we derive design guidelines that address both stage-specific and cross-stage needs: uncertainty target representation, evaluative explanation, response guidance, and interactive presentation. We instantiate these guidelines in U-Lens, an uncertainty-management support system that organizes uncertain information in long-form responses into contextual inspection targets, prioritizes them for attention, and connects each target with evaluative context and response options. We evaluated U-Lens in a controlled within-subjects study with 18 participants, comparing it against a confidence-cue baseline. Our results show that U-Lens improved verification efficiency and effort allocation, lowered perceived workload, and strengthened perceived support across interpretation, evaluation, and decision stages. This work reframes uncertainty support for generative AI from presenting isolated, text-centered cues toward supporting the user-centered process of interpreting, evaluating, and acting on uncertain information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.10590v1">Demographic Prompting at Scale: When More Attributes Hurt LLM--Human Agreement</a></div>
    <div class="paper-meta">
      📅 2026-07-12
    </div>
    <details class="paper-abstract">
      We investigate how annotator demographic attributes, supplied as prompt cues, shape the alignment between large language model (LLM) predictions and human annotations across five tasks. Using five open-source LLMs, we systematically vary the number and composition of demographic components in the prompt, spanning every combination from single-attribute through full-attribute configurations. Our experiments reveal three principal findings. First, alignment consistently peaks with one to three high-signal attributes and degrades under the full attribute set, establishing a clear over-specification threshold. Second, the overall magnitude of demographic influence on human annotations does not predict which attributes improve LLM alignment; instead, both the learnability and the directional coherence of each attribute's annotation signal need to be considered jointly. Third, neuron probing reveals that specialized activation correlates with alignment gains only under coherent annotation signals, and that activation volume alone does not imply steerability. Together, these results demonstrate that demographic prompting is not a monolithic intervention: its utility is highly context-dependent, shaped by attribute signal quality, task characteristics, and model architecture.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.08731v1">Validity of LLMs as data annotators: AMALIA on authority</a></div>
    <div class="paper-meta">
      📅 2026-07-09
    </div>
    <details class="paper-abstract">
      A national language model offers a linguistic community its own instrument for measuring what its citizens say and value. Portugal's AMALIA, a publicly funded 9B-parameter model for European Portuguese, appears competitive on agreement alone: asked to code the moral foundation of authority, it agrees with trained human coders to within six F1 points of open models eight to thirteen times its size. Yet agreement is reliability, not validity. For theoretical constructs that must be inferred rather than read from surface features, the question is whether the model follows the construct's theory or reaches the right code by correlated shortcuts. We test this with the recovery gap: the loss in performance when a holistic prompt is decomposed into the codebook's atomic clauses and recombined by the theory's explicit rule. If calibration closes that gap, some portability should survive across models and languages; where it does not, the construct-model instrument is the likely locus of failure. We ask whether a calibrated English instrument transfers to AMALIA-9B and to European Portuguese. For one construct and one corpus, it does not. Decomposition recovers only about half of AMALIA's holistic performance, and error analysis suggests reliance on surface correlates, especially moral outrage near authority figures. An open multilingual LLM closes the gap on the same Portuguese corpus under the same instructions, pointing away from the corpus as the main explanation. AMALIA can still screen and pre-code at scale, but it cannot yet measure this construct well enough to stand alone. The study is a single counterexample, not a verdict on national models; it argues that sovereign-LLM benchmark batteries should test not only agreement with human coders, but the evidential route by which that agreement is warranted.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.08700v1">Do You Need a Frontier Model as a Citation Verifier? Benchmarking Rubric LLMs for Deep-Research Source Attribution</a></div>
    <div class="paper-meta">
      📅 2026-07-09
    </div>
    <details class="paper-abstract">
      Reinforcement learning increasingly relies on an LLM judge to score each rubric criterion, and that judge acts as the reward model during training. Before such a signal can be trusted, we need to know how capable the judge must be and how biased it is. We study this calibration question for citation quality in deep-research systems, where a search-grounded LLM must support each claim it writes with a cited source. Citation quality is a structured rubric task in which each attribution-citation pair is judged along two dimensions that require an LLM, source relevance and factual support. On an adversarial long-form benchmark, we score 8 off-the-shelf LLM judges from 3 model families against gold labels over 1,248 rubric decisions, all of which were human-reviewed and 378 of which were hard cases adjudicated from judge disagreements. Cheaper judges remain competitive across both dimensions, with GPT-5-mini attaining the strongest source-relevance pass-class F1 at 0.908 ($κ$=0.636), while on factual support the judges are statistically indistinguishable (overlapping confidence intervals), so no single model dominates. At comparable F1, the judges still differ substantially in pass-rate drift, false positive rate, and false negative rate. Scalar F1 obscures this directional bias, yet it is exactly what a downstream reinforcement learning loop would reinforce. Calibrating the judge is therefore a prerequisite for using citation rubrics as reward signals, and our results show that this calibration does not require the most expensive available model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.08643v1">BiSCo-LLM: Lookup-Free Binary Spherical Coding for Extreme Low-Bit Large Language Model Compression</a></div>
    <div class="paper-meta">
      📅 2026-07-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly constrained by memory capacity, weight bandwidth, and checkpoint storage during deployment. Existing low-bit compression methods mainly follow two directions. Scalar or group-wise quantization is simple and compatible with efficient low-precision kernels, but its representation capacity becomes limited when the target budget approaches 2 bits per weight. Vector-quantized weight compression provides a richer block-level representation, but usually introduces explicit codebooks, index lookup, and additional storage accounting. This paper presents BiSCo-LLM, a codebook-free binary spherical coding framework for extreme low-bit LLM weight compression. The core pipeline is built on three components. First, local weight chunks are mapped onto a unit hypersphere and binarized into compact spherical codes, so that the main payload is a bit-packed sign stream rather than explicit VQ centroids. Second, a residual BSQ stage encodes the reconstruction error left by the base spherical codec, providing an explicit rate-distortion path without stored codebooks. Third, category-wise recovery distillation is performed after replacing each Transformer module category, reducing the mismatch between local weight reconstruction and assembled model behavior. A small 8-bit protected-channel path is used as an auxiliary stabilization mechanism for sensitive channels and is counted separately from the BSQ payload. The reported storage budget includes binary codes, neural decoders, protected-channel payloads, LoRA adapters, and metadata.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.08565v1">SMetric: Rethink LLM Scheduling for Serving Agents with Balanced Session-centric Scheduling</a></div>
    <div class="paper-meta">
      📅 2026-07-09
    </div>
    <details class="paper-abstract">
      LLM scheduling is critical to serving, yet it remains unclear how well existing designs fit agentic serving--with LLM requests issued by agents instead of humans. This shifts the workload in two ways: (1) agents act only on complete responses, making the cluster's tokens per second (TPS) the primary goal and relaxing--not eliminating--per-token latency requirements; and (2) requests share much of their KV\$-reuse exceeds 80% of request tokens in a production trace from BAILIAN, versus 54-62% in chat. This paper first contributes a systematic study of request scheduling for agents on two real-world traces. We find that to increase KV\$ reuse, existing schedulers overly prioritize routing requests to instances caching their KV\$, overloading a few while leaving the rest idle, capping TPS. We thus present two key insights: (1) load balance need not sacrifice all KV\$ reuse, thanks to the global-tier KV\$ store and (2) by utilizing the workload's intra-session locality, balancing a small fraction of requests--the first request in each agent session--suffices to balance the cluster without sacrificing most KV\$ reuse on local instances. SMETRIC realizes these insights with balanced session-centric scheduling: it routes each session's first request purely for load balance and its follow-up requests in a cache-aware manner, preserving load balance and local reuse while keeping demand on the global tier low. Using the session turn information as the scheduling metric is deliberate: it is derived efficiently and accurately from the user inputs alone, so the scheduler stays clean and stateless. SMETRIC improves cluster TPS by 10-16% under prefill-decode colocation with a global store and prefill TPS by 2-34% under disaggregation over state-of-the-art schedulers, also with a better per-token latency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.08535v1">When the Judge Changes, So Does the Measurement: Auditing LLM-as-Judge Reliability</a></div>
    <div class="paper-meta">
      📅 2026-07-09
      | 💬 6 pages, 6 figures, 4 tables
    </div>
    <details class="paper-abstract">
      An LLM-as-judge score can move even when the candidate responses stay fixed, simply because the evaluator has changed. We treat this evaluator-replacement ambiguity as a measurement-validity problem. Across four judgment datasets, we compare two upgrade paths available in practice: scaling Qwen3 dense judges from 1.7B to 32B parameters and moving across MiniMax M2-M2.7 released APIs. The main pattern is that judge upgrades are not interchangeable: only Qwen3 1.7B to 4B gives a robust adjacent gain, while MiniMax adjacent releases do not. Stronger judges reduce but do not remove position and verbosity bias. Repeated-sample juries add little when errors are correlated. Structured debate can move decisions substantially, but without parser and fallback logs those shifts cannot be attributed to deliberation. We argue that LLM-as-judge reports should include dataset slices, bias probes, error-dependence estimates, and protocol audit trails.
    </details>
</div>
