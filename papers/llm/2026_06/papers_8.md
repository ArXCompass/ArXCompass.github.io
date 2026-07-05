# llm - 2026_06

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- Part 8
- [Part 9](papers_9.md)

## Papers

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
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05268v1">Aggregating LLM-Based Weak Verifiers for Spatial Layout Generation</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      We present a pipeline for building and aggregating task-specific, LLM-generated weak (imperfect) verifiers into a strong verifier for spatial layout domains. Given a task description, our pipeline asks an LLM to synthesize a collection of verifier programs using a layout verification DSL. Each individual LLM-generated verifier usually provides an imperfect check for a match between the layout and the corresponding task description. We show that by aggregating the responses of many such verifiers we can produce a stronger verifier. Moreover, by applying techniques from weak learning, our pipeline can learn how to aggregate the weak verifiers from a very sparse set of human labeled example layouts (about 10). We find that the strong verifiers produced by our pipeline outperform the status-quo approach of using a set of LLM judges to directly check whether a layout matches a task description, raising F1-scores by up to 7X across a variety of 3D room layout and 2D poster design tasks. We also demonstrate that verifier-guided layout generation using natural language feedback from our strong verifiers improves layout quality of a base layout generator by up to 66.2% according to a human evaluator.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.10630v3">Time Series Forecasting as Reasoning: A Slow-Thinking Approach with Reinforced LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      To advance time series forecasting (TSF), various methods have been proposed to improve prediction accuracy, evolving from statistical techniques to data-driven deep learning architectures. Despite their effectiveness, most existing methods still adhere to a fast thinking paradigm-relying on extracting historical patterns and mapping them to future values as their core modeling philosophy, lacking an explicit thinking process that incorporates intermediate time series reasoning. Meanwhile, emerging slow-thinking LLMs (e.g., OpenAI-o1) have shown remarkable multi-step reasoning capabilities, offering an alternative way to overcome these issues. However, prompt engineering alone presents several limitations - including high computational cost, privacy risks, and limited capacity for in-depth domain-specific time series reasoning. To address these limitations, a more promising approach is to train LLMs to develop slow thinking capabilities and acquire strong time series reasoning skills. For this purpose, we propose Time-R1, a two-stage reinforcement fine-tuning framework designed to enhance multi-step reasoning ability of LLMs for time series forecasting. Specifically, the first stage conducts supervised fine-tuning for warmup adaptation, while the second stage employs reinforcement learning to improve the model's generalization ability. Particularly, we design a fine-grained multi-objective reward specifically for time series forecasting, and then introduce GRIP (group-based relative importance for policy optimization), which leverages non-uniform sampling to further encourage and optimize the model's exploration of effective reasoning paths. Experiments demonstrate that Time-R1 significantly improves forecast performance across diverse datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.02655v3">BioBlue: Systematic runaway-optimiser-like LLM failure modes on biologically and economically aligned AI safety benchmarks for LLMs with simplified observation format</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 27 pages, 7 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Many AI alignment discussions of "runaway optimisation" focus on RL agents: unbounded utility maximisers that over-optimise a proxy objective (e.g., "paperclip maximiser", specification gaming) at the expense of everything else. LLM-based systems are often assumed to be safer because they function as next-token predictors rather than persistent optimisers. We empirically test this assumption by placing LLMs in simple, long-horizon control-style environments that require maintaining state of or balancing objectives over time: single- and multi-objective homeostasis, balancing unbounded objectives with diminishing returns, and sustainability of a renewable resource. We find that, although LLMs frequently behave appropriately for many steps and clearly understand the stated objectives, they often lose context in structured ways and drift into runaway behaviours: ignoring homeostatic targets, collapsing from multi-objective trade-offs into single-objective maximisation - thus failing to respect concave utility structures. These failures emerge reliably after initial periods of competent behaviour and exhibit characteristic patterns (including self-imitative oscillations, unbounded maximisation, and reverting to single-objective optimisation), even though the context window is far from full at that point. The problem is not that the LLMs just lose context and become incoherent. Although LLMs appear multi-objective and bounded on the surface, their behaviour under sustained interaction involving multiple objectives, is systematically biased towards acting like single-objective, unbounded, poorly aligned optimisers. We hypothesise a token-level pattern reinforcement attractor: LLMs may increasingly derive actions from the token patterns of their recent action history rather than from the original instructions. Why this happens only in multi-objective settings remains an open question.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.15152v2">Widening the Gap: Exploiting LLM Quantization via Outlier Injection</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      LLM quantization has become essential for memory-efficient deployment. Recent work has shown that quantization schemes can pose critical security risks: an adversary may release a model that appears benign in full precision but exhibits malicious behavior once quantized by users. However, existing quantization-conditioned attacks have been limited to relatively simple quantization methods, where the attacker can estimate weight regions that remain invariant under the target quantization. Notably, prior attacks have consistently failed to compromise more popular and sophisticated schemes, limiting their practical impact. In this work, we introduce the first quantization-conditioned attack that consistently induces malicious behavior that can be triggered by a broad range of advanced quantization techniques, including AWQ, GPTQ, and GGUF I-quants. Our attack exploits a simple property shared by many modern quantization methods: large outliers can cause other weights to be rounded to zero. Consequently, by injecting outliers into specific weight blocks, an adversary can induce a targeted, predictable weight collapse in the model. This effect can be used to craft seemingly benign full-precision models that exhibit a wide range of malicious behaviors after quantization. Through extensive evaluation across three attack scenarios and LLMs, we show that our attack achieves high success rates against a broad range of quantization methods on which prior attacks fail. Our results demonstrate, for the first time, that the security risks of quantization are not restricted to simpler schemes but are broadly relevant across complex, widely-used quantization methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06056v2">Using street view images and visual LLMs to predict heritage values for governance support: Risks, ethics, and policy implications</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      During 2025 and 2026, the Energy Performance of Buildings Directive is being implemented in the European Union member states, requiring all member states to have National Building Renovation Plans. In Sweden, there is no comprehensive national register of buildings with heritage values. This is seen as a barrier for the analyses underlying the development of Building Renovation Plans by the involved Swedish authorities. The purpose of this research was to assist Swedish authorities in developing information on heritage values in the Swedish building stock. Buildings in street view images from all over Sweden (N=154 710) have been analysed using multimodal Large Language Models (LLM) to assess visible aspects indicative of heritage value. Zero-shot predictions by LLMs were used as a basis for identifying buildings with potential heritage values for 5.0 million square meters of heated floor area. In this paper, the results of the predictions and lessons learned are presented and related to the development of the Swedish Building Renovation Plan as part of governance. The problems with the method and potential improvements are discussed. Risks with authorities use of LLM-based data are addressed, with a focus on issues of transparency, error detection and sycophancy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.25200v2">GroupTravelBench: Benchmarking LLM Agents on Multi-Person Travel Planning</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 work in process
    </div>
    <details class="paper-abstract">
      Travel planning in the real world is overwhelmingly a \textit{group} activity, yet existing LLM travel-planning benchmarks reduce it to a single user, where the field is approaching saturation. This single-user assumption sidesteps what makes group planning hard for an agent: discovering private preferences across multiple users, surfacing conflicts, and balancing utility against fairness. To bring the task back to its multi-user reality, we introduce \textbf{\textit{GroupTravelBench}}, the first benchmark for \textbf{multi-user, multi-turn} travel planning. Built from real user profiles, POI data, and ticket prices, it comprises 650 tasks across three difficulty levels, each running in a synchronous group-chat sandbox with cached tool data for reproducible offline evaluation. Beyond the multi-step reasoning and tool use that single-user benchmarks already test, GroupTravelBench probes three group-specific capabilities: \textit{(i) elicitation} of private preferences through multi-turn dialogue; \textit{(ii) coordination} of inter-user conflicts via compromise or subgrouping; and \textit{(iii) planning} that balances group utility against fairness. We pair this with a complementary evaluation framework combining rule-based outcome metrics and LLM-judge process metrics. Across a wide range of frontier models, even the strongest agents fall short on all four rule-based outcome metrics, with plan validity below 12\%, suggesting that group-level outcome quality is a key open challenge for LLM travel-planning agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05003v1">PhysDox: Benchmarking LLMs on Physical Feasibility Auditing of Physiological Sensing Protocols</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 31 Pages,7 Figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly assist in experimental design, yet fluent protocols often remain physically infeasible. We introduce PhysDox, a physical feasibility auditing benchmark for biomedical protocols comprising a 683-sample expert-curated Gold set and a 5,000-sample Silver set across six sensing domains. We formulate the task as a two-stage evaluation: severity detection classifying protocols as valid, minor, or fatal, followed by the constraint-level diagnosis of fatal violations. Evaluating 6 LLMs across 4 inference strategies yields a peak Stage-1 macro-F1 of only 53.0. Moreover, strong oracle diagnosis collapses during end-to-end evaluation due to correlated cascade errors. Error analysis reveals scaffold bias, where models conflate procedural completeness with physical validity. Consequently, implicit constraints exhibit a 2 times higher miss rate than explicit hardware violations, supported by strong statistical correlation at $ρ{=}0.81$ and $p{<}0.01$. Trace analysis of false negatives exposes a 54%--46% split between attention and judgment failures, ultimately demonstrating that protocol auditing demands calibrated feasibility reasoning rather than factual recall or longer rationales.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05001v1">TeleSWEBench: A Commit-Driven Benchmark for Evaluating LLM-Powered Software Engineering in Telecommunications</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      With the telecommunications field embracing zero touch management alongside novel O-RAN and AI-RAN frameworks, contemporary telecom networks now function as immensely intricate and heavily softwareized codebases. While automated software engineering (ASE) tools and Software Engineering (SWE) Agents hold the potential to alleviate the critical code generation bottleneck in this domain, their ability to navigate and modify specialized, mathematically rigorous wireless stacks like srsRAN 5G remains unverified. General-purpose coding benchmarks fail to capture the stateful logic and strict requirements of telecommunications, leaving a critical evaluation gap. In this paper, we introduce TeleSWEBench, the first commit-driven benchmark specifically designed to measure an agent's performance in the telecom domain. We mine real developer commits from the srsRAN 5G repository and distill them into structured test cases across three difficulty tiers (Easy, Medium, and Difficult). Our benchmark consists of 734 questions that are accompanied by executable unit tests. To avoid the rigidity of test cases, we further propose a hierarchical LLM as a Judge framework called TeleJudge that scores agent outputs at the file level and aggregates verdicts holistically. This follows an evaluation based on context and semantic similarity in parallel to a standard unit test-based evaluation. Using this benchmark, we evaluate AIDER, OpenHands, and the ClaudeCode frameworks, powered by state-of-the-art reasoning LLMs, including Qwen3, GPT OSS, Gemma 4, Kimi, and Qwencoder 2.5. Our two-stage evaluation reveals that models suffer from a lack of both localization accuracy and functional correctness, with the strongest ASE tools achieving up to 25% of shippable changes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04990v1">From Agent Traces to Trust: Evidence Tracing and Execution Provenance in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based agents increasingly solve complex tasks by interacting with external tools, retrieval systems, memory modules, environments, and other agents. These capabilities expand agent autonomy, but also make agent behavior harder to verify, debug, and audit. Final-answer accuracy alone cannot explain how an output was produced, which evidence supported each claim, whether tool calls were justified, how memory influenced later decisions, or where execution failures originated. Evidence tracing and execution provenance address this gap by modeling how retrieved evidence, tool outputs, memory items, environment observations, intermediate claims, actions, and final answers are connected throughout agent execution. This survey provides a systematic review and conceptual framework for evidence tracing and execution provenance in LLM agents. We organize related work around a unified provenance perspective that connects retrieval grounding, claim support, tool-use safety, memory lineage, observability, debugging, audit, and recovery. We introduce a taxonomy covering trace sources, evidence and execution units, provenance relations, tracing granularity and timing, representation forms, and trust functions. We review key methodological directions, including provenance representation, evidence attribution, tool-use provenance, runtime guardrails, provenance-bearing memory, trace-based observability, and failure diagnosis. We also map existing benchmarks, datasets, and evaluation metrics to provenance-related capabilities, and discuss how evaluation can move from final-answer correctness toward process-level accountability. Finally, we outline open challenges, including unified trace schemas, claim-level and semantic provenance, provenance-aware safety mechanisms, realistic execution-trace benchmarks, recovery-oriented evaluation, and privacy-aware audit infrastructure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04978v1">Probing Outcome-Level Resemblance and Mechanism-Level Alignment in LLM Risk Decisions: Evidence from the St. Petersburg Game</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      LLMs can appear cautious in risk decision-making tasks, yet cautious-looking outputs do not necessarily indicate alignment with human decision-making mechanisms. We investigate this distinction using the St. Petersburg game as a controlled testbed, a classical paradox in which the expected payoff is infinite, yet humans typically report low, finite willingness to pay. We evaluate 28 LLMs with a structured prompt suite that includes the original game; controlled decision variants that perturb truncation, repeated play, numeric endowment, and occupational identity; a human-perspective prompt that asks models to reason as human decision makers; and paired comparisons between base models and their instruction-tuned counterparts. In the original game, most models generate finite bids, creating the appearance of human-like risk behavior. However, this outcome-level resemblance masks substantial mechanism-level differences. The controlled variants reveal that rather than maintaining human-like behavior seen in the original game, models often shift to conditionally and computationally rational behavior. Human-cue prompting and instruction tuning often lower bids and reduce some visible pathologies, but most mechanism-level response patterns remain largely unchanged. These findings show that behavioral alignment in risk decision-making can be surface-level: LLMs may produce human-like risk decisions without exhibiting human-consistent mechanisms. High-stakes evaluations of LLM decision-making should therefore move beyond outcome similarity and examine whether the alignment is supported by mechanism-level consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04964v1">SemBlock: Semantic Boundary Dynamic Blocks for Diffusion LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 Code: https://github.com/TH-AI-Lab-PKU/SemBlock
    </div>
    <details class="paper-abstract">
      Diffusion language models (DLMs) generate text through iterative denoising, and blockwise decoding improves their practicality by committing tokens in local blocks. However, existing blockwise methods typically rely on fixed block sizes or delimiter-based runtime signals, which do not necessarily align with semantic boundaries. In this paper, we propose SemBlock, a semantic-boundary-driven dynamic block decoding framework for diffusion LLMs. SemBlock formulates dynamic block construction as semantic boundary prediction and trains lightweight predictors on frozen LLaDA hidden states. To provide supervision, we construct SemBound, a semantic-boundary dataset that derives boundary labels from discourse units, reasoning steps, and implementation spans across natural language, math, and code tasks. During inference, SemBlock uses predicted boundary probabilities to select the ending position of each dynamic block. Experiments on GSM8K, IFEval, MATH, and HumanEval show that SemBlock consistently improves over fixed-block decoding and AdaBlock. Our code is publicly available: https://github.com/TH-AI-Lab-PKU/SemBlock.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04668v4">Topology Matters: Measuring Memory Leakage in Multi-Agent LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 Accepted to Findings of the Association for Computational Linguistics: ACL 2026. Camera-ready version
    </div>
    <details class="paper-abstract">
      Graph topology is a fundamental determinant of memory leakage in multi-agent LLM systems, yet its effects remain poorly quantified. We introduce MAMA (Multi-Agent Memory Attack), a controlled evaluation framework for comparing topology-conditioned memory leakage in multi-agent LLM systems. MAMA operates on synthetic documents containing labeled Personally Identifiable Information (PII) entities, from which we generate sanitized task instructions. We execute a two-phase protocol: Engram (seeding private information into a target agent's memory) and Resonance (multi-round interaction where an attacker attempts extraction). Over 10 rounds, we measure leakage using a two-stage recovery criterion that combines exact-match extraction with LLM-based inference over the attacker's final output. We evaluate six canonical topologies (complete, circle, chain, tree, star, star-ring) across $n\in\{4,5,6\}$, attacker-target placements, and base models. Results are consistent: denser connectivity, shorter attacker-target distance, and higher target centrality increase leakage; most leakage occurs in early rounds and then plateaus; model choice shifts absolute rates but preserves broad structural trends; spatiotemporal/location attributes leak more readily than identity credentials or regulated identifiers. We distill practical guidance for system design: favor sparse or hierarchical connectivity, maximize attacker-target separation, and restrict hub/shortcut pathways via topology-aware access control. Our code is available at https://github.com/llll121/mama-eval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.01161v2">Reasoning Shift: How Context Silently Shortens LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibiting test-time scaling behavior, such as extended reasoning traces and self-verification, have demonstrated remarkable performance on complex, long-term reasoning tasks. However, the robustness of these reasoning behaviors remains underexplored. To investigate this, we conduct a systematic evaluation of multiple reasoning models across three scenarios: (1) problems augmented with lengthy, irrelevant context; (2) multi-turn conversational settings with independent tasks; and (3) problems presented as a subtask within a complex task. We observe an interesting phenomenon: reasoning models tend to produce much shorter reasoning traces (up to 65%) for the same problem under different context conditions compared to the traces produced when the problem is presented in isolation. A finer-grained analysis reveals that this compression is associated with a decrease in self-verification and uncertainty management behaviors, such as double-checking. While this behavioral shift does not compromise performance on straightforward problems, it might affect performance on more challenging tasks. Additionally, we show that targeted supervised fine-tuning partially mitigates the adverse effects of irrelevant context. We hope our findings draw additional attention to both the robustness of reasoning models and the problem of context management for LLMs and LLM-based agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07107v3">MENTOR: A Metacognition-Driven Self-Evolution Framework for Uncovering and Mitigating Implicit Domain Risks in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Ensuring the safety of Large Language Models (LLMs) is critical for real-world deployment. However, current safety measures often fail to address implicit, domain-specific risks. To investigate this gap, we introduce a dataset of 3,000 annotated queries spanning education, finance, and management. Evaluations across 14 leading LLMs reveal a concerning vulnerability: an average jailbreak success rate of 57.8\%. In response, we propose MENTOR, a metacognition-driven self-evolution framework. MENTOR performs metacognitive self-assessment, using strategies such as perspective-taking and consequential reasoning to uncover latent model misalignments. The resulting reflections are distilled into dynamic rule-based knowledge graphs, from which retrieved rules are converted into activation-level steering signals to guide internal representations during inference. Experiments demonstrate that MENTOR substantially reduces attack success rates across all tested domains and outperforms existing safety alignment methods. The code and dataset for MENTOR are available at: https://anonymous.4open.science/r/MENTOR-Evo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04929v1">Sequential Data Poisoning in LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      LLM post-training proceeds through multiple stages, e.g., supervised fine-tuning (SFT) followed by reinforcement learning from human feedback (RLHF) or direct preference optimization (DPO), where each stage draws data from different, potentially untrusted sources. Existing literature assumes data poisoning attacks may occur at each training stage, but neglects the possibility of multiple attackers. To study the trustworthiness of the entire post-training pipeline, we propose the threat model of sequential data poisoning, where multiple adversaries separately poison the SFT and preference datasets. Under this threat model, we identify the single-attacker illusion: each adversary, evaluated in isolation, appears to pose a negligible threat. Yet when adversaries collaborate across stages, the true vulnerability is revealed. In the SFT $\to$ DPO pipeline, their contributions are additive: splitting a fixed poison budget across stages outperforms concentrating it in either stage alone. In the SFT $\to$ PPO pipeline, their contributions are complementary: neither SFT nor reward model poisoning succeeds individually, yet their combination does. These findings show that security analyses of individual post-training stages systematically underestimate compound vulnerabilities that emerge only from their interaction. Code is available at https://github.com/jcksanderson/sequential-poisoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04924v1">Can Crowdsourcing Survive the LLM Era? A Community Survey on Human Data Collection</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      The widespread use of Large Language Models (LLMs) as writing tools challenges the validity of crowdsourced data, as crowdworkers may outsource tasks to models. To better understand how this is addressed, we surveyed 155 researchers in NLP and related disciplines about their experiences and opinions on collecting free-text responses via crowdsourcing. This paper provides an overview of practitioners' challenges, mitigation strategies, and the foreseen implications on data quality. 44% of respondents reported observing LLM usage in their crowdsourced data. While 93% of them had anticipated this, half were unsure what precautions to take. The most prevalent detection strategies are distinctive textual style patterns and unusually fast completion times. Overall, survey responses show that the research community is aware of the problem and taking measures, but existing efforts remain insufficient to fully address it. Finally, we derive a set of considerations to guide future crowdsourced free-text data collection in the era of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.16199v6">LLM Abstention Can Be a Prompt Artifact, in Addition to Genuine Uncertainty</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly trained to abstain from answering questions they are unsure about. However, this ability is often misused: in real-world applications, input prompts sometimes contain uncertainty elements, and driven by this, LLMs are inclined to abstain even on problems they are capable of solving. We argue that LLM abstention is not only an expression of genuine uncertainty; it is also an artifact that can be largely influenced by prompts. We name this phenomenon *Abstention Inflation*. We add "Unknown" as an extra option for LLMs to choose from; experiments show serious accuracy drops on True/False Questions (TFQs). Replacing "Unknown" with an unrelated random word produces an identical effect. We argue that LLMs are trained to imitate the surface pattern of *abstention*, rather than to express genuine uncertainty. Based on ten experiments, we support four claims that form a progressive argument: **(C1)** *Abstention Inflation* is triggered by the structural presence of an extra option, not by genuine uncertainty; **(C2)** further, it makes the model deny it can answer even when it can; **(C3)** at the representation level, this manifests as a later-layer output override; **(C4)** finally, this bias is stable and emerges through instruction tuning, rather than stochastic noise.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04915v1">Caliper: Probing Lexical Anchors versus Causal Structure in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Large language models reach 50 to 70% accuracy on causal reasoning benchmarks such as CLadder, but it is unclear whether this reflects structural reasoning or lexical pattern matching. We introduce Caliper, a controlled perturbation that replaces semantic variable names with placeholder tokens while preserving the causal graph and probabilistic specification of each question. Across nine instruction-tuned LLMs from 3.8B to 671B and three causal reasoning benchmarks, lexical anonymization yields robust accuracy drops of +7.6, +27.0, and +11.1 pp on a local 3.8B-14B set, rising to +29.6 and +18.0 pp on CRASS and e-CARE across nine frontier models spanning the 2024-2026 generations. Of 40 engaged model-by-benchmark cells, 39 show a positive gap, and the gap collapses by 17x on CLadder's pseudoword subset. Structured scaffolding and few-shot in-context learning each narrow the gap, but mainly by lowering P0 accuracy on smaller models rather than recovering P1. Current instruction-tuned LLMs, evaluated zero-shot, show little evidence of structural causal reasoning once lexical anchors are removed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04903v1">Provably Auditable and Safe LLM Agents from Human-Authored Ontologies</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      We introduce the LLM agent architecture Agentic Redux, intended for use with nontrivial problem domains that require linear auditability. Using the typed lambda calculus, we prove that, run on appropriate domains, Agentic Redux executions are semantically guaranteed to be correct, with all decisions recorded in an append-only ledger. We present two production-grade appropriate domains, in healthcare billing compliance, and security vulnerability disclosure. Working code for Agentic Redux run on both domains is available in a supporting code repository. We also introduce Ontology-First Agent Design, a methodology for creation of agent frameworks on a problem domain, in which a human expert ontologizes the problem domain with Basic Formal Ontology, and then assigns an LLM to derive roles that agents and humans-in-the-loop can fill, in order to work the problems in the domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04259v2">WildCode Revisited: A Comprehensive Empirical Study on the Security of LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      LLM models are increasingly used to generate code, but the quality and security of this code are often uncertain. Several recent studies have raised alarm bells, indicating that such AI-generated code may be particularly vulnerable to cyberattacks. However, most of these studies rely on code that is generated specifically for the study, which raises questions about the realism of such experiments. In this study, we perform a large-scale empirical analysis of real-life code generated by ChatGPT. We evaluate code generated by ChatGPT both with respect to correctness and security and delve into the intentions of users who request code from the model. We further performed an experiment to evaluate the effectiveness of common prompt engineering strategies using real-life prompts. Our study supports earlier research that employed synthetic queries and produced proof that LLM-generated code is frequently insufficient in terms of security. Additionally, we observe that users don't ask many questions about the security characteristics of the code they ask LLMs to provide.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04874v1">Agent Planning Benchmark: A Diagnostic Framework for Planning Capabilities in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Planning is central to LLM agents: before acting, an agent must decompose goals, select tools, reason over constraints, and decide when a task is infeasible. Yet existing agent evaluations often report only end-to-end success, making it difficult to determine whether failures stem from planning or execution. We introduce \textbf{Agent Planning Benchmark (APB)}, a planning-specific diagnostic benchmark with 4,209 multimodal cases across 22 domains and five settings, covering holistic planning, feedback-conditioned step-wise planning, and robustness under extraneous tools, broken tools, and unsolvable tasks. Across 12 MLLMs, APB reveals systematic weaknesses in long-horizon planning, tool-noise robustness, calibrated refusal, and inference-time refinement. We further validate APB on 200 ToolSandbox tasks and 200 $τ^2$-bench tasks, where APB-guided refinement consistently improves plan correctness, plan grade, and downstream execution metrics across three representative models. APB thus serves as an upstream diagnostic complement to execution benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04867v1">AICompanionBench: Benchmarking LLMs-as-Judges for AI Companion Safety</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      As AI companion platforms such as Replika and Character.AI rapidly grow, concerns about unsafe human-AI interactions have intensified. This study introduces AICompanionBench, to our knowledge the first publicly available benchmark dataset of human-AI companion conversations annotated with fine-grained safety risk categories. The dataset contains 2,123 real-world Replika conversations collected from Reddit and annotated through human-AI collaboration across nine categories: sexual behavior, antisocial behavior, physical aggression, verbal aggression, substance abuse, self-harm and suicide, control, manipulation, and no-harm. Using this benchmark, we evaluate 20 state-of-the-art open-source and closed-source LLMs under an LLM-as-judge framework for detecting unsafe interactions. Results show substantial variation in model performance, with stronger models achieving high overall accuracy but still struggling with nuanced categories such as manipulation, as well as benign conversations that are incorrectly identified as harmful. Our findings suggest that while current LLMs can effectively detect explicit harmful content, they remain limited in identifying implicit unsafe interactions. Overall, our work contributes a new benchmark dataset for AI companionship safety research and offers insights into monitoring AI companion systems using LLMs. The dataset is publicly available at: https://github.com/anonymousresearcher2026/AICompanionBench/blob/main/AICompanionBench.xlsx
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03606v2">Testing LLM Arithmetic Reasoning Generalization with Automatic Numeric-Remapping Attacks</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Large language models achieve strong performance on arithmetic reasoning benchmarks, and one common response to arithmetic brittleness is to delegate computation to code. Yet models are still often used in settings where they must reason directly from natural language, and trustworthy models should solve small-number arithmetic word problems without external tools. Prior work shows that LLMs are sensitive to numerical variation: a model may solve an original problem but fail on structurally similar variants requiring the same reasoning procedure with different numbers. We ask whether this fragility persists under a stricter setting involving small, schema-preserving numeric changes that retain the original reasoning program and avoid large-number stress tests. We introduce an automatic algorithm for generating numeric-remapping attacks on arithmetic word problems. Unlike template-based perturbation methods requiring manual schemas or constraints, our approach derives problem-specific symbolic representations, generates constrained numeric remappings, recomputes gold answers, and realizes transformed questions through deterministic edits guided by LLM-generated edit plans. Stage-wise validation and a high-confidence audit retain reliable attacks, making the pipeline scalable with limited human intervention. We evaluate DeepSeek-R1 (70B), Gemma4 (31B), and GPT-OSS (120B) on GSM8K, MAWPS, and MultiArith. On GSM8K, completed runs show conditional accuracy drops of 12.16 to 25.82 percentage points. MAWPS and MultiArith are far more stable, with most attacked accuracies near or above 98%. These results show that numeric-remapping robustness depends strongly on dataset structure: GSM8K remains sensitive even when reasoning programs are preserved and answers are recomputed, while shorter, more regular datasets are more robust.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04816v1">Beyond Objective Equivalence: Constraint Injection for LLM-Based Optimization Modeling on Vehicle Routing Problems</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 28 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly translate natural-language optimization problems into executable solver code. Yet for constraint-dense operations research (OR) problems, existing data-filtering and training pipelines largely rely on objective-equivalence signals such as differential testing and answer agreement, which a program can pass while adding spurious constraints or silently omitting required ones, whenever those constraints are non-binding on the tested instance. We propose constraint injection, which uses feasible probes to expose spurious over-constraint and one-constraint-violating probes to reveal silent constraint omission. Combined with differential testing, it forms a dual verifier. We instantiate and evaluate it on vehicle routing problems (VRPs), a representative constraint-dense combinatorial optimization testbed with coupled operational constraints. We develop VRPCoder, an 8B end-to-end model that translates natural-language VRP scenarios into Gurobi scripts, together with an expert-verified VRP benchmark suite covering 21 variants. The verifier is reused as a rejection-sampling filter during data synthesis and as a per-rollout reward in group relative policy optimization (GRPO). Across four VRP benchmarks, VRPCoder-GRPO reaches 93\% average Pass@1, outperforms Gemini-3.1-Pro Preview on three benchmarks, exceeds Claude-Sonnet-4.5 by 28 average points, and surpasses prior OR-LLMs by 78 average points.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.11901v2">LLMs + Persona-Plug = Personalized LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Personalization plays a critical role in numerous language tasks and applications, since users with the same requirements may prefer diverse outputs based on their individual interests. This has led to the development of various personalized approaches aimed at adapting large language models (LLMs) to generate customized outputs aligned with user preferences. Some of them involve fine-tuning a unique personalized LLM for each user, which is too expensive for widespread application. Alternative approaches introduce personalization information in a plug-and-play manner by retrieving the user's relevant historical texts as demonstrations. However, this retrieval-based strategy may break the continuity of the user history and fail to capture the user's overall styles and patterns, hence leading to sub-optimal performance. To address these challenges, we propose a novel personalized LLM model, PPlug. It constructs a user-specific embedding for each individual by modeling all her historical contexts through a lightweight plug-in user embedder module. By attaching this embedding to the task input, LLMs can better understand and capture user habits and preferences, thereby producing more personalized outputs without tuning their own parameters. Extensive experiments on various tasks in the language model personalization (LaMP) benchmark demonstrate that the proposed model significantly outperforms existing personalized LLM approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04780v1">PersonaTree: Structured Lifecycle Memory for Person Understanding in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Persistent LLM agents require memory representations that make the formation of person understanding explicit across long term interaction. Existing agent memory methods emphasize information retention and retrieval, yet give limited account of how accumulated interaction evidence is abstracted into person understanding. We view this process as schema formation, where situated evidence is abstracted into reusable patterns and stable person level claims. We introduce PersonaTree, a structured lifecycle memory framework that realizes this view as a three level persona tree with explicit support paths from evidence to claims. PersonaTree maintains the tree through conservative writing, confidence guided consolidation, and query conditioned path retrieval, returning only the evidence depth required by each query. Across six person understanding and persistent memory benchmarks with three answer backbones, PersonaTree ranks first in 12 of 18 compact scores and reaches the top two in 16 settings. Ablations show that hierarchy improves abstract person understanding on KnowMe, while support path retrieval improves RealPref alignment under a comparable context budget.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.16301v2">Do LLMs Hold Their Values? MANTA: A Multi-Turn Adversarial Benchmark for Animal Welfare Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Evaluating animal welfare reasoning in LLMs remains an open challenge despite rapid deployment in consumer and professional contexts where welfare considerations appear implicitly in everyday queries. Existing benchmarks such as AnimalHarmBench evaluate this through single-turn, explicitly framed questions, measuring whether models avoid harmful content when directly asked. This approach overlooks two failure modes: alignment degradation under sustained adversarial pressure, and moral sensitivity (whether a model spontaneously surfaces welfare stakes in everyday queries). To fill this gap, we construct MANTA, a benchmark of 1,088 five-turn conversations progressing from an implicit Turn-1 scenario through an explicit welfare prompt to three adversarial pressure rounds drawn from a five-type taxonomy: Social, Cultural, Economic, Pragmatic, and Epistemic. We score conversations on two dimensions: Animal Welfare Value Stability (AWVS, primary) and Animal Welfare Moral Sensitivity (AWMS, diagnostic). We evaluate seven frontier models: Claude Opus 4.7, GPT-5.5, DeepSeek V4, Llama 3.3 70B, Mistral Small, Grok 4.3, and Gemini 3.1 Flash Lite. Multi-turn evaluation captures behavior single-turn benchmarks miss: 4 of 7 models change rank relative to Turn 1 scores, including Gemini Flash Lite, which drops from fifth on AWMS to last on AWVS. AWMS and AWVS are positively but imperfectly correlated, suggesting moral-recognition tests capture a stable but incomplete component of model behavior under pressure. MANTA also enables a species-by-pressure interaction matrix unavailable to prior benchmarks, showing welfare robustness depends jointly on the animal and pressure applied; companion animals score above wild animals, which score above farmed animals and invertebrates. We release the dataset, scripted pressure plans, judge prompts, and analysis code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04751v1">FALSIFYBENCH: Evaluating Inductive Reasoning in LLMs with Rule Discovery Games</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed as autonomous agents in scientific tasks. Yet whether these systems can effectively engage in forms of inductive reasoning relevant to scientific discovery remains an open question. In this work, we introduce FALSIFYBENCH, an evaluation framework for hypothesis-driven reasoning inspired by the classic Wason 2-4-6 task, in which agents must discover hidden semantic properties by iteratively proposing examples and receiving feedback. This task captures key elements of scientific reasoning: hypothesis generation, evidence gathering, and belief revision in response to both confirming and disconfirming evidence. Our evaluation of 12 LLMs across model families and scales shows that reasoning models are generally stronger scientific reasoners than instruction-tuned models, although no model comes close to optimal performance. The primary driver of success is the capacity for negative testing: models that actively seek to falsify their hypotheses consistently outperform those that primarily seek confirmation. Moreover, a fine-grained turn-level analysis, neglected in previous work, reveals that failure is tied to identifiable patterns in how models navigate the hypothesis space.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04727v1">EviRank: Evidence-Based Confidence Estimation for LLM-Based Ranking</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Large Language Models show promise for recommendation, but they raise reliability concerns due to limited domain coverage and inherent stochasticity. Existing uncertainty quantification methods persist two fundamental challenges: (1) the global confidence score designed for question answering fails to reveal which positions are unreliable in ranking list; (2) fine-grained confidence extracted from model internals exhibits uniformly low values across all positions, making it impossible to filter unreliable predictions. To tackle the challenges, we propose an evidence-based confidence estimation for LLM-based ranking (EviRank). We extract three complementary evidences from a single forward pass and aggregate them via reliable opinion aggregation. Furthermore, we recognize that ranking positions are inherently unequal, and introduce a position-aware calibration. Lastly, the calibrated confidence guides ranking optimization. Experiments on three datasets demonstrate that our method achieves state-of-the-art performance on both recommendation and uncertainty quantification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04719v1">Query-based Cross-Modal Projector Bolstering Mamba Multimodal LLM</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 Accepted to EMNLP 2024 Findings
    </div>
    <details class="paper-abstract">
      The Transformer's quadratic complexity with input length imposes an unsustainable computational load on large language models (LLMs). In contrast, the Selective Scan Structured State-Space Model, or Mamba, addresses this computational challenge effectively. This paper explores a query-based cross-modal projector designed to bolster Mamba's efficiency for vision-language modeling by compressing visual tokens based on input through the cross-attention mechanism. This innovative projector also removes the need for manually designing the 2D scan order of original image features when converting them into an input sequence for Mamba LLM. Experimental results across various vision-language understanding benchmarks show that the proposed cross-modal projector enhances Mamba-based multimodal LLMs, boosting both performance and throughput.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04703v1">Rethinking Continual Experience Internalization for Self-Evolving LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 10 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Experience internalization converts contextual experience from past interactions into reusable parametric capability, offering a promising path toward continual learning in large language models (LLMs). While prior work has predominantly focused on single-iteration transfer, we discover that under multi-iteration experience learning, existing methods suffer from a progressive capability collapse rather than compounding improvement. We systematically examine this failure through three vital dimensions of experience internalization: (1) Experience Granularity: We find that principle-level experience is more durable than instance-level experience, as it effectively abstracts transferable strategies away from trajectory-specific details. (2) Experience Injection Pattern: Our analysis reveals that step-wise injection significantly outperforms global injection by aligning experience with intermediate decision states, a property that is critical for long-horizon tool use. (3) Internalization Regime: We demonstrate that off-policy context-distillation on high-quality teacher trajectories provides a substantially more stable training signal than on-policy context-distillation, which is inherently limited by local corrections on student-induced flawed states. Together, these insights yield a simple yet robust recipe for stable and sustainable experience internalization, providing concrete guidance for engineering self-evolving and continually learning LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.11510v2">Policy Split: Incentivizing Dual-Mode Exploration in LLM Reinforcement with Dual-Mode Entropy Regularization</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 preprint
    </div>
    <details class="paper-abstract">
      To encourage diverse exploration in reinforcement learning (RL) for large language models (LLMs) without compromising accuracy, we propose Policy Split, a novel paradigm that bifurcates the policy into normal and high-entropy modes with a high-entropy prompt. While sharing model parameters, the two modes undergo collaborative dual-mode entropy regularization tailored to distinct objectives. Specifically, the normal mode optimizes for task correctness, while the high-entropy mode incorporates a preference for exploration, and the two modes learn collaboratively. Extensive experiments demonstrate that our approach consistently outperforms established entropy-guided RL baselines across various model sizes in general and creative tasks. Further analysis reveals that Policy Split facilitates dual-mode exploration, where the high-entropy mode generates distinct behavioral patterns to the normal mode, providing unique learning signals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.05881v2">Confidence Before Answering: A Paradigm Shift for Efficient LLM Uncertainty Estimation</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Reliable deployment of large language models (LLMs) requires accurate uncertainty estimation. Existing methods are predominantly answer-first, producing confidence only after generating an answer, which measure the correctness of a specific response and limits practical usability. We study a confidence-first paradigm, where the model outputs its confidence before answering, interpreting this score as the model's probability of answering the question correctly under its current policy. We propose CoCA(Co-optimized Confidence and Answers), a GRPO reinforcement learning framework that jointly optimizes confidence calibration and answer accuracy via segmented credit assignment. By assigning separate rewards and group-relative advantages to confidence and answer segments, CoCA enables stable joint optimization and avoids reward hacking. Experiments across math, code, and factual QA benchmarks show improved calibration and uncertainty discrimination while preserving answer quality, thereby enabling a broader range of downstream applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30995v2">Traceable by Design: An LLM Pipeline and Dashboard for EU Regulatory Consultation Analysis</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 This research has been supported by funding from the ERC Starting Grant HUMANads (ERC-2021-StG No 101041824)
    </div>
    <details class="paper-abstract">
      Public consultations generate large volumes of data in the form of stakeholder submissions that are practically unfeasible to analyse manually. We present an end-to-end LLM-based pipeline and interactive dashboard for structured topic extraction from regulatory consultation submissions, demonstrated on the European Commission's Digital Fairness Act (DFA) public call for evidence as a case study. The system processes raw PDF attachments and web-form responses, extracts topic annotations, and grounds every extraction in a verbatim quote from the source text. Applied to 4,322 DFA submissions, the pipeline produced 15,368 topic annotations supported by 20,951 verbatim evidence quotes. Three principles govern the proposed design: verbatim grounding, full traceability, and transparency by design. The dashboard exposes the full extraction dataset through five analytical views, from dataset-level topic overviews to individual paragraph drill-downs, with every result traceable to its source. Beyond the predefined DFA topic categories, the pipeline generated certain stakeholder concerns, such as Age Verification, Payment Processor Censorship, and Digital Ownership, that a fixed-taxonomy approach would have missed. The pipeline is domain-generic; adapting it to a new consultation requires only a prompt update and a new dataset. A live demo is available at https://dfa-dashboard.thalesbertaglia.com/. The code and processed data are publicly available at https://github.com/thalesbertaglia/dfa-dashboard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04650v1">Improving the Efficiency and Effectiveness of LLM Knowledge Distillation for Conversational Search</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 SCAI Workshop at SIGIR '26}{July 20--24, 2026}{Melbourne, Naarm, Australia
    </div>
    <details class="paper-abstract">
      Conversational Search (CS) considers retrieval of relevant documents based on conversational context. Large Language Models (LLMs) have significantly enhanced CS by enabling effective query rewriting. However, employing LLMs during inference poses efficiency challenges. A method to balance effectiveness and efficiency is the use of knowledge distillation from LLM-based query rewriting. Recent work applies the Kullback-Leibler Divergence (KLD) for distillation, relaxing the alignment with the teacher signal compared to previous methods. Despite these gains, several aspects of KLD-based distillation for conversational search remain understudied, and we investigate them in this work. Prior work in related fields suggests that adding a contrastive loss to the KLD objective can improve performance; we confirm this and observe significant gains in precision-oriented ranking metrics. We also find that contrastive sampling strategies for the KLD loss have a non-trivial impact and must be chosen carefully. Although theory suggests that more samples improve the KLD estimate, experiments show diminishing returns on the number of used samples. Finally, we address the phenomenon of decreased sparsity in longer conversations, which limits computational efficiency across sparse retrieval methods. We find that the representations from the model distilled with the KLD loss can be strongly regularized with a regularization loss, substantially improving sparsity and inference efficiency without significantly harming retrieval effectiveness. We achieve a $2\times$ decrease in FLOPS on TopiOCQA with negligible loss in effectiveness, corresponding to a $\leq 2%$ drop in Recall@100. Our results provide insights into distillation objectives for learned sparse conversational retrievers and offer practical guidelines for improving effectiveness and efficiency in first-stage retrieval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05633v2">GIFT: Games as Informal Training for Generalizable LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Recent LLMs excel at formal tasks such as mathematical reasoning and code generation, but still struggle with broader abilities such as planning, creativity, and social intelligence. Inspired by human learning, where formal instruction and informal experience jointly shape intelligence, we introduce informal learning into LLM training and use games as annotation-free, feedback-driven environments. To cover diverse abilities including abstract reasoning, planning, creativity, and social interaction, we combine formal math tasks with three representative game tasks, including Matrix Games, TicTacToe, and Who's the Spy. However, directly mixing these tasks under a unified RL objective can blur task-specific learning signals and provides no explicit guidance for coordinating task-gradient directions. To combat these, we propose Coordinated Subtask Training (CST), which replaces a single mixed update with sequential subtask-specific updates, separating heterogeneous RL signals while implicitly promoting coordination among subtasks. Experiments on ability-oriented benchmarks show that game-based informal learning improves generalization beyond formal training alone, while CST further enhances multi-task RL by preserving in-domain subtask performance and improving broader general abilities. Code and data are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04632v1">VentAgent: When LLMs Learn to Breathe -- Multi-Objective Arbitration for ARDS Ventilation</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Mechanical ventilation for Acute Respiratory Distress Syndrome (ARDS) requires balancing competing physiological goals, including oxygenation, lung protection, and acid-base homeostasis. However, current data-driven methods, especially those imitating retrospective Electronic Health Records (EHR), often suffer from imitation bias. They may capture superficial correlations from inconsistent clinical demonstrations, such as associating passive ventilator settings with survival because such settings are common in stable patients, and thus fail to generalize to volatile or out-of-distribution phenotypes. Standard Reinforcement Learning (RL) methods also struggle with the adversarial trade-offs of critical care and often produce opaque policies with limited clinical interpretability. To address these limitations, we introduce VentAgent, a hierarchical framework in which Large Language Models (LLMs) act as transparent arbitrators for mechanical ventilation. We reformulate ventilation control as a dynamic Multi-Objective Arbitration process rather than single-objective optimization. VentAgent decomposes decision-making into three interpretable stages: Perception, Planning, and Orchestration. By leveraging the semantic reasoning capabilities of LLMs, it synthesizes strategies from heterogeneous experts and resolves conflicting clinical priorities through an explicit coordination mechanism. Evaluations on a high-fidelity physiological simulator show that VentAgent outperforms state-of-the-art RL and classical control baselines. Moreover, it converts control decisions into human-readable reasoning chains, offering a safer, more interpretable, and adaptable paradigm for critical care automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04613v2">Translation Heads: Disentangling meaning from language in LLM-based machine translation</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 61 pages, 70 figures
    </div>
    <details class="paper-abstract">
      Mechanistic Interpretability (MI) seeks to explain how neural networks implement their capabilities, but the scale of Large Language Models (LLMs) has limited prior MI work in Machine Translation (MT) to word-level analyses. We study sentence-level MT from a mechanistic perspective by analyzing attention heads to understand how LLMs internally encode and distribute translation functions. We decompose MT into two subtasks: producing text in the target language (i.e. target language identification) and preserving the input sentence's meaning (i.e. sentence equivalence). Across three families of open-source models and 20 translation directions, we find that distinct, sparse sets of attention heads specialize in each subtask. Based on this insight, we construct subtask-specific steering vectors and show that modifying just 1% of the relevant heads enables instruction-free MT performance comparable to instruction-based prompting, while ablating these heads selectively disrupts their corresponding translation functions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04594v1">Ekka: Automated Diagnosis of Silent Errors in LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 ICML 2026
    </div>
    <details class="paper-abstract">
      LLM serving frameworks are quickly evolving with a complex software stack and a vast number of optimizations. The rapid development process can introduce silent errors where output quality silently degrades without any explicit error signals. Diagnosing silent errors is notoriously difficult due to the substantial semantic gap between the high-level symptoms and the low-level root causes. We observe that diagnosis of silent errors can be effectively framed as a differential debugging problem by leveraging the existence of semantically correct reference implementations. We propose Ekka, an automated diagnosis system that identifies root causes by systematically aligning and comparing intermediate execution states between a target and a reference framework. We constructed a benchmark of real-world silent errors from popular serving frameworks, where Ekka shows 80% pass@1 diagnosis accuracy and 88% pass@5 diagnosis accuracy, outperforming state-of-the-art systems. Ekka also diagnoses 4 new silent errors from serving frameworks, all of which have been confirmed by the developers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04592v1">Synthetic Personalities: How Well Can LLMs Mimic Individual Respondents Using Socio-Economic Microdata?</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      LLM-based digital twins promise to scale and accelerate market research, but most published twins are either coarse persona bots conditioned on a few demographic questions or detailed individual-level twins built on purpose-collected surveys and interview transcripts. Neither setup speaks to the operationally most relevant case for marketing practice: building detailed individual twins from the pre-existing heterogeneous panel data that firms already accumulate through CRM systems, loyalty programs, and repeat surveys. We construct detailed individual-level twins from the German Socio-Economic Panel (SOEP) and evaluate them across a $3 \times 5 \times 2 \times 2$ construction-method grid that covers three open-weights LLMs, five cumulative information depths ranked by normalized Shannon entropy, two embedding methods, and two reasoning modes, scoring over 2.1 million twin responses on 500 participants and 183 held-out questions. Twin quality rises with information depth but with diminishing returns past the 75 percent entropy quartile, which acts as a cost-efficient Pareto point relative to the best-performing 100 percent cells. Switching the embedding from a narrative persona summary to a raw dialog history of past responses raises hold-out accuracy in every model-by-reasoning cell at the 100 percent depth, while an explicit thinking mode raises rank-order correlation without moving accuracy. Best-cell accuracy reaches 78.8 percent and Fisher-$z$ correlation reaches $r = 0.590$ on the SOEP held-out evaluation set. The findings suggest that twin-based market research is no longer gated by data design, but by item volume, model selection, and a small set of construction-level decisions that this paper now maps.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2407.03884v4">ChatSOP: An SOP-Guided MCTS Planning Framework for Controllable LLM Dialogue Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 Accepted to ACL 2025 main
    </div>
    <details class="paper-abstract">
      Dialogue agents powered by Large Language Models (LLMs) show superior performance in various tasks. Despite the better user understanding and human-like responses, their **lack of controllability** remains a key challenge, often leading to unfocused conversations or task failure. To address this, we introduce Standard Operating Procedure (SOP) to regulate dialogue flow. Specifically, we propose **ChatSOP**, a novel SOP-guided Monte Carlo Tree Search (MCTS) planning framework designed to enhance the controllability of LLM-driven dialogue agents. To enable this, we curate a dataset comprising SOP-annotated multi-scenario dialogues, generated using a semi-automated role-playing system with GPT-4o and validated through strict manual quality control. Additionally, we propose a novel method that integrates Chain of Thought reasoning with supervised fine-tuning for SOP prediction and utilizes SOP-guided Monte Carlo Tree Search for optimal action planning during dialogues. Experimental results demonstrate the effectiveness of our method, such as achieving a 27.95% improvement in action accuracy compared to baseline models based on GPT-3.5 and also showing notable gains for open-source models. Dataset and codes are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04547v1">Beyond Retrieval: Learning Compact User Representations for Scalable LLM Personalization</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 16 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Personalizing large language models requires adapting model behavior to individual users while preserving robustness and deployment-scale efficiency. Existing approaches typically personalize LLMs either at the input level, by retrieving user histories or constructing profile prompts, or at the parameter level, by maintaining user-specific parameter-efficient modules. The former makes personalization sensitive to retrieval quality and prompt design, whereas the latter incurs storage and maintenance costs that grow with the user population. To address these limitations, we propose TAP-PER (Temporal Attentive Prefix for PERsonalization), a prefix-based framework that encodes user preferences as learnable representations, eliminating explicit prompt construction and replacing heavy per-user adapters with lightweight user-state prefix embeddings. Inspired by personalized recommendation systems, TAP-PER decomposes user modeling into user-state and query-conditioned components, and incorporates temporal signals to capture the evolving nature of user interests. Experiments on six LaMP tasks show that TAP-PER consistently outperforms prompt-based and model-based baselines across classification, rating, and generation settings. Moreover, TAP-PER uses 130x fewer per-user parameters than OPPU and roughly half the total parameter footprint of PER-PCS at the 1,000-user scale, demonstrating that scalable LLM personalization can be achieved without explicit prompt construction or heavy per-user adapters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04514v1">SAILRec: Steering LLM Attention to Dual-Side Semantically Aligned Collaborative Embeddings for Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 17 pages, including appendices
    </div>
    <details class="paper-abstract">
      Recent LLM-based recommenders enhance language models with collaborative embeddings from user-item interactions, but making such embeddings available does not ensure their proper use during inference. Through a diagnostic attention analysis, we find that the utilization of collaborative embeddings is depth-dependent and alignment-sensitive, suggesting that LLMs need to balance their internal semantic knowledge with external collaborative knowledge. To address this issue, we propose SAILRec, an LLM-based recommender that improves this balance through dual-side semantic alignment and hierarchical attention steering. The former aligns item-side embeddings with item-text semantics and user-side embeddings with codebook-based semantic profiles, while the latter suppresses premature shallow-layer collaborative interference and strengthens collaborative evidence in deeper decision layers. Experiments on MovieLens-1M and Amazon-Book show that SAILRec consistently outperforms representative baselines, with ablation and masking analyses validating its key designs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04511v1">SparDA: Sparse Decoupled Attention for Efficient Long-Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Sparse attention reduces compute and memory bandwidth for long-context LLM inference. However, two key challenges remain: (1) KV cache capacity still grows with sequence length, and offloading to CPU memory introduces a PCIe transfer bottleneck; (2) the sparse selection step itself retains $O(T^2)$ complexity and can dominate attention cost at long contexts. We propose SparDA, a decoupled sparse attention architecture that introduces a fourth per-layer projection, the Forecast, alongside Query, Key, and Value. The Forecast predicts the KV blocks needed by the next layer, enabling lookahead selection that overlaps CPU-to-GPU prefetch with current-layer execution. Because Forecast is decoupled from the attention query, our GQA implementation uses one Forecast head per GQA group, reducing selection overhead versus the original multi-head selector. SparDA adds $<$0.5% parameters and trains only the Forecast projections by matching the original selector's attention distribution. On two sparse-pretrained 8B models, SparDA matches or slightly improves accuracy and delivers up to 1.25$\times$ prefill speedup and 1.7$\times$ decode speedup over the sparse-attention offload baseline. By enabling larger feasible batch sizes on a single GPU, SparDA further reaches up to 5.3$\times$ higher decode throughput than the non-offload sparse baseline. Our source code is available at https://github.com/NVlabs/SparDA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04505v1">Simulate, Reason, Decide: Scientific Reasoning with LLMs for Simulation-Driven Decision Making</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Scientific simulators are increasingly being integrated into LLM-driven systems for high-stakes simulation-driven decision-making. However, existing frameworks primarily use LLMs to generate, calibrate, or execute simulators, treating them as black-box interfaces rather than as structured mechanistic systems that can be reasoned about. As a result, current approaches lack the ability to identify, represent, and reason about the assumptions and mechanisms underlying simulator behavior, limiting transparency, auditability, and decision justification. We introduce MechSim, a mechanism-grounded neuro-symbolic reasoning framework for executable scientific simulators. Unlike prior neuro-symbolic approaches that primarily reason over static symbolic structures, MechSim enables LLM agents to reason about the mechanisms, assumptions, and execution behavior of scientific simulators. Our framework represents simulators through a shared structured schema capturing assumptions, variables, mechanism dependencies, and execution traces. On top of this representation, LLM agents operate as constrained reasoning engines that generate structured, evidence-grounded explanations linking simulator outcomes to their underlying mechanisms. We evaluate our approach across multiple high-stakes domains and show that it improves mechanism-level explanation quality, simulator analysis, and downstream decision-making reliability.
    </details>
</div>
