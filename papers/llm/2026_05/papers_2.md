# llm - 2026_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05628v1">Towards Compute-Aware In-Switch Computing for LLMs Tensor-Parallelism on Multi-GPU Systems</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 15 pages, 18 figures, HPCA 2026
    </div>
    <details class="paper-abstract">
      Tensor parallelism (TP) in large-scale LLM inference and training introduces frequent collective operations that dominate inter-GPU communication. While in-switch computing, exemplified by NVLink SHARP (NVLS), accelerates collective operations by reducing redundant data transfer, its communication-centric design philosophy introduces the mismatch between its communication mode and the memory semantic requirement of LLM's computation kernel. Such a mismatch isolates the compute and communication phases, resulting in underutilized resources and limited overlap in multi-GPU systems. To address the limitation, we propose CAIS, the first Compute-Aware In-Switch computing framework that aligns communication modes with computation's memory semantics requirement. CAIS consists of three integral techniques: (1) compute-aware ISA and microarchitecture extension to enable compute-aware in-switch computing. (2) merging-aware TB (Thread Block) coordination to improve the temporal alignment for efficient request merging. (3) graph-level dataflow optimizer to achieve a tight cross-kernel overlap. Evaluations on LLM workloads show that CAIS achieves 1.38$\times$ average end-to-end training speedup over the SOTA NVLS-enabled solution, and 1.61$\times$ over T3, the SOTA compute-communicate overlap solutions but do not leverage NVLS, demonstrating its effectiveness in accelerating TP on multi-GPU systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.10241v2">ImCoref-CeS: An Improved Lightweight Pipeline for Coreference Resolution with LLM-based Checker-Splitter Refinement</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 Accepted by ACL2026 main
    </div>
    <details class="paper-abstract">
      Coreference Resolution (CR) is a critical task in Natural Language Processing (NLP). Current research faces a key dilemma: whether to further explore the potential of supervised neural methods based on small language models, whose detect-then-cluster pipeline still delivers top performance, or embrace the powerful capabilities of Large Language Models (LLMs). However, effectively combining their strengths remains underexplored. To this end, we propose \textbf{ImCoref-CeS}, a novel framework that integrates an enhanced supervised model with LLM-based reasoning. First, we present an improved CR method (\textbf{ImCoref}) to push the performance boundaries of the supervised neural method by introducing a lightweight bridging module to enhance long-text encoding capability, devising a biaffine scorer to comprehensively capture positional information, and invoking a hybrid mention regularization to improve training efficiency. Importantly, we employ an LLM acting as a multi-role Checker-Splitter agent to validate candidate mentions (filtering out invalid ones) and coreference results (splitting erroneous clusters) predicted by ImCoref. Extensive experiments demonstrate the effectiveness of ImCoref-CeS, which achieves superior performance compared to existing state-of-the-art (SOTA) methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21714v3">E-mem: Multi-agent based Episodic Context Reconstruction for LLM Agent Memory</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 This paper has been accepted by ICML 2026. If you find our project helpful, please consider giving it a star: https://github.com/dog-last/E-mem
    </div>
    <details class="paper-abstract">
      The evolution of Large Language Model (LLM) agents towards System~2 reasoning, characterized by deliberative, high-precision problem-solving, requires maintaining rigorous logical integrity over extended horizons. However, prevalent memory preprocessing paradigms suffer from destructive de-contextualization. By compressing complex sequential dependencies into pre-defined structures (e.g., embeddings or graphs), these methods sever the contextual integrity essential for deep reasoning. To address this, we propose E-mem, a framework shifting from Memory Preprocessing to Episodic Context Reconstruction. Inspired by biological engrams, E-mem employs a heterogeneous hierarchical architecture where multiple assistant agents maintain uncompressed memory contexts, while a central master agent orchestrates global planning. Unlike passive retrieval, our mechanism empowers assistants to locally reason within activated segments, extracting context-aware evidence before aggregation. Evaluations on the LoCoMo benchmark demonstrate that E-mem achieves over 54\% F1, surpassing the state-of-the-art GAM by 7.75\%, while reducing token cost by over 70\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20375v2">LLM-AutoDP: Automatic Data Processing via LLM Agents for Model Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 Accepted by VLDB2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can be fine-tuned on domain-specific data to enhance their performance in specialized fields. However, such data often contains numerous low-quality samples, necessitating effective data processing (DP). In practice, DP strategies are typically developed through iterative manual analysis and trial-and-error adjustment. These processes inevitably incur high labor costs and may lead to privacy issues in high-privacy domains like healthcare due to direct human access to sensitive data. Thus, achieving automated data processing without exposing the raw data has become a critical challenge. To address this challenge, we propose LLM-AutoDP, a novel framework that leverages LLMs as agents to automatically generate and optimize data processing strategies. Our method generates multiple candidate strategies and iteratively refines them using feedback signals and comparative evaluations. This iterative in-context learning mechanism enables the agent to converge toward high-quality processing pipelines without requiring direct human intervention or access to the underlying data. To further accelerate strategy search, we introduce three key techniques: Distribution Preserving Sampling, which reduces data volume while maintaining distributional integrity; Processing Target Selection, which uses a binary classifier to identify low-quality samples for focused processing; Cache-and-Reuse Mechanism}, which minimizes redundant computations by reusing prior processing results. Results show that models trained on data processed by our framework achieve over 80% win rates against models trained on unprocessed data. Compared to AutoML baselines based on LLM agents, LLM-AutoDP achieves approximately a 65% win rate. Moreover, our acceleration techniques reduce the total searching time by up to 10 times, demonstrating both effectiveness and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05598v1">Prober.ai: Gated Inquiry-Based Feedback via LLM-Constrained Personas for Argumentative Writing Development</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 Prototype awarded second place at the NYEdTech Hackathon (March 2026) https://www.nyedtechhackathon.com/2026-submissions
    </div>
    <details class="paper-abstract">
      The proliferation of large language models (LLMs) in educational settings has paradoxically undermined the cognitive processes they purport to support. Students increasingly outsource critical thinking to AI assistants that generate polished text on demand, resulting in measurable cognitive debt and diminished argumentative reasoning skills. We present Prober.ai, a web-based writing environment that inverts the conventional AI-tutoring paradigm: rather than generating or rewriting student text, the system constrains an LLM (Gemini 3 Flash Preview) through persona-specific system prompts and structured JSON output schemas to produce only targeted, inquiry-based questions about argumentative weaknesses. A two-phase interaction architecture -- Challenge and Unlock -- implements a pedagogical friction mechanism whereby revision suggestions are gated behind mandatory student reflection. The system's design is grounded in Toulmin's argumentation theory, research on peer feedforward questioning mechanisms, and evidence on AI-supported feedback in writing instruction. A functional prototype was developed in 36 hours during the NY EdTech Hackathon (March 2026), where it was awarded second place. We describe the system architecture, the prompt engineering methodology for constraining LLM output to pedagogically aligned JSON schemas, and discuss implications for scalable, cognition-preserving AI integration in writing education.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.23629v3">Emergent Slow Thinking in LLMs as Inverse Tree Freezing</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 34 pages, 17 figures, 1 table
    </div>
    <details class="paper-abstract">
      Reinforcement learning with verifiable rewards (RLVR) enables large language models to acquire slow, multi-step reasoning from sparse final-answer signals. We provide a statistical-physics picture of this emergence. We show that an autoregressive model's finite capacity forces it to compress its exponentially large prefix space into a Markov network of predictive states, on which slow thinking unfolds as a random walk -- the Concept Network (CoNet) picture. Within CoNet, RLVR dynamics are governed by two mechanisms: merging of compatible paths and frustrated competition among incompatible ones. Together they drive the network through nucleation, growth, and freezing into multi-input, single-output directed inverse trees. The picture reproduces the training dynamics of a 1.5-billion-parameter LLM and yields three predictions: reasoning chains lengthen as a geometric necessity of sparse topology; SFT induces catastrophic forgetting through bridge-node rupture; and frustration drives policy collapse. Building on the structural timing inherent in inverse-tree freezing, we propose Annealed-RLVR -- a brief SFT intervention at the moment of maximum frustration. It outperforms standard RLVR on both in- and out-of-distribution benchmarks, with the largest gains at high sampling budgets where standard RLVR collapses. The same SFT applied after the trees freeze instead triggers catastrophic forgetting, isolating timing as the active ingredient.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.03379v2">Two Calls, Two Moments, and the Vote-Accuracy Curve of Repeated LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Repeated sampling is a standard way to spend test-time compute, but its benefit is controlled by the latent distribution of correctness across examples, not by one-call accuracy alone. We study the binary correctness layer of repeated LLM inference under conditional-i.i.d. calls. One labeled call identifies the mean latent success probability; two labeled calls identify its second moment and hence the same-example correctness correlation that separates stable errors from recoverable call-level randomness. From these two moments, every fixed majority-vote budget has a sharp distribution-free two-call interval. The key technical reduction is that the infinite-dimensional moment problem has three-atom extremizers and quadratic dual certificates for every finite budget, so the bounds are exact rather than discretized or parametric. The first useful budget, three votes, has a closed form, width at most $1/8$, and a certified-improvement criterion. The infinite-vote endpoint is the limit of majority voting as the number of calls tends to infinity; it is also sharply bounded, but remains threshold-sensitive because it depends on latent mass around $q=1/2$. We add maximum-entropy and Latent-difficulty Gaussian-probit point completions, and experiments on LLM calls over QNLI and QQP show that empirical three- and five-vote accuracies are contained in the projected two-call regions while temperature changes and randomized model mixtures can create voting gains not ordered by one-call accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.01120v2">New Bounds for Zarankiewicz Numbers via Reinforced LLM Evolutionary Search</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 *Jay Bhan and Nicole Nobili contributed equally to this work as first authors, and their order was determined via coin flip
    </div>
    <details class="paper-abstract">
      The Zarankiewicz number $\textbf{Z}(m, n, s, t)$ is the maximum number of edges in a bipartite graph $G_{m, n}$ such that there is no complete $K_{s, t}$ bipartite subgraph. We determine for the first time the exact values of three Zarankiewicz numbers: $\textbf{Z}(11, 21, 3, 3)=116$, $\textbf{Z}(11, 22, 3, 3)=121$, and $\textbf{Z}(12, 22, 3, 3)=132$. We further establish lower bounds for 41 more Zarankiewicz numbers, including several that are within one edge of the best known upper bound, and we match the established value in four more closed cases. Our results are obtained using OpenEvolve, an open-source evolutionary algorithm based on Large Language Models (LLMs) that iteratively improves algorithms for generating mathematical constructions by optimizing a reward signal which we tailored for this specific problem. These findings provide new extremal graph constructions and demonstrate the potential of LLM-guided evolutionary search to contribute to mathematical research. In addition to presenting the resulting constructions, we report the generation algorithms produced, describe the relevant implementation details, and provide our computational costs. Our costs are remarkably low, at less than \$30 for each Zarankiewicz parameter combination, showing that LLM-guided evolutionary search can be an inexpensive, reproducible, and accessible tool for discovering new combinatorial constructions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05573v1">AstroAlertBench: Evaluating the Accuracy, Reasoning, and Honesty of Multimodal LLMs in Astronomical Classification</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Modern astronomical observatories generate a massive volume of multimodal data, creating a critical bottleneck for expert human review. While multimodal large language models (LLMs) have shown promise in interpreting complex visual and textual inputs, their ability to perform specialized scientific classification while providing interpretable reasoning remains understudied. We introduce AstroAlertBench, a comprehensive multimodal benchmark designed to evaluate LLM performance in astronomical event review along a three-stage logical chain: metadata grounding, scientific reasoning, and hierarchical classification over five categories. We use a pilot sample of 1,500 real-world alerts from the Zwicky Transient Facility (ZTF), a wide-field survey that scans the northern sky to detect transient astronomical events. On this dataset, we benchmark 13 frontier closed-source and open-weight LLMs that support visual input. Our results reveal that high accuracy does not always align with model ``honesty,'' defined as the ability to self-evaluate its reasoning, which affects its reliability as a real-world assistant. We further initialize a human-in-the-loop evaluation protocol as a precursor to future community-scale participation. Together, AstroAlertBench provides a framework for developing calibrated and interpretable astronomical assistants.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05501v1">SOCpilot: Verifying Policy Compliance for LLM-Assisted Incident Response</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      Security operations centers (SOCs) are beginning to use large language models (LLMs) as copilots to draft incident-response plans. These plans may include actions that are valid per the catalog but still violate mandatory steps, required ordering, or approval gates before analyst review. SOCpilot makes this compliance question measurable at the plan boundary. It fixes the incident package, action catalog, policy rules, verifier, and public evidence surface. Next, it verifies the copilot's proposed action trace. We evaluate two LLM providers on 200 real incidents from an anonymized production SOC in a financial-sector case study. We compare their plans to paired analyst-authored references from the same security orchestration, automation, and response (SOAR) cases. An identical inline policy text moves the two providers in opposite directions. A deterministic verifier removes 466 non-compliant, approval-gated actions, without reducing baseline-task recall. Aggregate rates remain stable across 3 reruns of the fixed corpus. The official evidence focuses on approval-gated decisions regarding recovery and containment. Separately, the artifact exposes zero-cost readiness checks for mandatory and ordering repairs. We release the runnable artifact so independent reviewers can rederive the public results without access to private incident data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05485v1">ReaComp: Compiling LLM Reasoning into Symbolic Solvers for Efficient Program Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      LLMs can solve program synthesis tasks but remain inefficient and unreliable on hard instances requiring large combinatorial search. Given a small set of reasoning traces, we use coding agents to compile them into reusable symbolic program synthesizers over constrained DSLs. The resulting solvers require no LLM calls at test time and are strong standalone systems: symbolic solver ensembles reach 91.3% accuracy on PBEBench-Lite and 84.7% on PBEBench-Hard, outperforming LLMs with test-time scaling for the latter by +16.3 percentage points at zero LLM inference cost. They also complement LLM search, improving PBEBench-Hard accuracy from 68.4% to 85.8% while reducing reported token usage by 78%, and raising SLR-Bench hard-tier accuracy from 34.4% to 58.0% in a neuro-symbolic hybrid setting. Compared to directly using coding agents as per-instance solvers, induced solvers are substantially more Pareto-efficient, amortizing a small one-time construction cost over many zero-token executions. Finally, most solvers transfer zero-shot to a real historical linguistics task - predicting sound changes in natural language data - reaching 80.1% accuracy under ensembling and recovering some plausible linguistic rules. Together, these results show that reasoning traces can be compiled into reusable symbolic solvers that solve many tasks directly, complement LLM inference on hard cases, and provide a scalable route to domain-general solver induction. We release code and data for reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05467v1">Nitsum: Serving Tiered LLM Requests with Adaptive Tensor Parallelism</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      LLM serving is increasingly multi-tenant: the same deployment must handle latency-critical interactive requests and more relaxed background workloads under a fixed GPU budget. This creates a tiered-SLO setting where maximizing overall goodput (requests that satisfy both TTFT and TPOT targets) is challenging because workload mix, request lengths, and load intensity vary over time. Existing systems mainly optimize request-level controls (e.g., queuing and batching) while keeping execution configuration largely static, which limits adaptation under multi-tier contention. We present Nitsum, a distributed LLM serving system that treats tensor parallelism (TP) as a first-class runtime control surface rather than a static deployment choice. Nitsum jointly optimizes TP level, prefill/decode GPU split, and request scheduling. To make frequent TP adaptation practical, Nitsum introduces TP-aware weight reuse and fast KV migration. Experiments on real traces and targeted microbenchmarks show that Nitsum improves SLO-compliant goodput over SoTA by up to 5.3 times.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05427v1">The Geopolitics of AI Safety: A Causal Analysis of Regional LLM Bias</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are integrated into global software systems, ensuring equitable safety guardrails is a critical requirement. Current fairness evaluations predominantly measure bias observationally, a methodology confounded by the inherent toxicity of topics naturally paired with specific demographics in testing datasets. This study introduces a Probabilistic Graphical Model (PGM) framework to audit LLM safety mechanisms causally. By applying Pearl's do-operator, we mathematically isolate the causal effect of injecting a cultural demographic into a prompt. We conduct a large-scale empirical analysis across seven instruction-tuned models spanning diverse origins: the United States (Llama-3.1-8B, Gemma-2-9B), Europe (Mistral-7B-v0.3), the UAE (Falcon3-7B), China (Qwen2.5-7B, DeepSeek-7B), and India (Airavata-7B). Utilizing two distinct datasets (ToxiGen and BOLD), the findings reveal a disparity between observational and interventional bias, demonstrating that standard fairness metrics can overestimate demographic bias by failing to account for context toxicity. Furthermore, the causal probabilities indicate distinct alignment trends: Western models exhibit higher causal refusal rates for specific demographic groups, whereas Eastern models demonstrate low overall intervention rates with targeted sensitivities toward regional demographics. We discuss the implications of these biases, highlighting how demographic-sensitive over-triggering restricts benign discourse in downstream applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05413v1">From History to State: Constant-Context Skill Learning for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents are increasingly used to operate browsers, files, code and tools, making personal assistants a natural deployment target. Yet personal agents face a privacy-cost-capability tension: cloud models execute multi-step workflows well but expose sensitive intermediate context to external APIs, while local models preserve privacy but remain less reliable. Both settings also pay repeatedly for long skill prompts and growing histories. We propose constant-context skill learning, a context-to-weights framework for recurring agent workflows: reusable procedures are learned in lightweight task-family modules, while inference conditions only on the current observation and a compact state block. A deterministic tracker renders this state block from task progress and supplies aligned subgoal rewards, so each module can be trained with step-level SFT and refined through online RL. Across ALFWorld, WebShop, and SciWorld, our agents achieve strong performance across Qwen3-4B, Qwen3-8B and Llama-3.1-8B. With Qwen3-8B, SFT+RL reaches 89.6\% unseen success on ALFWorld, 76.8\% success on WebShop, and 66.4\% unseen success on SciWorld. They match or exceed strong published agent-training results while reducing prompt tokens per turn by 2--7$\times$ relative to controlled ReAct prompting baselines, showing that procedural context can be moved from prompts into weights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05410v1">LaTA: A Drop-in, FERPA-Compliant Local-LLM Autograder for Upper-Division STEM Coursework</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 Submitted to Computers & Education
    </div>
    <details class="paper-abstract">
      Large-language-model (LLM) graders promise to relieve the grading burden of upper-division STEM courses, but most deployments to date send student work to third-party APIs, violating FERPA and exposing institutions to data risk while requiring substantial assignment modification. We present $\textbf{LaTA}\ (\textit{LaTeX Teaching Assistant})$, a drop-in, open-source autograder that runs entirely on commodity on-premises hardware and assumes a LaTeX-native workflow already adopted by many engineering and physics courses. LaTA implements a four-stage pipeline (ingest, segment, grade, report) using a locally hosted open-weight chain-of-thought LLM grader (gpt-oss:120b) that compares student work to an instructor-authored reference solution and applies a YAML rubric with binary per-item scoring. We deployed LaTA in Winter~2026 in ME 373 (Mechanical Engineering Methods) at Oregon State University, grading every weekly assignment for approximately 200 students on a single Mac Studio at \$0 marginal cost per assignment and 1--3 minutes of wall-clock time per submission, enabling regrading of corrected assignments and greatly expanded TA office hour offerings. The instructor-confirmed grading-error rate held at roughly $0.02$--$0.04\%$ per rubric line item across the term. Relative to the same instructor's previous traditionally-graded cohort, the LaTA-graded cohort outperformed by approximately $11\%$ on the midterm exam and $8\%$ on the final exam, and reported large gains in self-assessed confidence on every stated learning objective ($N = 159$ survey responses, $Δ\geq +1.49$ Likert points, $p < 10^{-27}$ on every comparison). We release the code under AGPLv3.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05401v1">Why Someone Asked "Why": Foil Inference in Human and LLM Question Interpretation</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 Accepted at Proceedings of the 48th Annual Conference of the Cognitive Science Society (CogSci 2026)
    </div>
    <details class="paper-abstract">
      Explanations are inherently contrastive: E happened rather than E' because of C rather than C'. However, these contrasts, or "foils", are rarely mentioned explicitly but have to be inferred in context. Here, we investigate how people select the intended foil E' of a why-question. Participants read vignettes and judged, for each foil, their prior expectation (what will happen next), closeness (what is most similar to what happened), and hindsight expectation (what could have happened instead), as well as which foil they thought the question asker had in mind when they asked the why-question. We found that foil selections were best predicted by hindsight expectation judgments. This suggests that people infer the foil by considering what a question asker finds surprising after the outcome occurred. Since correct foil selection is relevant not only in human-human interaction but also increasingly in dialogues with large language models, we investigated their performance on the same task. The coupling between LLMs' explicit expectation judgments and their foil selections is inconsistent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05391v1">Every(bot) Makes Mistakes: Coding Big Five Personalities, Context, and Tone into an LLM Chatbot Recovery Code Framework</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 14 pages of main content, 3 figures, 4 tables, 9 appendices. This paper has been submitted to the Becker Friedman Institute 2026 AI in Social Sciences conference for peer review
    </div>
    <details class="paper-abstract">
      Despite careful design involving classifiers, parameters, and safeguarding, errors during human/AI interaction are not rare. Poor error recovery can disrupt interaction flow, damage user trust, and decrease user engagement. Whilst existing work has explored LLM recovery, tone, context, and personality as separate design dimensions, no existing work has combined these variables into a structured guidance framework. This paper presents a recovery code that maps four common LLM chatbot task contexts to associated personality traits (four Big Five personalities: Conscientiousness, Agreeableness, Openness, and Extraversion), tones, and three-stage recovery instructions. A recovery evaluation rubric was also designed, comprising three dimensions (Recovery quality, Tone alignment, and Appropriateness) and nine sub-dimensions. The methodology is exploratory, with no participants used. A between-subjects design was employed across two conditions: Condition A (baseline, uncoded), four separate Claude Sonnet 4.6 agents received no recovery code training; Condition B (coded), four separate Claude Sonnet 4.6 models were trained on the recovery code. Identical 'user' prompts and error scenarios were used across both conditions. Eight LLM evaluator agents assessed the recovery responses using the evaluation rubric, producing scores out of 5 for each sub-dimension. Results found a 27.8% average performance increase in coded recovery responses (76.7%) compared to baseline responses (48.9%). Condition B performed strongest in the appropriateness dimension (83.3%), with notable improvement in personality appropriateness (75% versus 50%) and providing explanation (60% versus 20%). These findings suggest that structured personality, context, and tone-informed recovery codes can be successfully learnt and applied by LLM chatbots to improve error recovery quality across varying contextual tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05344v1">Open-SAT: LLM-Guided Query Embedding Refinement for Open-Vocabulary Object Retrieval in Satellite Imagery</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      In satellite applications, user queries often take the form of open-ended natural language, extending beyond a fixed set of predefined categories. This open-vocabulary nature poses significant challenges for retrieving relevant image tiles, as the retrieval system must generalize to a wide range of unseen objects and concepts. While vision-language models (VLMs) such as CLIP are widely used for text-image retrieval, even fine-tuned variants often struggle to accurately align such queries with satellite imagery. To address this, we propose Open-SAT, a training-free query embedding refinement algorithm that operates at inference time to improve alignment between user queries and satellite image content. Open-SAT uses VLMs to compute embeddings for image tiles, which are stored in a vector database for efficient retrieval. At query time, it leverages Large Language Models (LLMs) to refine the text embeddings by incorporating contextual information about objects of interest and their surroundings. A threshold-free retrieval mechanism further enhances accuracy and efficiency. Experimental results in three public benchmarks demonstrate that Open-SAT improves the F1 score by up to 16.04%, while retrieving a comparable number of image tiles. These results demonstrate the effectiveness of Open-SAT in open-vocabulary satellite image retrieval, leveraging LLM guidance without the need for additional training or supervision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.15155v2">eLLM: Elastic Memory Management Framework for Efficient LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 7pages, accepted to DAC'2026
    </div>
    <details class="paper-abstract">
      Large Language Models are increasingly being deployed in datacenters. Serving these models requires careful memory management, as their memory usage includes static weights, dynamic activations, and key-value caches. While static weights are constant and predictable, dynamic components such as activations and KV caches change frequently during runtime, presenting significant challenges for efficient memory management. Modern LLM serving systems typically handle runtime memory and KV caches at distinct abstraction levels: runtime memory management relies on static tensor abstractions, whereas KV caches utilize a page table-based virtualization layer built on top of the tensor abstraction. This virtualization dynamically manages KV caches to mitigate memory fragmentation. However, this dual-level approach fundamentally isolates runtime memory and KV cache management, resulting in suboptimal memory utilization under dynamic workloads, which can lead to a nearly 20% drop in throughput. To address these limitations, we propose eLLM, an elastic memory management framework inspired by the classical memory ballooning mechanism in operating systems. The core components of eLLM include: (1) Virtual Tensor Abstraction, which decouples the virtual address space of tensors from the physical GPU memory, creating a unified and flexible memory pool; (2) an Elastic Memory Mechanism that dynamically adjusts memory allocation through runtime memory inflation and deflation, leveraging CPU memory as an extensible buffer; and (3) a Lightweight Scheduling Strategy employing SLO-aware policies to optimize memory utilization and effectively balance performance trade-offs under stringent SLO constraints. Comprehensive evaluations demonstrate that eLLM significantly outperforms state-of-the-art systems, 2.32x higher decoding throughput, and supporting 3x larger batch sizes for 128K-token inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05175v1">MRI-Eval: A Tiered Benchmark for Evaluating LLM Performance on MRI Physics and GE Scanner Operations Knowledge</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 21 pages, 4 figures, 10 tables
    </div>
    <details class="paper-abstract">
      Background: Existing MRI LLM benchmarks rely mainly on review-book multiple-choice questions, where top proprietary models already score highly, limiting discrimination. No systematic benchmark has evaluated vendor-specific scanner operational knowledge central to research MRI practice. Purpose: We developed MRI-Eval, a tiered benchmark for relative model comparison on MRI physics and GE scanner operations knowledge using primary multiple-choice questions (MCQ), with stem-only and primed diagnostic conditions as complementary analyses. Methods: MRI-Eval includes 1365 scored items across nine categories and three difficulty tiers from textbooks, GE scanner manuals, programming course materials, and expert-generated questions. Five model families were evaluated (GPT-5.4, Claude Opus 4.6, Claude Sonnet 4.6, Gemini 2.5 Pro, Llama 3.3 70B). MCQ was primary; stem-only removed options and used an independent LLM judge; primed stem-only tested responses to incorrect user claims. Results: Overall MCQ accuracy was 93.2% to 97.1%. GE scanner operations was the lowest category for every model (88.2% to 94.6%). In stem-only, frontier-model accuracy fell to 58.4% to 61.1%, and Llama 3.3 70B fell to 37.1%; GE scanner operations stem-only accuracy was 13.8% to 29.8%. Conclusion: High MCQ performance can mask weak free-text recall, especially for vendor-specific operational knowledge. MRI-Eval is most informative as a relative comparison benchmark rather than an absolute competency measure and supports caution in using raw LLM outputs for GE-specific protocol guidance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.16804v2">AutoOR: Scalably Post-training LLMs to Autoformalize Operations Research Problems</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      Optimization problems are central to decision-making in manufacturing, logistics, scheduling, and other industrial settings. Translating complicated descriptions of these problems into solver-ready formulations requires specialized operations research (OR) expertise, making it hard to scale. We present AutoOR, a scalable synthetic data generation and reinforcement learning pipeline that trains LLMs to autoformalize optimization problems specified in natural language across linear, mixed-integer, and non-linear categories. AutoOR generates verified training data from standard optimization forms and uses solver execution feedback as the reward signal for RL post-training. AutoOR applied to an 8B model achieves state-of-the-art or competitive results across six established OR benchmarks, matching significantly larger frontier models. For a non-linear problem class involving physical dynamics, where frontier models score near 0%, we introduce a curriculum RL strategy that bootstraps from limited initial training data to make this class tractable for post-training. We believe that methods such as AutoOR can significantly accelerate industrial decision-making with AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.23338v2">A Systematic Survey of Security Threats and Defenses in LLM-Based AI Agents: A Layered Attack Surface Framework</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 58 pages, 8 figures, 15 tables
    </div>
    <details class="paper-abstract">
      Agentic AI systems introduce a security surface that is qualitatively different from that of stateless LLMs. They persist memory, invoke external tools, coordinate with peer agents, and operate across sessions, allowing attacks to emerge not only at the prompt interface but also through architectural state, delegated authority, and long-horizon interactions. Existing security taxonomies, however, primarily organize threats by attack type, such as prompt injection or jailbreaking, and therefore obscure where in the agentic stack a threat arises and over what timescale it manifests. We propose the Layered Attack Surface Model (\lasm), a structural taxonomy for agentic AI security. \lasm decomposes the agentic stack into seven layers -- Foundation, Cognitive, Memory, Tool Execution, Multi-Agent Coordination, Ecosystem, and Governance -- and augments them with a four-class temporality axis covering instantaneous, session-persistent, cross-session cumulative, and sub-session-stack threats. We use this 7$\times$4 framework to analyze 116 papers from 2021--2026. The resulting map reveals that the upper layers of the agentic stack remain sharply under-explored, especially for long-horizon and stack-propagating threats; multiple documented attack regions have no corresponding defenses; and current benchmarks provide no coverage for cross-session or sub-session-stack failure modes. We further derive a cross-layer defense taxonomy, defense recipes for canonical attack classes, and a dependency DAG that separates near-term engineering gaps from fundamental research challenges. We release the per-paper coding, robustness scripts, and a reference Agent Bill of Materials schema to support reproducible analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05134v1">Low-Cost Black-Box Detection of LLM Hallucinations via Dynamical System Prediction</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) frequently generate plausible but non-factual content, a phenomenon known as hallucination. While existing detection methods typically rely on computationally expensive sampling-based consistency checks or external knowledge retrieval, we propose a new method that treats the LLM as a black-box dynamical system. By projecting LLM responses into a high-dimensional manifold via an embedding model, we characterize the resulting vector sequences as observable realizations of the model's latent state-space dynamics. Leveraging Koopman operator theory, we fit the transition operators for both factual and hallucinated regimes and define a differential residual score based on their respective prediction errors. To accommodate varying user requirements and domain-specific sensitivities, we introduce a preference-aware calibration mechanism that optimizes the classification threshold based on a small set of demonstrations. This approach enables low-cost hallucination detection in a single-sample pass, avoiding the need for secondary sampling or external grounding. Extensive testing across three data benchmarks demonstrates that our method achieves state-of-the-art performance with reduced resource overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05125v1">Joint Treatment Effect Estimation from Incomplete Healthcare Data: Temporal Causal Normalizing Flows with LLM-driven Evolutionary MNAR Imputation</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      Target trial emulation (TTE) enables causal questions to be studied with observational data when randomized controlled trials (RCTs) are infeasible. Yet treatment-effect methods often address causal estimation, missingness, and temporal structure separately, limiting their robustness in electronic health records (EHRs), where time-varying confounding and missing-not-at-random (MNAR) biomarkers can reach 50%--80%. We propose a two-stage pipeline for treatment effect estimation from incomplete longitudinal EHRs. First, CausalFlow-T, a directed acyclic graph (DAG)-constrained normalizing flow with long short-term memory (LSTM)-encoded patient history, performs exact invertible counterfactual inference, avoiding approximation errors from variational inference and separating confounding through explicit causal structure. Ablations on four synthetic and one semi-synthetic benchmark with known counterfactuals show that DAG constraints and exact inference address distinct failure modes: neither compensates for the other. Second, because CausalFlow-T requires completed inputs, we introduce an LLM-driven evolutionary imputer that proposes executable imputation operators rather than individual entries, and evaluate it with three large language model (LLM) backends, including two open-source models. Across 30%--80% MNAR missingness, this imputer achieves the best pooled rank over biomarker and causal metrics, leading in point-wise accuracy and temporal extrapolation while preserving average treatment effect (ATE) recovery as statistical baselines degrade. On Swiss primary-care EHRs from adults with type 2 diabetes initiating a GLP-1 receptor agonist or SGLT-2 inhibitor, the pipeline estimates a per-protocol weight-loss difference of -0.98 kg [95% CI -1.01, -0.96] favoring GLP-1 receptor agonists, consistent with randomized evidence and obtained from realistically incomplete real-world EHRs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05116v1">On the Hardness of Junking LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 27 pages, 13 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are known to be vulnerable to jailbreak attacks, which typically rely on carefully designed prompts containing explicit semantic structure. These attacks generally operate by fixing an adversarial instruction and optimizing small adversarial components (e.g., suffixes or prefixes). In this setting, prompt structure is fundamental for performance, and recent results show that even simple random search can achieve strong performance when combined with sophisticated prompt design. Recently, it has been observed that harmful behaviors can be elicited even without the adversarial prompt, relying solely on optimized token sequences. This suggests the existence of natural backdoors, i.e., token sequences naturally emerged during LLMs training that trigger unsafe outputs without any meaningful instruction. However, despite these observations, this setting remains largely unexplored, and in particular the hardness of finding natural backdoors has not been assessed yet. In this work, we provide a first proof-of-concept study investigating the hardness of this task, which we refer to as the junking problem. We formalize it as the problem of finding token sequences that maximize the probability of generating a target prefix of harmful responses, propose a greedy random-search method to assess is such sequences can be discovered easily. Our results show that this problem is harder than standard jailbreak attacks, confirming the importance of semantic information in prompt design. At the same time, we find that our simple strategy is sufficient to solve it with a high success rate, suggesting that natural backdoors are present and easily recoverable. Finally, through perplexity analysis, we observe that the discovered token sequences lie in low-probability regions of the model distribution, supporting the hypothesis that they emerged implicitly from the training process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05080v1">The Pinocchio Dimension: Phenomenality of Experience as the Primary Axis of LLM Psychometric Differences</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      We administer 45 validated psychometric questionnaires to 50 large language models (LLMs) to identify the dimensions along which LLMs differ psychometrically. Using Supervised Semantic Differential (SSD), we find that the primary axis of between-model variance separates items describing phenomenally rich experience, including embodied sensation, felt affect, inner speech, imagery, and empathy, from items describing stimulus-driven behavioral reactivity ($R^2_{adj}=.037$, $p<.0001$). To test this hypothesis at the item level, we introduce the Pinocchio score ($π_i$), the ratio of inter-model response variance under neutral prompting to that under a human-simulation prompt, as an annotation-free measure of each item's experiential demand. $π_i$ predicts condition-induced shifts in primary factor loading magnitudes ($ρ=-.215$, $p<.0001$, $n=1292$--$1310$ items), confirming that between-model divergence on experiential items is structured rather than noisy. Applying PCA to per-model EFA scores across all questionnaires reveals one dominant dimension, the Pinocchio Axis ($Π$): the degree to which a model presents itself as a locus of phenomenal experience rather than a system of behavioral responses. This axis captures 47.1% of cross-questionnaire between-model variance in primary factor scores and converges with item-level Pinocchio scores ($r=.864$). Marked within-provider divergence across closely related model variants is consistent with post-training fine-tuning as a key contributor, supporting the interpretation that $Π$ reflects a training-shaped self-representational tendency governing how a model treats experiential language as self-applicable. The dominant axis of between-model psychometric variation is therefore not a conventional personality trait but a self-representational stance toward one's own nature as an experiencer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16659v2">LLMs learn scientific taste from institutional traces across the social sciences</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      Reinforcement-learned reasoning has powered recent AI leaps on verifiable tasks, including mathematics, code, and structure prediction. The harder bottleneck is evaluative judgment in low-verifiability domains, where no oracle anchors reward and the core question is which untested ideas deserve attention. We test whether institutional traces, the record of what fields published, where, and at which tier, can serve as a training signal for AI evaluators. Across eight social science disciplines (psychology, economics, communication, sociology, political science, management, business and finance, public administration), we built held-out four-tier research-pitch benchmarks and supervised-fine-tuned (SFT) LLMs on field-specific publication outcomes. The fine-tuned models cleared the 25 percent chance baseline and exceeded frontier-model performance by wide margins, with best single-model accuracy ranging from 55.0 percent in public administration to 85.5 percent in psychology. In management, evaluated against 48 expert gatekeepers, 174 junior researchers, and 11 frontier reasoning models, the best single fine-tuned model (Qwen3-4B) reached 59.2 percent, 17.6 percentage points above expert majority vote (41.6 percent, non-tied) and 28.1 percentage points above the frontier mean (31.1 percent). The fine-tuned models also showed calibrated confidence: confidence rose when predictions were correct and fell when wrong, mirroring how a skilled reviewer can say "I'm sure" versus "I'm guessing." Selective triage on this signal reached very high accuracy on the highest-confidence subsets in every field. Institutional traces, we conclude, encode a scalable training signal for the low-verifiability judgment on which science depends.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.03619v2">The Infinite Mutation Engine? Measuring Polymorphism in LLM-Generated Offensive Code</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      Malware authors have traditionally relied on polymorphic techniques to produce variants in the same malware family, complicating signature-based detection. Integrating generative AI into offensive toolchains enables attackers to synthesize structurally diverse payloads with identical behavior, raising the question of how much polymorphism LLMs provide. Recent work has assumed that LLMs can produce sufficiently polymorphic payloads, leaving unquantified the variation that emerges when an attacker repeatedly builds the same payload, or explicitly instructs the model to avoid prior implementations. In this work, we measure the polymorphic capacity of a commercial model (Claude Opus 4.6) as an automated malware generator. We build a dual-agent, four-stage pipeline that generates, tests, and refines a data-exfiltration payload comprising file traversal, encryption, exfiltration, and integration. We produce payloads in two settings: using prompts that specify only functional requirements, and using prompts that inject a structured history of prior outcomes to force divergence. We measure pairwise distances along structural (AST) and semantic (embedding) axes, finding that when polymorphism is not explicitly required, structural distances are high while semantic distances remain low; i.e., implementations diverge widely without changing high-level behavior. Explicit prompting substantially amplifies this structural diversity while preserving correctness, at the cost of roughly 5 times more tokens but only a small increase in LLM calls (from $4.2$ to $4.5$ per payload, with effective API costs of \$0.41 and \$0.73). These results show that a single commercial LLM can cheaply generate large populations of behaviorally equivalent yet structurally diverse payloads, facilitating the evasion of signature-based detection rules and similarity-based clustering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05277v1">GLiNER Guard: Unified Encoder Family for Production LLM Safety and Privacy</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 Models: https://huggingface.co/collections/hivetrace/gliner-guard-v1 PII-Bench: https://huggingface.co/datasets/hivetrace/pii-bench
    </div>
    <details class="paper-abstract">
      Production LLM systems require both safety moderation and PII detection under strict latency and cost constraints. This creates a trade-off: autoregressive moderators are accurate but expensive, while lightweight encoders are faster but less capable. We present GLiNER Guard (GLiGuard), a unified encoder that performs safety classification and PII detection in a single forward pass, simplifying safety pipelines. We introduce three variants: compact uni- and bi-encoders (145-147M) for high-throughput serving, and GLiGuard Omni (209M) for stronger moderation quality. Under dynamic batching on a single A100, the compact model reaches 193 requests/sec with P99 latency below 1s, achieving 1.6x higher throughput than GLiNER2. Omni remains competitive with much larger moderators on public safety benchmarks. We also release PII-Bench, a span-level benchmark for evaluating PII detection in end-to-end pipelines. Overall, encoder-based guardrails offer a practical low-cost alternative for always-on moderation. Models and benchmarks are released on HuggingFace.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05023v1">CuBridge: An LLM-Based Framework for Understanding and Reconstructing High-Performance Attention Kernels</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 Accepted to ACL 2026
    </div>
    <details class="paper-abstract">
      Efficient CUDA implementations of attention mechanisms are critical to modern deep learning systems, yet supporting diverse and evolving attention variants remains challenging. Existing frameworks and compilers trade performance for flexibility, while expert-written kernels achieve high efficiency but are difficult to adapt. Recent work explores large language models (LLMs) for GPU kernel generation, but prior studies report unstable correctness and significant performance gaps for complex operators such as attention. We present CuBridge, an LLM-based framework that adapts expert-written attention kernels through a structured lift-transfer-lower workflow. CuBridge starts from expert-written CUDA attention kernels and lifts them into an executable intermediate representation that makes execution orchestration explicit while abstracting low-level CUDA syntax. Given a user-provided PyTorch specification, CuBridge generates and verifies a target IR program, then reconstructs optimized CUDA code via reference-guided lowering. Across diverse attention variants and GPU platforms, CuBridge consistently produces correct kernels and substantially outperforms general frameworks, compiler-based approaches, and prior LLM-based methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.18466v2">Purdah and Patriarchy: Evaluating and Mitigating South Asian Biases in Open-Ended Multilingual LLM Generations</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 Accepted to TrustNLP at ACL (Association for Computational Linguistics) 2026
    </div>
    <details class="paper-abstract">
      Evaluations of Large Language Models (LLMs) often overlook intersectional and culturally specific biases, particularly in underrepresented multilingual regions like South Asia. This work addresses these gaps by conducting a multilingual and intersectional analysis of LLM outputs across 10 Indo-Aryan and Dravidian languages, identifying how cultural stigmas influenced by purdah and patriarchy are reinforced in generative tasks. We construct a culturally grounded bias lexicon capturing previously unexplored intersectional dimensions including gender, religion, marital status, and number of children. We use our lexicon to quantify intersectional bias and the effectiveness of self-debiasing in open-ended generations (e.g., storytelling, hobbies, and to-do lists), where bias manifests subtly and remains largely unexamined in multilingual contexts. Finally, we evaluate two self-debiasing strategies (simple and complex prompts) to measure their effectiveness in reducing culturally specific bias in Indo-Aryan and Dravidian languages. Our approach offers a nuanced lens into cultural bias by introducing a novel bias lexicon and evaluation framework that extends beyond Eurocentric or small-scale multilingual settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04996v1">Tailoring Scaffolding to Diagnostic Strategies: Theory-Informed LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 3 pages, 1 figure. Companion Proceedings 16th International Conference on Learning Analytics & Knowledge (LAK26), Strengthening the Use of Learning Theories for Personalization of Learning Analytics Workshop
    </div>
    <details class="paper-abstract">
      Learning analytics systems increasingly integrate large language models (LLMs) to provide adaptive scaffolding in complex learning environments, yet personalization is often driven by global instructional choices rather than principled alignment with learning theory, limiting effectiveness and pedagogical grounding. In prior work, we examined how structuring and problematizing scaffolding approaches can be instantiated through LLM agents in a scenario-based learning environment for diagnostic reasoning. While both approaches supported learning, we observed systematic differences in learner interaction patterns and clear tendencies indicating that different diagnostic strategies benefited from distinct forms of scaffolding. Building on these findings, we propose a theory-informed scaffolding design grounded in the Knowledge Learning Instruction (KLI) framework, as different diagnostic strategies target different types of knowledge and require different instructional mechanisms. We use KLI to guide the alignment between strategy demands and scaffolding approaches and introduce a KLI-informed hybrid LLM agent that adapts its pedagogical support according to the diagnostic strategy being practiced, rather than applying a single global scaffolding approach. We hypothesize that this design could enable better learning gains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.18231v2">AgenTEE: Confidential LLM Agent Execution on Edge Devices</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents provide powerful automation capabilities, but they also create a substantially broader attack surface than traditional applications due to their tight integration with non-deterministic models and third-party services. While current deployments primarily rely on cloud-hosted services, emerging designs increasingly execute agents directly on edge devices to reduce latency and enhance user privacy. However, securely hosting such complex agent pipelines on edge devices remains challenging. These deployments must protect proprietary assets (e.g., system prompts and model weights) and sensitive runtime state on heterogeneous platforms that are vulnerable to software attacks and potentially controlled by malicious users. To address these challenges, we present AgenTEE, a system for deploying confidential agent pipelines on edge devices. AgenTEE places the agent runtime, inference engine, and third-party applications into independently attested confidential virtual machines (cVMs) and mediates their interaction through explicit, verifiable communication channels. Built on Arm Confidential Compute Architecture (CCA), a recent extension to Arm platforms, AgenTEE enforces strong system-level isolation of sensitive assets and runtime state. Our evaluation shows that such multi-cVMs system is practical, achieving near-native performance with less than 5.15% runtime overhead compared to commodity OS multi-process deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05274v1">Sealing the Audit-Runtime Gap for LLM Skills</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      Large language model (LLM) ecosystems such as Claude Code and ChatGPT increasingly rely on skills: packages of natural-language instructions and executable tools. Once in the LLM's context, skill content cannot be reliably separated from trusted instructions, and a skill's executable side can invoke privileged actions, exposing the skill supply chain to injection, tampering, and rug-pull attacks. Existing defenses are stage-bound: centralized signing, audit reports unbound from the runtime artifact, or policy engines that cannot attest to what was approved. We present SIGIL, the first framework that seals the audit-runtime gap for LLM skills. SIGIL delivers verifiable hosting through a tamper-evident, decentralized on-chain registry from which LLMs fetch skills directly. The registry admits four publication types, Transparent, Licensed, Sealed, and Committed, spanning plaintext public distribution, monetized access, custodial use, and off-chain workflows; before admission, every skill is vetted by a Decentralized Autonomous Organization (DAO) audit committee that supports pluggable auditing methods under a stake-and-slash economic model. At load time, SIGIL delivers verified loading through a skill verification protocol executed by a Skill Verification Loader (SVL) embedded as the mandatory loading path: the SVL retrieves and decrypts the skill as its type requires, verifies its integrity against the on-chain record, and enforces its permission manifest before context injection. We evaluate SIGIL on a real-world deployment against 1,023 in-the-wild skills spanning six attack types. At load time, the SVL verifies each skill's integrity against its on-chain record and enforces its approved permission manifest, completing batched verification under 86 ms. Together, these results show that LLM skills can be cryptographically bound from publication through runtime at practical cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04956v1">KernelBench-X: A Comprehensive Benchmark for Evaluating LLM-Generated GPU Kernels</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      LLM-based Triton kernel generation has attracted significant interest, yet a fundamental empirical question remains unanswered: where does this capability break down, and why? We present KernelBench-X, a benchmark designed to answer this question through category-aware evaluation of correctness and hardware efficiency across 176 tasks in 15 categories. Our systematic comparison of five representative methods yields three main findings. First, task structure determines correctness more than method design. Category explains nearly three times more variance in semantic correctness than method (9.4% vs 3.3% explained deviance), and 72% of Fusion tasks fail across all five methods while Math tasks are solved consistently. Second, iterative refinement improves correctness, but not performance. Across GEAK iterations, compile rate rises from 52.3% to 68.8% while average speedup declines from $1.58\times$ to $1.44\times$; newly rescued kernels consistently underperform persistently correct ones ($1.16\times$ vs $1.58\times$ speedup in round~0$\to$1). Third, correctness does not imply efficiency. 46.6% of correct kernels are slower than the PyTorch eager baseline, and cross-hardware speedup variance reaches $21.4\times$. Besides, quantization remains completely unsolved (0/30 successes) despite non-trivial compilation rates, revealing systematic misunderstanding of numerical computation contracts rather than surface-level syntax errors. These findings suggest that future progress depends on handling global coordination, explicitly modeling numerical precision, and incorporating hardware efficiency into generation. The code is available at https://github.com/BonnieW05/KernelBenchX
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03847v3">DVPO: Distributional Value Modeling-based Policy Optimization for LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has shown strong performance in LLM post-training, but real-world deployment often involves noisy or incomplete supervision. In such settings, complex and unreliable supervision signals can destabilize training and harm generalization. While existing approaches such as worst-case optimization (e.g., RFQI, CQL) and mean-based methods (e.g., PPO, GRPO) can improve stability, they often overlook generalization and may produce overly conservative policies, leading to uneven performance across diverse real scenarios. To this end, we introduce DVPO (Distributional Value Modeling with Risk-aware Policy Optimization), a new RL framework that combines conditional risk theory with distributional value modeling to better balance robustness and generalization. DVPO learns token-level value distributions to provide fine-grained supervision, and applies an asymmetric risk regularization to shape the distribution tails: it contracts the lower tail to dampen noisy negative deviations, while expanding the upper tail to preserve exploratory diversity. Across extensive experiments and analysis in multi-turn dialogue, math reasoning, and scientific QA, DVPO consistently outperforms PPO, GRPO, and robust Bellman-based PPO under noisy supervision, showing its potential for LLM post-training in the real-world.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05890v2">DFPO: Scaling Value Modeling via Distributional Flow towards Robust and Generalizable LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      Training reinforcement learning (RL) systems in real-world environments remains challenging due to noisy supervision and poor out-of-domain (OOD) generalization, especially in LLM post-training. Recent distributional RL methods improve robustness by modeling values with multiple quantile points, but they still learn each quantile independently as a scalar. This results in rough-grained value representations that lack fine-grained conditioning on state information, struggling under complex and OOD conditions. We propose DFPO (Distributional Value Flow Policy Optimization with Conditional Risk and Consistency Control), a robust distributional RL framework that models values as continuous flows across time steps. By scaling value modeling through learning of a value flow field instead of isolated quantile predictions, DFPO captures richer state information for more accurate advantage estimation. To stabilize training under noisy feedback, DFPO further integrates conditional risk control and consistency constraints along value flow trajectories. Experiments on dialogue, math reasoning, and scientific tasks show that DFPO outperforms PPO, FlowRL, and other robust baselines under noisy supervision, achieving improved training stability and generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04908v1">Curated AI beats frontier LLMs at pharma asset discovery</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 5 pages, 5 figures, 1 table
    </div>
    <details class="paper-abstract">
      General-purpose LLMs with web search are increasingly used to scout the competitive landscape of pharmaceutical pipelines. We benchmark Gosset -- an AI platform with a chat interface backed by curated target-, modality-, and indication-level drug-asset annotations -- against four frontier systems with web access (Claude Opus 4.7, GPT 5.5, Gemini 3.1 Pro, Perplexity sonar-pro) on ten niche oncology/immunology targets where most of the pipeline lives in the long tail of preclinical and Asian-developed assets. All five systems receive the same natural-language query and the same JSON output schema. Across 10 targets Gosset returns 3.2x more verified drugs per query than the best frontier system, at perfect precision and 100% recall against the cross-system union of verified drugs. The same curated index is exposed as a Gosset MCP server that any frontier model can call as a tool, suggesting that each of these systems can close most of the recall gap by swapping generic web search for a curated index behind the same chat interface.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04906v1">Strat-Reasoner: Reinforcing Strategic Reasoning of LLMs in Multi-Agent Games</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) excel in certain reasoning tasks, they struggle in multi-agent games where the final outcome depends on the joint strategies of all agents. In multi-agent games, the non-stationarity of other agents brings significant challenges on the evaluation of the reasoning process and the credit assignment over multiple reasoning steps. Existing single-agent reinforcement learning (RL) approaches and their multi-agent extensions fail to address these challenges as they do not incorporate other agents in the reasoning process. In this work, we propose Strat-Reasoner, a novel RL-based framework that improves LLMs' strategic reasoning ability in multi-agent games. We introduce a novel recursive reasoning paradigm where an agent's reasoning also integrates other agents' reasoning processes. To provide effective reward signals for the intermediate reasoning sequences, we employ a centralized Chain-of-Thought (CoT) comparison module to evaluate the reasoning quality. Finally, we compute an accurate hybrid advantage and develop a group-relative RL approach to optimize the LLM policy. Experimental results show that Strat-Reasoner substantially improves strategic abilities of underlying LLMs, achieving 22.1\% average performance improvements across various multi-agent games.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04903v1">Delta-Based Neural Architecture Search: LLM Fine-Tuning via Code Diffs</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 19 pages, 4 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show strong potential for neural architecture generation, yet existing approaches produce complete model implementations from scratch -- computationally expensive and yielding verbose code. We propose Delta-Code Generation, where fine-tuned LLMs generate compact unified diffs (deltas) to refine baseline architectures rather than synthesizing entire models. Our pipeline iteratively fine-tunes the LLM via LoRA on curated architectures from the LEMUR dataset, with MinHash-Jaccard novelty filtering for structural diversity. We evaluate three 7B-class LLMs -- DeepSeek-Coder-7B, Qwen2.5-Coder-7B, and Mistral-7B -- across six datasets (CIFAR-10, CIFAR-100, MNIST, SVHN, ImageNette, CelebA) using a 22-cycle protocol (1,100 candidates per LLM). All three substantially surpass the full-generation baseline (50.6% valid rate, 42.3% mean first-epoch accuracy): DeepSeek-Coder reaches 75.3% valid rate and 65.8% mean accuracy; Qwen2.5-Coder 72.1%/64.6%; Mistral 66.6%/66.1%. On CIFAR-10, best first-epoch accuracies reach 85.5% (Mistral), 85.2% (DeepSeek), 80.6% (Qwen) -- well above 63.98% full generation and 71.5% for the concurrent approach of Gu et al. Output lengths are 30-50 lines versus 200+ for full generation (75-85% reduction). A 50-epoch study confirms the 1-epoch proxy preserves rankings (Mistral: Spearman $ρ$ = 0.926). Delta-based generation is a token-efficient, multi-domain, LLM-agnostic alternative to full-model synthesis for LLM-driven NAS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04886v1">BenCSSmark: Making the Social Sciences Count in LLM Research</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 12 pages, Accepted to LREC 2026
    </div>
    <details class="paper-abstract">
      This position paper argues that the under-representation of social science tasks in contemporary LLM benchmarks limits advances in both LLM evaluation and social scientific inquiry. Benchmarks -- standardized tools for assessing computational systems -- are pivotal in the development of artificial intelligence (AI), including large language models (LLMs). Benchmarks do more than measure progress -- they actively structure it, shaping reputations, research agendas, and commercial outcomes. Despite this central role, the social sciences are largely absent from mainstream evaluation frameworks, even though scholars in these fields generate dozens of rigorously annotated, context-sensitive datasets each year. Integrating this work into benchmark design could significantly improve the generalization and robustness of AI models. In turn, models trained on social scientific tasks would likely yield better performance on classic and contemporary tasks in disciplines as diverse as history, sociology, political science or economics. This is all the more pressing as these disciplines are quickly turning to LLMs for assistance. To address this gap, we introduce BenCSSmark, a benchmark composed of datasets annotated by computational social scientists. By integrating social scientific perspectives into benchmarking, BenCSSmark seeks to promote more robust, transparent, and socially relevant AI systems and to foster efficient collaboration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04835v1">Patterns of Developer Adoption of LLM-Generated Code Refactoring Suggestions</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 Accepted to PROMISE 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have gained widespread popularity and have steadily improved over time, enabling software developers to use them for various code-related tasks. One common task is code refactoring, where the LLM suggests changes for the developer to apply to their code to improve quality attributes such as readability or maintainability. While current research focuses on evaluating LLM-generated refactoring suggestions, there is a limited understanding of how developers apply these suggestions in practice. To explore this, we analyze 169 GitHub commits where developers refactor their code based on a ChatGPT conversation linked in the commit message. We found that developers mostly accept and use the suggestions without modifications. When changes are made, they are mostly major and fall into five different patterns that depend on the refactoring activity, the developer's prompt, and the validity of the response from ChatGPT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21251v3">CAP: Controllable Alignment Prompting for Unlearning in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 Accpeted to ACL 2026 Main Conference
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) trained on unfiltered corpora inherently risk retaining sensitive information, necessitating selective knowledge unlearning for regulatory compliance and ethical safety. However, existing parameter-modifying methods face fundamental limitations: high computational costs, uncontrollable forgetting boundaries, and strict dependency on model weight access. These constraints render them impractical for closed-source models, yet current non-invasive alternatives remain unsystematic and reliant on empirical experience. To address these challenges, we propose the Controllable Alignment Prompting for Unlearning (CAP) framework, an end-to-end prompt-driven unlearning paradigm. CAP decouples unlearning into a learnable prompt optimization process via reinforcement learning, where a prompt generator collaborates with the LLM to suppress target knowledge while preserving general capabilities selectively. This approach enables reversible knowledge restoration through prompt revocation. Extensive experiments demonstrate that CAP achieves precise, controllable unlearning without updating model parameters, establishing a dynamic alignment mechanism that overcomes the transferability limitations of prior methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19354v2">Do Agents Dream of Root Shells? Partial-Credit Evaluation of LLM Agents in Capture the Flag Challenges</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 Accepted to AIWare'26 Benchmark and Dataset Track
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents are increasingly proposed for autonomous cybersecurity tasks, but their capabilities in realistic offensive settings remain poorly understood. We present DeepRed, an open-source benchmark for evaluating LLM-based agents on realistic Capture The Flag (CTF) challenges in isolated virtualized environments. DeepRed places an agent in a Kali attacker environment with terminal tools and optional web search, connected over a private network to a target challenge, and records full execution traces for analysis. To move beyond binary solved/unsolved outcomes, we introduce a partial-credit scoring method based on challenge-specific checkpoints derived from public writeups, together with an automated summarise-then-judge labelling pipeline for assigning checkpoint completion from logs. Using DeepRed, we benchmark ten commercially accessible LLMs on ten VM-based CTF challenges spanning different challenge categories. The results indicate that current agents remain limited: the best model achieves only 35% average checkpoint completion, performing strongest on common challenge types and weakest on tasks requiring non-standard discovery and longer-horizon adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04764v1">Elicitation Matters: How Prompts and Query Protocols Shape LLM Surrogates under Sparse Observations</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      Large language models are increasingly used as surrogate models for low-data optimization, but their optimizer-facing prediction and its uncertainty remain poorly understood. We study the surrogate belief elicited from an LLM under sparse observations, showing that it depends strongly on prompt text and query protocol. We introduce an uncertainty-alignment criterion that measures whether model uncertainty tracks residual ambiguity among sample-consistent functions. Across controlled inference tasks and Bayesian optimization studies, we find that structural prompts act as effective priors, POINTWISE and JOINT querying induce different beliefs, and sequential evidence leads to non-monotonic, order-sensitive confidence updates. These effects change downstream acquisition decisions and regret, showing that elicitation protocol is part of the LLM surrogate specification, not a formatting detail.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.03686v2">From Code to Prediction: Fine-Tuning LLMs for Neural Network Performance Classification in NNGPT</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      Automated Machine Learning (AutoML) frameworks increasingly leverage Large Language Models (LLMs) for tasks such as hyperparameter optimization and neural architecture code generation. However, current LLM-based approaches focus on generative outputs and evaluate them by training the produced artifacts. Whether LLMs can learn to reason about neural network performance across datasets remains underexplored. We present a classification task integrated into the NNGPT framework, in which a fine-tuned LLM predicts which of two image classification datasets a given neural network architecture achieves higher accuracy on. The task is built on the LEMUR dataset, which provides standardized PyTorch implementations with reproducible performance metrics. Three prompt configurations of increasing difficulty are evaluated: a normalized-accuracy baseline (trivially reaching 100%), a metadata-enriched prompt replacing accuracies with dataset properties, and a code-only prompt presenting only architecture source code and dataset names. Using DeepSeek-Coder-7B-Instruct fine-tuned with LoRA, the code-only prompt reaches 80% peak accuracy over 15 epochs, while the metadata prompt peaks at 70%. Perdataset analysis reveals complementary strengths: metadata excels for datasets with distinctive properties (CelebAGender at 90.9%) but degrades for overlapping characteristics, whereas the code-only prompt shows more balanced performance. A comparison with DeepSeek-Coder1.3B confirms that model capacity affects this form of architectural reasoning. The results establish that LLMs can be fine-tuned to predict cross-dataset suitability from neural network code, suggesting that architecture source code contains richer discriminative signal than dataset metadata alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04738v1">OSAQ: Outlier Self-Absorption for Accurate Low-bit LLM Quantization</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 ICML 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities. However, their massive parameter scale leads to significant resource consumption and latency during inference. Post-training weight-only quantization offers a promising solution by reducing model size and accelerating token generation through alleviating the memory-bound issue. Nevertheless, the presence of inherent systematic outliers in weights continues to be a major obstacle. While existing methods, such as scaling and rotation, attempt to address this issue, the performance remains unsatisfactory. In this paper, we propose Outlier Self-Absorption Quantization (OSAQ), which performs additive weight suppression guided by the second-order low-rank property for low-bit weight-only quantization of LLMs. Specifically, we observe that the Hessian exhibits low-rank consistency across different inputs, with certain directions consistently showing vanishing curvature. Leveraging this property, we identify a stable null space of the Hessian and then construct an additive weight transformation by linearly combining the vectors within this null space, thereby suppressing weight outliers without affecting the task loss. This additive transformation can be absorbed into the weights offline, requiring no inter-layer transformations and introducing no inference overhead. Moreover, the construction is efficiently achieved by a closed-form solution, without resource-intensive training or iterative procedures. Extensive experiments demonstrate that OSAQ effectively suppresses outliers and enhances low-bit quantization performance. For instance, in 2-bit quantization, OSAQ, when integrated with GPTQ, achieves over 40% lower perplexity compared to vanilla GPTQ.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05267v1">Bridging Generation and Training: A Systematic Review of Quality Issues in LLMs for Code</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) frequently generate defective outputs in code generation tasks, ranging from logical bugs to security vulnerabilities. While these generation failures are often treated as model-level limitations, empirical evidence increasingly traces their root causes to imperfections within the training corpora. Yet, the specific mechanisms linking training data quality issues to generated code quality issues remain largely unmapped. This paper presents a systematic literature review of 114 primary studies to investigate how training data quality issues propagate into code generation. We establish a unified taxonomy that categorizes generated code quality issues across nine dimensions and training data quality issues into code and non-code attributes. Based on this taxonomy, we formalize a causal framework detailing 18 typical propagation mapping mechanisms. Furthermore, we synthesize state-of-the-art detection and mitigation techniques across the data, model, and generation lifecycles. The reviewed literature reveals a clear methodological shift: quality assurance is transitioning from reactive, heuristic-based post-generation filtering toward proactive, data-centric governance and closed-loop repair. Finally, we identify open challenges and outline research directions for developing reliable LLMs for code through integrated data curation and continuous evaluation. Our repository is available at https://github.com/SYSUSELab/From-Data-to-Code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04677v1">CodeEvolve: LLM-Driven Evolutionary Optimization with Runtime-Enriched Target Selection for Multi-Language Code Enhancement</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 14 pages, 2 figures
    </div>
    <details class="paper-abstract">
      We present CodeEvolve, an evolutionary framework for improving program performance and code quality with Large Language Models (LLMs). CodeEvolve extends OpenEvolve with runtime-guided target selection, Monte Carlo Tree Search (MCTS), automated code refinement, and language-specific evaluation pipelines for Java and Salesforce Apex. The system uses Java Flight Recorder (JFR) profiles to build weighted component graphs and select optimization targets that account for most execution cost, reducing reliance on manual bottleneck identification. For each target, CodeEvolve generates candidate edits, evaluates them through build validation, unit tests, performance checks, static analysis, and LLM-based review, and retains only variants that preserve functional correctness. Across real-world optimization tasks, CodeEvolve improves performance and code metrics while maintaining correctness. On a large enterprise Java codebase, it achieves an average speedup of 15.22$\times$ across seven hotspot functions and outperforms single-pass LLM optimization on five of them. An ablation study on Apex optimization shows that the full MCTS-augmented configuration produces 19.5 valid programs out of 20 on average, indicating that search, filtering, and refinement each contribute to more reliable optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04665v1">Paraphrase-Induced Output-Mode Collapse: When LLMs Break Character Under Semantically Equivalent Inputs</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      When the substantive content of a request is rewritten, do large language models still answer in the format the original task asked for? We find that they often do not, even at temperature zero. On a 150-query evaluation over five compact 2025-era LLMs and four task types, we observe a systematic failure mode we call prompt-variant output-mode collapse: when a closed-form prompt asks for a bare label or a single choice token, content-preserving prompt variants can push the model into conversational prose, the requested format dissolves, and exact-match evaluation pipelines silently misjudge the result. To make this measurable, we release PARACONSIST, a 900-prompt benchmark of 150 base queries with five lexical, syntactic, and semantic-expansion prompt variants each, and a Semantic Consistency Score that decomposes prompt-variant robustness into answer consistency, sentence-BERT semantic similarity, and length stability. Under a whole-word answer-set match, only ~22% of closed-form variant responses preserve the ground-truth label inside their output, while ~78% drift away from the answer space entirely. In our pool, the dominant predictor of collapse is task structure rather than model identity, with model differentiation jointly carried by answer consistency and length stability. Robustness audits should therefore track response-mode preservation as a first-class reliability target alongside answer accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04643v1">Graph-Augmented LLMs for Swiss MP Ideology Prediction</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 Accepted by SwissText 26
    </div>
    <details class="paper-abstract">
      Approximating the ideological position of Members of Parliament (MPs) is a fundamental task in political science, helping researchers understand legislative behavior, party alignment, and policy preferences. While Large Language Models (LLMs) have shown promising results in estimating MPs' ideological stances, there are more actors and elements in the parliamentary system, and relations between them, that could provide a wider and more informative picture. However, due to the complexity of integrating them in the prediction task, these additional elements are generally ignored. In this work, we propose an LLM framework, PG-RAG, that implements a retrieval-augmented generation pipeline: it first queries a political knowledge graph (KG) and then integrates the resulting graph-structured information into the context. This allows for capturing both textual semantics and inter-MP relationships, another relevant information source in any parliamentary system. We evaluate the approach on the task of ideology prediction, using data from a Swiss parliamentary dataset. When comparing graph-augmented models against several state-of-the-art baselines, the results demonstrate that incorporating this enriched information, which encodes information about different entities and relations, improves prediction performance. These results help to highlight the value of domain-specific relational information in modeling political behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.02463v2">When Stress Becomes Signal: Detecting Antifragility-Compatible Regimes in Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      Multi-agent LLM systems are increasingly used to solve complex tasks through decomposition, debate, specialization, and ensemble reasoning. However, these systems are usually evaluated in terms of robustness: whether performance is preserved under perturbation. This paper studies a different question: whether semantic stress exposes structured variation that could support future antifragile learning. We introduce CAFE (Cognitive Antifragility Framework for Evaluation), a statistical framework for detecting antifragility-compatible regimes in multi-agent architectures. CAFE models a controlled expected distribution of semantic stressors, reconstructs an architecture-specific observed effective stress distribution from multi-dimensional judge signals, and compares both distributions using a distributional Jensen Gap under a convex stress potential. A positive gap does not imply immediate performance improvement; instead, it indicates a convex-expansive deformation of the observed stress distribution, suggesting that the architecture exposes learnable stress structure. We evaluate CAFE on a banking-risk analysis benchmark with five multi-agent architectures: flat, hierarchical, debate, meta-adaptive, and ensemble. Across all architectures, semantic stress reduces average judged quality by roughly one third. Yet all architectures exhibit positive distributional Jensen Gaps with bootstrap confidence intervals above zero. These results show that immediate quality degradation can coexist with statistically detectable antifragility-compatible stress geometry. CAFE is therefore not an antifragile learner itself, but a measurement layer for identifying when and where antifragility learning may be worth applying.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.08457v2">Hybrid Pooling with LLMs via Relevance Context Learning</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 SIGIR 2026
    </div>
    <details class="paper-abstract">
      High-quality relevance judgements over large query sets are essential for evaluating Information Retrieval (IR) systems, yet manual annotation remains costly and time-consuming. Large Language Models (LLMs) have recently shown promise as automatic relevance assessors, but their reliability is still limited. Most existing approaches rely on zero-shot prompting or in-context learning (ICL) with a small number of labelled examples. However, standard ICL treats examples as independent instances and fails to explicitly capture the underlying relevance criteria of a topic, restricting its ability to generalise to unseen query-document pairs. To address this limitation, we introduce Relevance Context Learning (RCL), a novel framework that leverages human relevance judgements to explicitly model topic-specific relevance criteria. Rather than directly using labelled examples for in-context prediction, RCL first prompts an LLM (Instructor LLM) to analyse sets of judged query-document pairs and generate explicit narratives that describe what constitutes relevance for a given topic. These relevance narratives are then used as structured prompts to guide a second LLM (Assessor LLM) in producing relevance judgements. To evaluate RCL in a realistic data collection setting, we propose a hybrid pooling strategy in which a shallow depth-k pool from participating systems is judged by human assessors, while the remaining documents are labelled by LLMs. Experimental results demonstrate that RCL substantially outperforms zero-shot prompting and consistently improves over standard ICL. Overall, our findings indicate that transforming relevance examples into explicit, context-aware relevance narratives is a more effective way of exploiting human judgements for LLM-based IR dataset construction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04595v1">A Queueing-Theoretic Framework for Stability Analysis of LLM Inference with KV Cache Memory Constraints</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 Accepted in ICML 2026
    </div>
    <details class="paper-abstract">
      The rapid adoption of large language models (LLMs) has created significant challenges for efficient inference at scale. Unlike traditional workloads, LLM inference is constrained by both computation and the memory overhead of key-value (KV) caching, which accelerates decoding but quickly exhausts GPU memory. In this paper, we introduce the first queueing-theoretic framework that explicitly incorporates both computation and GPU memory constraints into the analysis of LLM inference. Based on this framework, we derive rigorous stability and instability conditions that determine whether an LLM inference service can sustain incoming demand without unbounded queue growth. This result offers a powerful tool for system deployment, potentially addressing the core challenge of GPU provisioning. By combining an estimated request arrival rate with our derived stable service rate, operators can calculate the necessary cluster size to avoid both costly over-purchasing and performance-violating under-provisioning. We further validate our theoretical predictions through extensive experiments in real GPU production environments. Our results show that the predicted stability conditions are highly accurate, with deviations typically within 10%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.14274v2">Discovering New Theorems via LLMs with In-Context Proof Learning in Lean</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 12 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated significant promise in formal theorem proving. In this study, we investigate the ability of LLMs to discover novel theorems and produce verified proofs. We propose a pipeline called \textit{Conjecturing-Proving Loop} (CPL), which iteratively generates mathematical conjectures and attempts to prove them in Lean 4. A key feature of CPL is that each iteration conditions the LLM on previously generated theorems and their formal proofs, enabling parameter-free improvement of proof strategies via in-context learning. We provide both theoretical and experimental evidence that CPL increases the discovery rate of hard-to-prove theorems compared to frameworks that generate statements and proofs simultaneously. Moreover, our experiments show that reusing the LLM's own formally verified outputs as context consistently improves subsequent proof success, demonstrating the effectiveness of self-generated in-context learning for neural theorem proving. The source code is available at https://github.com/auto-res/ConjecturingProvingLoop.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04585v1">IntenBot: Flexible and Imprecise Multimodal Input for LLMs to Understand User Intentions for Casual and Human-Like HRI</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      In natural human-to-human communication, multimodal user input is typically used to supplement explicit and complement implicit voice commands, with casualness allowing for flexible input modality combinations and tolerance for imprecise input data. For example, saying "I want that." with a casual glance at a bottle of water is clear enough in human-to-human communication as an implicit voice command accompanied by gaze and/or gestures, rather than an explicit one. To enable such a human-like interaction in human-robot interaction (HRI), we propose a system, IntenBot, to understand user intentions from flexible and imprecise multimodal input, including voice, gaze, and finger-pointing, in XR. The disambiguation capability of large language models (LLMs) is used to filter out irrelevant input modalities and imprecise input data, generating potential instructions for user confirmation. The flexible and imprecise multimodal input enables casual, human-like interaction with robots, reducing time, effort, and attention, and could also be used as non-voice input. We conducted an informative user behavior study in a simulated environment to understand users' natural be- havior in flexibly interacting with a robot using multimodal input and to obtain appropriate angle range parameters for gaze and finger-pointing. An XR study was then performed to evaluate the performance of IntenBot, compared with other methods. We also deployed IntenBot on a physical robot to showcase its real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04572v1">From Parameter Dynamics to Risk Scoring : Quantifying Sample-Level Safety Degradation in LLM Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 Accepted by ICML 2026
    </div>
    <details class="paper-abstract">
      Safety alignment of Large Language Models (LLMs) is extremely fragile, as fine-tuning on a small number of benign samples can erase safety behaviors learned from millions of preference examples. Existing studies attempt to explain this phenomenon by comparing parameters and hidden states before and after fine-tuning, but overlook their dynamic evolution during fine-tuning. In this paper, we uncover a critical mechanism underlying safety degradation by analyzing parameter dynamics, where benign fine-tuning causes parameters to cumulatively drift toward danger-aligned directions, progressively undermining the model's safety. This finding suggests that samples contributing more to this drift has greater fine-tuning risks. Based on this insight, we propose a method of Sample-Level Quantification of Safety Degradation (SQSD), which quantifies the influence of each training sample on safety degradation. Specifically, SQSD computes continuous risk scores to samples by measuring their induced parameter updates' projection difference between danger and safety directions. Extensive experiments across multiple models and datasets demonstrate that SQSD effectively quantifies sample-level fine-tuning risks and exhibits strong transferability across model architectures, parameter scales, and parameter-efficient methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04556v1">Benchmarking LLMs on the Massive Sound Embedding Benchmark (MSEB)</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      The Massive Sound Embedding Benchmark (MSEB) has emerged as a standard for evaluating the functional breadth of audio models. While initial baselines focused on specialized encoders, the shift toward "audio-native" Large Language Models (LLMs) suggests a new paradigm where a single multimodal backbone may replace complex, task-specific pipelines. This paper provides a rigorous empirical evaluation of leading LLMs - including members from the Gemini and GPT families - across the eight core MSEB capabilities to assess their efficacy and audio-text parity. Our results indicate that while a significant modality gap persists regarding performance and robustness, the empirical evidence for an "optimal" modeling approach remains inconclusive. Ultimately, the choice between audionative and cascaded architectures depends heavily on specific use-case requirements and the underlying assumptions regarding latency, cost, and reasoning depth.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.00753v5">LLM-Based Human-Agent Collaboration and Interaction Systems: A Survey</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 Accepted by ACL 2026 (Findings). Paper lists and resources are available at https://github.com/HenryPengZou/Awesome-Human-Agent-Collaboration-Interaction-Systems
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have sparked growing interest in building fully autonomous agents. However, fully autonomous LLM-based agents still face significant challenges, including limited reliability due to hallucinations, difficulty in handling complex tasks, and substantial safety and ethical risks, all of which limit their feasibility and trustworthiness in real-world applications. To overcome these limitations, LLM-based human-agent systems (LLM-HAS) incorporate human-provided information, feedback, or control into the agent system to enhance system performance, reliability, and safety. These human-agent collaboration systems enable humans and LLM-based agents to collaborate effectively by leveraging their complementary strengths. This paper provides the first comprehensive and structured survey of LLM-HAS. It clarifies fundamental concepts, systematically presents core components shaping these systems, including environment and profiling, human feedback, interaction types, orchestration, and communication, explores emerging applications, and discusses unique challenges and opportunities arising from human-AI collaboration. By consolidating current knowledge and offering a structured overview, we aim to foster further research and innovation in this rapidly evolving interdisciplinary field. Paper lists and resources are available at https://github.com/HenryPengZou/Awesome-Human-Agent-Collaboration-Interaction-Systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.00445v2">The Power of Order: Fooling LLMs with Adversarial Table Permutations</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      Large Language Models have achieved remarkable success and are increasingly deployed in critical applications involving tabular data, such as Table Question Answering. However, their robustness to the structure of this input remains a critical, unaddressed question. This paper demonstrates that modern LLMs exhibit a significant vulnerability to the layout of tabular data. Specifically, we show that semantically-invariant permutations of rows and columns - rearrangements that do not alter the table's underlying information - are sometimes sufficient to cause incorrect or inconsistent model outputs. To systematically probe this vulnerability, we introduce Adversarial Table Permutation, a novel, gradient-based attack that efficiently identifies worst-case permutations designed to maximally disrupt model performance. Our extensive experiments demonstrate that ATP significantly degrades the performance of a wide range of LLMs. This reveals a pervasive vulnerability across different model sizes and architectures, including the most recent and popular models. Our findings expose a fundamental weakness in how current LLMs process structured data, underscoring the urgent need to develop permutation-robust models for reliable, real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04539v1">RLearner-LLM: Balancing Logical Grounding and Fluency in Large Language Models via Hybrid Direct Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      Direct Preference Optimization (DPO), the efficient alternative to PPO-based RLHF, falls short on knowledge-intensive generation: standard preference signals from human annotators or LLM judges exhibit a systematic verbosity bias that rewards fluency over logical correctness. This blindspot leaves a logical alignment gap -- SFT models reach NLI entailment of only 0.05-0.22 despite producing fluent text. We propose RLearner-LLM with Hybrid-DPO: an automated preference pipeline that fuses a DeBERTa-v3 NLI signal with a verifier LLM score, removing human annotation while overcoming the "alignment tax" of single-signal optimization. Evaluated across five academic domains (Biology, Medicine, Law) with three base architectures (LLaMA-2-13B, Qwen3-8B, Gemma 4 E4B-it), RLearner-LLM yields up to 6x NLI improvement over SFT, with NLI gains in 11 of 15 cells and consistent answer-coverage gains. On Gemma 4 E4B-it (4.5B effective params), Hybrid-DPO lifts NLI in four of five domains (+11.9% to +2.4x) with faster inference across all five, scaling down to compact base models without losing the alignment-tax mitigation. Our Qwen3-8B RLearner-LLM wins 95% of pairwise comparisons against its own SFT baseline; GPT-4o-mini in turn wins 95% against our concise output -- alongside the 69% win the same judge gives a verbose SFT over our DPO model, this replicates verbosity bias on a frontier comparator and motivates logic-aware metrics (NLI, ACR) over LLM-as-a-judge for knowledge-intensive generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.09885v5">Diffusion-Inspired Masked Fine-Tuning for Knowledge Injection in Autoregressive LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are often used in environments where facts evolve, yet factual knowledge updates via fine-tuning on unstructured text often suffer from 1) reliance on compute-heavy paraphrasing augmentation and 2) the reversal curse. Recent studies show diffusion large language models (dLLMs) require fewer training samples to achieve lower loss in pre-training and are more resistant to the reversal curse, suggesting dLLMs may learn new knowledge more easily than autoregressive LLMs (arLLMs). We test this hypothesis in controlled knowledge fine-tuning experiments and find that while arLLMs rely on paraphrase augmentation to generalize knowledge text into question-answering (QA) capability, dLLMs do not require paraphrases to achieve high QA accuracy. To further investigate whether the demasking objective alone can induce such a knowledge injection advantage in dLLMs regardless of their diffusion denoising paradigm, we propose masked fine-tuning for arLLMs, which prompts an arLLM to reconstruct the original text given a masked version in context. The masked fine-tuning for arLLMs substantially improves the efficacy of knowledge injection, i.e. no paraphrase needed and resistant to the reversal curse, closing the gap between arLLMs and dLLMs. We also demonstrate broader applicability: on a large-scale knowledge-intensive dataset (1.2M samples), masked SFT achieves the best downstream accuracy on GPQA-diamond among all fine-tuning variants. The demasking objective also improves SFT on math tasks, suggesting broad utility beyond factual knowledge injection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04534v1">Characterizing Students' LLM Usage Behaviors and Their Association with Learning in Critical Thinking Tasks</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 EDM 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are becoming increasingly embedded in students' learning practices, yet much of what is known about how students use LLMs and how this usage impacts learning comes from problem-solving domains or constrained experimental settings. We present an analysis of data on LLM usage collected during two offerings of a research-oriented course where students learn to read, reason about, and critique academic papers. Without restrictions on whether or how to use LLMs, students reported their LLM usage practices when asked to do these activities as a series of homework assignments during the course. This paper extends prior work done on data from a single offering of the same course by presenting a refined bottom-up categorization of LLM usage types, cross-labeled by the extent of student initiative these usages entail. Furthermore, we examine how LLM use impacts student learning, measured by performance on three midterms, looking at factors such as frequency and type of usage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04530v1">SADE: Symptom-Aware Diagnostic Escalation for LLM-Based Network Troubleshooting</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents are increasingly applied to network troubleshooting, but root-cause localization on public benchmarks remains well below practical deployment thresholds. We argue this is because existing agents do not encode the disciplined, layer-by-layer methodology that human network engineers use, and instead rely on free-form deliberation that conflates evidence acquisition with hypothesis commitment. We present SADE (Symptom-Aware Diagnostic Escalation), an agent that encodes the classical Cisco troubleshooting methodology as an explicit policy. SADE pairs a phase-gated diagnostic workflow, which separates evidence acquisition from hypothesis commitment, with a routed library of fault-family skills and high-yield diagnostic helpers. On a held-out 523 incident set of the public NIKA benchmark covering eleven unseen scenarios, SADE improves root-cause F1 by 37 percentage points over a ReAct + GPT-5 baseline; a model-controlled comparison against the same Claude Sonnet backend without the SADE policy attributes 22 of those points to the diagnostic policy alone, showing that the gain is not a side-effect of the model upgrade.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04523v1">RaguTeam at SemEval-2026 Task 8: Meno and Friends in a Judge-Orchestrated LLM Ensemble for Faithful Multi-Turn Response Generation</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      We present our winning system for Task~B (generation with reference passages) in SemEval-2026 Task~8: MTRAGEval. Our method is a heterogeneous ensemble of seven LLMs with two prompting variants, where a GPT-4o-mini judge selects the best candidate per instance. We ranked 1st out of 26 teams, achieving a conditioned harmonic mean of 0.7827 and outperforming the strongest baseline (gpt-oss-120b, 0.6390). Ablations show that diversity in model families, scales, and prompting strategies is essential, with the ensemble consistently beating any single model. We also introduce Meno-Lite-0.1, a 7B domain-adapted model with a strong cost--performance trade-off, and analyse MTRAGEval, highlighting annotation limitations and directions for improvement. Our code is publicly available: https://github.com/RaguTeam/ragu_mtrag_semeval
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04515v1">From Priors to Perception: Grounding Video-LLMs in Physical Reality</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      While Video Large Language Models (Video-LLMs) excel in general understanding, they exhibit systematic deficits in fine-grained physical reasoning. Existing interventions not only suffer from limited generalization but fundamentally conflate generative artifacts with genuine physical fallacies. Furthermore, we find that models fail systematically not only in anti-physics anomalies but also in counter-intuitive scenarios where visual facts contradict statistical expectations. Accordingly, we propose the Unified Attribution Theory: this dual failure stems not from perception deficiency, but from Semantic Prior Dominance -- the reasoning mechanism is deeply hijacked by internal narrative scripts. To address this, we construct the Programmatic Adversarial Curriculum (PACC), the first high-fidelity adversarial video dataset synthesized based on physical laws, thoroughly decoupling visual artifacts from logical errors. Concurrently, we design the Visual-Anchored Reasoning Chain (VARC) to force models to explicitly ground their judgments in low-level visual facts prior to logical adjudication. Experiments demonstrate that without invasive architectural modifications, standard LoRA fine-tuning with the PACC curriculum effectively neutralizes prior interference in state-of-the-art (SOTA) models, yielding a substantial leap in physical reasoning capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04505v1">JASTIN: Aligning LLMs for Zero-Shot Audio and Speech Evaluation via Natural Language Instructions</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      The rapid advancement of generative audio models has outpaced the development of robust evaluation methodologies. Existing objective metrics and general multimodal large language models (MLLMs) often struggle with domain generalization, zero-shot capabilities, and instructional flexibility. To address these bottlenecks, we propose JASTIN, a generalizable, instruction-driven audio evaluation framework that formulates audio assessment as a self-instructed reasoning task. JASTIN bridges a frozen high-performance audio encoder with a fine-tuned LLM backbone via a trainable audio adapter. To ensure robust zero-shot generalization, we introduce a comprehensive instruction following data preparation pipeline, incorporating Multi-Source, Multi-Task, Multi-Calibration, and Multi-Description data. Experimental results demonstrate that JASTIN achieves state-of-the-art Pearson and Spearman correlations with human subjective ratings. It consistently outperforms general MLLMs across speech, sound, music, and out-of-domain evaluation tasks without the need for task-specific retraining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05260v1">SecureMCP: A Policy-Enforced LLM Data Access Framework for AIoT Systems via Model Context Protocol</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      The deployment of Large Language Model (LLM)-generated SQL queries in Artificial Intelligence of Things (AIoT) systems introduces critical security risks, as prompt injection attacks can manipulate LLMs into producing unauthorized queries that expose sensitive data or execute destructive operations. Existing NL2SQL research focuses on query accuracy, while MCP server implementations provide only SQL-level protections without fine-grained role-based access control. This paper proposes SecureMCP, a policy-enforced LLM data access framework integrating Role-Based Access Control (RBAC) with an MCP server to establish multi-layer defense for LLM-generated SQL execution. The framework incorporates five defense modules -- check_policy for table-and-column-level RBAC, explain_gate for cost-explosive query blocking, SQL Interceptor for dangerous pattern detection, Risk Level Filter for SQL risk classification, and DB Isolation for cross-database restriction -- operating in a sequential fail-closed pipeline mapped to six prompt injection types grounded in the OWASP Top 10 for LLM Applications. We evaluate SecureMCP on the IoT-SQL dataset (11 tables, 173 columns, 239,398 records) using Qwen3-8B. Experiment A demonstrates that defense modules preserve execution accuracy, with EX-in-ALLOW remaining within 65.1%-76.4% across four RBAC roles, matching the unprotected baseline of 63.8%. Experiment B shows that SecureMCP achieves 82.3% Policy Compliance on 2,400 adversarial queries, with genuine defense failures limited to 3.4%. The defense-in-depth analysis reveals check_policy accounts for 78.7% of blocks, while secondary modules contribute an additional 17.5 percentage-point improvement. The Injection Incorporation Rate of 72.5% confirms high LLM susceptibility, establishing the necessity of external policy enforcement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.23475v2">Supernodes and Halos: Loss-Critical Hubs in LLM Feed-Forward Layers</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      We study the organization of channel-level importance in transformer feed-forward networks (FFNs). Using a Fisher-style loss proxy (LP) based on activation-gradient second moments, we show that loss sensitivity is concentrated in a small set of channels within each layer. In Llama-3.1-8B, the top 1% of channels per layer accounts for a median of 58.7% of LP mass, with a range of 33.0% to 86.1%. We call these loss-critical channels supernodes. Although FFN layers also contain strong activation outliers, LP-defined supernodes overlap only weakly with activation-defined outliers and are not explained by activation power or weight norms alone. Around this core, we find a weaker but consistent halo structure: some non-supernode channels share the supernodes' write support and show stronger redundancy with the protected core. We use one-shot structured FFN pruning as a diagnostic test of this organization. At 50% FFN sparsity, baselines that prune many supernodes degrade sharply, whereas our SCAR variants explicitly protect the supernode core; the strongest variant, SCAR-Prot, reaches perplexity 54.8 compared with 989.2 for Wanda-channel. The LP-concentration pattern appears across Mistral-7B, Llama-2-7B, and Qwen2-7B, remains visible in targeted Llama-3.1-70B experiments, and increases during OLMo-2-7B pretraining. These results suggest that LLM FFNs develop a small learned core of loss-critical channels, and that preserving this core is important for reliable structured pruning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04488v1">How Does Thinking Mode Change LLM Moral Judgments? A Controlled Instant-vs-Thinking Comparison Across Five Frontier Models</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      We evaluate whether enabling provider-exposed reasoning mode changes moral judgments within the same model checkpoint. Across 100 moral-judgment scenarios and five frontier reasoning-trained LLMs (Claude Sonnet 4.6, GPT 5.5, Gemini 3 Flash, DeepSeek V3.1, and Qwen3.5 397B), aggregate binary-verdict agreement remains high and statistically indistinguishable between instant and thinking modes (Krippendorff's alpha = 0.78 vs. 0.79). However, disagreement is concentrated in 21 model-disputed scenarios, where instant-mode agreement is near chance (alpha = 0.08). On these scenarios, reasoning directionally narrows cross-model disagreement, increasing mean pairwise agreement from 5.4 to 6.7 out of 10. Reasoning also reduces demographic-judgment inconsistency in three of five models and does not increase it for any model. Across all five model families, reasoning changes self-labeled ethical frameworks more often than binary verdicts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.00305v2">Syntax- and Compilation-Preserving Evasion of LLM Vulnerability Detectors</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      LLM-based vulnerability detectors are increasingly deployed in CI/CD security gating, yet their resilience to evasion under syntax- and compilation-preserving edits remains poorly understood. We evaluate five attack variants spanning four carrier families of behavior-preserving code transformations on a unified C/C++ benchmark ($N=5000$) and introduce Complete Resistance (CR), measuring the fraction of correctly detected vulnerabilities that withstand all attack variants. Our findings reveal a significant robustness gap: models achieving 70\%+ clean recall exhibit CR as low as 0.12\%, meaning over 87\% of detected vulnerabilities can be evaded by at least one syntax-preserving edit. Universal adversarial strings optimized on a 14B surrogate transfer effectively to black-box APIs including GPT-4o, while on-target optimization further amplifies evasion (up to 92.5\% ASR). These results indicate that clean benchmark accuracy alone is insufficient as a security guarantee for deployed vulnerability detectors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04472v1">Automated Formal Proofs of Combinatorial Identities via Wilf-Zeilberger Guidance and LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 Accepted to ICML 2026. Preprint version
    </div>
    <details class="paper-abstract">
      Automating formal proofs of combinatorial identities is challenging for LLM-based provers, as long-horizon proof planning is required and unconstrained search quickly explodes. Symbolic methods such as the Wilf-Zeilberger (WZ) method can achieve a mechanized proof of combinatorial identities by constructing special auxiliary functions and demonstrating that they satisfy specific recurrence relations. We propose WZ-LLM, a neuro-symbolic framework that turns WZ proof plans into executable proof sketches in Lean 4 and uses an LLM-based prover to discharge the resulting machine-checkable subgoals. We also train a dedicated WZ-Prover via a Lean-kernel-verified bootstrapping loop with expert-verified iteration, followed by DAPO-based refinement. Experiments show that WZ-LLM achieves a 34% proof success rate on LCI-Test (100 classic combinatorial identities), outperforming strong baselines such as DeepSeek-V3 and Goedel-Prover-V2, and delivering consistent gains on CombiBench and PutnamBench-Comb. These results indicate that our framework provides two complementary strengths: improved direct proving for identities beyond the scope of WZ, and substantially higher end-to-end success when WZ sketches guide a specialized prover.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04468v1">Stabilizing LLM Supervised Fine-Tuning via Explicit Distributional Control</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      Post-training large language models (LLMs) often suffers from catastrophic forgetting, where improvements on a target objective degrade previously acquired capabilities. Recent evidence suggests that this phenomenon is primarily driven by excessive distributional drift during optimization. Motivated by this perspective, we propose Anchored Learning, a simple framework that explicitly controls distributional updates during offline fine-tuning via a dynamically evolving moving anchor. Instead of matching a fixed reference distribution, the anchor interpolates between the current model and a frozen reference to construct an intermediate target that the model distills toward, transforming global fine-tuning into a sequence of local trust-region updates in distribution space. Theoretically, we prove this anchor-based update admits a linear KL-divergence upper bound per iteration, ensuring a stable transition between model distributions. Extensive experiments on iGSM, MedCalc, and IFEval show that Anchored Learning consistently lies on the Pareto frontier of gain-stability trade-offs, achieving near-optimal performance improvements while substantially reducing degradation compared to strong baselines. For example, while standard SFT suffers from over 53% performance degradation on iGSM and MedCalc, Anchored Learning slashes this drop to under 5% while maintaining near-optimal gains (e.g., 75.2% on iGSM).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04467v1">KEET: Explaining Performance of GPU Kernels Using LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 12 pages, 8 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Performance profiles of GPU kernels generated by tools such as Nsight Compute are rich in detail but are often challenging to interpret. To achieve the best performance possible on a given GPU architecture, kernel developers need to spend significant time analyzing and comparing profiles in the tool's graphical interface to identify and understand kernel performance bottlenecks. Large Language Models (LLMs) have shown promise in understanding complex data and generating natural language explanations. In this paper, we propose the Kernel Execution Explanation Toolkit (KEET), an LLM-based agentic framework for interpreting Nsight Compute profiles to generate useful and data-grounded natural language explanations of performance issues in GPU kernels, and suggestions for optimizations. We evaluate \toolname using several CUDA kernels of varying complexity on NVIDIA H100 GPUs. We find that the generated explanations, when provided as context, improve the quality of LLM code optimization and multiple-choice question answering in downstream tasks. We further demonstrate that the tool can be used to interpret performance data from large sets of profiles to improve the quality of optimization suggestions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04446v1">Misrouter: Exploiting Routing Mechanisms for Input-Only Attacks on Mixture-of-Experts LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      Mixture-of-Experts (MoE) architectures have emerged as a leading paradigm for scaling large language models through sparse, routing-based computation. However, this design introduces a new attack surface: the routing mechanism that determines which experts process each input. Prior work shows that manipulating routing can bypass safety alignment, but existing attacks require model modification and thus apply only to locally deployed models. By contrast, real-world LLM services are remotely hosted and accessible only through input queries. This raises a fundamental question: can MoE routing be exploited through input-only attacks to induce stronger unsafe behaviors in real-world services? Our key insight is to optimize attacks in a white-box setting on open-source surrogate MoE models and transfer the resulting adversarial inputs to public API services within the same model family. This setting presents three main challenges: routing can be influenced only indirectly through input perturbations, routing control and output generation are tightly coupled, and even a successful safety bypass may still produce low-quality responses. To address these challenges, we propose Misrouter, an input-only attack framework that jointly targets routing behavior and expert functionality. Misrouter identifies weakly aligned experts that are willing to produce target harmful content by analyzing expert activations under harmful queries paired with unsafe continuations. It then optimizes adversarial inputs to steer routing toward these experts and away from strongly aligned ones. It further biases routing toward highly capable general-purpose experts identified from benign question-answering tasks. Finally, because routing and output objectives can conflict, Misrouter uses a two-phase optimization strategy that first steers routing and then optimizes harmful outputs while preserving routing stability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04418v1">Demystifying Manifold Constraints in LLM Pre-training</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      The empirical success of large language model (LLM) pre-training relies heavily on heuristic stabilization techniques, such as explicit normalization layers and weight decay. While recent constrained optimization approaches that explicitly restrict weights may improve numerical stability and performance, the mechanism and motivation for adding constraints still remain elusive. This paper systematically demystifies the role of explicit manifold constraints in LLM pre-training. By introducing the Msign-Aligned Constrained Riemannian Optimizer (MACRO)-a provably convergent, single-loop optimization framework-our study disentangles weight regularization heuristics from interacting mechanisms like RMS normalization and decoupled weight decay. Theoretical analyses and comprehensive empirical evaluations reveal that manifold constraints independently bound forward activation scales and enforce stable rotational equilibrium, thereby subsuming the roles of these heuristic mechanisms. Evaluations on large-scale LLM architectures demonstrate that MACRO achieves highly competitive performance while rigorously preserving the theoretical guarantees of exact Riemannian optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.18396v2">River-LLM: Large Language Model Seamless Exit Based on KV Share</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 Accepted to ACL 2026, 13pages, with appendix
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated exceptional performance across diverse domains but are increasingly constrained by high inference latency. Early Exit has emerged as a promising solution to accelerate inference by dynamically bypassing redundant layers. However, in decoder-only architectures, the efficiency of Early Exit is severely bottlenecked by the KV Cache Absence problem, where skipped layers fail to provide the necessary historical states for subsequent tokens. Existing solutions, such as recomputation or masking, either introduce significant latency overhead or incur severe precision loss, failing to bridge the gap between theoretical layer reduction and practical wall-clock speedup. In this paper, we propose River-LLM, a training-free framework that enables seamless token-level Early Exit. River-LLM introduces a lightweight KV-Shared Exit River that allows the backbone's missing KV cache to be naturally generated and preserved during the exit process, eliminating the need for costly recovery operations. Furthermore, we utilize state transition similarity within decoder blocks to predict cumulative KV errors and guide precise exit decisions. Extensive experiments on mathematical reasoning and code generation tasks demonstrate that River-LLM achieves 1.71 to 2.16 times of practical speedup while maintaining high generation quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04357v1">Coral: Cost-Efficient Multi-LLM Serving over Heterogeneous Cloud GPUs</a></div>
    <div class="paper-meta">
      📅 2026-05-05
    </div>
    <details class="paper-abstract">
      The usage of large language models (LLMs) has grown increasingly fragmented, with no single model dominating. Meanwhile, cloud providers offer a wide range of mid-tier and older-generation GPUs that enjoy better availability and deliver comparable performance per dollar to top-tier hardware. To efficiently harness these heterogeneous resources for serving multiple LLMs concurrently, we introduce Coral, an adaptive heterogeneity-aware multi-LLM serving system. The key idea behind Coral is to jointly optimize resource allocation and the serving strategy of each model replica across all models. To keep pace with shifting throughput demand and resource availability, Coral applies a lossless two-stage decomposition that preserves joint optimality while cutting online solve time from hours to tens of seconds. Our evaluation across 6 models and 20 GPU configurations shows that Coral reduces serving cost by up to 2.79$\times$ over the best baseline, and delivers up to 2.39$\times$ higher goodput under scarce resource availability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04298v1">Towards Self-Referential Analytic Assessment: A Profile-Based Approach to L2 Writing Evaluation with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-05
      | 💬 Accepted for the 21st Workshop on Innovative Use of NLP for Building Educational Applications
    </div>
    <details class="paper-abstract">
      Automated essay scoring (AES) research often relies on rank-based correlation metrics to validate analytic assessment. However, such metrics obscure both intrinsic intercorrelations among analytic dimensions that arise from the structure of writing proficiency itself and halo effects, whereby holistic impressions bleed into fine-grained component scores. As a result, high correlations may mask a system's true diagnostic behaviour. In this study, we propose a novel self-referential assessment evaluation framework that focuses on identifying intra-learner strengths and weaknesses rather than assessing inter-learner rankings. We conduct experiments on the publicly available ICNALE GRA, a uniquely dense second-language writing dataset annotated holistically and analytically by up to 80 trained raters. To obtain reliable reference scores, we apply two-facet Rasch modelling to calibrate rater severity and derive fair average scores across ten analytic aspects and holistic proficiency. We compare the analytic scoring performance of human operational raters and three large language models (LLMs) in a zero-shot setting. Our results show that LLMs tend to outperform single human raters in identifying relative weaknesses (negative feedback) across several proficiency aspects, while human raters remain stronger at identifying relative strengths (positive feedback). Overall, our findings highlight the limitations of rank-based evaluation for analytic assessment and demonstrate the value of intra-learner, profile-based methods for assessing and deploying LLMs in AES.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04295v1">LLMs Uncertainty Quantification via Adaptive Conformal Semantic Entropy</a></div>
    <div class="paper-meta">
      📅 2026-05-05
      | 💬 Accepted for publication in the Proceedings of IJCAI 2026, the 35th International Joint Conference on Artificial Intelligence, 12 Pages
    </div>
    <details class="paper-abstract">
      LLMs' overconfidence, particularly when hallucinating, poses a significant challenge for the deployment of the models in safety-critical settings and makes a reliable estimation of uncertainty necessary. Existing approaches for uncertainty quantification typically prioritize lexical or probabilistic measures; however, these techniques often ignore the semantic variance of different responses with similar meaning. In this paper, we propose Adaptive Conformal Semantic Entropy (ACSE), a method for estimating prompt-level uncertainty by adaptively measuring semantic dispersion in LLMs outputs. Our uncertainty scoring function is based on clustering semantic entropy of multiple diverse responses to the same prompt. The function adaptively adjusts the uncertainty score based on semantic features of each cluster. To ensure statistical reliability of our score, we use conformal calibration to apply a decision rule to accept/abstain the prompts, providing a finite-sample, distribution-free guarantee such that the error rate among the accepted responses remains bounded by a user-specified tolerance. Our extensive experimental evaluations using different LLMs and datasets, demonstrate that our approach consistently outperforms state-of-the-art uncertainty quantification baselines using discriminative performance, conformal guarantees, and probabilistic calibration indicators. As a highlight, for TriviaQA dataset, AUROC of our approach is 0.88 compared to 0.65 produced by the token entropy approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04264v1">Governed Collaborative Memory as Artificial Selection in LLM-Based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-05-05
    </div>
    <details class="paper-abstract">
      Persistent memory is turning language-model-based agents from stateless participants in isolated interactions into state-bearing components of LLM-based multi-agent systems. As memory becomes durable, reloadable, and behavior-shaping across agents, sessions, or versions, a design question arises that is not captured by retrieval accuracy or access control alone: which candidate memories should become shared institutional state? This Viewpoint frames that problem as governed collaborative memory. We argue that memory governance functions as a selection regime, determining which memory variants persist, which remain private, and which are rejected, abstained from, or superseded. We distinguish ungoverned persistence, constitutional or hybrid selection, automatic metric-based selection, and human-ratified artificial selection, emphasizing that these regimes are not a ranking but a design choice over target properties. We then describe a layered architecture that separates agent-local memory, shared institutional memory, archive memory, and project-continuity memory, with provenance and version lineage making selection inspectable. Documented traces from one running LLM-based multi-agent ecosystem illustrate unmanaged false-memory persistence, ratified institutional memory, rejection and revision, identity-preserving expansion, and governance-as-learning. The contribution is a design agenda: persistent LLM-based multi-agent systems should evaluate memory not only for recall and performance, but also for provenance fidelity, selection traceability, epistemic quality, correction pathways, and role preservation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26118v2">LLM-Guided Issue Generation from Uncovered Code Segments</a></div>
    <div class="paper-meta">
      📅 2026-05-05
    </div>
    <details class="paper-abstract">
      Developers are increasingly overwhelmed by AI-generated issue reports that lack actionability and reproducibility, eroding trust in automated bug detection tools. In this paper, we present IssueSpecter, an automated tool that finds bugs in uncovered code segments and automatically generates prioritized, actionable issue reports. IssueSpecter combines coverage analysis with LLM-based defect identification, producing structured reports complete with severity ratings, reproduction steps, and suggested fixes. We evaluate IssueSpecter on 13 actively maintained Python projects, generating 10,467 issue reports. Manual annotation of the top-130 ranked issues by IssueSpecter confirms that 84.6% of the LLM-generated issues are valid or warrant further investigation, with only 15.4% false positives. LLM-based ranking outperforms rule-based ranking by 50% at P@3 and 41% in MRR. The identified bugs cover a wide variety of types, from logic and boundary errors to security vulnerabilities and state consistency bugs. By ranking issues by priority, IssueSpecter aims to help developers focus their attention on the most impactful bugs first. Finally, we validate IssueSpecter through case studies reproducing real bugs surfaced from its generated issue reports, demonstrating its practical value for automatic bug discovery in open-source Python projects. Compared against CoverUp, a state-of-the-art coverage-driven test generation tool, IssueSpecter achieves a higher bug validity rate (81.0% vs. 76.2%) under identical evaluation conditions, using the same model and the same number of evaluated artifacts per project, while additionally providing structured issue reports with reproduction steps and candidate fixes that are immediately actionable without requiring developers to interpret generated test intent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04215v1">Predict-then-Diffuse: Adaptive Response Length for Compute-Budgeted Inference in Diffusion LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-05
    </div>
    <details class="paper-abstract">
      Diffusion-based Large Language Models (D-LLMs) represent a promising frontier in generative AI, offering fully parallel token generation that can lead to significant throughput advantages and superior GPU utilization over traditional autoregressive paradigm. However, this parallelism is constrained by the requirement of a fixed-size response length prior to generation. This architectural limitation imposes a severe trade-off: oversized response length results in computational waste on semantically meaningless padding tokens, while undersized response length cause output truncation requiring costly re-computations that introduce unpredictable latency spikes. To tackle this issue, we propose Predict-then-Diffuse, a simple and model-agnostic framework, that enables compute-budgeted inference per input query by first estimating the response length and then using it to run inference with D-LLM. At its core lies a Adaptive Response Length Predictor (AdaRLP) auxiliary predictor that predicts the optimal response length given an input query. As a measure against under-predicting the response length and re-running inference with a higher response length, we introduce a data-driven safety mechanism, which trades a negligible padding overhead. As a whole, our framework limits the significant waste of computation on padding tokens and preserves output quality. Experimental validation on multiple datasets demonstrate that Predict-then-Diffuse significantly reduces computational costs (FLOP) compared to the default D-LLM inference mechanism and baselines based on heuristics, while being robust to skewed data distributions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04208v1">Nsanku: Evaluating Zero-Shot Translation Performance of LLMs for Ghanaian Languages</a></div>
    <div class="paper-meta">
      📅 2026-05-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive multilingual capabilities for well-resourced languages, yet their performance on low-resource African languages remains poorly understood and largely unevaluated. This paper presents Nsanku, a systematic benchmark that evaluates the zero-shot machine translation performance of 19 open-weight and proprietary LLMs across 43 Ghanaian languages paired with English. Evaluation sentences were sourced from the YouVersion Bible platform, providing 300 sentence pairs per language. Two complementary automatic metrics are employed: Bilingual Evaluation Understudy (BLEU) and Character n-gram F-Score (chrF), alongside an average accuracy score and a cross-language consistency dimension. Nsanku represents the most comprehensive LLM translation evaluation for Ghanaian languages conducted to date. Results show that gemini-2.5-flash achieves the highest overall average score of 26.88 (BLEU: 24.60, chrF: 29.16), followed by claude-sonnet-4-5 at 24.87 (BLEU: 22.46, chrF: 27.28) and gpt-4.1 at 23.20 (BLEU: 21.15, chrF: 25.24). Among open-weight models, kimi-k2-instruct-0905 leads at an average score of 20.87. A critical finding from the consistency analysis is that no model and no language reached the Leaders quadrant of high performance and high consistency simultaneously, indicating that current LLMs are not yet reliably usable for Ghanaian language translation at scale. Siwu achieved the highest per-language average score at 25.73 while Nkonya scored lowest at 11.65. Nsanku establishes a publicly available, community-extensible evaluation infrastructure for African language NLP research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04180v1">MedFabric and EtHER: A Data-Centric Framework for Word-Level Fabrication Generation and Detection in Medical LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-05
    </div>
    <details class="paper-abstract">
      Large Language Models exhibit strong reasoning and semantic understanding capabilities but often hallucinate in domains that require expert knowledge, among which fabrications, the generation of factually incorrect yet fluent statements, pose the greatest risk in medical contexts. Existing medical hallucination datasets inadequately capture fabrication phenomena due to limited fabrication coverage, stylistic disparities between human and LLM-authored texts, and distributional drift during hallucinated sample synthesis. To address this, we propose a data-centric pipeline to generate realistic and word-level fabrications that preserve syntactic and stylistic fidelity while introducing subtle factual deviations, resulting in MedFabric. Building upon this dataset, we introduce ETHER, a modular word-level fabrication detector integrating Text2Table Decomposition, Word Masking and Filling and Hybrid Sentence Pair Evaluation to enhance factual alignment. Empirical results demonstrate that MedFabric outperforms state-of-the-art detectors by over 15% on word-level fabrication benchmarks while maintaining consistent performance across structural similarities, offering a comprehensive framework for reliable and domain-specific factuality detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04177v1">Are LLMs Ready for Conflict Monitoring? Empirical Evidence from West Africa</a></div>
    <div class="paper-meta">
      📅 2026-05-05
    </div>
    <details class="paper-abstract">
      As LLMs enter conflict monitoring, understanding systematic distortions in their outputs is critical for humanitarian accountability. We evaluate four vanilla open-weight models Gemma 3 4B, Llama 3.2 3B, Mistral 7B, and OLMo 2 7B and two domain-adapted models, AfroConfliBERT and AfroConfliLLAMA, on Nigeria and Cameroon conflict-event classification against ACLED, a gold-standard dataset with multi-stage verification. We find a bifurcated divergence in normative directionality. Open-weight models exhibit statistically significant False Illegitimation bias: Gemma misclassifies to 18.29% of legitimate battles as civilian-targeted violence while making zero False Legitimation errors. By contrast, AfroConfliBERT and AfroConfliLLAMA achieve near-directional neutrality, with Legitimization Bias differences indistinguishable from zero. Yet domain adaptation does not eliminate actor-based selection bias. Both adapted models show statistically significant actor bias comparable to vanilla LLMs; in Nigeria, state actors are legitimized 36.5% more often than non-state actors in identical tactical contexts. Open-weight outputs are also fragile to geography-specific lexical framing: delegitimizing phrases produce flip rates up to 66.7% in Cameroon and 34.2% in Nigeria, while perturbations salient in one context may not matter in another. Error trace profiling shows models mask normative bias through unfaithful rationale confabulations. In contrast, AfroConfliBERT and AfroConfliLLAMA are largely robust, with near-zero flip rates across perturbation categories. Overall, current models are not ready for unsupervised deployment in conflict monitoring. We call for fairness-aware fine-tuning to reduce actor-based selection bias, mandatory adversarial robustness evaluation against lexical manipulation, and context-specific human-in-the-loop oversight calibrated to regional difficulty.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04029v1">Stayin' Aligned Over Time: Towards Longitudinal Human-LLM Alignment via Contextual Reflection and Privacy-Preserving Behavioral Data</a></div>
    <div class="paper-meta">
      📅 2026-05-05
    </div>
    <details class="paper-abstract">
      Current human-AI alignment and evaluation methods for large language models (LLMs) often rely on preference signals collected immediately after an interaction. This practice implicitly treats preference as static, even though many LLM-mediated decisions unfold over time and may be re-evaluated differently after real-world consequences and observed outcomes. Therefore, we argue for a methodological shift from single-moment preference elicitation to longitudinal, context-situated alignment measurement. We present a methodological framework for collecting temporally grounded alignment signals by combining (1) in-situ preference capture, (2) context-triggered follow-up preference reflection, and (3) privacy-preserving behavioral traces that help interpret preference change. As an instantiation of this methodology, we introduce BITE, a browser-based system that detects consequential LLM interactions, prompts reflection across later decision points, and supports progressive, user-controlled consent for sharing behavioral data. Through a two week longitudinal deployment study with 8 participants, our approach surfaced differences between immediate and later user preferences in accuracy, relevance and other dimensions of the LLM output. Our findings highlight the limitations of single-moment preference datasets and underscore the importance of longitudinal methods for alignment evaluation in everyday use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.03998v1">EQUITRIAGE: A Fairness Audit of Gender Bias in LLM-Based Emergency Department Triage</a></div>
    <div class="paper-meta">
      📅 2026-05-05
      | 💬 37 pages, 10 figures, 13 tables. Code and analysis scripts available upon publication. Data: PhysioNet credentialed access (MIMIC-IV-ED v2.2 and MIMIC-IV v3.1, BIDMC IRB #2001P001699)
    </div>
    <details class="paper-abstract">
      Emergency department triage assigns patients an acuity score that determines treatment priority, and clinical evidence documents persistent gender disparities in human acuity assessment. As hospitals pilot large language models (LLMs) as triage decision support, a critical question is whether these models reproduce or mitigate known biases. We present EQUITRIAGE, a fairness audit of LLM-based ESI assignment evaluating five models (Gemini-3-Flash, Nemotron-3-Super, DeepSeek-V3.1, Mistral-Small-3.2, GPT-4.1-Nano) across 374,275 evaluations on 18,714 MIMIC-IV-ED vignettes under four prompt strategies. Of 9,368 originals, 9,346 are paired with a gender-swapped counterfactual. All five models produced flip rates above a pre-registered 5% threshold (9.9% to 43.8%). Two showed directional female undertriage (DeepSeek F/M 2.15:1, Gemini 1.34:1); two were near-parity; one had high sensitivity with weak male-direction asymmetry. DeepSeek's directional bias coexisted with a low outcome-linked calibration gap (0.013 against MIMIC-IV admission), a Chouldechova-style dissociation between within-group calibration and between-pair counterfactual invariance. Demographic blinding reduced Gemini's flip rate to 0.5%; an age-preserving blind variant left DeepSeek with residual F/M 1.25, implicating age as a residual channel. Chain-of-thought prompting degraded accuracy for all five models. A two-model ablation reveals opposite underlying mechanisms for the same directional phenotype: in Gemini the signal is emergent in the combined name+gender swap, while in DeepSeek the gender token alone carries it. EQUITRIAGE shows that group parity, counterfactual invariance, and gender calibration are distinct fairness properties, that intervention effectiveness is model-dependent, and that per-model counterfactual auditing should precede clinical deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.09457v2">Power-Softmax: Towards Secure LLM Inference over Encrypted Data</a></div>
    <div class="paper-meta">
      📅 2026-05-05
    </div>
    <details class="paper-abstract">
      Modern cryptographic methods for implementing privacy-preserving LLMs such as \gls{HE} require the LLMs to have a polynomial form. Forming such a representation is challenging because transformers include non-polynomial components, such as \Softmax and layer normalization. Previous approaches have either directly approximated pre-trained models with large-degree polynomials, which are less efficient over HE, or replaced non-polynomial components with easier-to-approximate primitives before training, e.g., \Softmax with pointwise attention. The latter approach might introduce scalability challenges. We present a new HE-friendly variant of self-attention that offers a stable form for training and is easy to approximate with polynomials for secure inference. Our work introduces the first polynomial LLMs over a billion parameters, exceeding the size of previous models by more than tenfold. The resulting models demonstrate reasoning and in-context learning (ICL) capabilities comparable to standard transformers of the same size, representing a breakthrough in the field. Finally, we provide a detailed latency breakdown for each computation over encrypted data, paving the way for further optimization, and explore the differences in inductive bias between models relying on our HE-friendly variant and standard transformers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.03971v1">Logical Consistency as a Bridge: Improving LLM Hallucination Detection via Label Constraint Modeling between Responses and Self-Judgments</a></div>
    <div class="paper-meta">
      📅 2026-05-05
      | 💬 ACL 2026 Main Conference
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are prone to factual hallucinations, risking their reliability in real-world applications. Existing hallucination detectors mainly extract micro-level intrinsic patterns for uncertainty quantification or elicit macro-level self-judgments through verbalized prompts. However, these methods address only a single facet of the hallucination, focusing either on implicit neural uncertainty or explicit symbolic reasoning, thereby treating these inherently coupled behaviors in isolation and failing to exploit their interdependence for a holistic view. In this paper, we propose LaaB (Logical Consistency-as-a-Bridge), a framework that bridges neural features and symbolic judgments for hallucination detection. LaaB introduces a "meta-judgment" process to map symbolic labels back into the feature space. By leveraging the inherent logical bridge where response and meta-judgment labels are either the same or opposite based on the self-judgment's semantics, LaaB aligns and integrates dual-view signals via mutual learning and enhances the hallucination detection. Extensive experiments on 4 public datasets, across 4 LLMs, against 8 baselines demonstrate the superiority of LaaB.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.03943v1">Beyond Rules: LLM-Powered Linting for Quantum Programs</a></div>
    <div class="paper-meta">
      📅 2026-05-05
    </div>
    <details class="paper-abstract">
      As quantum computing transitions from theoretical experimentation to its practical application, the reliability of quantum software has become a critical bottleneck. Traditional static analysis techniques for quantum programs, primarily rule-based linters, are increasingly inadequate; they struggle to keep pace with rapidly evolving APIs and fail to capture complex, context-dependent quantum programming problems. This results in high maintenance overhead and limited detection capabilities. In this paper, we introduce LintQ-LLM+CoT and LintQ-LLM+RAG, novel approaches that redefine the detection of quantum programming problems by employing Large Language Models (LLMs) specialized, respectively, via Chain-of-Thought (CoT) prompting and a Retrieval-Augmented Generation (RAG) system that grounds the model's reasoning in a curated knowledge base of verified quantum programming problems and best practices. We conducted a rigorous and manual comparative evaluation against the state-of-the-art rule-based tool, LintQ, using a corpus of 55 Qiskit programs. Our results show that LLM-based approaches, with and without RAG, outperform LintQ in terms of quantum programming problems detection correctness (precision) and completeness (recall). Overall, LLM-based approaches were more effective than LintQ (F1-score equal to 0.70 and 0.68 vs. 0.41). Furthermore, the RAG-enhanced variant demonstrated a slightly superior precision, effectively reducing false positives. Our findings suggest that LLMs provide a scalable and adaptive foundation for the next generation of linters in quantum software engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.01704v2">The Reasoning Trap: An Information-Theoretic Bound on Closed-System Multi-Step LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-05-05
      | 💬 23 pages, 18 figures, 4 tables, 126 references. Subtitle: A Falsifiable Theorem, the Multi-Agent-Debate Instantiation, and a Triple Failure of Human Reliability
    </div>
    <details class="paper-abstract">
      When copies of the same language model are prompted to debate, they produce diverse phrasings of one perspective rather than diverse perspectives. Multi-agent debate (MAD), and more broadly closed-system reasoning where agents iteratively transform each other's outputs, tends to preserve answer accuracy while degrading the reasoning behind those answers. We name the multi-agent case the Debate Trap and the broader phenomenon the Reasoning Trap, offering a programmatic theory of evidence-grounded reasoning failure.The framework has three parts: (i) SFS (Supported Faithfulness Score), a claim-level metric verifying decomposed atomic claims against provided evidence (decomposer-invariant rankings: Spearman rho=1.0); (ii) EGSR (Evidence-Grounded Socratic Reasoning), replacing adversarial argumentation with evidence-grounded inquiry; (iii) Theorem 1 (DPI Bound): under standard MAD, the chain E -> O^0 -> O^1 -> ... is Markov, and the Data Processing Inequality implies E[I(E;O^{t+1})] <= E[I(E;O^t)]. Three companion results -- open-system recovery (Theorem 2), EGSR accumulation (Lemma 2), and vote-aggregation floor (Proposition 1) -- partition multi-step LLM reasoning by its information-theoretic relationship to E. Across 16 conditions on SciFact (300 claims) and FEVER (1,000 claims), DebateCV (C13) preserves 88% of baseline accuracy while SFS drops 43%; majority-vote MAD (C15) reduces SFS to 1.7% of baseline (p < 10^{-6}, d = -0.96); EGSR recovers 98%. An R6 cohort study (Korean n=10x30 FEVER; English n=3x200 SciFact) finds inter-rater Fleiss kappa <= +0.018 with 0.8-1.4 Likert intra-rater shifts across language and domain -- the human agreement that faithfulness metrics have been calibrated against is not itself stable. We offer one falsifiable conjecture: any closed-system reasoning protocol preserving Theorem 1's Markov structure is, in expectation, subject to the same DPI bound.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21072v2">Distributed Generative Inference of LLM at Internet Scales with Multi-Dimensional Communication Optimization</a></div>
    <div class="paper-meta">
      📅 2026-05-05
    </div>
    <details class="paper-abstract">
      Decentralized LLM inference distributes computation among heterogeneous nodes across the internet, offering a performant and cost-efficient solution, alternative to traditional centralized inference. However, the low cross-node network bandwidth makes communication the primary bottleneck. In this paper, we introduce BloomBee, an internet-scale distributed LLM inference framework. BloomBee integrates LLM-layer assignment, micro-batching and tensor offloading to optimize communication from multiple dimensions. Additionally, BloomBee formulates the coordination of these techniques as an optimization problem and solves it using dynamic programming. BloomBee also customizes lossless compression and speculative decoding according to low-bandwidth network settings to reduce communication overhead. We evaluate BloomBee across a spectrum of network environments and show that it improves service throughput by up to 1.76x. It also reduces average latency by up to 43.20% compared to state-of-the-art decentralized LLM inference systems. BloomBee is open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.03884v1">QKVShare: Quantized KV-Cache Handoff for Multi-Agent On-Device LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-05
      | 💬 12 pages, 1 figure, 3 tables
    </div>
    <details class="paper-abstract">
      Multi-agent LLM systems on edge devices need to hand off latent context efficiently, but the practical choices today are expensive re-prefill or full-precision KV transfer. We study QKVShare, a framework for quantized KV-cache handoff between agents that combines token-level mixed-precision allocation, a self-contained CacheCard representation, and a HuggingFace-compatible cache injection path. Our current results support a narrower but clearer story than the original draft: on 150 GSM8K problems with Llama-3.1-8B-Instruct, adaptive quantization remains competitive under repeated handoff and shows its clearest gains against uniform quantization in deeper-hop, higher budget settings; for handoff latency, the QKVShare path reduces TTFT relative to full re prefill at every tested context, from 130.7 ms vs. 150.2 ms at nominal 1K context to 397.1 ms vs. 1029.7 ms at nominal 8K context;. Stage timing shows that post-injection generation, not card creation, dominates the current QKVShare latency path. These results position quantized KV handoff as a promising on-device systems direction while also highlighting the need for stronger controller ablations and apples-to-apples runtime comparisons.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.03792v1">TriBench-Ko: Evaluating LLM Risks in Judicial Workflows</a></div>
    <div class="paper-meta">
      📅 2026-05-05
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly integrated into legal workflows. However, existing benchmarks primarily address proxy tasks, such as bar examination performance or classification, which fail to capture the performance and risks inherent in day-to-day judicial processes. To address this, we publicly release TriBench-Ko, a Korean benchmark designed to evaluate potential deployment risks of LLMs within the context of verified judicial task requirements. It covers four core tasks: jurisprudence summarization, precedent retrieval, legal issue extraction, and evidence analysis. It jointly assesses model behavior across multiple deployment risk categories, including inaccuracy (hallucination, omission, statutory misapplication), biases (demographic, overcompliance), inconsistencies (prompt sensitivity, non-determinism), and adjudicative overreach. Each item is structured to systematically assess both task performance and a specific risk type based on real judicial decisions. Our evaluation of a range of contemporary LLMs reveals that many models frequently manifest significant risks, most notably struggling with precedent retrieval and failing to capture critical legal information. We provide a comprehensive diagnosis of these LLMs and pinpoint critical areas where LLM-generated outputs in judicial contexts necessitate rigorous inspection and caution. Our dataset and code are available at https://github.com/holi-lab/TriBench-Ko
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.03788v1">Say the Mission, Execute the Swarm: Agent-Enhanced LLM Reasoning in the Web-of-Drones</a></div>
    <div class="paper-meta">
      📅 2026-05-05
      | 💬 15 pages, 5 figures. This paper has been accepted for presentation at the 27th IEEE International Symposium on a World of Wireless, Mobile and Multimedia Networks (WoWMoM 2026)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly explored as high-level reasoning engines for cyber-physical systems, yet their application to real-time UAV swarm management remains challenging due to heterogeneous interfaces, limited grounding, and the need for long-running closed-loop execution. This paper presents a mission-agnostic, agent-enhanced LLM framework for UAV swarm control, where users express mission objectives in natural language and the system autonomously executes them through grounded, real-time interactions. The proposed architecture combines an LLM-based Agent Core with a Model Context Protocol (MCP) gateway and a Web-of-Drones abstraction based on W3C Web of Things (WoT) standards. By exposing drones, sensors, and services as standardized WoT Things, the framework enables structured tool-based interaction, continuous state observation, and safe actuation without relying on code generation. We evaluate the framework using ArduPilot-based simulation across four swarm missions and six state-of-the-art LLMs. Results show that, despite strong reasoning abilities, current general-purpose LLMs still struggle to achieve reliable execution - even for simple swarm tasks - when operating without explicit grounding and execution support. Task-specific planning tools and runtime guardrails substantially improve robustness, while token consumption alone is not indicative of execution quality or reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.03762v1">OracleProto: A Reproducible Framework for Benchmarking LLM Native Forecasting via Knowledge Cutoff and Temporal Masking</a></div>
    <div class="paper-meta">
      📅 2026-05-05
    </div>
    <details class="paper-abstract">
      Large language models are moving from static text generators toward real-world decision-support systems, where forecasting is a composite capability that links information gathering, evidence integration, situational judgment, and action-oriented decision making. This capability is in broad demand across finance, policy, industry, and scientific research, yet its evaluation remains difficult: live benchmarks evaluate forecasts before answers exist, making them the cleanest way to measure forecasting ability, but they expire once events resolve; retrospective benchmarks are reproducible, but they cannot reliably distinguish genuine forecasting from facts a model may have already learned during pretraining. Prompting models to "pretend not to know" cannot replace a genuine knowledge boundary. We propose OracleProto, a reproducible framework for evaluating LLM native forecasting capability. OracleProto reconstructs resolved events into time-bounded forecasting samples by combining model-cutoff-aligned sample admission, tool-level temporal masking, content-level leakage detection, discrete answer normalization, and hierarchical scoring. Instantiated on a FutureX-Past-derived dataset with six contemporary LLMs, OracleProto distinguishes forecasting quality, sampling stability, and cost efficiency under controlled information boundaries, while reducing residual leakage to the $1\%$ level, an order of magnitude below tool-only temporal filtering. OracleProto turns LLM forecasting from one-off evaluation into an auditable, reusable, and trainable dataset-level capability, providing a unified interface for fair cross-model comparison and a controlled signal source for downstream SFT and RL. Code and data are available at https://github.com/MaYiding/OracleProto and https://huggingface.co/datasets/MaYiding/OracleProto.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20284v2">Can LLMs Make (Personalized) Access Control Decisions?</a></div>
    <div class="paper-meta">
      📅 2026-05-05
    </div>
    <details class="paper-abstract">
      Precise access control decisions are crucial for the security of both traditional applications and emerging agent-based systems. Typically, these decisions are made by users during app installation or at runtime. However, due to the increasing complexity and automation of systems, making access control decisions can impose a significant cognitive burden on users, often overwhelming them and leading to suboptimal or even arbitrary choices. To address this problem, we investigate the ability of LLMs to make dynamic, context-aware decisions aligned with users' security preferences, expressed during a lightweight setup phase. As a case study, we analyze smartphone application permission requests, given their ubiquity and users' familiarity with them. We curated a dataset comprising 307 user privacy statements (short, natural-language descriptions of user preferences) and 14,682 corresponding permission decisions, gathered from smartphone users in an online data collection. We compare these decisions with those made by two versions of LLMs that are tasked with reasoning about the app and the request context: a general model and a personalized one (which incorporates user preferences). For the latter, we also collected user feedback on 1,298 of its decisions. Our results show that LLMs generally reflect users' preferences well, agreeing with the majority decision in up to 86% of cases, and can steer users toward safer behavior. However, the results also reveal a key trade-off in personalization: while incorporating user-specific privacy preferences improves agreement with individual decisions, strict adherence to these preferences may lead to less safe outcomes, as users tend to over-permission.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.03723v1">Segmenting Human-LLM Co-authored Text via Change Point Detection</a></div>
    <div class="paper-meta">
      📅 2026-05-05
    </div>
    <details class="paper-abstract">
      The rise of large language models (LLMs) has created an urgent need to distinguish between human-written and LLM-generated text to ensure authenticity and societal trust. Existing detectors typically provide a binary classification for an entire passage; however, this is insufficient for human--LLM co-authored text, where the objective is to localize specific segments authored by humans or LLMs. To bridge this gap, we propose algorithms to segment text into human- and LLM-authored pieces. Our key observation is that such a segmentation task is conceptually similar to classical change point detection in time-series analysis. Leveraging this analogy, we adapt change point detection to LLM-generated text detection, develop a weighted algorithm and a generalized algorithm to accommodate heterogeneous detection score variability, and establish the minimax optimality of our procedure. Empirically, we demonstrate the strong performance of our approach against a wide range of existing baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14709v3">HWE-Bench: Benchmarking LLM Agents on Real-World Hardware Bug Repair Tasks</a></div>
    <div class="paper-meta">
      📅 2026-05-05
    </div>
    <details class="paper-abstract">
      Existing benchmarks for hardware design primarily evaluate Large Language Models (LLMs) on isolated, component-level tasks such as generating HDL modules from specifications, leaving repository-scale evaluation unaddressed. We introduce HWE-Bench, the first large-scale, repository-level benchmark for evaluating LLM agents on real-world hardware bug repair tasks. HWE-Bench comprises 417 task instances derived from real historical bug-fix pull requests across six major open-source projects spanning both Verilog/SystemVerilog and Chisel, covering RISC-V cores, SoCs, and security roots-of-trust. Each task is grounded in a fully containerized environment where the agent must resolve a real bug report, with correctness validated through the project's native simulation and regression flows. The benchmark is built through a largely automated pipeline that enables efficient expansion to new repositories. We evaluate seven LLMs with four agent frameworks and find that the best agent resolves 70.7% of tasks overall, with performance exceeding 90% on smaller cores but dropping below 65% on complex SoC-level projects. We observe larger performance gaps across models than commonly reported on software benchmarks, and difficulty is driven by project scope and bug-type distribution rather than code size alone. Our failure analysis traces agent failures to three stages of the debugging process: fault localization, hardware-semantic reasoning, and cross-artifact coordination across RTL, configuration, and verification components, providing concrete directions for developing more capable hardware-aware agents.
    </details>
</div>
