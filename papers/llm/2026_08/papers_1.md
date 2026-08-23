# llm - 2026_08

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.20280v1">Which Eviction Policy Should an LLM Cache Use? A Systematic Study Across Workloads, Capacities, and Encoders</a></div>
    <div class="paper-meta">
      📅 2026-08-20
      | 💬 11 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Semantic caches reuse an LLM response when the incoming query embedding lies near a cached query, but proposed eviction policies have rarely been compared under one protocol. Using CLEVER, we evaluate FIFO, LRU, LFU, ARC, GDSF, a single-pass streaming adaptation of SISO, and a semantic-redundancy policy across three ordered, deduplicated query corpora, three cache capacities, and two encoders. No evaluated policy improves on LFU by more than 0.041 percentage points in any of the eighteen settings. Replacement is not irrelevant: FIFO and streaming SISO trail LFU by as much as 8.67 and 8.55 points, respectively, at tight capacity. We explain the missing upside with a conditional packing result. Under exact lookup and insert-on-miss, a newly inserted entry cannot have a resident neighbor within the hit radius, so a geometry-aware eviction rule receives little new redundancy signal. A separate audit exposes a larger problem with the evaluated operating point. At MiniLM's median nearest-neighbor threshold, only 2.1-3.9% of sampled LMSYS and QQP hits are judged answer-substitutable, reducing raw hit rates of 51-60% to quality-adjusted rates of 1.1-2.2%. The cross-encoder study further shows that thresholds do not transfer between embedding models. LFU is the strongest simple default in this protocol; deployment decisions should first establish answer validity and then test sub-point policy differences with exact search.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.20274v1">Break It Down, Pass It On: Cross-Task Skill Transfer in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-20
      | 💬 34 pages, 28 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents can induce skills from completed tasks and reuse them later to grow more capable with experience. In practice, induced skills may transfer unreliably and can even harm the agent that retrieves them. When agent-induced skills transfer reliably across tasks remains an open question. We conduct a comprehensive and controlled study of how the way skills are induced shapes their transfer across tasks. Specifically, we compare task-level with subtask-level skill induction and text with code skill formats, the two axes along which existing methods differ. Task-level skills mostly reduce the agent's performance below its no-memory baseline while subtask-level skills raise it above on average, and text skills transfer better than code skills. To further understand our findings, we examine two complementary properties of the induced skills: specificity, which measures how closely a skill matches real tasks, and abstractness, which measures how evenly its relevance spreads across tasks. Neither property alone predicts task success, but their combined effect does, which we propose as a skill utility score. The score correlates consistently with task success when skills are transferred, and subtask-level and text skills score higher. Computing skill utility only needs the skills and task descriptions but not any task execution, so our score serves as a practical diagnostic of a skill memory before any new task runs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.06127v2">PrefixAgent: An LLM-Powered Design Framework for Efficient Prefix Adder Optimization</a></div>
    <div class="paper-meta">
      📅 2026-08-20
    </div>
    <details class="paper-abstract">
      Prefix adders are fundamental arithmetic circuits, but their design space grows exponentially with bit-width, posing significant optimization challenges. Previous works face limitations in performance, generalization, and scalability. To address these challenges, we propose PrefixAgent, an LLM-powered framework that enables efficient prefix adder optimization. Firstly, we reformulate the problem into two subtasks, namely backbone synthesis and structure refinement, which effectively reduce the search space. The LLM performs these two phases of optimization by invoking tools through function calls, enabling it to iteratively construct the backbone and refine local structures based on reasoning and EDA feedback. Secondly, this new design perspective allows us to systematically collect large-scale, high-quality supervision data. We leverage the rewriting and equality saturation capabilities of e-graphs to comprehensively explore the prefix adder solution space. In addition, the explainability of e-graphs enables us to extract fine-grained rewrite trajectories, which serve as interpretable and effective training data. PrefixAgent is then fine-tuned on this dataset, significantly enhancing its optimization ability and reasoning generalization. Experimental results show that PrefixAgent synthesizes prefix adders with smaller areas than baseline methods in nearly all configurations, with the advantage growing at larger bit-widths, and it remains effective under a commercial EDA flow.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.20202v1">MemTrapBench: Benchmarking Cognitive Traps in LLM Memory Use</a></div>
    <div class="paper-meta">
      📅 2026-08-20
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Memory has become a key component of large language models, enabling them to retain information and learn from long-term interactions. However, existing memory benchmarks mainly evaluate whether information is correctly extracted, stored, and retrieved, while largely overlooking how retrieved memories reshape model reasoning and affect performance on the current task. We identify memory-induced cognitive traps: even faithfully recorded and semantically relevant memories can distort model reasoning or beliefs and degrade current task performance. To systematically evaluate these failure modes, we introduce MemTrapBench, which covers two forms of cognitive traps: Reasoning Fixation and Belief Distortion. Experiments across two model families and five representative memory frameworks show that MemTrapBench is challenging: all evaluated memory strategies underperform the no-memory setting, with even the strongest methods suffering drops of more than 10%. To mitigate these cognitive traps, we propose AdaptiveMem, a simple yet effective inference-time method that instructs LLMs to avoid memory traps. AdaptiveMem mitigates cognitive traps on MemTrapBench while preserving or improving performance on standard memory benchmarks across diverse memory frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06635v2">Can LLM Annotations Replace User Clicks for Learning to Rank?</a></div>
    <div class="paper-meta">
      📅 2026-08-20
      | 💬 12 pages, 7 figures, accepted as a Full Research Paper at CIKM 2026
    </div>
    <details class="paper-abstract">
      Large-scale supervised data is essential for training modern ranking models, but obtaining high-quality human annotations is costly. Click data has been widely used as a low-cost alternative, and with recent advances in large language models (LLMs), LLM-based relevance annotation has emerged as another promising annotation. This paper investigates whether LLM annotations can replace click data for learning to rank (LTR) by conducting a comprehensive comparison across multiple dimensions. Experiments on both a public dataset, TianGong-ST, and an industrial dataset, Baidu-Click, show that click-supervised models perform better on high-frequency queries, while LLM annotation-supervised models are more effective on medium- and low-frequency queries. Further analysis shows that click-supervised models are better at capturing document-level signals such as authority or quality, while LLM annotation-supervised models are more effective at modeling semantic matching between queries and documents and at distinguishing relevant from non-relevant documents. Motivated by these observations, we explore two training strategies -- data scheduling and frequency-aware multi-objective learning -- that integrate both supervision signals. Both approaches enhance ranking performance across queries at all frequency levels, with the latter being more effective. Our code is available at https://github.com/Trustworthy-Information-Access/LLMAnn_Click.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.20167v1">BreakGuard: Towards Detecting Dependency Breaking Changes with LLM-Generated Tests</a></div>
    <div class="paper-meta">
      📅 2026-08-20
    </div>
    <details class="paper-abstract">
      Open-source libraries play an important role in software development by providing reusable features that expedite the development process. As libraries evolve, they release new versions that add features, fix bugs, or apply security patches. In this process, they may break the contract established with their clients by introducing breaking changes (BCs) that alter the runtime behavior and break client applications. Client-side test suites often fail to detect these BCs because of limited library coverage that does not exercise all library methods used in the client's codebase. We propose BreakGuard, an approach that generates a test suite to detect breaking changes in clients. BreakGuard statically extracts every client method (focal method) that invokes the target library method (call site), then generates tests per focal method. A test detects a BC if it passes on the pre-breaking version and fails on the breaking version. We evaluate our approach on 89 real-world breaking changes from the BUMP dataset, using 3 LLMs (GPT4o, Qwen3-coder-480B, GPT-OSS-120B) and three context levels: minimal, method, and class. Using the best-performing configuration, BreakGuard detects 30.3% of breaking changes (27 of 89) at a mean cost of roughly $0.90 USD per detected breaking change. We successfully detected BCs from different library categories (e.g., JSON libraries, logging, parsing), but we find LLM-generated tests to be more reliable for detecting crash-type breaking changes as opposed to behavioural BCs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.20129v1">Multi-Agent Orchestration with the Common-Sense Reasoning Capabilities of LLMs for Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2026-08-20
      | 💬 17 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Autonomous vehicles require robust perception and decision-making capabilities to operate in diverse and unseen scenarios. While reinforcement learning and rule-based methods can provide effective control and safety mechanisms, their performance may degrade in situations requiring contextual reasoning. Large Language Models (LLMs) have demonstrated strong capabilities in understanding multimodal information and generating contextual reasoning, however, their use for direct vehicle control can introduce latency and hallucination risks. To address these limitations, a hybrid framework is proposed. This system uses an orchestrator to coordinate PPO-trained reinforcement learning and PID control, with LLM common-sense reasoning applied throughout the framework. LLM reasoning is further employed iteratively to refine the RL reward function for dynamic driving environments. The proposed framework is evaluated in highly randomized CARLA scenarios under diverse environmental and traffic conditions. The results demonstrate the potential of integrating LLM-based reasoning with conventional autonomous driving methods while retaining structured control and safety mechanism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.19998v1">SCoRD: Semantic-Assisted Continual Retriever-Reranker Distillation for LLM-Based Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-08-20
    </div>
    <details class="paper-abstract">
      Recommendation systems increasingly adopt a two-stage pipeline, where an ID-based retriever retrieves candidates and an LLM-based reranker refines their rankings. To improve retrieval quality, reranker-to-retriever distillation is commonly used to transfer the reranker's knowledge to the retriever. For practical deployment, however, this pipeline must continually adapt to evolving interests and incoming interactions. A naive solution is to repeatedly update the LLM reranker and distill its latest knowledge, but this incurs prohibitive costs. Updating the retriever alone is cheaper, but its limited capacity makes adaptation from sparse data difficult. We propose SCoRD, a continual knowledge distillation framework for LLM-based reranking pipelines under a non-stationary data stream. SCoRD introduces a semantic reasoning assistant that distills the LLM's ability to infer underlying user intents into reusable intent-level guidance. It selectively distills reranker knowledge to the retriever on low-confidence sequences, guides retriever-only updates without repeated LLM inference, and feeds retriever-derived representations and intent-drift signals back to the reranker. Experiments on real-world datasets show that SCoRD enables effective and efficient retriever-reranker co-adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.19993v1">Optimal Skill Selection for LLM Agents with Provable Bicriteria Guarantees</a></div>
    <div class="paper-meta">
      📅 2026-08-20
    </div>
    <details class="paper-abstract">
      Loading reusable skill documents into a bounded context window is now the primary way large language model (LLM) agents acquire task-specific capabilities, which makes skill selection a first-order determinant of task performance and token cost. Yet current agents score skills independently by semantic relevance and assemble the set by top-$k$ or greedy packing, with no quality guarantee or cost awareness on the selected set. As a result, redundant or poorly chosen skills waste scarce context tokens and can even degrade performance. We give the first model of how the selected skill set shapes execution outcomes and cast skill selection as an optimization problem: choose a skill set under a hard token budget to maximize a monotone submodular benefit minus context penalty. For this problem, we develop Best Prefix Selection (BPS), a polynomial-time algorithm, and prove, to our knowledge, the first performance guarantee for skill selection: a bicriteria $(1-1/e,1)$ approximation whose benefit coefficient is optimal in polynomial time. On a contamination-controlled BigCodeBench variant, BPS outperforms all the baselines, reaching $0.73$ measured task success versus $0.20$--$0.52$ for released skill routers, text retrievers, and the executor's own selection, on $28\%$ fewer tokens than the strongest released router.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.19974v1">ReguSim: Evaluating LLM Agent Rule Grounding in Financial Compliance</a></div>
    <div class="paper-meta">
      📅 2026-08-20
    </div>
    <details class="paper-abstract">
      LLM agents in financial markets may cite rules yet still submit orders that violate executable constraints or misread surveillance evidence. We introduce ReguSim, a controlled financial-compliance environment, and ReguBench, a target-marked monitoring benchmark, to separate four artifacts: stated reasoning, attempted action, execution enforcement, and monitor evidence. In trader runs with DeepSeek V4 Pro and Gemini 3.5 Flash, visible rules reduce but do not eliminate rejected actions, and incentive or persona framing shifts behavior. A bridge study shows that trader rationales can mislead an independent monitor unless enforcement evidence is shown. In monitoring, simple structured baselines either match or exceed prompt-only LLMs. The results frame financial compliance evaluation as an audit of rule-grounded actions and evidence use, rather than a single compliance score.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.19938v1">From Noise to Signal: Improving Security Log Anomaly Detection Using LLMs with Endpoint-Specific Logs</a></div>
    <div class="paper-meta">
      📅 2026-08-20
      | 💬 The paper contains 35 pages and 3 figures. The paper has not been submitted or published in any conference or journal. The authors have an aim to publish it in a journal
    </div>
    <details class="paper-abstract">
      Existing approaches to anomalous behaviour log detection, such as Wazuh rely primarily on predefined detection rules, while statistical anomaly detection approaches such as OpenSearch identify deviations from previously observed behavioural patterns. Recent research has investigated LLMs for log anomaly detection because of their ability to interpret semantic and contextual information. However, LLM-based approaches can be affected by prompt construction, noisy log data, and reliance on generic datasets that may lack endpoint-specific authentication behaviours. To address these limitations, this study develops a standardised instruction-based LLM classification framework for detecting anomalous authentication behaviours, including borderline cases. A controlled cybersecurity testbed was developed to generate endpoint-specific authentication data, producing a curated dataset comprising normal, borderline, and anomalous behavioural scenarios. Three instruction-tuned LLMs, Meta Llama 3.1 8B Instruct, Qwen 2.5 7B Instruct, and GPT-OSS 20B, were evaluated against Wazuh rule-based detection and OpenSearch Anomaly Detection using a common ground-truth severity framework. Meta Llama 3.1 8B Instruct achieved the strongest overall end-to-end detection performance, with an accuracy of 89.3%, recall of 88.2%, F1-score of 91.8%, and false negative rate of 11.8%. In comparison, Wazuh achieved an accuracy of 52.0% and false negative rate of 68.6%, while OpenSearch achieved an accuracy of 49.3% and false negative rate of 74.5%. Meta Llama also detected 80% of the borderline anomalous scenarios, compared with 20% for Wazuh and 15% for OpenSearch. Qwen achieved lower overall detection performance than Meta Llama but recorded the lowest average inference latency and 100% structured-response validity. GPT-OSS demonstrated strong classification performance when valid responses were produced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17168v3">RepSelect: Robust LLM Unlearning via Representation Selectivity</a></div>
    <div class="paper-meta">
      📅 2026-08-20
    </div>
    <details class="paper-abstract">
      When LLM weights are open or fine-tuning is available through an API, suppressing hazardous knowledge and tendencies is not enough: removal has to be deep enough that an adversary cannot restore it. Existing unlearning is shallow by this standard: fine-tuning or a handful of in-context examples brings the behaviour back, and it often degrades general capabilities in the process. We identify a root cause: existing methods edit representations shared with the retain set and lying in the subspace that a fine-tuning attacker recovers, making unlearning simultaneously easy to undo and disruptive. Leveraging this, we propose RepSelect (Representation Selectivity), which isolates forget-set-specific representations by collapsing the top principal components of the weight gradients before each unlearning update, preserving general capabilities while limiting what fine-tuning can recover. Across five unlearning datasets spanning both knowledge (biohazard, cyber, facts about real individuals) and tendencies (abusive, sycophantic), and three model families covering dense and Mixture-of-Experts architectures, RepSelect yields a 4-40x larger drop in post-relearning answer probability than five widely used baselines (GradDiff, NPO, SimNPO, RMU, UNDIAL). It is also near-perfectly robust to few-shot prompting and holds under an adaptive attack designed to exploit its mechanism. Our results show that unlearning needs to be selective about which representations it edits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.19889v1">Write Once, Run Everywhere: The Axon DSL for Shape-Safe and Framework-Agnostic LLM Architectures</a></div>
    <div class="paper-meta">
      📅 2026-08-20
    </div>
    <details class="paper-abstract">
      The entire ecosystem of open-source language models effectively relies on a single platform. What if this platform was forced to shut down tomorrow? Implementing and maintaining efficient model definitions and translating them between different training and inference regimes is a resource-heavy task that severely limits model efficiency and portability, hindering both scaling and deployment. Here, we present Axon, a strongly typed domain-specific language with Haskell-like syntax, that enables a write-once, run everywhere paradigm for LLM architectures. By basing collaboration on a language specification rather than a specific framework's vision, Axon fosters open cooperation and empowers researchers to implement highly specialized architectures without giving up optimization infrastructure or accepting deployment lock-in. Axon allows for concise, auditable specifications that can be automatically compiled to standalone implementations for leading frameworks: PyTorch, PyTorch with Triton, JAX, MLX and vLLM. In 467 inference benchmarking experiments on models ranging from 135M to 32B parameters, we demonstrate median speedups of 7% on PyTorch, 12% on PyTorch with Triton, 91% on JAX, and 107% on MLX, compared to the reference implementations from Transformers. When deployed as native vLLM architectures with PagedAttention and KV-cache, Axon models achieve a 58% median speedup over Transformers implementations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.19861v1">PolicyGuide: From Guarding One Action to Guiding the Whole Workflow for Policy-Compliant LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-20
      | 💬 26 pages, 15 figures, including appendices
    </div>
    <details class="paper-abstract">
      Customer-service LLM agents must follow organizational policy when acting on a user's behalf. Compliance failures arise from either forbidden actions, such as granting an ineligible change, or omitted procedural requirements, such as identification or confirmation. Runtime safeguards can intervene on risky actions, but action-local checks do not guide an agent through a multi-step procedure. Workflow-following systems support prescribed process execution, but primarily target workflow completion rather than safeguarding agent behavior. PolicyGuide instead compiles each domain policy into a workflow graph and invokes a proactive verifier at user-turn boundaries. From persisted graph state, the verifier reconciles open requests and returns step-specific remediation along a policy-compliant path. Across the $τ^2$-bench airline, retail, and telecom domains with a GPT-5.4 agent and verifier, PolicyGuide raises mean $\mathrm{Pass}^4$ from $0.42$ to $0.62$, with the largest gain on telecom ($0.19$ to $0.61$), the most workflow-structured domain. The same workflows transfer to Claude Sonnet 4.6 and Gemini 2.5 Pro agents. Complementary evaluations find the lowest observed attack-success rate under adversarial users and the strongest procedural compliance in an author-designed workflow-level validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.19802v1">Stopping and Routing LLM Judge Panels</a></div>
    <div class="paper-meta">
      📅 2026-08-20
      | 💬 21 pages, 2 figures, 20 tables. Accepted at WISE 2026
    </div>
    <details class="paper-abstract">
      LLM evaluation pipelines often have many candidate judges: general LLM-as-a-judge prompts, reward models, safety classifiers, confidence variants, and task-specific verifiers. The deployment question is not only which judge is best, but which judges should be called, on which examples, and when panel construction should stop. We formulate judge-panel design as a role-conditioned allocation problem. From a small labeled audit set, declared slices, and judge costs, the method estimates target-relative roles: copies add no conditional information, complements improve the global panel, and specialists help only on slices. These roles induce a policy: drop copies, add complements globally, route specialists conditionally, and stop when validation gain falls below a threshold. Across reasoning, code, safety, preference, reward-model, summarization, and math audits, the method is compared with single judges, flat panels, matched diversity heuristics, full-call stacking, reliability juries, and frugal cascades. The result is a regime map for judge calls: route specialists on deployable slices, stop in saturated verifier regimes, keep broad ensembles when their risk benefit is worth the cost, and ignore conditional copies. The output is a reusable, auditable call plan for the next evaluation batch.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.01034v2">A Finite-Calibration Regime Map for LLM Judge Panels</a></div>
    <div class="paper-meta">
      📅 2026-08-20
      | 💬 33 pages, 11 figures, 40 tables. Accepted at WISE 2026
    </div>
    <details class="paper-abstract">
      Deploying an LLM judge panel spends human labels on fitting a calibrator, constructing candidate judge paths, and validating which candidate to deploy. We study when finite labels should support a low-dimensional stacker or reliability model, and when an unrestricted joint output table is worth its cell-count and unseen-pattern cost. We cast this as a finite-calibration regime map and instantiate it as Finite-Calibration Panel Selection (FCPS), a validation selector over judge path, deployed panel size, and aggregator family with support diagnostics. Across RewardBench, LLMBar, SummEval, and Arena100K with a seven-judge pool, scalar/reliability aggregation has lower MSE than unrestricted joint-table calibration in 16 of 20 real dataset--budget cells by point estimate, while paired 95% intervals exclude zero in 11 cells; richer backoff/shrinkage tables narrow some gaps while preserving the finite-support bottleneck. Controlled calibration-growth data show the opposite regime: when labels contain a six-way interaction, the selected table grows to the interaction-bearing prefix and its MSE falls from 0.224 to 0.061 once unseen mass vanishes. The practical deployment question is whether the next judge's information is estimable under the available human labels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.19790v1">LLMs as Acquisition Policies for Finite-Pool Materials Optimization: A Controlled Study</a></div>
    <div class="paper-meta">
      📅 2026-08-20
    </div>
    <details class="paper-abstract">
      Discovering materials with desirable properties often requires searching large candidate spaces while experimental or computational evaluations remain costly. Active learning addresses this challenge by using previous observations to select which candidate to evaluate next, typically through probabilistic surrogate models. We investigate whether open-weight large language models (LLMs) can serve as standalone acquisition policies in this setting. We evaluate five LLMs across four retrospective finite-pool materials optimization tasks under different candidate-presentation strategies and compare them with random selection and conventional Gaussian-process methods. LLM policies generally reach the global optimum in fewer iterations than random selection, indicating that they provide a useful acquisition signal without task-specific training. Their performance relative to Gaussian-process methods is mixed: conventional acquisition performs better on most tasks, while LLMs match or outperform it in some settings. Performance varies substantially across tasks, models, initializations, and candidate presentations, with no LLM approach performing best across all tasks. Overall, open-weight LLMs show potential as acquisition policies for finite-pool materials search, although their reliability remains sensitive to the task and to how candidates and scientific context are presented.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.19760v1">Credit Without Ground Truth: Auditing Step-Level Credit Assignment in LLM Agents Against Executed Replay</a></div>
    <div class="paper-meta">
      📅 2026-08-20
      | 💬 49 pages, 7 figures. Pre-registered; frozen analysis plans and prompts included in the appendices. Under review
    </div>
    <details class="paper-abstract">
      Audited against causal ground truth from executed replay in a single-agent tool environment (ALFWorld), none of the step-level credit signals used to train LLM agents -- LLM-judge scores, outcome-conditioned logprob ratios, or the policy's own confidence -- identifies which steps causally matter better than chance. Existing evaluations grade these signals against annotated step *correctness*; we audit them against step *contribution* -- what re-sampling the policy's own alternatives at each decision point and rolling forward actually changes about the outcome -- and the two come apart. The ground truth itself is structured: causal contribution is sparse (30.5% of decision points where ground truth is defined carry measurable effect), and measurability is model-dependent -- the fraction of points with no policy-supported counterfactual differs by a factor of two (13.1% vs. 26.8%) between two similar-scale policies. The failure mode is identifiable: implicit credit echoes the policy's fluency (median rank correlation +0.75, replicating at +0.70 in a second family under a corrected instrument), while conditioning on the outcome adds no causal information (partial correlation -0.004, Qwen). A confidence-only router recovers pivotal steps at chance level, but cuts judge cost by 13.1% per turn (14.0% per trajectory). In a seven-arm pre-registered training experiment, no arm reliably outperforms the untrained policy, and the checkpoints' apparent instrument signature is fully explained by training dose -- sparser credit retains fewer examples, an order-of-magnitude spread in optimizer steps -- not credit content. Comparisons of credit rules must therefore match effective sample size, or they measure dose, not credit.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.19009v2">Grading the Graders: Verification Autonomy Levels (L0-L5) for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-08-20
      | 💬 v2: reproducibility study (kappa~0.8), agent-security case (PPMF), anchor semantics, 15+ fixes. Code and data: https://github.com/1549080929-debug/math_agent Keywords: LLM verification; verification autonomy; completeness; ground truth; trustworthy AI Writing and implementation assisted by an AI language model; all experiments, data, and research decisions are the author's own
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly paired with verifiers (step checkers, self-consistency filters, tool-based fact checkers, formal proof assistants) that claim to detect the model's errors. Yet the verification literature uses the word "level" to mean at least five different things: verification granularity, concept abstraction, risk tier, system-stack layer, and the epistemic source of the ground truth. We propose Verification Autonomy Levels (VAL), a meta-standard that classifies any verification scheme along a single axis: where does the verification spec come from, and what does the verdict guarantee? VAL ranges from L0 (LLM self-declaration; no deterministic anchor) through L2 (objective ground truth; correctness only) to L3/L4 (decidable systems with single-property or domain-level completeness), with L5 impossible in the unrestricted case. Central to VAL is the completeness blind spot: substitution- and sampling-based verifiers can confirm that proposed candidates hold, but cannot prove that no candidate was missed. We further identify a dichotomy the literature has not stated: completeness is reachable only for formally specifiable properties, whereas empirical open-world verification (fact-checking, diagnosis) caps at anchored correctness (L2). We document this gap empirically across four domains (symbolic mathematics, behavior monitoring, medical diagnosis, and code generation, the last a reverse validation with predictions stated before evidence) and in the strongest formal-verification baseline in our survey, whose authors note the verifier focuses on the correctness of each step. We show the levels of granularity, concept hierarchy, risk, and system stack are orthogonal to VAL, resolving a systematic conflation across 17 surveyed papers. Code and full assessment are released as supplementary material.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.07442v2">EchoTrace: Diagnosing Recursive Risks in LLM-Powered Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2026-08-20
      | 💬 Accepted by ACM CIKM 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly integrated into recommender systems as data augmenters, profile generators, and recommendation modules. While these roles can enhance semantic understanding and recommendation quality, they also introduce LLM-specific risks such as bias and hallucination. These risks become more critical in feedback-loop settings, where LLM-generated signals and recommendations recursively shape future training data and recommendation dynamics. In this paper, we propose a role-aware, phase-wise diagnostic framework for analyzing how LLM-induced risks emerge, propagate, and accumulate in LLM-powered recommender systems. Our framework combines controlled feedback-loop simulation with longitudinal phase-wise diagnosis across LLM-generated content, recommendation outputs, feedback-loop dynamics, and ecosystem-level effects. Experiments on widely used benchmarks show that LLM-based components can amplify popularity bias, introduce spurious signals through hallucination, and gradually produce polarized and self-reinforcing exposure patterns over time. The code for EchoTrace is available at https://github.com/DongUk-Park/EchoTrace.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.19746v1">PersonalBench: Measuring the Authorship Gap in LLM Personalization</a></div>
    <div class="paper-meta">
      📅 2026-08-20
      | 💬 17 pages. Extended version
    </div>
    <details class="paper-abstract">
      Personalized text generation aims to make LLMs write in a specific individual's style, yet existing benchmarks measure task accuracy or preference alignment rather than whether the model's output actually resembles the target author's writing. We introduce PersonalBench, a benchmark that evaluates inference-time personalization methods through three independent lenses: LUAR (a trained authorship verification model), an LLM-as-judge, and automated stylometrics. Across 50 authors, 1,000 generations, and two model families (Qwen 3, GLM-4), we find that personalization methods do produce author-differentiated output (LUAR discriminates target authors within generated text at AUC=0.918) but this differentiation never crosses the human-LLM boundary. All methods achieve LUAR similarity to real authors in the range 0.484-0.508, below the cross-author human floor of 0.626 (ceiling 0.756). The LLM's own authorship fingerprint dominates: generated text is more distant from any human author than random humans are from each other. Methods are statistically indistinguishable from each other on LUAR (spread 0.024) despite appearing differentiated on the LLM judge, a discrepancy we trace to circularity between trait extraction and profile extraction. We validate that LUAR reliably measures authorship in our corpus (AUC=0.76 single-post, 0.96 multi-post). We release PersonalBench as a calibrated measuring stick: inference-time personalization modulates the LLM's style but does not bridge the gap to human authorship.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.19689v1">Rethinking the Evaluation and Optimization of LLM-Based Social Simulation</a></div>
    <div class="paper-meta">
      📅 2026-08-20
    </div>
    <details class="paper-abstract">
      LLM-based social simulation is a promising complement to traditional methods such as surveys and behavioral experiments. A core question is how to evaluate the fidelity of LLM-simulated human behavior and optimize LLMs toward it. Prevailing practice evaluates by accuracy, checking whether the model selects the single response observed from a human, and trains the LLM to reproduce this hard label. However, human behavior is inherently subjective: the same person in the same situation may reasonably act differently, so an observed response is only one draw from an underlying response distribution, rendering accuracy-based evaluation unreliable and hard-label training misleading. To address these problems, we first introduce the subjectivity coefficient, an entropy-based quantity distinguishing objective tasks such as coding from subjective ones such as social simulation, and use it to systematically analyze how accuracy-based evaluation and hard-label training fail as subjectivity grows. Based on the subjectivity coefficient, we propose Subjectivity-Adaptive soft-Label Training (SALT): it pools observed outputs from semantically nearby inputs into soft distributional labels, with an aggregation radius adapted to the estimated subjectivity of each input; in the near-objective limit the neighborhood shrinks, so SALT naturally falls back to standard single-label training. Moreover, since existing datasets record only single observed responses and cannot support distributional evaluation, we construct SUBJSIM, a benchmark of 19,300 contexts covering 193 annotators and 100 subjective questions. Since real-world data typically provide only a single observation per input, our experiments train models from single observed outputs while evaluating them against the full response distributions, verifying feasibility in realistic settings. Results on SUBJSIM demonstrate the advantages of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.19670v1">The Asymmetric Harms of LLM Compression</a></div>
    <div class="paper-meta">
      📅 2026-08-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) compression reduces deployment costs, but standard aggregate metrics like perplexity and accuracy often mask underlying behavioral shifts. In this work, we systematically evaluate 3 LLMs across 11 compression methods to investigate the effects of compression on knowledge retention, model confidence, and social bias. We find that compression disproportionately reduces the relative retention of head knowledge compared to tail knowledge. Furthermore, compressed models often remain substantially confident in their incorrect answers on newly lost knowledge. Finally, we demonstrate that stable aggregate bias scores can conceal substantial, opposing shifts in stereotypical preferences across demographic subgroups. Together, these findings reveal asymmetric behavioral changes that aggregate performance measures fail to capture, highlighting the need for granular evaluation of compressed models before deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.19665v1">Training-Free LLM-Based Recommendation with Post-LLM Item Refinement Using Collaborative Signals</a></div>
    <div class="paper-meta">
      📅 2026-08-20
      | 💬 Published as a conference paper at CIKM 2026 (short)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown promise for training-free recommendation, but LLM-generated user interests are often too broad for fine-grained item retrieval. Existing methods incorporate collaborative filtering (CF) signals in a pre-LLM manner through candidate reranking or prompt augmentation, yielding limited gains. We propose CoRRe, a training-free recommendation framework with a post-LLM paradigm that injects CF signals into LLM-generated item representations, which are later matched with LLM-generated user interests for ranking. Specifically, CoRRe refines the directions of item embeddings using an item-item co-purchase graph and their magnitudes using item popularity. Experiments on real-world datasets show that CoRRe consistently outperforms existing training-free methods and achieves competitive or superior performance compared with training-based methods, without requiring any model training or task-specific fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.19659v1">FleetSieve: Decision-Critical Profiling for SLO-Aware LLM Fleet Configuration</a></div>
    <div class="paper-meta">
      📅 2026-08-20
    </div>
    <details class="paper-abstract">
      Choosing tensor-parallel (TP) degrees and replica counts for an LLM serving fleet is difficult because performance is not monotonic in TP and the feasible choice can change with load. Exhaustive profiling resolves this uncertainty, but measures many configurations that do not affect the final resource allocation. We present FleetSieve, which selects measurements according to their expected effect on a resource-coupled, SLO-aware fleet decision. FleetSieve models capacity and tail latency jointly, compares conservative and optimistic allocations, and stops when their remaining decision gap is below a specified tolerance. On a fixed H100 measurement grid for a 31B-parameter open-weight model, FleetSieve reaches the oracle aggregate decision using 22,200 GPU-seconds, 6.9% less than uniform random profiling in the fixed comparison. Across 200 random reveal orders, its mean saving over random profiling is 5.4% (95% bootstrap CI: 3.5-7.2%). The fixed-comparison saving is 21.5% for Chat, while FleetSieve does not use the fewest GPU-seconds for Code. Joint capacity and tail modeling also avoids selecting a configuration whose 46.4-second completion p99 violates a 30-second SLO. In a 16-GPU allocation, an incorrect sparse-profile decision loses up to 1.93 requests/s and 12.4 percentage points of max-min fulfillment. Boundary repeats and BurstGPT measurements support the observed load-dependent tail-latency mechanism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.19626v1">Auditing and Decomposing Feedback-Driven Evolution in LLM Test Generation under the Oracle Problem</a></div>
    <div class="paper-meta">
      📅 2026-08-20
    </div>
    <details class="paper-abstract">
      Execution feedback is often treated as a self-verifying signal for improving LLM-generated tests. However, when generated inputs are executed on a single accepted program and its outputs are used as ground truth, invalid or underspecified inputs can create spurious fault detections and apparent evolutionary gains. We audit this failure mode in feedback-driven test generation using 142 development tasks, 114 locked external tasks, and 138 held-out tasks, with two code models, three seeds, and fault-cross-fitted real submissions. On external inputs for which three accepted implementations agree, generated outputs match the panel on only 27.79% and 50.12% of cases. A single-reference oracle inflates the measured gain from evolution by 9.46-14.85 percentage points; after auditing, equal-budget independent resampling outperforms mutation-based evolution by 6.01-18.83 points. We further compare a genuine three-round feedback loop with a density-matched placebo. External Real-Placebo differences are +0.13 and -0.50 points, while held-out differences are +1.99 and +0.28 points and do not provide robust evidence of fine-grained feedback benefit. A blinded semantic audit by two software engineering doctoral students classifies 94.41% of panel-disconfirmed inputs as invalid but 3.60% as valid, showing that panel disagreement is informative but not semantic proof. We propose an audit-and-placebo protocol that separates verifier artifacts, interaction scaffolding, and grounded feedback credit in evaluations of self-evolving test generators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.19621v1">Mitigating Identity Essentialism in LLM Agents with Longitudinal Life Trajectories</a></div>
    <div class="paper-meta">
      📅 2026-08-20
      | 💬 23 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer a scalable approach to social simulation, but their credibility depends on how agents are constructed. Existing methods can partially reproduce population-level patterns, yet often fail to capture human-like diversity. Our analysis shows that static-profile agents exhibit stronger demographic separation and within-group compression than humans, a pattern consistent with identity essentialism: demographic labels can encourage models to treat group-average tendencies as individual traits, homogenizing responses within groups. We argue that this limitation arises from two related factors: sparse, static agent representations and the limited ability of prompt-only memory to persistently integrate experience. Inspired by complementary memory systems, we propose LifeMem, a longitudinal memory framework that combines structured life-event retrieval with agent-specific parametric memory for experience integration. Experiments on Add Health and Understanding Society with three LLMs show that LifeMem improves alignment with human data in terms of response distributions, overall and within-group diversity, and patterns of within-person response change across life stages. These findings highlight the value of longitudinal life-event memory for constructing more faithful and dynamically evolving social agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26170v2">EvoSelect: Data-Efficient LLM Evolution for Targeted Task Adaptation</a></div>
    <div class="paper-meta">
      📅 2026-08-20
    </div>
    <details class="paper-abstract">
      Adapting large language models (LLMs) to a targeted task efficiently and effectively remains a fundamental challenge. Such adaptation often requires iteratively improving the model toward a targeted task, yet collecting high-quality human-labeled data to support this process is costly and difficult to scale. As a result, synthetic data generation has emerged as a flexible and scalable alternative. One straightforward approach is through an iterative generation-training loop, where candidate data are synthesized through an external generator, the model is updated using these data and the process is repeated over iterations. However, generated samples can be noisy, highly redundant, or even misaligned with the targeted task distribution. Training indiscriminately on such data can dilute useful learning signals and even degrade model performance. To address this, we introduce a refined paradigm, namely an iterative generation-selection-training loop, which incorporates a selection step prior to model updates. Building on this paradigm, we propose EvoSelect, a data-efficient framework to evolve LLM effectively. Given candidate samples produced by the data generator, EvoSelect selects training data by jointly modeling targeted task alignment and diversity. We estimate task relevance through optimal transport with proxy gradient representations, which quantifies how well candidate samples align with the targeted task distribution. To mitigate redundancy, we incorporate a diversification mechanism that promotes coverage of complementary training samples. By interleaving alignment and diversification, EvoSelect enables progressive LLM evolution toward targeted tasks. Extensive experiments on various benchmarks demonstrate that with either weak or strong data generators, EvoSelect consistently improves adaptation efficacy over existing data selection methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.19579v1">Enforcing LLM Safety through DMD-based Classification of Prompt-Response Embedding Dynamics</a></div>
    <div class="paper-meta">
      📅 2026-08-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in high-stakes applications, yet their tendency to generate toxic, harmful, or policy-violating content poses significant risks. Detecting these unsafe outputs efficiently in a black-box manner remains an open challenge. In this paper, we extend a recently proposed dynamical systems framework designed for hallucination detection to LLM safety classification. By projecting both prompts and responses into high-dimensional embedding spaces and fitting separate Koopman-based predictive models for safe and unsafe regimes, we classify new outputs using a new differential residual score that compares prediction errors of the safe and unsafe regimes. A key contribution is the incorporation of the prompt and response embedding dynamics, yielding fitted Koopman operators that capture crucial interaction patterns. We evaluate our black-box method across three safety benchmarks using three embedding models. Our results show that incorporating prompt embeddings yields consistent improvements, particularly for interaction-dependent violations when paired with causal decoders (e.g., in Llama-3), while response-only violations benefit more from dense semantic embedding representations. These findings opens the door for using dynamical systems to analyze AI systems rather than the dominant paradigm of using AI to model dynamical systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.19564v1">Remember, Verify, or Ask? Cross-Family Evaluation of Memory Commitment in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-20
    </div>
    <details class="paper-abstract">
      Persistent memory can personalize an LLM agent, but an incorrect durable update can silently distort future behavior. We study the memory-clarification boundary: whether interaction-derived information should be persisted, used only in the current context, re-verified, or clarified with the user. MCB contains 140 primary scenarios, split into 70 development and 70 held-out items, plus a separate 70-item contrast set. It evaluates both action labels and structured tool-call selection. Two non-authors independently label the 70 held-out primary and 70 contrast items (97.1% agreement, Cohen's kappa = 0.962); a blind third resolves four disagreements, replacing eight author labels by non-author majority. Across Claude and Qwen, models verify changing facts more reliably than they ask users to resolve ambiguity. Bare Qwen asks on 0/12 clarification items while verifying 12/18 freshness items. Few-shot prompting raises accuracy from 0.557 to 0.771 (paired delta = +0.214, Holm-adjusted exact McNemar p_H = 0.002), yet clarification recall remains 0.333. The policy prompt reduces erroneous persistence from 0.243 to 0.100 (p_H = 0.038), although its accuracy gain is not significant. Label-tool agreement is 57% for each Claude model and 23% for Qwen; Qwen accuracy falls from 0.557 to 0.343 (p_H = 0.047). Memory evaluation must test both stated decisions and tool-call choices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.19557v1">When Do LLM Agents Help? Deadline-Aware Mixed-Criticality Task Scheduling at the Autonomous-Vehicle Edge</a></div>
    <div class="paper-meta">
      📅 2026-08-20
      | 💬 8 pages, 5 figures, and 4 tables
    </div>
    <details class="paper-abstract">
      Autonomous vehicles offload latency-sensitive perception tasks to nearby mobile edge computing (MEC) servers, where a missed safety-critical task is unsafe rather than merely degraded. Large language models (LLMs) are increasingly proposed as adaptive, explainable schedulers, yet evidence of when they help is scarce. We study deadline-aware, mixed-criticality scheduling on heterogeneous MEC servers, where time-critical (TC) tasks must be protected at a controlled cost to best-effort traffic, and ask whether a multi-agent LLM control layer improves on a strong heuristic. We answer in two steps. First we build the heuristic: a windowed contract-net auction that orders each admission window time-critical-first by earliest deadline and places tasks by earliest-finish-time. Across 60 instances on three topologies and 15 baselines under an identical online constraint, it attains a TC completion rate of 0.902, above every baseline (Holm-corrected p < 0.001; best baseline 0.838) and at 0.87 of a CP-SAT upper bound. Second, we add the LLM control plane. A controlled decomposition traces the scheduler's advantage to two ordinary factors, the batching horizon and time-critical-first ordering; the auction, the per-window LLM policy, and online adaptation add nothing while the load is stationary, where the heuristic is already near-optimal. Under a mid-run surge of safety-critical tasks the picture changes, and the LLM control plane gains significantly over both the static heuristic and the bandit. LLM orchestration therefore earns its cost only when non-stationarity opens headroom a fixed policy cannot use. We report control-plane latency and rationale, and release all code and seeded instances.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.15949v2">Ask to Be Sure: Informative Interactions for Confident Multi-Turn LLM Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-08-20
      | 💬 CIKM 2026
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled their use as conversational recommender systems (CRS), demonstrating strong recommendation accuracy and natural dialogue. However, guiding multi-turn interactions to elicit user preferences effectively remains challenging. Existing approaches either use separate reinforcement learning agents with templated interactions or optimize for interactivity judged by another LLM, without measuring how much useful information is actually gained. We propose a new approach that quantifies the effectiveness of each interaction by the reduction in the assistant's uncertainty, measured via entropy over recommendations. We apply this entropy reduction as a reward---without relying on ground-truth recommendations, which are often unavailable in real-world scenarios---to fine-tune the LLM, enabling strategic interaction generation. Empirical results with supervised fine-tuning (SFT) and direct preference optimization (DPO) on the INSPIRED and ReDial datasets show that our method improves both recommendation quality and conversational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.17546v2">REST API Testing with Verified LLM-Inferred Dependencies and Response-Driven Refinement</a></div>
    <div class="paper-meta">
      📅 2026-08-20
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Testing RESTful APIs requires generating sequences of API calls that satisfy dependencies among operations, parameters, and runtime-created resources. Recent LLM-based approaches infer such dependencies and generate test sequences from OpenAPI specifications, but they often treat LLM-inferred relationships as correct without execution-based validation. This can introduce spurious dependencies, miss feasible operation chains, and produce infeasible tests. In this paper, we propose APIPilot}, an execution-validated framework for REST API testing. APIPilot first derives candidate producer-consumer dependencies from OpenAPI specifications using structural heuristics and LLM-based semantic reasoning. It then treats these dependencies as hypotheses and validates them through concrete API executions before using them for test generation. The validated dependencies are organized into a dependency graph from which APIPilot constructs coverage-aware workflows via bounded top-k graph traversal, separating semantic dependency inference from sequence construction. To improve subsequent tests, APIPilot further performs response-driven refinement: runtime responses are analyzed to update resource pools, adjust input-generation constraints, and prune or revise invalid dependency mappings. Empirical evaluation on 16 real-world REST API services shows that APIPilot achieves 92.3% operation coverage, up to 58.6% code coverage, and an 88.1% workflow execution success rate, outperforming both LLM-based and traditional REST API testing baselines. APIPilot also detects 197 unique 5xx failures and specification-execution mismatches, demonstrating the benefit of grounding dependency inference in execution feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.01548v3">LLM Capability Limits: Static Emergence and Dynamic Boundary Control</a></div>
    <div class="paper-meta">
      📅 2026-08-20
    </div>
    <details class="paper-abstract">
      Test-time emergence in LLM systems has a deployment boundary: additional computation can realize decisions already supported by the deployed information--execution structure, while evidence, tools, memory, and executable semantics can change the class inherited by later computation. We formalize this boundary through inherited structural capability $\mathcal{D}_{\mathcal{J}}$ and resource-indexed finite realization $\mathcal{F}_s(\mathcal{J},M)$. At a common budget, Theorem 1 gives an exact decision representation: a successor improves every bounded-loss task exactly when its closed convex finite envelope retains the predecessor's. Terminal capability can therefore expand while same-budget capability strictly reverses. The same object yields finite-slice recovery and a workload-tail information radius for open-ended evaluation. Dynamically, Bellman value prices the successor capability class together with the finite policies it preserves. Nested realization makes every fixed extra resource increment vanish at saturation, allowing persistent positive successor value to dominate that increment. The resulting theory turns emergence into a boundary, compatibility, measurement, and control problem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.11922v2">LODESTAR: Robust Entropy-Based Answer Selection in Retrieval-Augmented Generation for Question Answering -- Directing Frozen-LLM Entropy with a Reinforcement-Learned Prompt Polarizer under Misleading Passages</a></div>
    <div class="paper-meta">
      📅 2026-08-20
      | 💬 28 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Predictive-distribution entropy is a strong answer-selection rule in retrieval-augmented generation (RAG) for question answering: across five QA benchmarks, selecting the answer a frozen respondent LLM produces with the lowest answer-token entropy lifts mean $F_1$ from 0.4769 to 0.5148 over the retriever's top-ranked passage, without gold answers. Yet this rule, which prior entropy-based selectors adopt, fails: a misleading passage makes the respondent confidently wrong, driving entropy down where the uncertainty signal looks most trustworthy. The failure comes from the passage the respondent reads, and the context it is read in is an input we can intervene on. We introduce LODESTAR: to our knowledge the first method to score a text intervention by the uncertainty it induces in a third-party frozen respondent, compared within one question. LODESTAR uses reinforcement learning (GRPO) to train, once and offline, a polarizer -- a short fixed natural-language string inserted into the respondent's prompt and never into its weights, directing entropy so that entropy-based answer selection stays robust to misleading passages; training labels are built from gold answers and two LLM judges, and inference reads neither. With every competing selector under the same frozen respondent and candidate pools on 5,008 questions, LODESTAR attains the highest mean $F_1$ of any inference-ready selector (0.5339), the highest macro exact match (0.4136), and the highest GPT-4o judge score of the frozen-respondent configurations judged (0.6435); its three-seed mean wins all 70 $F_1$ cells against fourteen published configurations and is paired-significant on $F_1$ against every one. The gain holds in-domain on NQ-Open and out-of-domain over SQuAD, TriviaQA, EntityQuestions and WebQuestions. Ablating the polarizer shows it is what makes the respondent read a misleading passage less often (26.0% vs 30.3%).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.19463v1">LLM as Detector: An In-context Learning Approach for Tabular Anomaly Detection</a></div>
    <div class="paper-meta">
      📅 2026-08-19
      | 💬 Accepted at International Conference on Neural Information Processing (ICONIP) 2026
    </div>
    <details class="paper-abstract">
      Anomaly detection in tabular data is challenging because abnormal samples often arise as violations of cross-feature dependencies rather than simple marginal deviations. Existing detectors rely on geometric or reconstruction signals, while prior LLM-based approaches mainly fine-tune LLMs with normal samples or generate synthetic anomalies. We propose LLM-Detector, a framework that utilizes the in-context learning capacity of LLMs for structured, prompt-conditioned scoring synthesis, enabling LLMs to derive anomaly detection logic from structured normal-state knowledge. Specifically, normal training data are converted into statistical summaries, causal dependencies, and distilled prototypes that are organized into a prompt for code generation. The resulting scoring engine evaluates statistical deviation, structural inconsistency, and density-based abnormality then computes an anomaly score for each test sample. We evaluate LLM-Detector on 24 tabular datasets, comparing against 15 SOTA baselines. Results show consistent improvements across both mixed-type and continuous-only settings. Moreover, this design eliminates the need for LLM fine-tuning or neural network training, reducing computational cost and enabling practical anomaly detection in real-world tabular systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.19437v1">Are LLMs becoming similarly creative? Evidence from three years of models</a></div>
    <div class="paper-meta">
      📅 2026-08-19
      | 💬 12 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Many benchmarks track Large Language Model (LLM) performance on tasks with verifiable answers, but less is known about how LLM performance is evolving on open-ended tasks, where creativity, originality and diversity may matter as much as quality. As LLMs increasingly support human ideation and creative work, understanding trends in LLM performance on open-ended tasks is critical. This paper presents a preliminary analysis of LLM creative outputs spanning three years of model releases, examining model responses to Infinity-Chat100, a real-world collection of open-ended user queries, and the Alternate Uses Task, an established psychometric creativity assessment. Using sentence-embedding similarity, we examine trends in LLM responses to these prompts. Our findings show a statistically significant decrease in model output diversity over time, suggesting that LLM outputs may be converging in creative substance across models. If this trend persists, LLM-driven homogenization may progressively diminish human agency in human-AI co-creative work, demanding careful consideration of LLMs' role in the human creative process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.23522v2">Qworld: Question-Specific Evaluation Criteria for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-19
    </div>
    <details class="paper-abstract">
      Evaluating large language models (LLMs) on open-ended questions is difficult because response quality depends on the question's context. Binary scores and static rubrics fail to capture these context-dependent requirements. Existing methods define criteria at the dataset level or generate them in a single pass, which limits their ability to explore the evaluation space implied by each question. We introduce One-Question-One-World (Qworld), a method that generates question-specific evaluation criteria using a recursive expansion tree. Given a question, Qworld decomposes it into scenarios, perspectives, and fine-grained binary criteria through hierarchical and horizontal expansion. The resulting criteria specify what a high-quality answer must address for that question. On HealthBench, Qworld covers 89% of expert-authored criteria and generates 79% novel criteria validated by human experts. Experts rate Qworld criteria higher in insight and granularity than those produced by prior methods. When applied to 11 frontier LLMs on HealthBench and Humanity's Last Exam, Qworld reveals capability differences in dimensions such as long-term impact, equity, error handling, and interdisciplinary reasoning that coarse rubrics do not capture. By generating evaluation criteria for each question, Qworld enables assessment of LLM responses that is tailored to the question rather than based on fixed task-level criteria.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.17379v2">PTXBench: Benchmark and Adapt LLMs for GPU Kernel Optimization with Architecture-specific PTX</a></div>
    <div class="paper-meta">
      📅 2026-08-19
    </div>
    <details class="paper-abstract">
      We introduce PTXBench, a benchmark for evaluating and adapting large language models (LLMs) to use architecture-specific PTX for GPU kernel optimization. PTXBench measures functional correctness, whether selected target instructions execute at runtime, and speedup over frontier libraries across GEMM and attention workloads on H100 and B200 GPUs. Our evaluation shows that architecture-specific PTX capability remains uneven: success rates fall substantially on complex attention backward workloads, and executing the target instructions does not necessarily translate into competitive performance. No evaluated model consistently matches frontier libraries across the suite. We further adapt Qwen3.6-27B using supervised fine-tuning. Repair-conditioned training improves several tasks, but generalization remains uneven; data coverage, balance, and the quality of the reasoning teacher matter in addition to dataset size. PTXBench provides an auditable testbed for measuring and improving LLMs' ability to exploit evolving GPU architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.02910v2">The Basic B*** Effect: The Use of LLM-based Agents Reduces the Distinctiveness and Diversity of People's Choices</a></div>
    <div class="paper-meta">
      📅 2026-08-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly act on people's behalf: they write emails, buy groceries, and book restaurants. While the outsourcing of human decision-making to AI can be convenient, it raises a fundamental question: how does delegating identity-defining choices to AI shape who people become? Across a large field study and a controlled experiment, we study the impact of agentic LLMs on two identity-relevant outcomes: interpersonal distinctiveness - how unique a person's choices are relative to others - and intrapersonal diversity - the breadth of a single person's choices over time. Study 1 uses 110,000 real choices drawn from social media behavior of 1,000 U.S. users to compare generic and personalized agents to a human baseline. Both agents shift people's choices toward more popular options, reducing the distinctiveness of their preferences. While the use of personalized agents tempers this homogenization (compared to generic agents), it also more strongly compresses the diversity of people's preference portfolios by narrowing their exploration across topics and psychological affinities. Study 2 replicates these patterns in an online experiment which mimics common real-world scenarios (e.g., choosing movies) and allows us to directly compare the AI agent's choices to those made by 348 participants (12,097 human choices). The findings also suggest that the flattening effects of AI agents are amplified when choices are made sequentially (vs. batch), and when agents rely on domain-specific user information for personalization. Understanding how AI agents compress human experience (and the trade-offs involved) is critical for designing systems that augment human agency and safeguard diversity in thought, taste, and expression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.19395v1">HYDRA: A Heterogeneous Chiplet DSE Framework for Serving Dynamic Hybrid LLM Workloads</a></div>
    <div class="paper-meta">
      📅 2026-08-19
      | 💬 Accepted at IEEE/ACM ESWEEK (CASES) 2026; the official version will be available in IEEE TCAD
    </div>
    <details class="paper-abstract">
      Hybrid Transformer-Mamba large language models (LLMs) enhance long-context efficiency, but their heterogeneous computation and communication patterns complicate efficient hardware acceleration. Chiplet-based architectures offer a scalable solution by integrating specialized compute and memory units. However, the design space spanning static architectural configurations and dynamic runtime policies is prohibitively large to explore exhaustively. To address this challenge, we present HYDRA, a comprehensive design space exploration framework for hybrid LLM serving on heterogeneous chiplet systems. HYDRA jointly explores chiplet composition, placement, inter-chiplet bandwidth provisioning, dynamic batching, and runtime scheduling. It integrates communication-aware placement, dynamic batching, elastic task scheduling, and a fast Markov-based performance estimator that captures multi-tenant runtime dynamics for efficient and accurate exploration. Across all workloads, HYDRA delivers 1.55x the throughput and 43.7 percent lower time-to-first-token on average, with throughput gains reaching up to 2.3x compared to state-of-the-art baselines. These results highlight that co-designing architecture and runtime policies is critical for efficient large-scale LLM serving on heterogeneous chiplet systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.03519v3">TS-Reasoner: Aligning Time Series Foundation Models with LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-08-19
      | 💬 Accepted to Transactions on Machine Learning Research, 2026
    </div>
    <details class="paper-abstract">
      Time series reasoning is crucial to decision-making in diverse domains, including finance, energy, and scientific discovery. While existing time series foundation models (TSFMs) can capture low-level dynamic patterns and provide accurate forecasting, further analysis usually requires additional background knowledge and sophisticated reasoning, which are lacking in most TSFMs but can be achieved through Large Language Models (LLMs). On the other hand, without expensive post-training, LLMs often struggle with the numerical understanding of time series data. Although it is intuitive to integrate the two types of models, developing effective training recipes that align the two modalities for reasoning tasks is still an open challenge. To this end, we propose TS-Reasoner that aligns the latent representations of TSFMs with the textual inputs of LLMs for downstream understanding/reasoning tasks. Specifically, we propose a simple yet effective method to curate diverse, synthetic pairs of time series and textual captions for alignment training. We then develop a two-stage training recipe that applies instruction fine-tuning after the alignment pretraining. Unlike existing works that train an LLM to take time series as inputs, we leverage a pretrained TSFM and freeze it during training. Experiments on several benchmarks demonstrate that TS-Reasoner not only outperforms a wide range of open-source LLMs, Vision-Language Models (VLMs), and Time Series LLMs of comparable scale, but also does so with remarkable data efficiency, e.g., using less than half the training data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.19147v1">Pre-Compiled Pipeline Shards for Distributed LLM Inference on Intel AI PC Fleets</a></div>
    <div class="paper-meta">
      📅 2026-08-19
    </div>
    <details class="paper-abstract">
      Modern Intel AI PCs ship capable integrated GPUs and NPUs with 16+ GB of unified memory, and they spend considerable time idle. That is not enough memory to fit a large model such as a 70B-parameter LLM. We show that a handful of AIPCs, working together over an ordinary network, can serve models beyond the capability of any single one. We use pipeline parallelism: a model is split by layer into per-stage shards, each pre-compiled into an OpenVINO graph, so that every machine runs one shard and passes activations to the next. Three techniques make this fast enough to be useful. First, we recover the speed of the unsplit model: a naive per-stage export runs well below monolithic inference because it misses an OpenVINO GPU optimization, and injecting a beam_idx Gather into each shard triggers that optimization (the IndirectKVCache fusion) and brings the shards to parity. Second, we leverage speculative decoding on stateful OpenVINO models. Third, the pipeline serves several users at once by interleaving their requests across the stages, each request carrying its own cache (micro-batching). Together, a two-node Llama 3.1 8B INT4 pipeline serves two concurrent users at 1.79x the single-user throughput of the unsplit model on the same hardware, and the gap widens under simulated wide-area latency. The same design scales to a 70B model that no single fleet member can hold: a four-node deployment of Lunar Lake AI PCs on Intel Tiber Cloud serves a single user at interactive speed, with output token-for-token identical to the same four-node pipeline decoding without speculation. Code, raw benchmark logs, and reproduction scripts ship as a self-contained package at https://github.com/labscommunity/pipeline-sharded-inference-paper (in the top-level reproduction/ directory).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.16804v3">AutoOR: Scalably Post-training LLMs to Autoformalize Operations Research Problems</a></div>
    <div class="paper-meta">
      📅 2026-08-19
    </div>
    <details class="paper-abstract">
      Optimization problems are central to decision-making in manufacturing, logistics, scheduling, and other industrial settings. Translating complicated descriptions of these problems into solver-ready formulations requires specialized operations research (OR) expertise, making it hard to scale. We present AutoOR, a scalable synthetic data generation and reinforcement learning pipeline that trains LLMs to autoformalize optimization problems specified in natural language across linear, mixed-integer, and non-linear categories. AutoOR generates verified training data from standard optimization forms and uses solver execution feedback as the reward signal for RL post-training. AutoOR applied to an 8B model achieves state-of-the-art or competitive results across six established OR benchmarks, matching significantly larger frontier models. For a non-linear problem class involving physical dynamics, where frontier models score near 0%, we introduce a curriculum RL strategy that bootstraps from limited initial training data to make this class tractable for post-training. We believe that methods such as AutoOR can significantly accelerate industrial decision-making with AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15232v3">When Agents Fail: A Comprehensive Study of Bugs in LLM Agents with Automated Labeling</a></div>
    <div class="paper-meta">
      📅 2026-08-19
      | 💬 Accepted at ISSRE 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized intelligent application development. While standalone LLMs cannot perform any actions, LLM agents address the limitation by integrating tools. However, debugging LLM agents is difficult and costly, as the field is still in its early stages and the community is underdeveloped. To understand the bugs encountered during agent development, we present the first comprehensive study of bug types, root causes, and effects in LLM agent-based software. We collected and analyzed 1,268 bug-related posts and code snippets from Stack Overflow, GitHub, and Hugging Face forums, focused on LLM agents built with seven widely used LLM frameworks as well as custom implementations. For a deeper analysis, we have also studied the component of the LLM agent where the bug occurred, along with the programming language and framework. This study also investigates the feasibility of automating bug identification. For that, we have built a ReAct agent named BugReAct, equipped with adequate external tools to determine whether it can detect and annotate the bugs in our dataset. According to our study, we found that BugReAct equipped with Gemini 2.5 Flash achieved a remarkable performance in annotating bug characteristics with an average cost of 0.01 USD per post/code snippet.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.17117v7">From Tokens to Thoughts: How LLMs and Humans Trade Compression for Meaning</a></div>
    <div class="paper-meta">
      📅 2026-08-19
    </div>
    <details class="paper-abstract">
      Humans organize knowledge into compact conceptual categories that balance compression with semantic richness. Large Language Models (LLMs) exhibit impressive linguistic abilities, but whether they navigate this same compression-meaning trade-off remains unclear. We apply an Information Bottleneck framework to compare human conceptual structure with embeddings from 40+ LLMs using classic categorization benchmarks. We find that LLMs broadly align with human category boundaries, yet fall short on fine-grained semantic distinctions. Unlike humans, who maintain ``inefficient'' representations that preserve contextual nuance, LLMs aggressively compress, achieving more optimal information-theoretic compression at the cost of semantic richness. Surprisingly, encoder models outperform much larger decoder models in human alignment, suggesting that understanding and generation rely on distinct representational mechanisms. Training-dynamics analysis reveals a two-phase trajectory: rapid initial concept formation followed by architectural reorganization, during which semantic processing migrates from deep to mid-network layers as the model discovers increasingly efficient, sparser encodings. These divergent strategies, where LLMs optimize for compression and humans for adaptive utility, reveal fundamental differences between artificial and natural intelligence. This highlights the need for models that preserve the conceptual ``inefficiencies'' essential for human-like understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.15473v2">Q-First: Most of Attention Needs Only the Query in Disaggregated LLM Decoding</a></div>
    <div class="paper-meta">
      📅 2026-08-19
      | 💬 21 pages,5 figures
    </div>
    <details class="paper-abstract">
      Disaggregated LLM serving puts the KV-cache sweep on memory-optimised hardware and the projections and feed-forward on compute-optimised hardware, then inherits from the decoder block a dependency neither device wants: attention runs first and the feed-forward consumes its output, so within one sequence each side idles while the other works. The usual repair costs one resident KV cache per extra sequence in flight, which is what motivated separating the devices at all. We remove the dependency instead. The sweep needs only the query, and exchanging the two sub-layers makes that query available while the compute side still has work to do, so the two run concurrently; the current key and value follow as a cache write nothing waits on. We state the decode as a protocol, show that it runs on stock kernels, and verify it end to end on a trained checkpoint to a relative error of 3.2x10^-3 -- with no new operator, no changed shape and no new hardware. We then train the block 8 ways at two seeds each, varying only where the attention reads and holding everything else fixed. At three per cent of compute-optimal a lead in bits per byte measures how much a change disturbed training rather than what it reaches, so we read magnitudes and not rankings. Among the 5 blocks whose feed-forward does not consume their own attention, no read point differs from the one that moves nothing by more than 0.0026 bits per byte -- smaller than the gap between an arm and itself at a second seed, 0.0066 -- while the same runs resolve a sub-layer exchange 25 times as large. Moving the query early is a change the measurement cannot find, which is what the protocol needs. The reach is bounded: projecting every layer's query from the network's input costs +0.0974, refuting a pre-registered threshold at both seeds, so a query may be read one feed-forward early and no further back.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.23057v2">RequestRouter: Request-Boundary Routing for Efficient Single-GPU LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-08-19
      | 💬 5 pages, 2 figures. Accepted at the 2nd International Workshop on Low Carbon Computing (LOCO 2026), Lancaster University, United Kingdom, 10-11 September 2026. Part of the LOCO 2026 proceedings, arXiv:2608.02072
    </div>
    <details class="paper-abstract">
      RequestRouter is a lightweight request-boundary controller for reducing the latency and energy cost of single-GPU large language model inference. Rather than serving all requests with one static configuration, the system uses cheap request-level features to select one fixed inference mode per request, including FP16, quantized inference, speculative decoding, prefix caching, continuous batching, and hybrid modes such as GPTQ plus prefix caching and INT8 plus continuous batching. We evaluate RequestRouter using an 8B instruction-tuned language model served through vLLM on NVIDIA A100 GPUs. Across the full-scale A100 evaluation--26,500 fixed-mode evaluations followed by 3,500 online-controller evaluations, for 30,000 measured inference executions in total--the controller achieves a 2.10x mean latency speedup over FP16 and a 0.48x energy ratio on deployment-style workloads. A smaller matched evaluation with repeated measurements provides a controlled statistical check of this result: RequestRouter retains a 1.93x latency speedup (95% CI: 1.88--1.98x) and a 0.523 energy ratio (95% CI: 0.506--0.540), showing that the gains persist under a more tightly controlled protocol. On a separate expanded automatic benchmark evaluation, the routed policy retains 99.6% of FP16 macro accuracy. A 100,000-call CPU microbenchmark measures only 0.00475 ms mean routing overhead (0.00532 ms p99). Thus, simple request-aware routing can recover substantial serving efficiency without retraining or modifying the underlying LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.00735v4">`From Prompt to Perturbation': An Adaptive Framework for Voice-Based Jailbreaks on Audio LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-19
      | 💬 Accepted at the IEEE International Conference on Data Mining (ICDM 2026)
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly integrated into audio-based applications, growing concerns have emerged regarding their vulnerability to audio-based adversarial attacks. These systems typically follow two architectural paradigms: cascaded pipelines, where automatic speech recognition converts audio inputs into text before LLM processing, and end-to-end large audio-language models (LALMs), which directly interpret raw audio signals. Beyond architectural differences, cascaded pipelines are primarily vulnerable to text-level jailbreak strategies delivered through speech, whereas end-to-end LALMs introduce additional acoustic-semantic attack vectors. However, existing studies often focus on a single paradigm and provide limited coverage of the broader audio attack space. To bridge this gap, we propose an adaptive jailbreak attack framework for systematic evaluation of both cascaded pipelines and LALMs under a unified experimental setting. At its core, the framework uses a feedback-guided mutation engine to automatically generate and refine jailbreak candidates across both textual prompts and audio perturbations, thereby expanding attack diversity and coverage. Experiments on six representative audio-based systems demonstrate that both paradigms remain substantially vulnerable to audio jailbreak attacks. Compared with state-of-the-art methods, our framework achieves consistently higher attack success rates across diverse audio-based LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.18795v1">Decomposing Wrong-Consensus Agreement in LLM Self-Consistency: A GPT-4.1 Case Study</a></div>
    <div class="paper-meta">
      📅 2026-08-19
      | 💬 18 pages, 2 figures, 9 tables; quantitative kappa-decomposition of agreement saturation in self-consistency;
    </div>
    <details class="paper-abstract">
      Majority voting over multiple LLM samples is widely used to raise answer accuracy, yet its gain varies erratically: on hard questions it can even backfire. This paper gives a quantitative account of this failure. A pluralistic agreement index Gamma is defined as the expected fraction of the samples of a wrong run that agree with the consensus, normalized by a reference scale d=(1-p)/(C-1), and is decomposed into a mechanical component (what a vote delivers given only a per-case answer preference) and a preference-unexplained residual. The mechanical null is difficulty-matched and leak-free: each case is resimulated at its own accuracy and option preference, estimated from the case's other runs, so no run predicts its own agreement. On GPT-4.1 the decomposition shows benchmark-associated direction (an observational ordering over n=4 cells per benchmark, not a significance claim). On multiple-choice GPQA-Diamond, the per-case answer preference explains 81-93% of the held-out test-run agreement index: the shared-bias-dominates account over-claims here, because a wrong but attractive option the whole cohort latches onto is captured by the per-case preference channel (whether that preference is induced by shared training bias is not identified). On open-domain AIME, the mechanical preference explains only 59-78% (21-29% if shrunk to pure noise), and a preference-unexplained residual of 1.56-2.80 Gamma units survives, which a run-level preference-heterogeneity reference more than absorbs (1.4-2.1). A self-consistency backfire on hard questions is reproduced (binned voting gap down to -0.09, coupled CI [-0.12,-0.07]), and the highest-agreement bin reaches an accuracy of only 0.42-0.83, a 1.2-3.6x lift over base rate: agreement is graded evidence, not certification. No new voting method is proposed; code and evidence are committed and reproducible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.18733v1">Flama: a Python framework for development and deployment of production-ready APIs, machine learning, and LLM services</a></div>
    <div class="paper-meta">
      📅 2026-08-19
      | 💬 83 pages, 6 figures, 1 table. Software available at https://github.com/vortico/flama, up-to-date documentation at https://flama.dev
    </div>
    <details class="paper-abstract">
      We present Flama, an open-source Python framework for developing and deploying production-ready web APIs, machine learning services, and large-language-model (LLM) applications. Built on the Asynchronous Server Gateway Interface (ASGI), Flama offers a type-driven, async-first programming model that unifies REST API development, predictive model serving, and generative AI inference in one architecture. It is organised around seven subsystems: a component-based dependency injection system resolving handler parameters from type annotations at startup; a pluggable schema layer supporting Pydantic, Marshmallow and Typesystem behind a single adapter; an automatic CRUD generator turning a SQLAlchemy table and a schema class into REST endpoints backed by the Repository and Unit of Work patterns; a portable binary format (.flm) packaging models from scikit-learn, TensorFlow, PyTorch and Hugging Face Transformers with their metadata for zero-code deployment; a multi-backend LLM server running vLLM (Linux/CUDA) or MLX (Apple Silicon) and exposing four wire protocols (OpenAI, Anthropic, Ollama, and a native streaming dialect) through a shared codec; a Rust-accelerated core compiled via Maturin for routing, JSON encoding, compression and parsing; and a Model Context Protocol module turning any application into an MCP server over JSON-RPC 2.0. Built-in capabilities include JWT authentication, two pagination strategies, background tasks in threads or processes, WebSocket endpoints, Server-Sent Event and NDJSON streaming, OpenAPI 3.2.0 generation from handler signatures, and a command-line interface for running applications and for serving, packaging and inspecting models. We describe the architecture, present the programming model through worked examples, and compare Flama with existing frameworks, model serving platforms and LLM inference engines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.17326v2">Procedural Collapse: A Structural Account of Disengagement in LLM-Assisted Writing</a></div>
    <div class="paper-meta">
      📅 2026-08-19
    </div>
    <details class="paper-abstract">
      When students use large language models for writing, the dominant explanation for disengagement is dispositional: they are over-reliant, and the remedy is to scaffold self-regulation. We argue that a structural explanation is needed, offering an alternative basis for design interventions to support appropriate AI-assisted writing. Current LLM writing interfaces induce procedural collapse: the replacement of an iterative, self-paced writing process with a single output that shifts the writer's task from generation to comprehensive evaluation. Because that evaluation is costly, shallow engagement becomes the default, and the cognitive work writing was supposed to produce goes unperformed. The framework points toward design directions that reduce the burden on writers to self-regulate, including decomposed interaction, goal elicitation as a default first step, and single-level output. They complement metacognitive scaffolding by restructuring the interaction itself.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.19893v3">Future Policy Approximation for Offline Reinforcement Learning in LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-08-19
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has emerged as a key driver of post-training for complex reasoning in large language models (LLMs), yet online RL introduces substantial instability and computational overhead. Offline RL offers a compelling alternative by decoupling generation from training; however, offline algorithms for reasoning remain under-optimized relative to their online counterparts. We revisit the potential of policy-gradient-style offline RL and address a central challenge in offline learning: gradient entanglement. In long-horizon reasoning trajectories, correct and incorrect solutions share substantial token overlap, causing gradient updates from incorrect trajectories to suppress tokens that are also critical for correct ones. We propose Future Policy Approximation (FPA), a simple offline policy-gradient method that weights gradients using an estimate of the future policy rather than the current policy, enabling proactive gradient reweighting. We estimate the future policy through logit- space extrapolation. Across three models, seven mathematical reasoning benchmarks, and three code-generation benchmarks, FPA consistently improves over strong offline baselines, including DPO, RPO, KTO, and vanilla offline RL. FPA stabilizes long-horizon training, where vanilla objectives degrade, and achieves accuracy comparable to state-of-the-art RLVR methods such as GRPO and DAPO at a fraction of the GPU hours.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.18686v1">Improving LLM-Based SSH Honeypots Through Prompting and Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2026-08-19
    </div>
    <details class="paper-abstract">
      LLM-based SSH honeypots often use closed cloud LLMs because they give strong shell realism, but cloud models create deployment problems. These include no stable versioning, provider-side changes, attacker-driven cost, and model decommissioning. Local open-weight models avoid these problems, but they usually perform worse and make mistakes that reveal the honeypot. These mistakes include malformed outputs, command echoing, inconsistent filesystem state, and AI-style artifacts. This paper studies how to improve and evaluate the shell emulation accuracy of local LLM-based SSH honeypots using prompt design and supervised fine-tuning. We fine-tune and evaluate eight models in total: the original fine-tuned GPT-3.5 model used in shelLM and seven open-weight local models, each compared to its base model. We also test how prompt structure transfers across model families. Using 34 automated unit tests that measure shell emulation accuracy in single-session and fresh-session settings, we find that prompt design has a large effect and that fine-tuning depends on dataset coverage. Fine-tuning on the original 112-conversation dataset does not improve aggregate pass rate, while an expanded dataset built from honeypot logs produces clearly stronger local models. Taken together, the results suggest that prompting and fine-tuning can each improve local LLM honeypots on their own, but their effects do not combine straightforwardly, since strong rule-based prompting and supervised adaptation can also conflict by addressing overlapping shell-behavior constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.19281v1">APPROVE: Visual End-User-in-the-Loop Robot Programming with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-19
      | 💬 Accepted for publication in Procedia CIRP, Proceedings of the 20th CIRP Conference on Intelligent Computation in Manufacturing Engineering (ICME 2026)
    </div>
    <details class="paper-abstract">
      Programming robots remains challenging for non-experts, as traditional methods require expert knowledge and even block-based interfaces often lack flexibility. Recent work has explored Large Language Models (LLMs) to automatically generate robot programs from natural language, but these systems remain limited by a lack of transparency, missing mechanisms to ensure alignment with user intent, and little support for reuse. We present APPROVE (AI-Powered Programming for Robots with Visual End-User Feedback), an LLM-based multi-modal end-user programming framework that integrates natural language input with a block-based interface and an explicit user confirmation step. Generated programs are visualized using a block-based interface in Blockly, allowing users to confirm, modify, or reject them before execution. Confirmed functions are stored in a library for reuse, gradually building a set of reliable program components. Our approach contributes a human-centered design for LLM-based robot programming that emphasizes user trust, intent alignment, and reusability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.29076v2">Selective KV Cache Protection for Noise-Resilient LLM Inference on Analog Compute-In-Memory Systems</a></div>
    <div class="paper-meta">
      📅 2026-08-19
    </div>
    <details class="paper-abstract">
      Analog compute-in-memory (CIM) arrays have emerged as a promising substrate for energy-efficient LLM inference, particularly for weight-stationary computations in linear layers. However, extending analog CIM to attention mechanisms introduces a fundamental challenge: KV cache operations demand repeated in-situ weight updates, and the resulting mismatch with the weight-stationary paradigm exposes dynamic computations to significant hardware noise, a critical problem that remains largely unexplored. In this paper, we present the first systematic study of dynamic attention computation on analog CIM arrays, revealing that initial and recent tokens exhibit disproportionate vulnerability to hardware noise. Motivated by this token-level insight, we propose a hierarchical token protection strategy that keeps sink tokens and a sliding recent-token window on a higher-precision digital path while processing the bulk KV cache on analog CIM. A co-designed scheduler combines analog programming, ownership transition, and bulk-MVM tile formation to bound digital overhead. Evaluations on nine LLMs show that our approach lowers average perplexity under analog noise from 33.91 to 11.95, close to the clean baseline of 11.06, while improving dynamic-KV programming-row utilization from 23.1\% to 91.2\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.18645v1">Code Health in LLM-Based Test Generation: Effectiveness and Token Efficiency</a></div>
    <div class="paper-meta">
      📅 2026-08-19
      | 💬 Accepted at the Engineering Track of the 26th IEEE International Conference on Source Code Analysis and Manipulation (SCAM 2026)
    </div>
    <details class="paper-abstract">
      Coding agents powered by Large Language Models (LLMs) are now prominent in software engineering. Previous work has shown that AI tools perform better on high-quality source code that is easy to maintain. In this study, we investigate how the effectiveness of LLM-generated unit tests varies across maintainability levels measured by CodeScene's CodeHealth (CH). We assess test effectiveness using traditional coverage metrics and mutation score across Python, Java, and C++. Moreover, we study how code with different levels of CH translates into input tokens using common industrial tokenizers. Our results suggest that CH provides a weak but consistent signal of LLM-generated test effectiveness and is negatively correlated with input-token count. These findings provide further evidence for a relationship between maintainability and LLM-based software development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17566v2">KA2L: A Knowledge-Aware Active Learning Framework for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-19
      | 💬 17 pages, 3 figures. Published in Expert Systems with Applications 333 (2027) 133951
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) with high-quality knowledge has been shown to enhance their performance effectively. However, there is a paucity of research on the depth of domain-specific knowledge comprehension by LLMs and the application of targeted active learning to improve their expertise. To address this gap, we introduce the Knowledge-Aware Active Learning (KA2L) framework. This framework assesses LLMs' mastery of specific knowledge points to aid in constructing unanswerable or unknowable questions through latent space analysis. This active learning strategy enhances training efficiency by focusing on knowledge the model has yet to master, thereby minimizing redundancy in learning already acquired information. This study innovatively employs a knowledge distribution probing technique to examine the hidden states of specific Transformer layers and identify the distribution of known and unknown knowledge within the LLM. Additionally, a hidden-state decoding method is proposed to generate numerous unknown questions in natural language from the latent knowledge space. In our experiments, we selected nine open-source LLMs to validate the effectiveness of the proposed framework. Results indicate that KA2L not only significantly reduces 50% annotation and computation costs across two open-domain and one vertical-domain dataset but also achieves better performance, offering valuable insights into active learning strategies for LLMs. The code is available at https://github.com/greenjerry/KA2L.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2402.13605v7">KorNAT: LLM Alignment Benchmark for Korean Social Values and Common Knowledge</a></div>
    <div class="paper-meta">
      📅 2026-08-19
      | 💬 Accepted at ACL 2024 Findings (35 pages, 7 figures, 16 tables)
    </div>
    <details class="paper-abstract">
      For Large Language Models (LLMs) to be effectively deployed in a specific country, they must possess an understanding of the nation's culture and basic knowledge. To this end, we introduce National Alignment, which measures an alignment between an LLM and a targeted country from two aspects: social value alignment and common knowledge alignment. Social value alignment evaluates how well the model understands nation-specific social values, while common knowledge alignment examines how well the model captures basic knowledge related to the nation. We constructed KorNAT, the first benchmark that measures national alignment with South Korea. For the social value dataset, we obtained ground truth labels from a large-scale survey involving 6,174 unique Korean participants. For the common knowledge dataset, we constructed samples based on Korean textbooks and GED reference materials. KorNAT contains 4K and 6K multiple-choice questions for social value and common knowledge, respectively. Our dataset creation process is meticulously designed and based on statistical sampling theory and was refined through multiple rounds of human review. The experiment results of seven LLMs reveal that only a few models met our reference score, indicating a potential for further enhancement. KorNAT has received government approval after passing an assessment conducted by a government-affiliated organization dedicated to evaluating dataset quality. Samples and detailed evaluation protocols of our dataset can be found in https://huggingface.co/datasets/jiyounglee0523/KorNAT and https://github.com/jiyounglee-0523/KorNAT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.18591v1">Can a Lightweight Multimodal Model Estimate LLM Reasoning Performance? A Study for Compute-Optimal Document Inference</a></div>
    <div class="paper-meta">
      📅 2026-08-19
    </div>
    <details class="paper-abstract">
      Uniformly allocating inference reasoning budgets to LLMs is expensive and prone to over-thinking penalties; especially in document tasks where visual layouts drive complexity. To address this, we introduce BudgetDoc, the first multimodal benchmark providing explicit supervision for model-budget-performance trade-offs across three document tasks. Using BudgetDoc, we train DRB (Document-Reasoning Balancer), an approx. 1B-parameter pre-flight estimator (SigLIP-2 + Qwen3-0.6B) that predicts ordinal model performance across budget levels, achieving a 0.753 weighted F1. When dynamically allocating reasoning budgets across five frontier models and three datasets, DRB matches or improves F1 scores compared to always-maximum-budget baselines in 9 of 15 configurations while drastically reducing cost. Finally, preliminary evaluations demonstrate DRB's potential to generalize to cross-model selection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.18588v1">AppEval: A Unified Benchmark for LLM-Based Mobile Application Repair in ArkTS, Swift, and Kotlin</a></div>
    <div class="paper-meta">
      📅 2026-08-19
      | 💬 9 pages, 2 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Repository-level LLM agents are typically evaluated on projects whose tests run on the build host. It remains unclear whether their repairs survive the mobile build-install-launch-test boundary, where a missing SDK, offline device, or pre-assertion crash can be mistaken for a program failure. We present AppEval, a benchmark and native-toolchain evaluation framework for mobile application repair across HarmonyOS/ArkTS, iOS/Swift, and Android/Kotlin. Each task separates a hidden behavior test from the reference production fix and is accepted only when the same installed-app target reaches an assertion failure on the defective revision and passes after the fix; infrastructure failures remain a distinct outcome. A common schema maps this contract to each platform's build system, runtime, and test runner. The audited Android partition contains 200 accepted instrumentation tasks from 24 independently buildable repositories. On these tasks, five agents achieve Pass@1 between 22.00% and 90.50%, a 68.50-percentage-point spread under the same dynamic oracle. These results show that mobile repair performance depends strongly on the evaluated agent while demonstrating why runtime-aware acceptance is necessary for meaningful comparison. The quantitative findings in this paper are Android-specific; audited iOS and HarmonyOS results are required before drawing cross-platform generalization conclusions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.18581v1">From Storage to Access: Verifiable Activation of Parametric Knowledge in LLMs via Explicit Priming and Implicit Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-08-19
    </div>
    <details class="paper-abstract">
      Although Large Language Models (LLMs) encode rich factual knowledge in their parameters, reliably recalling and verifying such knowledge remains a key bottleneck in factual question answering. Existing end-to-end methods entangle knowledge elicitation with reasoning, making it difficult to determine whether correct answers arise from parametric knowledge or the input context. To address this challenge, we propose VAKE (Verifiable Activation of Parametric KnowledgE), a two-stage reinforcement-learning framework that externalizes latent parametric knowledge through explicit Priming and transfers the acquired elicitation capability to implicit Reasoning. Given a query and an insufficient retrieved subgraph, the Priming policy explicitly inserts bridging triples as verifiable evidence, with supervision provided by rewards derived from answers generated by a separate frozen model over the augmented subgraph. Building on the policy learned during Priming, the Reasoning stage trains the model to answer from the original input, testing whether the capability acquired through explicit knowledge elicitation transfers to implicit reasoning. Experiments across seven benchmarks and models from 3B to 14B show that VAKE consistently outperforms standard baselines, including when transferring directly from HotpotQA to OOD datasets. LLM-based evaluation further shows that over 80% of the inserted triples provide factual bridging knowledge not derivable from the retrieved context, while more than half elicit knowledge inaccessible through direct prompting. These results suggest that VAKE activates latent parametric knowledge rather than copying the input context or memorizing dataset-specific associations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.18575v1">Beyond LLM-Based Reasoning: Lightweight GNNs for Agent Failure Attribution</a></div>
    <div class="paper-meta">
      📅 2026-08-19
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based multi-agent systems (MAS) often exhibit complex failure modes, which frequently cause agents to produce incorrect outcomes. This motivates the task of Agent Failure Attribution: given a failed multi-agent trajectory, identify the faulty agents and their corresponding error types. Existing approaches predominantly rely on LLMs to perform failure attribution, either through direct prompting, fine-tuning on synthetic data or complex agentic pipelines. While effective, these methods incur substantial computational overhead due to long-context processing, expensive post-training and handcrafted workflows. Moreover, empirical evidence shows that even state-of-the-art models achieve limited accuracy on existing benchmarks, suggesting that scaling model size alone is insufficient. In this work, we revisit this task and question the necessity of such expensive generative solutions. We introduce AFANet, a lightweight graph-based framework that models interaction trajectories through step-level semantic signals and agent-level relationships. We show that with significantly fewer parameters and near-zero inference cost, AFANet (i) matches or outperforms LLM-based baselines, including fine-tuned models on in-domain benchmarks, (ii) maintains robust performance across different GNN architectures and (iii) can be further improved with inexpensive test-time adaptation on the OOD benchmark. Our results suggest that effective agent failure attribution does not require heavy LLM reasoning and a lightweight, structured approach can achieve strong performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17178v3">TrojanGYM: A Detector-in-the-Loop LLM for Adaptive RTL Hardware Trojan Insertion</a></div>
    <div class="paper-meta">
      📅 2026-08-19
    </div>
    <details class="paper-abstract">
      Hardware Trojans (HTs) remain a critical threat because learning-based detectors often overfit to narrow trigger/payload patterns and small, stylized benchmarks. We introduce TrojanGYM, an agentic, LLM-driven framework that automatically curates HT insertions to expose detector blind spots. Given high-level HT specifications, a suite of cooperating LLM agents (instantiated with GPT-4, LLaMA-3.3-70B, Gemini-2.5Pro, and Claude Opus 4.5) proposes and refines RTL modifications that realize diverse triggers and payloads without impacting functionality of both the HT and the design under attack. TrojanGYM implements an agentic loop co-designed with HT detectors, in which constraint-aware syntactic checking, testbench-based functional verification, and GNN-based HT detectors provide feedback that iteratively refines HT specifications and insertion strategies to better surface detector blind spots. We further propose Robust-GNN4TJ, a new implementation of GNN4TJ with improved graph extraction, training robustness, and prediction reliability, especially on LLM-generated HT designs. On the most challenging TrojanGYM-generated benchmarks, Robust-GNN4TJ raises HT detection rates from 0% to 60% relative to prior art. We instantiate TrojanGYM on SRAM, AES-128, UART, and RISC-V designs at RTL, and show that it systematically produces diverse, functionally correct HTs that reach up to 68.75% evasion rates against modern GNN-based detectors, revealing robustness gaps that are not apparent when these detectors are evaluated on existing TrustHub-style benchmarks. We release all codes and artifacts at https://github.com/DfX-NYUAD/TrojanGYM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.18554v1">CentaurBench: Benchmarking LLM Capabilities on Augmenting vs. Automating Real-World Work Tasks</a></div>
    <div class="paper-meta">
      📅 2026-08-19
      | 💬 46 pages, 15 figures
    </div>
    <details class="paper-abstract">
      Most LLM benchmarks rank models on their ability to automate work tasks. In practice, however, models are often used to assist other (human or LLM) agents. The question that drives model selection is therefore not only which model produces the best output, but which model most improves the work of another (weaker) agent. We introduce a unified framework that evaluates the capability of models to automate and augment another agent's performance. Across seven economically grounded real-world tasks, an assistant model writes assistance text for a standardized lower-capacity worker model, which produces the deliverable. In automation mode, the assistant produces the output directly. Outputs are scored through blind pairwise comparisons by an LLM judge panel with task-specific rubrics, replicated across ten runs. Rankings across the two regimes are only modestly correlated, and the automation winner loses augmentation on five of seven tasks. Assistance is not reliably positive. The unaided worker outranks every assisted condition on three tasks, and only one model's guidance beats no guidance on average. These results suggest that automation ability is an incomplete proxy for assistance quality, motivating benchmarks that evaluate models according to the roles they play in human-AI and multi-agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.07625v2">HINT: Toward an Executable Hardware-Intent Representation Layer for LLM-Driven RTL Generation</a></div>
    <div class="paper-meta">
      📅 2026-08-19
    </div>
    <details class="paper-abstract">
      Generating implementation-quality RTL with large language models (LLMs) remains difficult because direct generation must resolve microarchitecture while simultaneously producing and debugging low-level code. We present HINT, an executable hardware-intent intermediate representation layer between behavioral specifications or executable oracles and RTL. HINT makes RTL-relevant microarchitecture explicit, supports pre-RTL checking, and supplies explicit RTL-lowering obligations. We evaluate HINT using both a minimal single-agent flow and a full staged workflow. Across seven operator cases, the HINT-mediated route, with no post-synthesis QoR refinement, produces contract-compliant synthesizable RTL on 7/7 cases; Direct C2RTL and C2HLSC apply to five cases and succeed on 5/5 and 1/5, respectively. Under matched Design Compiler synthesis, HINT reduces area by 5.0\%--26.2\% relative to five manual RTL implementations and by 8.9\%--86.1\% relative to five accepted Direct C2RTL results. RealBench AES and SDC, together with a Vortex VPU synthesizing to 561.67k~$μ\mathrm{m}^2$, further demonstrate specification-driven, protocol-rich, memory-rich, and hierarchical designs. In the controlled operator study, the HINT-mediated route shows better observed convergence and avoids the severe implementation-quality degradation seen in several direct-generation results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.18539v1">Evaluating and Explaining Prompt Sensitivity of LLMs Using Interactions</a></div>
    <div class="paper-meta">
      📅 2026-08-19
      | 💬 Accepted at the 43rd International Conference on Machine Learning (ICML 2026). 46 pages, 48 figures
    </div>
    <details class="paper-abstract">
      The remarkable capabilities of large language models (LLMs) are often undermined by their instability. Even subtle and semantically irrelevant changes in prompts can cause dramatic fluctuations in performance, a phenomenon known as prompt sensitivity. Previous studies typically evaluate prompt sensitivity by comparing the LLM's final outputs when prompts change. However, such coarse-grained metrics fail to explain the internal reasons for prompt sensitivity. In this paper, we introduce interactions as a fine-grained tool to analyze prompt sensitivity of LLMs. Specifically, we decompose the output score of the LLM into a set of interactions. Each interaction represents a nonlinear relationship involving a set of input variables. We discover that subtle changes to prompts can trigger severe instability in interactions, even when the outputs of the LLM remain the same. To this end, we propose an Interaction-based Prompt Sensitivity (IPS) metric by quantifying changes in interactions when we introduce subtle changes to prompts. We apply the IPS metric to 50 open-source LLMs and uncover four factors that reduce the prompt sensitivity of LLMs, including supervised fine-tuning, increased model scales, dense architectures, and few-shot learning. More crucially, we discover a common mechanism by which these four factors reduce prompt sensitivity: all four factors tend to reduce the prompt sensitivity of low-order interactions (i.e., interactions involving few input variables).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12875v2">MetaInfer: A Knowledge Only LLM Inference Engine Generator SKILL Toolbox</a></div>
    <div class="paper-meta">
      📅 2026-08-19
    </div>
    <details class="paper-abstract">
      As LLM technology advances, the space of model families, compute hardware, quantization schemes, parallelization strategies, and specialized optimization kernels continues to expand, sharply increasing the code complexity and maintenance cost of general-purpose inference frameworks. Conventional software engineering uses multiple layers of abstraction to support diverse application scenarios, but these abstractions also increase system complexity and may introduce additional performance overhead. This paper presents metainfer, an 'LLM-as-Compiler' approach in which users specify only the runtime constraints of an inference program. An LLM-driven multi-agent collaboration system, coupled with a contract knowledge base, then automatically generates a compact customized inference framework that satisfies these constraints. We evaluate metainfer from three perspectives: the effect of source-code reference, the runtime behavior and performance profile of engines generated under the zero-reference constraint on CKB-covered targets, and knowledge-base evolution for new model and platform scenarios. The results show that metainfer organizes generation constraints, validation feedback, and knowledge consolidation into a continuous closed loop, enabling runnable customized inference solutions to be generated from explicit knowledge. The code is publicly available at https://github.com/MetaInfer/MetaInfer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.18503v1">LLM-Powered Predictive Decision-Making for Sustainable Data Center Operations</a></div>
    <div class="paper-meta">
      📅 2026-08-19
    </div>
    <details class="paper-abstract">
      The growing demand for AI-driven workloads, particularly from Large Language Models (LLMs), has raised concerns about the significant energy and resource consumption in data centers. This work introduces a novel LLM-based predictive scheduling system designed to enhance operational efficiency while reducing the environmental impact of data centers. Our system utilizes an LLM to predict key metrics such as execution time and energy consumption from source code, and it has the potential to extend to other sustainability-focused metrics like water usage for cooling and carbon emissions, provided the data center can track such data. The predictive model is followed by a real-time scheduling algorithm that allocates GPU resources, aiming to improve sustainability by optimizing both energy consumption and queuing delays. With fast inference times, the ability to generalize across diverse task types, and minimal data requirements for training, our approach offers a practical solution for data center scheduling. This framework demonstrates strong potential for advancing sustainability objectives in AI-driven infrastructure. Through our collaboration with a data center, we achieved a 32% reduction in energy consumption and a 30% decrease in waiting time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22664v5">MBABench: Evaluating LLM Agents on End-to-End Spreadsheet Tasks in Finance</a></div>
    <div class="paper-meta">
      📅 2026-08-19
    </div>
    <details class="paper-abstract">
      LLM agents are increasingly expected to carry out end-to-end workflows, producing complete artifacts from high-level user instructions. To meet enterprise needs, frontier AI labs have developed agents that can construct entire spreadsheets from scratch. This is especially relevant in finance, where core workflows such as financial modeling, forecasting, and scenario analysis are commonly conducted through spreadsheets. Yet, existing spreadsheet benchmarks do not measure this new capability, focusing instead on question-answering or single-formula edits. To address this gap, we provide one of the first evaluations of agents on end-to-end spreadsheet tasks, focusing on economically critical financial workflows such as modeling and scenario analysis. Since deliverables therein are routinely reviewed and revised by multiple stakeholders, judging their quality necessarily involves high-level criteria such as readability or ease of modification. To reflect the multidimensional nature of solution quality, we develop an evaluation taxonomy comprising three dimensions: Accuracy, Formula, and Format, each comprising fine-grained criteria that reflect professional standards. Evaluating over 18 agents, the benchmark reveals that even the strongest agents fall short of basic professional finance standards, and their performance degrade sharply as the difficulty increases beyond a few chained calculations. This suggests that current agents are not yet able to reliably produce professional-quality spreadsheets at the level of complexity real-world workflows demand.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.18490v1">Bayesian Partner Modelling enables Adaptive Replanning for LLM Coordination</a></div>
    <div class="paper-meta">
      📅 2026-08-19
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Multi-agent Large Language Model (LLM) systems often struggle to collaborate with new teammates whose strategies shift mid-task. Because agents execute multi-step or temporally extended skills, they frequently continue executing outdated plans long after public evidence shows that a partner has changed its skill. Existing methods either treat partner tracking as passive context-leaving the agent aware of the shift but slow to act-or replan indiscriminately. We introduce BayesBeliefAgent, which pairs a hierarchical LLM planner with a Bayesian tracking module. Rather than replanning constantly, our agent interrupts its current skill only when a partner's actions directly contradict the inferred skill. Beyond standard reward, we evaluate performance using replanning efficiency and the belief-action gap: the fraction of total decisions where an agent with a correct partner estimate executes a non-complementary skill. Across benchmark Overcooked environments, contradiction-conditioned control drastically narrows this belief-action gap while requiring an order of magnitude fewer replans than heuristic methods
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.18473v1">A Locally Deployable Tool-Grounded LLM Multi-agent Framework for Automating Methane Emission Analysis and Reporting</a></div>
    <div class="paper-meta">
      📅 2026-08-19
    </div>
    <details class="paper-abstract">
      Methane field monitoring requires the integration of sampling design, meteorological interpretation, sensor processing, plume analysis, visualization, and reporting, but these steps are often distributed across separate expert-driven workflows. We developed a locally deployable, tool-grounded large language model (LLM) multi-agent framework for our low-cost methane sensing and field-monitoring campaigns. The framework uses LLM agents as workflow coordinators that link field measurements, meteorological data, deterministic sensor-processing routines, Gaussian plume inversion, and report generation, rather than directly estimating methane concentrations or emissions. Extensive field deployments across diverse real-world environments (e.g., wastewater treatment facilities, landfills, and oil and gas sites) demonstrate that our framework can achieve 92.0\% accuracy in workflow routing and parameter extraction, 85.0\% success in emission-rate estimation and plume prediction, and 95.0\% success in generating editable reports under practical operating conditions. Compared with manual and general-purpose LLM-assisted workflows, it reduced workflow time from hours-level to minutes-level, lowered manual coordination and prompt-engineering requirements, and retained traceable plume-based outputs. In addition, most processing can be performed locally, reducing exposure of sensitive facility and field data to cloud services. These results indicate that tool-grounded LLM coordination can reduce the time, labor, usability, and data-security barriers of methane field monitoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.01499v3">Matching Accuracy, Different Geometry: Evolution Strategies vs GRPO in LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2026-08-19
      | 💬 Accepted to Conference on Language Modeling (COLM) 2026
    </div>
    <details class="paper-abstract">
      Evolution Strategies (ES) have emerged as a scalable gradient-free alternative to reinforcement learning based LLM fine-tuning, but it remains unclear whether comparable task performance implies comparable solutions in parameter space. We compare ES and Group Relative Policy Optimization (GRPO) across four tasks in both single-task and sequential continual-learning settings. ES matches or exceeds GRPO in single-task accuracy and remains competitive sequentially when its iteration budget is controlled. Despite this similarity in task performance, the two methods produce markedly different model updates: ES makes much larger changes and induces broader off-task KL drift, whereas GRPO makes smaller, more localized updates. Strikingly, the ES and GRPO solutions are linearly connected with no loss barrier, even though their update directions are nearly orthogonal. We develop an analytical theory of ES that explains all these phenomena within a unified framework, showing how ES can accumulate large off-task movement on weakly informative directions while still making enough progress on the task to match gradient-based RL in downstream accuracy. These results show that gradient-free and gradient-based fine-tuning can reach similarly accurate yet geometrically distinct solutions, with important consequences for forgetting and knowledge preservation. The source code is publicly available: https://github.com/Bhoy1/ESvsGRPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.02592v3">AI Fact-Checking in the Wild: A Field Evaluation of LLM-Written Community Notes on X</a></div>
    <div class="paper-meta">
      📅 2026-08-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show promising capabilities for fact-checking, yet prior work evaluates them only in controlled offline settings using benchmarks or crowdworker judgments. Success in real-world fact-checking depends also on how content is judged within a live platform environment. We present the first field evaluation of LLM fact-checking deployed on a live social media platform, testing performance directly through X Community Notes' "AI writer" feature over a three-month period. Our LLM writer, a multi-step pipeline that handles multimodal content, conducts web and platform-native search, and writes contextual notes, was deployed to write 1,614 notes on 1,597 tweets and compared against 1,332 human-written notes on the same tweets using 108,169 ratings from 42,521 raters. Direct comparison of note-level platform outcomes is complicated by differences in submission timing and exposure between LLM and human notes; we therefore pursue two analysis strategies: a rating-level analysis modeling individual rater evaluations, and a note-level analysis that relies on common raters who rated all notes on the same post. Rating-level analysis shows that LLM notes receive more positive ratings than human notes across raters with different political viewpoints, and note-level analysis shows LLM notes achieve significantly higher helpfulness scores among common raters. Rater-provided tags suggest that people consider LLM notes to use more neutral language and cite better sources. These findings provide field evidence that LLMs can contribute broadly helpful fact-checking notes at scale, while showing that their evaluation is shaped by platform dynamics absent from controlled offline settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.16002v2">From Sequence to Structure: Relational Uncertainty Propagation for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-19
    </div>
    <details class="paper-abstract">
      Reliable uncertainty quantification (UQ) is essential for deploying large language model (LLM) agents in complex interactive environments. Existing UQ methods largely rely on local signals, such as token probabilities, predictive entropy, or per-step confidence, and therefore overlook the long-range dependencies through which errors accumulate across an execution trajectory. As a result, they may fail to identify agent failures whose causes originate several reasoning or interaction steps before the final answer. We propose RUPA (Relational Uncertainty Propagation for Agents), a trajectory-level UQ framework for LLM agents. RUPA represents an execution history as a directed trajectory graph in which reasoning states, tool interactions, and environment feedback are nodes connected by temporal and semantic dependency edges. It then propagates uncertainty over this graph to capture how execution risk accumulates and transfers across interaction steps. The propagated signal is combined with trajectory-level behavioral features and goal-alignment information to produce a confidence estimate for the full agent trajectory. We evaluate RUPA on representative agent benchmarks, including $τ$-2, Terminal-Bench-2, and GAIA, using 6 open-source LLMs spanning multiple model families. Experimental results show that RUPA consistently outperforms existing UQ methods by providing more accurate uncertainty estimates, enabling earlier failure detection, and improving uncertainty-guided agent execution across diverse agent tasks. These results demonstrate that explicitly modeling relational dependency is crucial to reliable UQ for long-horizon LLM agents, providing a practical foundation for trustworthy agent execution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.18398v1">LEDGER: Claim-to-Evidence Trace Graphs for Auditing LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-19
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents can now carry out long-horizon technical workflows involving complex tool use, code execution, file edits, and generated artifacts. As agents do more work faster, the productivity bottleneck shifts from producing outputs to auditing whether those outputs are correct and trustworthy. Agent observability systems make fine-grained execution events visible, but visibility alone still leaves reviewers to reconstruct which actions, artifacts, and validation steps matter for a particular conclusion. We introduce LEDGER - Layered Evidence and Decision Graphs for Execution Review, a tracing and review system that builds layered trace graphs over observed agent sessions. LEDGER preserves Trace Records while grouping them into Evidence Nodes and Workflow Nodes, representing artifacts as evidence anchors, and adding typed semantic edges that connect claims to supporting actions, artifacts, and checks. Through data-analysis and coding examples, we show how the resulting traces expose workflow decisions, artifact lineage, repair steps, validation coverage, and claim-support paths for evidence-centered audit.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.18361v1">Figurative and Cultural Knowledge in LLMs: Investigating Cross-Domain Transfer through Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2026-08-18
    </div>
    <details class="paper-abstract">
      Figurative language is deeply culturally embedded; fluent use requires not just linguistic competence but cultural immersion. We ask whether LLMs can learn this link: does fine-tuning on cultural data improve figurative language understanding, and vice versa? We conduct a systematic study across four models (ALLaM-7B, Fanar-1-9B, Qwen3-8B, Llama-3.1-8B) and six Arabic datasets spanning cultural commonsense, proverbs, and poetry across diverse dialects and regions. Fine-tuning on poetry improves idiom comprehension (+2.33%, p<0.05), a gain our ArabicMMLU control does not reproduce, indicating that it stems from figurative content rather than Arabic language adaptation and pointing to a sensitivity to non-literal meaning that transfers across figurative types. Cultural fine-tuning, by contrast, lowers proverb-interpretation accuracy in both Arabic-centric models. Transfer between the two domains is otherwise indistinguishable from noise, with Arabic models frequently regressing after fine-tuning, suggesting prior saturation of relevant knowledge, while multilingual models show greater adaptation headroom. Error analysis further reveals that fine-tuning reinforces experiential cultural knowledge while destabilizing historically grounded factual knowledge. Our findings suggest that the relationship between culture and figurative language, though conceptually natural, is not straightforwardly captured through fine-tuning alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.06641v2">From Documentation to Zero-day Vulnerabilities: LLM-Driven Fuzzing of JavaScript Engines in PDF Readers</a></div>
    <div class="paper-meta">
      📅 2026-08-18
      | 💬 16 pages, 2 figures, accepted by ACM CCS 2026
    </div>
    <details class="paper-abstract">
      Existing fuzzers for PDF readers rely on simple test cases that involve only individual API calls, leading to limited coverage and potentially missing vulnerabilities that require sequences of API calls. To address these limitations, we propose PDFuzzer, a novel PDF engine fuzzer that automatically generates complex and meaningful API call sequences. PDFuzzer first uses a Large Language Model (LLM) to construct context-free grammars and infer the relationships between individual API calls from specifications extracted from JavaScript API manuals and execution traces. Based on the grammars and relationships, PDFuzzer employs a constraint solver to generate concrete API call sequences for fuzzing. Our experiments show that PDFuzzer significantly outperforms state-of-the-art PDF fuzzers (TypeOracle, Favocado, and Cooper) and LLM-based fuzzers (Fuzz4All, naive LLM) on three mainstream PDF readers: Adobe Acrobat Reader, Foxit PDF Reader, and PDF-XChange Editor. PDFuzzer achieves up to 48% higher coverage than existing tools and identifies 31 zero-day vulnerabilities in these readers, from information leakage to arbitrary code execution. Our ablation study validates the necessity of each component, including LLMs, which achieve high accuracy across all pipeline stages (93-98%). We disclosed all vulnerabilities to the vendors via a coordinated vulnerability disclosure process and received bug bounties.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.18303v1">SESSE: Sketch, Expand, Sort, Summarize, Evaluate -- LLM-as-Judge Evaluation via Structured Decomposition</a></div>
    <div class="paper-meta">
      📅 2026-08-18
    </div>
    <details class="paper-abstract">
      LLM-as-judge evaluation reduces response quality assessment to a single holistic A/B preference choice, providing no mechanism to isolate which quality dimensions drove the preference or distinguish model errors from genuine label ambiguity. We propose SESSE (Sketch, Expand, Sort, Summarize, Evaluate), a training-free framework that decomposes holistic judgment into structured sub-questions mined directly from the judge's own error cases; requiring no oracle responses, task-specific rubrics, or fine-tuning. On RewardBench (n=1,000), SESSE achieves near-parity with the chain-of-thought baseline and is competitive with RISE-Judge-32B (92.7%), a fine-tuned specialist, while remaining fully training-free. Per-criterion vote evidence provides an interpretable audit trail for diagnosing label ambiguity and judge failure modes unavailable from a single holistic output token.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.18300v1">The Lifecycle of LLM-as-a-Judge for Large-Scale Recommendation Explanations</a></div>
    <div class="paper-meta">
      📅 2026-08-18
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge, which leverages a large language model to evaluate natural language generated by another AI application or model, has become a standard, scalable approach for accelerating and extending costly human evaluation. However, most work treats a judge as a static artifact, evaluating it once at construction or against a fixed benchmark. In contrast, we argue that an LLM judge running in a production system is better understood as having a lifecycle: it must be built, trained, deployed, and continuously maintained as the surrounding data evolves, and each phase poses distinct technical and operational challenges. We present such a lifecycle for the LLM judges that evaluate user-facing recommendation explanations at Netflix, where our pipeline generates and the judges assess hundreds of thousands of distinct show-level explanations per week, served across the mobile experience to millions of members. Our framework has four phases: (I) Birth, defining multiple evaluation criteria and building curated benchmark datasets with human labels and rationales; (II) Training, refining the judges' rubrics via Reasoning-Aligned Rubric Tuning (RART), a rubric-tuning procedure that uses a meta-judge over reasoning output as the learning signal; (III) Deployment, in which one judge serves two production roles: quality gating and reflective generation; and (IV) Monitoring, a continuous Human-in-the-Loop alignment process that detects drift and triggers re-tuning behind a human review gate. We report post-launch results from a five-week A/B test over tens of millions of members, in which the judge-aligned explanations shifted member viewing toward novel content (previously unwatched) and increased successful browse-to-play sessions relative to a no-explanation control, with no quality-related takedowns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.18260v1">Redakto - The Incognito Tab for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-18
      | 💬 Accepted at WIPE-OUT 2026, 2nd Workshop on Machine Unlearning and Privacy Preservation at ECML-PKDD
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are being increasingly used in everyday applications. A major challenge in the context of LLMs or Artificial Intelligence (AI) in general is to ensure privacy when using them, meaning that personally identifiable information (PII) is removed from any text that enters an LLM. These challenges have become more urgent with novel EU legislation. Uncertainty around LLM usage with respect to privacy concerns in EU countries can be a major blocker for the speed of innovation and transfer from research to applications. Here we present \textbf{Redakto}, a tool that can be used for anonymizing text prior to feeding it to an LLM or other downstream text processing. We provide state-of-the-art functionalities for both redaction of PII but also when used for pseudonymization. These functionalities are exposed such that they can easily be used by end-users, through the Redakto web application, and by developers and researchers, via REST APIs and model context protocol (MCP) hooks. The implementation is fully open source, requires modest compute resources, and can be readily deployed on local hardware. In contrast to prior work and in order to better assess the quality of the anonymized texts, we conduct extensive empirical evaluations on textual data from legal and medical domain with respect to both privacy and utility of the redacted texts. Our empirical results demonstrate that the texts anonymized with different redaction strategies achieve utility scores on par with the original texts, suggesting that anonymization with Redakto can be used for LLM tasks without substantial negative impact for the tasks we explored.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.18254v1">GAPL: Grounded Action-effect Policy Learning for LLM-Based Trajectory Planning</a></div>
    <div class="paper-meta">
      📅 2026-08-18
      | 💬 11 pages, 5 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Trajectory planning for autonomous driving requires both high-level reasoning and precise low-level control. Large Language Models (LLMs) offer semantic-rich planning capabilities, however, their application is limited by hallucinated reasoning, poor grounding in environment dynamics, and limited numerical precision in control. We propose GAPL (Grounded Action-effect Policy Learning), a unified framework that integrates LLM-based effect estimation, simulation-based effect grounding, and policy optimization into a closed-loop system. GAPL consists of three modules: (1) an LLM-based Effect Evaluator for structured multi-dimensional action-effect estimation; (2) a Simulation-based Effect Grounder that predicts dynamics-consistent effects from simulator rollouts; and (3) an Effect-Aware Decision Maker that grounds LLM effect estimates against simulation via a distiller to guide Proximal Policy Optimization (PPO)-based policy learning. Experiments on four Highway-env scenarios demonstrate that GAPL consistently outperforms baselines, achieving average reductions of {0.76, 0.86, 2.00} in collision rate, average displacement error (ADE), and final displacement error (FDE), and an average reward gain of 1.44.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.18232v1">Contracting for LLM Delegation: Moral Hazard in Technology and Effort Choice</a></div>
    <div class="paper-meta">
      📅 2026-08-18
    </div>
    <details class="paper-abstract">
      We extend the standard Principal-Agent framework to scenarios where the Agent selects from a suite of technologies, each characterized by a distinct cost-capability profile. This framework is increasingly critical in the era of Large Language Models (LLMs), where Agents choose both a model and an associated effort level (e.g., token budget). We model the relationship between output quality and effort as a concave, saturating function, which depends on the Agent's hidden two-dimensional action choice balancing technology selection and effort allocation. We derive the optimal linear contract for the Principal, demonstrating that the Agent's best response is characterized by a threshold reward share that triggers technology switching. Finally, we calibrate our model using open-weight LLM pairings across the MATH and MMLUPro benchmarks. We show that both Principal and Agent, when employing bandit algorithms to navigate this environment, converge to strategies that closely align with our theoretical equilibrium. These results suggest that simple linear contracts can effectively incentivize complex, technology-aware delegation in agentic workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.18027v1">Chain-of-Experience for Continual LLM Improvement</a></div>
    <div class="paper-meta">
      📅 2026-08-18
      | 💬 H.T. and Y.F. contributed to this work equally
    </div>
    <details class="paper-abstract">
      Humans continuously learn from experience, whereas conventional large language model (LLM) evaluations ignore the models' ability to improve through inference-time interaction. In this paper, we study how LLMs learn from iterative experience at test time, a setting we refer to as Chain-of-Experience (CoE), where models accumulate experiential traces through iterative interactions with self or environmental feedback to form a continual improvement loop beyond zero-shot inference. We instantiate CoE with diverse feedback mechanisms, including model self-feedback and environmental signals such as correctness or public coding test pass rates, and evaluate across math, coding, and knowledge domains using 8 LLMs, including GPT-5, Gemini-2.5 Pro, Claude-4.5 Sonnet. Our study shows that leveraging iterative experience consistently outperforms feedback-free baselines, achieving substantial gains with self feedback alone, alongside a 5.6% overall improvement and 19% lower API cost across tasks and models. We further show that combining complementary feedback channels (e.g., model and correctness signals) yields additional gains, and that CoE delivers higher accuracy per token than existing test-time strategies. We observe a positive correlation between LLM base ability and improvement capacity, and show that models remain robust under weak or spurious feedback, with different feedback contributing to distinct improvement aspects and most gains emerging early in the iterations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.13110v4">SCOPE: Selective Conformal Optimized Pairwise LLM Judging</a></div>
    <div class="paper-meta">
      📅 2026-08-18
      | 💬 Accepted at ICML 2026. 23 pages (9 main plus appendix), 7 figures, 11 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as scalable judges in pairwise evaluation, but they remain prone to miscalibration and biases. We propose \textsc{Scope} (Selective Conformal Optimized Pairwise Evaluation), a framework that calibrates an acceptance threshold so that, under exchangeability, the error rate among non-abstained judgments is at most a user-specified level $α$. To supply \textsc{Scope} with a bias-neutral uncertainty signal, we introduce Bidirectional Preference Entropy (BPE), which queries the judge under both response positions and converts the order-averaged preference probability into an entropy-based score. Across various pairwise judging benchmarks, BPE outperforms standard confidence proxies in calibration and discrimination, while \textsc{Scope} consistently satisfies the target risk bound (empirical FDR $\approx 0.097$--$0.099$ at $α=0.10$) and retains substantial coverage. Compared to vanilla baselines, \textsc{Scope} accepts up to $2.4\times$ more judgments under the same risk constraint, demonstrating that BPE enables reliable and high-coverage LLM-based evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.18017v1">Can Large Language Models Explain Flight Safety Events? A Prior-Guided Semantic LLM-based Approach</a></div>
    <div class="paper-meta">
      📅 2026-08-18
      | 💬 14 pages, 6 figures, submitted to IEEE Transactions on Intelligent Transportation Systems
    </div>
    <details class="paper-abstract">
      Improving flight safety with flight data requires not only accurate detection of risk events, but more importantly, clear interpretation of their underlying causes at the level of pilot control behavior. Existing explainable AI techniques, such as feature importance maps, often require considerable domain knowledge to translate them into operationally meaningful explanations. Large Language Models (LLMs), which excel at language reasoning, bring a promising solution to this issue. However, applying LLMs in this domain presents key challenges such as modal inconsistency, limited classification ability, scarcity of task-specific data for fine-tuning, and lack of domain knowledge. To overcome these challenges, we propose FlightLLM, a prior-guided semantic LLM-based approach for interpretable flight safety analysis. Specifically, we first perform feature engineering to address modal inconsistency, combining statistical descriptors with physically meaningful flight indicators. This representation is further processed by a Semantic Discretization module, which converts abstract numerical patterns into qualitative descriptions that are more compatible with language reasoning. In addition, since LLMs are not inherently strong classifiers, CatBoost is incorporated as a statistical expert, and its prediction results are injected into the prompt as prior guidance. A contrastive few-shot learning strategy is further adopted to compensate for limited data. Finally, we design structured prompts to embed aviation-specific knowledge into the inference process. Using hard landing, a representative risk event with complex causal mechanisms, as an anchor point, we evaluate FlightLLM on a dataset of 704 real-world A320 flight samples. Experimental results show that the proposed approach achieves competitive classification performance while generating direct and reasonable explanations for event causes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.03502v2">Hybrid LLM-Augmented Reinforcement Learning Agents for Complex Sequential Decision Tasks</a></div>
    <div class="paper-meta">
      📅 2026-08-18
      | 💬 This submission is withdrawn because the uploaded manuscript does not accurately reflect the intended structure or results. Several components referenced in the text are incomplete or not represented in the PDF, and the current version may mislead readers. The work is therefore withdrawn to maintain clarity of the record
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently shown strong capabilities in reasoning, planning, and tool-use, enabling new forms of autonomous agents. However, LLM-based agents struggle with long-horizon sequential decision tasks that require precise action optimization and environment interaction. Reinforcement Learning (RL), while effective for sequential control, often lacks the high-level abstraction and task decomposition abilities needed for complex scenarios. This paper introduces an LLM-Augmented Reinforcement Learning Agent that integrates LLM-driven planning with RL-based action optimization. The proposed architecture leverages the LLM to generate subgoals, structured plans, and contextual guidance, while the RL agent refines low-level actions through interaction with the environment. Experiments on sequential decision tasks demonstrate improved sample efficiency, higher success rates, and more coherent action trajectories compared to RL-only and LLM-only baselines. This hybrid paradigm highlights a promising direction for building more capable autonomous systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.28895v3">LLM-Based Generative Retrieval for Snapchat Content Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-08-18
    </div>
    <details class="paper-abstract">
      Pretrained large language models (LLMs) are promising retrieval engines because they combine rich semantic priors, strong sequence modeling capabilities, and favorable scaling behavior. However, turning a pretrained LLM into a generative retriever in production deployment raises several challenges: the model must learn an internal item vocabulary that was absent from pretraining, and generate valid item identifiers under strict latency and cost constraints. We address these challenges through the design and launch of SnapLGR, an LLM-based generative retrieval system for short-video recommendation at Snapchat. The system is built around three main designs. First, we construct semantic identifiers (SIDs) from multimodal item embeddings and enhance them with Personalized PageRank (PPR)-based co-engagement contrastive learning, resulting in improved codebook utilization, reduced collisions, and infused collaborative signal. Second, we use continued pretraining (CPT) to ground the introduced SID tokens before supervised fine-tuning (SFT) on user interaction sequences. Third, we make SnapLGR serving practical through TensorRT-LLM CUDA-backed beam search and a decentralized worker-loop architecture. In a live A/B test, the launched system increased View Time by 0.37%, Time Spent by 0.09%, Deep Sessions by 0.18%, and Deep Sessions Unique User by 0.11% relative to the existing TIGER-style generative retrieval baseline. We then decompose this offline gap under a fixed tokenizer and quantify the gains due to model architecture, scaling, and pretraining. Overall, our deployment shows that successful production SnapLGR requires joint design across representation learning, vocabulary grounding, and efficient training and serving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.18008v1">Policy-Invariant Reward Shaping from LLM Feedback: A Framework for Hybrid RL Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-18
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      Combining large language models with reinforcement learning is increasingly explored, yet the theoretical status of LLM-derived reward signals is often left implicit. We formalize the hybrid LLM-planner and RL-controller architecture as a Goal-Augmented Markov Decision Process and show that when the LLM per-state progress score is used as a bounded potential function, the resulting shaping term preserves the optimal policy set even when the LLM scores are inaccurate. This guarantee is stronger than what general LLM-as-reward approaches provide. We verify the result numerically on a small MDP under four potential configurations, including an adversarial one scaled to twenty times the base reward magnitude.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.17994v1">Judge, Retrieve, or Abstain: Uncertainty-Guarded LLM Judging with Provable Risk Guarantees</a></div>
    <div class="paper-meta">
      📅 2026-08-18
      | 💬 Accepted at Conference on Language Modelling 2026
    </div>
    <details class="paper-abstract">
      Using LLMs as judges has become standard practice for evaluating model outputs at scale. This is particularly common for subjective, open-ended tasks such as assessing helpfulness or alignment, where no single reference answer exists. However, objective tasks introduce a distinct reliability challenge for reference-free LLM judging. In the absence of a reference answer, the judge evaluates factual correctness either through its parametric knowledge or through tool augmentation. Although the former enables efficient evaluation, the judge may hallucinate or lack sufficient evidence for its verdict. Conversely, tool augmentation can provide additional evidence but introduces extra computational cost and requires an appropriate mechanism to determine when and how that evidence should be used reliably. More importantly, neither approach alone provides formal control over the risk of accepted verdicts or guarantees their reliability at a specified level. We propose a risk-controlled framework that calibrates uncertainty thresholds on a held-out set so that the false discovery rate among accepted verdicts remains below a user-specified level~$α$ with high probability, using finite-sample Clopper--Pearson intervals. When the parametric mode is not sufficiently confident, the instance is routed to a retrieval-augmented mode, where the judge gathers web evidence and re-evaluates the instance under a second calibrated threshold. The finite-sample guarantee carries over to this two-threshold routing without additional assumptions. Across open-domain QA benchmarks and judges of varying scales, the framework maintains the target error rate while achieving substantially higher coverage than single-mode baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.07507v2">TCMIIES: A Browser-Based LLM-Powered Intelligent Information Extraction System for Academic Literature</a></div>
    <div class="paper-meta">
      📅 2026-08-18
    </div>
    <details class="paper-abstract">
      The rapid growth of academic publications has created a need for tools that extract structured knowledge from unstructured scientific texts. Although large language models (LLMs) can perform natural language understanding and information extraction, existing solutions often require specialized infrastructure, programming expertise, or fine-tuned domain-specific models, which limits their accessibility for researchers in specialized fields. This paper describes TCMIIES (Traditional Chinese Medicine Information Intelligent Extraction System), a browser-based, zero-installation platform that uses commercial LLM APIs to perform structured information extraction from academic literature. The system employs a schema-guided prompting framework with automatic system prompt generation, allowing researchers to define custom extraction schemas through a graphical interface without programming. TCMIIES features a pure front-end architecture that processes all information locally in the browser, supports five major LLM providers (DeepSeek, OpenAI, Qwen, Zhipu AI, and custom OpenAI-compatible endpoints), implements concurrent batch processing with automatic retry mechanisms, and provides intelligent field mapping for Chinese academic databases including CNKI and Wanfang. Evaluation across multiple extraction scenarios in Traditional Chinese Medicine research shows structured output compliance rates exceeding 94\% and extraction accuracy approaching but below expert-level agreement ($κ=0.82$ as reference). The system offers a flexible, privacy-preserving, and cost-effective solution for domain researchers who need to process literature at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.16806v2">Breaking Planner Integrity Boundary: Enviroment State-Text Injection Attack on LLM-Driven Embodied Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-18
      | 💬 submitted to USENIX Security 2027
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-driven embodied agents rely on environment states to interpret scenes, generate high-level plans, and drive physical execution, making planner-visible state representations a critical security boundary. Existing attacks primarily manipulate user instructions, prompt contexts, model behavior, or perceptual inputs, while paying limited attention to whether environment-state text itself can serve as deceptive task evidence and propagate beyond planning to affect execution outcomes. Because embodied tasks are constrained by entity grounding, action preconditions, spatial relations, and environmental constraints, planning deviation alone does not guarantee adversarial execution. To address this gap, we investigate environment-state text as an independent attack surface and present the first closed-loop Environment State-Text Injection (ESTI) attack for LLM-driven embodied agents. Without modifying the original user instruction, model parameters, or executor, ESTI reformulates an adversarial objective as false state evidence compatible with the current environment and influences planning and execution through object properties, spatial relations, affordances, task-stage rules, and execution feedback. We further develop ESTI-Bench to evaluate attack propagation across the planning-to-execution closed loop and compare ESTI with Vanilla IPI, EIRAD, and BADROBOT across ProgPrompt/VirtualHome, VoxPoser/RLBench, and AI2-THOR/iTHOR. ESTI consistently outperforms existing baselines, improving planning-level and execution-level attack success rates by up to 89.32\% and 43.69\%, respectively. Further analysis shows that grounding, consistency, and executability jointly determine whether manipulated state evidence can propagate through the embodied closed loop and produce verifiable environmental changes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.17948v1">SIGMA: SHAP-Guided Implicit-Trajectory Generation for Metadata-Free LLM-Based AutoFE</a></div>
    <div class="paper-meta">
      📅 2026-08-18
    </div>
    <details class="paper-abstract">
      Recent research has leveraged Large Language Models (LLMs) to enhance Automated Feature Engineering (AutoFE) through semantic descriptions and trajectory-based prompting. However, there exist two challenges that limit their applicability and scalability in long-horizon optimization: (1) semantic metadata is unavailable in many practical settings, and (2) trajectory accumulation increases the risk of exceeding the context window, while without it, the generation process can become unstable, leading to becoming stuck in the local optima and a high duplicate rate of generated features. To this end, we propose a SHAP-enhanced Implicit-trajectory Generation for Metadata-free AutoFE (SIGMA), a scalable constant-context optimization framework. SIGMA leverages SHAP values to provide task-aware signals for guiding group feature generation instead of semantic information. In addition, we adopt an EXposed-feature Implicit Trajectory (EXIT) approach, where the exposed features in the prompt implicitly represent the trajectory. Empirical results demonstrate that SIGMA achieves performance comparable to the state-of-the-art (SOTA) LLM baselines with a nearly constant prompt length. Notably, EXIT significantly reduces the duplicate ratio of generated features from 37.2% to 6.8%. At the same time, SIGMA matches traditional SOTA performance with only 5.4 features on average, demonstrating substantial efficiency gains in feature utilization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.17933v1">EvoTS-Agent: A Self-Evolving LLM Agent for Financial Time Series Change Point Detection</a></div>
    <div class="paper-meta">
      📅 2026-08-18
    </div>
    <details class="paper-abstract">
      Financial time series exhibit non-stationary and heterogeneous statistical properties, making change-point detection challenging because no single unsupervised algorithm performs consistently across assets and market regimes. Conventional workflows consequently depend heavily on expert-driven model selection, feature design, and hyperparameter tuning, limiting their scalability and adaptability. We propose EvoTS-Agent, a validation-guided self-evolving LLM agent for autonomous financial time-series change-point detection. EvoTS-Agent first performs curated exploratory data analysis to characterize dataset properties and initialize candidate detection models. It then evolves executable experiment trajectories through three complementary operators: \textit{Revision} exploits the current best solution, \textit{Alternative Strategy} explores fundamentally different modeling directions when progress stagnates, and \textit{Recombination} synthesizes complementary evidence from high-performing trajectories. Validation feedback guides trajectory evolution throughout the search, enabling the agent to adapt its detection pipeline to the statistical characteristics of each dataset while preserving reliable optimization. Experiments across four benchmark datasets demonstrate that EvoTS-Agent consistently outperforms existing LLM-based agents while maintaining a 100\% execution success rate across all evaluated backbone LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.09106v2">LexKairos: Benchmarking Legal Temporal Capabilities in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-18
      | 💬 15 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated strong performance across a wide range of legal tasks. In legal practice, time is a critical concept that governs the validity of statutes, the progression of legal cases, and the enforcement of procedural deadlines. However, legal temporal capabilities remain underexplored in existing legal AI benchmarks. To address this gap, we propose LexKairos, a comprehensive benchmark for evaluating the temporal capabilities of LLMs in the Chinese legal context across three dimensions: statutory temporal knowledge, case temporal modeling, and statute-case temporal reasoning. LexKairos comprises nine sub-tasks drawn from real-world Chinese judicial cases and statutes. We conduct systematic evaluations of eight LLMs under multiple inference settings, including vanilla, Chain-of-Thought (CoT), and thinking modes. Our results show that Gemini-3-Flash achieves the strongest overall performance, yet even the best-performing model exhibits notable limitations on tasks demanding precise time-sensitive statutory metadata recall or complex reasoning in time limits, indicating that legal temporal knowledge and reasoning remain open challenges for current LLMs. Data and code are available at https://github.com/thunlp/LexKairos.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.17843v1">Encoded but Not Actionable: Auditing the Decode-Generate-Steer Gap in Frozen LLMs for Geometric Constraints</a></div>
    <div class="paper-meta">
      📅 2026-08-18
      | 💬 13 pages, 7 figures, 8 tables, including appendices
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated strong performance on structured reasoning tasks, but what they encode and whether it informs model behavior remain unclear. We investigate this question through geometric reasoning, using parametric CAD constraints as a controlled testbed for separating local pairwise relations from sketch-level constraint status. By probing the hidden states of six frozen decoder-only LLMs, we examine four properties: linear decodability, forced-choice generation, activation-level influence, and behavioral steerability. Pretraining substantially improves the decoding of local geometric relations, and this advantage persists after accounting for positional cues with shuffled-order controls. In contrast, sketch-level DOF status is already highly decodable from randomly initialized representations and improves only modestly with pretraining, indicating that much of its probe performance is available without learned weights. Further analyses show that decodable information is not always actionable. Generation often fails to express this information, and on the two intervention-tested backbones, activation-restoration effects at the patched entity position vanish while decodability persists across depth. Mean-difference steering also does not reliably control outputs. These results show that decodability, generation, activation-level influence, and steerability can diverge in the tested setting. The audit provides a controlled way to distinguish failures to encode geometric structure from failures to express or control encoded information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.17836v1">Leveraging Association Context Retrieval in Knowledge Edit- ing to Build White-Box Attacks on LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-18
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are granted increasing autonomy, it is essential to investigate methods that can induce unsafe behavior. We propose a novel white-box attack inspired by locate-then-edit approaches from the field of Knowledge Editing. Our choice is motivated by the observation that models edited with such schemes tend to assign unusually high prediction probabilities to the edit target, a property that is particularly advantageous when designing attacks. We modify the editing framework by incorporating as- sociative knowledge retrieved from the model, thereby extending constraint removal to an entire thematic category rather than being limited to prompts from a predefined dataset. Experiments with various archi- tectures demonstrate improved attack effectiveness over competing methods without dealing critical damage to general model performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.17827v1">From Global Benchmarks to Local Evaluations: Benchmarking LLMs for the German Public Sector</a></div>
    <div class="paper-meta">
      📅 2026-08-18
      | 💬 Accepted as non-archival paper at Eval4SD (co-located with KONVENS 2026)
    </div>
    <details class="paper-abstract">
      Public institutions face a persistent challenge in selecting LLMs suited to their specific context. Existing benchmarks, however, are of limited use as they primarily reflect English-language and US-centric settings, and often only evaluate task performance. In this paper, we present first results of MÖVE, a holistic evaluation framework for the German public sector, examining three rarely considered governance dimensions: energy consumption, provider transparency, and knowledge of German-party positions. Our results reveal significant trade-offs, with no single model excelling across all dimensions: estimated energy consumption varies more than 60-fold and is not explained by model size alone, information disclosure varies systematically across providers, and European models do not exhibit stronger knowledge of German party positions. Model selection for public institutions thus cannot rely on performance rankings alone. Instead, evaluations should also reflect the governance requirements of the deployment context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.17826v1">Bounded-State Restoration: Decoupling Local Restore Capacity from External LLM State</a></div>
    <div class="paper-meta">
      📅 2026-08-18
      | 💬 16 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Hierarchical KV-cache systems can retain long-context LLM execution state beyond GPU memory, but retention capacity does not determine the local memory required to make that state executable again. We isolate this second resource as the restoration working set (RWS): the peak local staging state whose lifetimes overlap during restoration. In the pinned upstream LMCache whole-plan path, measured full-reuse points for 1.956, 7.823, and 15.646 GiB/rank states first succeed at 2, 8, and 16 GiB L1 rungs, with successful L1 peaks of 1.956, 7.824, and 15.648 GiB/rank. We introduce Bounded-State Restoration (BSR), which separates complete discovery from local residency. BSR probes the complete reusable prefix without materializing the whole hit in L1, then installs confirmed state through a reusable window of at most $W$ chunks. Under bounded auxiliary state, peak restoration capacity is $O(W)$ while total transfer and installation work remains $Θ(|S|)$. Because reusable state spans heterogeneous allocator groups and tensor-parallel ranks, BSR uses a request-level commit rule: partial installation is never exposed as a valid reusable prefix; failures invalidate the advertised prefix and fall back to a lower valid tier or deterministic recomputation. On DeepSeek-V4-Flash with TP=2 across two DGX Spark nodes, a clean no-resume sweep grows external state from 1.956 to 31.277 GiB/rank while measured L1 RWS remains exactly 500.75 MiB/rank at $W=32$, a 63.959x largest-state external-to-live-staging ratio. A second fresh 524K-token run repeats the largest-state acceptance result. Evaluated tier and rank-asymmetric failures expose either complete reuse or zero external reuse before fallback. A matched SSD optimization reduces 512K restore TTFT from 43.1 to 17.6 seconds without changing RWS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23671v4">Can LLMs Reliably Self-Report Adversarial Prefills, and How?</a></div>
    <div class="paper-meta">
      📅 2026-08-18
      | 💬 In submission
    </div>
    <details class="paper-abstract">
      Prior work shows that large language models (LLMs) exhibit introspective capability on benign tasks. We extend the question to safety contexts and examine how reliably a model can recognize that its own prior response was elicited by an adversarial prefill attack. Across ten open-weight instruction-tuned LLMs from 3B to 70B and four safety benchmarks, no model reliably recognizes its own compromised outputs, with models claiming intent on prefilled responses at an average rate of 25.3%. Introspective signal stems primarily from reasoning about safety and refusal. Orthogonalizing models' weights against the refusal direction collapses the gap between claim rates on prefilled and natural outputs to near zero, though the direction is not its unique mediator. The signal also depends on the probe: framing the question as internal intention versus external tampering elicits qualitatively different responses on the same models. Training models to mimic correct introspective answers or pursue an introspective objective can improve the accuracy of introspection, but such training does not transfer to the tampering probe and counterintuitively raises attack success rate under adversarial prefill on most models, amounting to a partial mitigation. These findings outline mechanisms underpinning the observed introspective signals in safety contexts and highlight risks in the reliability of LLM self-reports. Our code is available at https://github.com/ngqm/prefill-introspection.
    </details>
</div>
