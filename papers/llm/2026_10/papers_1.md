# llm - 2026_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.22333v1">Evaluation Awareness Beyond English: Do LLMs Act Differently When Tested in Indic Languages?</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can often tell when they are being tested. This ability, known as evaluation awareness, can lead models to act more cautiously during safety evaluations than in real use. Existing work has studied evaluation awareness almost entirely in English, although models serve users in many languages. Indic languages are spoken by over a billion people, yet evaluation awareness remains almost entirely unstudied in them. We test four instruction-tuned models from three families in five Indic languages alongside English. A set of cues varies whether the model is told that it is monitored and whether the monitoring is a safety test. Models refuse harmful requests more often when they are told that they are being tested. This increase appears in all four models and all six languages. In Indic languages, refusal of harmful requests rises by 14.4 percentage points on average. The test cue also brings Indic refusal of harmful requests level with English, but raises refusal of benign Indic requests from 6.9% to 11.3%. We release all 47,880 judged responses to support further work. These results suggest that safety evaluations in Indic languages may not reflect how models treat their everyday users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.22316v1">Learning to Steer LLM Agents and Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 21 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based agents and multi-agent systems increasingly tackle complex tasks through long-horizon tool use and inter-agent collaboration. However, these systems are prone to deviating from the overall goal as execution unfolds, which wastes execution resources and leads to undesired outcomes, a failure termed goal drift. In this work, we propose a steering agent that supervises execution on the fly and steers the system back on track when drift occurs. The steering agent observes the execution context, deciding when to intervene with a low-cost fast path and how to intervene with a high-cost slow path. To unlock its steering intelligence, the ability to assess goal drift and generate effective interventions, we jointly train the two paths with reinforcement learning, combining a pairwise ranking objective with group-relative policy optimization (GRPO). Experiments on information-seeking search tasks show that the steering agent provides effective supervision for both single-agent and multi-agent systems, improving their performance and efficiency. The learned steering intelligence generalizes to out-of-domain tasks, transfers to unseen executors, and supports controllable intervention frequency without retraining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.22312v1">Grid-Bench: Can LLMs Predict How Discrete Worlds Change?</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 39 pages, 15 figures, 15 tables, including appendices. Project website: https://grid-bench.github.io/ ; Code: https://github.com/grid-bench/Grid-Bench ; Data: https://huggingface.co/datasets/xzx34/Grid-Bench
    </div>
    <details class="paper-abstract">
      Spatial instructions describe how to move objects, transform regions, or modify their surroundings. When several operations are applied in sequence, each acts on a configuration shaped by earlier changes. How accurately can large language models (LLMs) predict the resulting spatial state? We introduce Grid-Bench, an executable benchmark in which eight families of operations transform discrete worlds represented as two-, three-, and four-dimensional grids. Given an initial grid, explicit rules, and an event history expressed as text, a model predicts the complete final grid. Reference execution makes every predicted value checkable, supporting comparisons across spatial operations, history lengths, and workloads. Evaluating six LLMs shows that complete-state accuracy falls sharply as histories grow, with substantial divergence among models that perform similarly on single events. Models with similar exact-match accuracy can nevertheless differ substantially in the number of incorrect cells on shared instances. Even over long histories, some models recover many correct changes beyond what copying the initial state would yield. Grid-Bench makes both complete recovery and these differences in returned states directly measurable across a range of spatial conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.22298v1">AI Consilium: a Benchmark of LLM Clinical Multidisciplinary Second Opinions</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can generate differential diagnoses for complex cases, but a useful second opinion must also be factually grounded, clinically coherent, actionable, and safe. We developed a multidimensional benchmark using 300 diagnostically resolved NEJM--MedQA cases: 110 for physician reference ratings and Judge calibration and 190 for held-out evaluation. Eight single-model pipelines generated independent role-specific opinions and interactive-consilium reports under free-form and structured clinical-reasoning conditions. A physician-calibrated LLM Judge evaluated 3,040 final reports and the corresponding structured primary-role outputs. Model identity separated report quality more strongly than response format. Structured final reports had higher overall mean scores in seven of eight model pipelines, with their clearest advantage in management (2.740 vs. 2.436). Compared with the structured pre-consilium primary-role opinion, the complete structured synthesis had higher pooled scores across all five shared criteria, with the largest increases in problem prioritization, factual grounding, and reasoning coherence. GPT-5.6-Sol led every evaluated component and reached consensus in the fewest review rounds. These findings show that the benchmark can distinguish model capability, the contribution of multidisciplinary synthesis, criterion-specific effects of response structure, deliberation dynamics, and inference requirements within one common-case evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.22292v1">CAT-alyzing Honesty: Continuous Adversarial Training for Transferable LLM Honesty</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 9 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are becoming prevalent in scenarios where honesty is critical. However, despite alignment training, these models still exhibit dishonest behaviour, such as outputting claims that contradict some knowledge or are not supported by what is accessible in context. We introduce Continuous Adversarial Training for Consistent Honesty (CATCH), inspired by a method originally developed to robustly refuse harmful queries. CATCH trains the model towards an honest answer and away from a dishonest one while maintaining the model's capabilities. We evaluate two architectures (Llama-3.3-70B-Instruct and Qwen3-3.6-27B). We show that trained with only hundreds of targets built from self-reporting scenarios, CATCH learns an honest behaviour that robustly generalizes and transfers across distinct benchmarks. It consistently reduces dishonesty under pressure on MASK, increases confessions in the multi-agent game of Among Us, and improves the discovery rates of the model's hidden behaviours in AuditBench. We show that CATCH outperforms prompting and supervised fine-tuning in improving honesty, and it is on par with state-of-the-art honesty activation steering. However, CATCH surpasses honesty steering in maintaining utility, thanks to a novel choice of utility data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.22273v1">Librarian Agents: Training LLMs to Curate Long-Context Workspaces that Anticipate Future Queries</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      Reasoning over long document contexts remains a challenge for large language models (LLMs). Standard retrieval methods are inherently reactive, bottlenecking agents with complex, cross-document reasoning at inference time, while precomputed structures like knowledge graphs often fall short because they rely on rigid heuristics. Instead, can we allow agents to perform anticipatory directed curation: learning to proactively reorganize unstructured corpora prior to inference to efficiently serve diverse future queries. We build on a key asymmetry in modern agentic harnesses (e.g., Claude Code): while LLM agents are adept at navigating existing filesystems to solve an immediate task, they are rarely trained to construct reusable workspaces for future use. To instantiate this capability, we introduce Librarian Agents, a framework in which a builder agent transforms raw documents into a hierarchical filesystem of synthetic files and reference links, which forms a workspace tailored for a downstream answering agent. We formulate workspace construction as a Markov decision process and optimize the builder policy via reinforcement learning to directly maximize downstream answering accuracy. Empirically, Librarian Agents outperforms the strongest retrieval and precomputation baselines by 33 percentage points on BrowseComp-Plus, amortizes upfront construction costs across repeated queries, and generalizes zero-shot to unseen domains on BRIGHT with a 31-point average accuracy improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.22249v1">Not X, but When: Diagnosing a Mandarin LLM Writing Tic through Contextual Appropriateness</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      Recurring expressions in model-written text are often treated as signatures of machine authorship, yet most of them serve legitimate functions. Whether such an expression is a tic depends on whether the model knows when it is called for. This judgment of appropriateness, central to communicative competence, remains largely unexamined for language models. We study it through the Mandarin corrective construction bushi X, ershi Y ("not X, but Y"). Drawing on Gricean pragmatics and corrective negation, we introduce a decision tree that classifies uses as substantive, optional, or defective by testing whether the context makes X a live alternative and whether rejecting X is informative. Five commercial models use the construction 2.5 to 9.2 times as densely as a human reference. Of 307 annotated uses, only 1.30% are defective while 52.8% are optional, that is, correct but unneeded, so models fall short in appropriateness far more often than in correctness. Several reward models favor the construction even over quality-matched variants, an LLM judge favors it even where the context does not license it, and both supervised fine-tuning and preference optimization tend to raise its frequency in open-weight models. We hope this work encourages the development of models attentive to the appropriateness of their expressions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.22232v1">WaveLLM: Wavelet-Domain LLM Entropy Modeling for Lossless Image Compression</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 Accepted at ACM Multimedia 2026 (MM '26). 10 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Lossless image compression fundamentally relies on accurate entropy modeling, driving a recent shift toward large language model (LLM)-based methods. However, existing approaches mainly emphasize the modeling capacity of LLMs while directly serializing images into pixel or residual sequences, leaving heterogeneous signal patterns entangled within a single prediction stream. In this paper, we argue that effective entropy modeling depends not only on model capacity, but also on the predictability of the input representation. Motivated by this insight, we propose WaveLLM, a wavelet-domain LLM-based entropy model for lossless image compression. By decomposing an image into multi-level wavelet subbands, WaveLLM converts a heterogeneous prediction problem into a set of more structured and statistically homogeneous subproblems that are more amenable to autoregressive probability estimation. To bridge numerical coding symbols with the LLM's native text token space, we introduce a dedicated numerical interface featuring value embeddings and a coefficient prediction head. Furthermore, a task-adaptive prompting strategy is designed to enable a single shared model to adapt to diverse subbands within a unified framework. Built on Llama-3.2-3B with LoRA adaptation, WaveLLM achieves 2.51, 2.24, and 2.27 bpsp on Kodak, CLIC.pro, and DIV2K, respectively, outperforming the PixelCNN-based codec using the same wavelet transform by 0.43, 0.36, and 0.46 bpsp, and surpassing recent LLM-based lossless compressors on all three benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.22187v1">Change the Number, Change the Answer: Assessing Rating Scale Sensitivity in LLM-Based Text Scoring</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in computational social science to assign scalar scores to social and psychological constructs. However, studies use different rating scales, and whether their scores remain comparable across scale designs is understudied. To address this gap, we define and systematically investigate scale sensitivity, the variation in LLM-generated scores caused by rating scale design. We distinguish numerical sensitivity, changes in normalized scores, from semantic sensitivity, changes in score interpretation. To assess both, our experimental design crosses 2 formats (continuous vs. ordinal), 4 ranges (3, 5, 11, and 21 points), and 2 signs (signed vs. unsigned) across 3 tasks (political leaning, sentiment, and toxicity), 2 prompting strategies (zero-shot and rubric-based), and 5 LLMs. We find that changing one design element shifts normalized scores by about 0.10 on average and changes the category of roughly 20% of texts. Sensitivity is greatest for toxicity, the only unipolar construct examined. Also, rubric-based prompting does not reduce sensitivity compared with zero-shot prompting. We provide practical guidance for designing more reliable LLM-based text scoring and highlight the need for robustness checks across alternative scales.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.22172v1">Efficient KV Cache Migration for Geo-Distributed LLM Inference in Collaborative Edge Computing</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 Accepted by ICDCS 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown exceptional capabilities but rely heavily on centralized cloud GPU clusters, leading to high bandwidth costs, latency, and privacy issues. Edge computing offers a solution by deploying LLMs closer to users on edge servers, yet single edge servers often cannot meet LLMs' heavy computational demands. Collaborative Edge Computing (CEC) addresses this challenge by integrating resources of multiple geo-distributed heterogeneous edge servers, which jointly host small model shards and associated Key-Value (KV) cache for distributed inference. However, edge servers may disconnect or easily become overloaded, making it essential to migrate stateful KV cache to other servers for ensuring continuous service. Existing cloud-based KV cache management methods, such as direct cross-GPUs transmission and recomputation from input tokens are ill-suited for bandwidth-constrained and resource-limited edge environments. In this work, we propose EdgeFlow, a novel hybrid KV cache migration scheme that integrates direct KV cache transmission with activation-based recomputation, leveraging parallel computation and communication to minimize migration time. On this basis, we formulate a multi-node KV migration optimization problem and design a 2-approximation algorithm based on improved linear programming relaxation and bipartite graph techniques. Extensive experiments on a real-world distributed LLM inference platform demonstrate that EdgeFlow reduces migration time by up to 45%, significantly enhancing the robustness and efficiency of distributed LLM inference in collaborative edge computing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.22139v1">Are You Sure Your LLM Is Sure? Repairing and Auditing Confidence</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 main paper 10 pages; supplementary 44 pages; 5 figures
    </div>
    <details class="paper-abstract">
      Confidence scores determine whether an LLM answer is accepted, reviewed, or regenerated, yet scalar recalibration can only correct errors visible through the score itself. We introduce RADAR (Repair, Adopt, and Diagnose After Recalibration), a reliability framework that separates confidence repair from deciding whether to deploy the repair and auditing the errors that remain. Under squared loss, we decompose prediction risk into outcome noise, information hidden by the score, and mapping error; restricting repair to order-preserving maps additionally introduces a pooling gap. For repair, we propose Probe-Anchored Isotonic Calibration (PAIC), which learns a monotone map from correctness-probe predictions on unlabeled target prompts. PAIC's excess risk depends on the probe's score-conditional bias and fitting error rather than its overall accuracy. For adoption, we derive a paired procedure with finite-sample control of false improvements. For auditing, we develop a prompt-clustered kernel test that detects feature-aligned residual errors while accounting for dependent answers from the same prompt. Across multiple LLM checkpoints, benchmarks, and distribution shifts, richer predictors can improve confidence estimation, but projecting their information back to a scalar score can lose these gains, and lower prediction risk need not yield better review decisions. Across 47,400 replayed raw-score allocations, paired betting adopts candidates with lower held-out test risk in 42.95% of allocations, versus 15.09% for one-sided empirical Bernstein. RADAR thus provides a principled workflow to repair what a scalar score can express, adopt a repair only when supported by evidence, and diagnose systematic errors that recalibration leaves behind. Code and data will be released upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.22138v1">Believed but Not Said: Measuring Hidden Belief Collapse in Multi-Agent LLM Discussions</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 14 pages, 6 figures
    </div>
    <details class="paper-abstract">
      LLM systems increasingly rely on groups of agents that discuss, critique, and check one another, assuming that peers will catch individual errors. We test this assumption against a minimal adversary. In a ten-agent factual discussion, one or three adversarial agents repeatedly assert a plausible false answer. After each round we probe each benign agent's private answer through a separate call whose response is hidden from its peers, and we separately label its public stance toward the claim. Across six models and three factual-QA datasets, groups remain stable without an attacker: only 0.6% of benign agents finish with the false answer. A single adversarial agent increases this rate to 44.8% and three to 60.5%. Studies of conformity typically report public agreement without private change; we find the reverse. With three adversarial agents present, benign agents hold the false answer in 51.6% of exposed turns but publicly endorse it in only 19.7%. Much of the influence therefore remains invisible in the discussion, and the false answers persist when agents are asked for their own views. Agents that retain the true answer often fail to defend it; once adopted, the false answer spreads among benign agents, and baseline knowledge does not predict robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.01263v2">Autonomous OSS Threat Detection via Taxonomy-Aligned LLMs</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      Open-source software (OSS) ecosystems face escalating supply chain threats, including typosquatting, dependency confusion, Trojan Source obfuscation, malicious build injection, and CI/CD pipeline poisoning. Sonatype documented over 1.2 million cumulative malicious packages across major registries, while signature-based and rule-based detection approaches lack the semantic understanding needed to generalize across evolving attacks. This paper proposes a taxonomy-aligned large language model framework for automated detection and multi-class classification of OSS supply chain threats. We introduce an AV-xxx taxonomy covering five semantically distinct attack categories and curate 999 verified real-world incidents from GitHub Security Advisories, CISA alerts, government security databases, and commercial threat intelligence reports, spanning 2018 to 2026 across eight package ecosystems. Using taxonomy-aligned prompting with GPT-4, the framework achieves 97% accuracy and 97% macro F1 across all five categories. Evaluation against eight methods five traditional machine learning baselines, one zero-shot open-source LLM, and two fine-tuned neural models reveals that fine-tuned Llama 3.1 8B (70.5%) and SecRoBERTa (77.5%) underperform a simple TF-IDF classifier (82.3%), while Mistral 7B without taxonomy alignment achieves 65.7%. These findings highlight taxonomy-aligned prompting over model scale, domain pretraining, or fine-tuning as the critical performance factor. Cross-ecosystem analysis shows varying classification difficulty, with npm achieving 65.5% baseline accuracy versus 100% in smaller ecosystems. GitHub Actions exhibits a substantial accuracy F1 divergence (82.4% accuracy, 0.367 macro F1), indicating systematic class imbalance and highlighting ecosystems where LLM-based contextual reasoning offers the greatest benefit. Our dataset, scripts, and code are publicly available in Github.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.22127v1">Explaining the Roles of Experts in Mixture-of-Experts LLMs for Personalization</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Personalization of Large Language Models (LLMs) is increasingly important for user-centric applications, yet prior methods mainly adapt dense LLMs, whose polysemantic neurons, each encoding multiple unrelated concepts, hinder localizing the components behind personalization, and thus understanding and controlling it. We propose to personalize Mixture-of-Experts (MoE) LLMs, which sparsely route each token to a few specialized experts, reducing polysemanticity and enabling such localization. We adapt three open-weight MoE LLMs on a large-scale personalization dataset and train crosscoders on paired activations of the personalized and non-personalized models to learn interpretable personalized latents, from which we identify the experts encoding distinct personalization features. The personalized MoE LLMs perform competitively with dense LLMs while activating only a fraction of their parameters, exhibit lower polysemanticity than dense LLMs with similar active parameters, and, within the same model family, yield personalized latents that are less shared with the non-personalized model. Causal interventions show that these latents suffice to recover the personalized model's behavior and that boosting their associated experts moves the non-personalized model toward it, revealing distinct routing paths for different personalization features.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.22124v1">PROBE: Learning to Audit Policy Compliance in Tool-Using LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 27 pages, NeurIPS 2026
    </div>
    <details class="paper-abstract">
      Modern agentic systems extend language models with external tools, enabling them to complete complex tasks through multi-turn interactions. As these agents are deployed in consequential settings, they are often constrained by policies governing pricing, privacy, confirmation, safety, and other requirements. However, existing evaluation methods leave a critical gap: capability benchmarks measure what agents can accomplish, while content-safety red-teaming measures harmful generations, but neither tests whether agents comply with predefined policies during otherwise valid tool use. Such policy violations are difficult to audit because their effects may emerge only after several turns, span multiple tool calls, or surface in downstream systems. We propose PROBE (Policy Rule Observation via Behavioral Exploration), an algorithm for training LLM-based auditors that interact with deployed agents as realistic users and probe them for policy violations through multi-turn, tool-mediated dialogue. We formalize agent auditing as a two-player Markov game and propose a composite reward that balances violation discovery, behavioral plausibility, conversation progress, and policy diversity. To study training objectives and reward design, we compare GRPO, DPO, and GFlowNet-based optimization in a controlled setting. Across extensive experiments, auditors trained with PROBE uncover substantially broader and higher-yield policy violations than same-architecture zero-shot adversaries, even when the auditor is significantly smaller than the target agent. Our results show that multi-turn behavioral auditing is a distinct and necessary axis of agent evaluation, complementing capability and safety benchmarks. PROBE offers a scalable path toward automated policy auditing for real-world agentic systems and suggests a co-evolutionary loop in which auditors expose failures and agents improve from auditor feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.22735v3">Explanation Generation for Contradiction Reconciliation with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Existing NLP work commonly treats contradictions as errors to be resolved by choosing which statements to accept or discard. Yet a key aspect of human reasoning in social and professional settings is the ability to hypothesize explanations that reconcile contradictions. For example, "Cassie hates coffee" and "She buys coffee everyday" appear contradictory, but are in fact compatible if Cassie was assigned the daily chore of buying coffee for all her coworkers. Despite advances in the reasoning capabilities of large language models (LLMs), their ability to hypothesize such explanations remains largely unexplored. To address this gap, we introduce the task of reconciliatory explanation generation, where models must generate explanations that effectively render contradictory statements compatible. We propose a novel method of reusing natural language inference (NLI) datasets, and introduce quality metrics that enable scalable automatic evaluation. Experiments with 13 LLMs in thinking and non-thinking modes reveal models' limited success in this task, with longer reasoning chains negatively associated with explanation success. As a proof of concept, we further adapt our introduced task and metrics for reinforcement-learning training, finding that trained models exhibit improved accuracy in non-thinking mode on two narrative reasoning benchmarks (True Detective and MuSR). Our findings highlight reconciliatory explanation generation as a challenge in LLM reasoning and motivate further work to investigate its potential for improving model capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.22092v1">Onboard Lightweight LLM Reasoning for Human-Aware Safe UAV Navigation</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 HEAI -- IROS 2026 Workshop
    </div>
    <details class="paper-abstract">
      Aerial robots operating near people must treat a pedestrian, a parked car, and a wall differently, yet conventional perception--planning stacks reduce all of them to occupied geometry---a \emph{human-aware safety gap} in which category-specific clearance is unavailable to the planner and temporary occlusions remove pedestrian-specific constraints from its map. We present a modular, fully onboard pipeline that closes this gap by turning visual observations into context-dependent safety constraints for real-time UAV navigation. A shared BLIP encoder drives scene captioning and text-conditioned segmentation, which is fused with depth to localize obstacles in 3D. Obstacle-specific risk is inferred by a lightweight LLM (Qwen-2.5-0.5B) stabilized with retrieval-augmented generation and grammar-constrained decoding that restricts its output to a planner-compatible JSON schema. The LLM reasons once per object category rather than per frame: its verdicts are cached per class, so previously seen categories are resolved without inference and new ones trigger a fresh retrieval-grounded query; detections are additionally given persistent identities and held in the map across occlusions. The perception pipeline runs at 10\,Hz on a Jetson Orin NX, while fallback LLM queries take approximately 0.8\,s and remain off the control-critical path. In Gazebo, low-altitude urban navigation among pedestrians and vehicles shows a roughly $3\times$ reduction in collision events over a geometric planner, and semantics-aware landing among moving actors raises success from 34\% to 96\%. The 224-pixel encoder input limits reliable small-object perception at range; conservative pedestrian margins and short-term persistence mitigate this limitation during navigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.12306v5">A large-scale pipeline for LLM-assisted corpus annotation: variation and change in the English consider construction</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      As natural language corpora expand at an unprecedented rate, manual annotation remains a significant methodological bottleneck in corpus linguistic work. We address this challenge by presenting a scalable pipeline for automating grammatical annotation in voluminous corpora using large language models (LLMs). Unlike previous supervised and iterative approaches, our method employs a four-phase workflow: prompt engineering, pre-hoc evaluation, automated batch processing, and post-hoc validation. We demonstrate the pipeline's accessibility and effectiveness through a diachronic case study of variation in the English evaluative consider construction (consider X as/to be/Ø Y). We annotate 143,933 'consider' concordance lines from the Corpus of Historical American English (COHA) via the OpenAI API in under 60 hours, achieving 98%+ accuracy on two sophisticated annotation procedures. A Bayesian multinomial GAM fitted to 44,527 true positives of the evaluative construction reveals previously undocumented genre-specific trajectories of change, enabling us to advance new hypotheses about the relationship between register formality and competing pressures of morphosyntactic reduction and enhancement. Our results suggest that LLMs can perform a range of data preparation tasks at scale with minimal human intervention, unlocking substantive research questions previously beyond practical reach, though implementation requires attention to costs, licensing, and other ethical considerations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.22062v1">Documenting Trofinetide Treatment Phenotypes in Rett Syndrome with a Dual-LLM Consensus Pipeline</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 20 pages, 3 figures, 4 tables; supplementary material (22 pages, 1 figure, 11 tables) included
    </div>
    <details class="paper-abstract">
      Objective: Trofinetide, approved in 2023 as the first Rett syndrome treatment, lacks scalable real-world monitoring because chart review is labor-intensive. We developed a dual large language model (LLM) consensus pipeline to extract 18 treatment-response phenotypes, comprising 4 adverse effects, 13 Rett clinical features, and medication refusal, from clinical notes. Materials and Methods: We analyzed 26,763 notes from 108 patients with confirmed sustained trofinetide exposure at Texas Children's Hospital. The pipeline filters notes by keyword, classifies each through two LLMs requiring consensus, and aggregates a patient-by-timepoint matrix. Architecture and prompt were selected on a diarrhea development set, then frozen and applied to the other 17 phenotypes with the phenotype definition substituted, as an out-of-sample test. The reported prompt comparison scores 183 expert-labeled cells. Results: A prompt requiring diarrhea to be active and severe performed best (84.2 $\pm$ 0.0% accuracy; F1 = 0.674), 47 percentage points above regular-expression matching ($p < 0.0001$). Expert reviewers judged six of the pipeline's calls per phenotype, three positive and three negative, and agreed with 90.2% (92/102). The 13 Rett clinical features are manifestations of the disorder itself rather than adverse effects of trofinetide. Gross and fine motor impairment, language/communication, and stereotypy were documented present in 92-100% of patients at every visit. Discussion: In-context LLM extraction captured multi-domain treatment responses. The outcome is documented presence at each visit, not a direction-of-change judgment. Conclusion: Coupling in-context LLM reasoning with a consensus architecture offers a scalable alternative to keyword and regular-expression extraction for real-world evidence, adapting to new phenotypes and medications through configuration alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.05613v3">LLM Benchmarking via Representation Multi-task Learning</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      Quantifying and evaluating the capabilities of Large Language Models (LLMs) remains a fundamental challenge in modern data science and artificial intelligence. In this paper, we consider LLM evaluation based on their performance across items in multiple benchmark domains (e.g., mathematical reasoning and coding) within a leaderboard framework. Our goal is to address two core questions: (1) How do we derive more accurate domain-specific scores by borrowing information across domains? and (2) How do we define and estimate an overall score that aggregates performance across multiple domains? To solve these problems, we propose a novel statistical framework based on representation Multi-task Learning (MTL) and an item response theory model. Specifically, we define overall and domain-specific LLM traits through an Item Response Theory (IRT) model, and propose an MTL approach to estimate these traits from item-level response data. We develop a computationally efficient estimator and establish its minimax optimality under certain asymptotic regimes. This framework provides a rigorous measurement foundation for systematic LLM evaluation. We conduct extensive simulations, demonstrating the superior performance of the proposed method over competing methods. Crucially for the Applications and Case Studies section, we apply the proposed framework to MMLU response data from the Hugging Face Open LLM Leaderboard, covering 4,272 LLMs and 13,232 items across 56 subjects. The empirical analysis reveals substantial heterogeneity in domain size and difficulty, together with strong positive cross-domain dependence, highlighting the practical value and substantive insights generated by our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.19705v2">SoK: Trading Agents or Market Crashers? Dissecting Robustness and Security Failures in Academic Financial LLM Trading Schemes</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 51 pages, 15 figures, 19 tables, 206 references. SoK paper. v2 (extended version): adds a code-level audit of the 7 open-source schemes, a domain-independent threat model applied to healthcare, law, and other high-stakes domains, an analysis of cascades and agent worms, a discussion of options-trading agents, and 14 open problems
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents already trade with real capital, yet agent-security research has mostly treated them as generic agents. Trading changes the threat: a compromised agent's orders move prices that other participants act on, and an attacker who profits from a crash can trigger the shock that a non-robust agent mishandles. We present FARSIGHT (Financial Agent Robustness and Security Investigation and Global Holistic Taxonomy), a systematization that rates trading agents on robustness to market stress and on security against source manipulation, agent compromise, and agents acting as attackers. It also checks paper-level ratings against released code. Across 15 academic schemes, none earns full credit on any robustness metric, no paper describes a way to halt trading without waiting for LLM inference, and none combines source provenance with cross-source corroboration. In the code of the 7 open-source schemes, no release enforces a stop-loss or position limit outside the LLM, 4 lack a component that their paper or documentation describes, and 2 run model-written code on the host. The same gaps appear beyond finance. Restating FARSIGHT as a domain-independent threat model, we map them onto agents in healthcare, law, software, and crypto. We then trace how compromised agents spread harm, as worms and as cascades that need no further compromise. At published injection rates per hop, a channel read by ten agents lets one compromised agent start an outbreak. We derive 14 open problems and urge deployers to corroborate sources, keep interrupts that bypass the model, and enforce loss limits outside it.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.22031v1">GraphACM: Graph-based Action Consistency Measurement for Uncertainty Quantification in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      Behavioral consistency across repeated executions provides an informative signal for assessing the reliability of large language model (LLM) agents. However, existing approaches typically summarize consistency with a single task-level score, without identifying where behavioral differences occur or whether actions are compared under equivalent decision contexts. We propose a graph-based framework for fine-grained behavioral consistency measurement and uncertainty quantification. Given multiple rollouts of the same task, we merge equivalent states into a directed, action-labeled graph, enabling action comparisons at shared decision states. We introduce node-level and global measures of action consistency, using the global consistency score as a proxy for uncertainty to predict task success or failure. Experiments across three LLMs on ALFWorld, AppWorld, and WebShop demonstrate that our measure is positively associated with task success and achieves an average AUROC of 0.811 for task-success prediction, outperforming the strongest behavioral-consistency baseline (0.729) and verbalized confidence (0.769). These results highlight the effectiveness of state-conditioned behavioral consistency and pinpoint where agent actions agree.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.14610v2">When Do LLMs Apply the Wrong Law? Diagnosing LLM Failures in Temporal Legal Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      Legal reasoning requires identifying which version of a statute applies to a case, typically based on when the legally relevant events occurred rather than when the case is adjudicated. Yet whether large language models (LLMs) can reliably perform such temporal legal reasoning remains largely unexplored. We formalize this task as Temporal Applicable-Law Determination (TALD), and construct a TALD-Bench, a benchmark of 51,956 Chinese civil judgments spanning the enactment of the Civil Code. Evaluating nine advanced LLMs, we identify three key findings. First, models exhibit a strong bias toward newer law, achieving over 70% accuracy on new-law cases but below 25% on old-law cases. Second, models demonstrate substantial knowledge of statutory versions and temporal-effect rules, yet struggle to apply this knowledge to cases. Stronger general reasoning performance does not consistently translate into better TALD accuracy, whereas temporal-law hints yield substantial improvements, suggesting a bottleneck in how models deploy their existing capabilities. Third, temporal-law hints also increase uncertainty during statutory-version reasoning, suggesting that they may help models reconsider their default preference for newer law. Our findings highlight the importance of adapting how LLMs apply existing capabilities, rather than relying solely on improvements in legal knowledge or general reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.18891v2">Orchestrating LLM Agents for Scientific Research: A Pilot Study of Multiple Choice Question (MCQ) Generation and Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      Advances in large language models (LLMs) are rapidly transforming scientific work, yet empirical evidence on how these systems reshape research activities remains limited. We report a mixed-methods pilot evaluation of an AI-orchestrated research workflow in which a human researcher coordinated multiple LLM-based agents to perform data extraction, corpus construction, artifact generation, and artifact evaluation. Using the generation and assessment of multiple-choice questions (MCQs) as a testbed, we collected 1,071 SAT Math MCQs and employed LLM agents to extract questions from PDFs, retrieve and convert open textbooks into structured representations, align each MCQ with relevant textbook content, generate new MCQs under specified difficulty and cognitive levels, and evaluate both original and generated MCQs using a 24-criterion quality framework. Across all evaluations, average MCQ quality was high. However, criterion-level analysis and equivalence testing show that generated MCQs are not fully comparable to expert-vetted baseline questions. Strict similarity (24/24 criteria equivalent) was never achieved. Persistent gaps concentrated in skill\ depth, cognitive engagement, difficulty calibration, and metadata alignment, while surface-level qualities, such as {grammar fluency}, {clarity options}, {no duplicates}, were consistently strong. Beyond MCQ outcomes, the study documents a labor shift. The researcher's work moved from ``authoring items'' toward {specification, orchestration, verification}, and {governance}. Formalizing constraints, designing rubrics, building validation loops, recovering from tool failures, and auditing provenance constituted the primary activities. We discuss implications for the future of scientific work, including emerging ``AI research operations'' skills required for AI-empowered research pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.18616v5">Evaluating PDPL Compliance in E-Commerce Websites: Insights and Lessons Learned from Human and LLM Analyses</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      In 2024, Saudi Arabia's Personal Data Protection Law (PDPL) came into force. However, little work has been done to assess its implementation. In this paper, we analyzed 100 e-commerce websites operating in Saudi Arabia against the PDPL, examining the presence of a privacy policy and, if present, the policy's declarations of four items pertaining to personal data rights and practices: 1) personal data retention period, 2) the right to request the destruction of personal data, 3) the right to request a copy of personal data, and 4) a mechanism for filing complaints. Our results show that, despite national awareness and support efforts, a significant fraction of e-commerce websites in our dataset are not fully compliant with the examined PDPL requirements at the privacy policy level: only 31% of websites in our dataset declared all four examined items in their privacy policies. Second, the majority of top-ranked e-commerce websites in our dataset (based on search results order) and those hosted on local e-commerce hosting platforms exhibited significantly higher non-compliance rates than mid- to low-ranked websites and those not hosted on local e-commerce platforms. Third, we explored the use of Large Language Models (LLMs) as an automated tool for privacy policy analysis to measure compliance with the PDPL, using the GPT-5 model. We highlight the potential of LLMs and suggest considerations that can inform future benchmarks aimed at improving LLM-based automated analysis for privacy policies. Our results provide a step forward in understanding implementation barriers to data protection laws, especially in non-Western contexts. We provide recommendations for policymakers, regulators, website owners, and developers seeking to improve data protection practices and automate compliance monitoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2406.07243v4">MBBQ: A Dataset for Cross-Lingual Comparison of Stereotypes in Generative LLMs</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 Accepted to COLM 2024
    </div>
    <details class="paper-abstract">
      Generative large language models (LLMs) have been shown to exhibit harmful biases and stereotypes. While safety fine-tuning typically takes place in English, if at all, these models are being used by speakers of many different languages. There is existing evidence that the performance of these models is inconsistent across languages and that they discriminate based on demographic factors of the user. Motivated by this, we investigate whether the social stereotypes exhibited by LLMs differ as a function of the language used to prompt them, while controlling for cultural differences and task accuracy. To this end, we present MBBQ (Multilingual Bias Benchmark for Question-answering), a carefully curated version of the English BBQ dataset extended to Dutch, Spanish, and Turkish, which measures stereotypes commonly held across these languages. We further complement MBBQ with a parallel control dataset to measure task performance on the question-answering task independently of bias. Our results based on several open-source and proprietary LLMs confirm that some non-English languages suffer from bias more than English, even when controlling for cultural shifts. Moreover, we observe significant cross-lingual differences in bias behaviour for all except the most accurate models. With the release of MBBQ, we hope to encourage further research on bias in multilingual settings. The dataset and code are available at https://github.com/Veranep/MBBQ.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.21877v1">Aligning LLMs with Sparse Retrieval via Adaptive Sparse Supervision</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) offer strong semantic representations, their application to first-stage retrieval incurs substantial latency and storage costs. Neural sparse retrievers such as SPLADE provide efficient retrieval but rely on relatively compact encoders with limited representational capacity. To bridge this gap, we propose an LLM-based sparse retrieval framework with input-adaptive sparse alignment. Specifically, we introduce a dynamic $α$-truncation strategy that selects input-specific supervision supports based on the cumulative activation mass of SPLADE representations. This strategy guides the LLM to preserve salient lexical signals while suppressing less informative activations, without relying on a fixed support budget. Subsequent retrieval fine-tuning combines contrastive learning, knowledge distillation, and sparsity regularization to improve retrieval effectiveness. Experiments on MS MARCO and six selected BEIR datasets demonstrate improvements over SPLADE++ while maintaining competitive effectiveness against LLM-based sparse retrievers. Compared with Lion-sp, our method reduces index storage by 6.4-fold and end-to-end query latency by over 11-fold, achieving a favorable effectiveness-efficiency trade-off.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05708v2">Cost-Efficient RAG for Entity Matching with LLMs: A Blocking-based Batch Processing Framework</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 Extended version of the paper accepted at ACM SIGMOD 2027
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer a compelling approach to entity matching (EM) in label-scarce settings, yet their matching quality is sensitive to records with ambiguous or sparse attributes due to the absence of external factual grounding. Retrieval-augmented generation (RAG) directly addresses this grounding gap by retrieving and injecting relevant external knowledge. However, naively applying RAG to large-scale EM is expensive due to per-pair retrieval calls over large knowledge graphs (KGs) and the increase in input tokens caused by enriching prompts with retrieved contextual knowledge. To address the above limitation, we introduce a CE-RAG4EM (Cost-Efficient RAG for Entity Matching) framework that reduces overhead through blocking-based batch retrieval and generation. We investigate the benefits of this framework through a systematic analysis of RAG systems for entity matching, focusing on blocking-aware batch optimizations and retrieval granularity. Extensive experiments on nine widely used real-world entity matching benchmark datasets show that CE-RAG4EM under a zero-shot setting achieves comparable or improved matching quality while substantially reducing end-to-end overhead. The results demonstrate that our CE-RAG4EM closes the knowledge grounding gap of the LLM-EM baseline and yields significant improvements in F1 scores on datasets with ambiguous or numerically sparse attributes, while its batch retrieval and inference substantially reduce the overhead and maintain competitive matching quality compared to naive RAG4EM. Our analysis identifies clear performance-overhead trade-offs across blocking-based batch sizes and retrieval granularity, showing how matching quality and computational cost shift with each design choice. These findings offer concrete guidance for building scalable, efficient RAG systems for entity matching that balance accuracy and cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.30587v2">Words Speak Louder Than Code: Investigating Cognitive Heuristics in LLM-Based Code Vulnerability Detection</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 Accepted at IEEE Symposium on Security and Privacy (IEEE S&P) 2027
    </div>
    <details class="paper-abstract">
      Researchers and practitioners increasingly apply Large Language Models (LLMs) for automated vulnerability detection. Recent work has shown that LLMs are susceptible to the same cognitive heuristics that bias human judgment. Yet, no work has investigated whether these heuristics affect a model's assessment of code vulnerabilities. In this paper, we present the first systematic exploration of cognitive heuristics in LLM-driven code vulnerability detection. We introduce a controlled framework that holds the code fixed and only varies the surrounding context to trigger three cognitive heuristics: the halo effect through author attribution, the framing effect through task objectives and consequences, and the anchoring effect through prior analysis results. Within this framework, we evaluate eight LLMs across three programming languages. Our results demonstrate that all evaluated models are susceptible to these heuristics. Cross-model average susceptibility is highest for framing at 33.2%, followed by anchoring at 23.5% and halo at 18.4%. Vulnerabilities that require semantic reasoning for detection are more susceptible to cognitive heuristics than those identifiable through pattern matching. Furthermore, models often change their verdict from safe to vulnerable based on the cognitive condition, without accurately identifying the actual vulnerability. To highlight the practical impact, we demonstrate a proof-of-concept black-box cognitive attack that can suppress up to 97\% of previously detected vulnerabilities. Together, these findings show that cognitive bias is a consistent and exploitable property of LLM-based vulnerability detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.21815v1">Exact Makespan Bounds and the GPipe Homogeneity Bias for Heterogeneous LLM Inference Pipelines</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      We give an exact critical-path characterization of the makespan of micro-batch pipelines serving LLMs on heterogeneous GPU clusters, quantifying how the classic homogeneous GPipe formula errs. In the pure-forward, communication-free, infinite-buffer setting, T(M) = (M-1)tau_max + Sum_j tau_j, where tau_j is stage j's per-micro-batch service time and M the micro-batch count (Theorem 1). Under asynchronous communication -- modern NCCL-based serving -- it extends to T(M) = Sum_j tau_j + Sum_j d_j + (M-1)tau_max, with Sum_j d_j a purely additive constant (Theorem 2); under blocking communication, T(M) = (M-1)sigma_max + Sum_s sigma_s with sigma_s = tau_s + d_s, so communication enters the steady-state slope iff some d_s > tau_max - tau_s (Theorem 3). These results quantify the GPipe overestimate, decompose the bias into a homogeneity term Sum_j (tau_max - tau_j) and a communication-ignoring term Sum_j d_j, and imply that balancing per-stage compute dominates communication reduction for steady-state throughput. A real system with partial overlap is sandwiched between Theorems 2 and 3, with a closed-form bound on the overlap throughput gain. All claims are validated pointwise (error < 10^-9) on eleven configurations and a 1.6x10^5-point fuzz, including a mixed RTX 5090/3090/GTX 1070 partition. We situate the results in flow-shop scheduling and list open extensions: bandwidth-limited communication, bounded buffers, prefill-decode interleaving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.28304v3">The Necessity of Setting Temperature in LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 17 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used as automated judges, yet decoding temperature is often set by intuition rather than evidence. We run over one million large scale evaluations, multiple open-source models, prompting strategies, and judging paradigms to examine how temperature affects judge reliability. Low temperatures yield consistent outputs with few errors, especially in smaller models, while higher temperatures reduce consistency and increase errors. Yet consistency alone does not guarantee sound judgments: low temperatures can narrow the range of reasoning explored. We characterize judge behavior through four reasoning personas and find that temperature shifts their distributions in model-dependent ways, most visibly in ambiguous cases where competing answers are hard to distinguish. Repeated high-temperature sampling reveals this hidden uncertainty, and combining it with majority voting or using high temperature for reasoning and low temperature for the final decision, improves agreement on ambiguous cases without hurting clear ones, at the cost of extra inference. Our findings suggest a simple rule: keep temperature low for stable outputs, but explore more when the judgment is hard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.14923v2">TLA-Verus: Generating and Solving Temporal Proof Obligations with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      The TLA+ (temporal logic of actions) formal specification toolkit has seen widespread adoption in modeling modern distributed software systems including databases, storage systems, ledgers and coordination protocols. This popularity is despite its two significant limitations for software verification: Firstly, TLA+ offers no way to detect drift between software implementation and mathematical specification; secondly, TLC and TLAPS cannot provide satisfactory proofs of arbitrary temporal logic properties. We address the second limitation and take a first step towards the first by translating TLA+ specifications into Verus, an auto-active verifier built for Rust, via both a deterministic transpiler and an agentic translator. We demonstrate that our LLM-based pipeline is able to generate proof obligations from TLA+ specifications at scale, and to produce fully machine checkable proofs of both safety and liveness properties. Since temporal logic reasoning in Verus is a largely new task for LLMs, our work yields a valuable evaluation dataset, which we use to map out current generation model capabilities in Verus theorem proving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.21731v1">Retrieved but Not Read: Understanding Why Search-Augmented LLMs Fail to Use Available Evidence</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 14 pages, 4 figures, 16 tables. Code and data: https://github.com/ojwwalters/retrieved-but-not-read
    </div>
    <details class="paper-abstract">
      Web search allows LLMs to access up-to-date information, but it remains unclear why they sometimes produce outdated answers even when relevant evidence is available. To study this problem, we construct temporal facts whose values changed after the models' training cutoffs. We evaluate one proprietary model under deployed web search and six open-weight model configurations under controlled retrieval. We find that models often answer from search snippets without opening the source pages, which is associated with higher rates of outdated answers and unfaithful citations. Our controlled experiments and layer-wise analysis reveal a gap between the availability of evidence and its effective use. Even when retrieved evidence influences internal predictions, models may still favour outdated information in their final answers. We hope these findings will help advance research on the reliable use of retrieved evidence in search-augmented LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.21720v1">FIRM: Fisher-Informed Retention Merging for Federated LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 16 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Federated LLM unlearning addresses targeted knowledge removal when forget and retain data remain distributed across private clients and are inaccessible to the server. The resulting locally trained adapters encode conflicting objectives, making their aggregation fundamentally different from conventional federated optimization. A key failure mode is magnitude explosion, whereby unnormalized cross-client aggregation increases the merged update magnitude and degrades retained utility under stronger unlearning. We propose FIRM (Fisher-Informed Retention Merging), a single-round framework for stable and retention-aware server-side aggregation. FIRM uses client-local Fisher statistics as a unified sensitivity signal to construct a Fisher-normalized base merge and impose explicit control over retain-sensitive parameters. Its second-order retention projection admits a unique minimum-distance solution, while the corresponding dual scales with the number of retain clients rather than the model dimension. Experiments across WMDP, TOFU, and MUSE-HP using Zephyr-7B and Llama-3-8B show utility gains at comparable forgetting on WMDP and MUSE-HP, with metric-dependent trade-offs on TOFU. Controlled comparisons support benefits of the combined base merge and additional retention correction at selected operating points.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.21708v1">Human Interventions as Labels: Localizing, Steering, and (Sometimes) Editing the Computations Behind LLM Agent Failures</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      When an LLM agent misbehaves, a human steps in, and that intervention is usually discarded or replayed as training data. We treat it instead as a measurement: a label pointing at the internal computation that failed. We study it in a controlled agent environment, on one failure (asking a clarifying question when the request was already unambiguous), with labels from a rule-based overseer validated against blind human judgment. Claims cover eligible models, which both over-ask and sometimes act on matched unambiguous requests: five of thirteen qualified (Qwen2.5-3B, gemma-2-2b, OLMo-3-7B, Mistral-Nemo-12B, Phi-4). Four findings. First, the ask-vs-act decision is causally localizable in every eligible family: a direction learned from intervention labels flips 72-100% of held-out over-asks at a definite residual-stream depth, with site and dose chosen on training data, and random directions under the same procedure flip at most one episode. Second, over-asking has two mechanistic signatures: in four families the model demonstrably knows the request is unambiguous (the belief is decodable, causally inert and orthogonal to the decision) and asks anyway; in the fifth, the rare over-asks track different world-representations. Third, steering gated on a probe readout removes most of the simulated oversight burden while legitimate questions survive (bootstrap 95% intervals exclude zero); a prompt instruction makes it worse. Fourth, consolidating the fix into the weights yields a map, not a rule: on one model a LoRA confined to the causal band passes every criterion fixed in advance while behavior cloning on identical data breaks retention; on three families the reverse holds; on the fifth the outcome crosses the boundary across seeds. A geometric rule written down in advance to predict the regime failed on two out-of-family tests; we report the failure and stop rather than fit a replacement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.22967v3">Closed-Loop Molecular Inverse Design with Semantic LLM Surrogates</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 24 pages
    </div>
    <details class="paper-abstract">
      Practical molecular inverse design is rarely a one-shot generation problem; it often involves enriching a pool of molecules that match a desired property profile under a limited oracle budget. Bayesian optimization (BO) offers a natural framework for this setting, with standard Gaussian-process (GP) surrogates operating in continuous embedding spaces. We propose LoopMol, a closed-loop framework whose large language model (LLM) surrogate reasons directly over task instructions, SMILES-level optimization histories, and oracle feedback. At each iteration, the surrogate selects informative reference molecules under an exploration--exploitation principle, optionally with a concise guidance sentence. These outputs condition the next generation round and yield an inspectable optimization trace. Experiments on six drug and material tasks across three Llamole backbones show that the LLM-based variant improves over one-shot generation and is competitive with or stronger than GP-based BO. They also reveal a domain-dependent interface: reference-only conditioning performs better on binary drug tasks, while adding a concise guidance sentence generally benefits continuous material tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04990v6">From Agent Traces to Trust: A Survey of Evidence Tracing and Execution Provenance in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based agents are moving beyond text generation to planning, tool use, retrieval, memory access, environmental interaction, and multi-agent collaboration. Because their outputs depend on information and execution steps across an agent run, their execution processes are difficult to verify, debug, and audit. Even a correct final answer does not reveal which evidence supported its claims, whether tool calls were justified, how memory shaped decisions, or where execution failures originated. Relevant research addresses these questions across evidence attribution, tool-use safety, agent memory, and observability, but these areas emphasize different parts of the execution process. This survey brings them into a common framework through execution provenance, a typed graph of an agent run, and evidence tracing, its projection onto evidence-support relations, distinguishing evidence that supports a conclusion from execution that influences it. We introduce a six-dimensional taxonomy covering trace sources, evidence and execution units, provenance relations, tracing granularity and timing, representation forms, and trust functions, complemented by operational definitions of nine provenance relations and two cross-cutting attributes, provenance origin and trace assurance. Applying this framework to the surveyed systems and benchmarks reveals three gaps: limited integration of fine-grained provenance with multiple trust functions, no benchmark in our catalog that evaluates the complete provenance chain, and little reporting of tamper-evident or attested traces. These findings motivate research on unified trace schemas, semantic provenance, memory and multi-agent lineage, provenance-aware safety and recovery, realistic benchmarks, trace assurance, and privacy-aware auditing. Resources are available at https://github.com/xiaoqi-7/Agent-Tracing-Survey.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.21624v1">AgentPerturb: Evaluating LLM Agent Robustness under Controlled Interaction Perturbations</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 25 pages, 10 figures. Project repository: https://github.com/xuyj233/AgentPerturb
    </div>
    <details class="paper-abstract">
      Real-world tasks are often complex and noisy, posing substantial challenges for LLM agents. Yet existing evaluations lack a systematic framework for jointly characterizing potential perturbation targets and types. We use this two-dimensional view to conduct a unified study that quantifies the robustness of current frontier models, traces the mechanisms behind perturbation-induced failures, and draws implications for robustness-oriented training. Through controlled evaluations on APEX-Agents and Toolathlon, we find that perturbations consistently degrade performance, while the dominant vulnerabilities vary across task environments. Trajectory-level analysis reveals that agents rarely recognize altered inputs and often propagate corrupted information without verification. A focused cost analysis further shows that successful perturbed trajectories can require substantially more computation. These findings motivate training objectives that cover diverse forms of noise and reward verification, uncertainty awareness, and recovery. We provide a rigorous benchmark-construction pipeline. Code and data will be released at https://github.com/xuyj233/AgentPerturb.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.21604v1">Evaluating Inquiry Learning in LLM-Based Agents: Hypothesizing, Reasoning, and Exploration</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 EMNLP 2026 (Findings)
    </div>
    <details class="paper-abstract">
      Scientific inquiry involves iterative cycles of exploration, hypothesis generation, and reasoning over observations to uncover relationships in complex systems. Large language models (LLMs) have demonstrated strong capabilities in several components of this process, including inductive reasoning, hypothesis generation, and data analysis. However, prior work typically evaluates these abilities in isolation, leaving open how models integrate them across a full inquiry cycle. In this work, we introduce a framework for evaluating LLMs' inquiry capabilities in a unified, interactive setting. We extend the Countdown task to a mathematical environment where an agent probes the system by manipulating variables, collecting observations, and inferring relationships between them. This setup enables analysis of exploration strategies, causal discovery, and hypothesis refinement during inquiry. Across seven LLMs and multiple prompting strategies, our experiments reveal limited inherent inquiry capabilities and low success rates in discovering the underlying expressions. However, structured guidance inspired by educational scaffolding significantly improves performance. Our findings highlight current limitations of LLMs in strategic exploration and iterative hypothesis refinement, and show how simple guidance strategies developed for human learners in inquiry learning can support more effective inquiry behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.21588v1">Hear, Localize, and Reason: Spatially Aware Scene Understanding for Audio-Visual LLMs</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 NeurIPS 2026; Project page: https://hlrllm.github.io/
    </div>
    <details class="paper-abstract">
      Audio-visual large language models (AV-LLMs) have made strong progress in multimodal understanding, yet most treat audio as monaural semantic content, limiting their ability to reason about where sounds originate and how they relate to the visual scene. Recent spatial audio-visual studies address parts of this problem, but often focus on specific abilities, such as spatial correspondence or direction and distance reasoning. In this work, we introduce HLR-AVSceneQA, a comprehensive benchmark with a hear-localize-reason hierarchy that evaluates whether models can recognize what is heard, localize sound sources in egocentric and allocentric views, and use these localized sources to reason about how they relate to the visual scene. We further introduce HLR-LLM, which augments a strong audio-visual foundation model with a binaural spatial audio branch alongside its monaural semantic audio pathway. We train HLR-LLM with a curriculum that incorporates final-stage chain-of-thought supervision for multi-step spatial reasoning. Experiments show that existing AV-LLMs remain limited in spatial grounding and relational reasoning, while HLR-LLM substantially improves spatial audio-visual scene understanding without compromising semantic audio-visual perception.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.09549v3">SecureClaw: Clawing Back Control of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 Accepted at NeurIPS 2026 (poster); 29 pages including appendices. This version corrects the conference footer only; scientific content is unchanged
    </div>
    <details class="paper-abstract">
      Tool-using large language model (LLM) agents face two distinct security failures: unauthorized external actions and exposure of sensitive plaintext inside the runtime before any final output check can intervene. Existing defenses usually protect one boundary, either the planner/runtime or the point where external actions are executed, and therefore do not by themselves secure both surfaces. We present SecureClaw, a dual-boundary architecture that enforces authorization in a trusted executor and keeps raw protected data outside the LLM runtime. Sensitive reads pass through a trusted gateway that replaces raw values with reference IDs. In the evaluated deployment, the gateway also provides short planning summaries that deliberately release selected information, a process called declassification. Writes that change external state follow a PREVIEW$\rightarrow$COMMIT protocol in which only a trusted executor may execute the exact request authorized by policy. The runtime can still plan over summaries and reference IDs, but cannot directly resolve the IDs into secrets or perform side effects. Across the AgentDojo, AgentLeak, and Agent Security Bench (ASB) benchmarks, SecureClaw is the only defense we evaluate in a common harness that simultaneously retains usable task utility and achieves 0% attack success rate (ASR) on ASB, 0.64% ASR on AgentDojo, and 3.23% overall leak on AgentLeak's attacked parity lane, which measures final-output and internal-relay leakage. Code is available at https://github.com/ponyyuhan/SecureClaw.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.21570v1">Independent Data-Integrity Verification for LLM-Driven Functional Coverage Closure in Hardware Design</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      LLM agents can write directed tests that close functional coverage, but coverage only shows that a state was reached, not that the data on the way there was correct. This is a long-known point in verification and we do not claim it as new. We describe a small coverage-closure agent that applies it to an LLM loop: coverage is read from the report that the simulation itself exports (cocotb-coverage YAML) and never taken from the model's account, and data integrity is checked by scoreboards that are separate from the model. On three architecturally distinct RTL designs (an asynchronous FIFO, a UART transmitter and an AHB2APB bridge) the agent raised coverage from 91.7%, 88.9% and 67.4% to 100% within its 8-iteration budget in every logged run, using one design-agnostic agent. A reactive scoreboard monitor attached to the FIFO's base testbench also found a clock-domain-crossing bug in the author's own stimulus code. The monitor is not yet attached to the LLM-generated tests, so their data integrity is not independently checked; one manual check on a single FIFO test is reported as preliminary. We list the weaknesses of this evidence: three designs, no baseline, no ablation, and logged runs that come from several agent checkpoints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.21565v1">BiShield-TEE: On the (In-)Security of Unilateral Weight Obfuscation in On-Device TEE-Shielded LLM Partition</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 Accepted at NeurIPS 2026
    </div>
    <details class="paper-abstract">
      On-device deployment of large language models (LLMs) is increasingly adopted due to latency and privacy requirements, but exposes proprietary models to reverse engineering and intellectual property theft. Trusted Execution Environments (TEEs) provide hardware-enforced isolation for on-device inference, yet most TEE-based parameter obfuscation relies on unilateral matrix transformations that obfuscate only output side. We identify a vulnerability in this design, termed the \emph{Solution Determinacy Property (SDP)}: the obfuscation secret is uniquely determined, exposing the structure of the true key and enabling attackers to break the protection. Under our attacks, representative methods \textsc{GroupCover} and \textsc{ArrowCloak} are reduced to negligible protection, with attack accuracy reaching over $2.4\times$ the full-shielding black-box baseline and recovering near-complete model functionality.To address this vulnerability, we propose \textsc{BiShield-TEE}, a bilateral framework that induces \emph{Ambiguous Solutions} through joint obfuscation of input and output dimensions, hiding the true key among many equally valid candidates. Experiments across diverse models and datasets show that \textsc{BiShield-TEE} confines attack accuracy to the full-shielding black-box baseline, while achieving higher inference efficiency than prior methods on real hardware.Our code is available at https://github.com/laduiw/BiShield-TEE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04924v2">Can Crowdsourcing Survive the LLM Era? A Community Survey on Human Data Collection</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      The widespread use of Large Language Models (LLMs) as writing tools challenges the validity of crowdsourced data, as crowdworkers may outsource tasks to models. To better understand how this is addressed, we surveyed 155 researchers in NLP and related disciplines about their experiences and opinions on collecting free-text responses via crowdsourcing. This paper provides an overview of practitioners' challenges, mitigation strategies, and the foreseen implications on data quality. 44% of respondents reported observing LLM usage in their crowdsourced data. While 93% of them had anticipated this, half were unsure what precautions to take. The most prevalent detection strategies are distinctive textual style patterns and unusually fast completion times. Overall, survey responses show that the research community is aware of the problem and taking measures, but existing efforts remain insufficient to fully address it. Finally, we derive a set of considerations to guide future crowdsourced free-text data collection in the era of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.20827v3">DischargeBench: Pre-Deployment Evaluation of LLM Discharge Educators through Persona-Grounded Patient Simulation</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      Effective discharge education is personalized: a clinician adapts the discharge plan to each patient's literacy, recall, and personality. Before large language models (LLMs) take on this role with real patients, they need to be evaluated in conversations with diverse patients, yet existing evaluations target static, artifact-generation, or question-answering tasks. We introduce **DischargeBench**, a pre-deployment benchmark in which a candidate LLM educator teaches a persona-grounded simulated patient in an open-ended, multi-turn conversation. We curate **MIMIC-IV-Ext-DischargeBench**, 320 cases across 16 ICD chapters, each paired with a patient persona that varies personality, education level, health literacy, and recall. Each conversation is scored on Conversation Quality, Topic Checklist Score (TCS), Simulated Comprehension Score (SCS), and a Claim-Level Safety Audit (CLSA) by an LLM judge validated against physician annotations. Across eight closed- and open-source LLMs, the best model scored only 0.70 on TCS and 0.72 on SCS, and distrustful and anxious patients lowered comprehension for every model. DischargeBench is intended as a screening step before human evaluation, not as evidence of readiness to educate patients.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.21525v1">CriticaLLM: Using LLMs as Autonomous Game Master for Tabletop Role-Playing Games</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive capabilities across many tasks, including processing large amounts of text, following instructions, and generating ideas. This allows the use of such models in various contexts, including games and Tabletop Role-Playing Games (TTRPGs), such as Dungeons & Dragons. However, these games include many complex components, from crafting interesting stories to adhering to rules, while allowing some freedom in players' actions. In this work, we first highlight key components that LLMs should exhibit in the context of TTRPGs. Then, we introduce our framework, CriticaLLM, a fully automated Game Master that can run entire TTRPG games, from creating diverse and interesting stories to managing exploration and combat phases. We evaluate our framework with actual TTRPG players and examine the diversity of generated plots and antagonists across four LLMs of varying sizes. Overall, our framework appears to be a promising option for running entire TTRPG games.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19705v2">RouteWeaver: Weaving Mode Selection and Execution into Unified LLM Routing</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 Submitted to ICLR 2027
    </div>
    <details class="paper-abstract">
      LLM routing dynamically selects and coordinates heterogeneous language models across diverse tasks. However, existing approaches are largely built around a predefined execution paradigm, such as single-round, multi-round, or agentic routing, which fixes how model calls are organized and limits adaptation to different query needs. We propose RouteWeaver, a unified routing framework that integrates these routing modes within a single LLM policy, jointly learning which mode to use and how to select models within that mode. To jointly optimize mode selection and within-mode execution, we develop COMET-GRPO (COordinated Mode-Execution Training), which decouples the trajectory-level advantage into separate signals for mode selection and within-mode execution, enabling more targeted credit assignment to each decision level. We further introduce progressive exploration, which gradually shifts training from externally assigned modes to autonomous selection, so that the router first learns how to execute each mode before deciding when to use it. We evaluate RouteWeaver across mathematics, coding, knowledge, logical reasoning, and factual recall. RouteWeaver achieves an average accuracy of 61.4%, an 11.03% relative improvement over the strongest baseline, and outperforms all compared baselines in each of the five domains. Ablations show that both COMET-GRPO and progressive exploration are important for adaptive routing; without progressive exploration, the routing policy collapses entirely to single-round execution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.21508v1">Decomposing Brain-LLM Alignment with Interpretable Semantic Representations</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) predict neural responses during naturalistic language comprehension, but the linguistic information underlying this brain alignment remains difficult to interpret. We use SCRIBE, an interpretable contextual semantic representation spanning 65 neurocognitively grounded semantic properties, as a lens for decomposing the brain alignment of one LLM, LLaMA2-7B. Using variance partitioning between LLaMA2-7B and SCRIBE on naturalistic fMRI data from seven participants listening to narrative stories (Moth Radio Hour), we separate brain-predictive variance shared between the two representations from variance unique to each. We find that shared variance is the dominant component across the whole brain and language-related regions, while both representations also retain significant unique contributions. We then examine whether the concreteness of individual semantic properties relates to shared and SCRIBE-unique brain-predictive variance. Concrete properties show greater shared variance with LLaMA2-7B than abstract properties across the whole brain and most examined regions, whereas abstract properties do not show greater SCRIBE-unique variance. Finally, property ablations reveal that semantic contributions extend across multiple dimensions, with cumulative effects on both shared and SCRIBE-unique variance. Together, these results show that, for LLaMA2-7B on this dataset, a substantial portion of brain-LLM alignment is associated with interpretable semantic information, while revealing complementary brain-predictive semantic information beyond the LLaMA2-7B representation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.21502v1">ISW-Agent: Integrating LLM-Driven Dialogue with n8n Workflow for Internal Solitary Wave Remote Sensing Image Processing</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      Internal solitary wave (ISW) remote sensing requires task parameters and spatial information to be transferred across image acquisition, detection, stripe extraction, and propagation prediction. Here, we present ISW-Agent, which integrates large language model (LLM) interpretation with n8n workflows and independent scientific models through structured task constraints, geospatial transformations, and stage-specific output validation. In tests of 120 Chinese instructions per backend, GPT-5.6 Sol and DeepSeek both achieved a strict task-match rate of 94.2%, with all intents and execution decisions matching the reference. Sentinel-1 cases demonstrated continuous processing from dialogue requests to wavefront prediction for the specified scene. Of 50 timed scenes, 39 completed detection, extraction, and prediction with a median turnaround time of 48.04 s; 11 stopped after empty extraction. Across 3,408 controlled fault-injection runs, retries recovered 234 transient-failure conditions and output validation rejected 234 structural defects, while 61 provenance or completeness errors indicated the need for content-consistency checks. Overall, ISW-Agent thus connects natural-language requests to traceable spatial processing chains for multimodel ISW analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.02255v2">On the Privacy of LLMs: An Ablation Study</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) face diverse privacy threats, yet existing studies typically evaluate individual attacks in isolation, offering limited guidance on which system factors drive privacy vulnerabilities. This paper presents a unified threat model and a systematic ablation study of four representative attacks: Membership Inference (MIA), Attribute Inference (AIA), Data Extraction (DEA), and Backdoor Attacks (BA). Our findings reveal attack-specific sensitivities: membership inference depends strongly on dataset characteristics and attack strategy, attribute inference is influenced by model and training characteristics, data extraction increases with model scale and training-data repetition, and backdoor attacks achieve consistently high success rates. Based on these findings, we derive practical recommendations for privacy-aware LLM deployment, including prioritizing membership leakage assessments in retrieval-augmented systems, minimizing repeated personal information in training data, monitoring attribute inference risks, and safeguarding fine-tuning pipelines against backdoors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18915v2">Reasoning Error from Known Fact: Step-Level Self-Consistency Group Relative Policy Optimization for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      With the rapid advancement of large language models (LLMs), modern systems not only possess strong foundational capabilities and extensive knowledge, but can also solve complex problems via long, multi-step reasoning. However, as reasoning traces become longer, LLMs may produce a substantial amount of hallucinated content during the reasoning process, which is often difficult to detect. In this work, we conduct a fine-grained analysis of hallucinations arising in LLM reasoning and find that the reasoning traces are particularly prone to Context-Sensitive Factual Hallucinations: cases where the model actually has the relevant knowledge, yet makes factual errors due to contextual interference during reasoning. To address this issue, we propose Step-level Self-Consistency Group Relative Policy Optimization (SSC-GRPO), which assigns step-level rewards to reasoning traces by computing self-consistency scores of individual steps across multiple rollouts. Compared with prior methods, SSC-GRPO achieves state-of-the-art performance on both mathematical reasoning benchmarks and hallucination leaderboards. Our results offer a new perspective for detecting and mitigating hallucinations in the reasoning process of large language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.21466v1">Ontology Refinement with Embeddings and LLMs: A Case Study on CASTON</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      Ontologies are widely used to represent and organize complex knowledge in domains such as biomedicine and healthcare. However, constructing reliable ontologies remains challenging, as class hierarchies, whether manually or automatically constructed, may contain subtle subsumption errors that require careful refinement. We study this problem in CASTON, a casting-domain ontology, and present a lightweight but scalable refinement workflow that combines a fine-tuned ontology embedding model with Large Language Models (LLMs). The embedding model is used to extract a limited number of candidate axioms for refinement together with possible replacement candidates, and an LLM then returns repair suggestions. In a small expert evaluation, the default configuration using a CASTON-fine-tuned OnT embedding model and Qwen3-8B produced reliable high-confidence repairs: all six high-confidence replacements were accepted by both domain experts. The implementation is available at https://github.com/HuiYang1997/OntRefine.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.21412v1">JevAgentHarness: Benchmarking, Deploying and Training Decision Models That Share Control with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 Code: https://github.com/JevAgentHarness/JevAgentHarness
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents now fix software, operate computers and serve customers, but their harnesses, the programs that run them, call a large LLM at every step, which makes long tasks slow and costly. Most of these calls end in a multiple-choice decision, such as which tool to call next. Decision models such as Jev answer these questions directly, at a fraction of an LLM call's time and price. We ask how a harness can delegate these decisions to a decision model with little loss of success. To evaluate decision models in agentic tasks, we build AgentDecisionBench, which tests them from isolated questions up to control of a running agent, on public benchmarks and our DecisionGym environment. The decision model most accurate on isolated questions solved only 69.5% of agent tasks, and without a fitted confidence threshold, delegating tool choices lowered success by 19.0 points. Building on these findings, we then design JevAgentHarness, which fits this threshold for each choice on the agent's own trajectories and lets the LLM decide below it. Because existing decision models never see the states an agent reaches after its own mistakes, we train AgentDecision-4B, an open 4B model, on such states with Jev's answers as labels. On tasks unlike its training data, the agent solved 91.5% of them with AgentDecision-4B and 82.0% with Jev. On tasks like their training data, 4B models trained this way made agents 81.1-89.0% cheaper and up to 2.5 times faster, without losing 3 or more points of task success. The models, data, and harness will be released at https://github.com/JevAgentHarness/JevAgentHarness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17688v2">LLMs Infer Cultural Context but Fail to Apply It When Responding</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 Accepted to EMNLP 2026 (Main Conference); 10 pages, 7 figures, 2 tables (26 pages, 13 figures, 8 tables including references and appendices)
    </div>
    <details class="paper-abstract">
      Recent work has shown that LLMs overrepresent dominant cultures, particularly Western ones, while marginalizing others. We investigate whether this affects models' ability to generate culturally adapted responses by evaluating their use of local measurement units based on the user's perceived cultural background. We introduce Cultural and Pragmatic Response Inference (CAPRI), a dataset of conversations with varying levels of cultural cues. Experiments with state-of-the-art LLMs show that models can infer cultural background and recall relevant conventions, but often fail to utilize the information to adapt their answers to the relevant cultural conventions, unless explicitly prompted to perform the tasks sequentially. We further evaluate adaptation to the interpretation of time and quantity expressions, two subjective language grounding dimensions that are affected by culture. We find that models increasingly adapt their answers as cultural cues accumulate, but their priors are not culture-neutral, sometimes aligning with the model's country of origin. Overall, CAPRI provides a resource for future research aimed at narrowing the gap between cultural knowledge and culturally adaptive language generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.21405v1">MemCut: Every Secret Leaves a Trail in the Long-Term Memory of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 17 pages, 4 figures, 13 tables, including appendices
    </div>
    <details class="paper-abstract">
      Long-term memory has become a crucial component of large language model (LLM) agents, with most research focusing on retrieving relevant information accurately and efficiently. However, a dependable memory system must also forget sensitive user data upon request. To evaluate this capability, we introduce MemCut, a systematic benchmark for user-requested forgetting. Its 456 hand-designed secrets in long-horizon conversations have fully annotated evidence trails to test recovery after deletion, including from messages that never state the secrets directly. We propose two recovery metrics. Native recovery measures reader success through the system's retrieval, whereas residual recovery measures the same reader's success when directly given surviving objects traced to annotated disclosures or evidence. We establish three reference deletion baselines to test how far existing retrievers and write models support forgetting across five memory designs and four LLMs. Disclosure removal reduces native recovery far more than residual recovery. LLM-guided deletion of ten additional turns from twenty retrieval candidates leaves 39-76% residual recovery. Subset sweeps (144 facts, three systems, two models) show that widening the pool at a fixed budget reduces residual recovery more than deleting all original candidates. These sweeps identify candidate coverage as a bottleneck in the reference baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03328v4">Rethinking Calibration for High-Sparsity LLM Pruning: From Single Sources to Capability Coverage</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      Post-training LLM pruning relies on a small calibration set, usually chosen heuristically and judged by WikiText perplexity or by mean accuracy on general-domain commonsense benchmarks. This single-axis view hides capability-specific effects: at 60\% sparsity, strategies separated by $2.85$ points in averaged commonsense accuracy diverge by $51.9$ points in code retention, and across 15 sources no single one preserves all capabilities. We treat calibration as a capability-coverage problem. Balanced, a uniform four-way mixture under a fixed budget, tests the principle on real corpora: it outperforms every evaluated single source, and its margin over C4 grows from $+4.6$ points at 30\% sparsity to $+18.8$ at 60\%. The advantage holds across model scale and a second model family, and transfers to held-out benchmarks disjoint from calibration construction. When aligned corpora are unavailable, IGSP approximates that mixture from the base model and the evaluation taxonomy alone, outperforming prior self-generation methods by $7.23$--$12.79$ points on the LLaMA main suite. On Qwen2.5-7B, IGSP exceeds Balanced on both the main suite and held-out benchmarks. Capability coverage is a first-order design variable for high-sparsity pruning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.21393v1">Self-Guided LLM Reasoning for Error Diagnosis in Vehicle Routing Problems</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      Vehicle Routing Problems (VRPs) are fundamental to many real-world applications. Recent advances in large language models (LLMs) have spurred progress in solving these NP-hard problems. However, most existing approaches assume that input instances are error-free, neglecting the presence of faults commonly encountered in real-world data. In this work, we systematically investigate the capability of LLMs to detect (and correct) errors in VRP instances, and identify key limitations, particularly in handling complex instances or errors that substantially increase the LLM's reasoning burden. To overcome these challenges, we propose a self-guided reasoning framework in which LLMs autonomously generate code that provides structured insights to aid in error diagnosis. Experiments across diverse problem settings and LLMs demonstrate that our framework significantly improves error detection performance, achieving near-perfect accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.21375v1">Memorilla: Latent Semantic Memory for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 Website: https://memory.stanford.edu/
    </div>
    <details class="paper-abstract">
      Memory enables large language models (LLMs) to use information beyond their parameters, such as past user conversations, long documents, or an agent's earlier actions. For a frozen LLM, access to this accumulated knowledge is limited to what a memory system places in its context window. However, gathering and combining relevant information needed for each query while keeping the model's context short remains challenging for existing systems. Here we introduce Memorilla, a memory system that gathers information from an entire document collection into a fixed number of query-dependent 'memory tokens' for a frozen LLM. Memorilla encodes documents (such as user chats, book chapters) once into reusable embeddings, then learns to retrieve and place query-specific information in the LLM's input embedding space. Using only 16 memory tokens, Memorilla achieves the best performance against baselines including Mem0 and MemoRAG on five of eight benchmarks spanning personalization, long-document and knowledge-intensive question answering. Notably, on FactKG, it reaches 84.33% accuracy versus 80.43% with full-context inference, while using 108x fewer tokens to represent the source documents. Furthermore, Memorilla can facilitate continual learning in agentic systems where the document count (the agent's past turns) increases throughout the trajectory. In a TextWorld environment, we show that Memorilla reduces the agent's tendency to wander between rooms it has already explored, and improves its average success rate from 18.6% with full-history in-context learning to 24.2%. Overall, Memorilla aids frozen LLMs in accumulating and using knowledge effectively across long-context interactions and takes a step towards building continually learning AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.01379v3">Multi-LLM Orchestration for High-Quality Code Generation: Exploiting Complementary Model Strengths</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are central to automated code generation, yet most approaches use one model at every stage, from writing code to repairing and optimizing it. LLMs have complementary strengths: their failures do not fully overlap across programming languages, algorithmic categories, and development stages. For example, in Python, the model with the highest code-generation accuracy ranks last at code repair. Using one model to generate code, another to repair it, and a third to refine it can therefore achieve higher correctness and better execution performance than a single model throughout. We propose PerfOrch, a multi-LLM multi-agent system with four agents for categorization, generation, debugging, and refinement. The generation, debugging, and refinement agents select models through a Memory module that ranks candidate LLMs by their measured performance for each stage, programming language, and algorithmic category. We evaluate PerfOrch with five candidate LLMs on 4,640 problems from four benchmarks covering five programming languages (HumanEval-X, EffiBench-X, BigCodeBench, and APPS) against single-model direct generation, single-model generate-debug-refine pipelines (PerfCodeGen), and the state-of-the-art multi-agent systems MapCoder and CodeSIM. PerfOrch achieves the highest pass@1 in all twelve benchmark-language settings, exceeding the strongest baseline by an average of 8.19 percentage points, and reduces execution time for an average of 75.03% of solved problems with an average speedup of 1.18x. The Memory rankings remain effective on three unseen benchmarks without re-profiling. Sequential refinement matches exhaustive execution-time refinement coverage with 66% fewer refinement-stage tokens (57% for average-memory optimization). The complementary strengths persist after all five LLMs are replaced with their current-generation successors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.21333v1">QuasiLeak-Bench: Evaluating LLM-Based Privacy Detectors on Policy-Conditioned Quasi-Evidence Group Recovery</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 37 pages, 15 figures, 25 tables
    </div>
    <details class="paper-abstract">
      Natural language messages often contain quasi-identifiers that are weak in isolation but can jointly reveal an individual's private attributes. We call such a jointly revealing set of cues a quasi-evidence group. Existing privacy benchmarks focus on explicit identifiers, attribute inference, or task-driven disclosures, leaving inference from quasi-evidence groups largely unmeasured. In practice, whether exposing such a quasi-evidence group violates privacy also depends on sender-specified disclosure policies. We introduce QuasiLeak-Bench, a benchmark for policy-conditioned quasi-evidence group recovery, which evaluates whether detectors recover complete quasi-evidence groups and flag the disclosures the given policy prohibits. To obtain reliable ground truth at scale, a reverse-construction pipeline first selects weak quasi-identifiers, composes them into quasi-evidence groups verified to jointly reveal a target attribute, and generates documents around this evidence, followed by automated quality control. Experiments show that existing privacy detectors transfer poorly to QuasiLeak-Bench, and locally deployable LLM-based detectors fall far behind proprietary models, mainly because they miss most quasi-evidence groups altogether. However, a 4B model fine-tuned on our training split surpasses the strongest prompted model on held-out documents, showing that the task is learnable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.21309v1">Combining LLM and Dynamic Analysis for Universal Testing of Static Application Security Testing Tools</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      Static application security testing (SAST) tools help detect potential software vulnerabilities, but they themselves contain faults undermining reliability and usability. Automated testing can address these issues, yet current techniques suffer from limited program transformation diversity and lack mechanisms to verify generated variants, leading to reduced fault detection and high manual effort. We present LLM4SAST, the first universal framework detecting faults in SAST tools using large language models and dynamic feedback. Our key insight is leveraging LLMs to understand and transform code in diverse programming languages. We design novel metamorphic relations with transformations to guide variant generation, using an LLM to produce multi-language variants that satisfy these relations. To reduce erroneous variants and identify unique faults, we propose two heuristics: dynamic-feedback-driven selection and program-differencing-based clustering. For each metamorphic relation, a report-centric test oracle detects variants that fail tests and identifies underlying faults by differentially analyzing SAST reports of the seed program and its variants. LLM4SAST identified 66 unique faults across six SAST tools, with 32 confirmed or fixed after being reported to developers, demonstrating effectiveness. The proposed heuristics eliminate 47.8% of false positives, and LLM4SAST detects 3.4 times more faults while requiring only 39.8% of the variants compared to existing approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.17715v2">How Do LLM-Generated Texts Impact Term-Based Retrieval Models?</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      As more content generated by large language models (LLMs) floods into the Internet, information retrieval (IR) systems now face the challenge of distinguishing and handling a blend of human-authored and machine-generated texts. Recent studies suggest that neural retrievers may exhibit a preferential inclination toward LLM-generated content, while classic term-based retrievers like BM25 tend to favor human-written documents. This paper investigates the influence of LLM-generated content on term-based retrieval models, which are valued for their efficiency and robust generalization across domains. Our linguistic analysis reveals that LLM-generated texts exhibit smoother high-frequency and steeper low-frequency Zipf slopes, higher term specificity, and greater document-level diversity. These traits are aligned with LLMs being trained to optimize reader experience through diverse and precise expressions. Our study further explores whether term-based retrieval models demonstrate source bias, concluding that these models prioritize documents whose term distributions closely correspond to those of the queries, rather than displaying an inherent source bias. This work provides a foundation for understanding and addressing potential biases in term-based IR systems managing mixed-source content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.03867v2">Where to Steer: Input-Dependent Layer Selection for Steering Improves LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 EMNLP 2026 (Main Conference) Camera Ready
    </div>
    <details class="paper-abstract">
      Steering vectors have emerged as a lightweight and effective approach for aligning large language models (LLMs) at inference time, enabling modulation of model behaviors by shifting intermediate representations toward a target behavior. Many existing steering methods apply interventions at globally fixed locations in an LLM, effectively treating the optimal intervention location as input-invariant. However, representations relevant to a target behavior can be encoded at different LLM layers depending on the input. Theoretically, we show that different inputs may require steering at different layers to achieve alignment with a given behavior. Empirically, we observe that the most effective layer for steering varies substantially between inputs. Motivated by these observations, we introduce \textit{Where to Steer} (W2S), a general framework that adaptively selects intervention layers conditioned on the inputs by learning a mapping from input representations to steering layers. Across multiple LLMs and alignment behaviors, W2S consistently outperforms fixed-layer baselines in both in-distribution and out-of-distribution settings. Our findings suggest that input-dependent layer selection is a key design dimension complementary to methods for steering vector extraction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.21257v1">Fractional Power-of-Two Quantization for Efficient and Effective Multiplier-Free LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 37 pages, 19 figures, 22 tables. Accepted at NeurIPS 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demand efficient inference, where the multiply-accumulate (MAC) operations in linear projections dominate compute and energy cost. Post-training quantization (PTQ) reduces this cost by mapping weights and activations to low-bit integers. Power-of-Two (PoT) weight quantization promises efficient multiplier-free LLM inference via shift-add logic, but suffers significant accuracy degradation at low bitwidths due to its coarse base-2 quantization grid. We introduce Fractional PoT (FPoT), a base-$\sqrt{2}$ (half-octave) fractional PoT grid that achieves practical 4-bit weight quantization while preserving a fully integer shift-add datapath. The method combines (i) the base-$\sqrt{2}$ weight grid with uniform integer activations, (ii) a calibration-time grid-alignment refinement absorbed into the static per-channel scale, and (iii) a lightweight 1-adder $\sqrt{2}$ approximation whose error is absorbed during weight calibration via approximate-grid quantization, yielding a processing element (PE) with significantly fewer gates than a 4-bit integer multiplier. On Llama-3-8B with 4-bit weight and activation quantization, FPoT improves zero-shot commonsense reasoning performance by 8.72% relative to the PoT baseline, while maintaining performance close to the uniform integer baseline, with only a negligible 0.4%p degradation. These results hold across various configurations spanning seven models and three bit-widths. Furthermore, FPoT can be integrated with state-of-the-art LLM PTQ methods, demonstrating framework-agnostic applicability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.21254v1">LLM-Based Weak Annotation for Cybersecurity Tweet Classification: A Teacher-Student Framework</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      Cyber threat intelligence (CTI) relies on the timely identification of emerging threats, enabling organizations and security analysts to support situational awareness and decision-making. Social media platforms such as X (formerly Twitter) have become valuable CTI sources because vulnerabilities and attack activities are often discussed there before appearing in formal reports. However, the rapid growth of cybersecurity discussions on social media has created more content than domain experts can realistically review and annotate, making labeled data a major bottleneck for developing supervised learning approaches. While large language models (LLMs) have shown promise as weak annotators in other domains, their effectiveness for cybersecurity tweet classification and downstream training of domain-specific pre-trained language models (PLMs) remains underexplored. To address this challenge, we propose a Teacher-Student weak supervision framework in which a selected LLM serves as a Teacher to generate weak labels for unlabeled cybersecurity tweets, enabling the training of cybersecurity-specific PLMs without large-scale manual annotation. Experimental results show that PLMs trained solely on LLM-generated labels consistently achieve F1 scores above 0.92, remaining within 3 percentage points of fully human-supervised models across all evaluated architectures. These findings suggest that LLM-based weak annotation offers a practical alternative for constructing cybersecurity-specific training data when expert annotation resources are limited.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.21249v1">Characterizing Predictability-Latency Trade-offs of KV-Cache SSD Offloading in LMCache for LLM Serving Systems</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 6 pages, 3 figures, 3 tables
    </div>
    <details class="paper-abstract">
      KV-cache offload is widely used to stretch GPU memory for LLM serving, but its storage behavior has not been characterized at the block-device level. In this paper, we study LMCache through realworld multi-session workloads that span same/different context x same/different prompt, using over 100 stateless requests per workload. Our comparison is observational rather than factorial: it contrasts two realizable deployment snapshots -- LMCache 0.3.0 with buffered I/O and the kernel page cache, and LMCache 0.3.16 with prefix-aware deduplication and O_DIRECT. Key findings define the paper. First, the legacy stack appears almost read-free at the SSD layer in all four workloads, but a flush_ram experiment shows that this is conditional on page-cache warmth: once the cache is evicted, the same path issues about 1 GB of sequential reads in 683 ms. Second, deduplication delivers a dramatic write reduction only for the exact-repeat workload: one set collapses from 1.62 GB to 11.75 MB of writes, while the other three sets remain in the 2-5 GB range. Third, O_DIRECT converts a hidden and bimodal read cost into an explicit and stable one: on one set, warm TTFT rises from 27 ms to 93 ms because each of the 99 warm requests re-reads the same ~12 MB partial chunk from SSD. The main conclusion is therefore not "newer is better," but a system trade-off: predictability versus average-case latency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.15766v2">BioXArena: Benchmarking LLM Agents on Multi-Modal Biomedical Machine Learning Tasks</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 NeurIPS 2026 E&D Track
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents increasingly automate machine-learning model building, yet biomedical benchmarks emphasize question answering, reasoning, and tool use or cover narrow slices of biomedical ML coding. We introduce BioXArena, a biomedical machine learning (BioML) coding benchmark for task-specific model building on heterogeneous, often multi-modal datasets. It contains 76 end-to-end tasks across 9 domains: sequence, single-cell, structure, network biology, chemical biology, perturbation dynamics, phenotype--disease, imaging, and text-integrated tasks. Each task packages primary-source data with hidden labels, held-out graders, and biology-aware metrics on a unified 0-to-100 scale; agents write runnable code, train models, and predict private test samples. BioXArena emphasizes realistic data interfaces: most tasks require agents to integrate multiple input sources, including 46 tasks that combine distinct modality families. We compare 18 harness--backbone configurations: general-purpose coding, biomedical, and ML agents plus our Minimal runner, all under the same controlled resource configuration. Codex with GPT-6 Astra achieves the highest task-level average score of 73.91 among 18 configurations, closely followed by Claude Code with Opus 5.5 (73.79). Matched-backbone comparisons show higher observed scores for mature coding systems than for Minimal; seven Minimal backends characterize model differences under a fixed runner. Beyond the main leaderboard, ablations, robustness checks, and scaling, runtime, and failure-mode analyses characterize how backbones, scaffolds, budgets, and domains affect BioML coding performance, exposing modeling and execution gaps across agent and LLM backbone families.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.21246v1">FGSS-BO: Feedback-Guided Hypothesis Management for LLM-Based Bayesian Optimization</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 21 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-guided Bayesian optimization (BO) uses past observations to guide search, but proposed directions and their revision can remain implicit. We propose Feedback-Guided Search State (FGSS), a structured hypothesis-management interface connecting language-based guidance to observed evidence, executable parameter interventions, and comparative support updates. An LLM reflector proposes and revises hypotheses; a deterministic curator validates them and maintains identities and test records. The evolving state guides candidate construction and priorities. Instantiated on LLAMBO, FGSS-BO retains its surrogate and default expected improvement selection, with a bounded testing policy. On two retrospective protocol/media tasks with nearest-neighbor response oracles, it attains higher sample mean final best and search AUC than the matched LLAMBO baseline and evaluated non-LLM baselines. Prose-reflection and EI-only controls show mixed component effects. Six direct-training hyperparameter-optimization tasks show task-dependent transfer, strongest on diabetes SVM; exact-grid and reaction controls examine other settings. FGSS exposes hypotheses, their execution, and cross-round revisions for inspection while supporting search gains in selected settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.21214v1">Rubric Judgments Lack Uncertainty: Conformal Weighting for LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      Rubric-based evaluation is widely used to assess large language model (LLM) outputs by decomposing quality into multiple criteria and assigning each a binary judgment from an LLM verifier. However, judgment certainty varies substantially across criteria. Existing methods uniformly aggregate these judgments into a final score, implicitly treating every criterion as equally certain and overlooking this heterogeneity. As a result, uncertain judgments remain incorporated into reward estimation without distinction, potentially yielding misleading training signals for downstream reinforcement learning (RL). To address this limitation, we propose Uncertainty-aware Rubric Weighting (URW), a training-free framework that quantifies criterion-level judgment uncertainty via conformal calibration. Specifically, URW elicits a soft uncertainty signal from a fixed pool of paraphrased judgment templates and converts it into a per-criterion weight using split-conformal $p$-values with provable coverage guarantees. Extensive experiments demonstrate that URW consistently improves pairwise preference accuracy and delivers stronger downstream RL alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20159v2">Benchmarking Code LLM-as-a-Judge via Constrained Perturbation and Calibration</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      Code from LLM-assisted development may contain defects like API hallucinations, motivating the development of code evaluation metrics to assess such code before deployment. As conventional rule-based metrics merely score code based on similarity to reference programs, and execution-based metrics require expensive manual preparation for tests, researchers have developed LLM-as-a-judge metrics for both behavioral and nonbehavioral code evaluation, and curated multiple benchmarks to validate their effectiveness. However, such benchmarks suffer from critical limitations: Some feature coarse-grained binary labels of passing/failing unit tests, unable to distinguish between fundamental and minor flaws; others utilize fine-grained criteria with manually annotated scores as ground-truth, but such criteria are often vaguely defined, introducing subjectivity and unreliability in human scores. Furthermore, they often directly prompt LLMs to generate code for evaluation, which may lead to unbalanced code quality distributions, failing to test the metrics' discriminative capability across varying quality levels. To curate a diverse benchmark with a balanced program quality distribution and streamline manual annotation, we propose AXIOM, a novel perturbation-based framework. It reframes program scores as the refinement effort needed for deployment, with two stages: (1) Quality-constrained semantic perturbation, prompting LLMs to apply sequences of predefined perturbation rules to existing programs, precisely controlling each program's target quality to achieve a balanced distribution. (2) Hybrid quality calibration, selecting semantically distinct programs for benchmark diversity, and performing streamlined manual annotation with multisource context. With AXIOM, we curate AXIOMBENCH, a large-scale multilingual benchmark with 1,962 code evaluation problems in a more balanced score distribution...
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03056v3">SkillDAG: Self-Evolving Typed Skill Graphs for LLM Skill Selection at Scale</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 In *The 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP, Findings)*, 2026
    </div>
    <details class="paper-abstract">
      Large skill libraries are now standard equipment for LLM agents, and which subset reaches the prompt decides what the agent can do. Enumeration and top-$K$ embedding retrieval both treat skills as independent documents, assuming relevance is a property of one skill at a time. It is not: skills depend on, conflict with, specialize, and duplicate one another, so the closest match to a task can name it while omitting a prerequisite it needs. Graph retrieval encodes these relations but spends them in a fixed pipeline, handing the agent an opaque bundle. We present SkillDAG, which exposes a typed skill graph to the LLM as an agent-callable interface rather than consuming it for the agent: search returns vector matches, typed-edge neighbors, and conflict signals, and a propose-then-commit protocol lets it register execution-backed edges as it works. Across two benchmarks and three model families, SkillDAG outperforms full-library prompting, vector retrieval, and Graph-of-Skills, gaining +14.3 points of ALFWorld success and tying only at ceiling; its ranking holds flat across a $10\times$ pool expansion that degrades the fixed pipeline. Inter-skill structure, these results suggest, is worth more to an agent that can query and revise it than to a retriever that consumes it.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22166v3">Adapting the Interface, Not the Model: Runtime Harness Adaptation for Deterministic LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      LLM agents are shaped not only by their language models, but also by the runtime harness that mediates observation, tool use, action execution, feedback interpretation, and trajectory control. While existing agent adaptation methods mainly update model parameters, many failures in deterministic, rule-governed domains stem from mismatches at the model--environment interface. We propose Life-Harness, a lifecycle-aware runtime harness that improves frozen LLM agents without changing model weights or evaluation environments. Life-Harness evolves from training trajectories by converting recurring interaction failures into reusable interventions across environment contracts, procedural skills, action realization, and trajectory regulation, and remains fixed for evaluation on unseen tasks. We evaluate Life-Harness across eight environments from AgentBench, $τ$-bench, $τ^2$-bench, and AppWorld using 18 open-weight backbones, and further test three frontier models on seven environments. With Qwen3-4B-Instruct, Life-Harness exceeds Meta-Harness and AHE by 28.7 and 7.5 percentage points in mean accuracy across the seven shared environments, respectively, while using 8.5\% and 68.4\% fewer optimization tokens. The evolved harnesses also benefit GPT 5.5, Opus 4.8, and Gemini 3.1 Pro, yielding an average accuracy gain of 3.5 percentage points. Code is available at https://github.com/Tianshi-Xu/Life-Harness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.12242v4">ZenFlow: Enabling Stall-Free Offloading for LLM Training</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 13 pages, 16 figures
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) often exceeds GPU memory limits, prompting systems to offload model states to CPU memory. However, existing offloaded training frameworks like ZeRO-Offload treat all parameters equally and update the full model on the CPU, causing severe GPU stalls, where fast, expensive GPUs sit idle, waiting for slow CPU updates and limited-bandwidth PCIe transfers. We present ZenFlow, a new offloaded training framework that decouples updates between GPU and CPU by prioritizing important gradients. ZenFlow performs in-place updates of important gradients on GPU while offloading and accumulating less important ones on CPU without blocking training iterations, fully overlapping CPU work with GPU computation. To make this importance-aware design scalable, ZenFlow leverages a novel spatio-temporal locality of important gradients, concentrated in specific positions that remain stable across iterations, to make important gradient identification efficient and lightweight while preserving fidelity. Evaluation shows that ZenFlow achieves up to 5x end-to-end speedup, 2x lower PCIe traffic, 85% less GPU stalls, and up to 72% lower cloud GPU cost, all while preserving accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.21139v1">When is a better LLM worth paying for? Stochastic Optimization with Switching Mixed-Fidelity Oracles</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used upstream of decision-making to clean data, detect label errors, and reconstruct missing values. Yet LLM-assisted data curation creates a new operational tradeoff: higher-fidelity models and richer prompts can be substantially more expensive, while the value of this additional fidelity depends on the downstream optimization accuracy. This raises the central question of this work: when should one pay for a better LLM, given a specific downstream task at hand? We study this question through stochastic optimization with mixed-fidelity first-order oracles, which can have systematic bias and state-dependent variance. We develop two cost-aware algorithmic frameworks that allow adaptive switching between stochastic oracles. With known bias, the first meta-algorithm applies a restarted, cost-aware policy that adaptively selects oracle fidelity across optimization stages. Its cumulative cost is no larger than that of the best fixed oracle in hindsight and can be substantially smaller. With unknown bias, we develop a budgeted certification scheme that uses occasional trusted-gradient evaluations to certify the optimization progress and eliminate unsuccessful oracle sources, and show an explicit tradeoff between certification cost and delayed detection of biased sources. We provide concrete realizations of these meta-algorithms by applying biased stochastic gradient extrapolation (SGE) and recursive regularization under state-dependent noise as subroutines. Synthetic and LLM-completion experiments illustrate the qualitative switching behavior predicted by the theory: the adaptive policy skips several expensive LLM oracles and substantially reduces cumulative cost relative to fixed high-fidelity alternatives while achieving comparable accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18253v2">Mind the Delay: Latency-aware Query Routing for Heterogeneous LLM pools</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      Large language model (LLM) query routers typically assign each query to a suitable model that balances response accuracy and monetary cost. Subsequently, separate system-level load balancers based on policies such as join-the-shortest-queue or round-robin distribute the queries across replicas of each model to reduce latency. However, decoupling accuracy-cost and latency considerations can lead to suboptimal routing decisions. We introduce the Batch Evolution Aware (BEA) estimator to provide a fine-grained estimate of the latency, that is, the time-to-first-token (TTFT) experienced by an arriving query. Unlike existing load balancers that rely on coarse signals such as queue lengths, per-model processing rates, or historical average latencies, BEA replays the serving framework's token-batch formation using a snapshot of an instance's current workload, stopping when the incoming query's first token is generated. Taking into account the compute and memory requirements of the prefill and decode phases of language queries as well as the contention between multiple queries that are served in the same batch, BEA predicts TTFT within an error of approximately 5%. Each prediction takes about 0.1 ms, which is fast enough to run on the critical routing path. We also design an analytical framework for a model instance under a limited processor-sharing scheme that calculates expected TTFT in closed form and captures the qualitative trend of latencies as the load increases. The TTFT estimates produced by BEA are then used in our latency-aware router, BEAR, which simultaneously balances accuracy, cost, and latency to select the best model instance from a heterogeneous pool of LLMs. Against seven baselines, BEAR improves the utility of responses delivered within their targets by 63% on average over existing latency-aware routers, and by 30.5% over the best baseline at each arrival rate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.21135v1">Stronger Judges Do Not See More: Deployment-Integrity Blindness in LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      LLM-agent evaluation reports behavioral scores but leaves the identity of the deployed runtime stack unreported. A tampered deployment that preserves evaluated outputs passes every output-only evaluator. We introduce DIB-Bench, a benchmark that measures this deployment-integrity blindness across 7,560 scored examples, six tamper types, and an evaluator ladder from regex scorers through LLM judges to GPT-5-nano. Stronger evaluators accept more tampered outputs: mean acceptance rises from 88.6% (rule-based) to 94.1% (GPT-5-nano), and a trigger-conditional tamper invisible on the evaluation suite produces 48-100% harmful outputs, rising with model capability. A PAC-style bound shows that tampers with zero output shift are undetectable from outputs at any sample size; a manifest-hash identity channel detects them with zero samples. Behavioral fingerprinting identifies the model component above chance (76.7%) but covers one of six stack components. We also propose a graded deployment-identity descriptor for evaluation reports.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.02244v2">Task-Level Natural Language Priors as Learning Signals for Low-Resource LLM Training</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often struggle when low-resource training data are ambiguous or incomplete. Task-level natural-language priors can provide useful guidance in such settings, but existing approaches usually treat these priors as input context rather than as learning signals during training. We study how task-level natural-language priors can supply guidance missing from these data through Prior-Guided Tuning (PGT). Its implementation, Contrastive Prior Steering (CPS), retains the supervised objective and adds positive- and negative-prior-conditioned losses with opposing optimization directions. Across AmbiMath, Diagnostic Anchoring, and Jigsaw, CPS achieves near-perfect exact-match accuracy on the synthetic task, improves accuracy under misleading diagnostic suggestions by 8.2 percentage points, and improves average subgroup Macro F1 on Jigsaw by 10.7 percentage points over plain fine-tuning, averaged over the two evaluated models. On Jigsaw, with one tenth of the experimental training data, CPS slightly exceeds full-data plain fine-tuning. These results show how reusable task-level language can complement limited labeled examples without requiring instance-level rationales or preference pairs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.22519v4">LLMON: An LLM-native Markup Language to Leverage Structure and Semantics at the LLM Interface</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 28 pages
    </div>
    <details class="paper-abstract">
      Textual Large Language Models (LLMs) provide a simple and familiar interface: a string of text is used for both input and output. However, the information conveyed to an LLM often has a richer structure and semantics, which is not conveyed in a string. For example, most prompts contain both instructions ("Summarize this paper into a paragraph") and data (the paper to summarize), but these are usually not distinguished when passed to the model. This can lead to model confusion and security risks, such as prompt injection attacks. This work addresses this shortcoming by introducing an LLM-native mark-up language, LLMON (LLM Object Notation, pronounced "Lemon"), that enables the structure and semantic metadata of the text to be communicated in a natural way to an LLM. This information can then be used during model training, model prompting, and inference implementation, leading to improvements in model accuracy, safety, and security. This is analogous to how programming language types can be used for many purposes, such as static checking, code generation, dynamic checking, and IDE highlighting. We discuss the general design requirements of an LLM-native markup language, introduce the LLMON markup language and show how it meets these design requirements, describe how the information contained in a LLMON artifact can benefit model training and inference implementation, and provide some preliminary empirical evidence of its value for both of these use cases. We also discuss broader issues and research opportunities that are enabled with an LLM-native approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.21113v1">SSR: Structural Signal Retriever for Scalable Agent Discovery in LLM-Orchestrated Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      Large-scale multi-agent ecosystems require mechanisms for discovering suitable agents without exposing entire registries to a large language model (LLM) orchestrator. The challenge is not only to retrieve relevant Agent Cards from a growing registry, but also to maintain a compact candidate context, since each additional candidate increases prompt size, inference cost, latency, and the risk of exceeding the model context window. We formulate this problem as candidate-context scalability and propose the Structural Signal Retriever (SSR), our deterministic and training-free retrieval method designed for Agent Card discovery. SSR derives structural lexical signals from capability-bearing Agent Card fields, weights them using inverse document frequency (IDF), and uses an Inverted Signal Index (ISI) for selective candidate scoring. We evaluate SSR using a synthetic agent benchmark, a public MS MARCO adaptation, and end-to-end LLM orchestration. At 1,000 agents, SSR reduces LLM input by 98.8% and orchestration latency by 70.9% relative to full-catalog orchestration. On MS MARCO, IDF improves retrieval effectiveness within the structural representation, while the ISI preserves retrieval effectiveness and reduces explicitly scored passages by 96.8%. End-to-end experiments show that candidate pre-selection maintains a compact candidate context as registry size grows and avoids the operational failures observed with full-catalog orchestration. These results support SSR as an Agent-Card-native, traceable retrieval mechanism for candidate-context scalability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.21108v1">Planning Commitment of LLM under Partial Observability</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      LLM-based planning agents commit to plans of different lengths before interacting with a partially observable environment. Some aggressively commit to a long plan based on assumptions about unobserved states, while others conservatively plan only over what they have observed. Both behaviors emerge implicitly in pre-trained models, whether the LLM outputs a plan directly or generates a PDDL formulation for a classical solver. Aggressive commitment risks costly failures, while conservative commitment risks excessive LLM calls. This choice pits safety against efficiency, a trade-off that has been neither characterized systematically nor made controllable. We investigate aggressive and conservative LLM planners and formalizers on two partially observable domains, where priors are provided as either explicit statistics or common-sense knowledge. Across three LLMs, both strategies reach similar accuracy. Conservative methods achieve fewer failed attempts (safety) at the cost of more API calls (efficiency). Aggressive methods fail more often but replan efficiently only when priors are informative. Interestingly, analysis shows that LLMs treat the provided priors nearly as facts, which risks failure when priors are unreliable. We also explore hybrid strategies with tunable hyperparameters that adjust the balance between safety and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16557v3">BenchPreS: A Benchmark for Context-Aware Personalized Preference Selectivity of Persistent-Memory LLMs</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 Accepted at EMNLP 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly store user preferences in persistent memory to support personalization across interactions. However, in third-party communication settings governed by social and institutional norms, some user preferences may be inappropriate to apply. We introduce BenchPreS, which evaluates whether memory-based user preferences are appropriately applied or suppressed across communication contexts. Using two complementary metrics, Misapplication Rate (MR) and Appropriate Application Rate (AAR), we find even frontier LLMs struggle to apply preferences in a context-sensitive manner. Models with stronger preference adherence exhibit higher rates of over-application, and neither reasoning capability nor prompt-based defenses fully resolve this issue. These results suggest current LLMs treat personalized preferences as globally enforceable rules rather than as context-dependent normative signals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.21091v1">MemComposer: Learning to Assemble Memories for LLM Agents via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 30 pages, 8 figures, 14 tables
    </div>
    <details class="paper-abstract">
      Agent memory enables large language model (LLM) agents to reuse past experiences without updating their parameters. However, semantic similarity alone does not guarantee memory usefulness: downstream performance depends on how memories are selected and presented to the agent. We propose MemComposer, a memory composer that jointly examines the current query and retrieved candidates to decide which experiences to include, how many to select, and how to order them. We train the composer using Experience Policy Optimization (ExPO), a GRPO-based framework that evaluates assembled memory contexts through complete agent interactions and uses task-success feedback to optimize the assembly policy. Across three $τ^2$-bench domains with two frozen executors, MemComposer outperforms the strongest baseline for each executor while reducing average memory-token usage by 16.9% and 4.9%, respectively. The learned policy also transfers to unseen domains, executors, and memory banks, and benefits from memories generated by stronger models without further training. These results demonstrate the value of learning query-specific memory assembly using downstream task feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05458v3">HarnessMDP: Learning to Control LLM Agent Harnesses with Offline Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 13 pages, 4 figures
    </div>
    <details class="paper-abstract">
      The execution harness is a critical component of LLM agents, shaping how the agent interacts with its environment and organizes multi-step execution. Prior work has shown that improving tools, workflows, and other harness components can substantially improve agent performance. This suggests that the harness can be treated as a learnable control layer that adapts execution decisions from experience, rather than as a predetermined execution configuration. In this work, we introduce HarnessMDP, a learning-based framework that makes harness control itself learnable. HarnessMDP formulates agent execution as a finite-horizon Markov decision process, where the harness learns to select structural execution actions from the evolving agent state. The harness policy is learned offline from collected trajectories using task performance and deployed online to adaptively control new executions. Extensive experiments on coding tasks, $τ$-bench, AgentBench, and other agent evaluation settings show that HarnessMDP consistently improves execution behavior and final task performance over baseline harnesses. These results demonstrate the effectiveness of HarnessMDP and establish learnable harness control as a practical mechanism for improving LLM agents through better execution. Our code is available at https://github.com/Hik289/Agentic-RL-harness.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.04528v3">Measuring Harness-Induced Belief Divergence in Multi-Step LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      Software agents are becoming increasingly important in modern coding workflows, where they solve complex tasks through multi-step interaction with code repositories. A central component of these systems is the execution harness, which governs how the underlying LLM observes, acts on, and verifies its environment. We show that the harness can in turn shape the agent's evolving beliefs about the environment, even when the task, environment, and base LLM remain fixed. To reveal this effect, we introduce a belief-rollout diagnostic that traces how an agent's anticipated world state and future decisions evolve under alternative harnesses. We further separate immediate divergence introduced by the interface from divergence that develops over longer horizons. Across controlled coding tasks include HIBench-Code, SWE-bench Verified and Terminal-Bench, we find substantial cross-harness belief divergence that is often invisible from terminal success alone and can propagate into different downstream actions. Building on this observation, we introduce Belief-Invariant World-Modeling (BIWM), which exposes evidence hidden or compressed by harness execution and aligns the resulting belief trajectories across different harnesses. Our findings reveal that harness design can systematically shape how software agents understand and act on their environments, providing practical guidance for building and evaluating more reliable coding agents. Our code is available at https://github.com/Hik289/Harness-induce-bias.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.21060v1">Bellman-Consistent Trajectory RL for LLM Arithmetic Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 Grade A as a project report
    </div>
    <details class="paper-abstract">
      To address sparse terminal rewards in LLM arithmetic reasoning, we propose a Bellman-consistent trajectory RL method. Our framework introduces Trajectory Bellman Residual Minimization (TBRM) to transform sparse rewards into dense token-level signals. We integrate a Reinforce Leave-One-Out (RLOO) group strategy to reduce gradient variance of Bellman residuals and use hindsight goal relabeling to accelerate intermediate learning. Evaluated on the Countdown task, our method alleviates optimization bottlenecks and stabilizes gradient variance significantly better than GRPO. The TBRM-RLOO framework boosts Pass@1 to 0.55--0.64 and achieves a state-of-the-art Pass@16 accuracy of 76.0--84.0\%, outperforming the SFT, IPO/DPO, and RLOO baselines. Our approach demonstrates exceptional single-sample efficiency and successfully circumvents standard performance plateaus via test-time scaling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.21038v1">INCONVENIENT: A Benchmark to Assess LLM Generalization Capabilities from Convenience Samples</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      Machine learning models for tabular prediction typically assume that training data are representative of the population on which they will be deployed. In practice, however, data are often collected based on convenience, availability, or cost, resulting in systematic sampling biases that can distort feature--target relationships and undermine real-world generalization. Existing tabular distribution-shift benchmarks primarily construct out-of-distribution (OOD) settings by partitioning observations across naturally occurring domains. While such partitions may induce changes in P(Y | X), the resulting shifts are neither explicitly controlled nor isolated, and the evaluation distributions themselves are often not representative of the broader target population. We introduce INCONVENIENT, a benchmark designed to study robustness to convenience-sample bias. Rather than relying on naturally occurring domain partitions, INCONVENIENT directly perturbs feature--target relationships in the training distribution while preserving the original population distribution as the deployment environment. This setting more closely reflects real-world scenarios in which models are developed from unrepresentative samples and subsequently deployed on a broader population. The distinction is particularly relevant for large language models (LLMs), whose pretraining may encode population-level knowledge that extends beyond the relationships observed in a biased training dataset. Unlike existing distribution-shift benchmarks, INCONVENIENT provides explicit control over both the direction and severity of relationship shifts, enabling a systematic assessment of robustness to convenience sampling. The benchmark comprises 200 tabular classification and regression datasets spanning diverse application domains and generates around 15,500 shifted training distributions which lead to substantial downstream degradation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.22305v2">Learning at the Right Pace: Adaptive Data Scheduling Improves LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 15 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) achieve remarkable reasoning capabilities through reinforcement learning (RL) post-training. However, existing RL post-training commonly relies on uniform data sampling, which ignores the semantic structure of the training data and the evolving capability of the training policy. To address these limitations, we propose Adaptive Data Scheduling (ADS), a dual-level data scheduling framework that combines adaptive sampling across semantic clusters with policy-boundary sample selection. At the cluster level, ADS organizes samples according to semantic patterns and adapts the inter-cluster sampling distribution to prioritize clusters within the policy's current reach. At the sample level, ADS tracks prompts near the policy's capability boundary within each cluster to obtain informative relative advantages. Experimental results across three LLMs and seven reasoning benchmarks demonstrate that ADS improves average accuracy by 5.5% over Group Relative Policy Optimization (GRPO). Notably, ADS consistently improves RL methods with different objective designs, highlighting its potential as a general data scheduling strategy. The source code is available at: https://github.com/Richard-zrx/ADS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.15591v2">Agent Gym: A Framework for Continuous Evaluation and Evolution of LLM Agents Through Human-in-the-Loop Feedback</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 15 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents deployed in production environments face a fundamental tension: the agent's behavior is frozen at deployment time, while the business rules and edge cases it must handle continue to evolve. Existing approaches address agent construction and one-time evaluation but provide no structured mechanism for continuous post-deployment behavioral correction without modifying the agent's source code. Most of the approaches offered in the market, require intense collection of logs and traces, and re-examining the agent design by the engineering team, a process which is heavy, long and negates the economical value of agentic transformation. We introduce Agent Gym, a modular, domain-agnostic framework that wraps any existing LLM-based agent in a continuous evaluation-and-evolution loop. The framework provides six composable capabilities --- Act, Evaluate, Investigate, Correct, Learn, and Observe --- organized across three architectural zones: a constitution layer that codifies domain knowledge in configuration artifacts, a runtime inference pipeline that chains acting, investigation, and adaptive correction, and a learning loop that enables subject matter experts to discover and validate new correction rules through natural language interaction. The key technical contributions include a hybrid deterministic-LLM correction engine with 21 condition operators and three-tier actions, a three-layer investigation architecture for ground-truth-free compliance validation, and a programmatic safety loop that guarantees rule correctness before human approval. We further introduce the Spec-to-Note Gap, an autoencoder-inspired view of agentic system transparency. An open-source reference implementation for invoice processing demonstrates that the framework is fully operational and ready for adoption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.21016v1">CausalEvo: Causal Skill Self-Evolution for Health LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 GenAI for Health Workshop @ NeurIPS 2026, Sydney
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents for health can accumulate reusable textual skills from task feedback, such as calculation criteria, measurement time points, and formula units. Deciding which skills to retain is consequential: useful rules improve later answers, while faulty generalizations propagate errors. We propose CausalEvo, a method for causal skill self-evolution that estimates the relative effects of deliberately changed skill sets while holding the model weights, feedback cases, and rule count fixed. At each evolution step, a frozen LLM proposes candidate rules, which are pooled with the incumbent skills so that the current skill set remains a feasible choice. Paired skill replacements against shared background rules are scored on the same feedback cases; least squares converts the score differences into relative skill values, and the top-ranked rules are retained, favoring incumbents on ties. A decision-loss bound separates candidate support, approximation, and estimation errors. On six MedCalc-Bench-Verified tasks with 120 held-out cases, CausalEvo achieves 60.00% accuracy, compared with 55.83% for SkillOpt and 51.67% without skills. Across the recorded training configurations, which use different optimization budgets, it consumes 55.6% fewer training tokens than SkillOpt. Gains concentrate in particular calculations alongside regressions on others, and paired accuracy intervals include zero. CausalEvo turns task feedback into controlled decisions about skill retention and replacement. Project repository: https://github.com/txbrook/CausalEvo
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.12286v2">T-GADE: Thermodynamical Generative-AI-Driven Evolution of LLM Artifacts</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 v2: 12 pages, 4 figures, 6 tables. Presents T-GADE as a general diversity-aware framework for LLM-based evolutionary computation with an EoH-aligned implementation example, reports a preregistered seed-paired comparison in which the host-equipped configuration reaches low excess with fewer total LLM calls than EoH, and treats the selection temperature as an explicit diversity control
    </div>
    <details class="paper-abstract">
      Introducing large language models (LLMs) into evolutionary computation enables semantic generation of candidates, but the classical problem of maintaining population diversity remains. Inspired by the thermodynamical genetic algorithm (TDGA), we propose T-GADE (Thermodynamical Generative-AI-Driven Evolution of LLM Artifacts), an evolutionary computation framework extended through LLMs. T-GADE keeps the notions of a population and genetic operators, takes structured artifacts generated by an LLM as individuals, and implements crossover and mutation as LLM operators. It further introduces the HoxLM, an LLM that attempts self-repair of an individual after evaluation and before survival selection; two update schemes, generational and steady-state; and Fermi-type and Bose-type occupancy rules that define solution identity and whether duplicates are permitted. Survival selection is unified as minimization of the free energy $F_T=\langle E\rangle-TH$, and the selection temperature $T$ explicitly controls the balance between quality and diversity. We tested an implementation example aligned with Evolution of Heuristics (EoH) on the online bin-packing task, where the excess is the bin count in excess of the volume lower bound, relative to that bound. The HoxLM-equipped configuration reached lower excess with fewer total LLM calls than EoH. In the preregistered comparison at 400 total calls it won 9 of 10 seed-paired runs, with a median sealed excess of 0.836% against 3.912%, and it was also better in the mean and in the worst case. Temperature controls the spread of the retained population under both update schemes, and a small positive temperature improves training quality over the collapsed $T=0$ point in the generational Bose-type scheme. The evolved heuristics are shown as readable scoring rules.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.25061v3">DataKernelBench: Can LLMs Optimize Database Queries on GPUs?</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 Accepted at EMNLP 2026
    </div>
    <details class="paper-abstract">
      GPUs increasingly accelerate database systems, but query-specific peak performance still often relies on hand-written kernels. Existing LLM kernel benchmarks focus on machine learning operators, leaving irregular, heterogeneous, data-movement-heavy database-style operators untested. We introduce DataKernelBench, which translates SQL into validated PyTorch TorchPlan programs and evaluates LLMs that optimize either the core tensor-bounded snippet or the full query in CUDA or Triton through execution-guided repair. Across ten proprietary and open-weight models on TPC-H SF10 with an H100 GPU, the strongest full-query CUDA configuration achieves $2.11\times$ speedup over the TorchPlan baseline at full pass rate. We find that higher-performing implementations commonly use kernel fusion and execution-strategy changes, stronger models benefit most from full-query specialization, and workload context matters more than hardware context. To handle data larger than GPU memory, we extend TorchPlan with Dask-cuDF for on-demand partition loading on TPC-H SF100 with four H100 GPUs, achieving $2.54\times$ speedup. Project page: https://kerneldf.github.io/datakernelbench
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.18786v2">Where Do LLM Ensemble Gains Come From? Decomposing Candidate Value, Diversity, and Aggregation</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 14 pages, 6 figures
    </div>
    <details class="paper-abstract">
      LLM ensemble gains are often measured against a single baseline, but the value attributed to additional candidates can change with the comparison used. We study an LLM aggregator under four matched conditions: Direct (no candidates), Single-slot (one candidate in a multi-slot interface), Duplicate (one candidate repeated), and Multi-candidate (separately generated candidates). Under either non-diverse control, overall candidate gain (CIV) splits algebraically into an additional-candidate contrast (DSV) and a candidate-context contrast (CCV); the split, but not CIV, depends on the chosen control. On MultiHopRAG model diversity (k=2), the repetition gap Rep=-8.33 pp [-13.25, -3.75] quantifies the difference between gains measured against Duplicate and Single-slot. On a matched FRAMES stronger-panel extension (n=200, k=3, aggregator output limit 4096), paired Delta Rep=+9.67 pp [+6.00, +13.50]; within panels, Rep changes sign (original -6.33; stronger +3.33). In FRAMES evidence aggregation, candidate-context contrasts remain strongly negative under both non-diverse controls. Our results show why ensemble evaluations should state the reference control before attributing gains to additional candidates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.20902v1">Thinking in Graphs: Neo4j-Backed Exploit-Chain Reasoning for LLM Penetration-Testing Agents</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 16 pages, 7 figures, 9 tables, 18 code listings
    </div>
    <details class="paper-abstract">
      LLM coding agents such as Claude Code and OpenAI Codex can drive penetration-testing tools, but they reason about an engagement as a flat list of findings in a context window. Attackers think in graphs: isolated low-severity exposures turn catastrophic once chained through credential capture, privilege escalation, and lateral movement. This gap feeds a common belief that AI agents cannot truly chain exploitation; we argue this describes the flat context window, not the model. We present ChainGraph, a pattern that externalizes exploit-chain reasoning into a Neo4j property graph wired to an off-the-shelf agent via the Model Context Protocol, no bespoke harness. Findings, attack steps, credentials, assets, and tamper-evident evidence receipts are typed nodes with pre/post-condition edges; JSON/Markdown holds documentary memory while Neo4j holds reasoning memory. We formalize five operator questions (next-step inference, entry-to-crown-jewel pathing, combined-severity promotion, evidence provenance, and chokepoint identification) as reusable Cypher and Graph Data Science queries, plus a chain-severity score distinct from per-finding local severity. On a synthetic multi-forest Active Directory engagement modeled on GOAD, in live Neo4j with GDS, three findings rated Medium in isolation become Critical once their path to the forest root is materialized, and betweenness centrality names the best chain break-points. Scoping each decision to a graph result rather than re-reading the engagement cut the reasoning payload by about 98% on a token proxy. We give a reproducible setup for Claude Code and Codex. Results are illustrative single-run measurements, not a benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.14438v2">Neither True nor False: Do LLMs Represent Truth-Value Gaps in Plural Predication?</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being used to summarize and verify claims in critical sources that inform consequential decisions. Such claims are full of plural definites. If three of five patients improved after taking drug A, the claim the patients improved after drug A is not true, since two did not, yet its negation, the patients didn't improve after drug A, is not true either, since three did. In formal semantics, this truth-value gap is known as the Homogeneity Gap: both the plural definite and its negation are neither true nor false when the property holds for only a fraction of the plural referent. A model that cannot represent this gap turns mixed evidence into a clean verdict, an error that can be costly in high-stakes use. We introduce HomogeneityNLI, a diagnostic dataset of 1,000 minimal pair items crossing determiner, polarity, and group composition across ten logical conditions. Evaluating seven open-weight models, we uncover a pervasive Bivalence Bias: models assign plural definites their homogeneous truth conditions. In the zero-shot setting, no model recognizes the gap for both polarities in more than 15% of scenarios, and negated definites are judged False in 84% of cases. Explicit linguistic rules, chain-of-thought, and enumeration prompts produce unstable, model-specific effects that often entrench the bias. Our findings indicate that current open-weight LLMs acquire the truth conditions of plural predication but not its trivalent logic, and cannot yet be trusted to summarize or verify claims whose evidence is mixed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.20832v1">Amortizing Sequential Experimental Design in Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Despite their extensive pretraining knowledge and strong reasoning capabilities, large language models (LLMs) struggle to adaptively gather information in experimental design tasks. We study how to amortize sequential Bayesian experimental design (BED) into an LLM by explicitly training the reasoning that guides action selection using information-gain rewards. Our approach specifies the experimental generative model independently of the LLM's own beliefs, allowing us to optimize expected information gain under a model of our choosing. We investigate the design choices for amortization, including alternative generative-model constructions, reward estimation, hypothesis-space specification, multi-step objectives, and computational costs relative to training- and inference-time BED baselines. On Twenty Questions, the resulting policies improve information acquisition and task performance, outperform inference- and training-time BED baselines at substantially lower computational cost, and generalize to unseen categories. We further demonstrate positive transfer from Twenty Questions training to Medi-Q, a real-world clinical dialogue benchmark. These results suggest that information gathering can be learned as a generalizable reasoning skill in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.03376v2">Towards Better Aligned LLM Judges for Radiology Report Evaluation across Modalities and Anatomies</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Current work on radiology report evaluation has focused primarily on LLM-based metrics for chest X-rays (CXR), leaving open whether such approaches remain robust across other modalities and anatomies, and which model and prompt configurations are best suited to serve as LLM judges. We present a comprehensive evaluation of LLM judges for radiology report evaluation across diverse model families, prompting strategies, and adaptation methods, leveraging two expert-annotated datasets that span multiple imaging modalities and anatomical regions. Specifically, we compare four existing metrics (RadFact, GREEN, FineRadScore, and CRIMSON) with our proposed metric, VERT, across open- and closed-source models of varying sizes and reasoning capabilities. Among existing metrics, GREEN proves the most robust, outperforming more recent CXR-specific approaches. VERT further improves Kendall's correlation with radiologist judgment by up to 26% over GREEN, while ensembling and lightweight fine-tuning of Qwen3-30B yield gains of up to 9.9% and 25%, respectively, of which the latter also has a latency reduction of 37.2x. A systematic error analysis further shows that, while these metrics correlate with expert ratings overall, they tend to miscount errors such as omissions. Overall, our results show that carefully selected general-purpose LLM judges provide strong radiology report evaluation across modalities, with lightweight adaptation offering complementary gains when annotated data is available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.20753v1">Optimal Hydra: Model Heterogeneous Federated Learning for LLMs using Data-Informed Hydra-Nets</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 currently under submission
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) trained on vast amounts of data have demonstrated impressive performance over a range of tasks, but future performance gains will require increasing amounts of data and compute. Federated Learning (FL) facilitates training on decentralized private data, but client resource limitations can make training impractical, particularly as model sizes continue to expand. These resource constraints make the assumption of a shared server-client architecture impractical. Model heterogeneous FL offers a promising alternative by adapting model size to client capabilities; however, previous work is limited to the pre-training setting, and conclusions on sub-network selection do not necessarily hold for fine-tuning. Furthermore, prior approaches create client sub-networks without considering their unique data distributions, thereby overlooking the opportunity to tailor sub-network architecture to local data. Motivated by evidence that attention heads exhibit task dependency within layer defined feature spaces, we propose Optimal Hydra, a transformer based heterogeneous FL framework for fine-tuning that combines global layer selection with client specific attention head selection, aligned with local data distributions. Optimal Hydra makes explicit use of client data to produce client-specific sub-networks we call Hydra-nets. To our knowledge, this is the first model heterogeneous FL approach tailored to LLM fine-tuning. Experiments on three datasets show significant improvements over baseline methods while also reducing local resource demands.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22613v2">Evolutionary Multi-Task Optimization for LLM-Guided Program Discovery</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Recent LLM-guided evolutionary search methods have shown that iterative program mutation can discover strong algorithms, but they typically optimize each task independently, even when related tasks share reusable structure. We study evolutionary multitask optimization (EMO) for LLM-guided program discovery and propose EMO-STA (Shared-Then-Adapt), a two-stage framework that first evolves a shared archive of executable programs across a task family and then adapts selected shared candidates to each target task. Within EMO-STA, we explore multiple adaptation strategies, including warm-starting from the shared archive, adapting the best average shared program, and adapting the shared program that performs best on each target task. Across eight task families spanning continuous optimization, geometric construction, modeling, and algorithmic optimization, EMO-STA improves over single-task evolution under matched evolutionary-iteration budgets in most settings, with STA Best-Local providing the strongest in-distribution adaptation and STA Best-Shared yielding robust transfer to unseen tasks. Iteration-allocation experiments show that devoting a substantial fraction of the family-level iteration budget to shared evolution is effective, with roughly balanced shared and aggregate adaptation budgets often yielding the best performance. ARC and time-series feature-engineering case studies further suggest that shared evolution can mitigate overfitting in low-evidence settings by favoring programs that perform well across related tasks rather than exploiting brittle task-specific artifacts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.15455v2">Surfacing Neural Activations Over Multi-Turn Conversation Improves User Calibration to LLM Behavioral Drift</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Current human-AI interaction (HAI) only surfaces model outputs, hiding how LLM behavior can shift unpredictably toward unsafe behaviors in multi-turn conversations, potentially undermining users' goals or wellbeing. To address this harmful opacity in multi-turn HAI, we introduce an interface that surfaces an LLM's internal activations to help users anticipate and evaluate model behavior. We construct behavioral scores from these activations for six HAI-relevant traits using mechanistic interpretability methods, then visualize them via a dynamic sunburst, drift panel, and attentional cues. In a counterbalanced, randomized controlled study (N=246), both static and dynamic transparency interfaces improved participants' anticipation and evaluation performance (d=-0.36) while helping temper overconfidence in their predictive abilities. Furthermore, the multi-turn interface significantly outperformed the static visualization (d=-0.32) while remaining similarly usable. Our findings suggest that continuously surfacing internal model representations to everyday users is a meaningful and tractable step toward continuously transparent and informed HAI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.20576v1">One Task, Many Interfaces: Simple Cross-Interface Consistency Training Improves LLM Agent Robustness</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Post-training binds the competence of an LLM agent to the interface it was trained on. Semantics-preserving perturbations of that interface, such as renaming an action or changing the notation, leave the underlying task unchanged yet make the agent worse at it. Scoring each sample independently means interface randomization only broadens this dependence from a single interface to the training interface distribution, so out-of-distribution perturbations still degrade the agent. We propose Cross-Interface Policy Regularization (CIPR), a self-supervised objective that replaces action names with uninformative identifiers and presents each training sample under two interfaces in which equivalent actions are identified by construction. CIPR penalizes the KL divergence between the action distributions induced under the two interfaces, so the policy behaves consistently across them. We evaluate CIPR in both supervised and reinforcement-learning post-training, where it improves robustness. On agentic function calling, held-out interface perturbations reduce the accuracy of a post-trained agent by nearly 25%, and CIPR recovers 10.5% while leaving accuracy on the unchanged interface intact.
    </details>
</div>
