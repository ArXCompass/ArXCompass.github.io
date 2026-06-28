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

## Papers

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
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04296v1">The Saturation Trap and the Subjectivity of Intervention Timing: Why Affect-Based Triggers and LLM Judges Fail to Time Interventions on Autonomous Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-02
      | 💬 11 pages, 5 tables. Code and data:https://github.com/2025eb1100268-tech/intervention-timing-saturation-trap
    </div>
    <details class="paper-abstract">
      As autonomous AI agents move from conversational systems to long-horizon software execution, runtime safety layers that decide when to interrupt an agent have become essential. We study this timing problem using a continuous 18-dimensional affective-dynamics engine (HEART) as a diagnostic probe, evaluating four intervention trigger families - absolute state thresholds, composite state-action patterns, regex reasoning-feature extraction, and zero-shot LLM-as-judge - against human-annotated intervention points on SWE-bench-Verified debugging traces. We report three findings. First, a State Saturation Trap: agents show no recovery signal under sustained difficulty, so modeled frustration quickly crosses the threshold and stays at its maximum, converting threshold-on-state triggers from moment detectors into near-constant indicators that fire on 39-83% of actions across five trajectories. Second, a capability-and-context floor for LLM judges: a small model (gpt-5.4-mini) never fires, while frontier and cross-vendor models escape the zero-firing floor only with full-trajectory context, and even then reach only F1 0.17-0.40 at up to 90x the cost. Third, and most importantly, the supervised target is not reproducible among humans: three trained annotators using one rubric on a 56-action trajectory agree on where to intervene only slightly above chance (location Krippendorff's alpha = +0.047; best pairwise Cohen's kappa = +0.349) and not at all on intervention type (pause degenerate; clarify below chance; reflect only alpha = +0.226). We conclude that intervention timing is a low-reliability construct, making single-annotator F1 an unsuitable optimization target. Our contribution is the joint mapping of this problem across human inter-rater reliability, four detector architectures, a cross-model LLM-judge sweep, and a reproduced saturation effect, rather than any single detector's accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.06911v2">TamperBench: Systematically Stress-Testing LLM Safety Under Fine-Tuning and Tampering</a></div>
    <div class="paper-meta">
      📅 2026-06-02
      | 💬 25 pages, 15 figures
    </div>
    <details class="paper-abstract">
      As increasingly capable open-weight large language models (LLMs) are deployed, improving their tamper resistance against unsafe modifications, whether accidental or intentional, becomes critical to minimize risks. However, there is no standard approach to evaluate tamper resistance. Varied datasets, metrics, and tampering configurations make it difficult to compare safety, utility, and robustness across different models and defenses. To address this, we introduce TamperBench, the first unified framework to systematically evaluate the tamper resistance of LLMs. TamperBench (i) curates a repository of state-of-the-art weight-space fine-tuning attacks, latent-space representation attacks, and alignment-stage defenses; (ii) enables realistic adversarial evaluation through systematic hyperparameter sweeps per attack-model pair; and (iii) provides both safety and utility evaluations. We use TamperBench to evaluate 21 open-weight LLMs, including defense-augmented variants, across nine tampering threats using standardized safety and capability metrics with hyperparameter sweeps per model-attack pair. The results provide insights including effects of post-training on tamper resistance, that jailbreak-tuning is typically the most severe attack, and that current alignment-stage defenses largely fail to withstand attack sweeps. Code is available at https://github.com/criticalml-uw/TamperBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04274v1">Long Live Fine-Tuning: Task-Specific Transformers Outperform Zero-Shot LLMs for Misinformation Response Classification on Reddit</a></div>
    <div class="paper-meta">
      📅 2026-06-02
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become default tools for online information verification, an implicit assumption follows them: that scale and general capability are sufficient for nuanced classification of misinformation discourse. We test this assumption directly on 900 Reddit comments spanning three PolitiFact-verified misinformation claims (environment, health, immigration), labelled as belief (propagates the claim), fact-check (corrects it), or other. We compare nine models across three paradigms -- BART-MNLI, three Llama variants, three commercial frontier LLMs (Claude Haiku 4.5, Gemini Flash Lite 2.5, Claude Sonnet 4.6), and fine-tuned DistilBERT and RoBERTa -- under universal and topic-specific label schemas. The assumption does not hold. Fine-tuned RoBERTa reaches 0.62 macro-$F_1$ against a best zero-shot result of 0.50 (Claude Haiku 4.5), at a fraction of the per-query cost; the supervised advantage is concentrated on the belief class, the implicit, affective category every zero-shot model under-detects. Scaling does not help: Llama-3-8B matches Llama-3-70B, and Claude Sonnet 4.6 underperforms the smaller Haiku under generic labels, collapsing belief detection to 0.17 and refusing outright on a subset of comments flagged as sensitive. This is a safety-alignment artefact, not a capacity limit. Label schema and topic jointly shape zero-shot performance, with the same model varying by more than 0.13 macro-$F_1$ across topics under matched labels. In a verification context, where missing belief is the costlier error, task-specific fine-tuning remains the more reliable choice despite the proliferation of large generative models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04272v1">RL Excursions during Pre-Training: Re-examining Policy Optimization for LLM training</a></div>
    <div class="paper-meta">
      📅 2026-06-02
    </div>
    <details class="paper-abstract">
      The standard LLM training pipeline applies reinforcement learning (RL) only after pre-training and supervised fine-tuning (SFT). We question this status quo by training a LLM from scratch and applying RL, SFT, and SFT followed by RL directly to intermediate pre-training checkpoints. We find that RL is effective very early, and often matches the full SFT$\to$RL pipeline early as well. Through experiments on harder problems, we find that targeted pre-training data composition is a strong lever for RL effectiveness, even more so than model scale. Beyond reasoning accuracy, applying RL directly to base checkpoints expands the model's distribution; the sharpening effect reported in recent work arises only when RL follows SFT. The general capabilities of the model remain essentially unchanged by RL, while they degrade following SFT. Finally, we merge RL and SFT objectives by parallel averaging, which outperforms across all other training methods discussed, across metrics, while preserving general capabilities. Together, these results suggest that LLM training might benefit from an expanded use of RL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04262v1">Can I Take Another Dose? Evaluating LLM Decision-Making Under Temporal Uncertainty in OTC Dosing QA</a></div>
    <div class="paper-meta">
      📅 2026-06-02
      | 💬 16 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used for everyday health questions, including whether a user can safely take another dose of an over-the-counter (OTC) medication. Yet this common safety-relevant setting remains underexplored in existing medical QA evaluations, where correct answers require tracking dose timing, computing rolling 24-hour intake, following product-label constraints, and handling incomplete medication histories. We introduce DOSEBENCH, a focused benchmark of 81 curated OTC dosing scenarios focused on adult acetaminophen and ibuprofen use, with manually annotated gold references. We evaluate four LLMs across repeated runs using metrics for decision correctness, consistency, explanation verifiability, failure types, and confidence-related signals, resulting in 1,620 model responses. Our results show that models frequently struggle with rolling-window reasoning and ambiguity-sensitive cases and that stable or confident-looking responses can still violate dosing constraints. These findings suggest that OTC dosing QA provides a narrow yet practical testbed for evaluating temporal reasoning, constraint following, and safety-relevant uncertainty handling in medical QA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04246v1">StepPRM-RTL: Stepwise Process-Reward Guided LLM Fine-Tuning for Enhanced RTL Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-06-02
      | 💬 6 pages, 2 figures, DAC'2026
    </div>
    <details class="paper-abstract">
      Automatic generation of RTL code for digital hardware designs remains challenging due to long-horizon reasoning, multi-step dependencies, and strict correctness constraints in Verilog and VHDL. We present StepPRM-RTL, a novel framework that combines stepwise trajectory modeling, process-reward modeling (PRM), and retrieval-augmented fine-tuning (RAFT) to enhance both the functional correctness and reasoning fidelity of LLM-based RTL code generation. StepPRM-RTL constructs stepwise reasoning trajectories from canonical solutions, where each step contains a rationale and incremental code modification. A Process Reward Model (PRM) evaluates intermediate steps, providing dense feedback that guides reinforcement-style updates during RAFT fine-tuning. Monte Carlo Tree Search (MCTS) explores alternative reasoning paths, enriching the training dataset with high-quality trajectories. This integration of stepwise and outcome-aware rewards allows the model to learn both how and why to construct correct RTL, improving long-horizon reasoning beyond standard supervised or outcome-based training. Experimental evaluation on benchmark Verilog and VHDL datasets demonstrates that StepPRM-RTL outperforms the best prior methods by over 10\% in functional correctness and reasoning fidelity metrics. Ablation studies confirm that the combination of PRM-guided rewards and stepwise trajectory exploration is key to its performance. StepPRM-RTL generalizes across RTL languages and provides a scalable framework for high-fidelity, interpretable code generation, establishing a new standard for LLM-assisted hardware design automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04226v1">PerceptTwin: Semantic Scene Reconstruction for Iterative LLM Planning and Verification</a></div>
    <div class="paper-meta">
      📅 2026-06-02
      | 💬 Accepted at ICRA 2026 (Vienna); published on arxiv for archival purposes. See also https://percept-twin.github.io/
    </div>
    <details class="paper-abstract">
      Simulation environments are useful for both robot policy learning and planning verification and validation. Traditionally, the process of creating a simulation was onerous. Creating a bespoke simulation environment for each individual environment that a robot would operate in was simply infeasible. In this work, we introduce PerceptTwin, a fully automatic pipeline that constructs interactive simulations directly from semantic scene representations produced by a robot's perception stack. PerceptTwin combines open-vocabulary object maps with 3D asset generation, affordance prediction, and commonsense condition checking. These interactive simulations can be used to validate and refine plans before they are executed on the robot hardware. Borrowing from the AI alignment literature, we also introduce an LLM judge that verifies plan correctness and alignment with human preferences. Experiments show that PerceptTwin feedback allows LLM planners to refine plans, enhance safety, and resist harmful black-box prompting attacks. In our suite of tasks, PerceptTwin improves plan success by an average of approximately 39% for GPT5, GPT5Mini, and GPT5Nano planners. Additionally, PerceptTwin also improves human plan verification by up to 18% on average for plans that fail due to unfilled skill preconditions. Our results demonstrate the potential of open-vocabulary scene simulation from robot perception as a foundation for safer, more reliable robot planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.01146v2">PersistBench: When Should Long-Term Memories Be Forgotten by LLMs?</a></div>
    <div class="paper-meta">
      📅 2026-06-02
      | 💬 76 pages, 34 figures, ICML (2026)
    </div>
    <details class="paper-abstract">
      Conversational assistants are increasingly integrating long-term memory with large language models (LLMs). This persistence of memories, e.g., the user is vegetarian, can enhance personalization in future conversations. However, the same persistence can also introduce safety risks that have been largely overlooked. Hence, we introduce PersistBench to measure the extent of these safety risks. We identify two long-term memory-specific risks: cross-domain leakage, where LLMs inappropriately inject context from the long-term memories; and memory-induced sycophancy, where stored long-term memories insidiously reinforce user biases. We evaluate 18 frontier and open-source LLMs on our benchmark. Our results reveal a surprisingly high failure rate across these LLMs - a median failure rate of 53% on cross-domain samples and 97% on sycophancy samples. To address this, our benchmark encourages the development of more robust and safer long-term memory usage in frontier conversational systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04197v1">Exploring the Topology and Memory of Consensus: How LLM Agents Agree, Fragment, or Settle When Forming Conventions</a></div>
    <div class="paper-meta">
      📅 2026-06-02
      | 💬 Submitted to the Journal of Artificial Societies and Social Simulation (JASSS)
    </div>
    <details class="paper-abstract">
      How much should an LLM agent remember, and how should multi-agent systems be connected when trying to reach consensus? We show these two design choices interact in a way that flips the sign of memory's effect on coordination. Across 432 simulation runs of a networked Naming Game on eight fixed 16-agent topologies, we vary memory depth and network structure. Longer memory slows the time to reach steady state in decentralized networks but accelerates it in centralized ones; the same parameter pushes the system in opposite directions depending on topology. Critically, "faster settling" in centralized networks means locking in to a fragmented plateau more quickly, not reaching system-wide consensus, which can be used to generate diverging opinions. We further document a memory-mediated speed-unity trade-off: centralized networks consistently preserve more competing conventions than decentralized networks, but their settling speed depends sharply on memory. At the agent level, within-network analyses show that high-betweenness bridges suffer a brokerage penalty while agents in locally clustered neighborhoods achieve higher coordination success. Finally, in search of analytically tractable generative mechanisms, we find that agents' choices are well captured by Fictitious Play, indicating belief-based rather than reward-based adaptation. The practical implication: memory depth and communication topology should be co-designed, not optimized in isolation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04141v1">Caught in the Act(ivation): Toward Pre-Output and Multi-Turn Detection of Credential Exfiltration by LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-02
    </div>
    <details class="paper-abstract">
      LLM agents often place sensitive credentials in the same context window as untrusted retrieved content, creating a direct path for indirect prompt injection to induce credential exfiltration. We study this failure mode through three complementary defenses. First, we ask whether activation probes can detect credential access before output tokens are emitted. Second, we construct honeytokens from format-specific character models and calibrate detection with split conformal prediction. Third, we treat multi-turn exfiltration as a cumulative information-flow problem and track an estimated leakage budget across conversation turns. In controlled experiments on open-weight models, activation features separate benign and credential-seeking prompts with high accuracy, including under held-out encoding transformations. In a small synthetic multi-turn suite, cumulative accounting detects attacks that per-turn detectors miss. These results are preliminary: the multi-turn benchmark is in-house and small, the activation method requires white-box access, and the information estimator provides a practical signal rather than a formal upper bound. Still, the results suggest that credential-exfiltration defenses should combine pre-output monitoring, calibrated canary detection, and temporal leakage accounting rather than relying only on text-level output filters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04118v1">Computational conceptual history of scientific concepts: From early digital methods to LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-02
      | 💬 19 pages, chapter in the book Understanding Science with Large Language Models? (pp. 383-412). transcript. Edited by Arno Simons, Adrian Wüthrich, Michael Zichert, Gerd Graßhoff (eds.)
    </div>
    <details class="paper-abstract">
      This article situates large language models (LLMs) within the longer history of computational approaches to concept analysis in the history, philosophy, and sociology of science (HPSS). We examine what LLMs add to existing methods, how they inherit longstanding problems, and review recent case studies that employ them. In the first part, we reconstruct computational conceptual history before LLMs by bringing together three strands of work: early digital methods in HPSS, distributional approaches from digital history and related research, and lexical semantic change detection. We provide an overview of the main challenges and opportunities, focusing on corpus construction, operationalization and modelling choices, and evaluation and interpretation. In the second part, we turn to the era of LLMs, starting with a short introduction to LLMs before reviewing LLM-based work on lexical semantic change detection and relevant case studies in HPSS. We then revisit the earlier methodological questions, showing how issues of corpus construction, model choice and training data, operationalization trade-offs, and evaluation and interpretation play out in LLM-based workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30021v2">Recovering Diversity Without Losing Alignment: A DPO Recipe for Post-Trained LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-02
      | 💬 Under Review. 26 pages, 3 figures, 16 tables
    </div>
    <details class="paper-abstract">
      Many open-ended instructions have multiple valid answers that users can benefit from seeing, but post-training often narrows an LLM's output space toward a small set of canonical responses. We introduce REDIPO, an offline DPO data-construction pipeline for recovering distinct valid answer modes while preserving the alignment benefits of the instruct model. For each prompt, REDIPO samples responses from both base and instruct models, rewrites base-model responses with the instruct model, filters candidates for safety and instruction-following quality, and builds preference pairs that favor marginally diverse responses among candidates with similar instruction-following reward. Across Qwen3-4B, OLMo-3-7B, and LLaMA-3.1-8B, REDIPO improves NoveltyBench distinct_k by 134%, 33%, and 44% relative to the instruct checkpoints, while DivPO changes diversity by 0%, -6%, and -4% on the same models. These gains largely maintain MTBench, IFEval, and Arena-Hard performance, and reduce direct-category HarmBench attack success rate. Ablations show that marginal-diversity pair selection and base-response rewriting drive the diversity gains, while filtering and quality-bounded pairing help maintain alignment. Overall, our results show that diverse valid answers from base-model generations can be reintroduced through carefully constructed preference data while retaining the alignment benefits of post-training. We release our code and data at https://github.com/vsamuel2003/ReDiPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03967v1">AlignAtt4LLM: Fast AlignAtt for Decoder-Only LLMs at IWSLT 2026 Simultaneous Speech Translation Task</a></div>
    <div class="paper-meta">
      📅 2026-06-02
      | 💬 Accepted to IWSLT 2026
    </div>
    <details class="paper-abstract">
      We describe AlignAtt4LLM, an IWSLT 2026 simultaneous speech translation system for English to German, Italian, and Chinese. The system is a synchronous cascade: Qwen3-ASR with forced alignment produces an incrementally updated source transcript, and Gemma-4 E4B-it translates that prefix under an MT-side AlignAtt policy. To our knowledge, this is the first application of AlignAtt to a decoder-only LLM, where the encoder-decoder cross-attention used by earlier AlignAtt systems is absent. We recover a usable policy by proposing (1) an explicit source span in the prompt, (2) offline selection of translation-specific alignment heads, (3) selective qk-fast replay of the draft-to-source attention block, and (4) runtime query/key capture that preserves model outputs bit-identically. On the IWSLT 2026 development set, AlignAtt4LLM outperforms the supplied baselines for the European target languages, English to German and English to Italian, in both the low-latency regime around 2 seconds and the high-latency regime below 4 seconds CU-LongYAAL. Results for English to Chinese are more mixed, but the method is not tied to Gemma-4: because AlignAtt4LLM only requires a deterministic prompt layout, calibrated attention heads, and query/key capture, the same policy can be reapplied to stronger translation-focused decoder-only MT backbones for non-European target languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.02260v2">Position: Adversarial ML for LLMs Is Not Making Any Progress</a></div>
    <div class="paper-meta">
      📅 2026-06-02
      | 💬 Accepted at ICML 2026 Position Paper Track
    </div>
    <details class="paper-abstract">
      In the past decade, considerable research effort has been devoted to securing machine learning (ML) models that operate in adversarial settings. Yet, progress has been slow even for simple "toy" problems (e.g., robustness to small adversarial perturbations) and is often hindered by non-rigorous evaluations. Today, adversarial ML research has shifted towards studying larger, general-purpose language models. In this position paper, we argue that the situation is now even worse: in the era of LLMs, the field of adversarial ML studies problems that are (1) less clearly defined, (2) harder to solve, and (3) even more challenging to evaluate. As a result, we caution that yet another decade of work on adversarial ML may be failing to produce meaningful progress.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.06254v6">EviRerank: Adaptive Evidence Construction for Long-Document LLM Reranking</a></div>
    <div class="paper-meta">
      📅 2026-06-02
    </div>
    <details class="paper-abstract">
      Decoder-only LLM rerankers struggle with long documents: inference is costly and relevance signals can be diluted by irrelevant context. Motivated by a diagnostic attention analysis suggesting that appended irrelevant context can weaken query-focused interactions, we propose EviRerank, an evidence-based long-document reranking framework for decoder-only LLMs. EviRerank first scores document blocks with a lightweight selector, such as BM25, a bi-encoder, or a cross-encoder. It then constructs a compact reranking context under a hard token cap by dynamically budgeting evidence blocks with Adaptive Evidence Budgeting (AEB) and adding a compact global cue via Summary Augmentation (SA). Finally, the compact evidence context is reranked with a decoder-only LLM. Across TREC DL'19, DL'22, DL'23, and MLDR-zh, EviRerank consistently outperforms full-document LLM reranking and strong block-selection baselines while reducing input length. RankZephyr-7B validation further confirms transfer to listwise reranking. On TREC DL'19, EviRerank reaches up to 0.744 nDCG@10 and 0.307 MAP, improving over RankLLaMA while using a compact evidence context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03910v1">NetKV: Network-Aware Decode Instance Selection for Disaggregated LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-06-02
    </div>
    <details class="paper-abstract">
      Disaggregated LLM inference forces the KV cache to traverse the datacenter network before decoding begins, so transfer time enters directly into the Time to First Token (TTFT) budget. Current schedulers route on compute load and prefix-cache locality alone, ignoring the topological distance and dynamic congestion between prefill and decode instances. We close this gap with a thin operator-to-scheduler interface, the network cost oracle, and we prove that ignoring the network term renders cache-aware-only scheduling arbitrarily suboptimal as context length grows. NetKV, the O(|D|) per-request greedy that consumes this oracle, has tier rankings that are provably robust to stale telemetry. On a 64-GPU four-tier fat-tree simulator driven by Mooncake traces, NetKV reduces mean TTFT by up to 21.2% over round-robin and 17.6% over a tuned cache+load-aware scheduler, lifts SLO attainment by up to 20.1 percentage points, and keeps the Time Between Tokens overhead below 0.5 ms in every condition tested, with no changes to the transport, inference engine, or hardware.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03895v1">Agent libOS: A Library-OS-Inspired Runtime for Long-Running, Capability-Controlled LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-02
      | 💬 14 pages, 1 figure, 2 tables
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents are evolving from request-response assistants into long-running software actors: they maintain state across model calls, fork subtasks, wait for external events, request human authority, generate tools, and perform side effects that must be resumed and audited. This paper presents Agent libOS, a library-OS-inspired runtime substrate for LLM agents. Agent libOS runs above a conventional host operating system; it does not implement hardware drivers, kernel-mode isolation, or a POSIX-compatible operating system. Instead, it treats an agent as an AgentProcess: a schedulable execution subject with process identity, parent-child lineage, lifecycle state, a tool table derived from an AgentImage, typed Object Memory, explicit capabilities, human queues, checkpoints, events, and audit records. Its central design rule is tools are libc-like wrappers; runtime primitives are the authority boundary. Filesystem access, object access, sleeps, human approval, JIT tool registration, and external side effects are checked at primitive boundaries under explicit capabilities and policy. We describe the design, threat model, Python prototype, and safety-oriented evaluation. The current prototype implements async scheduling, namespace-local Object Memory, runtime-integrated human approval, one-shot permission grants, per-process working directories, shell and image-registration primitives, Deno/TypeScript JIT tools over a libOS syscall broker, filesystem/object bridge tools, an injectable Resource Provider Substrate, deterministic demos, real-model smoke scripts, and 123 regression tests at the time of writing. Rather than improving planner accuracy, Agent libOS demonstrates a runtime substrate in which long-running LLM agents can be scheduled, authorized, resumed, and audited without treating tool dispatch as the trust boundary.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03876v1">From 'What' to 'How' and 'Why': Sharing LLM-Generated Retrospective Summaries of Older Adults' Passive Tracking Data with Remote Family Members</a></div>
    <div class="paper-meta">
      📅 2026-06-02
    </div>
    <details class="paper-abstract">
      With the growing prevalence of modern ubiquitous computing technologies, multi-modal tracking systems hold promise for providing timely awareness and reassurance to stakeholders such as remote family members (RFMs) of older adults, who play a central role in care coordination. However, combining heterogeneous data streams into high-level, meaningful content - such as retrospective summaries - remains challenging. While recent work has demonstrated the promise of large language models (LLMs) for interpreting multi-modal tracking data, less attention has been given to generating narrative accounts for stakeholders like RFMs, who possess rich personal knowledge of older adults and strong emotional responsibility, yet have limited visibility into their daily lives and limited capacity for caregiving. In this work, we explore how LLMs can be used to generate retrospective summaries from multi-modal tracking data for RFMs of older adults. We leveraged and customized an existing system, Vital Insight, to generate initial summaries on different dates and data availability scenarios as technology probes, and conducted interviews with 11 RFMs to gather feedback. Based on these insights, we redesigned the system into a multi-layer, multi-agent, insight-driven summary approach that builds from objective statistics and descriptions to enriched, context-aware narratives. We then compared the redesigned summaries with the initial versions through a survey with the same 11 RFMs and found significant improvements in satisfaction, perceived helpfulness, trust, and willingness to receive the summaries. We conclude by presenting design implications for AI-generated summaries for RFMs and broader contexts, emphasizing the need to support RFMs' sensemaking shift from simply presenting ''What'' data were collected, to explaining ''How'' is my loved one doing and ''Why''.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03867v1">A Training-Free Mixture-of-Agents Framework for Multi-Document Summarization using LLMs and Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2026-06-02
      | 💬 Accepted by Neural Computing and Applications
    </div>
    <details class="paper-abstract">
      Multi-Document Summarization (MDS) plays a critical role in distilling essential information from collections of textual data. Existing approaches often struggle to capture complex inter-document relationships, rely heavily on large amounts of labeled data for supervised training, or exhibit limited generalization across domains and languages. To address these limitations, we present a training-free mixture-of-agents framework for MDS that leverages the complementary strengths of large language models (LLMs) and knowledge graphs. Our approach decomposes summarization into specialized agent tasks: extractive selection, knowledge-aware abstraction, and iterative refinement, each operating without task-specific fine-tuning. We unify their outputs using a multi-perspective consistency mechanism guided by LLMs. Experiments across four datasets in English and Vietnamese demonstrate state-of-the-art or competitive performance, validating the effectiveness and adaptability of our modular design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03866v1">Taiji: Pareto Optimal Policy Optimization with Semantics-IDs Trade-off for Industrial LLM-Enhanced Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-06-02
      | 💬 8 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Scaling recommender systems via large language models (LLMs) has become a prominent trend in the industry. However, aligning the LLM's semantic space with the recommender's ID space via post-training (e.g., SFT and RL) remains challenging. Existing LLM4Rec paradigms are bottlenecked by two main issues: (1) the difficulty of measuring and improving chain-of-thought (CoT) quality in open-domain recommendation during SFT, and (2) the neglect of the trade-off between LLM semantic rewards and recommendation preference rewards during RL alignment. Inspired by these challenges, we present Taiji, a novel LLM-as-Enhancer framework designed for industrial recommender systems. To overcome the SFT bottleneck, we utilize reverse-engineered reasoning and open-ended rejection sampling to generate high-quality, domain-specific CoT data. To resolve the RL alignment issue, we propose Pareto Optimal Policy Optimization (POPO), which adaptively adjusts cross-domain reward weights. Theoretically, it achieves an optimal trade-off between the semantic world knowledge of LLMs and the collaborative ID features representing online user preferences. Extensive offline evaluations and online A/B tests validate the effectiveness of Taiji. Deployed on Kuaishou's advertising platform since May 2026, Taiji currently serves over 400 million users daily, yielding significant commercial revenue and demonstrating its robust scalability in web-scale environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03852v1">FLARE: Fine-Grained Diagnostic Feedback for LLM Code Refinement</a></div>
    <div class="paper-meta">
      📅 2026-06-02
    </div>
    <details class="paper-abstract">
      Large language models often generate code with bugs. Existing methods rely on feedback signals such as test failures and self-critiques to iteratively refine the generated code. Such signals are either too coarse-grained or too high-level, which is not sufficient to inform the model where to fix the bug. In this work, we present Flare, an iterative framework with a lightweight diagnostic model that predicts line-level suspiciousness signals for bug localization and code refinement. Given the inherent uncertainty of diagnostic predictions, Flare searches over the top-k suspicious regions and selects the best candidate according to execution outcomes. Experiments on LiveCodeBench and BigCodeBench with five base LLMs show that, even without candidate search (k=1), Flare outperforms the strongest baseline with an absolute improvement from 1.72% to 7.42%. Furthermore, searching over 10 candidates yields an average improvement of 8.50% compared with no candidate search. When evaluated in isolation, our lightweight diagnostic model achieves the best performance compared with recent fault localization methods, demonstrating that it can provide reliable fine-grained guidance for code refinement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13663v2">SAIL: Sound Abstract Interpreters with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-02
      | 💬 43 pages, 21 figures
    </div>
    <details class="paper-abstract">
      How to construct globally sound abstract interpreters to safely approximate program behaviors remains a bottleneck in abstract interpretation. In this paper, we show the potential of using state-of-the-art LLMs to automate this tedious process. Focusing on the neural network verification area, we synthesize non-trivial sound abstract transformers across diverse abstract domains using LLMs to search within infinite space from scratch. We formalize the synthesis task as a constrained optimization problem, for which we design a novel mathematically grounded cost function that measures the degree of unsoundness of each generated candidate transformer, while enforcing hard syntactic and semantic validity constraints. Building on this formulation, we introduce SAIL, a novel unified framework that combines model generation, syntactic and semantic validation, and cost-function-based refinement to synthesize globally sound abstract transformers. Evaluation results show that SAIL not only matches the performance of manually designed transformers, but also is able to synthesize sound and high-precision transformers that do not exist in the literature for complex non-linear operators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.02240v2">AgentRedBench: Dynamic Redteaming and Integration-Aware Defense for LLM Agents over SaaS Integrations</a></div>
    <div class="paper-meta">
      📅 2026-06-02
    </div>
    <details class="paper-abstract">
      Indirect prompt injection in tool-use agents is a concrete production threat: LLM agents read from integrations (third-party services such as Gmail, Salesforce, or Jira accessed through tool calls) whose response content the user neither writes nor controls. Existing benchmarks under-measure the threat: most cover only a handful of integrations with the same attack payload replayed across runs, and open-source guards are trained on chat-style data rather than tool-response content. We introduce AGENTREDBENCH, a dynamic LLM-driven redteaming benchmark of 215 subtle underspecified authorization (attacks at the boundary of what the user's request authorises) scenarios across 24 enterprise integrations in nine functional families and five attack types. Across an eight-model panel (Anthropic, OpenAI, Google), no-guard ASR (attack success rate) ranges from 32% (Claude Sonnet 4.6) to 81% (Gemini 3 Flash). To keep the scenario set out of training corpora and preserve headline ASR meaning over time, we release the codebase, integration schemas, and AGENTREDGUARD model openly; the canonical scenarios are evaluated through a maintainer-mediated channel with immutable versioning. We release AGENTREDGUARD alongside the benchmark: a guard trained on an integration-diverse corpus of adversarial tool-response content. AGENTREDGUARD cuts panel ASR from 69.9% to 2.4% at 0.37% false-positive rate, outperforming every open-source baseline with non-trivial detection (Llama Guard, PromptGuard 2, ProtectAI) on both axes. Cross-integration and cross-attack type holdouts both confirm the gain transfers beyond the training subset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03770v1">E2LLM: Towards Efficient LLM Serving in Heterogeneous Edge/Fog Environments</a></div>
    <div class="paper-meta">
      📅 2026-06-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become integral to modern applications, yet their deployment remains challenging. Beyond executing the models themselves, practical deployment must address cost efficiency, low latency, and optimal resource utilization. Conventional approaches typically assume that an entire model can be hosted on a single device, which does not hold in many real-world scenarios, particularly in Edge and Fog environments where device resources are constrained. In this paper, we introduce E2LLM, a framework designed to enable efficient LLM deployment in such resource limited settings. Rather than simply partitioning a single model across all available devices, E2LLM replicates the full model across multiple groups of devices (replicas) and applies model parallelism within each replica. Each replica is assigned a specialized role PREFILL or DECODER based on its efficiency in handling input and output tokens. This separation leverages the inherent differences between these two phases of LLM inference. To effectively organize devices, we utilize a Genetic Algorithm to form clusters that maximize system performance. Within each cluster, we apply Dynamic Programming to determine an optimal partitioning strategy that minimizes bottlenecks in model-parallel execution. Experimental results demonstrate that our approach adapts robustly to varying workloads, including scenarios with significant variation in input and output token lengths. Compared to the Splitwise baseline, E2LLM reduces average waiting time by over 50% under high-demand conditions
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03761v1">Framing Migration News with LLMs: Structured CoT as a Support for Human Interpretation</a></div>
    <div class="paper-meta">
      📅 2026-06-02
    </div>
    <details class="paper-abstract">
      Frame analysis of migration news is a socially consequential task: media scholars and researchers who study how migration is narrated need tools that are not only accurate, but transparent, auditable, and accessible within the resource constraints typical of academic research groups. Existing LLM-based approaches rely on proprietary APIs and large models that raise concerns about data privacy, reproducibility and equitable access among media researchers. This work studies how a locally deployable open-source LLM can support interpretable frame analysis as an assistive tool. We introduce a Structured Chain-of-Thought (SCoT) prompting approach using Llama3-8B, enabling step-by-step justifications grounded in predefined framing categories. This structured design allows users to audit model outputs and examine alternative interpretations in a task that is inherently subjective. We evaluate our approach on a dataset of migration-related news and show that SCoT improves classification performance over zero-shot and few-shot baselines while remaining feasible on a single GPU. Then, we conduct a human-centered evaluation in which annotators assess the coherence and influence of "the model's reasoning". Results indicate that SCoT explanations are generally perceived as logical (mean score 4.1/5, though with notable variation across texts) and can prompt reflection on initial interpretations, even when disagreement persists. Our findings highlight both the potential and risks of LLM-assisted frame analysis. While structured reasoning can increase the traceability of model outputs and support critical interpretation, it can also influence human judgment in subtle ways. By enabling local deployment and emphasizing human-in-the-loop interaction, this work contributes to discussions on responsible and accessible computational tools for the study of socially impactful media narratives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02380v5">LLMs, Reasoning and Plagiarism</a></div>
    <div class="paper-meta">
      📅 2026-06-02
      | 💬 The authors explicitly reserve all rights in this work. No permission is granted for the reproduction, storage, or use of this document for the purpose of training artificial intelligence systems or for text and data mining (TDM), including but not limited to the generation of embeddings, summaries, or synthetic derivatives. Claude and Gemini were used in writing this manuscript
    </div>
    <details class="paper-abstract">
      Recent reports claim that Large Language Models (LLMs) derive new science and exhibit human-level general intelligence. Such claims are entangled with two different narratives about what LLMs do: one in which they are an engine of synthesis that genuinely reasons to new knowledge, and one in which they retrieve and re-emit the work of others without attribution. In the scientific setting these are best understood as a contrast between \emph{reasoning} and \emph{plagiarism}. Finding where the truth lies between these two narratives is very challenging, as central components of the model -- the training data and the interaction transcript -- remain opaque. Thus claims of LLM reasoning do not satisfy Popper's refutability principle. We propose guidelines for transparency and reproducibility that will allow reasoning claims to be studied using the scientific method. The dominance of the reasoning narrative, we suggest, is in practice encouraging plagiarism in the scientific literature; we discuss what might be done about it.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03739v1">Entropy Gate: Entropy Quenching for Near-Lossless Token Compression in LLM Pipelines</a></div>
    <div class="paper-meta">
      📅 2026-06-02
    </div>
    <details class="paper-abstract">
      LLM pipelines waste substantial token budgets on low-information content: repeated context, verbose responses, and redundant boilerplate. We introduce Entropy Gate, a token compression framework applying entropy quenching $-$ a thermodynamic process that progressively freezes out low-energy tokens while preserving semantic fidelity. Each token receives a multi-factor information energy $E(t)$ combining statistical, structural, and positional components. An adaptive quenching schedule $T(τ) = T_0 / (1 + ατ)$ removes tokens whose Boltzmann survival probability $p_i = \exp(-E_i / kT)$ falls below threshold, with a fidelity gate halting compression when energy-weighted similarity drops below $θ$. We prove token selection by descending $E(t)$ maximizes expected semantic preservation, that quenching produces nested survival sets, and that achievable compression approaches the information-theoretic limit $\text{CR} \to 1 - I(P; T)/H(P)$. A Phase 1 heuristic achieves 40-60% compression across five prompt categories while maintaining $S_E > 0.80$, with energy-squared amplification $E \to E^2$ adding 10-25 percentage points. Context deduplication adds 50-70% savings on repeated blocks. Output-side quenching, motivated by findings that brevity improves accuracy, further reduces response overhead. Combined with external memory, reduction composes multiplicatively to 88-96% for agentic workloads. The framework is stateless, model-agnostic, and deploys as an OpenAI-compatible HTTP proxy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04067v1">Need to Know: Contextual-Integrity-Grounded Query Rewriting for Privacy-Conscious LLM Delegation</a></div>
    <div class="paper-meta">
      📅 2026-06-02
    </div>
    <details class="paper-abstract">
      As LLMs become increasingly woven into everyday workflows, user queries sent to cloud hosted LLMs routinely mix task-essential content with task non-essential sensitive disclosures, yet type based PII redaction is context agnostic and may raise two issues: over disclosing untyped sensitive context and over removing answer bearing spans. We recast privacy preserving query rewriting under Contextual Integrity: a span should be forwarded only if it is necessary for the task. We introduce DelegateCI-Bench, the first task based Contextual Integrity benchmark for privacy-conscious delegation, comprising 3,167 samples that combine high quality synthetic data spanning 11 tasks and 20 task types, WildChat based real user queries, and a medical challenge set with dense sensitive information. Building on this benchmark, we propose a CI-guided reinforcement learning framework that converts essential and non-essential sensitive spans into verifiable optimization signals, and train a query rewriter to preserve task critical information while suppressing unnecessary sensitive disclosure. Experiments show that our learned rewriter achieves the best privacy-utility tradeoff, achieving up to +10.1 average utility over on-device baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.31381v2">LLM Judges Inconsistently Disagree Across Safety Criteria and Harm Categories</a></div>
    <div class="paper-meta">
      📅 2026-06-02
      | 💬 8 pages plus appendices, under review
    </div>
    <details class="paper-abstract">
      We evaluate the consistency of automated judges in conducting a multi-dimensional safety evaluation in a reference-free setup. Our results indicate that Large Language Models are unreliable judges in identifying safety issues related to machine-generated advice in regulated domains such as finance, although they are more reliable at identifying more overt forms of unsafe/harmful content such as violence. The degree of inconsistency in a model's judgments can vary significantly by the chosen safety criteria and can be impacted by the language of the content and its linguistic style as well. Finally, there is high disagreement among different judges for the same output, across domains, safety criteria, and languages. These findings provide new insights on the practice of using LLMs as evaluators and offer several recommendations for practitioners on how to use automated judges in practical scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.31530v2">UNISON: A Unified Sound Generation and Editing Framework via Deep LLM Fusion</a></div>
    <div class="paper-meta">
      📅 2026-06-02
    </div>
    <details class="paper-abstract">
      We present UNISON, a latent diffusion framework that unifies speech generation, sound generation, and audio editing within a single model. A single model handles text-to-audio, text-to-speech, zero-shot speaker cloning, mixed speech-and-sound generation, scene-level audio editing, speech-in-scene editing, and timed temporal composition, all of which share a single set of weights. Our architecture features two core designs: (1) Layer-wise deep LLM fusion, which injects hidden states from uniformly sampled layers of a frozen MLLM into corresponding MM-DiT blocks via learned projections, providing depth-matched semantic conditioning that improves instruction following over single-layer baselines; and (2) a unified multi-task architecture where task identity is encoded solely by a channel-wise mask and source audio is provided through VAE-encoded channel concatenation. Training is stabilized by an online GPU-side multi-task data synthesis pipeline with task-homogeneous batching and a two-stage curriculum. With 621M--732M trainable parameters, UNISON achieves results competitive with or exceeding task-specialist models across evaluated domains, while being roughly $4\times$ smaller than comparable unified systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.02173v2">The Efficiency vs. Accuracy Trade-off: Optimizing RAG-Enhanced LLM Recommender Systems Using Multi-Head Early Exit</a></div>
    <div class="paper-meta">
      📅 2026-06-02
    </div>
    <details class="paper-abstract">
      The deployment of Large Language Models (LLMs) in recommender systems for predicting Click-Through Rates (CTR) necessitates a delicate balance between computational efficiency and predictive accuracy. This paper presents an optimization framework that combines Retrieval-Augmented Generation (RAG) with an innovative multi-head early exit architecture to concurrently enhance both aspects. By integrating Graph Convolutional Networks (GCNs) as efficient retrieval mechanisms, we are able to significantly reduce data retrieval times while maintaining high model performance. The early exit strategy employed allows for dynamic termination of model inference, utilizing real-time predictive confidence assessments across multiple heads. This not only quickens the responsiveness of LLMs but also upholds or improves their accuracy, making it ideal for real-time application scenarios. Our experiments demonstrate how this architecture effectively decreases computation time without sacrificing the accuracy needed for reliable recommendation delivery, establishing a new standard for efficient, real-time LLM deployment in commercial systems.
    </details>
</div>
