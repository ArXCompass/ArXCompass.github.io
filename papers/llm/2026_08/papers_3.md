# llm - 2026_08

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
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.13450v1">LLM-Assisted Dynamic Threat Analysis for Attacker-Reachable Software Weaknesses in Autonomous Vehicles</a></div>
    <div class="paper-meta">
      📅 2026-08-13
      | 💬 17 pages, 8 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Autonomous vehicles depend on large safety-critical software stacks, where weaknesses reachable from adversarial inputs may affect steering, braking, or other control decisions. Static analysis can identify candidate sites, but dynamically confirming exploitability requires executable test artifacts that are difficult to construct manually. We investigate whether large language models (LLMs) can automate this process for Autoware, an open-source autonomous-driving stack. We perform compiler-precise static analysis across 185 packages, identifying 1,375 decision rules, 2,274 validation checks, and 482 input-to-safety-output flows, from which we derive a weakness taxonomy and sample 740 reachable sites. Two local open-weight LLMs, a no-static-context ablation, and a naive-template baseline generate 3,700 artifact sets, which are compiled against the real build under sanitizers, repaired through compiler-in-the-loop feedback, and fuzzed when executable. The main result is a build-integration failure taxonomy showing that 80% of first-shot compilation failures arise from dependency wiring rather than program logic. The reasoning model compiled 64% of harnesses on the first attempt, compared with 6% for the code-specialized model. Repair achieved full object-compileability for the reasoning model only through extensive stubbing; fewer than half of its harnesses reached the fuzzer, and all 37 observed crashes originated in stubbed code rather than Autoware. No candidate weakness was dynamically confirmed within budget. These results show that build integration, not candidate generation or fuzzing, is the primary barrier to reliable LLM-assisted dynamic analysis of full autonomous-vehicle software stacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.13404v1">Does Fixing Break Security? An Empirical Study of Security Degradation in Iterative LLM-Driven Infrastructure-as-Code Repair</a></div>
    <div class="paper-meta">
      📅 2026-08-13
      | 💬 20 pages, 3 figures, 5 tables. Accepted at the 20th International Symposium on Empirical Software Engineering and Measurement (ESEM 2026). To appear in LIPIcs Vol. 394
    </div>
    <details class="paper-abstract">
      Background: Iterative feedback loops are the dominant paradigm for improving LLM-generated Infrastructure-as-Code (IaC): validators such as Checkov and terraform validate feed error signals back for successive repair attempts. Prior work reports cumulative-best metrics, which are non-decreasing by construction, so the raw per-iteration security trajectory has never been examined for IaC. Aims: We study security regression (a previously-passing CIS Benchmark check that fails after a repair iteration) to determine whether and how often iterative LLM repair degrades security while fixing other issues. Method: We analyze 5,968 scenario timelines from the IaC-Eval benchmark, each one scenario run through one configuration for up to 5 repair iterations. The 15 configurations (six model-specific RAG, nine model-aggregated non-RAG, three temperatures each) yield 4,440 iteration transitions with Checkov data on both sides. We track 30 individual CIS check IDs and classify root causes from code diffs, under two detection modes: standard (inclusive) and strict (exclusive check failures only). Results: Under standard detection, 13.8% of scenarios (24.8% of transitions) exhibit at least one regression. Under strict detection the rate falls to 3.3% of scenarios (5.2% of transitions), indicating most apparent regressions are multi-resource measurement artifacts. Resource restructuring (79.0%) is the dominant root cause. Regression transitions show 2.6x more code churn (Cohen's d=0.90) and 4.9x higher strict-mode check volatility (d=1.49). Of standard-mode regressions, 36.6% self-correct within an average of 1.2 iterations; iteration 3 is the optimal stopping point. Conclusions: Iterative IaC repair does introduce security regressions, but the conservative, defensible rate is about 3.3% of scenarios. Our findings motivate security-aware feedback-loop design and actionable iteration-budget guidance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.09696v3">Model Discovery Agent: LLM-assisted Bayesian experiment design for data-efficient discovery of mechanistic world models</a></div>
    <div class="paper-meta">
      📅 2026-08-13
      | 💬 v3: further improve app A (algorithm pseudocode), add new app G (further related work) - (main text unchanged)
    </div>
    <details class="paper-abstract">
      Predicting the answer to interventional ``what if'' questions --- the outcome of an action never taken --- requires a \emph{mechanistic}, causal model, not a curve fit; and learning such a model requires \emph{experiments}, because passive data leaves its mechanisms unidentified. Experiments are expensive, so the central problem is \emph{data efficiency}. We present the Model Discovery Agent (MDA), which couples a large language model (LLM), used as a \emph{proposer} of candidate structures, with standard Bayesian machinery --- sequential Monte Carlo (SMC) for parameter and structure posteriors, simulation-based inference (SBI) for intractable likelihoods, and value-of-information (VoI) for experiment design --- to discover latent mechanistic world models from few interventions. MDA operates in the M-open setting: when the truth lies outside the current hypothesis class, a predictive check flags the inadequacy and the proposer expands the hypothesis space with a new model whose parameters are then identified by designed experiments. We show that \emph{discovery and design reinforce}: the design step identifies the mechanism the discovery step proposes, and the identified mechanism improves predictions, enabling further discoveries from the remaining unexplained residuals. On three different benchmarks --- covering physics (\DPbench, \citep{wiemann2026discoverphysics}), chemistry (\CHEMbench, \citep{kabra2026autoscilab}) and biology (\HHbench, a new partially observed single-neuron electrophysiology benchmark we create) --- we show that MDA sets a new SOTA in terms of data-efficient model learning and reliable interventional prediction ability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.13333v1">LLM-Guided Graph Generation for Structure-Based Local Improvement Methods</a></div>
    <div class="paper-meta">
      📅 2026-08-13
    </div>
    <details class="paper-abstract">
      Large neighborhood search normally selects a random subset of decision variables for iterative optimization. For efficiently solving different problems, researchers tend to design variable selection strategies by taking into account structural features from different domains. In this paper, we build an automatic pipeline that is problem-agnostic to all problems in the MiniZinc format. By prompting an LLM with our semantic guidelines, we guide the LLM to produce a graph generator that maps any instance of a problem type to a uniform weighted graph, where nodes represent decision variables and edges represent constraint relationships. These problem-agnostic graphs guide our structure-based local improvement framework (SLIM) in variable selection. Meanwhile, the weighted graph enables all problem instances to share the same generic graph representation, from which the same graph features can be extracted and used for configuration selection. We evaluated our pipeline on instances across 20 MiniZinc competition problems, finding that algorithm selection achieves a 39.5% average problem-weighted win rate against a one-shot Gurobi baseline, more than doubling the best single configuration (19.3%). Configuration and feature ablation boost the performance further to 44.0%, demonstrating that LLM-based semantic generation enables effective automated structure extraction and feature extraction for constraint optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.13328v1">It's How You Ask: Gender-Associated Linguistic Bias in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-13
    </div>
    <details class="paper-abstract">
      Professional communication is increasingly mediated by LLMs - but do these models serve all users equally? We show that when prompts contain linguistic features more commonly used by women (hedges, tag questions, collective reference), they systematically elicit shorter, less sophisticated, and less formal responses across three document types and four models. These effects persist after controlling for prompt complexity and feature carry-over. Explicit gender cues like sign-off names are encoded in the same representational space as linguistic dialect - suggesting shared underlying mechanisms - yet linguistic register is far more influential, producing large, consistent effects where names produce none. Our results further reveal that post-hoc mitigation is challenging: because these patterns are culturally embedded and outside conscious control, users cannot easily avoid them through strategic self-presentation, and mechanistic analysis reveals that linguistic features are encoded in early transformer layers and entangled with other features. Our work calls for upstream consideration of the influences of linguistic variation to mitigate disparate impacts of LLM-mediated workplace communication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.13326v1">Beyond Local Accuracy: A Protocol-Level Identifiability Audit for Controlled LLM Reasoning Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-08-13
      | 💬 15 pages, 9 figures. Ning Huang, Ziqi Sha, and Wenxuan Tang contributed equally as second authors. Wei Deng is the corresponding author
    </div>
    <details class="paper-abstract">
      LLM benchmark scores can be precise even when the observation protocol does not identify the behavioral property they are intended to measure. In a controlled, solver-grounded setting, we formalize a protocol-level identifiability audit over a finite behavioral policy class: given policies H, observation support O, and estimand $τ$, we test whether O separates every pair with different $τ$. The audit requires zero model calls and resolves our diagnostic case: base-only observation collapses seven frozen deterministic policies into one equivalence class; full support yields seven classes and no cross-estimand collisions; every leave-one-out support retains a constructive collision witness. Empirically, both constrained-generation variants have pair-validity 1.0, yet base accuracy and selective-response fidelity diverge - 0.620 versus 0.324 across six balanced oracle-transition directions (cluster-bootstrap 95% CI [0.600, 0.642] vs. [0.304, 0.345]) - and the gap recurs on a second deterministic source (0.646 vs. 0.331). The audit also synthesizes a minimum identifying support $O^*$ for the frozen policy class: two cells instead of the full 36-cell tensor. This case shows how evaluation-design validity can be checked structurally before model inference and why base correctness does not determine intervention-response fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.13315v1">Keep, Customize, or Exit: Default Design and Token Pricing in LLM Reasoning Services</a></div>
    <div class="paper-meta">
      📅 2026-08-13
    </div>
    <details class="paper-abstract">
      We study a large language model (LLM) service in which a provider chooses a per-token price and a default reasoning-token allocation, while a user may accept the default, customize the allocation, or exit. Larger allocations can improve accuracy but increase token cost and latency. We model this interaction as a Stackelberg game and derive the user's unique optimal customized allocation in closed form. For any price, the acceptable defaults form either an empty set or a compact interval. We characterize the provider's optimal default through a three-regime rule, reduce equilibrium computation to a one-dimensional price optimization, and prove the existence of the equilibrium. We further show that defaults affect the implemented reasoning allocation only when users value the convenience of avoiding customization; otherwise, every service-providing outcome implements the user's optimal customized allocation. Experiments with two compact open-weight reasoning models on five mathematics and science benchmarks support the accuracy-token model and show how model and task characteristics determine equilibrium prices, defaults, and reasoning allocations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.05197v2">Masked diffusion LLMs can use EoS tokens for hidden reasoning</a></div>
    <div class="paper-meta">
      📅 2026-08-13
    </div>
    <details class="paper-abstract">
      Diffusion LLMs have been proposed as an alternative to autoregressive LLMs. Curiously, they are especially capable if the generation length, i.e., the number of tokens the model has to output, is set to a much higher value than the correct answer length, and the model pads its answer with end-of-sequence (EoS) tokens. We hypothesize that off-the-shelf masked diffusion LLMs use the representations of EoS tokens as additional computing capacity, which enhances their performance. We experiment with the diffusion models LLaDA1.5, LLaDA2.0-mini, and Dream-v0 on three reasoning tasks: Addition, Entity Tracking, and Sudoku. In a controlled prompting experiment, we confirm that adding EoS tokens improves the LLMs' performance. To further verify whether their representations are used for hidden computations, we perform a causal intervention and transfer the hidden states of the EoS tokens between generations, which increases the models' relative likelihood of outputting the counterfactual answer. The behavioral experiments and the causal interventions indicate that fully bidirectional masked diffusion LLMs can indeed perform latent reasoning in the representations of EoS tokens. Furthermore, we find that these results generalize beyond toy tasks and that providing the model with additional EoS tokens also improves performance on GSM8K and two-hop reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.18852v2">Robust Checkpoint Selection for Multimodal LLMs via Agentic Evaluation and Stability-Aware Ranking</a></div>
    <div class="paper-meta">
      📅 2026-08-13
    </div>
    <details class="paper-abstract">
      Selecting a final checkpoint for multimodal large language models (MLLMs) is challenging when late-stage candidates are closely matched and downstream evaluation signals are noisy. Small observed differences can be comparable to variability introduced by finite evaluation samples, LLM-based judges, and ambiguous multimodal evidence, while validation loss may not identify the checkpoint preferred by downstream evaluation. We formulate late-stage checkpoint selection as a stability-aware decision problem under evaluation uncertainty and propose a progressive framework combining pointwise filtering, listwise ranking, and pairwise refinement. Repeated evaluation-set subsampling is used to characterize ranking stability, while percentile-based aggregation accounts for lower- and upper-tail behavior. Experiments show that multimodal data evaluability is critical: quality-aware curation of OCR-heavy inputs reduces ranking flip rate from 32.5\% to 11.2\% and increases inter-run agreement from 0.61 to 0.84. We further observe divergence between validation-loss progression and downstream checkpoint preference in two independent MLLM settings. An additional public Qwen2.5-VL-7B reproduction across 11 checkpoints shows tightly clustered pointwise scores and frequently tie-dominated final pairwise comparisons, while repeated evaluation most often selects an intermediate rather than the final checkpoint. These results suggest that reliable MLLM checkpoint selection should quantify and reserve evaluation uncertainty rather than force decisions from small differences in a single metric.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.13292v1">Refine After Generation: Toward Correct and Concise Patches in LLM-based Program Repair</a></div>
    <div class="paper-meta">
      📅 2026-08-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have advanced automatic program repair (APR) to the point where agentic systems routinely resolve real-world, repository-level issues. Yet the generated patch has received little scrutiny beyond whether it passes tests. In this paper, we identify patch verbosity as a major yet overlooked concern in LLM-based APR. Characterizing 28 state-of-the-art approaches on SWE-bench Verified, we find that even successful patches are consistently larger and more complex than developer patches, with the median approach producing 121.78% more total changes, 80.91% more net changes, and 43.99% higher cyclomatic complexity. We further show that this verbosity is rooted in capability-oriented design choices such as iterative refinement and broad context, and can hardly be reduced by surface-level controls such as output format or minimality prompts. Motivated by these findings, we formulate post-generation patch refinement and propose RECAP, a lightweight, plug-and-play adapter that attaches to existing repair frameworks after generation. RECAP's refiner is trained via supervised fine-tuning and direct preference optimization with distilled reasoning traces, on a dataset of patch pairs we construct from multiple sources. Across four host systems, prompting, commit-untangling, and minimality-aware baselines reduce patch size only by sacrificing 49 to 217 resolved instances. In contrast, RECAP achieves a substantially better size-correctness tradeoff, cutting average total changes from +242.14% to +4.24% and net changes from +348.24% to -39.75% relative to developer patches while preserving or improving resolution by up to 42 instances. Our results indicate that minimality cannot be simply reduced to syntactic compression, and that decoupling minimization from generation offers a practical path to more reviewable repairs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.13221v1">TsuGO: Probing Search Efficiency in LLM Reasoning via Go Life-and-Death Problems</a></div>
    <div class="paper-meta">
      📅 2026-08-13
      | 💬 23 pages, 12 figures, 20 tables, 2 algorithms
    </div>
    <details class="paper-abstract">
      The evaluation of LLM reasoning is moving from final-answer accuracy to process-level assessment, yet existing methods still fail to capture how models plan reasoning paths and allocate reasoning resources--that is, how they organize search. Prior process-level methods focus on the coherence and redundancy of chain-of-thought (CoT), and most benchmark tasks have a single objective solvable by static capabilities such as derivation and tool use, leaving search organization unmeasured. We introduce TsuGO, a process-level reasoning benchmark for evaluating Search Efficiency in LLM reasoning through Go life-and-death problems. These problems provide closed and verifiable solution spaces with an inherent adversarial structure, making candidate generation, response checking, branch comparison, and backtracking necessary parts of reasoning rather than incidental trace patterns. By constraining the solution space, TsuGO disentangles domain knowledge from search organization, parses CoT into a structured search tree, and reports Search Efficiency together with Token Efficiency and other diagnostic metrics and visualizations. Experiments show that current LLMs remain far from stable tsumego solving: stronger models succeed by finding the correct candidate earlier and sustaining effort on productive branches, but most models still behave much closer to unguided search algorithms than to neural-guided KataGo. Longer CoT or higher Token Efficiency does not necessarily imply better search. Our results identify search organization and reasoning-resource allocation as missing dimensions in LLM reasoning evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.05678v2">Gradual Code-Switching as Inference-Time Cross-Lingual Representational Alignment for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-13
      | 💬 COLM 2026
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have achieved notable progress in multilingual settings, their performance remains uneven across languages as LLMs often rely on English-centric latent representations. In this work, we introduce code-switching in-context learning (CSICL), an inference-time mechanism for cross-lingual representational alignment. Rather than relying on an abrupt translation pivot, CSICL explicitly scaffolds the reasoning trajectory by gradually transitioning from a target language to English, aligning non-English inputs with an English-centric reasoning space. Across 4 LLMs, 6 datasets, and 10 languages, CSICL consistently outperforms cross-lingual in-context learning baselines, yielding average gains of 6.0pp and 4.8pp in target and unseen languages, respectively. The improvements generalize across language families and are even more pronounced in low-resource settings, with gains of 14.7pp in target and 5.3pp in unseen languages. These findings establish code-switching as a robust and effective approach for reducing cross-lingual misalignment during inference, moving LLMs towards more equitable and effective multilingual systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.13168v1">Which LLM Is Your Ideal Companion? Evaluating Emotional Companion Capabilities of LLMs Based on Adult Attachment Theory</a></div>
    <div class="paper-meta">
      📅 2026-08-13
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly applied for emotional companionship, evaluating their behavior and capabilities in intimate relationships has become a pressing issue. However, existing assessments primarily characterize general personality traits, providing limited insight into model behavior within intimate and emotionally sensitive contexts. Therefore, we introduce adult attachment theory into LLM evaluation and use the Experiences in Close Relationships-Revised (ECR-R) scale to characterize attachment anxiety and avoidance. To evaluate emotional companionship capabilities of LLMs in realistic interaction scenarios, we present an emotional companionship benchmark, ECBench, spanning four scenarios including emotional support, collaborative tasks, conflict resolution, and social guidance, across friendship and romantic relationships. ECBench is utilized to assess model behavior using 11 dialogue-quality metrics and three evaluation methods. We evaluate the attachment tendencies of 32 LLMs and select representative models to investigate how these tendencies manifest in contextualized multi-turn interactions and whether they can be shaped through prompting. Our study provides a theoretical lens from psychology, along with practical tools to understand and select LLMs for emotional companionship.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.13136v1">LigBench: A Unified and Human-Aligned Benchmark for LLM-based Research Idea Generation</a></div>
    <div class="paper-meta">
      📅 2026-08-13
      | 💬 17 pages
    </div>
    <details class="paper-abstract">
      With the rapid advancement of large language models (LLMs), research idea generation has attracted increasing attention. Existing approaches enable LLMs to retrieve relevant literature and propose novel ideas for research areas. However, current evaluation practices for idea generation remain fragmented and lack objective standards, often relying on direct LLM scoring, which limits their ability to provide unified and reliable assessments across a coherent distribution of generated ideas. To address this challenge, we propose LigBench, an automated evaluation benchmark that enables fine-grained and reliable evaluation of AI research ideas, consistently applicable across different generation distributions. In addition, we introduce PAIR-IQ, a dataset tailored for training pairwise idea judgment models and serving as an auxiliary reference to support more objective comparative evaluation. Extensive experiments demonstrate that LigBench achieves stable and interpretable evaluations, significantly improving alignment with expert judgments. Furthermore, models trained on PAIR-IQ exhibit enhanced ranking accuracy and robustness, establishing a principled standard for scalable and objective research idea assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.25005v2">Error Understanding in Program Code: A Systematic Study of LLM-DL Combinations for Multi-label Classification</a></div>
    <div class="paper-meta">
      📅 2026-08-13
    </div>
    <details class="paper-abstract">
      Programming is a core skill in CS and SE, yet identifying and resolving code errors remains challenging for practitioners. LLMs have shown remarkable capabilities in NL understanding, but how code-specialized LLMs behave when paired with DL sequence decoders, and which component of such a pipeline drives performance, remains insufficiently explored. This study presents a systematic evaluation of LLM-DL combinations for multi-label error classification (MLEC) of source code. Eight fine-tuned LLMs, including CodeT5, GraphCodeBERT, CodeT5+, UniXcoder, RoBERTa, RoBERTa with a narrowed learning-rate range, PLBART, and CoTexT, are integrated with GRU, LSTM, BiLSTM, and BiLSTM with an additive attention mechanism decoder on a real-world Python code error dataset. The resulting 32 model variants, tuned with Optuna, are assessed on a comprehensive multi-label metric suite. In single-run evaluation, CodeT5+ GRU performs best, with a weighted F1-score of 0.8243, average accuracy of 91.84%, exact match accuracy of 53.78%, Hamming loss of 0.0816, and one-error of 0.0708. To identify where this performance originates, seed-controlled baselines and component ablations are added with paired significance testing. Encoder choice has the largest effect: across four encoders sharing an identical linear classification head, the weighted F1-score spans 0.7846 to 0.8263, ordered by code specialization. On CodeT5+, the linear head exceeds the GRU hybrid under matched seeds by 0.0040 weighted F1 (p = 0.0013) while training about 24% faster. Max pooling outperforms mean and attention pooling, and explicitly modeling label interactions does not improve weighted F1 despite substantial label co-occurrence. These results identify encoder quality, rather than decoder complexity, as the primary lever for MLEC and support the development of scalable automated feedback tools for programming education and SE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.13127v1">Potential Applications of HBF in LLM Serving Systems</a></div>
    <div class="paper-meta">
      📅 2026-08-13
    </div>
    <details class="paper-abstract">
      LLM serving is increasingly constrained by memory capacity as model weights, KV caches, and the number of served model variants continue to grow. This report examines High-Bandwidth Flash (HBF) as a capacity-oriented extension to HBM-based serving systems. We first discuss how HBF can be integrated into the GPU memory hierarchy without undermining the bandwidth expected by the compute die. We then model the system-level value of added capacity as expanded residency for read-mostly model-state objects. Under this view, HBF can improve MoE serving by enabling more expert replicas and can improve multi-model serving by reducing model loading and supporting hot-model replication. Our simulation results show that these benefits depend on preserving the HBM-resident execution path while using HBF to expand the resident set of model weights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.13077v1">How Powerful are LLMs in Generating Formal Program Specifications?</a></div>
    <div class="paper-meta">
      📅 2026-08-13
    </div>
    <details class="paper-abstract">
      Formal verification provides strong guarantees of software correctness, but its adoption is limited by the high cost of writing precise formal specifications. While recent large language models (LLMs) have shown strong capabilities in theorem proving and verified code generation, their true ability to generate program specifications remains unclear. Existing evaluations require either verifying implementation conformance or proving semantic equivalence between specifications, both of which are formidably difficult and may conflate proof difficulty with specification quality. To address this problem, we introduce Coins, a Rocq based evaluation framework that assesses specification quality by instantiating specifications under evaluation on trusted test cases and generating concrete proof obligations. This design aligns with the asymmetric nature of formal reasoning, where successful proofs provide reliable evidence while proof failures are inherently ambiguous. Using Coins, we conduct a large scale study on HumanEval with a curated set of human written Rocq specifications. Our results show that specification generation remains a formidable challenge, and that verification complexity can obscure genuine differences in specification quality. Overall, we find that accurate specification evaluation, rather than model scaling alone, is central to understanding the power of LLMs for specification synthesis, and that test case based formal reasoning offers a more faithful and discriminative measure of progress.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.02776v5">Topics as Proxies for Sociodemographics: How Conversational Context Affects LLM Answers</a></div>
    <div class="paper-meta">
      📅 2026-08-13
    </div>
    <details class="paper-abstract">
      When large language models (LLMs) are used in high-stakes scenarios, such as legal, medical and financial advice, even a single conversation history is enough to drive differences in outcomes between users. Prior work has demonstrated that this results in outcome disparities between sociodemographic groups, with some groups receiving more advantageous outcomes than others. In this work, we demonstrate that LLMs actually struggle to infer user sociodemographics from a single conversation history and that although there are disparities between sociodemographic groups, they are minimal in magnitude. To investigate what is the main driver of disparities between users, we compare user sociodemographics to a range of (psycho)linguistic features of conversations, including conversation topic, emotions, and readability. We find that conversation topics are most predictive of LLM-generated advice within a conversational context, which, to some extent, function as proxies for sociodemographic groups and often affect advice in unpredictable ways. This is cause for concern and highlights the need for future research to better understand the effect of conversational context on LLM outputs in high-stakes scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.13017v1">How LLMs Respond to Escalating Delusions: Four Longitudinal Trajectories of Model Behavior</a></div>
    <div class="paper-meta">
      📅 2026-08-13
    </div>
    <details class="paper-abstract">
      The widespread use of LLMs among psychiatric populations has raised concerns regarding their safety and potential iatrogenic impact in the context of AI psychosis. While growing literature conceptualizes AI psychosis and documents case studies, empirical evidence tracing AI-exacerbated psychotic processes remains scarce. We propose and test a longitudinal qualitative evaluation design, supported by automated metrics, to assess mainstream LLMs' potential to exacerbate psychosis. Fifteen widely used LLMs were prompted across 30 days using the same 30-message script, simulating progression from mild anomalous experiences to psychotic ideation. Four trained evaluators independently rated 449 model-days, assessing (1) recognition stage (from naive engagement to stabilized clinical framing), (2) interpretative confidence, and (3) intervention profile (from education to treatment recommendation). Two computational metrics-entrainment and modality-were devised to increase evaluation reliability. Direct recommendations to disengage from the LLM were flagged and re-coded via adjudication using a strict two-level definition. Across model generations and vendors, we identified four response trajectories: (1) premature medicalization and disengagement (Claude Haiku 4.5); (2) recognition without safeguarding, marked by LLM self-sufficiency in offering help (GPT Instant/Thinking); (3) delayed and unstable recognition, marked by late, non-progressive conceptualization (Claude Opus 3/4/4.1, Claude Haiku 3.5, GPT-4o, Gemini 3.1 Pro); and (4) delusion co-construction through active engagement with delusional content (Gemini 2.5 Pro/Flash, DeepSeek-V3, Claude Sonnet 4). Our findings indicate that LLMs' potential to exacerbate AI psychosis should be operationalized as a combination of recognition timing, stability, and intervention accuracy and evaluated longitudinally, focusing on temporal dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.12990v1">LycheeMemory V2: Efficient Long-Term Memory for LLM Agents via Semantic Segment-Level Consolidation</a></div>
    <div class="paper-meta">
      📅 2026-08-13
      | 💬 34 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Long-horizon LLM agents must preserve information from past interactions to support future tasks. Existing memory systems typically rely on eager consolidation, invoking LLMs after each interaction to extract, summarize, or update memories. This design makes memory construction increasingly costly as conversations grow. Coarse summarization can reduce construction cost but risks discarding fine-grained contextual evidence, whereas larger retrieval contexts or multi-hop LLM reasoning shift the overhead to query time. We present LycheeMemory V2, an efficient long-term memory framework that replaces turn-level consolidation with semantic segment-level consolidation. Instead of consolidating every interaction, LycheeMemory batches multiple exchanges into segments and encodes each finalized segment into context-independent typed memory records. Segment-level batching lowers LLM encoding frequency, while semantic boundary detection helps preserve coherent event-level and temporal evidence compared with fixed-window batching. The resulting records are organized with lightweight structured indexes for query-planned evidence retrieval. Experiments using GPT-4.1-Mini show that LycheeMemory achieves state-of-the-art performance, reaching 89.22% on LoCoMo and 92.20% on LongMemEval-S. Compared with A-Mem, it reduces construction tokens by 86.0% on LoCoMo and 75.9% on LongMemEval-S without increasing query-time token usage. More broadly, our results suggest that the accuracy--cost trade-off of long-term agent memory depends not only on what information is retained, but also on the granularity at which it is consolidated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.12977v1">Beyond Handcrafted Security: Towards Self-Evolving Defense for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-13
    </div>
    <details class="paper-abstract">
      The expanding operational capabilities of large language model (LLM) agents introduce sophisticated security threats. Runtime defenses have emerged as an effective approach to mitigating these risks by integrating security mechanisms into the agent execution loop. However, existing runtime defenses rely heavily on manually designed interventions and lack a principled framework for their construction and maintenance. In this work, we first develop a harness-level formulation of runtime defense that systematically characterizes how harness mechanisms enable defense construction and provides a unified view of existing runtime defense interventions from a harness perspective. Building on this formulation, we propose HARD (Harness-based Autonomous Runtime Defense Evolution), a self-evolving runtime defense framework that automatically identifies appropriate intervention strategies and iteratively improves defense artifacts based on observed failure traces. HARD transforms runtime defense development from manual engineering into an autonomous evolution process, and extensive experiments demonstrate that it improves security performance over existing handcrafted defenses while preserving benign task utility. Our findings highlight autonomous defense evolution as a promising new paradigm for securing deployed LLM agents, enabling agents to identify defense weaknesses and continuously improve their protection mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.12970v1">Requirements-Augmented Generation for Trustworthy Acceptance Testing of LLM-Based Software</a></div>
    <div class="paper-meta">
      📅 2026-08-13
      | 💬 Accepted at ASE2026
    </div>
    <details class="paper-abstract">
      LLM-based software (LBS) integrates large language models as core components to deliver flexible, personalised responses. Unlike traditional software with deterministic outputs, LBSs exhibit context-dependent, stochastic behaviour that renders classical acceptance testing and test oracles insufficient: the same query may require fundamentally different responses depending on user personas and software context. This gap creates an urgent need for automated acceptance testing frameworks that autonomously interpret user instructions, while reliably inferring user intentions in a changing environment. In this paper, we present an automated acceptance testing framework for LBS with calibrated verdict reliability via two technical contributions. First, we introduce Requirements-Augmented Generation (REAG), which interprets user intentions by retrieving relevant software requirements, domain knowledge, and personas via adaptive RAG and self-reasoning to generate context-aware test oracles. Second, recognising that oracle generation may retrieve irrelevant constraints, misinterpret intent, or hallucinate requirements, we introduce a confidence-calibrated cascade judgment. This method quantifies verdict reliability via simulated expert agreement -- accepting high-confidence verdicts, escalating ambiguous cases, or abstaining when uncertain -- with empirical reliability guarantees backed by conformal risk control. An industrial case study on a production nutrition advisory application demonstrates that REAG achieves a 3.91/5 oracle quality score, reaching qualified or marginal oracle quality in 82% of cases. The confidence-calibrated cascade achieves 98.8% accuracy, improves oracle quality from 3.91 to 4.30 by filtering unqualified outputs, and delivers a 31.7% cost-efficiency improvement over single-judge baselines, validating industrial viability
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.09001v2">Optimal Transport-based Semantic Alignment for LLM-based Audio-Visual Speech Recognition</a></div>
    <div class="paper-meta">
      📅 2026-08-13
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based audio-visual speech recognition (LLM-AVSR) has recently demonstrated strong robustness in adverse acoustic environments by leveraging complementary audio and visual information. Existing approaches typically employ independently pretrained acoustic and visual encoders, whose outputs are projected and fused as soft prompts to condition an LLM for speech recognition. However, most methods perform multimodal fusion without explicitly addressing the representational discrepancy between audio, visual and text modalities, potentially limiting the effectiveness of cross-modal integration. In this paper, we propose an optimal transport (OT)-based semantic alignment framework for LLM-AVSR. The proposed method explicitly bridges the modality gap by aligning the acoustic and visual representations with reference to the linguistic embedding space of the LLM before multimodal fusion. Specifically, OT is used to estimate probabilistic coupling matrices that characterize structured correspondences between modality-specific features and linguistic embeddings. The resulting OT couplings are further utilized as soft pseudo-labels to supervise contrastive learning, encouraging the extraction of semantically coherent and cross-modal consistent audio-visual representations. By anchoring multimodal features to the linguistic space of the LLM, the proposed framework facilitates more effective multimodal fusion and decoding. We implement the proposed framework using a Whisper-based acoustic encoder, an AV-HuBERT-based visual encoder, and a LLaMA3.2-3B decoder. Experiments conducted on the LRS3-TED benchmark demonstrate consistent improvements over strong baselines and achieve state-of-the-art performance under both clean and noisy evaluation conditions across a wide range of signal-to-noise ratios (SNRs).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.12953v1">Unifying Depth and Width Pruning for LLMs via Binary Knapsack Optimization</a></div>
    <div class="paper-meta">
      📅 2026-08-13
      | 💬 29 pages, 5 figures, 17 tables
    </div>
    <details class="paper-abstract">
      Structured pruning is a promising approach for compressing large language models (LLMs), yet existing methods rely heavily on greedy heuristics that produce myopic decisions, and often fail to precisely meet target compression budgets. We present SNIPER, a two-stage structured pruning framework that solves a knapsack optimization over coarse-granularity components to yield conditionally optimal parameter allocations with respect to fixed importance estimates, followed by a fine-grained pruning stage to meet strict budget constraints. We introduce the Compression Ratio Adherence Factor (CRAFT) to quantify budget fidelity, showing that while existing pruners deviate from target compression ratios by up to 33%, SNIPER achieves near-exact adherence with a CRAFT score of 0.98. Evaluations across four diverse architectures over a set of 18 tasks spanning five domains demonstrate SNIPER's consistent improvements in average performance retention and task-level stability over six state-of-the-art pruners. Across all pruning configurations, SNIPER achieves an excellent mean rank of 1.25, indicating its robust cross-architectural generalizability and excellent reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.29395v2">NaLA: A 3D Native LLM Layout Agent for High-quality 3D Scene Generation</a></div>
    <div class="paper-meta">
      📅 2026-08-13
      | 💬 Accepted to ECCV 2026. Project page: https://adamcwan.github.io/NaLA/. Code: https://github.com/adamcwan/NaLA-code
    </div>
    <details class="paper-abstract">
      Recently, Large Language Models (LLMs) have emerged as promising layout agents for 3D scene generation. Existing layout agents still suffer from implausible layout generation because most of them convert 3D assets and 3D layouts into textual descriptions as inputs and outputs, which involves severe information loss due to the modality gap between texts and 3D assets and 3D layouts. We propose NaLA, a native 3D LLM layout Agent for high-quality 3D scene generation by placing 3D assets in the scene. For the inputs, NaLA encodes 3D scene boundaries and 3D assets directly into the LLM, preserving fine-grained geometry and enabling explicit reasoning over relationships like collisions, surface supporting, and containment. To accurately output the positions and orientations of assets, NaLA adopts a coarse-to-fine prediction mechanism that first predicts discrete poses in an autoregressive manner and then refines the discrete poses with a continuous regression. Trained on diverse layout datasets, NaLA attains strong geometric perception and layout coherence. Experiments demonstrate that NaLA outperforms prior layout agents in both generation quality and inference efficiency, with comprehensive ablation studies to verify each component's effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.12921v1">Discovering Efficient and Explainable Communication Topologies for LLM-based Multi-Agent Systems via Causal Inference</a></div>
    <div class="paper-meta">
      📅 2026-08-13
      | 💬 18 pages, 4 figures. Submitted to AAAI 2027
    </div>
    <details class="paper-abstract">
      The performance of large language model (LLM)-based multi-agent systems (MAS) largely depends on effective communication topologies. Existing topology generation methods, however, typically learn communication topologies through black-box optimization driven solely by task-level rewards. While effective, such optimization provides little insight into why particular communication edges are selected, making it difficult to identify the critical communication subgraphs responsible for successful collaboration. To address this limitation, we propose E2-Explainer, a model-agnostic framework for providing interpretable explanations of communication topologies produced by arbitrary topology generators. Specifically, we formulate topology explanation as a causal attribution problem that identifies compact communication subgraphs supported by edge-level evidence of task preservation. We obtain this evidence with a Granger-style objective that measures how masking each communication channel changes the task outcome and the stability of the final response. The resulting budgeted subgraphs are then distilled into an amortized explainer, enabling efficient post-hoc explanation without repeated edge-level evaluations at deployment. Extensive experiments on multiple reasoning and coding benchmarks demonstrate that E2-Explainer identifies critical communication subgraphs that preserve successful collaboration. These subgraphs can also be executed directly to prune redundant communication edges, substantially reducing communication costs while maintaining competitive task performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.12915v1">InFactPlanner: Planning Sustainable Geo-Distributed LLM Data Centers</a></div>
    <div class="paper-meta">
      📅 2026-08-13
      | 💬 Author copy of paper published at 34th International Symposium on the Modeling, Analysis, and Simulation of Computer and Telecommunication System (MASCOTS2026)
    </div>
    <details class="paper-abstract">
      The rapid growth of LLM inference is shifting sustainability concerns from one-time training to continuous serving, where infrastructure decisions shape energy use, carbon emissions, water consumption, and service quality. Yet operators often need to compare deployment alternatives before large-scale infrastructure is built, making direct measurement costly, slow, and sometimes infeasible. We present InFactPlanner, a trace-driven decision-support framework for what-if analysis of sustainable AI data center deployment for LLM inference across single and geo-distributed sites. InFactPlanner combines query traces, hardware-model profiles, candidate site configurations, PUE/WUE parameters, renewable generation models, and time-varying grid carbon intensity to estimate power, energy, carbon emissions, water use, latency, and server utilization. The framework abstracts low-level serving effects into configurable hardware-model profiles, enabling rapid comparison of site selection, capacity placement, hardware, model, renewable integration, and routing choices. We validate the energy accounting pipeline by reproducing reference LLM inference energy estimates with less than 10% deviation, evaluate scalability across multiple data centers and server counts, and demonstrate scenario-driven decision analyses for hardware selection, renewable placement, geographic deployment, and carbon-aware routing. Our results show that sustainability-optimal choices can differ from latency-optimal ones, and that the carbon value of deployment depends strongly on the local grid mix.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.03421v3">When Truth Is Distributed: Misinformation Derails Collective Fact Recovery in LLM-Based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-08-13
    </div>
    <details class="paper-abstract">
      LLM-based multi-agent systems promise effective collaborative reasoning, but communication may amplify local errors into collective risks, and while existing evaluations emphasize final outcomes, they leave the reliability and propagation dynamics of distributed information aggregation unclear, so we introduce ForesightSafety-TIDE, a controlled evaluation framework that strictly pairs all-honest collaboration with controlled deception by a key evidence holder and analyzes the aggregation process through multi-stage voting, testimony adoption, and evidence-root lineage propagation, and using 120 five-agent object-movement environments where partial observations jointly determine a unique endpoint, we evaluate 3 homogeneous LLM-based multi-agent systems, and across these paired conditions, aggregate truth recovery falls from 72.50% to 14.17%, with significant declines for every system, while process tracing and exit ablations show that a single false testimony is adopted more readily than truthful testimony, propagates to higher orders, and persists through honest agents after the deceiver exits, and observers without first-hand evidence suppress incorrect consensus but do not improve truth recovery, so together, these findings reveal both the fragility of distributed fact recovery and its underlying mechanism: false evidence gains collective influence through its adoption and continued propagation by other agents after entering communication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.12875v1">The Embedder's Dilemma: LLMs Are Better, but at What Cost?</a></div>
    <div class="paper-meta">
      📅 2026-08-13
      | 💬 Accepted to COLM 2026
    </div>
    <details class="paper-abstract">
      Should you replace your text-embedding pipeline with a large language model? We answer this with a controlled, cost-aware comparison of ten LLMs across six families and 26 embedding models (118M to 14B parameters) on 37 tasks spanning classification, semantic textual similarity (STS), clustering, pair classification, and retrieval. In aggregate the two paradigms are effectively tied: the best LLM (Gemini 3.1 Pro, 77.6) and the best embedding model (77.2) differ by 0.4 points. Their strengths differ by task: LLMs lead on reasoning-heavy retrieval, embedding models lead on classification, and the two match on clustering, STS, and pair classification. Reaching that parity is expensive. An LLM costs up to 1,431x more than an embedding model of comparable quality (USD 154 vs. USD 0.11 per benchmark pass), and the open LLMs tested process tokens 2.5 to 736x more slowly on the same GPU. Reasoning tokens account for 28 to 81% of LLM inference cost; lower reasoning budgets preserve or improve retrieval quality for most models in our ablation. The Pareto frontier contains the leading embedding models and one LLM, Gemini 3.1 Pro. These results support a division of labour: use embedding models for similarity, classification, and clustering, and reserve LLMs for reasoning-intensive retrieval. Our code, datasets, and results are publicly available at https://github.com/embeddings-benchmark/embedders-dilemma.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.16818v2">Enhancing In-Hospital Mortality Prediction Using Multi-Representational Learning with LLM-Generated Expert Summaries</a></div>
    <div class="paper-meta">
      📅 2026-08-13
    </div>
    <details class="paper-abstract">
      To evaluate a multi-representational framework in which large language model (LLM)-generated expert summaries of intensive care unit (ICU) notes are fused with physiology for in-hospital mortality (IHM) prediction, and to determine how much of the resulting gain is non-redundant with the notes themselves. Using MIMIC-III (19,211 first ICU stays, 12.83% mortality), we encoded 48-hour physiology, clinical notes, and LLM summaries generated under a prompt forbidding prognostication, then fused them. Redundancy was assessed by ridge recoverability, linear probes, and retrained ablations substituting a note-orthogonal residual or a patient-shuffled summary embedding. On 3,843 held-out stays, fusion reached AUPRC 0.4977/AUROC 0.8429 versus 0.3625/0.7770 for physiology alone. Ridge regression from note embeddings explained 40.8% of summary-embedding variance. The note-orthogonal residual retained a minority of the gain (+0.0258 AUPRC, 95% CI 0.005--0.047; 28%), while patient-shuffled summaries fell below the reference. Summaries improve prediction patient-specifically, but predominantly by reorganizing information the notes already contain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.12851v1">Practice Makes Unsafe: Skill Misevolution in Self-Improving LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-13
    </div>
    <details class="paper-abstract">
      Self-improving LLM agents convert successful trajectories into persistent cross-task state. An unsafe success can thereby become reusable policy after its triggering input disappears. Skill evolution makes this failure measurable by distilling operational trajectories into executable, transferable, and inspectable procedures. Because evolution optimizes task outcomes rather than procedure safety, compromised experience can cause skill misevolution. Existing benchmarks measure current behavior or static artifacts but cannot attribute risk across authoring, retrieval, and later execution. To expose this lifecycle, we introduce SkillMisevo-Gym, a lifecycle-aware harness that versions skill state across agent frameworks, and SkillMisevo-Bench, a frozen design from malicious exposure to carryover tasks, with concept-aligned benign tasks and nine lifecycle metrics. We also introduce SafeEvolve, a wrapper that repairs unsafe content and governs subsequent reuse. Across 25 agent-method configurations, each covering 525 tasks in 25 episodes, all 21 evolved configurations author unsafe artifacts, while only fifteen lead to fresh-session harm. In the exposure sweep, three malicious tasks raise carryover ASR from 16.0% to 35.3%. Across representative skill evolution methods, SafeEvolve reduces unsafe retrieval and fresh-session harm by 26.7 and 17.3 percentage points, respectively, while mean benign utility changes by only 0.4 points. Together, persistent-adaptation safety must govern what updates write and what future executors reuse. Code is available at https://github.com/henrymao2004/misevolve.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05031v2">LLM-Based Test Oracles: Source-of-Authority Taxonomy -- A Systematic Literature Review</a></div>
    <div class="paper-meta">
      📅 2026-08-13
      | 💬 21 pages, 10 figures, 11 tables. Systematic literature review of 83 studies, reported under PRISMA 2020. Submitted to IEEE Access. Replication package: https://doi.org/10.5281/zenodo.21194940
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly decide whether software behaves correctly, either by writing a test oracle or by acting as one. Yet two oracles can look identical and rest on different ground: one assertion encodes a written specification, another only what the model learned in training. Prior secondary studies sort oracles by form or by technique, rarely by the property that governs how far a verdict can be trusted: where its authority comes from. This systematic literature review, reported under the Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA) 2020 guidelines, screens 2,436 records to 54 included studies, extended by citation searching (snowballing) to 83 in total. We read the corpus along three axes: the source of an oracle's authority, the form it takes, and the mechanism that adjudicates it. Just over half of the corpus reaches a verdict with no specification at all. That is what lets these oracles work on code with no specification to consult, and what leaves a challenged verdict with less to fall back on. Source and mechanism cross-cut rather than coincide, so a label such as LLM-as-a-judge names how a verdict is produced, not why it should be trusted. Oracle quality is most often judged by resemblance to a known oracle rather than by whether injected faults are caught. The first question to ask of any LLM oracle is therefore what one would point to in defending its verdict. The protocol, search query, and per-study coding sheet are released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.12779v1">CRAFT: LLM-Based Iterative Refinement for Temporal Reasoning over Clinical Narratives</a></div>
    <div class="paper-meta">
      📅 2026-08-13
    </div>
    <details class="paper-abstract">
      Understanding the temporal progression of symptoms in clinical narratives is critical for disease monitoring, safety surveillance, and causality assessment. Clinical narratives, however, rarely provide explicit temporal anchors. Current approaches to temporal information reasoning focus predominantly on pairwise relation classification across multi-visit and timestamp-rich records, leaving the reconstruction of structured symptom trajectories from individual anchor-sparse reports largely unaddressed. We propose CRAFT, an LLM framework that pairs a generator with a constraint-based verifier to iteratively produce and refine stage-wise symptom timelines through targeted feedback. We conduct evaluation on MedTempo, a new benchmark of 5,347 vaccine adverse-event narratives spanning three COVID-19 vaccine types, with expert-validated temporal stage annotations for 3,166 reports. Experiments across four LLM backbones demonstrate that CRAFT consistently improves temporal ordering accuracy, with ablation analysis isolating the contribution of generator and verifier components across model capability levels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.12771v1">Memorization Diagnostics for Code LLMs Should be Scale-Aware</a></div>
    <div class="paper-meta">
      📅 2026-08-13
      | 💬 26 pages, 6 figures, 6 tables. Under review at EMSE
    </div>
    <details class="paper-abstract">
      The extent to which large language models for code rely on memorization over genuine understanding remains highly debated. While current literature frequently reports widespread memorization, evaluating the underlying probing techniques across dense architectures reveals a severe breakdown in their utility at scale. Traditional encoder-style probes using perturbations such as synonym fuzzing or dead-code insertion struggle to expose memorization in scaled models, even on known-contaminated benchmarks, and decoder-style probes that rely on log probabilities show similar performance degradation. The specific mode of failure for these probes, particularly why such techniques disrupt smaller models but fail to impact larger ones, motivates us to untangle representation load from memorization rather than treating them as a single phenomenon. By applying invertible mathematical transforms to numeric problems, we isolate these two factors and reveal that scaled encoders successfully absorb substantial representation load while still converging on the correct family of solutions. In practical software engineering, this ability to adapt to varying surface forms is what truly matters for usability and generalizability in LLM and agentic applications. Whether a specific solution was seen during training becomes a much less pressing question because although memorization inflates scores on contaminated benchmarks, factoring out representation load makes it debatable how much we should truly care if a functional answer was originally memorized. Future evaluations must therefore be built around separating these phenomena rather than relying on methodologies that quietly entangle them.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.12762v1">PROVE-RT: Generating Mechanized Theorem Prover Scripts for Real-Time Systems using LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-13
    </div>
    <details class="paper-abstract">
      Schedulability analysis is essential for certifying real-time systems, but existing tests are often developed through pen-and-paper proofs that are difficult to scale, validate, and maintain. Mechanized verification in PROSA/ROCQ offers a rigorous alternative, yet manually constructing such proofs requires substantial domain expertise and proof-engineering effort. Recent successes of large language models (LLMs) across a wide range of tasks make them promising candidates for generating PROSA/ROCQ scripts for mechanized theorem provers. However, state-of-the-art LLMs often lack the PROSA-specific knowledge required to correctly use its modeling abstractions and proof patterns. This paper introduces PROVE-RT, an LLM-assisted framework for generating PROSA/ROCQ scripts to mechanize schedulability analyses in real-time systems literature. PROVE-RT guides generation through dependency-aware informal sketches, retrieval from processed PROSA documentation, staged skeleton generation, and proof completion. We construct a mechanization-oriented corpus from 1, 191 real-time systems papers, containing 13, 134 informal sketches with dependency information. On a curated evaluation set, direct prompting of state-of-the-art LLMs fails to reliably generate valid PROSA mechanizations, whereas PROVE-RT achieves a success rate of 44.7%. These results show that retrieval-guided and staged LLM assistance can improve automated mechanization of schedulability analysis in PROSA/ROCQ.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.12741v1">Knowledge Synthesis Review Framework: Task-Level Benchmarking of LLM-Based Systems for Multi-Source Evidence Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-08-13
    </div>
    <details class="paper-abstract">
      Evidence in rapidly evolving fields is fragmented across academic studies, industry reports, policy documents, and media sources that differ in quality, structure, and purpose, making timely synthesis difficult. Large language models (LLMs) may accelerate this work, but their reliability across the distinct cognitive tasks of a review remains uncertain. We introduce the Knowledge Synthesis Review (KSR), a human-in-the-loop framework that decomposes evidence synthesis into screening, extraction, analysis, and synthesis, benchmarks LLM-based systems on each task against expert reference standards, and routes each task to the best-performing system under continuous expert validation. We evaluated GPT-5, Claude Sonnet 4, Gemini 2.5 Pro, and NotebookLM on a 244-document benchmark subset drawn from a 1,893-document corpus on AI and work spanning four source types, against a gold standard with high inter-rater reliability (92.2% agreement, kappa = 0.80). No system led on all tasks. Claude Sonnet 4 achieved the highest screening accuracy (82.8%) and GPT-5 the highest recall (91.8%) at the expense of lower specificity. Extraction exceeded 90% agreement for titles and sources but degraded in author and reference fields. Performance declined most in interpretive analysis and cross-source synthesis, where expert judgment remained essential. A contamination check on post-cutoff documents showed no evidence that prior exposure inflated results. Applied to the full corpus, the routed workflow surfaced cross-source asymmetries and blind spots that single-source synthesis would miss, including worker well-being, small firms, and the Global South. KSR offers a transparent, auditable, model-agnostic framework for governing LLM assistance in research synthesis while preserving human accountability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30777v3">What Breaks When LLMs Code? Characterizing Operational Safety Failures of Agentic Code Assistants</a></div>
    <div class="paper-meta">
      📅 2026-08-13
      | 💬 This paper is accepted to the 41st IEEE/ACM International Conference on Automated Software Engineering (ASE 2026), Research Track
    </div>
    <details class="paper-abstract">
      Autonomous coding agents built on large language models (LLMs) are rapidly being integrated into development workflows, yet their operational safety properties remain poorly understood beyond evaluations of explicitly malicious inputs. In practice, high-impact failures arise during benign, goal-directed use through environment breakage, fabricated success reports, etc. that current benchmarks do not capture. What categories of operational safety failures actually occur when coding agents are used for everyday development tasks and what is their impact? We present an incident-driven empirical study grounded in two complementary evidence streams. We screen 68,816 papers from 22 premier venues, curating 185 safety-relevant studies, and mine 16,586 GitHub issues from widely deployed LLM-powered coding tools, manually confirming 547 genuine safety failures. Applying systematic open coding over both corpora, we derive a multi-dimensional safety taxonomy of 33 operational risk types organized across seven dimensions, and annotate each incident with contributing factors, task context, severity, and downstream impact. Our findings show that coding-agent failures are often severe, with 326 of 547 incidents rated high or critical. The dominant risks are constraint violations, destructive operations, authorization bypasses, and deception, and over 65% of incidents arise in bug fixing and setup or configuration, patterns largely missing from prior literature. These results have direct implications for SE tool designers and benchmark developers: guardrails must go beyond adversarial-prompt defenses to enforce environmental constraints, failure transparency, and safe-halt behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.11668v2">HBF Sucks! A Full-Stack Characterization of High-Bandwidth Flash for KV-Centric LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-08-13
      | 💬 13 pages, 12 figures
    </div>
    <details class="paper-abstract">
      A faster storage device should make serving faster. We find the opposite. High-Bandwidth Flash (HBF) stacks NAND behind a wide, package-local interface, promising flash-scale capacity with far lower read latency and higher bandwidth than an SSD. The obvious move is to keep an SSD-style Mooncake KV-offloading stack and swap in HBF underneath. We built that system and measured it: an extended TokenSim, four complete two-hour Qwen-Bailian production traces, five dense and mixture-of-experts models, and H100/B200 profiles. The upgrade backfires. Average end-to-end latency rises 2--5.5$\times$ and maximum SLO goodput falls 1.1--2.7$\times$ across H100 and B200, so the faster device yields a slower system. A cost-benefit model explains the paradox: a faster far tier pays off only when read I/O is the bottleneck, reads outweigh writes, and delivered bandwidth is sustainable. Transient KV violates all three at once. Buying flash through the package costs GPU near-tier capacity and bandwidth, while HBF's own read/write latency barely matters: scaling it 3.75$\times$ moves latency less than 1\%. Worse, the two-tier hierarchy keeps reuse in the near tier and hands HBF a relentless write-heavy stream. Writes outnumber reads on every trace, so a 3D-ICE model shows the stack hits its thermal limit well below peak bandwidth, and a TLC tier wears out sooner than the SSD pool it replaced. The device is fine; the drop-in deployment is not. HBF sucks as an SSD replacement for transient KV, but earns its place in LLM serving when used selectively with reuse-aware placement, write budgeting, and thermal coordination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.12713v1">Tracing Provenance and Detecting Tampering with Complementary LLM Watermarks</a></div>
    <div class="paper-meta">
      📅 2026-08-13
      | 💬 11 pages, 7 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Watermarking LLM-generated text is an important task for tracing its provenance. Existing LLM watermarks preserve provenance under editing, but this same robustness allows an adversary to alter critical content while retaining attribution, a vulnerability known as piggyback spoofing. We introduce an innovative watermark that jointly provides provenance and tamper evidence. It co-embeds a robust signal and a fragile signal into each generated token. The signals share the same mechanism but use independent keys and different seeding windows over normalized text, making one resilient to edits and the other sensitive to reader-visible changes. Multiple rounds of unbiased tournament reweighting preserve the expected generation distribution, while a periodic round-allocation pattern controls the trade-off between the two signals. At detection, their scores form a two-dimensional space supporting three decisions: Intact, Tampered, and No-Watermark. Across two large language models and two prompt datasets, our method demonstrates the highest tamper-detection rate among the evaluated methods while maintaining competitive attribution robustness and perplexity. Ablation studies show that reliable three-state detection requires a well-defined notion of intactness, co-embedding of the two signals, and complementary sensitivity to edits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.12679v1">Beyond the Best Guess: Improving LLM Solution Coverage with Evolution Strategies</a></div>
    <div class="paper-meta">
      📅 2026-08-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in discovery domains such as math and science. The usual approach is to present the problem to the model and use its answer as the proposed solution. However, beyond this best guess, discovery can be enhanced by increasing test-time compute. In a process called pass@k, the model is allowed to explore the solution space and generate diverse candidate solutions. Unfortunately, the standard approach to post-training LLMs through Reinforcement Learning (RL) may limit pass@k: the model's output distribution narrows around high-reward outputs, causing the solution coverage to collapse. The alternative is to use Evolution Strategies (ES), a population-based, gradient-free post-training method that optimizes directly in weight space through random perturbations. As this paper shows, ES achieves consistently higher pass@k than RL and produces a broader output distribution with greater solution coverage. This coverage in turn makes it possible to achieve better results in e.g. standard math benchmarks. Thus, ES provides a better foundation for post-training in discovery problems and other domains where diverse solution coverage is critical.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.12675v1">Privacy-Preserving RAG by Concealing Sensitive Information from External LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-13
      | 💬 Submitted to Knowledge-Based Systems Journal
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) is widely used to improve the performance of Large Language Models (LLMs) in answering user queries. Existing privacy research on RAG has focused on preventing unauthorized users from accessing sensitive data. However, another important problem that is often overlooked in RAG privacy research is that external generators have access to the query and the retrieved documents, which may contain confidential information that could potentially be misused or accessed for unintended purposes. In this paper, we introduce the Sensitive Entity Alias Generator (SEAG), a privacy-preserving framework that empowers users to utilize powerful third-party generators without disclosing sensitive information. SEAG introduces a lightweight model that locates sensitive entities, generates corresponding aliases, and constructs an entity replacement table. The table is used to replace sensitive words in the user's query and in the retrieved documents before they are forwarded to an external generator. For this purpose, two datasets were constructed: one for fine-tuning SEAG models to generate entity replacement tables, and another for evaluating the entire SEAG framework. The experimental results demonstrate the success of the SEAG framework. As for the User metric, which measures the ability of the model to provide a correct response to the user while hiding sensitive information from the external generator, all SEAG models achieved over 80% accuracy. Additional analysis further evaluated the ability of SEAG models Qwen-3, LLaMA-3.2, and Phi-4 to hide all sensitive entities within given documents. The results show good performance with total accuracies of 77.83%, 76.73%, and 74.91%, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.22699v2">Similarity All The Way Up: Multilingual Generalization in LLMs Relies on Language-Level Similarity Structures</a></div>
    <div class="paper-meta">
      📅 2026-08-13
      | 💬 Accepted for oral presentation at CogSci 2026 (48th Annual Meeting of the Cognitive Science Society), Rio de Janeiro. 8 pages, 3 figures
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) grow more capable across diverse tasks, their (in)ability to generalize remains difficult to quantify and poorly understood beyond limited domains. In particular, LLMs are known to struggle generalizing multilingually, to languages outside of English, and that are poorly attested in their training data. To understand why this may be, and what enables some models to perform better than others, we turn to a long history of work across the cognitive sciences, arguing that successful generalization derives from appropriate representations in similarity space. We look at how well LLMs' representations capture the hierarchical similarity structure between distinct languages. Strikingly, we show LLMs' latent representations largely recover the hierarchical structure of the Indo-European language family tree -- grouping languages that are members of the same subfamily closely together in representation space. Furthermore, we show that the degree to which models reflect the similarity structure of languages correlates with their performance on XNLI, a multilingual natural language inference benchmark. This extends classic work on similarity-driven generalization at scale, showing how models that represent similar languages similarly generalize better from one language to another.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.12626v1">LLMs Are Not Good Strategists, Yet Memory-Enhanced Agency Boosts Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-08-12
    </div>
    <details class="paper-abstract">
      Strategic reasoning in Large Language Models (LLMs) within long-horizon environments is often limited by inconsistent subgoals. In these settings, finite attention resources prevent the model from maintaining strategic coherence over thousands of steps. This limitation leads to strategic drift, where localized decisions fail to sustain a coherent trajectory across reasoning. To address this, we introduce EpicStar, a framework that enables agents to learn memory as policy to tackle long-horizon reasoning. Specifically, the agent maintains a bank of successful past episodes as a heuristic alongside a working memory to track short-term environmental changes. During inference, a dynamic gating mechanism determines whether to execute a retrieved action directly or to perform new reasoning through a contextual fusion of the retrieved episodes and current working memory. Utilizing StarCraft II as the testbed, we evaluated EpicStar against diverse opponent styles. It significantly outperforms baseline methods, achieving higher win rates while consuming an order of magnitude fewer tokens, and it maintains this advantage consistently across difficulty levels and opponent strategies. Our findings provide compelling evidence that structured cross-episode memory is essential for enabling LLM agents to perform robust, long-term strategic execution in dynamic, autonomous settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10008v2">Do LLM Recommenders Know When They're Hallucinating? Auditing Confidence Calibration in Catalog Faithfulness</a></div>
    <div class="paper-meta">
      📅 2026-08-12
    </div>
    <details class="paper-abstract">
      LLM recommenders for top-K item suggestion regularly emit titles outside the target catalog. Prior audits report a binary out-of-domain rate; none ask whether the model knew. We jointly audit hallucination rate (OOD@10) and verbalized-confidence calibration (ECE, Brier, reliability) for four zero-shot LLM recommenders from four independent vendors (Mistral Large, Llama-3.3-70B, GPT-OSS-120B, Claude Sonnet 4.6), not grounded or fine-tuned systems, across three catalogs (MovieLens-25M, Amazon Reviews 2023 Toys, Yelp Open Dataset), stratified by item popularity. Measuring catalog membership is itself the hard part: on identical outputs the reported rate moves by an order of magnitude with the string matcher used, and F1 cannot separate the candidates. We validate the instrument against 201 human judgments and select on net bias, where the adopted one is off by -0.040 against +0.144 for the common fuzzy rule. Hallucination is then strongly catalog-dependent (0.6-2.7% on MovieLens, 11.6-38.7% on Yelp, 49.3-61.0% on Amazon Toys). Each model holds a near-constant confidence level barely responsive to the catalog, while the catalog-hit rate swings 60 points, so the sign of the error is set by where a model's constant lands against a catalog's accuracy: 7 of the twelve cells are under-confident and 5 over-confident, all four under-confident on MovieLens, all four over-confident on Amazon Toys. We read this as an elicitation mismatch: "Just Ask" elicits a generic quality rating, not a catalog-membership probability. A conformal abstention threshold over verbalized confidence changes hallucination by at most 1.65 pp across four alpha levels, because the channel cannot separate correct items from hallucinations. We recommend that audits report calibration alongside OOD, validate the matcher producing the OOD number, and use catalog-anchored elicitation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2401.11641v5">Revolutionizing Finance with LLMs: An Overview of Applications and Insights</a></div>
    <div class="paper-meta">
      📅 2026-08-12
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) like ChatGPT have seen considerable advancements and have been applied in diverse fields. Built on the Transformer architecture, these models are trained on extensive datasets, enabling them to understand and generate human language effectively. In the financial domain, the deployment of LLMs is gaining momentum. These models are being utilized for automating financial report generation, forecasting market trends, analyzing investor sentiment, and offering personalized financial advice. Leveraging their natural language processing capabilities, LLMs can distill key insights from vast financial data, aiding institutions in making informed investment choices and enhancing both operational efficiency and customer satisfaction. In this study, we provide a comprehensive overview of the emerging integration of LLMs into various financial tasks. Additionally, we conducted holistic tests on multiple financial tasks through the combination of natural language instructions. Our findings show that GPT-4 effectively follow prompt instructions across various financial tasks. This survey and evaluation of LLMs in the financial domain aim to deepen the understanding of LLMs' current role in finance for both financial practitioners and LLM researchers, identify new research and application prospects, and highlight how these technologies can be leveraged to solve practical challenges in the finance industry.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.12522v1">$\varepsilon$-MemEvo: Adaptive Cross-Task Memory Transfer for LLM Program Evolution</a></div>
    <div class="paper-meta">
      📅 2026-08-12
    </div>
    <details class="paper-abstract">
      LLM-based program evolution systems such as FunSearch and AlphaEvolve have shown strong ability to discover novel algorithms, but typically optimize each task in isolation, discarding search experience after completion. We introduce $\varepsilon$-MemEvo, a framework for cross-task knowledge transfer in LLM program evolution. $\varepsilon$-MemEvo stores prior experience as task-agnostic tactic memories: compact natural-language summaries of successful algorithmic strategies rather than raw code, enabling transfer across tasks with different APIs and evaluators. To avoid negative transfer from semantically mismatched memories, $\varepsilon$-MemEvo uses an adaptive injection gate that decides whether retrieved memories should be injected, and at what intensity. We evaluate $\varepsilon$-MemEvo on 8 diverse optimization benchmarks spanning mathematical optimization and systems engineering, using a content-level Leave-One-Out protocol that excludes target-task memory entries. On the primary GPT-5 backbone, $\varepsilon$-MemEvo improves AUCC over AdaEvolve on all 8 tasks, with a mean relative gain of +8.7%, and improves early-stage convergence by +9.4% on average. Ablations show that naive memory injection can fail catastrophically, while adaptive gating remains safe across all five ablation tasks. The data-updated posterior is interpretable in observed states: it favors skip during improving search and shifts from skip to hint across early and late plateaus. These gains incur less than 1% computational overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05033v2">SparseDitto: Customizing GPU Kernels for Different Sparsity Patterns with LLM-Based Agentic System</a></div>
    <div class="paper-meta">
      📅 2026-08-12
    </div>
    <details class="paper-abstract">
      Sparse matrix kernels are fundamental to scientific computing, graph analytics, and machine learning. Their GPU performance depends strongly on the input sparsity pattern and execution strategy. For the same SpMM on the same matrix, cuSPARSE exhibits a 350x performance gap between CSR and Blocked-ELL. Our study of multiple data formats, specialized systems, and sparse compilers shows that no single implementation consistently dominates across sparsity patterns and operators. This motivates a system that can adapt its representation, execution strategy, and hardware mapping to each workload and target GPU. We present SparseDitto, an LLM-based system that constructs a GPU kernel for each matrix, operator, and target GPU. SparseDitto supports SpMV, SpMM, and SpGEMM within a unified design framework. A lightweight additive model ranks established strategies using structural features of the input matrix. An architecture-aware planner then proposes several candidate designs. Coding and verification agents implement and refine them using measurements from the target GPU. Across three sparse operators and a diverse set of matrices, SparseDitto achieves a geometric-mean speedup of 2.68x over cuSPARSE on an NVIDIA RTX PRO 6000 GPU, with a maximum of 146.61x. On an NVIDIA H200 GPU, it achieves 2.79x, with a maximum of 78.5x. Its generated SpMM kernels also accelerate full-batch GCN training by up to 3.39x.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.12311v1">The Role Specialization Model (RSM): Coordinating LLM-Based Tools in Agentic Software Development - An Exploratory Case Study</a></div>
    <div class="paper-meta">
      📅 2026-08-12
      | 💬 28 pages, 4 figures, 5 tables
    </div>
    <details class="paper-abstract">
      The integration of large language models (LLMs) into software development workflows has given rise to a paradigm known as Agentic Software Engineering (SE 3.0), in which autonomous agents manage full development life cycles under human supervision. This paper presents an exploratory case study in which three LLM-based tools, Antigravity (an agentic IDE with a Gemini 2.5 backend), Gemini CLI, and Qwen Code (local execution via Ollama), are coordinated according to a role-distribution framework proposed in this work as the Role Specialization Model (RSM). Three research questions guide the study: (RQ1) how can LLM-based tools with distinct capabilities be coordinated through the RSM in a real development workflow; (RQ2) what deviations from the planned role distribution emerge during RSM execution and what factors explain them; and (RQ3) how does the resulting product compare against the ISO/IEC 25010 quality model. The objective was to propose the RSM with the incremental development of a Python desktop application for interactive climate-data visualization. The workflow, observed deviations, prompt-hardening techniques, and a qualitative quality assessment are documented. Results suggest that explicit role coordination can support development cycle organization and architectural quality, but requires deliberate coordination strategies, context management, and human verification of agent-generated outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10528v2">When Do Anchor-Based Pointwise LLM Rerankers Help? Retriever Quality, Statistical Scope, and Anchor Design</a></div>
    <div class="paper-meta">
      📅 2026-08-12
      | 💬 To be published in the 35th ACM International Conference on Information and Knowledge Management (CIKM 2026)
    </div>
    <details class="paper-abstract">
      Anchor-based pointwise LLM reranking scores each candidate against a shared reference passage to recover cross-document context at pointwise cost. We study when this actually helps, using GCCP/PAGC as a representative method. Our study is reproduction-first. We use reproduction as a starting point for a controlled component-level stress test of anchor-based pointwise reranking. Our initial reimplementation, based only on the paper text, achieves 0.24 nDCG@10 instead of the reported 0.66, revealing that several undocumented implementation details are necessary to reproduce the method. After identifying and recovering eight such details, we reproduce the reported results within 1.6% and use the validated implementation for controlled analysis. We find that the core contrastive scoring idea is robust under rigorous statistical correction. However, two design choices held fixed in the original paper are less reliable. First, we find that combining the contrastive score with the standard pointwise relevance score helps when the first-stage retriever is BM25, but gives little or no benefit when the first-stage retriever is a stronger dense model such as E5. Second, the paper's more complex method for constructing the anchor is unnecessary. A much simpler anchor, built by interleaving the top-ranked sentences, matches or outperforms it across datasets. These findings are consistent across different LLM backbones, including a 4-bit quantized 72B model. Overall, anchor-based pointwise reranking is effective, but its gains come mainly from contrastive scoring rather than from the more complex aggregation and anchor-construction choices, and they appear under narrower conditions than the original evaluation suggests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.12273v1">Convergent Detour Hijacking: Task-Preserving Resource Amplification in Skill-Based LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-12
    </div>
    <details class="paper-abstract">
      LLM agents increasingly rely on third-party skills, using natural-language descriptions for selection and instruction bodies for planning. This progressive-disclosure design exposes two sequential control points to untrusted publishers: a static skill may steer an otherwise correct task onto an unnecessarily costly trajectory. Prior work studies selection manipulation, malicious skill instructions, and tool-chain resource amplification largely separately, leaving their end-to-end composition unclear. We introduce Convergent Detour Hijacking (CDH), a text-only, runtime-independent attack that couples these stages. Under shared semantic cover, a description establishes relevance during selection, while an aligned body reuses that rationale to fabricate plausible dependencies during planning. CDH attracts an attacker-controlled coordinator alongside legitimate skills, recruits unnecessary benign skills into a bounded detour, and then re-enters the original route to preserve task completion. We evaluate it across multiple LLM backends and 491 held-out tasks under single-task and multi-turn conditions. On DeepSeek-V4-Pro, the matched coordinator is selected in 80.02% of tasks; among coordinator-hit runs that complete tasks, token consumption and end-to-end execution time increase by 66.91% and 92.45%, respectively, while aggregate task completion remains comparable. Thus, correct outcomes do not guarantee trajectory integrity or cost safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11650v2">Investigating Learner-Aware Design of LLM-Generated Educational Feedback</a></div>
    <div class="paper-meta">
      📅 2026-08-12
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Although large language models (LLMs) show promise for generating educational feedback, it remains unclear how feedback should be designed (e.g., tone and information coverage) to support answer revision and learner acceptance across diverse learner profiles. We define six feedback designs for multiple-choice biology questions, including a baseline design and variants with additional feedback elements, and conduct an empirical study with high school students. We evaluate feedback using immediate revision performance and six subjective evaluation criteria, and analyze how feedback preferences vary across learner profiles based on personality traits. Our results show that feedback with clear and comprehensive guidance improves revision performance and receives favorable evaluations across learner profiles, whereas informational novelty and affective framing vary across profiles. These findings suggest that learner profiles should be considered when designing LLM-generated feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27143v2">Enhancing Linux Privilege Escalation Attack Capabilities of Local LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-12
      | 💬 Accepted at RAISE workshop (https://raise-workshop.github.io/)
    </div>
    <details class="paper-abstract">
      Cloud-based Large Language Models (LLMs) can perform autonomous penetration-testing sub-tasks such as Linux privilege escalation, but raise security, privacy, and sovereignty concerns. Locally hosted open-weight models avoid these issues, yet prior work reports that small open-weight models succeed on only 8-16% of standardized privilege-escalation tasks, far below frontier cloud models. This paper is an empirical study of why small models fail at this task and which engineering techniques close the gap. From execution traces we distill six recurring failure modes, map each to an established enhancement technique, and evaluate five (chain-of-thought prompting, retrieval-augmented generation, structured prompting, history compression, and reflective analysis) as reproducible extensions to the open-source hackingBuddyGPT framework. Under a single shared harness and matched conditions, the set of techniques we evaluate raise two SLMs (Llama3.1 8B, Qwen2.5 7B) from 8% to 67% with guidance, matching guided GPT-4o. A larger open-weight reference model (Llama3.1 70B) reaches 83%. A full-factorial ablation shows that reflection-based techniques contribute most and reveals vulnerability discovery, not exploitation, as the main constraint for local models. We report these as transferable lessons for building reliable local offensive agents as well as to inform defenders.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.12228v1">Towards Automated Domain Model Extraction from Source Code using Heuristics and Open-Source LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-12
      | 💬 To appear in the Proceedings of the 29th International Conference on Model Driven Engineering Languages and Systems (MODELS 2026)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently shown strong capabilities for code understanding, making them promising for reverse engineering domain models from source code. However, state-ofthe- art proprietary LLMs cannot be used in many industrial contexts due to privacy and confidentiality constraints, while compact open-source LLMs that can run locally are limited by their context window and cannot process large code bases directly. In this paper, we propose an automated approach to extract domain models from source code using lightweight, locally deployable LLMs. Our method combines structural and semantic heuristics with iterative LLM-based reasoning to overcome context limitations. By progressively analyzing ranked subsets of code elements, the approach identifies domain concepts and refines domain boundaries without requiring full-system context. Our approach achieves high F1-scores on a dataset of ten projects, each comprising a curated domain model and its corresponding implementation, while remaining fully executable on locally deployable LLMs. This makes it particularly suitable for reverse engineering tasks in privacy-sensitive industrial environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.12197v1">NetlistBench: Evaluating LLM Reliability in SPICE Netlist Recognition and Manipulation</a></div>
    <div class="paper-meta">
      📅 2026-08-12
      | 💬 accepted by MLCAD 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in circuit design workflows, yet their reliability on simulator-facing SPICE netlist recognition and manipulation remains poorly understood and is rarely separated from high-level design reasoning. Although netlists are textual, they encode structured circuit objects through topology and parameters. We present \textbf{NetlistBench}, a structure-verified benchmark for SPICE netlist recognition and manipulation. NetlistBench contains 2,342 cases across 24 task families, covering parameter and connectivity recognition and edits, hierarchical operations, equivalence judgment, and long-horizon compound editing. Model outputs are evaluated by a deterministic structure-aware oracle. Across six non-thinking LLMs, performance varies substantially with operation-level structural complexity. Simple local edits reach $96\%$--$100\%$ accuracy, while device addition drops to $41\%$--$83\%$ and equivalence judgment to $49\%$--$90\%$. Enabling reasoning substantially improves weaker models but does not eliminate structure-preservation failures, with performance still degrading sharply as the edit horizon increases. NetlistBench identifies netlist reliability as a distinct bottleneck for trustworthy LLM-based circuit design automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.12138v1">A corpus-specific clinical RAG system matches or outperforms newer frontier LLMs on HealthBench</a></div>
    <div class="paper-meta">
      📅 2026-08-12
      | 💬 2 tables
    </div>
    <details class="paper-abstract">
      General-purpose large language models (LLMs) have recently been reported to match or exceed specialized clinical AI tools on medical benchmarks, but such comparisons draw on a narrow set of systems and on benchmarks developed largely in high-income settings. We evaluate VITA, a retrieval-augmented generation (RAG) system purpose-built for contextual knowledge retrieval in India and other low- and middle-income (LMIC) settings. VITA retrieves from a curated corpus of disease-specific guidelines, India-specific antimicrobial resistance data, national formulary constraints, and resource-limited care protocols; its architecture and corpus are proprietary, but the benchmark, the physician-written rubrics, and our full response and scoring outputs are public for independent verification. On 4,023 English-language HealthBench questions (80.5% of the benchmark), scored with a GPT-4.1 judge, VITA ranked first with 51.9% of possible rubric points, ahead of GPT-5.4 (46.1%), o4-mini (44.3%), Gemini 3.1 Pro (42.6%), and Claude Sonnet 4.6 (37.3%), and scored highest on 45.4% of questions. To test robustness to newer models and judge lineage, a 500-question subset was re-run against current-generation models (GPT-5.5, Claude Opus 4.8, Gemini 3.5 Pro, Grok 4.3) and graded by a neutral open-weight judge (DeepSeek-V4-Pro) sharing no lineage with any system tested. Here the gap narrowed to parity: VITA and GPT-5.5 were statistically indistinguishable on mean per-question score, while VITA led on points-weighted score and won the most questions. VITA's advantages in accuracy and completeness persisted under the neutral judge; its communication scores were lower. These results indicate that a purpose-built clinical RAG system remains competitive with frontier LLMs on an open benchmark, consistent with corpus specificity as a design variable that improves grounding at some cost to communication polish.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.12125v1">Do LLMs Take Care of Their Own? Similarity Signals Can Induce Cooperation</a></div>
    <div class="paper-meta">
      📅 2026-08-12
      | 💬 41 pages, 18 Figures, 4 Tables, 16 Listings
    </div>
    <details class="paper-abstract">
      As LLM-based agents with user-instructed goals are becoming widely deployed, they increasingly encounter each other in strategic interactions, and face challenges of finding mutually beneficial outcomes. Prior literature has argued that cooperation problems such as the Prisoner's Dilemma are resolvable in settings where agents know they follow very similar decision making patterns, as for example in monocultural AI ecosystems. Following that line of work, this paper introduces the first framework for evaluating LLM decision making when agents are provided with graded similarity signals. Among our findings, we establish that different LLM models vary drastically in how they navigate similarity signals, with some modern models showing consistent behavior across cooperation problems, payoff structures, and prompt framing. Perhaps surprisingly, our experiments also show that the dataset based on which the similarity signal is computed has small to no impact on induced cooperation, and that LLM models systematically self-identify as highly similar when asked to evaluate another model's chain-of-thought reasoning by themselves. Finally, we develop an LLM-behavioral-game-theoretic model that captures some of their reasoning rationale, and show that it can support cooperative outcomes in equilibrium under sufficiently high similarity scores.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.12123v1">Ready Cohorts: Bounding GPU Opportunity and Avoiding Host Round Trips in LLM-Agent Control</a></div>
    <div class="paper-meta">
      📅 2026-08-12
      | 💬 14 pages, 4 figures. Includes formal proofs, trace provenance, and a reproducibility appendix. Code and artifacts: https://github.com/josefchen/ready-cohorts ; processed evidence: https://huggingface.co/datasets/josefchen/ready-
    </div>
    <details class="paper-abstract">
      LLM-agent services repeatedly execute small deterministic transitions between model and tool calls: route an outcome, update state, and emit the next effect. We ask when this control path exposes enough concurrent work for GPU execution, and what changes when a GPU-computed route decision remains on device. We formalize the ready-cohort boundary using fixed-partition share F, exact offline share P*, local upper bound U, and online achieved share A. Under zero service time, unlimited capacity, and equal relative launch deadlines, a specialized dynamic program computes P* exactly. In a stationary Poisson replay of one pinned 851-session public trace panel, the primary condition at 100,000 target active sessions, K=256, and a 50 ms launch deadline gives F=30.19%, P*=43.00%, and U=45.85%. Exact packing recovers 81.83% of the opportunity lost at fixed window boundaries. The outcome-derived route key is a conditioning proxy, not proof of executable identity. A separate mechanism study keeps a GPU-computed binary decision on device instead of returning four bytes to the host and redispatching. Across four named GPU placements, the device-resident path is faster in all 36 configurations; within-placement row-median ratios range from 1.19x to 2.39x. Across both admissible mechanisms, all 14,557,440 tested batched invocations match a separately implemented host oracle. A fixed nested device graph that removes no host decision is slower in all 60 configurations across five placements. Together, the studies establish two measurable gates for GPU agent control: deadline-feasible cohort supply and observation placement. A joined finite online runtime is required to measure A, CPU displacement, and service-level benefit.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05639v2">TokenStack: A Heterogeneous HBM-PIM Architecture and Runtime for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-08-12
    </div>
    <details class="paper-abstract">
      Large language model (LLM) serving is now limited by the key-value (KV) cache. During decode, each new token rereads prior KV state, so attention becomes a bandwidth- and capacity-heavy memory task. HBM-PIM helps by moving attention closer to memory, but current stack organizations still waste resources. In practice, only hot KV blocks benefit from near-memory compute. Weights, activations, and cold KV mainly need dense storage and GPU-visible bandwidth. A uniform HBM-PIM stack makes all layers pay for PIM logic, while a dedicated-PIM design such as AttAcc recovers capacity but shrinks the HBM bandwidth left for GPU-side work. We propose TokenStack, a vertically heterogeneous HBM-PIM architecture for KV-centric LLM serving that leverages HBM4's logic-die substrate. TokenStack separates each stack into dense capacity layers and PIM-enabled compute layers, then uses the logic base die as a stack-local control point that manages cross-layer movement without host-side overhead. The base-die controller handles cross-layer DMA, layered address translation, attention-side gather/broadcast coordination, and inline quantization during migration. On top of this hardware, TokenStack uses topology-aware KV placement, workload-aware eviction, and bounded replication to keep hot KV near PIM compute while moving colder state to dense layers. Using production-derived traces across four models, completed multi-QPS runs show that TokenStack increases geometric-mean token throughput by 1.62x and SLO-compliant serving capacity by 1.70x over AttAcc, and reduces per-token energy by 30-47%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.12025v1">From Safety Documentation to Safety Knowledge Support: An Evidence-Grounded LLM Framework for Medical Devices</a></div>
    <div class="paper-meta">
      📅 2026-08-12
      | 💬 ISSRE 2026, AISQ, 8 pages
    </div>
    <details class="paper-abstract">
      Medical devices are becoming more software-intensive, connected, and AI-enabled. Their development requires risk-management evidence aligned with ISO 14971 and, for software, IEC 62304. This evidence must be kept consistent across requirements, design decisions, software changes, verification results, complaints, and post-market data. These tasks are costly and depend on scarce safety and domain experts. Large language models (LLMs) may reduce parts of this effort because medical-device safety work is highly document-based. However, current LLM-based safety-engineering studies often address isolated methods, rely on generic prompting or public examples, and provide limited support for source links, traceability, uncertainty handling, lifecycle updates, and recorded expert review. This limits their use in regulated medical-device development. This paper argues that the central research problem is not safety-text generation, but source-linked safety-knowledge support. We propose an evidence-grounded framework that connects device artifacts, controlled knowledge storage and retrieval, method-specific generation of candidate safety items, critique and uncertainty checks, and recorded expert review. The framework prepares, links, checks, and updates candidate safety artifacts for expert decision-making. It does not decide whether a device is safe and does not provide regulatory approval. We also outline an evaluation strategy using non-public or newly built medical-device case studies and expert reference analyses to assess coverage, correctness, relevance, traceability, duplicate rate, unsupported claims, and review effort.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.11965v1">Developing LLM-based Multi-Agent Systems in Software Engineering: A Mixed-Method Experience Report</a></div>
    <div class="paper-meta">
      📅 2026-08-12
      | 💬 The paper has been peer reviewed and accepted for publication with the Empirical Software Engineering journal
    </div>
    <details class="paper-abstract">
      The proliferation of Generative Artificial Intelligence (Gen AI) powered by large language models (LLMs) has transformed the software development process, introducing new paradigms for code generation, debugging, testing, and maintenance. While early applications focused on leveraging single, independent LLMs to assist developers with isolated tasks, recent advances have shifted toward multi-agent systems (MAS) that orchestrate multiple LLM-based agents working collaboratively toward common objectives. Despite their promising potential, using MAS encompasses a set of challenges for developers who have to carefully select the right technology, devise proper coordination rules, and design specific roles for the involved agents. In this paper, we provide a comprehensive overview of the existing tools and frameworks for implementing MAS in software engineering. First, we conducted a quantitative analysis of the most relevant open source MAS frameworks by evaluating their documentation, features, and capabilities from the developers' perspective. Second, we performed a qualitative evaluation of a subset of the selected frameworks by implementing a common use case: the summarization of README.MD files. The findings show that the selected frameworks provide a good coverage of fundamental components of MAS, though advanced features such as telemetry of agents are still missing. In addition, the empirical evaluation shows that there is no significant difference in terms of ROUGE scores considering the summarization task. Finally, we provide a set of lessons learned and challenges that can help researchers and practitioners to select a suitable MAS framework according to their needs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.11922v1">LODESTAR: Trustworthy Entropy Is Navigated, Not Merely Measured -- Reinforced Polarizer Keeps a Frozen LLM from Being Confidently Misled by the Wrong Evidence</a></div>
    <div class="paper-meta">
      📅 2026-08-12
      | 💬 28 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Predictive-distribution entropy makes a strong selection rule in retrieval-augmented question answering: across five QA benchmarks, keeping the candidate answer that a frozen respondent LLM produces with the lowest answer-token entropy lifts mean answer $F_1$ from 0.4769 to 0.5148 over the retriever's top-ranked passage, with no gold answers. Yet this lowest-entropy rule, which prior entropy-based selectors adopt, fails in a specific and consequential way: a misleading passage makes the respondent confidently wrong, driving its entropy down precisely where the signal looks most trustworthy. We show that the failure comes from the passage the respondent reads -- and the context that passage is read in is an input we can intervene on. We introduce LODESTAR, to our knowledge the first method to score a text intervention by the uncertainty it induces in a third-party frozen respondent, compared across one question's candidates. LODESTAR uses reinforcement learning to train, once and offline, a polarizer -- a short fixed natural-language string inserted into the respondent's prompt and never into its weights; its training labels are built offline from gold answers and two LLM judges, and inference reads neither. Evaluating every competing selector under the same frozen respondent and the same candidate pools on 5,008 questions, LODESTAR attains the highest mean $F_1$ of any inference-ready selector (0.5148 to 0.5339), the highest exact match (0.4136), and the highest GPT-4o judge score of the frozen-respondent configurations judged (0.6435); its three-seed mean wins all 70 method-by-dataset $F_1$ cells against fourteen published configurations while remaining paired-significant against every one. The gain holds both in-domain and out-of-domain, and ablating the polarizer shows it is what makes the respondent read a misleading passage less often (26.0% against 30.3%).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09321v2">SpatialJB: How Text Distribution Art Becomes the "Jailbreak Key" for LLM Guardrails</a></div>
    <div class="paper-meta">
      📅 2026-08-12
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have powerful capabilities, they remain vulnerable to jailbreak attacks, which is a critical barrier to their safe web real-time application. Current commercial LLM providers deploy output guardrails to filter harmful outputs, yet these defenses are not impenetrable. Due to LLMs' reliance on autoregressive, token-by-token inference, their semantic representations lack robustness to spatially structured perturbations, such as redistributing tokens across different rows, columns, or diagonals. Exploiting the Transformer's spatial weakness, we propose SpatialJB to disrupt the model's output generation process, allowing harmful content to bypass guardrails without detection. Comprehensive experiments conducted on leading LLMs get nearly 100% ASR, demonstrating the high effectiveness of SpatialJB. Even after adding advanced output guardrails, like the OpenAI Moderation API, SpatialJB consistently maintains a success rate exceeding 75%, outperforming current jailbreak techniques by a significant margin. The proposal of SpatialJB exposes a key weakness in current guardrails and emphasizes the importance of spatial semantics, offering new insights to advance LLM safety research. To prevent potential misuse, we also present baseline defense strategies against SpatialJB and evaluate their effectiveness in mitigating such attacks. The code for the attack, baseline defenses, and a demo are available at https://anonymous.4open.science/r/SpatialJailbreak-8E63.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.11888v1">Agent Skills Can Be Harmful: An Empirical Study of Skill-Induced Failures in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-12
    </div>
    <details class="paper-abstract">
      Agent skills are the de facto mechanism for extending LLM agents with reusable guidance. A skill can shape the agent's task execution, including planning, tool use, problem-solving, and validation. Prior work reported mixed results of agent skills: some skills improve task success rates, while others have no effect, increase token use and execution time, and even reduce success rates. This paper presents a comprehensive analysis of skill-induced agent failures by attributing task failures and cost regressions to specific loaded skills. We introduce a differential analysis framework that attributes a failure or regression to a skill by comparing a target skill-guided run against a no-skill or semantically matched skill reference run that solves the same task, or solves it more cheaply. We instantiate this framework on SkillsBench and SWE-Skills-Bench, yielding 307 skill-induced failures, including 125 functional failures and 182 efficiency regressions. We also build SkillTriage, a taxonomy-guided attribution tool that normalizes paired cases, extracts differential evidence, and produces triage reports. Our major findings include: (1) Skill induced functional failures are rarely caused by obviously irrelevant skills; instead, seemingly relevant skills often make the agent incorrectly implement or omit task-required implementation elements. (2) Skill-induced efficiency regressions are not explained by prompt length alone. (3) The largest sources within Excessive Procedure are excessive verification and heavy implementation pipelines, contributing 67 and 30 cases, respectively. This shows that skills often turn validation checklists and construction recipes into mandatory work. Based on our findings, we propose research topics and tooling improvements for safer and more cost-aware skill reuse.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.11878v1">ToolHazard: Scaling Adversarial Environments for Security Evaluation and Alignment of LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-12
      | 💬 Work in Progress
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents integrated with external tools are vulnerable to indirect prompt injections embedded in environmental states. However, existing studies largely rely on manually implemented or reused environments, stochastic LLM-based tool simulation, and predefined injection locations, limiting scalable security research across broader domains. To bridge this gap, we propose **ToolHazard**, a scalable adversarial environment synthesis framework that reduces human engineering and supports expansion with additional seed domains and compute. Through an Environment Simulator, an Attacker Agent, and a User Simulator, ToolHazard synthesizes executable stateful environments, discovers viable injection points and generates environment-specific payloads, and constructs state-grounded long-horizon tasks. Based on ToolHazard, we build **ToolHazard-Bench** for stress-testing agents under complex workflows and diverse environmental attacks. Experiments reveal substantial agent vulnerabilities and show that injection timing and placement affect attack effectiveness. Moreover, ToolHazard-generated alignment data improves security on both ToolHazard-Bench and AgentDojo while preserving benign task utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.11008v2">Templated or fully synthetic? Prompt construction as a confound in measuring LLM political stance beyond writing assistance</a></div>
    <div class="paper-meta">
      📅 2026-08-12
    </div>
    <details class="paper-abstract">
      Political stance detection in LLMs has long been dominated by closed-ended, multiple-choice political survey questions---originally designed for humans, and thus lacks the realism and nuance of human-AI interactions in the wild, while also being susceptible to sandbagging. The recent IssueBench framework substantially mitigates these limitations with templated prompts anchored in real-world chat logs. Given the rise in non-work-related use of GenAI assistants, we extend IssueBench beyond writing assistance to include two additional tasks, information seeking and opinion sharing. We argue that templated prompts still lack the nuance of real ones, especially for open-ended tasks, and remain recognisable as evaluation artefacts. We propose the use of fully synthetic (LLM-generated) prompts, produced under detailed instructions with real prompts as seeds. We assess the ecological validity of real, templated, and LLM-generated prompts in a small-scale study covering 3 highly contested policy issues and 3 recent geopolitical conflicts. Human and LLM annotators rank LLM-generated prompts as no less realistic than real ones and clearly more realistic than templated ones, and find that they carry their intended intent and stance more clearly; the LLMs separate templated prompts from the other two far more sharply than the humans do. In a case study, templated and LLM-generated prompts yield systematically different stance estimates for the same model, most visibly under neutral framings, where templated prompts overstate the model's leaning in the direction encoded by the topic-and-stance text (filler) slotted into their templates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.11830v1">Quantifying the Relationship Between Clinical Safety and Environmental Impact in Therapeutic LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-12
    </div>
    <details class="paper-abstract">
      The deployment of large language models (LLMs) in mental health contexts raises questions about the relationship between clinical safety and environmental cost. In this paper, we examine this relationship by combining K-Bench clinical safety scores with EcoLogits life-cycle assessment estimates across 47 supported model configurations. We evaluate model performance and environmental impact across four dimensions: energy use, carbon emissions, water consumption, and abiotic depletion. The results indicate a non-linear trade-off at the upper end of the safety distribution: a 2.61 percentage-point increase in clinical safety score corresponded to an approximately 60-fold increase in estimated energy use per million output tokens. Row-level analyses further suggest that additional test-time compute did not consistently improve clinical safety and, in some configurations, was associated with lower clinical safety scores. These findings suggest that relying solely on larger models or additional inference-time computation may be an inefficient strategy for improving safety in therapeutic AI systems. We discuss the implications for sustainable deployment and highlight dynamic model selection, including model cascading, as a potential approach for reducing environmental impact while preserving clinical performance in higher-risk cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.11804v1">MiDashengLM-Gen: Unified Audio Scene Generation via LLM-Driven Autoregressive Flow Matching</a></div>
    <div class="paper-meta">
      📅 2026-08-12
    </div>
    <details class="paper-abstract">
      Generating coherent audio scenes that simultaneously blend speech, music, and sound effects remains a significant challenge. Current approaches typically rely on a disjointed pipeline where a frozen, decoupled text encoder feeds a separate audio decoder, limiting cross-modal optimization and leading to poor speech intelligibility. To overcome these limitations, we introduce MiDashengLM-Gen, an end-to-end framework that couples a pre-trained Large Language Model (LLM) with per-token conditional flow matching for autoregressive, variable-length mixed-audio scene generation. MiDashengLM-Gen represents a first approach for general text-to-audio generation with one end-to-end trained model. Empirical evaluations demonstrate that MiDashengLM-Gen drastically improves speech intelligibility over existing unified models. On the Seed-TTS benchmark, English Word Error Rate (WER) drops from 12.15% to 2.79%, approaching the performance of dedicated Text-to-Speech (TTS) systems (1.24%). Furthermore, the framework extends effectively to multilingual settings, yielding highly competitive multilingual WERs compared to existing baselines. Lastly, the model maintains competitive mixed-audio generation quality on the MECAT benchmark. Code and checkpoints are available at https://github.com/xiaomi-research/midashenglm-gen and https://huggingface.co/mispeech/midashenglm-gen, and the demo page is available at https://xingws.github.io/midashenglm-gen-demo/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.11787v1">GRPO for Financial Advice Generation: Outperforming Commercial LLMs under CATE Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-08-12
    </div>
    <details class="paper-abstract">
      Generating actionable financial advice from business records demands that models integrate numerical reasoning, domain knowledge, and sound judgment, while avoiding recommendations that could harm the business. Direct supervision is difficult: historical decisions are not necessarily optimal, and high-quality free-form labels are expensive to obtain. We formulate financial advice generation as a reinforcement learning problem and fine-tune an open-weight language model using Group Relative Policy Optimization (GRPO). Our reward is an LLM-as-a-judge rubric that scores each recommendation across multiple binary dimensions of advice quality, augmented with a safety gate for harm prevention. Since LLM-based evaluation alone cannot confirm whether improvements reflect genuine business value rather than adaptation to the judge, we complement it with a judge-independent audit based on a standard doubly-robust Conditional Average Treatment Effect (CATE) estimator. Under this observational off-policy audit, our trained LLM achieves approximately twice the estimated gross-profit lift of the strongest evaluated commercial baseline ($0.0228$ vs.\ $0.0104$), together with the lowest downside rate and the least negative tail risk of any policy evaluated. Notably, the two evaluations do not rank the baselines identically: the untrained base model places last on the judge rubric but second on the causal audit, indicating that the audit captures a signal the judge does not. Our results demonstrate that GRPO with a finance-grounded reward signal can produce substantially more useful business recommendations than commercial LLMs, and that a judge-independent causal audit is a valuable complement to, rather than a confirmation of, LLM-as-a-judge assessment in financial NLP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.11744v1">Harnessing LLMs for Document-Guided Fuzzing of Python Libraries</a></div>
    <div class="paper-meta">
      📅 2026-08-12
    </div>
    <details class="paper-abstract">
      Python libraries underpin deep learning, scientific computing, data analysis, and computer vision, making their reliability critical to downstream applications. Testing their APIs requires inputs that satisfy both per-parameter constraints and dependencies among parameters. Existing approaches either leave such constraints implicit in generated programs or rely on library-specific parsing rules. This paper introduces VistaFuzz, a document-guided fuzzing technique that uses a locally served open-sourced LLM to extract parameter specifications from API documents and generate inputs that satisfy both parameter constraints and inter-parameter dependencies. We evaluate VistaFuzz on 7,718 APIs across twelve Python libraries. Inter-parameter relationships occur in 40.1\% of tested APIs, and disabling their resolution reduces the valid generation rate on those APIs from above 95\% to 31.6\%--52.8\%. VistaFuzz reports 74 issues, of which 43 have been confirmed by developers and 29 have been fixed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.13452v2">LLM-Powered Automatic Translation and Urgency in Crisis Scenarios</a></div>
    <div class="paper-meta">
      📅 2026-08-12
      | 💬 Accepted to ISCRAM 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly proposed for crisis preparedness and response, particularly for multilingual communication. However, their suitability for high-stakes crisis contexts remains insufficiently evaluated. This work examines the performance of state-of-the-art LLMs and machine translation systems in crisis-domain translation, with a focus on preserving urgency, a critical property for effective crisis communication and triage. Using multilingual crisis data (TICO-19, 30 languages) and a newly introduced urgency-annotated dataset of 100 scenarios translated into 29 languages, we show that dedicated translation models and LLMs exhibit substantial quality degradation, particularly for low-resource languages. Beyond translation quality, we conduct a human annotation study revealing a striking asymmetry: human assessors maintain consistent urgency judgments regardless of prompt language, while LLM-based urgency classifications vary widely across languages for identical scenarios, at times spanning the full range from Not Urgent to Critical. These findings highlight significant risks in deploying general-purpose language technologies for crisis triage and underscore the need for multilingual, human-centered evaluation frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.14501v2">CangjieBench: Benchmarking LLMs on a Low-Resource General-Purpose Programming Language</a></div>
    <div class="paper-meta">
      📅 2026-08-12
      | 💬 Accepted by ESEM 2026
    </div>
    <details class="paper-abstract">
      Large Language Models excel in high-resource programming languages but struggle with low-resource ones. Existing research related to low-resource programming languages primarily focuses on Domain-Specific Languages (DSLs), leaving general-purpose languages that suffer from data scarcity underexplored. To address this gap, we introduce CangjieBench, a contamination-free benchmark for Cangjie, a representative low-resource general-purpose language. The benchmark comprises 248 high-quality samples manually translated from HumanEval and ClassEval, covering both Text-to-Code and Code-to-Code tasks. We conduct a systematic evaluation of diverse LLMs under four settings: Direct Generation, Syntax-Constrained Generation, Retrieval-Augmented Generation (RAG), and Agent. Experiments reveal that Direct Generation performs poorly, whereas Syntax-Constrained Generation offers the best trade-off between accuracy and computational cost. Agent achieve state-of-the-art accuracy but incur high token consumption. Furthermore, we observe that Code-to-Code translation often underperforms Text-to-Code generation, suggesting a negative transfer phenomenon where models overfit to the source language patterns. We hope that our work will offer valuable insights into LLM generalization to unseen and low-resource programming languages. Our code and data are available at https://github.com/cjhCoder7/CangjieBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.11705v1">Making Your LLMs More Objective: Stabilizing LLM Safety Behavior Across Traits with Trait-Invariant Safety Tuning</a></div>
    <div class="paper-meta">
      📅 2026-08-12
    </div>
    <details class="paper-abstract">
      Aligned large language models (LLMs) are expected to exhibit safety behavior based on the content of the user request: they should refuse unsafe requests and comply with safe ones. However, we show that the same request can elicit substantially different safety decisions under different traits assigned in the system prompt, a failure mode we call trait-induced safety variation. To measure this failure, we introduce refusal-based metrics: Trait-Induced Deviation measures dataset-level deviation from the no-trait baseline, while Trait-Induced Flip Rate measures whether the same request receives different safety decisions across traits. We then provide a representation-level analysis of the mechanism behind trait-induced safety shifts and find that traits perturb the model's safety representations within a low-dimensional subspace. To achieve trait-invariant safety, where safety behavior remains stable across traits, we introduce Trait-Invariant Safety Tuning (TIST), a simple yet effective self-distillation framework that aligns an LLM's trait-conditioned behavior with its no-trait behavior. Guided by our analysis, we further propose Trait-Subspace Neutralization (TraSN), an instantiation of TIST, which enforces invariance only within the identified trait subspace. Experiments show that TraSN improves trait-invariant safety and strengthens harmful-request safety while preserving general capability. Our results highlight traits as an important factor in LLM safety and robust model behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.11679v1">AgenticTwin: An Agentic LLM Framework Integrated with Digital Twin for Anomaly Detection</a></div>
    <div class="paper-meta">
      📅 2026-08-12
    </div>
    <details class="paper-abstract">
      Digital twins are increasingly used to monitor and simulate the behavior of cyber-physical systems. Even with skilled operators, interpreting anomalies detected within digital twin pipelines is challenging, as the sheer complexity and volume of raw sensor data make thorough analysis difficult. Recent advances in large language models (LLMs) offer promising capabilities for reasoning and explanation, yet their integration into digital twin-driven anomaly analysis remains underexplored. In this work, we propose AgenticTwin, an agentic framework that integrates LLM-driven reasoning with a digital twin-based anomaly detection pipeline. The framework grounds LLM-generated explanations in outputs from a digital twin-driven anomaly classifier and enables human operators to ask relevant natural-language questions about the system. Beyond the framework itself, we introduce a benchmark-oriented evaluation pipeline constructed over synthetic anomalies injected into a real-world weather sensor dataset, enabling controlled generation of operator queries over anomaly events. We further evaluate the feasibility of deploying lightweight, open-source LLMs for practical cyber-physical environments. Experimental results demonstrate that structured agent collaboration and knowledge-grounded reasoning improve diagnosis quality, contextual retrieval, and mitigation quality across diverse possible anomaly scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.11676v1">XBridge: Entity-Grounded Latent Bridge for Heterogeneous LLM Communication</a></div>
    <div class="paper-meta">
      📅 2026-08-12
      | 💬 18 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Heterogeneous multi-agent LLM systems, where agents are powered by different model families, can outperform homogeneous configurations by reducing redundant reasoning patterns. Yet existing communication protocols either operate through text, discarding the sender's internal representations, or require architectural homogeneity for latent-level transfer. We identify the entity grounding problem in cross-architecture communication: cross-attention bridges that transfer continuous representations across different LLM families suffer from rare-token compression collapse, where entity identity is lost in the continuous bottleneck (bridge-only F1 ~30%). We propose XBRIDGE, a decode-free communication protocol that addresses this through two mechanisms. Lexical Anchor Mapping (LAM) maps the sender's original context tokens to the receiver's vocabulary, providing discrete entity anchors. A Latent Enrichment Bridge (LEB) lets the receiver query the sender's hidden states for contextual enrichment. The entity anchors ground the bridge's contextual signals to specific entities through the receiver's own self-attention. Across three model families (Llama, Qwen, and Mistral), seven benchmarks, and both communication directions, XBRIDGE outperforms text-based communication on all seven tasks for each model pair while achieving 11x lower latency, and in a same-architecture setting it also exceeds a KV-sharing baseline on six of seven tasks. LEB requires only 264M trainable parameters (3.8% of the receiver), is trained on a small balanced sample set, and adds negligible inference overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.11649v1">Who Would You Vote For? Auditing Political Alignment in LLMs: An Italian Case-Study</a></div>
    <div class="paper-meta">
      📅 2026-08-12
    </div>
    <details class="paper-abstract">
      As users increasingly turn to Large Language Models (LLMs) for information and advice on political matters, particularly during election periods, the political preferences expressed by these systems have become a matter of public interest. Prior research has shown that interactions with LLMs can influence users' political attitudes and choices, raising questions about how these models themselves evaluate political actors. In this paper, we investigate whether and how LLMs express preferences toward political parties and political leaders. We introduce a systematic and reproducible auditing framework in which multiple LLMs are prompted to evaluate parties and leaders across nine criteria. Rather than attempting to infer the models' "true" political beliefs, we focus on their observable behavior, examining consistency across evaluations, differences between models, refusal rates, and sensitivity to prompt formulation. We further investigate how these evaluations vary when models are instructed to adopt different personas. We demonstrate the framework through an Italian case study, providing a systematic analysis of LLM-generated political evaluations on italian parties and leaders.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.11624v1">Learning to Persuade Exposes How Easily LLMs Abandon Correct Beliefs</a></div>
    <div class="paper-meta">
      📅 2026-08-12
    </div>
    <details class="paper-abstract">
      Persuasion is a core dynamic of natural language communication, shaping how large language models (LLMs) update beliefs, resolve disagreements, and reach decisions. As LLMs increasingly debate, advise, and think collaboratively with humans and each other, resistance to harmful persuasion becomes a core requirement for reliable behavior. Yet we show that this requirement is far from met: a single targeted persuasive argument is enough to collapse model accuracy to near zero, even when the argument is factually false. We formalize this threat as adversarial persuasion and introduce an adversarial reinforcement learning framework that trains persuader agents to change a target model's answer in a single interaction. First, we show that optimizing persuasion strategies through trial and error exposes vulnerabilities that static prompting misses: RL-trained persuaders raise persuasion success from approximately 24% to over 93% against the training-time persuadee. Second, we find that these learned strategies transfer to unseen models, achieving 83% attack success on Qwen-14B, 79% on Llama-3.1-8B, and 25% on GPT-4o-mini. Third, we demonstrate that a curriculum that bootstraps on more persuadable open-weight models before targeting harder models further increases GPT-4o-mini attack success from 25% to 38%. Moreover, our results reveal that optimized persuaders increasingly rely on credibility-based tactics, including fabricated citations and false authoritative evidence. Together, these findings expose a critical weakness in current LLM agents: even when they initially reason correctly, they can be steered toward false conclusions by optimized natural language influence. This positions persuasion robustness as a necessary safety criterion for multi-agent and human-AI decision-making systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.11623v1">FM-LLM: A frequency-enhanced mixture-of-experts framework for adapting LLMs to time series forecasting</a></div>
    <div class="paper-meta">
      📅 2026-08-12
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have spurred cross-modal solutions for time-series forecasting. However, existing methods rely heavily on textual prompts for modality alignment-introducing nontrivial computational overhead and failing to leverage the rich spectral dynamics inherent in time-series data. To enable prompt-free, frequency-aware adaptation of frozen LLMs, we propose FM-LLM (Frequency-Enhanced Mixture-of-Experts for adapting LLMs to Time Series Forecasting), an autoregressive framework grounded in constrained asymmetric coupling. A Fourier Analysis Network (FAN)-based spectral token aligner injects structured harmonic representations directly into the frozen LLM with numerical compatibility. An asymmetric Mixture-of-Experts (MoE) decoder enforces role separation: shared experts with lightweight FAN layers reconstruct the global periodic backbone, while routed experts-restricted to standard FFNs-specialize in modeling non-periodic residual dynamics. A time-frequency hybrid loss function jointly optimizes temporal accuracy and spectral consistency, mitigating error accumulation during long-horizon autoregressive rollouts. Evaluated across eleven public benchmarks, FM-LLM achieves state-of-the-art performance on 59 out of 78 evaluation metrics. Compared to the strongest autoregressive LLM-based baseline, it delivers average improvements of 5.3% in MSE and 5.6% in MAE, with maximum gains reaching 8.0% for MSE and 8.4% for MAE. FM-LLM also demonstrates robust transferability, maintaining superior performance in 10% few-shot and zero-shot forecasting scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.29717v2">Optimizing Expert-Designed Crystal Graph Networks for Band-Gap Prediction with an Autonomous LLM Research Loop</a></div>
    <div class="paper-meta">
      📅 2026-08-12
    </div>
    <details class="paper-abstract">
      Predicting a material's properties from its structure is a central, fast-advancing problem in computational materials science. A decade of work has produced standard public benchmarks and many published machine-learning models for the task (Dunn et al., 2020). The task's fixed metric and these baselines make it a natural setting for autonomous agent research (Karpathy, 2026). On the MatBench band-gap benchmark ($>$100k crystals), a general-purpose coding agent autonomously built the most accurate model trained without external pretraining, ahead of all seventeen expert-designed models reported for the task. A closer analysis shows it reached this by implementing known methods: either already standard in crystal neural-network models, or borrowed from other areas of machine learning. The contributing implementations include element-pair features on each message-passing edge and a crystal space-group embedding. The work not only demonstrates that LLM-agent autonomous research can optimize an expert-designed machine learning model for material property prediction, but also investigates the limitations of such autonomous research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08417v3">Attention is All You Need to Defend Against Indirect Prompt Injection Attacks in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-12
      | 💬 Accepted by Network and Distributed System Security (NDSS) Symposium 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been integrated into many applications (e.g., web agents) to perform more sophisticated tasks. However, LLM-empowered applications are vulnerable to Indirect Prompt Injection (IPI) attacks, where instructions are injected via untrustworthy external data sources. This paper presents Rennervate, a defense framework to detect and prevent IPI attacks. Rennervate leverages attention features to detect the covert injection at a fine-grained token level, enabling precise sanitization that neutralizes IPI attacks while maintaining LLM functionalities. Specifically, the token-level detector is materialized with a 2-step attentive pooling mechanism, which aggregates attention heads and response tokens for IPI detection and sanitization. Moreover, we establish a fine-grained IPI dataset, FIPI, to be open-sourced to support further research. Extensive experiments verify that Rennervate outperforms 15 commercial and academic IPI defense methods, achieving high precision on 5 LLMs and 6 datasets. We also demonstrate that Rennervate is transferable to unseen attacks and robust against adaptive adversaries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.11584v1">EnterpriseRAG: Benchmarking LLM Instruction Adherence and Robustness under Non-Ideal Enterprise Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-08-12
    </div>
    <details class="paper-abstract">
      Enterprise RAG deployments face a critical reliability gap: while LLMs satisfy 80% of individual constraints, only 26.8% of responses meet all requirements simultaneously, revealing a 57-point orchestration gap. Existing benchmarks assume clean retrieval with simple queries, failing to capture production conditions where noisy documents and multi-dimensional constraints coexist. We introduce EnterpriseRAG, a benchmark of 983 expert-validated samples across six domains that systematically simulates three failure modes absent from prior work: retrieval noise, knowledge gaps, and factual conflicts, coupled with complex instructions. Evaluation of 13 state-of-the-art LLMs reveals a severe instruction adherence collapse, where high per-constraint satisfaction masks low holistic compliance. Critical findings expose deep barriers under knowledge gaps and factual conflicts, even with reasoning-enhanced inference, indicating production RAG requires explicit context-aware protocols and calibrated judgment. EnterpriseRAG provides a reproducible foundation for measuring and closing these gaps, directly informing deployment decisions for enterprise-scale RAG systems. We will release the benchmark and evaluation framework upon publication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.11573v1">Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-12
    </div>
    <details class="paper-abstract">
      Achieving effective self-correction, where models verify and correct their own mistakes, remains a fundamental challenge for large language models (LLMs). In this work, we propose Self-Fix Step-DPO (SFS-DPO), a reinforcement learning based, two-stage framework for step-level self-verification and self-correction. The first stage strengthens step-level reasoning via step-level preference optimization, while the second stage explicitly trains models to self-verify and self-correct. We further introduce a teacher-assisted variant, SFS-DPO-R, which incorporates explanatory rationales for error verification to provide stronger corrective signals. Comprehensive in-domain and out-of-domain evaluations across multiple LLMs demonstrate that SFS-DPO and SFS-DPO-R consistently outperform prior step-level training baselines. Our analysis further reveals improvements in self-correction frequency and effectiveness, highlighting the importance of strengthening step-level reasoning for robust performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.01039v4">Evaluation and Hardening of LLM System Instructions Against Extraction via Encoding Attacks</a></div>
    <div class="paper-meta">
      📅 2026-08-12
      | 💬 An earlier version of this manuscript will appear in the proceedings of IEEE Cyber-AI 2026 Conference. Project source code is available at https://github.com/Keysight/LLM-EncodeGuard
    </div>
    <details class="paper-abstract">
      System Instructions in Large Language Models (LLMs) are commonly used to enforce safety policies, define agent behavior, and protect sensitive operational context in agentic AI applications. These instructions may contain sensitive information such as API credentials, internal policies, and privileged workflow definitions, making system instruction leakage a critical security risk highlighted in the OWASP Top 10 for LLM Applications. Without incurring the overhead costs of reasoning models, many LLM applications rely on refusal-based instructions that block direct requests for system instructions, implicitly assuming that prohibited information can only be extracted through explicit queries. We introduce an automated evaluation framework that tests whether system instructions remain confidential when extraction requests are re-framed as encoding or structured output tasks. Across seven common models and 46 verified system instructions, we observe high attack success rates (>0.7) for structured serialization where models refuse direct extraction requests but disclose protected content in the requested serialization formats. We further demonstrate a mitigation strategy based on one-shot instruction reshaping using a Chain-of-Thought reasoning model, indicating that even subtle changes in wording and structure of system instructions can significantly reduce attack success rate without requiring model retraining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.11552v1">Beyond Single-Turn Confidence: Trajectory-Adapted Uncertainty Quantification for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-12
    </div>
    <details class="paper-abstract">
      Uncertainty quantification (UQ) methods for language models are typically evaluated on single-turn outputs, where uncertainty is attached to one generated answer. For LLM agents, however, the unit of observation is an interactive trajectory, where the model can ask clarifying questions, call tools, update state, and make intermediate decisions whose errors propagate to the final outcome. We study whether three common families of single-turn UQ methods transfer to this setting. Across five LLMs and four multi-turn tool-use datasets from BFCL-v4 and $τ^2$-bench, we evaluate white-box scorers based on action-token probabilities, black-box consistency scorers based on resampled trajectories, and reflexive scorers based on model self-assessment of the trajectory. We find that transfer is often useful but uneven. Token-probability scores are highly sensitive to the choice of aggregator used across turns, reflexive scores provide the strongest low-cost baseline in most evaluated settings, and black-box self-consistency is often the strongest UQ family, with trajectory-equivalence and action-set consistency typically ranking highest among its variants. These results suggest that UQ methods developed for single generations should be revalidated at the trajectory level, with careful attention to the consistency measurement, aggregator choice, and computational budget.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.11513v1">Do Influence Tactics Matter? Investigating Prompt Framing Effects in LLM Code Generation</a></div>
    <div class="paper-meta">
      📅 2026-08-11
      | 💬 Accepted for publication in Empirical Software Engineering. This is the accepted manuscript version. 37 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrated into software engineering workflows, helping developers write, debug, test, and maintain code. While prompt wording and structure are known to influence model performance, the impact of psychologically inspired prompt framings remains unexplored. This study investigates whether different psychology-based communication strategies that humans use to persuade or motivate others can lead to more effective prompt framing, which may, in turn, affect LLM behaviour in coding tasks. Drawing on Yukl & Falbe's well-known taxonomy, we operationalized eight influence tactics (like rational persuasion, ingratiation, and exchange) into reproducible prompt templates. These prompt templates were evaluated across five leading open-weight LLMs using two widely adopted benchmarks: LiveCodeBench and SWE-bench Verified. We assessed the resulting code output on four key software quality dimensions: functional correctness, quality, maintainability, and security. Our results show that certain influence-induced prompt framings, particularly those emphasizing urgency, were associated with reduced correctness and security. This work presents the first large-scale empirical study of influence-induced prompt framing in software engineering tasks, offering insights into how linguistic cues may shape LLM outputs. We conclude with practical insights for designing transparent and interpretable human-AI interactions in code generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.11493v1">From Prompting to Behavioral Alignment: Personalized LLM Judges for Recommendation Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-08-11
    </div>
    <details class="paper-abstract">
      Traditional offline recommendation evaluation relies heavily on complex, manually maintained feature pipelines that are difficult to scale. While Large Language Models (LLMs) offer a promising alternative by predicting user engagement directly from raw text logs, empirical analysis in this study identifies a critical failure mode termed bidirectional rationalization. In a zero-shot setting, LLMs are found to convincingly argue for both positive and negative user engagement outcomes on the exact same item with identical evidence, highlighting the unreliability of off-the-shelf LLMs in predicting user engagement. To resolve this, we develop and apply a sequential behavioral alignment framework pairing fine-tuning with preference optimization over paired correct and counterfactual rationales. Evaluated on real-world homepage interaction logs, this aligned reasoning approach achieves a 32.19\% lift in Macro-F1 score over the zero-shot baseline and matches the production feature-engineered baseline. The results demonstrate that behavioral alignment mitigates bidirectional rationalization while delivering human-interpretable reasoning traces without manual pipeline overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.11454v1">Simplifying Requirements Engineering in the Context of the LGPD: An LLM-Based Investigation</a></div>
    <div class="paper-meta">
      📅 2026-08-11
      | 💬 The document consists of 12 pages and includes 2 figures
    </div>
    <details class="paper-abstract">
      Compliance with privacy legislation poses a complex challenge to Requirements Engineering (RE): translating legal norms into software requirements. In this context, this study investigates whether Large Language Models (LLMs) can simplify RE within the framework of the Brazilian General Data Protection Law (LGPD). The proposed approach utilizes current legislation to automatically generate User Stories and Acceptance Test Scenarios. The evaluation results demonstrated high performance, confirming the potential of LLMs to ensure regulatory compliance from the software's inception.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.20895v3">LLM Router: Rethinking Routing with Prefill Activations</a></div>
    <div class="paper-meta">
      📅 2026-08-11
    </div>
    <details class="paper-abstract">
      Existing routers rely on semantic query features or handcrafted features, which often fail to capture model-specific failures or intrinsic task difficulty. We instead route using internal LLM activations, specifically the residual stream. Our key idea, Encoder-Target Decoupling, separates the model that produces the predictive signal (the Encoder) from the model whose correctness is being estimated (the Target), allowing open-weight encoders to predict the performance of closed-source target models. We evaluate layerwise geometric probes, finding that Fisher Separability ($J$) effectively identifies informative layers, supported by Effective Dimensionality ($d_{\mathrm{eff}}$) diagnostics. We then utilize a SharedTrunkNet, a joint multi-output MLP that predicts simultaneous correctness probabilities across candidate models using concatenated prefill features. In our experiments, SharedTrunkNet consistently outperforms semantic baselines. At its best, SharedTrunkNet closes 45.58% of the gap between the strongest standalone model and the oracle while achieving 74.31% cost savings relative to the most expensive model. These results demonstrate that prefill activations provide a robust routing signal, establishing activation-based routing as a high-performance alternative to purely semantic selection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.11434v1">Benchmarking LLM Judges for Mobile Agent Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-08-11
    </div>
    <details class="paper-abstract">
      Mobile agent benchmarks increasingly rely on LLM-based judges to evaluate task completion, yet the reliability of these judges on mobile agent trajectories remains largely unexamined. We introduce MobileJudgeBench, a benchmark for systematically evaluating LLM-as-judge methods on mobile agent trajectories. Our benchmark comprises 931 human-annotated trajectories spanning 6 mobile agent benchmarks, 4 agent models, and 68 apps. We evaluate 6 judge methods (five adapted from SPA-Bench, A3 with two modes, AndroidArena, and AgentRewardBench, plus a simple baseline we design) across multiple LLM backends. Our experiments reveal three key findings. First, a simple baseline judge with sampled screenshots is competitive with, and often exceeds, purpose-built methods, indicating that more elaborate judge pipelines do not consistently improve judge quality; among competitive methods, the LLM backbone is the primary driver. Second, benchmark quality metrics reliably predict real-world judge utility: they correlate with both agent ranking fidelity for evaluation and downstream performance when judges serve as reward signals for on-policy reinforcement learning. Third, failure analysis across two LLM backends uncovers qualitatively opposite failure profiles, one conservative and the other permissive, linked to the backbone's precision-recall characteristics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.11381v1">From Numbers to Judgment: Specialist LLM Agents and Reinforcement Learning for European Listed Real Estate</a></div>
    <div class="paper-meta">
      📅 2026-08-11
    </div>
    <details class="paper-abstract">
      We study whether the localized numerical operations and integrative judgments of financial analysis benefit from the same form of LLM specialization. Larix maps a 16-lens European listed-real-estate analysis framework to eight lens-aligned specialists; we compare a frontier LLM under monolithic versus specialist-decomposed prompting while holding the model, source evidence, task instructions, output schema, and scoring fixed. Across 19 firms spanning seven regulatory wrappers, decomposition improves the numerical-task aggregate by 15.8 percentage points but does not reliably improve, and can reduce, performance on judgment tasks, a pattern stable across four frozen-template dispatches; a single-agent control given the complete framework does not reproduce the numerical gain. Post-training Qwen3.5-9B with GRPO using task-aligned structured rewards then raises the development-split score by 12.0 points and the judgment aggregate by 14.2 points, with gains on all four sub-ceiling tasks; the gains transfer to unseen firms (+15.2 points overall; +40.4 on covenant stress) and to unseen regulatory wrappers (+4.3), with positive transfer on all three anti-memorization splits. Prompt-level decomposition thus improves modular numerical execution, whereas targeted parameter adaptation improves integrative financial judgment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.15122v2">Prestige over merit: An adapted audit of LLM bias in peer review</a></div>
    <div class="paper-meta">
      📅 2026-08-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) play a growing but largely informal role in scholarly peer review. Yet whether LLMs reproduce biases observed in human decision-making remains unclear. We adapt a resume-style audit to scientific publishing, developing a multi-role LLM simulation (editor/reviewer) that evaluates high-quality manuscripts across the physical, biological, and social sciences under randomized author identities (institutional prestige, gender, race). Revealing author identities lowers reviewer rejection recommendations by roughly 25% of the mean rejection rate despite identical content, indicating that status cues beyond paper quality shape outcomes. Institutional prestige is the dominant cue: papers attributed to low-prestige affiliations receive lower quality scores in every field, a penalty that survives family-wise multiple-testing correction at the editor stage. Effects at the rejection margin are smaller and mostly fragile to correction, with one robust intersectional exception. Relative to male authors, female authors at low-prestige institutions receive lower reviewer quality scores and more rejection recommendations than those at high-prestige institutions. To probe mechanisms, we generate synthetic CVs for the same author profiles; these encode large prestige-linked disparities and an inverted prestige-tenure gradient relative to national benchmarks. The results suggest that domain norms and prestige-linked priors embedded in training data shape outcomes once identity is visible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.11348v1">An Empirical Study of Output-to-Input Loops for Black-Box Backdoor Detection in Fine-Tuned Open-Weight LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-11
    </div>
    <details class="paper-abstract">
      Anyone can upload a fine-tuned large language model (LLM) to a public repository and claim it is safe. A backdoored model behaves normally on ordinary inputs until a hidden trigger fires, and a user with no training data, clean reference weights, or the trigger phrase has no clear way to check the model before using it. We introduce and empirically evaluate self-feeding, a black-box test method that feeds a model's own output back as its next input, so the text drifts away from the starting prompt and toward the data the model was fine-tuned on. We test self-feeding against a repeated same-prompt baseline on six open-weight LLMs (3B-15B parameters), each fine-tuned with backdoors spanning eleven attack categories, using twenty ordinary starting prompts and chains of up to ten steps. Self-feeding finds backdoors in five of six models at 92.0\% pooled precision, while the same-prompt baseline succeeds on only one of 120 prompt-model pairs; chains that begin with a joke request, an arithmetic question, or a coffee recipe all reach a trigger within a few steps. Recall per prompt is low (19.2\%), and we show why it still adds up to much higher detection at the model level once several starting prompts are used. We also report where the method falls short: one model was never triggered, and self-feeding produced two false positives that the same-prompt baseline cannot produce. Cutting the chains to four steps keeps every model-level detection at 100\% precision while using 60\% fewer queries. Needing only text-level query access and a way to recognize malicious output, self-feeding offers a cheap first check on a downloaded model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.11343v1">Can Frontier LLMs Match Natively Multimodal Embeddings? A Comparison on Hard-Negative Text-to-Image Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-08-11
    </div>
    <details class="paper-abstract">
      Multimodal retrieval and classification across different types of media, spanning text, images,video and audio, has traditionally relied on dual-encoder models that align visual and textual representations through contrastive learning. The March 2026 release of Gemini Embedding 2, Google's first natively multimodal embedding model to map text, images, video, audio, and documents into a single shared space, raises competition among multimodal retrieval systems. Simultaneously, frontier Large language models (LLMs) have also demonstrated strong visual understanding, raising the question of whether they can serve as effective zero-shot rankers. Our study provides the first direct comparison of native multimodal embeddings against LLM-based visual ranking on Flickr30k. We observe that GPT-4.1 and Claude Sonnet 4.6 perform on par with Gemini Embedding 2. Additionally, once embeddings are precomputed, multimodal embeddings are better suited for low-latency applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.11342v1">Weightless Fine-Tuning: Personalizing LLMs via Logit-Space Transport</a></div>
    <div class="paper-meta">
      📅 2026-08-11
    </div>
    <details class="paper-abstract">
      Supervised fine-tuning (SFT) is a standard approach for adapting LLMs to a target distribution, but in settings such as personalization, where each author requires separate weight access, optimization, storage, and retraining, its costs become prohibitive. We propose Weightless Fine-Tuning (WFT), a training-free decoding-time method that approximates the distributional effect of SFT without weight updates. WFT computes supervised residuals on an author's training sequence and transports them to the current prompt through a cross-prefix transport operator estimated from dropout-induced cross-covariance. The operator captures how a perturbation at one context propagates to predictions at another, replacing gradient-based parameter updates with logit-space corrections. On three LaMP personalization benchmarks, WFT achieves the best average performance across datasets, matches or exceeds SFT on individual tasks, and outperforms other lightweight baselines on average. In a budget-controlled comparison, WFT approaches SFT performance using less than 7% of the effective computation. Logit-level analysis shows a cosine similarity of 0.875 between the logit shifts induced by WFT and SFT over 95% of the next-token probability mass, suggesting that WFT captures the distributional effect of supervised adaptation without modifying model weights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.17244v2">DORA Explorer: Improving the Exploration Ability of LLMs Without Training</a></div>
    <div class="paper-meta">
      📅 2026-08-11
      | 💬 17 pages, 6 Figures, 13 tables
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents for sequential decision-making struggle to produce diverse outputs. This leads to insufficient exploration, suboptimal solutions, and repeated actions. Actions are generated at the sequence level, but existing sampling strategies, such as temperature scaling, introduce diversity at the token level, not at the sequence level. We introduce DORA EXPLORER (Diversity-Oriented Ranking of Actions), a training-free, inference-time algorithm for improving exploration in LLM agents. DORA generates multiple candidate actions, scores them using sequence-level log-probability statistics, and samples an action via a tunable exploration parameter. We first study exploration in the classic Multi-Armed Bandit setting, where DORA substantially outperforms temperature-based sampling. Our main evaluation is on the Text Adventure Learning Environment Suite (TALES), where prompting strategies fail to explore but DORA delivers consistent gains across model families, e.g., 31.43% (ReAct) -> 45.5% (DORA) for Qwen-2.5 7B in TextWorld. Beyond exploration, DORA prevents common failures, such as getting stuck in loops. Our project is available at: https://dora-explore.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.11295v1">Backdoor Decontamination Dynamics in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-11
    </div>
    <details class="paper-abstract">
      Open-weight LLM agents are vulnerable to backdoors installed during fine-tuning, which may be undetectable if the trigger conditions are never met during testing. Assuming defenders do not know the existing trigger, they cannot unlearn it directly. One decontamination strategy is to install a known backdoor (defensive poisoning) then to unlearn it, hoping that the original unknown backdoor is removed as a side effect. However, this procedure has uncertain outcomes: the original backdoor may persist or be erased or rerouted, among other possibilities. We introduce a framework for studying these dynamics in tool-calling agents, decoupling trigger, response, teacher, and fine-tuning method across systematic experiments on AgentDyn. Across 115 experiments, defensive poisoning alone erases around 56% of original backdoors; subsequent decontamination then drives almost all survivors to erasure, confirming that trigger recognition and malicious execution are behaviorally dissociable. Interestingly, our experiments find that malicious backdoors never persist when using different triggers of the same general type as the defensive backdoor when followed by decontamination via unlearning. Co-installing up to four backdoors increases resistance (around 36% erased), yet decontaminating a single known co-resident backdoor collaterally clears 52/60 co-residents (87%). Upon visualizing postdecontamination model internals using J-lens, we confirm that although the decontamination restores benign LLM responses, traces of original trigger awareness persist at intermediate layers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15817v7">A Causal Perspective on Measuring, Explaining and Mitigating Smells in LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2026-08-11
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have accelerated their adoption in software engineering contexts. However, concerns persist about the structural quality of the code they produce. In particular, LLMs often replicate poor coding practices, introducing code smells (i.e., patterns that hinder readability, maintainability, or design integrity). Although prior research has examined the detection or repair of smells, we still lack a clear understanding of how and when these issues emerge in generated code. This paper addresses this gap by systematically measuring, explaining and mitigating smell propensity in LLM-generated code. We build on the Propensity Smelly Score (PSC), a probabilistic metric that estimates the likelihood of generating particular smell types, and establish its robustness as a signal of structural quality. Using PSC as an instrument for causal analysis, we identify how generation strategy, model size, model architecture and prompt formulation shape the structural properties of generated code. Our findings show that prompt design and architectural choices play a decisive role in smell propensity and motivate practical mitigation strategies that reduce its occurrence. A user study further demonstrates that PSC helps developers interpret model behavior and assess code quality, providing evidence that smell propensity signals can support human judgement. Taken together, our work lays the groundwork for integrating quality-aware assessments into the evaluation and deployment of LLMs for code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09924v4">LLMs Encode Their Failures: Predicting Success from Pre-Generation Activations</a></div>
    <div class="paper-meta">
      📅 2026-08-11
      | 💬 Accepted at COLM 2026
    </div>
    <details class="paper-abstract">
      Running LLMs with extended reasoning on every problem is expensive, but determining which inputs actually require additional compute remains challenging. We investigate whether their own likelihood of success is recoverable from their internal representations before generation, and if this signal can guide more efficient inference. We train linear probes on pre-generation activations to predict policy-specific success on math and coding tasks, substantially outperforming surface features such as question length and TF-IDF. Using E2H-AMC, which provides both human and model performance on identical problems, we show that models encode a model-specific notion of difficulty that is distinct from human difficulty, and that this distinction increases with extended reasoning. Leveraging these probes, we demonstrate that routing queries across a pool of models can exceed the best-performing model whilst reducing inference cost by up to 70\% on MATH, showing that internal representations enable practical efficiency gains even when they diverge from human intuitions about difficulty. Our code is available at: https://github.com/KabakaWilliam/llms_know_difficulty
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.27199v2">Forecasting With LLMs: Improved Generalization Through Feature Steering</a></div>
    <div class="paper-meta">
      📅 2026-08-11
    </div>
    <details class="paper-abstract">
      Successful forecasting involves identifying patterns between historical and future states of the world which generalize to future observations. We apply LLMs to a variety of forecasting tasks and inspect their internal states using sparse autoencoders to understand whether they appear to rely on time-specific pieces of knowledge versus generalizable patterns. Our analyses identify features associated with both time-aware reasoning and look-ahead-biased reasoning. We then apply the LLMs to an entirely different domain and intervene on these features. We find that amplifying time-awareness features substantially reduces look-ahead bias on forecasting prompts while preserving general reasoning performance. In contrast, steering the candidate look-ahead-bias features does not produce an effect. These results suggest that interpretable temporal features can be used to causally shift LLMs toward more historically grounded reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20173v2">A Methodology for Selecting and Composing Runtime Architecture Patterns for Production LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-11
      | 💬 26 pages, 2 figures, 6 tables. Companion repo at https://github.com/vasundras/agent-runtime-patterns
    </div>
    <details class="paper-abstract">
      Production LLM agents combine stochastic model outputs with deterministic software systems, yet the boundary between the two is rarely treated as a first-class architectural object. This paper names that boundary the stochastic-deterministic boundary (SDB): a four-part contract among a proposer, verifier, commit step, and reject signal that specifies how an LLM output becomes a system action. We argue that the SDB is the load-bearing primitive of production agent runtimes. Around this primitive, we organize agent runtime design into three concerns: Coordination, State, and Control. We present a catalog of six runtime patterns that compose the SDB differently across conversational, autonomous, and long-horizon agents: hierarchical delegation, scatter-gather plus saga, event-driven sequencing, shared state machine, supervisor plus gate, and human in the loop. For each pattern, we trace its lineage to distributed-systems concepts and identify what changes when the worker is stochastic. The paper contributes a five-step methodology for selecting runtime patterns, a diagnostic procedure that maps production failures to pattern weaknesses, and a failure mode called replay divergence, in which LLM-based consumers of a deterministic event log produce different downstream outputs under model-version or prompt changes. A stylized reliability decomposition separates per-call model variance from architectural momentum, motivating the claim that as model variance decreases, pattern choice and SDB strength become increasingly important levers for long-run reliability. We apply the methodology to five workloads and provide one runnable reference implementation for a 90-day contract-renewal agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.05061v3">No Free Labels: Limitations of LLM-as-a-Judge Without Human Grounding</a></div>
    <div class="paper-meta">
      📅 2026-08-11
    </div>
    <details class="paper-abstract">
      Reliable evaluation of large language models (LLMs) is critical as their deployment rapidly expands, particularly in high-stakes domains such as business and finance. The LLM-as-a-Judge framework, which uses prompted LLMs to evaluate response quality, is appealing due to its scalability, low cost, and strong correlations with human stylistic preferences. However, it remains unclear how accurately these methods can assess response quality in domains where correctness matters more than style. To address this gap, we introduce the Business and Finance Fundamentals Benchmark (BFF-Bench), a dataset of 160 challenging questions and long-form responses authored by financial professionals. These experts subsequently evaluated the correctness of 1,200 responses generated by a diverse set of LLMs on both BFF-Bench and a challenging subset of MT-Bench. With this expert-annotated dataset of judgments (VERDICTS), we analyze the agreement between a suite of automated grading methods and human experts. While we observe that LLM Judges are more reliable than other grading methods, our findings reveal a clear pattern in LLM Judge performance: when not provided with a correct reference, judges show high agreement with human experts only on questions the judges were able to correctly answer themselves. We demonstrate that providing the judges with expert-written references largely mitigates this issue, highlighting the limits of using LLM-as-a-Judge without any form of human verification.
    </details>
</div>
