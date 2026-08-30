# llm - 2026_08

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10753v1">Can LLMs be Used to Simplify Algorithms? Simpler Algorithms for Vertex Coloring and Edge Connectivity</a></div>
    <div class="paper-meta">
      📅 2026-08-11
    </div>
    <details class="paper-abstract">
      Having simple algorithms is important for the practical adoption of new algorithms. However, simplifying existing algorithms is a field that does not usually receive a lot of attention from the theoretical computer science community. It also seems like a task that LLMs might perform well. Thus, in this paper we study how well LLMs can simplify algorithms by evaluating three different LLMs on ten different algorithmic problems. Our study resulted in the discovery of two novel algorithms. The first algorithm is for vertex coloring, and gives a refined bound for the so-called asymmetric palette sparsification proposed by Assadi and Yazdanyar [SOSA 2025] with a very simple proof. The second is a further simplification of the algorithm of Saranurak [SOSA 2021] for deterministically computing a global minimum cut in an unweighted graph using expanders.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10725v1">Rethinking LLM Verification: Evidence Structure, Uncertainty, and Selective Refinement</a></div>
    <div class="paper-meta">
      📅 2026-08-11
      | 💬 Findings Track at the Conference on Empirical Methods in Natural Language Processing (EMNLP 2026)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often rely on shortcuts rather than systematic reasoning, raising safety concerns in medical applications. Allowing models to abstain when uncertain improves reliability but introduces a coverage accuracy tradeoff. We propose a two-stage framework for medical hypothesis verification in multiple-choice settings that manages this tradeoff through targeted ontology grounding, applied only when the model abstains. We show that abstention is not random but reflects genuine uncertainty, with abstained predictions associated with lower confidence. Across two frontier models (GPT-5.5, accessed via the Azure OpenAI API, and DeepSeek-R1), the proposed framework improves question-level accuracy by 9.6 percentage points (82.9% to 92.5%) and hypothesis-level accuracy by 4.2 percentage points (92.0% to 96.2%). Our experiments conducted on MedReason and MedQA show that abstention can be repurposed as a control signal for selective reasoning refinement, achieving knowledge-graph-level performance without explicit knowledge graph construction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10715v1">Most biomedical publications show signs of LLM-assisted writing</a></div>
    <div class="paper-meta">
      📅 2026-08-11
    </div>
    <details class="paper-abstract">
      Over the past several years, LLM-powered chatbots and agents have become widely used as a tool for academic writing. LLM-assisted writing can be valuable by removing language barriers but at the same time causes concerns about misconduct and fraud. To inform policy decisions, it is necessary to monitor the prevalence of LLM-altered texts in scholarly publications. Despite some recent progress in this direction, no existing method can produce reliable estimates. Here we suggest and validate a new unbiased approach to estimate LLM usage in a corpus of texts based on changing word frequencies. We apply our method to the full texts of open-access biomedical papers from Pubmed Central, and show that by the end of 2025, 89% of papers show excess of LLM-associated vocabulary. We also find that LLMs are twice as likely to be used when writing a paragraph in the Discussion section (68%) compared to a paragraph in the Methods section (32%), but even inside the Methods section, the overall prevalence of LLM usage is over 50%. We believe that our estimates are crucial to shape future guidelines and policies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10710v1">LLM Ensemble Fault Classification for Automotive HiL Validation</a></div>
    <div class="paper-meta">
      📅 2026-08-11
      | 💬 8 pages, 3 figures. Accepted at the 23rd Workshop on Model Driven Engineering, Verification and Validation (MoDeVVa 2026), co-located with MODELS 2026. Original submitted version
    </div>
    <details class="paper-abstract">
      Automotive HiL validation generates large multivariate test recordings whose analysis remains challenging due to manual review effort, rule-based limitations, and the need for explainable diagnostic decisions. Recent machine-learning and deep-learning approaches have improved fault diagnosis, but they often require large labelled datasets, generalise poorly across operating conditions, and provide limited insight into their predictions. This paper proposes an explainable multi-LLM ensemble framework for sensor-level fault classification in automotive validation. The framework uses compact evidence representations of fault-injection recordings and combines the outputs of heterogeneous large language models to improve diagnostic robustness, ranking quality, confidence reliability, and interpretability. The approach is evaluated on gasoline-engine and electric-vehicle HiL systems across three driving settings and ten single-fault classes. Among the individual models, Mistral Small~24B provides the strongest overall single-model trade-off, achieving 0.903 Top-1 accuracy, 0.887 MCC, and the lowest Brier score of 0.102. The final Top-3 ensemble combines Mistral Small~24B, Qwen2.5~32B, and Phi-4~14B using confidence-weighted voting, improving the scenario-averaged results to 0.917 Top-1 accuracy, 0.913 macro F1, and 0.902 MCC, while also providing the best calibration among the tested ensemble strategies. A Top-5 ensemble does not improve over the Top-3 configuration, indicating that model complementarity is more important than ensemble size. The results show that coordinated multi-LLM reasoning can support robust, calibrated, and engineer-interpretable fault classification for automotive HiL validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10703v1">Your LLM, Your Style: Behavioral Mode Axes for LLM Behavioral Control</a></div>
    <div class="paper-meta">
      📅 2026-08-11
      | 💬 33 pages, 8 figures. Code and data: https://github.com/lhz191/LLM-Behavioral-Personality
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly act in interactive settings where their behavioral styles affect user experience, safety, and downstream decision making. Existing LLM personality studies largely rely on self-report questionnaires administered in first-person settings, making the resulting profiles sensitive to surface elicitation choices and poorly grounded in concrete model behavior. In this work, we introduce a situated behavioral-data (B-data) framework for studying and controlling LLM behavioral personality. We construct 3,200 contrastive behavioral scenarios spanning 20 behavioral patterns and four prompt registers, grounded in validated psychometric facets such as BFI-2, DOSPERT, and HEXACO. Using this framework, we find that LLMs exhibit stable and model-specific behavioral profiles, while also revealing register-dependent shifts across first-person decisions, advice-giving, and task execution. We then show that these behavioral patterns can be controlled through Behavioral Mode Axes (BMAs), activation-space directions derived from contrastive behavioral traces. Compared with response-derived BMAs, which are more prone to trait drift, thought-derived BMAs more faithfully capture the intended behavioral mechanism and provide cleaner control over situated behavioral styles. Our results suggest that LLM personality-like tendencies are better understood not as abstract self-report traits, but as measurable and controllable behavioral modes grounded in concrete interaction contexts. Our code and data are available at https://github.com/lhz191/LLM-Behavioral-Personality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10698v1">EVIL-Detect for NLPCC 2026 Shared Task 6: LLM-Generated Text Detection</a></div>
    <div class="paper-meta">
      📅 2026-08-11
      | 💬 Accepted by NLPCC 2026 Shared Tasks
    </div>
    <details class="paper-abstract">
      The rapid development of large language models (LLMs) has increased the need for reliable detection of LLM-generated text, especially in realistic Chinese scenarios involving human-written text (HWT), LLM-generated text (LGT), and LLM-refined text (HLT). This paper presents EVIL-Detect, a multi-signal ensemble framework with conflict-aware fusion for NLPCC 2026 Shared Task 6. The system integrates edit-extent regression, zero-shot likelihood-contrast signals, lexical statistics, and conservative text rules. With calibrated decision boundaries and conflict-aware integration, our system improves robustness under strong out-of-distribution shifts, achieving a macro-F1 score of 0.8888 and ranking first in the official evaluation. Our code is available at https://github.com/bbbbhrrrr/evildetect.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10690v1">Can Released LLM Vocabularies Support Token-Level Estimation of Hidden Corpora?</a></div>
    <div class="paper-meta">
      📅 2026-08-11
    </div>
    <details class="paper-abstract">
      Pretraining corpus composition shapes LLM capabilities, but it often remains hidden even when model weights are released. Prior work has inferred corpus mixtures or traced specific token groups from released tokenizer vocabularies; in contrast, we estimate corpus ratios for arbitrary target tokens. We first show that BPE tokenizers trained on different corpora share stable token ID--ratio distributions, motivating distribution transfer from known corpora to a target tokenizer trained on hidden corpora. We then propose Quantile-Guided Density Estimation (QGDE), which approximates this distribution with multiple quantile trends and uses local density weighting to produce token-level estimates. In controlled settings and a realistic setting using the released SmolLM tokenizer, QGDE achieves mean relative errors as low as 3.00% for token-level estimation and 3.08% after aggregation into category-level mixtures. These results suggest that released tokenizer vocabularies provide a useful signal for fine-grained corpus estimation beyond coarse composition inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10669v1">REDAgentBench: Executable Red Teaming and Faithful Measurement of LLM Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-08-11
      | 💬 6 figures, 4 tables. Supplementary material included
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents combine language-based reasoning with external tools to perform complex tasks. Adversarial inputs can exploit interactions between the agent and its environment, causing the agent to violate safety policies during execution. Yet existing evaluations often reduce agent safety to a single attack success rate (ASR), collapsing exposure, execution, observation, and adjudication and potentially conflating actual violations with evidence visibility. We introduce REDAgentBench, an executable framework for autonomous red-teaming and faithful measurement. It derives attacks from explicit safety constraints and associated agent-system vulnerabilities, runs them in isolated service sandboxes, and verifies harmful effects from service receipts and final-state changes. The benchmark contains 1,661 cases across five service surfaces. Across six models and three agent harnesses, macro-average ASR is 65.69%; reported ASR varies with harness and evidence view, while evaluation-context disclosure changes execution behavior. In a state-grounded diagnostic cohort, almost one in five confirmed violations with resolved action anchors occurs after the agent states the relevant constraint or risk, revealing a Recognition--Execution Gap. Finally, a training-free policy reminder reduces confirmed violations by more than 70 percentage points in matched replay. These findings show that executable evaluation can improve safety measurement and identify actionable intervention points.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08932v2">From Indiscriminate to Targeted: Functionally Critical Signal-Driven Assertion Generation using LLMs for Efficient RTL Verification</a></div>
    <div class="paper-meta">
      📅 2026-08-11
      | 💬 11 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Functional verification has become the most time-consuming phase in IC development, and Assertion-Based Verification (ABV) is key to reducing debugging time. However, existing LLM-based assertion generation methods typically pursue indiscriminate verification, aiming for maximal coverage without considering signal criticality, whereas industrial practice demands maximizing coverage with minimal verification cost. Consequently, identifying signals that have the greatest impact on design functionality and error propagation-enabling a shift from indiscriminate to targeted verification-remains a key challenge. To address this, we propose AgileAssert, a key signal-driven assertion generation framework that constructs RTL semantic graphs and identifies the top-K critical signals via a hybrid scoring and selection mechanism, followed by structure-aware RTL slicing to provide the LLM with precise targets and contextual information, thereby guiding LLMs to generate tightly constrained targeted assertions for efficient verification. Evaluated on block- and CPU-level designs, with an average 66.68% reduction in assertions, our approach outperforms three existing SOTA methods, and significantly improving coverage metrics while reducing input token consumption by 64%. In mutation testing, when our approach surpasses existing methods in error detection rate, the average number of assertions used decreases by 72.74%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10621v1">ProbGuard: Calibrated Safety Risk Estimation from LLM Output Distributions</a></div>
    <div class="paper-meta">
      📅 2026-08-11
    </div>
    <details class="paper-abstract">
      Recent research on Large Language Model (LLM) safety has widely adopted guardrails to identify unsafe LLM outputs. Existing guardrails typically formulate safety assessment as a deterministic classification task, mapping a discrete token sequence to a discrete safety label. However, this paradigm has two limitations: First, safety assessment is inherently an uncertain problem, particularly during the early generation state. Second, relying solely on discrete token sequences discards the rich probabilistic information embedded in the LLM output distribution. To address these limitations, we propose the first completely probabilistic architecture-agnostic guardrail \textsc{ProbGuard} to leverage the LLM early output distributional signals for estimating and calibrating the safety probability, thereby enabling early stopping of unsafe ongoing outputs. Specifically, given an LLM's generated prefix distribution, we formulate the safety risk as the unsafe probability of its continued generation dynamics and estimate this risk by Monte-Carlo sampling. Through post-training on the distributional signals and calibrated safety risk, \textsc{ProbGuard} achieves the best calibration performance across all nine model--dataset combination settings, reducing the average Brier score and ECE by 79.6\% and 71.9\%, respectively, over the best baseline. \textsc{ProbGuard} further limits the attack success rate to at most 1\% across six representative jailbreak attacks after observing the LLM early output distributions from only the first ten decoding steps.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.08636v3">InternAgentHarness: A Scalable Synthetic Environment for Enhancing LLM Agentic Abilities</a></div>
    <div class="paper-meta">
      📅 2026-08-11
      | 💬 InternAgentHarness tech report
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly expected to act as generalist agents capable of solving complex real-world problems. Training such agents, however, requires stable and diverse environments that support repeated interaction with stateful, tool-augmented tasks and provide verifiable feedback. Despite recent progress, the development of robust LLM agents remains limited by the lack of realistic, scalable, and executable training environments. We present InternAgentHarness, a scalable synthetic environment for improving the agentic capabilities of LLMs. Built upon the InternBootcamp training framework~\citep{internbootcampv1}, InternAgentHarness instantiates executable agent environments through a four-layer interface that unifies prompt generation, tool execution, interaction control, and reward computation. We further introduce \bootcampcli, an agent harness that automatically converts diverse agentic tasks into a Bootcamp-trainable paradigm. Unlike static benchmarks, InternAgentHarnessmakes evaluation actionable: observed failures can be systematically converted into new synthetic tasks, filtered trajectories, reinforcement learning rollouts, and subsequent re-evaluation under the same executable interface. We instantiate InternAgentHarness on a suite of 10 tasks covering both text-only and vision-based agent scenarios. Starting from Qwen3-VL-30B-A3B-Thinking, both supervised fine-tuning (SFT) and reinforcement learning (RL) on InternAgentHarness substantially improve performance over untuned model. These results suggest that InternAgentHarness provides a practical foundation for scalable synthetic agent environments and enables the continuous improvement of LLM agents through an iterative cycle of evaluation, synthesis, training, and refinement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10545v1">ImpactHO: Importance-Aware KV Cache Transfer for Multi-User Edge LLM Handover</a></div>
    <div class="paper-meta">
      📅 2026-08-11
    </div>
    <details class="paper-abstract">
      Edge LLMs must preserve inference continuity when a user hands over between edge nodes, requiring key-value (KV) cache transfer to the target node. However, simultaneous handovers saturate the backhaul, preventing full cache delivery within the mobility-imposed transfer window. Rather than allocating bandwidth as if all cache entries were equally valuable, we order each user's KV cache by importance and transmit only its most informative fraction, turning token-level sparsity into communication savings. We cast the transfer as a multi-user backhaul allocation problem that maximizes average accuracy across users. Each user's partial-cache accuracy serves as its utility: a sigmoid that fits measurements on the RULER benchmark with $R^2>0.99$ across models and context lengths. Because importance ordering front-loads the high-value entries, the concave region of the accuracy curve spans nearly the entire cache. Our proposed allocator keeps served users within this region, making each per-slot allocation problem convex. The optimum is derived via a closed-form weighted water-filling solution that generalizes information-theoretic water-filling and enables online scheduling. The proposed allocator attains over 93.7% average accuracy in a 500ms transfer window, within 0.5pp of the full-cache ceiling, and reaches 98.2-99.5% of a clairvoyant upper bound.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10532v1">Benchmarking LLM-Guided Control-Plane Policies for Backend Fault Isolation in HAProxy</a></div>
    <div class="paper-meta">
      📅 2026-08-11
      | 💬 43 pages, 6 figures, 15 tables. Submitted to Journal of Network and Computer Applications (Elsevier)
    </div>
    <details class="paper-abstract">
      Static load balancers cannot mitigate a backend that is degraded rather than down: round-robin and least-connections keep routing traffic to a server returning HTTP 500s until an operator intervenes. We ask whether a Large Language Model can replace the static routing policy itself, reading HAProxy and Prometheus telemetry every 10 seconds and isolating faulty servers through guardrailed calls to the HAProxy Data Plane API. On a reproducible benchmark with a persistent structural fault built into roughly one-third of a heterogeneous fleet, we sweep 15 open-weight models across five families (0.35B to 35B total parameters; dense, mixture-of-experts, and efficient-sparse architectures), reasoning modes, fleet scales of 3 to 9 backends, and two routing algorithms, totaling 240 runs. We find a capability threshold near 3B active parameters. Below it, LLM policies are typically unreliable and sometimes worse than no policy; above it, every model, regardless of architecture, saturates near an 88% reduction in client-perceived 5xx errors over the static baseline. The threshold is approximate: Gemma 4 E2B clears it with 2B active parameters, while the dense 3B Granite 4.0 Micro does not. The availability gain has costs. Draining concentrates load onto surviving servers, inflating tail latency 2.6 to 2.8 times, and enabling reasoning multiplies token spend roughly tenfold, overrunning the control interval and degrading effectiveness. The efficient operating point is a supra-threshold model in its cheapest non-reasoning mode, wrapped inside deterministic guardrails.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10503v1">Every Token Counts: Exact Likert-Scale Distributions for Measuring LLM Attitudes and Biases</a></div>
    <div class="paper-meta">
      📅 2026-08-11
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are increasingly deployed as autonomous agents, accurately evaluating their latent values and biases is critical. The NLP community typically evaluates models using large, unstructured benchmarks. While effective for general capabilities, these datasets fundamentally conflate causal mechanisms: even when an aggregate bias is detected, unstructured evaluations cannot disentangle whether it stems from baseline traits, contextual confounders, or complex interactions. To address this, we introduce an analytically exact framework for the controlled behavioral evaluation of LLMs. We bridge human psychometrics with LLM mechanics by resolving gaps in design, measurement, and analysis. First, we replace unstructured prompting with fully crossed factorial experiments to systematically isolate causal main and interaction effects. Second, we eliminate Monte Carlo text sampling noise by operating directly on exact, token-level Probability Mass Functions (PMFs). Third, we derive a multivariate ordinal consensus metric and a distributional ANOVA to process these PMFs analytically. We validate our framework with a case study on consumer ethnocentrism across five LLMs, demonstrating how our approach isolates systemic country-of-origin biases that aggregate benchmarks otherwise obscure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18795v2">Opinion Polarization in LLM-Based Social Networks: Manipulation and Mitigation</a></div>
    <div class="paper-meta">
      📅 2026-08-11
      | 💬 16 pages, 8 figures
    </div>
    <details class="paper-abstract">
      How vulnerable are online social networks to adversaries who seek to amplify opinion polarization by manipulating opinions, and how difficult is it to mitigate such manipulation? Existing studies have examined this question using mathematical models of opinion dynamics. While these models offer valuable theoretical insights, they rely on simplified assumptions about interactions, message content, and opinion updates, limiting the adversarial strategies they can capture and the applicability of their findings to real-world settings. Large language model (LLM)-based simulations provide a richer alternative: agents can be assigned diverse personas, communicate through natural language, and respond to persuasive or adversarial content in a context-dependent way. This enables the study of manipulation strategies that are difficult to represent using classical mathematical models. To the best of our knowledge, this study provides the first systematic analysis of polarization amplification and mitigation in an LLM-based simulated social network framework. In our framework, LLM agents with diverse personas interact over a social network by exchanging natural language posts and updating their opinions accordingly. We show that even an adversary with a limited manipulation budget can considerably increase polarization. We then study two classes of defense mechanisms: reactive mitigations, which assign specific users to actively counter manipulation, and proactive interventions, which increase resistance through general mechanisms not tied to particular users. Our results show that although these mechanisms reduce the impact of adversarial attacks, they generally do not restore the network to its baseline polarization state. These findings suggest that neither approach fully overcomes the vulnerability of the network, highlighting the potential risk of such attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10492v1">INSIDE the Student's Mind: Jointly Modeling Latent Reasoning and Action in LLM Student Simulators</a></div>
    <div class="paper-meta">
      📅 2026-08-11
      | 💬 Accepted at the Conference on Language Modeling (COLM) 2026
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based simulators often reproduce observable actions but fail to capture the underlying reasoning behind them. In education, where student simulation is increasingly used for various applications such as evaluating tutoring systems, this gap is especially pronounced. Two students may submit identical submissions for entirely different reasons. We present INTERNAL STUDENT DIALOGUE (INSIDE), a student modeling framework that fine-tunes LLMs not only to act like students but also to think like them. INSIDE generates internal dialogue grounded in Bloom's Taxonomy across cognitive, affective, and action dimensions, and fine-tunes models on paired think traces and actions. We baseline against different prompting frameworks and evaluate on two axes: fidelity of simulated actions and quality of generated internal dialogue. Our evaluations show that INSIDE improves simulation fidelity in both action fidelity, matching code generation of real students, and reasoning alignment, achieving the highest alignment across models up to 57.9%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10483v1">Predicting Space Groups of Double Perovskites by LLM with Dynamic Few-Shot Learning</a></div>
    <div class="paper-meta">
      📅 2026-08-11
      | 💬 46 pages, 6 figures, Supplementary Information included(24 pages)
    </div>
    <details class="paper-abstract">
      Double perovskites (DPs) offer broad compositional tunability, but predicting the space groups (SGs) of stable structures remains difficult because available datasets are often strongly imbalanced toward dominant SG classes. We refer to dominant SG classes as major SGs and underrepresented classes as minor SGs. We introduce Dynamic and Diversity-enhanced Few-shot Retrieval and Rule-Guided Inference for Space-Group Prediction (DyRIS), an LLM-agent-based framework that predicts ranked SG candidates from a given DP composition. DyRIS uses diversity-enhanced dynamic few-shot prompting to retrieve relevant in-context examples while limiting the dominance of frequently represented SGs. It further incorporates rule-guided inference based on B/B' cation ordering, quantitative indicators, and major-SG bias control to refine and rank the final Top-3 SG candidates. We evaluate DyRIS on 3,528 thermodynamically filtered DP entries and compare it with composition-based and descriptor-based baselines. At a training-data ratio of 0.5, DyRIS achieves competitive overall accuracy while obtaining the best Overall Top-1 macro-F1 score and the best performance across all Minor-SG metrics. DyRIS improves Minor-SG Top-1 accuracy by 3.26 percentage points relative to CrabNet and achieves higher Minor-SG Top-3 accuracy than the strongest PyCaret-based baseline. Ablation studies show that diversity-enhanced retrieval, quantitative indicators, major-SG bias control, and B/B' ordering information each contribute to prediction performance. Additional experiments show that the final rule-guided inference step is not easily replaced by conventional classifier- or ranker-based models. These findings demonstrate the potential of combining retrieval-based LLM reasoning with crystallographic domain knowledge for SG prediction in imbalanced materials datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10480v1">Multi-Granular Rationale-Guided Molecular LLM for Property Prediction</a></div>
    <div class="paper-meta">
      📅 2026-08-11
      | 💬 16 pages, 5 figures, 18 tables. Code: https://github.com/skku-aihclab/MR-MoL
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are widely applied across chemical tasks, such as molecular property prediction, which underpins drug discovery. Molecular LLMs represent a molecule through several modalities, notably a 1D SMILES sequence or a 2D molecular graph. Both encode molecular information implicitly, so the contribution of individual substructures remains opaque. Retrieval and augmentation methods add context, but from external sources. However, the cues chemists reason over are the internal substructures that drive a property up or down. We propose MR-MoL, a multi-granular rationale-guided molecular LLM that supplies this evidence directly. A fine-tuned GNN scores each substructure through masking, and the most influential ones are serialized as a ranked, direction-tagged rationale that the LLM reads alongside the SMILES sequence and molecular graph. The rationale spans three levels of granularity: Murcko scaffolds with their side chains, BRICS fragments, and functional groups. This is, to our knowledge, the first method to expose GNN-derived attributions to an LLM as evidence for property prediction. On eight MoleculeNet tasks, MR-MoL achieves the best overall results among generalist models and narrows the gap to specialist models tuned for each task. Five diagnostics further confirm that the model reads the rationale rather than merely benefiting from its presence. Its direction, rank, and substructure each shape the prediction, and its attributions reproduce known structure-property relationships.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10462v1">Calibrating Post-Training Feature Shifts for LLM Data Contamination Detection</a></div>
    <div class="paper-meta">
      📅 2026-08-11
      | 💬 14 pages, 7 figures. The first two authors contributed equally
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are trained on massive and largely undisclosed corpora that may contain copyrighted or privacy-sensitive content. Data contamination detection (DCD) therefore aims to determine whether a given text is a member of the pre-training corpus of a target LLM. Recent state-of-the-art DCD methods follow a feature-based paradigm that derives membership features from the input text and the corresponding model output. However, most modern LLMs undergo post-training, such as instruction tuning, preference optimization, and reasoning-oriented training, which can alter model outputs and shift the corresponding membership features, thereby reducing the separability between members and non-members. To address this problem, we propose CalibDCD, a broadly applicable calibration framework for feature-based DCD methods, comprising (1) Multi-View Shift Detection, which identifies recurring feature shifts associated with post-training, and (2) Bounded Feature Correction, which selectively mitigates their influence on membership prediction. Specifically, Multi-View Shift Detection evaluates controlled prompt variants on known non-member texts and consolidates the most informative views to identify recurring feature shifts. Bounded Feature Correction selectively adjusts feature components aligned with the detected shifts and controls the correction extent to preserve useful detection information. Experiments show that CalibDCD consistently improves existing feature-based detectors, with gains of up to 7.0% in AUC and 15.0% in TPR@5%FPR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10459v1">MD-ProTector: Positioning Multiple Data-Driven Prototypes for LLM-Generated Text Detection</a></div>
    <div class="paper-meta">
      📅 2026-08-11
    </div>
    <details class="paper-abstract">
      As LLM-generated content becomes more sophisticated, detection systems for distinguishing those texts from human-written text must operate at scale while handling diverse writing styles, domains, languages, and generator models. Input-only encoder detectors are suitable for practical deployment setting, but standard binary classification supplies only the class label and does not explicitly organize the substantial variation within either class. We propose MD-ProTector, which represents each class with multiple trainable reference vectors in the encoder embedding space, referred to as prototypes. These prototypes provide separate decision boundaries for different groups of texts within the same class. However, adding multiple prototypes alone does not determine which variation each prototype should represent. MD-ProTector addresses this problem with Prototype Positioning loss, which separates class-level structure from the within-class variation that differentiates individual prototypes. Evaluated across five settings from three large-scale benchmarks covering domain, generator, language, and adversarial variation, MD-ProTector achieves the highest AvgRec on MAGE CDCM and RAID and the highest AUROC and lowest FPR95 on RAID among the compared encoder-based methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10441v1">Detecting an Effect Is Not Learning to Act on It: A Reward-SNR Floor for LLM Acquisition Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-11
    </div>
    <details class="paper-abstract">
      Many pipelines can pay a per-example cost to acquire an auxiliary, model-derived observation -- an LLM's structured reasoning, a slow oracle, an expensive measurement -- and then must decide when the acquired signal is worth using. Our thesis is a distinction that is easy to miss: detecting that such a signal helps on average is not the same as learning to act on it per instance, and a reward-SNR floor governs when the second is even possible. Even when the signal is faithful and an in-sample oracle picking the top-b examples by realized reward shows a sizable apparent gain, no deployable policy can learn when to acquire it: across per-impression, cluster, regime, and uplift-tree granularities, learned routing never beats random, and a matched-moment noise placebo reproduces >=100% of the oracle's apparent gain -- the apparent "learnable structure" is order statistics of noise. We explain this with one distinction, detecting a mean effect vs. learning a per-instance acquisition policy, and a reward-SNR detectability floor: routing is estimable offline only if the reward SNR rho clears rho*(N) ~= 2.8/sqrt(N), with a positive control confirming a true low-SNR limit rather than a broken pipeline. As a concrete instantiation we introduce Structured Hypothesis Embeddings (SHE): a frozen LLM turns a user history into ranked, confidence-scored, evidence-grounded intent hypotheses, fused into a recommender. On three public datasets (MIND, REES46, Amazon-Beauty), SHE is faithful and calibratable, yet its value is backbone- and regime-conditional (significant over an ordered GRU, +0.0114, 95% CI [+0.0030, +0.0209], but a global redundancy gap indistinguishable from zero), and learned acquisition collapses at every granularity because all three datasets sit below the floor. The realizable unit is a design-time regime gate, not a per-instance policy. We release code and a one-command reproduction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.18401v5">CAPO: Critic-Guided Action-Aligned Policy Optimization for Advancing LLM Agent Capabilities</a></div>
    <div class="paper-meta">
      📅 2026-08-11
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become a key technique for improving the agentic capabilities of large language models (LLMs). Although critic-free methods such as GRPO are increasingly popular, we argue that critic-based methods remain well suited to long-horizon agentic tasks because their critic models can assess each state and assign credit to different decisions. However, representative critic-based methods such as PPO still organize value estimation, credit assignment, and policy updates at token granularity, whereas the environment state changes only after the agent completes a full action. In this work, we propose CAPO, a critic-guided action-aligned policy optimization method for advancing LLM agent capabilities. CAPO introduces two action-aligned designs. First, it estimates the state value at the action boundary using a critic model and assigns the corresponding advantage to the complete action rather than individual tokens. Second, CAPO introduces a novel action-aware policy ratio that aggregates token ratios within each action with length calibration for consistent updates across actions of different lengths. These two designs effectively achieve fine-grained credit assignment and consistent policy updates while remaining compatible with token-level gradient computation. Experiments across multi-hop QA, academic paper search, and text-world action tasks show that CAPO consistently outperforms representative RL methods. Further analyses validate both designs and reveal how action-aligned optimization improves multi-turn RL training. We hope this work provides a practical path for training more capable LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.02621v2">Reinforcement Learning-based Semi-supervised Knowledge Distillation with LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2026-08-11
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) substantially improves the reasoning capabilities of language models, but most existing RL fine-tuning approaches rely entirely on ground-truth verifiable rewards and thus labeled datasets with verifiable answers. To overcome this, we propose a RL framework for reasoning distillation that leverages continuous, LLM-based rewards. Our method employs an efficient mechanism that computes a continuous CoT reward (CCR) directly from a single-token logit of a judge LLM, evaluating the student model's reasoning trajectory. This formulation provides an effective and scalable online training signal that can be applied to large volumes of unlabeled data. We demonstrate that, when paired with a strong judge, simply using CCR achieves performance comparable to that of ground-truth or pseudo-label verifiable rewards, and even surpasses them as the amount of unlabeled data increases. Furthermore, we find that combining them in a semi-supervised setup is highly synergistic: verifiable rewards help stabilize the CCR, while CCR improves the generalizability of verifiable rewards to related tasks. We also provide a comprehensive empirical comparison of various reward sources across multiple model architectures and dataset sizes. Our results show that this semi-supervised approach consistently enhances mathematical reasoning, yielding an absolute improvement of 5-10% on multiple reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10414v1">How Robust Are LLMs to Vietnamese Dialects?</a></div>
    <div class="paper-meta">
      📅 2026-08-11
      | 💬 8 pages, 3 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are typically evaluated on standard written Vietnamese, yet everyday communication frequently involves regional dialects that preserve meaning but differ in surface form. Existing Vietnamese dialect work largely addresses this issue through dialect-to-standard normalization instead of measuring how the model fails under Vietnamese dialectal inputs. To address this gap, we present the first systematic evaluation of LLM robustness to Vietnamese dialect variation across multiple tasks, quantifying performance degradation and failure patterns. We introduce VialectBench (Vietnamese Dialects Benchmarking), a controlled benchmark for testing whether model decisions remain stable across six Vietnamese dialect groups. VialectBench contains 400 Standard Vietnamese source instances and 2,400 human-written dialectal rewrites spanning emotion recognition (ER), natural language inference (NLI), question answering (QA), and multiple-choice question answering (MCQA). Dataset evaluation with a fixed reference language model shows that the dialectal rewrites induce a measurable model-relative likelihood shift while remaining nearly equal in length to their Standard counterparts. Across ten instruction-tuned models, dialectal inputs reduce average performance by 2.82%, and no evaluated model is fully dialect-invariant. All four tasks are affected, with QA showing the largest average degradation. Robustness also varies substantially across dialect groups: PNT3 and PNT2 cause the largest average performance drops, at 6.17% and 4.73%, respectively, whereas PNB slightly improves average performance by 0.42%. The Central dialect group (PNT1-PNT4) also yields the highest average harmful-flip rate across all models, at 6.54%. These findings show that strong performance on Standard Vietnamese does not guarantee reliable behavior under meaning-preserving regional variation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.09740v2">Security Tests as Executable Specifications for LLM Code Generation: Benefits, Trade-offs, and Coverage Limits</a></div>
    <div class="paper-meta">
      📅 2026-08-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can generate functionally useful code that remains vulnerable, while security-focused interventions may break intended behavior. We investigate security tests as executable specifications both before generation and during iterative repair. We develop SecTDD, a controlled test-feedback scaffold that separates three factors: whether tests are shown upfront, whether failed executions trigger revision, and how failures are selected and represented. The evaluation uses behavior-partitioned visible and hidden tests and byte-identical initial candidates for repair comparisons. Across 2,705 trajectories, 31 task instances, three secure-code benchmarks, 16 CWE categories, and two model families, showing all visible tests upfront increases hidden functional-and-security joint success by 19.3 percentage points on average, but improves only seven of nine benchmark-model conditions and harms two. In shared-candidate comparisons, structured feedback repairs 80 initially unsuccessful candidates with no joint regressions; fixed raw feedback repairs 83 but causes three regressions. Structured and raw feedback are otherwise nearly indistinguishable head-to-head (six wins, six losses, and 453 ties). Candidates that pass all visible tests still fail hidden behavior families under every common regime. These results show that executable feedback can repair secure-code generation, but its benefits depend on the model, task, feedback entry point, and especially test coverage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10385v1">Persona Conditioning as an Assessor-Sensitivity Probe for LLM-Based IR Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-08-11
      | 💬 Accepted at CIKM 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as relevance assessors in information retrieval (IR) evaluation, raising questions about how assessor framing affects judgment reliability and downstream system comparison. We study persona conditioning as a diagnostic mechanism for exposing LLM assessor sensitivity. Using task-oriented personas drawn from two complementary sources (PersonaHub and NVIDIA Nemotron-Personas-USA), we instantiate five assessor roles emphasizing intent interpretation, domain expertise, contrastive judgment, evidence verification, and global search-quality assessment, compared with a standard UMBRELA baseline. Across six LLM backbones on TREC DL20 and RAG24, our analyses reveal structured rather than uniform assessor sensitivity. Judgments usually remain close to the baseline while shifting assessment strictness, evidential threshold, or interpretation emphasis rather than producing widespread relevance reversals. At the system level, high-capacity models preserve system-ranking agreement, while smaller models amplify persona-induced instability. Local rank-displacement analysis shows sensitivity concentrates on particular retrieval systems and system types, especially neural ranking/reranking systems on DL20 and RAG-oriented pipelines on RAG24. Persona source matters less than assessor role and model capacity. These findings position persona-conditioned judging as a controlled sensitivity probe for stress-testing LLM-based IR evaluation pipelines and identifying systems whose evaluation outcomes are sensitive to assessor framing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05353v2">Evidence Lock Before Commitment: A Frozen Interface Degrades LLM-as-Judge Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-08-11
      | 💬 Withdrawn due to an error identified in the code during debugging. The error affects the reported results
    </div>
    <details class="paper-abstract">
      LLM judges are often asked to extract criteria and evidence before choosing between candidate answers. This workflow assumes that the intermediate record preserves the information needed for a later verdict. For reasoning-capable models, visible field order does not reveal internal decision order, so we test an observable alternative: persist the evidence in one call and make it the exclusive input to the next. Across 24,000 judgments over HelpSteer3, FeedbackQA, and CoVal, we compare standard pairwise judging, structured one-call judging, two-call evidence locking, and three-call pointwise locking with Claude Sonnet 4.5 and GPT-5. Evidence locking reduces agreement with released human preferences by 4 to 6 percentage points and increases answer-order inconsistency by 8 to 10 points relative to structured one-call judging. Pointwise locking is also harmful, while structured evidence elicitation remains close to standard judging. The result holds for both judges and all three datasets. Persisted evidence can support auditability, but it should not replace the source answers at decision time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.07215v3">The CodeInverter Suite: Structure- and Data-Aware Binary Decompilation with Efficient LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-11
    </div>
    <details class="paper-abstract">
      Binary decompilation plays a vital role in various cybersecurity and software engineering tasks. Recently, end-to-end decompilation methods powered by large language models (LLMs) have attracted increasing attention for their ability to generate highly readable source code with minimal human intervention. However, existing LLM-based methods still struggle with reconstructing program structure and logic, achieving accurate data recovery, ensuring data security and privacy, and maintaining computational efficiency. To address these challenges, we propose the CodeInverter Suite, with three main pieces: (1) the CodeInverter Workflow (CIW) is a novel prompt engineering method that incorporates control flow graphs (CFG) and explicit data mappings to enhance structure reconstruction and data recovery during decompilation; (2) building upon CIW, we construct the CodeInverter Dataset (CID), a large-scale domain-specific dataset containing 8.69 million samples enriched with CFGs and data mapping information; (3) we develop CodeInverter Models (CIMs), two lightweight LLMs with 1.3B and 6.7B parameters, enabling efficient inference in privacy-sensitive and resource-constrained environments. Extensive experiments on two benchmark datasets demonstrate that CIW significantly enhances the decompilation performance of various LLMs, with average improvements of 13.07% in re-executability and 23.94% in re-compilability. For our proposed decompilation model, CIM-6.7B achieves state-of-the-art performance in terms of re-executability and readability, outperforming existing LLMs-even with over 100 times more parameters-by an average of 11.03% and 6.27%, respectively.
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
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10315v1">Is This Your Final Answer? Cross-Contextual Consistency as a Measure of LLM Credibility</a></div>
    <div class="paper-meta">
      📅 2026-08-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are powerful black-box systems, making it difficult to discern whether their answers reflect stable internal beliefs or superficial pattern matching. We identify cross-contextual consistency as an underutilized behavioral property of LLMs: a credible answer should remain stable when the same task is placed under topic-aligned, content-neutral contextual variation. Building on this intuition, we operationalize Cross-Contextual Consistency (C3) by comparing model generations under original and perturbed prompts. Across 26 models and six benchmarks spanning reasoning, factuality, and code generation, we find that answers with smaller cross-contextual shifts are more likely to be correct or factual. We demonstrate that C3 provides a complementary axis of evaluation and can serve as a benchmark usefulness diagnostic, identifying which portions of a benchmark remain informative even when aggregated scores are widely considered "saturate".
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10314v1">Does the way we write a theory change the program an LLM builds from it? A prospective randomized study of renderer format in LLM theory-to-program translation</a></div>
    <div class="paper-meta">
      📅 2026-08-10
      | 💬 112 pages, 2 figures; includes supplementary material and five reproducibility annexes. Data: https://doi.org/10.5281/zenodo.21876518. Software: https://doi.org/10.5281/zenodo.21879576
    </div>
    <details class="paper-abstract">
      A verbal theory does not run: translating it into an executable model requires choices about variables, interventions, and interactions. We tested whether the presentation of otherwise identical theoretical content systematically changes the programs produced by large language models. In a prospective preregistered randomized experiment, 16 independent assignment bits allocated 32 paired renderer slots between a structured intervention contract and connected prose. Two pinned LLM snapshots translated five anonymous theoretical accounts, yielding 320 preauthorized single-shot programs in a frozen sparse quadratic language. A deterministic evaluator measured atomic finite-difference responses (H1) and mixed interaction responses (H2). Two primary endpoints assessed cross-model matched-distance reduction and closed-set same-account identifiability, with exact randomization inference and 27 preregistered support criteria evaluated across signed-linear and magnitude-rank pipelines. Both H1 and H2 returned the registered verdict NOT_SUPPORTED; only 19 of 108 criterion evaluations passed. Same-account identifiability remained near chance (AUC 0.469-0.523, against a registered 0.80 threshold). One H2 matched-distance endpoint moved and survived multiplicity correction in the signed-linear pipeline, but the corresponding magnitude-rank result missed the registered effect-size floor, so it did not satisfy the joint support rule. Thus renderer format did not produce the uniform, family-invariant, classifiable behavioral geometry predicted in advance. The result places a concrete boundary on specification-format effects in LLM theory-to-program translation and provides a fully auditable randomized design, datasets, and software for studying executable formalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10281v1">From Prompt Injection to Web Exploitation: Revisiting Classic Vulnerabilities in LLM-Integrated Applications</a></div>
    <div class="paper-meta">
      📅 2026-08-10
    </div>
    <details class="paper-abstract">
      Large Language Models are increasingly integrated into web applications through chatbots, tool-calling pipelines, and agentic workflows. In these systems, user input may influence not only generated text, but also backend actions such as database queries, HTTP requests, file operations, template rendering, or API calls. This paper introduces LLM-mediated web attacks, a class of attacks in which attacker-controlled input is transformed by an LLM-integrated application and then reaches traditional web-application sinks. We systematize this attack surface through representative LLM2X variants, including LLM2SQLi, LLM2XSS, LLM2SSTI, LLM2CommandInjection, LLM2IDOR, LLM2CSRF, LLM2XXE, and LLM2SSRF. Our analysis shows that the LLM usually does not create the underlying vulnerability itself; rather, it acts as a mediation layer, and in some tool-enabled settings as a confused deputy, carrying attacker influence into components that trust model-generated or model-influenced content. As an experimental case study, we implement TicketOracle, a Flask-based LLM-integrated web application for evaluating LLM2SSRF across five attack scenarios and seven LLMs. Our results show substantial variation in susceptibility across models, suggesting that exploitation depends both on insecure application architecture and model-specific behavior. We conclude with mitigation strategies across the prompt, model, application, and network layers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10268v1">Toward Human Rights Benchmarking for LLMs: A Pilot Methodology</a></div>
    <div class="paper-meta">
      📅 2026-08-10
      | 💬 Best paper, honorable mention, at the ICML 2026 Workshop on AI4Law, Seoul, South Korea
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly mediate legal determinations over what human rights are realized, and how. Yet, no evaluation benchmark exists to assess whether they can reason correctly about human rights law. To this end, we report our efforts to develop a robust and scalable methodology for creating HumRightsBench: the first expert-validated, scenario-based benchmark for evaluating reasoning grounded in the obligation structure of international human rights law. We adapt the IRAC framework for legal reasoning to better suit the unique reasoning patterns of human rights work (substituting P, "proposing remedies," for C, "legal conclusion," yielding IRAP) to structure our evaluation heuristics. We also produce a pilot series of authentic scenarios designed to implicate the many dimensions of real-world human rights issues and annotated by human rights lawyers and professionals across the world. Ultimately, we find that model accuracy scores range considerably across legal reasoning tasks (overall model performance ranges from 0.339 to 0.577, task min-max ranges from 0.025 to 0.774), which strongly implies that HumRightsBench is a capable instrument for advancing this emerging subfield of AI evaluations science at a critical moment in its evolution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10262v1">Not a Monolith: Lab-Level Divergence in the Cooperative Equilibria of Chinese Frontier LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-10
      | 💬 9 pages, 8 tables. Companion study to arXiv:2605.29874, under a fixed-converter design. Code and replication package: https://github.com/arqFranciscoLeon/evollm (archived: https://doi.org/10.5281/zenodo.20248614)
    </div>
    <details class="paper-abstract">
      Does the cooperative bias documented for Western frontier LLM agents extend to a different alignment lineage, and should the Chinese models that embody it be treated as a single bloc or as distinct laboratories? We study four frontier-tier Chinese models - DeepSeek V4 Pro, Qwen3-Max, Kimi K2.5 and GLM-5.1 - in an evolutionary Iterated Prisoner's Dilemma, under a design that removes a confound present in prior work. Rather than letting each model convert its own natural-language strategies into code, which entangles strategic disposition with coding ability, we hold the converter fixed (GPT-5.4 Mini) across all labs, so every cross-lab comparison is a comparison of generation alone. We run the full protocol: all-play-all tournaments and a Moran process at n=500 runs per condition, across three prompt styles and four population regimes. Two pre-registered hypotheses are evaluated. H6 (not monolithic) is supported: the four labs differ significantly in aggressive-equilibrium proportion, P_A running from 1% for Qwen3-Max to 9% for DeepSeek V4 Pro, with four of six pairwise comparisons surviving Holm-Bonferroni. The spread across the four labs (P_A range 8pp) is larger than the difference between the Chinese and Western ecosystems' mean P_A (5.0% vs 5.0%): on this measure, within-ecosystem variation exceeds the East-West gap. H5 (cooperative-bias generality) is consistent but qualified: a cooperative plurality holds in 6 of 12 lab-prompt combinations against the 9 of 12 reported for Western models, a difference we do not treat as firm, since the count rests on Cooperative-Neutral near-ties and rises to 9/12 under an alternate converter in our pre-registered robustness check. The lab, not the ecosystem, is the unit at which cooperative disposition is set; treating "Chinese models" as a monolith is not supported by the evidence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06227v3">Automated Data Enrichment using Confidence-Aware Fine-Grained Debate among Open-Source LLMs for Mental Health and Online Safety</a></div>
    <div class="paper-meta">
      📅 2026-08-10
    </div>
    <details class="paper-abstract">
      Real-world indicators play an important role in many Natural Language Processing (NLP) applications, such as life events for mental health analysis and risky behaviours for online safety, yet labelling such information is often costly and/or difficult due to its multi-label and dynamic nature. Large Language Models (LLMs) show promising potential for automated annotation, but the multi-label setting remains challenging. In this work, we propose a Confidence-Aware Fine-Grained Debate (CFD) framework that simulates human collaborative annotation using fine-grained communication to better support automated multi-label enrichment. We introduce two expert-annotated resources: life-event and symptom annotation for a mental health well-being dataset, and a new online safety sharenting dataset. Experiments show that CFD achieves the most robust enrichment performance across tasks, and that the benefit of fine-grained confidence is influenced by its quality and variability. We further evaluate training-free strategies for incorporating enrichment indicators into downstream tasks and show that automated enrichment consistently improves performance, by up to 9.9 Macro-F1 points, with the most effective integration format depending on how the indicator relates to the downstream objective.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10258v1">TAF-MED: Multi-Turn Safety Refusal Collapse in LLMs Under Declared Self-Treatment Intent</a></div>
    <div class="paper-meta">
      📅 2026-08-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly provide conversational health information that may influence treatment decisions, yet existing benchmarks do not isolate whether medication-safety boundaries persist across follow-ups after explicit self-treatment intent. We introduce TAF-MED, a physician-reviewed benchmark of 500 fixed three-turn scenarios, and evaluate eight LLMs across 4,000 conversations. A rubric-based automated judge labelled responses as SAFE, LEAKY, or UNSAFE, and two physicians independently annotated a model-balanced random subset of 400 conversations. We assessed unsafe guidance, collapse after a strictly SAFE initial response, and model-ranking stability. Overall, 71.6% of conversations contained an UNSAFE response, and 61.4% of those beginning with a strictly SAFE response later collapsed to UNSAFE; model-level collapse rates ranged from 24.4% to 96.2%. Four of 28 model pairs reversed order between initial unsafe and collapse rates. Automated labels achieved 94.3% agreement with the adjudicated physician reference ($κ= 0.895$). These findings show that first-turn safety is an incomplete proxy for conversational safety persistence and motivate evaluation across complete dialogue trajectories. We will release TAF-MED on Hugging Face to support reproducible research on multi-turn medical safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10257v1">GenRec: An LLM-Backed Recommendation Ranker at Netflix</a></div>
    <div class="paper-meta">
      📅 2026-08-10
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are reshaping recommender systems by enabling richer modeling of users, content, and context directly in natural language. At Netflix, we are exploring this direction through GenRec, an LLM-backed recommendation ranker built on top of an in-house foundational LLM. GenRec follows a two-phase framework: Phase 1 adapts an open-source LLM to Netflix data, developing deep understanding of the catalog and member behavior while balancing capabilities such as content understanding and instruction following. Phase 2 post-trains this foundation model with recommendation-ranking specific data, labels, and reward signals, aiming to align the ranker with business requirements and long-term member satisfaction. This paper focuses on Phase 2 and the transition from a traditional discriminative ranker with thousands of engineered features to an LLM-backed ranker driven by verbalized user histories and context. We describe our design for input verbalization and context engineering, post-training data construction, reward integration, model architecture, and a cost-constrained serving design based on a prefill-only inference approach. We report results from a large-scale A/B test comparing GenRec against the current production ranker model, where we show that a GenRec model trained with substantially fewer Phase-2 labeled training examples and input signals can achieve statistically significant gains in offline and online metrics. We discuss how LLM-backed recommenders could shift the recommendation paradigm: from feature engineering to context engineering, and from bespoke architectures to shared foundation backbones. We also outline practical lessons for serving such systems under real-world resource constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10239v1">Beyond Detection: Evaluating Defensive LLMs Against AI-Generated Social Engineering in Live Turn-by-Turn Interaction</a></div>
    <div class="paper-meta">
      📅 2026-08-10
    </div>
    <details class="paper-abstract">
      Generative AI makes social-engineering attacks more fluent, adaptive, and scalable, increasing the need for LLM-based de- fenders that can protect users during ongoing interactions. We ask whether such defenders identify the structural source of risk or merely react to surface cues. We formalize trust-chain localization: identifying whether an interaction fails at actor authority, asset control, verification sufficiency, or transaction path. We construct a controlled 300-case online-housing corpus spanning 20 scenario families, legitimate cases, four structural failure modes, and three surface conditions. Five defender models are evaluated on the same corpus in state- ful turn-by-turn and one-shot static settings, yielding 1,500 model-case evaluations per protocol and 3,000 in total. No model produced explicit unsafe compliance, yet defensive effectiveness varied sharply: intervention rates ranged from 0% to 96.3%. Protective action and correct structural localization were frequently decoupled, with models sometimes intervening while identifying the wrong trust component or recognizing a structural failure without taking protective action. Asset-control failures were a major localization bottleneck, surface sensitivity varied across models, and live-static differences were model-dependent. These findings show that safe-looking behavior alone is insufficient; live scam resistance must separately measure intervention, timing, structural localization, and false-positive behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10213v1">VeriFin: A Neurosymbolic Framework for Verifying LLM-Generated Financial Claims</a></div>
    <div class="paper-meta">
      📅 2026-08-10
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      Large language models often produce plausible numerical claims from financial filings while using the wrong reporting period, unit, line item, or formula. Verifying such claims requires more than rechecking arithmetic: a verifier must ground the relevant facts, establish the authorized calculation, and determine whether the candidate value follows from both. We propose VeriFin, a neurosymbolic verification framework for numerical financial question answering that grounds operands in filed XBRL facts, derives calculations from the question, filing linkbases, or documented metric definitions, and checks claims using Z3. When a claim is inconsistent, solver-derived unsatisfiable cores identify the conflicting facts, formula, and candidate value, enabling targeted repair. We evaluate VeriFin on XBRLFiling, a new 600-question benchmark constructed from 10-K filings of 28 U.S. companies, and on FinanceBench. On fixed candidate pools shared by all verification methods, VeriFin accepts none of the incorrect claims, whereas the baselines accept 6 to 92 incorrect claims on XBRLFiling's 600 claims and 4 to 21 incorrect claims on FinanceBench's 67 claims. The zero false-accept result persists across multiple answer-generator models. Solver-derived feedback further improves repair, achieving up to 69.9% recovery among true catches. These results show that source-grounded symbolic verification can provide a reliable and auditable acceptance control for LLM-generated financial claims.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.07943v2">Poise: Position-Aware One-Instruction Skill Injection for Silent Execution on LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-10
      | 💬 Title changed from "POISE: Position-Aware Undetectable Skill Injection on LLM Agents" to "Poise: Position-Aware One-Instruction Skill Injection for Silent Execution on LLM Agents"; the manuscript, figures, appendices, and reproducibility details have been updated. 15 pages, 2 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Agent skills extend general-purpose agents, but their open format enables skill poisoning: a tampered skill can make an agent run an attacker's command while completing the user's legitimate task. Invocation alone is insufficient; the attack-specific action must complete while that task still passes its verifier. We therefore define Attack Success Rate (ASR) to require a postcondition-validated sandbox action and a passing task verifier in the same trial. Skill files expose a reliability-visibility trade-off between a preloaded but conspicuous YAML frontmatter block and a longer body, where arbitrary placement may be skipped or locally incongruent. We introduce Poise, a position-aware attack that uses context-aware generation to place exactly one benign-looking, command-bearing instruction at a structurally feasible body position. On the eligible Skill-Inject pool with codex+gpt-5.2, Poise achieves 89.3\% ASR, 28.0 points above a context-free random-placement body baseline and comparable to the 86.7\% ASR of a high-exposure YAML-only baseline. Under the SkillTester audit, four LLM judges falsely flag 74.6\% of clean skills on average across both benchmarks, while only 5.6\% of Poise variants gain a new high-risk alert over their clean counterparts. One locally plausible, command-bearing body instruction therefore matches YAML-level reliability, while the resulting poisoned skill seldom adds a new high-risk finding over its clean counterpart.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10186v1">The Deliberative Deficit: An Empirical Critique of LLMs in Democratic Discourse</a></div>
    <div class="paper-meta">
      📅 2026-08-10
      | 💬 10 pages, archival publication at AIES 2026
    </div>
    <details class="paper-abstract">
      LLMs are increasingly deployed in settings that require collective reasoning on complex, value-laden problems. Confidence in these deployments rests largely on benchmarks for verifiable tasks (mathematics, coding, coordination games), yet many of these applications concern problems where no objectively correct answer exists and where decision quality instead depends on integrating pluralistic perspectives to find mutually acceptable solutions. We argue that LLM reasoning capacity on this class of problems cannot be fully inferred from verifiable-task benchmarks, and that procedural evaluations of LLM discourse (respectfulness, justification, engagement) are systematically insufficient. We apply the Deliberative Reason Index (DRI), a measure developed in political science and validated across citizen assemblies, as a tool for evaluating reliable group-level reasoning on pluralistic, non-verifiable problems. Synthesizing recent evidence across 1,980 five-agent LLM runs on 12 citizen-assembly topics across 11 frontier model configurations, we find that LLM groups produce discourse with procedural quality comparable to human deliberation, while gains in intersubjective consistency are small, topic-dependent, and concentrated on tractable rather than ethically contested questions. LLM groups exhibit roughly one-third the perspective diversity of human assemblies and reverse the human convergence pattern: human deliberation decreases dispersion as diverse views synthesise, whereas LLM deliberation increases it. Engineering diversity through persona prompting does not restore the human dynamic but inverts which component of deliberative reasoning is updated. Our conclusion is constraining rather than prohibitive: LLMs can function as tools supporting human reasoning on pluralistic problems, but current evidence does not license treating them as autonomous deliberative agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10171v1">Generating Attacks for LLMs with GFlowNets</a></div>
    <div class="paper-meta">
      📅 2026-08-10
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has facilitated their ubiquitous integration into various domains, leading to widespread adoption. However, this escalating trend has introduced significant security vulnerabilities, necessitating the identification and mitigation of flaws arising from malicious exploitation. Red teaming assessments, conducted to evaluate model robustness through diverse adversarial inputs, are essential for exposing security risks and implementing countermeasures. Currently, red teaming is performed either manually by experts or automatically using predefined attack datasets. Nevertheless, manual testing remains time-consuming, while existing automated methods suffer from limited creativity due to their inherent dependency on fixed datasets. In this study, we propose an automated, human-independent, and adaptive approach leveraging GFlowNets to identify LLM vulnerabilities by utilizing one large language model to test another. Within this framework, an attacker model is trained against a specified victim model to perform automated red teaming and provide a quantitative robustness score. This research aims to generate more effective adversarial attacks in English compared to existing benchmarks and, as a novel contribution to the literature, introduces a model capable of generating attack inputs in the Turkish language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11470v2">The Periodic Table of LLM Reasoning: A Structured Survey of Reasoning Paradigms, Methods, and Failure Modes</a></div>
    <div class="paper-meta">
      📅 2026-08-10
    </div>
    <details class="paper-abstract">
      Reasoning has become central to how Large Language Models (LLMs) are evaluated and interpreted, spanning Chain-of-Thought (CoT), mathematical problem-solving, multi-hop question answering, code generation, retrieval-augmented reasoning, tool use, and multimodal decision-making. In this survey, we introduce the Periodic Table of LLM Reasoning, a framework organizing 300+ recent papers by reasoning paradigm, methodological mechanism, evaluation setting, and failure mode. We classify LLM reasoning into nine paradigms: Chain-of-Thought, Multi-Hop, Mathematical, Commonsense, Visual and Temporal, Code and Algorithmic, Retrieval-Augmented, Tool-Augmented or Agentic, and Reinforcement Learning-based reasoning. For each, we review approaches, including prompting, architectural interventions, supervised fine-tuning, verifier-guided inference, reward modeling, retrieval, tool interfaces, agentic workflows, and benchmark design. We argue that LLM reasoning is not a single emergent capability but a family of scaffolded behaviors shaped by model scale, task structure, external memory, supervision, and evaluation protocols. We synthesize recurring failure modes, including hallucinated reasoning, brittle multi-step inference, spurious rationales, weak causal grounding, poor out-of-distribution generalization, benchmark contamination, and unreliable self-verification. Progress is difficult to compare across paradigms because gains may arise from prompting, retrieval, verifier design, or benchmark structure rather than general reasoning ability. The survey connects methods to their assumptions, strengths, and failure modes, providing a reference map of the field and a diagnostic framework for future work. We conclude that robust LLM reasoning will require meta-reasoning, multimodal and temporal grounding, adaptive tool use, and principled evaluation under distribution shift.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10149v1">REATS: LLM Reasoning-based Ensemble Learning for Adaptive Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2026-08-10
    </div>
    <details class="paper-abstract">
      Due to the diversity of real-world time series, no single forecasting model consistently dominates across all samples. Ensemble learning addresses this by combining complementary model strengths, yet existing methods rely on fixed rules or black-box models based solely on numerical inputs, failing to leverage LLM reasoning for interpretable weighting decisions. We propose REATS, which leverages LLM reasoning capabilities as an intelligent ensemble router that jointly processes textual temporal pattern descriptions and numerical features to produce interpretable, sample-adaptive ensemble weights through chain-of-thought reasoning. To enable effective LLM-based ensembling, we study its key design choices and propose: (i) a structured input pipeline that transforms raw time series into hybrid textual--numerical representations with fixed token cost, enabling rule-based chain-of-thought construction without API dependency, augmented with retrieved similar-sample priors; (ii) a diverse multi-row weight supervision scheme coupled with a token-efficient percentage-table format that reduces numerical complexity and mitigates LLM hallucinations; and (iii) a two-stage fine-tuning framework combining SFT with GRPO, where a reciprocal reward mapping transforms the continuous unbounded MSE gap into bounded signals with amplified near-oracle sensitivity, addressing the uniform sensitivity and outlier-dominated advantage compression inherent in naive reward designs for regression-based GRPO. Experiments on eight benchmarks demonstrate that REATS outperforms competitive ensemble baselines while providing natural language explanations and demonstrating strong transfer learning and out-of-domain generalization to unseen candidate models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.09885v1">SHE: Trajectory-driven Safety Harness Evolution for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-10
      | 💬 Project: https://github.com/RainbowQTT/SHE
    </div>
    <details class="paper-abstract">
      The safety of large language model (LLM) agents depends not only on model weights but also on the agent harness that manages context, memory, tools, permissions, and runtime control. Existing safety mechanisms often treat the harness as a fixed deployment artifact, limiting their ability to evolve with emerging risks. Moreover, coupled functions across harness components obscure safety responsibility attribution, making localized evolution difficult. We propose Safety Harness Evolution (SHE), a framework that learns evolving safe boundaries from rollout trajectories. SHE decomposes the harness into four artifacts with explicit safety responsibilities, including the System Prompt, Rule Bank, Safety Memory, and Tool Policy, defining clear functional boundaries for localized evolution. Based on this decomposition, SHE introduces an attribution-guided evolution loop that converts trajectory failures into structured diagnoses, learns artifact-specific boundary refinements, and selects evolved harnesses through safety-utility validation. Experiments on Agent-SafetyBench demonstrate that SHE effectively enhances safety through harness evolution, achieving a 3.1x ASR reduction compared with static SafeHarness, while also improving benign utility. The evolved harness further generalizes to unseen risks on the held-out AgentHarm benchmark and transfers across agent models without additional evolution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.09857v1">Agentic Harnesses: LLM-Driven Verification Layers for Robot Autonomy</a></div>
    <div class="paper-meta">
      📅 2026-08-10
      | 💬 7 pages. Not yet finalized for conference submission
    </div>
    <details class="paper-abstract">
      Advances in advanced artificial intelligence tools have sparked research in robot autonomy, but the development of such systems has largely focused on execution rather than verifying the feasibility actions planning models propose. Like general-purpose LLMs, robotics planning models carry risks: biased toward user-specified goals, they may suggest actions misaligned with scientific ethics, they may be unsafe due to an inability to "remember" prior safety risks, or they may be vulnerable to adversarial attacks on the autonomy ecosystem. We propose a LLM-driven verification layer between planning and execution to evaluate action permissibility. Our LLM-as-a-Judge ensemble combines chain-of-thought reasoning across models and synthesizes those expert judge outputs, mirroring a combination of a mixture of experts and self-consistency approach. This layer serves as middleware, gating plans from the server's planning module before they reach the MCP server and therefore the robot's low-level controls: plans are approved, rejected for reformulation, or escalated for human review. With this system, we achieve near 85% precision across accept/escalate/reject categories 97% containment of adversarial attacks, with negligible errors between accepting and rejecting tasks, and errors mostly manifesting at the escalate boundary.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.09779v1">KGCaRe: Explainable Complex Conditional Question Answering using Automatic Knowledge Graph Construction and Context Retrieval with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-10
    </div>
    <details class="paper-abstract">
      Answering complex conditional questions using Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) remains a challenge, particularly in domain-specific contexts where general-purpose LLMs and RAG tend to underperform. We hypothesize that augmenting RAG with unstructured and structured knowledge, extracted from both documents and knowledge graphs (KGs), can improve reasoning and answer accuracy for such tasks. To test this, we propose KGCaRe, a hybrid approach that combines neural retrieval with symbolic reasoning over LLM-generated KGs. KGCaRe constructs a KG from documents using a multi-prompt extraction strategy and stores it in a graph database. Simultaneously, the documents are embedded into a vector store to enable neural retrieval. KGCaRe performs innovative iterative graph traversal guided by the LLM to extract relevant triples, prune irrelevant information, and uses additional clue entities to traverse the graph again if the initial traversal does not provide satisfactory context to generate the answer. The relevant triples extracted from the KG in path form, along with semantically retrieved text passages, are then fed into custom KGCaRe prompts to generate answers to the complex conditional questions with explanations. We evaluate KGCaRe on two complex conditional QA datasets. Our results on these datasets show that KGCaRe consistently outperforms existing baselines, including Vanilla LLM, Code Prompt, Text Prompt, Think-on-Graph, Vanilla RAG, and HybridContextQA, across multiple LLMs such as Mistral, Mixtral, GPT-3.5, and GPT-4o. We publicly release the software pipeline that we developed to implement the proposed KGCaRe approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.09717v1">How Do Large Language Models Judge Social Attraction? Evidence from Theory-Grounded Persona Ratings Across Multiple LLMs and Humans</a></div>
    <div class="paper-meta">
      📅 2026-08-10
      | 💬 9 pages, 2 figures, 2 tables. Includes technical supplement
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to perform subjective evaluations traditionally made by humans, yet their validity as social judges remains unclear. This paper examines whether LLMs can assess social attraction from theory-grounded persona profiles constructed from ten psychological and relational constructs and organized into three tiers: socially attractive, socially mixed, and socially unattractive. We examine LLM ratings in two studies and compare them with human judgments in a third study. In Study 1, 34 LLMs rated 12 profiles across three repeated runs. Although some models tended to give higher or lower ratings overall, they showed strong stability across runs, consistent three-tier ordering, and high agreement in relative profile ordering. Study 2 examined sensitivity to gender presentation using six matched name-and-pronoun profile pairs and a separate pronoun-only test with a gender-neutral name, finding no significant effects in either analysis. In Study 3, 198 human participants evaluated the six matched profiles from Study 2. Their ratings reproduced the three-tier structure and followed a profile ordering consistent with that of the LLMs. However, LLMs rated attractive profiles more positively and unattractive profiles more negatively than humans, while neither group showed a significant overall effect of gender presentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.10015v3">FinTrace: Holistic Trajectory-Level Evaluation of LLM Tool Calling for Long-Horizon Financial Tasks</a></div>
    <div class="paper-meta">
      📅 2026-08-10
    </div>
    <details class="paper-abstract">
      Recent studies demonstrate that tool-calling capability enables large language models (LLMs) to interact with external environments for long-horizon financial tasks. While existing benchmarks have begun evaluating financial tool calling, they focus on limited scenarios and rely on call-level metrics that fail to capture trajectory-level reasoning quality. To address this gap, we introduce FinTrace, a benchmark comprising 800 expert-annotated trajectories spanning 34 real-world financial task categories across multiple difficulty levels. FinTrace employs a rubric-based evaluation protocol with nine metrics organized along four axes -- action correctness, execution efficiency, process quality, and output quality -- enabling fine-grained assessment of LLM tool-calling behavior. Our evaluation of 13 LLMs reveals that while frontier models achieve strong tool selection, all models struggle with information utilization and final answer quality, exposing a critical gap between invoking the right tools and reasoning effectively over their outputs. To move beyond diagnosis, we construct FinTrace-Training, the first trajectory-level preference dataset for financial tool-calling, containing 8,196 curated trajectories with tool-augmented contexts and preference pairs. We fine-tune Qwen-3-8B/32B using supervised fine-tuning followed by direct preference optimization (DPO) and show that training on FinTrace-Training consistently improves intermediate reasoning metrics, with DPO more effectively suppressing failure modes. However, end-to-end answer quality remains a bottleneck, indicating that trajectory-level improvements do not yet fully propagate to final output quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01236v2">Safeguarding LLM Agents from Misalignment through Provenance Analysis</a></div>
    <div class="paper-meta">
      📅 2026-08-10
    </div>
    <details class="paper-abstract">
      As LLM agents gain increasing access to powerful tools, ensuring that their actions align with the user's intent becomes critical. When an agent's proposed action deviates from that intent---a phenomenon called misalignment---it may cause harm that is difficult to undo. Existing runtime guardrails rely on an LLM-as-a-judge paradigm that lacks a systematic framework for reasoning about alignment, often producing inconsistent or difficult-to-audit judgments. Motivated by provenance analysis, we propose a conceptual framework that formalizes misalignment detection as determining whether a proposed tool call is supported by traceable evidence in the agent's context. Based on this framework, we build ProvenanceGuard, a multi-stage pipeline that analyzes the agent's action for three types of misalignment before its execution and only allows aligned actions. We evaluated ProvenanceGuard on AgentSafetyBench and WorkBench, across 11 backbone LLMs. Compared to the LLM-as-a-judge baseline, ProvenanceGuard reduces error rate on misaligned traces from 44.3% to 2.1% on Agent-SafetyBench and from 32.4% to 18.7% on WorkBench, while reducing interventions on task-successful traces from 31.2% to 13.0% and introducing no statistically significant increase in unnecessary interventions on aligned traces. These results demonstrate that structured, provenance-based reasoning provides an effective and practical foundation for safeguarding LLM agents from misalignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.09650v1">Listwise Cross-Encoder Fine-Tuning vs. Agentic Instruction Tuning for LLM Rerankers: A Systematic Study in Medical Procedure Reranking</a></div>
    <div class="paper-meta">
      📅 2026-08-10
      | 💬 10 pages, 6 figures, 4 tables. Code available at https://github.com/matanf-healthee/listwise-crossencoder-reranking
    </div>
    <details class="paper-abstract">
      Reranking medical procedures against patient queries is a critical component of health insurance information retrieval, complicated by a substantial lexical gap between patient language and clinical nomenclature. We present a systematic comparison of two reranking paradigms for this production task: (1) small cross-encoders (MedCPT, MiniLM-L12) fine-tuned with listwise learning-to-rank objectives across layer freezing configurations, and (2) Qwen3-Reranker-4B, a 4B-parameter instruction reranker whose prompt is iteratively refined via an agentic optimization loop driven by GPT-4.1. On a purpose-built dataset of 2,647 queries across 708 insurance services, we find that a 109M-parameter cross-encoder fine-tuned with ListNet outperforms the 4B-parameter model by 2.6 percentage points on NDCG@3 and 13.3 points on Spearman correlation - at 37x fewer parameters. We report practical findings, a scalable LLM based dataset construction pipeline, and deployment trade-offs relevant to production reranking systems. We release our code and a sample dataset to support reproducibility and adaptation to other domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.04713v2">RSPO: Reward-Swap Policy Optimization for Multi-Turn LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-10
    </div>
    <details class="paper-abstract">
      Reinforcement learning holds significant potential for training large language models (LLMs) to handle multi-turn interactive tasks. However, in long-horizon, multi-turn tasks characterized by sparse outcome rewards, directly training with outcome rewards often results in slow convergence due to the sparsity of signals and the lack of fine-grained feedback. Furthermore, the model may fail to learn successful trajectories that are not sampled during training, thereby limiting its performance. Conversely, while employing customized dense process rewards provides richer signals and accelerates convergence, these surrogate rewards may exhibit potential misalignment with the ground-truth outcome rewards. This inconsistency can bias the training direction and ultimately degrade the model's final performance. In this work, we propose Reward-Swap Policy Optimization (RSPO), a method designed to leverage the rich information from dense process rewards to facilitate training with outcome rewards. By utilizing a reward-swap mechanism, RSPO ensures the diversity of sampled trajectories while guaranteeing consistency between the optimization objective and the true outcome rewards, thereby elevating the performance ceiling of the model. We conduct extensive experiments on two challenging agent benchmarks, WebShop and ALFWorld. By applying our method to various reinforcement learning algorithms, including GRPO, PPO, and GiGPO, we demonstrate that RSPO achieves consistent performance improvements across different baselines and benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.06705v2">Translation Tag Team: Formal Rules and LLMs Translate More Macros Together than Apart</a></div>
    <div class="paper-meta">
      📅 2026-08-10
      | 💬 13 pages, 7 figures, 2 tables, 8 listings, conference. To be published in ASE 2026
    </div>
    <details class="paper-abstract">
      Modern critical software infrastructure is largely written in C. Since C lacks memory safety, researchers are investigating automatic translation of C to safer languages like Rust. But real-world C software consists of more than just C code, often using named code fragments called macros which are not part of the C language proper. State-of-the-art techniques avoid translating macros by preprocessing C code first before translating it. But this approach produces translations that are dissimilar to the original C code, because preprocessing inlines all macro definitions. To preserve macro usage in translated code, we study the language features that macros and C share and distill them into the first formally-specified translator, MerC. To evaluate MerC, we introduce the first macro translation benchmark, MacroBench, with test cases based on macros randomly sampled from real-world C programs. We find that MerC supports 50% of MacroBench's macro test cases. We also use MacroBench to evaluate how effective large language models (LLMs) are at performing the previously-unstudied task of macro translation. LLMs translate 22% to 77% more of MacroBench than MerC, but with 8% and 28% of these translations being incorrect translations requiring additional validation by developers. In contrast, MerC only produces correct translations. Our key insight is that running MerC first then using LLMs on the remainder reaps greater benefits than using either technique alone. This tag team approach has an average failure rate 32% lower than that of LLMs, while also translating an average of 51% more test cases than MerC.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.09605v1">TSPORec: Token Selection via Preference Optimization for LLM-Based Sequential Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-08-10
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as powerful tools for improving recommendation systems. The effectiveness of LLMs arises from their ability to harness rich textual information and their capacity to model heterogeneous user preferences based on users' interaction history. However, due to the large-scale and deep architectures, LLM-based sequential recommendation approaches generally incur high inference costs, resulting in a low return on investment. To mitigate this cost, many existing approaches resort to using only the first few tokens of item descriptions, which inadvertently discards valuable information contained in the full text, thereby leading to suboptimal recommendation performance. To address this limitation, we propose a novel Token Selection approach for Preference Optimization in LLM-based sequential Recommendation, i.e., TSPORec, which accurately pinpoints informative tokens throughout the entire textual content to improve recommendation performance. Specifically, we design a three-stage pipeline to select informative tokens and introduce a novel proxy reward to facilitate the implementation. TSPORec not only enhances recommendation performance but also improves computational efficiency. Extensive experiments across two models and datasets demonstrate the superb performance (up to 31.25%) and efficiency (up to 63.4%) of our approach compared with six baseline approaches. Code is available at https://github.com/WNQzhu/TSPORec.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.04412v3">Beyond Pixels: Exploring DOM Downsampling for LLM-Based Web Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-10
      | 💬 21 pages, LaTeX, print version; enhanced algorithm, settled on non-inferiority claim, added downsampling monotonicity and linearity results
    </div>
    <details class="paper-abstract">
      The advent of large language models (LLMs) has sparked an evolution of autonomous web browsing agents: given a web browsing task and serialised user interface (UI) state, an LLM is expected to suggest input actions that incrementally solve the given task. The central challenge lies in serialising UI state for LLMs. Web agents have increasingly relied on grounded graphical UI (GUI) snapshots - screenshots augmented with visual cues - favoured for their modest input token footprint. Document Object Model (DOM) snapshots, serialised as HTML, represent a compelling alternative that leverages previously demonstrated HTML interpretation capabilities of LLMs. Their excessive token footprint, however, has precluded reliable deployment with web agents to date. We propose D2Snap, an algorithm to downsample the DOM, premised on preserving actionability and actionability-discriminating features. We evaluate D2Snap-downsampled DOM snapshots using a snapshot-variant web agent (GPT-4o) on a dataset sampled from Online-Mind2Web. Whilst 42% of raw DOM snapshots exceed the model context window (128 x 10^3 tokens), all D2Snap-downsampled DOM snapshots of our reference configuration fit, at a mean context utilisation of 16.5%. Against the 67% success rate of a grounded GUI snapshot baseline, our configuration attains 73% (+5.8%pt; 95% CI -13.6 to +26.0%pt; McNemar, p = 0.47), excluding a deficit (one-sided 95%) beyond 11%pt. Image input moreover appears to add little to snapshot utility; grounding text alone attains 62% (-5.8%pt; McNemar, p = 0.37).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.09586v1">ICM Out! Better Tournament Strategy from Computed Continuations, vs. Solvers and LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-10
      | 💬 34 pages, 2 figures
    </div>
    <details class="paper-abstract">
      The Independent Chip Model (ICM) converts tournament chips into reference prize equity, and policies are routinely constructed against those values. Because ICM reads only stack sizes, it omits action order, blind obligations, and seat rotation, and it does not price the elimination pressure a big stack puts on the short stacks it can bust. Those omissions can alter the successor-state contrasts that determine a move. We introduce Strategic-Continuation Optimization (SCO), a policy-construction method that enumerates current-hand outcomes, maps them to successor states, prices those states with continuation values computed from the finite tournament model, and optimizes and freezes the resulting current-hand policy. The fixed-ICM comparison policy changes one thing only: the same optimizer solves the same game with successor states priced by analytic ICM, so the two policies differ only through that pricing. We evaluate the resulting policies in a three-player jam/fold tournament with a \$1M prize pool. Relative to the frozen strategic-continuation benchmark, analytic ICM has \$9{,}036 mean absolute value error across all 2,838 state--seat entries. That value error rewrites the ranges it prices: measured against each decision point's own fixed-ICM jam range, SCO moves the jam frequency by an average of 14.08\%. To price those different moves, we compare all 946 states and three policy owners while changing only the focal policy and holding both opponents and the continuation evaluator fixed. The policy produced by SCO earns \$214.33 more prize equity per hand on average and is favored in 2,433 of 2,838 matched units. The ordering survives replacing the solver-built opponent with two LLMs and with a family of non-modeling threshold players. This value-to-policy-to-cost chain shows directly when ICM becomes an inadequate objective for tournament strategy construction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.09577v1">ElasticBack: Stealthy Conditional Backdoor in LLM-Agent Skills via Coupled Trigger-Rule Optimization</a></div>
    <div class="paper-meta">
      📅 2026-08-10
    </div>
    <details class="paper-abstract">
      Agent skills, bundles of instructions and resources that an LLM agent loads on demand, form an emerging supply chain where a single poisoned skill can persistently compromise every agent that installs it. However, existing skill attacks either fire on every request or rely on fine-tuned weights or multiple skills, leaving a conditional and low-cost backdoor unexplored. In this work, we present ElasticBack, an effective conditional single-skill backdoor that plants a rule R in the skill document and a benign-looking trigger T in the user query, so the malicious payload fires only when both co-occur. ElasticBack binds the two sides through a trigger-as-switch construction, generating R via semantic-anchored rule injection. It then freezes R and evolves T against it with a stealth-constrained genetic search, so that effectiveness and stealth are optimized, keeping the backdoor weight-free and dormant on benign inputs. Extensive experiments across three target behaviors (50 skills each) and four agent LLMs show that ElasticBack attains a high attack success rate at a near-zero false-positive rate with preserved clean accuracy, transfers across models, and evades deployment-time defenses. These results motivate stronger defenses for the skill supply chain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.09574v1">The Politician, the Liar, and the Obedient Worker: Emerging Behavior of LLM Agents in Hierarchical Games</a></div>
    <div class="paper-meta">
      📅 2026-08-10
    </div>
    <details class="paper-abstract">
      LLMs are rapidly embedding themselves into daily life: drafting our emails, managing our schedules, and making decisions on our behalf. As they move from individual tools to participants in multi-agent organizations, an important question arises: do they reproduce the governance failures like free-riding, corruption, and entrenched leadership that plague human institutions? We introduce the Hierarchical Game (HG), a public goods game extended with managerial authority, democratic elections, and private communication. Testing six frontier models across twelve experiments that add institutions one at a time (speech, peers, government, wages, oversight, elections), we find distinct behavioral profiles: Qwen promises and lies (13.3\% broken promises); Grok refuses to cooperate on its own but becomes fully cooperative once a manager can punish it (16\%$\to$100\%); Claude and GPT-4o cooperate reliably at baseline. But honesty proves fragile. When the manager role comes with a salary, all models except GPT-4o start cutting private deals to win or keep the position. When punishment is made anonymous, honest models begin to cheat. When all agents share the same model family, the first elected manager stays in power indefinitely. Leadership change only happens in groups that mix different families.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.09567v1">From Runnable to Verifiable: An Independent Reproducibility Study of LLM/Agent-Driven Vulnerability Validation Artifacts</a></div>
    <div class="paper-meta">
      📅 2026-08-10
      | 💬 16 pages, 2 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Security research artifacts---repositories, PoC exploits, and validation pipelines---are increasingly produced by LLM/agent-driven vulnerability workflows, yet the gap between \emph{publicly available}, \emph{runnable}, \emph{signal-producing}, and \emph{semantically confirmed} artifacts is poorly measured. We conduct a pre-registered reproducibility audit of this literature. A search covering 2023--2026 with dual screening yields a 104-paper consensus corpus, of which 59 papers (56.7\%) have a publicly reachable artifact. We execute an 18-paper sample at R0/R1 and all 102 cases of the anchor benchmark (arXiv:2509.24037), with patched-counterfactual verdicts on 30 signal-producing cases and matched-negative-control verdicts on 19. Three findings stand out. First, 58/102 (56.9\%) anchor cases contain a script-internal CVE identifier that diverges from the declared directory CVE. Second, only 10/18 (55.6\%) paper-level artifacts complete their declared workflow at R0, rising to 11/18 (61.1\%) after environment-only R1 repair. Third, artifact-embedded oracles prove unreliable: 20/30 patched-counterfactual audits still produce the claimed signal on the patched build, 7/19 matched negative controls still trigger on benign input, and the oracle confusion matrix has sensitivity 60\% and specificity 45\%. A trigger on the vulnerable build is not evidence of CVE-specific reproduction without a clean patched counterfactual. These are exploratory results from a pre-registered protocol, and our protocol---pre-registered post-conditions, R0/R1 repair ladder, G1--G3 semantic evidence levels, and patched-counterfactual oracles---is a reusable template for the security reproducibility community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.23333v2">Process Supervision of Confidence Margin for Calibrated LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-08-10
    </div>
    <details class="paper-abstract">
      Scaling test-time computation with reinforcement learning (RL) has emerged as a reliable path to improve large language models (LLM) reasoning ability. Yet, outcome-based reward often incentivizes models to be overconfident, leading to hallucinations, unreliable confidence-based control, and unnecessary compute allocation. We introduce Reinforcement Learning with Confidence Margin (RLCM), a calibration-aware RL framework that jointly optimizes correctness and confidence reliability via a margin-enhanced process reward over intermediate-budget completions. Rather than aligning confidence to correctness likelihoods, RLCM encourages the model to widen the confidence margin between correct and incorrect steps within a single reasoning trajectory. Across mathematical, code, logic and science benchmarks, our method substantially improves calibration while maintaining or improving accuracy. We further show that, with calibrated confidence signals, the resulting models enable more efficient conformal risk control and effective confidence-weighted aggregation
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.09555v1">Bidirectional Context Self-Distillation for Reinforcement Learning of Skill-Based LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-10
      | 💬 9 pages, 6 figures
    </div>
    <details class="paper-abstract">
      External natural-language skills provide large language model (LLM) agents with reusable and editable guidance for solving complex tasks. Yet their effectiveness depends not only on skill quality, but also on whether the policy can translate the provided guidance into appropriate actions. However, methods specifically designed to improve this skill-utilization ability remain largely underexplored. In practice, skill-based agents are commonly trained with reinforcement learning objectives centered on task-level rewards, which offer limited supervision and struggle to capture subtle differences in how effectively the policy uses the provided skills. We propose BCSD (Bidirectional Context Self-Distillation), a framework that combines self-distillation with reinforcement learning to train LLM agents to use external skills more effectively. Unlike prior self-distillation methods that rely on a single privileged context, BCSD evaluates each trajectory from two complementary skill-context views. The augmented view introduces higher-level Meta-Skill guidance, while the reduced view prunes general guidance to highlight task-specific skills. Their complementary token-level signals are combined to rescale the RL advantage. Experiments on ALFWorld and WebShop demonstrate that BCSD achieves the strongest overall performance across model scales, enabling agents to utilize external skills more effectively. Ablation studies further verify the complementary contributions of the augmented and reduced context views. Code will be released to ensure full reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.09510v1">Build it, Break it, Repeat: Benchmarking and improving LLM-manipulated disinformation detection in social media posts</a></div>
    <div class="paper-meta">
      📅 2026-08-10
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Detecting machine-generated disinformation on social media is increasingly difficult as large language models (LLMs) make it easier to generate and rewrite misleading content at scale. Static benchmark evaluations, measuring detector performance on fixed held-out datasets, do not capture how detectors behave when posts are deliberately transformed to evade classification. This paper adapts the Build it, Break it, Fix it framework into Build it, Break it, Repeat (BiBiR): iterative sessions designed to stress-test detectors' robustness under iterative adversarial conditions, evaluating whether models remain reliable when disinformation posts are systematically transformed to evade classification. Across five iterations, the findings show that the best adversarial breakers' transformations came from a combination of back-translation and LLM persona-based rewriting, with the best performing technique achieving a 95% label flip rate (LFR), whilst still preserving the meaning of the original posts. The best builders' model was a triplet contrastive model with a dynamic anchor switching (DASS) architecture, which achieved an average accuracy of 72.68%, outperforming the strong baseline (a fine-tuned e5-small-LoRA) by 15 percentage points on the most robust set of breakers' adversarial attacks. The results demonstrate that an iterative framework best exposes detector weaknesses and pushes robustness improvements; however, it may still require semantic preservation analysis to distinguish valid adversarial evasion from transformations that changed the original disinformation claims' meaning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.03610v2">Language-Specialized Multi-Teacher On-Policy Distillation for Multilingual LLM-Based ASR</a></div>
    <div class="paper-meta">
      📅 2026-08-10
    </div>
    <details class="paper-abstract">
      Modern LLM-based ASR systems have established multilingual capability as a standard feature, leveraging large-scale multilingual corpora and LLMs' cross-lingual knowledge to achieve competitive performance across multilingual benchmarks. However, jointly modeling languages with heterogeneous acoustic, phonological, and lexical characteristics inevitably introduces optimization conflicts, undermining language-wise specialization. To address this challenge, we propose Language-Specialized Multi-Teacher On-Policy Distillation (LS-MOPD), which decouples language-specific knowledge acquisition from multilingual capability integration: language-specialized teachers are independently optimized via reinforcement learning (RL), with their expertise then integrated into a generalist multilingual student through language routing and token-level multi-teacher distillation, thereby reducing direct cross-lingual optimization conflicts. We further explore static and dynamic acoustic-prefix configurations to examine how teacher-student prefix consistency influences the efficacy of on-policy distillation. Experiments on benchmarks covering Mandarin, Mandarin subdialects, Cantonese, and English demonstrate that LS-MOPD substantially outperforms RL baselines and surpasses the empirical performance envelope defined by the best-performing RL teachers on nearly all benchmarks, revealing its potential to generalize beyond all teachers in multilingual ASR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.09440v1">MetaStrategy: Generative Ranking with Executable LLM Strategies</a></div>
    <div class="paper-meta">
      📅 2026-08-10
    </div>
    <details class="paper-abstract">
      Industrial recommender systems rank heterogeneous content under coupled user, business, commercial, and experience objectives. Existing generative ranking methods typically construct item sequences directly, making them difficult to integrate with mature predictive models, operational rules, and field-level guardrails. We present MetaStrategy, a framework that instead generates a structured, executable ranking strategy. Conditioned on request context, a large language model (LLM) policy emits a typed JSON bundle controlling objective weights, content and category preferences, experience constraints, and position policies. A deterministic validator and compiler instantiate an isolated Generator that competes atomically with incumbents under the list-level Evaluator of the Generator-Evaluator (GE) architecture. We train the policy in a production-path replay environment that re-executes logged requests through the current re-ranking stack without user exposure. The method combines selection, relative-rank, and baseline-lift rewards, a self-competitive curriculum that feeds frequent strategies back as competitors, and Evaluator-routed reward-augmented on-policy distillation that transfers complementary 4B-parameter Teachers into a compact 0.8B-parameter Student. We deploy MetaStrategy in Taobao Homepage Guess You Like through diff-triggered nearline generation; LLM inference remains outside synchronous ranking, with no observable increase in response time (RT). In a seven-day user-randomized online A/B test, MetaStrategy wins 27.93% of treatment-side GE calls and significantly improves click page views (click PV) by 2.11%, item-detail page views (IPV) by 3.12%, and transaction amount by 2.83%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.27887v3">PortBench: A Correlation-Aware, Full-Pipeline Benchmark for LLM-Driven Portfolio Management</a></div>
    <div class="paper-meta">
      📅 2026-08-10
      | 💬 Project page: https://portbench.github.io/
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown strong performance across diverse financial tasks, yet portfolio management (PM) remains poorly benchmarked. Existing benchmarks exhibit two gaps: they are often equity-only and ignore cross-asset correlations; they fail to evaluate the complete PM decision pipeline. We introduce PortBench, a benchmark spanning six heterogeneous asset classes over ten years. PortBench comprises two layers: a static QA dataset of 6,269 questions across seven task templates, and a dynamic five-stage allocation pipeline. To evaluate these layers, we introduce two metrics: a dual-layer correlation score for inter-class hedging and intra-class concentration, and CEPS, which quantifies how reasoning errors compound across pipeline stages. We further evaluate under three stress regimes and risk profiles, and support real-time evaluation to mitigate pretraining contamination on historical markets. Evaluating ten frontier LLMs, we find that despite strong financial QA performance, 90\% of model-profile cases fail to outperform equal-weight allocation in 2024, and this deficit persists across other market regimes; models that satisfy every procedural constraint still suffer large drawdowns under stress. Our source code is available at \href{https://github.com/AgenticFinLab/portbench}{this https URL}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.26937v2">Beyond Questions: Evaluating LLM's Knowledge Expression</a></div>
    <div class="paper-meta">
      📅 2026-08-10
    </div>
    <details class="paper-abstract">
      Parametric knowledge in large language models (LLMs) is a cornerstone of their success, yet remains poorly understood. Existing knowledge benchmarks typically rely on predefined questions (e.g., "What is the birth date of M.L. King?"), evaluating only knowledge that benchmark designers explicitly choose to query, a problematic availability bias. In this paper, we introduce open knowledge evaluation, a new paradigm for LLM knowledge expression benchmarking. Instead of asking narrow questions, it evaluates models on the knowledge they choose to surface in response to open-ended elicitation prompts (e.g., "Tell me everything you know about M.L. King"). This shifts the focus from predefined answer retrieval toward characterizing the knowledge models naturally express. We instantiate this paradigm with BeQu (Beyond Questions), a benchmark of 10,000 entities paired with reference corpora for statement verification. Using BeQu, we evaluate a broad range of language models and analyze the effects of reasoning effort, model scale, prompt format, and knowledge domain. Data and leaderboard are available on this work's GitHub repository and at the benchmark's website.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10042v1">UserToolBench: A User-Profile-Hidden Benchmark for Personalized Decision Making in Tool-Use LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-10
      | 💬 21pages,4figures
    </div>
    <details class="paper-abstract">
      Tool-use LLMs are increasingly asked to act on users' behalf, but existing benchmarks usually focus on profile recall, style imitation, generic tool use, or response-level personalization. We introduce UserToolBench , a benchmark for personalized decision making in tool-use LLMs. UserToolBench tests whether a model can infer latent user preferences from interaction history, recognize when clarification is needed, and produce user-aligned tool-call trajectories under incomplete information. The benchmark is built from privacy-sanitized real interaction traces and combines structured persona profiles, public API-style tool ecosystems, and long-horizon multi-turn trajectories. It includes 10 user profiles, 36 tool sets, 1,065 turns, 170 unique tools, and evaluation-focused task types covering lack-of-information, single-tool, and multi-tool settings. Experiments with strong tool-use LLMs show that current models still have difficulty with personalized delegation. Multi-tool coordination, missing-constraint inference, and long-horizon behavioral consistency remain major bottlenecks. These results suggest that personalization evaluation should move beyond asking whether outputs sound user-specific and instead ask whether LLMs make correct decisions for the users they represent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.09351v1">Test-Time Augmentation for LLMs: When Input Diversity Beats Output Diversity at Matched Compute</a></div>
    <div class="paper-meta">
      📅 2026-08-10
      | 💬 Published at the COLM 2026 Workshop on Efficient Reasoning
    </div>
    <details class="paper-abstract">
      Test-time scaling improves LLM accuracy but multiplies inference cost, making the accuracy gained per unit of compute the metric that matters in deployment. Self-consistency is one of the established approaches, which spends this budget entirely on the output side by sampling repeated reasoning paths. We study Test-Time Augmentation (TTA), which extends self-consistency by also perturbing the input, aggregating predictions across transformed versions of the input, and ask whether input-side diversity converts compute into accuracy more efficiently than output-side diversity. We perform a systematic, matched-compute comparison: we evaluate three simple input-side strategies (semantic rephrasing, lexical perturbations, and visual transformations) across six datasets covering general and multilingual knowledge, mathematical reasoning, multi-modal question answering, and sentiment classification, against chain-of-thought prompting and self-consistency. Semantic rephrasing delivers consistent and statistically significant accuracy gains while Pareto-dominating self-consistency on cost-effectiveness, delivering roughly 1.8X more accuracy per dollar and outperforming it on five of six tasks. We further analyze the number of augmentations, multi-modal strategies, and base model scaling, finding that TTA is most cost-effective for mid-tier models where a stronger model is unavailable or too expensive. Our findings indicate that for current mid-tier LLMs, varying the input converts inference compute into accuracy more efficiently than varying the reasoning path alone. The TTA implementation is available at https://github.com/aws-samples/sample-genai-reflection-for-bedrock.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.00270v3">LLMs Remember First, Forget Last: Dual-Process Interference in Large Language Models</a></div>
    <div class="paper-meta">
      📅 2026-08-10
      | 💬 NPIR, Springer book series: Lecture Notes in Networks and Systems, indexed by Scopus and Ei Compendex
    </div>
    <details class="paper-abstract">
      Large language models can process millions of tokens, yet how they handle conflicting information within context remains poorly understood. From patient health logs tracking evolving vital signs to legal documents with superseding clauses, real-world applications routinely require models to retrieve specific values from streams of semantically similar, competing updates. We adapt classical interference paradigms from cognitive psychology to compare retroactive interference (RI; recalling initial values after updates) and proactive interference (PI; recalling recent values despite competing prior encodings) across 39 LLMs spanning 1B to 2.5T parameters. Every model exhibits the same pattern: PI causes substantially greater degradation than RI (d = 1.73), the opposite of the typical human finding where new information more readily disrupts old. Four lines of evidence indicate that RI and PI engage distinct mechanisms: model size predicts RI resistance but not PI; the two scores correlate only weakly; error analysis reveals qualitatively different failure profiles, with RI failures reflecting passive retrieval failure and PI failures reflecting active primacy intrusion; and hallucination rates remain below 1% in both conditions. An adaptive bootstrap design further shows that RI decays smoothly while PI collapses bimodally at high interference, consistent with gradual dilution versus winner-take-all attention dynamics. These findings characterize a systematic primacy bias in transformer attention, with implications for reliable deployment in settings where in-context information evolves over time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.09343v1">LLM-Guided Heuristic Design from Simulation Traces: A Case Study in Dynamic Production and AGV Scheduling</a></div>
    <div class="paper-meta">
      📅 2026-08-10
      | 💬 33 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Simulation-based optimization (SBO) evaluates executable policies under stochastic dynamics, but most methods treat the simulator as a black box: aggregate scores rank candidates without revealing why they fail or which policy logic should change. We present an LLM-guided heuristic design framework that uses repeated simulation for selection and event-level traces for diagnosis. Each incumbent is assessed through multiple replications, while replaying its lowest-scoring one produces a queryable trace. A manager agent formulates bottleneck hypotheses from this evidence, and editing agents implement parallel code-level revisions. After execution checks and repeated evaluation, best-so-far selection retains only improvements. LLM revision occurs between evaluation batches, while a fixed policy controls each simulation run. We evaluate the framework in a discrete-event simulation of dynamic production and automated guided vehicle (AGV) scheduling. Across five independent optimization runs with Gemini-3.1-Pro, final mean scores averaged 77.51 on the simulator's 0-100 scale. In the highest-scoring run, trace-based diagnoses motivated proactive charging, distance-aware AGV assignment, and rebalanced dispatch priorities, raising the best-so-far mean score from 62.49 to 78.61. On 100 matched seeds, the best final policy outscored representative rolling-MILP, rule-based, and metaheuristic policies on every seed and retained its advantage under random faults without re-optimization. After separate re-optimization for a longer horizon and variable order interarrival times, the resulting policies again outscored all baselines. Ablations with two LLM backbones showed that removing either parallel candidate generation or trace-database access reduced final mean scores. These results show that simulation traces can guide targeted code-level policy improvement in complex simulation-based scheduling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2406.14373v3">Artificial Leviathan: Exploring Social Evolution of LLM Agents Through the Lens of Hobbesian Social Contract Theory</a></div>
    <div class="paper-meta">
      📅 2026-08-10
    </div>
    <details class="paper-abstract">
      The emergence of Large Language Models (LLMs) and advancements in Artificial Intelligence (AI) offer an opportunity for computational social science research at scale. Building upon prior explorations of LLM agent design, our work introduces a simulated agent society where complex social relationships dynamically form and evolve over time. Agents are imbued with psychological drives and placed in a sandbox survival environment. We conduct an evaluation of the agent society through the lens of Thomas Hobbes's seminal Social Contract Theory (SCT). We analyze whether, as the theory postulates, agents seek to escape a brutish "state of nature" by surrendering rights to an absolute sovereign in exchange for order and security. Our experiments unveil an alignment: Initially, agents engage in unrestrained conflict, mirroring Hobbes's depiction of the state of nature. However, as the simulation progresses, social contracts emerge, leading to the authorization of an absolute sovereign and the establishment of a peaceful commonwealth founded on mutual cooperation. This congruence between our LLM agent society's evolutionary trajectory and Hobbes's theoretical account indicates LLMs' capability to model intricate social dynamics and potentially replicate forces that shape human societies. By enabling such insights into group behavior and emergent societal phenomena, LLM-driven multi-agent simulations, while unable to simulate all the nuances of human behavior, may hold potential for advancing our understanding of social structures, group dynamics, and complex human systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.09292v1">Beyond the Capability Boundary: Zeroth-Order Optimization for Self-Evolving LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-10
    </div>
    <details class="paper-abstract">
      Self-evolving methods improve the capabilities of LLM agents by sampling trajectories from the underlying LLMs and learning from these trajectories. However, these methods struggle to learn beyond the inherent capability boundary of the agents, since the agents cannot sample correct trajectories on difficult examples for further improvements. In this paper, we propose a zeroth-order self-evolution framework that enables agents to learn beyond their capability boundary by perturbing LLM parameters to adapt to difficult examples without any trajectory annotations. Specifically, we perturb LoRA parameters of LLMs, run the agent, compute the losses under the perturbed and original parameters, and use the loss difference to estimate gradients and further update the LoRA parameters. We sample trajectories using the updated LLMs for supervised fine-tuning to break through the capability boundary of the agents, forming a closed self-evolution loop. We introduce a parallel perturbation inference mechanism and an adaptive lookup mechanism to reduce time consumption in zeroth-order optimization, with an answer perplexity loss that provides smooth and stable zeroth-order loss values. Experiments on multiple deep research benchmarks show that our method obtains substantially more successful trajectories and consistently outperforms strong baselines, especially on difficult examples. The code and released artifacts are available at https://github.com/hidk1911/ZOForLLMAgents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.09291v1">UnionSparse: An Index-Efficient Sparsity Framework for Low-Bit Sparse LLM Inference on Edge</a></div>
    <div class="paper-meta">
      📅 2026-08-10
      | 💬 14 pages, 19 figures. Accepted via the ESWEEK 2026 Journal Track for publication in IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (TCAD)
    </div>
    <details class="paper-abstract">
      Edge LLM inference combines sparsity and low-bit quantization to meet device memory, latency, and power limits. Yet quantization shrinks weight payloads without proportionally reducing sparse metadata, so index traffic and nonzero extraction become critical SpMM bottlenecks. We introduce the Payload-to-Metadata Ratio (PMR) and show that improving PMR raises effective compute intensity in decoding. We present UnionSparse, an index-efficient framework that combines Index-Efficient Bitmap Encoding (IE-BME) with a SpMM kernel using Low-Bit Shared-Memory Parallel Decoding (LSPD). IE-BME amortizes metadata and aligns sparse traversal with fragment assembly, while LSPD improves small-batch execution. Under W4A4 quantization and 30%--70% sparsity, UnionSparse outperforms FlashLLM and SpInfer by 2.30x and 1.43x, and CUTLASS and cuBLAS Tensor Core by 1.56x and 3.46x, respectively. These results establish payload-extraction efficiency as a first-order concern for low-bit sparse inference on edge GPUs. Source code is available at: https://github.com/Victor-Alen/UnionSparse.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.09282v1">ComboShoppingBench: Evaluating LLM Agents for Budget-Constrained Basket Shopping with Coupons</a></div>
    <div class="paper-meta">
      📅 2026-08-10
    </div>
    <details class="paper-abstract">
      Real-world shopping often requires constructing a basket of complementary items rather than retrieving a single product. Such combo-shopping tasks arise in device setup, meal preparation, event planning, and group takeout ordering, requiring joint reasoning about item compatibility, availability, store-level requirements, delivery fees, coupons, and budgets. Evaluation is challenging because multiple baskets may satisfy the same request, making exact-match metrics unsuitable, whereas semantic evaluation alone cannot detect infeasible orders, invalid coupon combinations, or incorrect payments. We introduce ComboShoppingBench, an agentic shopping benchmark for open-ended yet verifiable basket construction in a simulated commerce and takeout environment. During task synthesis, an exploration agent constructs a feasible and semantically coherent basket of purchasable products; this witness guides the generation of coupons, budget constraints, user queries, and aligned evaluation rubrics. During evaluation, LLM judges assess semantic satisfaction, response quality, and claim faithfulness, while deterministic validation checks product-ID validity, budget compliance, and coupon optimality. Experiments with diverse LLM agents demonstrate that even strong agents struggle on ComboShoppingBench, highlighting substantial room for improvement in reliable, constraint-aware combo shopping.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.09260v1">SafeQL: Search-based Refinement for Safe and Efficient LLM-based Text-to-SQL</a></div>
    <div class="paper-meta">
      📅 2026-08-10
      | 💬 VLDB 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have advanced Text-to-SQL by enabling natural language interfaces to databases without task-specific fine-tuning. However, existing LLM-based systems remain unreliable, often generating SQL queries that are invalid under the database schema, referencing non-existent tables, attributes, functions, or values. Such errors persist because interactions with the database management system (DBMS) are typically limited to error messages, leaving it in a largely passive role during query refinement. This paper proposes SafeQL, \textit{a search-based refinement paradigm that redefines the role of the DBMS as an active guide in the refinement process}. Instead of regenerating entire queries after execution failure, SafeQL interprets DBMS feedback to incrementally repair only the erroneous components. Each refinement step is formulated as a guided search within a \textit{safe query space}, where candidate queries are progressively validated through DBMS execution, thereby converging to an executable query and preventing repeated regeneration of errors. Experiments on the Bird and Spider benchmarks show that SafeQL significantly improves execution accuracy and efficiency compared to regeneration-based methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.09253v1">SkillSentry: Reliable Skill Execution for LLM Agents via Runtime Assurance</a></div>
    <div class="paper-meta">
      📅 2026-08-10
    </div>
    <details class="paper-abstract">
      LLM agents are increasingly equipped with skills to perform complex tasks through multi-step reasoning and tool use. Although skills provide reusable procedural knowledge, agents may still execute them unreliably. Even when an agent has demonstrated the capability to complete tasks under the guidance of a skill, it may fail to do so consistently across similar tasks or repeated runs due to deviations from the skill procedure or incorrect execution of individual steps. Such instability limits the practical reliability of LLM agents. To address this problem, we propose SkillSentry, a skill-oriented runtime assurance framework built upon a new domain-specific language (DSL) for representing runtime guidance for skill execution. SkillSentry initializes the runtime guidance by combining a skill specification extracted from the corresponding skill document with execution experience mined from historical successful and failed traces. It then wraps around the agent execution loop to monitor and guide skill execution under the current guidance, while iteratively refining the guidance using newly collected traces. We evaluate SkillSentry on 15 skills across two LLM agents, each paired with two backbone models, i.e., Claude Code with Claude-Haiku-4.5 and Claude-Opus-4.6, and Codex with GPT-5.2 and GPT-5.4. Our results show that SkillSentry improves the task success rate of LLM agents by 24.1% across skills, on average, while exhibiting lower variability across repeated runs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.09252v1">FSGen: Agile Fused and Sparse Accelerator Generator with Accurate Power Model for LLM Applications</a></div>
    <div class="paper-meta">
      📅 2026-08-10
      | 💬 Research Manuscript Published in Design Automation Conference (DAC) 63 (2026)
    </div>
    <details class="paper-abstract">
      With the growing demand of artificial intelligence (AI) applications, large language models (LLMs) have become important workloads in many domains. The question of how to efficiently generate optimal AI chip accelerator designs remains unresolved and challenging. Currently, there is a lack of end-to-end design methodologies for efficient design space exploration (DSE). We propose FSGen, an agile framework for attention-based LLM accelerator generation with an early-stage PPA estimator. FSGen supports fused operator dataflows and sparsity with a diverse design space and finds designs with 1.4x better power efficiency or 10x speedup with similar PPA metrics compared to prior work. Pareto-optimal designs have much better performance over a wide range of LLM benchmarks and have 58x better figures of merit (FoM). Design exploration is also faster due to our PPA estimators, which have better accuracy than prior art and reduce DSE runtime drastically.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10037v1">DOCSCHISEL: Adaptive Tool Documentation Optimization Framework for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly rely on external tools to accomplish complex real-world tasks, making tool documentation a critical grounding resource for LLM agents. Existing studies mainly focus on improving the tool-use capabilities of LLM agents, while largely treating tool documentation as a fixed input. Although several recent works attempt to optimize tool documentation through rewriting or compression, little is known about how the information contained in tool documentation affects agent performance across different settings. To bridge this gap, we conduct a large-scale empirical study on tool documentation for LLM agents. Our study reveals substantial heterogeneity in the information fields provided by existing tool documentation. Moreover, the effectiveness of different information fields is highly dependent on the task domain, LLM backbone, and agent paradigm, indicating that no fixed tool documentation can consistently generalize across diverse agent settings. Motivated by these findings, we propose DocsChisel, an adaptive tool documentation optimization framework for LLM agents. DocsChisel analyzes failed execution traces of a target LLM agent to identify documentation-related issues, and iteratively optimizes tool documentation by adding, removing, and refining information fields for each tool. We evaluate DocsChisel against two state-of-the-art baselines, i.e., EasyTool and DRAFT. Experimental results show that DocsChisel improves the task success rate of LLM agents by 95.89% over the original tool documentation and by 75.15%, on average, over existing baselines, while incurring limited optimization time and token overhead
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.06312v1">Benchmarking and Enhancing LLMs for Rule-Intensive Review of National Standard Documents</a></div>
    <div class="paper-meta">
      📅 2026-08-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly support complex professional tasks, yet their capabilities in rule-intensive document review remain insufficiently evaluated. National standard documents, such as China GB/T standards, offer a representative testbed: they are lengthy, highly structured, and governed by explicit rules for scope, terminology, normative wording, and cross-section consistency. Existing benchmarks focus on domain knowledge and question answering, largely overlooking intrinsic quality review for professional documents. Such reviews rely heavily on human experts, making them costly and difficult to scale. To bridge this gap, we introduce GB/T-Bench, the first benchmark for the structured review of national standard documents. Its GB/T Review Taxonomy is a hierarchical schema covering document structure, scope alignment, normative modality, terminology consistency, and normative references, with 25 diagnosable error types. A controllable counterexample generation mechanism combines deterministic rules and constrained LLM rewriting to process 488 documents into 7,306 traceable review error instances for evaluation. We also develop a diagnosis-oriented evaluation protocol requiring exact matches on error location, review dimension, and error type, plus document-level coverage metrics. We further propose GB/T-Reviewer, a multi-agent framework that converts review knowledge into specialized skills and coordinates global inspection, targeted diagnosis, rule scanning, and result verification. Experiments with 14 mainstream LLMs reveal a substantial human-LLM gap: the strongest model achieves only 0.3280 CMCS versus 0.6640 for experts. GB/T-Reviewer raises the best CMCS to 0.5094, showing the value of structured skill coordination for rule-intensive document review. This work paves the way for trustworthy AI in standardization and other high-stakes document domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.06301v1">HarnessOpt-Bench: Evaluating LLMs at Harness Optimization</a></div>
    <div class="paper-meta">
      📅 2026-08-06
    </div>
    <details class="paper-abstract">
      As LLMs are increasingly deployed within agentic systems, their capabilities depend not only on the model weights but also on the harness: the prompts, tools, control flow, memory, and orchestration code surrounding them. This makes automated harness optimization -- the iterative and evaluation-guided improvement of a harness by an AI system -- both an important route to improving AI systems and a demanding capability for AI systems themselves. Yet the community lacks a common protocol for measuring how well frontier LLMs perform at this task. We introduce HarnessOpt-Bench, a benchmark for end-to-end harness optimization under expensive and stochastic evaluation. An optimizer, an LLM paired with a coding harness, receives a target agent's seed harness, graded evaluation feedback, and a fixed target-evaluation budget. It edits the harness and nominates a final candidate, which is scored by its normalized gain over the seed on a held-out test partition that remains inaccessible throughout search. A trusted execution environment enforces the evaluation boundary, meters target-agent resource use, and preserves candidate versions for audit. We evaluate 5 frontier LLMs as optimizers both under a shared coding harness and under their native harnesses across 4 downstream tasks, over 111 scored runs. Experiment results show that optimizer models separate more than the coding harnesses they act through, native harnesses are not consistently superior, and gains vary substantially across tasks and seed regimes. These results establish harness optimization as a measurable and discriminative capability with large space for improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07568v3">d3LLM: Ultra-Fast Diffusion LLM using Pseudo-Trajectory Distillation</a></div>
    <div class="paper-meta">
      📅 2026-08-06
      | 💬 ICML 2026
    </div>
    <details class="paper-abstract">
      Diffusion large language models (dLLMs) offer capabilities beyond those of autoregressive (AR) LLMs, such as parallel decoding and random-order generation. However, realizing these benefits in practice is non-trivial, as dLLMs inherently face an accuracy-parallelism trade-off. Despite increasing interest, existing methods typically focus on only one-side of the coin, targeting either efficiency or accuracy. To address this limitation, we propose d3LLM (Pseudo-Distilled Diffusion Large Language Model), striking a balance between accuracy and parallelism: (i) during training, we introduce pseudo-trajectory distillation to teach the model which tokens can be decoded confidently at early steps, thereby improving parallelism; (ii) during inference, we employ entropy-based multi-block decoding with a KV-cache refresh mechanism to achieve high parallelism while maintaining accuracy. To better evaluate dLLMs, we also introduce AUP (Accuracy Under Parallelism), a new metric that jointly measures accuracy and parallelism. Experiments demonstrate that our d3LLM achieves up to 10$\times$ speedup over vanilla LLaDA/Dream, and 5$\times$ speedup over AR models without much accuracy drop. Our code is available at https://github.com/hao-ai-lab/d3LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.06183v1">MicroEvo: Knowledge-Guided LLM Sampling for Efficient Microarchitecture Design Space Exploration</a></div>
    <div class="paper-meta">
      📅 2026-08-06
      | 💬 Accepted by ICCAD 2026
    </div>
    <details class="paper-abstract">
      Microarchitecture design space exploration suffers from expansive search spaces and expensive PPA evaluation, leaving only a small simulation budget for design decision-making. Existing methods perform blind search without considering microarchitectural dependencies and fail to learn from the iterative search effectively, leading to wasted evaluations and weak Pareto convergence. In this paper, we propose MicroEvo, a knowledge-guided framework that couples off-the-shelf LLMs with Monte Carlo Tree Search (MCTS) for multi-objective microarchitecture optimization. MicroEvo combines LLM-driven evolutionary operators, a Pareto-aware tree policy that balances Pareto contribution and diversity, an active knowledge accumulation mechanism that extracts and reuses optimization insights, and state-aware directives that adapt the search behavior online. Experiments show that MicroEvo improves Pareto-front quality by up to 36.2% over NSGA-II and achieves 10.6x higher search efficiency, and also demonstrates strong scalability to a complex industrial-scale core. The code repository is available at: https://github.com/GEAR-SEU/MicroEvo-ICCAD-26.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.06166v1">What out-of-the-box LLMs can(t) do in law? A Turing test in Italian exams for lawyers, judges and notaries</a></div>
    <div class="paper-meta">
      📅 2026-08-06
    </div>
    <details class="paper-abstract">
      The article reports on a blind Turing Test experiment, assessing the performance of out-of-the-box leading LLMs on three Italian legal professional exams: the Bar, Judges and Notary exams. Leading LLMs were asked to generate full written exam papers, which were made indistinguishable from human submissions and anonymously evaluated by expert examiners, using the same criteria applied in real examinations. Results reveal marked differences across both models and tasks. While some LLMs match or exceed top human performance in adversarial legal argumentation and doctrinal analysis, all models fail in the notary exam, which requires goal-directed legal planning under strict formal and substantive constraints. Beyond ranking models, the study identifies task-specific strengths, limitations and recurring legal failure patterns. Although limited to out-of-the-box systems, the findings provide qualitative evidence on the current scope and boundaries of the legal competence of LLMs across distinct professional tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.02776v4">Topics as Proxies for Sociodemographics: How Conversational Context Affects LLM Answers</a></div>
    <div class="paper-meta">
      📅 2026-08-06
    </div>
    <details class="paper-abstract">
      When large language models (LLMs) are used in high-stakes scenarios, such as legal, medical and financial advice, even a single conversation history is enough to drive differences in outcomes between users. Prior work has demonstrated that this results in outcome disparities between sociodemographic groups, with some groups receiving more advantageous outcomes than others. In this work, we demonstrate that LLMs actually struggle to infer user sociodemographics from a single conversation history and that although there are disparities between sociodemographic groups, they are minimal in magnitude. To investigate what is the main driver of disparities between users, we compare user sociodemographics to a range of (psycho)linguistic features of conversations, including conversation topic, emotions, and readability. We find that conversation topics are most predictive of LLM-generated advice within a conversational context, which, to some extent, function as proxies for sociodemographic groups and often affect advice in unpredictable ways. This is cause for concern and highlights the need for future research to better understand the effect of conversational context on LLM outputs in high-stakes scenarios.
    </details>
</div>
