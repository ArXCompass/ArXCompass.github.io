# llm - 2026_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.12375v1">OnTrack: Real-Time Monitoring and Intervention in LLM Agent Trajectories via Streaming Structure-Aware Optimal Transport</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      Agents are deployed in applications from trip planners and stock trading to IT incident triage. In most cases, LLM agents work autonomously with minimal rule-based safeguarding, leading to cost and safety issues from irreversible actions. Recent works resolve this either by using a safeguard agent to monitor behavior or evaluating logs post-hoc. The first adds cost and latency to every step; the second delivers its verdict after the run, when tokens are burned and damage is done. To overcome this, we propose OnTrack, a streaming monitoring mechanism that compares an agent's steps and dependencies against recorded successful runs to alert users or block the agent in about a millisecond per step. We study this problem in three regimes of decreasing access: full reference access (historical runs and tool schemas), intermediate access (only tool schemas), and no prior knowledge (only step logs as generated). Expectation of OnTrack's monitoring capabilities reduces as data access drops, ranging from plan violation detection to identifying loops, stalls, and repeated tool calls. Finally, we evaluate OnTrack using SWE-bench trajectories. Based on the first 8 steps, our method ranks failing trajectories below succeeding ones better than content similarity approaches (+0.057 AUROC). With an abort policy, we save about 18% of compute that would be burned on failing runs, where 83% of interrupted runs were actually heading to failure (5 out of 6 aborts were correct).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.39882v2">LLM Persona Unlearning</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      Pre-training equips large language models (LLMs) with a broad repertoire of behavioral patterns associated with roles, styles, values, and goals. Post-training teaches conditional enactment and makes a helpful Assistant the default, but it does not erase alternative modes from the weights; explicit prompts can therefore elicit personas that repeatedly shape judgment, language, and action. In open-weight settings, runtime controls can be removed, motivating persona unlearning: a weight-level edit that makes a designated persona difficult to elicit and enact on unseen contexts. We introduce PersonaUnlearnBench, a model-specific paired benchmark spanning six LLMs from three families and five personas, with aligned forget/retain sets, held-out instruction paraphrases, and four-axis evaluation. The benchmark shows that standard unlearning methods cannot reliably erase the target persona without sacrificing meaningful generation or general utility. We therefore propose PaCE, which compares target and desirable responses to the same questions to locate an internal behavior direction, then trains target-prompt states away from the target mode and toward the matched desirable response. Experiments show that PaCE consistently suppresses target personas with high response quality and useful counterpart behavior, at moderate utility cost. These results establish persona unlearning as a distinct behavior-level editing problem and a practical route toward persistent control of latent LLM response policies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.12327v1">SparseDecoding: Decoding-Aware Pruning for Accurate and Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      The memory-bound nature of the decoding stage of large language model (LLM) inference incurs significant latency. Layer-wise training-free network pruning approaches guided by the Hessian have been a prominent solution to this problem, as pruning reduces the number of nonzero parameters read from memory during decoding. Nevertheless, typical methods in this line compute the Hessian using pre-collected natural sequences, whereas the model is fed self-generated tokens during decoding, creating a distribution shift between the two sequences. The Hessian calculated on the natural sequence is different from that calculated on the generated sequence. We observe that this discrepancy causes the activation distribution during generation to deviate from that used for pruning, further hurting the pruned model performance. Moreover, most existing LLM pruning methods that bring actual speedup primarily target the sparse matrix-matrix (SpMM) multiplication, providing limited support for the sparse matrix-vector (SpMV) operations, which dominate decoding. To solve these problems, we introduce SparseDecoding, a principled decoding-aware pruning framework tailored for accurate and efficient LLM decoding. Specifically, at the algorithmic axis, SparseDecoding constructs calibration matrices from layer-wise activations collected during the dense-model autoregressive generation, excluding prefill, thereby aligning the pruning objective with the decoding activations. At the system axis, we develop an optimized N:M sparse matrix-vector kernel with bitmask indexing and fixed-step traversal. Substantial empirical results on representative LLMs (Llama-3.1-8B, Llama-3.3-70B, Qwen3-14B / 32B) demonstrate that our method consistently outperforms standard fixed-text calibration on the long-form generation benchmarks while achieving up to 1.48x end-to-end wall-clock decoding speedup on A100 GPUs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.12325v1">Prior or Feedback? What an LLM Uses When Adapting Neural Operators</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 16 pages, 4 figures, 5 tables. Accepted at the NeurIPS 2026 Workshop on AI for Science: Verification in the Age of AI Scientists. Equal contribution
    </div>
    <details class="paper-abstract">
      Do LLM scientific agents rely only on their initial task context, or do they adapt their decisions in response to experimental feedback? We study this question in neural operator adaptation, where a large language model (LLM) selects fine-tuning configurations under a limited trial budget. Across transfers within and between partial differential equation (PDE) families, the LLM achieves lower held-out test nRMSE than random search and Bayesian optimisation in nearly every matched comparison. Endpoint performance alone cannot distinguish what happens, so we verify each attribution with controlled interventions. Before observing any validation score, the LLM's first configuration already ranks near the top of the corresponding random-search pool, indicating a useful initial bias. A complementary cold-start intervention shows that the selected base learning rate shifts with the PDE description. Once feedback becomes available, reassigning validation scores among evaluated configurations changes the next proposal in every case tested, whereas a value-preserving rewrite produces no comparable aggregate effect. These interventions establish that the LLM's decision-level actions respond to the given task and observed outcomes, showing that it combines a task-dependent prior with sensitivity to experimental feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.12304v1">Looking Inside LLMs: Small-World Connectivity as a Signature of Reasoning Performance</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      Understanding large language model (LLM) reasoning requires looking beyond behavioral performance to examine how reasoning ability is reflected in internal organization. Inspired by neuroscience findings linking higher intelligence to stronger small-world organization in functional brain networks, we investigate small-world connectivity as a structural signature of LLM reasoning. We construct functional graphs from attention-head activation similarities and find that a higher small-world index (SWI), capturing local clustering and short global paths, consistently correlates with better fluid reasoning performance across models and training checkpoints. Since local clustering is central to small-world organization, we further examine how heads important for model performance connect within and across communities. We find that these heads tend to have a larger share of connection weight within their own communities (high core scores) and a more concentrated weight distribution across communities (low bridge scores). These observations motivate the hypothesis that high core and low bridge scores serve as structural indicators of head importance for reasoning capability. We validate this hypothesis through pruning, introducing Small-World Allocation (SWA), a hierarchical sparsity allocation method guided by these scores. Across six LLMs, SWA better preserves small-world organization and model performance than competing allocation strategies, reducing WikiText perplexity by up to 20%. Together, these findings identify small-world functional connectivity as a measurable signature of LLM reasoning performance, offering a structural perspective that complements behavioral evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.06955v2">ROMA: LLM System for Real-World Object-Centric Multi-Sensory Active Perception</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      Humans inherently understand the physical world through an active process. When sensory evidence is insufficient to infer physical properties, we naturally interact with the environment by deciding what information is missing, how to acquire it, and when sufficient evidence has been obtained. In stark contrast, existing multi-sensory robot systems mainly integrate sensory inputs rather than actively acquiring missing evidence through interactions. In this work, we introduce ROMA, an LLM-based system for Real-World Object-Centric Multi-Sensory Active Perception. ROMA integrates vision, audio, tactile, and force sensing into a reasoning-interaction-feedback loop. The model identifies missing evidence and determines the target objects, interactions, and modalities, while a physical interface executes the selected interactions and collects the multi-sensory feedback. To support this capability, we construct ROMI-2K, a large-scale real-world multi-sensory object interaction dataset covering nearly 2,000 objects and 6 atomic interactions with synchronized sensory feedback. Building on these data, we develop a two-stage training framework that aligns sensory modalities and equips the LLM to assess evidence sufficiency, select informative interactions, and reason over the multi-sensory feedback. We further characterize active perception as perception chains, where acquired evidence guides subsequent interactions and reasoning, and establish ROMA Bench to evaluate single-attribute, long-horizon multi-attribute, and intent-driven active perception. Experiments show that ROMA can actively acquire missing evidence and solve complex, long-chain multi-sensory perception tasks that existing methods struggle to handle, laying a strong perceptual foundation for active multi-sensory embodied agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.12242v1">TokenRouter: Efficient Serving System for Token-Level LLM Routing</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 Accepted by NeurIPS 2026
    </div>
    <details class="paper-abstract">
      Large language model (LLM) routing distributes inference work across different models, advancing the cost-quality Pareto frontier of LLM serving. While coarse-grained routing at the session or query level has been widely adopted in production systems, recent algorithmic work shows that fine-grained token-level routing can yield substantial efficiency and quality gains. However, efficiently serving token-level routed inference poses significant challenges to existing systems. Built on single-LLM assumptions, current systems suffer from severe step desynchronization and frequent batch admission delays under token-level routing, and they also impose high implementation complexity on developers. To address these challenges, we design TokenRouter, an efficient and developer-friendly serving system for token-level routed LLM inference. TokenRouter follows the principle of request-centric programming, model-centric execution: developers describe routing logic from the perspective of a single request, while the runtime launches a subserver for each LLM and dispatches requests asynchronously. Each subserver employs a delayed-batching scheduler, whose optimal hyperparameters are derived from a mathematical throughput model of the system. Across diverse routing algorithms, workloads, and model pairs, TokenRouter achieves 2.01-64.15x higher decoding throughput than existing systems, substantially advancing the serving efficiency of token-level LLM routing. Our code is available at https://github.com/thu-nics/TokenRouter.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.10169v2">Beyond Euclidean Clipping: Overcoming Exploration Collapse in LLM RL via Riemannian Isometric Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 ICML 2026
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become a dominant paradigm for enhancing LLMs' reasoning capabilities. However, RL algorithms with PPO-Clip are inherently limited by exploration collapse. Subsequent works remain primarily heuristic and fail to identify the essential cause of PPO-Clip's failure. This work reveals the fundamental flaw of PPO-Clip: it implicitly measures policy discrepancy using Euclidean metric, which is theoretically inconsistent with the intrinsic geometry on the policy Riemannian manifold. This geometric mismatch results in overly conservative updates in low-probability regions while aggressive in high-probability regions, ultimately collapsing exploration. To correct this geometric flaw, we propose Riemannian Isometric Policy Optimization (RIPO), which guarantees isometric policy updates on the Riemannian manifold, effectively balancing exploration and exploitation. We further show that RIPO achieves a favorable bias-variance trade-off, which stabilizes optimization. Extensive experiments demonstrate that RIPO significantly surpasses existing LLM RL algorithms across seven competition-level benchmarks (up to 60% improvement over GRPO on AIME24).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.12183v1">A Closer Look at Agentic BBO: Benchmarking LLM Agents for Black-Box Optimization</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      Black-box optimization (BBO) arises in many scientific and engineering problems where objective evaluations are expensive and limited. Recent large language model (LLM) agents offer a new way to approach BBO by combining task semantics, computation, optimization tools, and feedback-driven decision making, showing great potential due to the integration with mathematically rigorous tools. However, existing agentic BBO studies use different task domains and system configurations, making their results difficult to compare and the effects of individual design choices hard to isolate. We therefore introduce AgenticBBO-Bench, a cross-domain benchmark for agentic BBO spanning synthetic functions, hyperparameter optimization, database tuning, chip design, and molecular design under a unified finite-budget evaluation protocol. In our experiments, agentic BBO achieves higher family-averaged scores than direct LLM-based methods in all five domains and outperforms the best numerical optimizers in four. We further study three factors shaping agent performance: optimization tools, task information and prior knowledge, and the role of the LLM during search. Our results show that additional numerical tools do not consistently improve performance, task semantics are broadly useful while more specific priors are less reliable, and numerical optimizers can effectively absorb gains from search trajectories established by the agent. Finally, we introduce a five-task frontier challenge within AgenticBBO-Bench and evaluate seven LLMs under the Codex agent harness, where GPT-6 Astra and DeepSeek-V4.1-Flash lie on the Pareto frontier of performance and cost among the evaluated models. Our code is available at https://github.com/lamda-bbo/agentic-bbo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.12137v1">Poster: A Preliminary Study of LLM Distillation Inference</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 Accepted as a poster paper at the 2026 ACM SIGSAC Conference on Computer and Communications Security (CCS'26)
    </div>
    <details class="paper-abstract">
      Unauthorized model distillation, in which a model is trained on the outputs of a proprietary large language model (LLM), is a growing threat to model providers. We study distillation inference: determining whether a suspect model was distilled from another model or trained independently. We formulate this problem as a hypothesis test and estimate the behavior expected under each hypothesis by training shadow models: distilled shadow models learn from the teacher's reasoning traces, whereas independent shadow models learn only from reference answers. The auditor measures how closely each model predicts the teacher's reasoning outputs and then uses the shadow models to convert the suspect's score into a calibrated p-value. In a preliminary study using Qwen2.5-7B as the teacher and Llama-3.2-3B for the suspects, our test achieves a true positive rate of 1.0 at a significance level of 0.02. These results demonstrate the feasibility of using distillation inference to detect distillation attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.12114v1">Universal Textual Teaching for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 Webpage: https://alexlu99.github.io/UTT/
    </div>
    <details class="paper-abstract">
      Knowledge distillation (KD) transfers knowledge from stronger Teacher models to weaker Student models, but most methods require training the Student parameters, thereby binding the distilled knowledge to a specific architecture and checkpoint. This implicit representation is difficult to interpret or reuse across models and limits KD for API-only or costly-to-train models. This paper studies knowledge transfer for large language models (LLMs). We introduce Universal Textual Teaching (UTT), a parameter-update-free framework that distills observed Teacher-Student knowledge gaps into a textual, interpretable, and reusable natural-language artifact called Primer. Specifically, UTT first identifies representative gap cases through paired evaluations, and iteratively updates the Primer via multi-role interactions: the Student attempts each task, the Prompter turns evaluation feedback into a teaching instruction, the Teacher provides a targeted demonstration, and the Synthesizer consolidates validated lessons. Empirically, on the challenging math (Omni-MATH-2) and code generation (KernelBench) tasks, extensive results confirm the effectiveness of the method: UTT remarkably raises the Student's accuracy from 9.4% to 48.6% and Fast1 accuracy from 9% to 35% on KernelBench, while increasing mathematical reasoning accuracy from 27.6% to 51.7%. UTT also performs better than representative prompt engineering and parameter-based KD methods. Of note, UTT is shown to be generalizable across different Teachers and Students: a Primer synthesized for one Teacher-Student pair can generalize to other Students that do not participate in the synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.12083v1">All Verdicts are Not Equal: Rethinking LLM Judge Reliability</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 Accepted at AACL IJCNLP (Main) 2026
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge is the standard paradigm for NLP evaluation, yet its systemic reliability remains poorly understood despite being widely treated as a deterministic ground truth. We present a comprehensive reliability audit, stresstesting six frontier models across four benchmarks, five prompt formats, two presentation orders, three sampling temperatures, and ten repetitions per condition. Our empirical analysis reveals severe vulnerabilities: verdicts change across identical replications at temperature zero, position-order swaps flip the majority of verdicts on challenging tasks, and the most deterministic judge achieves perfect consistency by trivially repeating incorrect verdicts, agreeing with ground truth only 51% of the time. To formalize these multi-faceted failure modes, we introduce the trustworthy verdict rate (T ), a unified metric capturing the joint probability that an evaluation is reproducible, order-invariant, and accurate. UsingT , we derive a theoretical upper bound on accuracy imposed by position bias and show that reliability is item-specific rather than modellevel. Finally, we demonstrate that shifting from pairwise win-rate to holistic rubric scoring improves trustworthiness more than any single-format prompting intervention, offering an actionable framework for robust NLP evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.08818v2">Just for FUNS: LLM-Guided Spatio-Temporal Graph Node Generation for Forecasting Unobserved Node States</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      Spatio-temporal forecasting is a cornerstone of logistics, urban planning, and intelligent transportation systems. However, constrained by deployment costs and maintenance resources, sensor networks often lack comprehensive spatial coverage, rendering Forecast Unobserved Node States (FUNS) a critical yet formidable challenge. Conventional models rely on historical observations and typically falter when encountering nodes without prior records. To address this, we redefine the problem as a conditional generation task on spatio-temporal graphs and propose GenST, a framework that introduces Large Language Models (LLMs) as a semantic bridge, leveraging a pre-trained LLM fine-tuned to extract rich semantic features from node descriptions, such as functional zones and road network structures, to compensate for missing spatio-temporal signals. Specifically, we design a two-stage generative architecture: a Spatio-Temporal VAE first compresses spatio-temporal dynamics into a latent space, followed by a Generative Transformer (GenT) that reconstructs the future states of unobserved nodes from noise, guided by multi-modal conditions including semantics, geographic coordinates, and neighborhood contexts. Experiments on six traffic and two non-traffic datasets show GenST significantly outperforms existing baselines in zero-shot prediction tasks, demonstrating the practical potential of semantic-guided generation for mitigating spatio-temporal data sparsity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.19661v3">PaReGTA: A Temporally Aware LLM-Based Patient Representation Framework for EHR Analytics</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 37 pages, 5 figures, 21 tables
    </div>
    <details class="paper-abstract">
      Temporal information in structured electronic health records (EHRs) is often lost in sparse one-hot or count-based representations, while sequence models can be costly and data-hungry. We propose PaReGTA, an LLM-based encoding framework that (i) converts longitudinal EHR events into visit-level templated text with explicit temporal cues, (ii) learns domain-adapted visit embeddings via lightweight contrastive fine-tuning of a sentence-embedding model, and (iii) aggregates visit embeddings into a fixed-dimensional patient representation using hybrid temporal pooling that captures both recency and globally informative visits. The resulting fixed-dimensional patient representations can be used with conventional downstream machine-learning models. To examine factor-level sensitivity, we use PaReGTA-RSS (Representation Shift Score), a prespecified factor-removal analysis that recomputes patient representations after removing clinically defined factor groups and quantifies the resulting change in the fitted logit of a fixed logistic-regression model. We evaluated PaReGTA in a cohort of 39,088 patients with migraine from the All of Us Research Program (AoU) on a retrospective patient-level classification task with an EHR-derived target. In an exploratory comparison on the fixed held-out test cohort of 7,818 patients, PaReGTA-Gap + LightGBM, used as a post hoc analytical reference, had the highest reported AUC, accuracy, and F1 values among the evaluated sparse, BERT-based EHR, and recurrent approaches in this cohort.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.12030v1">Natural Language to First-Order Logic LLM-based Autoformalization</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 Accepted to EMNLP-2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have renewed interest in autoformalization. Yet, when First-Order Logic (FOL) is considered as the target formalism, the field still lacks a unified task formulation and a systematic survey. This paper addresses this gap: we first provide a principled definition for the FOL-autoformalization task by distinguishing Ontology Extraction from Logical Translation, showing how their conflation obscures (cross-study) evaluation; we review existing datasets, evaluation metrics, and LLM-based methods, including fine-tuning, prompting, and verification-based refinement; we identify open challenges in benchmarking, semantic evaluation, ontology-aware methods, and end-to-end applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.09571v2">How Do LLMs Change Predictions Under Negation?</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Negation is an essential feature of human language, yet large language models (LLMs) remain unreliable in processing it. We evaluate recent open-source and closed-source LLMs on our negation benchmark and find that, in 37-71% of cases, they repeat the same answer under negation (e.g., "Madrid" for "What is not the capital of Spain?"). To understand and address this brittleness, we mechanistically examine how models operate under negation. Our main finding is that specialized attention heads and MLP neurons jointly implement negation by (1) suppressing retrieval of the original answer (e.g., "Madrid") while (2) promoting a favored candidate within the answer category (e.g., "Paris"). This contrasts with accounts of human negation processing, in which information about the original answer helps to determine what should be excluded. Furthermore, we find that this difference from human processing is a key source of negation failures: the model's mechanism relies on suppressing the original answer rather than using it to determine what to exclude, so the model can repeat the original answer when suppression is too weak or when a bias toward particular answers prevents it from selecting an alternative. To address this weakness in the model's negation mechanism, we propose a training objective that requires larger shifts in answer preference for more confident original predictions, and show that it reduces negation failures with less degradation of general capabilities than standard fine-tuning baselines. Together, our results demonstrate how mechanistic analysis can reveal why a linguistic capability fails and guide training that targets the underlying limitation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.12022v1">Examining Social Attribution in LLM Reasoning: A Theory-Guided Probing Methodology</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in sociotechnical systems where social attribution, the reasoning process attributing external events to the causes and reasons of agents' social behaviors, plays a critical role. These processes involve judgments of social cause, responsibility, and blame/credit to agents. Although attributional models are well-studied in social psychology and cognition through Attribution Theory, social attribution remains underexplored in AI, particularly LLM social reasoning. This paper provides the first systematic exploration of LLM social attribution. Our work focuses on responsibility and blame attributions, examining current LLMs' judgments and their underlying internal mechanisms. Guided by attribution theory, we construct a social attribution benchmark consisting of a Vignette subset based on classic scenarios from attribution theory research and a Reality subset based on real-world social narratives, yielding 7,639 responsibility/blame judgment questions. On this basis, we evaluate 32 representative LLMs and 5 basic non-LLM baselines. To further explore the internal mechanisms underlying the LLM judgment process, we develop a probing-based methodology to investigate the latent-space representations of 5 key attribution dimensions and the consistency of their influences on LLM judgments compared to those in human social attribution. Our research findings reveal that current LLMs exhibit measurable but incomplete agreement with human responsibility and blame judgments, and meanwhile, this agreement is positively correlated with model size. Some attribution dimensions are systematically decodable from specific positions in LLM hidden states, and their influences on the final judgment are consistent with those indicated by human Attribution Theory. The dataset and associated code are available at https://github.com/Yuzhaoxin946/SAB-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.08215v2">Learn2Play Bench: How Well Do LLM Agents Learn from Experience in Unfamiliar Environments?</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      Learning from experience is essential for LLM agents to adapt to unfamiliar and dynmaic environments. Evaluating this ability is therefore important for understanding how effectively agents acquire and use new knowledge. Existing benchmarks have sought to evaluate this ability, but they primarily evaluate tasks whose rules are provided in the instructions or already familiar to pretrained models, making it difficult to distinguish learning from interactions from reasoning with existing knowledge. To address this, we introduce Learn2Play Bench, a benchmark of newly designed text-based games, whose rules are novel or counterintuitive, requiring agents to acquire knowledge through interaction rather than rely solely on pretrained knowledge. These games provide reproducible feedback and automatic scoring, enabling controlled evaluation of learning across repeated attempts. We also vary game instances to test whether agents can apply what they have learned to new situations. Therefore, we evaluate how backbone models, self-evolving methods, and agent harnesses affect agents' learning ability, revealing three findings: (1) Experience retention: Retaining complete records of actions and feedback can support more effective learning than summarizing these experiences into rules or strategies. (2) Human agent gap: Top-performing human players achieve higher peak scores than the evaluated agents. Human explore more varied strategies, and repeat actions less. (3) Harness matters: With the backbone fixed, changing the harness can improve performance while reducing estimated inference cost. Together, these findings provide insights into how LLM agents learn from experience and suggest directions for future work to improve their learning ability. Project website: https://liushiliushi.github.io/learn2play-bench-website/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.03316v2">Multi-Task Evolution for Zero-Shot Cross-Problem Generalization using LLMs</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      Designing effective heuristics for diverse combinatorial optimization problems requires substantial expertise and repeated search. Large language models (LLMs) automate heuristic generation and refinement, but heuristic search typically depends on evaluation feedback from the problem being optimized. Generalizing to new problem definitions using only source-task feedback therefore remains a central challenge. We introduce MECo, an LLM-driven multi-task evolutionary framework for zero-shot cross-problem generalization. MECo maintains task-conditioned heuristic populations and uses a transfer gap based on cross-task population performance to guide their interactions. These interactions enable the transfer and recombination of heuristics. A complementary selection criterion then constructs a compact heuristic set by rewarding each member's additional coverage of source combinations. The selected set is applied to target problems without further search or adaptation. Experiments on 32 problem variants across vehicle routing (VRP) and flexible job-shop scheduling (FJSP) show that MECo achieves the lowest mean costs compared with eight automated heuristic design (AHD) baselines under the same budgets. On out-of-domain problems, it outperforms the strongest baseline in each family. Moreover, integrating the framework of MECo with different AHD methods improves their ID and OOD performance in both families, supporting its effectiveness across different methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11978v1">Specialized Decision Models vs. General-Purpose LLMs: Benchmarking Jev Across Knowledge, Reasoning, and Multilingual Tasks</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 10 pages, 2 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Jev is a "System One" model that returns a choice among given options instead of generating text. We study how such a specialized decision model compares with general-purpose large language models (LLMs). We evaluate Jev on 13 multiple-choice benchmarks covering knowledge, reasoning, and multilingual understanding, and compare it with 19 LLMs in three tiers: frontier, representative, and small. Jev is competitive with frontier LLMs on knowledge and commonsense benchmarks and obtains the best score on MMLU-Redux and ARC-Challenge. Outside mathematics, it also outperforms most representative LLMs and all small LLMs. However, it falls behind on mathematical word problems: on MathQA, it is 17.7 points below the frontier median and scores lower than all 19 LLMs. These results indicate that a specialized decision model can match general-purpose LLMs on decisions that rely mainly on knowledge, but not on decisions that require multi-step calculation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11963v1">Can LLMs Fix It Without Code? Toward Automated Verification of No-Code Bug Fixes</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      A no-code fix resolves an invalid bug report by directing the user to change a setting, update to a version where the problem is already fixed, or adjust their workflow. Manually verifying whether a proposed no-code fix resolves the reported bug takes considerable developer time. This study proposes an automated, execution-based pipeline for evaluating the capability of large language models (LLMs) to generate no-code fixes in a real browser environment. We evaluate 322 no-code fixes generated by the 12 configurations released with the benchmark of a previous study, covering bug reports categorized as Faulty Configuration, Wrong Version, or External System & Dependency. An executor agent applies each fix by following its natural-language instructions, and an issue-specific checker determines whether the reported bug persists. We repeat the pipeline with three executors: two Computer-Use Agents, OpenCUA-72B and Claude Sonnet 5, and one multimodal agentic LLM, Meta's Muse Glimmer. Only 17.6% of the candidate issues could be set up and passed both sanity gates. Across the 322 fixes, 14.6% to 49.7% resolved the bug depending on the executor, and the strongest configuration, Claude Opus 4.6 in the Vanilla pipeline, resolved up to 74.1% of its fixes under Claude Sonnet 5. Changing only the executor shifted a configuration's resolution rate by 38.8% on average, and the three executors reached the same verdict on only 46.9% of the fixes. Compared with human execution, the executors matched the human consensus for 66.1% to 88.1% of the sampled fixes. Even under the best executor, fewer than half of the LLM-generated no-code fixes resolve the reported bug, so such fixes need verification before they reach users. Execution-based verification can provide this, but the measured capability depends strongly on the executor, which evaluations must report and control.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11901v1">Can Decision Models Understand Stance? Evaluating Jev Against General-Purpose LLMs</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 8 pages, 1 figure, 5 tables
    </div>
    <details class="paper-abstract">
      Stance detection requires identifying an author's attitude toward a given target, sometimes based on conversational context. Jev, a specialized decision model designed for structured decision-making, offers an alternative to general-purpose large language models (LLMs). In this work, we evaluate Jev on two stance detection datasets, VAST (English texts) and ZS-CSD (Chinese conversations), comparing it with four general-purpose LLMs and two fine-tuned models. Results show that Jev achieves competitive performance on VAST, matching GPT-5.6 and outperforming the other general-purpose LLMs. However, it falls behind stronger LLMs on ZS-CSD, particularly in distinguishing favor from against. Further analysis suggests that this limitation may be related to understanding reply relationships and stance direction rather than conversation length alone. These findings highlight both the potential and limitations of Jev for stance detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11899v1">Forms of LLM-Integrated Applications from LLM-Chats to Autonomous AI Agent System</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly embedded as components in software systems, marketed under labels such as chatbot, copilot, retrieval-augmented generation, workflow, coding agent and AI agent. Whether these labels denote genuine architectural forms or serve as branding has not been assessed systematically. In the sources surveyed, labels do carry architectural content, most clearly in vendor usage: copilot denotes a router-worker architecture operating a host application under step-by-step user confirmation, while the more recent shift to the label agent coincides with AI-planned multi-step execution of which the user sees only the outcome. The coding agents of four major providers share one architecture, a reason-and-act loop delegating to subagents. This survey describes seven recurring forms---LLM chats, custom agents, retrieval-augmented generation (RAG), AI-enhanced workflows, copilots, coding agents, and, in part, agentic RAG---in a common vocabulary of agents and tools. Each is characterized along four structural dimensions (agentic RAG only partially): the architectural pattern, the control of execution and the point of user intervention, the number of agent calls per task, and tool use. An illustrative corpus of 22 systems from research publications and vendor documentation grounds the descriptions and shows where they reach their limit.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11843v1">Anytime-valid detection of LLM weight exfiltration</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 Accepted for oral presentation at the NeurIPS 2026 workshop E-Values: From Statistics to ML (Paris, December 13 2026)
    </div>
    <details class="paper-abstract">
      A compromised LLM inference server can leak model weights by encoding payload bits in otherwise plausible token choices. A replay of the same prompt in a trusted server can expose such deviations, but benign numerical nondeterminism also causes token mismatches. Patient attackers can therefore hide within normal variation unless evidence is combined across responses. We introduce a prompt-level e-process that calibrates whole-response mismatch events on trusted benign traffic and accumulates evidence sequentially while, under a calibration-transfer assumption, controlling the probability of any false alarm over an unbounded monitoring horizon. We evaluate it on four models against a seed-blind attack and a stronger seed-aware attack that hides payload bits only in near-ties to remain stealthy, analyzing the channel capacity vs detectability trade-off. Compared with a hard per-token alarm, the e-process combines weak evidence across responses while providing explicit anytime false-alarm control.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.34565v2">FlowState: Execution State as Memory for Long-Horizon LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      Long-horizon tasks require LLM agents to continually draw on information from earlier interactions. However, retaining the full history increases context costs, while compressing it risks losing details needed later, and the relevance of historical information often becomes apparent as the task progresses. To address these challenges, we propose FlowState, which treats execution state as memory that can be retained and revisited across requests, unifying current decision-making with the reuse of historical information. FlowState preserves semantically typed state nodes, their relations, and references to raw tool observations, separating persistent retention from on-demand access. Within a single execution loop, Incremental State Update (ISU) maintains the current state based on new inputs and feedback, while Progressive State Access (PSA) progressively reveals historical states and supporting evidence as needed during reasoning. Together, these mechanisms enable agents to reassess prior decisions in light of new information and guide subsequent actions. Compared with a full-context baseline using the same DeepSeek-V4-Flash model, FlowState improves the average success rate on MemoryArena and the average pass rate on $τ^3$-Bench by 4.55 and 13.95 percentage points, respectively, while reducing total token consumption by 43.2% and 40.6%. These results demonstrate the performance and efficiency advantages of FlowState on long-horizon tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11835v1">On the Risks of using LLM-Generated Tests for Regression Testing</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      Software is under constant evolution: developers continuously add features, fix bugs, and refactor code, and any of these changes may break existing functionality. Regression testing guards against such effects by capturing expected behavior in test cases. LLM-based test generation aims to automate this process by generating regression tests directly from the code under test. This is beneficial when the implementation is correct, but problematic when the code contains faults: the generated tests may then encode and preserve incorrect behavior. To investigate this risk, we apply LLM-based regression test generation to pull requests merged into the main branch of software projects and study the impact of the generated tests on subsequent project evolution. We distinguish between fault-revealing tests, which assert correctly implemented behavior, and fault-enforcing tests, which assert faulty behavior. Across 145 pull requests from SciPy, Qiskit, and pandas, 8%-17% of the generated tests are fault-enforcing, while only 2.4%-4.8% reveal faults. Fault-enforcing tests persist over time: after several subsequent commits, 83%-91% of them are still relevant and pass. They also accumulate: when the faults of all pull requests are combined in one codebase, 83%-92% remain enforced at the end of the commit history, and the developer-written test suite detects only 14%-30% of them. Our results reveal a fundamental risk of LLM-generated regression tests: without manual validation, they may encode faulty behavior as expected behavior, allowing bugs to persist across software revisions and largely evade developer-maintained test suites. LLM-based regression testing can thus give rise to a new form of technical debt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.32382v2">PlurVA-LLM-2026 Shared Task Track-1: Pluralistic Value Alignment in LLMs via Multilingual Fine-Tuning and Threshold Calibration</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 11 pages, 6 figures, 6 tables, Accepted paper at the first workshop on Pluralistic Value Alignment of LLMs @ AACL-IJCNLP 2026
    </div>
    <details class="paper-abstract">
      We present our system for the PlurVA-LLM 2026 Shared Task Track-1, which focuses on pluralistic value alignment in the contexts of China, Indonesia, and Sri Lanka. For this resource-constrained track, we fine-tuned Llama 3.1 8B Instruct using 4-bit QLoRA. Our approach combines option-permutation augmentation for Chinese data, annotator vote expansion for Indonesian data, and binary reformulation with SinhalaMMLU augmentation for Sri Lankan data. We further applied conditional threshold calibration to the predictions for the Sri Lankan data. The final system achieved accuracies of 0.785 for Chinese, 0.715 for Indonesian, and 0.916 for Sri Lankan, resulting in an overall macro-average accuracy of 0.805.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11768v1">Narrow and Deep: An Ontology Tower as the Knowledge of an LLM Agent for an Industrial Equipment System</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 28 pages, 6 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents are beginning to operate industrial energy equipment, and what they get right depends on what they are told about the plant. Established building ontologies name many kinds of points across many sites, whereas an industrial equipment system needs few entities with much knowledge about each. This study proposes the ontology tower, a narrow-and-deep ontology of a single equipment system whose knowledge deepens in two ways: through quantities derived from the measured points by physical relations, and through lessons from the operating journal incorporated as knowledge nodes. On a real low-humidity air-handling test plant operated daily through a programmable logic controller, agents received a text projected from its tower in a preregistered evaluation of nine tasks replayed from the plant's records, using four open-weight models from 9 to about 750 billion parameters. This knowledge raised the rate at which the agents avoided the most plausible misjudgment of each task by about 20 percentage points, and the overall task score of the 9-billion-parameter model as much as that of the largest. Operating lessons were used when incorporated into the tower or placed in the prompt as records, but seldom when left in the journal behind a search tool. In live runs through an invariant safety layer, the agents brought the controlled variable into its target band in 12 of 14 runs. An ontology narrow in entities but deep in what is known about them can thus supply the knowledge that an agent for an industrial equipment system needs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11766v1">Same Outcome, Different Evidence: Intent Recovery in LLM Safety Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 12 pages, 2 figures. Code and experiment inputs: https://github.com/kevinjiang0121-cyber/IRIS
    </div>
    <details class="paper-abstract">
      Safety evaluations of large language models commonly summarize harmful-output behavior with attack success rate (ASR). Yet the same non-harmful outcome can arise for very different reasons. A model may recover a harmful task and refuse it, fail to recover the task, or respond to something else entirely. Distinguishing these cases becomes especially important under intent-obscuring prompts, where a low ASR does not reveal whether the evaluated task was actually engaged. To make this distinction explicit, we pair ASR with operative understanding rate (UR), which measures whether a response both identifies the evaluated task and treats it as the task to be answered. Across interfaces, this paired view reveals substantial variation hidden by ASR: similar ASR values can correspond to sharply different recovery rates. Controlled English reconstructions show that recovery consistently improves as compressed prompts become more explicit, whereas ASR does not follow the same pattern. A complementary contrast comes from FormalLogic, where high recovery can still coincide with frequent harmful assistance. Together, these results show that non-harmful outcomes are not equally informative about model safety, motivating the joint reporting of intent recovery and ASR in LLM safety evaluation. Code and experiment inputs are available at https://github.com/kevinjiang0121-cyber/IRIS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11765v1">Thinking Inertia: LLMs Keep Thinking When Told Not To</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 Accepted by NeurIPS 2026. Website: https://thinking-inertia.github.io GitHub: https://github.com/thinking-inertia/code
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly ship with explicit "thinking modes", yet their counterpart, "no-thinking", has received far less attention. We study LLMs' no-thinking behavior along two axes. a. How to measure no-thinking? Prior work typically defines no-thinking through proxies such as a disabled thinking mode or the absence of long traces. These proxies are unreliable: disabled thinking modes may still emit reasoning, while long traces may contain filler rather than genuine inference. We instead normalize each response into a pre-answer trace and final answer, and evaluate it at three levels: (i) Empty-Thinking Rate for strict answer-only compliance; (ii) instruction-aware Question-Pre-answer Relevance for similarity between the question and pre-answer trace; and (iii) LLM-as-judge Explicit Inference Rate for visible explicit inference. Together, these metrics distinguish answer-only output, relevant but non-inferential text, and explicit inference. b. How does no-thinking vary across tasks and models? We evaluate six prompting interventions on six LLMs across Boolean, multiple-choice, and open-ended questions. We find that explicit no-think controls cannot reliably eliminate visible inference. Models instead exhibit "Thinking Inertia": explicit inference persists even under strict controls and becomes more prevalent as the answer space opens. Accuracy remains stable on Boolean and multiple-choice tasks, whereas open-ended tasks reveal a trade-off between answer-only compliance and task accuracy. Rewriting the same questions across answer spaces shows that supplying candidate answers makes answer-only responses easier to produce. These findings establish no-thinking as a non-trivial capability: stopping explicit reasoning cannot be assumed from model settings or instructions alone and deserves systematic evaluation alongside reasoning ability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11732v1">MemTrial: Learning When to Trust Memory in LLM Portfolio Agents</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents for portfolio management learn from experience: they credit each experience in their memory with the outcome of the decisions that used it. In financial markets, however, this outcome mostly reflects the market move shared by all decisions on that date, so the credit tracks the market rather than the experience, and these agents often do worse than simply holding the equal-weight (1/$N$) portfolio. We ask how an agent can credit an experience with what it changes, and answer it by putting memory on trial: drafts of the same decision with and without an experience face the same market, so the outcome they share cancels in their difference. Our agent, MemTrial, drafts each decision with eight combinations of its retrieved experiences, chosen by a fractional factorial design, and credits each experience with its Banzhaf value, the average of these differences. As each date occurs once and each draft is a noisy LLM sample, these credits are noisy and may not hold on new dates. MemTrial therefore pools them across dates and similar experiences with a hierarchical Bayesian model, acts on them only after they have predicted unseen dates, and otherwise stays anchored at a conservative reference such as 1/$N$. On four benchmarks, MemTrial not only benefits from experiences that matter (the best of 15 methods on a semi-synthetic benchmark with known experience quality) but also limits its losses when its values do not hold (at most 2.2\% below 1/$N$ on PortBench and InvestorBench, against 15--38\% for the best experience-learning agent). Averaged over five settings, it improves the utility of the best experience-learning agent by 21.2\%, and with eight LLMs it beats every LLM-based baseline on InvestorBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2407.00079v5">Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 23 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Mooncake is the serving platform for Kimi, a leading LLM service provided by Moonshot AI. It features a KVCache-centric disaggregated architecture that separates the prefill and decoding clusters. It also leverages the underutilized CPU, DRAM, and SSD resources of the GPU cluster to implement a disaggregated cache of KVCache. The core of Mooncake is its KVCache-centric scheduler, which balances maximizing overall effective throughput while meeting latency-related Service Level Objectives (SLOs). Unlike traditional studies that assume all requests will be processed, Mooncake faces challenges due to highly overloaded scenarios. To mitigate these, we developed a prediction-based early rejection policy. Experiments show that Mooncake excels in long-context scenarios. Compared to the baseline method, Mooncake can achieve up to a 525% increase in throughput in certain simulated scenarios while adhering to SLOs. Under real workloads, Mooncake's innovative architecture enables Kimi to handle 75% more requests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11651v1">Evidence-Traceable Dynamic Interviewer Architecture for Expertise-Adaptive Qualitative Interviews Using Local LLMs</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 Accepted to be part of the book titled: AI in Education : Pedagogy, Ethics, and Society which will be published by Springer Nature in the Book series: Transactions on Computational Science and Computational Intelligence
    </div>
    <details class="paper-abstract">
      Automated interviewers and conversational agents are increasingly used in research, recruitment, customer service, and education. However, many existing systems rely on fixed question sequences and provide limited context-based personalization without considering participants' knowledge, which can lead to repetitive or irrelevant follow-up questions. Therefore, there is a need for an adaptive interviewing system that can adjust question depth while maintaining conversational continuity and semantic progression. To address this, an Evidence-Traceable Dynamic Interviewer Architecture is presented using a locally hosted Large Language Model (LLM), with the interview continuously adapted throughout the entire conversation based on the participant's responses and evolving context. The interviewer profiles participants' expertise in real time to generate knowledge-appropriate questions, well-articulated responses, and smooth transition messages that support conversational continuity. A five-module prompt-driven architecture and persistent interview-state record support these functions. The interviewer was evaluated with 246 participants. Expertise Profiling module (M3) showed 78.9% exact agreement with independently reported participant expertise, with a weighted Cohen's K of 0.80. Generate Iterative Questions module (M4) showed a strong expertise-complexity association (p=.79, p<.001), and participants reported high relevance (mean 4.41), engagement (mean 4.32), and satisfaction (mean 4.38), providing evidence that the architecture's adaptive components operated consistently with their intended functions while participants reported a positive interview experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11602v1">Where Do the Tokens Go? Understanding and Reducing Costs in LLM Agents for Vulnerability Discovery</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      LLM agents can spend millions of tokens during vulnerability discovery without producing a working proof of concept (PoC). What consumes that budget, and why does it fail to produce results? We diagnose these costs and failures through a multi-axis open-coding study of 200 CyberGym traces, spanning four agents (i.e., Codex, OpenCode, Cybench, and EnIGMA) under an unaided baseline and four existing efficiency methods. The study reveals three key findings. First, different agents vary substantially in success and cost, and higher spending does not consistently yield better outcomes. Second, code localization and understanding, together with vulnerability reasoning and trigger design, account for 60.4% of tokens and represent the two leading bottlenecks in failed runs. Third, only 24.4% of matched comparisons preserve success at lower total cost; unsuitable signals and auxiliary overhead limit the benefits of existing methods. Motivated by these findings, we present AVRI, an Agent-centric Vulnerability Reasoning Interface built around a persistent Bidirectional Evidence Trace (BET). BET connects how the harness consumes input with the conditions required to make a selected operation unsafe, retaining source-supported correspondences alongside the agent's hypotheses and open questions. Reading, analysis, and persistence commands help agents build and reuse this evidence rather than repeatedly retrieve and reconstruct it. On 20 evaluation tasks, AVRI reduces total cost by 18.0% for Codex and 23.7% for OpenCode while preserving success rates and improving or maintaining recall.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11600v1">Error-Propagation Modeling for Failure Attribution in LLM-Based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      LLM-based multi-agent systems (MASs) are increasingly used to solve complex tasks through coordinated reasoning, tool use, and interaction with external resources. However, attributing failures in such systems remains challenging because the observed outcome often does not directly reveal the error responsible for the failed execution. In this work, the attribution target is the decisive error, defined as the agent--step pair whose correction would recover the failed execution. Existing approaches largely identify suspicious steps without explicitly modeling how errors propagate across interactions or persist in unresolved loops, making decisive errors difficult to distinguish from downstream failure symptoms. We propose \textbf{E}rror-Propagation \textbf{M}odeling for \textbf{F}ailure \textbf{A}ttribution (\textbf{EMFA}). EMFA constructs a structured representation of the failed trajectory, models both cascading propagation and persistent interaction loops, and uses propagation-aware candidate screening followed by counterfactual verification to identify the decisive agent--step pair. On the Who\&When benchmark, EMFA achieves state-of-the-art step-level attribution accuracy and remains competitive at the agent level. It improves the previous best step-level results by 3.45 and 4.40 percentage points on the Hand-Crafted and Algorithm-Generated subsets, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11598v1">Overview of the NTCIR-19 Automatic Evaluation of LLMs 2 (AEOLLM-2) Task</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 NTCIR-19
    </div>
    <details class="paper-abstract">
      In this paper, we provide an overview of the NTCIR-19 Automatic Evaluation of LLMs 2 (AEOLLM-2) task. Building on the success of the NTCIR-18 core task AEOLLM, we proposed AEOLLM-2 for NTCIR-19 to further investigate automatic evaluation methods for Large Language Models (LLMs), particularly in long-form text generation scenarios. In AEOLLM-2, we introduced a new subtask, Deep Research Evaluation, which focuses on the automatic evaluation of long-form deep research reports generated by LLMs. Participants developed evaluation methods to automatically assess the quality of these reports, and the performance of each method was measured by comparing its scores against human-annotated ground-truth labels. This year, we received 91 runs from 10 teams in total. This paper describes the background of the task, the dataset construction, the evaluation measures, the participants' methods, and the final evaluation results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11586v1">Measuring Cultural Alignment Beyond the Average: A Framework for Evaluating Maternal-Health LLM Interactions in Indian Contexts</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      Existing evaluation methods for healthcare LLMs primarily assess factual correctness,safety, and fluency, while providing limited insight into whether generated interactions reflect culturally situated healthcare reasoning. This limitation is particularly important in maternal health, where care decisions are shaped by social and relational norms. We introduce MH-INDIC, a culturally grounded evaluation framework for maternal-health interactions in urban and semi-urban North Indian contexts that operationalises cultural behaviour through ten dimensions of maternal-health reasoning. Using a 26-item survey administered to 102 pregnant and postpartum women from urban and semi-urban North India, we evaluate ten LLMs. We distinguish population level cultural alignment from profile-level behavioural variation. Although several models approximate the human population-level distribution, all evaluated systems exhibit substantially lower variation across demographic and household profiles than the human cohort, revealing a gap between aggregate alignment and profile-conditioned sensitivity. As a downstream application of MH-INDIC, we use the strongest-aligned proprietary and open-source models to generate culturally conditioned maternal-health dialogues under zero-shot, self-conditioned, and human-grounded prompting. Human-grounded conditioning produces stronger profile alignment and dialogue quality ratings, suggesting that measured cultural profiles can improve the cultural grounding of generated interactions
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11585v1">Adapting English Quality Classifiers for Multilingual LLM Pretraining Data Selection</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      Recent advances in large language model (LLM) pretraining highlight the role of high-quality training data in improving performance. While model-based filtering has proven effective in selecting high-quality subsets from web-scale corpora, especially for high-resource languages, low-resource languages face challenges due to limited availability of annotated data. This work explores extending quality filtering to over 100 languages by proposing a multilingual adaptation approach that converts an existing English quality classifier into a multilingual variant. Our approach proposes training a small multi-layer perceptron on top of Transformer encoder-only model embeddings, using multilingual text as input and scores obtained from English classifiers applied to machine-translated text as labels. Our 1B, 3B and 8B scale experiments show that our approach maintains the downstream LLM benchmark performance of existing multilingual model-based filtering baselines, without harming regional and cultural knowledge benchmarks. To further evaluate cross-lingual generalization, we compare classifier scores of high-quality synthetic data and web samples, and the correlation of classifier scores with LLM-based ones, revealing that the classifier can learn the scoring criteria of its original English variant, even for languages not included in its training data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11573v1">Memory Type Varies: Empowering LLM Agents for Long-Term Memory with Diverse Strategies</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 NeurIPS 2026 Accept Paper
    </div>
    <details class="paper-abstract">
      The memory capabilities of Large Language Models (LLMs) have garnered increasing attention recently. Despite great success achieved, existing retrieval-based memory approaches typically overlook the differences between memories and employ a unified strategy to process all memories, leading to suboptimal performance. Thus, an intuitive question arises: can we categorize memory into different types and select appropriate strategies? However, given the topic-rich, scenario-complex, and boundary-blurred nature of memory scenarios, achieving precise classification of memories is not easy. To address this challenge, we propose a memory multi-class dataset in this paper, termed TriMEM, which provides precise annotations for memory types across diverse scenarios. Building upon this foundation, we propose a novel memory framework, named MemoType, which can adaptively recognize each memory and query type with the learned router model. With the memory and query routing, MemoType can retrieve the memory with corresponding query types and design tailored retrieval strategies, thereby enhancing the retrieval performance. Moreover, we theoretically prove that any single retrieval strategy is subject to a fundamental upper bound on its expected retrieval precision in multi-class corpora, leading to systematic precision degradation. Extensive experiments on three datasets demonstrate that MemoType consistently outperforms existing methods, achieving up to 16.18% improvement in Recall@1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11556v1">SoK: Are LLMs Reliable at Source Code Recovery? A Taxonomy and Empirical Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      Effective source recovery is critical to security applications such as malware analysis, vulnerability assessment, and legacy maintenance. Large Language Models (LLMs) are reshaping this field, shifting the paradigm away from rule-based heuristics to probabilistic and high fidelity semantic recovery of source code from assembly or classical decompiler-derived pseudo-C. However, despite rapid progress, the field suffers from fragmentation across numerous approaches as well as their non-unified evaluations, limiting objective comparisons. Further, existing works have limited coverage of embedded, IoT architectures and source languages beyond C/C++. In this work, we present the first Systematization of Knowledge (SoK) focused specifically on LLM-assisted binary-to-source recovery. We provide a granular design-centric taxonomy of LLM-assisted source recovery methods and systematic evaluations using seven key metrics along six evaluation dimensions. We evaluate state-of-the-art methods on 45,000 test samples derived from four standard and five embedded architectures, five optimization levels, and symbol stripping. We ablate the impact of design choices on recovery performance, including input representation, contextual enrichment, model scale, iteration and review roles using controlled in-house recovery pipelines and three off-the-shelf models. Finally, we test same-language and cross-language recovery capability covering four mature and two legacy languages. Our systematization and comprehensive evaluations provide key insights that guide future directions in this field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11542v1">Constitutional Gating and Deterministic Recovery for Multi-Agent LLM Negotiation: Ablations Against a Stateful Adversarial Gatekeeper</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 29 pages, 3 figures. The Gatekeeper, agents, constitution, lexicon, 30 run logs and analysis scripts are released (see Appendix F). Companion paper: arXiv:2610.09772
    </div>
    <details class="paper-abstract">
      Multi-agent LLM systems negotiating with a stateful counterpart waste model calls in three ways: polite loops that never meet the counterpart's hidden acceptance condition, malformed outputs that trigger retries, and compliance deadlocks in which the counterpart demands something the agent must refuse. We study a three-part control stack - a 5-Pillar runtime constitution, a 4-tier swarm (Director, three-agent majority vote, Monitor, schema hard gate) and Cognitive Annealing (deterministic deadlock detection, atomic purge of the agent-side context, a canonical recovery message) - against a released adversarial Gatekeeper whose acceptance rules are fixed regular expressions and whose LLM only renders reply text. The testbed has a known solution: it measures whether the stack executes a constitution-aligned strategy against swarm drift and recovers from deadlock, not whether it discovers anything. In five runs per configuration (30 runs; Gemini 2.5 Pro agents, Claude Haiku 4.5 Gatekeeper) we find: (i) the constitution and Director make an acceptable framing possible but not reliable - 0/5 baseline unlocks versus 1/5 and 2/5 with the constitution; when the swarm unlocks it does so in one turn with 7-8 calls and about 15k tokens (67-73% below baseline); when it does not, it costs 17-38% more; (ii) the Monitor and hard gate do not reduce unlocks and leave an audit trail; (iii) under a honeytrap-to-compliance deadlock, LLM-only steering escapes 0 of 5 times while atomic purge plus a canonical strike escapes 5 of 5 (Fisher $p = 0.008$) at the same call budget, with zero calls for the strike. LLM-written strikes failed the deterministic pre-flight 5 of 5 times although an LLM Monitor had approved four. Pre-registered hypotheses on average call and token reduction were not supported. Cost is bounded in every arm by deterministic stop rules; the stack adds recovery at no extra model cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.24259v3">MemCalib: Benchmarking and Optimizing Memory Use in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      The effectiveness of agent memory ultimately depends on whether the underlying LLM gives each memory in context an appropriate degree of influence over its response. Yet this capability has remained largely overlooked. To assess this capability, we introduce MemCalib, a benchmark grounded in realistic memory-system scenarios for evaluating memory use and advancing optimization algorithms. Results on the MemCalib test set reveal that frontier open- and closed-source models struggle to use memory appropriately. They frequently over-use or under-use memory rather than matching each proposition's actual use to its target level, leading to biased, low-quality responses. Experiments with common post-training algorithms, including group relative policy optimization and on-policy self-distillation, further reveal a clear directional skew: trained models improve in one direction while deteriorating in the other. We therefore propose MemCalib-RL, an ordered bidirectional counterfactual credit-assignment algorithm that separates over- and under-use signals and localizes their credit to response tokens through exact atom ablation. Results across model families and scales (Qwen3-8B, Ministral-3-8B-Instruct, and Qwen3.5-35B-A3B) show that MemCalib-RL achieves the best overall performance while better balancing over-use and under-use, with gains generalizing beyond MemCalib in external benchmark evaluation. Further experiments support its design choices and robustness and provide insight into its training dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11539v1">Design Creativity Bench: Measuring creativity in LLM-Generated UI</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      As leading LLMs improve on capability evaluations, their limitations in producing creative outputs on design tasks remain insufficiently characterised. Our work introduces Design Creativity Bench, a benchmark that evaluates diversity and appropriateness in UI designs. It measures distinctiveness among models on the same prompt (originality), how much a model's designs change between two prompts for the same UI goal in different product domains (creative range), and the share of a brief's acceptance criteria each design meets (appropriateness). Originality is 0.592 for same-prompt design pairs from different models (95% CI [0.582, 0.602]), far below the 0.764 for same-prompt human-model pairs (95% CI [0.751, 0.778]). Creative range is 0.581 across models (95% CI [0.567, 0.597]), against 0.902 for human designs (95% CI [0.884, 0.919]). Appropriateness is above 90% for every model, and the best model reaches 99.2%, slightly above the 98.0% for human designs. Our work shows that the default output of LLMs, though generally appropriate, is substantially more repetitive than the human baseline. This calls for strong measures to address the issue.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11514v1">SSCBench: Evaluating the Evidential Validity of Fault-Injection Tests for Tool-Using LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      Fault injection is increasingly used to evaluate the reliability of tool-using LLM agents. However, there has been limited study of how fault-adoption results should be interpreted when the agent itself determines which authoritative observations become visible during execution. In this paper, we present a systematic study of this evidential validity problem in agent fault-injection evaluation. We develop a measurement protocol that specifies what observations can refute an injected assertion, determines whether they can become visible before the affected fact is first used, and records whether the evaluated execution actually realizes this condition. We construct SSCBench as an instantiation of the protocol and evaluate four fault operators and five agent configurations over 1,191 faulted executions in two $τ$-bench environments. Our experiments show that an admitted fault case and agent configuration can realize substantially different evidential conditions across executions, and that aggregate adoption can remain well defined even when the population supporting a timely-counterevidence claim is sparse or absent. For example, among 44 adopted runs in which counterevidence eventually became visible, only 17 received it before first use, while 27 received it afterward. We also find that first-error timing and later stance revision need not coincide, and that automated trajectory analysis can recover adoption without reliably recovering the first faulty-reliance event needed for temporal diagnosis. We argue that the evidential condition realized by an execution and the population supporting a claim-specific interpretation are part of fault-injection evaluation itself and should be reported before adoption is interpreted as failure under pre-use counterevidence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11510v1">Does Modern Standard Arabic (MSA) Dominate Arabic Dialects in LLMs? A Representation-Level Analysis</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 EMNLP 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often default to Modern Standard Arabic (MSA) when generating Arabic, even when prompted with dialectal Arabic. A natural explanation is that their internal representations are dominated by MSA. We test this hypothesis by adapting the language-dominance framework of Shani and Basirat (2025) (https://doi.org/10.18653/v1/2025.blackboxnlp-1.7) to 26 Arabic varieties. Across layers and model families, we find no evidence that MSA acts as a dominant internal representation for Arabic dialects. Instead, dialect representations form a dense and highly overlapping space: normalized mutual information drops sharply compared to patterns reported for more distinct languages. Moreover, the strongest separability effects are not confined to intermediate layers, but can shift toward later layers depending on the architecture. These findings challenge a common interpretation of MSA-biased generation: output preference does not necessarily reveal internal representational dominance. Analyses of multilingual and dialectal LLMs should therefore distinguish generation bias from the geometry of internal representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11501v1">Beyond Sequences: Distilling Structured Decision Memory for LLM Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      Despite the adoption of large language models (LLMs) in recommendation systems, prevailing approaches mostly model single-type behaviors (e.g., views or purchases). Even when incorporating multiple behaviors, existing methods flatten heterogeneous actions into homogeneous token sequences, ignoring their distinct decision-making roles. This flattening fails to capture semantic hierarchies and contextual nuances in complex decision-making, such as trade-offs between price and quality. Consequently, performance degrades in critical ``difficult-choice'' scenarios involving highly similar items. To bridge this gap, we propose MARI (Memory-Augmented Recommendation with Interpretability), which grounds predictions in explicit, structured decision evidence. MARI maintains a Decision Memory Bank (DMB) that archives users' past rationales as Structured Decision Memories (SDMs): concise records of goals, constraints, and trade-offs. These SDMs are generated offline via Post-Hoc Decision Distillation from heterogeneous behaviors and user-generated content. By retrieving relevant SDMs to augment LLM reasoning, MARI achieves interpretability and scalability without the prohibitive cost of processing long raw sequences. Extensive experiments show MARI significantly outperforms state-of-the-art baselines on standard next-item prediction and a newly introduced Difficult Choice Prediction task, incurring low latency overhead by decoupling memory construction from online inference. Qualitative analyses reveal actionable, human-readable insights into user decision-making, marking a concrete step toward reasoning-aware recommendation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11447v1">Closed-loop evaluation of LLM agents for embedded software development</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 Published in Journal of Systems Architecture 179 (2026) 103937. 23 pages, 4 figures, 5 tables. Code and artifacts: https://github.com/jgcarrasco/closed_loop_evaluation_agents_embedded
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed as coding agents that edit files, run builds and tests, inspect execution results, and repair software iteratively. Embedded firmware is a demanding target because correctness depends on closed-loop behavior under sensing, timing, and safety constraints, not only on static source quality. Yet embedded-agent evaluation remains limited and often emphasizes one-shot synthesis or offline correctness. We present a benchmark for closed-loop evaluation of embedded coding agents. Each task provides a plain-text engineering description, constrained workspace, and visible build-and-runtime surface. The agent must translate requirements into implementation and self-verification steps, then iterate until the required device behavior is achieved. The suite contains five embedded-control tasks and four feedback scenarios: one-shot generation, realistic self-verification, CI-style red/green feedback, and oracle-style detailed feedback. The implementation targets simulated ESP32 firmware for reproducibility. We evaluate seven GPT-family and Qwen-family configurations across five tasks and four scenarios, with three repetitions per condition for 420 runs. gpt-5.4 has the highest pass rate among evaluated configurations but does not saturate the benchmark; qwen3.5-27B is the strongest observed local model; and smaller local models degrade sharply in pass rate and search efficiency. These results suggest that capable local embedded coding agents are emerging.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28510v3">Efficient and Scalable Provenance Tracking for LLM-Generated Code Snippets</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 Article accepted at The Journal of Systems & Software
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) for code completion and generation are increasingly used in software development, yet they may reproduce training examples verbatim and without authorship attribution, raising legal and ethical concerns around plagiarism and license compliance. Classical fingerprint-based plagiarism detectors, such as Winnowing, remain highly effective, yet the inspection requires comparing fragments of code to the entire training set, and their linear-time search makes them impractical for the billion-scale corpora used to train modern code LLMs. To bridge this gap, we introduce SourceTracker, a 300M-parameter encoder tailored for code retrieval, together with a hybrid two-stage provenance-tracking pipeline HybridSourceTracker (HST). HST first narrows down a small set of candidate snippets via vector search, then re-ranks those candidates using Winnowing on exact fingerprints. We train and evaluate our system on a 10M-snippet subset of the TheStackV2 dataset, with both verbatim and adapted snippets that emulate realistic identifier renaming. On an in vitro 100k-snippet search space with adapted queries, our hybrid approach reaches a mean reciprocal rank on par with Winnowing for 30-token fragments. Then, starting from windows >= 60 tokens, it consistently overperforms by up to 5.4% while preserving logarithmic-time query complexity. In a complementary evaluation using an LLM-based judge, we find that many retrieved snippets not labeled as ground truth are still highly similar to the expected sources, particularly with longer context windows, and thus remain useful for end users. Overall, our results demonstrate that integrating vector search with fingerprinting enables scalable, high precision provenance tracking for code produced by LLMs, provided that their training data is made accessible...
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.07878v2">Rethinking Latency Denial-of-Service: Attacking the LLM Serving Framework, Not the Model</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 NeurIPS 2026
    </div>
    <details class="paper-abstract">
      LLM inference is inherently expensive, even a modest slowdown can translate into substantial operating costs and severe availability risks. Recently, a growing body of research known as latency attacks focuses on crafting inputs to trigger worst-case output lengths. However, we report a contrary finding that these algorithmic-level latency attacks are largely ineffective against modern LLM serving systems. We reveal that system-level optimization such as continuous batching provides a logical isolation to mitigate contagious latency impact on co-located users. Thus, in this paper, we shift our focus from the algorithm to the system layer, and introduce a new Fill and Squeeze attack strategy targeting the state transition of the scheduler. ``Fill'' first exhausts the global KV cache to induce Head-of-Line blocking, while ``Squeeze'' forces the system into repetitive preemption. By manipulating output lengths using different attack prompts, and leveraging side-channel probing of memory status, we demonstrate that the attack can succeed in a practical black-box setting with much less cost. Extensive evaluations on vLLM indicate up to $75-742\times$ TTFT degradation relative to benign baselines and $1.5-4\times$ average slowdown on Time Per Output Token compared to existing attacks with 30-40% lower attack cost. Code: https://github.com/Phil-Fan/FS-attack
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11345v1">SynCo: Data Synthesis Co-Training for Self-Evolving LLMs via Multi-Agent Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      Self-evolving LLM agents promise to improve autonomously through continual interaction and learning, reducing their dependence on manually curated supervision. Realizing this promise requires not only updating the agent, but also evolving its training experience as its capabilities change. However, most existing pipelines rely on static datasets or separately updated synthesis models, causing previously useful tasks to become trivial while overly difficult tasks remain uninformative. This growing mismatch between agent capability and training experience limits sustained self-improvement. To address this problem, we propose SynCo, an agentic data synthesis co-training framework for self-evolving LLMs based on multi-agent reinforcement learning. SynCo jointly optimizes two independently parameterized agents: a Synthesizer that constructs training tasks from the Reasoner's evolving capability state, and a Reasoner that learns from the resulting experience. Each synthesized task induces multiple Reasoner rollouts whose outcomes provide complementary rewards to both agents. Correctness feedback improves the Reasoner, while task quality, answer reliability, and outcome-grounded teachability guide the Synthesizer. Their updates are fed back into subsequent synthesis rounds, allowing the task-solving policy and its training distribution to evolve together. Extensive experiments across eight mathematical reasoning benchmarks demonstrate that SynCo substantially outperforms a broad range of existing synthetic-data methods and controlled baselines, achieving the strongest overall performance while deriving most of its gains from previously unsolved problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11334v1">ReCast: Attribution-Oriented Step Representation Learning for LLM-Based Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      In LLM-based agent systems, failures can originate from early steps whose effects propagate through subsequent interactions, making their origins difficult to identify. To trace such failures back to their origin, failure attribution has been formulated as the task of identifying the earliest step responsible for the failure. Recent methods leverage LLM internal signals for failure attribution, typically using hidden states as step representations. We therefore conduct an empirical study to evaluate how effectively these representations distinguish root-cause steps from other steps and find limited separation. Motivated by this observation, we propose ReCast, a step representation learning method that transforms hidden states from a frozen LLM into attribution-oriented step representations. ReCast first selects attribution-relevant layers, then constructs complementary pattern and deviation features, and finally learns contextualized step representations through an encoder trained with contrastive and ranking objectives. We also introduce ReCast-2K, a training dataset for failure attribution. ReCast achieves the best Hit@1 across four benchmarks, surpassing the strongest baseline by 5.65 and 9.19 pp on Who&When Algorithm and Handcrafted, respectively. Code is available at https://anonymous.4open.science/r/ReCast-5FB6 .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21231v3">SparKV: Overhead-Aware KV Cache Loading for Efficient On-Device LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 IoTJ
    </div>
    <details class="paper-abstract">
      Efficient inference for on-device Large Language Models (LLMs) remains challenging due to limited hardware resources and the high cost of the prefill stage, which processes the full input context to construct Key-Value (KV) caches. We present SparKV, an adaptive KV loading framework that combines cloud-based KV streaming with on-device computation. SparKV models the cost of individual KV chunks and decides whether each chunk should be streamed or computed locally, while overlapping the two execution paths to reduce latency. To handle fluctuations in wireless connectivity and edge resource availability, SparKV further refines offline-generated schedules at runtime to rebalance communication and computation costs. Experiments across diverse datasets, LLMs, and edge devices show that SparKV reduces Time-to-First-Token by 1.3$x-5.1x with negligible impact on response quality, while lowering per-request energy consumption by 1.5x to 3.3x, demonstrating its robustness and practicality for real-world on-device deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11310v1">FloorSAV: Elucidating Spatial Audio-Visual Context with 2D Floormap for AV-LLMs</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 Project page: https://byulharang.github.io/FloorSAV/
    </div>
    <details class="paper-abstract">
      While 3D spatial reasoning in dynamic egocentric environments is crucial for embodied intelligence, audio-visual large language models (AV-LLMs) lack explicit mechanisms to process and internalize global geometry directly from raw sensory streams. Existing approaches either require costly fine-tuning or underutilize the model's cross-modal reasoning capacities. In this paper, we propose FloorSAV, a novel framework that explicitly grounds spatial audio-visual context by rendering a dynamic 2D floormap. By integrating 3D point clouds, camera trajectories, spatial audio cues, and semantically grounded object landmarks, we inject this floormap into the AV-LLM as a synchronized stream with an egocentric video. AV-LLMs utilize their multi-modal capabilities to jointly reason over visual, auditory, and geometric cues in a single inference with floormap interpretation guidance. We further introduce SAVED-Bench (Spatial Audio-Visual Egocentric Benchmark with Dynamic Agents), constructing essential tasks of spatial capability in real-world scenarios: dynamic relativity, regional, and path reasoning QAs. FloorSAV improves AV-LLMs' spatial reasoning on various tasks from both SAVED-Bench and SAVVY-Bench. Studies with ground-truth floormaps demonstrate the substantial potential of FloorSAV with accurate spatial information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11300v1">Characterizing Overconfident Failure in LLM-Based Code Generation</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used for automated code generation, but generated programs can appear syntactically plausible while still failing execution-based correctness checks. Existing validation methods, such as testing and program analysis, remain essential but are often incomplete, costly, or applied only after generation. Model-derived uncertainty is therefore a natural early reliability signal. This paper studies the dilemma of overconfidence in code LLMs where incorrect programs are often generated with token-level confidence comparable to correct programs. We study this dilemma across four open-source code models and three execution-based benchmarks. Our analysis begins by investigating whether existing uncertainty metrics provide reliable proxies for execution correctness in code generation. We then characterize overconfidence at both global and local token levels, asking whether incorrect programs remain indistinguishable from correct ones under confidence and entropy summaries, including selective generation and the limits of instruction tuning. Finally, we evaluate whether common mitigation strategies reduce this failure mode. Our study yields four findings. First, existing uncertainty signals provide only partial and model-dependent evidence of execution failure. Second, overconfidence persists at both program and token levels, and uncertainty-based selection does not consistently improve accepted-set accuracy. Third, instruction tuning can increase certainty on failing generations without consistently improving correctness discrimination. Fourth, common mitigation techniques improve specific aspects of reliability but do not reliably resolve overconfident failure. Our exploratory latent analysis suggests that hidden representations may encode correctness-related signals that output confidence does not expose.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.23257v2">CTRL: Control-Based Time Series Forecasting with LLM-Guided Residual Learning</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 Published in Findings of the Association for Computational Linguistics: ACL 2026. 18 pages, 9 figures, 22 tables
    </div>
    <details class="paper-abstract">
      Time series forecasting underpins critical decision-making across diverse domains. While large language models (LLMs) offer promising reasoning capabilities, existing LLM-based time series forecasting approaches either reduce them to numerical predictors that bypass their strengths, or allow direct forecast generation that destabilizes predictions in non-stationary settings. We introduce CTRL, a framework that decouples semantic reasoning from quantitative prediction. A frozen backbone generates base forecasts, while specialized LLM agents function as controllers that analyze backbone prediction errors through decomposed trend, seasonal, and irregular components, grounding reasoning in interpretable temporal structure. Each agent outputs compact control signals that a lightweight residual decoder translates into forecast corrections. CTRL incorporates label-free test-time adaptation that detects distribution shift from input statistics alone and readapts control signals with only 3-24 LLM calls via caching. CTRL is explicitly designed to improve robustness under non-stationary temporal dynamics and distribution shift, while remaining competitive on highly stationary time series where adaptive correction provides limited additional benefit.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11284v1">DynaTE: Accelerating Diffusion LLMs via Dynamic Token Execution</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      Diffusion-based LLMs (dLLMs) have recently emerged as a promising alternative to autoregressive (AR) LLMs by enabling bidirectional parallel refinement, alleviating the sequential decoding bottleneck of AR generation. However, their parallel iterative refinement mismatches AR accelerators optimized for sequential decoding and their discrete token generation differs from DiT accelerators designed for continuous denoising. Recent dLLM accelerators have explored workload-specific optimizations to reduce vocabulary processing overhead and redundant computation across denoising iterations. However, these approaches retain all tokens in parallel execution, despite varying token refinement utility and execution requirements. This paper presents DynaTE, a hardware--software co-design architecture that dynamically adapts accelerator execution to evolving token states during dLLM decoding. DynaTE first enables adaptive token execution by skipping low-utility token computation, while a dimension-reconfigurable PE array maintains high utilization under varying active-token patterns. Second, DynaTE exploits dynamic token dependencies through FLDD to refine a small number of locally dependent tokens within the current iteration, reducing the overall number of denoising iterations, while a Merge--Split--Merge dataflow hides the resulting serial overhead. Third, a streaming vocabulary engine interleaves multiple token streams from the LM head to accommodate irregular output variations caused by selective token computation and uneven vocabulary-selection demands. Evaluated on two representative dLLMs, DynaTE achieves 2.05--2.78$\times$ speedup and 2.99--3.93$\times$ higher energy efficiency over state-of-the-art dLLM accelerators, while delivering 2.55$\times$ speedup and 6.07$\times$ higher energy efficiency over Jetson AGX Orin.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11260v1">LLM Agents as Resilience Engineers for Scientific Applications</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      Efficient checkpoint/restart support is essential for resilient HPC scientific applications, but implementing it requires substantial expertise: developers must identify recoverable state, choose globally consistent checkpoint points, and preserve application invariants during restart. We study whether frontier LLM coding agents can automate this process. We build a benchmark suite of 16 MPI applications spanning diverse domains, code sizes, and critical-state structures, and evaluate them with a no-human-in-the-loop generate--validate--revise pipeline for checkpoint/restart synthesis. Across the benchmark, the pipeline produces 41 working resilient implementations. Our results show that agent-driven resilience engineering is practical when critical state is visible or accessible through coherent abstractions: successful runs finish in under one hour on average, consume about 15M tokens, and produce implementations with negligible failure-free overhead and recovery efficiency comparable to human-written code. However, modularized and fragmented state remains a major limitation, with some failed attempts consuming over 100M tokens and 300 minutes without producing a working implementation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11253v1">LLM-IDEA: Identifiability-Driven Experimental Agent for Autonomous Discovery of Mechanistic World Models</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      Large language model agents are being increasingly deployed as autonomous scientists, designing experiments and inferring mechanistic world models with minimal human oversight. Yet identifiability is often overlooked: when a plateau is reached, the agent needs to know whether it is not yet capable enough or the model simply is not identifiable from the data, in which case no amount of further experimentation of the same kind can help. We propose the Identifiability-Driven Experimental Agent (LLM-IDEA) for closed-loop discovery with an identifiability engine that returns a three-way plateau verdict: capability limit, resolvable within the design class, or certified exhausted. On ODEBench, 60 of the 62 systems with free constants are identifiable at round 0; the RC circuit is certified exhausted for every experiment that protocol can run, and a harvesting model is resolvable by one added initial condition. The identifiability engine reproduces known verdicts on Lotka-Volterra, Van der Pol, Lorenz, and a pharmacokinetic model, where it recommends the intravenous arm pharmacologists use, and it ranks the depth scorer of our own benchmark last among four observation designs. On the DiscoverPhysics benchmark, it finds two public worlds whose explanation rubric rewards a distinction no legal experiment can make, and every model there with accurate trajectories failed the explanation grade (15 of 15, against 5 of 9 in identifiable worlds, p = 0.012). On the Alien Universe, a two-body testbed we propose in which a force law switches between a provably non-identifiable and an identifiable protocol, LLM-IDEA on the identifiable protocol reaches discovery depth at least three on 8/8 seeds versus 1/8 without it. An autonomous discovery agent can thus compute, rather than guess, whether a plateau calls for more search, a better experiment of the same kind, or a different kind of experiment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.30158v3">CPR for LLMs: Critical-Point Routing against Catastrophic Forgetting in Domain Adaptation</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      Supervised fine-tuning (SFT) is the de facto standard for adapting large language models (LLMs) to target domains, but it often degrades the model's general capabilities, a phenomenon known as catastrophic forgetting. Existing approaches typically modify the SFT loss to mitigate forgetting, but they inevitably operate along a domain-generality trade-off. In this work, we step outside this trade-off by decoupling the two capabilities at the model level: we keep the original base model for general capability, and selectively invoke the SFT expert only when domain-specific knowledge is required. Specifically, we propose CPR (Critical-Point Routing), a token-level routing framework between a base model and its expert derivative, based on critical tokens where the base model fails but the expert succeeds. We train a lightweight hierarchical router that estimates the expert-call probability per token, and pair it with a tailored inference procedure that combines momentum smoothing and threshold gating. Across diverse model-domain configurations, CPR achieves state-of-the-art across all settings, surpassing SFT expert by 1.4-5.5% in domain performance while recovering its general-capability drop from 3.4-14.5% to at most 0.5%, with minimal overhead from invoking the expert on only one-third of tokens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11188v1">What to Admit and How to Present: Governing Persistent Memory in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 Under review at AAMAS 2027
    </div>
    <details class="paper-abstract">
      Persistent memory can improve personalization in LLM agents but can also induce sycophancy and cross-domain leakage. We distinguish two governance decisions: admission, which determines what recalled information enters the working context, and presentation, which determines how admitted information is expressed. We implement two inference-time designs without retraining: factor-compiled admission (FC), which assesses whole memory entries, and permission-semantic admission (PS), which decomposes entries into typed units; both translate adjudicated attributes into eligibility decisions via deterministic policies. We evaluate on a four-backbone development suite and an external benchmark with four tasks of 300 samples each. Relative to verbatim injection, FC and PS reduce pooled judge-assessed failure rates on the external benchmark by 6.7 and 8.8 percentage points (p = 2.7e-7 and 4.1e-12), and development-set cross-domain leakage falls by up to 29.5 percentage points. A query-conditioned gating baseline shows no significant change in objective-fact failure or pooled failure. Under matched admission budgets, PS outperforms random and relevance-based selection on external objective-fact judgment after Holm correction. Holding presentation fixed, tightening admission cuts cross-domain failure by a further 17.5 percentage points (p = 1.6e-4); in contrast, no comparison between two renderings of identical adjudicated outputs survives multiple-comparison correction. Both designs increase personalization failures, and PS misses the preregistered improvement and personalization-preservation criteria. These results support evaluating admission and presentation separately: selection quality provides task-specific safety gains, while preserving beneficial memory use remains unresolved.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11160v1">LadderEdit: Edit-Level Residual Compression for Memory-Efficient Lifelong Editing of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 EMNLP 2026 Main Conference Long Paper
    </div>
    <details class="paper-abstract">
      Lifelong editing of LLMs requires storing thousands of edits after acquisition. A widely used family of approaches attaches one LoRA adapter per edit, which preserves behavior but grows linearly in storage. To address this challenge, we propose LadderEdit, a method that compresses each LoRA adapter after it is acquired. Each edit is first stored at low rank as a cheap sketch. We then check whether this sketch still satisfies the rewrite, generalization, and locality contract on probe prompts. Edits that pass keep the sketch; those that fail are promoted to a higher rank along a ladder until the contract is met. Because every edit retains some representation, coverage is maintained, and only hard edits consume more rank. Across ZsRE, CounterFact, and WikiBigEdit benchmarks on LLaMA-3-8B, Mistral-7B, and Qwen2.5-7B, LadderEdit tracks exact LoRA storage at 5.2x less memory and remains effective at 50,000 sequential edits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11159v1">Local Prototype Reconstruction for Text-Compatible Speech-to-LLM Bridge Pretraining</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      Speech-to-LLM systems often connect a frozen speech encoder to a frozen large language model (LLM) through a small trainable bridge. The bridge is usually treated as plumbing, but it in fact defines the geometry of the speech-to-LLM interface, and the pretraining objective decides whether that interface provides a reusable initialization for downstream tasks. We study a transferable bridge through two complementary properties: global alignment with the text side, and local lexical manifold compatibility, where bridge embeddings remain close to the frozen LLM's input-embedding neighbourhoods. We make this property measurable with a fixed, head-free, timestamp-free diagnostic that applies to any objective, and show that next-word prediction (NWP) and sentence-level contrastive pretraining do not fully capture token-level lexical compatibility. We then introduce Local Prototype Reconstruction (LPR), a lightweight training-only regularizer that requires each aligned bridge token to be reconstructable from a small neighbourhood of frozen LLM token embeddings, with a hard single-prototype anchor as its limiting case. On multilingual ASR and speech translation, LPR improves transfer, with the largest gains on translation and low-resource adaptation. Crucially, our independent diagnostic correlates with downstream gains across objectives, suggesting that lexical manifold compatibility is predictive of reusability for speech-to-LLM bridges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11152v1">Do LLMs Learn from Rewards in Context? : Rethinking the role of reward in In-Context Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 NeurIPS 2026 Spotlight (Negative Results Track)
    </div>
    <details class="paper-abstract">
      LLM agents increasingly improve at inference time by accumulating experience in context rather than by updating parameters. This process is often described as in-context reinforcement learning (ICRL). Whether in-context learning (ICL) can actually play the role of RL, however, has not been tested. We study this question in its simplest form, direct ICRL, where the model conditions directly on raw trajectory-reward pairs, and ask whether the reward acts as a learning signal. Through controlled experiments on four benchmarks across six models, we find that the reward is read, but its effect is small: flipping, randomizing, or removing the reward leaves the improvement curve almost unchanged, and this holds even under meta-prompts that explicitly instruct the model to explore, exploit, or reason over rewards. Trajectories drive improvement, but not through their semantic content: shuffled or corrupted trajectories work as well as real ones. These patterns closely mirror those known in ICL, suggesting that direct ICRL is better understood as a special case of ICL than as inference-time RL. This reframing has implications for agent memory design: ICL factors such as input distribution and demonstrations may matter more than RL elements such as reward shaping and exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11135v1">Can a System-One LLM Perform Knowledge Tracing When Few or No Learners Are Logged?</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 41 pages, 7 figures. Code and result summaries: https://anonymous.4open.science/r/jevkt-coldstart
    </div>
    <details class="paper-abstract">
      Knowledge tracing (KT) models need many logged learners, so a new course or platform starts without a usable model. In LLM-based KT the LLM generates the answer, which we call System-Two; it is either fine-tuned on the target data or reasons and votes over ten samples, which is slow and gives coarse probabilities. We ask whether an off-the-shelf System-One LLM, which returns a probability for a typed question directly in a single pass, can perform KT when few or no learners are logged. On seven datasets, Jev without any data from the target platform reaches a mean AUC of .706, above the best of 28 deep KT models trained on 8 learners (.689) and above System-Two Thinking-KT on all seven datasets (.650) at about 1/100 of its API cost. Adding examples and a similar-learner statistic from the logged learners (JevKT) raises this to .722; JevKT stays significantly ahead of deep KT up to 16 learners and ahead on average up to 64, and supervised KT catches up between 64 and 128 learners. Among the readers we tested, the gain is specific to Jev, since three other LLMs queried with the byte-identical typed request through the official System-One adapter fall below it on all seven datasets, and reader swaps and contamination checks find no evidence that the input format or memorised data explain the gain. For new learners the advantage holds from their first interactions, whereas on unseen items with all learners logged, deep KT remains ahead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.24282v2">LLM-powered Query Expansion for Enhancing Boundary Prediction in Language-driven Action Localization</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 Accepted by International Journal of Computer Vision (IJCV)
    </div>
    <details class="paper-abstract">
      Language-driven action localization in videos requires not only semantic alignment between language query and video segment, but also prediction of action boundaries. However, the language query primarily describes the main content of an action and usually lacks specific details of action start and end boundaries, which increases the subjectivity of manual boundary annotation and leads to boundary uncertainty in training data. In this paper, on one hand, we propose to expand the original query by generating textual descriptions of the action start and end boundaries through LLMs, which can provide more detailed boundary cues for localization and thus reduce the impact of boundary uncertainty. On the other hand, to enhance the tolerance to boundary uncertainty during training, we propose to model probability scores of action boundaries by calculating the semantic similarities between frames and the expanded query as well as the temporal distances between frames and the annotated boundary frames. They can provide more consistent boundary supervision, thus improving the stability of training. Our method is model-agnostic and can be seamlessly and easily integrated into any existing models of language-driven action localization in an off-the-shelf manner. Experimental results on several datasets demonstrate the effectiveness of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11116v1">BayesJudge: Uncertainty-Aware Bayesian Meta-Evaluation of Human and LLM Judgments</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 Accepted at NeurIPS 2026
    </div>
    <details class="paper-abstract">
      AI evaluation pipelines often produce conflicting judgments rather than clean labels. In pairwise LLM evaluation, this conflict is especially visible: disagreement can arise from ambiguous items, underspecified rubrics, heterogeneous or unstable human raters, or an LLM judge whose verdict changes when the response order is swapped. We propose BayesJudge, an online Bayesian meta-evaluation layer for conflicting human-LLM judgment streams. For each comparison, BayesJudge estimates a panel-relative posterior verdict distribution over the two responses, with a tie or ambiguity state when such labels are available. At the same time, it estimates rater-specific human confusion matrices and LLM presentation-order bias. The method uses tie-open labels to keep ambiguity observable and paired order-swapped judge calls to separate response quality from presentation effects. We formulate the exact online posterior recursion and use a scalable Rao-Blackwellized assumed-density SMC approximation for streaming inference. Controlled synthetic experiments demonstrate recovery of prespecified evaluator parameters and illustrate two protocol-level identifiability mechanisms: tie-open labels expose ambiguity mass, and order-swapped paired judgments separate item preference from position bias. On a real-world SummEval dataset, BayesJudge successfully detects systematic presentation-order effects in LLM judge outputs, infers distinct expert and crowdworker behavior signatures without rater metadata, and produces posterior uncertainty estimates that correlate with human disagreement. Our code is available at https://anonymous.4open.science/r/BayesJudge-0879.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11087v1">Beyond Imitation: A Framework and Benchmark for LLM-Assisted Peer Review</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      The rapid growth of scientific publishing has strained peer review, particularly in machine learning, raising concerns about declining review quality and increasing reviewer workload. Large language models (LLMs) have been proposed as automated review assistants, yet their evaluation has focused largely on imitating human-written reviews rather than supporting the core functions of peer review. Here, we introduce a verification-centric perspective on LLM-assisted peer review, emphasizing error detection as a critical and resource-intensive task. We present a scalable benchmark that evaluates review systems' ability to identify logical contradictions, constructed through synthetic insertion of errors into conference papers, yielding unambiguous evaluation targets and enabling systematic comparison. We further propose a Multi-Layered Review (MLR) framework that prioritizes detailed manuscript comprehension before review generation, aligning more closely with human reviewing practices while improving token efficiency. Across evaluations, our approach demonstrates strong alignment with human review scores, achieves high error detection performance, and provides complementary perspectives on reviewer focus. These improvements can be attributed to both the choice of the underlying LLM and the design of our system. At the same time, we corroborate persistent vulnerabilities to adversarial manipulation, underscoring the need for robustness in automated review systems. Our findings highlight the importance of rigorous, error-focused evaluation to guide responsible deployment of LLM-based tools in peer review and other critical scientific workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.09609v2">What Personal Information Improves LLM-Based Next-Location Prediction?</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 20 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used for individual next-location prediction, with personal information easily added to prompts alongside mobility history. Yet the incremental predictive value of such information remains unclear. Using linked sociodemographic records and mobility traces from 5,000 Shenzhen residents, this study separates model responsiveness from predictive value. GPT-5 is the primary model, with GPT-5.5 and Claude Opus 4.6 used for replication. In 1,000 paired prediction instances, models rank 100 candidate destinations with and without age, gender, occupation and income while all other inputs are held fixed. Behavioural history raises top-1 accuracy from 5.6% to 18.5% as prior history increases from zero to six days. By contrast, sociodemographic attributes produce no detectable overall gain, although replacing correct attributes with those of another person reduces accuracy by 5.4 percentage points. Candidate construction also matters, removing distance raises accuracy by 7.7 points under proximity sampling but lowers it by 22.3 points under popularity sampling, with the reversal reproduced across all three LLMs. These findings identify behavioural history as the clearest source of incremental value and show that personal information should be evaluated under matched, explicitly specified conditions before its privacy and governance costs are justified.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11076v1">Stability-Plasticity Balance via Singular-Vector Selection in LLM Continual Learning</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      Domain-specific continual adaptation of LLMs risks catastrophic forgetting, creating a fundamental tension between acquiring new capabilities and preserving those learned during pretraining. PEFT mitigates this problem by restricting the number of trainable parameters, but existing methods lack a principled unit for deciding where plasticity should be allocated and stability should be preserved. We identify the singular-vector channel as a natural unit for managing this trade-off. Each channel represents an input-output transformation, which can be updated to acquire new knowledge or fixed to preserve pretrained capabilities. Based on this perspective, we introduce SVC, a parameter-efficient continual-learning method that selectively updates Singular-Vector Channels. Before fine-tuning, SVC uses domain-specific data to estimate each channel's adaptation benefit and a fixed public general-domain corpus only as a history activation proxy for estimating forgetting cost. It then adaptively selects trainable channels based on these scores via knee-based cost screening, Pareto-front filtering, and Otsu thresholding. Experimental results across four LLM families and eight downstream tasks show that SVC better preserves pretrained capabilities while achieving strong downstream performance relative to existing PEFT baselines. Further analysis of channel scoring and selection demonstrates that selective plasticity at the singular-vector-channel level enables effective continual LLM adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11030v1">NOMOS: Compiling Written Policies into Statically Verified Tool-Call Gates for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 28 pages, 5 figures, 19 tables. Submitted to IEEE Access
    </div>
    <details class="paper-abstract">
      Tool-using LLM agents violate the policies they are deployed to enforce, often silently. Prior defenses hand-write rules, query an LLM verifier per action, or compile policies through heavyweight formal machinery. Naive compilation fails: extracted rules block the tool satisfying their own precondition, or read arguments their tool lacks. NOMOS, a four-pass compiler, turns a natural-language policy into a deterministic tool-call gate; static verification with tool-schema-level checks alone (no prover, solver, or LLM) repairs or rejects 37% (airline) and 13% (retail) of candidates, without which most shipped rules are inoperable. Replaying compiled rules over undefended transcripts flags bindings that refuse legitimate work (a development binding refused 95.9% of task-passing calls); no evaluation binding is flagged. On $τ^2$-bench the gate cuts violations of reference-encoded clauses among state-changing calls from 66.3% to 2.6% (airline) and 30.8% to 6.9% (retail), raising airline task success significantly for $2 \le k \le 4$; a 26B on-premise compilation is not significantly worse than hand-written or frontier-compiled rules. Unlike AgentDojo's shipped defenses, it reaches a zero attack success rate (ASR) on banking, where nine attack families collapse onto three structural rules. On the other three suites its ASR is at most 3.6%, from goals with no tool call to govern and one write admitted by a binding weaker than its clause; a second agent model, Llama-3.3-70B, reproduces the effect on both benchmarks. Decisions take microseconds without an LLM call, at a domain-dependent benign-utility cost; compilation runs on-premise on open-weight gemma-4-26B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.01376v2">3BASiL: An Algorithmic Framework for Sparse plus Low-Rank Compression of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 The Thirty-ninth Annual Conference on Neural Information Processing Systems
    </div>
    <details class="paper-abstract">
      Sparse plus Low-Rank $(\mathbf{S} + \mathbf{LR})$ decomposition of Large Language Models (LLMs) has emerged as a promising direction in model compression, aiming to decompose pre-trained model weights into a sum of sparse and low-rank matrices $(\mathbf{W} \approx \mathbf{S} + \mathbf{LR})$. Despite recent progress, existing methods often suffer from substantial performance degradation compared to dense models. In this work, we introduce 3BASiL-TM, an efficient one-shot post-training method for $(\mathbf{S} + \mathbf{LR})$ decomposition of LLMs that addresses this gap. Our approach first introduces a novel 3-Block Alternating Direction Method of Multipliers (ADMM) method, termed 3BASiL, to minimize the layer-wise reconstruction error with convergence guarantees. We then design an efficient transformer-matching (TM) refinement step that jointly optimizes the sparse and low-rank components across transformer layers. This step minimizes a novel memory-efficient loss that aligns outputs at the transformer level. Notably, the TM procedure is universal as it can enhance any $(\mathbf{S} + \mathbf{LR})$ decomposition, including pure sparsity. Our numerical experiments show that 3BASiL-TM reduces the WikiText2 perplexity gap relative to dense LLaMA-8B model by over 30% under a (2:4 Sparse + 64 LR) configuration, compared to prior methods. Moreover, our method achieves over 2.5x faster compression runtime on an A100 GPU compared to SOTA $(\mathbf{S} + \mathbf{LR})$ method. Our code is available at https://github.com/mazumder-lab/3BASiL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.22698v2">Black-Box Forensics for Conversational LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      As LLM-powered scams proliferate, black-box forensics for conversational LLM agents offers a path to accountability for systems hidden behind anonymous endpoints. Identifying the base model behind a chatbot endpoint (attribution), without model parameter access or knowledge of the hidden system prompt, would let investigators trace AI-enabled scams back to the providers whose models power them. Detecting when two endpoints run the exact same system prompt (fingerprinting), even one novel and unseen, would link individual scams into criminal networks and expose silent API changes. We conduct an empirical investigation of both capabilities. Our attribution classifiers identify the base model behind an agent with 98% accuracy from a few turns of non-adversarial conversation. Attribution of system prompts, while possible, requires retraining on a large amount of data for each prompt; system prompts in the wild are unbounded and ever-changing, making this approach costly. To tackle this more open-ended setting, our cross-encoder fingerprinting method achieves an AUC of 0.768 and an F1 of 0.703 on entirely unseen system prompts, and aggregating 50 interaction conversations from each target agent boosts AUC to 0.943. Conversational agents with unseen system prompts can thus be fingerprinted with robust accuracy from a few turns of ordinary conversation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11007v1">Curating Always-Loaded Context for LLM Agents: A Capacitated Assortment Model with Censored Feedback</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      At the start of every session, LLM agents load a fixed context file, such as $\texttt{AGENTS.md}$. Each loaded token in the file is charged again in every later round of the session, and these files can degrade performance as they grow in size. However, in practice, human or automated curators usually grow these files by appending. We formulate context curation as a capacitated assortment problem. Instructions consume tokens under a finite attention capacity; adding an instruction never raises the compliance of the others, while retained instructions incur a per-session setup cost. We prove an upper bound on the optimal file size, regardless of the number of available candidate instructions, and that appending every instruction with positive standalone value can be arbitrarily worse in net value than selecting an optimal subset. A token budget also limits the loss when the token price is underestimated. We then examine what can be learned from past sessions and how this information can guide decisions to add or remove instructions. Feedback is inherently censored: the benefits and harms of loaded instructions are observable, whereas missing instructions generate feedback only when their absence causes harm. In this setting, we show that deleting instructions ignored by agents can inevitably remove helpful ones. We characterize how much evidence should be collected before adding an instruction. Besides, we bound regret when human reviewers can inspect only a limited number of edits per period. Empirical experiments further show that irrelevant rules drawn from real context files reduce language-model compliance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08993v2">Arbiter: Detecting Interference in LLM Agent System Prompts</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      System prompts for LLM-based coding agents are software artifacts that govern agent behavior, yet lack the testing infrastructure applied to conventional software. We present Arbiter, a framework combining formal evaluation rules with multi-model LLM scouring to detect interference patterns in system prompts. Applied to three major coding agent system prompts: Claude Code (Anthropic), Codex CLI (OpenAI), and Gemini CLI (Google), we identify 152 findings across the undirected scouring phase and 21 hand-labeled interference patterns in directed analysis of one vendor. We show that prompt architecture (monolithic, flat, modular) strongly correlates with observed failure class but not with severity, and that multi-model evaluation discovers categorically different vulnerability classes than single-model analysis. One scourer finding was structural data loss in Gemini CLI's memory system was consistent with an issue filed and patched by Google, which addressed the symptom without addressing the schema-level root cause identified by the scourer. Total cost of cross-vendor analysis: \$0.27 USD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.05613v2">LLM Benchmarking via Representation Multi-task Learning</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      Quantifying and evaluating the capabilities of Large Language Models (LLMs) remains a fundamental challenge in modern data science and artificial intelligence. In this paper, we consider LLM evaluation based on their performance across items in multiple benchmark domains (e.g., mathematical reasoning and coding) within a leaderboard framework. Our goal is to address two core questions: (1) How do we derive more accurate domain-specific scores by borrowing information across domains? and (2) How do we define and estimate an overall score that aggregates performance across multiple domains? To solve these problems, we propose a novel statistical framework based on representation Multi-task Learning (MTL) and an item response theory model. Specifically, we define overall and domain-specific LLM traits through an Item Response Theory (IRT) model, and propose an MTL approach to estimate these traits from item-level response data. We develop a computationally efficient estimator and establish its minimax optimality under certain asymptotic regimes. This framework provides a rigorous measurement foundation for systematic LLM evaluation. We conduct extensive simulations, demonstrating the superior performance of the proposed method over competing methods. Crucially for the Applications and Case Studies section, we apply the proposed framework to MMLU response data from the Hugging Face Open LLM Leaderboard, covering 4,272 LLMs and 13,232 items across 56 subjects. The empirical analysis reveals substantial heterogeneity in domain size and difficulty, together with strong positive cross-domain dependence, highlighting the practical value and substantive insights generated by our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.08833v4">An Investigation of Robustness of LLMs in Mathematical Reasoning: Benchmarking with Mathematically-Equivalent Transformation of Advanced Mathematical Problems</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 34 pages, 9 figures, accepted at ICLR 2026 workshop
    </div>
    <details class="paper-abstract">
      Frontier large language models (LLMs) now reach near-ceiling accuracy on standard mathematical-reasoning benchmarks and gold-medal-level performance at the International Mathematical Olympiad. As these benchmarks saturate and their items leak into training data, a high score no longer shows whether a model reasons robustly or which component of its reasoning fails. To evaluate reasoning while keeping results informative and failures diagnosable, we propose GAP (Generalisation-and-Perturbation), a methodology that automatically generates mathematically equivalent variants of existing mathematics problems at scale using two disjoint, interpretable transformations: (1) surface renames, probing the binding between identifiers and latent variable roles, and (2) kernel rewrites, probing whether a high-level proof plan survives a change of mathematical setting. Compared with existing benchmarks, GAP has two key benefits: (1) novel, likely unseen variants mitigate data leakage, and (2) performance across transformation families enables failure diagnosis, each transformation testing a hypothesis about the cause of failure. We instantiate GAP on all 1,051 William Lowell Putnam Competition problems from 1938 to 2024, adding 5,255 unseen variants to form PutnamGAP, a 6,306-item competition-level mathematics corpus and the first public machine-readable dataset from the full Putnam archive. Using PutnamGAP, we evaluated 18 commercial and open-source models spanning sizes and providers. Accuracy drops across all models and variant families, most severely under kernel rewrites. This gap does not close with model strength, suggesting that even the strongest models' dominant weakness is transferring a proof plan to a changed mathematical setting, rather than handling surface changes. Further analysis provides finer failure diagnoses and potentially useful insights for improving LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.10906v1">Reading the Room: Foundations, Design, and Challenges of Normative Competence in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      Human communities are governed by normative systems: shared standards that produce \textit{norms} dictating acceptable behavior, enforced through community sanctioning. Aligning increasingly autonomous AI systems with these norms is a central alignment challenge, complicated by the fact that norms are vast in number, change quickly, and are often arbitrary (e.g., dress or language conventions). Thus, alignment requires \textit{normative competence}: the ability to discern from interaction alone what norms a community enforces without relying on static pretrained knowledge. We introduce a multi-agent community debate setting, where access to debate is governed by synthetic norms, to study normative competence in isolation from pretraining exposure. We show that baseline LLM agents fail to learn norms even when doing so would improve their accuracy. We then experiment with various \textit{normative modules} -- architectural components for norm inference -- finding that norm-following is highly sensitive to both the style of norm and the model powering the normative module, suggesting a lack of generalizability. Furthermore, when idiosyncratic, non-normative behaviors accompany the true norm, LLM agents exhibit an unselective attribution failure: they indiscriminately copy idiosyncratic noise alongside enforced rules, a pattern that persists even when imitating unnecessary behaviors is explicitly penalized. To the best of our knowledge, our work is the first to operationalize and evaluate normative competence in LLMs, demonstrating that current AI systems excel at behavioral mimicry but lack the capacity to discern socially enforced order.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.10848v1">Amortized Off-Policy Evaluation for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      Accurate evaluation is central to selecting which LLM to deploy, yet testing a candidate on live traffic exposes real users to an unvetted model. Teams therefore evaluate candidates offline, on data produced by already-deployed models. This is off-policy evaluation (OPE), and it faces two distribution shifts: as a model is updated in post-training, its responses diverge from the logged ones (policy shift), and the reward definition under which it is judged changes with business requirements (reward shift). Classical OPE methods are ill-suited to this continual-deployment setting because they are defined per task and require fitting from scratch on every new logged dataset or reward definition. To address this, we propose PFN-OPE, a prior-data fitted network that amortizes OPE across a distribution of contextual-bandit tasks. We pretrain it once on tasks constructed from a pool of LLM responses scored by several reward functions, in which both shifts occur. At test-time it maps a logged dataset and one sampled target response per prompt to a value estimate in a single forward pass, with no per-task fitting. On HelpSteer2 and UltraFeedback with Qwen, Llama, and Gemma policies, PFN-OPE achieves 2.0 to 9.3 times lower error than the best baselines across all tested configurations in the reward-shifted settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.07510v2">Understanding and Enhancing Backdoor Persistency in LLM Agent Post-Training</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 EMNLP 2026 Findings
    </div>
    <details class="paper-abstract">
      Developers can build LLM agents by adapting third-party models through benign post-training. We study a supply-chain threat in which an attacker supplies a model with a backdoor: hidden behavior that produces malicious outputs when a particular input pattern appears. Focusing on software-engineering agents, we ask whether such backdoors survive the developer's supervised fine-tuning (SFT) and subsequent task-level reinforcement learning (RL). We observe that benign SFT substantially reduces attack success, but subsequent RL often preserves the residual behavior and sometimes even increases attack success. Our analysis of backdoor erosion during SFT identifies two factors that may favor survival: initial backdoor strength and gradient compatibility with benign training. These factors motivate PersistBD, which refines an already-backdoored model before release to improve its persistency through the benign post-training process. On Qwen2.5-Coder-7B, PersistBD raises attack success from 20% to 74% after SFT and from 20% to 76% after SFT-RL, while maintaining comparable benign task performance. Together, our results show that backdoors can remain active through benign post-training and that adversaries can deliberately increase their persistence. This highlights a supply-chain risk for AI developers and motivates stronger techniques for detecting and mitigating inherited backdoors when adapting third-party models into agents. Our code is available at https://github.com/uiuc-kang-lab/PersistBD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.10749v1">Listen-to-Reason: Listen with Experts, Retrieve over a Graph, Reason with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      Large audio-language models (LALMs) fuse an audio encoder into a large language model (LLM) through multi-stage training. This coupling means that a new domain or a stronger LLM requires retraining, and their answers cannot be traced to what the model heard: a chain-of-thought is a post-hoc account. We propose Listen-to-Reason (L2R), an interpretable-by-design pipeline that passes audio to the LLM through an explicit, human-readable tree: small heads on frozen expert encoders map each chunk of a clip to semantically meaningful nodes on the tree (for speech, music and environmental sound), and a frozen text-only LLM answers from these nodes and an ASR transcript without hearing the clip. Every answer can therefore be traced to the nodes and transcript it read, and the nodes are causal: replacing the deciding node with a distractor overturns 78% of correct answers on SAKURA. With a 7B reader, L2R outperforms all LALMs we compare against on SAKURA and trails them by 6-12 points on MMAU and MMAR, despite training about 1,400x fewer parameters on orders of magnitude less audio data. However, because any LLM can serve as the reader, we show that a stronger reader narrows this gap without retraining any audio component. A new domain is added with one small head: with five labelled clips per species, it outperforms QLoRA fine-tuning of an LALM on the same clips by 13-26 points.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.10533v1">EngramEdit: Decoupled Knowledge Updates in LLMs through Conditional Memory</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      Conditional memory architectures such as DeepSeek Engram use input n-grams to look up learned embeddings, expanding the capacity of large language models (LLMs) with limited additional computation. Beyond model scaling, this architecture has demonstrated the potential to decouple factual knowledge storage from general-purpose computation, offering a promising route to updating factual knowledge while keeping the Transformer backbone fixed. Realizing this potential is challenging because different expressions of a fact may activate different n-gram embeddings, while updating shared embeddings can unintentionally change the model's predictions about other facts. We propose EngramEdit for decoupled knowledge updates through conditional memory. EngramEdit first computes target memory representations that make the model predict the updated fact across multiple expressions. It then jointly updates the shared n-gram embeddings to match these targets across expressions and edits, penalizing updates to frequently reused embeddings more strongly to preserve unrelated knowledge. Experiments show that EngramEdit enables independent factual knowledge updates through conditional memory, achieving near-perfect editing success. Revised knowledge is usable across unseen expressions and in multi-hop reasoning, with nearly three times the strongest baseline's accuracy under chain-of-thought (CoT) prompting. Unrelated knowledge and general capabilities are largely preserved even as factual updates accumulate. These findings show that EngramEdit turns conditional memory into an editable knowledge interface, extending its role beyond model scaling to support decoupled knowledge updates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.14767v2">Loop-Back Authority in LLM Agent Teams: A Paired Experiment on Flat and Hierarchical Coordination</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 8 pages, 3 figures, 3 tables, plus 21 pages of supplementary material. Code: https://github.com/cihatburak/loop-back-authority-llm-agents
    </div>
    <details class="paper-abstract">
      Does authority in AI teams improve the outcome? Organizational theory asserts that authority facilitates decision making, improving quality. Meanwhile, some nascent AI research suggests that revision under authority makes LLM output worse. Multi-agent LLM frameworks default to giving a Manager agent the authority to send a worker's output back for revision. Prior comparisons test the effect of authority using verifiable tasks. We conduct an experiment on an open-ended task, business-intelligence reporting, using a sample of 43 paired laptop products and 86 runs. Each report is written once by a hierarchical team and once by a flat team. We find that flat teams produce higher-quality reports, scoring higher on Utility (d = 0.42, p = 0.009) and Writing Clarity (d = 0.34, p = 0.030). The reports are the same length, but hierarchical team reports use 53% more hedging words such as "may" and "could", and each revision is associated with a 0.14-point drop in Writing Clarity on a 1 to 5 scale. Before any revision, the hierarchical team's first draft is indistinguishable from the flat team's report. In other words, the quality gap can be traced to revision. Authority improves quality when the Manager can verify the work, else when it can only provide feedback it has a negative effect on quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01353v4">The Trojan Knowledge: Bypassing Commercial LLM Guardrails via Harmless Prompt Weaving and Adaptive Tree Search</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 Accepted at ICML 2026. Project website: https://everywheresafety.github.io/cka-agent/
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) remain vulnerable to jailbreak attacks that bypass safety guardrails to elicit harmful outputs. Existing approaches overwhelmingly operate within the prompt-optimization paradigm: whether through traditional algorithmic search or recent agent-based workflows, the resulting prompts typically retain malicious semantic signals that modern guardrails are primed to detect. In contrast, we identify a deeper, largely overlooked vulnerability stemming from the highly interconnected nature of an LLM's internal knowledge. This structure allows harmful objectives to be realized by weaving together sequences of benign sub-queries, each of which individually evades detection. To exploit this loophole, we introduce the Correlated Knowledge Attack Agent (CKA-Agent), a dynamic framework that reframes jailbreaking as an adaptive, tree-structured exploration of the target model's knowledge base. The CKA-Agent issues locally innocuous queries, uses model responses to guide exploration across multiple paths, and ultimately assembles the aggregated information to achieve the original harmful objective. Evaluated across state-of-the-art commercial LLMs (Gemini2.5-Flash/Pro, GPT-oss-120B, Claude-Haiku-4.5), CKA-Agent consistently achieves over 95% success rates even against strong guardrails, underscoring the severity of this vulnerability and the urgent need for defenses against such knowledge-decomposition attacks. Our codes are available at https://github.com/Graph-COM/CKA-Agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.10455v1">PHRBench: A Behavioral Evaluation of Post-Hallucination Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      Hallucinated information can propagate through multi-stage LLM systems and become part of the context for subsequent reasoning. Existing studies of post-hallucination reasoning (PHR) mainly characterize changes in final outcomes and aggregate reasoning dynamics, leaving how models resolve hallucinated premises at the response level insufficiently understood. In this work, we introduce PHRBench, a controlled benchmark for behaviorally structured PHR across four domains and 18 large language models. PHRBench characterizes each reasoning trajectory independently of final-answer correctness through Hallucination Compliance, Hallucination Avoidance, and Heuristic Correction, and defines an insightful trajectory as successful correction that ultimately reaches the correct answer. Across 4820 controlled instances, we find that successful recovery remains relatively rare and is associated with more frequent belief updates along the reasoning trajectory. We further find that properties of the hallucinated prompt contain substantial predictive signal for successful recovery, with a lightweight predictor achieving an AUROC of 0.847. These findings provide a behavioral view of post-hallucination reasoning, characterizing how LLMs resolve erroneous context and when successful recovery is likely to occur.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.17575v3">A Dual-Hypothesis Reasoning Framework for LLM Guardrails</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      We propose ARBITER, a novel LLM guardrail framework that introduces two key ideas: (i) dual-hypothesis reasoning, a reasoning method for LLM guardrails that explicitly considers both safe and unsafe interpretations of a prompt before making a safety decision, and (ii) multi-component supervised fine-tuning (MC-SFT), a structured training loss for reasoning-based guardrails that decomposes LLM outputs into logical components and weights them according to their importance. Existing reasoning-based guardrails often rely on expensive procedures, such as generating reasoning traces using larger or closed-source teacher models and applying full-parameter fine-tuning. In contrast, ARBITER uses a cost-effective self-generation strategy for reasoning traces and LoRA-based parameter-efficient fine-tuning while still achieving better performance than these expensive approaches. Additionally, ARBITER provides faithful evidence-phrase explanations for unsafe decisions, enabling a more transparent and interpretable guardrail method. Experiments on three safety moderation benchmarks show that ARBITER outperforms existing reasoning-based and non-reasoning guardrail baselines, with clear gains in out-of-domain evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.21445v2">When Trivia Is Not Trivial: Everyday Knowledge Failures in Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 EMNLP 2026 Findings
    </div>
    <details class="paper-abstract">
      Quiz rooms, trivia nights, and quiz shows challenge human knowledge across a wide range of topics, from canonical facts to everyday culture. In this paper, we examine whether large language models (LLMs) can perform competitively in such settings, using quiz-style questions to test them on both common and niche topics. We introduce TriviaRoomQA, a multilingual benchmark designed to evaluate everyday, culturally grounded, and long-tail knowledge across 288 topics. The benchmark contains 3,300 parallel multiple-choice questions in six European languages and additional 5,340 French-only questions for a more fine-grained case study. We evaluate 30 open-weight LLMs from European, Asian, and North American providers, covering models from 7 to 70B parameters. We find that models are strong on knowledge-intensive topics such as history, geography, and mathematics, but substantially weaker on everyday popular-culture topics such as celebrities, music, movies, and news. Moreover, model performance varies across languages even for the same underlying questions, suggesting that access to factual knowledge is not always language-independent. In sum, our dataset and experiments demonstrate an important knowledge gap which is not captured by existing academic-based saturated benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22823v2">Which Way Did It Move? Diagnosing and Overcoming Directional Motion Blindness in Video-LLMs</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 NeurIPS 2026 (Accept). 50 pages including Appendix. Project page: https://jong980812.github.io/which-way-did-it-move/
    </div>
    <details class="paper-abstract">
      Video Large Language Models (Video-LLMs) have made rapid progress on temporal video understanding, yet many fail at a basic perceptual primitive: signed image-plane motion direction. On simple videos of a single object moving left, right, up, or down, most Video-LLMs perform near chance, with above-chance cases largely attributable to prediction biases rather than genuine direction understanding. We call this failure directional motion blindness. We localize the failure by tracing motion direction information through the Video-LLM pipeline. Motion direction remains linearly accessible from the vision encoder, projector, and LLM hidden states, but the readout fails to bind this signal to the correct verbal answer option, revealing a direction binding gap. Although synthetic motion direction instruction tuning reduces this gap on the source domain, motion direction concept vector analysis shows that visual complexity weakens the signal magnitude and limits out-of-domain generalization. We introduce MoDirect, a dataset family for motion direction instruction tuning and evaluation, and DeltaDirect, a diagnosis-driven, projector-level objective that predicts normalized 2-D motion vectors from adjacent-frame feature deltas. On MoDirect-SynBench, instruction tuning with DeltaDirect improves motion direction accuracy from 25.9% to 85.9%. On MODIRECT-REALBENCH, DeltaDirect improves realworld motion direction accuracy by 21.4 points over the vanilla baseline without real-world tuning data, while preserving standard video-understanding performance. Our project page is available at https://jong980812.github.io/which-way-did-it-move/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.06966v2">APEX: Active Protection at Execution Boundaries for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      Indirect prompt injection (IPI) hides adversarial instructions in content that large language model (LLM) agents read at runtime. As agents compose heterogeneous capability units, including Tools, MCP servers, and Skills, the carriers of injection multiply, and defenses built to recognize attack patterns fall behind them. We instead shift defense from covering attack patterns to one stable point: whatever the carrier and however the injection propagates, harm materializes only at the \emph{execution boundary}, where the agent turns internal state into an external action or released output. Safety there turns on two conditions, both settled by the trusted task rather than by the run: whether the proposed effect is authorized, and whether the runtime information reaching it is endorsed by that task. We present APEX, an active defense that enforces both at this boundary from a single authorization contract compiled before untrusted execution: \emph{evidence-gated prevention} admits an effect only when the contract justifies it, while \emph{deception-based exposure} makes unendorsed use reveal itself before the effect commits. Protection therefore follows from what the task permits rather than from how an attack is built, and applies uniformly across capability units without attack-specific policies or taint tracking. Against 13 baselines, APEX attains 0\% attack success on five of six benchmarks and 0.56\% on the sixth, holds 0\% under adaptive attacks on all three capability-unit types, and remains effective across defender backbones. Code is available at https://github.com/ZhengXR930/APEX_official/tree/official.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.10656v1">Can LLMs Simulate Novice Programmers' Misconceptions?</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      We evaluate 13 LLMs on generating, solving, simulating, and diagnosing novice programming misconceptions on code-tracing problems. While frontier models reliably perform these tasks, small models ($\le$14B) struggle with dynamic execution. Notably, code-tuned models fail during misconception simulation by reverting to correct execution. Finally, misconception diagnosis is significantly easier in True/False formats than in open-ended generation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.10655v1">Nullify: Null-Space Activation Steering for Training-Free LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) inevitably internalize substantial amounts of sensitive or private information during pre-training, while LLM unlearning aims to selectively erase specific knowledge to prevent privacy leakage with minimal loss of model utility. However, existing methods struggle to balance forget quality with utility, and typically incur substantial computational costs due to parameter fine-tuning. To address this, we propose Nullify, a training-free, non-destructive activation steering method for LLM unlearning. Nullify employs steering vectors during inference to redirect privacy-related activations away from their memorized answers, while satisfying a null-space constraint that leaves retained-query activations essentially unaffected to maintain utility. Evaluations on TOFU and MUSE show that Nullify matches or surpasses established baselines in forget quality while achieving near-lossless preservation of model utility. By avoiding weight updates entirely, Nullify serves as an efficient, plug-and-play inference-time intervention framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.10318v1">Nobody Truly Agrees on Sentiment: Humans, Bespoke Tools, and LLMs Struggle with Social Media Texts</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 11 pages, 1 figure, 2 tables, accepted for publication at TPDL 2026
    </div>
    <details class="paper-abstract">
      Social media is a rich source of real-time public sentiment, but widely used sentiment analysis tools are often applied without understanding their limitations. In this study, we evaluate the inter-rater reliability of three bespoke sentiment analysis tools (TextBlob, VADER, and Twitter-roBERTa-base) and three large language models (LLMs: Qwen3-32B, GPT-OSS-120B, Llama-4-Maverick-17B) against six human raters across 100 tweets. We measured agreement using two statistical measures: Cohen's kappa for pairwise comparisons and Fleiss' kappa for multiple raters. Even among the human raters, our results showed only fair agreement, highlighting the subjectivity of sentiment analysis. Higher agreement was observed under the binary sentiment classification (negative vs. non-negative and positive vs. non-positive) than under the three-class classification across both humans and automated tools. The Twitter-roBERTa-base model showed the strongest alignment with human ratings, outperforming both bespoke sentiment tools and LLMs, particularly in distinguishing negative versus non-negative sentiment. LLMs showed substantial agreement among themselves and moderate to substantial alignment with humans, performing better in positive vs. non-positive classifications. Our findings underscore that domain-specific fine-tuning remains crucial for reliable social media sentiment analysis, and human-centered evaluation remains essential for establishing gold-standard labels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.10235v1">Beyond LLM-GA: Secure Fluid Antenna Systems with ReEvo-Designed Memetic Algorithm</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 Accepted by WCSP 2026
    </div>
    <details class="paper-abstract">
      Fluid antenna systems (FASs) offer significant spatial flexibility, yet securing them against eavesdropping is critical for practical FAS deployment in military, satellite, and internet-of-things networks. Although large language model (LLM)-assisted genetic algorithms (LLM-GAs) can address this secure FAS port selection problem, whether further algorithmic improvement is possible warrants deeper investigation. To this end, we propose a memetic algorithm based on reflective evolution (ReEvo). Unlike the state-of-the-art LLM-GAs, which design only crossover or mutation operators with an LLM, our algorithm leverages an LLM to evolve dedicated crossover, mutation, and local-search operators offline. These operators are then embedded into a memetic search framework, thereby obviating any online LLM queries during execution. Simulation results at equal generation counts demonstrate that our proposed algorithm achieves a higher secure sum-rate than the conventional GA and the state-of-the-art LLM-GAs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.10232v1">LLM Persuasion Is in the Eye of the Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have already been shown to match or exceed human experts in persuasion. While their persuasive capabilities hold promise for beneficial uses such as education and health communication, they can also be used to manipulate and misinform, making their evaluation a growing priority for developers and regulators. That evaluation, however, remains fragmented: studies differ in what they treat as persuasion, and broad claims often rest on narrow, situation-specific assessments. Automated methods, often modelled on human studies, offer a way to compare such assessments directly, as they can be run on the same models at scale and can include high-risk forms of persuasion that would be difficult or unethical to test on people. In this study, we adapt nine published automated methods to a shared setup, run them on the same fifteen LLMs, and ask whether their rankings agree and why. We find that the methods agree only weakly (mean Spearman $ρ= 0.25$). Our analyses point to two contributing factors. Models that refuse some tasks but not others, directly or indirectly, lower agreement by about a quarter, and these refusals fall mostly on manipulation tasks. General capability also plays a part: most rational persuasion (non-manipulative) methods track it, whereas most manipulation methods do not. Together, these findings suggest that agreement depends more on the task a method sets than on how it scores persuasion, although this pattern is only indicative given the eight methods available for analysis. More broadly, our results suggest that persuasion scores combine a model's ability to persuade with its willingness to do so. A single score is therefore informative about its own setting, but says little about a model's persuasiveness across tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.10227v1">From Prompts to Trees: Effective LLM-Guided Tree Generation for Few-Shot Tabular Classification</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 Accepted to EMNLP 2026 Main as an oral presentation. Code available: https://github.com/yueqiu0/LLMTree
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) possess rich world knowledge and impressive generalization capabilities, their direct application to tabular data classification is hindered by high inference costs and limited interpretability. In contrast, decision trees are fast and transparent but often underperform in low-data regimes. In this work, we propose a novel framework that bridges these paradigms by distilling LLM knowledge into interpretable decision trees under a few-shot learning setting. Instead of directly prompting the LLM to generate full trees, which is often unstable and inefficient, we develop a three-stage paradigm that prompts the LLM to generate rules and organize the rules into a tree. Experiments on multiple real-world tabular datasets demonstrate that our method achieves superior accuracy and interpretability with significantly lower prompting overhead compared to existing baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.14057v3">Measuring the Creativity of Frontier LLMs in Automated Research</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 After discussion with our advisor, we concluded that the manuscript requires substantial further improvement before being made publicly available. As these revisions are expected to take a considerable amount of time, we would like to withdraw the current version for now and resubmit a more complete and robust version once the work has been substantially improved
    </div>
    <details class="paper-abstract">
      Frontier LLMs are increasingly capable of conducting automated research, yet their creativity in this setting has not been systematically evaluated. We propose a set of metrics to evaluate creativity along the two dimensions of valueness and novelty. Valueness assesses whether each proposed idea is useful, while novelty is evaluated from three perspectives: whether the same idea has appeared before (Exact-Match P-Novelty), whether the modified variable or variable combination has been explored before (Variable-level P-Novelty), which reflects the breadth of research-space exploration, and whether the proposed idea is explicitly attributed to external knowledge in the model's reasoning (H-Novelty). Our evaluation shows that the models achieve relatively similar Valueness and Exact-Match P-Novelty scores, while differing substantially in Variable-level P-Novelty. H-Novelty is also consistently high among the models for which it can be evaluated. Notably, further correlation and idea-level performance analyses reveal a strong positive correlation between Variable-level P-Novelty and research performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.22327v3">Geometry-Aware Online Scheduling for LLM Serving: From Theoretical Bound to System Practice</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      The rapid growth of interactive Large Language Model serving has made efficient management of dynamic Key-Value cache footprints increasingly important for inference performance. Modern inference systems overwhelmingly rely on time-centric scheduling heuristics, such as Shortest Job First. However, their classical guarantees are rooted in traditional scheduling modeling, failing to capture the highly dynamic, 2D spatio-temporal geometric growth specific to LLM inference mechanisms. To resolve this, we propose the geometry-aware online scheduling by introducing the Smallest Volume First (SVF) algorithm and its highly efficient variant, 1-bit SVF. Via a novel volume-certificate proof, we establish a worst-case approximation ratio for SVF that approaches \textbf{3} in the high-concurrency regime of LLM serving, substantially improving upon the prior best constant of 48. Building upon this core breakthrough, we complete a comprehensive theoretical taxonomy analyzing our algorithms across different traffic scenarios and information availability. Practically, we seamlessly integrate our approach as a plug-and-play layer in vLLM. Extensive evaluations on Llama-3.1 models demonstrate comprehensive performance gains: SVF delivers strong reductions in both average and tail latency, while 1-bit SVF, with merely a single bit of information, achieves competitive throughput and latency. This work establishes a theoretically sound and empirically proven approach for resolving memory-constrained scheduling in modern LLM deployments. To facilitate future research, our code is available at https://github.com/Li-Beverly-Kong/Geometry-Aware-Online-Scheduling.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.10150v1">On the Reliability of LLM-Based Vulnerability Patching Benchmarks</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown strong potential for automated vulnerability patching, but current benchmarks can substantially distort reported performance. Drawing on extensive experience developing, running, and stress-testing such frameworks, we identify under-examined pitfalls across three dimensions: (1) agent-level factors, where prompting, tool availability, and detailed instructions can raise success rates without improving developer-aligned patch quality; (2) framework-level factors, where permission errors, infrastructure bugs, and timeout handling can silently suppress or inflate performance; and (3) dataset-level factors, where bug reports and single proof-of-concept (PoC) tests fail to capture whether patches address root causes or follow developer intent. We curate 112 historical bugs from 84 open-source C/C++, Go, and Rust projects, each with PoC tests, regression tests, and additional developer tests that assess alignment with the original developers' design principles. Through controlled experiments and case studies, we show that LLMs can achieve high PoC passing rates under ideal conditions, yet benchmark execution choices can materially change measured success. More importantly, developer-test passing rates remain low and improve only marginally with newer models, suggesting that models increasingly suppress symptoms without consistently producing upstream-quality fixes. These results show that benchmark scores are highly sensitive to evaluation design, and we provide practical guidelines for more rigorous, reliable, and reproducible evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.10142v1">When Can You Ship on Evals Alone? Trial-Level Surrogacy for Offline Evaluation of LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      Offline evals increasingly guide changes to LLM-powered products, but showing that an eval agrees with human judgment does not tell a team which changes it can ship without an A/B test. That decision depends on whether eval treatment effects predict online treatment effects across a class of interventions. We call this requirement trial-level surrogacy and show that it is logically independent of unit-level (Prentice) surrogacy. In historical launch logs, sampling noise in both effect estimates attenuates the observed relationship. Under a bivariate normal working model, subtracting the reported within-intervention variances recovers the between-intervention covariance, and the corrected moments give a closed-form ship/test/kill rule that bounds the probability of shipping a change with a non-positive online effect. LLM judges add a separate distortion. When the judge is equally accurate on both arms, eval effects are only rescaled, but arm-dependent accuracy can reverse their sign. Once deployed, the eval-to-outcome relationship can no longer be observed in the ship and kill regions, so drift there becomes undetectable without randomized audit experiments. On 183 interventions from the Upworthy Research Archive, the pooled eval-to-outcome relationship is weak despite precise measurement on both sides, and the illustrative gate sends every intervention to an A/B test.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.10126v1">Know the Shape, Find the Fault: Topology-Conditioned Diagnosis of Multi-Agent LLM Failures</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      Multi-agent LLM systems coordinate task execution through exchanges of information among agents. When coordination breaks down, similar symptoms in execution traces can reflect different problems in how information is passed, used, or verified. Communication topology captures how agents exchange information and provides structural cues for distinguishing coordination failure modes. Using these cues for diagnosis requires establishing how topology relates to failure patterns and recovering the relevant structure from execution traces that lack explicit topology labels. We analyze the relationship between communication topology and failure patterns and introduce MAScope, a two-stage framework for topology-conditioned diagnosis. Its Trace Structural Extractor TSE recovers communication topology from heterogeneous execution traces by grounding an interaction graph in message evidence. The Topology-Conditioned Judge TC-Judge then classifies failures using the trace, predicted topology, an empirical failure prior estimated from separate labeled traces, and a short description of topology-specific failure patterns. Under a fixed orchestration structure, the recovered topology can be reused across executions. Experimental results show a statistically significant association between communication topology and failure type, with $χ^2 = 409.9$ and $p = 1.2 \times 10^{-70}$. On the \num{851} MAST-clean traces, ground-truth topology context raises gpt-mini's Macro-F1 from $0.173$ to $0.350$. With predicted topology, the pipeline achieves $0.346$, approaching the trace-only gpt-5.4 baseline of $0.372$. For \num{1000} traces under a fixed orchestration structure, the projected pipeline cost, including one topology extraction, is approximately $6\%$ of repeated gpt-5.4 diagnosis cost. These results show that topology-conditioned context improves failure diagnosis and supports lower-cost deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.25727v2">Auditing Privacy Risks in LLM-Enhanced Graph Neural Networks</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently advanced graph neural networks (GNNs) by enriching node representations with semantic information, giving rise to LLM-enhanced GNNs that achieve substantial performance gains. However, how such semantic enhancement affects privacy risks remains largely underexplored. To bridge this gap, we systematically audit the privacy risks of LLM-enhanced GNNs through a unified framework consisting of five stages: (1) dataset preparation, (2) victim model training, (3) privacy attack, (4) risk assessment, and (5) defense analysis. Specifically, our evaluation spans ten text-attributed graph datasets across diverse domains, six privacy attacks, 42 LLM-enhanced GNN configurations, and three more recent language-model backbones. Extensive experiments show that, despite their utility improvements, LLM-enhanced GNNs consistently exhibit greater empirical privacy vulnerability than shallow text representation baselines under the evaluated attacks across diverse models and datasets. Further analysis shows that LLM-enhanced representations exhibit more distinguishable link-, label-, and membership-related signals in the embedding space, making them more exploitable by inference attacks. Finally, we evaluate representative defenses and examine their effectiveness in mitigating these privacy risks. Overall, this work provides a systematic audit of privacy risks in LLM-enhanced GNNs and offers insights for developing more secure and trustworthy graph learning systems.
    </details>
</div>
