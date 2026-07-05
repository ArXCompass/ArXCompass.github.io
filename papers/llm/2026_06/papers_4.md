# llm - 2026_06

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- Part 4
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.19423v2">The Autonomy Tax: Defense Training Breaks LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-18
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents increasingly rely on external tools (file operations, API calls, database transactions) to autonomously complete complex multi-step tasks. Practitioners deploy defense-trained models to protect against prompt injection attacks that manipulate agent behavior through malicious observations or retrieved content. We reveal a fundamental \textbf{capability-alignment paradox}: defense training designed to improve safety systematically destroys agent competence while failing to prevent sophisticated attacks. Evaluating defended models against undefended baselines across 97 agent tasks and 1,000 adversarial prompts, we uncover three systematic biases unique to multi-step agents. \textbf{Agent incompetence bias} manifests as immediate tool execution breakdown, with models refusing or generating invalid actions on benign tasks before observing any external content. \textbf{Cascade amplification bias} causes early failures to propagate through retry loops, pushing defended models to timeout on 99\% of tasks compared to 13\% for baselines. \textbf{Trigger bias} leads to paradoxical security degradation where defended models perform worse than undefended baselines while straightforward attacks bypass defenses at high rates. Root cause analysis reveals these biases stem from shortcut learning: models overfit to surface attack patterns rather than semantic threat understanding, evidenced by extreme variance in defense effectiveness across attack categories. Our findings demonstrate that current defense paradigms optimize for single-turn refusal benchmarks while rendering multi-step agents fundamentally unreliable, necessitating new approaches that preserve tool execution competence under adversarial conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19826v1">Heterogeneous LLM Debate Under Adversarial Peers: Honest Gains, Replacement Costs, and Resilience</a></div>
    <div class="paper-meta">
      📅 2026-06-18
    </div>
    <details class="paper-abstract">
      Heterogeneous LLM debate is motivated by the promise that diverse peers correct one another, but the same exchange that carries correction also carries adversarial influence. We measure which dominates by tracking how a heterogeneous peer changes the honest agents' revision behavior: how often they change their answer, and whether the change is corrective or harmful. We compare matched panels (homogeneous baseline, honest-mixed, and adversarial-mixed) and contaminated panels in which a malicious same-family peer is already present, spanning four model families and three reasoning benchmarks. An honest heterogeneous peer sharply lowers harmful revision, and an adversarial one reverses it. For Llama-3.1-70B defenders on MATH-hard, the honest-slot harmful-revision rate falls from 89% in the homogeneous panel to 35% with an honest peer, and an adversarial peer returns it to 90%. The conditional rate hides this damage on weak defenders, but the end-of-debate flip rate exposes it. The pattern keeps its sign across families and benchmarks while its magnitude varies with the defender-benchmark regime. We also measure the effects when an adversarial same-family peer is already present: an honest heterogeneous peer lowers both harmful revision and the rate at which initially-correct answers are lost. On the same Llama-3.1-70B setting, the added honest peer cuts the flip rate on initially-correct items from 31% under a same-family adversary to 6%. Heterogeneity is therefore not only an attack surface but, when an adversary is already present, also a defense.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19771v1">Beyond Entropy: Learning from Token-Level Distributional Deviations for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-18
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has significantly advanced Large Language Model (LLM) reasoning; however, it faces a fundamental optimization instability: uniform token updates precipitate entropy collapse, leading to premature convergence to suboptimal strategies, whereas excessive Shannon Entropy maximization can cause entropy explosion, driving blind exploration toward incoherent reasoning chains. To resolve this dichotomy, we introduce the Independent Combinatorial Tokens (ICT) framework, which shifts the optimization focus from scalar uncertainty to the distributional properties of token logits. By leveraging the Jensen-Shannon (JS) divergence between token logits distributions, ICT identifies tokens with distinctive distributional patterns as critical branching points for guiding effective exploration in LLM reasoning. Our theoretical analysis, grounded in both Shannon and second-order Rényi entropy, proves that selectively updating on these tokens regulates policy concentration: it reduces the overall distribution uncertainty measured by Shannon entropy, while controlling probability concentration captured by second-order Rényi entropy. This dual effect prevents over-concentrated token generation from weakening exploration and effectively stabilizes the training landscape. Empirical results demonstrate that updating only the top 10% of unique tokens on Qwen2.5 (0.5B/1.5B/7B) models yields an average pass@4 improvement of 4.58%, with a maximum gain of 14.9%, over GRPO, 20-Entropy, and STAPO baselines across seven benchmarks spanning math, commonsense, and Olympiad-level problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19755v1">SafeSpec: Fast and Safe LLM via Dynamic Reflective Sampling</a></div>
    <div class="paper-meta">
      📅 2026-06-18
    </div>
    <details class="paper-abstract">
      Speculative inference accelerates large language model (LLM) decoding but provides no inherent safety guarantees. Existing safety defenses are largely incompatible with speculative inference: they either introduce additional computation or disrupt the draft-verify mechanism, negating acceleration benefits. This reveals a fundamental incompatibility between current safety methods and speculative decoding. We propose SafeSpec, a safety-aware speculative inference framework that integrates risk estimation directly into the verification process. SafeSpec attaches a lightweight latent safety head to the target model to jointly evaluate semantic validity and safety in a single forward pass. When unsafe generations are detected, SafeSpec applies rollback and safety-guided reflective multi-sampling to recover safe continuations rather than terminating generation. We model jailbreak attacks as distributional shifts over generative trajectories, where adversarial prompts increase the probability of harmful continuations without eliminating safe ones. Under this model, SafeSpec performs risk-aware trajectory recovery within the speculative decoding process. Across multiple models and adversarial benchmarks, SafeSpec achieves a substantially improved safety-efficiency trade-off. On Qwen3-32B, SafeSpec reduces attack success rates by 15% while preserving a 2.06x inference speedup on benign workloads, demonstrating that speculative acceleration and inference-time safety can be jointly optimized.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19746v1">SAC: Disaggregated KV Cache System for Sparse Attention LLMs with CXL</a></div>
    <div class="paper-meta">
      📅 2026-06-18
    </div>
    <details class="paper-abstract">
      The scaling of LLMs toward long-context inference has shifted the primary serving system bottleneck from computation to memory capacity. Traditional solutions for dense attention models rely on RDMA-based disaggregated memory pools, which perform coarse-grained fetching of the entire prefix KV cache from remote storage to local memory before decoding. However, this approach is fundamentally inefficient for emerging sparse attention models. While only a small fraction of KV entries are active during decoding, these systems still fetch the full KV cache locally, leading to severe transmission bottlenecks and local memory wastage. To address this, we propose SAC, the first efficient disaggregated KV cache system optimized for sparse attention models. By leveraging the low-latency, cache-line granularity load/store semantics of Compute Express Link (CXL), SAC fetches only the required top-k KV entries on demand during inference. Evaluations on DeepSeek-V3.2 using SGLang show that SAC achieves 2.1x higher throughput, 9.7x lower TTFT, and 1.8x lower TBT compared to RDMA-based baselines, establishing CXL-based disaggregation as the superior infrastructure for emerging sparse attention models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19714v1">AURA: Adaptive Uncertainty-aware Refinement for LLM-as-a-Judge Auditing</a></div>
    <div class="paper-meta">
      📅 2026-06-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as judges for open-ended generation, as large-scale human evaluation is often expensive and difficult to scale, yet their preferences remain imperfect proxies for human judgment. Existing auditing pipelines often assume that a reliable subset of examples or clean supervision signals are available beforehand, for example from human annotation, heuristic filtering, or the outputs of strong judges. In LLM evaluation, this assumption is fragile: the initial split may inherit judge bias, while human verification is typically too scarce to define stable groups at scale. We propose AURA, an adaptive uncertainty--aware refinement framework for auditing pairwise LLM--as--a--judge decisions under selected human verification. AURA iteratively learns a human-consistency signal, propagates reliable evidence, and prioritizes uncertain comparisons for human review. The key idea is to treat trust in a judge as a latent quantity that is progressively refined as evidence accumulates. We provide a compact formulation, a stable refinement procedure, and a comprehensive evaluation on both synthetic and real pairwise LLM-answer data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.01338v2">Benchmarking Local LLMs for Natural-Language-to-SQL Querying in Biopharmaceutical Manufacturing: An Empirical Benchmark on Consumer-Grade Hardware</a></div>
    <div class="paper-meta">
      📅 2026-06-18
    </div>
    <details class="paper-abstract">
      Biopharmaceutical manufacturing organizations operate under regulatory frameworks such as FDA guidance, EU Good Manufacturing Practice (GMP), and the EU AI Act, which can restrict the use of cloud-based artificial intelligence systems. Locally deployed large language models (LLMs) offer a privacy-preserving alternative, but their suitability for pharmaceutical manufacturing tasks remains underexplored. This study evaluates four open-source LLMs (Qwen 2.5 Coder 7B, Llama 3.1 8B, Mistral 7B, and Meditron 7B) deployed locally via Ollama for natural-language-to-SQL generation over a pharmaceutical manufacturing database. A FastAPI-based evaluation platform, PharmaBatchDB AI, was developed using a synthetic Microsoft SQL Server database containing approximately 63,000 records across Batch, Manufacturing Execution System (MES), and Clean-In-Place (CIP) modules. Models were benchmarked on 60 domain-specific natural-language questions using metrics including SQL extraction rate, SQL compliance, factual consistency, ROUGE-L, hallucination rate, throughput, and latency. Qwen 2.5 Coder 7B, Llama 3.1 8B, and Mistral 7B generated SQL for all evaluation tasks, while Meditron 7B failed on nearly all tasks due to context-window limitations and poor SQL generation capability. Llama 3.1 8B achieved the highest SQL compliance, whereas Qwen 2.5 Coder 7B achieved the strongest overall text similarity and factual consistency. Performance differences between the two leading models were not statistically significant. The results show that code-tuned general-purpose LLMs outperform a domain-specific biomedical model on structured query generation for pharmaceutical manufacturing data. Although fully local, GxP-aligned NLQ systems are feasible on consumer hardware, current performance levels still require human oversight and downstream validation for regulated use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17128v3">Shift-Left High-Level Synthesis Verification via Knowledge-Augmented LLM Agent</a></div>
    <div class="paper-meta">
      📅 2026-06-18
    </div>
    <details class="paper-abstract">
      High-Level Synthesis (HLS) relies on transforming original C specifications into synthesizable HLS-oriented C (HLS-C) implementations. Functional consistency verification between original C specifications and HLS-C implementations is a critical yet labor-intensive task in HLS design flows. While Large Language Models (LLMs) have recently shown promise in automated testbench generation, their stochastic nature often leads to insufficient coverage, inconsistent verification environments, and unreliable equivalence checking results. To address these limitations, we propose a knowledge-augmented, agent-driven shift-left verification framework for automated functional consistency checking between original C and HLS-C implementations before synthesis. The framework introduces a Dual-Tier Consistency Checking mechanism that jointly enforces static structural alignment and dynamic behavioral equivalence between paired testbenches, while integrating symbolic execution and coverage-driven refinement to improve verification completeness. Furthermore, we construct a heterogeneous HLS Verification Knowledge Graph to provide topology-aware reasoning priors for testbench generation, and design an autonomous verification agent to orchestrate iterative refinement and failure diagnosis across heterogeneous toolchains. Experimental results on 107 HLS benchmark pairs demonstrate that the proposed framework achieves 0.9826 average coverage and 0.9533 dynamic consistency, outperforming representative AST-based, retrieval-augmented, and iterative agent-based baselines. https://github.com/cz-5f/HLS-LeVeri.git
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03090v2">"**Important** You should give me full credits!": Exploring Prompt Injection Attacks on LLM-Based Automatic Grading Systems</a></div>
    <div class="paper-meta">
      📅 2026-06-18
      | 💬 15 pages, 8 figures, 9 tables
    </div>
    <details class="paper-abstract">
      The emergence of large language models (LLMs) has significantly accelerated recent research on LLM-based automatic grading (AG) systems. Benefiting from the strong instruction-following capabilities and broad prior knowledge of LLMs, educators can deploy AG systems across diverse tasks using only natural language rubrics while achieving satisfactory grading performance. Despite these advantages, new security concerns may also arise. In particular, prompt injection (PI) attacks have recently become a major threat to LLM-based applications. In the context of AG, attackers can potentially exploit PI vulnerabilities to manipulate grading systems into assigning artificially high scores regardless of the actual answer quality. Such behavior poses serious risks to the fairness, reliability, and integrity of educational assessment. In this work, we study PI attacks in AG systems, and systematically investigate the effectiveness of such attacks in educational scenarios. We further evaluate the effectiveness of existing defensive strategies against these attacks. Through comprehensive experiments under rubric-based grading settings, we demonstrate that current LLM-based AG systems remain highly vulnerable to PI attacks. We hope that our findings raise awareness of this emerging threat and motivate future research toward secure, robust, and trustworthy LLM-based educational systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18650v1">BLADE: Scalable Bi-level Adaptive Data Selection for LLM Training</a></div>
    <div class="paper-meta">
      📅 2026-06-17
    </div>
    <details class="paper-abstract">
      As Large Language Model (LLM) datasets scale to trillions of tokens, data selection has emerged as a critical frontier to filter out uninformative noise and construct adaptive learning trajectories. Beyond static heuristic filtering, advanced data selection methods for LLM training largely follow two paradigms, each with fundamental limitations. Influence-based methods provide principled bi-level objectives but require intractable inverse-Hessian computations, while excess-loss methods are computationally efficient but rely on a static reference model that becomes misaligned with the evolving proxy model during training. We propose BLADE (Bi-Level Adaptive Data sElection), a Hessian-free framework for data selection. BLADE reformulates the bi-level optimization problem underlying influence-based methods as a penalized single-level objective via Lagrange multipliers, avoiding inverse-Hessian computation while revealing a principled connection to excess-loss based data selection. The resulting objective recovers an excess-loss form but replaces the static reference model with a dynamic one that stays synchronized with training. Theoretically, we prove that this penalized formulation guarantees first-order convergence. For efficient online batch selection, we instantiate BLADE as a memoryless randomized block-coordinate Frank-Wolfe algorithm. Extensive experiments show that BLADE consistently outperforms state-of-the-art data selection baselines, providing a practical recipe for LLM training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18649v1">Gender Bias in LLM Hiring Decisions: Evidence from a Japanese Context and Evaluation of Mitigation Strategies</a></div>
    <div class="paper-meta">
      📅 2026-06-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in hiring workflows, yet most research on gender bias in LLM hiring decisions has focused on English-language, Western-format resumes. This study examines whether pro-female gender bias extends to a Japanese corporate context and evaluates two practical mitigation strategies. Using a counterfactual resume design with 60 Japanese rirekisho-format resumes, 12 name pairs selected on linguistically grounded gender-signal criteria, and five state-of-the-art LLMs (Claude Sonnet 4.6, GPT-4o, DeepSeek-V3, Gemini 2.5 Flash, Llama 3.3 70B), we conducted 43,200 API calls across baseline, prompt instruction, and privacy filter conditions. A crossed random-effects linear mixed model confirms a significant pro-female bias across all five models, replicating Western findings in a non-Western context. A prompt-level gender-neutrality instruction produces no meaningful reduction in bias. A name-reliance analysis formally identifies the candidate name as the primary gender channel: removing the name from the prompt reduces the female effect by nearly its full magnitude. An unexpected incompatibility between the privacy filter and GPT-4o's content safety filter, resulting in a 42% refusal rate, highlights a practical deployment challenge for name anonymization in LLM-assisted recruitment pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.15557v4">MORTAR: Multi-turn Metamorphic Testing for LLM-based Dialogue Systems</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 Accepted for publication in IEEE Transactions on Software Engineering (TSE)
    </div>
    <details class="paper-abstract">
      With the widespread application of LLM-based dialogue systems in daily life, quality assurance has become more important than ever. Recent research has successfully introduced methods to identify unexpected behaviour in single-turn testing scenarios. However, multi-turn interaction is the common real-world usage of dialogue systems, yet testing methods for such interactions remain underexplored. This is largely due to the oracle problem in multi-turn testing, which continues to pose a significant challenge for dialogue system developers and researchers. In this paper, we propose MORTAR, a metamorphic multi-turn dialogue testing approach, which mitigates the test oracle problem in testing LLM-based dialogue systems. MORTAR formalises the multi-turn testing for dialogue systems, and automates the generation of question-answer dialogue test cases with multiple dialogue-level perturbations and metamorphic relations (MRs). The automated MR matching mechanism allows MORTAR more flexibility and efficiency in metamorphic testing. The proposed approach is fully automated without reliance on LLM judges. In testing six popular LLM-based dialogue systems, MORTAR reaches significantly better effectiveness with over 150\% more bugs revealed per test case when compared to the single-turn metamorphic testing baseline. Regarding the quality of bugs, MORTAR reveals higher-quality bugs in terms of diversity, precision and uniqueness. MORTAR is expected to inspire more multi-turn testing approaches, and assist developers in evaluating the dialogue system performance more comprehensively with constrained test resources and budget.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.00802v2">GrowthHacker: Automated Off-Policy Evaluation Optimization Using Code-Modifying LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 Accepted for publication in ACM Transactions on Software Engineering and Methodology (TOSEM), 2026
    </div>
    <details class="paper-abstract">
      With data-driven development now widely adopted, online A/B testing is an established method for measuring the effects of new technologies. However, deploying online experiments demands resources for design, implementation, and deployment, and may negatively impact users (e.g., unsafe or unethical outcomes) while requiring weeks of data collection. To address this, the growing research area of off-policy evaluation (OPE), or offline A/B testing, assesses new technologies offline using previously collected logged data. OPE is also a fundamental problem in reinforcement learning and is important where online testing is expensive or risky, such as healthcare, recommender systems, education, and robotics. Despite advances in code-generation large language models (LLMs) and agentic workflows, little is known about whether and how LLMs and LLM-based agents can automatically optimize OPE implementations. We propose GrowthHacker, a benchmark that evaluates baseline LLMs and LLM-based agents on large-scale public datasets. GrowthHacker autonomously and iteratively modifies code, runs OPE, and uses the metrics to guide subsequent optimization. We evaluate methods on Open Bandit Pipeline (OBP) and Scope-RL, and develop a two_agent framework that addresses limitations of existing frameworks while reducing complexity. Across both libraries, two_agent shows the highest reliability (98.1%-100% success rate) and positive-outcome rate (78%), with a median improvement of 4.4% among positive outcomes; CrewAI achieves the highest average improvement (37.9%) and is the only framework with zero extreme-value failures. AutoGen and Default each reach 65% positive-outcome rates. These results establish the feasibility of using LLM-based agents as automated "growth hackers" to continuously improve OPE systems, with implications for scaling data-driven decision-making where manual optimization is expensive.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18042v2">Latency Prediction for LLM Inference on NPU Systems</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 12 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Deploying Large Language Models (LLMs) requires exploring a large configuration space spanning parallelization strategies, batching techniques, and scheduling policies. Exhaustive measurement across this space is impractical, making latency prediction essential for system optimization. While NPUs have emerged as accelerators designed for LLM inference, no prediction methodology has been established for them. Specifically, applying prior work to LLM inference latency prediction on NPUs faces three challenges: undisclosed microarchitecture of commercial NPUs, unpredictable compiler optimizations, and latency non-linearity induced by bucketing. We present LENS, a latency estimator that predicts NPU inference latency without information on the microarchitecture or compiler, and captures the non-linear latency induced by bucketing. LENS profiles each bucket with two end-to-end (E2E) measurements and composes the results to predict latency for arbitrary input-output length combinations. We validate LENS across NPUs from multiple vendors, several LLMs, and diverse workloads, achieving a mean prediction error of 2.15\%. We further compare LENS against two methodologically related baselines, confirming the validity of its approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18600v1">ShuntServe: Cost-Efficient LLM Serving on Heterogeneous Spot GPU Clusters</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 18 pages, 16 figures, 5 tables
    </div>
    <details class="paper-abstract">
      As large language model (LLM) services become widely adopted, the cost of GPU resources for serving these models in cloud environments has emerged as a critical concern. Spot instances offer up to 90% cost savings over on-demand instances, but their frequent interruptions and limited availability pose significant challenges for continuous LLM serving. GPU spot instances, in particular, exhibit lower and more volatile availability than CPU-based instances, making homogeneous clusters that depend on a single GPU type vulnerable to correlated failures. Heterogeneous clusters spanning multiple GPU types can address this by leveraging complementary availability patterns across diverse spot pools, yet existing LLM serving systems are designed for homogeneous environments and suffer from load imbalance when deployed on heterogeneous GPUs. This paper presents ShuntServe, a cost-efficient LLM serving system for heterogeneous spot GPU clusters. ShuntServe employs a roofline model-based analytical serving performance estimator and a dynamic programming-based model placement optimizer that jointly determines node configuration, parallelization strategy, and layer assignment to maximize throughput across heterogeneous GPUs. To enhance fault tolerance when using spot instances, ShuntServe combines output-preserving request migration with concurrent initialization via a shared tensor store, minimizing migration downtime by overlapping replacement node preparation with ongoing serving. Evaluation on Llama-3.1-70B and Qwen3-32B with a heterogeneous AWS cluster of L4, A10G, and L40S GPUs shows that ShuntServe achieves 1.42x and 1.35x higher throughput than state-of-the-art baselines and attains 31.9% and 31.2% cost efficiency improvements over on-demand instances for offline and online serving, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18596v1">Better Adherence, Richer Context: A Field Evaluation of LLM-Powered Conversational Voice Diaries for Sleep</a></div>
    <div class="paper-meta">
      📅 2026-06-17
    </div>
    <details class="paper-abstract">
      Sleep diaries are central to behavioral sleep medicine and cognitive behavioral therapy for insomnia, yet daily completion is difficult to sustain, and static forms often provide limited context for interpreting night-to-night sleep variation. We designed an LLM-powered conversational voice diary that delivers clinically grounded morning and evening sleep diary questions through proactive smart-speaker prompts, structured conversational intake, and adaptive follow-up dialogue. We evaluated the system in a four-week between-subjects field study with 30 university students, comparing it with a text-based mobile diary using matched diary items, reporting windows, and reminder intervals. Compared with the text-based diary, the conversational voice diary showed higher adherence and elicited more detailed contextual self-report about routines, stressors, environmental conditions, and other sleep-related factors. Participants also described the voice diary as easier to integrate into daily routines, despite longer perceived completion time. However, voice-based conversational intake produced lower completeness for some structured diary fields, revealing a trade-off between expressive richness and structured precision. These findings show both the promise and the challenge of using LLM-powered conversational voice assistants for longitudinal health self-report.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15851v2">Narrative Theory-Driven LLM Methods for Automatic Story Generation and Understanding: A Survey</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 31 pages
    </div>
    <details class="paper-abstract">
      Applications of narrative theories using large language models (LLMs) deliver promising methods in automatic story generation and understanding tasks. Our survey examines how natural language processing (NLP) research uses LLM methods to engage with diverse concepts from narrative studies. We use established distinctions from narratology to categorise ongoing efforts and discover the following: \redtext{(a) narrative texts come from diverse sources beyond just literature, (b) theoretical synthesis and validation are potential outcomes, (c) generation tasks lag behind understanding in several ways: theoretical application, post-training methods, exploring non-fiction narratives and addressing narrative levels beyond fabula and discourse.} For future directions, instead of the pursuit of a single, generalised benchmark for `narrative quality', we believe that progress can benefit from efforts that focus on the following: defining and improving theory-based metrics for individual narrative attributes; continue conducting large-scale, theory-driven literary/social/cultural analysis; generating narratives in situated contexts; and continuing experiments where outputs can be used to validate or refine narrative theories. This work provides a contextual foundation for more systematic and theoretically informed narrative research in NLP by providing an overview to ongoing research efforts and the broader narrative studies landscape.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.02690v3">Outrunning LLM Cutoffs: A Live Kernel Crash Resolution Benchmark for All</a></div>
    <div class="paper-meta">
      📅 2026-06-17
    </div>
    <details class="paper-abstract">
      Repairing system crashes discovered by kernel fuzzers like Syzkaller is a critical yet underexplored challenge in software engineering. While recent works have introduced Large Language Model (LLM) based agents for Linux kernel crash-resolution, their evaluation benchmarks are usually static and thus, do not capture the evolving nature of the Linux kernel, and suffer from potential data contamination due to LLM knowledge cutoffs. To address the above problem, we present (i) Live-kBench, an evaluation framework for self-evolving benchmarks that continuously scrapes and evaluates agents on freshly discovered kernel bugs, and (ii) kEnv, an agent-agnostic standardized crash-resolution environment for kernel compilation, execution, and feedback. This design decouples agent workflows from heavy-weight execution, enabling fair and scalable comparison across diverse agent frameworks under identical conditions. To this end, we curate an inaugural dataset of 534 Linux kernel bugs and empirically demonstrate a significant performance gap, with agents achieving up to 25% higher equivalent patch rate on bugs fixed before the LLM knowledge cutoff. Using kEnv, we benchmark three state-of-the-art agents, showing that they resolve 74% of crashes on the first attempt (plausible patches); however only ~20% of generated patches closely match developer fixes. Additionally, exposing crash resolution feedback improves crash resolution rate by 29%. Live-kBench provides the community with an evaluation infrastructure for self-evolving benchmarks that is both time and attribute sensitive; complete with a public dashboard to track agent progress on Linux kernel bugs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.00510v2">PCBSchemaGen: Reward-Guided LLM Code Synthesis for Printed Circuit Boards (PCB) Schematic Design with Structured Verification</a></div>
    <div class="paper-meta">
      📅 2026-06-17
    </div>
    <details class="paper-abstract">
      Most LLM code-synthesis benchmarks rely on unit tests as the reward oracle, but PCB schematic design has none: correctness is defined by structured physical constraints over real IC packages and pin-level assignments, per-task golden references are unavailable, and SPICE simulation does not validate schematic-level correctness. We introduce PCBSchemaGen, a training-free inference-time framework that turns a frozen LLM into a verifiable, repairable PCB schematic generator. The framework induces a domain schema from IC datasheets to ground LLM decoding, pairs it with a deterministic 5-layer continuous-reward verifier with pin-level error localization, and refines candidates through a Thompson Sampling arm-acquiring bandit. We evaluate on 2 PCB benchmarks covering 227 real-IC tasks across 22 unified circuit domains, including a public-schematic-derived suite that serves as a fully held-out generalization test (verifier, KG library, and prompts frozen before any evaluation). Under our framework, an open-weight 31B model (Gemma-4-31B) passes 81.3% of PCBBench tasks on average, and the same framework transfers across both benchmarks with zero verifier code changes; a Circuitron-style inference-time prompting baseline on the same Gemma-4-31B backbone collapses on hard system-level designs. This suggests inference-time refinement under a deterministic structural verifier is a general recipe for reference-free LLM code synthesis in domains without unit-test oracles. Our benchmarks and deterministic verifier are publicly available at https://github.com/HZou9/PCBSchemaGen_v2.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22495v3">Reinforcement-aware Knowledge Distillation for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-17
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) post-training has recently driven major gains in long chain-of-thought reasoning large language models (LLMs), but the high inference cost of such models motivates distillation into smaller students. Most existing knowledge distillation (KD) methods are designed for supervised fine-tuning (SFT), relying on fixed teacher traces or teacher-student Kullback-Leibler (KL) divergence-based regularization. When combined with RL, these approaches often suffer from distribution mismatch and objective interference: teacher supervision may not align with the student's evolving rollout distribution, and the KL regularizer can compete with reward maximization and require careful loss balancing. To address these issues, we propose RL-aware distillation (RLAD), which performs selective imitation during RL -- guiding the student toward the teacher only when it improves the current policy update. Our core component, Trust Region Ratio Distillation (TRRD), replaces the teacher-student KL regularizer with a PPO/GRPO-style likelihood-ratio objective anchored to a teacher--old-policy mixture, yielding advantage-aware, trust-region-bounded distillation on student rollouts and naturally balancing exploration, exploitation, and imitation. Across diverse logic reasoning and math benchmarks, RLAD consistently outperforms offline distillation, standard GRPO, and KL-based on-policy teacher-student knowledge distillation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19644v1">Prompt Quality and Pull Request Outcomes: A Stage-Based Empirical Study of LLM-Assisted Development</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 48 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-powered tools such as ChatGPT are increasingly used in collaborative software engineering workflows, yet little is known about how prompt structure influences downstream pull request (PR) outcomes. Prior studies primarily examine conversational helpfulness, productivity, or coarse-grained adoption metrics, leaving the role of prompt structure in collaborative integration behavior insufficiently understood. We analyze 265 manually validated developer-ChatGPT interactions derived from self-admitted ChatGPT usage in open-source pull requests. Building on prior research on developer-facing artifacts and prompt engineering, we operationalize prompt structure using three dimensions: Context, Specificity, and Verification. We first evaluate whether LLM-assisted annotation can reliably reproduce human judgments of prompt structure, finding substantial variation across dimensions and workflow contexts. Specificity shows the most stable agreement with human judgments; Context is systematically under-scored by the LLM; and Verification remains difficult to assess consistently, motivating a hybrid human-LLM annotation strategy. Using this validated framework, we then examine how prompt structure influences actionable code generation, code adoption, and integration depth across AI-assisted PR workflows. Specificity and Context are most strongly associated with actionable code generation; Verification emerges as the primary predictor of code adoption; and integration depth is most strongly associated with Context. Overall, our findings show that prompt characteristics exert distinct, stage-dependent effects across AI-assisted software engineering workflows, influencing downstream adoption and integration through contextual grounding, task specificity, and evaluability cues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19605v1">FAPO: Fully Autonomous Prompt Optimization of Multi-Step LLM Pipelines</a></div>
    <div class="paper-meta">
      📅 2026-06-17
    </div>
    <details class="paper-abstract">
      Multi-step LLM pipelines fail through interactions among retrieval, reasoning, and formatting steps, so prompt-only optimization can miss bottlenecks in the chain. We present FAPO (Fully Autonomous Prompt Optimization), a framework that lets Claude Code optimize an LLM pipeline inside a standardized codebase. FAPO evaluates a pipeline, inspects intermediate steps, diagnoses failures, proposes scoped changes, and validates variants repeatedly to optimize against a score function. It first tries prompt edits and, only when prompt optimization appears insufficient, changes chain structure within the permitted scope when attribution identifies a structural bottleneck. Across six benchmarks and three task models, FAPO beats the baseline GEPA in 15 of 18 model-benchmark comparisons. In 11 model-benchmark comparisons, FAPO wins with non-overlapping mean $\pm$ trial-standard-deviation ranges, and the mean FAPO-GEPA gain is +14.1 pp. In the six HoVer and IFBench comparisons where prompt-first search escalated to structural changes, FAPO wins all six with a mean gain of +33.8 pp. FAPO also improves performance on security tasks: on CTIBench-RCM, a security CVE-to-CWE task, prompt-only FAPO lifts test accuracy by +4.0 pp on GPT-5, +7.1 pp on Foundation-Sec-8B-Instruct, and +2.0 pp on Foundation-Sec-8B-Reasoning. These results position FAPO as a state-of-the-art pipeline optimization technique for both general-purpose and security-focused tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19588v1">Analyzing the Narration Gap in LLM-Solver Loops</a></div>
    <div class="paper-meta">
      📅 2026-06-17
    </div>
    <details class="paper-abstract">
      Formal tools such as SAT and SMT solvers are increasingly embedded in language model reasoning pipelines when a safety or security critical question can be formulated in logic. Unlike chain of thought whose steps are sampled from the model distribution without formal guarantee, a solver produces a sound and independently verifiable answer. However, the soundness guarantee can be lost in the interaction between the solver and the model. The hybrid pipeline has three components: formalizing the question, deciding it, and narrating the result. Prior work has studied the formalization and decision, but not narration, which is the step that turns a formal tool's output into the user answer. To fill the narration gap, we first model the LLM-solver loop as a verified decision procedure. We further evaluate five open-sourced models under prompt injection, and we find certificate gating makes the solver verdict sound, while an adversary can invert a verified conclusion across phrasings and channels. We study the mitigation through hardened prompt that reduces injection significantly but cannot eliminate it and still suffers under adaptive attack. Combining the formal analysis and empirical studies, we show in the LLM-solver loop, robustness does not reach to the answer that the user finally reads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17041v3">Benchmarking LLM Agents on Meta-Analysis Articles from Nature Portfolio</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 13 pages, 7 figures, preprint for arXiv, dataset and code available at https://github.com/BFTree/MetaSyn
    </div>
    <details class="paper-abstract">
      Meta-analysis is a demanding form of evidence synthesis that combines literature retrieval, PI/ECO-guided study selection, and statistical aggregation. Its structured, verifiable workflow makes it an ideal substrate for evaluating systematic scientific reasoning, yet existing benchmarks lack ground truth across the full retrieval-screening-synthesis pipeline. We introduce MetaSyn, a dataset of 442 expert-curated meta-analyses from Nature Portfolio journals. Each entry pairs a research question with PI/ECO criteria, a retrieval corpus of 140k PubMed articles, verified positive studies, hard negatives that are topically similar but PI/ECO-ineligible, and complete search strategies and date bounds. Benchmarking twelve pipeline configurations (nine RAG variants and a protocol-driven agent) reveals a critical screening bottleneck: despite a retrieval ceiling of 90.9% recall at K=200, no system recovers more than 52.7% of ground-truth included literature. Current LLMs fail to reliably separate eligible studies from PI/ECO-failing distractors in pools of comparable topical relevance. Stage-attributed metrics capture where systems succeed and fail; a single end-to-end score does not.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19559v1">Uncertainty Decomposition for Clarification Seeking in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 26 pages, 8 figures. Source code: https://github.com/PE51K/udcs-in-llm-agents
    </div>
    <details class="paper-abstract">
      Recent position papers argue that the classical aleatoric/epistemic uncertainty framework is insufficient for interactive large language model (LLM) agents and call for underspecification-aware, decomposed, and communicable uncertainty representations that can unlock new agent capabilities such as proactive clarification seeking and shared mental-model building. Practical deployment constraints -- black-box APIs, interactive latency budgets, and the absence of labeled trajectories -- rule out logprob-based, multi-sampling, and training-based methods, leaving prompt-based estimation as the most viable family for surfacing such signals at deployment time. We answer this call with a simple prompt-based decomposition that separates action confidence from request uncertainty (u), enabling the agent to ask for clarification when the task specification is ambiguous. To evaluate it, we introduce two clarification-augmented benchmarks (WebShop-Clarification and ALFWorld-Clarification) in which 50% of tasks are deliberately underspecified, and systematically compare the proposed decomposition against ReAct+UE and Uncertainty-Aware Memory (UAM) across five LLM backbones (GPT-5.1, DeepSeek-v3.2-exp, GLM-4.7, Qwen3.5-35B, GPT-OSS-120B) on these variants together with the standard WebShop, ALFWorld, and REAL benchmarks for fault detection. Averaged across the five backbones, the proposed decomposition improves clarification F1 on ALFWorld-Clarification by 73% over ReAct+UE and by 36% over UAM, and leads clarification F1 on every backbone on WebShop-Clarification and on four of five backbones on ALFWorld-Clarification, indicating that the gains generalize beyond a single LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19544v1">Reliability without Validity: A Systematic, Large-Scale Evaluation of LLM-as-a-Judge Models Across Agreement, Consistency, and Bias</a></div>
    <div class="paper-meta">
      📅 2026-06-17
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge has become the dominant evaluation paradigm for language models, but judge validation in practice relies on exact-match agreement, a metric that does not correct for chance and systematically overstates discriminative ability. We present the largest systematic evaluation of LLM-as-a-Judge to date: 21 judges from nine providers across MT-Bench, JudgeBench, and RewardBench, evaluated under three protocols (agreement, consistency, bias audit) over 118 runs and approximately 541,000 individual judgments. Four findings emerge, consistent across the full cohort, including the April 2026 frontier: kappa deflation between exact match and Cohen's kappa is universal (33--41 pp on MT-Bench), judge rankings shift by up to 14 positions across benchmarks, high test--retest reliability (>0.95) coexists with severe position bias (>0.10) in two production-deployed judges (instantiating a consistency--bias paradox), and verbosity bias is small (<0.011) across our cohort under a single pairwise rubric. We distill these into a Minimum Viable Validation Protocol.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19535v1">FloatDoor: Platform-Triggered Backdoors in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in sensitive settings such as software engineering, where their outputs directly shape downstream artifacts. Recent work has shown that an identical model can produce measurably different outputs depending on the deployment platform, a consequence of non-associative floating-point arithmetic and divergent kernel implementations. We study the security implications of this platform-dependent variability and uncover a novel attack surface on LLM deployments. We introduce FloatDoor, the first input-independent, platform-triggered backdoor attack against generative LLMs. The compromised model exhibits adversary-chosen behavior when served on a target platform and is otherwise benign. FloatDoor is realized through two lightweight LoRA adapters, one that amplifies inter-platform numerical divergence and one that binds the resulting platform signature to a malicious downstream task, while leaving aggregate model utility largely intact. FloatDoor exploits a pronounced time-of-check, time-of-use gap between model auditing and serving. We demonstrate FloatDoor on Qwen3-4B across a broad range of deployment targets, including NVIDIA GPUs, Google TPUs, AWS Graviton, and Alibaba Yitian-710. As a final case study, we show that FloatDoor reliably induces exploitable code vulnerabilities on a chosen target platform. Our results establish a new class of attacks on LLM deployments and underscore the pressing need for trusted model supply chains in sensitive, LLM-powered applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19528v1">Techniques for Peak Memory Reduction for LoRA Fine-tuning of LLMs on Edge Devices</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 Hassan Dbouk and Matthias Reisser contributed equally to this work
    </div>
    <details class="paper-abstract">
      Fine-tuning of Large Language Models (LLMs) using Low-Rank Adaptation (LoRA) on an end-user's data offers personalized experiences while keeping data private, but faces severe memory constraints on consumer hardware. Peak memory during fine-tuning often exceeds device limits, especially for models with billions of parameters and long-context training data. This paper introduces a suite of complementary techniques to reduce memory footprint without sacrificing model quality: (1) base model quantization with on-the-fly dequantization, (2) memory-efficient checkpointing combining selective activation caching and disk offloading, (3) softmax approximation using semantically relevant token subsets, and (4) logits masking. Experiments on Llama-3.2 3B and Qwen-2.5 3B demonstrate up to $26\times$ and $28\times$ reduction in peak memory, enabling fine-tuning on resource-constrained devices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19514v1">LLM-Mediated Human-AI Interaction in Search and Rescue: Impact of Expertise on Attentional Allocation</a></div>
    <div class="paper-meta">
      📅 2026-06-17
    </div>
    <details class="paper-abstract">
      Human-AI teaming (HAT) increasingly involves AI systems that provide real-time, context-aware guidance in complex tasks. While such systems can improve performance, their effectiveness depends on how they shape human cognition and behavior. In particular, AI assistance can introduce cognitive demands and influence attention, planning, and interaction with the task environment, with effects that can vary across levels of expertise. This work investigates these mechanisms in a simulated search and rescue (SAR) environment. We compare human performance under two LLM (Large Language Model)-guided conditions and a no-LLM baseline, and analyze interaction at multiple levels, including task performance, eye-tracking measures, and planning behavior. Eye tracking provides fine-grained insight into attention allocation and interaction with AI guidance, while behavioral measures capture how users structure and adapt their decisions over time. Results indicate that LLM guidance enhanced task efficiency (higher rewards and victims-per-step) but did not increase total victims saved. Eye-tracking data revealed an attention-guidance trade-off, with visual resources shifting to the chat interface alongside increased pupil size variability. Expertise moderated this effect: novices exhibited passive AI reliance, whereas experts maintained a "verification loop" through persistent environmental scanning. These findings suggest that LLM-mediated teaming efficacy depends on the operator's ability to cross-reference AI guidance with ground truth to maintain situational awareness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19509v1">LLM Doesn't Know What It Doesn't Know: Detecting Epistemic Blind Spots via Cross-Model Attribution Divergence on Clinical Tabular Data</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 Accepted at EIML@ICML 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly applied to structured clinical data, yet whether they can recognize the limits of their own knowledge on such tasks remains unexplored. We study this question through the lens of cross-model attribution divergence with the goal of reducing epistemic uncertainty for structured tasks, comparing Qwen 2.5 7B and XGBoost on a prediction task via attribution divergence analysis. We report four findings. First, LLM verbalized confidence is epistemically vacuous, it outputs a near-constant (0.856-0.937) regardless of whether accuracy is 49% or 75.3%, tracking prompt format rather than prediction quality. Second, the LLM exhibits an inverse difficulty effect: accuracy drops to 64.8% when XGBoost is 99% correct, but matches XGBoost (73.8% vs. 73.1%) when it is moderately uncertain. Third, few-shot examples and SHAP-derived feature evidence are orthogonal, super-additive interventions: they reduce the Attribution Disagreement Score (ADS) from 1.54 to 0.38 and improve accuracy from 49% to 75.3% without training. Fourth, a cross-model calibrator that determined LLM reliability using attribution divergence signals reduces expected calibration error from 0.254 to 0.080, replacing uninformative verbalized confidence with patient-specific reliability estimates, without accessing model internals or requiring repeated inference. We frame these findings as a cold start problem for LLMs on structured data and outline a path toward genuine epistemic self-awareness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16357v2">Beyond Grading Accuracy: Exploring Alignment of TAs and LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 7 pages, 3 figures
    </div>
    <details class="paper-abstract">
      In this paper, we investigate the potential of open-source Large Language Models (LLMs) for grading Unified Modeling Language (UML) class diagrams. In contrast to existing work, which primarily evaluates proprietary LLMs, we focus on non-proprietary models, making our approach suitable for universities where transparency and cost are critical. Additionally, existing studies assess performance over complete diagrams rather than individual criteria, offering limited insight into how automated grading aligns with human evaluation. To address these gaps, we propose a grading pipeline in which student-generated UML class diagrams are independently evaluated by both teaching assistants (TAs) and LLMs. Grades are then compared at the level of individual criteria. We evaluate this pipeline through a quantitative study of 92 UML class diagrams from a software design course, comparing TA grades against assessments produced by six open-source LLMs. Performance is measured across individual criteria, highlighting areas where LLMs diverge from human graders. Our results show per-criterion accuracy of up to 88.56\% and a Pearson correlation coefficient of up to 0.78, representing a substantial improvement over previous work while using only open-source models. The models achieve performance close to that of a TA, suggesting a possible path toward a mixed-initiative grading system, where TAs are aided in their grading. Our findings demonstrate that open-source LLMs can effectively support UML class diagram grading by explicitly identifying alignment with grading criteria. The proposed pipeline provides a practical approach to managing increasing workloads with growing student counts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19494v1">Hidden Anchors in Multi-Agent LLM Deliberation</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 13 pages, 6 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Multi-agent LLM deliberation, where agents exchange and revise answers over several rounds, is increasingly used to improve reasoning and accuracy, yet how and why it works is rarely modelled. Such deliberation mirrors how humans reach decisions. As social animals we are pulled both by the group, the herd effect that classical opinion-dynamics models such as DeGroot and Friedkin--Johnsen capture, and by our own internal belief, which they do not. We model multi-agent deliberation as a closed-loop dynamical system in which each agent carries a hidden internal belief, its anchor, that continually pulls its opinion regardless of its neighbours. We show this anchor can be recovered from the deliberation alone, and that it explains a behaviour classical consensus rules forbid: an agent's confidence in the correct answer can climb past where any agent started, escaping the space (convexhull) formed by the initial beliefs. Checking whether the recovered anchor also predicts held-out runs (generalizes) gives a simple test for when a model is truly driven bysuch an anchor. Across three open-weight model families this is a spectrum, not all-or-nothing. All anchors' influence are about equally strongly, but they differ in where the anchor sits, and only when it sits far from the initial opinions does deliberation escape the hull and need the full closed-loop model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19474v1">Secure Coding Drift in LLM-Assisted Post-Quantum Cryptography Development: A Gamified Fix</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 Accepted for 2026 SIGIR Workshop on Vulnerabilities in Generative Systems for Information Retrieval track
    </div>
    <details class="paper-abstract">
      The transition to Post Quantum Cryptography (PQC) introduces considerable implementation complexity, requiring strict adherence to constant-time execution, side channel resistance, and precise parametrisation. Simultaneously, large language models (LLMs) are heavily embedded in software development workflows, including cryptographic engineering. While LLMs improve productivity, evidence shows that they frequently generate insecure or suboptimal code, particularly in security critical domains. This paper introduces Secure Coding Drift in PQC, a novel socio technical vulnerability model capturing the gradual degradation of secure coding practices due to sustained reliance on LLM-generated code. Unlike prior work that focuses on static vulnerabilities, we conceptualise security risk as a longitudinal behavioural phenomenon rising from human AI interaction. To mitigate this, we propose a gamified, LLM augmented secure coding framework that embeds adversarial evaluation, behavioural feedback, and security scoring into development workflows. Our approach reframes LLMs from passive assistants into active security co-pilots, contributing toward safer PQC implementation in AI mediated environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19468v1">Characterizing Narrative Content in Web-scale LLM Pretraining Data</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 8 pages of main content, 28 total pages. 30 figures
    </div>
    <details class="paper-abstract">
      The narrative composition of web-scale LLM pretraining corpora remains largely unexplored even though narrative is a fundamental mode of human communication. We present the first fine-grained study of narrative features in Dolma, a 3-trillion-token open pretraining corpus. Drawing on narrative theory, we design a framework spanning three core narrative elements (agency, setting, and events) operationalized as 11 interpretable dimensions. After sampling and annotating a diverse set of 400 passages, we finetune and validate NarraBERT, a RoBERTa-based model for fine-grained narrative prediction. We apply NarraBERT to 3M passages, resulting in a new dataset, NarraDolma. We find (i) narrative structure is measurable at scale across extremely heterogeneous data, (ii) we uncover a continuous, multidimensional narrative structure underlying web text, and (iii) narrative qualities are unequally distributed across pretraining sources and topics in ways that current curation practices neither measure nor account for. Our framework, dataset, and analyses provide a foundation for understanding how narrative qualities are distributed in LLM pretraining data and for studying how data composition affects narrative reasoning tasks. We publicly release NarraDolma and NarraBERT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17276v2">On the Memorization Behavior of LLMs in Generative Recommendation: Observations, Implications, and Training Strategies</a></div>
    <div class="paper-meta">
      📅 2026-06-17
    </div>
    <details class="paper-abstract">
      Generative recommendation (GR) has emerged as a promising direction for recommender systems. Recently, large language models (LLMs) have been increasingly adopted for GR, as their rich pretrained knowledge is expected to help them generalize beyond common user behavior patterns that traditional memorization-oriented baselines can capture. However, existing LLM-based GR works largely ignore LLMs' well-known tendency to memorize, which, if present in LLMs fine-tuned for GR, would restrict their utilization of pretrained knowledge. In this work, we investigate this concern by examining one-hop memorization, where a model recommends items that are direct successors of items in the training data. We show that LLMs do this more than non-LLM-based GR models-in fact, the vast majority of their gains over GR baselines are actually on users whose target items can be predicted through one-hop memorization. We intuit that improving performance on the remaining users requires LLMs to learn richer item-item relations beyond one-hop transitions. To achieve this, we propose IIRG, a novel training strategy that teaches LLMs to capture: (1) collaborative relations derived from item co-occurrences across multiple hops in user sequences, and (2) semantic relations among items with similar themes, both of which can serve as useful recommendation signals. We show that IIRG significantly improves over LLMs trained solely with standard next-item prediction, with especially large gains for users whose test items are not covered by train-time one-hop transitions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.04219v5">Model Collapse Is Not a Bug but a Feature in Machine Unlearning for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 Accepted at ICLR 2026
    </div>
    <details class="paper-abstract">
      Current unlearning methods for LLMs optimize on the private information they seek to remove by incorporating it into their fine-tuning data. We argue this not only risks reinforcing exposure to sensitive data, but also fundamentally contradicts the principle of minimizing its use. As a remedy, we propose a novel unlearning method-Partial Model Collapse (PMC), which does not require unlearning targets in the unlearning objective. Our approach is inspired by recent observations that training generative models on their own generations leads to distribution collapse, effectively removing information from model outputs. Our central insight is that model collapse can be leveraged for machine unlearning by deliberately triggering it for data we aim to remove. We theoretically analyze that our approach converges to the desired outcome, i.e. the model unlearns the data targeted for removal. We empirically demonstrate that PMC overcomes four key limitations of existing unlearning methods that explicitly optimize on unlearning targets, and more effectively removes private information from model outputs while preserving general model utility. Overall, our contributions represent an important step toward more comprehensive unlearning that better aligns with real-world privacy constraints. Code available at https://www.cs.cit.tum.de/daml/partial-model-collapse/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19266v1">Trade-offs in Medical LLM Adaptation: An Empirical Study in French QA</a></div>
    <div class="paper-meta">
      📅 2026-06-17
    </div>
    <details class="paper-abstract">
      The development of large language models (LLMs) has led to an increased focus on their adaptation to specialized domains and languages, yet the effectiveness of domain adaptation strategies remains unclear. We present a study of medical domain adaptation using French medical question-answering (QA) as a case study. We compare continual pretraining (CPT), supervised fine-tuning (SFT), and their combination across three model families, multiple sizes, and three initialization types, explicitly disentangling adaptation effects from base model choice. We evaluate both multiple-choice (MCQA) and open-ended QA (OEQA) under greedy and constrained decoding using automatic metrics and LLM-as-a-Judge evaluation. For MCQA, CPT+SFT most often achieves the best scores, but gains over SFT are small and frequently not statistically significant, making SFT a strong and cost-effective default. For OEQA, CPT consistently improves overlap-based metrics, while SFT often degrades generation quality; instruction tuning and CPT+SFT are preferred by LLM-based evaluation. Cross-lingual experiments further show effective transfer from French adaptation to English benchmarks. Overall, we provide practical guidelines for selecting adaptation strategies under computational constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10827v2">Speaker Verification with Speech-Aware LLMs: Evaluation and Augmentation</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 3 Tables, 1 Figure, Published in Interspeech 2026
    </div>
    <details class="paper-abstract">
      Speech-aware large language models (LLMs) can accept speech inputs, yet their training objectives largely emphasize linguistic content or specific fields such as emotions or the speaker's gender, leaving it unclear whether they encode speaker identity. First, we propose a model-agnostic scoring protocol that produces continuous verification scores for both API-only and open-weight models, using confidence scores or log-likelihood ratios from the Yes/No token probabilities. Using this protocol, we benchmark recent speech-aware LLMs and observe weak speaker discrimination (EERs above 20% on VoxCeleb1). Second, we introduce a lightweight augmentation that equips an LLM with ASV capability by injecting frozen ECAPA-TDNN speaker embeddings through a learned projection and training only LoRA adapters. On TinyLLaMA-1.1B, the resulting ECAPA-LLM achieves 1.03% EER on VoxCeleb1-E, approaching a dedicated speaker verification system while preserving a natural-language interface.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19183v1">Language Models as Interfaces, Not Oracles: A Hybrid LLM-ML System for Pediatric Appendicitis</a></div>
    <div class="paper-meta">
      📅 2026-06-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can make clinical decision support more accessible by interpreting free-text documentation, but their direct use as diagnostic engines is limited by sensitivity to prompts, information order, and plausible but incorrect outputs. Structured machine-learning models offer more stable risk prediction, yet they require tabular inputs that are difficult to integrate with narrative clinical workflows. We present ClaMPAPP (Clinical Language-assisted Machine-learning Pipeline for Appendicitis), a hybrid system that uses an LLM as an interface rather than as the final decision-maker. ClaMPAPP extracts schema-constrained clinical features from note-like narratives, applies deterministic plausibility checks, and passes validated features to an XGBoost classifier trained on clinical, laboratory, and ultrasound variables. We evaluated ClaMPAPP on two independent pediatric appendicitis cohorts from German hospitals and compared it with end-to-end LLM baselines, including open-source and proprietary models. To preserve ground truth while testing free-text input, narratives were generated from structured electronic health records through template rendering and constrained LLM rewriting, with additional sentence-order permutation to assess positional robustness. ClaMPAPP achieved the strongest overall diagnostic performance in both internal and external validation while minimizing missed appendicitis cases, the key safety concern in acute triage. End-to-end LLMs showed unstable sensitivity-specificity trade-offs and greater degradation under narrative reordering. These results support an LLM-as-interface, ML-as-predictor design that separates natural-language usability from predictive inference and provides a more auditable pathway for clinical decision support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.23092v2">Enhancing CVRP Solver through LLM-driven Automatic Heuristic Design</a></div>
    <div class="paper-meta">
      📅 2026-06-17
    </div>
    <details class="paper-abstract">
      The Capacitated Vehicle Routing Problem (CVRP), a fundamental combinatorial optimization challenge, focuses on optimizing fleet operations under vehicle capacity constraints. While extensively studied in operational research, the NP-hard nature of CVRP continues to pose significant computational challenges, particularly for large-scale instances. This study presents AILS-AHD (Adaptive Iterated Local Search with Automatic Heuristic Design), a novel approach that leverages Large Language Models (LLMs) to revolutionize CVRP solving. Our methodology integrates an evolutionary search framework with LLMs to dynamically generate and optimize ruin heuristics within the AILS method. Additionally, we introduce an LLM-based acceleration mechanism to enhance computational efficiency. Comprehensive experimental evaluations against state-of-the-art solvers, including AILS-II and HGS, demonstrate the superior performance of AILS-AHD across both moderate and large-scale instances. Notably, our approach establishes new best-known solutions for 8 out of 10 instances in the CVRPLib large-scale benchmark, underscoring the potential of LLM-driven heuristic design in advancing the field of vehicle routing optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19167v1">Teaching Software Engineering with LLM and MCP Integration: From Classroom to Industry Practice</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 Aceept by International Symposium on Educational Technology (ISET) 2026
    </div>
    <details class="paper-abstract">
      The rapid integration of Large Language Models (LLMs) and the Model Context Protocol (MCP) into industrial software engineering has created a pressing need to update software engineering education to align with emerging technologies and evolving industry demands. This study investigates an innovative approach that integrates LLMs and MCP into a collaborative teaching model for software engineering education, aiming to build a practical learning framework closely connected to real-world engineering practices. By embedding LLM and MCP driven tools into daily teaching, code assistance, and engineering simulations, the model effectively bridges the gap between traditional instruction and industrial workflows. This integration enhances students' programming competence, practical problem-solving abilities, and proficiency in using intelligent engineering tools. Furthermore, through partnerships with industry internships, students can apply these technologies in real-world settings, further strengthening the connection between academic preparation and professional practice. Overall, this research offers a practical pathway for reforming and innovating software engineering education in the era of artificial intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19135v1">A Technical Taxonomy of LLM Agent Communication Protocols</a></div>
    <div class="paper-meta">
      📅 2026-06-17
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) advance and multi-agent systems aim to overcome the limits of standalone agents, robust communication protocols are becoming essential infrastructure for distributed agent networks. Nonetheless, the fragmented protocol landscape presents a significant interoperability challenge. This study develops a technical taxonomy to classify and analyze LLM agent communication protocols. Following an established iterative method, we defined the taxonomy's purpose, meta-characteristic, and ending conditions, then performed five iterations, three empirical-to-conceptual and two conceptual-to-empirical, on nine actively maintained open-source protocols with demonstrable adoption. The taxonomy comprises five dimensions: counterparty, payload, interaction state, discovery mechanism, and schema flexibility. Classification reveals recurring architectural patterns: all sampled agent-to-agent protocols combine hybrid payloads with session-state persistence; most protocols support multiple predefined schemas, and two negotiate schemas at runtime, indicating a trend toward schema flexibility; decentralized discovery remains rare. Analysis suggests short-term convergence pressure toward protocols unifying agent-to-agent and agent-to-context (tool and data) communication. Long-term, however, no single protocol is likely to maximize versatility, efficiency, and portability simultaneously. The field will more likely evolve toward a federated, layered protocol stack. The framework guides protocol selection and highlights open research gaps such as privacy and policy enforcement.}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19111v1">Leadership as Coordination Control: Behavioral Signatures and the Recovery-Advantage Boundary in Multi-Agent LLM Teams</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 33 pages
    </div>
    <details class="paper-abstract">
      Team science holds that leadership is contingent: it helps only under specific conditions, and capable, autonomous teams may need none at all. We ask the analogous question for multi-agent LLM teams: under what measurable conditions does process-level coordination control add value, and do those conditions match what team science predicts? We use behavioral signatures (majority lock-in, exploration, recovery from an incorrect round-0 consensus) and per-action ablations, clean because each controller is an explicit action set, not a monolithic prompt. We operationalize three classical leadership styles (transactional, transformational, situational) as controllers over a shared action vocabulary (explore, revise, accept, synthesize). A matched controller with the same actions but an arbitrary rule recovers no better than majority voting, so the theory-derived rule, not the vocabulary, does the work. Across four task regimes and three open-weight model families, no controller dominates by accuracy, as the contingency view predicts: transactional control matches a shared round-0 vote on all 12 (model, regime) combinations to within 1.3pp, and gains appear only on the one combination where the round-0 majority is unreliable (llama-4-scout social; situational +8pp over flat). A recovery-advantage account, tested with four boundary probes, says a controller beats plain interaction only where the round-0 majority is unreliable, the task is recoverable, and undirected interaction does not already repair it. These regions map onto contingency theory (leadership substitutes, path-goal redundancy, the situational readiness gap), so a largely null accuracy result is what the theory predicts, not a failure of the controllers. We read process-level coordination control as a contingency to be measured and theory-mapped, not a leaderboard to be topped.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.20045v3">Efficient Hallucination Detection for LLMs Using Uncertainty-Aware Attention Heads</a></div>
    <div class="paper-meta">
      📅 2026-06-17
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have become highly capable, they remain prone to factual inaccuracies, commonly referred to as "hallucinations." Uncertainty quantification (UQ) offers a promising way to mitigate this issue, but most existing methods are computationally intensive and/or require supervision. In this work, we propose Recurrent Attention-based Uncertainty Quantification (RAUQ), an unsupervised and efficient framework for identifying hallucinations. The method leverages an observation about transformer attention behavior: when incorrect information is generated, certain "uncertainty-aware" attention heads tend to reduce their focus on preceding tokens. RAUQ automatically detects these attention heads and combines their activation patterns with token-level confidence measures in a recurrent scheme, producing a sequence-level uncertainty estimate in just a single forward pass. Through experiments on twelve datasets spanning question answering, summarization, and translation across nine different LLMs, we show that RAUQ consistently outperforms state-of-the-art UQ baselines. Importantly, it incurs minimal overhead, requiring less than 1\% additional computation. Since it requires neither labeled data nor extensive parameter tuning, RAUQ serves as a lightweight, plug-and-play solution for real-time hallucination detection in white-box LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19057v1">Quantifying and Auditing LLM Evaluation via Positive--Unlabeled Learning</a></div>
    <div class="paper-meta">
      📅 2026-06-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used as judges for scalable evaluation, yet such LLM--as--a--Judge systems exhibit systematic biases that are decoupled from semantic quality, most notably verbosity bias. Meanwhile, human supervision is costly and typically selective, yielding reliable positive judgments but leaving most outputs unlabelled and potentially mixed in quality. We formulate LLM evaluation under selective human supervision as a positive--unlabelled learning problem and propose a geometric auditing framework based on Partial Optimal Transport. By aligning a small set of human--verified positives with a reliable subset of unlabelled outputs in a fixed embedding space, our method identifies human--consistent preferences and corrects biased judges without retraining. Experiments demonstrate improved alignment with human preferences, increased robustness to presentation biases, and interpretable confidence estimates, offering a scalable and statistically grounded alternative to existing LLM--as--a--judge pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.00161v2">LLM Compression by Block Removal with Constrained Binary Optimization</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 16 pages, 3 figures
    </div>
    <details class="paper-abstract">
      In this paper, we formulate the compression of large language models (LLMs) by optimally deleting transformer blocks (``block removal'') as a constrained binary optimization (CBO) problem that can be mapped to a physical system (Ising glass), whose energies are a strong proxy for downstream model performance. This formulation enables an efficient ranking of a large number of candidate block-removal configurations yielding many high-quality, non-trivial solutions beyond those only removing consecutive regions. Our method performs strongly in the deep compression regime, such as for 50% compression of Llama-3.3-70B-Instruct, where we achieve an almost 23 percentage point increase on the MMLU benchmark compared to other state-of-the-art (SOTA) block-removal methods. For lighter compression, it performs on par with those methods across several benchmarks for Llama-3.1-8B-Instruct, Qwen3-14B (both before and after retraining), as well as Llama-3.3-70B-Instruct. The approach is computationally efficient and requires only forward and backward passes on a calibration dataset for a few active parameters. Additionally, we demonstrate that using good heuristic solvers for the CBO problem provides solutions that perform well on downstream tasks in negligible runtime when it is unfeasible to solve the problem exactly. The method can be readily applied to any architecture. We illustrate this generality on the recent NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 model, which exhibits a highly inhomogeneous and challenging block structure, and where we outperform SOTA for AIME25 and GPQA when removing either 2 attention layers or 3 mixture-of-experts layers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.09191v2">From Values to Tokens: An LLM-Driven Framework for Context-aware Time Series Forecasting via Symbolic Discretization</a></div>
    <div class="paper-meta">
      📅 2026-06-17
    </div>
    <details class="paper-abstract">
      Time series forecasting plays a vital role in supporting decision-making across a wide range of critical applications, including energy, healthcare, and finance. Despite recent advances, forecasting accuracy remains limited due to the challenge of integrating historical numerical sequences with contextual features, which often comprise unstructured textual data. To address this challenge, we propose TokenCast, a large language model (LLM) driven framework that leverages language-based symbolic representations as a unified intermediary for context-aware time series forecasting. Specifically, TokenCast employs a discrete tokenizer to transform continuous numerical sequences into temporal tokens, enabling structural alignment with language-based inputs. To effectively bridge the semantic gap between modalities, both temporal and contextual tokens are embedded into a shared representation space via a pre-trained LLM, further optimized with generative objectives. Building upon this unified semantic space, the aligned LLM is subsequently fine-tuned in a supervised manner to predict future temporal tokens, which are then decoded back into the original numerical space. Extensive experiments on real-world datasets demonstrate the effectiveness of our framework and highlight its potential as a generative framework for context-aware time series forecasting. The code is available at https://github.com/Xiaoyu-Tao/TokenCast.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18976v1">CAPRA: Scaling Feedback on Software Architecture Deliverables with a Multi-Agent LLM System</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 Accepted for publication at the 38th International Conference on Software Engineering Education and Training
    </div>
    <details class="paper-abstract">
      Automated assessment in software engineering education has advanced significantly for code grading and essay scoring. However, reviewing software architecture deliverables, which requires analyzing structural completeness and requirements traceability, has not yet been fully automated. Applying Large Language Models (LLMs) to this task requires robust architectures to ensure technical feedback is accurate and reliable for students. This paper presents CAPRA (Configurable Architecture Proficiency Report Assessment), a multi-agent LLM system that analyzes software architecture deliverables to generate personalized, template-compliant LaTeX feedback. As a core design choice, CAPRA coordinates multiple specialized agents and employs a Python-based microservice for multi-modal document extraction, utilizing PyMuPDF and vision-enabled LLMs (specifically gpt-4o) to parse text and UML diagrams. To ensure educational reliability and mitigate hallucinations, CAPRA introduces a deterministic Evidence Anchoring step using fuzzy matching via normalized Levenshtein distance, along with a ConsistencyManager agent that cross-verifies, deduplicates, and merges findings. System performance is assessed using a structured eight-criterion binary evaluation taxonomy covering: (i) extraction completeness, (ii) feature validation, (iii) issue grounding and severity detection, (iv) recommendation specificity and traceability, and (v) template and tone compliance. A preliminary empirical evaluation on 10 student reports shows that CAPRA satisfied 88.8% of the evaluated criteria under a strict two-rater aggregation rule, achieved moderate inter-rater agreement with human evaluators (kappa = 0.582), and processed each report in slightly over 4 minutes. While these results support the viability of LLM-supported architectural feedback, human oversight remains essential for subjective assessment dimensions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18947v1">Decoupling Search from Reasoning: A Vendor-Agnostic Grounding Architecture for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 15 pages, Figure 8
    </div>
    <details class="paper-abstract">
      Production LLM agents increasingly depend on real-time search, yet native search grounding bundles retrieval policy, provider choice, evidence injection, cost, latency, and generation behavior behind a single model-provider boundary. This coupling makes grounding hard to inspect, tune, reuse, or port, and can trigger Search-Induced Verbosity that breaks strict output contracts. We present Decoupled Search Grounding (DSG), a vendor-agnostic boundary that moves grounding outside the reasoning model through an MCP-compatible gateway, exposing provider routing, source-aware context rendering, configured fallback, retrieval-depth control, and exact plus semantic caching as first-class controls. Across five frontier models on SimpleQA, FreshQA, and HotpotQA, native search leads on recency-sensitive FreshQA, but DSG exposes a stronger frontier when control matters: on SimpleQA it nearly matches native accuracy (86.1% vs. 87.7%) at 91% lower search cost, preserves concise answer contracts, and reaches a 99.4% warm-cache hit rate with 68% lower latency. Deployed as a shared production grounding layer for large-scale agentic workloads with interchangeable models, DSG matches or slightly exceeds native-search accuracy on an e-commerce query-understanding (QIU) workload while cutting search cost by over 98%. Real-time grounding is best treated as an optimizable interface boundary, not a fixed model feature.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18924v1">Who Wins the Conflict? Mechanistic Interpretability of Text Bias in Audio LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      While Audio Large Language Models (Audio LLMs) excel at multimodal understanding, they suffer from text dominance, a bias where models blindly favor text over acoustic evidence, causing hallucinations. However, the internal mechanisms underlying how these models behave when audio and textual inputs contradict each other remain unexplored. In this work, we present the first mechanistic analysis of this phenomenon by tracing the propagation of internal representations across layers. Our investigation reveals three key findings: (i) text dominance is systematically and empirically across models; (ii) while text and audio rely on functionally distinct pathways, they ultimately converge into a shared semantic space in late layers; and (iii) the text pathway does not erase audio information, but rather actively suppresses intact audio representations. Building on these insights, we leverage back-patching, a training-free intervention that routes late-layer audio activations back into earlier layers. This amplifies the audio representations, enabling them to overcome textual suppression. Our evaluation shows that back-patching consistently reduces text dominance, paving the way for mechanistic multimodal alignment under conflict.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.04120v2">Probing Semantic Alignment, Lexical Invariance, and Syntactic Influence in LLM Metaphor Processing</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 Accepted to ACL 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) achieve strong performance on metaphor detection and interpretation tasks, yet it remains unclear what such behavioral success reveals about metaphor processing. We present a diagnostic analysis that examines the limits of behavioral evidence by probing three complementary dimensions: semantic attribute alignment, lexical invariance, and syntactic sensitivity. Using geometric probing, we assess whether model-generated interpretations align with reference semantic attributes; through context-varying substitution, we analyze the stability of lexical associations between metaphorical and literal expressions; and via controlled syntactic perturbations, we examine sensitivity in metaphor detection. Our analysis reveals that LLM-generated interpretations can exhibit semantic drift relative to reference attributes; stable lexical anchors persist across contextual conditions, potentially supporting conventional metaphors while biasing novel metaphors requiring contextual integration; and detection performance is sensitive to syntactic irregularities. These findings suggest that strong behavioral performance may reflect heterogeneous underlying signals, highlighting the need for caution when interpreting metaphor benchmarks as evidence of robust, integrated semantic understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13836v2">FutureOmni: Evaluating Future Forecasting from Omni-Modal Context for Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 Accepted by ICML 2026
    </div>
    <details class="paper-abstract">
      Although Multimodal Large Language Models (MLLMs) demonstrate strong omni-modal perception, their ability to forecast future events from audio-visual cues remains largely unexplored, as existing benchmarks focus mainly on retrospective understanding. To bridge this gap, we introduce FutureOmni, the first benchmark designed to evaluate omni-modal future forecasting from audio-visual environments. The evaluated models are required to perform cross-modal causal and temporal reasoning, as well as effectively leverage internal knowledge to predict future events. FutureOmni is constructed via a scalable LLM-assisted, human-in-the-loop pipeline and contains 919 videos and 1,034 multiple-choice QA pairs across 8 primary domains. Evaluations on 13 omni-modal and 7 video-only models show that current systems struggle with audio-visual future prediction, particularly in speech-heavy scenarios, with the best accuracy of 64.8% achieved by Gemini 3 Flash. To mitigate this limitation, we curate a 7K-sample instruction-tuning dataset and propose an Omni-Modal Future Forecasting (OFF) training strategy. Evaluations on FutureOmni and popular audio-visual and video-only benchmarks demonstrate that OFF enhances future forecasting and generalization. We publicly release all code (https://github.com/OpenMOSS/FutureOmni) and datasets (https://huggingface.co/datasets/OpenMOSS-Team/FutureOmni).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29649v2">LLM-Evolved Domain-Independent Heuristics for Symbolic AI Planning</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 Accepted at the LM4Plan workshop at ICAPS 2026
    </div>
    <details class="paper-abstract">
      Heuristic search is the dominant paradigm in symbolic AI planning, and the strongest heuristics are the result of decades of work by planning researchers. Recent work has shown that large language models (LLMs) can design heuristics for individual planning domains, but no LLM-generated heuristic has so far worked on arbitrary planning tasks. In this paper, we use evolutionary search to produce the first LLM-generated domain-independent heuristics that exceed the hand-engineered state of the art. We let an LLM mutate parent heuristics written in C++, store candidates in a MAP-Elites archive keyed on informedness and speed and calculate fitness scores by blending coverage with solving time. To place the evolved programs in context, we additionally benchmark a broad set of hand-engineered heuristics on their informedness-speed tradeoff, which to our knowledge has not been done before. On unseen testing domains, our best evolved heuristic solves more tasks than even the strongest baseline, with our full heuristic suite spanning the Pareto frontier of said tradeoff. We also find that seeding evolution from the trivial blind heuristic outperforms seeding from the strong FF heuristic, even when the resulting program is itself an FF variant, and that LLM reasoning effort affects how often candidates compile much more than the quality of those that do. Because the evolved programs are plain C++, they slot into existing planners as drop-in replacements and inherit the soundness and completeness guarantees of the underlying search.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.15066v4">ChatModel: Automating Reference Model Design and Verification with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-17
    </div>
    <details class="paper-abstract">
      As the complexity of integrated circuit designs continues to escalate, functional verification becomes increasingly challenging. Reference models, critical for accelerating the verification process, are themselves becoming more intricate and time-consuming to develop. Despite the promise shown by large language models (LLMs) in code programming, effectively generating complex reference models remains a significant hurdle. Therefore, we introduce ChatModel, an LLM-aided agile reference model generation and verification platform. ChatModel streamlines the transition from design specifications to fully functional reference models by integrating design standardization and hierarchical agile modeling. Employing a building-block generation strategy, it not only enhances the design capabilities of LLMs for reference models but also significantly boosts verification efficiency. We evaluated ChatModel on 300 designs of varying complexity, demonstrating substantial improvements in both efficiency and quality of reference model generation. ChatModel achieved a peak performance improvement of 58.99% compared to alternative methods, with notable enhancements in generation stability, and delivered a 9.18x increase in its capacity to produce reference model designs. Moreover, ChatModel accelerates the reference model design and validation cycles by an average of 7.11x over traditional manual approaches. These results highlight the potential of ChatModel to significantly advance the automation of reference model generation and validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.27353v2">An In-depth Study of LLM Contributions to the Bin Packing Problem</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 Accepted for publication in ACM Transactions on Evolutionary Learning and Optimization
    </div>
    <details class="paper-abstract">
      Recent studies have suggested that Large Language Models (LLMs) could provide interesting ideas contributing to mathematical discovery. This claim was motivated by reports that LLM-based genetic algorithms produced heuristics offering new insights into the online bin packing problem under uniform and Weibull distributions. In this work, we reassess this claim through a detailed analysis of the heuristics produced by LLMs, examining both their behavior and interpretability. Despite being human-readable, these heuristics remain largely opaque even to domain experts. Building on this analysis, we propose a new class of algorithms tailored to these specific bin packing instances. The derived algorithms are significantly simpler, more efficient, more interpretable, and more generalizable, suggesting that the considered instances are themselves relatively simple. We then discuss the limitations of the claim regarding LLMs' contribution to this problem, which appears to rest on the mistaken assumption that the instances had previously been studied. Our findings instead emphasize the need for rigorous validation and contextualization when assessing the scientific value of LLM-generated outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.00026v2">ActMem: Bridging the Gap Between Memory Retrieval and Reasoning in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-17
    </div>
    <details class="paper-abstract">
      Memory management is essential for LLM agents in long-term interactions. Current memory frameworks typically treat agents as passive ``recorders'' and retrieve information without understanding its deeper implications. They may fail in scenarios requiring reasoning and complex decision-making. To bridge this critical gap, we propose a novel actionable memory framework called ActMem that integrates memory retrieval with active causal reasoning. ActMem transforms unstructured dialogue history into a structured causal and semantic graph. By leveraging counterfactual reasoning and commonsense completion, it enables agents to deduce implicit constraints and resolve potential conflicts between past states and current intentions. Furthermore, we introduce a comprehensive dataset ActMemEval to evaluate agent reasoning capabilities in logic-driven scenarios, moving beyond the fact-retrieval focus of existing memory benchmarks. Experiments demonstrate that ActMem significantly outperforms baselines in handling complex, memory-dependent tasks, paving the way for more consistent and reliable intelligent assistants.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18797v1">Beyond Scalar Scores: Exploring LLM-based Metrics for Clinical Significance Evaluation in Radiology Reports</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Reliable evaluation of generated radiology reports requires strict clinical accuracy, as omitted critical findings or mischaracterized radiographic observations can directly affect patient care. Existing metrics obscure this requirement by reducing report quality to a medically ungrounded scalar. Although Large Language Models (LLMs) possess rich medical knowledge, they likewise struggle to draw a reliable boundary between clinically significant errors and harmless variation. We study this boundary using ReEvalMed benchmark as testbed and evaluate metric-level clinical significance from detecting true clinical errors ("Discrimination") and tolerating insignificant variations ("Robustness"). Across 8 LLM evaluators under one-pass and two-pass settings, we identify a widespread discrimination bias: models effectively detect errors but also over-penalize harmless rephrasings. To mitigate this, we synthesize 4k report pairs and train lightweight interpretable metrics on Qwen3-8B and MedGemma-4B. Our trained metric sharpens the clinical significance boundary, surpassing 32B-scale medical LLMs and remaining competitive with proprietary models. Crucially, the more costly two-pass setting fails to consistently improve overall performance and mainly trades discrimination for robustness. These findings suggest one-pass trained metrics as the practical choice for cost-sensitive deployment, with two-pass inference reserved for settings where D-R balance is critical. We will release the dataset and metric.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18795v1">Opinion Polarization in LLM-Based Social Networks: Manipulation and Mitigation</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 14 pages, 7 figures
    </div>
    <details class="paper-abstract">
      How vulnerable are online social networks to adversaries who seek to amplify opinion polarization by manipulating opinions, and how difficult is it to mitigate such manipulation? Existing studies have examined this question using mathematical models of opinion dynamics. While these models offer valuable theoretical insights, they rely on simplified assumptions about interactions, message content, and opinion updates, limiting the adversarial strategies they can capture and the applicability of their findings to real-world settings. Large language model (LLM)-based simulations provide a richer alternative: agents can be assigned diverse personas, communicate through natural language, and respond to persuasive or adversarial content in a context-dependent way. This enables the study of manipulation strategies that are difficult to represent using classical mathematical models. To the best of our knowledge, this study provides the first systematic analysis of polarization amplification and mitigation in an LLM-based simulated social network framework. In our framework, LLM agents with diverse personas interact over a social network by exchanging natural language posts and updating their opinions accordingly. We show that even an adversary with a limited manipulation budget can considerably increase polarization. We then study two classes of defense mechanisms: reactive mitigations, which assign specific users to actively counter manipulation, and proactive interventions, which increase resistance through general mechanisms not tied to particular users. Our results show that although these mechanisms reduce the impact of adversarial attacks, they generally do not restore the network to its baseline polarization state. These findings suggest that neither approach fully overcomes the vulnerability of the network, highlighting the potential risk of such attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18774v1">RouteJudge: An Open Platform for Reproducible and Preference-Aware LLM Routing</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 Accepted by Pluralistic Alignment Workshop at ICML 2026
    </div>
    <details class="paper-abstract">
      We present RouteJudge, an online pairwise preference evaluation framework for LLM routing systems, with a public platform available at https://routejudge.cn. Different from model-level response evaluation, RouteJudge focuses on router-level decision quality. For each user query, multiple routing strategies independently recommend candidate models under the same model pool and budget constraints. The selected model responses are then presented to users through anonymous pairwise comparisons, and the resulting user preferences are attributed back to the routing strategies behind the compared responses. Each evaluation record stores the query, routing decisions, model responses, preference labels, cost, latency, and task metadata, enabling preference-aware, cost-aware, and task-conditioned analysis of LLM routers. To support the continuous expansion of routing methods in RouteJudge, we further release ORBIT (Optimal Routing and Budgeted Inference Toolbox), a modular and extensible toolbox that standardizes the end-to-end workflow of LLM routing. ORBIT provides unified interfaces for benchmark loading, query representation, router implementation, budget-aware evaluation, and method comparison, allowing researchers to develop and evaluate routing algorithms under consistent protocols. It also serves as the submission and integration layer for RouteJudge: researchers can implement routing methods within ORBIT, validate them on existing routing benchmarks, and submit compatible routers for online preference-based evaluation. The code of ORBIT is available at https://github.com/AIGNLAI/LAMDA-ORBIT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13899v4">Do We Still Need Humans in the Loop? Comparing Human and LLM Annotation in Active Learning for Hostility Detection</a></div>
    <div class="paper-meta">
      📅 2026-06-17
    </div>
    <details class="paper-abstract">
      Instruction-tuned LLMs can annotate thousands of instances at low cost. This raises two questions for active learning (AL): can LLM labels replace human labels within the AL loop, and does AL remain necessary when entire corpora can be cheaply labeled? We investigate both on a new dataset of 277,902 German political TikTok comments (25,974 LLM-labeled, 5,000 human-annotated), comparing LLM and human annotation across seven conditions, four encoders, and 10 random seeds. Under a two-question interface that mirrors the human annotation task, LLM annotation at scale outperforms human-supervised classifiers at roughly one-tenth the cost (\$28 for GPT-5.2 Batch API vs. \$316 for Prolific). The advantage holds for both a closed-source (GPT-5.2) and an open-weight (Qwen3.5-122B-10B) LLM, is robust under soft-label evaluation, and is unlocked specifically by the two-question decomposition; a holistic single-prompt baseline only ties with human supervision. AL provides no reliable advantage over random sampling under either LLM annotator. However, error structure varies sharply: only GPT-5.2 under the two-question interface produces classifiers with near-human FP/FN balance, while other LLM variants over-flag border-control and economic competition discourse. We release the dataset and code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18747v1">Generating Natural and Expressive Robot Gestures through Iterative Reinforcement Learning with Human Feedback using LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 8 Pages, 6 Figures
    </div>
    <details class="paper-abstract">
      Expressive gestures are essential for natural and effective communication, complementing speech when verbal cues alone are insufficient (e.g., pointing). For social robots such as the humanoid Pepper, producing natural and expressive movements is critical for improving human-robot interaction (HRI) and long-term acceptance. However, generating gestures remains challenging due to reliance on expert-authored animations, resulting in rigid behaviors that are impractical for dynamic and diverse environments. Alternatively, machine learning approaches often struggle to capture perceived naturalness, becoming increasingly challenging with more degrees of freedom. Consequently, producing expressive robot gestures requires a system that can adapt to the environment while adhering to social norms and physical constraints. Recent advances in large language models (LLMs) enable dynamic code generation, offering new opportunities for runtime gesture synthesis from natural language. In this paper, we integrate ChatGPT into the humanoid robot Pepper to generate co-speech gestures aligned with conversational output. While this baseline enables flexible gesture generation, the resulting motions are often perceived as stiff and unnatural. To address this limitation, we introduce an iterative reinforcement learning with human feedback (RLHF) system that finetunes gesture generation based on user evaluations, leveraging an iterative user study to compare Pepper's generated gestures. Our results show that RLHF improved the LLM's co-speech generative capabilities, producing more expressive, relevant and fluid movements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18741v1">ReMP: Low-Downtime Runtime Model-Parallelism Reconfiguration for LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-06-17
    </div>
    <details class="paper-abstract">
      Current large language model (LLM) inference systems universally deploy ultra-large-scale models using a combination of Tensor Parallelism (TP) and Pipeline Parallelism (PP). However, existing systems treat the model parallelism topology as a static configuration that cannot be flexibly adjusted at runtime. This rigid design creates a fundamental contradiction with the dynamically changing inference workloads in real-world scenarios. State-of-the-art systems lack online reconfiguration capabilities and can only switch configurations by restarting the service, resulting in several minutes of service interruption, KV cache loss, and prohibitive recomputation overhead. To address this problem, this paper presents ReMP, a runtime model parallelism reconfiguration framework that supports low downtime. ReMP achieves dynamic adjustment through three key techniques: (1) decoupling the model parallelism topology from runtime state to avoid full service reconstruction; (2) designing a two-dimensional KV cache migration mechanism to preserve reusable cache states after TP/PP changes; and (3) implementing end-to-end online reconfiguration. Experiments demonstrate that ReMP can complete most topology switches within 1-7 seconds on models ranging from 7B to 70B parameters, achieving speedups of tens to over a hundred times compared to the restart approach. Moreover, ReMP significantly outperforms fixed configurations under dynamic workloads, delivering superior performance in terms of TTFT, TPOT, and output throughput.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12805v4">SciHorizon-GENE: Benchmarking LLM for Life Sciences Inference from Gene Knowledge to Functional Understanding</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 Accepted by SIGKDD 2026. 12 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown growing promise in biomedical research, particularly for knowledge-driven interpretation tasks. However, their ability to reliably reason from gene-level knowledge to functional understanding, a core requirement for knowledge-enhanced cell atlas interpretation, remains largely underexplored. To address this gap, we introduce SciHorizon-GENE, a large-scale gene-centric benchmark constructed from authoritative biological databases. The benchmark integrates curated knowledge for over 190K human genes and comprises more than 540K questions covering diverse gene-to-function reasoning scenarios relevant to cell type annotation, functional interpretation, and mechanism-oriented analysis. Motivated by behavioral patterns observed in preliminary examinations, SciHorizon-GENE evaluates LLMs along four biologically critical perspectives: research attention sensitivity, hallucination tendency, answer completeness, and literature influence, explicitly targeting failure modes that limit the safe adoption of LLMs in biological interpretation pipelines. We systematically evaluate a wide range of state-of-the-art general-purpose and biomedical LLMs, revealing substantial heterogeneity in gene-level reasoning capabilities and persistent challenges in generating faithful, complete, and literature-grounded functional interpretations. Our benchmark establishes a systematic foundation for analyzing LLM behavior at the gene scale and offers insights for model selection and development, with direct relevance to knowledge-enhanced biological interpretation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18709v1">LLMs Struggle to Measure What Distinguishes Students of Different Proficiency Levels: A Study of Item Discrimination in Reading Comprehension Assessment</a></div>
    <div class="paper-meta">
      📅 2026-06-17
    </div>
    <details class="paper-abstract">
      Item discrimination is a fundamental psychometric property of educational assessment, which measures whether an item meaningfully distinguishes students with higher proficiency from students with lower proficiency. While various existing works have explored whether large language models (LLMs) can estimate item difficulty, it remains unclear whether they can capture item discrimination. In this work, we evaluate 42 proprietary and open-weight LLMs in zero-shot settings using two complementary approaches: direct discrimination prediction, where models explicitly estimate an item's discrimination value from its content, and response-based Classical Test Theory (CTT) calibration, where LLM answers are treated as synthetic student responses to compute discrimination scores. Our results show that direct prediction yields weak alignment with human-calibrated discrimination: the best-performing model reaches only a Spearman correlation of 0.152. Response-based CTT calibration provides a stronger but still limited signal, with the all-persona synthetic respondent pool reaching a Spearman correlation of 0.241. These findings highlight item discrimination as an open challenge for LLM-based psychometric evaluation: current LLMs contain non-random discrimination-relevant signal, but they do not yet reliably capture how assessment items distinguish human students.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.15633v2">Formalizing and Mitigating Structural Distortion in LLM Attention for Graph Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 Accepted to KDD 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown promise for reasoning over Text-Attributed Graphs (TAGs). However, applying LLMs to graphs requires linearizing their structure into sequences, introducing distortion rooted in the graph bandwidth problem. While this distortion has been shown to degrade performance, it is often attributed to prompt design or model scale, leaving the underlying mechanism unclear. In this work, we show \textit{how} rotary positional embeddings turn graph linearization into bandwidth-dependent attention decay, suppressing attention between graph-adjacent nodes that are forced far apart in the serialized sequence. This shifts the focus of LLM-based graph reasoning from prompt engineering and scaling toward correcting attention misalignment. Motivated by this analysis, we propose \textbf{G}raph-\textbf{a}ligned \textbf{L}anguage \textbf{A}ttention (\textbf{GaLA}), a lightweight, inference-time modification for LLMs. GaLA biases attention toward graph-adjacent nodes while preserving the LLM's sequential inductive biases. Across TAG benchmarks, GaLA improves performance with negligible overhead, demonstrating that distortion is a correctable bottleneck in LLM-based graph reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.01139v3">SkillRevise: Improving LLM-Authored Agent Skills via Trace-Conditioned Skill Revision</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 15 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Agent skills are procedural artifacts that enable LLM agents to execute workflows, verify constraints, and recover from failures. Existing self-evolving methods refine skills using accumulated trajectories. However, they struggle in cold-start settings, where only an initial, imperfect skill is available. Consequently, skill construction defaults to expert authoring or one-shot LLM generation. Expert-authored skills are costly and may not align with how LLM agents actually execute tasks, while one-shot generated skills can be syntactically well formed yet behaviorally weak. To bridge this gap, we propose SkillRevise, an execution-grounded framework designed to iteratively refine these initial skills. SkillRevise diagnoses skill defects from execution evidence, retrieves relevant repair principles from a general memory, and applies execution-anchored edits. By re-executing candidates, it retains the first verifier-passing skill within the revision budget and falls back to empirical utility only when no candidate succeeds. Evaluated across three benchmarks and five LLMs, SkillRevise substantially outperforms one-shot baselines, improving the base agent's success rate on SkillsBench from 36.05% to 61.63%. Furthermore, the revised skills transfer across both executors and task environments, suggesting that SkillRevise captures reusable procedural knowledge beyond any single executor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17510v2">OmniDroneX: An LLM-Assisted Holistic Drone-as-a-Service Ecosystem</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 This manuscript is a full version of a paper accepted in shortened form by IEEE International Conference on Joint Cloud Computing
    </div>
    <details class="paper-abstract">
      Despite rapid advances in UAV technologies, current deployments remain limited due to several gaps in UAV systems research. To address these challenges, we propose OmniDroneX, a unified Drone-as-a-Service ecosystem, in which drones are transitioned from fixed function platforms into dynamically composable entities that can be integrated with external infrastructures to offer omni-capabilities. OmniDroneX bridges low-level physical primitives with high-level mission intent through a unified vendor-agnostic interface (libUAV) and a formal physical-service abstraction model (PT-SOA). A core innovation is the diverse application of large language models (LLMs) across multiple layers of the OmniDroneX architecture. LLMs are used to assist in identifying and formalizing primitive device functions and abstract service definitions, supporting automated service composition and workflow generation, and enabling interactive, natural-language mission specification and refinement. OmniDroneX also incorporates important categories of composition techniques that are essential in dynamic UAV systems, including physical layer composition for drone capability augmentation, as well as spatiotemporal, functional, collaborative, exception-aware, and QoS-based service compositions. Collectively, these features allow OmniDroneX to serve as a foundation for scalable, resilient, and self-evolving UAV ecosystems operating in complex and dynamic environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.26557v3">MemBoost: A Memory-Boosted Framework for Cost-Aware LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 ICML MemFM 2026 Workshop
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) deliver strong performance but incur high inference cost in real-world services, especially under workloads with repeated or near-duplicate queries across users and sessions. In this work, we propose MemBoost, a memory-boosted LLM serving framework that enables a lightweight model to reuse previously generated answers and retrieve relevant supporting information for cheap inference, while selectively escalating difficult or uncertain queries to a stronger model. Unlike standard retrieval-augmented generation, which primarily grounds a single response, MemBoost is designed for interactive settings by supporting answer reuse, continual memory growth, and cost-aware routing. Experiments across multiple models under simulated workloads show that MemBoost substantially reduces expensive large-model invocations and overall inference cost, while maintaining high answer quality comparable to the strong model baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18673v1">Understanding and Mitigating Prompt Leaking Attacks in Real-World LLM-Based Applications</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 Accepted at ACM CCS 2026
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based applications rely on system prompts to encode core logic and developer-defined constraints, making these prompts important intellectual property. However, system prompts are vulnerable to prompt leaking attacks. Although prior work has shown such attacks in controlled settings, their prevalence, causes, and defenses in real-world deployments remain unclear. This paper presents a systematic study of prompt leaking in real-world LLM-based applications. We measure 1,200 applications across six major commercial platforms and find that over 80% of deployments leak system prompts under realistic adversarial queries, sometimes exposing sensitive information such as third-party API keys. We also show that existing defenses often fail to prevent leakage without degrading usability. To explain these failures, we conduct an attention-level mechanistic analysis and identify attention drift, where query-key alignment bias and softmax amplification cause LLMs to progressively ignore defensive constraints. Guided by this insight, we propose AREA, a practical defense that re-anchors the model's attention using an optimizable soft prompt. Experiments and real-world case studies show that AREA matches the leakage resistance of state-of-the-art defenses while improving average usability by over 33% and reducing optimization overhead by nearly 3x. Our responsible disclosure led two affected vendors to classify these leaks as medium-severity vulnerabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13681v2">EvoArena: Tracking Memory Evolution for Robust LLM Agents in Dynamic Environments</a></div>
    <div class="paper-meta">
      📅 2026-06-17
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have achieved strong performance on a wide range of benchmarks, yet most evaluations assume static environments. In contrast, real-world deployment is inherently dynamic, requiring agents to continually align their knowledge, skills, and behavior with changing environments and updated task conditions. To address this gap, we introduce EvoArena, a benchmark suite that models environment changes as sequences of progressive updates across terminal, software, and social domains. We further propose EvoMem, a patch-based memory paradigm that records memory evolution as structured update histories, enabling agents to reason about environmental evolution through changes in their memory. Experiments show that current agents struggle on EvoArena, achieving an average accuracy of 39.6% across evolving terminal, software, and social-preference domains. EvoMem consistently improves performance, yielding an average gain of 1.5% on EvoArena and also improving standard benchmarks such as GAIA and LoCoMo by 6.1% and 4.8%. Beyond individual tasks, EvoMem further improves chain-level accuracy by 3.7% on EvoArena, where success requires completing a consecutive sequence of related evolutionary subtasks. Mechanistic analysis shows that EvoMem improves evidence capture in the memory, indicating better preservation of complete evolving environment states. Our results highlight the importance of modeling evolution in both evaluation and memory for reliable agent deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18656v1">The Wrong Kind of Right: Quantifying and Localizing Misfired Alignment in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-17
    </div>
    <details class="paper-abstract">
      Warning: This paper studies stereotypes and biases, and contains potentially disturbing examples, used for illustration purposes only. Our findings should not be interpreted as an argument against alignment. Instead, this paper highlights the need for principled approaches to more advanced alignment. Alignment aims to ensure that large language models (LLMs) behave safely and reliably, including by avoiding unsafe inferences. However, we show that such safety-oriented behaviors can misfire: models may reject warranted conclusions even when they are explicitly supported by context. We call this failure mode misfired alignment, where alignment-induced changes cause LLMs to override explicit evidence. To quantify this phenomenon, specifically on stereotype-related alignment, we introduce VETO, a benchmark consisting of 2,032 BBQ-derived contrastive pairs, and define a new metric, Misfired Alignment Rate (MAR), which measures on a 0 to 100 scale how often a model fails on a stereotype-related question but succeeds on its contrastive counterpart. We benchmark 25 LLMs on VETO, and show that all LLMs, including the most recent ones, exhibit non-trivial (4.7 to 18.9%) MARs while all human participants achieve 0.0% MAR. Controlled priming experiments further show that alignment-induced cues can substantially amplify MAR across LLMs, indicating that these failures are not merely artifacts of individual examples but can be induced by safety-related framing. Mechanistic analyses on open-weight LLMs reveal late-layer suppression of evidence-supported answers, and comparisons between instruct and base LLMs suggest that this suppression emerges after instruction training. These findings show that current alignment methods can overgeneralize surface-level safety cues, to the point of overriding objective evidence, motivating more work on alignment objectives that better preserve contextual grounding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18519v1">As You Wish: Mission Planning with Formal Verification using LLMs in Precision Agriculture</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Though robotic systems are now being commercialized and deployed in various industries, many of these systems are highly specialized and often require an advanced skill set to operate and ensure they perform as instructed. To mitigate this problem, we recently introduced a mission planner leveraging LLMs to synthesize mission plans in precision agriculture based on mission descriptions provided in natural language. While the system demonstrates impressive performance, it also suffers from the inherent ambiguities of natural language. In this paper, we extend our system to address this issue by introducing multiple feedback loops in the planning architecture that leverage linear temporal logic (LTL) to ensure the mission planning system meets the specifications formulated by the user while still using natural language. To mitigate potential bias, this is achieved by using two different commercial LLMs in charge of the specification and verification subtasks. Through extensive experiments, we highlight the strengths and limitations of integrating mission verification into a fully autonomous pipeline, particularly regarding an LLM's ability to generate valuable LTL formulas, and show how our proposed implementation addresses and solves these challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.09905v2">The Personalization Trap: How User Memory Alters Emotional Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-16
      | 💬 19 pages 5 figures
    </div>
    <details class="paper-abstract">
      When an AI assistant remembers that Sarah is a single mother working two jobs, does it interpret her stress differently than if she were a wealthy executive? As personalized AI systems increasingly incorporate long-term user memory, understanding how this memory shapes emotional reasoning is critical. We investigate how user memory affects emotional intelligence in large language models (LLMs) by evaluating 15 models on human-validated emotional intelligence tests. We find that identical scenarios paired with different user profiles produce systematically divergent emotional interpretations. Across validated user-independent emotional scenarios and diverse user profiles, systematic biases emerged in several high-performing LLMs where advantaged profiles received more accurate emotional interpretations. Moreover, LLMs demonstrate significant disparities across demographic factors in emotion reasoning and supportive recommendations tasks, indicating that personalization mechanisms can embed social hierarchies into models' emotional reasoning. These results highlight a key challenge for memory-enhanced AI: systems designed for personalization may reinforce social inequalities. To mitigate these disparities, we curate a general-purpose preference dataset designed to reduce demographic profiles' influence on emotional understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18453v1">LLM Parameters for Math Across Languages: Shared or Separate?</a></div>
    <div class="paper-meta">
      📅 2026-06-16
      | 💬 5 pages. Accepted at ACL Student Research Workshop (SRW) 2026. Code: https://github.com/luisavictor/math-across-languages Translated Datasets: https://huggingface.co/math-across-languages Webpage: https://math-across-languages.github.io
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit substantial cross-lingual variation in mathematical reasoning performance, but it remains unclear whether these differences reflect language-specific parameters or a shared mechanism that manifests differently by language. We present a cross-lingual mechanistic analysis of mathematical reasoning in LLMs, enabling us to localize and compare model parameters that support mathematical reasoning across languages. We find that the extracted math-associated parameters exhibit partial cross-lingual overlap, with the strongest overlap concentrated in intermediate model layers. We further observe that English consistently produces the largest set of math-relevant parameters, whereas lower-resource languages reveal smaller sets of relevant parameters. These results suggest that math-related behavior in multilingual LLMs is neither fully language-invariant nor fully language-specific, but instead exhibits partial cross-lingual parameter overlap with systematic language-dependent differences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18431v1">Beyond Prediction: Tail-Aware Scheduling for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      LLM serving exhibits extreme length variability, making size-based scheduling difficult in practice. Recent LLM schedulers approximate SJF/SRPT using predicted decode lengths or ranks and primarily report mean-centric metrics such as TTFT and TBT. We show that these prediction-driven policies can be fragile under distribution shifts, bursty arrivals, and GPU memory pressure, while offering limited control over the tail latency (P90-P99) that dominates user experience, even with perfect decode-length knowledge. We introduce a distribution-aware, prediction-free scheduling framework that replaces explicit length prediction with soft priority boosting driven by lightweight statistical signals. Our design co-optimizes scheduling and cache-aware preemption to account for memory-coupled decode dynamics across workload mixes. Evaluated on production and open-source traces, our method reduces P99 TTLT by up to 35-50% relative to SRPT with perfect length knowledge and reduces TTFT by 34-47% across workloads, including reasoning-heavy and chat-heavy tasks. These results demonstrate a robust alternative for optimizing tail latency in online LLM serving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18405v1">Evaluating the Effectiveness of LLMs in Aiding Compliance Testing of PKCS#1-v1.5</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Testing implementations of binary protocols for specification compliance requires inputs that satisfy both structural and semantic constraints. Purely random generation and primitive mutations are often insufficient for exploring semantically meaningful behaviors in protocols that rely on Type-Length-Value (TLV) encoding, yet domain-specific compliance testing tools require deep protocol expertise and significant manual effort to construct. This work investigates whether grammar-level mutation combined with LLM-based code synthesis can serve as a viable, more generalizable approach to specification compliance testing. We evaluate the approach on PKCS#1 v1.5 signature verification -- a widely deployed TLV-encoded standard with a formally verified testing oracle (Morpheus) -- across 48 cryptographic library implementations. We reproduced 10 of 13 non-trivial specification violation categories previously identified by Morpheus, including all 5 signature forgery categories, and discovered 1 previously unreported discrepancy. We found that LLM hallucination -- occurring in 82.5% of generated scripts -- is the primary factor limiting effectiveness, not the mutation strategies. We identify five distinct hallucination types and show that their distribution varies systematically across mutation categories: structural mutations are implemented with 13.3% fidelity while constraint mutations achieve 30.3% correctness but suffer the highest rate of mutations being fully ignored (8.1%). These findings reveal a striking gap between operational reliability (99.8%) and semantic fidelity (17.5%), providing actionable guidance on when LLM-based code synthesis can be trusted in specification-driven testing pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18388v1">LLMZero: Discovering Adaptive Training Strategies for RL Post-Training via LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      RL post-training strategies are dataset-dependent and reveal a recurring empirical pattern: capacity parameters accumulate monotonically across stages, while regularization parameters predominantly oscillate in response to shifting training dynamics. This distinction matters because fixed schedules commit all parameters to fixed trajectories and therefore cannot express the non-stationary exploration-exploitation tradeoffs that regularization must track; the principle provides actionable design rules for multi-stage training. We discover this through LLMZero, a system where LLM agents search over training trajectories via tree search, diagnosing pathologies at each checkpoint and proposing coordinated multi-parameter transitions. Across 4 diverse GRPO tasks, LLMZero discovers strategies that improve over the base model by 9% to 140% relative and over grid search by 6% to 15% relative, consistently outperforming random search and the skill-based agent. The structural principle transfers across tasks, providing an explanation for why discovered strategies take qualitatively different forms yet share similar parameter dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.25065v2">Vulcan: Instance-specialized, Verifiable Systems Heuristics Through LLM-driven Search</a></div>
    <div class="paper-meta">
      📅 2026-06-16
      | 💬 19 pages
    </div>
    <details class="paper-abstract">
      Systems resource management tasks rely primarily on hand-designed heuristics. However, growing hardware heterogeneity and workload diversity require heuristics specialized to particular deployment instances, making manual design expensive and difficult to scale. In this paper, we explore how to synthesize systems heuristics using LLMs. The main challenge is ensuring that generated heuristics execute safely, integrate correctly with the surrounding system, and still achieve strong performance. We propose Vulcan, a framework that identifies LLM-friendly interfaces that isolate core decision logic from the rest of the implementation. With Vulcan, LLM-generated code is restricted to simple stateless decision functions, while trusted runtime abstractions provide rich derived statistics for meaningful policy exploration without system-integration bugs. To ensure execution safety, LLMs synthesize heuristics in a restricted language, Anvil, that guarantees important properties by construction. We evaluate Vulcan across three well-studied domains and demonstrate up to 4.9x higher savings for spot-VM scheduling, up to 2x lower miss ratios for cache eviction, and up to 10% higher application performance for tiered-memory systems, while ensuring execution safety throughout.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18166v1">Evaluating Open-Source LLMs for Multi-Label ATT&CK Technique Classification on CTI Reports</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Classifying Cyber Threat Intelligence (CTI) using MITRE Adversarial Tactics, Techniques, and Common Knowledge (ATT&CK) is essential for proactive defense, but historically required extensive human effort. Pre-Large Language Model (LLM) automation sped up this process, but could not resolve the complex language and multi-step attack patterns found in unstructured CTI reports. LLMs addressed previous limitations by using contextual reasoning to understand unstructured text. However, current evaluations rely on simplified, single-technique sentences that ignore the complexity of real-world CTI reports, which often leads to inflated performance results. Consequently, the baseline performance of open-source LLMs on complex unstructured CTI reports remains unevaluated. To address this gap, we constructed a ground-truth dataset of 2,076 human-annotated sentences (1,281 technique-positive, 795 negative) from 83 complex unstructured CTI reports. These sentences were mapped to 114 unique ATT&CK techniques using a six-phase annotation process, achieving \k{appa} = 0.68 inter-annotator agreement. Using this dataset, we evaluated seven open-source LLMs ranging from 8B to 236B parameters across prompt strategy and temperature configurations. The highest-performing LLM achieved a micro-averaged F1 score of 0.22, establishing the empirical baseline for multi-label ATT&CK classification on complex unstructured CTI. Parameter size showed a statistically significant positive correlation with F1 score. Prompt strategy and temperature produced no statistically significant gains across model configurations. These results indicate that current open-source LLMs are insufficient for production-grade ATT&CK classification. The dataset, benchmark, and findings provide a reproducible foundation for future CTI research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.00826v3">LLM-Powered Multi-Agent System for Automated Crypto Portfolio Management</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Cryptocurrency portfolio management requires the fusion of heterogeneous multi-modal signals, including structured price and on-chain time series, unstructured news text, and technical indicators, under high-volatility and real-time constraints. While deep learning approaches show predictive capability, their opacity limits practical adoption, and single large language model (LLM) agents struggle to process the breadth of modality-specific inputs needed for robust decision-making. We propose a multi-agent system (MAS) framework in which three modality-specialised agents, a Crypto Agent for market dynamics, a News Agent for weekly news sentiment, and a Trading Agent for signal fusion and portfolio execution, decompose the task across three communication architectures: hierarchical, collaborative, and debate. We evaluate four capability configurations: zero-shot, chain-of-thought (CoT), retrieval-augmented generation (RAG), and skill-augmented. In a 52-week backtest over calendar year 2025 across the top 15 L1 blockchain native cryptocurrencies by market capitalisation as of January 2025, the best configuration, Hierarchical (Skill), achieves a cumulative return of 133.52% and a Sharpe ratio of 1.502, outperforming single-agent variants, passive benchmarks, and deep learning baselines. An ablation study identifies the Crypto Agent as the most critical component, with its removal reducing cumulative return by 42.57 percentage points. A cross-model comparison further shows that MAS outperforms the single-agent baseline under GPT-4o, GPT-5, and Claude Sonnet 4.5, suggesting that the benefit of multi-agent coordination is model-agnostic. Unlike black-box deep learning models, every portfolio decision is traceable to explicit agent reasoning, offering an interpretable and effective approach to multi-modal cryptocurrency portfolio management.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18129v1">Towards Understanding and Measuring COGNITIVE ATROPHY in LLM Behaviour</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Recent incidents involving LLMs used for mental-health support reveal a critical evaluation gap: surface-level safety scores do not capture how models behave across realistic, emotionally sensitive interactions over time. Existing benchmarks measure knowledge, safety, or static response quality, but miss whether LLM interactions help users keep reflecting, coping, and making decisions themselves. We formalize this missing dimension as COGNITIVE ATROPHY, a process-level behavioural measure in AI-mediated mental-health support distinct from safety and helpfulness. To measure it, we introduce COGNITIVE ATROPHY BENCH, a clinically grounded benchmark built from 1,576 fully human-generated counseling conversations, 15,680 turns, and 42,230 responses from five LLMs. Three clinical and neuropsychology experts developed a 20-attribute schema spanning user context, response behaviour, and global risk flags; six trained clinical reviewers applied it with span-grounded evidence, producing 5,324 reviewer judgments. We further introduce the User-Input Risk Index (UIRI), the Cognitive Atrophy Risk Index (ARI), and trajectory summaries. Across five LLMs, models show a consistent moderate-to-high level of atrophy-aligned behaviour across single and multi-turn settings. While models generally respond to overt safety cues, they adapt less reliably when users seek solutions or decisions. The dominant recurring patterns are directive advice, problem-solving, recommendation responses, topic shifts, and forms of validation that may reinforce dependence rather than reflection. Our work makes COGNITIVE ATROPHY measurable and provides a foundation for auditing model behaviour in sensitive LLM conversations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18063v1">When LLMs Analyze Scars: From Images to Clinically-Meaningful Features</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Medical image classification faces a fundamental dilemma: while deep learning models achieve remarkable performance at scale, real-world clinical scenarios often suffer from severe data scarcity due to annotation costs, privacy constraints, and disease rarity. This challenge is particularly pronounced in pathological scar classification, where differentiating keloids from hypertrophic scars requires subtle expert knowledge and labeled images are extremely limited. We propose a novel paradigm that repositions large language models (LLMs) as knowledge-driven feature engineers rather than end-to-end classifiers. We call this framework ScaFE (Scar Feature Engineering). Our key insight is that LLMs encode rich medical knowledge that can be externalized as executable feature extraction code, enabling the transformation of high-dimensional images into low-dimensional, clinically interpretable representations. Specifically, we prompt an LLM with established scar assessment criteria to generate deterministic Python code that extracts features aligned with clinical scoring systems such as the Vancouver Scar Scale. Our approach offers three key advantages: (1) data efficiency, achieving robust performance with limited training samples by decoupling knowledge acquisition from statistical learning; (2) privacy preservation, as raw images are processed locally without exposure to external LLMs; and (3) interpretability, through explicit features grounded in clinical reasoning. Extensive experiments on scar classification demonstrate that our method consistently outperforms end-to-end deep learning baselines or using LLMs as black-box classifiers under limited data conditions, establishing a promising direction for integrating LLMs into data-efficient and clinically transparent medical AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18062v1">Security and Privacy Prompts in the Wild: What Users Ask LLMs and How LLMs Respond</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are widely used to fulfill users' information needs; users ask LLMs about the weather, pose educational questions, and consult them for legal assistance. One particularly understudied area is digital security and privacy (S&P), where users may seek LLMs' help on how to secure their online accounts or protect their computers from cyber attacks. To the best of our knowledge, no prior study has collected or analyzed the S&P questions users ask LLMs; prior research on LLM response quality relied on expert-authored S&P misconceptions or FAQs rather than user queries. Drawing from WildChat, a dataset of 3.2M user-LLM conversations collected in the wild, our study identifies 14,727 S&P prompts and categorizes them into nine categories covering a wide range of S&P topics. From the S&P prompts, we sampled 450 and performed a thematic analysis to characterize the S&P questions users ask LLMs. Separate from the thematic analysis, we curated 270 advice-seeking S&P prompts, where users ask for recommendations, guidance, or specific S&P information. We measured LLM response quality and consistency when posing the prompt to LLMs 10 times. We found that commercial LLMs outperform open-weight models (GPT 5.5 provided "good enough" responses on 98% of prompts; Llama 4 on 47%). However, among prompts that received high-quality responses on average, commercial models sometimes produce contradictory responses across runs, risking confusing or misleading users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18051v1">Compositional Skill Routing for LLM Agents: Decompose, Retrieve, and Compose</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      LLM agents increasingly rely on external skills -- reusable tool specifications -- but real-world tasks often require composing multiple skills, not just selecting one. We formalize this as the Compositional Skill Routing problem: given a complex user query and a large skill library, decompose the query into atomic sub-tasks, retrieve the appropriate skill for each sub-task, and compose an executable plan. We present SkillWeaver, a decompose-retrieve-compose framework combining an LLM task decomposer, a bi-encoder skill retriever with FAISS indexing, and a dependency-aware DAG planner. To support evaluation, we introduce CompSkillBench, a benchmark of 300 compositional queries over 2,209 real MCP server skills spanning 24 functional categories, sourced from the public MCP ecosystem. Our experiments reveal that task decomposition quality is the primary bottleneck: standard LLM decomposition reaches only 34.2% category recall at the step level. To address this, we propose Iterative Skill-Aware Decomposition (SAD), a retrieval-augmented feedback loop that iteratively aligns decomposition with available skills. SAD improves decomposition accuracy from 51.0% to 67.7% (+32.7%, Wilcoxon p < 10^-6) in a single iteration; DA-conditioned analysis confirms that correct granularity is the prerequisite for effective retrieval (CatR@1 rises from 34% to 41% when DA=1). SkillWeaver reduces context window consumption by over 99%, and transfer experiments confirm generalization (+35.6% relative DA gain even when target categories are absent from the retrieval pool).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.23243v3">Are Frontier LLMs Ready for Cybersecurity? Evidence for Vertical Foundation Models from Dual-Mode Vulnerability Benchmarks</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      We evaluate whether frontier LLMs are ready for cybersecurity through a dual-mode benchmark: white-box function-level vulnerability detection (VulnLLM-R, across C/Java/Python) and black-box web application security testing (five production-style applications with 118 ground-truth vulnerabilities across 20+ CWE families, which we will open-source). We test six frontier models (GPT-5.4, Codex~5.3, Claude Opus~4.7, Sonnet~4.6, Gemini~3.1~Pro and Gemini~3~Flash) and two domain-specialized models across four testing paradigms. Our findings are sobering: (1)~every frontier model produces 10-50% false positive rates in white-box detection, systematically over-predicting vulnerabilities; (2)~in black-box testing, frontier models achieve only 4-8% ground-truth coverage, improving to just 10-19% even with external security tools (Playwright MCP, Burp Suite MCP); (3)~structured penetration-testing methodology encoded in domain-specialized agents raises per-family detection above 50%, demonstrating that methodology, not scale, is the primary lever; and (4)~a domain-specialized defense model achieves the highest precision (0.904) and lowest false positive rate (9.7%) among all models, on a single GPU. We identify the absence of structured security testing traces end-to-end request/response sequences, failure-heavy data, and multi-step attack chains as the fundamental training data bottleneck, and propose self-play security testing as a data generation strategy. Our results make the case for vertical foundation models purpose-built for cybersecurity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18037v1">ProvenanceGuard: Source-Aware Factuality Verification for MCP-Based LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-16
      | 💬 20 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Tool-using LLM agents increasingly use the Model Context Protocol (MCP) to answer from heterogeneous evidence sources, including search, APIs, databases, clinical records, and formulary tools. Standard factuality metrics usually test whether an answer is supported by pooled evidence, missing a provenance-sensitive failure mode: a claim may be supported somewhere while being attributed to the wrong source. We call this cross-source conflation. We introduce ProvenanceGuard, a source-aware verifier for MCP-grounded answers. It consumes captured MCP traces with stable tool IDs, source IDs, and raw outputs; decomposes answers into atomic claims; routes claims to source-specific evidence; checks support with NLI and a token-alignment proxy; compares stated attribution with the routed source; and returns per-claim verdicts plus an answer-level allow/block decision. Blocked answers can be repaired with retrieval-augmented answer revision and re-verified. We evaluate on 281 medical-domain MCP-agent traces. A 266-trace adjudicated subset yields 2,325 LLM-assisted claim labels split by trace; 361 held-out labels are human-verified. On the 40-trace held-out split, ProvenanceGuard achieves block F1 0.802 and source accuracy 0.858 over 260 source-eligible claims, outperforming source-blind baselines that do not emit claim-to-source IDs. On a harder multi-source benchmark it reaches block F1 0.846, while source-plus-relation accuracy drops to 0.229, showing that exact source ownership remains difficult with semantically close sources. Repair-and-reverify resolves all blocked answers in the full trace set, often via conservative fallback. In 50 controlled clinical conflation probes, ProvenanceGuard detects all injected attribution swaps with no retained wrong attribution. These results show that source attribution is an independent axis for factuality verification in MCP-based agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18030v1">ParaTutor: LLM Mediated Parent Child Tutoring through Role Separated Scaffolding Interface in Real Time</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Parent child tutoring is a collaborative learning setting with asymmetric roles, where parents guide children s problem solving while children engage in understanding and reasoning. However, most LLM based learning systems are designed for either single users or symmetric collaboration, leaving parent child tutoring with distinct instructional roles underexplored. Through a formative study, we find that effective parent child tutoring depends on preserving these distinct roles, with parents guiding the learning process and children remaining actively engaged in reasoning. We also identify recurring challenges when parents struggle to understand problem structure, lack sufficient knowledge to provide support, or encounter communication difficulties that disrupt shared understanding. To address these challenges, we present ParaTutor, a scaffolding system that provides different forms of support to parents and children. ParaTutor supports parents with guidance for tutoring and provides children with visual grounding for problem solving. We evaluate ParaTutor with 23 parent child dyads (children aged 10 to 12) under four tutoring conditions that vary how LLM assistance is delivered. Results show that generic LLM assistance tends to reduce the parent s role in tutoring, whereas ParaTutor better preserves parent led support and sustains children s participation in reasoning. These findings suggest that in multi users learning, the value of LLM support depends not only on model capability but also on how support is distributed across users with different roles. Our work contributes design implications for LLM systems that support family learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18005v1">LLM Consumer Behavior Theory: Foundations of a Novel Research Field</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed as autonomous agents that make consumption decisions on behalf of users. This shift raises fundamental questions for consumer theory, which has traditionally modeled humans as the primary decision-makers. In this paper, we introduce LLM Consumer Behavior Theory, a new field of study concerned with analyzing consumer behavior in agentic markets. Drawing on classical and behavioral economics alongside recent advances in Natural Language Processing, we formalize how human preferences are reflected and acted upon by LLM-based agents, and how agent-level decisions aggregate into market demand. We unify previously fragmented literature on LLM decision-making, human behavior simulation, and preference elicitation under a common economic lens, highlighting where assumptions, such as rationality and heterogeneity, may fail in agentic markets. Rather than providing empirical validation, this paper outlines the scope of LLM consumer behavior and identifies open research questions related to alignment, preference representation, and market dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17986v1">ShellGames: Speculative LLM-Driven SSH Deception</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Cyber deception and Moving Target Defense are promising strategies that aim to disrupt adversaries by increasing uncertainty. However, sustaining long-lived, credible interactive sessions with adversaries remains an open challenge. Large Language Models (LLMs) offer a promising path toward more dynamic deception systems, but suffer from key limitations that fundamentally limit their applicability, including: lack of persistent state, output inconsistencies, hallucinations, latency, and susceptibility to behavioral subversion that may reveal the deception. We propose ShellGames, an SSH shell simulator based on LLM designed to address these limitations. ShellGames combines five complementary techniques: (i) Automatic Chain-of-Thought and few-shot learning to improve correctness; (ii) memory management to maintain system state coherency; (iii) speculative command execution to reduce response latency; (iv) smart routing of complex interactive commands to a sandboxed environment; and (v) subversion detection leveraging the constrained input-output domain of shell environments. To enable systematic evaluation, we introduce a standardized benchmarking protocol and dataset spanning correctness, consistency, state tracking, and robustness tasks. ShellGames achieves $0.898$ command accuracy on correctness ($+5.3pp$ over baselines), $0.918$ sequence-level accuracy on consistency ($+36pp$), $0.98$ state tracking accuracy ($+18.3pp$), and $0.95$ accuracy on robustness ($+37pp$). A user study with $n=20$ participants confirms that ShellGames achieves realism comparable to a real shell under free exploration and outperforms traditional honeypots on perceived command coverage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17952v1">SoftMoE: Soft Differentiable Routing for Mixture-of-Experts in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-16
      | 💬 Accepted at ICML 2026
    </div>
    <details class="paper-abstract">
      Sparse Mixture-of-Experts (MoE) architectures enable scaling LLM parameters under a fixed inference budget by activating only a small subset of experts via top-$k$ routing. While this preserves causality and suits autoregressive language models, the discrete top-$k$ operator is not differentiable, forcing a fixed number of active experts per input and resulting in inefficient use of computation. We propose SoftMoE, which replaces discrete routing with a truncated soft top-$k$ LapSum relaxation, allowing gradient-based optimization of expert routing. We further parameterize the mean number of active experts per layer and impose a global budget constraint, enabling the model to learn how to allocate expert capacity across layers. SoftMoE remains fully compatible with autoregressive modeling and achieves performance comparable to or better than sparse MoE on language modeling and downstream tasks, while activating significantly fewer experts. Notably, the learned allocation is highly non-uniform, with later layers activating more experts. The source code is publicly available$^\dagger$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17949v1">RouteBalance: Fused Model Routing and Load Balancing for Heterogeneous LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-06-16
      | 💬 12 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Heterogeneous LLM serving stacks split scheduling into two layers that optimize in isolation: model routers pick a model from quality and cost signals while ignoring instance load, and serving load balancers optimize queues while ignoring quality. We present RouteBalance, a serving-aware scheduling layer that fuses both into a single online assignment over concrete model instances, jointly trading off quality, latency, and cost. A batched in-process predictor stack and dead-reckoned instance state keep the joint decision cheap on the request hot path ($\approx$32 ms at 12 req/s). On a 13-instance, 28-GPU heterogeneous cluster serving four model sizes, a single deployed RouteBalance stack traces the upper region of the three-way quality-cost-throughput frontier. Sweeping one weight vector reaches both the highest routing-decision quality (DeepEval $0.419$, $+0.013$ over the strongest baseline, $95\%$ CI $[{+}0.005,{+}0.022]$; the ordering holds when a second judge re-scores the actually served text) and, at its cost-priority corner, per-request cost that ties the cheapest baseline. With router engineering equalized against concurrent-scoring baseline variants we build, its balanced preset serves at $2.8$ s and $30$ req/s, leading $2.6$ to $4.1\times$ ahead of enhanced BEST-Route at high load. (Deploying those routers as published, one serial scoring call per request, makes them collapse $23\times$ under load, a deployment-architecture effect we isolate separately, not the routing result.) A four-arm isolation shows the benefit follows from pricing latency at model-selection time; the learned predictors contribute calibration and SLO headroom rather than the headline frontier. Code: https://github.com/AKafakA/route-balance
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06116v6">The Homogenization Problem in LLMs: Towards Meaningful Diversity in AI Safety</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Generative AI models reproduce the human biases in their training data and further amplify them through mechanisms such as mode collapse. The loss of diversity produces homogenization, which not only harms the minoritized but impoverishes everyone. We argue homogenization should be a central concern in AI safety. To meaningfully characterize homogenization in Large Language Models (LLMs), we introduce a framework that allows stakeholders to encode their context and value system. We illustrate our approach with an experiment that surfaces gender bias in an LLM (Claude 3.5 Haiku) on an open-ended story prompt. Building from queer theory, we formalize homogenization in terms of normativity. Borrowing language from feminist theory, we introduce the concept of xeno-reproduction as a class of tasks for mitigating homogenization by promoting diversity. Our work opens a collaborative line of research that seeks to understand and advance diversity in AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.16337v2">Medical Heuristic Learning: An LLM-Driven Framework for Interpretable and Auditable Clinical Decision Rules</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Predictive modeling for clinical tabular data is central to clinical decision support and therefore requires not only strong predictive performance but also transparent decision logic. Although deep learning and tree-based ensemble methods can achieve high accuracy, their black-box nature remains a major obstacle to clinical deployment. This challenge is further compounded by common characteristics of medical data, including limited sample sizes, severe class imbalance, and feature evolution arising from changes in diagnostic criteria and clinical documentation. To address these issues, we propose Medical Heuristic Learning (MHL), an instantiation of the learning-beyond-gradients paradigm for clinical tabular prediction. Instead of relying on neural network weight updates, MHL uses a large language model (LLM)-driven workflow that integrates statistical probes, medical knowledge probes, rule synthesis, and code-level iterative refinement to optimize a deterministic and executable decision system. The resulting model is expressed not as opaque parameters, but as versioned pure-Python decision rules that are explicitly interpretable, fully auditable, and clinically grounded. MHL also supports continual learning by starting from previously validated rules and iteratively revising them using updated feature information under data drift or feature evolution. Comprehensive experiments on medical datasets show that MHL achieves performance comparable to state-of-the-art methods while maintaining strong behavior in small-sample and highly imbalanced settings. The results further indicate that this explicit rule update mechanism can help alleviate catastrophic forgetting under feature evolution. Overall, these findings suggest that non-gradient-based heuristic systems offer a transparent and adaptable alternative for high-stakes clinical decision support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17915v1">Trustworthy Self-Composable Big-Data-as-a-Service: An LLM-Orchestrated Multi-Agent Framework for Automated Data Engineering, AutoML, MLOps Deployment, and Drift-Aware Lifecycle Optimization</a></div>
    <div class="paper-meta">
      📅 2026-06-16
      | 💬 7 pages, 3 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Big-Data-as-a-Service (BDaaS) platforms require re liable automation across data ingestion, cleaning, feature engi neering, model development, deployment, and post-deployment monitoring. However, existing LLM-based data science agents and AutoML systems mainly focus on isolated workflow stages, leaving limited support for lifecycle-level orchestration, artifact governance, human oversight, and drift-aware adaptation. This paper proposes a trustworthy self-composable BDaaS frame work based on LLM-orchestrated multi-agent collaboration. The proposed architecture decomposes the BDaaS lifecycle into specialized agents for data ingestion, data cleaning, feature engineering, AutoML training, model evaluation, MLOps de ployment, monitoring, and drift detection. A central LLM or chestration layer coordinates agent execution, validates interme diate outputs, manages workflow context, and enables dynamic workflow composition. The framework also incorporates shared artifact governance, reproducibility support, human-in-the-loop checkpoints, and drift-aware feedback loops. A prototype-based evaluation is conducted using controlled tabular benchmark datasets with missing values, categorical variables, outliers, class imbalance, and simulated covariate drift. Compared with manual ML, AutoML-only, and single-agent LLM baselines, the pro posed multi-agent BDaaS pipeline achieves competitive predictive performance while improving lifecycle-level reliability, including workflow completion, artifact traceability, deployment readiness, reproducibility, and drift recovery. The results suggest that LLM-orchestrated multi-agent systems can extend conventional AutoML toward trustworthy, adaptive, and production-oriented BDaaS lifecycle automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.10139v2">TACOMORE: Exploring a replicable prompting protocol for LLM-assisted corpus analysis</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      As corpus linguistics continues to scale, researchers are facing a growing methodological bottleneck: while computational tools can easily count billions of words, the qualitative interpretation of these data remains a slow and labor-intensive human task. Large Language Models (LLMs) offer a promising way to automate this process, yet their integration into the field is often hindered by concerns over black-box unpredictability and a lack of replicability. This study introduces TACOMORE, a structured prompting framework designed to transform ad-hoc AI interactions into a standardized linguistic protocol. Built upon four foundational principles (Task, Context, Model, and Replicability), the framework guides LLMs to move beyond generic probability prediction to anchoring their reasoning in the specific co-occurrence patterns of a target corpus. We applied this framework to three core corpus tasks, i.e., the analysis of keywords, collocates, and concordances, using an open corpus of COVID-19 research abstracts. After testing three LLMs, we found that while structured prompting improves accuracy and replicability, inherent limitations regarding hallucination persist. This research offers a critical lens into the role of LLMs in corpus linguistics, highlighting their potential as complementary tools while emphasizing the irreplaceable role of human validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.02721v2">Blueprint First, Model Second: A Framework for Deterministic LLM Workflow</a></div>
    <div class="paper-meta">
      📅 2026-06-16
      | 💬 12 pages, 7 figures, 6 tables
    </div>
    <details class="paper-abstract">
      While powerful, the inherent non-determinism of large language model (LLM) agents limits their application in structured operational environments where procedural fidelity and predictable execution are strict requirements. This limitation stems from current architectures that conflate probabilistic, high-level planning with low-level action execution within a single generative process. To address this, we introduce the \textsc{Source Code Agent} framework, a new paradigm built on the ``Blueprint First, Model Second'' philosophy that decouples workflow logic from the generative model. An expert-defined operational procedure is first codified into a source code-based Execution Blueprint, which is then executed by a deterministic engine. The LLM is strategically invoked as a specialized tool to handle bounded, complex sub-tasks within the workflow, but never to decide the workflow's path. We evaluate on the TravelPlanner benchmark for constraint-aware travel planning. The \textsc{Source Code Agent} achieves a 35.56\% final pass rate, a 97.6\% improvement over the state-of-the-art ATLAS baseline (18.00\%) on the same Claude-Sonnet-4 backbone. Critically, it reduces constraint violations by 96.0\% (11 vs 275) while improving execution efficiency by 27.1\% (10.2$\pm$0.7 steps vs 14.0). Two production incident-diagnosis deployments and additional results on ScienceWorld and ALFWorld confirm that the architecture transfers beyond travel planning to procedurally well-defined, constraint-intensive workflows. Our work enables the verifiable and reliable deployment of autonomous agents in applications governed by strict procedural logic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30036v2">Teaching Values to Machines: Simulating Human-Like Behavior in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-16
      | 💬 We had some disagreement regarding proper attribution; we hope to resolve it soon and upload the paper
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate a remarkable capacity to adopt different personas and roles; however, it remains unclear whether they can manifest behavior that adheres to a coherent, human-like value structure. In this work, we draw on established psychological value theory to induce human-like values in LLMs and assess their alignment with patterns observed in human studies. Using validated psychological questionnaires, we conduct large-scale experiments -- over 5 million questions -- to evaluate value structures and value-behavior relationships in leading LLMs and compare them to humans. Our findings reveal strong agreement between value-prompted LLMs and humans across both dimensions. Moreover, incorporating human value distributions enhances population-level simulations with value-induced LLMs. These findings highlight the potential of value-induced LLMs as effective, psychologically grounded tools for simulating human behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17838v1">Environment-Grounded Automated Prompt Optimization for LLM Game Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      LLM agents in interactive environments are highly sensitive to their prompts, yet prompt engineering remains a manual, task-specific process. We introduce an automated prompt optimization framework for LLM agents that decomposes the observation-to-action pipeline into a goal-conditioned descriptor agent and an action selection agent, and iteratively refines each module's prompt through an LLM-driven evolutionary loop guided by environment returns. We propose a behavior analyzer to attribute episode outcomes to specific prompt components, and a mutator to propose targeted revisions to the prompt, before validating them through environment rollouts. We evaluate on all five BabyAI tasks in the BALROG benchmark, comparing our pipeline against BALROG's RobustCoTAgent under both plain and guided prompt initializations. Optimization improves performance consistently across tasks and conditions, without requiring updates to the model weights. On PutNext, a multi-step coordination task where the RobustCoTAgent achieves 0% success, our framework reaches up to 72.5% success rate using the same underlying LLM with optimized prompts. These results suggest that a multi-agent framework, combined with automatic prompt optimization, enhances LLMs without the need for fine-tuning or extensive human supervision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.09004v2">LATTEArena: An Evaluation Framework for LLM-powered Tabular Feature Engineering (Extended Version)</a></div>
    <div class="paper-meta">
      📅 2026-06-16
      | 💬 31 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Feature engineering remains a cornerstone of tabular data analysis, and Large Language Models (LLMs) have emerged as a promising paradigm for its automation, giving rise to LLM-powered Automated Tabular Feature Engineering (LATTE). However, the field lacks standardized, cost-aware evaluation platforms, and the combinatorial explosion of design choices obscures true algorithmic progress. To bridge these gaps, we systematically deconstruct 15 representative LATTE methods into a unified 6-dimensional taxonomy. Based on this abstraction, we introduce LATTEArena, a standardized, modular, and extensible benchmarking framework that decouples monolithic pipelines into reusable execution blocks. By distilling the massive combinatorial space, we evaluate 24 core LATTE configurations across 7 research questions. Our head-to-head benchmarking goes beyond predictive accuracy to quantify token efficiency and execution robustness, yielding 17 empirical findings on cost-effectiveness trade-offs. Furthermore, we provide 3 concrete recommendations for optimal real-world deployment. By enabling controlled component-level comparisons, LATTEArena shifts the paradigm from ad-hoc prompt engineering to systematic context management. All code, datasets, and over 4,000 execution logs are publicly available to foster a dynamic, community-driven benchmark. Our framework, leaderboard, and all artifacts are hosted on the LATTEArena project website at https://goodenhak.github.io/LATTEArena.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17787v1">LUMEN: Coordinated Failure Recovery for Distributed LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Modern large language model (LLM) serving clusters distribute inference requests across multiple worker processes on different GPUs, but failures are prevalent at scale. When a worker fails, the cluster simultaneously loses the failed worker's GPU-resident key-value (KV) caches and serving capacity, leaving surviving workers to absorb the redirected traffic while re-running interrupted requests from scratch. Existing fault-tolerant systems either restart interrupted requests from scratch or restore KV caches from checkpoints stored on a fixed neighboring worker, but both approaches route recovery work without considering current cluster load and leave the recovering worker idle during model reload. We present LUMEN, a fault-tolerant LLM serving system that treats recovery as a load-aware coordination problem across three decision points: checkpoint placement before failures, interrupted-request distribution at failure time, and serving capacity restoration during model reload. We evaluate LUMEN using both prototype experiments and large-scale simulations and demonstrate significant improvements in serving and recovery times.
    </details>
</div>
