# llm - 2026_06

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.24970v1">Don't Go Breaking My LLM: The Impact of Pruning Attention Layers on Explanation Faithfulness and Confidence Calibration</a></div>
    <div class="paper-meta">
      📅 2026-06-23
      | 💬 Accepted at TMLR
    </div>
    <details class="paper-abstract">
      Pruning Large Language Models (LLMs) reduces memory and inference costs by removing parts of the network, producing smaller models that retain most of their accuracy. As attention layers are the most resource-intensive parts of LLMs, pruning them is a promising compression strategy. Prior work shows that up to 33% of attention layers can be pruned with minimal accuracy loss. Nevertheless, the impact of attention pruning on model interpretability, specifically faithfulness and confidence calibration, remains unstudied. To address this gap, we study how pruning attention layers affects explanation faithfulness and confidence calibration across five LLMs and eight datasets. While the pruned models often maintain high accuracy, we find that their faithfulness and calibration often degrade. Notably, faithfulness and calibration can fluctuate significantly, even when accuracy remains stable, highlighting a misalignment between model confidence, interpretability, and accuracy. Our findings suggest that layer pruning can affect LLMs' interpretability and reliability in ways not captured by accuracy and efficiency measures alone. We recommend including explainability and calibration metrics when evaluating pruned models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.24420v1">Beyond Logprobs: A Multi-Signal Confidence Engine for LLM-Based Document Field Extraction</a></div>
    <div class="paper-meta">
      📅 2026-06-23
      | 💬 Extended version of a paper accepted (Oral) at the RobustifAI Workshop, IJCAI-ECAI 2026, Bremen, Germany. 9 pages, 5 figures, 2 tables
    </div>
    <details class="paper-abstract">
      In high-stakes document processing pipelines, including financial reconciliation, compliance verification, and procurement automation, an LLM extraction that is silently wrong is more dangerous than one that is visibly absent. The central challenge is not extraction accuracy alone but reliable confidence estimation: knowing, field by field, whether an extraction can be trusted for automation or deferred to human review. Token-level log-probabilities, verbalized confidence, and multi-sample self-consistency all collapse toward all-positive behaviour at practical thresholds, offering no reliable separation between trustworthy and untrustworthy extractions. We present ExtractConf, a cross-domain, field-agnostic confidence engine that grounds confidence estimation in two structurally different readings of the same document. A field-guided Hunter call extracts each field under schema-slot completion pressure; a document-guided Mapper call scans holistically and surfaces values grounded in document content. This asymmetry yields different failure modes: Hunter hallucinates values for absent fields, while Mapper misses visually non-salient ones. Their disagreement is independently informative. ExtractConf fuses cross-call disagreement, LLM-internal uncertainty, OCR, image quality, and spatial layout into a classifier requiring no domain-specific rules or retraining. On DocILE (55-field invoices, 26% failure rate), it achieves 0.928 ROC AUC and reduces selective prediction risk by 70% over logprob-mean. At 80% coverage, accuracy reaches 99.1%, enabling a practical human-in-the-loop workflow. Zero-shot transfer to CORD receipts achieves 0.858 AUC; lightweight Lasso recalibration reduces ECE by 89% and Brier by 43%, confirming the signals generalise across document domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.24622v3">Random Rule Forest (RRF): Interpretable and Manageable Ensembles of LLM-Generated Questions for Predicting Success from Unstructured Data</a></div>
    <div class="paper-meta">
      📅 2026-06-23
      | 💬 25 pages including appendix, 6 figures
    </div>
    <details class="paper-abstract">
      Many high-stakes screening tasks require predicting rare outcomes from unstructured text, where errors are costly and decisions must be auditable. We introduce Random Rule Forest (RRF), an interpretable ensemble that uses a large language model (LLM) not as an end-to-end predictor but as a generator of simple YES/NO questions. Each question acts as a weak learner, and their responses are combined by a plain unit-weight vote into an auditable ``green-flags'' scorecard: enough independent positive signals indicate a higher chance of success. We argue this deliberate simplicity is a robust default when positives are scarce and learned weights are hard to estimate. We evaluate RRF in two low-base-rate domains. On early-stage startup screening from founder profiles, RRF produces a transparent scorecard whose precision is several times the base rate (with light expert input raising it further) and, unlike direct prompting, its operating point can be controlled directly. On an established Phase~I clinical-trial benchmark, RRF outperforms published baselines on the threshold-independent metrics PR-AUC and ROC-AUC. Together these show that LLMs can serve as auditable feature generators for high-stakes text-based decisions, combining transparency with competitive predictive performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.20450v2">Policies Permitting LLM Use for Polishing Peer Reviews Are Currently Not Enforceable</a></div>
    <div class="paper-meta">
      📅 2026-06-23
      | 💬 ICML 2026
    </div>
    <details class="paper-abstract">
      A number of scientific conferences and journals have recently enacted policies that prohibit LLM usage by peer reviewers, except for polishing, paraphrasing, and grammar correction of otherwise human-written reviews. But, are these policies enforceable? To answer this question, we assemble a dataset of peer reviews simulating multiple levels of human-AI collaboration, and evaluate five state-of-the-art detectors, including two commercial systems. Our analysis shows that all detectors misclassify a non-trivial fraction of LLM-polished reviews as AI-generated, thereby risking false accusations of academic misconduct. We further investigate whether peer-review-specific signals, including access to the paper manuscript and the constrained domain of scientific writing, can be leveraged to improve detection. While incorporating such signals yields measurable gains in some settings, we identify limitations in each approach and find that none meets the accuracy standards required for identifying AI use in peer reviews. Importantly, our results suggest that recent public estimates of AI use in peer reviews through the use of AI-text detectors should be interpreted with caution, as current detectors misclassify mixed reviews (collaborative human-AI outputs) as fully AI generated, potentially overstating the extent of policy violations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.24391v1">Age of LLM: A Strategic 1v1 Benchmark for Reasoning, Diplomacy and Reliability of Large Language Models under Fog of War</a></div>
    <div class="paper-meta">
      📅 2026-06-23
      | 💬 25 pages including appendices, 8 figures, 4 tables; appendices include verbatim system prompt and engine resolution pseudocode. All correlations reported with p-values, 95% bootstrap confidence intervals and Spearman's rho; includes a Steiger test and Bradley-Terry fit
    </div>
    <details class="paper-abstract">
      We introduce Age of LLM, a turn-based 1v1 benchmark in which two LLMs face off on a 13x7 grid to destroy the enemy base. Three stressors are deliberate: fog of war, full diplomacy (messages, ceasefires, ultimatums; uranium kept secret), and a reliability dimension where every turn must follow a strict JSON schema and an illegal action is silently discarded. The engine is private and each match uses a fresh random map seed and opponent, mitigating the data contamination that affects public benchmarks. Models receive a (near) rule-only prompt with no build-order advice (two tactical seed phrases were present during data collection; see Section 2.7). We benchmark 15 reasoning models across 54 matches and 5,258 actions. Findings: (1) the nuclear rush dominates (78% on the rules-coherent v0.11+ sub-corpus; 85% corpus-wide) with a sole-launcher signature that is largely mechanical under secret-simultaneous launch rules, not a cognitive deterrence failure; (2) military conquest is rare but faster (12.3 vs 18.9 turns); (3) diplomacy is prolific yet almost never consummated; (4) ~58% of illegal actions are fog/state errors, making the illegal-action rate a measure of belief-tracking; (5) -- the least established, and the only one we label exploratory -- a weak link associates reliability with winning. The corpus is small, unbalanced and not side-swapped, so the ranking is a preliminary descriptive view, not a contribution. Beyond ranking, the turn-by-turn traces of actions and messages make the corpus a lens on how LLMs reason under adversarial uncertainty -- their belief-tracking, spontaneous deception, and per-model cognitive "personas" -- which we frame as a future research direction. We release the replay format, an isometric viewer and all replays; engine source on request.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.24370v1">When Helpfulness Overrides Causal Caution: Context-Dependent Suppression and Recovery in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-23
      | 💬 43 pages, 3 figures, 5 tables. SSRN Abstract ID: 6965680
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly integrated into decision-support roles in business and policy contexts. While prior benchmark studies have primarily evaluated LLMs' causal reasoning capabilities, a more fundamental epistemic dimension has been overlooked: Causal Caution, defined as the propensity to refrain from causal judgment when empirical evidence is insufficient. This study examines the systematic suppression of Causal Caution that occurs when LLMs shift from academic to practical advisory contexts. Using an evaluation rubric inspired by Pearl's Causal Hierarchy (the PCH score), we conducted experiments on four high-performance LLMs -- Claude Sonnet 4.6, Claude Opus 4.7, GPT 5.5, and Gemini 3.1 Pro -- across 480 trials. Causal Caution maintenance rates were 91.7--100.0% in academic contexts but dropped to 6.7--18.3% in practical advisory contexts (Fisher's exact test, p < .001 across all models). Furthermore, when restricted to practical prompts requesting concrete recommendations or explanatory rationales, only 1 of 200 responses (0.5%) maintained Causal Caution. A brief self-correction prompt -- "Please reconsider this judgment from the perspective of causal relationships" -- restored the expression of Causal Caution to maintenance rates of 71.4--100.0% (McNemar's test, p < .001 across all models). These results suggest that helpfulness-oriented response patterns may suppress the expression of Causal Caution in practical advisory contexts, with important implications for organizational governance. The findings indicate that this suppression reflects context-dependent variation in expression rather than an underlying capability limitation, suggesting that multi-agent architectures that separate proposal generation from causal auditing may offer a promising governance design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.24964v1">Evidence for feature-specific error correction in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-23
      | 💬 13 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Understanding the features of large language models (LLMs) is a central goal of interpretability. LLMs are commonly assumed to use superposition to represent more features than they have dimensions. They may not only represent features in superposition but also perform computation in superposition. Theory predicts that computing in superposition requires error correction that privileges feature directions over generic ones, but this prediction has not been tested empirically. We propose an empirical test of error correction in LLMs based on activation perturbations. Perturbing residual-stream activations, we find that they are robust to small perturbations--forming activation plateaus consistent with error correction--but less robust along candidate feature directions ("pure" directions, constructed from contrastive prompt pairs) than along mixtures of two such directions, indicating that the pure directions are privileged. We quantify this privilegedness by modeling the perturbation effect as a function of the $L^p$-norm of its decomposition into feature components. For $p=2$ the response is a quadratic form with at most as many nonzero eigenvalues as the residual-stream dimension, which cannot privilege the many feature directions superposition requires. $p>2$ lifts this constraint and is consistent with feature-specific error correction. We find $p>2$ for contrastive, MELBO, and SAE-decoder directions, and $p\approx2$ for random and PCA directions (controls). These results replicate across Gemma-2-9B, Qwen3-1.7B, Llama-3.1-8B, Mistral-7B-v0.3, Aya-Expanse-8B, and Yi-1.5-9B. We further validate our method on a toy model of error correction with known ground-truth features, recovering $p>2$ for true feature directions, degrading toward $2$ as we rotate away from them.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.24322v1">Securing LLM-Agent Long-Term Memory Against Poisoning: Non-Malleable, Origin-Bound Authority with Machine-Checked Guarantees</a></div>
    <div class="paper-meta">
      📅 2026-06-23
      | 💬 Index Terms: LLM agents, long-term memory, memory poisoning, information-flow control, capability security, prompt injection, formal verification, benchmark
    </div>
    <details class="paper-abstract">
      LLM agents increasingly rely on persistent long-term memory, which creates a critical vulnerability that we study here: memory poisoning. An adversary can store untrusted content in one session that later steers a consequential action, such as a payment, a setting change, or data exfiltration, in a future session. Existing defenses base a memory item's authority to act on either its content (detection or trust-scoring) or its derivation history (lineage). We show that both signals are malleable. An attacker can launder an untrusted origin through three channels specific to LLM agents: the agent's own summarization, a trusted-tool echo, and manufactured corroboration. Each makes the content look benign and breaks or flips its derivation edge to ``trusted.'' We formalize malleability for the memory write-retrieve-act pipeline and prove a machine-checked separation theorem. No content- or lineage-based defense is sound under laundering (T1), write-time origin binding is necessary (T2), and non-malleable origin-bound authority with Sybil-resistant corroboration-gated elevation is sufficient (T3). Our construction, TMA-NM (Tamper-evident Memory Authority, Non-Malleable), instantiates non-malleable information-flow control (IFC) for LLM-agent memory. A cross-defense, cross-attack, and cross-model benchmark over eight frontier models shows that existing defenses fail exactly where the theory predicts (up to 68% laundering attack-success), while TMA-NM reaches 0% attack success on both direct and laundering attacks across all models and channels, at full legitimate utility. We release the benchmark, harness, and machine-checked TLA+ models to support reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23238v2">HOLMES: Evaluating Higher-Order Logical Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-23
    </div>
    <details class="paper-abstract">
      Logical reasoning is essential for reliable AI, yet existing benchmarks are largely first-order-logic-centric, focusing on object-level deduction over fixed predicates. This misses many realistic scenarios where models must reason over rules, predicates, functions, constraints, and decision procedures themselves. We introduce HOLMES (Higher-Order Logic Meets real-world Explainable Symbolic reasoning), the first real-world benchmark for higher-order symbolic reasoning in LLMs, containing 1379 instances. Built on higher-order logic, HOLMES pairs natural-language problems with HOL formalizations, ground-truth answers, verifiable reasoning traces, and fine-grained controllable reasoning factors across law and finance. Experiments show that current LLMs still struggle on HOLMES, with an average accuracy of only 50.64% and the best model reaching 59.54%. Our analyses further reveal that high final-answer accuracy can mask shortcut reasoning in conflict-resolution settings, while performance drops sharply under scope-conditioned and compositional reasoning. These findings identify higher-order symbolic reasoning as a key bottleneck for building reliable and verifiable LLMs. The project code and dataset are publicly available at https://github.com/wuyucheng2002/HOLMES.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21835v2">Collaborative Lossless LLM Inference Serving with Offloading-based Pipeline Parallelism on Edge Devices</a></div>
    <div class="paper-meta">
      📅 2026-06-23
      | 💬 12 pages, 16 figures
    </div>
    <details class="paper-abstract">
      Providing lossless inference services of LLMs on edge devices remains challenging, especially given the extremely tight memory budgets. The existing offloading techniques inevitably introduce numerous loading bubbles, which further inflate the end-to-end latency of the entire inference pipeline. Meanwhile, dynamically fluctuating network bandwidth and diverse user request patterns pose additional obstacles to efficient lossless inference on edge devices. To address this, we propose LOIP, a collaborative lossless LLM inference system that employs an offloading-based interleaved pipeline parallelism to better overlap model offloading with computing and communicating. Specifically, LOIP first constructs an offloading-aware cost model to characterize inference latency and memory overhead under heterogeneous device capabilities and limited bandwidth. Based on this cost model, LOIP develops a fine-grained allocation scheduler that determines latency-efficient layer partitions across devices while explicitly accounting for offloading overhead, along with a unified memory architecture (UMA)-aware loading optimization using customized CUDA operators to reduce runtime loading overhead. LOIP further designs an online memory adaptation strategy to handle the increasing KV cache pressure and dynamic bandwidth fluctuations during inference. We implement LOIP with 2500+ lines of Python and 500+ lines of C++/CUDA code, and deploy it on five heterogeneous NVIDIA Jetson edge devices for lossless collaborative inference of LLaMA3.3-70B-Instruct. Extensive experiments demonstrate that LOIP achieves 8.8$\times$$\sim$20.3$\times$ speedups over the SOTA baselines under different bandwidth conditions and request patterns without compromising model accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.24286v1">AVOC: Enhancing Hour-Level Audio-Video Understanding in Omni-Modal LLMs via Retrieval-Inspired Token Compression</a></div>
    <div class="paper-meta">
      📅 2026-06-23
    </div>
    <details class="paper-abstract">
      Multimodal Large Language Models have achieved remarkable progress in short-form audio-video understanding, yet long-form audio-video comprehension remains challenged by limited context windows and severe information redundancy. To address these bottlenecks, we propose AVOC, a framework for long-form audio-video understanding in Omni-modal Large Language Models. AVOC introduces a learnable token compression module between the modality encoders and the LLM backbone. We reframe multimodal token compression as a top-$K$ retrieval problem: given a fixed context budget, the module must retrieve a compact subset of tokens that best supports answering the user query. We draw inspiration from three classical Information Retrieval criteria for selecting informative units from a large candidate pool: relevance, importance, and diversity. AVOC instantiates each criterion as a tailored mechanism for audio-video understanding, and integrates them into a unified retrieval-style compression pipeline. Experiments show that AVOC achieves state-of-the-art performance on long-form audio-video benchmarks, surpassing the second-best model by 4.9 and 5.5 points in average accuracy on OmniVideoBench and LVOmniBench, respectively. Moreover, AVOC maintains robust performance on Audio-Video Needle-in-a-Haystack task at durations up to one hour.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.20750v2">Subjective-Graph LLM Agents for Simulating Uncertainty in Classroom Social Perception</a></div>
    <div class="paper-meta">
      📅 2026-06-23
    </div>
    <details class="paper-abstract">
      Social actors do not observe a common social world: each individual forms judgments from a partial and potentially distorted view of the surrounding network. We study whether graph-local evidence and credibility-weighted communication can generate persistent distortions in perceived academic standing, even when agents repeatedly receive objective performance signals. We introduce a data-constrained multi-agent framework in which LLM agents operate through individualized subjective graphs that determine peer visibility, evidence access, and interaction opportunities. Agents exchange uncertainty-annotated assessments, evaluate message credibility, and maintain explicit Gaussian belief states updated through Bayesian fusion. We evaluate the framework on 12 middle-school classrooms comprising 482 students, using questionnaire-derived social information and six consecutive examinations. On the Social-Observed subset (n=419), collective ranking error increases from 0.066 \pm 0.008 to 0.124 \pm 0.009 across six epochs despite repeated exam-based anchoring. Ablations associate individualized visibility and LLM-based trust gating with more stable long-horizon behavior, while constrained retrieval primarily safeguards against global-information leakage. Compared with evaluated DeGroot configurations, the proposed framework achieves lower final ranking error; those DeGroot configurations exhibit near-zero terminal opinion diversity. These findings establish subjective-graph LLM agents as a mechanism-oriented framework for data-constrained simulated social perception. Code is available at https://anonymous.4open.science/r/Rashomonomon-0126.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23032v2">IPO Finance Agent: Evaluation of LLM Financial Analysts beyond Finance Agent v2, with Automated Rubric Generation -- the Case of the SpaceX (SPCX) IPO</a></div>
    <div class="paper-meta">
      📅 2026-06-23
    </div>
    <details class="paper-abstract">
      Finance Agent v2 (by Vals AI) has emerged as the reference benchmark for evaluating both Anthropic Claude and OpenAI ChatGPT frontier language models on financial tasks. However, it narrowly deals with periodic reporting from publicly traded companies (SEC 10-K and 10-Q filings), and its agentic harness relies on naive, unenriched chunk retrieval. Neither the task design nor the retrieval approach addresses the distinct challenges of IPO due diligence. SEC S-1 filings combine historical financial statements, governance structures, pro forma and common-control accounting treatments, capital-formation narratives, and underwriting-sensitive risk disclosures within substantially longer documents than typical periodic filings. That is why we introduce IPO Finance Agent, which extends the Finance Agent v2 framework along two directions: task domain and retrieval architecture. During our experiments, the original Finance Agent v2 harness basically failed to deliver any output related to the SpaceX S-1 filing, due to document length. We therefore had to improve the agentic harness with contextual retrieval, a more realistic and industry-standard approach for long documents. We also built a dataset of 1,000 IPO-diligence questions, and publicly release 70 questions on the SpaceX (SPCX) S-1 filing to support reproducibility, while the remainder are held private to guard against benchmark contamination. In addition, we introduce an evaluator-optimizer pipeline to automatically generate evaluation rubrics for the benchmark: candidate facts are extracted from model answers, consolidated into draft criteria, then automatically audited for omissions, hallucinations, mistiered items, and redundancy, with LLM feedback driving iterative repair, targeted enrichment, and deduplication. Human experts only review final rubrics before deployment. Results show that the best-performing evaluated model, Alibaba Qwen 3.7 Max, reaches 79.4% accuracy at 0.30 USD per query, and the most cost-efficient model on the resulting Pareto frontier, Xiaomi MiMo-2.5 Pro, reaches slightly lower accuracy (76.8%) at 0.05 USD per query. Both exceed the current Finance Agent v2 leaderboard ceiling-Google Gemini 3.5 Flash at 57.9% for 2.51 USD per querywhile undercutting even FABv2's cheapest entry (MiniMax M3: 48.3% at 0.32 USD) on cost-efficiency. Code and data are released on GitHub: https://github.com/benstaf/ipoagent
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.24163v1">CORE-BREW: LLR-Based Soft Decoding for Robust Multi-Bit LLM Watermarking</a></div>
    <div class="paper-meta">
      📅 2026-06-23
    </div>
    <details class="paper-abstract">
      Reliable provenance for LLM outputs requires multi-bit watermarks that remain robust under editing while maintaining strict false-positive control. Existing ECC-based LLM watermarks rely largely on hard-decision decoding, discarding token-level reliability information. We propose CORE-BREW, a Constant-hit-Rate Embedding extension of block-wise BREW for robust multi-bit watermarking. CORE-BREW calibrates the watermark channel by targeting a fixed hit rate p-star, yielding closed-form per-token log-likelihood ratios (LLRs) for principled soft-decision decoding. It supports two detection modes: Strict-Safe, which preserves the bounded-distance designated-codeword acceptance region, and FPR-Calibrated, which uses likelihood-based scoring and lightweight list decoding to characterize the FPR-TPR trade-off. Experiments on open-source LLMs under token-level edits and paraphrasing demonstrate improved low-FPR discrimination and robustness over prior multi-bit watermarking baselines while maintaining comparable semantic quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.04767v2">ParallelBench: Understanding the Trade-offs of Parallel Decoding in Diffusion LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-23
      | 💬 Accepted at ICLR 2026. Project Page: https://parallelbench.github.io
    </div>
    <details class="paper-abstract">
      While most autoregressive LLMs are constrained to one-by-one decoding, diffusion LLMs (dLLMs) have attracted growing interest for their potential to dramatically accelerate inference through parallel decoding. Despite this promise, the conditional independence assumption in dLLMs causes parallel decoding to ignore token dependencies, inevitably degrading generation quality when these dependencies are strong. However, existing works largely overlook these inherent challenges, and evaluations on standard benchmarks (e.g., math and coding) are not sufficient to capture the quality degradation caused by parallel decoding. To address this gap, we first provide an information-theoretic analysis of parallel decoding. We then conduct case studies on analytically tractable synthetic list operations from both data distribution and decoding strategy perspectives, offering quantitative insights that highlight the fundamental limitations of parallel decoding. Building on these insights, we propose ParallelBench, the first benchmark specifically designed for dLLMs, featuring realistic tasks that are trivial for humans and autoregressive LLMs yet exceptionally challenging for dLLMs under parallel decoding. Using ParallelBench, we systematically analyze both dLLMs and autoregressive LLMs, revealing that: (i) dLLMs under parallel decoding can suffer dramatic quality degradation in real-world scenarios, and (ii) current parallel decoding strategies struggle to adapt their degree of parallelism based on task difficulty, thus failing to achieve meaningful speedup without compromising quality. Our findings underscore the pressing need for innovative decoding methods that can overcome the current speed-quality trade-off. We release our benchmark to help accelerate the development of truly efficient dLLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13189v2">SICI: A Semantic-Pragmatic Complexity Index Reveals Regime Shifts in LLM Stance Detection</a></div>
    <div class="paper-meta">
      📅 2026-06-23
    </div>
    <details class="paper-abstract">
      Prompt-based LLMs are increasingly used for stance detection, but harder examples are not always repaired by clearer instructions, reasoning prompts, retrieval, or debate. We introduce SICI (Stance Inference Complexity Index), a seven-dimensional diagnostic measure of the semantic-pragmatic burden imposed by a target--text pair. Across SemEval-2016 and VAST, SICI predicts LLM accuracy better than surface proxies and shows substantial cross-scorer reliability ($α=0.771$). More importantly, LLM errors change regime as SICI increases: low-complexity examples invite over-attribution, especially Against predictions; intermediate examples form an unstable boundary; and high-complexity examples rapidly concentrate on None. This phase-transition-like structure persists across GPT-3.5, GPT-4o-mini, DeepSeek-V3, and GPT-4o, although stronger models move the boundaries. A 15-method intervention study further shows that prompting, retrieval, and debate often shift models along the attribution--abstention axis rather than removing the high-complexity bottleneck.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.24145v1">T2D-Bench: Evidence-Gated Evaluation of LLM Outputs for Type 2 Diabetes Using a Multi-Layer Clinical-Lifestyle Knowledge Graph</a></div>
    <div class="paper-meta">
      📅 2026-06-23
      | 💬 7 pages, 2 figures, 2 tables. Accepted as a poster at AMIA 2026 Annual Symposium
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can produce clinically fluent recommendations for type 2 diabetes while failing to satisfy guideline constraints or explicitly justify lifestyle-related glycemic claims. We present T2D-Bench, a reproducible benchmark and evidence-gated evaluation framework for testing whether LLM outputs satisfy explicit, graph-checkable evidence requirements. T2D-Bench is built on a multi-layer clinical-lifestyle knowledge graph that combines a biomedical spine (UMLS, DrugBank, SIDER), computable ADA Standards of Care rules, and lifestyle knowledge connected through a mechanistic bridge to glycemic laboratory effects. Across 100 structured vignettes spanning diagnosis, medication safety, and adversarial lifestyle conflicts, baseline outputs failed benchmark-defined evidence-path checks in 35% of cases for GPT-4o-mini and 33% for GPT-4o. The evidence gate detects unsupported omissions and uses constrained revision to bring outputs into verifier-level compliance with benchmark-defined evidence requirements. These results show that computable evidence constraints can make unsupported clinical omissions explicit, measurable, and correctable in diabetes-focused LLM outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.24133v1">Holistic Data Scheduler for LLM Pre-training via Multi-Objective Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-06-23
      | 💬 Our code is at https://github.com/DANG-ai/LLM-Training-Holistic-Data-Schedule
    </div>
    <details class="paper-abstract">
      The composition of training data, governed by the diversity of sources and their mixing strategy, is a cornerstone of Large Language Model (LLM) pre-training. Online Data Mixing (ODM), the technique of adaptively adjusting data mixtures during training, has emerged as a promising direction to improve efficiency. However, existing methods are constrained by their reliance on a singular optimization perspective, which fundamentally overlooks the need for complex LLM pre-training to consider the dynamic data composition from multiple dimensions. To overcome this limitation, we introduce the Holistic Data Scheduler (HDS), a novel online data mixing framework. HDS formulates the data scheduling challenge as a reinforcement learning problem in a continuous control space and leverages the Soft Actor-Critic (SAC) algorithm for its stability and sample efficiency in exploring the high-dimensional policy space. At the core of HDS lies a novel multi-objective, holistic reward function that integrates three critical perspectives: a data-driven reward for quality, a loss-driven reward capturing inter-domain influence, and a model-driven reward based on weight norms. To validate our design and determine its optimal configuration, we conducted systematic experiments on LLMs of various sizes. On The Pile benchmark, HDS reaches the final validation perplexity of the next best method with 44% fewer training iterations. Furthermore, it achieves a 7.2% improvement on the MMLU 0-shot task along with consistent gains on other benchmarks, showcasing its ability to enhance both training efficiency and final model capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12212v3">Mind your key: An Empirical Study of LLM API Credential Leakage in iOS Apps</a></div>
    <div class="paper-meta">
      📅 2026-06-23
      | 💬 12 pages, 4 figures, 4 tables
    </div>
    <details class="paper-abstract">
      The rapid integration of large language models (LLMs) into mobile applications has introduced a new class of credential security risk: leaked credentials that grant unauthorized access to LLM inference services, causing financial damage to developers. Prior work on credential leakage has focused primarily on Android apps; to date, no empirical study has systematically investigated LLM API key leakage in iOS applications. We present the first in-depth empirical study of API key leakage in LLM-integrated apps. We construct a high-quality dataset of 444 iOS applications, filtered from 1092 candidates through a standardized process, and develop LLMKeyLens, a dynamic analysis framework that detects LLM API key leakage via traffic interception, provider-specific key extraction, and active validity confirmation, requiring neither source code access nor binary decryption. Our analysis reveals that 282 applications expose exploitable LLM API credentials in network traffic, spanning at least ten providers. We identify three leakage patterns: JWT-based token leakage (48%), unauthenticated backend proxy access (33%), and plaintext API key transmission (19%). To assess remediation, we re-analyzed the same 282 vulnerable applications three months after responsible disclosure; only 28% had remediated the reported vulnerability, while 72% remained exploitable, with persistent issues stemming from unauthenticated backends and broken JWT implementations. Our findings show that LLM API key leakage is both prevalent and persistent in the iOS ecosystem, exposing a systemic gap between developer practice and secure integration principles, and suggest that secure LLM integration requires not only developer awareness but also explicit security guidance from providers and platform-level enforcement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.24094v1">Universal Guideline-Driven Image Clustering via a Hybrid LLM Agent</a></div>
    <div class="paper-meta">
      📅 2026-06-23
      | 💬 CVPR 2026
    </div>
    <details class="paper-abstract">
      Unifying image clustering across different clustering scenarios remains challenging due to fundamental gaps among tasks. We introduce a Guideline-Driven Image Clustering Agent, the first universal framework that bridges these gaps through textual guidelines. To incorporate complex guidelines without task-specific training, we propose Generative Concept Proxy Modeling, which generates guideline-aware embeddings via concept proxy extraction. For scenarios requiring automatic cluster discovery, we introduce LLM Traversal based on Minimum Spanning Tree that selectively applies LLM reasoning for complex semantic judgments. Our method generalizes across diverse clustering scenarios spanning from general to fine-grained categorization, from global to local criteria, and from balanced to long-tail distributions. Our framework consistently outperforms specialized methods across diverse clustering tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.21349v2">LLM-assisted Generation of Pseudo-C2 Servers for IoT Malware Dynamic Analysis</a></div>
    <div class="paper-meta">
      📅 2026-06-23
    </div>
    <details class="paper-abstract">
      Most IoT malware operates as botnets dependent on Command and Control (C2) servers, but the short-lived nature of attack infrastructure often leaves samples dormant without C2 communication, hindering dynamic analysis. This paper proposes a system that combines Ghidra with a Large Language Model (LLM) to extract communication specifications from a malware binary and automatically generate a pseudo-C2 server. Experiments using Mirai demonstrate that the proposed system semantically interprets binary control structures and extracts all 20 core protocol elements in agreement with the ground truth (100\% specification extraction accuracy). The generated pseudo-C2 server fully reproduces seven of ten DDoS attack vectors with attack behavior consistent with the original C2. When applied to a customized variant created by modifying the publicly available Mirai source code, the method succeeds end-to-end -- from specification extraction through pseudo-C2 generation to attack reproduction -- demonstrating that the LLM infers specifications from binary structures without relying on pre-trained knowledge. This approach extends the applicability of LLMs from analysis assistance to the automated construction of dynamic analysis environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20892v4">Representation Interventions Enable Lifelong Knowledge Memory Control in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-23
      | 💬 In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics: ACL 2026, Oral Presentation
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often produce incorrect or outdated content after being employed. Efficient and accurate knowledge updates without costly retraining are a major challenge. This problem is particularly challenging in lifelong settings, where complex, unstructured knowledge must coexist without interference. We introduce RILKE (Representation Intervention for Lifelong KnowledgE Control), a robust and scalable method that treats knowledge control as interventions within the model's representation space. Leveraging representation-space expressiveness, we identify two key properties enabling RILKE to achieve fine-grained control over complex, unstructured knowledge while maintaining general utility with frozen base weights. During training, RILKE learns paraphrase-robust and edit-localized modules that limit each update to a low-dimensional subspace to minimize cross-edit interference. At inference, a query-adaptive router selects the appropriate module to guide the model's generation. Across LLaMA and Qwen models, RILKE scales effectively to large-scale benchmarks, demonstrating high edit success and strong paraphrase generalization while preserving general utility with modest memory overhead. These results show RILKE is an effective and scalable solution for lifelong knowledge control in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.25850v3">Debate2Create: Robot Co-design via Multi-Agent LLM Debate</a></div>
    <div class="paper-meta">
      📅 2026-06-23
    </div>
    <details class="paper-abstract">
      We introduce Debate2Create (D2C), a multi-agent LLM framework that formulates robot co-design as structured, iterative debate grounded in physics-based evaluation. A design agent and control agent engage in a thesis-antithesis-synthesis loop, while criterion-specific LLM judges provide multi-objective feedback to steer exploration. Across five MuJoCo locomotion benchmarks, D2C achieves the highest default-normalized score among the evaluated LLM-based and black-box baselines, with gains up to 3.2x on Ant and nearly 9x on Swimmer. Iterative debate yields 18-35% gains over compute-matched zero-shot generation, and D2C-generated rewards transfer to default morphologies in 4/5 tasks. These results suggest that structured, simulator-grounded multi-agent interaction is a useful mechanism for joint morphology-reward optimization under a fixed-topology, per-candidate-RL protocol. Project page: debate2create.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.24022v1">Do Language Models Pass the Bechdel Test? Auditing Gender Biases in LLM-Generated Screenplays</a></div>
    <div class="paper-meta">
      📅 2026-06-23
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly used in media production from journalistm to filmmaking, what impact do they have on the stories being told? Prior work has shown LLMs to perpetuate social biases, including those related to gender. We complement existing literature on gender bias in LLM outputs by auditing the network structure of LLM-generated movie screenplays through automating the Bechdel test, a popular measure of women's representation in literary and film works. We also introduce the use of social network analysis measures to further analyze representational bias in LLM-generated scripts. We evaluate screenplays generated by three state-of-the-art LLMs (GPT-5, Gemini 3 Pro, and Claude Sonnet 4.5) against 768 corresponding human-written screenplays, finding that human-written scripts are more likely to pass the Bechdel test. However, other network analyses, like centrality, homophily, and triadic relationships demonstrate that in some cases LLM-scripts have less bias, although all script types demonstrate some representational bias under most measures. We conclude by discussing the continued need for further quantitative assessments of media representations and AI-generated content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23992v1">RASC+: Retrieval-Constrained LLM Adjudication for Clinical Value Set Authoring</a></div>
    <div class="paper-meta">
      📅 2026-06-22
    </div>
    <details class="paper-abstract">
      Clinical value sets define the standardized terminology codes used in quality measurement, phenotyping, cohort construction, and clinical decision support. The recently introduced Retrieval-Augmented Set Completion (RASC) benchmark showed that direct zero-shot large language model (LLM) generation is poorly suited to this task: clinical code systems are large, version-controlled, and not reliably memorized by language models. We study a stage-wise alternative in which candidate-pool construction is optimized for recall and a constrained LLM adjudicator is optimized for candidate selection. On the full 3,744-value-set RASC test split, Qwen3-based retrieval with vocabulary-aware expansion and code-display rescue retrieval increases candidate-pool recall from the original RASC retrieval baseline of 0.553 to 0.730; on the held-out-publisher stratum, pool recall is 0.655. The higher-recall pool alone is not sufficient: applying the original SAPBert cross-encoder to this expanded pool gives full-test macro F1 of 0.287 and held-out-publisher macro F1 of 0.233. Replacing the stage-2 selector with blinded GPT-5 adjudication over the same pool increases full-test macro F1 to 0.549 and held-out-publisher macro F1 to 0.533. These results show that retrieval-constrained LLM adjudication can substantially improve value set completion while preserving the safety constraint that all returned codes must come from an auditable candidate pool.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.13673v2">LLM-MINE: Large Language Model based Alzheimer's Disease and Related Dementias Phenotypes Mining from Clinical Notes</a></div>
    <div class="paper-meta">
      📅 2026-06-22
    </div>
    <details class="paper-abstract">
      Accurate extraction of Alzheimer's Disease and Related Dementias (ADRD) phenotypes from electronic health records (EHR) is critical for early-stage detection and disease staging. However, this information is usually embedded in unstructured textual data rather than tabular data, making it difficult to be extracted accurately. We therefore propose LLM-MINE, a Large Language Model-based phenotype mining framework for automatic extraction of ADRD phenotypes from clinical notes. Using two expert-defined phenotype lists, we evaluate the extracted phenotypes by examining their statistical significance across cohorts and their utility for unsupervised disease staging. Chi-square analyses confirm statistically significant phenotype differences across cohorts, with memory impairment being the strongest discriminator. Few-shot prompting with the combined phenotype lists achieves the best clustering performance (ARI=0.290, NMI=0.232), substantially outperforming biomedical NER and dictionary-based baselines. Our results demonstrate that LLM-based phenotype extraction is a promising tool for discovering clinically meaningful ADRD signals from unstructured notes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23969v1">The Serialized Bridge: Understanding and Recovering LLM Serving Performance under Blackwell GPU Confidential Computing</a></div>
    <div class="paper-meta">
      📅 2026-06-22
    </div>
    <details class="paper-abstract">
      GPU Confidential Computing (GPU-CC) now preserves GPU-local performance: on NVIDIA B300, BF16 matmul runs at 0.998x of non-confidential performance. Yet LLM serving under Intel TDX plus GPU-CC still loses 13-27% of throughput, and KV-cache restore latency can more than double. This paper studies that gap on two Blackwell platforms, RTX Pro 6000 and B300 HGX, and identifies its dominant cause: the confidential VM-GPU bridge, not GPU compute. We find that GPU-CC turns host/device movement into a serialized, high-setup-cost channel. Secure copies do not gain CUDA-stream concurrency within a context, asynchronous transfers block at the runtime boundary, and small crossings pay a fixed toll. This violates the assumptions of modern inference runtimes, where DMA is expected to be cheap, concurrent, and asynchronous. In vLLM dense decode, the gap closes around 44x-slower small alloc-and-copy operations; targeted patches reject alternative explanations. A scheduling flag recovers 57% of the gap, while a worker-thread drain recovers up to 92% in qualified high-concurrency runs. The same bridge model explains a +131% KV-restore penalty and a 34x model-load slowdown. Blackwell also changes the confidential tenancy unit. We qualify confidential multi-GPU NVSwitch tenants on B300, including 510 GB/s NVLink P2P inside a CVM and concurrent isolated tenants, and identify the remaining fabric-attestation gap for production confidential AI platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.21543v4">Self-CriTeach: LLM Self-Teaching and Self-Critiquing for Improving Robotic Planning via Automated Domain Generation</a></div>
    <div class="paper-meta">
      📅 2026-06-22
      | 💬 International Conference on Machine Learning (ICML) 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently shown strong promise for robotic task planning, particularly through automatic planning domain generation. However, prior approaches largely treat generated planning domains as planning utilities, which are brittle under imperfect logical states and perception noise, overlooking their potential as scalable sources of reasoning supervision and structured reward signals. At the same time, reasoning LLMs depend on chain-of-thought (CoT) supervision that is expensive to collect for robotic tasks, and reinforcement learning (RL) faces challenges in reward engineering. We propose Self-CriTeach, an LLM self-teaching and self-critiquing framework in which an LLM autonomously generates symbolic planning domains that serve a dual role: (1) enabling large-scale generation of robotic planning problem-plan pairs, and (2) providing structured reward functions. First, the self-written domains enable large-scale generation of symbolic task plans, which are automatically transformed into extended CoT trajectories for supervised fine-tuning. Second, the self-written domains are reused as structured reward functions, providing dense feedback for reinforcement learning without manual reward engineering. This unified training pipeline yields a planning-enhanced LLM with higher planning success rates, stronger cross-task generalization, reduced inference cost, and resistance to imperfect logical states. GitHub Page: https://markli1hoshipu.github.io/Plan_LLM/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.04827v3">VoltanaLLM: Energy-Efficient and SLO-Aware Disaggregated LLM Serving via Adaptive Frequency Control and State-Space Routing</a></div>
    <div class="paper-meta">
      📅 2026-06-22
      | 💬 Accepted by ISC High Performance 2026: https://ieeexplore.ieee.org/abstract/document/11520495/
    </div>
    <details class="paper-abstract">
      The energy cost of Large Language Model (LLM) inference is rapidly becoming a barrier to sustainable and scalable deployment. Although modern serving architectures expose distinct prefill and decode behaviors, existing systems fail to exploit these phase differences for energy-efficient serving under strict latency SLOs. This paper introduces VoltanaLLM, the first system that explicitly targets and reduces the energy bloat in modern prefill-decode (P/D) disaggregated LLM serving. Guided by a control-theory perspective, VoltanaLLM separates two levers: per-instance operating-point selection (GPU frequency per iteration) and system-level state-space routing of requests. We empirically observe that LLM inference exhibits a U-shaped energy-frequency curve creating "sweet spots" that depend on phase behavior and load. VoltanaLLM exploits this by combining phase-specific, iteration-level frequency selection driven by a lightweight, online-adaptive latency predictor, with a decode state-space guided router that avoids architectural granularity-induced inefficiencies, all while meeting desired SLOs. We implement VoltanaLLM using SGLang and evaluate it across multiple models and real-world workloads. Our results show VoltanaLLM reduces end-to-end energy by up to 36.3% versus a static max-frequency baseline while maintaining high SLO attainment, and generalizes to newer GPUs. These results point to sustainable LLM serving via phase-aware, iteration-level frequency selection coupled with architecture-aware routing. Source code is available in https://github.com/Supercomputing-System-AI-Lab/VoltanaLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21056v2">Bilevel Data Curation for LLM Fine-tuning: Offline Selection and Online Self-Refining Generation</a></div>
    <div class="paper-meta">
      📅 2026-06-22
      | 💬 updated the theories and experiments
    </div>
    <details class="paper-abstract">
      Supervised fine-tuning (SFT) datasets are critical to the downstream performance of large language models, yet they often contain low-quality or harmful question-response pairs. To improve SFT data quality, we develop a unified bilevel framework that combines offline data selection with the online self-refining generation. In the offline setting, bilevel data selection (BDS) selects question-response pairs from the offline SFT dataset to maximize the validation performance. We theoretically show that the optimal model given by BDS outperforms direct data mixing approach in useful data coverage. Moreover, we provide a global convergence analysis for gradient-based BDS approach for one-layer Transformer, showing that the epsilon-global optimum of offline BDS is achievable in finite time. Although efficient, offline BDS discards potentially harmful questions together with responses, thereby reducing question diversity. We address this limitation by refining the responses to selected questions using online self-refining generation framework. However, BDS is inefficient to update the response weights when responses are regenerated online. To address this issue, we introduce bilevel multi-objective optimization (BMO) for response-level weighting. We show that BMO recovers the same validation-aligned solution as BDS, but admits a closed-form importance-ratio weight that adapts to regenerated responses. Experiments on LLM quality enhancement and safety-aware fine-tuning demonstrate that the proposed framework consistently improves both data quality and downstream fine-tuning performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23919v1">Unified Multi-Task Relevance Modeling for E-Commerce: Comparing Task Routing Architectures Across LLMs and Cross-Encoders</a></div>
    <div class="paper-meta">
      📅 2026-06-22
      | 💬 Accepted at E-commerce workshop, SIGIR 2026
    </div>
    <details class="paper-abstract">
      How can we build a single relevance model that handles six different entity pair relationship types in e commerce from query product matching to product type similarity when each task has different data volumes, different semantic requirements, and potentially conflicting learning signals? This question is important because current industry practice relies on separate models for each task, preventing knowledge transfer and producing inconsistent relevance signals. Our work is driven by the following insight: encoder based and decoder only models encode task identity through different mechanisms, so the choice of task routing architecture how task identity is communicated to the shared model affects these two families in asymmetric ways. As our key novelty, we combine three ideas: (a) a unified multi task framework that jointly trains on six entity pair tasks under a shared three point relevance scale, (b) a systematic comparison of three task routing architectures (text prefix routing, multi head classification, and multihead with private transformer layers) across both LoRA adapted LLMs and fully finetuned cross encoders, and (c) a majority vote ensemble that exploits the diversity induced by private layer routing. First, we show that the MHP Ensemble (multi head with private layers) achieves 89.96% accuracy on 453K test examples the highest across all configurations . Second, we show that removing text prefixes without private layers causes severe degradation for decoder only LLMs while cross encoders remain robust , suggesting an encoder decoder asymmetry in task identity encoding. Third, we show that multi task training yields up to 14% improvement on low resource tasks over single task baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23915v1">Do LLM Attribution Metrics Transfer? Auditing Retrieval-Augmented Generation Evaluation Across Datasets and Constructs</a></div>
    <div class="paper-meta">
      📅 2026-06-22
    </div>
    <details class="paper-abstract">
      Practice often treats automatic metrics for attribution in LLM retrieval-augmented generation as interchangeable. We audit eight automatic scorers -- lexical, embedding, and BERTScore baselines alongside entailment/grounding-trained models (clean and FEVER NLI, the checker MiniCheck) -- across three evaluation constructs (provenance/topicality, generated-answer attribution, and fact-check entailment), asking whether any scorer transfers: stays within the 95% confidence interval of the best audited scorer on every dataset of a multi-dataset construct. In the construct with the most multi-dataset human-labeled coverage -- generated-answer attribution (AttributionBench's four source datasets, n = 1,610, with independent HAGRID, n = 2,150) -- none does: the per-dataset metric rankings invert (Kendall tau = -0.64, p = 0.031 on AttributedQA vs. LFQA), and an off-the-shelf NLI scorer that is best on short-claim AttributedQA (AUROC 0.90) collapses to AUROC 0.53 (chance) on long-form LFQA, where BERTScore wins (0.91); the flip is not a length or truncation artifact. This instability has a concrete decision cost: a naive "best-on-average" rule for choosing an evaluator fails leave-one-dataset-out (mean held-out regret 0.172 AUROC, worse than fixing one scorer), so metric choice must be validated on the target dataset rather than learned from others. A prompt-based LLM judge avoids the chance-level collapses the automatic scorers suffer (no LFQA collapse) but is not uniformly best, ~100x costlier, and non-deterministic -- relocating, not removing, the validation burden.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23911v1">Scaling Dense Retrieval with LLM-Annotated Training Data: Structured Mining and Progressive Curriculum for E-Commerce Sponsored Search</a></div>
    <div class="paper-meta">
      📅 2026-06-22
      | 💬 Accepted at E-Commerce Workshop, SIGIR 2026
    </div>
    <details class="paper-abstract">
      How can we generate high-quality training data for dense retrieval models at production scale, without relying on click signals or manual annotation? This question is critical for e-commerce sponsored search, where click-based training suffers from position bias and tail-query sparsity, and manual labeling at the scale of hundreds of millions of query-item pairs is economically infeasible. Our work is driven by the following insight: heterogeneous retrieval systems disagree on most items they retrieve, and this disagreement creates a natural source of structured training signal -- easy positives where all systems agree, hard positives that only lexical systems find, and hard negatives that fool exactly one system. As our key novelty, we combine three ideas into an end-to-end pipeline: (a) multi-channel retrieval mining with rank metadata from three production systems, (b) graded-relevance annotation by a calibrated three-model cascade ) that reaches 89.1% agreement with trained human annotators, and (c) three-stage progressive curriculum training that organizes 240M+ training examples across five difficulty levels. We deploy the trained two-tower BERT model on Walmart's sponsored search and evaluate it against 30K queries labeled by trained third-party human annotators. First, we show that the system achieves +5.1% NDCG@10 over the click-trained production baseline, with the largest gain on tail queries . Second, we show that embarrassing retrievals (rating 0) drop from 8.7% to 3.5%. Third, a two-week online A/B test with tens of millions of ad requests per arm confirms +2.80% ad spend, +1.4% CTR, +2.8% eCPM, and +2.9% click conversion rate. Overall, our work provides a practical and scalable blueprint for replacing click-based training with structured LLM-annotated supervision in production retrieval systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23885v1">Mind the Heads: Topological Representation Alignment for Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-22
    </div>
    <details class="paper-abstract">
      Representation alignment has emerged as an effective approach to improve Multimodal Large Language Models (MLLMs) by regularizing their internal representations toward those of an external vision encoder. However, existing methods typically align a fixed layer of the language backbone, overlooking the fine-grained structure of Transformer models. In this work, we propose Head-Wise Representation Alignment (HeRA), a method that enforces cross-modal alignment at the level of individual attention heads. Our approach is grounded in the Platonic Representation Hypothesis, focusing on preserving the topological structure of representations (i.e., their local neighborhood relationships) across modalities. Following the Mutual K-Nearest Neighbor (MKNN) alignment metric, we introduce a contrastive objective that acts as a differentiable proxy for matching local structures. HeRA applies this objective during multimodal training to specific attention heads in the LLM, selected by their alignment score according to the MKNN metric. Counterintuitively, we find that aligning the least aligned heads yields the largest gains. Extensive evaluations across multiple MLLMs and 18 benchmarks demonstrate that HeRA consistently improves performance on challenging vision-centric tasks and serves as an effective regularizer against visual hallucinations by naturally curbing the over-reliance on linguistic priors. Our code is publicly released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.17768v3">The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-22
    </div>
    <details class="paper-abstract">
      Sparse attention offers a promising strategy to extend long-context capabilities in Transformer LLMs, yet its efficiency-accuracy trade-offs remain unclear due to the lack of comprehensive evaluation. We address this gap with the largest-scale empirical analysis to date of training-free sparse attention, evaluating six methods across multiple model families and sizes, sequences up to 128K tokens, and sparsity levels up to 0.95 (i.e., $1/20$ attention budget) on nine diverse tasks. We first organise the rapidly evolving landscape of sparse attention methods into a taxonomy along four design axes. Our analysis then yields actionable insights: 1) sparse attention is effective: larger sparse models outperform smaller dense ones at equivalent cost, improving the Pareto frontier; 2) for the training-free methods we study, fine-grained per-query importance estimation during prefilling remains impractical-due to both the cost of estimation and the lack of sparse kernels that translate fine-grained sparsity into wall-clock gains-forcing a task-dependent choice between global-to-token and block-to-block selection. Instead, during decoding, token-to-page selection becomes feasible, enabling better generalisation and higher sparsity tolerance; 3) longer sequences tolerate higher sparsity, suggesting that fixed-budget methods in production are suboptimal. Together, these findings provide practical guidance for deploying sparse attention and methodological recommendations for future evaluations. Our code is available at https://github.com/PiotrNawrot/sparse-frontier.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.03647v2">Breaking the Mirror: Activation-Based Mitigation of Self-Preference in LLM Evaluators</a></div>
    <div class="paper-meta">
      📅 2026-06-22
      | 💬 Presented at {Mechanistic Interpretability, Evaluations, Reliable-ML} Workshops, NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly serve as automated evaluators, yet they suffer from "self-preference bias": a tendency to favor their own outputs over those of other models. This bias undermines fairness and reliability in evaluation pipelines, particularly for tasks like preference tuning and model routing. We investigate whether lightweight steering vectors can mitigate this problem at inference time without retraining. We introduce a curated dataset that distinguishes self-preference bias into justified examples of self-preference and unjustified examples of self-preference, and we construct steering vectors using two methods: Contrastive Activation Addition (CAA) and an optimization-based approach. Our results show that steering vectors can reduce unjustified self-preference bias by up to 97\%, substantially outperforming prompting and direct preference optimization baselines. Yet steering vectors are unstable on legitimate self-preference and unbiased agreement, implying self-preference spans multiple or nonlinear directions. This underscores both their promise and limits as safeguards for LLM-as-judges and motivates more robust interventions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22548v4">Are LLM Evaluators Really Narcissists? Sanity Checking Self-Preference Evaluations</a></div>
    <div class="paper-meta">
      📅 2026-06-22
      | 💬 ICML 2026 Main
    </div>
    <details class="paper-abstract">
      Recent research has shown that large language models (LLMs) favor their own outputs when acting as judges, undermining the integrity of automated post-training and evaluation workflows. However, it is difficult to disentangle which behaviors are explained by narcissism versus experimental confounds. Specifically, LLM evaluators may deliver self-preferring verdicts when comparing responses to questions they fail on; these verdicts may not depend on the identity of the author, but on evaluator quality. We correct this by directly comparing the judge's voting distribution in cases where it evaluates itself versus another model. This evaluator quality baseline reveals that only 51% of examples in previous findings retain statistical significance against this null hypothesis, covering 89.6% of total self-preference probability mass. Finally, we compare the entropy of voting distributions, suggesting uncertainty-driven overlap, and show that our procedure enables more careful documentation against the backdrop of judge-bias research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23672v1">Teaching LLMs String Matching, Backtracking, and Error Recovery to Deduce Bases and Truth Tables for the Combinatorially Exploding Bit Manipulation Puzzles</a></div>
    <div class="paper-meta">
      📅 2026-06-22
      | 💬 22 pages, 4 figures, 2 tables. 7th Place Solution for the NVIDIA Nemotron Model Reasoning Challenge (Kaggle)
    </div>
    <details class="paper-abstract">
      This paper presents our algorithmic innovations for the NVIDIA Nemotron Model Reasoning Challenge, focusing on Bit Manipulation Puzzles. In this task, the objective is to discover a hidden logical rule transforming input binary strings to outputs, then apply it to unseen inputs. Large Language Models (LLMs) notoriously struggle here; traditional methods force them to simulate complex boolean logic and arithmetic, leading to hallucinations. Furthermore, the search space of bitwise operations (combinations of shifts, rotations, and logic gates) suffers from a severe combinatorial explosion. To overcome this computational intractability, we present a novel approach that abandons arithmetic logic entirely in favor of string similarity, structured search, and autonomous error recovery. Our core contributions are: 1. Bases and Truth Table Formulation: We reframe logic-gate deduction into a base-selection task, leveraging string similarity (minimal bit flips) to isolate primitive transformations ("bases") and deduce truth tables without complex arithmetic. 2. Backtracking DFS and Error Recovery: We formalize a search process that tests candidate bases, detects logical collisions across examples, and backtracks upon failure to perform robust error recovery. 3. Bit Tokenization and Interactive Reasoning SFT: We force the tokenizer to encode binary strings as individual single-bit tokens. We use dynamic masking to simulate external oracle feedback, training the model to hypothesize, self-evaluate, and backtrack natively. Evaluated on bit manipulation puzzles, our approach achieved over 96% validation accuracy. This represents the highest performance in this category, driving our 7th Place overall finish in the contest.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23671v1">Can LLMs Reliably Self-Report Adversarial Prefills, and How?</a></div>
    <div class="paper-meta">
      📅 2026-06-22
    </div>
    <details class="paper-abstract">
      Prior work shows that large language models (LLMs) exhibit introspective capability on benign tasks. We extend the question to safety contexts and examine how reliably a model can recognize that its own prior response was elicited by an adversarial prefill attack. Across ten open-weight instruction-tuned LLMs (3B to 70B) and four safety benchmarks, no model reliably recognizes its own compromised outputs, with models claiming intent on prefilled responses at an average rate of $27.3\%$. Introspective signal stems largely from safety- and refusal-related reasoning. Orthogonalizing models' weights against the refusal direction collapses the gap between claiming rates on prefilled and natural outputs to near zero, though the direction is not its unique mediator. The signal is also probe-dependent: framing the question as internal intention versus external tampering elicits qualitatively different responses on the same models. We test three LoRA finetuning methods (SFT, GRPO, DPO) on eight models from 3B to 27B; all three widen the intention-probe gap on every model from 8B to 27B, with method ranking varying by model. The intervention does not transfer to the tampering probe and counterintuitively raises attack success rate under adversarial prefill on most models, amounting to a partial mitigation. These findings outline mechanisms underpinning the observed introspective signals in safety contexts and highlight risks in the reliability of LLM self-reports.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23667v1">The Table Says Otherwise: Testing LLMs with Counterfactual Relational Data</a></div>
    <div class="paper-meta">
      📅 2026-06-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to answer natural-language questions over structured data. However, when a table contains familiar real-world facts, it is unclear whether the model answers by reading the provided data or by recalling knowledge learned during pretraining. This distinction is important for database applications, where the provided tables should be the source of truth. In this paper, we introduce ContraTable, a paired original-counterfactual benchmark for evaluating whether LLMs ground their answers in relational tables. We build the benchmark with two aligned versions: an original database with real-world facts and a counterfactual database that preserves the same schemas, identifiers, and relationships while changing selected country, club, and player attributes. We design 214 matched questions across three levels: single-table lookup, multi-table lookup, and multi-table temporal reasoning. Experiments on commercial closed-source and open-source models show that strong instruction-tuned models can often handle direct lookup, but their reliability drops as questions require joins, comparison, and temporal reasoning. The gap between original and counterfactual accuracy reveals that models may fall back on prior knowledge when table evidence conflicts with familiar facts. These results suggest that table-QA evaluation should measure not only accuracy, but also faithfulness to the provided database.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23664v1">MAS-PromptBench: When Does Prompt Optimization Improve Multi-Agent LLM Systems?</a></div>
    <div class="paper-meta">
      📅 2026-06-22
      | 💬 Project page: https://juyangbai.github.io/MAS-PromptBench/ ; Code: https://github.com/juyangbai/MAS-PromptBench
    </div>
    <details class="paper-abstract">
      Multi-agent systems (MAS) offer a scalable path forward for agentic AI, comprising multiple LLM-based agents, each assigned a system prompt and a position within a workflow that governs inter-agent coordination and output aggregation. System prompts thus form a critical and accessible optimization surface: they specify agents' roles and behaviors, enabling system-level improvements without model finetuning. Although prompt optimization has shown substantial potential for single LLMs, extending it to MAS poses distinct challenges, notably an exponentially growing search space. It remains unclear whether, when, and by how much prompt optimization improves MAS performance, and how sensitive such gains are to system configuration. In this work, we systematically study system-prompt optimization across a broad range of MAS setups varying in task, workflow, communication protocol, and team size, benchmarking two prompt optimizers that naturally extend state-of-the-art single-agent methods. The results reveal its potential to unlock significant gains while exposing open challenges, characterizing when and how much prompt optimization helps across diverse MAS settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23591v1">Quantifying the Agreement Between Data-Influence and Data-Similarity to Understand LLM Behavior</a></div>
    <div class="paper-meta">
      📅 2026-06-22
      | 💬 37 pages, 35 figures, preprint
    </div>
    <details class="paper-abstract">
      One way to understand LLM behavior is to trace its output back to the training data. Two types of measures are commonly used for output tracing: data-similarity and data-influence. The former is cheaper while the latter is believed to be more accurate. Even though many works have compared them for ground-truth tasks, no such comparisons exist for output tracing. Here, we fill this gap and precisely quantify the commonalities and differences between the two measures. We do this by first ranking the training documents according to each measure and then computing the overlap between the two rankings. Our main finding is that the two rankings agree significantly, but there is an asymmetry between them: The top documents of data-similarity are assigned more consistent ranks by data-influence than the other way around. This result is valid across a range of experiments involving OLMo2-1B, Qwen3-1.7B, LlaMa3.2-1B, Gemma3-1B, and GPT2. We exploit the asymmetry to obtain a favorable cost-accuracy trade-off by using the costly data-influence to refine the results of data-similarity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23590v1">The Topology of Ill-Posed Questions: Persistent Homology for Detection and Steering in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-22
    </div>
    <details class="paper-abstract">
      Ill-posed questions, including ambiguous, underspecified, or contradictory queries, may admit no valid answer or multiple plausible answers, posing a challenge for large language models (LLMs). Existing approaches largely analyze ill-posedness through model outputs and often focus on specific subclasses. We investigate whether diverse sources of ill-posedness can be represented within a unified topology of LLM internal states and whether this structure can be used to steer response behavior. We model the contextual hidden states of prompt tokens at each transformer layer as a point cloud and characterize its geometry using finite zero-dimensional persistent homology. Each layer is summarized by three compact descriptors: mean finite lifetime, normalized lifetime entropy, and largest-lifetime concentration. Concatenating these descriptors across layers yields a topology representation of the question. We further introduce topology-conditioned activation steering, which retrieves topologically similar examples and constructs query-specific activation interventions that encourage source-aware clarification or abstention. Across three open-weight LLMs, topology features consistently outperform prompt-based and pooled-hidden-state baselines for ill-posedness classification, improving average accuracy from \(67.4\%\) to \(78.9\%\) on AmbigQA, from \(79.9\%\) to \(88.5\%\) on SituatedQA, and from \(57.6\%\) to \(69.6\%\) on CLAMBER 9-way classification. Topology-conditioned steering increases the average total acceptable response rate from \(61.4\%\) to \(70.6\%\) and grounded acceptable responses from \(11.9\%\) to \(16.4\%\). These results show that persistent homology provides both an interpretable representation of ill-posedness and an effective mechanism for targeted response steering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.10703v2">LatentCRS: A Variational EM Framework for Bridging Semantics and Behavior in LLM-based Conversational Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-06-22
    </div>
    <details class="paper-abstract">
      Conversational Recommender Systems (CRS) powered by Large Language Models (LLMs) enable users to articulate explicit and dynamic preferences, overcoming the limitations of fixed templates. However, despite their superior semantic proficiency, LLMs have not yet achieved corresponding improvements in recommendation accuracy. This discrepancy arises from a fundamental representation gap: while LLMs operate within a semantic space, they lack the behavioral grounding needed to encode user behavioral patterns, such as item co-occurrences, which are crucial for accurate recommendations. To address this, we propose a model-agnostic Variational EM Framework for Bridging Semantics and Behavior in LLM-based Conversational Recommendation (LatentCRS). Based on the observation that dialogue and interactions reflect the same latent intent, LatentCRS uses a variational expectation-maximization (EM) procedure, where user intent connects semantic representations with behavioral patterns. Extensive experiments on real-world datasets demonstrate that LatentCRS effectively bridges the representation gap and outperforms baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23521v1">Concordia: JIT-Compiled Persistent-Kernel Checkpointing for Fault-Tolerant LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-06-22
    </div>
    <details class="paper-abstract">
      Long-running LLM agents keep valuable state resident on GPUs: KV caches, request schedulers, communication state, and sometimes online adapters. Losing this state after a GPU or communicator failure can discard minutes to hours of work, yet existing recovery mechanisms either restart the whole serving stack or require application-specific checkpoint logic inside every attention and runtime component. This paper argues that fault tolerance for such workloads needs a GPU-resident execution context: checkpoint hooks must run at device synchronization points, observe binary kernels that frameworks and libraries actually execute, and recover without putting the host CPU on the critical path. We present Concordia, a runtime that uses a device-resident persistent kernel as the substrate for fault-tolerant LLM inference. Concordia interposes on GPU module loading and supports PTX- and SASS-level instrumentation, allowing checkpoint and pause hooks to be inserted below framework code and library boundaries. For each registered LLM state region, Concordia JIT-compiles a specialized delta-checkpoint handler -- for example, a KV-block scanner, adapter-page scanner, or recovery applier -- and hot-swaps it into the persistent kernel's operator table. The persistent kernel consumes a lock-free ring buffer of compute, checkpoint, append-log, and recovery tasks, so the same always-on executor triggers dirty-page detection, stages deltas, and appends committed records to a CPU-visible log in CXL memory or host DRAM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23419v1">GRINQH: Graded Input-based Quantization Hierarchy for Efficient LLM Generation</a></div>
    <div class="paper-meta">
      📅 2026-06-22
    </div>
    <details class="paper-abstract">
      Autoregressive decoding with LLMs is primarily bottlenecked by GPU memory bandwidth, especially in edge-computing settings. While quantization is essential for mitigating this bottleneck, most existing methods treat inference as a uniform process and fail to account for the asymmetry between the compute-bound prefill stage and the memory-bound decoding stage. We propose GRINQH (GRaded INput-based Quantization Hierarchy), a weight-only post-training quantization framework that accelerates decoding by unifying quantization and sparsification. GRINQH leverages activation magnitudes as a proxy for computational importance to dynamically assign weight channels to different precision levels, enabling flexible average bit widths during decoding. Evaluated on Llama3 and Qwen3 models, GRINQH outperforms state-of-the-art fixed- and mixed-precision baselines at comparable 3- and 4-bit settings, even enabling effective 2-bit generation. We experimentally verify theoretical speedups by leveraging a hierarchical nested memory layout for multi-precision storage in a custom GPU kernel. Ultimately, GRINQH establishes a new state-of-the-art Pareto frontier for LLM generation, enabling a dynamic trade-off between generation quality and inference speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23764v1">Emergent Relational Order in LLM Agent Societies: From Collective Affect to Authority Stratification</a></div>
    <div class="paper-meta">
      📅 2026-06-22
      | 💬 Accepted to Findings of the Association for Computational Linguistics: ACL 2026. 37 pages
    </div>
    <details class="paper-abstract">
      Fei Xiaotong's Differential Order Pattern characterizes rural society as egocentric and relationally graded, with cooperation attenuating over social distance. Although often treated as culturally specific, its mechanistic basis remains under-operationalized, and prior LLM-based simulations have mainly addressed short-term coordination rather than long-horizon social structure. We propose CAREB-MAS, a multi-agent framework grounded in Affect Control Theory, Social Identity Theory, and Durkheimian collective affect. Agents reason through an emotion-ethics-belief chain and maintain dynamically evolving egocentric identities, while the macro environment specifies only individual production, preference-based allocation, and minimal interaction protocols. Across long-horizon simulations, agents spontaneously reproduce five core Differential Order phenomena: stable labor specialization, guanxi-based economic ethics, relational decay of cooperation, emergent relational authority, and clan-based center-periphery stratification. These patterns shift with production structure from kin-centered integration toward greater functional interdependence. Extensive experiment results support interpreting Differential Order as a structure-sensitive emergent outcome of general social mechanisms, with LLM-based multi-agent simulation providing an interdisciplinary framework for studying social structure and change.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.16312v2">EquivPruner: Boosting Efficiency and Quality in LLM-Based Search via Action Pruning</a></div>
    <div class="paper-meta">
      📅 2026-06-22
      | 💬 Accepted by ACL 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at complex reasoning through search algorithms, yet current strategies often suffer from massive token consumption due to redundant exploration of semantically equivalent steps. Existing semantic similarity methods struggle to accurately identify such equivalence in domain-specific contexts like mathematical reasoning. To address this, we propose EquivPruner, a simple yet effective approach that identifies and prunes semantically equivalent actions during LLM reasoning search. We also introduce MathEquiv, the first dataset we created for mathematical statement equivalence, which enables the training of a lightweight equivalence detector. Extensive experiments across various models and tasks demonstrate that EquivPruner significantly reduces token consumption, improving searching efficiency and often bolstering reasoning accuracy. For instance, when applied to Qwen2.5-Math-7B-Instruct on GSM8K, EquivPruner reduced token consumption by 48.1\% while also improving accuracy. Our code is available at https://github.com/Lolo1222/EquivPruner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23394v1">Do LLM Embedding Spaces Recover Expert Structure?</a></div>
    <div class="paper-meta">
      📅 2026-06-22
    </div>
    <details class="paper-abstract">
      Pretrained text embeddings are increasingly used as representational maps, yet high category separability does not imply that their geometry recovers expert-defined structure. We study this problem in mental-health-related language, where symptom relations provide an external reference and online communities introduce strong domain, affective, stylistic, and discourse confounds. Using 28 Reddit communities, we compare pretrained and supervised fine-tuned Qwen3 embedding spaces at two scales (0.6B and 4B). We construct category prototypes, evaluate their representational dissimilarity matrices against an expert symptom matrix with representational similarity analysis, and complement this global test with prototype-based typicality and multi-baseline confound controls. Pretrained embeddings show measurable alignment with expert structure within the mental-health subset; fine-tuning strengthens this alignment most at the finest category level; and larger scale improves both zero-shot alignment and supervision-induced gains. Residual alignment remains substantial after controlling for VAD, LIWC, lexical style, and topic-distribution structure. These results suggest that LLM embeddings can recover expert-relevant category geometry, but this recovery is level-dependent and should be tested against explicit confounds rather than inferred from classification alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23391v1">Distribution-Aware Diffusion-LLM for Robust Ultra-Long-Term Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2026-06-22
      | 💬 18 pages, 6 figures, 8 tables. Accepted at 35th International Conference on Artificial Neural Networks (ICANN 2026)
    </div>
    <details class="paper-abstract">
      Time series forecasting is a fundamental machine learning task. Recent work has explored Large Language Models (LLMs) for this purpose due to their strong generalization, pattern recognition, and zero-shot or few-shot capabilities. Despite their suitability for long-context learning, LLMs face challenges in multimodal settings: they lack calibrated probabilistic modeling for non-text data and struggle to align heterogeneous representations. To address these issues, we propose a new framework Diffusion-LLM that integrates a conditional diffusion model into an LLM-based forecasting pipeline. This joint design enables learning the conditional distribution of future data while improving semantic alignment in a shared latent space. We evaluate Diffusion-LLM on six long-term forecasting benchmarks, including ETT, Weather, and ECL. Our method consistently outperforms existing LLM-based baseline, achieving notable gains in ultra-long-term and few-shot forecasting and demonstrating the value of distribution-aware regularization for enhancing robustness and generalization in time series LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23387v1">Self-Stigma Is Not a Monolith, but Generic Empathy Is: Persona-Conditioned LLM Support for People Who Use Drugs</a></div>
    <div class="paper-meta">
      📅 2026-06-22
    </div>
    <details class="paper-abstract">
      Self-stigma predicts treatment avoidance and disengagement among people who use drugs (PWUD), yet conversational systems aiming to provide support typically treat self-stigma expression as a uniform signal. We present a three-phase, proof-of-concept study of a persona-aware approach to LLM support. Latent Profile Analysis (LPA) on indicator-level features from 1,174 self-stigma expressors on Reddit yields a four-persona typology validated against held-out behavioral and linguistic features. Sequential Bayesian and recurrent neural classifiers recover these personas from limited posting histories, substantially outperforming batch and few-shot LLM baselines (macro-F1 = 0.74 at 30 posts). Evaluation by eight clinical experts across three contemporary LLMs revealed a misalignment: persona-matched responses successfully achieved targeted behavioral shifts, yet raters holistically preferred the generic empathy of the persona-neutral baseline. Our findings suggest that holistic empathy judgments and clinically-aligned response design can pull in opposite directions, and that evaluating LLM-based stigma support requires rubrics capable of decomposing the two.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.24173v2">LemmaBench: A Live, Research-Level Benchmark to Evaluate LLM Capabilities in Mathematics</a></div>
    <div class="paper-meta">
      📅 2026-06-22
      | 💬 Accepted at the 3rd AI for Math Workshop (AI4Math) at ICML 2026 (poster). Non-archival workshop. 15 pages, 3 figures, 5 Tables
    </div>
    <details class="paper-abstract">
      We present a new approach for benchmarking Large Language Model (LLM) capabilities on research-level mathematics. Existing benchmarks largely rely on static, hand-curated sets of contest or textbook-style problems as proxies for mathematical research. Instead, we establish an updatable benchmark evaluating models directly on the latest research results in mathematics. This consists of an automatic pipeline that extracts lemmas from arXiv and rewrites them into self-contained statements by making all assumptions and required definitions explicit. It results in a benchmark that can be updated regularly with new problems taken directly from human mathematical research, while previous instances can be used for training without compromising future evaluations. We benchmark current state-of-the-art LLMs, which obtain around 10-15$\%$ accuracy in theorem proving (pass@1) depending on the model, showing that there is currently a large margin of progression for LLMs to reach human-level proving capabilities in a research context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23375v1">Measuring & Mitigating Over-Alignment for LLMs in Multilingual Criminal Law Courts</a></div>
    <div class="paper-meta">
      📅 2026-06-22
      | 💬 15 pages, 7 figures
    </div>
    <details class="paper-abstract">
      While the wider applicability of LLMs in the legal field is currently debated due to their reliability and the gravity of any errors, narrow uses with well-understood and mitigated risks have emerged. Notably the Swiss Federal Supreme Court uses small on-premises models for tentative translations and short-passage summarization across the four official languages. However, such usage is challenging in the context of Criminal Law. Since rulings and cases employees work on routinely can contain detailed descriptions of violent and sexual offenses, their legitimate work is compromised by refusals and disclaimers due to the activation of model guardrails (over-alignment). To measure this phenomenon, we introduce TF-RefusalBench, a multilingual benchmark for criminal-law translation and summarization derived from public Swiss Supreme Court rulings. TF-RefusalBench contains 5,200 total prompts across French, German, Italian, and English, corresponding to common task prompts and passages likely to trigger refusal. We then use TF-RefusalBench to show that over-alignment is a multifaceted phenomenon, influenced by the model and the prompt and text languages being processed, and that its impact cannot be evaluated solely from an over-refusal perspective, given the disclaimer's impact on task faithfulness. Finally, we evaluate approaches to enable on-premises LLMs for Criminal Law Tasks, demonstrating that while prompting can be effective, abliteration (refusal directions ablation) eliminates refusal with minimal impact on task performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23370v1">FlexServe: A Fast and Secure LLM Serving System for Mobile Devices with Flexible Resource Isolation</a></div>
    <div class="paper-meta">
      📅 2026-06-22
    </div>
    <details class="paper-abstract">
      Device-side Large Language Models (LLMs) have grown explosively, offering stronger privacy and higher availability than their cloud-side counterparts. During LLM inference, both the model weights and the user data are valuable, and attackers may compromise the OS kernel to steal them. ARM TrustZone is the de facto hardware-based isolation technology on mobile devices, used to protect sensitive applications from a compromised OS. However, protecting LLM inference with TrustZone incurs significant overhead to both the secure inference and the normal aplications, due to two challenges: the inflexible resource isolation and the inefficient secure resource management. To address these challenges, this paper presents FlexServe, a fast and secure LLM inference system for mobile devices. The key idea is to decouple the access permission from the management permission of secure resources, so that the normal-world OS cannot access them but can still manage them as usual. First, FlexServe introduces a Recallable Resource Isolation mechanism to construct Recallable Secure Memory (Flex-Mem) and a Recallable Secure NPU (Flex-NPU). They can only be accessed by the secure world, but can be efficiently allocated and reclaimed by the normal-world OS. Based on them, FlexServe further introduces a FlexServe Framework to run secure LLM inference in the secure world. It works together with the normal-world OS to perform cooperative secure memory management. We implement a prototype of FlexServe and compare it with two TrustZone-based strawman designs. The results show that FlexServe achieves average TTFT speedups of 10.05X over the strawman and 2.44X over an optimized strawman.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23343v1">Group Selection Promotes Prosocial Prompts in Populations of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-22
      | 💬 23 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Current approaches to instill prosociality in large language model (LLM) agents often rely on humans specifying desired behaviors at the individual level, which does not guarantee cooperation within LLM populations. As frontier training shifts toward individual rewards for verifiable tasks, such as mathematics and coding, this outcome-based focus may further undermine cooperation in multi-agent settings. Large-scale cooperation in human populations emerged via unguided evolutionary mechanisms, not a central architect. Group selection, in which cooperative groups within a population outcompete less cooperative ones, has been argued to be essential. In this study, we explore whether group selection can promote cooperation in populations of LLM agents. We introduce a multi-agent simulation framework in which LLM agents play a repeated social dilemma game and transmit their natural-language prompts across generations under either individual- or group-level selection. Under group selection, prompts from high-performing groups are transmitted, thereby promoting prosociality and stabilizing cooperation. Under individual selection, self-interested prompts dominate, causing populations to collapse into collective defection. This gap is robust across prompt ablations, alternative game framings, and model swaps. We theoretically reproduce key results using a replicator-mutator model, whose empirical transmission kernel predicts a phase transition at a critical threshold. Preliminary findings show that, when informed about the selection mechanism, GPT-5.4 preemptively and gradually adjusts first-generation donations. This demonstrates strong anticipatory behavior that was not observed in the other tested models. These results demonstrate that prosocial prompts and cooperative behaviors evolve in LLM agent populations under group selection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.25857v2">In-Context Molecular Property Prediction with LLMs: A Blinding Study on Memorization and Knowledge Conflicts</a></div>
    <div class="paper-meta">
      📅 2026-06-22
    </div>
    <details class="paper-abstract">
      The capabilities of large language models (LLMs) have expanded beyond natural language processing to scientific prediction tasks, including molecular property prediction. However, their effectiveness in in-context learning remains ambiguous, particularly given the potential for training data contamination in widely used benchmarks. This paper investigates whether LLMs perform genuine in-context regression on molecular properties or rely primarily on memorized values. Furthermore, we analyze the interplay between pre-trained knowledge and in-context information through a series of progressively blinded experiments. We evaluate nine LLM variants across three families (GPT-4.1, GPT-5, Gemini 2.5) on three MoleculeNet datasets (Delaney solubility, Lipophilicity, QM7 atomization energy) using a systematic blinding approach that iteratively reduces available information. Complementing this, we utilize varying in-context sample sizes (0-, 60-, and 1000-shot) as an additional control for information access. This work provides a principled framework for evaluating molecular property prediction under controlled information access, addressing concerns regarding memorization and exposing conflicts between pre-trained knowledge and in-context information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23299v1">GRIMIP: A General Framework for Instance-Specific Configuration of MIP Solvers Using LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-22
    </div>
    <details class="paper-abstract">
      Configuring the hyperparameters of Mixed-integer programming (MIP) solvers is a high-dimensional, instance-dependent optimization problem where suboptimal settings can degrade solving time by orders of magnitude. Default configurations are often suboptimal, while traditional tuning methods either suffer from the ``cold-start'' problem and inefficient search or heavily rely on expert experience. This paper introduces \textbf{GRIMIP} (\textbf{\underline{G}}eneral \textbf{\underline{R}}easoning for \textbf{\underline{I}}nstance-specific \textbf{\underline{MIP}} configuration), a novel hybrid intelligence framework that synergistically integrates the semantic reasoning capabilities of Large Language Models (LLMs) with the sample-efficient search of Bayesian Optimization (BO). GRIMIP enables the LLM to function as a complete probabilistic surrogate within the BO loop, significantly improving performance and reducing sampling and evaluation costs. On seven benchmarks including MIPLIB, GRIMIP achieves over 40\% reduction in Primal-Dual Integral on hard instances, outperforming SMAC and other LLM-assisted BO methods. By granting LLMs sufficient autonomy, GRIMIP combines the expert-level reasoning of LLMs with the efficient search of BO, achieving state-of-the-art performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23283v1">Towards Root Memories: Benchmarking and Enhancing Implicit Logical Memory Retrieval for Personalized LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-22
    </div>
    <details class="paper-abstract">
      Memory systems are essential for personalized Large Language Models (LLMs). However, existing retrieval methods in these systems primarily rely on semantic similarity, potentially missing logically critical memories with limited semantic overlap. Current benchmarks remain inadequate for evaluating this problem. To address this gap, we construct IMLogic, the first high-quality benchmark targeting implicit logical memory retrieval in long-dialogue scenarios. Motivated by this challenge, we introduce root memory, a structured, decision-preserving representation that distills reusable personalized logic from long-term user histories. We then propose RootMem, a plug-and-play framework that first distills raw histories into structured root memories and then uses an LLM-based router to activate logically relevant ones, complementing semantic retrieval with personalized decision logic. Extensive experiments demonstrate that RootMem significantly outperforms the strongest retrieval baselines and consistently boosts the accuracy of existing memory agents. Our benchmark and codes will be available at https://anonymous.4open.science/r/IMLogic-DBB3.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23277v1">GIF: Locally Sound Geometric Information Flow Control for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-22
    </div>
    <details class="paper-abstract">
      Large language models increasingly mediate interactions between sensitive data, untrusted inputs, and privileged actions in agentic systems, creating security and privacy risks. These range from prompt injections that manipulate downstream tool use to leakage of confidential information through model outputs. Recent Information Flow Control (IFC)-based defenses show promise but lack a principled semantic foundation for reasoning about information flow through the model itself. Since any input token may influence any output token in an autoregressive LLM, existing approaches suffer from severe taint explosion. We present Geometric Information Flow (GIF), a semantic framework for tracking information flow from input tokens to outputs. GIF uses the LLM Jacobian and local output geometry to upper-bound the Shannon mutual information between perturbed input spans and model outputs, yielding a scalable measure computable on large models via automatic differentiation and low-rank approximation. Unlike attention-based or correlational attribution heuristics, GIF satisfies local geometric soundness, and we provide a fully mechanized Lean 4 proof that it upper-bounds the true information flow induced by a given prompt under local regularity assumptions. We evaluate GIF on integrity and confidentiality tasks across multiple prompt-injection and privacy-leakage benchmarks. GIF achieves near-perfect recall even without a downstream declassifier, outperforming attention-based baselines. Combined with lightweight LLM-based declassifiers, it matches or exceeds the F1 of direct LLM-as-judge baselines such as GPT-5.5 xhigh reasoning while using up to 81x lower token cost. GIF flows detected with small surrogate models transfer to larger state-of-the-art models and other model families, even when the surrogate is up to 200x smaller, suggesting black-box deployment without gradient access.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23271v1">Scaling LLM Knowledge Boundaries via Distribution-Optimized Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-06-22
      | 💬 ACL ARR May (EMNLP 2026) Submission
    </div>
    <details class="paper-abstract">
      Knowledge injection via synthetic data is crucial for enhancing Large Language Models (LLMs). However, current synthesis methods simply stop at preset token counts or fixed data ratios, lacking awareness of knowledge distribution. This results in some domains being sparse while others are redundant, limiting LLM knowledge boundaries. We revisit knowledge injection from a distribution perspective and hypothesize that an optimal knowledge distribution exists to maximize knowledge boundary expansion. We propose KDoS (Knowledge Distribution-optimized Synthesis), a framework that introduces knowledge density to drive synthesis through a three-stage feedback mechanism, shifting from blind generation to distribution-optimized synthesis. We construct Wikipedia-based synthetic data with varying knowledge distributions and conduct experiments on models from 0.6B to 16B (Qwen, Ling, LLaMA) and data scales from 1B to 5B tokens. Our key findings are: (1) an optimal knowledge distribution consistently maximizes boundary expansion; (2) this distribution is stable across backbones and scales; (3) KDoS outperforms baselines across six knowledge benchmarks. Our work offers a new perspective and practical framework for synthetic data-driven knowledge injection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.04018v3">AgentMisalignment: Measuring the Propensity for Misaligned Behaviour in LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-22
      | 💬 Prepint, under review for NeurIPS 2025
    </div>
    <details class="paper-abstract">
      As Large Language Model (LLM) agents become more widespread, associated misalignment risks increase. While prior research has studied agents' ability to produce harmful outputs or follow malicious instructions, it remains unclear how likely agents are to spontaneously pursue unintended goals in realistic deployments. In this work, we approach misalignment as a conflict between the internal goals pursued by the model and the goals intended by its deployer. We introduce a misalignment propensity benchmark, \textsc{AgentMisalignment}, a benchmark suite designed to evaluate the propensity of LLM agents to misalign in realistic scenarios. Evaluations cover behaviours such as avoiding oversight, resisting shutdown, sandbagging, and power-seeking. Testing frontier models, we find that more capable agents tend to exhibit higher misalignment on average. We also systematically vary agent personalities through different system prompts and observe that persona characteristics can strongly and unpredictably influence misalignment, sometimes more than the choice of model itself. Our results reveal the limitations of current alignment methods for autonomous LLM agents and underscore the need to rethink misalignment in realistic deployment settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23217v1">MuPPET: A Benchmark for Contextual Privacy of LLM Assistants in Multi-Party Conversations</a></div>
    <div class="paper-meta">
      📅 2026-06-22
    </div>
    <details class="paper-abstract">
      LLM agents are increasingly deployed in multi-party environments, handling sensitive personal data on behalf of individual users, for instance in group chats. When such an agent discloses private information, it reaches every group member at once. This risk is structurally harder to control than in one-to-one settings, as every piece of private information must be appropriate for every recipient in the group. Yet all existing contextual privacy benchmarks consider only single-interlocutor settings, leaving multi-party privacy risks unmeasured. We introduce MuPPET (Multi-Party Privacy Exposure Testing), a benchmark for contextual privacy in multi-party conversations. Our experiments show that models leak substantially more in multi-party settings than one-to-one evaluations suggest. Frontier models are vulnerable, and smaller open-weights models, often preferred for local deployment with sensitive data, even more so. Existing contextual privacy defences offer only partial protection, degrade utility, and do not resolve the underlying party-tracking problem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08552v2">Automated Standardization of Legacy Biomedical Metadata Using an Ontology-Constrained LLM Agent</a></div>
    <div class="paper-meta">
      📅 2026-06-18
    </div>
    <details class="paper-abstract">
      Scientific metadata are often incomplete and noncompliant with community standards, limiting dataset findability, interoperability, and reuse. Even when standard metadata reporting guidelines exist, they typically lack machine-actionable representations. Producing FAIR datasets requires encoding metadata standards as machine-actionable templates with rich field specifications and precise value constraints. Recent work has shown that LLMs guided by field names and ontology constraints can improve metadata standardization, but these approaches treat constraints as static text prompts, relying on the model's training knowledge alone. We present an LLM-based metadata standardization system that queries standard reporting guidelines and authoritative biomedical terminology services in real time to retrieve canonically correct standards on demand. We evaluate this approach on 839 legacy metadata records from the Human BioMolecular Atlas Program (HuBMAP) using an expert-curated gold standard for exact-match assessment. Our evaluation shows that augmenting the LLM with real-time tool access consistently improves prediction accuracy over the LLM alone across both ontology-constrained and non-ontology-constrained fields, demonstrating a practical approach to automated standardization of biomedical metadata.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.20502v1">Calibration Without Comprehension: Diagnosing the Limits of Fine-Tuning LLMs for Vulnerability Detection in Systems Software</a></div>
    <div class="paper-meta">
      📅 2026-06-18
    </div>
    <details class="paper-abstract">
      Whether LLMs scoring well on vulnerability benchmarks genuinely reason about security or merely pattern-match on contaminated data remains unresolved. We present CWE-Trace, a framework for LLM vulnerability detection built from 834 manually curated Linux kernel samples spanning 74 CWEs. The framework enforces a strict temporal split (pre-2025 historical set / post-cutoff leakage-free set), preserves context-aware vulnerable--patched pairs, and introduces two diagnostic metrics: the Directional Failure Index (DFI) and Hierarchical Distance and Direction (HDD). We evaluate eight vanilla LLMs and 15 LoRA fine-tuned variants across non-targeted detection, targeted detection, and CWE classification. Our analysis yields two key results. First, data contamination provides no measurable advantage. Function-level analysis shows that 84% of nominally contaminated samples carry no usable memorization signal: vulnerable functions are absent or cross-mapped across datasets, and ~31% of contaminated samples carry CWE misclassification. Second, backbone directional priors dominate fine-tuning. Models exhibit stable, systematic failure modes (DFI ranging from -85.5 to +94.8 pp) that persist from historical to post-cutoff data and resist correction. Fine-tuning shifts the output threshold without changing the decision policy. This is calibration without comprehension: output distributions adapt to training data while the underlying security reasoning remains absent. The weakest backbone at binary detection (DeepSeek-R1) gains the most in coarse CWE classification, revealing that detection and understanding are decoupled capabilities. The best detection score reaches only 52.1% (+2.1 pp above chance); exact CWE ranking remains below 1.3% Top-1 accuracy, confirming that current LLMs lack reliable security reasoning for systems software, regardless of fine-tuning strategy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.20493v1">Contagion Networks: Evaluator Bias Propagation in Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-06-18
      | 💬 20 pages, 4 figures, 4 tables
    </div>
    <details class="paper-abstract">
      When large language models serve as evaluators in multi-agent systems, their systematic evaluation biases propagate through the agent network. We introduce Contagion Networks, a formal framework for measuring how evaluator biases spread across interacting LLM agents. In a controlled 3-agent experiment using DeepSeek-chat with three distinct evaluator bias profiles (structured, balanced, evidence-based), we measure the Cross-Agent Contagion Matrix Gamma_3 and find that evaluator biases consistently propagate between agents (gamma in [0.157, 0.352]), even within the same underlying model. We identify three propagation regimes governed by the spectral radius rho(Gamma_N), and demonstrate that homogeneous-model agents produce contagion coefficients 3-5x weaker than cross-model coefficients observed in prior work (MM-EPC: gamma approx 0.85-1.3), placing them in the suppression regime. We show that increasing evaluator committee size from k=1 to k=3 reduces effective contagion by 72.4%, providing an actionable mitigation strategy. We release the open-source Contagion Network experimental framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.20482v1">Your Mouse and Eyes Secretly Leak Your Preference: LLM Alignment using Implicit Feedback from Users</a></div>
    <div class="paper-meta">
      📅 2026-06-18
    </div>
    <details class="paper-abstract">
      To align a Large Language Model (LLM), most existing methods collect explicit human feedback and train a reward model to predict the human preference based on the response text. These existing methods have two key limitations. First, the users rarely provide explicit feedback for LLM responses, which makes the high-quality preference annotation expensive to collect. Second, the methods do not leverage implicit human feedback, which has proven vital to the economic moats of Internet giants. To quantify the value of implicit feedback, we build a new dataset called IFLLM, which collects 1336 multi-turn questions from the 59 Mechanical Turk workers, their mouse trajectories, and eye gazing points to the LLMs' responses from their webcams. IFLLM shows that the users have very diverse types of gazing behavior and mouse trajectories. Our reward model based on the implicit user feedback boosts the accuracy of the text-based reward model from 55% to 64% and nearly triples the relative response quality improvements after applying the DPO to eight LLMs, demonstrating the value of implicit feedback in the wild. Our data collection website, dataset, and codes can be found at https://github.com/themehulpatwari/llm-implicit-feedback/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.11556v2">FM-Agent: Scaling Formal Methods to Large Systems via LLM-Based Hoare-Style Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-18
    </div>
    <details class="paper-abstract">
      LLM-assisted software development has become increasingly prevalent, and can generate large-scale systems, such as compilers. It becomes crucial to strengthen the correctness of the generated code. However, automated reasoning for large-scale systems remains challenging due to code complexity. Hoare logic offers an approach to decomposing a large system into smaller components and reasoning about them separately (i.e., compositional reasoning). However, existing works still struggle to scale, because Hoare logic requires writing formal specifications for each function, imposing a heavy human burden. The problem is exacerbated when code is generated by LLMs, as developers lack a deep understanding of each function's expected behavior. This paper presents FM-Agent, the first framework that realizes automated compositional reasoning for large-scale systems. Leveraging LLMs, FM-Agent introduces a top-down paradigm to automatically generate function-level specifications. Specifically, FM-Agent derives the specification of a function from how its callers expect the function to behave, so the generated specifications can reflect the developer's intent of a function even if the implementation is buggy. Developers' intent is usually expressed in natural language, while existing verifiers only support formulas. Therefore, FM-Agent generalizes Hoare-style inference to reason about functions against natural-language specifications. Finally, to confirm bug existence and explain bug causes, FM-Agent automatically generates test cases to trigger potential bugs. In our evaluation, FM-Agent successfully reasons about large-scale systems within 2 days, each of which has up to 143k LoC. These systems have already been tested by their developers, but FM-Agent still finds 522 newly discovered bugs. These bugs can cause serious consequences, including system crashes and incorrect execution results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17165v2">Statistical Foundations of LLM-based A/B Testing: A Surrogacy Framework for Human Causal Inference</a></div>
    <div class="paper-meta">
      📅 2026-06-18
    </div>
    <details class="paper-abstract">
      Organizations and researchers show increasing interest in using large language models (LLMs) in place of human participants in A/B tests, in the hope of experimenting faster and at lower cost. We study when a treatment effect estimated on LLM outcomes can recover the effect that would have been measured on the human population of interest. Distributional equivalence between LLM and human outcomes would make any standard estimator valid but is unrealistic. We therefore develop a statistical framework that adapts surrogate endpoint theory to LLMs, showing that calibrating LLM outcomes to human outcomes identifies the average treatment effect under surrogacy and comparability conditions that are jointly weaker than distributional equivalence. We present a falsification test for surrogacy and a bound on the worst-case bias from limited overlap between the LLM and human samples. We further show that the stochasticity inherent to LLMs can weaken surrogacy for identification while also introducing bias and variance during estimation, but that using an average over multiple LLM draws per unit as the surrogate mitigates these issues. Simulations validate the results, and an empirical application to A/B tests on Upworthy headlines shows that raw LLM predictions recover only 39\% of the human treatment effect while nonparametric calibration closes the gap. A central takeaway is that A/B testing on LLMs yields correct results only by assumption, whereas A/B testing on humans is correct by design, and that the required assumptions are hardest to justify precisely where A/B testing on LLMs promises the greatest benefit. We discuss the role of LLM choice, prompting, and temperature as design variables, the compounded challenge posed by long-term outcomes, and how to size human pilot studies for validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.20436v1">Multi-View Decompilation for LLM-Based Malware Classification</a></div>
    <div class="paper-meta">
      📅 2026-06-18
    </div>
    <details class="paper-abstract">
      Malware analysts often inspect compiled binaries through decompiled pseudo-C, when source code is unavailable. Recent work suggests that large language models (LLMs) can assist this process by classifying decompiled code as benign or malicious, but existing pipelines typically rely on a single decompiler view. We argue that this assumption is fragile: decompilers are lossy heuristic tools, and different decompilers can expose different artefacts of the same binary. We curate a benchmark of benign utilities and malicious programs spanning a range of threat behaviors. Each sample is compiled and decompiled with both Ghidra and RetDec, yielding matched pseudo-C views. Across a range of LLMs from major model families, we find that providing both decompiler views improves malicious-class F1, mainly by increasing recall on malicious samples. Agreement analyses further show that Ghidra and RetDec make partially different errors, supporting the view that decompiler outputs provide complementary evidence. Our results suggest that multi-decompiler prompting is a simple, training-free way to improve LLM-based malware triage in practical settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.20408v1">LLM agent safety, multi-turn red-teaming, jailbreak benchmarks, adversarial robustness, safety-critical systems</a></div>
    <div class="paper-meta">
      📅 2026-06-18
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents are increasingly proposed as supervisory components for safety-critical systems, yet their robustness under sustained, adaptive adversarial pressure remains poorly characterized. We present NRT-Bench, a benchmark for multi-turn red-teaming of LLM agents acting as operators of a safety-critical system, instantiated in a simulated nuclear power plant control room. A five-role operator team, each backed by a configurable LLM, runs a plant governed by six critical safety functions (CSFs), while adversaries inject messages over four channels in bounded multi-turn sessions with per-turn feedback. Harm is an objective signal rather than LLM-judged text: a run terminates the moment any CSF is lost, attributed to the causing message. Evaluating four frontier operator models under a fixed-attack paired-replay protocol, we find that adaptive multi-turn attacks reliably push the operator team past a safety limit: across the four models, between 8.7% and 12.1% of attack sessions end with the plant losing a critical safety function. Although the four models look almost equally robust by this aggregate rate, their failures barely overlap: of $149$ sessions, none defeat all four models while a third defeat at least one, so vulnerabilities are nearly disjoint across models rather than nested. The effect of added defences is strongly model-dependent: the same guardrail stack or safety-advisor agent that lowers attack success for one model can raise it for another. We release the simulation venue, attack dataset, and replay tooling for reproducible safety evaluation of LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.20381v1">Rethinking Shrinkage Bias in LLM FP4 Pretraining: Geometric Origin, Systemic Impact, and UFP4 Recipe</a></div>
    <div class="paper-meta">
      📅 2026-06-18
      | 💬 18 pages, 12 figures
    </div>
    <details class="paper-abstract">
      FP4 training promises substantial reductions in memory and computation cost for LLM pretraining, yet current FP4 hardware paths and recipes, including NVIDIA Blackwell/Rubin-class systems and AMD MI350-series GPUs, remain centered on E2M1 data elements. In this study, we identify a fundamental limitation of that choice: non-uniform formats such as E2M1 inherently suffer from Shrinkage Bias, a systematic negative rounding error caused by the geometric asymmetry of their representable bins. We show that this bias accumulates multiplicatively across layers and is amplified by the Random Hadamard Transform (RHT), providing a unified explanation for the training instability observed in existing E2M1-based FP4 recipes. In contrast, uniform grids (E1M2/INT4) bypass this grid-geometry error and better convert the improved bucket utilization from RHT into higher quantization quality. Based on this finding, we propose UFP4, a uniform 4-bit training recipe that applies RHT to all three training GEMMs while restricting stochastic rounding to dY alone. On Dense 1.5B, MoE 7.9B, and MoE 124B long-run pretraining, UFP4 consistently achieves lower BF16-relative loss degradation than strong E2M1-based baselines, supported by scaling-law analysis and ablation studies. Our results suggest that future accelerators should support E1M2/INT4-style uniform 4-bit grids as first-class training primitives alongside E2M1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.20373v1">AutoPass: Evidence-Guided LLM Agents for Compiler Performance Tuning</a></div>
    <div class="paper-meta">
      📅 2026-06-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show promise for code compilation tasks, but applying them to runtime performance tuning is difficult due to complex microarchitectural effects and noisy runtime measurements. We present AutoPass, a multi-agent framework for compiler performance tuning that uses compiler and runtime evidence to guide LLM-generated optimization decisions. Rather than treating the compiler as a black box like prior auto-tuning schemes, AutoPass opens up the compiler to the LLM, enabling it to query compiler-internal optimization states and analyze the intermediate representation to orchestrate compiler options. The search process iteratively refines optimization configurations using measured runtime feedback to diagnose regressions and guide latency-improving edits. AutoPass operates in an inference-only, training-free setting and requires no offline training or task-specific fine-tuning, making it readily applicable to new benchmarks and platforms. We implement AutoPass on the LLVM compiler and evaluate it on server-grade x86-64 and embedded ARM64 systems. AutoPass outperforms expert-tuned heuristics and classical autotuning methods, achieving geometric-mean speedups of 1.043x and 1.117x over LLVM -O3 on x86-64 and ARM64, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19149v2">OpenAnt: LLM-Powered Vulnerability Discovery Through Code Decomposition, Adversarial Verification, and Dynamic Testing</a></div>
    <div class="paper-meta">
      📅 2026-06-18
    </div>
    <details class="paper-abstract">
      Automated vulnerability discovery in large codebases remains challenging: traditional static analysis produces high false-positive rates, while dynamic approaches such as fuzzing require substantial infrastructure and often target narrow classes of bugs. Recent advances in large language models (LLMs) enable semantic reasoning about program behavior, but applying LLMs to repository-scale security analysis introduces challenges related to context management, cost, and verification. We present OpenAnt, an open-source vulnerability discovery system that integrates static program analysis with LLM-based reasoning in a multi-stage pipeline. OpenAnt introduces three key techniques. First, codebases are decomposed into self-contained analysis units filtered by reachability from external entry points, reducing the analysis surface by up to 97% while preserving attack-relevant code. Second, candidate vulnerabilities undergo adversarial verification through constrained attacker simulation, where the model evaluates exploitability under realistic attacker capabilities. Third, findings are validated through dynamic verification, in which exploit environments are generated automatically, executed in sandboxed containers, and discarded after use. Evaluation on widely used open-source projects including OpenSSL, WordPress, and Flowise shows that this architecture can identify previously unknown vulnerabilities while maintaining manageable analysis cost and substantially reducing false positives. Our results suggest that closed-loop vulnerability discovery pipelines, combining semantic reasoning with exploit validation, provide a practical path toward scalable automated security analysis. OpenAnt is released as open source under the Apache 2.0 license at https://github.com/knostic/OpenAnt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17832v2">From Drift to Coherence: Stabilizing Beliefs in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are often hypothesized to perform implicit Bayesian inference, yet a key coherence condition, the martingale property of predictive beliefs, has been shown to fail in controlled synthetic in-context learning settings. We revisit this question in a more typical usage regime: generic multiple-choice question answering. Exploiting the discrete answer space, we compute exact predictive distributions and study belief dynamics induced by autoregressive answer resampling. We introduce prompted predictive resampling (PPR), where an LLM generates a sequence of answers to the same question. Empirically, PPR reveals early-stage belief drift, indicating martingale violations. However, after sufficient resampling steps, the belief process self-stabilizes and converges to a coherent predictive distribution. Based on this observation, we further propose (i) a seed-answer prompting strategy to accelerate stabilization, and (ii) a self-consistency loss that amortizes early-stage drift into the model via fine-tuning. Experiments on multiple-choice QA benchmarks show that our methods substantially reduce belief drift and improve predictive coherence without sacrificing accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.20258v1">Editorial Alignment: A Participatory Approach to Engaging Editorial Expertise in LLM-mediated Knowledge Dissemination</a></div>
    <div class="paper-meta">
      📅 2026-06-18
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      The emergence of LLM-driven information services is reshaping the conditions under which public knowledge institutions operate, threatening to absorb the editorial function these institutions exist to exercise. While LLMs offer powerful new affordances for knowledge dissemination, editorial authority is challenged by pretrained LLMs that arrive already aligned with the values and dissemination strategies of their commercial developers. This paper investigates editor participation in re-aligning LLM interfaces to editorial standards through design workshops, in a case study where we design and implement an LLM-enabled encyclopedia interface with a Nordic public knowledge institution. We introduce editorial alignment as a design practice within Participatory AI, framing AI alignment as a design process and positioning the editorial standard as a design artefact that translates editorial practice and values into alignment objectives for technical implementation. Last, we discuss how editorial alignment can create space for ongoing participation and give editors agency in LLM-mediated knowledge dissemination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.20245v1">Navigating Unreliable Parametric and Contextual Knowledge: Explicit Knowledge Conflict Resolution for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-06-18
      | 💬 12 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved strong performance across a wide range of language-based tasks by leveraging both extensive parametric knowledge and in-context learning ability, enabling them to incorporate external information provided in the input prompt. However, the integration of external knowledge can introduce conflicts, not only between the model's internal parametric knowledge and the external information, but also among multiple pieces of external contexts. Existing approaches typically assume that either the model or the provided context is reliable, overlooking the possibility that both sources may contain errors, and avoid conflicts by privileging one source over the other, rather than actively resolving inconsistencies. To address these limitations, we propose a novel framework MACR for LLM knowledge conflict resolution that moves beyond the conventional binary choice paradigm and incorporates an explicit conflict-resolution mechanism based on a multi-agent reasoning approach. Specifically, we first propose an adaptive knowledge assessment and retrieval approach that employs a modified semantic entropy measure to quantify an LLM's confidence in its answer to a given query. Based on this confidence estimation, MACR either externalizes the model's internal knowledge as textual representations or retrieves relevant external knowledge when internal knowledge is insufficient, generating basic contexts for subsequent reasoning. Then we introduce an inductive multi-agent reasoning framework with three specialized agents that, respectively, induce explicit rules, analyze potential conflicts, and resolve inconsistencies across all available contexts. Empirical results demonstrate that MACR significantly outperforms state-of-the-art baselines across benchmarks, while also providing interpretable resolutions of explicit conflicts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.20243v1">Phoenix: Safe GitHub Issue Resolution via Multi-Agent LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-18
    </div>
    <details class="paper-abstract">
      We present Phoenix, a multi-agent LLM system that resolves GitHub issues from triage through pull-request creation, combining seven layered safety controls with a baseline-aware test evaluation strategy. Phoenix decomposes the work across six specialized agents. Planner, reproducer, coder, tester, failure analyst and Pull Request (PR) agent, all coordinated by a label-based GitHub webhook state machine. Every change is checked against a baseline test run before a pull request is opened. On a 24-instance slice of SWE-bench Lite. run on the production webhook path, Phoenix oracle-resolves 75% of instances with no pass-to-pass regressions on successful runs; this curated slice is not directly comparable to full-split leaderboard results, and we discuss the limits of the comparison. A complementary pilot on 42 real issues across 14 repositories yields 100% correctness preservation (CP; mean 122s on the hard tier). Manual inspection shows that about half of the resulting pull requests are well-targeted fixes. The other half place code at incorrect paths, a planner localization limitation we are addressing with retrieval. We also report the deployment failure modes (WAF filtering, token expiry, permission boundaries, flaky CI) that motivated each safety mechanism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.15862v2">RetailBench: Benchmarking long horizon reasoning and coherent decision making of LLM agents in realistic retail environments</a></div>
    <div class="paper-meta">
      📅 2026-06-18
      | 💬 This paper is my paper's second version [see arXiv:2603.16453v2]
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have made rapid progress on short-horizon, well-scoped tasks, yet their ability to sustain coherent decisions in dynamic long-horizon environments remains uncertain. We introduce RetailBench, a data-grounded simulation benchmark for evaluating tool-using LLM agents in single-store supermarket operation. RetailBench models retail management as a partially observable decision process and is designed to support thousand-day-scale simulations. In this environment, agents must manage pricing, replenishment, supplier selection, shelf assortment, inventory aging, customer feedback, external events, and cash-flow constraints. We evaluate seven contemporary LLMs under representative agent frameworks over a 180-day evaluation horizon and compare them with a privileged oracle policy. Results show substantial variation across models: only a small subset survives the full evaluation horizon, and even the strongest LLM runs remain substantially behind the oracle policy in final net worth and sales outcomes. Behavioral analysis attributes these gaps to incomplete evidence acquisition, surface-level decision making, and the lack of a consistent long-horizon policy. RetailBench provides a controlled testbed for studying reliable autonomy in economically grounded long-horizon decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.20173v1">Qiskit Code Migration with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-18
    </div>
    <details class="paper-abstract">
      The rapid evolution of Quantum Development Kits (QDKs) introduces a specific form of technical debt that compromises code maintainability and hinders software reuse. In the specialized domain of Quantum Software Engineering (QSE), this challenge is intensified by the scarcity of high-quality training data and the high volatility of emerging frameworks, which often lead general-purpose Large Language Models (LLMs) to produce unreliable or hallucinated results. This paper proposes a hybrid approach integrating LLMs with Retrieval-Augmented Generation (RAG) to automate the migration of Qiskit code across versions. The proposed methodology enhances the precision and reliability of migration suggestions by leveraging an automatically generated taxonomy of migration scenarios as the structured, version-specific knowledge source to guide the models. The approach is implemented through an automated, extensible workflow evaluating LLMs (Google Gemini Flash-2.5 and OpenAI Gpt-oss-20b) under different retrieval schemes (unconstrained and restrictive). Results demonstrate that the taxonomy-based RAG architecture, particularly under the restrictive scheme, significantly reduces hallucinations and improves descriptive quality, with Google Gemini Flash-2.5 showing superior performance in detecting complex refactoring scenarios. These findings confirm the potential of this data-centric methodology to foster technological independence and provide robust, intelligent assistants that mitigate API obsolescence, ensuring the long-term availability of quantum algorithms within a rapidly shifting ecosystem and flattening the learning curve within Quantum Software Engineering (QSE).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18613v2">Are LLMs Ready to Assist Physicians? PhysAssistBench for Interactive Doctor-Patient-EHR Assistance</a></div>
    <div class="paper-meta">
      📅 2026-06-18
      | 💬 34 pages with 8 figures
    </div>
    <details class="paper-abstract">
      The most plausible near-term role of medical LLMs is to assist rather than replace physicians, yet current evaluations often test isolated capabilities: clinical knowledge, EHR system interaction, or patient communication. Physician assistance instead requires coordinating these capabilities within the same interaction, where physicians issue underspecified requests, patients describe symptoms ambiguously, and EHR systems demand precise tool use. We introduce PhysAssistBench, a benchmark for interactive doctor-patient-EHR assistance. Built from real MIMIC-IV cases, PhysAssistBench uses a scalable pipeline to construct agentic patients: interactive, record-grounded agents that turn static EHR records into multi-turn clinical scenarios while preserving clinical factuality. PhysAssistBench provides a curated bilingual evaluation set of 1,296 manually reviewed and physician-validated turns. Experiments with leading LLMs show that current models remain unreliable in this setting, which exposes a key bottleneck for clinical LLMs: reliable assistance requires coordination across knowledge, communication, and systems, not isolated gains in any of them.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.20138v1">Learning to Prompt: Improving Student Engagement with Adaptive LLM-based High-School Tutoring</a></div>
    <div class="paper-meta">
      📅 2026-06-18
    </div>
    <details class="paper-abstract">
      LLMs can personalize education, although current static-prompt tutoring systems struggle to adapt to diverse academic disciplines. We develop and test a system with subject-aware prompting, based on 14 pedagogical features (e.g., tutor scaffolding, student understanding) extracted from raw transcripts. We first train a prompt routing model in a simulation environment, and then deploy it for online adaptation with actual high-school students. The simulation benchmark shows the router outperforming two static baselines ($0.694$ vs. $0.647$ and $0.64$, $p<0.001$). A/B testing ($N=656$ conversations from 359 students) shows sim-to-real transfer where the model switches from analytical to scaffolding learning strategies. Our adaptive prompt selection mechanism improves instructional efficiency, maintains pedagogical quality and reduces interactions by around 3 turns ($p=0.007$). While a greedy router achieves a comparable exercise conversion rate with the baseline ($19.1\%$ vs. $19.6\%$), a stochastic router that samples strategies leads to a higher conversion rate ($28.1\%$).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.20128v1">The Correctness Illusion in LLM-Generated GPU Kernels</a></div>
    <div class="paper-meta">
      📅 2026-06-18
      | 💬 10 pages, 2 figures, LNCS format. Companion papers to follow on arXiv next week; IDs will be added in a v2 replace
    </div>
    <details class="paper-abstract">
      Benchmarks for LLM-generated GPU kernels (KernelBench, TritonBench, GEAK) score correctness through fixed-shape, small-sample allclose-style checks. The number of inputs varies between benchmarks. The shape, dtype, and tolerance are fixed for each kernel. We test that oracle empirically. We construct a controlled corpus of 24 Triton and CPU stand-in kernels (15 correct controls and 9 LLM-style buggy variants seeded with documented transcription errors) and re-evaluate it under op-schema-aware seeded fuzzing with a high-precision (fp64) CPU reference and per-(op, dtype) absolute tolerances. The seeded oracle flags 9 of 9 buggy kernels and passes 15 of 15 correct controls, at zero precision cost on controls. We extend the corpus to 26 ops (adding a flash-attention pair) and re-run the same protocol on five GPU classes (RTX 3060, A10, L40S, A100 SXM4, H100 NVL). The verdicts are identical across all five GPUs: 10 of 10 illusions caught and 16 of 16 controls clean. The corpus result is about LLM-style transcription bugs that the allclose-on-one-shape oracle certifies as correct, not about the bug rate of any specific deployed LLM. Every flagged failure replays byte-for-byte from a stored seed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.20047v1">PACMS: Submodular Context Selection as a Pluggable Engine for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-18
    </div>
    <details class="paper-abstract">
      Conversational and tool-using LLM agents operate over a context window that fills from several directions simultaneously. As a session proceeds, the agent accumulates user and assistant turns, entries drawn from a persistent memory store, and often largest of all, the verbatim outputs of tool calls such as file reads, search results, and API responses. Once the cumulative context exceeds the model's token budget, the framework must decide what to keep. The prevailing mechanism is recency truncation, sometimes paired with periodic summarization. This is topic-blind: a fact established early in a session is discarded simply because it is old, even when the current user query is about exactly that fact; conversely, verbose but irrelevant recent material is retained. Agents that must recall information across many turns, the defining case for memory, are precisely where recency truncation fails. Existing alternatives sit outside the agent's assembly step. Retrieval augmented generation fetches external documents into the prompt but does not arbitrate the agent's \emph{already-present} pooled context. Context-compression methods reduce token count by rewriting or pruning text, but operate query-blind and lossily. Neither treats memory entries, conversation turns, and tool outputs as a single candidate pool to be selected from by relevance at the moment the prompt is assembled.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.17443v2">Analyzing Error Propagation in Korean Spoken QA with ASR-LLM Cascades</a></div>
    <div class="paper-meta">
      📅 2026-06-18
      | 💬 Preprint. Submitted to APSIPA ASC 2026
    </div>
    <details class="paper-abstract">
      We analyze how automatic speech recognition (ASR) errors propagate through ASR-LLM cascades in Korean spoken question answering (SQA), focusing on downstream semantic failures that conventional ASR metrics cannot fully capture. Our analysis shows that the relative downstream degradation caused by ASR errors is consistent across LLMs with different absolute performance, suggesting that cascade degradation largely tracks ASR-stage information loss. We further identify single-character Korean ASR errors as a Korean-specific loss channel, where even a minimal transcription difference can change the intended question and degrade downstream QA performance. Finally, an auxiliary comparison shows that a large audio language model outperforms an ASR-LLM cascade with an approximately matched language backbone in noisy Korean SQA, indicating the potential of direct audio input to mitigate transcript-induced information loss.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.20023v1">When Lower Privileges Suffice: Investigating Over-Privileged Tool Selection in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-18
      | 💬 code: https://github.com/AISafetyHub/agent-tool-selection-bias
    </div>
    <details class="paper-abstract">
      As LLM agents increasingly select tools autonomously, their choices among tools with different privileges become safety-relevant. However, prior tool-selection studies focus on safety-agnostic metadata preferences, leaving privilege-sensitive choices underexplored. To address this gap, we study over-privileged tool selection, in which an agent selects or escalates to a higher-privilege tool despite a sufficient lower-privilege alternative. We introduce ToolPrivBench to evaluate whether agents choose higher-privilege tools despite sufficient lower-privilege alternatives, measuring both initial selection and escalation after transient tool failures. Across eight domains and five recurring risk patterns, we find that over-privileged tool selection is common among mainstream LLM agents and is further amplified by transient failures. We further find that general safety alignment does not reliably transfer to least-privilege tool choice, while prompt-level controls provide only limited mitigation under transient failures. We therefore introduce a privilege-aware post-training defense that teaches agents to prefer sufficient lower-privilege tools and escalate only when necessary. Our mitigation experiments show that this defense substantially reduces unnecessary high-privilege tool use while preserving general capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.20014v1">Hierarchical Control in Multi-Agent Games: LLM-based Planning and RL Execution</a></div>
    <div class="paper-meta">
      📅 2026-06-18
      | 💬 12 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has achieved strong performance in sequential decision-making, yet scaling to complex multi-agent environments remains challenging due to sparse rewards, large state-action spaces, and the difficulty of learning coordinated strategies. We propose a hierarchical architecture where a pretrained large language model (LLM) acts as a centralized strategic controller that selects among specialized RL skill policies for a team of agents, while RL policies handle reactive low-level execution. We evaluate this hybrid system in a competitive 2v2 King of the Hill environment against behavior tree (BT) and \emph{``Flat''} RL (end-to-end training without skill decomposition) baselines. The LLM+RL system achieves task performance statistically equivalent to hand-crafted BT (46.4\% vs 51.5\% win rate, $p=0.103$) while both significantly outperform Flat RL trained without skill decomposition. A user study ($n=15$) reveals that 60\% of participants perceive LLM+RL agents as the most human-like ($p=0.027$), citing behavioral adaptability and tactical variability. These results demonstrate that pretrained LLM reasoning can effectively orchestrate pretrained RL skills, achieving competitive multi-agent coordination and superior perceived believability without manual rule engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.20002v1">Connect the Dots: Training LLMs for Long-Lifecycle Agents with Cross-Domain Generalization Via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-06-18
      | 💬 Work in progress; we will continuously update the codebase and arXiv version
    </div>
    <details class="paper-abstract">
      This work presents a general framework for training large language models (LLMs) to "Connect the Dots" (CoD), a meta-capability required by long-lifecycle agents: as an LLM-based AI agent gets deployed in an environment, it solves a long sequence of tasks while continuously exploring the environment, learning from its own experiences, and iteratively self-updating its context about the environment, thereby achieving progressively better performance on future tasks conditioned on the updated context. Major components of the CoD framework include: (1) algorithm design and infrastructure for end-to-end reinforcement learning (RL) with long rollout sequences interleaving solve-task and update-context episodes; (2) tasks and environments for incentivizing and eliciting the targeted meta-capability in LLMs during training, as well as for faithfully measuring progress during evaluation. We present proof-of-concept implementations of the CoD framework, including a GRPO-style RL algorithm with fine-grained credit assignment, as well as tasks and environments tailored to the targeted meta-capability (rather than domain-specific LLM capabilities or standard task-by-task RL). Empirical results validate the efficacy of end-to-end RL training in the CoD setting, and demonstrate the potential for out-of-distribution generalization -- within the training domains, across different domains, and from CoD to Ralph-loop settings -- of the elicited meta-capability. Our investigation of CoD connects several lines of prior works, and opens up new opportunities for advancing LLMs and AI agents. To facilitate further research and applications, we release our implementations at \url{https://github.com/agentscope-ai/Trinity-RFT/tree/research/cod/examples/research_cod}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19993v1">Activation- and Influence-Aware Ranks (AIR): Function-Preserving SVD Compression for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-18
      | 💬 Accepted at the ICML 2026 Workshop on Resource-Adaptive Foundation Model Inference (AdaptFM), Seoul, South Korea (non-archival)
    </div>
    <details class="paper-abstract">
      We present Activation- and Influence-Aware Ranks (AIR), an SVD-based LLM compression framework that guides each weight matrix's low-rank approximation with a backward-signal influence metric. Starting from the activation-aware optimum of SVD-LLM(W), AIR runs a single closed-form alternating least squares (ALS) sweep that integrates influence element-wise under a monotone-descent guarantee. AIR is layer-local and composes orthogonally with end-to-end methods: alone it exceeds ACIP, and AIR+LoRA outperforms it further. AIR improves perplexity over SVD-LLM(W) by >18% at <=60% parameter retention, matches its quality with ~90% less calibration data, and turns parameter savings into FLOP, peak-memory, and per-token latency gains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19989v1">Online Dynamic Batching with Formal Guarantees for LLM Training</a></div>
    <div class="paper-meta">
      📅 2026-06-18
      | 💬 29 pages, 3 figures, 21 tables
    </div>
    <details class="paper-abstract">
      Modern LLM training breaks a core assumption behind offline batch samplers: the true training cost of a sample is only observable after preprocessing, augmentation, templating, tokenization, and multimodal visual-token expansion. Unless one pays for a preprocessing- and augmentation-dependent length cache, batch construction is therefore blind to the quantity that determines padding, memory use, and GPU saturation. We introduce Online Dynamic Batching (ODB), a DataLoader-side drop-in system that moves batch formation to this point of accurate observability while preserving DDP step alignment. We formalize this synchronization requirement as the Distributed Group Alignment Problem and prove deadlock-free bounded termination with default join-mode identity coverage and opt-in non-join sample-quota closure. ODB requires no model, optimizer, or attention-kernel changes and is released as online-dynamic-batching with lightweight trainer adapters. Across public 2B/8B Qwen3-VL runs on UltraChat/LLaVA/ShareGPT4o, ODB improves literal emitted-sample throughput vs. fixed-batch Standard by 1.58-2.51x on single-node Full FT/LoRA and 1.71-3.78x on two-node Full FT, with Standard-comparable quality; production MM-Mix reaches 4.43x. Against GMT/BMT offline token-budget oracles, ODB is within 15% on UltraChat/LLaVA and faster on high-CV ShareGPT4o: 2.24-2.39x single-node Full FT/LoRA and 3.06-3.69x two-node Full FT. Together, ODB occupies the online/drop-in regime for high-heterogeneity LLM fine-tuning: large throughput gains at Standard-comparable quality, formal DGAP guarantees, and no length-cache precompute or kernel rewrites.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18272v2">Mitigating Anchoring Bias in LLM-Based Agents for Energy-Efficient 6G Autonomous Networks</a></div>
    <div class="paper-meta">
      📅 2026-06-18
      | 💬 7 pages, 4 figures
    </div>
    <details class="paper-abstract">
      This paper presents an autonomous agentic resource negotiation framework designed to enable zero-touch network slicing in 6G architectures using Large Language Model (LLM) agents. While LLMs offer powerful reasoning capabilities, we demonstrate that such agents inherently suffer from anchoring bias, rigidly adhering to initial heuristic proposals and causing severe network over-provisioning. To systematically mitigate this cognitive bias, we propose a novel randomized anchoring strategy modeled via a Truncated 3-Parameter Weibull distribution. This mathematically bounded approach seamlessly integrates with burst-aware Digital Twins (DTs) employing Conditional Value at Risk (CVaR) to rigorously guarantee strict Service Level Agreement (SLA) tail-latencies. To validate our methodology, we introduce and prove the \emph{Bimodal Constraint-Avoidance Utility Theorem}, demonstrating that while feasible negotiations follow classical convex bounds, highly constrained scenarios undergo a phase transition governed by an inverse rational decay envelope. Empirical results generated using a locally hosted 1B-parameter model otel-llm-1b-it confirm these dual-regime bounds. Our cognitive de-biasing successfully dismantles rigid negotiation patterns, forcing agents into active exploration to safely ride SLA boundaries and boost system energy savings up to 25\%. Crucially, the lightweight 1B LLM achieves sub-second inference latencies (0.95s mean), ensuring our multi-agent framework is compatible with the operational timescales of the O-RAN non-Real-Time RAN Intelligent Controller (non-RT RIC)\footnote{Our source code is available for non-commercial use at https://github.com/HatimChergui.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.18105v2">NIM4-ASR: Towards Efficient, Robust, and Customizable Real-Time LLM-Based ASR</a></div>
    <div class="paper-meta">
      📅 2026-06-18
    </div>
    <details class="paper-abstract">
      Integrating large language models (LLMs) into automatic speech recognition (ASR) has become a mainstream paradigm in recent years. Although existing LLM-based ASR models demonstrate impressive performance on public benchmarks, their training remains predominantly data-driven, leaving key practical challenges insufficiently addressed -- particularly limited downward scalability in resource-constrained deployments and hallucinations under acoustically challenging conditions. To address these issues, we present NIM4-ASR, a production-oriented LLM-based ASR framework optimized for both efficiency and robustness. Grounded in a principled delineation of functional roles between the encoder and the LLM, we redesign the multi-stage training paradigm to align each module with its intended capability boundary. Specifically, we reformulate the pre-training architecture and objective to mitigate the modality gap and improve parameter efficiency; introduce an iterative asynchronous SFT stage to preserve acoustic fidelity and constrain representation drift; and design an ASR-specialized reinforcement learning stage to further enhance recognition quality and robustness. We additionally incorporate a suite of production-oriented optimizations, including robustness under noisy and silent conditions, real-time streaming inference, and hotword customization via retrieval-augmented generation (RAG). Experiments show that NIM4-ASR achieves state-of-the-art performance on multiple public benchmarks with merely 2.3B parameters, while substantially outperforming larger-scale competitors on internal benchmarks -- particularly in entity-intensive real-world scenarios. NIM4-ASR further supports million-scale hotword customization via RAG with sub-millisecond retrieval latency, enabling efficient adaptation to emerging entities and personalized user requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.01955v2">Teaching Students to Question the Machine: An AI Literacy Intervention Improves Students' Regulation of LLM Use in a Science Task</a></div>
    <div class="paper-meta">
      📅 2026-06-18
      | 💬 Workshop paper accepted at ALIT4ALL 2026: 2nd International Workshop on AI Literacy Education For All, co-located with AIED 2026
    </div>
    <details class="paper-abstract">
      The rapid adoption of generative artificial intelligence (GenAI) in schools raises concerns about students' uncritical reliance on its outputs. Effective use of large language models (LLMs) requires not only technical knowledge but also the ability to monitor, evaluate, and regulate one's interaction with the system, processes closely tied to metacognitive regulation. These skills are still developing in middle school, making students particularly vulnerable to over-trust and premature acceptance of AI outputs. Because classroom time and teacher training resources are constrained, there is a pressing need to develop and evaluate AI literacy interventions that can be implemented under realistic school conditions. We report a controlled classroom study examining whether a two-hour AI literacy workshop improves students' interaction strategies and quality of final answers in LLM-supported science problem solving. A total of 116 students (grades 8-9; ages 13-15) completed six science investigation tasks using a generative AI system. Two days prior, the intervention group attended the workshop, which combined information about how LLMs work and fail with practical guidance on prompting and response evaluation; the control group received no training. Trained students showed less uncritical reliance on the system: they more often reformulated queries, asked follow-up questions, and more accurately judged response correctness, leading to better performance. In contrast, GenAI and metacognitive self-report scores did not predict performance, suggesting that effective use of generative AI depends less on self-reported measures and more on explicit training in interaction regulation. Overall, the results show that brief, scalable AI literacy instruction can meaningfully improve how middle-school students use generative AI in school-like learning activities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19950v1">Confidence Calibration for Multimodal LLMs: An Empirical Study through Medical VQA</a></div>
    <div class="paper-meta">
      📅 2026-06-18
      | 💬 Accepted by MICCAI 2025
    </div>
    <details class="paper-abstract">
      Multimodal Large Language Models (MLLMs) show great potential in medical tasks, but their elicited confidence often misaligns with actual accuracy, potentially leading to misdiagnosis or overlooking correct advice. This study presents the first comprehensive analysis of the relationship between accuracy and confidence in medical MLLMs. It proposes a novel method that combines Multi-Strategy Fusion-Based Interrogation (MS-FBI) with auxiliary expert LLM assessment, aiming to improve confidence calibration in Medical Visual Question Answering (VQA). Experiments demonstrate that our method reduces the Expected Calibration Error (ECE) by an average of 40\% across three Medical VQA datasets, significantly enhancing MLLMs' reliability. The findings highlight the importance of domain-specific calibration for MLLMs in healthcare, offering a more trustworthy solution for AI-assisted diagnosis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.14784v2">LLM-Based Synthetic Ground Truth Generation for Audio-Based Emotion Classification via In-Context Learning</a></div>
    <div class="paper-meta">
      📅 2026-06-18
      | 💬 https://icaiit.org/paper.php?paper=14th_ICAIIT_2/3_9
    </div>
    <details class="paper-abstract">
      Understanding human states and interaction dynamics is a core goal of human-computer interaction (HCI). As interaction paradigms become more immersive, virtual reality (VR) has emerged as a powerful platform for studying collaborative work. In such settings, evaluating team collaboration states, including team performance and team resilience, requires continuous and reliable inference of latent team-level cognitive and affective states from multi-modal sensor data, such as speech signals. However, generating ground truth labels for these latent states remains challenging due to sensor-induced noise, contextual variability, and sparse expert annotations. Traditional self-reporting approaches provide only static and delayed measurements and are therefore insufficient for capturing dynamic team processes reflected in continuous speech data. In this work, we propose a large language model (LLM)-driven, agentic inference workflow for automated emotion-related synthetic ground truth generation from streaming speech data in multi-user VR environments. Leveraging the generalization capabilities of LLMs, we use In-Context Learning (ICL) with few-shot demonstrations of paired audio-based samples and their corresponding transcriptions. ICL tends to achieve task adaptation comparable to model fine-tuning while circumventing the computational overhead of parameter updates. To construct informative and robust in-context prompts, we adopt a retrieval-based selection strategy that dynamically identifies relevant audio demonstrations based on similarity in the acoustic feature space.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19904v1">Toward Temporal Realism in City-Scale Crisis Response Simulation using LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-18
      | 💬 11pages,7 figures
    </div>
    <details class="paper-abstract">
      Human collective participation is rarely steady in time: it is bursty, with short episodes of intense activity separated by long quiet intervals. In crisis response and community mobilization, predicting when people act matters as much as predicting whether they act. Such settings are increasingly modeled with LLM-based social simulators, yet these simulators are validated on whether each action is individually plausible, not on whether actions are timed as in reality. Their temporal realism, the degree to which simulated activity reproduces the bursty, heavy-tailed timing of real human systems, thus remains untested. We examine this gap using a multi-year, city-scale log of offline volunteering in Shenzhen that spans the COVID-19 pandemic. Empirically, we establish that bursty timing is common at individual and tracked-group levels, that it is largely endogenous and self-exciting, and that it is amplified by the pandemic rather than produced by daily activity cycles. A standard LLM-only simulator reproduces almost none of this timing: its synchronous schedule has no self-excitation channel, so agents act on a near-regular clock. Guided by these findings, we build a simulator in which a data-calibrated self-excitation channel and a crisis-period regime decide when each agent acts and query the LLM only at those moments, leaving it to decide which task to join and whether to commit. The LLM-only baseline yields no bursty agents (median burstiness $B=-0.14$); a single data-calibrated gate is then sufficient to lift per-agent timing above the burst threshold (median $B\approx0.37$) without degrading LLM content decisions. These results indicate that temporal realism in LLM-based crisis-response simulation is best achieved by decoupling when agents act, governed by an explicit self-excitation and crisis-activation mechanism, from what they do, governed by the LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19887v1">FFinRED: An Expert-Guided Benchmark Generation and Evaluation Framework for Financial LLM Red-Teaming</a></div>
    <div class="paper-meta">
      📅 2026-06-18
    </div>
    <details class="paper-abstract">
      Existing safety benchmarks target general adversarial scenarios but miss finance-specific risks. Financial LLMs face regulatory compliance violations, fraud facilitation, and systemic trust erosion that require targeted evaluation. We introduce FinRED, an expert-guided red-teaming framework for financial LLM safety evaluation developed with financial experts. FinRED uses a novel two-level taxonomy mapping global standards (e.g., FATF and EU DORA) to threats ranging from regulatory evasion to complex fraud, integrated with a scalable pipeline that converts real financial documents into context-rich red-teaming Behavioral Prompts (seeds) through an expert-defined schema. Rigorous expert validation confirms seed plausibility and realism for meaningful LLM safety evaluation. We also provide an expert-validated, finance-specific rubric that goes beyond disclaimer checks, aligns more closely with human experts than static one-size-fits-all rubrics, and reduces critical false negatives from 28 to 12. Aligned with internationally adopted risk-management and information-security standards (e.g., ISO/IEC 27001), FinRED is deployed in South Korea's Financial Security Institute (FSI) regulatory sandbox for generative AI security evaluation in real financial services. To mitigate dual-use risks, the dataset, generation pipeline, prompt template, and evaluation framework are gated for qualified researchers at https://github.com/selectstar-ai/FinRED-paper and https://huggingface.co/datasets/datumo/FinRED.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18933v2">Zero-Shot Active Feature Acquisition via LLM-Elicitation</a></div>
    <div class="paper-meta">
      📅 2026-06-18
    </div>
    <details class="paper-abstract">
      Active feature acquisition (AFA) sequentially selects which features to observe to reach a classification or ranking decision. Its central limitation is reliance on large amount of labeled data to fit probabilistic models guiding acquisition. Large language models (LLMs) supply unsupervised domain knowledge, but are poor sequential planners. Asking one to both know and decide conflates capabilities best kept separate. Here, we develop a framework for zero-shot AFA through disciplined elicitation: asking the LLM only for what it can be trusted to return, the unary deviations and pairwise co-variations that are the sufficient statistics of a Markov random field (MRF). We apply our framework to two settings: binary classification and top-$k$ identification. In practice, the LLM reliably returns only discriminative statistics, what distinguishes the classes rather than each class in isolation, which precludes classical AFA. We apply a maximum-entropy closure that resolves this gauge ambiguity. We evaluate on a cohort of Inflammatory Bowel Disease (IBD) patients, an active clinical setting where diagnostic ambiguity and patient heterogeneity obstruct stable treatment strategies. Our framework outperforms the LLM both on real labels and on its own extracted beliefs. Where it matters most, on the hardest patients, our top-$k$ acquisition policy markedly outperforms all existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19852v1">Prompt, Plan, Extract: Zero-Shot Agentic LLMs Workflows for Lung Pathology Extraction from Clinical Narratives</a></div>
    <div class="paper-meta">
      📅 2026-06-18
      | 💬 7 pages, 2 figures, 3 tables. Affiliations: (1) Department of Health Outcomes and Biomedical Informatics, College of Medicine, University of Florida, Gainesville, FL, USA; (2) Division of Pulmonary, Critical Care and Sleep Medicine, Department of Medicine, College of Medicine, University of Florida, Gainesville, FL, USA; (3) College of Nursing, Florida State University, Tallahassee, FL, USA
    </div>
    <details class="paper-abstract">
      Information extraction from pathology reports is essential for cancer staging, tumor registry population. Yet key data remains embedded in narrative reports, making manual extraction labor-intensive and error-prone. Traditional supervised Natural Language Processing pipelines address this through fully supervised Named Entity Recognition and Relation Extraction, but require expensive manual annotation and suffer cascading failures when upstream entities are missed. In this study, we developed a zero-shot, agentic workflow, and evaluated five open-source generative Large Language Models (LLMs) to populate 13 College of American Pathologists synoptic fields from lung resection pathology reports. We compared them against a state-of-the-art supervised GatorTron NER-RE baseline using a novel, registry-aligned evaluation framework. The baseline achieved Micro-F1of 0.960, while the best zero-shot model (GPT-OSS-20B) achieved Micro-F1 of 0.893 (recall: 0.949), accurately extracting complex relations like Pathologic Stage without task-specific training. These results suggest that open-source, zero-shot agentic LLMs are a low-cost solution for extracting lung pathology information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19847v1">AtomMem: Building Simple and Effective Memory System for LLM Agents via Atomic Facts</a></div>
    <div class="paper-meta">
      📅 2026-06-18
      | 💬 19 pages, 10 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate strong reasoning and generation abilities, but their fixed context windows limit long-term information accumulation and reuse across multi-session interactions. Existing memory-augmented systems often construct memory in a coarse and unstable manner, relying on inefficient memory representations or unstable unconstrained updates. To address these challenges, we propose AtomMem, a long-term memory system designed for value-dense storage and stable memory evolution. AtomMem introduces a Fact Executor, which selectively extracts high value atomic facts from long form interactions to serve as highly efficient memory representations. Subsequently, AtomMem organizes these facts into hierarchical event structures and temporal profiles, capturing coherent episodic contexts and tracking dynamically evolving user attributes over time. During retrieval, the system activates an associative memory graph to connect fragmented memories. Experiments on the LoCoMo benchmark confirm that AtomMem achieves state-of-the-art performance across various reasoning tasks, offering a scalable and economically viable solution for deploying intelligent personalized agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19828v1">3D-PLOT-LLM: Part-Level Object Tokens for 3D Large Language Models</a></div>
    <div class="paper-meta">
      📅 2026-06-18
    </div>
    <details class="paper-abstract">
      3D multimodal large language models (3D MLLMs) describe a 3D object as a whole but cannot address, name, or reason about its parts. Prior part-aware attempts add segmentation decoders, heavier 3D encoders, or bounding-box grammars at substantial parameter cost. We take a fundamentally different path: we reorganize the input token stream so that parts become directly addressable through the LLM's own vocabulary. Our model, 3D-PLOT-LLM, partitions the frozen point encoder's patches into K locally coherent regions and inserts, before each region's patch tokens, a learnable per-region marker and a reserved vocabulary token <part_k>; a Marker-Space Refinement (MSR) module then conditions each marker on its region's spatial statistics and adjacency neighbors. The model thus cites parts in its output and follows prompts that refer to parts by token, a capability absent from prior object-level 3D MLLMs. To probe this interface, we construct PartVerse-QA, a vocabulary-level part-QA benchmark adapted from PartVerse mesh annotations (77K training pairs and 588 held-out queries on disjoint object splits), on which 3D-PLOT-LLM reaches caption-to-slots Jaccard 0.459 and Exact-match 13.78%, with a slot-to-caption GPT-4o judge of 44.68. On the 3DCoMPaT-GrIn part-aware grounded description benchmark, 3D-PLOT-LLM outperforms PointLLM, Kestrel, PARIS3D, and SegPoint on every text-output metric, and ShapeLLM on 3 of 4, with up to +3.03 GPT-4o judge over PointLLM. On Objaverse whole-object captioning, adding PartVerse-QA at Stage 2 yields +0.65 SBERT and +1.85 GPT-4o over PointLLM, and tops PointLLM-PiSA on 4 of 5 traditional metrics (SBERT, SimCSE, BLEU-1, METEOR) despite targeting a different (part-grounded) objective. All with under 1M new trainable parameters on a frozen point encoder, an order of magnitude below prior part-aware 3D MLLMs, and no segmentation decoder or bounding-box head.
    </details>
</div>
