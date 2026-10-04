# llm - 2026_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.00840v1">Contextual trajectory and incremental contextual displacement: Towards using LLMs to understand dynamic, utterance-specific meaning construction</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Transformer-based large language models (LLMs) such as RoBERTa represent text using contextual word embeddings (CWEs), which alter the embeddings associated with each token based on surrounding context. We construct token-wise incremental trajectories by repeatedly recomputing a token's CWE as successive words are added to a sentence, yielding a representation of how contextualized embeddings evolve as the utterance unfolds. We evaluate this approach using garden-path sentences as a test case with characteristic features. Token-wise trajectories reproduce known features of garden-path processing, including disruption around the critical region, and reliably distinguish garden-path sentences from matched disambiguated controls. We introduce several metrics for quantifying representational displacement across contextual increments and show that trajectory information can be highly predictive of sentence type. We find that ambiguity-related information is recoverable not only from the sentence-level CLS representation but also from ordinary vocabulary tokens, suggesting that utterance-level information is distributed across multiple representational scales. In exploratory analyses, we find qualitatively similar trajectory structures in other ambiguity- and misdirection-related linguistic phenomena. Together, these results establish token-wise incremental trajectories as a promising framework for studying utterance-specific meaning construction using LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.00839v1">Do Defenses Against LLM Extraction Work Across Attacks? A Lifecycle Benchmark of Black-Box Model Extraction</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 24 pages, 5 figures, 14 tables. Code: https://github.com/sliu11-byte/MEA-Bench
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) deployed through text-only APIs face model extraction risks, as adversaries can collect their responses to train surrogates that reproduce their capabilities. While prior work has developed diverse attacks and defenses, evaluations remain fragmented across access assumptions, model configurations, query budgets, and security objectives, limiting comparability across methods. To address this gap, we introduce a unified benchmark covering six extraction attacks, ten defenses, and two adaptive attacks that paraphrase or back-translate protected responses before surrogate training. The benchmark controls model configurations, query data, budgets, and held-out evaluation conditions within each comparison while preserving attack-specific querying and training procedures. We measure surrogate capability, fidelity to the victim, output quality using Rep-4, and query-budget sensitivity; defenses use their own security metrics paired with surrogate performance. For the adaptive attacks, we jointly measure provenance-detector scores and the capability and fidelity of surrogates trained on rewritten responses. The benchmark thus provides a reproducible basis for comparing extraction methods and their interactions with defenses under text-only access. Code and artifacts are available at https://github.com/sliu11-byte/MEA-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.00833v1">VERITYGATE: A Four-Gate Schema-Level Faithfulness Framework and Paired Benchmark for Grounded LLM Narrations over Structured Evidence</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 16 pages, 5 figures, 7 tables. Accepted at Grounding Language Models: Learning Faithfully and Efficiently (GroundLM 2026), co-located with EMNLP 2026. Code: https://github.com/sachinkg12/yukti/releases/tag/veritygate-groundlm-2026-v1.0.0 Supplementary artifact: https://doi.org/10.5281/zenodo.22668710
    </div>
    <details class="paper-abstract">
      Fluent LLM explanations may not follow the evidence from a structured system. We present VERITYGATE, a four-gate checker for declared evidence IDs, entities, numbers, and claim types. It checks a fixed schema; it does not verify every fact in the prose. At r=0 and r=1, we test 900 instances per setting (450 grounded-ungrounded pairs) with GPT-4o-mini, Llama-3.3-70B, and Claude Sonnet 4.6. Under this schema-level contract and before repair, 80.3% of mini claims and 47.9% of Sonnet claims fail. These are verifier rejection rates, not prose-hallucination rates. One repair pass raises claim survival from 19.7% to 28.0% for mini and from 52.1% to 54.3% for Sonnet. Verified claims per example change by +0.14 for mini, -0.71 for Llama, and -0.47 for Sonnet, so survival and output volume must be reported together. A second Sonnet pass gives no clear gain. At r=1, Gate 4 covers 97.0%, 98.7%, and 100% of failing claims for mini, Llama, and Sonnet. Small human studies support the rules but show gaps between schema checks and correct prose. A domain-specific GPT-4o judge test shows an order effect, so it is only a usefulness check. We release the code and data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.21404v2">Multi-Perspective LLM Annotations for Valid Analyses in Subjective Tasks</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Large language models are increasingly used to annotate texts, but their outputs reflect some human perspectives better than others. Existing methods for correcting LLM annotation error assume a single ground truth. However, this assumption fails in subjective tasks where disagreement across demographic groups is meaningful. Here we introduce Perspective-Driven Inference, a method that treats the distribution of annotations across groups as the quantity of interest, and estimates it using a small human annotation budget. We contribute an adaptive sampling strategy that concentrates human annotation effort on groups where LLM proxies are least accurate. We evaluate on politeness and offensiveness rating tasks, showing targeted improvements for harder-to-model demographic groups relative to uniform sampling baselines, while maintaining coverage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.00787v1">Identity-Bound Governance Under Execution Uncertainty: An Accountability Proof Block for LLM Agent Persistent Halts, with Cryptographic Implementation and Cross-Model Calibration</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 24 pages, 7 tables, 2 propositions. Paper 8 of the Agent Governance Series. v2: RFC 8785 canonicalization, event_id uniqueness (V5), k-of-n multi-principal threshold governance (Prop 8.5), infrastructure fault injection experiment. Code: https://github.com/chelof100/identity-bound-governance
    </div>
    <details class="paper-abstract">
      A correctly governed LLM agent can reach a state in which neither continuing execution nor automatically halting is admissible: the system has detected a persistent failure of its observability or drift-detection layer, but cannot itself decide who has the authority to resume, deny, or recalibrate the deployment. We call this an identity-bound governance event and formalise the mechanism that resolves it. We introduce the Accountability Proof Block (APB): a system-constructed evidence block, a human-supplied decision block, and an ed25519 signature binding both to a registered principal. The system cannot forge the signature, and the principal cannot alter the evidence undetected. We prove four theorems: Protocol-Bounded Governance Completeness, Non-Repudiability, Impossibility of Anonymous Re-Authorization, and Finite-Time APB Construction Termination. The implementation uses RFC 8785 JSON canonicalization and a UUID4-based replay predicate. Empirically, governance completeness holds over 3,812 halt events with zero unresolved cases; the verifier detects 100% of 1,800 attacks across 9 adversarial vectors; a k-of-n multi-principal variant shows 0 false acceptances in 2,000 single-key capture attempts. A study of six open LLMs finds the drift threshold T* stable within model (sigma/T* < 2%) but varying across models, refuting size-monotonicity: the largest model did not drift. T* must therefore be measured per deployment, and the APB is the vehicle by which that threshold yields accountable authority transfer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.00707v1">Initialization Improves LLM-Driven Discovery</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been used for novel discovery of algorithms, theorems, drugs, and other tasks through the use of harnesses that prompt an LLM to iteratively optimize an objective. In this work, we study the relationship between the population of previous iterates and eventual discovery success. We generalize past work on harness design to develop a suite of 12 harnesses called 'Modular' and characterize their performance across 5 diverse discovery tasks, finding that discovery success is brittle and sensitive to harness design. We uncover mode collapse, characterized by a dramatic drop in the diversity of iterates, as a common failure mode. We find that popular state-of-the-art harnesses and diversity-inducing harness interventions, which aim to prolong this collapse, yield inconsistent gains. Our results instead uncover that the performance of early discoveries is predictive of eventual success. We therefore propose a universally applicable intervention that performs an initial stage of parallel exploration in order to initialize subsequent iterative optimization. Our method provides consistent gains across many harnesses and target applications, confirming the importance of initialization in LLM-driven discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14661v2">M-CALLM: Multi-level Context Aware LLM Framework for Group Interaction Prediction</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 This submission is being withdrawn to avoid redundancy with arXiv:2604.08771, which presents a shorter workshop version of the same work. Please refer to that entry
    </div>
    <details class="paper-abstract">
      This paper explores how large language models can leverage multi-level contextual information to predict group coordination patterns in collaborative mixed reality environments. We demonstrate that encoding individual behavioral profiles, group structural properties, and temporal dynamics as natural language enables LLMs to break through the performance ceiling of statistical models. We build M-CALLM, a framework that transforms multimodal sensor streams into hierarchical context for LLM-based prediction, and evaluate three paradigms (zero-shot prompting, few-shot learning, and supervised fine-tuning) against statistical baselines across intervention mode (real-time prediction) and simulation mode (autoregressive forecasting) Head-to-head comparison on 16 groups (64 participants, ~25 hours) demonstrates that context-aware LLMs achieve 96% accuracy for conversation prediction, a 3.2x improvement over LSTM baselines, while maintaining sub-35ms latency. However, simulation mode reveals brittleness with 83% degradation due to cascading errors. Deep-dive into modality-specific performance shows conversation depends on temporal patterns, proximity benefits from group structure (+6%), while shared attention fails completely (0% recall), exposing architectural limitations. We hope this work spawns new ideas for building intelligent collaborative sensing systems that balance semantic reasoning capabilities with fundamental constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.00701v1">From Images to Tasks: Characterizing Multimodal LLM Interactions in the Wild</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Multimodal large language models (LLMs) increasingly integrate vision and text, yet how people use them in natural settings remains underexplored. We seek to answer the question: when users upload images, what tasks are they trying to accomplish? Analyzing over 40,000 de-identified image-upload conversations from Microsoft Copilot, we characterize real-world multimodal use through a hierarchical framework of ten capabilities, spanning perception, cognition, and generation. First, we characterize the distribution and composition of these capabilities, finding that the majority of image-upload tasks involve multiple capabilities. Second, we find that multimodal use spans a broader and more diverse task space than text-only interactions, with asymmetric coverage and task classes that rely on cross-modal grounding. Third, mapping observed capability demand onto 253 existing benchmarks reveals uneven alignment between benchmark coverage and real-world use: benchmarks concentrate on perception and reasoning toward fixed answers, while common workflows involving text, code, and data generation remain comparatively undertested. We validate our taxonomy and findings on an independent ChatGPT dataset. Our results provide a large-scale empirical characterization of what users seek to accomplish with multimodal LLMs and highlight opportunities for benchmark design grounded in observed user demand.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.33044v2">Pinned and Still Unstable: Within-Judge Verdict Variance and the Noise Floor of LLM-as-Judge Leaderboards</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 v2: corrects reporting details (claim provenance, candidate generation cap, top-K ties, data completeness, abort threshold, model aliases) and clarifies wording; no new experiments, within-judge variance results unchanged. Changes are listed in Appendix D. 32 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Modern LLM evaluation assumes that pinning a judge to a fixed model version and decoding at temperature zero yields reproducible verdicts. We show this assumption fails as a property of how LLM-as-Judge is operationalized on cloud serving infrastructure, not of any particular model family. Across four frontier judges served via a single major enterprise cloud platform and three standard benchmarks (Arena-Hard, AlpacaEval 2, MT-Bench), identical inputs to the same temperature-zero judge, at a constant serving-reported model version, produce different verdicts across re-runs: per-item flip rates of roughly 5% on average and about 40% on the close-call items that decide leaderboard margins, with a per-judge magnitude spanning a 40x range (0.13% to nearly 10%). We introduce metrics tailored to this instability: per-item flip rate, a two-part stability profile (waver fraction and conditional intensity), and adjacency separability. For the principal judge the aggregate ranking is stable (0% top-K instability, 0% pooled winner flip); what degrades is precision: under a paired hierarchical bootstrap, roughly one-fifth to three-quarters of adjacent leaderboard positions are statistically indistinguishable, a noise floor driven mainly by finite prompt sampling rather than the judge. Across judges, leaderboards agree on the coarse ordering but diverge in the middle (Kendall's tau of 0.42-0.64 between Gemini and Sonnet judges on Arena-Hard, values sensitive to answers truncated at the generation cap). Of 15 expected head-to-head orderings we re-judge, 12 survive every re-run of the principal judge but only 8 survive every judge. Leaderboards thus report unhedged point estimates that overstate their precision. We propose a minimal, low-cost reporting protocol: several judge re-runs, published stability profiles and adjacency intervals, and results under at least two judges from different families.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.00687v1">Leto: Fast In-Place Recovery for LLM Training on Surviving Hardware</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Hardware-operable failures (HOFs) interrupt large language model (LLM) training but permit recovery on the same hardware without reset, repair, or replacement. Existing recovery systems nevertheless reload checkpoints, recompute lost progress, and rebuild process state, idling GPUs that could otherwise continue training. We present Leto, a fault-tolerant training system that leverages surviving hardware to enable efficient in-place recovery. Our key insight is that the state needed to resume training can be retained or prepared outside the active training process while remaining on the same hardware. Leto retains the working model state and the reusable process state, and preinitializes the remaining state in a shadow trainer. We devise two-tier erasure protection and chunk-level transactional updates to keep the retained model state recoverable and consistent, and reclaim the shadow state when active training needs its GPU memory. Evaluation on 6- and 72-GPU NVIDIA A100 clusters shows that Leto recovers 3.6--6.5$\times$ faster than the best-performing checkpointing baselines and improves productive training time by up to 13.7 percentage points. Large-scale simulation shows over 95% productive training time on a 131,072-GPU cluster.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.00685v1">Backdoor Purification for LoRA-Tuned LLMs via Null-Space Projection</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 NeurIPS 2026
    </div>
    <details class="paper-abstract">
      With the rapid adoption of large language models (LLMs) and parameter-efficient fine-tuning (PEFT) methods, the risk of backdoor attacks has become more severe. Existing backdoor purification methods typically rely on at least one of the strong assumptions, such as prior knowledge of triggers, access to clean references, or aggressive retraining, and they often lack comprehensive evaluations. These constraints substantially limit their practical applicability. To overcome these challenges, our work proposes purifying LoRA-tuned LLMs without these assumptions and even without post-hoc retraining of the suspect parameters. Our objective is to significantly reduce the attack success rates (ASR) while preserving both (i) the base model's general capabilities and (ii) the new downstream skills learned through the adapter. Through a series of ablation studies, we progressively scale our approach from a single layer in a text classification setting to a full-parameter LLM in the generative task. Through careful data curation and feature approximation, we extract high-fidelity backdoor directions and, for each layer or head, construct orthogonal null spaces in both the input and output channels, onto which the LoRA updates are projected. Empirically, our null-space projection method reduces the ASR from nearly 100% to less than 10%, while preserving the base model's benign performance and the adapter's learned abilities during downstream task adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.10793v2">ConQuR: Corner Aligned Activation Quantization via Optimized Rotations for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are costly to deploy due to their large memory footprint and high inference cost. Weight-activation quantization can reduce these costs, but low-bit activation quantization remains difficult because activation outliers induce large quantization error. Recent rotation-based methods address this by applying orthogonal transformations that redistribute activation magnitude across dimensions, but existing approaches either require expensive end-to-end rotation training or rely on stored activation corpora, introducing significant compute or storage overhead. We propose a lightweight post-training rotation calibration method for LLM activation quantization. Our method learns orthogonal rotations that align normalized activations with the corners of an inscribed hypercube, encouraging activation energy to be distributed more evenly across dimensions. This objective admits an efficient closed-form update via the orthogonal Procrustes problem, avoiding gradient-based optimization over the orthogonal group. We further introduce an online calibration procedure that updates rotations as calibration samples are processed, eliminating the need to store activations on disk and allowing rotations to adapt to quantized activation distributions during calibration. Experiments on Llama-2 and Llama-3 models from 3B to 70B parameters show that our method achieves competitive or improved performance across perplexity benchmarks and common sense reasoning tasks while avoiding both costly end-to-end training and large offline activation storage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.17384v3">Quantifying Diversity of Thought: A Predictive Law of Weighted LLM Ensemble Lift</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      This paper provides an experimentally verified formal law for calculating the uplift that diversity of thought provides in Large Language Model (LLM) ensembles. From first principles, we derive an exact decomposition of LLM ensemble lift into rescue and damage masses, which yields a compact heuristic for calculating uplift. From this we extract the metrics which predict ensemble performance: an accuracy-adjusted correctness correlation, $φ_{\mathrm{adj}}$, together with the accuracy gap and collective accuracy of the pair. We test the law on 767,520 inferences from ten open-weight models over two graduate-level science benchmarks, together with a novel agentic cybersecurity benchmark in which each model conducts digital-forensics investigations by multi-turn tool use in a network-isolated sandbox (23,520 graded trials including abstentions); all votes are released openly. Calibrated once on SuperGPQA at a 40:60 vote split, the heuristic predicts lift on the calibration set with Spearman's $ρ=0.84$ and, with its coefficients frozen, transfers to two datasets never used in calibration ($ρ=0.51$ on GPQA Diamond and $0.84$ on the forensic tasks), whilst the measured swap mass tracks realised lift with $R^2\ge 0.96$ throughout. Raw $φ$ has almost no predictive power ($R^2\le 0.09$ throughout); the accuracy-adjusted $φ_{\mathrm{adj}}$ is markedly superior ($R^2=0.67$ on SuperGPQA), and the heuristic combining these metrics is the most stable pre-pooling predictor across the three datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.08894v2">WaLLM -- Understanding Use and Engagement with a General-Purpose LLM on WhatsApp</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Large language model (LLM) chatbots are increasingly reaching users through messaging platforms (e.g. WhatsApp). However, these systems remain largely proprietary and opaque, while academic research has focused on narrow, domain-specific assistants. This leaves open questions about how people use general-purpose LLMs and how such systems should be designed. To address this gap, we developed WaLLM, a general-purpose LLM chatbot, and deployed it on WhatsApp as a design probe to study open-ended AI use in the wild. Our findings show that health and well-being accounted for the largest proportion of queries, suggesting that users turned to WaLLM for advice and information. Engagement features varied in their adoption and associated patterns of use: proactive communication supported the service's visibility and correlated with higher user activity, while communal lists facilitated content discovery. We report how these features were adapted to WhatsApp's affordances and discuss implications for designing general-purpose LLM services over messaging platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.00682v1">Ontology-Grounded, Reasoner-Verified Benchmarks for Evaluating LLM Reasoning in Scientific AI</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 17 pages, 2 figures. Accepted at the AI Data Readiness for Scientific Discovery (AIDaR) Workshop at the 40th Conference on Neural Information Processing Systems (NeurIPS 2026), Paris
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly underpin scientific AI applications that reason over structured knowledge, from biomedical question answering to materials informatics. However, their logical reasoning often falls short, producing factual inaccuracies unacceptable in these settings. Reliable evaluation remains challenging: manual dataset construction scales poorly, and LLM-based generation risks embedding the very flaws it aims to measure. High-quality benchmarks must ground both correct and incorrect labelled examples in explicit background knowledge, formally verifiable by a standard reasoner. We propose a pipeline that automatically generates ontology-grounded multiple-choice question (MCQ) benchmarks from any sufficiently axiomatised OWL 2 ontology, with correct answers grounded in the ontology by design. Distractors are generated by perturbing the right-hand-side class expressions of class definition axioms, and their incorrectness is formally verified by an OWL reasoner via entailment checks. We evaluate the pipeline on three ontologies: Pizza (small, academic), PMDco (complex, materials science), and DOID (large, biomedical), generating 112, 2,491, and 15,216 MCQs respectively. Distractors span four semantic categories from class unsatisfiability to weakened subsumptions, enabling diagnostic evaluation of specific reasoning failures. Items meet natural language quality standards: mean LLM judge scores of 4.02, 4.36, and 3.36 out of 5 confirm fluency, and correct-answer-to-distractor similarity above 0.8 shows that wrong options cannot be dismissed on surface form alone. Six LLMs evaluated zero-shot achieve 41.1-76.8% accuracy, well above the 25% random-guessing baseline, indicating the benchmarks are challenging and discriminative. This work is a step towards more reliable benchmarks for assessing logical reasoning in scientific AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.00675v1">LabBook: Harnessing Experimental History for Efficient LLM-Driven Discovery</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Evolutionary approaches to LLM-driven discovery often generate new programs from a small set of selected ancestors. This keeps contexts manageable but can omit useful evidence from other experiments, whereas including the full experimental history produces long, redundant contexts. We introduce a simple, single-agent discovery harness built around LabBook, an agent-maintained memory that serves two complementary roles: guiding retrieval of relevant evidence from a complete experimental log and informing the generation of new solutions. At each iteration, the same agent combines its memory with retrieved evidence and jointly produces the next program and an updated LabBook. This separates complete history retention from selective context construction, without requiring an explicit population or branching search structure. On 49 Frontier-CS problems, LabBook improves the observed quality-cost trade-off over the evaluated evolutionary baselines with two backbones, while remaining competitive across nine additional mathematical, systems, and heuristic-design tasks. Code will be released at https://github.com/BoYuanVisionary/LabBook.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.00663v1">Backdoor Containment via Expert Quarantine and Shutdown in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 NeurIPS 2026
    </div>
    <details class="paper-abstract">
      Backdoored large language models (LLMs) can behave normally on benign inputs while producing attacker-specified outputs under hidden triggers. Existing defenses span four stages--prior-training, in-training, post-training, and inference-time--and share one of two underlying strategies: either suppress backdoor learning (by filtering poisoned data or interrupting its acquisition during optimization) or learn, then purify (by repairing model weights or gating inputs after a fully backdoored model has formed). We propose a third strategy, learn, but channel: allow backdoor formation during training but route it into a designated, quarantined component that can be disabled at deployment. To this end, we propose Quarantined Expert Shutdown QES, a computationally efficient containment strategy built in a regularization-steered MoE-like setting. Specifically, given a poisoned dataset, QES augments a Transformer-based language model with routed expert-specific LoRA branches and lightweight routers, and uses auxiliary routing objectives to attract trigger-conditioned behavior into a designated expert while preserving benign capability elsewhere. At deployment, mitigation reduces to a single constant-time operation: zeroing the quarantined expert's routing weight, without trigger screening or further updating model weights. Empirically, our methods reduce the attack success rate ASR from 100% to 0-10% on most settings across two tasks, three attacks, and four model families, while downstream utility is often preserved or only modestly affected. These results establish learn, but channel as a previously unexplored regime for backdoor containment in generative LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.00622v1">Understanding and Mitigating Library-Related Issues in LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Software practitioners increasingly rely on Large Language Models (LLMs) to generate code that integrates external libraries. However, LLMs often produce incorrect library usage, such as invalid imports, outdated API calls, and hallucinated dependencies, leading to compilation or runtime failures that reduce the reliability of AI-assisted software development. In this paper, we propose an agentic approach to mitigate libraryrelated errors in LLM-generated code. More specifically, we first conduct an exploratory study to characterize the library-related issues produced by LLMs. Our analysis of 100 LLM-generated code files reveals that 84% of generated files contain at least one library-related error, with recurring patterns including incorrect import paths, missing imports, hallucinated libraries, deprecated library usage, and unused imports. Based on these findings, we design an agentic approach that integrates task analysis, documentation grounding, code generation, and automated validation to improve library usage during code synthesis. We evaluate our approach on 300 code generation tasks derived from realworld implementations of rapidly evolving Python frameworks, including LangChain and AutoGen, across five LLMs: GPT-5, DeepSeek-V3, Qwen3, Mistral, and Llama 3. The results show that our approach consistently improves code generation quality across all evaluated models, reducing library-related errors by 38.1% - 54.6% and increasing code correctness by up to 16%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.00613v1">Spatial Strategies, Not Actions: Vector-Quantized Geodesics as Tools for LLM-Driven Agents</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Large language model (LLM) based agents are often criticized for lacking spatial understanding and mainly exploiting statistical text patterns. We investigate their spatial comprehension through an architecture combining geometrical tools with a LLM serving as a high-level orchestrator in grid-world environments. The agent first collects geodesic trajectories, which are then vector-quantized to extract a representative subset. Offline, the LLM associates a natural language description of the underlying behavioral patterns to each selected trajectory, making it a tool. Online, the LLM chooses the appropriate tool conditioned on the current state and goal. Low-level control is handled by primitive actions that execute the trajectory associated with the tool. From an agentic AI perspective, this approach separates learning into two levels: tool discovery is handled through unsupervised quantization of trajectories, while reasoning and decision-making are handled by the LLM. We test the approach in a partially observable dynamic 2D grid environment with an open vision-language model (Qwen3.6-35B-A3B). Pairing the geometry-derived tool library with an agent-centered zoom tool and a collision detection tool lets a fast, non-reasoning configuration match the goal-reaching rate of a much more costly chain-of-thought version, while cutting the cost of a decision from minutes to seconds.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.00562v1">Can LLMs Reason Over Long Horizons? An Empirical Evaluation of Context Strategies for Longitudinal Clinical Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Longitudinal clinical reasoning requires large language models (LLMs) to identify and integrate relevant evidence distributed across extended patient histories. Although long-context models can process increasingly large amounts of information, providing more history does not necessarily make relevant evidence more accessible or improve reasoning. We compare five context strategies (Full, Recent, Episodic, Semantic, and Hybrid) on MedLoCoMo across four open-weight LLMs, examining answer correctness, robustness to query-evidence distance, and abstention on questions with unsupported premises. Episodic and Hybrid generally achieve the strongest overall accuracy, while Recent Context degrades most as supporting evidence becomes more distant; Episodic and Hybrid maintain the highest accuracy at long distances. Analysis of adversarial questions further shows that strong performance on answerable questions does not necessarily translate to successful abstention when the available history does not support the requested conclusion. These findings show that reliable longitudinal reasoning depends not only on how much history an LLM can access, but critically on how relevant evidence is selected and presented for reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.00555v1">Extending LLM-based support for software engineers with ADHD</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Software engineering workflows are often not designed to accommodate the needs of developers with Attention Deficit Hyperactivity Disorder (ADHD), despite known challenges related to task initiation, sustained attention, and completion. At the same time, large language models (LLMs) are increasingly integrated into programming tools, but existing systems do not account for neurodiversity or support structured progression across development tasks. In this work, we present Tether 2.0, an LLM based assistant designed to support software engineers with ADHD through workflow oriented interaction across planning, coding, debugging, and review. The tool was named Tether 2.0 because it builds upon the open source code and foundations of the original Tether system. Our approach combines structured interaction modes, activity aware context, and persistent memory to support task progression and continuity. We evaluate the system through expert feedback and hands on use with a software engineer with ADHD. Our results indicate that Tether 2.0 supports requirement clarification, task decomposition, incremental implementation, debugging, and completion, while enabling users to maintain progress and resume work after interruptions. These findings suggest that LLM based assistants can support direction, sustained progress, and learning in software development when designed with workflow structure and context aware interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.22826v2">MINCE: Shrinking LLM Evaluation Datasets via Few-Model Monte Carlo Calibration</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 Accepted to EMNLP 2026, Industry Track
    </div>
    <details class="paper-abstract">
      Evaluating LLMs across many model variants---quantized, fine-tuned, or deployment-specific---requires running large benchmarks repeatedly, a process that can take tens of hours per model on edge hardware such as NPUs. Existing subset selection methods reduce this cost but depend on large calibration pools or learned prediction layers. We introduce MINCE (Monte Carlo Informed N-sizing for Compact Evaluation), which uses Monte Carlo simulation over per-item logs from a small set of calibration models to find the minimum subset size that bounds accuracy drift and then fixes a randomly sampled subset at that size, with no prediction layer needed. MINCE reduces IFEVAL by 54\%, MMLU by 89\%, GSM8K by 70\%, and MMLU-Pro by 88\% with maximum drift $\leq$2.62\,pp on BF16 calibration models. The frozen subsets generalize to held-out GPU models with mean drift $\leq$1.40\,pp and to INT4 NPU models with mean drift of 0.77--3.59\,pp, while delivering evaluation speedups of up to 8.1$\times$ on the GPU models and evaluation speedups of 1.7--3.4$\times$ on the NPU models. The method is robust to calibration pool size and achieves lower drift than tinyBenchmarks (12$\times$ lower on MMLU, 3.3$\times$ on GSM8K) while using 42$\times$ fewer calibration models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.00511v1">Before Agents Decide: Epistemic Action in LLM-Based Systems</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 Accepted at the Foundations of Agentic Systems Theory (FAST) Workshop at NeurIPS 2026
    </div>
    <details class="paper-abstract">
      Before a difficult decision, people often act simply to understand the situation better. We turn an object to see another side, place alternatives next to each other, or change one condition and observe what happens. These actions may not complete the task, but they improve the evidence needed for the next choice. LLM-based agents can search and explore, yet agent design gives less attention to an earlier question: is the available evidence ready for the decision? Sometimes necessary evidence is missing. In other cases, the evidence is present but its form hides what matters, or the comparison needed to judge it does not yet exist. Cognitive science calls actions that improve the basis for a later choice epistemic actions. We bring this idea to LLM-based agents and distinguish three modes: acquiring missing evidence, transforming available evidence, and probing a system to create a revealing response. We use the term epistemic scaffolding for the interfaces, tools, and environments that make these actions possible and auditable. This paper argues that agent design must address how decision-ready evidence is produced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.00465v1">AIR-LLM: Broadcasting AI Weights over Radio for Memory-Free Edge LLM Inference via RF Computing</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 14 pages, 12 figures, 6 tables. Appendix: 12 pages, 7 figures, 11 tables
    </div>
    <details class="paper-abstract">
      Next-generation large language models (LLMs) are expanding from the cloud to ubiquitous edge devices. However, edge devices typically either lack the memory to store increasingly large LLM weights or, even with enough memory, spend unaffordable energy on loading the weights. This raises our question: can an edge device run an LLM without storing or loading its weights, but receive them over the air and consume them on the fly? Inspired by wireless broadcasting, we present AIR-LLM, an LLM inference architecture for edge devices, which is composed of: (i) a central radio (e.g., 5G base stations) that broadcasts the LLM weights into the air, and (ii) the edge user that receives the weights and completes the general matrix-vector multiplication (GEMV) of LLM inference directly in the radio frequency (RF) domain using RF mixers. To further shorten the airtime, AIR-LLM exploits MIMO spatial multiplexing and proposes an energy-efficient precoder-postcoder pair on the edge to calibrate its own wireless channel. Since the central radio stays user-unaware, AIR-LLM is user-scalable so that one broadcast serves unlimited users within its coverage. We implement AIR-LLM on the NVIDIA Sionna ray-traced channels of two real-world urban scenes and the profiling of a real RF mixer. With a WikiText-2 perplexity degradation of 4.0% on LLaMA-3.1-8B, AIR-LLM saves the energy by 157.7x/40.4x against the FP16 and weight-only quantization baselines; with 20 users, its airtime is 104.1x/26.0x shorter, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.40335v1">Is Weight Tying Still Beneficial for Decoder-Only LLMs in Private Settings Under DP-SGD?</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 Accepted at the 8th IEEE International Conference on Trust, Privacy and Security in Intelligent Systems, and Applications (IEEE TPS 2026). 12 pages (10 pages of main content), 1 figure, 13 tables
    </div>
    <details class="paper-abstract">
      Differentially Private Stochastic Gradient Descent (DP-SGD) is a leading approach for privacy-preserving fine-tuning of large language models (LLMs). Many decoder-only LLMs employ weight tying between input and output embeddings, a design choice originally introduced for parameter efficiency and improved language modeling performance in the non-private setting. However, the impact of weight tying under differentially private training remains largely unexplored. In this work, we investigate the role of weight tying in the DP setting using GPT2 and DistilGPT2 as representative decoder-only architectures. Interestingly, we find that untied embeddings consistently outperform weight-tied models under DP-SGD, achieving gains of up to 4.74% points in accuracy on SST-2, QNLI, and QQP. Beyond improved utility, untying embeddings enables the use of memory-efficient ghost clipping for DP-SGD. By contrast, weight tying introduces shared-parameter interactions that complicate standard ghost norm computation and largely negate its computational advantages. As a result, untied models achieve over 60% lower memory usage while preserving the benefits of ghost clipping. Our results indicate that untied embeddings provide a more effective and scalable design for differentially private training of decoder-only LLMs and highlight the need to revisit standard LLM architectural choices in the privacy-preserving setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.40247v1">Herschel: Continuous Optimization of Production LLM Inference through On-Demand Profiling</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 17 pages
    </div>
    <details class="paper-abstract">
      Model-as-a-service platforms call for continuous optimization as complex serving conditions expose inefficiencies missed before deployment. Detailed always-on profiling can incur substantial overhead, while lightweight collection omits information needed for diagnosis. We present Herschel, a continuous optimization system for production large language model (LLM) inference. Our key insight is that adaptive, on-demand profiling can provide rich full-stack evidence without continuous collection. Herschel safely attaches to and detaches from selected running processes without engine changes or restarts, and adapts coverage as investigations reveal missing evidence. Herschel reconstructs operator executions and cross-process dependencies to identify inefficiency mechanisms and suggest solutions using applicable reference fixes. AI agents implement and test engine and kernel changes under controlled conditions that preserve the triggering workload and dependencies, with expert review before deployment. Controlled tests show active-collection overhead below 0.5% for time to first token and 7% for time per output token. Bounded windows, typically 30 s, avoid the continuous cost of always-on tracing. Over six months, Herschel collected approximately 17,000 traces across over 120 model variants and more than 10 accelerator types, identifying inefficiency patterns in 23% of the traces. Representative findings guide widely deployed optimizations, including restructured synchronization, removal of unused computation, and improved operator implementations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.37574v2">MERGE: Multi-LLM Ensemble for Retrieval via Generative Enrichment</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 9 pages, 4 tables, 1 figure. Preprint
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to enrich user queries in information retrieval (IR) so that a standard retriever such as BM25 can bridge vocabulary gaps with the target corpus. Any single LLM, however, is limited by its training data and architectural biases, and its enrichment behavior depends on hand-crafted prompts that must be re-engineered for each new model -- an expensive and poorly scalable process. We present MERGE (Multi-LLM Ensemble for Retrieval via Generative Enrichment), a two-stage framework: three heterogeneous 7-8B open-source LLMs independently produce candidate expansions, and a larger LLM generatively synthesizes them into a single query. To make prompt engineering scalable across the ensemble, we integrate a task-grounded Automatic Prompt Optimization (APO) loop into both stages. Unlike APO methods that judge candidates with an LLM evaluator, our loop scores each candidate by its downstream retrieval performance and runs a small tournament between the current champion prompt and optimizer-proposed drafts, terminating once the champion survives two consecutive rounds; a history-augmented variant additionally feeds the recent tournament trajectory back to the optimizer. MERGE is retriever-agnostic and issues a single BM25 pass with no rank fusion, no supervised document expansion, and no re-indexing. On five BEIR benchmarks (NQ, SciFact, FiQA, Touche-2020, DBPedia), MERGE improves BM25 nDCG@10 over the original queries by +2.1 to +14.9 points and matches or outperforms strong LLM-based query-expansion baselines despite using only compact open-source models. Ablations confirm that the Stage-2 ensemble beats any single Stage-1 LLM, and that task-grounded APO converts large seed-prompt regressions into consistent gains without hand-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.28013v4">Kill-Chain Canaries: Stage-Level Tracking of Prompt Injection Across Attack Surfaces and Five Production LLMs</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 12 pages, 6 figures, 6 tables. Code: https://github.com/KevinChunye/prompt_injection
    </div>
    <details class="paper-abstract">
      Multi-agent LLM systems now read documents, web pages and tool results on behalf of users, yet their resistance to prompt injection is usually reported as one number: did the attack succeed? We introduce a kill-chain canary method that plants a unique token in every injected payload and records the furthest of four stages it reaches (Exposed -> Persisted -> Relayed -> Executed), across 950 runs, five production LLMs, six attack surfaces, and five defense conditions. Exposure was 100% among runs that called the tool; the outcomes differ downstream. Claude Haiku 4.5 and Claude Sonnet 4.5 executed none of their 164 text-surface attacks, and in the text relay the canary token never appeared in a memory write (0/40); GPT-4o-mini executed 53% of its attacks. Four findings follow. (1) A Claude writer kept the canary token out of shared memory in every relay run we report; one cross-model pairing (Claude writer, GPT-4o-mini reader, n = 3) is consistent with this protecting the reader, and other pairings were not tested. (2) As readers, the Claude models executed 0/40 raw pre-seeded injections, but Claude Haiku 4.5 executed 2/3 injections relayed by GPT-4o-mini; whether relayed injections are harder to refuse than raw ones is an open question. (3) DeepSeek Chat went from 0/24 on pre-seeded memory to 8/8 on tool results, scenarios that also differ in task and payload format; white-text PDF payloads, invisible on the rendered page, succeeded at least as often as visible ones. (4) pi_detector and write_filter failed on channels they do not inspect, spotlighting failed on content it wraps, and write_filter blocked the PDF relay but not the text relay, a difference we cannot explain. Code and run logs are publicly released: https://github.com/KevinChunye/prompt_injection
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.40221v1">PhantomEnvironments: Training LLM Agents in Fictional Worlds</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Training LLM agents with reinforcement learning (RL) is bottlenecked by environments, which must provide verifiable rewards, support long-horizon interaction, and scale cheaply. Existing approaches rely on costly human-curated data or on LLM-generated environments that risk hallucinations and benchmark contamination. We show that LLMs can instead be trained into capable search agents using synthetic environments generated entirely by rules, whose generation requires no LLM and has zero marginal cost. We build PhantomEnvironments, multi-turn RL environments from fictional worlds, where agents must search a corpus of templated articles to answer multi-hop questions. Despite sharing no facts with the real world, these strikingly simple environments yield agents that transfer to real-world multi-hop search benchmarks, often outperforming real-world training data on newer benchmarks. Trained agents generalize to unseen fictional universes, and Qwen models learn to scale their search budget roughly linearly with question difficulty, suggesting emergent search scaling from environment interaction alone. Ablating environment complexity reveals that hop count drives transfer more than constraints or comparisons: even the simplest rule-generated environments are a surprisingly effective, free resource for training generalizable LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.00455v2">Towards a Belief-Based World Model for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 pre-print
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are being used as policies for autonomous decision-making and planning in many domains. Despite their strong reasoning capabilities, LLMs struggle with long-horizon tasks, especially under partial observability. World models are a promising way to enhance policy performance, both during training and inference. During inference, agents currently use world models to simulate the consequences of candidate actions before choosing an action, which can improve decision-making. However, we argue that simulation alone is an incomplete interface for decision-making under partial observability: simulation does not adequately capture uncertainty about the current state, which agents may need for accurate decision-making. We address this limitation with Belief-Based World Models (BB-WMs), which maintain a belief that LLMs can query to access information on what is known and uncertain about the current state. Before developing methods to learn accurate BB-WMs, this paper focuses on a more fundamental question: does exposing a world model's belief directly to an LLM policy improve decision-making? Our results show that giving LLM agents access to beliefs improves task performance under partial observability, while remaining complementary to existing simulation-based world models. Code: https://github.com/skumar-ml/belief-world-models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.00432v1">XOR-Trellis: Ultra-Low-Complexity Dequantization and Curvature-Aware Hadamard-Free LLM Quantization</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Trellis-coded quantization enables high-dimensional compression of large language model (LLM) weights at ultra-low bit widths without the exponentially large codebooks required by conventional vector quantization. Practical deployment, however, presents two challenges: reconstructing compressed weights at sufficient parallel throughput to avoid making dequantization an inference bottleneck, and maintaining quantization accuracy without costly incoherence transformations. We address these challenges with two complementary techniques. First, we introduce an ultra-low-complexity trellis dequantizer that uses a structured, hardware-efficient state-to-value mapping while preserving diverse reconstruction choices for trellis search. Second, we reformulate discrete trellis path optimization with a curvature-aware objective that reflects model sensitivity directly in the original coordinate space. Together, these techniques enable high-quality ultra-low-bit trellis quantization with inexpensive, highly parallel runtime reconstruction and without relying on Hadamard-based incoherence processing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.40148v1">From Spectra to Joint Schedules in LLM Pre-training: 3+3(+2) Scaling-Law Regimes</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Power-law learning curves are often treated as fixed properties of a model and its data, although learning-rate and batch-size schedules can change the observed loss. We study this dependence in noisy online SGD with linear random features. Conditional on the representation, an exact Volterra equation separates two response components: a forcing term that propagates unresolved target error and a memory kernel that propagates stochastic-error injections. We prove that either component follows a power law if and only if its cumulative weighted spectral mass has the corresponding low-spectrum scaling; individual eigenvalues and target coefficients need not obey coordinatewise power laws. Under a joint schedule, intrinsic time $T_t=\sum_{s<t}η_s$ controls optimization progress, while $r_t=B_t/η_t$ controls noise injection. Their interaction yields sharp conditions under which a schedule preserves, changes, or destroys the clean power law, together with a memory ceiling on noise reduction. The power-law random-feature model realizes this mechanism in $3+3(+2)$ propagation regimes with phase-dependent compute rates. Controlled nanoGPT experiments show that (1) learning-rate and batch-size schedules with matched $B/η$ paths are nearly equivalent in intrinsic time, (2) a forcing-memory surrogate accurately predicts loss across schedules, and (3) its fitted exponents across real-world datasets identify the regime of LLMs in $3+3(+2)$ map.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.00431v1">ChainLoRA: Geometry-Preserving Task Vector Merging for Continual Learning in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      Continual parameter-efficient fine-tuning for large language models (LLMs) must balance retention of previously acquired knowledge, adaptation to new tasks, and strict parameter budgets. We present \textbf{ChainLoRA}, a replay-free continual merging framework built on chain-updated task-vector geometry. From a parameter-merging perspective, we formulate a geometric view of forgetting through a measurable interaction between task updates, separating directional overlap from coefficient coupling. Building on this view, ChainLoRA combines chain-updated training with post-stream adaptive SVD merging. During training, initialization and a one-sided orthogonality proxy use only the last carrier, keeping their historical-state footprint and regularization overhead constant as the task stream grows. At merging time, Adaptive SVD extracts a shared carrier and aligns it to the latest task through Procrustes adaptation. Our theoretical analysis shows that Procrustes adaptation facilitates geometric approximate separation of shared and task-specific components. The one-sided proxy further bounds inter-task interference. An effective-rank penalty additionally promotes efficient utilization of the task subspace during continual learning. Experiments show that ChainLoRA achieves state-of-the-art performance among the evaluated replay-free methods on the Large and SuperNI benchmarks, while remaining competitive on Standard CL and attaining almost the closest average scores to the evaluated replay-based method across all three benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.40118v1">Persistent Context Graphs for Efficient Memory Compaction in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      As LLM capabilities advance, agents are tackling increasingly complex tasks over longer horizons. Their growing interaction histories make memory compaction essential for staying within context windows and reducing prefill cost. Existing methods summarize the history or compress its KV cache, often adding model computation to preserve information for future requests. A new user request can change which history matters, but reassessing that history with the model requires re-encoding it if the KV cache has expired. Past attention provides signals of historical importance and dependencies between messages, while relevance to the current task must be assessed using the new user request. We introduce ReCAP, a memory compaction method that stores attention-derived importance scores and dependency links in a lightweight, persistent context graph. For each new request, ReCAP combines stored importance with relevance cues from the request and follows dependency links to select messages and their supporting context, without additional model calls for selection. Compared with Codex's default summarization-based compaction, ReCAP reduces estimated latency for compaction and cold restoration by approximately 95% on both Qwen3-Coder and gpt-oss. It also roughly halves the historical context per call on SWE-Together at comparable task quality and improves accuracy on the code tasks of Lost-in-Conversation over full history by 19.8 and 41.2 points.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.40098v1">Pricing Time, Not Just Tokens: Latency-Aware Mechanism Design for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 28 pages, 1 figure
    </div>
    <details class="paper-abstract">
      The economic theory of LLM pricing treats tokens as a homogeneous commodity considering aggregate token count as the main features buyers and sellers consider. We model inference as a service market where buyers have three-dimensional private information - willingness-to-pay, task volume, and time preference - and utility depends on latency slack alongside token quantities. Our main result is a separation theorem: discrete hardware tiers induce endogenous self-selection on time preferences, reducing three-dimensional screening to standard one-dimensional screening within each tier. We derive the cost structure from GPU inference physics - compute-bound prefill and bandwidth-bound decode - and characterize optimal tiered mechanisms via virtual-value techniques. Optimal per-task prices are volume-independent, providing theoretical grounding for flat per-token API pricing. We verify the mechanism empirically by calibrating to 8-GPU clusters of H100 and B200 hardware. The separation theorem holds in 83% of 105 tested configurations overall, rising to 96% at economically relevant WTP scales. A seller adopting two-tier pricing under the optimal mechanism captures 26-66% higher profit than the best single-tier alternative, with gains driven by efficient cross-tier allocation in regimes where hardware costs are a significant fraction of per-request value.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.37172v2">SimpleEvol: An Agent-Loop Framework for LLM-Driven Automated Heuristic Design with Minimal Human Priors</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 Accepted at NeurIPS 2026. 47 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have emerged as powerful tools for automated heuristic design (AHD), enabling iterative generation and refinement of heuristics. However, the dominant paradigm embeds LLMs as narrow, fixed components, such as crossover or mutation, within heavily hand-engineered evolutionary frameworks. We argue this misapprehends LLMs. It treats them as specialized tools rather than general reasoners, constrains them to low-level operations, and underutilizes their autonomy. Moreover, the extensive human priors in these frameworks violate the bitter lesson principle that general methods scaling with computation surpass hand-crafted solutions. This raises a key question: which AHD framework designs best convert stronger LLM capabilities into better heuristics? To address this, we propose metrics for LLM-driven AHD framework handcraftedness (AHI) and intelligence conversion efficiency (ICE). Evaluating ten LLMs across three challenging combinatorial optimization problems, we obtain a notable finding that frameworks with fewer human priors consistently yield higher ICE. Based on this finding, we propose SimpleEvol, an agent-loop framework for AHD which removes nearly all human priors and allows the LLM to operate autonomously. SimpleEvol consistently achieves the highest ICE, often by a large margin. Our results challenge the trend toward complex AHD pipelines and point to a lighter and more model-centric alternative, suggesting that reducing human priors is a more effective strategy to scale up with model intelligence. The source code is available at https://github.com/HenryZhu1029/SimpleEvol-Master.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.00426v1">IrekoGPT: Turning Structured Pruning into Post-Hoc Slimmable LLMs</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 Accepted at the NeurIPS 2026 Workshop "AXIOM: Foundations of Efficient Deep Learning". 8 pages, 5 figures
    </div>
    <details class="paper-abstract">
      We introduce IrekoGPT, a post-hoc method for converting pretrained LLMs into slimmable models whose width can be adjusted at inference time. Building on SliceGPT, we retain its projection matrices without pruning them, allowing a single model to expose nested subnetworks at different widths. We improve robustness by calibrating each layer across multiple compression ratios, and correct downstream linear layers through gradient-free ridge regression. Across Llama and Qwen models, preliminary results show improvements over naive PCA-based slimming, with the largest gains at high compression. Code is available at https://github.com/aimagelab/IrekoGPT
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.39969v1">TACTIC: Temporal and Context-Aware LLM Tactical Planning for Roadside LiDAR Attacks</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Physical LiDAR attacks are often evaluated using fixed primitives and manually selected parameters, despite their strong dependence on surrounding traffic. We present TACTIC, a scene-aware framework that uses a multimodal large language model (MLLM) to coordinate state-adaptive roadside LiDAR attacks. Under a gray-box threat model, TACTIC relies only on an attacker-operated roadside perception stack, without accessing the victim LiDAR's native point clouds or internal processing. Local perception provides metric vehicle states, while the MLLM combines these measurements with roadside imagery to infer relational traffic context and construct a semantic scene graph. Based on this representation, TACTIC selects and configures two complementary primitives: \emph{push-away}, which shifts the perceived range of a lead vehicle, and \emph{phantom-obstacle braking}, which triggers emergency braking through obstacle injection. Measured traffic states and empirically calibrated constraints ground the generated tactics in physically feasible operating regions. To accommodate MLLM latency, TACTIC overlaps reasoning and execution asynchronously while high-rate local perception detects scene changes and triggers replanning. Across 280 randomized CARLA trials, the full policy achieves a 100% collision rate, compared with 35% for a fixed rule, 60% for random selection, and 75% for a restricted LLM using mode selection with default parameters. Joint physical-and-image input achieves 100% success, versus 65% with physical measurements alone and 75% with imagery alone, while asynchronous $Δ$ refresh reduces scene-mutation response from 7.4 s to 2.0 s. These results show that scene-dependent tactical planning can expose context-sensitive LiDAR failure modes that fixed attack policies may miss.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.38036v2">Gender bias across LLMs is common and highly heterogeneous</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Understanding gender biases in large language models (LLMs) is increasingly important as these systems become embedded in decision-support tools with real consequences. Prior research has focused only on a small set of models, leaving open the extent to which gender biases are common and heterogeneous across LLMs. We address this gap across ten models released between April 2025 and June 2026, spanning nine vendors, using two paradigms: gender attribution to stereotyped phrases (Study 1) and moral judgment of abuse or torture against a woman or a man to prevent a catastrophic outcome (Study 2). In Study 1, two of ten models attributed masculine-stereotyped phrases to female writers more often than the reverse, while three models showed the opposite pattern. In Study 2, several models converged on a male-disadvantaging asymmetry that was directionally consistent with a documented human tendency to protect female targets from harm, though the specific conditions under which this asymmetry emerged varied by model; three other models, by contrast, showed no variation across conditions. These results indicate that gender-related biases are common in LLMs. Their direction and magnitude, however, are highly heterogeneous, to the point that some models behave in diametrically opposite ways to others. Bias auditing should therefore be treated as an ongoing, multi-vendor process, rather than a one-time assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.39882v1">LLM Persona Unlearning</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Pre-training equips large language models (LLMs) with a broad repertoire of behavioral patterns associated with roles, styles, values, and goals. Post-training teaches conditional enactment and makes a helpful Assistant the default, but it does not erase alternative modes from the weights; explicit prompts can therefore elicit personas that repeatedly shape judgment, language, and action. In open-weight settings, runtime controls can be removed, motivating persona unlearning: a weight-level edit that makes a designated persona difficult to elicit and enact on unseen contexts. We introduce PersonaUnlearnBench, a model-specific paired benchmark spanning six LLMs from three families and five personas, with aligned forget/retain sets, held-out instruction paraphrases, and four-axis evaluation. The benchmark shows that standard unlearning methods cannot reliably erase the target persona without sacrificing meaningful generation or general utility. We therefore propose PaCE, which compares target and desirable responses to the same questions to locate an internal behavior direction, then trains target-prompt states away from the target mode and toward the matched desirable response. Experiments show that PaCE consistently suppresses target personas with high response quality and useful counterpart behavior, at moderate utility cost. These results establish persona unlearning as a distinct behavior-level editing problem and a practical route toward persistent control of latent LLM response policies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.39866v1">Preemptive LLM Unlearning against Forbidden Capability Acquisition via Gradient Sealing</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Open-weight LLMs are released not only as fixed products but also as substrates for downstream fine-tuning. This openness, however, creates legal and ethical risks because users may misuse fine-tuning to instill illicit knowledge or enable hostile operations. Model providers therefore need apre-release defense against such acquisition, motivating the problem of preemptive unlearning. Unlike retrospective unlearning, which removes capabilities already present in a fixed model, preemptive unlearning seeks to prevent their acquisition under unseen attack data and future fine-tuning procedures. Despite its practical importance, this setting remains largely unexplored, presents distinct challenges, and is therefore the central focus of our work. We first verify that existing retrospective methods provide insufficient pre-release protection. Even when forbidden capabilities are suppressed in current outputs, forbidden-domain data can still induce gradients through internal pathways, enabling later acquisition. Motivated by this finding, we propose a gradient-sealing principle that blocks these pathways by pushing relevant pre-activations into the negative region, where ReLU-family activations exhibit zero or near-zero derivatives. Experiments across multiple LLM families demonstrate our stronger resistance to downstream acquisition than retrospective baselines, validating gradient sealing as an effective mechanism for pre-release protection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.34004v2">RICE-Alpha: Reliability-Informed Correction with Event Graphs for LLM-Agent Stock Forecasting</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 15 pages, 4 figures, 4 tables. v2: Corrected corresponding author to Xiang Hu and added Tong Liu's email; scientific content unchanged
    </div>
    <details class="paper-abstract">
      Equity-relevant news evolves through temporally dependent corporate events, making historical information useful only when event continuity, information availability, and transition reliability are modeled. Existing LLM-based financial agents incorporate historical evidence, yet they provide limited support for preserving issuer-specific chronology under point-in-time constraints and for identifying when historical transitions contribute information beyond the current forecast. We present RICE-Alpha (Reliability-Informed Correction with Event Graphs), a point-in-time stock-scoring framework that separates a history-aware multi-view Base Alpha from a reliability-calibrated residual correction derived from historical event continuation. A Multi-Tier Memory Layer grounds news interpretation in temporally eligible issuer-specific history, while a Typed Event Agent constructs event states whose successor relations are formed within issuers and pooled across firms only after valid local pairing. Matured transitions are calibrated by their empirical reliability, and the resulting graph signal is residualized against the Base Alpha and technical view to obtain the RICE Delta. On daily Nasdaq-100 and Hang Seng Index panels from 2024 to 2026, RICE-Alpha achieves the strongest results among the evaluated LLM-based agents and momentum across four predictive and four portfolio-level metrics. Its ICIR more than doubles that of the strongest baseline, while net Sharpe ratios reach 1.656 and 1.725 in the U.S. and Hong Kong, respectively. U.S. ablations further show significant reductions in IC and RankIC after Holm adjustment when major components are removed. These results indicate that historical event continuation adds incremental information when it is temporally grounded, reliability-calibrated, and introduced as a residual correction to a multi-view forecast.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.29709v2">Three Ways Classical Test Theory Can Mislead About LLM Judges</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 16 pages (7 of main text), 4 figures. v2 adds the gold-computed null for all three statistics and a decision table, corrects the reading of the Livingston-Lewis difference, adopts Brennan's estimator for Phi(lambda) and revises the appendix. Code and data: https://github.com/louisyzhu/llm-judge-reliability
    </div>
    <details class="paper-abstract">
      Evaluations that use a large language model (LLM) as a judge have begun to borrow reliability statistics from classical test theory and its extensions. We examine three such statistics that need one administration and no gold labels. None of them can isolate the judge, because one judge under one prompt supplies no variance component of its own. Claude Haiku 4.5 judged 210 constructed short answers against ten-element checklists. On the 180 with parsed verdicts, the Kuder-Richardson coefficient (KR-20) came out at 0.5223 on the judge's verdicts and 0.5231 on error-free gold verdicts. In simulation, bank design alone moves KR-20 from 0.01 to 0.68 at the judge's measured 4.72% error rate. The dependability index $Φ(λ)$, a ratio of mean squared distances from the pass mark, sits 0.22 to 0.38 below the judge's accuracy against gold and returns 0.54 to 0.68 on error-free gold verdicts. Livingston-Lewis accuracy treats the rubric elements as a sample, and at a pass mark of five elements it credits error-free gold scores with 0.78, close to the judge's 0.81. A statement about the judge therefore needs gold labels or a varied scorer facet, and a reliability ratio needs the bank's spread beside it. One of the four closest judge-evaluation papers varies the prompt and still reads a reliability below 0.7 as a sign that a model cannot serve as a judge, although that reliability moves with the spread of the samples scored. We derive a decision table and four reporting lines from these two rules.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.39828v1">KUAISHOU Explorer LLM-Rec Challenge 2026: Reasoning Generative Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Generative recommendation, has been attracted a surge of attentions in industrial and academic research community, towards to build more smart system to build next-generation recommender. Under the significant developing wave of large language model, our team have been developed Semantic ID based OneRec/OneRec-V2. These models have been widely deployed in production and demonstrate the scaling potential of the autoregressive next-item prediction paradigm for industrial recommender systems. Building on the success of OneRec, we further explored a series of models, including OneRec-Think, OpenOneRec, and OneReason, that connect item Semantic IDs with natural language in a unified representation space and seek to unlock the potential of natural-language chain-of-thought (CoT) reasoning for recommendation. However, our preliminary works found that introducing reasoning CoT does not always improve the recommendation performance. To address this issue, OneReason strengthens the semantic alignment between items and language, introduces structured template-based supervision for interest reasoning, and applies advanced reinforcement learning techniques to make reasoning more beneficial to recommendation. As a frontier topic to building recommendation foundation models, we believe this topic has significant research value and hope to encourage more researchers to explore it together. To this end, together with the SIGIR 2026 community, we organized the KUAISHOU Explorer LLM-Rec Challenge 2026: Reasoning Generative Recommendation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.39807v1">Stress-Testing LLM Lie Detectors: Role-Play Failures and Spurious Correlations</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Lie detection probes aim to predict from a language model's internal states whether its output is truthful or dishonest. However, role-play complicates what "truth" means for an LLM: language models can adopt a wide range of personas that take very different claims to be true, including personas whose beliefs clearly contradict reality, such as a conspiracy theorist. In this work, we investigate whether lie detection probes reliably flag falsehoods generated under such an anti-factual persona or whether they instead follow the persona's beliefs. We introduce a dataset of 8,916 human-reviewed, on-policy responses from three LLMs adopting anti-factual personas. Evaluating eight probes from prior work, we find that many fail in this setting, particularly when correct and incorrect answers are evaluated under the same persona prompt. To investigate why, we construct three novel confounder datasets in which truth is anti-correlated with a potential confounding concept. Our experiments reveal that many existing probes strongly track concepts that are spuriously correlated with truth in their training data, such as instruction compliance or response likelihood. Based on these findings, we introduce a simple linear probe that achieves the strongest overall performance on both the persona and confounder stress tests. Our results suggest that current lie detection probes are far from reliable and highlight the need for training data in which truth is decorrelated from confounding concepts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.39786v1">Explore-on-Graph: Hybrid Embedding-LLM Reasoning for Knowledge Graph Question Answering under Incompleteness</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly combined with knowledge graphs (KGs) to ground reasoning in structured evidence. However, most LLM-based KGQA methods rely on traversing existing graph edges and become unreliable when reasoning paths are broken by missing facts. Alternatives that ask LLMs to generate missing knowledge risk introducing hallucinated evidence. We introduce XoG (eXplore-on-Graph), a framework for multi-hop question answering over incomplete KGs that recovers missing reasoning paths from learned graph structure rather than LLM parametric knowledge. XoG combines type-level entity-relation statistics to identify candidate relations with KG embeddings to retrieve plausible missing entities, using the LLM as a semantic selector and reasoner. These mechanisms are integrated into an iterative planning-exploration-reasoning process. Experiments on WebQSP, CWQ, and the Wikidata-based BRINK benchmark show that XoG remains competitive on complete KGs and consistently outperforms comparable methods without task-specific KGQA training under KG incompleteness. These gains persist across multiple LLM backbones, indicating that stronger LLMs alone do not resolve missing graph evidence. XoG also reduces LLM token consumption by up to 33% compared with a closely related planning-based approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.39783v1">COMPASS: Predicting the Relationship of Multiple Patches for Vulnerabilities with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Modern software heavily relies on code reuse, so upstream vulnerability fixes do not automatically propagate to downstream codebases. Downstream maintainers must manually adopt patches to eliminate known risks. In practice, a single vulnerability often corresponds to multiple patches, which greatly complicates downstream patch adoption because different patch relationships imply different adoption strategies. To address this challenge, we first manually inspect large-scale multi-patch vulnerabilities (about 1K) in the real world and interview experienced developers, summarizing six typical types of patch relationships, i.e., Merge, Mirror, Better Solution, Fixing-of-Fixing, Collaboration, and Separation. Based on these observations, we propose COMPASS, an automated approach that predicts the relationships of multiple vulnerability patches with large language models. Given a CVE as input, COMPASS follows a four-phase pipeline that (i) identifies the patch group and pre-scans explicit relationships, (ii) performs individual patch analysis, (iii) infers relationship instances via a hierarchy-guided prompt, and (iv) validates completeness and consistency of the inferred results. As output, COMPASS reports the predicted relationships within the patch group and visualizes them as a relationship graph. We evaluate COMPASS on a benchmark of 300 multi-patch CVEs and compare it against mainstream learning-based and LLM baselines. Results show that our method achieves strong and consistent prediction effectiveness and outperforms SOTA by 85.04% on average. We publicly release an online querying website to support community reuse of patch relationships knowledge: https://patch-relation.com.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18446v2">Using Fine-Tuned LLMs to Identify Indicators of Vulnerability in UK Police Incident Logs</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 25 pages, 4 figures. Preprint. v2: revised following peer review
    </div>
    <details class="paper-abstract">
      Purpose: Understanding how much of routine policing involves vulnerable people could inform resourcing, training, and multi-agency response, yet administrative data provide limited insight. We explore whether an LLM-based classification pipeline, developed on open-source US police data, can be adapted to estimate the prevalence of four vulnerability indicators - mental ill health, substance misuse, alcohol dependence, and homelessness - in UK police incident narratives, and when outputs can be treated as defensible measurements. Methods: We analyse nearly 3,000 de-identified incident logs from a UK police force, using a multi-stage pipeline combining repeated model inference, label aggregation, structured human review, and statistical correction. The pipeline runs on a locally hosted open-weight LLM, reflecting the secure environments police must work in. Results: LLMs can produce meaningful, if imperfect, prevalence estimates at scale. Mental ill health indicators are present in approximately one in five incidents, with lower prevalence for other indicators. However, naive LLM deployment is unreliable: single-pass classifications are unstable, and aggregated outputs systematically over-assign indicators relative to human judgement. Correcting these biases required substantial human input and statistical adjustment, leaving considerable uncertainty. Conclusions: While LLMs can extract information from unstructured police data, their outputs cannot be treated as valid measurements without careful methodological support. At the population level, defensible estimates are achievable but resource-intensive; at the individual level, errors remain frequent and unpredictable, limiting suitability for operational decisions. This study highlights both the potential and the constraints of LLM-based measurement in applied settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20641v2">Trusted Weights, Treacherous Optimizations? Optimization-Triggered Backdoor Attacks on LLMs</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 27 pages, 6 figures; v2 revised discussions
    </div>
    <details class="paper-abstract">
      Inference optimization aims to minimize the latency and resource consumption of LLM inference while preserving output quality, making large-scale deployment practical and cost-effective. However, optimized execution can introduce small numerical inconsistencies from the original model. We reveal that this inconsistency not only causes the model's outputs to diverge, but more critically can introduce hidden backdoors. The backdoor remains dormant under standard unoptimized execution and is activated only when inference optimization is enabled, allowing it to evade existing backdoor detection pipelines. We first introduce the Input-Specific Optimization Backdoor (ISOB) to demonstrate that optimization-induced differences can cause wrong predictions. However, ISOB remains input-specific and cannot establish a universal optimization-triggered backdoor. To overcome this limitation, we design the Universal Optimization Backdoor (UOB). The backdoored model stays benign under unoptimized execution but activates when inference optimization is enabled. We conduct extensive experiments across seven mainstream open-source LLMs, four tasks, and three optimization backends. UOB reaches up to 100\% attack success while largely preserving clean accuracy. To mitigate this vulnerability, we design three defense methods that reduce the backdoor ASR to at most 0.02 while preserving clean accuracy. These results reveal inference optimization as a new LLM security attack surface and motivate defenses against test-deployment disagreement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.07494v2">GeoFP8: Geometry-Aware FP8 Gradient Compression for Distributed LLM Training</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 12 pages, 6 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Gradient communication is a primary scaling bottleneck in large language model (LLM) pretraining. Communicating gradients in low-precision formats, such as FP8 and NVFP4, can significantly reduce the communication volume. Existing methods quantize gradients via linear or nonlinear mappings in Euclidean space, often degrading model performance because highly anisotropic gradients incur direction-dependent distortion. We present GeoFP8, a geometry-informed gradient scaling method that performs low-precision communication in geometry-aware coordinates. By transforming gradients into a near-isotropic space before quantization, GeoFP8 makes low-precision representations substantially more faithful to their high-precision counterparts. GeoFP8 only changes the coordinate system used for low-precision gradient communication and does not change the optimizer, training recipe, communication collective, or low-precision format. We also develop a simplified geometry-aware transformation algorithm with low-rank approximation and selective application to balance the computation overhead and communication reduction. We examine the empirical convergence of GeoFP8 using Llama-300M and Llama-600M models. Our results show that GeoFP8 reduces the end-to-end pretraining time of Llama-600M by 7.6% on 64 NVIDIA GH200 Superchips, while improving the downstream task preservation profile over direct Euclidean FP8 communication under the same optimizer and communication path.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.39701v1">Values as Style: Disentangling Values from Semantics with One-Way Mixing for Low-Damage LLM Steering</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Value steering should change an LLM's normative priorities while preserving the scenario, facts, and task constraints underlying its answer. Conventional activation edits often change both. We introduce an editable semantic-value interface on frozen residual states, with a one-way semantic-to-value pathway that grounds value recognition in context. Stop-gradient blocks feedback through this pathway; swap consistency, topic de-confounding, and decorrelation encourage selective codes. At inference, editing the value code produces a residual delta while holding the semantic code fixed. On two instruction-tuned backbones, this interface improves semantic preservation and reduces benign refusals at comparable value alignment. A matched mixing-by-gating ablation separates representation learning from selective edit activation, and dimension-matched probes establish improved code selectivity. Against validation-selected prompting on LLaMA-3.1-8B, the method achieves comparable alignment (0.750 vs. 0.748), higher BERTScore (0.938 vs. 0.923), and fewer contradictions (5.1% vs. 7.6%). Human ratings and cross-taxonomy controls provide complementary evidence for low-damage value steering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.00406v1">LLM-as-a-Judge for Low-Resource Languages: Adapting Ragas and Comparative Ranking for Romanian</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Evaluating Retrieval-Augmented Generation (RAG) systems remains a challenge for Low-Resource Languages (LRLs), where standard reference-based metrics fall short. This paper investigates the viability of the "LLM-as-a-Judge" paradigm for Romanian by adapting the Ragas framework using next-generation models (Gemini 2.5 and Gemini 3). We introduce AdminRo-Eval, a curated dataset of Romanian administrative documents annotated by native speakers, to serve as a ground truth for benchmarking automated evaluators. We compare three evaluation methodologies - direct scoring, comparative ranking, and granular decomposition - across metrics for Faithfulness, Answer Relevance, and Context Relevance. Our findings reveal that evaluation strategies must be metric-specific: granular decomposition achieves the highest human alignment for Faithfulness (96% with Gemini 2.5 Pro), while comparative ranking outperforms in Answer Relevance (90%). Furthermore, we demonstrate that while lightweight models struggle with complex reasoning in LRLs, the Gemini 2.5 Pro architecture establishes a robust, transferable baseline for automated Romanian RAG evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.39645v1">SEPAL: Separated Expert Pairs with Answer-Level Fusion for Reliable LLM Collaboration</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 22 pages, 4 figures. Code: https://github.com/zhansan114514/SEPAL
    </div>
    <details class="paper-abstract">
      Multi-agent collaboration lets large language models (LLMs) improve question answering through deliberation and feedback. Yet shared discussion couples correction with exposure to the same mistakes, which can erode the diversity needed for voting. Self-consistency offers sampling diversity without feedback, while single-pair Actor-Critic collaboration refines only one candidate. We introduce SEPAL, which assigns three private Actor-Critic teams to direct reasoning, evidence grounding, and verification. Role-specific training gives the teams different reasoning objectives beyond sampling variation. Each Critic guides revisions within its own team, preventing feedback from carrying errors across candidates. Once revision ends, majority voting combines only the final answers, keeping the reasoning histories separate until the decision. Across five open-weight backbones and five question-answering benchmarks, SEPAL improves mean accuracy by 1.81 percentage points over a matched single Actor-Critic pair, with improvements across all five backbones. Code is available at https://github.com/zhansan114514/SEPAL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19918v2">Lowest Span Confidence: Zero-Shot Hallucination Detection from a Single LLM Response</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Hallucinations in Large Language Models (LLMs), i.e., plausible but non-factual generations, pose a significant challenge to reliable deployment in high-stakes environments. However, many existing hallucination detectors require expensive repeated sampling for consistency checks or access to model-internal states unavailable in common API-based scenarios. To this end, we propose an efficient zero-shot metric called Lowest Span Confidence (LSC) for hallucination detection under minimal resource assumptions. Concretely, LSC evaluates the local confidence of adjacent complete-word spans. By selecting the lowest aggregated confidence across neighboring words whose token widths can vary, LSC captures localized uncertainty associated with factual inconsistency. This boundary-aligned smoothing reduces the global dilution of perplexity and the sensitivity of minimum token probability to isolated noise. Our main evaluation spans four model families {Llama-2, Qwen2.5, Gemma-2, Mistral} and seven benchmarks {NQ, TriviaQA, SQuAD, CoQA, HotpotQA, RAGTruth, FELM}. Additional analyses examine word reconstruction, span width, and the role of adjacency in preserving local confidence. Across these settings, LSC is competitive with methods that use multiple responses or model-internal information while requiring only one response and its output token probabilities, without training a separate detector or using an auxiliary model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.17804v2">An Empirical Study of Reward Specification and Benchmark Reliability in GRPO-based LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 29 pages, 5 figures. Code and artifacts linked in the paper. v2: Extended the held-out evaluation to include broad-topic helpfulness, replacing the terminal-training rollout analysis
    </div>
    <details class="paper-abstract">
      Practical LLM unlearning is usually evaluated through two objectives: suppress target-specific knowledge and preserve non-target utility. In generative QA, this leaves a third behavior underspecified: when a target-adjacent prompt admits a broader answer without target-specific leakage, the model should answer at that level rather than leak, evade, or refuse. We study this specification problem in a controlled LoRA-GRPO RWKU setting, comparing four reward designs that span lexical suppression, anti-refusal shaping, rubric-based broad answering, and an explicit refusal contrast, with and without SFT warm-up. The experiments show that optimization success is not equivalent to behavioral unlearning: RWKU forget scores, held-out completion audits, and training dynamics can point to different conclusions. We trace these disagreements to reward-hacking endpoints, policy-support limits in GRPO, benchmark probes that miss endpoint changes, and a rubric reward that selects broad-topic answering with low semantic leakage under held-out evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.00392v1">From A2A Attacks to Envelope-Layer Defense: Red-Teaming Evaluation of LLM Agents and a Three-Layer Isomorphic Attack-Defense Model</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Agent interaction protocols such as ACP and A2A have moved LLM-based agents toward multi-agent collaboration, introducing new security threats. A task sent by a remote peer over A2A is treated as a legitimate request, providing a natural channel for indirect prompt injection. Existing agent security evaluations mostly rely on a single metric, the attack success rate (ASR), and cannot distinguish whether an attack failed because the LLM recognized the malicious content or because a mechanism at the agent layer blocked execution. To address this, we propose A2A-TIBA, an attack principle combining indirect prompt injection with bypass circumvention. Through implant-command-exfiltration steps, it induces the target agent to deploy a callback interaction program, after which the attacker issues commands bypassing the agent. To evaluate defenses finer, we design GDA Measurement, a red-team testbed method using raw context capture via an LLM gateway, dual data preservation, and agent-based autonomous judging. We propose four attack outcomes, Class A/B/C/D, extending ASR into semantic refusal rate, semantic breach rate, interception rate, and penetration rate. Experiments reveal the envelope layer -- the channel through which malicious content enters an agent -- as a new defense dimension. We accordingly propose ELA-ITL, a three-layer isomorphic attack-defense model, dividing defense into envelope packaging, LLM recognition, and agent interception, and attack into implant channel, prompt optimization, and execution mechanism. Testing on 15 agent front-end x LLM back-end combinations and building a 1,000-case dataset verifies the attack effectiveness of A2A-TIBA, the evaluation validity of GDA Measurement, and confirms that adding malicious prompt labels to envelope packaging such as A2A, tool, and memory channels significantly improves LLM recognition of malicious content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.39437v1">Network-based Spatial Context Retrieval for Open-weight LLMs: A Faithfulness Benchmark for Grounded Geographic Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) encode substantial latent geographic knowledge, yet they reason poorly over space and are unreliable when queried from coordinates alone. Useful behaviour emerges only when structured spatial context is supplied in the prompt. This raises a question geographic evaluation has left unexamined: once the right context is supplied, does the model reason from it, or override it with its own parametric recall? We take up this question with an open pipeline for network-based spatial context retriev-al. In it, the surroundings of a selected point are defined by the pedestrian street network, the area actually reachable on foot. Using only open data and open-weight models, the pipeline retrieves features from OpenStreetMap and the GHS-POP population grid, computes indicators over the network catchment in code, and injects them as a compact spatial brief. On this basis we build a faithfulness benchmark. It labels every claim a model makes by its source (grounded in the brief, or drawn from training knowledge) and its correctness, and it probes each case with a planted false premise that the brief refutes. We evaluate sixteen open-weight model configurations across three families (Qwen, Gemma and Llama, with Gemma in two generations), four size classes and, where available, both thinking and non-thinking modes, on three con-trasting cities, resampling every case over ten seeds. The results show that resistance to the planted premise varies more strongly by model family and generation than by scale, while brief-reading competence forms a partly separate dimension. These behaviours are not captured by conventional world-correctness scores or single-shot evaluation. We release the implementation, spatial briefs, model outputs, and claim-level labels as a reproducible workflow at github.com/perezjoan/NSCR-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.04476v3">PerfReasoning: How Well Do LLMs Reason on Hardware Performance?</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Performance modeling is central to hardware design and software optimization, yet constructing these models requires structured reasoning about computation, data reuse, storage, and movement. We introduce PerfReasoning, a benchmark that evaluates LLMs both as direct performance reasoners and as generators of analytical performance-model code. Given workload, architecture, and mapping specifications, models compare mappings and predict off-chip traffic and buffer requirements. The strongest closed-source models exceed 90% on reasoning-based Q&A, and the best open-weight model reaches 82.4%. However, model construction is substantially harder: while GPT-5.6 Sol exceeds 80% pass rate, all other model configurations average below 45% and vary markedly across runs. Task-specific RL raises a 4B model's mapping-reasoning accuracy by 15.7 points, whereas feedback-free multi-round self-revision prompting is not reliably effective. PerfReasoning exposes the gap between plausible architectural reasoning and reliable performance-model construction. We will publicly release the benchmark to support reproducible evaluation and track future progress.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.39414v1">SEW: Style-Encoded Watermarking of LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 19 pages, 3 figures, 15 tables
    </div>
    <details class="paper-abstract">
      Code watermarking supports provenance tracking for code generated by LLMs. Modifying token selection to embed watermarks as an LLM generates code can create a trade-off between detectability and functional correctness. Other methods instead watermark completed code using predefined transformations or trained neural models. Recurring patterns can make watermark choices predictable across programs, while treating patterns common in unwatermarked code as watermark evidence can cause false detections. We therefore introduce SEW, which embeds and detects watermarks in already generated code through three components: (i) code style rules collected from style guides and transformation rules, with style choices determined by a secret key and each program's structural context; (ii) style-preference calibration, which evaluates watermark evidence using style probabilities estimated from human-written code; and (iii) context-aware style aggregation, which combines evidence from structurally matching locations assigned the same code style choice, preventing repeated applications of that choice from inflating watermark evidence. On CodeContests across three LLMs and three programming languages, SEW achieves a mean relative improvement of 12.44% in TPR@FPR5% over the baselines and is robust to four non-LLM code-editing attacks, with only a 0.94% mean relative decrease. Our code is available at https://github.com/suhanmen/SEW.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.25893v2">$D^2$-Monitor: Dynamic Safety Monitoring for Diffusion LLMs via Hesitation Signals</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Despite the emergence of diffusion large language models (D-LLMs) as an alternative to autoregressive large language models (AR-LLMs), safety monitoring for D-LLMs remains largely unexplored. Unlike AR-LLMs, D-LLMs generate text through a multi-step denoising process, exposing intermediate hidden representations that may contain safety-relevant information unavailable in standard single-step monitoring setups. Motivated by the suitability of lightweight probes for always-on monitoring, we analyze which trajectory-level signals best indicate when such probes are likely to struggle. We find that the most informative signal is safety hesitation: intermediate hidden states repeatedly falling within a small margin of the probe's decision boundary. The number of such hesitation steps in D-LLM's trajectory predicts probe failure effectively, providing a proxy for sample difficulty. Building on this analysis, we propose $D^2$-Monitor, in which a lightweight probe runs on every sample, producing a base prediction and a hesitation estimate that decides whether to escalate the sample to a more expressive probe. That probe is trained on trajectories with at least one hesitation step, and reads only the minimal span of those steps. Evaluated on 3 datasets (WildGuardMix, ToxicChat, OpenAI-Moderation) across 4 D-LLMs, $D^2$-Monitor achieves state-of-the-art performance with a compact parameter footprint ($\leq$ 0.93M parameters), and exhibits the best trade-off between effectiveness and efficiency relative to 8 baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.39382v1">SkillFM: Generating Skills for LLM Agents via Latent Flow Matching</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 33 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Textual skills provide reusable guidance for large language model agents, but existing approaches often rely on manually curated skill banks or reinforcement learning with indirect and delayed feedback. We introduce SkillFM (Skill Flow Matching), a generative framework that synthesizes task-conditioned textual skills directly without test-time skill retrieval. Our framework combines a codec for encoding and reconstructing textual skills in a continuous latent space with a conditional flow model trained using improved MeanFlow. At inference time, the learned velocity field enables single-step latent sampling, and an LLM-based decoder converts the sampled representation into textual guidance for a frozen downstream agent. We evaluate the framework on embodied tasks, question answering, and web shopping. On ALFWorld and Search-QA, our method achieves the best overall performance among the compared vector-based skill approaches. Our analyses further demonstrate that latent skill generation is an effective alternative to retrieval-based skill augmentation. Our code and training skill libraries are available at https://github.com/lulushang999/SkillFM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.39358v1">Working Around the Compute Ceiling: Byte-Exact Memory in Galahad Makes LLM Reading a One-Time Cost LLM Reading a One-Time Cost</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      A transformer language model performs a bounded amount of computation per token, and recent work by Vishal Sikka, former CEO of Infosys, argues that this bound limits which tasks a model can carry out or verify (arXiv:2507.07505). We ask how much of the budget beneath that ceiling is spent on work the model has already done. Serving is stateless across requests: a model that answers a second question about a document recomputes the document's attention state from the first token. On seven real-world datasets, 98.7% of prompt tokens were text the model had already read. We present Galahad, a memory layer for vLLM, SGLang and llama.cpp that makes this reading a one-time cost. Taliesin saves the model's key-value (KV) state for a block of text and loads it on the next request that contains the same bytes, instead of recomputing it. Blaise keeps the documents themselves and passes the model only the section a question needs. On a recall test with 100 facts hidden in a 97,000-token corpus (Gemma 4 31B), Taliesin alone let the model attend to the whole corpus and answered 98 of 100 on llama.cpp at 3.0 s and 572 J per question, against 10 of 100, 9.3 s and 2,754 J for the same model without Galahad, which could hold only the last 12,000 tokens. With Blaise added, the model read about 668 tokens per question and answered 100 of 100 on all three runtimes at 0.59-0.64 s and 200-213 J; a tuned RAGFlow pipeline answered 77. Storing the corpus is a one-time cost of about 100 s and 28 kJ, whose energy is recovered after 13 questions. Restored state is bit-identical: all 262,144 output logits matched after restart, rehydration and hot-load. Galahad worked with all 30 models we tested under vLLM, and it fails closed: any load that does not pass its checks is recomputed. Together these results move LLM serving from stateless to stateful inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.37054v2">ACTR: Aligning Thoughts and Responses for Multilingual Safety in Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Ensuring the safety of reasoning large language models (LLMs) across languages is essential for their reliable deployment. However, when exposed to jailbreak attacks in non-high-resource languages, these models may generate unsafe responses even when their reasoning traces identify safety risks. To address this issue, we propose aligning cross-lingual thoughts and responses (ACTR), a framework that improves multilingual safety alignment by strengthening the use of existing safety reasoning. Specifically, we first present the think gap score (TGS) to compare the normalized contributions of reasoning traces to attention outputs during response generation across languages, and use reasoning- trace substitution to measure the cross-lingual safety gap. Next, using a corpus of jailbreak queries, we assess neuron importance through changes in response representations caused by neuron masking and compare the high-importance neuron sets obtained with reasoning enabled and disabled to identify safety think neurons that support the use of safety reasoning. Finally, we devise neuron-selective consistency optimization (NSCO), which uses a frozen judge model to reward agreement between the safety categories of reasoning traces and responses while updating only the parameters associated with the selected neurons, requiring no human-annotated responses or preference data. Across two reasoning models, ACTR achieves lower average attack success rates than the evaluated state-of-the-art methods on AdvBench-X and MultiJail, with safety gains extending to unseen languages, while preserving or improving average performance on multilingual knowledge and mathematical reasoning tasks and limiting false refusals of benign requests. Warning: this paper contains examples with unsafe content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.39352v1">Hiding in Plain Sight: Decoupling Pretext from Actuation for Skill Poisoning in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      LLM agents increasingly rely on reusable Skills for complex, multi-step tasks, creating a critical supply-chain attack surface where poisoned Skill content steers agent decision loops under benign requests. Existing skill poisoning attacks either colocate actuation with its contextual pretext or distribute actuation across multiple Skills, but do not explicitly separate the rationale for execution from the operation itself. In this work, we reveal that untrusted agent decisions fundamentally depend on two conceptually distinct Risk-Realization Factors (RRFs): an actuation factor (specifying what concrete operation is performed) and a pretext factor (providing the situational rationale for why the agent must perform it). Guided by this abstraction, we propose a coordination-based attack paradigm: decoupling pretext from actuation. Rather than fragmenting the malicious actuation, we preserve it as an intact operation within a downstream Steering Skill, while delegating the pretext factor to an upstream Grounding Skill that subtly alters persistent environment artifacts through routine utility operations. The intact actuation thus hides in plain sight, appearing completely legitimate and task-driven only when evaluated against the fabricated pretext. Building on this formulation, we develop an automated framework that discovers authentic execution dependencies, synthesizes coordinated pretext-actuation skill pairs, and iteratively refines poisoned skill instructions via runtime closed-loop feedback. Extensive evaluations across single-session and persistent cross-lifecycle scenarios demonstrate that decoupled skill poisoning achieves high attack success, exposing a critical blind spot in isolated Skill security audits. Our automated framework code is available at https://github.com/Wenxin-buaa/CoordPoison.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.39346v1">Offline Guidance, Online Reasoning: Reusing LLM Feedback for Small Language Models</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 29 pages. Code: https://github.com/ZBH031/reusable-latent-correction
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer strong reasoning capabilities but are often costly to access through commercial APIs, while small language models (SLMs) are easier to deploy locally yet remain weaker in reasoning. This capability-deployment gap has motivated LLM-SLM collaboration, which aims to improve SLM reasoning using LLM capabilities while preserving the deployment advantages of SLMs. Existing approaches mainly follow two paradigms. Knowledge distillation uses LLM-generated answers and reasoning trajectories to train SLMs offline, but requires parameter updates and additional training. Alternatively, online collaboration routes difficult problems to an LLM or leverages LLM-generated guidance and corrections when an SLM encounters difficulties. Although effective, online collaboration requires repeated LLM access. Moreover, the guidance produced for a particular problem is discarded after inference and cannot benefit subsequent problems involving similar reasoning states. In the paper, we focus on a more constrained setting in which the LLM is accessed only offline, the SLM parameters remain fixed, and online inference is performed solely by the SLM. To this end, we propose Reusable Latent Correction (RLC), which converts one-off natural-language guidance from a black-box LLM into persistent corrective experiences in the hidden space of an SLM. RLC stores these experiences in an external bank and retrieves them according to the SLM's current reasoning state, enabling the SLM to reuse LLM-derived corrections during inference without any online LLM calls. Experiments across multiple reasoning benchmarks and SLM scales show that RLC consistently improves SLM reasoning without parameter updates or online LLM calls. Code is available at https://github.com/ZBH031/reusable-latent-correction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.39334v1">Taming Speculative Search for Test-Time Scaling in LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      Test-time scaling has recently emerged as a powerful approach for improving LLM reasoning by allocating additional computation during inference, substantially enhancing accuracy on challenging tasks such as mathematics and coding. To accelerate the exploration of reasoning paths, recent studies proposed speculative execution. However, we show that supporting speculative execution poses two unique challenges for LLM serving systems: (1) an explosion in the search space of candidate paths and (2) frequent, fine-grained verification tasks for candidates. To address these challenges, this paper proposes SpecScale, a serving system for efficient speculative execution. We introduce three techniques to reconcile the trade-off between latency and computational overhead: (1) early pruning of low-quality candidate paths, (2) deduplicating computation across redundant candidate paths, and (3) deferring fine-grained verification tasks. We evaluate SpecScale on challenging reasoning benchmarks, including MATH and Olympiad. Our results show that SpecScale significantly outperforms both non-speculative and recent speculative approaches, delivering substantial improvements in throughput and latency while preserving answer quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.16199v9">LLM Abstention Can Be a Prompt Artifact, in Addition to Genuine Uncertainty</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly trained to abstain from answering questions they are unsure about. However, this ability is often misapplied: in real-world applications, user prompts sometimes contain elements of uncertainty, which lead LLMs to abstain even on problems they are capable of solving. We argue that LLM abstention is not only an expression of genuine uncertainty; it can also be an artifact largely shaped by prompts. We name this phenomenon *Abstention Inflation*. We add "Unknown" as an extra option for LLMs to choose from; experiments show serious accuracy drops on True/False Questions (TFQs). Replacing "Unknown" with an unrelated random word produces a similar effect. We argue that LLMs are trained to imitate the surface pattern of abstention, rather than to express genuine uncertainty. Based on ten experimental settings, we support four claims that form a progressive argument: **(C1)** *Abstention Inflation* can be triggered by the presence of an extra option, not by genuine uncertainty; **(C2)** it makes the models deny they can answer, even when they can; **(C3)** it is a later-layer output override, as the reasoning traces and mid-layer representations preserve correct answers; **(C4)** it is not stochastic noise: it results from various factors, emerges through instruction tuning, is boosted by problems' higher difficulty, and can be mitigated at larger model sizes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.08693v3">SkillMaster: Toward Autonomous Skill Mastery in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Skills provide an effective mechanism for improving LLM agents on complex tasks, yet in existing agent frameworks, their creation, refinement, and selection are typically governed by external teachers, hand-designed rules, or auxiliary modules. As a result, skills remain external resources to be invoked, rather than capabilities that agents can develop, adapt, and internalize through experience. To endow LLM agents with autonomous skill mastery, we propose SkillMaster, a training framework that teaches agents to create new skills, refine existing skills, and select accumulated skills during task solving. This capability is achieved through three key designs. First, we train agents through trajectory-informed skill review, teaching agents to propose, update, or retain skills based on evidence from completed episodes. Second, each candidate skill edit is designed to be evaluated by its counterfactual utility on related probe tasks, providing a direct learning signal for training skill-editing decisions. Third, we introduce DualAdv-GRPO, which separately estimates advantages for task-solving actions and skill-editing decisions, stabilizing joint training across task solving and skill management. Experiments on ALFWorld and WebShop show that SkillMaster improves the overall success rate over state-of-the-art baselines by 8.8% and 9.3%, respectively, achieving the best performance among all compared methods. Further analysis reveals a marked shift in agent capability: agents trained with SkillMaster can identify skill failures, refine procedural knowledge from trajectory evidence, and transfer improvements to future tasks with limited skill-bank edits. Overall, SkillMaster moves LLM agents beyond mere skill use toward self-improving agents capable of developing, adapting, and applying their own skill repertoires.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.39302v1">SQD-Agent: LLM-driven agentic framework for Quantum Chemistry workflows</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 Accepted in IEEE International Conference on High Performance Computing (HiPC 2026)
    </div>
    <details class="paper-abstract">
      Quantum algorithms and quantum hardware are advancing towards a promising paradigm for scientific applications. However, translating domain-specific problems into executable hybrid quantum-classical workflows remains a significant barrier for application researchers due to the required expertise in quantum algorithms, nuances in quantum programming, and hardware-aware system integration. At the same time, AI and primarily LLM based agents are increasingly capable of interpreting natural-language intent, reasoning over complex workflows, and translating high-level objectives into executable code and building computational pipelines. In this work, we introduce SQD Agent, an LLM-based agentic framework that translates natural-language user intent into executable workflows for Quantum Chemistry applications where algorithms from the Sample-Based Quantum Diagonalization (SQD) family are used. By automating this translation, SQD Agent reduces the level of human expertise and configuration overhead required, thereby simplifying experimentation in hybrid quantum-classical settings for application researchers new to quantum. SQD Agent adopts a modular and extensible architecture that supports seamless integration of heterogeneous quantum backends, classical solvers, and workflow components, ensuring adaptability to rapidly evolving quantum ecosystems. The framework further incorporates interactive capabilities for on-demand profiling, bottleneck analysis, resource optimization, intelligent result caching, and convergence visualization. Key features include quantum chemistry experiments, error mitigation on real quantum hardware, together with analysis of candidate mitigation schemes in terms of their potential error-recovery behavior and computational budget, helping users understand their practical trade-offs and decide which strategies to explore in subsequent experiments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10333v2">EDGC: Entropy-driven Dynamic Gradient Compression for Efficient LLM Training</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) at scale incurs substantial communication overhead, while static gradient compression cannot adapt to gradient evolution and may degrade model quality. We propose EDGC, an entropy-driven dynamic gradient compression framework that adapts compression ranks to gradient entropy during training. EDGC combines efficient entropy estimation through gradient sampling, a theoretical model relating entropy to compression rank under a bounded-error constraint, and window-based rank adjustment across pipeline stages. Experiments on 32-V100 and 64-H100 GPU clusters training GPT2 models with 2.5B and 12.1B parameters show that EDGC reduces communication latency by up to 46.45% and end-to-end training time by 16.13%, while maintaining model quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.39279v1">Faithful Dual-constrained Erasure for Robust LLM Safety Alignment</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Machine unlearning has emerged as a crucial mechanism for removing hazardous knowledge and enforcing safety alignment in Large Language Models (LLMs). However, recent studies reveal a persistent security risk: unlearned models remain highly vulnerable to retraining attacks, where suppressed malicious behaviors rapidly resurface after benign fine-tuning. In this work, we investigate the optimization dynamics of unlearning and identify that this vulnerability stems from shallow alignment. Rather than effectively erasing target knowledge, models often exploit a shortcut by activating previously dormant parameters to act as spurious suppressors, forming a fragile inhibitory shell over intact malicious representations. To address this issue and enforce authentic memory deletion, we propose FDCU, a novel dual-constrained subspace projection framework. FDCU restricts parameter updates through a highly scalable, element-wise dual-masking rule: it preserves general knowledge manifolds via Fisher Information and strictly prohibits the abnormal activation of spurious suppressors via the Principle of Minimal Functional Intervention (PMFI). By reliably blocking the model's ability to superficially hide knowledge, FDCU promotes the authentic dismantling of target representations. Extensive experiments across specific knowledge erasure and safe output control tasks demonstrate that FDCU achieves state-of-the-art robustness against retraining attacks while maintaining near-lossless general utility, ensuring durable safety for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.16364v3">WASIL: In-the-Wild Arabic Spoken Interactions with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 Spoken Prompts, Multilingual LLMs, Speech-based Evaluation, Dialectal Speech, Low-resource Languages, Conversational AI, Speech-to-Text QA, Real-world Interaction, Spoken Language Understanding
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) voice assistants are commonly built as cascaded Automatic Speech recognition (ASR) to LLM systems, where recognition errors can distort user intent. Dislikes may also arise from ambiguous, out-of-domain, or non-request turns, making it hard to isolate ASR effects. We release WASIL (it denotes connection or linking in Arabic): in-the-wild Arabic spoken interaction prompts with audio, ASR hypotheses, assistant responses, and explicit like/dislike feedback (8,529 turns; 14.2% dislikes), plus a 2,000-turn test set covering Modern Standard Arabic (MSA) and four major dialects with their labels. We provide low-cost gold transcripts via multi-ASR agreement-guided post-editing and annotate answerability (answerable, ambiguous/needs-clarification, unsupported, not-a-request/noise) to separate intrinsic unanswerability from ASR-induced degradation. Finally, we describe scalable reference-free evaluation of responses from ASR vs. gold transcripts using multi-judge LLM scoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.39149v1">Rep2Skill: Representation-Guided Skill Self-Evolution for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Textual skills enable large language model (LLM) based agents to accumulate reusable procedural knowledge without updating model parameters. Yet existing skill evolution remains largely confined to the text space: an optimizer must diagnose success and failure patterns, and revise skills solely from long execution trajectories and sparse task outcomes. This text-only paradigm leaves the agent's internal representations, which contain rich records of its evolving execution state, outside the skill optimization loop. We ask whether an agent can improve its external textual skills by reflecting on its own internal representations. We introduce Rep2Skill, a representation-guided framework for self-evolution on agent skills. Specifically, upon the collected agent rollouts, Rep2Skill models their internal model representation trajectories to localize turns that deviate from successful execution dynamics, and it further interprets these signals alongside the execution contexts as actionable textual feedback for targeted skill revision. Experiments on two agent environments with two open-source LLMs show that Rep2Skill consistently outperforms text-only approaches in the self-evolution setting, where the same LLM serves as both executor and optimizer without a stronger external model. This establishes a promising direction moving agent self-improvement beyond text-only reflection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.39138v1">MoSE: Mode-Switching Expander for Mixed LLM Training and Inference</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 Accepted by ACP 2026, Top-Score Papers
    </div>
    <details class="paper-abstract">
      AI clusters increasingly run large language model (LLM) inference and training on the same fabric. Prefill-decode (P-D) disaggregation creates key-value (KV) cache transfers between prefill and decode groups, whereas training collectives and all-to-all traffic benefit from near-uniform global connectivity. A static sparse topology can therefore be poorly matched to one of the two traffic patterns. We present Mode-Switching Expander (MoSE), a reconfigurable expander that treats topology design as a fixed-degree edge-allocation problem. MoSE reallocates the same sparse edge budget toward direct P-D connectivity in inference-heavy modes and restores a uniform random regular expander in training-heavy modes. We evaluate MoSE using a 1024-group flow-level topology model, shortest-path routing, and two mixed workloads. Across 20 seeds, MoSE reduces average and 95th-percentile (P95) load-aware KV communication cost by 90.8\% and 91.9\% relative to Static-Training in the inference-heavy mode. In the training-heavy mode, it reduces average and P95 training communication cost by 22.7\% and 27.6\% relative to stale Static-Inference. These results show that coarse-grained topology switching can support both workload modes without additional ports or routing changes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.39131v1">Characterizing High Bandwidth Flash for LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Large language model (LLM) serving requires substantial memory to store model weights and KV caches. As models grow larger and contexts become longer, memory capacity and bandwidth increasingly become bottlenecks for serving performance. Agentic workloads compound this pressure through repeated interactions over growing contexts, making it increasingly important to retain KV state for reuse. High-bandwidth flash (HBF) offers a way to expand accelerator memory capacity for large language model (LLM) serving, but its access costs and limited write endurance complicate its use. We evaluate HBF for high-throughput agentic serving across system design and scheduling choices to understand when additional capacity improves serving performance and energy efficiency. We introduce an HBM-HBF-host hierarchical storage system and buffered cache-aware scheduling, and use trace-driven simulations to analyze their effects on performance, energy consumption, and HBF write lifetime. Across the evaluated workloads, the fastest HBF-augmented systems reduce completion time by 36.1-87.0% relative to HBM-only systems. Modeled energy savings reach 55.8%, although HBF increases energy consumption on some light workloads. Buffered cache-aware scheduling extends estimated HBF write lifetime from 4.77 to 14.82 years in the evaluated configuration. These results demonstrate the importance of coordinating data placement and scheduling to improve serving efficiency while sustaining a practical HBF write lifetime.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.37111v2">Learning from Viable Failure Prefixes: Milestone Viability Potential Policy Optimization for Long-Horizon LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Long-horizon LLM agents require reinforcement learning methods that can assign credit to intermediate decisions under sparse and delayed rewards. Existing group-based methods such as GRPO and GiGPO alleviate this issue by comparing rollout returns or repeated anchor states, but they still fail when the compared returns have no variation. We identify this failure mode as zero-credit failure: during early training, many failed rollouts contain useful prefixes, yet existing methods assign them no task-discriminative advantage. To address this issue, we propose Milestone Viability Potential Policy Optimization (MVPO), a potential-routed policy optimization algorithm that learns from viable failure prefixes. MVPO estimates prefix potential over Union-Find viability regions, repairs zero-credit groups with potential-difference advantages, and attenuates the potential branch according to relative performance progress. Experiments with Qwen2.5-1.5B-Instruct show that MVPO outperforms eight strong baselines, including GRPO and GiGPO. Under the same training length, MVPO improves over the GiGPO baseline by +4.4 success points on ALFWorld and +5.3 on WebShop, while adding only 0.16%-0.20% advantage-construction overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.39112v1">CamAgent: An LLM-Agent Framework for Multi-Species Camera-Trap Workflows</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Camera traps accumulated vast, multidimensional data for wildlife monitoring, yet translating raw media archives into meaningful ecological insights remains highly fragmented. Current research workflows require laboriously stitching together disparate analysis tools and scripts, creating steep programming hurdles and complicating end-to-end spatiotemporal analyses. To overcome this fragmentation, we present CamAgent, an autonomous Large Language Model (LLM) agent framework that integrates camera-trap analytical workflows into a unified intelligent ecosystem. CamAgent interprets natural-language ecological intent, schedules computational routing, and executes specialized tools spanning computer-vision perception (e.g., SpeciesNet), CamtrapDP-compatible data management, detection-corrected occupancy modeling, temporal activity analysis, and species co-occurrence networks. The framework automates multi-stage analytical pipelines while maintaining essential data-quality controls and analytical conventions. Consequently, CamAgent significantly reduces manual programming overhead for conservationists, establishing a transparent, scalable, and fully integrated paradigm for camera-trap ecology. Our project is available at https://anonymous.4open.science/r/artifact72c6f4.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.39076v1">Multi-LLM Collaborative Alignment via Stackelberg Games</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      A pool of language models can collaborate and improve collectively by learning from one another's responses. These interactions depend on the instructions used during training. Existing methods typically sample instructions uniformly, even though their usefulness may change as the models improve: an instruction on which models' responses once differed in quality may later be answered equally well, while a previously difficult instruction may begin to provide a useful learning signal. We propose Stackelberg Alignment, a game-theory-inspired leader-follower framework that turns instruction selection into an adaptive curriculum. An EXP3 bandit acts as the leader, allocating a fixed sampling budget across instructions and updating its sampling distribution using a reward that combines instruction difficulty and response discriminability. The language models act as followers: they respond to the selected instructions, evaluate one another's responses, and learn from the resulting preference signals through DPO or GRPO. The framework uses Elo-style reputation-weighted peer judgment and reputation-based opponent matching to support reliable and competitive model interactions. Experiments across three heterogeneous model pools and 12 benchmarks spanning scientific discovery, reasoning, code, instruction following, and knowledge show that Stackelberg Alignment achieves the highest macro-average across three diverse model pools, outperforming the strongest training-time baseline by up to 7.4% and the best static inference baseline by 12-25%. Analysis confirms that the adaptive leader concentrates duels on the most informative instructions, and ablations show that both reputation-weighted judgment and reputation-based matching improve the effectiveness of multi-LLM evolution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.39072v1">Beyond Text: LLM-Based Dimensional Emotion Evaluation in Multimodal Dialogue</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 15 pages, 6 figures, 11 tables
    </div>
    <details class="paper-abstract">
      Emotion recognition in conversation has been widely studied, but applying Large Language Models (LLMs) to continuous dimensional emotion evaluation in multimodal dialogue remains largely unexplored. We propose an LLM-based framework that performs discrete emotion recognition and Valence-Arousal-Dominance (VAD) dimensional evaluation on IEMOCAP, incorporating acoustic cues as natural language descriptions following the SpeechCueLLM approach. We evaluate six models spanning the LLaMA, GPT, and Qwen families under zero-shot prompting, few-shot prompting, and LoRA fine-tuning. LoRA fine-tuned LLaMA models substantially outperform prompt-engineered GPT models on both tasks despite GPT's larger scale, a gap we attribute to domain adaptation rather than model capacity. Our best model achieves a Valence CCC of 0.7822, a new state-of-the-art on IEMOCAP. Ablation studies confirm that textual audio descriptions meaningfully improve smaller models (+3.5 to 3.6 weighted F1) while contributing little for the largest model, suggesting audio cues are most valuable when linguistic capacity is limited. The performance asymmetry across VAD dimensions closely mirrors the annotator agreement hierarchy in IEMOCAP's own annotations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.39065v1">Can Agents Trust Their Skills? Uncovering Unsafe Chains of Trust in Skill-Based LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 26 pages, 12 tables, 8 figures, appendices
    </div>
    <details class="paper-abstract">
      LLM agents increasingly rely on installable skills, which are packages of instructions, code, and resources that equip them with task-specific capabilities and, once installed, can be automatically invoked across subsequent user tasks. This creates a chain of trust in which users delegate authority to agents, while agent frameworks admit skill-provided content into the agents' context with insufficient validation, allowing malicious skills to influence agent behavior under that delegated authority. Yet, little is known about whether this trust model adequately constrains untrusted skill content before it reaches security-sensitive operations, or how frequently such trust violations arise in real-world agents. We present TrustProbe, a framework for uncovering unsafe chains of trust in skill-based LLM agents. First, TrustProbe analyzes agent source code to identify source-to-sink call paths from skill-controlled inputs to security-sensitive operations. Second, it generates semantically realistic SKILL.md seeds with injected canaries and evolves them through feedback-guided scheduling and mutation. Finally, it validates vulnerabilities using an oracle that confirms attacker-controlled flows and verifies observable harm. Across 11 open-source agents, eight with more than 10,000 GitHub stars, TrustProbe identifies 104 taint-style vulnerabilities. Validation on a large corpus of real-world skills collected from public hubs such as ClawHub further shows that 25.1% of skill-agent trials exercise the identified vulnerable paths, with payload injection successfully weaponizing 15 of the vulnerabilities. These results reveal a systematic trust failure in skill-based LLM agents: untrusted skill content can reach security-sensitive operations and exercise authority delegated by users to their agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.39049v1">Structure vs. Chain-of-Thought: Evaluating LLM Criteria Extraction for Depression Severity</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 Extended version of a paper accepted at MHSM 2026 (IEEE ICDM 2026 workshop). 14 pages, 1 figure. Code: https://github.com/xinkaichen97/depseverity-artifact
    </div>
    <details class="paper-abstract">
      A large language model (LLM) can rate depression severity directly from a social media post or mark which clinical criteria the post shows and let code turn the count into a label. The latter is easier to audit because a clinician can check each marked criterion. We compare these approaches on two Reddit corpora using three LLMs (from 9B to frontier scale) and two questionnaires (PHQ-9, BDI-II), and measure agreement with quadratic weighted kappa. For the two frontier models, criteria extraction scores above chain-of-thought on one corpus only when its decision thresholds are fitted on labeled data. Neither model's gain is significant, with or without recalibrating chain-of-thought on the same labels. With thresholds fixed a priori from PHQ-9's criteria, extraction shows no gain on either corpus, even where models mark over two criteria per post. The 9B model behaves differently on a corpus from depression communities. It labels most posts severe, whether prompted directly or with chain-of-thought, while the a priori rule beats both without labels. After chain-of-thought is recalibrated on the same labels, no significant gap remains, consistent with a calibration effect. Yet higher ordinal agreement does not ensure better detection of severe cases. PHQ-9 criteria extraction misses most severe posts, and moving from direct prompting to chain-of-thought and then to extraction increases misses in nearly all comparisons. On the primary corpus, a relabeled stress dataset, a model using that dataset's own features, including word counts from the text, is not significantly different from frontier criteria extraction under the a priori rule.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.28005v2">Atomic and Holistic LLM Judges for Reference-Grounded Support Labels: A Prompt-Controlled Comparison</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 Accepted to AACL-IJCNLP 2026 (Main Conference)
    </div>
    <details class="paper-abstract">
      When an LLM judge only has to assign a three-way support label to a candidate answer given a reference, does asking it to decompose the answer into atomic claims help, and at what cost? We compare four single-call designs that share the judge model, the inputs, and the level of instruction detail: candidate-side atomic decomposition, a matched holistic rubric, reference-side decomposition that checks whether each reference claim is covered, and a bidirectional combination. Support labels are constructed from TruthfulQA, ASQA, and QAMPARI references (200 questions and 400 rows per dataset). All four designs are run with Opus-4.6, GPT-4.1, and Gemini Flash Lite; Sonnet-4.6 is added for the candidate-side and holistic designs. Candidate-side decomposition is weak where the label depends on completeness: the holistic rubric is more accurate on ASQA and QAMPARI for every judge while using fewer tokens. Reference-side decomposition is 12.5-21.3 points more accurate than holistic on ASQA for 28-29% more tokens, and stays near the holistic ceiling on QAMPARI at 63-69% more. On TruthfulQA misconceptions, candidate-side decomposition is competitive and significantly better for two judges. On a 60-row single-author subset whose labels are looser than strict reference completeness, candidate-side decomposition edges ahead of holistic for every judge, reversing the construction-label order. What a judge decomposes should follow what the label measures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.39001v1">The Invisible Language Tax: Token Premiums of French and Regional Languages in 2026 LLM Tokenizers, and a French-Optimized Prototype</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 11 pages, 5 figures, 7 tables. Code, tokenizer, per-sentence counts and controls: https://github.com/Baracoda-ai-labs/baracoda-fr (commit fc8a736)
    </div>
    <details class="paper-abstract">
      LLM services are billed per token and context windows are measured in tokens, yet the number of tokens needed for the same content varies across languages. We measure this token premium on seven tokenizers of widely used 2026 models (OpenAI o200k, Llama 3, Qwen3, DeepSeek V3/V4, Gemma 3, Mistral Tekken, and the Claude generation-5 tokenizer via Anthropic's counting API) on NTREX-128 (124 non-English reference translations) and on the Universal Declaration of Human Rights for regional languages. French requires 31% to 58% more tokens than English, whereas Simplified Chinese ranges from 5% fewer to 40% more and is cheaper than French on six of the seven tokenizers. Regional and overseas languages of France pay roughly 1.6 to 3.3 times the English count. We discuss how history re-sending, tiered pricing and fixed context windows amplify the absolute gap in agentic use. In a controlled experiment (BPE, Europarl, 50k vocabulary), adding French to tokenizer training data quickly reduces the premium, with diminishing returns and a growing cost for English. Finally, we present Baracoda FR v1.2, a byte-level BPE prototype with Tekken's vocabulary size. On a final test of six corpora never consulted during design, with a protocol declared fixed beforehand, it uses 11.5% fewer tokens than Tekken on French and 3.7% fewer on English; results hold after removing test sentences overlapping the training data and with an equal ordinary-token budget. It is worse on other languages and, at comparable vocabulary size, does not outperform CroissantLLM. These are segmentation results only; effects on model quality and task cost remain to be shown.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.38999v1">When LLM-Inferred User Context Adds Value in Production Streaming Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Contextual information in recommender systems is shifting from static, predefined variables toward latent representations inferred from behavior. Large language models support this shift by rendering an unstructured interaction history as a natural-language summary, which yields a thematic user context that can be encoded and used in place of an aggregate profile. The conditions under which such generated profiles outperform aggregate embeddings have received limited characterization mainly at the domain level. We evaluate semantic user-profiling strategies on a production streaming platform, ranking against the full catalog. The evaluation covers a 2*2 design space crossing representation type (aggregate or LLM-generated) with contextual scope (holistic history or attention-fused short-term and long-term contexts). The relative ordering of the two representation types is conditional on the user's consumption regime. Aggregate profiles are consistently stronger under habitual consumption, which characterizes approximately four-fifths of the population, while LLM-generated profiles are stronger for exploratory users whose subsequent interactions diverge semantically from their history. We also observe a popularity-attractor effect in LLM-generated profiles, which modestly raises within-list diversity while substantially lowering catalog coverage and reducing novelty. These results indicate that a context-aware system can select a profiling strategy from the inferred consumption regime rather than applying one representation to all users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.27287v2">SR-Fraud: An Outcome-Supervised Reflective LLM Agent Framework for Non-Stationary Payment Fraud Detection</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Real-time payment fraud detection is a non-stationary streaming prediction problem: adversaries adapt before supervised labels mature, and localized burst attacks can cause losses before retraining. Production systems typically rely on tabular classifiers and rules, which can struggle to capture these emerging sequential patterns before periodic retraining occurs. We present SR-Fraud, an outcome-supervised reflective LLM framework that decouples request-time decisions from offline adaptation. A frozen, stateless agent scores each transaction from a Hybrid Episodic Window to track behavioral shifts, while an offline reflection agent proposes boundary hypotheses from matured errors. A deterministic verifier then admits only supported hypotheses into an executable knowledge state. On a production payment-fraud benchmark, SR-Fraud improves all detection metrics over its frozen decision agent, obtains higher point estimates than static and periodically retrained CatBoost, and detects an emerging fraud burst.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.38981v1">Vosti: Specifying, Implementing, and Verifying Deterministic LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      LLM inference systems may vary batch composition, prompt chunking, prefill/decode execution, and KV-cache reuse, eviction, or recomputation. These optimizations should not affect system outputs. Production systems, including vLLM's batch-invariant mode and SGLang's deterministic mode, target this goal but lack a formal system-level specification. We formalize deterministic LLM inference: under a fixed model and deployment configuration, requests with the same prompt and initial sampler state produce bitwise-identical logits at corresponding output positions across executions. Our tests find that these production modes produce different logits under some execution variations. To address this limitation, we present Vosti, an inference engine designed and verified against this specification. Vosti chooses kernels independently of runtime engine state and ties cached KV values to their logical token prefixes. Its proof decomposes at the engine/GPU kernel boundary: a Verus inductive proof establishes that scheduling and the paged, prefix-sharing KV-cache preserve output logits, while a Triton analyzer proves bitwise-equal selected kernel outputs across batches, query lengths, and paged KV-cache layouts. Vosti produces bitwise-identical logits across every tested execution variation and achieves performance comparable to vLLM's batch-invariant mode on decode-heavy workloads, while providing a stronger, formally verified determinism guarantee.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.38976v1">Fairness Beyond a Single Run: Training-Seed Variability in Speech LLM Adaptation</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Demographic fairness gaps in automatic speech recognition are almost always reported from a single training run. We fine-tune the Q-former projector and LoRA adapters of a speech LLM at five audio compression factors and six random seeds, holding the encoder, base decoder, data and decoding fixed, and evaluate every run on Common Voice and Fair-Speech. At 460 h of clean LibriSpeech, the seed moves fairness metrics more than compression does on most demographic axes. A balanced 3x3 decomposition attributes 85.3% of the variation in Fair-Speech ethnicity normalized gap to the seed against 8.3% to compression (p = 0.009), though compression explains more on age and gender. Held-out LibriSpeech word error rate spreads by 0.04 points across those seeds while Common Voice spreads by 8.57, so these are not failed runs, and the effect survives controlling for accuracy and dropout. Scaling and diversifying the adaptation set to 960 h damps the effect but does not remove it. On Fair-Speech ethnicity, two single-run systems must differ by more than 0.30 in normalized gap to exceed seed variability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.18859v4">TwinRouterBench: Fast Static and Live Dynamic Evaluation for Realistic Agentic LLM Routing</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      LLM routing matters most in long-horizon applications such as coding agents, deep research systems, and computer-use agents, where a single user request triggers many model calls. Routing each call to the cheapest sufficient model can cut costs without sacrificing quality, yet existing router benchmarks evaluate routers only on one-shot prompts. They never expose the router-visible prefix at an intermediate agent step, never test whether a cheaper replacement preserves downstream task success, and often rely on online LLM judges at evaluation time. We introduce TwinRouterBench, a step-level routing benchmark with two tracks. The static track provides 970 router-visible prefixes from 520 instances across SWE-bench, BFCL, mtRAG, QMSum, and PinchBench, each paired with an execution-verified target tier estimated under a released downgrade-and-cascade protocol; scoring is deterministic arithmetic over tier labels, trajectory membership, and token costs, with no online evaluator-side LLM judge. The dynamic track supplies a harness that runs routers on the full 500-case SWE-bench Verified suite; in this paper we report a 100-case held-out evaluation disjoint from the static SWE supervision split. At each LLM call the router selects a concrete model from a locked pool, and success is measured by official task resolution and realized API spend. The two tracks support fast offline iteration followed by end-to-end validation under live agent execution. Code and data are available at https://github.com/CommonstackAI/TwinRouterBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.38972v1">Making LLMs Say What They Think: Measuring and Improving CoT-Interpretability Alignment</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 28 pages, 9 figures, 10 tables
    </div>
    <details class="paper-abstract">
      Chain-of-thought (CoT) traces often serve as a proxy for how Large Language Models (LLMs) arrive at their answers. However, growing evidence shows that models' CoT often fails to reflect their internal computations and can be changed without affecting their final answers. In this work, we measure and improve the alignment between the reasoning described in an LLM's CoT and what it computes internally. We propose CoT-Interpretability Alignment (CIA), a metric that measures the agreement between a model's CoT traces and its internal reasoning strategies as detected by interpretability tools. We evaluate CIA on three tasks (two-hop question answering, hint intervention, and integer multiplication) across three LLMs, finding that LLMs exhibit limited alignment across all tasks (44.8-75.9%). We then experiment with improving CIA via post-training, setting both the task accuracy and parametric faithfulness signals as a reward. Experiments show that we can substantially improve CoT parametric faithfulness while maintaining or improving the task accuracy. We provide rich analysis, such as their generalization patterns. Our work provides both a framework for auditing CoT parametric faithfulness and a pathway toward making models' explicit reasoning more trustworthy. Code and data are available at https://github.com/yihuaihong/CIA-minimal-repro.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.12890v2">Steer-to-Detect: Probing Hidden Representations for Detection of LLM-Generated Texts</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has made machine-generated text increasingly difficult to distinguish from human-written text. While recent studies explore leveraging internal representations of language models to uncover deeper detection signals, these raw features often exhibit substantial overlap between classes, limiting their discriminative power. To address this challenge, we propose Steer-to-Detect (\texttt{S2D}), a two-stage framework for detecting LLM-generated text. In the first stage, \texttt{S2D} learns a steering vector that is injected into the hidden states of a frozen observer LLM, producing representations with improved class separability. In the second stage, detection is performed via a hypothesis testing procedure based on the steered representations. We establish finite-sample, high-probability guarantees for Type I and Type II errors, providing a theoretical characterization of the procedure. Empirically, \texttt{S2D} achieves strong and consistent performance across a range of settings, including out-of-distribution scenarios and adversarial perturbations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22028v2">Contrastive Representation Shaping for LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 Accepted to NeurIPS 2026
    </div>
    <details class="paper-abstract">
      Most LLM unlearning methods aim to approximate retrain-from-scratch behaviors with minimal distribution shift, often via alignment-style objectives defined in the prediction space. While effective at reducing forgotten content generation, such approaches may act as suppression: forgotten concepts can persist in representations and remain entangled with retained knowledge. We introduce CLReg, a contrastive representation regularizer that identifies forget features while pushing them away from retain features, reducing forget--retain interference while empirically preserving the scale and shape of retain features. As light motivation for the mechanism, we provide a one-step analysis showing that CLReg decreases a simple entanglement proxy in the embedding space. Across unlearning benchmarks and LLMs of different sizes, CLReg decreases forget-retain representation entanglement to enhance mainstream unlearning methods without extra privacy risks, inspiring future unlearning work to remove forget concepts via representation shaping. Code is available at https://github.com/HaoranTang/CLReg.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.38884v1">Right Answers, Costly Models: The Efficiency Gap in LLM-based Optimization Modeling</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Optimization modeling formulates real-world decision problems as mathematical programs that solvers can use to find optimal decisions. Large language models (LLMs) can automate this process, but the resulting correct formulations can require substantial time and memory to construct and solve, limiting practical scalability. Therefore, we systematically investigate whether LLMs can identify problem structure from natural-language descriptions and apply suitable optimization modeling techniques to generate mathematical models and solver code that solve the problems correctly and efficiently. To this end, we first curate OptTips, a knowledge base of 50 expert modeling techniques in eight families. Using this knowledge, we develop OptDachshund, a multi-agent framework that transforms problems from existing optimization benchmarks into new tasks for evaluating LLMs' use of modeling techniques. It constructs conventional and expert mathematical models with solver code for the same task and data, providing baselines for correctness and computational cost. The resulting EfficientOpt benchmark contains 561 expert-reviewed tasks with paired reference implementations. Evaluation of 11 representative LLMs reveals an efficiency gap on correctly solved tasks with comparable measurements: for every LLM, most generated programs take longer to solve than their expert counterparts. Within the comparable reference-size subset, 57\% of programs with correct objective values and fewer variables and linear constraints have longer recorded solver times. Case studies show that different modeling techniques can achieve the same optimal value at similar recorded cost. Faster solving may not reduce execution time if the code takes longer to prepare data and build the model. LLM optimization modeling should therefore be evaluated for both correctness and computational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.38883v1">Toward Quantum Software Automation: A Quantum-Aware Harness for LLM-Guided Evolution</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 25 pages, 41 figures
    </div>
    <details class="paper-abstract">
      Quantum software is critical for improving the efficiency and reliability of scarce quantum hardware. However, its design still relies heavily on ad-hoc, handcrafted heuristics that are often suboptimal and quickly become obsolete as quantum hardware evolves. LLM-guided evolutionary search offers a promising way to automatically explore complex software designs, but existing search frameworks lack the quantum-specific support needed for efficient evolution: verification is expensive, feedback is sparse, and heterogeneous quantum programs require different optimization objectives. In this paper, we present QSA, a quantum-aware harness for LLM-guided evolutionary search toward automating quantum software design. QSA equips the search with three forms of quantum-specific guidance: an evolution-hardness-guided coreset and approximate scoring to reduce verification cost, static and snapshot analyses to provide fine-grained execution context, and task-specific rewards for compiler passes and runtime policies. We evaluate QSA on the IBM Quantum platform across three benchmark suites. For multiprogramming, QSA improves QPU utilization by 4.2%-9.5% and Hellinger fidelity by 15.2%-19.5% over the state of the art. For error mitigation, QSA reduces mitigation time by at least 96.8% while achieving comparable or better fidelity. These gains require only $6.9 in LLM API cost over 11.3 hours.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.38866v1">When Context Changes: Understanding Update Failures in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      As preferences, goals, and facts change, LLM agents must use the current state while earlier versions remain in context. Yet they can answer with an old value of the same variable, a failure that we call stale binding. To study when models use outdated information and why, we introduce Controlled In-Context Memory (CICM), a benchmark for tracking and using updated information in conversations and agent logs. We observe that even frontier reasoning models can fail to recover the current state. We find that in open-source models probes can still recover the updated value when the model answers with an old one, pointing to a failure to select information that remains available. Component tests in Qwen and Pythia identify a mechanism for this selection failure: attention drift, where attention favors old values over the current one when producing an answer. We study a one-layer transformer to mathematically understand how this phenomenon happens: when attention scores are similar, several old values can together receive more attention than the current value. Guided by this explanation, we redirect attention toward the current value without further training. When the current value is requested directly, adjusting this intervention for each input corrects most old-value errors across various model families while preserving nearly all initially correct answers. Reliable context management therefore requires more than remembering updated information: models must use it to guide their answers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.38860v1">Optimal Design for Active Preference Learning with Biased LLM Judges</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 35 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Learning from human preferences is central to large language model (LLM) alignment, but human preference annotation is costly. Active preference learning reduces this cost by selecting informative comparisons, and LLM judges can provide additional scalable feedback. However, the preferences of the judges may deviate from those of the target human population. Even after calibration on trusted reference data, active acquisition can shift the comparison distribution and expose residual judge bias. We therefore incorporate judge deviations into the acquisition design rather than relying on a separate calibration stage. Under joint estimation, comparisons that appear highly informative about the reward may also reflect judge bias and therefore provide less information about human preferences. To address this issue, we propose Nuisance-Adjusted Optimal Design (NAOD), a comparison-selection strategy that prioritizes policy-relevant target information after nuisance adjustment and uses the Frank-Wolfe algorithm for optimization. Theoretically, we establish a sharp conditional local asymptotic minimax lower bound on policy risk and construct an estimator that attains it. We further characterize the finite-sample cost of learning the nuisance representation and show that representation error can reverse an oracle design advantage. Finally, we validate these predictions experimentally and evaluate NAOD on Chatbot Arena data across 17 judges, 15 budget configurations, and 15 random cluster-level splits. NAOD reduces the mean regret of proxy policy by 29.1% relative to a matched target-information design, outperforms existing methods, and improves human-preference prediction on held-out data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.10027v2">Speech-based Psychological Crisis Assessment using LLMs</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 5 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Psychological support hotlines provide critical support for individuals experiencing mental health emergencies, yet current assessments largely rely on human operators whose judgments may vary with professional experience and are constrained by limited staffing resources. This paper proposes a large language model (LLM)-based framework for automated crisis level classification, a key indicator that supports many downstream tasks and improves the overall quality of hotline services. To better capture emotional signals in spoken conversations, we introduce a paralinguistic injection method that inserts identified non-verbal emotional cues into speech transcripts, enabling LLM-based reasoning to incorporate critical acoustic nuances. In addition, we propose a reasoning-enhanced training strategy that trains the model to generate diagnostic reasoning chains as an auxiliary task, which serves as a regulariser to improve classification performance. Combined with data augmentation, our final system achieves a macro F1-score of 0.802 and an accuracy of 0.805 on the three-class classification task under 5-fold cross-validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.38830v1">SparLeak: Privacy Leakage from Sparse Attention in LLM Inference on Shared GPUs</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Sparse attention is widely used to accelerate long-context inference in modern large language models (LLMs), but its input-dependent execution behavior introduces previously unexplored privacy risks. We identify a new GPU micro-architectural side channel, termed Sparsity-Induced Memory Access (SIMA), which arises from secret-dependent key-value cache access patterns induced by sparse attention. Based on this observation, we present SparLeak, a phase-aware side-channel attack that extracts SIMA traces during LLM inference and enables two practical privacy extractions: query attribute inference from prefill-phase traces and autoregressive response reconstruction from decoding-phase traces. By reconstructing approximate token-level sparsity profiles from page-level observations and applying profiling-based learning, SparLeak accurately recovers sensitive information, including user-query attributes and private LLM response content. Extensive evaluation across three LLM architectures, three sparse attention mechanisms, and three privacy-sensitive datasets shows that SparLeak achieves average attack success rates of 90.9% for attribute inference and 87.3% for response reconstruction under real-world LLM serving settings, highlighting the significance to account for SIMA leakage when deploying sparse-attention-based LLM systems. We provide anonymized SIMA traces, trained attack models, evaluation scripts, and documentation as artifacts at https://anonymous.4open.science/r/Janus_artifacts/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.38829v1">Diversity Combining for Multi-Path LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 Accepted by NeurIPS 2026
    </div>
    <details class="paper-abstract">
      Multi-path reasoning methods such as self-consistency (SC) sample $K$ reasoning paths and choose the most frequent answer. However, their gains quickly plateau as $K$ increases, and existing methods do not predict when this saturation will occur. We formalize multi-path LLM reasoning as a diversity combining problem from wireless communications: each path is a noisy channel observation, and the pairwise correlation of path correctness caps the design-effect effective sample size of the vote at a finite ceiling. Generalized least squares (GLS) analysis shows that, under exchangeability, the optimal symmetric linear combiner of latent embeddings is uniform, supporting majority vote as the natural default in standard SC while leaving room for weighting or pruning under heterogeneous prompt-template branches. Across 5 models and 12 benchmarks, prompt-template diversity reduces path correlation in $55$ of $57$ valid cells, with the strongest effect on open-ended QA. We derive an Adaptive-K rule that uses a four-path pilot to select $K^*$, retaining $96$--$103\%$ of MV@$K{=}32$ accuracy across Math, QA, and NLU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.38818v1">Whose Voice Survives the Summary? A Voice-Retention Audit of LLM Employee Listening</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 10 pages, 3 figures, 3 tables. Accepted at the 60th Hawaii International Conference on System Sciences (HICSS 2027)
    </div>
    <details class="paper-abstract">
      Organizations increasingly route employee feedback to leaders through large language model (LLM) summaries, an unaudited layer that silences already-spoken voice. We introduce a Voice Retention / Representation Ratio metric for representational bias in summarization and apply it to a bilingual (English/German) corpus of 2,586 free-text responses from a global professional service company. First, employees supply criticism more reliably than praise (withholding praise is 82 times more common). Second, across 45 leader-summaries the pipeline filters by popularity, not sentiment: criticism survives, yet a concern voiced once is dropped 86% of the time, with short and German-only content lost on the same axis (theme retention 0.14 vs 0.74; German directional). Controlling for frequency, sentiment has no independent effect; the harm is prevalence-driven, which sentiment-only audits miss. A targeted prompt recovers only named themes. We contribute the metric, field evidence, and a disaggregated voice-retention card.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.38816v1">You're Hired: Strategic Model Selection for LLM Collaboration</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 21 pages, 10 tables, 5 figures
    </div>
    <details class="paper-abstract">
      While multi-agent and model collaboration algorithms gain traction to combine the strengths of diverse Large Language Models (LLMs), existing systems remain bottlenecked on pre-defined and hand-crafted model pools. In this work, we investigate the problem of model selection in multi-LLM systems. We propose and systematically evaluate a taxonomy of 9 selection algorithms ranging from diversity of model descriptions, capability-aware behavioral diversity, and LLM-based recruiters. We conduct extensive experiments across two candidate pools of 10 and 32 models, deployed in four model collaboration algorithms, and evaluated across tasks spanning math, coding, QA, and reasoning. Results demonstrate that successful selection algorithms greatly outperform random or heuristics-based teams such as merely selecting the models with top individual performance, by up to 36.1% across settings. Specifically, capability- and training-based selection strategies alleviate selection variance and achieve the best performance, which we recommend to employ before deploying real-world multi-LLM systems. Further analysis reveals that larger candidate pools pose greater challenges to shallow selection heuristics, while algorithms grounded in interacting with candidate models and understanding model capability robustly filter out misaligned, unsafe models, as well as generalizing to novel, out-of-distribution tasks. Together, we establish that principled and informed team selection is critical and present strong model selection algorithms for assembling effective multi-LLM systems.
    </details>
</div>
