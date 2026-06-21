# llm - 2026_06

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
