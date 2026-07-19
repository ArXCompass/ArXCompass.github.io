# llm - 2026_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- Part 6

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26666v2">PersistentKV: Page-Aware Decode Scheduling for Long-Context LLM Serving on Commodity GPUs</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 7 pages, 3 tables; workshop paper
    </div>
    <details class="paper-abstract">
      Autoregressive large language model (LLM) serving is increasingly limited by key-value (KV) cache movement rather than dense matrix multiplication. Modern paged-attention systems reduce fragmentation, and mature kernels like FlashInfer provide highly optimized decode attention. However, the best single-kernel implementation is not always the best serving schedule: low-active long-context decode can under-utilize GPUs, while mixed sequence lengths introduce tension between many exact-length launches and coarse padded batches. We present PersistentKV, a native block-table decode attention engine and page-aware scheduling study for grouped-query attention (GQA). PersistentKV maps work by KV-head group, executes directly over native page tables, and adds a compact workqueue schedule executing only non-empty row-KV-head-sequence-split tasks. On an RTX 3060 (FP16, page size 16, Hq=32, Hkv=8, d=128), a calibrated roofline-style policy selects FlashInfer for small active batches, PersistentKV sequence splitting for batch size 1 (B1) long-context steps, and PersistentKV workqueue scheduling for supported B8 long-context GQA steps. With cost-model constants fixed on calibration traces, five held-out seeds improve mean wall decode-token throughput by 1.04x to 1.08x on B8 bimodal, uniform, and Zipf-like workloads, and by 1.40x on a B1 bucketed trace. For the B4 boundary case and uncalibrated GQA ratios, the policy avoids regressions by routing to FlashInfer. We also report an attention-plus-MLP timing proxy and workload counters showing workqueue scheduling reduces launch fan-out from 16.00 to 2.00 launches per step on held-out bimodal B8. These results show that work assignment is a decisive serving-system variable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17041v5">Benchmarking LLM Agents on Meta-Analysis Articles from Nature Portfolio</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 13 pages, 7 figures, preprint for arXiv, dataset and code available at https://github.com/BFTree/MetaSyn
    </div>
    <details class="paper-abstract">
      Meta-analysis is a demanding form of evidence synthesis that combines literature retrieval, PI/ECO-guided study selection, and statistical aggregation. Its structured, verifiable workflow makes it an ideal substrate for evaluating systematic scientific reasoning, yet existing benchmarks lack ground truth across the full retrieval-screening-synthesis pipeline. We introduce MetaSyn, a dataset of 442 expert-curated meta-analyses from Nature Portfolio journals. Each entry pairs a research question with PI/ECO criteria, a retrieval corpus of 140k PubMed articles, verified positive studies, hard negatives that are topically similar but PI/ECO-ineligible, and complete search strategies and date bounds. Benchmarking twelve pipeline configurations (nine RAG variants and a protocol-driven agent) reveals a critical screening bottleneck: despite a retrieval ceiling of 90.9% recall at K=200, no system recovers more than 52.7% of ground-truth included literature. Current LLMs fail to reliably separate eligible studies from PI/ECO-failing distractors in pools of comparable topical relevance. Stage-attributed metrics capture where systems succeed and fail; a single end-to-end score does not.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.10540v3">FusionFactory: Fusing LLM Capabilities with Multi-LLM Log Data</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has created a diverse landscape of models, each excelling at different tasks. This diversity drives researchers to employ multiple LLMs in practice, leaving behind valuable multi-LLM log data. This naturally leads to the question of whether such logs can be fully leveraged to fuse LLMs' complementary capabilities. Although prior work has explored various strategies for integrating multiple LLMs, we argue that practical fusion must meet two essential requirements: (1) compatibility with real-world serving scenarios (e.g., local and API-based serving), and (2) flexibility to operate at different stages of the LLM pipeline to meet varied user needs (e.g., fine-tuning and inference stages). To this end, we introduce LLMFusionBench, a large-scale benchmark for LLM fusion that spans 14 tasks across five domains, with responses from 20 open-source LLMs (8B--671B) totaling 103M tokens. Building on LLMFusionBench, we propose FusionFactory, a systematic framework with three elaborated levels: (1) query-level fusion via tailored LLM routers, (2) thought-level fusion leveraging retrieved abstract reasoning templates, and (3) model-level fusion via distillation from top-ranked responses. Experiments show that FusionFactory consistently outperforms the best individual LLM across all 14 benchmarks, with the optimal fusion configuration varying across benchmarks, highlighting the promise of multi-LLM log data as a practical foundation for fusing diverse LLM capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.24661v3">Measuring Reasoning Quality in LLMs: A Multi-Dimensional Behavioral Framework</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      Despite remarkable progress on reasoning benchmarks, current LLM evaluation practice remains anchored to final-answer correctness, providing limited insight into how models reason, how reliably they behave under contextual variation, or how efficiently they reach conclusions. This paper proposes a unified multi-dimensional framework for measuring LLM reasoning quality from a behavioral perspective, operationalizing six theoretically grounded dimensions rooted in cognitive science: Correctness (CQ), Consistency (CS), Robustness (RS), Local Logical Coherence (LS), Efficiency (ES), and Stability (SS). The framework introduces deployment-aware aggregation, enabling context-specific model selection beyond accuracy-based leaderboards. Experiments across multiple LLMs and benchmarks reveal behaviors systematically concealed by single-metric evaluation, including the orthogonality of local logical coherence and correctness, deployment-context-dependent ranking inversions, and non-trivial dimensional profiles in small locally-deployed models. Discriminant validity analysis confirms that the proposed dimensions capture largely non-redundant signals. The resulting pipeline provides a foundation for diagnosing LLM reasoning behavior across deployment contexts, with domain-specific validation as a direction for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.31163v2">ComplianceGate: Classifier-Gated Multi-Tier LLM Routing for Inference in Regulated Industries</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      Large language models deployed in regulated industries operate under two constraints: compliance enforcement and cost efficiency. Personally identifiable information (PII) in user queries can reach model endpoints before the system determines whether that data should leave its jurisdictional boundary. Serving all queries through a single large model consumes full GPU capacity regardless of query complexity while offering no mechanism for geographic routing. Mixture-of-Experts architectures do not address this routing occurs between expert layers within the model after data has already arrived at the endpoint, with all experts loaded in memory regardless of query complexity. We propose a classifier-gated routing architecture that enforces compliance by design. A trained encoder classifier sits before any decoder inference, evaluating each query for complexity and data sensitivity, then routing it to an appropriately sized dense model in the appropriate geographic location. PII-containing queries route to local endpoints before any LLM computation begins, making data residency violations structurally impossible. Simple queries reach small, fast models at a fraction of the cost. Our evaluation on 600 queries demonstrates 39% median latency reduction, 33-52% cost savings depending on query distribution, and generation throughput of 122-200 tokens/second versus 50-64 for the baseline. The encoder classifier achieves 99.2% accuracy with near-perfect PII recall at 7ms inference overhead, establishing pre-inference classification as a practical path to compliance-by-design LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.00596v1">Semantic-Guided Reading Order Reconstruction in Historical Armenian Newspapers with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 International Conference on Pattern Recognition, 2026, Lyon, France
    </div>
    <details class="paper-abstract">
      This paper addresses reading order reconstruction in historical Armenian newspapers, which combine complex layouts with limited language resources. We introduce a new annotated dataset of 66 pages and compare geometric heuristics, YOLO-based layout parsing, an end-to-end document model ECLAIR, and a hybrid method combining semantic zone detection with a generative LLM. Our hybrid method achieves the lowest error rates of all evaluated approaches, reducing ordering errors by up to 76% over the strongest geometric baseline, and remains robust in multi-page settings and under noisy OCR. Rather than targeting production the method is designed as a data bootstrapping strategy enabling rapid annotation in highly under-resourced scenarios. Alongside the dataset, we release a specialized Tesseract OCR model for historical Armenian print.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.19294v5">Maximizing Mutual Information Between Prompt and Response Improves LLM Performance With No Additional Data</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 International Conference on Machine Learning 2026
    </div>
    <details class="paper-abstract">
      While post-training has successfully improved large language models (LLMs) across a variety of domains, these gains heavily rely on human-labeled data or external verifiers. Existing data has already been exploited, and new data is expensive to collect. Moreover, true intelligence goes far beyond verifiable tasks. Therefore, we need self-improvement frameworks that are less dependent on external signals and more broadly applicable to both verifiable and non-verifiable domains. We propose **Mutual Information Preference Optimization (MIPO)**, a contrastive data augmentation method that constructs preference pairs by generating a positive response conditioning on the correct prompt, and a negative response by conditioning on a random, unrelated prompt. We show that using Direct Preference Optimization to learn from this paired data maximizes pointwise mutual information *under the base LLM* between prompts and model responses. Experiments with with 1-7B parameter Llama and Qwen instruct models show that MIPO achieves 3-16% gains (and 51% increase for Qwen2.5-1.5B-Instruct) on personalization compared to prompting baselines. Surprisingly, MIPO can also be useful in verifiable domains, such as math and multiple-choice question answering, yielding 1-20% gains *without any additional data or external supervision*. These results suggest a promising direction for self-improvement using intrinsic signals derived from contrastive data pairs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.00555v1">Rise From The Ashes: LLM-based Static Analysis for Deep Learning Framework Bugs</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      Deep learning (DL) frameworks are critical AI infrastructures that often hide bugs with serious security implications. While dynamic approaches such as fuzzing are effective in uncovering these bugs, they require real test execution and incur high computational costs. Static analysis is a natural complement because it can detect bugs without runtime execution, offering fast and scalable testing. Unfortunately, there is still limited work targeting static analysis for DL frameworks due to their multilingual architectures and tensor-related program state. We present Phoenix, the first LLM-based static analysis technique for DL frameworks. Our key insight is that cross-language tensor flows in DL frameworks can be modeled, together with concrete code context, as a structured semantic bridge intermediate representation (SBIR) that LLMs can analyze for potential bugs in tensor semantic propagation. We implement this insight through a multi-agent workflow. A summarization agent first distills bug summaries from historical bug-fix patches and CWE rules. Guided by each summary, an extraction agent identifies bug-relevant repository symbols for code retrieval, and a generation agent synthesizes grounded SBIRs from the retrieved context. Finally, an analysis agent is leveraged to check SBIRs and report potential bugs. Our evaluation shows that Phoenix is a practical complement to dynamic DL framework testing for bug finding. To date, Phoenix has found 31 real new bugs in PyTorch for different heterogeneous hardware backends (Intel CPU, NVIDIA CUDA, and Apple MPS). Among them, 20 submitted bug-fixing patches have been merged into upstream.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.00501v1">BaseRT: Best-in-Class LLM Inference on Apple Silicon via Native Metal</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      We present BaseRT, a native Metal inference runtime for large language models (LLMs) on Apple Silicon, and report the highest inference throughput on this hardware to date. Existing runtimes, including llama.cpp and MLX-based frameworks, incur overhead from abstractions not designed for Metal's execution model or Apple Silicon's unified memory topology. By building natively on Metal with chip-specific kernel fusion, unified memory-aware optimisation, and custom dispatch logic, BaseRT recovers performance that framework-based approaches leave on the table. BaseRT supports a wide range of model families across eight quantisation formats (Q2 to FP16) on all Apple M-series devices. In this paper, we evaluate the Qwen3, Llama 3.2, and Gemma 4 families at Q4 and Q8 quantisation on M3 and M4 Pro devices. BaseRT achieves up to 1.56x higher decode throughput than llama.cpp and up to 1.35x higher than MLX, with substantially larger margins on prefill for mixture-of-experts models, delivering consistent best-in-class throughput from sub-1B to 30B parameter models. These results establish Apple Silicon as a more capable inference platform than previously reported, with direct implications for the emerging edge inference paradigm: as privacy requirements, latency constraints, and cloud cost pressures drive inference toward on-device deployment, performance-optimised local runtimes are a critical enabling layer for this transition. BaseRT is publicly available at https://github.com/basecompute/baseRT
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01277v1">Cognitive Firewall: A Proactive, Zero-Trust, Multi-Gate Framework for LLM Safety</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can be induced to produce harmful content through multi turn strategies in which no single user message appears clearly unsafe. Existing runtime safeguards commonly evaluate prompts or responses as isolated messages, which limits their ability to recover ac-cumulated intent, verify asserted authority, or detect harmful objectives decomposed across a dialogue. This paper presents the Cognitive Firewall, a proactive runtime oversight framework that interposes an independent oversight model between a user and a protected target mod l. The framework decomposes safety assessment into four categorical gates: an intent gate that identi-fies the operational objective of a request, a zero trust context gate that treats claimed roles and permissions as unverified evidence, a consistency gate that detects escalation and decomposition across turns, and an output risk gate that inspects candidate responses before release. Gate decisions are combined through escalation rather than score averaging, allowing any confident danger signal to block an interaction while preserving an auditable rationale. Experiments on four jailbreak benchmarks and a benign safety test set show that the Cognitive Firewall substantially reduces attack success across single turn, multi turn, authority based, and human crafted attacks. It lowers attack success to 2 percent or below on three attack sets and to 14 percent on the most difficult human crafted set, while maintaining an 8 percent over refusal rate. These results indicate that decomposed, conversation level oversight can improve proactive containment and auditability for LLM safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.00481v1">Beyond the Prompt: Jailbreaking Function-Calling LLMs via Simulated Moderation Traces</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      Jailbreak attacks remain a critical threat to the safe deployment of large language models (LLMs). While prior work has primarily studied attacks and defenses at the prompt level, we show that this prompt-centric paradigm overlooks a structural vulnerability in stateful, function-calling environments. In such applications, developer-defined schemas, structured arguments, and untrusted tool outputs are interleaved into a single shared model context. This architecture expands the attack surface by blurring the boundary between trusted control logic and untrusted data, allowing adversarial intent to be distributed across a multi-turn execution path. We exploit this architectural flaw through SMT, a black-box attack framework based on Simulated Moderation Traces. Departing from purely prompt-based interactions, SMT constructs a multi-turn trajectory that simulates a legitimate moderation-auditing workflow. Within this trajectory, a fabricated moderation frame leverages red-team testing as a pretext to elicit harmful generations. The subsequent validation feedback treats safety refusals as execution failures, prompting refinements that gradually weaken the model's safety constraints and ultimately trigger harmful outputs. Extensive empirical evaluations on prominent commercial LLMs from five different providers across two standardized safety benchmarks show that SMT consistently achieves the highest average attack success rate and HarmScore while requiring a near-minimal number of queries, substantially outperforming existing baselines. These findings demonstrate that prompt-level sanitization alone is fundamentally insufficient for defending tool-enabled LLM systems and highlight the urgent need for context-aware validation across schemas, arguments, tool outputs, and accumulated conversation state. The code is available at https://github.com/liujlong27/SMT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13349v2">When Less Latent Leads to Better Relay: Information-Preserving Compression for Latent Multi-Agent LLM Collaboration</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      Communication in Large Language Model (LLM)-based multi-agent systems is moving beyond discrete tokens to preserve richer context. Recent work such as LatentMAS enables agents to exchange latent messages through full key-value (KV) caches. However, full KV relay incurs high memory and communication cost. We adapt KV-cache eviction methods to this setting and introduce \textbf{Orthogonal BackFill (OBF)} to mitigate information loss from hard eviction. OBF injects a low-rank orthogonal residual from discarded KV states into the retained KV states. We evaluate OBF against full KV relay on nine benchmarks spanning mathematical reasoning, expert and commonsense QA, and coding. With only 9.9%-20.2% of the prompt KV states retained, H-OBF delivers between $97%$ and $120%$ of full KV relay's per-benchmark accuracy across the nine benchmarks. This suggests that more information does not necessarily lead to better communication; preserving the most useful information matters more. Our codebase is included in the supplementary material. Our codebase is publicly available on https://github.com/markli404/When-Less-Latent-Leads-to-Better-Relay.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.00454v1">Agri-SAGE: Simulation-Grounded Multi-Agent LLM for Context-Aware Agricultural Advisory Generation</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      Agricultural advisory systems face a fundamental tension: static agronomic guidelines offer consistent, evidence-based recommendations, yet remain blind to in-season variability and dynamic uncertainties. Recent advisory systems powered by LLMs are liable for a different risk of generating recommendations that are agronomically credible but physiologically unconvincing. Agri-SAGE is a closed-loop framework designed to resolve the above two limitations by integrating retrieval-grounded multi-agent LLM reasoning with APSIM-based biophysical simulation, to generate and validate agronomic advisories. To assess this framework, we evaluate three reasoning approaches, namely Plan-and-Solve, Tree of Thoughts, and Reflexion, over a 10-year retrospective analysis. All three significantly outperform static PoP (Package-of-Practice) baselines, with Tree of Thoughts achieving impressive peak yields. At the same time, Reflexion achieves comparable agronomic outcomes at substantially lower computational cost by leveraging cross-seasonal episodic memory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.00448v1">Real-Time Hard Negative Sampling via LLM-based Clustering for Large-Scale Two-Tower Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      The two-tower model has been widely used for large-scale recommendation systems, particularly in the retrieval stage. Industry standards for training two-tower models typically involve in-batch and/or out-of-batch negative sampling. However, these methods often produce easy negatives that models can quickly learn, failing to sufficiently challenge the model. To address this issue, a novel self-supervised hard negative sampling technique is proposed that leverages a large language model (LLM) to generate hard negatives from the same cluster during model training. By utilizing the LLM to learn media representations, the proposed approach ensures that the generated negatives are more challenging and informative. This real-time sampling framework is designed for seamless integration into production models, capable of handling billions of training data points with minimal computational complexity. Experiments on public datasets, along with deployment to a large-scale online system, demonstrate that the proposed negative sampling technique outperforms widely used industry methods. Furthermore, analysis in industrial applications reveals that this sampling method can help break inherent feedback loops in recommendations and significantly reduce popularity bias.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.17314v4">Clotho: Measuring Task-Specific Pre-Generation Test Adequacy for LLM Inputs</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 FSE 2026 Camera Ready version
    </div>
    <details class="paper-abstract">
      Software increasingly relies on the emergent capabilities of Large Language Models (LLMs), from natural language understanding to program analysis and generation. Yet testing them on specific tasks remains difficult and costly: many prompts lack ground truths, forcing reliance on human judgments, while existing test adequacy measures typically rely on output uncertainty and thus are only available after full inference. A key challenge is to assess how useful a test input is in a way that reflects the demands of the task, ideally before even generating any output. We introduce Clotho, a task-specific, pre-generation test adequacy measure that estimates input difficulty directly from LLM hidden states. Given a large pool of unlabelled inputs for a specific task, Clotho uses a Gaussian Mixture Model (GMM) to adaptively sample the most informative cases for human labelling. Based on this reference set the GMM can then rank unseen inputs by their likelihood of failure. In our empirical evaluation across eight benchmark tasks and three open-weight LLMs, Clotho can predict failures with a ROC-AUC of 0.716, after labelling reference sets that are on average only 5.4% of inputs. It does so without generating any outputs, thereby significantly reducing LLM execution costs compared to output-based uncertainty or confidence measures. Comparison of Clotho and these post-generation adequacy measures shows that the two approaches complement each other. Crucially, we show that adequacy scores learnt from open-weight LLMs transfer effectively to proprietary models, extending the applicability of the approach. When prioritising test inputs for proprietary models, Clotho increases the average number of failing inputs from 18.7 to 42.5 out of 100, compared to random prioritisation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17673v6">Bridging Symbolic Control and Neural Reasoning in LLM Agents -- The Structured Cognitive Loop</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 This update clarifies the theoretical architecture by separating Regulation as the Soft Symbolic Control layer from Control as a deterministic runtime engine, while adding explicit discussion of how the current implementation should be interpreted in light of that distinction
    </div>
    <details class="paper-abstract">
      Large language model agents suffer from architectural fragilities such as entangled reasoning and execution, memory volatility, and uncontrolled action sequences. We introduce Structured Cognitive Loop (SCL), a modular agent architecture that separates cognition into Retrieval, Cognition, Control, Action, and Memory (R-CCAM). SCL introduces Regulation as a dedicated governance layer through which Soft Symbolic Control applies symbolic constraints to probabilistic inference, while Control remains a distinct deterministic runtime engine for duplicate-call prevention, error limits, and termination judgment. Through multi-step conditional reasoning experiments, we show that SCL achieves zero policy violations, prevents redundant tool calls, and maintains complete decision traceability. We position SCL within hybrid intelligence, distinguish it from prompt-centric, memory-only, and neuro-symbolic approaches, and derive three design principles for trustworthy agents: modular decomposition, adaptive symbolic governance, and transparent state management. With an open-source implementation and a live GPT-4o-powered travel planning agent, this work offers a practical path toward reliable, explainable, and governable LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.00403v1">A Penny for Your Prompts: Experiments Detecting and Mitigating LLM Usage by Survey Respondents</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 Published at SOUPS 2026 (Symposium on Usable Privacy and Security)
    </div>
    <details class="paper-abstract">
      Large language models are increasingly used by participants on crowdsourcing platforms when responding to surveys, potentially undermining the validity of collected data. Our study aims to quantify the prevalence of this behavior and investigate methods to detect and prevent it. In a series of surveys (N = 250), we examined conditions such as platform choice, survey length, requests not to use AI, and disabling copy-paste functionality. We were able to identify distinct characteristics of LLM-assisted responses and found that their frequency varied widely, from under 10% on Prolific to over 80% on Mechanical Turk. Mitigation measures reduced LLM usage but did not necessarily improve data quality. No participants employed browser-use agents at the time of our survey, but we report on our own detection experiments. We recommend that researchers actively screen survey responses for LLM usage by recording and analyzing keystroke data and crafting instructions and questions aimed at AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10531v2">LC-QAT: Data-Efficient 2-Bit QAT for LLMs via Linear-Constrained Vector Quantization</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 Accepted by ICML 2026
    </div>
    <details class="paper-abstract">
      Quantization-aware training (QAT) is essential for extremely low-bit large language models (LLMs). Current QAT methods are mainly based on scalar quantization (SQ), which enables efficient optimization but suffers from severe performance degradation at 2-bit precision. On the other hand, vector quantization (VQ) provides substantially higher representational capacity, but its discrete codebook lookup prevents end-to-end training. We propose LC-QAT, a 2-bit weight-only VQ-QAT framework that represents quantized weights via a learned affine mapping over discrete vectors, which yields a high-quality PTQ initialization and enables fully differentiable end-to-end optimization without explicit codebook lookup in the training forward pass. This strong post-training initialization makes LC-QAT highly data-efficient. Experiments across diverse LLMs demonstrate that LC-QAT consistently outperforms state-of-the-art QAT methods while using only 0.1%--10% of the training data. Our results establish LC-QAT as a practical and scalable solution for extreme low-bit model deployment. Codes are publicly available at https://github.com/AI9Stars/UniSVQ.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.04140v2">Selective Expert Guidance for Effective and Diverse Exploration in Reinforcement Learning of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 Accepted by ICLR 2026
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has become a widely adopted technique for enhancing the reasoning ability of Large Language Models (LLMs). However, the effectiveness of RLVR strongly depends on the capability of base models. This issue arises because it requires the model to have sufficient capability to perform high-quality exploration, which involves both effectiveness and diversity. Unfortunately, existing methods address this issue by imitating expert trajectories, which improve effectiveness but neglect diversity. To address this, we argue that the expert only needs to provide guidance only at critical decision points rather than the entire reasoning path. Based on this insight, we propose MENTOR: Mixed-policy Expert Navigation for Token-level Optimization of Reasoning, a framework that provides expert guidance only at critical decision points to perform effective and diverse exploration in RLVR. Extensive experiments show that MENTOR enables models capture the essence of expert strategies rather than surface imitation, thereby performing high-quality exploration and achieving superior overall performance. Our code is available online.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.12887v2">EcoGEO: Trajectory-Aware Evidence Ecosystems for Web-Enabled LLM Search Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      Web-enabled LLM agents are changing how online information influences search outcomes. Existing Generative Engine Optimization (GEO) studies mainly focus on individual webpages. However, agentic web search is not a single-document setting: an agent may issue queries, crawl pages, follow links, reformulate searches, and synthesize evidence across multiple browsing steps. Influence therefore depends not only on page content, but also on how pages are organized, connected, and encountered along the agent's browsing trajectory. We study this shift through Ecosystem Generative Engine Optimization (EcoGEO), which treats GEO as an environment-level influence problem for web-enabled LLM agents. To instantiate this perspective, we propose TRACE, a Trajectory-Aware Coordinated Evidence Ecosystem. Given a recommendation query and a fictional target product, our method builds a controlled evidence environment that coordinates an agent-facing navigation entry page with heterogeneous support pages. These pages use shared terminology, internal links, and consistent product attributes to introduce, verify, and reinforce the target product. We evaluate our method on OPR-Bench, a benchmark for open-ended product recommendation. Experiments show that it consistently outperforms page-level GEO baselines in final target recommendation. Trajectory-level metrics further show increased initial target-result crawls, target-specific follow-up searches, and internal-link crawls, suggesting that the gains come from shaping the agent's evidence-acquisition process rather than merely adding more target-related content. Overall, our findings support an ecosystem research paradigm for GEO, where web-enabled LLM agents are studied in relation to the broader evidence environments that guide search, browsing, and answer synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06160v2">Evaluating Implicit Biases in LLM Reasoning through Logic Grid Puzzles</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 26 pages (including appendix)
    </div>
    <details class="paper-abstract">
      While recent safety guardrails effectively suppress overtly biased outputs, subtler forms of social bias emerge during complex logical reasoning tasks that evade current evaluation benchmarks. To fill this gap, we introduce a new evaluation framework, PRIME (Puzzle Reasoning for Implicit Biases in Model Evaluation), that uses logic grid puzzles to systematically probe the influence of social stereotypes on logical reasoning and decision making in LLMs. Our use of logic puzzles enables automatic generation and verification, as well as variability in complexity and biased settings. PRIME includes stereotypical, anti-stereotypical, and neutral puzzle variants generated from a shared puzzle structure, allowing for controlled and fine-grained comparisons. We evaluate multiple model families across puzzle sizes and test the effectiveness of prompt-based mitigation strategies. Focusing our experiments on gender stereotypes, our findings highlight that models consistently reason more accurately when solutions align with stereotypical associations. This demonstrates the significance of PRIME for diagnosing and quantifying social biases perpetuated in the deductive reasoning of LLMs, where fairness is critical.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04424v2">Gavel: Agent Meets Checklist for Evaluating LLMs on Long-Context Legal Summarization</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 webpage at https://yao-dou.github.io/gavel/
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) now support contexts of up to 1M tokens, but their strengths and weaknesses on complex long-context tasks remain unclear. To study this, we focus on multi-document legal case summarization, where a single case often spans many documents exceeding 100K tokens. We systematically evaluate 12 frontier LLMs with Gavel, which consists of Gavel-Ref, a reference-based evaluation framework with checklist, residual-fact, and writing-style evaluations, and Gavel-Agent, a reference-free agent for evaluating factual coverage directly from source documents. Our results show that current models are more prone to omitting key information than hallucinating. They all perform well on simple checklist items, such as filing date, but struggle with rare and complex items, such as settlements. Performance also declines as case length increases. To meta-evaluate Gavel, we collect 160 hours of human annotations. Gavel-Agent reduces token usage by at least 36% compared to end-to-end and chunk-by-chunk methods while achieving competitive performance. Gavel-Agent also generalizes to the medical domain, performing the best with at least 77% fewer tokens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.00297v1">EPC: A Standardized Protocol for Measuring Evaluator Preference Dynamics in LLM Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 10 pages, 3 tables
    </div>
    <details class="paper-abstract">
      When LLM agents use evaluator feedback to adapt their behavior in closed loops, evaluator biases propagate through the agent's strategy distribution -- a phenomenon known as evaluator preference coupling. Prior work has documented coupling across multiple evaluator families and model versions, but the field lacks a standardized protocol that enables third-party researchers to (i) reproduce coupling measurements, (ii) compare results across evaluators and time points, and (iii) detect measurement decay as proprietary evaluators silently update. This paper provides the protocol. We specify EPC (Evaluator Preference Coupling) -- a detailed, RFC-style protocol specification for the four-phase isolation paradigm, covering executor and evaluator configuration, strategy and task design, the TTRL update rule, metric computation (gamma, JSD, ECE, Brier), and output schema. We accompany the protocol with a versioned Reference Snapshot v1.0: coupling measurements for eight evaluator conditions (N=122 unique experimental repetitions across GPT-4o, Qwen, DeepSeek, and others) derived from five independent studies, annotated with evaluator version identifiers, API endpoints, and measurement dates. The snapshot is explicitly time-bound: all values are conditional on specific model versions and are expected to decay as proprietary evaluators update. We define a versioning convention (vX.Y-Z, encoding protocol version, snapshot version, and evaluator generation) and provide a usage guide covering adoption, interpretation, and known pitfalls. The protocol, reference snapshot, and implementation code are released as open infrastructure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.00292v1">An LLM-Based Framework for Intent-Driven Network Topology Design</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 submitted to IEEE CNSM 2026
    </div>
    <details class="paper-abstract">
      Designing deployable and resilient network topologies from natural language requirements remains a challenging problem in network automation. This work investigates the ability of Large Language Models (LLMs) to generate structurally valid and constraint-compliant network topologies through a constraint-driven pipeline combining hierarchical modeling and systematic validation. The framework is evaluated via a multimodel comparison of proprietary and open-weight LLMs across four realistic network scenarios released as a public dataset. We assess structural correctness using node and edge F1-scores against reference topologies, and evaluate resilience through server and content connectivity metrics. In addition, we analyze common failure modes, including interface mismatches and directional inconsistencies in generated topologies. Overall, this work provides a systematic benchmark for understanding how LLMs handle structural and resilience constraints in topology synthesis, and supports informed model selection for AI-driven network design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10687v3">Who Gets the Reward & Who Gets the Blame? Evaluation-Aligned Training Signals for Multi-LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 Accepted at the NeurIPS 2025 Workshop on Bridging Language, Agent, and World Models for Reasoning and Planning (LAW 2025)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) in multi-agent systems (MAS) have shown promise for complex tasks, yet current training methods lack principled ways to connect system-level evaluation with agent- and message-level learning. We propose a theoretical framework that unifies cooperative game-theoretic attribution with process reward modeling to transform system evaluation to agent credit to response-level signals. Unlike prior approaches that rely only on attribution (Shapley) or step-level labels (PRM), our method produces local, signed, and credit-conserving signals. In success cases, Shapley-based credit assignment fairly allocates outcomes across agents and is refined into per-message rewards that promote cooperation while discouraging redundancy or sabotage; in failure cases, first-error localization yields repair-aware preferences that penalize harmful steps while rewarding corrective attempts. The resulting signals are bounded, cooperative, and directly compatible with reinforcement- or preference-based post-training, providing a unified and auditable pathway from global evaluation to local supervision in LLM multi-agent training. Our contribution is conceptual: we present a theoretical foundation and training signals, leaving empirical validation for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19775v2">From Actions to Understanding: Conformal Interpretability of Temporal Concepts in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 Accepted at the Mechanistic Interpretability Workshop, 43rd International Conference on Machine Learning, Seoul, South Korea, 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed as autonomous agents capable of reasoning, planning, and acting within interactive environments. Despite their growing capability to perform multi-step reasoning and decision-making tasks, internal mechanisms guiding their sequential behavior remain opaque. This paper presents a framework for interpreting the temporal evolution of concepts in LLM agents through a step-wise conformal lens. We introduce the conformal interpretability framework for temporal tasks, which combines step-wise reward modeling with conformal prediction to statistically label model's internal representation at each step as successful or failing. Linear probes are then trained on these representations to identify directions of temporal concepts - latent directions in the model's activation space that correspond to consistent notions of success, failure or reasoning drift. Experimental results on two simulated interactive environments, namely ScienceWorld and AlfWorld, demonstrate that these temporal concepts are linearly separable, revealing interpretable structures aligned with task success. We further show preliminary results on improving an LLM agent's performance by leveraging the proposed framework for steering the identified successful directions inside the model. The proposed approach, thus, offers a principled method for early failure detection as well as intervention in LLM-based agents, paving the path towards trustworthy autonomous language models in complex interactive settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01485v1">CoPersona: Collaborative Persona Graphs for Robust LLM Personalization</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 Accepted at KDD '26. 12 pages, 5 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Real-world LLM personalization is often constrained by sparse and skewed user histories: most users provide only a handful of interactions, while even frequent users' logs capture an incomplete and biased view of their preferences. As a result, weakly observed user attributes are difficult to infer, leading to brittle personalization when test-time requests shift toward under-supported facets. Motivated by this limitation, we present CoPersona, a graph-based collaborative personalization framework that completes sparse user profiles by borrowing signals from behaviorally similar peers. However, directly transferring signals is difficult because uneven facet coverage introduces bias into interaction histories, obscuring user similarity in the unstructured global space. To address this issue, CoPersona decomposes interaction histories into multiple facet-level representations and explicitly models peer-to-peer, facet-level alignment through a multiplex persona graph. To effectively leverage peer information at inference time, we employ a dual-branch architecture that combines non-parametric peer retrieval with parametric graph reasoning. Experiments across multiple domains and model scales demonstrate consistent improvements over strong baselines, validating CoPersona as an effective approach for robust LLM personalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.05953v2">SCALEFeedback: A Large-Scale Dataset of Synthetic Computer Science Assignments for LLM-generated Educational Feedback Research</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      Using Large Language Models (LLMs) to give educational feedback to students for their assignments has attracted much attention in the AI in Education (AIED) field. Yet, there is currently no large-scale open-source dataset of student assignments that includes detailed assignment descriptions, rubrics, and student submissions across various courses. As a result, research on generalisable methodology for automatic generation of effective and responsible educational feedback remains limited. In this paper, we introduce a synthetic computer science university assignment dataset for LLM-based educational feedback research, called SCALEFeedback (Synthetic Computer science Assignments for LLM Educational Feedback Research). The dataset is generated via Sophisticated Assignment Mimicry (SAM) framework specifically designed to synthesise this dataset and that utilizes one-to-one LLM-based imitation from real assignment descriptions, rubrics, and student submissions. Our open-source dataset contains 10,000 synthetic student submissions spanning 155 assignments across 59 university-level computer science courses. Technical validation confirmed that the synthetic dataset closely resembles real data while successfully eliminating personally identifiable information present in the source material. The creation of this dataset is a valuable contribution to researchers who aim to develop LLM-based generalisable methods for offering high-quality, automated educational feedback in a scalable way.
    </details>
</div>
