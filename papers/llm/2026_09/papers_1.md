# llm - 2026_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.04159v1">SENTINEL-RL: Offloading Topological Reasoning from LLM Agents in the Security Operations Center</a></div>
    <div class="paper-meta">
      📅 2026-09-03
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents are increasingly proposed as autonomous SOC analysts, but two limitations make them unreliable at enterprise scale: a finite context window cannot hold a multi-thousand-host authentication graph, and free-form generation offers no guarantee that a recommended containment action is consistent with the topology it operates on. We present Sentinel-RL, an agentic-SOC architecture that decouples topological reasoning from semantic reasoning: a heterogeneous graph attention encoder summarizes the live authentication subgraph into a fixed-dimensional state, a Proximal Policy Optimization (PPO) policy maps this state to a constrained set of investigative actions, and an LLM agent loop is restricted to consuming the policy's recommendations and producing analyst-readable narratives gated by a critic. We instantiate the system on the LANL Comprehensive, Multi-Source Cyber-Security Events dataset and the Indiana University Quartz HPC cluster, reporting four results: (i) a two-phase CREATE ingestion pattern loads a 24M-edge authentication subgraph into Neo4j in 14.2 minutes on a single 32-core node, roughly 24x faster than the canonical MERGE-based pipeline; (ii) a sliding-window alert engine reliably trips a 25-event/10-second threshold in <=2.5 s across 50 trials; (iii) PPO training over 200 iterations converges to a mean episodic return of 8.74+/-0.31, with held-out precision of 0.91 and recall of 0.87 on labeled red-team events; and (iv) the integrated containment loop completes a full detect-investigate-recommend-human-approve cycle in a median of 6.3 s. We contribute a reusable engineering pattern (the hot-node deadlock workaround), a portable HPC deployment pattern (anchor-node co-location), and an enterprise-readiness analysis covering false-positive economics, reversibility guarantees, audit compliance, and the human-approval boundary.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.25175v3">EasySteer: A Unified Framework for High-Performance and Extensible LLM Steering</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 EMNLP 2026 System Demonstrations. Code: https://github.com/ZJU-REAL/EasySteer Demo: https://www.youtube.com/watch?v=3rRGzZmhrXg
    </div>
    <details class="paper-abstract">
      Large language model (LLM) steering has emerged as a promising paradigm for controlling model behavior at inference time through targeted manipulation of hidden states, offering a lightweight alternative to expensive retraining. However, existing steering frameworks suffer from critical limitations: computational inefficiency, limited extensibility, and restricted functionality that hinder both research progress and practical deployment. We present EasySteer, a unified framework for high-performance, extensible LLM steering built on vLLM. Our system features modular architecture with pluggable interfaces for both analysis-based and learning-based methods, fine-grained parameter control, pre-computed steering vectors for eight application domains, and an interactive demonstration system. Through deep integration with vLLM's optimized inference engine, EasySteer achieves 10.8-22.3$\times$ speedup over existing frameworks. Extensive experiments demonstrate its effectiveness in overthinking mitigation, hallucination reduction, and other key applications. EasySteer transforms steering from research technique to production-ready capability, establishing critical infrastructure for deployable, controllable language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.04127v1">Epistemic Warrant for LLM Recommendations: Characterizing the Basis for Reliance When Ground Truth Is Unavailable</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 43 pages
    </div>
    <details class="paper-abstract">
      Large language models are increasingly used to support organizational decisions, yet users often lack a principled basis for assessing whether to rely on a specific recommendation. Existing approaches typically evaluate broad model properties, such as reliability, uncertainty, or robustness, or focus on user trust, rather than the underlying basis for relying on an individual recommendation. Adapting theoretical foundations from epistemology, we introduce epistemic warrant, a decision-level construct that characterizes the stability of a model's preference and the scope over which that preference holds. We operationalize this construct through a four-tier reliance certificate for pairwise recommendations, distinguishing among unstable, context-dependent, locally supported, and broadly supported recommendations. We validate the construct using contemporary methodologies: known-groups tests successfully recover expert-prespecified warrant orderings, and stronger warrants systematically align with independent consensus from crowd workers. Furthermore, we demonstrate that epistemic warrant provides information distinct from verbalized confidence and is not readily explained by decision difficulty. Ultimately, this framework offers a theoretically grounded, implementable approach for characterizing the warrant of individual LLM recommendations when objective ground truth is unavailable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.04098v1">Why Gated DeltaNet Survives 4-Bit Quantization: NVFP4 W4A4 for the Recurrent Half of a Hybrid 27B LLM</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 14 pages, 2 figures, 6 tables. Quantized checkpoint: https://huggingface.co/minima-ai/mnma_qwen3.8_27b_nvfp4
    </div>
    <details class="paper-abstract">
      Hybrid LLMs pair softmax attention with linear-attention layers such as Gated DeltaNet (GDN), whose recurrent state summarizes the context in fixed size. Early community 4-bit quantizations of Qwen3.8-27B (48 GDN layers, 16 attention layers) left the GDN block in 8- or 16-bit precision -- especially its decay and write-strength gates -- on the intuition that errors in a recurrence accumulate over long contexts. We test that intuition by building Minima: NVFP4 W4A4 on all 496 linear layers, GDN included. Across perplexity at 4K/32K, MMLU-Pro, GSM8K, AIME'25, GPQA-Diamond, LiveCodeBench, and RULER retrieval to 64K, Minima matches BF16 within seed noise (5-task average -0.52) while being the smallest (17.5 GiB) and fastest-prefill (+14-19%) recipe we compare, and its 32K perplexity gap shrinks with position. A four-part mechanism study explains why: (i) NVFP4's 16-element block scaling localizes the residual stream's extreme outliers, equalizing activation error across layer roles; (ii) the supposedly fragile gate projections are the least sensitive -- softplus/exponential and sigmoid parameterizations compress ~11% GEMM error to ~2% output error; (iii) the delta-rule recurrence holds injected noise at a flat plateau over 32K tokens and forgets a state impulse within hundreds of steps, because each write overwrites the state along the current key direction; (iv) the per-token quantization cost washes out with context instead of compounding. We also repair a global-scale mismatch that arises when per-module-calibrated NVFP4 checkpoints are served by kernels that fuse those modules into one GEMM, and show calibrated FP8 KV-cache scales are performance-free. The result: a practical recipe -- quantize everything, ship KV scales -- and a mechanistic account of why the recurrent half of a hybrid LLM is the easy half to quantize. Checkpoint: https://huggingface.co/minima-ai/mnma_qwen3.8_27b_nvfp4
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.04055v1">LabelMate: An LLM-Driven Framework for Refined Issue Report Labeling</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 38 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Software users often submit issue reports to a product's issue tracking system to report defects, suggest enhancements, or raise other product-related concerns. Labeling these issue reports supports effective planning and improves community engagement. However, many issue reports remain unlabeled due to the substantial manual effort required to design an appropriate label taxonomy, then assign suitable labels from this taxonomy to new issue reports. Existing automated labeling approaches attempt to mitigate these challenges. However, they suffer from key limitations, such as extensive manual intervention, the assignment of generic labels, and a dependence on existing labeled datasets. To address these limitations, we propose LabelMate, a novel Large Language Model (LLM)-driven framework that (1) derives a comprehensive, project-specific label set from historical issue reports and (2) automatically assigns relevant labels to new issue reports without requiring any pre-labeled training data. We evaluate LabelMate on 16,500 issue reports from 30 popular and diverse GitHub repositories. Based on this dataset, our approach generates a coherent list of 275 labels and achieves an average labeling accuracy of 89.84%, a statistically significant improvement over existing generic label assigning approaches. These results demonstrate that LabelMate offers an efficient, domain-adaptive solution to streamline the issue labeling process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.04010v1">Unlocking Lossless Speedups in LLMs via Discrete Diffusion</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 Code and Checkpoints at https://s-sahoo.github.io/uno/
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) owe much of their success to next-token prediction (NTP), but their autoregressive (AR) structure requires slow, sequential token generation. To overcome this bottleneck, we introduce diffusion-augmented LLMs, a new class of models that defines an AR model distribution while using diffusion to draw multiple tokens in parallel from that distribution. We decouple the parameters of these models into two sets: AR weights, trained using the standard NTP objective, and lightweight diffusion weights, trained to generate multiple tokens simultaneously. The diffusion weights are learned through a simple Diffusion Distillation phase that adds negligible overhead to existing LLM training pipelines. We also introduce $Ψ$-Spec, a family of samplers that enables lossless acceleration and inference-time scaling at a fixed context length. Unlike speculative decoding, our method requires no separate draft model. Unlike diffusion LLMs (d-LLMs), it accelerates generation without sacrificing the quality of the underlying AR model. The resulting models, called Uno, can be trained from scratch or built by augmenting existing open-weight AR LLMs. Uno achieves higher throughput than leading speculative-decoding methods at every evaluated batch size and delivers up to $3\times$ speedups over the base AR model, including at the largest batch size supported by the device. Notably, our 8B Uno model outperforms the leading open d-LLM, the 26B DiffusionGemma, and the proprietary Mercury 2 across all evaluated benchmarks in agentic tool use, coding, and long-context reasoning. We release code and checkpoints at: https://s-sahoo.github.io/uno/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03999v1">Shifting from Injection to Interaction: Rethinking Web Security in the Age of LLMs and Beyond</a></div>
    <div class="paper-meta">
      📅 2026-09-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are becoming integral to web applications and browser agents, transforming online interactions while introducing new attack vectors and reshaping longstanding web vulnerabilities. Classical threats such as cross-site scripting (XSS) can be amplified through LLM-mediated interactions, while LLM-specific vulnerabilities can propagate across web applications, introducing attacks such as prompt injection. Securing modern web systems therefore requires understanding interactions between traditional and LLM-specific threats across the system lifecycle. Unlike prior surveys treating web and LLM security separately, this survey provides a unified analysis of how LLMs amplify web vulnerabilities across client-side, server-side, and pipeline layers while evaluating defenses and their limitations. The analysis examines extending NIST and ISO/IEC AI security frameworks to the security needs of LLM-enabled web environments. Three unresolved challenges are identified: adversarial natural-language instructions, autonomous agent security, and post-deployment security through continuous monitoring and adaptation. An LLM-aware monitoring and control framework is proposed, integrating semantic input validation, prompt integrity protection, output isolation, agent governance, and runtime monitoring. This unified perspective characterizes the evolving threat landscape and outlines future directions for secure AI-enabled web systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.08693v4">Recognition Without Mitigation: Ethical Frameworks in Autonomous Offensive-LLM Agent Research</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 Accepted at AutonomousCyber 2026
    </div>
    <details class="paper-abstract">
      Large language models have moved from advising on offensive security to autonomously conducting it. A growing literature presents agents that execute reconnaissance, exploitation, and privilege escalation against real or simulated targets. Such an agent is a deployable, re-pointable capability that could be used by a malicious actor against a non-consenting third party. Papers that introduce these prototypes therefore carry an ethical burden, which top security venues have begun to encode as hard policy in their 2026 ethics mandates. We present a systematic audit of ethics reporting based on 54 papers describing autonomous offensive-LLM penetration-testing prototypes (2023-2026), assembled from peer-reviewed venues as well as from pre-prints. We score each against an eleven-dimension instrument derived both top-down from the Menlo Report, and bottom-up from 2026 security venue ethics mandates. Our central result is a recognition-without-mitigation gap: dual-use risk is reported as recognized in 57% of papers, but a concrete mitigation is reported in only 15%, roughly a 4:1 gap. Safeguards commonly protect the experiment, not the public. Guardrail bypasses are deployed by 17% of papers but not disclosed to LLM providers. Measured against the new mandates, the corpus defines a pre-regulation baseline in which current practice does not meet the substantive requirements. We argue this audit is itself defensive intelligence on the offensive-agent ecosystem, and provide an ethics statement checklist for authors working on autonomous offensive agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16027v3">Deja Vu in Plots: Leveraging Cross-Session Evidence with Retrieval-Augmented LLMs for Live Streaming Risk Assessment</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 SIGIR'26 Full Paper
    </div>
    <details class="paper-abstract">
      The rise of live streaming has transformed online interaction, enabling massive real-time engagement but also exposing platforms to complex risks such as scams and coordinated malicious behaviors. Detecting these risks is challenging because harmful actions often accumulate gradually and recur across seemingly unrelated streams. To address this, we propose CS-VAR (Cross-Session Evidence-Aware Retrieval-Augmented Detector) for live streaming risk assessment. In CS-VAR, a lightweight, domain-specific model performs fast session-level risk inference, guided during training by a Large Language Model (LLM) that reasons over retrieved cross-session behavioral evidence and transfers its local-to-global insights to the small model. This design enables the small model to recognize recurring patterns across streams, perform structured risk assessment, and maintain efficiency for real-time deployment. Extensive offline experiments on large-scale industrial datasets, combined with online validation, demonstrate the state-of-the-art performance of CS-VAR. Furthermore, CS-VAR provides interpretable, localized signals that effectively empower real-world moderation for live streaming.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03955v1">Two-Stage Reinforcement Learning for Sound and Adversarial Test Generation in Code LLMs</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 21 pages, 7 figures. Accepted to Findings of the Association for Computational Linguistics: EMNLP 2026
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has substantially advanced code generation with large language models (LLMs) through executable feedback. The feedback for coding problems mainly comes from specific test cases, where high-quality test cases are often scarce since they should be both sound and discriminative. We thus turn to study the auto-generation of test cases using the learned model. We find this is naturally an adversarial RL problem: the model is expected to generate effective test cases as counterexamples, depending on the solver's current failure modes. We propose Test Cases Scaling (TCS), a two-stage RL framework for effective test generation. Both stages train a test generator from a rolling policy-aligned buffer: Stage 1 generates tests consistent with the reference solution, and Stage 2 restricts the buffer to current failure modes and learns counterexample tests. Across TACO and LiveCodeBench, TCS improves both pass@1 and inference-time answer selection according to generated tests. We find the learned test generator also enables effective selection among other LLM outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09108v2">Evolving Excellence: Automated Optimization of LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2026-09-03
    </div>
    <details class="paper-abstract">
      Agentic AI systems built on large language models (LLMs) offer significant potential for automating complex workflows, from software development to customer support. However, LLM agents often underperform due to suboptimal configurations; poorly tuned prompts, tool descriptions, and parameters that typically require weeks of manual refinement. Existing optimization methods either are too complex for general use or treat components in isolation, missing critical interdependencies. We present ARTEMIS, a no-code evolutionary optimization platform that jointly optimizes agent configurations through semantically-aware genetic operators. Given only a benchmark script and natural language goals, ARTEMIS automatically discovers configurable components, extracts performance signals from execution logs, and evolves configurations without requiring architectural modifications. We evaluate ARTEMIS on four representative agent systems: the \emph{ALE Agent} for competitive programming on AtCoder Heuristic Contest, achieving a \textbf{$13.6\%$ improvement} in acceptance rate; the \emph{Mini-SWE Agent} for code optimization on SWE-Perf, with a statistically significant \textbf{10.1\% performance gain}; and the \emph{CrewAI Agent} for cost and mathematical reasoning on Math Odyssey, achieving a statistically significant \textbf{$36.9\%$ reduction} in the number of tokens required for evaluation. We also evaluate the \emph{MathTales-Teacher Agent} powered by a smaller open-source model (Qwen2.5-7B) on GSM8K primary-level mathematics problems, achieving a \textbf{22\% accuracy improvement} and demonstrating that ARTEMIS can optimize agents based on both commercial and local models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03923v1">Speak for Me: Giving LLMs the Situational Awareness to Participate in a Meeting</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 Accepted at EMNLP 2026 Main
    </div>
    <details class="paper-abstract">
      In online meeting delegation, LLM agents fail to recognize when to speak. With no structured way to track stances, coverage, and floor, they miss the moments where they should contribute. Prompt-only delegates stay silent on 51.4% of the absent participant's talking opportunities on the AMI corpus. We present CAPA (Collaborative Agent Predictive Architecture), an architecture for online meeting delegation. A Perceiver updates the meeting state from each observed turn. A Predictor forecasts how the conversation will continue. A Controller decides whether to speak and which proposition to surface. A Generator phrases the chosen contribution in the participant's style. Two judges score the forecast and the action against the next observed turn. A Recalibrator updates the meeting state from those verdicts for future decisions. To evaluate online delegation, we introduce an episode-level protocol that scores whether, when, and what a delegate contributes around the participant's actual idea units. The protocol's schema-constrained LLM judges align with human annotations at Cohen's kappa = 0.71. On 137 AMI meetings, CAPA reduces the silence rate from 51.4% to 2.5%, doubles credited recovery (26.1 --> 52.2), and keeps hallucination at 0.6%. The failure mode shifts from omission to selection, with each residual near-miss attributable to a specific module of the architecture. Mechanism ablations identify the meeting state as the lever that closes the recognition gap, where raw-context scaling alone does not.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.20721v4">User Perceptions vs. Proxy LLM Judges: Privacy and Helpfulness in LLM Responses to Privacy-Sensitive Scenarios</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 Published as a main conference paper at ACL 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are rapidly being adopted for tasks like drafting emails, summarizing meetings, and answering health questions. In these settings, users may need to share private information (e.g., contact details, health records). To evaluate LLMs' ability to identify and redact such information, prior work introduced real-life, scenario-based benchmarks (e.g., ConfAIde, PrivacyLens) and found that LLMs can leak private information in complex scenarios. However, these evaluations relied on proxy LLMs to judge the helpfulness and privacy-preservation quality of LLM responses, rather than directly measuring users' perceptions. To understand how users perceive the helpfulness and privacy-preservation quality of LLM responses to privacy-sensitive scenarios, we conducted a user study ($n=94$) using 90 PrivacyLens scenarios. We found that users had low agreement with each other when evaluating identical LLM responses. In contrast, five proxy LLMs reached high agreement, yet each proxy LLM had low correlation with users' evaluations. These results indicate that proxy LLMs cannot accurately estimate users' wide range of perceptions of utility and privacy in privacy-sensitive scenarios. We discuss the need for more user-centered studies to measure LLMs' ability to help users while preserving privacy, and for improving alignment between LLMs and users in estimating perceived privacy and utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03894v1">CROCODIL: Cross-Model Code Editing with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 EMNLP 2026 Findings
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become ubiquitous tools for code generation and editing. However, development teams often use multiple LLM assistants. Different developers may prefer different models, and individual developers may switch between models across different coding sessions. Because of this, the edits any one model makes are frequently applied to foreign code originally generated by another model. These LLMs are often trained on different datasets, and as a result have different stylistic preferences. Do LLMs behave differently when they edit foreign code originally written by a different LLM with a different coding style? We find that models tend to make more, and often excessive, edits on foreign code. We introduce CROCODIL (Cross-model Code Editing with LLMs), a post-training framework for reducing excessive edits while preserving functional correctness. CROCODIL's similarity reward penalizes large changes, while its execution reward scores build and test success. We use the product of these two rewards to encourage the policy to decrease the edit size without decreasing the edit task success rate. CROCODIL is available at https://github.com/EngineeringSoftware/Crocodil.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03874v1">STAIR (STructure Aware Information Retriever): A novel dataset and LLM based retriever for document structure augmentation</a></div>
    <div class="paper-meta">
      📅 2026-09-03
    </div>
    <details class="paper-abstract">
      Retrieval Augmented Generation (RAG) is a key component for generating accurate and hallucination free answers using Large Language Models (LLMs). LLMs are improving at handling long context, but still suffer from "lost in the middle" problem. Thus, precise and accurate retrieval is important. Current retrievers chunk long context into length-based manageable chunks - in the process throwing away rich and informative semantic global structure in the corpus. We introduce a novel retrieval system STAIR that empowers an LLM to exploit global structure in a corpus such as a Table of Contents (ToC) to efficiently store and retrieve information from its model parameters. Our thorough and careful ablation studies with a finetuned Differentiable Search Index (DSI) system show that ToC helps build a low hallucination (less than 0.05%) generative Information Retrieval (IR) system and can generalize to examples where very few training samples are available. To further research in this novel direction of ToC based retrieval we release SearchTome - a diverse benchmark created from 18 books across 6 diverse domains to further research in this novel direction. STAIR achieves a high Recall@1 score of 82.6% on SearchTome as compared to DSI (76.9%), where the difference is found to be statistically significant. STAIR easily beats other strong baselines such as BM25 (59.5%), DPR (68.7%) and out-of-the-box Mistral (13.8%).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.00262v6">LLM-Driven Cost-Effective Requirements Change Impact Analysis</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 33 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Requirements are inherently subject to change throughout the software development lifecycle. Within the limited budget available to requirements engineers, manually identifying the impact of such changes on other requirements is error-prone and effort-intensive, especially in regulated domains. This can lead to overlooked impacted requirements, which, if not properly managed, can cause serious issues in downstream tasks. Inspired by the growing potential of large language models (LLMs) across diverse domains, we propose ProReFiCIA, an LLM-driven approach to automatically identify impacted requirements when changes occur. We conduct an extensive evaluation of ProReFiCIA using several LLMs and prompt variants tailored to this task. Using the best LLM-prompt combination, ProReFiCIA achieves 85.7% recall on an unseen industrial dataset, demonstrating its effectiveness in identifying impacted requirements. Further, the cost of applying ProReFiCIA remains small, as the engineer only needs to review the predicted impacted requirements, which represent 3.0% of the entire set of requirements. Lastly, incorporating domain knowledge via RAG increases recall to 95.8% while slightly raising the cost to 3.4%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03815v1">Inferring Hidden User Models from the Behavior of Personalized LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 19 pages, 6 figures, and 5 tables
    </div>
    <details class="paper-abstract">
      Recent personalized LLM agents increasingly transform information retained in memory into compressed or structured representations, which we call user models, to guide later decisions. When source wording is removed from the state reachable through the ordinary interface, these models are commonly treated as more privacy-preserving because direct memory-extraction attacks lose the text they target. Yet we argue that user models expose a new attack surface because an attacker can still recover the private information from the personalized choices they shape, even when source records and backend state remain inaccessible. We therefore introduce UMPeek, a black-box attack based on hypothesis-guided adaptive probing to infer such hidden user model. It forms hypotheses from choices left open by a request, switches among ordinary follow-up tasks, and retains only claims supported and not contradicted by visible behavior. We conduct an extensive benchmark evaluation across diverse personalization tasks and user-model backends against existing attacks. We further validate UMPeek in real-world systems using information confirmed to be retained, and we evaluate defenses against its adaptive probing. Overall, UMPeek outperforms existing attacks in both benchmark and real-world comparisons and continues to recover user information under response-level defenses, showing that keeping records and backend state inaccessible does not guarantee semantic privacy when retained information shapes visible behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12821v2">GeoNatureAgent Benchmark: Benchmarking LLM Agents for Environmental Geospatial Analysis Across Frontier and Open-Weight Foundation Models</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 4 pages, 4 figures. Short paper, ACM SIGSPATIAL 2026. v1 is an extended 10-page preprint
    </div>
    <details class="paper-abstract">
      Environmental scientists spend disproportionate effort on data wrangling rather than analysis. New AI agents can be a helpful tool, but no benchmark exists to evaluate AI agents that automate environmental geospatial workflows through structured tool calling against real APIs. We introduce the GeoNatureAgent Benchmark, the first benchmark for environmental analysis agents that operate via structured tool calls to a production-style geospatial API. The benchmark comprises 93 tasks across 18 categories. Tasks are evaluated against an open, self-hostable geospatial API that serves three environmental indicators across Spain and Portugal via sixteen tools. We evaluate nine frontier and open-weight LLMs, reporting capability and per-case cost as orthogonal axes. Results manifest that (1) Claude Sonnet 4 achieves the highest capability at 60.8% +/- 0.8%, followed closely by DeepSeek V3.2 at 56.3% +/- 3.1%, while no other model exceeds 51%; (2) the cost-accuracy Pareto frontier is occupied mostly by open-weight models, with DeepSeek V3.2 offering 93% of Claude's capability at 11.6x lower cost; and (3) structured tool calling against a real API provides a more discriminative measure of real-world agent capability, with mean accuracies 25-35 percentage points below those reported on general-purpose GIS benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.02837v2">Fixing FOLIO and MALLS: Verified Annotations and an LLM-assisted Framework to Focus Human Relabeling</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 Accepted to EMNLP-2026
    </div>
    <details class="paper-abstract">
      Accurate translation from Natural Language to First-Order Logic (NL-to-FOL) underpins neurosymbolic AI systems and Natural Language Inference (NLI), making the quality of NL-to-FOL benchmarks essential---yet these datasets have never been rigorously audited. Our first contribution is to present a systematic human inspection of the validation split of \textsf{FOLIO} and a subset of \textsf{MALLS} test instances, finding that approximately 42.5\% and 42\% of entries, respectively, contain incorrect FOL formalizations (i.e., ground truth labels), with additional rates of ambiguous NL sentences (17.8\% and 51\%) and incorrect NLI labels in \textsf{FOLIO} (8.4\%). Our second contribution is to develop and release corrected ground truths for such datasets, showing that annotation errors distort model evaluation on a reference benchmark task: testing three state-of-the-art LLMs (Gemma~4 31B-it, Qwen3-30B-A3B, and GPT-4o-mini) with the corrected ground truths yields accuracy gains from +11 to +23 percentage points. Motivated by these findings, we propose an LLM-based framework to support humans in manual reviewing NL-to-FOL datasets. By directing reviewers toward the most error-prone instances, we empirically show that it is possible to achieve 90\% dataset accuracy after reviewing fewer than 20\% of instances, compared to over 76\% required by unguided review. We release all human-verified annotations and the code for our framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03721v1">Can LLMs Extract Architectural Design Decisions from Source Code Commits? - A Preliminary Exploratory Study</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 Accepted at IdeaArch Workshop at ECSA 2026
    </div>
    <details class="paper-abstract">
      Context: Architectural Design Decisions (ADDs) capture the rationale behind the structure and evolution of software systems but are rarely documented explicitly, and are often hidden inside source code commits. Recovering them is important for Architectural Knowledge Management (AKM). Problem: Extracting ADDs from commits is challenging due to their implicit and unstructured nature. Large Language Models (LLMs) have shown strong capabilities in understanding code and text, yet their effectiveness for this task remains underexplored. Study: We present a preliminary study using four LLMs (Gemini 3 Pro, DeepSeek R1, Kimi K2, Qwen3) with zeroshot and fewshot prompting on 30 developer-written ADDs from open-source projects. We score outputs with ROUGE-L, BLEU, METEOR, and BERTScore, and one author manually reviews the Gemini outputs. Results: All models reach a BERT-F1 above 0.81, and fewshot prompting improves alignment (Gemini BERT-F1: 0.828 to 0.847). However, the generated ADDs are often too long, implementation-focused, and miss the rationale behind the decision. This highlights opportunities for architecture-aware LLM systems and automated AKM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28791v4">Skill-Conditioned Gated Self-Distillation for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 Accepted by EMNLP 2026 Findings. Code is available at https://github.com/walawalagoose/SGSD
    </div>
    <details class="paper-abstract">
      On-policy self-distillation (SD) improves LLM reasoning by using teacher-side privileged information (PI) to turn sparse verifier outcomes into dense token-level supervision. Existing methods usually assume trusted PI, such as reference answers or successful traces. We ask whether PI can instead come from an experience-derived skill bank, where retrieved skills are compact and reusable but may also be irrelevant or misleading. We propose Skill-Conditioned Gated Self-Distillation (SGSD), which formulates skill-based SD as teacher hypothesis validation rather than unconditional imitation. SGSD retrieves skill-mistake pairs, constructs a multi-teacher pool, and lets all skill-conditioned teachers score the same plain-prompt student rollout. The verifier validates each teacher's polarity: supporting a success or suppressing a failure gives positive supervision, while the opposite stance is reversed. A robust gated objective then distills informative teacher-student disagreements while suppressing uncertain or extreme signals. Experiments on multiple mathematical reasoning benchmarks show that SGSD consistently improves over GRPO and remains competitive with answer-conditioned OPSD under a weaker PI assumption. For example, on Qwen3-1.7B, SGSD outperforms GRPO by 6.2% and OPSD by 1.7% on average on AIME24, AIME25, and HMMT25.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.02031v2">Learning-Based Collaborative MEC for LLM Inference with Soft-Deadline Awareness via Transformer-Enhanced PPO</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 7 pages, 5 pages
    </div>
    <details class="paper-abstract">
      This paper investigates collaborative mobile edge computing (MEC) servers for large language model (LLM) inference under soft deadline constraints. In this system, to improve the quality of service, computations are expected to be completed within their deadlines. However, due to dependencies among tasks or subtasks, any missed deadline can lead to catastrophic consequences for the entire request. In this context, this work proposes an extended deadline mechanism with constrained flexibility. The main challenges lie in handling large-scale computations under strict latency constraints while limiting the number of allowable deadline extensions, especially in the presence of task dependencies within each request. To tackle these challenges, we develop a transformer-enhanced proximal policy optimization (PPO) framework that enables efficient collaboration among MEC servers. The proposed approach aims to maximize the number of tasks completed within their deadlines while minimizing the use of deadline extensions. By capturing temporal dependencies and cross-server interactions, the transformer improves decision-making for task migration. Simulation results demonstrate that the proposed method significantly outperforms conventional PPO and heuristic-based approaches in terms of task completion rate and overall system efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03687v1">A Circuit for Plural Reference: How LLMs Represent and Retrieve Singular and Plural Entities</a></div>
    <div class="paper-meta">
      📅 2026-09-03
    </div>
    <details class="paper-abstract">
      Coreference resolution is an important task in contextual reasoning. In this paper, we investigate the mechanism for representing and retrieving singular and plural entities for plural reference. We use a combination of mechanistic interpretability and attention pattern analysis to study the process in which LLMs predict a pronoun to refer back to previously mentioned entities. Using a range of causal intervention techniques, we find a set of attention heads that are responsible for (1) representing coreference information in the input, (2) identifying entities that form a plural reference, (3) transferring the information to the component that is responsible for selecting the antecedents and predicting the pronoun. We also find that LLMs align with humans in preference for plural pronoun. Specifically, entities in a plural construction are more likely to be referred to as a plural entity if they are ontologically similar and are linked by the conjunction "and".
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03674v1">LLM4AIGQ: LLM-based AI Guidance Query Generation Framework for Multi Interest Mining</a></div>
    <div class="paper-meta">
      📅 2026-09-03
    </div>
    <details class="paper-abstract">
      Guidance queries stimulate user consumption by extracting preferences to provide search queries with guidance value, playing a crucial role in the e-commerce field. Traditional AI-generated queries (AIGQ) generation primarily relies on a two-stage "Query-to-AI-Generated-Query" (Q2AIGQ) association paradigm, first recalling user primary search queries from user profiles, historical behavior sequences, item-side information, and the current query through multi-path retrieval, then generalizing AIGQ via rule-based methods. This approach suffers from semantic drift due to information cascade loss; additionally, primary search query derivation heavily depends on "user-item" co-occurrence relationships, lacking exploration of user multi-interests, resulting in guidance queries with low value and mismatched purchase intent. To address the expressive limitations of traditional co-occurrence-based retrieval, we propose LLM4AIGQ, an LLM-based solution for generating AI guidance queries tailored to users' multi-interests. This approach segments user interests by integrating user profiles and historical interaction sequences, infers specific consumption intents for each sub-interest, and subsequently generates corresponding AIGQ. In terms of model training, we employ a post-training pipeline comprising Supervised Fine-Tuning (SFT), Reinforcement Learning (RL), and Direct Preference Optimization (DPO) to enhance the model's capability in generating AIGQ. We also introduce a multi-level reward design to satisfy the requirements of multi-objective optimization and long-chain reasoning in practical applications. Regarding deployment, we adopt a nearline-generation and online-read architecture to meet latency constraints. Extensive experimental analyses demonstrate that our model achieves robust performance in both offline evaluations and online A/B tests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28553v2">Refusal Before Decoding: Detecting and Exploiting Refusal Signals in Intermediate LLM Activations</a></div>
    <div class="paper-meta">
      📅 2026-09-03
    </div>
    <details class="paper-abstract">
      In this paper, we investigate whether refusal behavior can be predicted from LLM intermediate activations before decoding using linear probes trained on residual stream activations at each transformer block. We find that refusal is linearly decodable well before the final layer, indicating that safety-relevant behavior is represented in intermediate activations before output generation. To test whether this signal is actionable, we introduce Mechanistic AutoDAN, a probe-guided variant of AutoDAN that replaces full-model fitness evaluation with partial forward passes and probe-based scoring inside a genetic prompt search loop. Across the evaluated models, our method achieves attack success rates competitive with vanilla AutoDAN while reducing per-iteration search time by up to 72%, and probe-guided prompts match or exceed AutoDAN's cross-model transfer in several configurations. We further find that the usefulness of probe guidance increases with model scale. Our results suggest that refusal-relevant information is decodable from intermediate activations and can serve as an effective search signal in an AutoDAN-style discrete jailbreak optimization loop, especially for larger and more robust models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03597v1">KhatianDoc: A Human-Verified Benchmark Diagnosing Multimodal LLM Failure on Bengali Legal Land Records</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 12 pages, 5 figures, 11 tables, NLLP Workshop @ EMNLP 2026. Dataset: https://huggingface.co/datasets/RaiyanKhaan/KhatianDoc
    </div>
    <details class="paper-abstract">
      Land ownership in Bangladesh is recorded in Ana-Ganda-Kora-Kranti-Til, a base-16 positional fraction system with dedicated Unicode glyphs, no mainstream font, and no coverage in any OCR pipeline or tokenizer. The handwritten records that carry these fractions, RS Khatians, are the authoritative title record for millions of parcels and a frequent subject of civil litigation, yet no benchmark has asked whether a machine can read one. We introduce KhatianDoc, a four-task benchmark built from 107 real RS Khatian records from the Vumi (land) Office of Munshiganj, Bangladesh: symbol recognition, base-16-to-decimal conversion, structured field extraction, and legal document question answering over 1,634 QA pairs. Ground truth was transcribed by hand, verified by a land-law practitioner to full agreement, and anonymized through positional tokens that keep the referential distinctions multi-hop questions depend on. We evaluate six multimodal LLMs (8B to 72B+, open and closed) under a fixed zero-shot protocol. Five QA categories, 39.3% of our stratified set, return zero correct answers from every model; on the arithmetic task, every model that emits a number does worse than a constant-mean baseline, with exact- and near-match scores coinciding: decorrelation, not approximation. Auditing our own metrics surfaced two artifacts in opposite directions: we correct a refusal-scoring bug and report the fixed scores beside the originals, and flag an inflated metadata metric as an upper bound. KhatianDoc documents not a performance gap but the absence of a capability, with verified ground truth for future systems. Code and data, with a redacted image release, are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03592v1">Code Transformation Rule Synthesis using LLMs: Potential and Limits</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 Accepted at the 41st IEEE/ACM International Conference on Automated Software Engineering (ASE 2026)
    </div>
    <details class="paper-abstract">
      Due to their black-box nature, LLMs suffer from limited explain- ability and a lack of determinism. Their usage cost can also rise, particularly with repetitive tasks on large codebases. To mitigate this, we conduct a novel empirical study targeting three domain- specific languages for transformation rules, namely Comby, GritQL, and Ast-Grep. We evaluate three LLMs (GPT-5.4, GPT-oss-120B, and Llama3.1-8B) on six diverse datasets covering four software- evolution tasks: API misuse correction, program repair, API migra- tion, and language version migration. Our results provide evidence that transformation rule synthesis moves beyond proof-of-concept with strong frontier models. GPT-5.4 achieves consistently high rule applicability rates and produces transformations closest to the ground truth across most benchmarks. Smaller and open-weight GPT-oss-120B and Llama3.1-8B models remain effective for simpler, localized changes but struggle with complex migration scenarios. We also observe non-negligible generalizability through the usage of meta-variables and through a high reuse score in the first quartile of many datasets. Finally, when compared to the anti-unification algorithm, LLMs outperform it in correctness, but underperform in rule applicability. Overall, our results show great potential for LLMs to generate sound, correct, generalizable, and reusable rules.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03588v1">KC-Bench: A Dynamic Interactive Benchmark for Evaluating Knowledge Conflicts in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-09-03
    </div>
    <details class="paper-abstract">
      As LLMs increasingly act through tools, they must reconcile user instructions, parametric knowledge, and dynamic environmental observations before taking actions. We introduce KC-Bench, a controlled multi-turn benchmark for measuring this capability across world-knowledge conflicts, input inconsistencies, and multi-source temporal conflicts. Its 238 tasks are manually screened from more than 1,000 generated candidates and combine a user simulator, stateful tools, deterministic environment assertions, an open-source natural-language evaluator, and human trajectory verification. Evaluation of nine models, including DeepSeek-V4-Flash, GLM-5.2, and MiniMax-M3, shows substantial cross-domain variation: no model handles factual correction, identity consistency checking, and temporal conflict resolution reliably across all settings. In the simulated environments, missed conflicts can propagate to tool calls or synthetic protected-data flows. KC-Bench isolates this model-level behavior rather than ranking complete agent frameworks, and provides a reproducible diagnostic for developing conflict-aware reasoning and execution safeguards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05274v2">Sealing the Audit-Runtime Gap for LLM Skills</a></div>
    <div class="paper-meta">
      📅 2026-09-03
    </div>
    <details class="paper-abstract">
      Large language model (LLM) ecosystems such as Claude Code and ChatGPT increasingly rely on skills: packages of natural-language instructions and executable tools. Once in the LLM's context, skill content cannot be reliably separated from trusted instructions, and a skill's executable side can invoke privileged actions, exposing the skill supply chain to injection, tampering, and rug-pull attacks. Existing defenses are stage-bound: centralized signing, audit reports unbound from the runtime artifact, or policy engines that cannot attest to what was approved. We present SIGIL, the first framework that seals the audit-runtime gap for LLM skills. SIGIL delivers verifiable hosting through a tamper-evident, decentralized on-chain registry from which LLMs fetch skills directly. The registry admits four publication types, Transparent, Licensed, Sealed, and Committed, spanning plaintext public distribution, monetized access, custodial use, and off-chain workflows; before admission, every skill is vetted by a Decentralized Autonomous Organization (DAO) audit committee that supports pluggable auditing methods under a stake-and-slash economic model. At load time, SIGIL delivers verified loading through a skill verification protocol executed by a Skill Verification Loader (SVL) embedded as the mandatory loading path: the SVL retrieves and decrypts the skill as its type requires, verifies its integrity against the on-chain record, and enforces its permission manifest before context injection. We evaluate SIGIL on a real-world deployment against 1,023 in-the-wild skills spanning six attack types. At load time, the SVL verifies each skill's integrity against its on-chain record and enforces its approved permission manifest, completing batched verification under 86 ms. Together, these results show that LLM skills can be cryptographically bound from publication through runtime at practical cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10187v3">HOMURA: Taming the Sand-Glass for Time-Constrained LLM Translation via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-09-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable strides in multilingual translation but are hindered by a systemic cross-lingual verbosity bias, rendering them unsuitable for strict time-constrained tasks like subtitling and dubbing. Current prompt-engineering approaches struggle to resolve this conflict between semantic fidelity and rigid temporal feasibility. To bridge this gap, we first introduce Sand-Glass, a benchmark specifically designed to evaluate translation under syllable-level duration constraints. Furthermore, we propose Homura, a reinforcement learning framework that explicitly optimizes the trade-off between semantic preservation and temporal compliance. By employing a constrained reinforcement learning objective featuring a novel dynamic syllable-ratio reward, Homura effectively "tames" the output length. Experimental results demonstrate that Homura significantly outperforms strong baselines, achieving precise length control that respects linguistic density hierarchies without compromising semantic adequacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01023v2">AKRASIA: Stealthy Backdoor Attack on Reasoning-based Code LLMs</a></div>
    <div class="paper-meta">
      📅 2026-09-03
    </div>
    <details class="paper-abstract">
      We present AKRASIA, a stealthy, inference-time backdoor attack against reasoning-based Code LLMs. AKRASIA aims to achieve a backdoor target (e.g., malicious code execution) in reasoning LLMs while evading automated defenses and human inspection. To achieve this, AKRASIA probes the victim LLM to construct a code-level backdoor trigger. It then employs in-context learning for backdoor learning, and model unfaithfulness to conceal the backdoor trigger, and generate plausible reasoning. We evaluate AKRASIA using four backdoor targets six (6) reasoning LLMs, three coding tasks/datasets and three defense methods. AKRASIA has up to 99.34% average attack success rate on SOTA LLMs and mantains up to 97.23% average accuracy. AKRASIA evades the SOTA defense, retaining up to 98.82% average ASR in most (14/18) defense settings. It evades human inspection, successfully hiding the backdoor trigger and reasoning steps in up to 80% of settings. Our findings motivate the need to defend LLMs against reasoning backdoors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2408.09199v3">TC-RAG:Turing-Complete RAG's Case study on Medical LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 version 2.0; ACL 2025
    </div>
    <details class="paper-abstract">
      In the pursuit of enhancing domain-specific Large Language Models (LLMs), Retrieval-Augmented Generation (RAG) emerges as a promising solution to mitigate issues such as hallucinations, outdated knowledge, and limited expertise in highly specialized queries. However, existing approaches to RAG fall short by neglecting system state variables, which are crucial for ensuring adaptive control, retrieval halting, and system convergence. In this paper, we introduce the TC-RAG through rigorous proof, a novel framework that addresses these challenges by incorporating a Turing Complete System to manage state variables, thereby enabling more efficient and accurate knowledge retrieval. By leveraging a memory stack system with adaptive retrieval, reasoning, and planning capabilities, TC-RAG not only ensures the controlled halting of retrieval processes but also mitigates the accumulation of erroneous knowledge via Push and Pop actions. In the case study of the medical domain, our extensive experiments on real-world healthcare datasets demonstrate the superiority of TC-RAG over existing methods in accuracy by over 7.20\%. Our dataset and code have been available at https://github.com/Artessay/TC-RAG .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03529v1">KnowFeat: Knowledge-Guided Feature Engineering via LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 12 pages, 13 tables, 2 figures. Under review
    </div>
    <details class="paper-abstract">
      Automated feature engineering with large language models (LLMs) can produce semantically meaningful features for tabular data, yet existing methods lack structured domain knowledge, rigorous verification, and explainable provenance. We propose KnowFeat, a knowledge-guided feature engineering framework that organizes domain knowledge into five types -- schema metadata, regulatory indicators, detection rules, expert opinions, and court document evidence -- and injects them as structured context into an LLM agent. A three-stage verification pipeline filters candidates through code execution, statistical quality checks, and model effectiveness evaluation. Every accepted feature carries a provenance record tracing its design to specific knowledge assets. Under a strict held-out protocol that eliminates feature-selection leakage, KnowFeat ranks first (avg. rank 2.3) across twelve public benchmarks among seven methods (one-sided Wilcoxon p=0.017), with a peak gain of +11.6 pp AUC on a telecom churn dataset. On a real-world Bitcoin anti-money laundering (AML) dataset (Elliptic) and a synthetic digital currency AML benchmark (SimECNY), KnowFeat maintains competitive detection performance with full provenance traceability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03511v1">Lost in Reordering: Structural Sensitivity of Multilingual LLMs under Semantics-Preserving Perturbations</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 Accepted at EMNLP 2026 (Findings - Long paper)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate strong multilingual reasoning performance, yet their robustness to semantics-preserving structural variation remains underexplored, particularly for relatively free word-order languages. We investigate the structural sensitivity of multilingual LLMs using two linguistically grounded perturbation settings in Hindi and Malayalam: constrained constituent reordering and active-passive voice transformation. We introduce a benchmark dataset IndicReStruct, with two variants, GSM8K-Reordered and GSM8K-Voice, constructed from GSM8K while preserving semantic meaning. Across six state-of-the-art LLMs and multiple prompting strategies, we observe consistent and significant degradation in mathematical reasoning performance under structurally perturbed inputs. To further understand these failures, we perform qualitative error analysis and mechanistic interpretability experiments using residual-stream activation patching. Our analyses show that reasoning failures frequently arise from disruptions in entity-quantity alignment and that intermediate transformer layers contribute most strongly toward reasoning restoration. Overall, our findings suggest that current multilingual LLMs remain highly sensitive to surface syntactic realization and lack robust compositional invariance under structurally different but semantically equivalent inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.20047v2">PACMS: Submodular Context Selection as a Pluggable Engine for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-09-03
    </div>
    <details class="paper-abstract">
      Conversational and tool-using LLM agents operate over a context window that fills from several directions simultaneously. As a session proceeds, the agent accumulates user and assistant turns, entries drawn from a persistent memory store, and often largest of all, the verbatim outputs of tool calls such as file reads, search results, and API responses. Once the cumulative context exceeds the model's token budget, the framework must decide what to keep. The prevailing mechanism is recency truncation, sometimes paired with periodic summarization. This is topic-blind: a fact established early in a session is discarded simply because it is old, even when the current user query is about exactly that fact; conversely, verbose but irrelevant recent material is retained. Agents that must recall information across many turns, the defining case for memory, are precisely where recency truncation fails. Existing alternatives sit outside the agent's assembly step. Retrieval augmented generation fetches external documents into the prompt but does not arbitrate the agent's \emph{already-present} pooled context. Context-compression methods reduce token count by rewriting or pruning text, but operate query-blind and lossily. Neither treats memory entries, conversation turns, and tool outputs as a single candidate pool to be selected from by relevance at the moment the prompt is assembled.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03494v1">GrowPage: On-Demand KV Budgeting for Efficient LLM Reasoning Serving</a></div>
    <div class="paper-meta">
      📅 2026-09-03
    </div>
    <details class="paper-abstract">
      Long-output reasoning has made the key--value (KV) cache a critical memory bottleneck for efficient LLM serving. Existing KV compression methods usually rely on a predefined per-request budget and adjust only which KV states are retained, leaving the total capacity fixed throughout decoding. However, reasoning workloads exhibit substantial demand variation: different requests require different KV capacities, and the attention demand of an individual request evolves during generation. We introduce \textbf{GrowPage}, an on-demand KV budgeting framework that treats KV capacity as a runtime resource. GrowPage maintains lightweight dual-timescale query summaries to capture recent and long-term attention behaviors, and uses their relative attention working sets to estimate demand evolution. At each capacity boundary, GrowPage either compresses KV states within the current allocation or acquires an additional physical page when broader demand emerges. By integrating with PagedAttention's page-level memory abstraction, GrowPage preserves continuous batching and prefix caching. Experiments on reasoning benchmarks across multiple models show that GrowPage achieves a superior performance--throughput trade-off over existing approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.19030v5">RECAST: Expanding the Boundaries of LLMs' Complex Instruction Following with Multi-Constraint Data</a></div>
    <div class="paper-meta">
      📅 2026-09-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly expected to tackle complex tasks, driven by their expanding applications and users' growing proficiency in crafting sophisticated prompts. However, as the number of explicitly stated requirements increases (particularly more than 10 constraints), LLMs often struggle to accurately follow such complex instructions, which limits their applicability in complex real-world scenarios. To the best of our knowledge, existing datasets do not exceed 10 constraints per instance. To address this challenge, we propose RECAST, an efficient and scalable framework for synthesizing datasets where each example incorporates far more constraints than those in existing benchmarks, aiming to challenge and extend the boundaries of models' ability to follow complex instructions. These constraints are extracted from real-world prompt-response pairs to ensure practical relevance. Using this framework, we construct RECAST-30K, a large-scale, high-quality dataset comprising 30k instances spanning 19 constraint types. Experimental results demonstrate that models finetuned on RECAST-30K substantially improve in following complex instructions while maintaining their general capabilities without degradation. Moreover, RECAST enables automatic verification of constraint satisfaction via rule-based validators for quantitative constraints and LLM-based validators for qualitative ones; the verifiability provided by RECAST enables the design of reward functions for reinforcement learning, which further boosts model performance on complex and challenging tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.03936v2">Can Dialects Be Steered Like Languages? Sparse Neurons and Distributed Directions in Arabic LLMs</a></div>
    <div class="paper-meta">
      📅 2026-09-03
    </div>
    <details class="paper-abstract">
      Dialectal data are scarce relative to Modern Standard Arabic (MSA), causing Arabic LLMs to overproduce MSA and struggle with dialectally accurate generation. This raises a fundamental interpretability question about where and how dialectal features are encoded within model internals and whether these representations can improve dialect generation without fine-tuning. We study two inference-time approaches as interpretability probes and control mechanisms. First, neuron-level analysis identifies sparse populations that encode dialect-specific features and tests whether amplifying or suppressing them steers model outputs toward target dialects. Second, vector steering extracts dialect-specific activation directions and injects them during inference, motivated by feature entanglement at the neuron level. We find that these neurons are real but only partially explanatory. They occupy under 1\% of MLP dimensions but span only 5\% to 21\% of the residual dialect direction. This limited coverage is causally consequential. Neuron steering reinforces dialect in some varieties when the prompt is already dialectal but cannot induce it from MSA prompts, whereas vector steering succeeds in both settings. Arabic dialects are therefore steerable mainly through distributed rather than localized representations
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03432v1">Decoupled Analysis-Judging: An Automated Creativity Evaluator Using LLMs in Complex Multi-step Creativity Tasks</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 Accepted to EMNLP 2026
    </div>
    <details class="paper-abstract">
      Automated evaluation of creativity tasks remains challenging for LLM-as-a-Judge, as LLM is susceptible to biases such as verbosity bias and leniency bias. Such limitations are particularly evident in Contextually-Grounded and Procedurally-Structured Tasks (CGPST), a complex multi-step creativity task where inter-step dependencies, highly subjectivity, and wide scoring ranges lead to more unstable and biased judgments. Existing approaches either rely on task-specific training or directly apply LLM-as-a-Judge, both of which struggle to ensure reliable evaluation under such complexity. To bridge these gaps, we propose CreaEval, an automated creativity evaluator for CGPST that decouples typical LLM-as-a-Judge into analysis and judging. Correspondingly, CreaEval involves two critical phases: Memory-augmented Analysis, a SoT-LLM converts multi-step responses into structured evaluation evidence, incorporating cross-step memory; and Evidence-based Judging, a Judge-LLM uses the extracted evidence for judging without accessing raw responses. Comprehensive experiments show that CreaEval achieves an average performance improvement of 22.74% over the second-best baselines across CGPST and two classic simple creativity tasks, demonstrating its generalizability. The code is available at https://github.com/Jaong/CreaEval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03407v1">Caught in the Story: Narrative Captivity in Multi-turn LLMs Conversation</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 Accepted by EMNLP 2026 findings
    </div>
    <details class="paper-abstract">
      People increasingly turn to large language models (LLMs) for everyday advice, making ethically charged interpersonal problems a practical moral-advisory context. Most prior work has studied this context through single-turn judgments or pressure-laden rebuttals, assumptions that poorly match how guidance is sought in real-world contexts. These assumptions leave unclear whether narration alone, without an explicit opposing position, can shift model judgments during multi-turn moral consultation. Yet real-world moral-conflict conversation often elicits one party's self-justifying account, which can unfold over multiple turns and create information asymmetry. We introduce \textbf{narrative captivity}, a failure mode in which a model treats an unopposed one-sided account as complete and aligns with the narrator's interpretation without seeking missing perspectives. To measure this phenomenon, we build a benchmark of $5{,}078$ interpersonal-conflict scenarios spanning six moral dimensions. Across 17 LLMs, narrative captivity is widespread: end-state judgments under multi-turn narration shift by 25 percentage points on average beyond the matched single-turn baseline. Stage-level analysis identifies preference optimization as a major contributor, while four inference-time strategies provide only partial mitigation. We hope our project fosters LLM advisors that preserve independent judgment in real-world consultation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03383v1">TIGPO: Temporal Instance-Graph Policy Optimization for Long-Horizon LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-09-03
    </div>
    <details class="paper-abstract">
      Graph-based policy optimization improves credit assignment for long-horizon LLM agents by organizing rollout trajectories into state-transition graphs. However, existing methods construct graphs independently within each policy update, discarding transitions discovered by earlier policies and limiting advantage estimation to small, batch-local rollout groups. We propose \emph{Temporal Instance-Graph Policy Optimization} (TIGPO), which extends graph-based credit assignment across policy updates. TIGPO maintains a persistent transition graph for each task, allowing valid transitions discovered by different policy versions to jointly determine credit for current rollouts. To actively reconnect current exploration with historical experience, TIGPO allocates a fixed rollout budget between Exploration slots for ordinary task sampling and Revisit slots for delayed reattempts of previously explored tasks. For each revisit, TIGPO pairs the current rollout group with its corresponding earlier Exploration group to construct a cross-temporal reference. The enlarged reference is designed to stabilize relative advantage estimation under small rollout groups, while comparison on the same task directly captures policy improvement across training stages. Historical transitions and scores serve only as structural and detached statistical references and are never replayed in the policy loss. Experiments on ALFWorld and WebShop demonstrate that TIGPO consistently outperforms prior group-based and graph-based policy optimization methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03350v1">From Zero to Hero: An Open LLM Ecosystem for Armenian</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 18 pages, 4 figures, 13 tables. Data and model: https://huggingface.co/collections/COPA-AI/armenian-llm-ecosystem. Code: https://github.com/COPATeam/armenian_llm_ecosystem
    </div>
    <details class="paper-abstract">
      Pretraining data for Armenian, a morphologically rich and low-resource language, is scarce, and no open Armenian LLM has been released with the data and recipe needed to reproduce it. To address this gap, we curate and release two datasets. ArmWeb is an extensively validated corpus of 4.37M Armenian news documents. ArmSTEM is a parallel English-Armenian collection of 373K math and science problems with step-by-step solutions, translated into Armenian and verified through both answer-preserving LLM judgment and human evaluation. Continued pretraining of Gemma-4-E4B on these datasets yields arm-gemma-e4b, which outperforms every existing open Armenian model as well as its unadapted base, and is the first open Armenian LLM with complete training data and recipe. Our ablations show that news-only continued pretraining improves fluency while eroding knowledge, a pattern we also observe in existing Armenian models, and that a small share of verified translated STEM data reverses the loss. We further find that the largest public Armenian corpora overlap web-derived evaluation panels heavily, including a train/test self-overlap inside FineWeb-2. We openly release all data, models, and code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03340v1">Fresh Memory, Stale Plans: Dependency-Scoped Validation for Distributed LLM-Agent Memory</a></div>
    <div class="paper-meta">
      📅 2026-09-03
    </div>
    <details class="paper-abstract">
      Distributed LLM-agent teams can read the latest shared facts and still act on an obsolete plan. A planner may derive an action from requirement $r_3$, another agent may commit $r_4$, and an executor may receive $r_4$ without replacing the plan derived from $r_3$. We call this \emph{stale-plan execution}: state freshness does not establish that the plan authorizing an action remains valid. We introduce PlanFence, a dependency-scoped action-validation protocol. Plans cite the exact public records they used, and an executor validates only the records that can affect the pending external action, replanning once or blocking when validation is incomplete. In 30 controlled live workflows with a post-plan revision, a freshness-only executor acts on the obsolete plan in every task, whereas PlanFence completes all tasks without an invalid action. Controlled replay reveals two conditional boundaries: proactive synchronization yields lower coordination stall at low churn, while PlanFence avoids repeated update-path coordination as churn grows and avoids validating unrelated state as the shared keyspace grows. These are controlled safety and systems-cost results, not general task-accuracy gains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.22793v2">TRACE: A Self-Evolving Skill Bank for Consistent, Limit-Aware LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 9 pages, 5 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Reliable deployment of LLM agents in user-facing products depends not on raw task-solving ability but on consistency and limit-awareness: behaving the same way across repeated trials, and recognizing when a request cannot, or cannot yet, be safely fulfilled. CAR-bench exposes this reliability gap in the domain of in-car assistants: an LLM-simulated user issues incomplete or ambiguous requests, requiring the agent to resolve uncertainty through multi-turn dialogue and tool use while strictly adhering to domain policies. Even frontier models show a substantial gap between what they can solve at least once (Pass@3) and what they solve consistently across trials (Pass^k). We bridge this gap with TRACE (TRAjectory-Contrastive Evolution), which iteratively improves a skill-based agent's behavioral knowledge without modifying model weights. This knowledge is organized as a Skill Bank of modular, retrievable skills, each encoding a self-contained set of tool-use rules and behavioral guidelines. TRACE evolves this bank through an agentic self-evolution loop: after each evaluation round, it groups trajectories by the skills invoked and refines each skill by contrasting successful and failed behaviors. The updated bank then guides subsequent rounds, while during deployment the Actor performs state-conditioned skill orchestration at every turn. On GPT-5.5, TRACE improves consistency (Pass^3) by 34.6 points, from 59.9% to 94.5%, while shrinking the gap between potential and reliable performance to just 4.0 points. On the official hidden set, TRACE achieved first place using GPT-5.6-Sol, attaining a Pass^3 score of 70%-a 40% relative improvement over the baseline. These results show that TRACE converts high model potential into stable, consistent performance gain. Project homepage: https://darwin-agent.github.io/Car-bench-TRACE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03313v1">SelfDR: Self-Distillation from Reasoning for LLM-Based Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 12 pages, 5 figures, CIKM'26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently emerged as powerful backbones for recommendation. To better elicit their capabilities, reasoning has been widely incorporated to help LLMs interpret rich textual signals and improve recommendation accuracy. However, explicitly generating intermediate reasoning traces often incurs substantial computational costs, which limits practical deployment in real-world recommender systems. To address this challenge, we propose SelfDR, a Self-Distillation from Reasoning framework for LLM-based Recommendation. SelfDR distills an LLM's own reasoning-enhanced predictions to produce recommendations directly, improving recommendation effectiveness while maintaining inference efficiency. All components in the framework are built on the same base LLM, without relying on any external models. Specifically, the teacher recommender is constructed by training a reasoner with downstream performance as the reward, enabling it to generate targeted rationales that are later incorporated into the teacher's input. A student recommender for direct recommendation, with the same underlying model, then learns from the teacher through self-distillation with a dynamic weighting strategy. Extensive experiments on three public datasets validate the effectiveness, rationality, and efficiency of SelfDR. Codes are available at https://github.com/JiangDeccc/SelfDistillation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03311v1">DoPR: Reusable Compressed Document Prefixes for Efficient LLM Reranking</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 15 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are effective rerankers, but pointwise reranking repeatedly processes the same document across different queries, causing substantial redundant document-side computation. We propose \textbf{DoPR}, a compressed document prefix framework that decouples offline document processing from online reranking. DoPR first selects query-independent document representations and converts them into compressed document prefix states, which are precomputed offline and reused whenever the document is retrieved. During online reranking, the model scores each query-document pair by processing only the query and scoring token, with document information supplied by the stored prefix states. This design reduces online cost through both document-side compression and cross-query prefix-state reuse. Experiments on TREC DL, BEIR, and BRIGHT with Qwen3 models from $0.6$B to $8$B show that DoPR achieves up to 8.0$\times$ online document-side memory reduction and up to 8.04$\times$ latency speedup, while retaining \textbf{97.1\%-99.5\%} of the average NDCG@10 of matched full-document rerankers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10866v2">Short-Window Sliding Learning for Real-Time Violence Detection via LLM-based Auto-Labeling</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 5 pages, 2 figures. Accepted paper for the IEIE (Institute of Electronics and Information Engineers) Fall Conference 2025. Presentation on Nov 27, 2025
    </div>
    <details class="paper-abstract">
      This paper proposes a Short-Window Sliding Learning framework for real-time violence detection in CCTV footages. Unlike conventional long-video training approaches, the proposed method divides videos into 1-2 second clips and applies Large Language Model (LLM)-based auto-caption labeling to construct fine-grained datasets. Each short clip fully utilizes all frames to preserve temporal continuity, enabling precise recognition of rapid violent events. Experiments demonstrate that the proposed method achieves 95.25\% accuracy on RWF-2000 and significantly improves performance on long videos (UCF-Crime: 83.25\%), confirming its strong generalization and real-time applicability in intelligent surveillance systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.00669v2">Medical Reasoning in the Era of LLMs: A Systematic Review of Enhancement Techniques and Applications</a></div>
    <div class="paper-meta">
      📅 2026-09-03
    </div>
    <details class="paper-abstract">
      The proliferation of Large Language Models (LLMs) in medicine has enabled impressive capabilities, yet a critical gap remains in their ability to perform systematic, transparent, and verifiable reasoning, a cornerstone of clinical practice. This has catalyzed a shift from single-step answer generation to the development of LLMs explicitly designed for medical reasoning. This paper provides the first systematic review of this emerging field. We propose a taxonomy of reasoning enhancement techniques, categorized into training-time strategies (e.g., supervised fine-tuning, reinforcement learning) and test-time mechanisms (e.g., prompt engineering, multi-agent systems). We analyze how these techniques are applied across different data modalities (text, image, code) and in key clinical applications such as diagnosis, education, and treatment planning. Furthermore, we survey the evolution of evaluation benchmarks from simple accuracy metrics to sophisticated assessments of reasoning quality and visual interpretability. Based on an analysis of 60 seminal studies from 2022-2025, we conclude by identifying critical challenges, including the faithfulness-plausibility gap and the need for native multimodal reasoning, and outlining future directions toward building efficient, robust, and sociotechnically responsible medical AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09341v2">Auditing Multi-Agent LLM Reasoning Trees Outperforms Majority Vote and LLM-as-Judge</a></div>
    <div class="paper-meta">
      📅 2026-09-03
    </div>
    <details class="paper-abstract">
      Multi-agent systems (MAS) can substantially extend the reasoning capacity of large language models (LLMs). Most MAS frameworks aggregate agent outputs via simple majority voting, discarding the evidential structure of reasoning traces. Majority voting is brittle under confabulation consensus, where agents share correlated biases and converge on the same incorrect rationale. We introduce AgentAuditor, which moves beyond frequency-based aggregation by organizing agent traces into a Reasoning Tree that explicitly represents agreements and divergences in their reasoning. AgentAuditor resolves conflicts by comparing branch-level evidence at critical divergence points, turning global adjudication into efficient, localized verification. We further propose Anti-Consensus Preference Optimization (ACPO), which trains the adjudicator with evidence-verified preference supervision to reduce conformity to misleading majority cues. Across four MAS frameworks and multiple reasoning benchmarks, AgentAuditor consistently improves aggregation performance over majority voting, with gains of up to 5% absolute accuracy while remaining token-efficient.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03230v1">Two Truths and A Lie? Benchmarking Off-the-Shelf LLMs for Requirements Quality Assessment: Performance, False Alarms, and Misses</a></div>
    <div class="paper-meta">
      📅 2026-09-03
    </div>
    <details class="paper-abstract">
      Requirements engineering (RE) governs the quality of everything downstream in systems engineering (SE); defective requirements that survive review cycles propagate into design rework, schedule delays, and cost overruns. Because requirements are often written in natural language, recent advances in generative AI have raised expectations that large language models (LLMs) can absorb requirement quality assessment, a task otherwise slow and human expertise-intensive. Yet empirical evidence on whether LLMs can be trusted to do so remains scarce. This study presents the first benchmarking analysis of off-the-shelf LLM performance for requirement quality evaluation. Against an expert-derived ground truth built on INCOSE quality criteria, we evaluate ten models spanning two families (OpenAI and Anthropic) and five generations each, across one hundred independent runs, two requirement sets, and five sampling temperatures. Four contributions follow. First, we quantify a strongly asymmetric error profile: across all models and runs, the best-performing Anthropic model detects a median of only 47% of expert-identified issues while false-flagging 11%. Second, performance degrades significantly where SE judgment is required, as necessity and correctness issues are almost always missed. Third, generational progress is non-monotonic, so newer models cannot be assumed better. Fourth, this error behavior shifts only modestly and non-monotonically across sampling temperatures, indicating characteristic model deficiencies rather than inherent stochasticity. Off-the-shelf LLMs are therefore not yet trustworthy autonomous evaluators. Findings also warrant caution for Agentic AI developers: orchestrating these LLM modules in specialized architectures risks compounding these deficiencies rather than correcting them. Their defensible near-term role is human-in-the-loop decision support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05460v2">LLM Evaluation as Tensor Completion: Low Rank Structure and Semiparametric Efficiency</a></div>
    <div class="paper-meta">
      📅 2026-09-03
    </div>
    <details class="paper-abstract">
      Large language model (LLM) evaluation platforms increasingly rely on pairwise human judgments. These data are noisy, sparse, and non-uniform, yet leaderboards are reported with limited uncertainty quantification. We study this as semiparametric inference for a low-rank latent score tensor observed through pairwise comparisons under Bradley-Terry-Luce-type models. This places LLM evaluation in a new tensor completion setting with structured observations, non-uniform sampling, and pairwise contrasts. Our target is a smooth functional $ψ(T^\star)$, including linear estimands such as ability gaps and nonlinear ones such as win probabilities. We derive the information operator on the low-rank tangent space, the efficient influence function, and the semiparametric efficiency bound, then construct a one-step debiased estimator with asymptotic normality. A central challenge is that the information operator is anisotropic and does not commute with the tangent-space projection, creating a bottleneck absent from isotropic models. We introduce a score-whitening method that equalizes local Fisher information and restores stable inference at the optimal sample-complexity scale. Our results provide a principled framework for uncertainty quantification in LLM evaluation and more broadly for inference on low-rank structures from pairwise data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03221v1">Counterfactual Fairness Audits of Multi-Step Clinical LLM Agents Require a Measured Per-Action Instability Floor</a></div>
    <div class="paper-meta">
      📅 2026-09-02
      | 💬 13 pages, 1 figure, 2 tables. Code and data: https://github.com/rohithreddybc/FairMedAgent
    </div>
    <details class="paper-abstract">
      Counterfactual audits are the standard tool for checking whether a clinical agent treats demographically distinct but clinically identical patients differently. They report a flip rate: how often an action changes when only the patient descriptor changes. We show that this quantity is uninterpretable on its own. Re-running an identical condition ten times over sixteen vignettes (same narrative, same descriptor string, nothing varied) moved a clinical agent's action in 8.7% of outcome-vignette cells, and instability was heterogeneous across actions by a factor of eight, from 0.022 for ICU escalation to 0.179 for controlled-substance caution. No demographic contrast in our data was distinguishable from that floor. A second model gives a pooled floor of 6.7% and ranks the six actions almost identically (Spearman 0.94, exact p=0.017), so the floor is not one system's artefact. Majority-vote aggregation over five draws removes 39% of it and then flattens, and a null simulation attributes the residue to heterogeneous per-cell rates, so replication mitigates without eliminating. Any counterfactual fairness estimate reported without a per-action floor beside it therefore cannot be read as evidence of disparity. The measurements were taken with FairMedAgent, an evaluation harness for disparity in the actions of clinical LLM agents whose estimand, the within-range counterfactual flip rate, counts only flips between actions a published decision rule admits and a clinician has adjudicated. That estimand requires band adjudication, which is under way; no disparity result is claimed here. Each synthetic vignette runs a six-stage trajectory (five model-facing decisions around a deterministic environment step) under fixed-form conditions spanning race, sex, age, insurance, English proficiency, and their intersections. The harness, the floor protocol, and every analysis script are released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03218v1">The Analyst in the Prompt: Role, Retrieval, and Memory Biases in LLM Financial Analysis</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly use user context such as memory, profiles, and role prompts to personalize their responses. This personalization can affect evidence-based judgment: the same evidence may lead to different conclusions under different user contexts. Finance provides a high-stakes setting to study this problem because decisions often depend on interpreting long and complex documents. We test this using 3,575 SEC filings across twelve LLMs. We compare persona-conditioned retrieval, neutral retrieval, and memory-framed context to separate the effect of evidence selection from the effect of interpretation. We find that most user-context spillover comes from how models interpret the same evidence under different roles, rather than from retrieving different evidence. We then test two simple mitigation strategies: expressing the same investor mindset as a user profile instead of an assistant role, and separating evidence-based and personalized outputs. Both reduce spillover, but neither removes it completely, and their effectiveness varies substantially across models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06350v2">EDIT: Evidence-Diagnosed Intervention Training for Rule-Faithful LLM Grading</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      Reliable rubric grading requires more than accurate score prediction. Each judgement must be grounded in the mark scheme and evidence from the student answer. Existing credit-assignment and intervention methods, primarily designed for self-contained reasoning tasks such as mathematics reasoning, struggle in this setting because they do not identify where grading reasoning goes wrong or how the model's belief about the final mark changes during reasoning. We propose Evidence-Diagnosed Intervention Training (EDIT), a two-phase framework for training more rubric-faithful LLM graders. First, EDIT-SFT locates problematic reasoning steps using internal model signals: posterior belief over the final mark and input-grounding scores. It then revises only these local steps with help from a rubric checklist. Second, EDIT-RL calibrates the grader with belief-guided reward shaping, penalising large harmful belief drifts while still allowing helpful exploration. Experiments on two real-world, multi-subject grading benchmarks demonstrate that EDIT consistently outperforms strong supervised fine-tuning and reinforcement learning baselines on both in-domain and out-of-domain splits, with ablation studies confirming that internal-state diagnostics drive these gains. Under deterministic rubric-edit interventions, EDIT-SFT is the most rule-responsive of all evaluated systems, and EDIT-RL largely retains this responsiveness while improving accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03213v1">LLMs Learn Better In-Context from Rules than from Examples</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit in-context learning capabilities, where they can learn new tasks from prompt contexts without weight updates. We compare the learning efficacies of two prominent modes of in-context learning: (1) learning from descriptions of rules (instruction following); and (2) learning from examples of input-output demonstrations (few-shot prompting). Through five learning tasks that cover diverse domains (games, arithmetic, linguistic inferences), we compare two modes of learning (rules vs. examples) specifying the same underlying task. We furthermore explore model and task properties that modulate the learning efficacies. We find that models generally learn more reliably from rules than from examples alone, and additional examples on top of rules or simply scaling up the number of examples do not lead to consistent and significant gains. Instruction tuning amplifies the benefit of rule-based learning while keeping example-based learning capacities intact. Surprisingly, we find no privileged effect of example-based learning in base models, and rules still lead to gains in algebraic task domains. Overall, the comparative efficacy of rules over examples is larger when the task recruits algebraic abstractions and computations, and smaller when the task requires distributional sensitivity and/or recruits parametric knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29395v2">Low Rank for Rank: Uncertainty-Aware Task-Specific LLM Ranking under Sparse Pairwise Comparisons</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      Pairwise human-preference platforms such as Chatbot Arena have become central to large language model (LLM) evaluation, yet reliable task-specific ranking remains challenging. Global leaderboards mask task heterogeneity, while ranking each fine-grained task independently is unstable under sparse, imbalanced comparisons. We propose a low-rank framework for task-specific LLM ranking from sparse pairwise comparisons, modeling the task-by-model ability matrix $Θ^\star \in \mathbb{R}^{d_t \times d_m}$ as low rank so that information is shared across related tasks while task-specific differences are preserved. We first develop a max-norm ($\ell_\infty$) accurate estimator for the latent scores, combining a convex initializer with alternating-minimization refinement, and prove task-wise top-$K$ recovery guarantees under sparse sampling. Our main contribution is an uncertainty quantification framework for task-specific ranking. We construct cross-fitted one-step debiased estimators for fixed score contrasts -- such as the task-specific ability gap between two models -- yielding asymptotically valid confidence intervals that attain the semiparametric efficiency bound. We then extend the inference to the high-dimensional ranking regime, where per-task ranks and top-$K$ membership are determined by many dependent score-gap hypotheses. Using Gaussian and multiplier-bootstrap calibration, we obtain simultaneous confidence sets for per-task ranks and valid top-$K$ membership tests across many tasks and models. Experiments on synthetic data and Chatbot Arena show that low-rank sharing improves sample efficiency over independent task-wise Bradley-Terry estimation and produces tighter, better-calibrated ranking certificates, with the largest gains in the sparse regime typical of real LLM benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05031v3">LLM-Based Test Oracles: Source-of-Authority Taxonomy -- A Systematic Literature Review</a></div>
    <div class="paper-meta">
      📅 2026-09-02
      | 💬 21 pages, 10 figures, 11 tables. Systematic literature review of 83 studies, reported under PRISMA 2020. Published in IEEE Access. Replication package: https://doi.org/10.5281/zenodo.21194940
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly decide whether software behaves correctly, either by writing a test oracle or by acting as one. Yet two oracles can look identical and rest on different ground: one assertion encodes a written specification, another only what the model learned in training. Prior secondary studies sort oracles by form or by technique, rarely by the property that governs how far a verdict can be trusted: where its authority comes from. This systematic literature review, reported under the Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA) 2020 guidelines, screens 2,436 records to 54 included studies, extended by citation searching (snowballing) to 83 in total. We read the corpus along three axes: the source of an oracle's authority, the form it takes, and the mechanism that adjudicates it. Just over half of the corpus reaches a verdict with no specification at all. That is what lets these oracles work on code with no specification to consult, and what leaves a challenged verdict with less to fall back on. Source and mechanism cross-cut rather than coincide, so a label such as LLM-as-a-judge names how a verdict is produced, not why it should be trusted. Oracle quality is most often judged by resemblance to a known oracle rather than by whether injected faults are caught. The first question to ask of any LLM oracle is therefore what one would point to in defending its verdict. The protocol, search query, and per-study coding sheet are released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03177v1">Frontier LLMs are effective batch optimizers: Assessing reasoning models in continuous and discrete settings</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      Frontier large language models (LLMs) have become attractive priors for optimization due to their large-scale pretraining that enables them to navigate a variety of optimization settings. However, the effectiveness of modern reasoning LLMs in batch optimization settings remains underexplored. Here we investigate the performance of the current generation of frontier LLMs as batch optimizers in both continuous and discrete settings. We find that while LLMs are competitive zero-shot batch optimizers for numerical test functions, their performance is brittle compared to classical non-LLM optimization approaches. However, LLM priors are significantly better in semantically rich settings, indicating that their batch optimization behavior is highly effective when navigating and reasoning over the discrete spaces most similar in structure to their pretraining data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.03313v4">LLM as GNN: Graph Vocabulary Learning for Text-Attributed Graph Foundation Models</a></div>
    <div class="paper-meta">
      📅 2026-09-02
      | 💬 EMNLP 2026
    </div>
    <details class="paper-abstract">
      Text-Attributed Graphs (TAGs), where each node is associated with text descriptions, are ubiquitous in real-world scenarios. They typically exhibit distinctive structure and domain-specific knowledge, motivating the development of a Graph Foundation Model (GFM) that generalizes across diverse graphs and tasks. Despite large efforts to integrate Large Language Models (LLMs) and Graph Neural Networks (GNNs) for TAGs, existing approaches suffer from decoupled architectures with two-stage alignment, limiting their synergistic potential. Even worse, existing methods assign out-of-vocabulary (OOV) tokens to graph nodes, leading to graph-specific semantics, token explosion, and incompatibility with task-oriented prompt templates, which hinders cross-graph and cross-task transferability. To address these challenges, we propose PromptGFM, a versatile GFM for TAGs grounded in graph vocabulary learning. PromptGFM comprises two key components: (1) Graph Understanding Module, which explicitly prompts LLMs to replicate the finest GNN workflow within the text space, facilitating seamless GNN-LLM integration and elegant graph-text alignment; (2) Graph Inference Module, which establishes a language-based graph vocabulary ensuring expressiveness, transferability, and scalability, enabling readable instructions for LLM fine-tuning. Extensive experiments demonstrate our superiority and transferability across diverse graphs and tasks. The code is available at this: https://github.com/agiresearch/PromptGFM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03160v1">No country for old linguists: LLM-brain alignment underdetermines neural computation</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      Nastase et al. (2026) argue that large language models (LLMs) may illuminate language processing because both rely on distributed, context-sensitive representations shaped by statistical learning. Their rejection of simple cortical "boxology" is persuasive, and they articulate a strong case for the value of LLM-brain alignment research. The key question is what kind of inference LLM-brain alignment licenses. My claim here will be narrow: representational alignment can in principle constrain mechanistic hypotheses, but it does not by itself identify a mechanism. Nastase et al. acknowledge that an encoding model can capture features represented in neural activity without establishing a shared architecture or algorithm. Yet the authors sometime move from alignment to "shared computational principles" and ultimately to LLMs as mechanistic models of natural language. Indeed, their methodological caveat that alignment does not establish a shared architecture or algorithm sits uneasily with their conclusion that LLMs might instantiate the same computational principles as biological brains and provide a "fully mechanistic model" of language. I discuss what I consider to be problems of logical, causal, and computational underdetermination in Nastase et al.'s (2026) proposal.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03156v1">Compound Prompt Constraints in LLM Code Generation: A Factorial Study of Format, Persona, and Urgency</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in software engineering pipelines for code generation, where production prompts often combine multiple constraints. This paper presents a full-factorial empirical study of how output formatting, persona assignment, and urgency framing jointly affect LLM code-generation reliability. We evaluate all 27 combinations in a controlled 3x3x3 design and decompose each compound condition into an additive prediction and a residual interaction term that captures super-additive degradation. The study uses all 164 HumanEval+ problems across five OpenAI models from the GPT-4o family, GPT-4.1 family, and o3-mini, yielding 22,140 greedy-decoding evaluations. A format-aware extraction pipeline separates formatting failures from reasoning failures, and significance is assessed with McNemar's test, odds ratios, and 95% confidence intervals. Results show that compound constraints can produce architecture-dependent degradation not predictable from single-factor experiments. The GPT-4o family exhibits consistent super-additive effects, with pass@1 reductions 3-12 percentage points beyond additive predictions; the largest interaction is -12.2 pp on GPT-4o-mini for JSON + expert persona + moderate urgency. JSON combinations generally produce larger interactions than XML. In contrast, the GPT-4.1 family is largely resistant, while o3-mini shows a qualitatively different pattern in which structured output constraints can improve performance. These findings show that vulnerability is architecture-dependent rather than size-dependent, that individually neutral or beneficial constraints can combine to cause substantial degradation, and that compound-prompt testing should be standard in reliability assessment for LLM-assisted engineering pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03151v1">BASP: Communication-Efficient Batch-Aware Sequence Parallelism for LLM Training</a></div>
    <div class="paper-meta">
      📅 2026-09-02
      | 💬 Accepted by IEEE HPEC 2026
    </div>
    <details class="paper-abstract">
      Long-context reasoning for large language models (LLMs) is becoming increasingly important, but training over long sequences remains challenging due to massive memory and communication requirements. Sequence parallelism has emerged as an essential technique for addressing bottlenecks in long sequence LLM training. However, we observe that existing sequence parallelism methods are batch-agnostic and apply uniform sequence partitioning across all batch sizes, resulting in inefficient communication. In this paper, we introduce Batch- Aware Sequence Parallelism (BASP), a sequence parallelism approach that leverages batch structure to reduce communication overhead. BASP exploits batch structure by partitioning GPUs into disjoint sequence-parallel groups according to the micro- batch size. This design reduces the all-to-all communication group size, thereby localizing communication and improving training efficiency. Experimental results on an NVIDIA A100 cluster show that BASP improves end-to-end training time by up to 1.17 - 1.31x in Llama and Qwen models compared to standard sequence parallel baselines, while preserving identical model accuracy and memory usage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29843v2">HARP: Hadamard-Preconditioned Adaptive Rotation Processor for Extreme LLM Quantization</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      Post-training quantization (PTQ) is essential for deploying LLMs under memory and bandwidth constraints. However, extreme low-bit quantization remains highly sensitive to activation outliers and anisotropic weight curvature. Existing incoherence-based PTQ methods mitigate this issue with fixed randomized Hadamard transforms (RHTs), which improve quantization robustness but cannot adapt the rotated basis to the layer, calibration distribution, or quantizer. We introduce HARP (Hadamard-preconditioned Adaptive Rotation Processor), a learnable structured two-sided orthogonal processor that replaces fixed Hadamard mixing while preserving exact full-precision equivalence. HARP represents each rotation as a product of sparse butterfly-like block-orthogonal stages, supports non-power-of-two dimensions through Mixed-Radix schedules, and initializes to the RHT processor up to a fixed permutation. Fitted only on calibration data, HARP adapts the quantization basis to each layer and backend. Across 2--4-bit settings on Llama models from 1B to 70B, HARP consistently improves perplexity and yields its clearest zero-shot gains at 2 bits; a 2-bit Qwen3-8B experiment shows the same transfer beyond the Llama family. HARP also preserves deployment efficiency: on Llama 2 7B at 2 bits, it reaches 128 tok/s, retaining 90% of RHT throughput (142 tok/s) and running approximately $2.1\times$ faster than FP16 (61 tok/s).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.01736v4">Argument Collapse: LLMs Flatten Long-Form Public Debate</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      As LLMs are increasingly used to draft publicfacing arguments, they may flatten public debate by repeatedly introducing the same polished, plausible arguments. We study argument collapse, the tendency of essays generated by different LLMs to converge to a smaller set of main arguments, sub-arguments, and paragraph-level structures. We compare 1,039 human responses from 195 New York Times (NYT) debates, 448 human responses from 61 longer-form Boston Review (BR) forums, and 23,381 LLM-generated essays. In the NYT corpus, 65.3% of human main arguments are unique within a debate, compared to 3.4% of LLM main arguments. Asking LLMs to generate diverse answers adds variation, but a typical model recovers only about half of the distinct human main arguments, with much of the added variation falling outside the observed human argument space. Collapse also appears in sub-arguments, where among essays with the same main argument, 41.0% of human subarguments are unique versus 9.1% from LLM responses. Qualitatively, LLMs often reuse generalized and hedged sub-arguments, while humans prefer more concrete and topic-specific ones. Structure-wise, LLM-generated essays tend to follow a more fixed arc, often opening with a direct claim and moving quickly toward proposals. The same patterns hold in longer BR essays, suggesting that argument collapse extends beyond short-form responses. Finally, human-preference evaluators favor both common arguments and LLM essays, which could reinforce argument collapse.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.10078v5">Human Psychometric Questionnaires Mischaracterize LLM Behavior</a></div>
    <div class="paper-meta">
      📅 2026-09-02
      | 💬 Accepted at EMNLP 2026 (Main)
    </div>
    <details class="paper-abstract">
      We examine whether human psychometric questionnaires can serve as reliable tools for characterizing and predicting LLM behavior in everyday user interactions. We analyze eight open-source LLMs by comparing their value and personality profiles derived from two different methods: Likert self-reports on established questionnaires (PVQ-40/21 and BFI-44/10) and generation probabilities over value-laden responses to everyday user queries. The two profiles diverge substantially. Within-construct item consistency, often cited as evidence of stable LLM dispositions, disappears in generation probabilities. We find that established questionnaire items contain explicit lexical cues that allow models to recognize the target construct and respond in alignment-consistent, socially desirable ways, whereas realistic user queries contain far less recognizable cues. In addition, demographic persona prompts shift models' responses to human questionnaires in ways consistent with real human patterns, but no such shifts appear in the generation setting, highlighting that human questionnaires overestimate LLMs' ability to faithfully reproduce expected psychological traits when role-playing demographic personas. Overall, our study indicates that questionnaire scores alone should not be treated as evidence of LLMs' response tendencies in realistic user interactions, and supports generation-probability profiling with ecologically valid items as a complementary behavioral measure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03079v1">LeanStream: A Speculate-and-Refine Streaming Framework for Efficient on-Device LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-09-02
      | 💬 6 pages, 13 figures. To appear in the Proceedings of the 32nd Annual International Conference on Mobile Computing and Networking (MobiCom '26)
    </div>
    <details class="paper-abstract">
      On-device LLM inference is attractive for privacy and responsiveness, but remains challenging on mobile and embedded devices because model weights far exceed available DRAM. Prior systems exploit activation sparsity and offload weights to SSD or flash storage, but face a fundamental systems trade-off: accurate sparse execution decisions require the latest context, whereas efficient computation-I/O overlap requires early prediction. As a result, existing designs either serialize execution or incur redundant weight fetches, extra computation, and large cache overheads. We present LeanStream, a streaming speculate-and-refine framework for efficient on-device LLM inference. LeanStream progressively refines computation, loading, and cache-retention priorities using partial GPU results, enabling fine-grained overlap between GPU execution and storage I/O. We implement LeanStream on both mobile and embedded platforms. Compared with prior on-device LLM inference systems, LeanStream reduces memory usage by 4.8$\times$ to 7.5$\times$ at the best throughput achieved by prior work, while further improving token generation throughput by 1.6$\times$ to 2.1$\times$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16848v2">Mediocrity is the key for LLM as a Judge Anchor Selection</a></div>
    <div class="paper-meta">
      📅 2026-09-02
      | 💬 ACL 2026
    </div>
    <details class="paper-abstract">
      The ``LLM-as-a-judge'' paradigm has become a standard method for evaluating open-ended generation. To address the quadratic scalability costs of pairwise comparisons, popular benchmarks like Arena-Hard and AlpacaEval compare all models against a single anchor. However, despite its widespread use, the impact of anchor selection on the reliability of the results remains largely unexplored. In this work, we systematically investigate the effect of anchor selection by evaluating 22 different anchors on the Arena-Hard-v2.0 dataset. We find that the choice of anchor is critical: a poor anchor can dramatically reduce correlation with human rankings. We identify that common anchor choices (best-performing and worst-performing models) make poor anchors. Because these extreme anchors are consistently better or worse than all other models, they are seldom indicative of the relative ranking of the models. We further quantify the effect size of anchor selection, showing it is comparable to the selection of a judge model. We conclude with actionable recommendations. First, we conduct a power analysis, and compute sufficient benchmark sizes for anchor-based evaluation, finding that standard benchmark sizes are insufficient for pairwise evaluation and fail to distinguish between competitive models reliably. Second, we provide guidelines for selecting informative anchors to ensure reliable and efficient evaluation practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.02859v1">User Feedback Provides a Unique Signal that LLMs Can not Detect</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      Harnessing naturally occurring feedback from user interactions offers a promising learning signal for Large Language Models (LLMs). However, recent studies suggest this feedback is inherently noisy and difficult to leverage effectively. We challenge this conception by demonstrating that user feedback is a highly actionable signal for improvement, and that its perceived ineffectiveness stems from a systematic bias in current evaluation paradigms. To isolate the usefulness of feedback, we construct synthetic data with a definitive ground truth, alongside naturalistic data to validate that our findings hold in real-world scenarios. By comparing model revisions generated with and without access to feedback across both settings, we show that feedback-informed revisions resolve targeted issues at significantly higher rates than baseline revisions. Finally, we expose the root of the evaluation bias: when a model successfully fixes an issue exclusively due to feedback, LLM judges frequently fail to identify the genuinely corrected response, systematically preferring inferior baseline outputs instead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.02852v1">The Implications of Linguistic Illegibility for LLM Security</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      LLMs are trained to generate natural language. However, various strands of evidence indicate that an LLM's externalized linguistic outputs and mechanistically-extracted linguistic features can be an unreliable lens for understanding internal model computation. We introduce the term ``linguistic illegibility'' to broadly refer to scenarios in which an LLM's externalized or mechanistically-probed language artifacts fail to represent how the model actually thinks. We argue that the specter of linguistic illegibility is unavoidable for LLMs whose internal computations are not directly expressed via language, but rather math over activation spaces (with lossy translations between activation spaces and natural language happening at the bookends). If linguistic illegibility is always possible, then security mechanisms that rely on a model's linguistic self-reporting (e.g., chain-of-thought monitoring, constitutional self-critique, activation probing for linguistically-defined feature vectors) can never be completely sound; the model sandbox will always need isolation techniques whose guarantees do not depend on reading a model's linguistic state at all. We argue that observing a model's outputs using taint tracking is a promising approach for an effective sandbox: regardless of how a model linguistically self-reports, a taint tracking policy can define, a priori, various pieces of system state that should never be influenced by model-produced data. We also discuss several additional sandboxing mechanisms (e.g., robust virtualization, third-party auditing of sandboxing configurations) which collectively provide a critical floor beneath linguistic monitoring, and would have mitigated recent sandbox exploits by frontier models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.23286v2">Adaptive Graph-of-Islands Evolution for Automatic Feature Engineering with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      Automatic feature engineering (AutoFE) for tabular data requires discovering informative transformations from a large program space. Existing approaches suffer from three limitations: classical methods rely on fixed operator libraries with limited expressivity, LLM-based methods generate proposals from static prompts without retaining search experience, and evolutionary methods use fixed migration policies that ignore task-specific cross-family transfer utility. We introduce TOPOFE, a framework that formulates AutoFE as graph-structured multi-island evolutionary program search. The transformation space is partitioned into semantically coherent families, each explored by an island through LLM-guided mutation and crossover. Each island maintains a Prompt Adaptation Memory that accumulates accept/reject feedback to steer proposals toward productive regions without parameter updates. To coordinate global exploration, TOPOFE dynamically learns a directed topology graph whose edge weights encode transfer utility between transformation families. Cross-island transfer is triggered by adaptive saturation detection and performed through LLM-mediated hybrid synthesis, enabling discovery of compositional feature programs that cannot emerge from isolated local search. Experiments on 29 tabular datasets show that TOPOFE consistently outperforms most state-of-the-art AutoFE methods on classification and regression tasks. Beyond predictive performance, TOPOFE produces feature sets with lower redundancy and higher representational coverage, while the learned topology graph acquires meaningful task-specific transfer structure correlated with downstream gains. The discovered feature programs transfer reliably across diverse predictors and LLM backbones, demonstrating that improvements arise from TOPOFE's structured search and adaptive coordination rather than backbone-specific generation capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19086v2">MUCOCO: Automated Consistency Testing of Code LLMs</a></div>
    <div class="paper-meta">
      📅 2026-09-02
      | 💬 MuCoCo has been accepted into EMNLP Findings 2026
    </div>
    <details class="paper-abstract">
      Code LLMs often portray inconsistent program behaviors. Developers typically employ benchmarks to assess Code LLMs, but most benchmarks are hand-crafted, static and do not target consistency property. In this work, we pose the scientific question: how can we automatically discover inconsistent program behaviors in Code LLMs? To address this challenge, we propose an automated consistency testing method, called MUCOCO, which employs semantic-preserving mutation analysis to expose inconsistent behaviors in code LLMs. Given a coding query, MUCOCO automatically transforms its program into semantically equivalent programs (aka mutants) and detects inconsistencies between the mutants and the original program (e.g., different output or test failure). We evaluate MUCOCO using four (4) coding tasks and seven (7) LLMs. Results show that MUCOCO is effective in exposing inconsistency and outperforms the closest baseline (TURBULENCE). About one in seven (15%) inputs generated by MUCOCO exposed inconsistencies. Our work motivates the need to test Code LLMs for consistency property
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.02805v1">Large Language Models (LLMs) for Telecom Root Cause Analysis (RCA): A Structured Reasoning Framework for Evidence-Grounded Diagnosis</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      Root cause analysis (RCA) is a critical task in telecom network operations, but diagnosing performance degradations in modern 5G and emerging 6G networks remains challenging due to complex cross-layer dependencies. While large language models (LLMs) offer promising capabilities for reasoning and knowledge integration, directly applying vanilla LLMs to telecom RCA often leads to hallucination, unstable reasoning, and poor alignment with structured network evidence. This work first reviews the evolution of telecom RCA from rule-based and machine learning (ML) approaches to emerging LLM-enabled techniques, and provides an overview of recent paradigms, including structured reasoning, retrieval-augmented knowledge grounding, agentic orchestration, and verifiable reasoning. Building upon these insights, we propose a structured reasoning framework for LLM-enabled telecom RCA that aligns diagnostic reasoning with telecom-specific evidence and domain knowledge. The proposed approach first organizes heterogeneous network telemetry into canonical contexts, and then enforces decision-path reasoning during diagnosis, and finally generates evidence-grounded explanations for reliable fault identification. Experimental results on two 5G RCA datasets, TeleLogs and TelecomTS, demonstrate that the proposed framework consistently improves diagnostic accuracy and decision consistency compared with baseline techniques. These cross-dataset results highlight the importance of structured reasoning design for practical LLM-based RCA systems in next-generation telecom networks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.24780v2">LivingArena: Do LLMs Know What Other LLMs Don't? Peer-Probing as Scalable Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      Fixed benchmarks are costly to renew and cannot adapt their questions to model-specific failures. We ask whether LLMs can instead discover one another's weaknesses and turn those observations into an evaluation process. To study this question, we introduce \textbf{LivingArena}, an automated peer-probing framework in which models take turns testing one another. Using the interaction history, each questioner identifies potential weaknesses of its opponent and constructs targeted, verifiable questions to probe them. A 3,600-round tournament of ten models reveals a clear role asymmetry: strong answerers are not always reliable questioners, because they may generate internally inconsistent tests or fail to verify their own reference answers. After a questioner exposes an answerer's failure, it is more likely to pursue the same capability domain, while the answerer's weakness recurs on independently generated questions, including questions written by different models. These findings show that peer probing can reveal persistent model-specific weaknesses while separately evaluating answering and reliable test construction. By automating this process and allowing test difficulty to evolve with model capabilities, LivingArena provides a "living" benchmark for model development, red-teaming, and capability-aware multi-agent coordination. We publicly release our code: https://github.com/galaxyChen/LivingArena
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.16337v4">Medical Heuristic Learning: An LLM-Driven Framework for Interpretable and Auditable Clinical Decision Rules</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      Predictive modeling for clinical decision support requires both strong predictive performance and transparent, auditable, and human-reviewable decision logic. Although deep learning and tree-based ensemble methods can achieve high accuracy, their black-box nature remains a major obstacle to trustworthy clinical deployment. Moreover, clinical prediction often operates under practical constraints, including limited sample sizes, severe class imbalance, and feature evolution arising from changes in diagnostic criteria or clinical documentation practices. We propose Medical Heuristic Learning (MHL), a constrained paradigm for LLM-assisted rule learning. Rather than relying on updates to implicit model weights, MHL integrates statistical probes, medical knowledge probes, initial rule synthesis, and iterative rule optimization to construct an executable rule-based expert system. The resulting rule system is expressed entirely using the native logical and control-flow constructs of a programming language. Valid rule versions are recorded and retained along the search trajectory, making the decision logic explicit, interpretable, and auditable. MHL also supports continual learning by using previously validated rules as a starting point and iteratively revising them in response to updated feature information under data drift or feature evolution. MHL is not tied to any specific programming language. Comprehensive experiments on medical datasets show that MHL achieves predictive performance comparable to that of state-of-the-art methods, performs favorably in small-sample and highly imbalanced settings, and supports the transfer and adaptive revision of validated rules under feature evolution. Overall, these findings suggest that non-gradient-based heuristic systems offer an approach to balancing predictive performance and transparency in clinical decision support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.02750v1">Bilevel Coordinated Reflection: A Game-Theoretic Approach to Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      Multi-agent LLM systems commonly use an orchestrator to decompose a task for a team of workers and then improve through textual reflection. Despite strong empirical results, these systems lack a unified account of coordination, memory improvement, and the role of external verification. We model orchestrator-worker interaction as a bilevel coordination game: under bounded coupling, the workers' local-update game is an approximate potential game whose equilibrium slack is controlled by decomposition quality. We then analyse reflection as stochastic movement over semantic memory states. For free-form reflection, we derive a finite-time upper bound, prove worst-case tightness, and give a positive lower bound under a falsifiable persistent-harm condition. We further prove an information-theoretic impossibility result: no gate that observes only the generated transcript can improve uniformly over text-indistinguishable environments, whereas an environment-grounded gate can. Motivated by this separation, we introduce Stochastic Reflective Memory Ascent (SRMA), which accepts a candidate memory only after a grounded evaluation risk strictly decreases. Under calibration and non-degenerate corrective mass, SRMA converges exactly, geometrically or polynomially; matching constructions show that both rate regimes are order-tight. We also provide confidence gating for stochastic evaluation and re-anchoring guarantees for piecewise-stationary environments. Experiments instantiate these objects with environment-grounded metrics and test the predicted coordination and drift laws. On 500 SWE-bench instances, the complete Kimi-based system resolves 72.2% versus a 70.8% public mini-SWE-agent reference. Code: https://github.com/YihangChen9/Bilevel-Coordinated-Reflection
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.02745v1">Incremental Pooled LLM Evaluation for Cost-Effective Retrieval Model Selection</a></div>
    <div class="paper-meta">
      📅 2026-09-02
      | 💬 10 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Selecting a retrieval model for a production RAG system requires reliable comparative evaluation, but obtaining relevance judgments at scale is expensive and difficult to repeat as new candidate systems arrive. We study pooled LLM evaluation, in which an LLM judges the union of documents retrieved by the current set of candidate systems, and the pool is then expanded incrementally as new systems are introduced by judging only the new documents they contribute. These judgments are reused to evaluate all systems on a common basis. We validate this approach on four retrieval benchmarks with 11 systems spanning dense, sparse, and hybrid configurations, and deploy it to compare 62 retrieval configurations for a financial news QA system. Pooled LLM rankings correlate strongly with gold-standard evaluation across datasets, and 97% of pairwise system orderings are preserved once bootstrap uncertainty in the qrels is taken into account. In production, document overlap yields 65-80% judgment reuse and up to 4.9x lower evaluation cost, allowing teams to benchmark new retrieval candidates without re-judging previously assessed documents. These results suggest pooled LLM evaluation is a practical and cost-effective workflow for incremental retrieval model selection in deployed systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.02730v1">CORAL: An LLM-Native Harness for Production Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2026-09-02
      | 💬 Accepted by RecSys '26 OARS Workshop
    </div>
    <details class="paper-abstract">
      Production recommender systems shape what billions of people see, and sustaining their performance requires continual optimization: as content, user behavior, and upstream models shift, the choices governing retrieval, ranking, and serving must be revisited. Traditionally, human engineers test such changes through online experiments--a slow, reactive process limited by engineering effort, leaving parts of the system unrevised as conditions change. Although large language models have been applied to ranking, user modeling, and offline model development, few systems place an agent in a continual closed loop that acts on a live recommender and learns from the measured effects of its decisions. We present CORAL (Constraint-Optimized Recommender via an Agentic Loop), an LLM-native harness that closes this loop: each cycle, the agent observes operating signals, reasons over a memory of past decisions and outcomes, and invokes tools--including a numerical optimizer that keeps changes within a fixed operating budget--to reconfigure the recommender, with measured outcomes informing the next cycle. We formulate this as a partially observed, non-stationary, constrained optimization problem in which the policy improves in context, without parameter updates, from its prior actions. Across two large-scale social platforms, evaluated with A/B experiments, the same harness improves engagement at no additional serving cost on one and reduces serving cost without degrading engagement on the other, spanning the engagement-efficiency frontier. Performance improves as the loop iterates, suggesting that a single agentic loop can automate continual optimization work traditionally performed by human algorithm engineers under explicit guardrails.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.10103v2">LLM Watermarking as Big Data Provenance: A Deployment-Oriented Systematization</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become widely deployed, their outputs can be copied, transformed, and redistributed at scale without reliable evidence of origin, creating risks for trust, accountability, intellectual property (IP) protection, and high-stakes decision-making. LLM watermarking addresses this problem by embedding detectable signals into text during or after generation. However, existing methods vary in design assumptions, threat models, and evaluation criteria, while deployment choices such as watermark placement, detection authority, and key management affect reliability, security, and scalability. This paper systematizes LLM watermarking as provenance infrastructure for large-scale data ecosystems. We organize existing approaches along four deployment dimensions: insertion point, verification authority, operational state, and transformation threat model, and relate them to the big data requirements of Volume, Velocity, Variety, Veracity, and Value. We further introduce a Big Data Watermarking Readiness framework centered on four deployment workloads: online generation, streaming detection, transformation pipelines, and ecosystem governance. The framework connects these workloads to system-level requirements including throughput, false-positive control, robustness, cross-domain reliability, governance, and downstream utility. Our analysis highlights a gap between benchmark performance and deployment readiness: false positives accumulate at scale, repeated transformations weaken watermark signals, computational overhead can limit online deployment, and centralized verification can create governance bottlenecks. We conclude with an evaluation blueprint and research directions for scalable, trustworthy provenance in big data ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.02679v1">From Tokens to Semantics: Leveraging Complementary Signals for Hallucination Detection in Black-Box LLMs</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      When LLMs support public-facing or high-stakes workflows, missed fabrications can harm users and institutions, while false alarms consume limited human-review capacity. When no trusted context or reference document is available, we study two signals accessible through black-box model APIs: semantic entropy, which measures disagreement among sampled response meanings, and uncertainty derived from token log-probabilities. Their failure modes can be complementary: semantic entropy becomes uninformative when responses form one semantic cluster, while token uncertainty can miss consistently confident errors. We extend token-based uncertainty detection by aggregating token-level signals across sampled responses through our TopK method, evaluate the hybrid CoCoA method, which combines target-response uncertainty with semantic dissimilarity, and propose and study two supervised methods: Gated, which routes single-cluster cases to an aggregated-token-feature classifier, and Stacked, which learns jointly from semantic uncertainty and broader token features. We evaluate seven benchmarks, including five public benchmarks (four text datasets and multimodal handwritten-cheque extraction) and two constructed benchmarks (Financial Summaries and Long-Text QA), using four language models. In our evaluation across models and datasets, Stacked gave the best performance in nearly half of the cases, while TopK and CoCoA remain competitive without supervised training labels, although their thresholds require careful calibration. No method is universally strongest. We therefore evaluate performance at false-positive-rate budgets from 1% to 15%, assess their sensitivity to generation and calibration choices, and examine variation across dataset characteristics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.22230v2">Whitewashing Hate, Smearing Harmless Content: Annotator-Style Rebuttal Attacks on LLM-Based Moderation</a></div>
    <div class="paper-meta">
      📅 2026-09-02
      | 💬 We identified errors in the experimental setup and analysis that affect several key results and conclusions. As substantial re-analysis is required and the conclusions may change, we respectfully request withdrawal of the current version
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used for hate speech moderation, often within human--AI workflows in which reviewers provide feedback before a final decision. Such feedback introduces two manipulation directions: whitewashing hateful content as normal and smearing normal content as hateful. This study examines the susceptibility of initially correct model judgments to annotator-style rebuttals and analyzes whether attack effectiveness differs across manipulation directions. We introduce a rejudge protocol that extends direct contradiction with decision-boundary perturbations and adversarial rationales. Experiments with multiple LLMs on two hate speech datasets show that annotator-style rebuttals substantially degrade moderation performance, with stronger effects in multi-turn settings. The results further reveal stable, model-specific asymmetries between whitewashing and smearing across attack configurations, indicating distinct directional vulnerability patterns. Explicit reasoning prompts and defensive instructions reduce these effects but do not eliminate them. These findings highlight the need for direction-aware safeguards and dedicated feedback-robustness evaluation in human--AI moderation workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.02580v1">Competitive Market Behavior of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed as economic agents, yet there is little evidence whether LLM agents are suited for participating in market mechanisms designed for humans, and whether these mechanisms deliver desired outcomes when faced with LLM agents. We address this question by replicating seminal economic experiments, replacing human subjects with LLM agents. We place agents in a double auction environment, which is a widely-used market mechanism. We check whether such a market is able to deliver an efficient allocation of resources, thereby testing a novel dimension of alignment of LLM agents -- their compatibility with a fundamental market mechanism. We find that markets populated by LLM agents exhibit slower or no convergence towards market equilibrium, thus providing less efficient allocations than markets populated by humans. We then analyze agents' individual trading decisions and find substantial heterogeneity both across model families and market roles. We also run a lexical analysis of Chain-of-Thought (CoT) traces generated by the agents. We find that the decision to execute a trade rather than continue incrementally adjusting prices is associated with a shift from strategic considerations toward urgency. We publicly release our testing framework, which can be used for future evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03043v2">The Geometry of LLM-as-Judge: Why Inter-LLM Consensus Is Not Human Alignment</a></div>
    <div class="paper-meta">
      📅 2026-09-02
      | 💬 Accepted in The 2026 Conference on Empirical Methods in Natural Language Processing
    </div>
    <details class="paper-abstract">
      LLM judges now score most open-ended NLP output, and their mutual agreement is routinely read as evidence that the scores can be trusted. That reading is unsafe: judges may agree because they capture quality, or because they share the same blind spots, and agreement statistics alone cannot tell these apart. We develop a geometric test that can. Treating each judge's scores as a vector, we measure spread, effective rank, the angle to human scores, and the judge-judge, judge-human, and human-human agreement triple for 42 judges on two community-built Indic benchmarks covering four domains and eight languages. Every comparison is reference-matched: a judge and a held-out rater are scored against the same two-rater mean, since an averaged reference otherwise flatters judges by several degrees. On subjective rubrics, judges agree with one another as much as humans do yet reach only 58-66% of human agreement and often concentrate on an axis humans do not weight. On the one rubric with a verifiable answer, most of that gap closes. Ensembles converge on the judges' shared axis rather than the human one, and training widens scores without rotating them. Inter-judge agreement is evidence of human alignment only after this check passes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.02553v1">The Shape of Ownership: Verifying LLM Provenance through Semantic Structures</a></div>
    <div class="paper-meta">
      📅 2026-09-02
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly redistributed, adapted, and served behind opaque APIs, model ownership can no longer be established reliably by inspecting model internals or deployment records. This creates a need for behavioral signatures that remain observable through black-box interaction. Yet most existing black-box fingerprints instantiate ownership signals through fixed query-key associations, reducing model identity to sparse memorized associations detached from ordinary behavior and limiting both robustness and stealth (e.g., fine-tuning or quantization) and stealthiness. A stronger fingerprint should instead be distributed, naturally elicited, and expressed at a higher semantic level. To this end, we introduce PROSE (Provenance through Relational Organization of Semantic Expression), replacing fixed query sets with a target semantical domain and brittle response keys with semantic structures internalized as domain-conditioned response behavior. Specifically, the fingerprint is encoded in how the model semantically organizes its in-domain conclusions, rather than in particular tokens or prescribed outputs. PROSE constructs a private bank of domain-specific semantic templates, internalizes them through mixed fine-tuning on structurally verified and clean responses, and verifies ownership by detecting the designated structures in responses to held-out natural queries. Extensive experiments across multiple model architectures, scales, and target domains show that PROSE achieves a 100% fingerprint detection rate on unmodified models with no observed false positives, preserves model utility, and retains strong detectability under downstream modifications and output transformations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.02548v1">Learn from Whoever Is Right: Answer-Verified Multi-Teacher Distillation for Multi-Domain LLMs</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      Modern large language models (LLMs) rely on reinforcement learning to build strong capabilities in individual domains, but integrating those capabilities into a single deployable model remains challenging. By routing each sample to the teacher whose domain matches it, existing approaches let a domain label decide which teacher provides supervision. However, domain expertise holds only on average: the matched teacher is not always correct on a given sample, while a teacher from another domain sometimes is. The reliable teacher therefore has to be identified per sample, not per domain. In this paper, we introduce Multi-Teacher Self-Distillation Policy Optimization (MT-SDPO), an on-policy distillation method that unifies several frozen teachers into one student model. MT-SDPO consists of three components: (1) self-anchors, where a rollout is supervised by a correct rollout from its own group; (2) answer-verified eligibility, where a teacher may supervise a sample only if its own answer passes a verifier; and (3) privileged distillation, which merges the anchor and all verified feedback into one context that an exponential moving average self-teacher reads and the student does not, thereby keeping one policy at deployment. Across five students from three model families, MT-SDPO lifts the weakest domain of Qwen3-8B by 14.79 points and narrows its domain gap by 74.7%, a better balance than serving one matched teacher per domain. Verified reliability, not domain membership, should decide who teaches. Code is available at https://github.com/hexixiang/MT-SDPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.02514v1">AceSpec: An Asymmetric Edge-Cloud Collaborative Framework for Communication-Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      Deploying Large Language Models (LLMs) on edge devices typically relies on model compression or split inference. However, compression degrades reasoning capabilities, while split inference suffers from severe Wide Area Network (WAN) communication bottlenecks. Edge-cloud speculative decoding emerges as a promising alternative, leveraging an edge small model to draft tokens for cloud verification. Yet, over volatile WANs, inevitable prediction rejections trigger catastrophic pipeline stalls and network-wide rollbacks, neutralizing collaborative gains. To overcome this, we propose AceSpec, an asymmetric edge-cloud collaborative framework. AceSpec utilizes un-saturated edge compute to proactively construct a probabilistic state cache, effectively transforming network-wide pipeline flushes into $\mathcal{O}(1)$ local memory lookups. To preserve bandwidth, it employs an asymmetric communication protocol that transmits minimal main-chain indices uplink and compact sparse distributions downlink. Furthermore, we introduce a network-aware, Lagrangian-optimized resource allocation strategy that dynamically maximizes the local cache hit rate. Evaluations demonstrate that AceSpec achieves up to a 3.52$\times$ throughput speedup and exhibits exceptional bandwidth immunity, sustaining near-peak inference performance even under severely constrained 50 Kbps WAN conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10118v2">Reviewing the Reviewer: LLM-Assisted Reviewer Feedback Generation for Guideline Compliance</a></div>
    <div class="paper-meta">
      📅 2026-09-02
      | 💬 Accepted at EMNLP Main, 2026
    </div>
    <details class="paper-abstract">
      Peer review is central to scientific quality, yet reliance on simple heuristics, namely lazy thinking and non-specific critiques, has threatened review quality. Prior work frames lazy thinking detection as single-label classification and stops at detection, yet review segments often exhibit multiple co-occurring issues, and reviewers benefit more from actionable, guideline-aware feedback than from labels alone. We further show that off-the-shelf LLMs prompted for feedback frequently rewrite the entire review or address the authors rather than the reviewer, motivating an inference-time approach. We introduce an LLM-driven framework that decomposes reviews into argumentative segments, identifies issues violating ACL Rolling Review (ARR) guidelines, and generates targeted feedback using issue-specific templates refined by a novel iterative, reranking-based generation algorithm. In a controlled rewriting study, our feedback reduces guideline violations by up to 92.4\%. We also release LazyReviewPlus, the first multi-label dataset of 1,309 sentences annotated for detecting lazy thinking and lack of specificity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.06992v2">GPTKB 2.0: Browsing, Querying, and Auditing a Disambiguated LLM-Derived Knowledge Base</a></div>
    <div class="paper-meta">
      📅 2026-09-02
      | 💬 Accepted to EMNLP 2026 Demo Track
    </div>
    <details class="paper-abstract">
      We present a web demo for exploring a large-scale disambiguated knowledge base (KB) materialized from a large language model (LLM). GPTKB 2.0 contains 38.4M triples over 1.6M canonical entities, together with 207.6K consolidated relations and 66K consolidated classes. Unlike prior LLM-derived knowledge bases that largely identify entities by surface strings, GPTKB 2.0 performs context-guided disambiguation during recursive KB construction, separating homonyms and merging synonymous mentions as facts are elicited. The demo makes this process inspectable: users can browse entities, follow links across the KB, and audit the provenance of individual facts, including surface forms, candidate matches, source triples, and disambiguation decisions. The interface further supports structured SPARQL queries, natural-language questions translated to SPARQL, and entity linking from user-provided text to canonical GPTKB 2.0 entries. GPTKB 2.0 is available at https://gptkb.org/, with the full KB downloadable for offline use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.17039v3">Evaluating the Evaluator: Summarization Metrics and LLM-Judges beyond English</a></div>
    <div class="paper-meta">
      📅 2026-09-02
      | 💬 SEPLN 2026
    </div>
    <details class="paper-abstract">
      Automatic text summarization relies on automatic evaluation to quickly determine the quality of summarization models via automatic metrics and LLM-as-a-Judge models. However, these techniques require meta-evaluation to ensure that they capture human judgments correctly. In this paper, we explore this meta-evaluation beyond English by generating a new multilingual summary meta-evaluation dataset (BASSE), which comprises human judgments on 2,040 abstractive summaries, generated either manually or by five Large Language Models (LLMs) with four different prompts. For each summary, annotators evaluate five criteria on a 5-point Likert scale: coherence, consistency, fluency, relevance, and 5W1H. We then benchmark automatic summarization metrics and LLM-as-a-Judge models. Our results show that currently proprietary judge LLMs have the highest correlation with human judgments, followed by criteria-specific automatic metrics, while open-sourced judge LLMs perform poorly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.02482v1">How LLMs Build Fictional Worlds: Setting and Narrative Space in AI-Generated Creative Storytelling</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      In this paper, we analyze how Large Language Models (LLMs) employ worldbuilding strategies, focusing on setting as one measurable dimension of storyworld construction. We compare 1,000 AI-generated stories per model in English and German with human-authored fiction from Project Gutenberg. Building on prior work, we operationalize setting through five types of narrative space: "action", "perceived," "visual," "descriptive" and "no space", identified using fine-tuned BERT classifiers for German and English. We generate narratives using GPT 4.1, LlaMA 3.3, Mistral 3.2, and Gemma 3 and compare their spatial distributions to a human-authored baseline. We find that human-authored texts predominantly employ "action space," grounding narratives in embodied character-environment interaction, whereas LLMs systematically overproduce "perceived space," emphasizing atmosphere and affect. This divergence remains stable across narrative time. Overall, our findings show that LLMs exhibit worldbuilding patterns that differ consistently from human-authored fiction in ways that are both model-specific and language-sensitive.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.02473v1">Learning to Fuse LLMs with Ontology Rankers for Rare-Disease Diagnosis</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      Ontology rankers remain useful for rare-disease diagnosis because each candidate can be traced to matched patient phenotypes. Large language models (LLMs) can generate differential diagnoses from the same patient description, but their predictions lack an equally clear evidence trail. Rather than asking which system should replace the other, we ask whether an LLM can improve the ranker without giving up its evidence. Our behavior-based fusion model examines the two ranked lists, their agreement, and the ontology support behind each candidate, and learns how much to rely on each system for the individual case. Before comparison, we remove a documented test-set leakage pathway caused by benchmark cases and ontology annotations being derived from the same publications. Across eight open LLMs, fusion improves Phenomizer Recall@1 by 7.86 percentage points on Phenopacket Store and 20.18 points on RAMEDIS. When paired with DeepSeek-V4-Flash through an API, a fusion model trained only on the other LLMs improves Recall@1 from 0.1657 to 0.2176, a 5.19-point gain, without retraining. For 90.8% of correct fused diagnoses, the disease retains candidate-level ontology evidence that can be inspected. These results show that LLMs can strengthen an established diagnostic tool without discarding the structured evidence that makes it useful.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.24661v4">Measuring Reasoning Quality in LLMs: A Multi-Dimensional Behavioral Framework</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      Despite remarkable progress on reasoning benchmarks, current LLM evaluation practice remains anchored to final-answer correctness, providing limited insight into how models reason, how reliably they behave under contextual variation, or how efficiently they reach conclusions. This paper proposes a unified multi-dimensional framework for measuring LLM reasoning quality from a behavioral perspective, operationalizing six theoretically grounded dimensions rooted in cognitive science: Correctness (CQ), Consistency (CS), Robustness (RS), Local Logical Coherence (LS), Efficiency (ES), and Stability (SS). The framework introduces deployment-aware aggregation, enabling context-specific model selection beyond accuracy-based leaderboards. Experiments across multiple LLMs and benchmarks reveal behaviors systematically concealed by single-metric evaluation, including the orthogonality of local logical coherence and correctness, deployment-context-dependent ranking inversions, and non-trivial dimensional profiles in small locally-deployed models. Discriminant validity analysis confirms that the proposed dimensions capture largely non-redundant signals. The resulting pipeline provides a foundation for diagnosing LLM reasoning behavior across deployment contexts, with domain-specific validation as a direction for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.00030v4">SignBind-LLM: Multi-Stage Modality Fusion for Sign Language Translation</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      Current sign language translation (SLT) systems attempt to learn all aspects of signing---manual gestures, high-speed fingerspelling, and asynchronous non-manual facial cues---within a single end-to-end network. Learning multiple tasks without detailed supervision leads to poor recognition of fingerspelled proper nouns and technical terms, and leaves rich disambiguating information from lip movements largely unexploited. We introduce SignBind-LLM, a modular framework that addresses these limitations through three dedicated expert streams: one for continuous signing, one for fingerspelling, and one for lipreading. Each expert is pre-trained independently using CTC on approximately two million automatically generated pseudo-gloss sequences, removing the need for manual gloss annotation. A lightweight transformer with learned temporal alignment fuses the expert outputs, and a pre-trained language model translates the resulting pseudo-gloss sequences into fluent spoken English. At matched decoder scale (250M parameters), our architecture already surpasses all prior methods, confirming that the gains are architectural rather than a consequence of scaling the language model. Scaling to a larger decoder sets a new state-of-the-art across How2Sign: 23.1, BOBSL: 7.0, and ChicagoFSWild+: 73.6%, while requiring significantly lower training cost than prior approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12406v2">Isolation as a First-Class Principle for LLM-Agent System Safety: Concepts, Taxonomy, Challenges and Future Directions</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      The capability of LLM agents to function as the ``brain'' of a system fundamentally expands the scope of analysis beyond a standalone model. Consequently, safety is no longer only about input--output content alignment. It also concerns system behavior and real-world execution outcomes. However, the current literature is fragmented across attack types, applications, and benchmarks. This makes it hard to explain why failures such as prompt injection, tool misuse, and memory poisoning often share the same structural cause, and how they spread through an agent workflow. In this survey, we treat isolation as a first-class principle for LLM-agent system safety. By isolation, we refer to the separation of user inputs, tool access, execution channels, inter-agent communication, and environment-originated context. We organize the literature with a boundary-centric taxonomy of five boundaries: user-agent, agent-tool, agent-execution, agent-agent, and system-environment. This view helps identify where the loss of isolation first occurs, how compromise propagates across boundaries, and which defenses are most relevant at each interface. We also summarize cross-boundary failure paths, discuss open challenges, and outline a research agenda for isolation-by-construction in future agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15662v4">Stepwise Think-Critique: Interleaved Reasoning and Self-Critique in a Single LLM</a></div>
    <div class="paper-meta">
      📅 2026-09-02
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Human beings solve complex problems through critical thinking, where reasoning and evaluation are intertwined to converge toward correct solutions. However, most existing large language models (LLMs) treat the reasoning and verification as separate processes: they either generate reasoning without explicit self-checking or rely on external verifiers to detect errors post hoc. The former lacks immediate feedback, while the latter increases system complexity and hinders synchronized learning. Motivated by human critical thinking, we propose Stepwise Think-Critique (STC), an end-to-end trainable framework in which a single LLM emits a structured, step-level critique inline with each reasoning step. STC is trained with reinforcement learning that complements reasoning rewards with a critique-consistency reward derived from final-answer correctness, jointly optimizing reasoning correctness and critique reliability. On five mathematical reasoning benchmarks, STC improves Pass@1 by 7.2% over the 1.5B base model and attains 67.4% step-level critique F1, surpassing seven external process reward models evaluated at their per-dataset oracle thresholds---a step toward LLMs with built-in critical thinking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04165v3">PoC-Gym: Towards More Reliable LLM-Assisted Proof-of-Concept Exploit Generation</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      Recently Large Language Models (LLMs) have been used in security-related tasks, including generating proof-of-concept (PoC) exploits. Several LLM-assisted approaches have been proposed; they typically generate PoCs from vulnerability descriptions and use additional guidance. But, such approaches are often ineffective because the signals-such as printed markers, generated files, or runtime side effects-that they use for validation may not imply that the vulnerability is triggered. Research for more reliable PoC generation is in need but yet remains challenging. We propose PoC-Gym, a pipeline for LLM-based PoC generation for Java security vulnerabilities. PoC-Gym uses both static and dynamic information, e.g., CVE-tailored prompts, static traces, and coverage-based feedback, and iteratively generates PoC candidates. Each candidate goes through a series of validations: whether the execution is complete, manifests a success signal, and reaches the sink of the target trace. We evaluate PoC-Gym using 20 Java CVEs. Across 338 runs, 116 candidates pass PoC-Gym's runtime validation and 65 candidates pass post-hoc validation against the ground-truth vulnerable locations, covering 12 of the 20 CVEs. On the 14-CVE overlap with FaultLine, the strongest PoC-Gym configuration is post-hoc valid for 8 CVEs, while FaultLine reports success for 5 CVEs under its original evaluation criterion. But, given the complexity of PoC generation, PoC-Gym also generates many runtime-valid but post-hoc-invalid PoCs. To better understand how to achieve more reliable PoC generation, we present an in-depth analysis of such PoCs and identify common sources of failures. We believe that our work provides insights for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.02379v1">MultiGhostBench: A Multilingual Benchmark for Long-Form LLM-Generated Text Attribution under Distribution Shifts</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      While existing work on LLM authorship attribution (AA) has made progress, available benchmarks remain limited, often focusing on English, controlled settings, or relatively outdated models, with the few multilingual studies considering only relatively short texts. We introduce MultiGhostBench, a multilingual benchmark comprising 928 books generated by five recent LLMs across six languages and three scripts, with an average length of approximately 59K words per book. The benchmark supports evaluation under domain, author, and language shifts. Evaluation of representative AA methods shows that no single method consistently performs best across settings, and performance generally degrades under distribution shifts. Transformer-based detectors can retain generator-related information across languages, although transfer effectiveness varies by language pair, whereas statistical and fingerprint-based detectors are more language-dependent. We envision MultiGhostBench as a valuable resource for the development and evaluation of robust AA methods. The dataset and code can be found at https://github.com/GrecoMT/MultiGhostBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18579v2">ICE: Intervention-Consistent Explanation Evaluation with Statistical Grounding for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      Evaluating whether explanations faithfully reflect a model's reasoning remains an open problem. Existing benchmarks use single interventions without statistical testing, making it impossible to distinguish genuine faithfulness from chance-level performance. We show that faithfulness is not a fixed property but an operator-dependent quantity that changes with the intervention method used to measure it. We introduce ICE (Intervention-Consistent Explanation), a framework that evaluates explanations against random baselines of equal size under multiple operators. Evaluating 7 LLMs across 4 tasks with deletion and retrieval infill operators, we find that switching operators crosses the positive-evidence threshold in 18% of configurations (5 of 28 attention comparisons), with gaps reaching 44 percentage points. Randomized baselines detect anti-faithfulness (explanations worse than random) in nearly one-third of English deletion configurations, invisible without random comparisons. These patterns persist across 6 non-English languages and 2 attribution methods. The methodology generalizes to step-level chain-of-thought evaluation, where preliminary results on 3 frontier models suggest that high accuracy does not imply faithful reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.02353v1">LLM-Driven Joint Evolution of Coupled Heuristics Components for Routing Optimization</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      Heuristic design for combinatorial optimization remains heavily reliant on expert knowledge, while existing large language model (LLM)-enhanced evolutionary methods typically evolve isolated algorithmic components, even when one determines the search state on which another operates. This paper proposes LLM-driven Heuristic Components Joint Generation (LLM-HCJG), a population-based framework that jointly generates and co-evolves interdependent heuristic components under a shared design blueprint. Applied to guided local search (GLS), LLM-HCJG couples solution initialization with penalty construction and embeds the generated pair into an enhanced online search mechanism. The resulting form is further transferred from the traveling salesman problem (TSP) to the capacitated vehicle routing problem (CVRP). Theoretical analysis establishes the non-separable state-transition effects between the two components and the advantage in generation consistency. Across synthetic instances and 41 public TSPLIB/CVRPLIB benchmarks, LLM-HCJG attains consistently low optimality gaps, including best or tied-best results on 28 of 29 TSPLIB instances and all 12 CVRPLIB instances. Ablation and structural analyses further indicate that these gains are associated with cross-component compatibility and alignment rather than isolated-component recombination. These results support effective cross-instance transfer within the evaluated routing settings under limited-sample, modest-cost training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.02265v1">CAPTURE: Disentangling Preference Drift from Memory Poisoning in Personalized LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-09-02
      | 💬 Under review at ICLR 2027
    </div>
    <details class="paper-abstract">
      Personalized language agents use persistent memory to adapt to users over time, but the same mechanism creates an attack surface. When new information conflicts with stored preferences, an agent must distinguish genuine preference drift from temporary context shifts, ambiguity, or adversarial memory poisoning. We formulate this problem as a continuous-time partially observable decision process over a latent user state and show why rules based only on recency and provenance are insufficient. CAPTURE addresses this ambiguity with a neural differential-equation belief tracker, a multi-timescale memory ledger, uncertainty-triggered clarification, and counterfactual auditing of cited memories. On 480 held-out episodes from 96 users, CAPTURE achieves a 71.5% win rate, compared with 69.3% for an identically supervised baseline and 66.1% for the strongest heuristic baseline. It limits fixed-policy poisoning success to 11.5% while accepting 83.5% of genuine preference updates. Under an adaptive attacker with access to the released weights, attack success rises to 24.7%, exposing a real adaptation-security tradeoff. We further evaluate the frozen system zero-shot on an independently constructed benchmark and replay longitudinal interaction histories from 40 users collected over two to three weeks. These results suggest that modeling preference authenticity explicitly can improve both personalization and robustness in memory-augmented LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.02264v1">Codebook Agent: Amortized Topology Design for LLM Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      Adapting the communication topology of an LLM multi-agent system to each query improves both accuracy and efficiency, yet current designers treat this as conditional graph generation: a variational, autoregressive, or diffusion decoder searches the $N \times N$ adjacency space, and a graph-network proxy trained on utility and a structural cost such as edge count ranks the sampled candidates. We argue that this formulation is misaligned with the problem. Empirically, topologies that survive a reward filter collapse to about six distinct graphs even when the codebook capacity grows from 8 to 64; edge count is negatively correlated with measured token consumption (Pearson $r \approx -0.4$), so sparsifying the graph makes inference more expensive; and a message-passing scorer over agent-profile nodes is adjacency-invariant whenever agents share a profile---the default configuration of published benchmarks---so it cannot rank candidates at all in that regime. These three facts motivate Codebook Agent: a vector-quantized autoencoder compresses successful topologies into a query-independent 16-entry codebook; a reward-weighted MLP maps the query embedding to a distribution over codes; and an MLP proxy that reads the flattened adjacency, regressed on measured utility and per-task normalized token cost, reranks the top decoded candidates in a single batched forward pass. With no iterative search and no message passing at test time, Codebook Agent is the most accurate method on all six benchmarks we compare (84.6 average against 83.0 for the strongest prior designer), emits a topology in 2.4 ms, and uses 21.9--33.2% fewer LLM tokens.
    </details>
</div>
