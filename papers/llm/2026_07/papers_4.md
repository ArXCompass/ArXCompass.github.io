# llm - 2026_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- Part 4

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01457v1">Grounded Optimization: A Layered Engineering Framework for Reducing LLM Hallucination in Automated Personal Document Rewriting</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 13 pages, 1 figure. Equal contribution by both authors. Code and data: https://github.com/shashank-indukuri/grounded-optimization
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly applied to resume optimization for applicant tracking systems, introducing hallucination failures distinct from general text generation: anachronistic technology injection, cross-domain terminology contamination, structural mutation, and content fabrication. We present Grounded Optimization, a five-layer framework combining temporal context validation, deterministic contamination detection, structural invariant enforcement, prompt-level grounding, and an evaluator agent. In ablation experiments across three LLMs, four temperature settings, and six layer configurations on 25 synthetic resumes spanning 14 industries, undefended baselines produce 2.48-5.36 detected hallucinations per resume. Among detectors independent of the active defenses, temporal hallucinations are reduced by 50-95% across all conditions; overall detected hallucination rate falls to 0.04-0.24. Prompt-level grounding alone achieves zero detected hallucinations at low temperature with a capable instruction-following model; higher temperatures and weaker models reveal the need for the deterministic layers as a complement. We release the contamination taxonomy, evaluation code, and raw data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01440v1">FaithMed: Training LLMs For Faithful Evidence-Based Medical Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 15 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Faithful reasoning is essential in medicine, where clinical decisions require transparent justification grounded in reliable evidence. Current medical LLMs either lack active access to evidence or use retrieved evidence without supervising how it should be appraised and applied during reasoning. To address this, we formalize evidence-based medicine principles as process-level criteria and introduce FaithMed, a framework that combines clinician-designed, automatically refined rubrics with reinforcement learning using step-level process reward assignment and advantage grouping. Across seven medical benchmarks, FaithMed improves over agentic-search baselines (+9% on average) and outcome-only RL (+5.8%), while raising average evidence-based medicine rubric scores over agentic-search Qwen3 baselines (+15.5%). This work demonstrates that explicit step-level supervision can improve both task success and the faithfulness of the reasoning process. Code is available at https://github.com/cxcscmu/FaithMed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01431v1">IsoSci: A Benchmark of Isomorphic Cross-Domain Science Problems for Evaluating Reasoning versus Knowledge Retrieval in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      We introduce ISOSCI, a benchmark of isomorphic cross-domain science problem pairs that separates reasoning ability from domain knowledge retrieval in LLM evaluation. Each pair shares identical logical structure but requires different domain-specific knowledge, enabling controlled attribution of reasoning-mode gains. Across five model pairs spanning four model families, we find that 91.3% of reasoning-mode gains are knowledge-dependent rather than structure-invariant (63/69 gains; Wilson 95% CI [82.3%, 96.0%]), directly challenging the assumption that chain-of-thought reasoning improves short-horizon procedural scientific problem-solving. Reasoning toggles on highly capable models provide less than 5 percentage points accuracy gain across all domains, and a reasoning-specialized model (o3-mini) that outperforms its standard counterpart on GPQA Diamond (+19.2 percentage points) underperforms on ISOSCI (-24.7 percentage points), showing that benchmark choice determines conclusions about reasoning utility. We release ISOSCI at https://huggingface.co/datasets/isosci/isosci
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02813v3">HAL: Inducing Human-likeness in LLMs with Alignment</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      Aligning language models to qualitative behavioral traits, such as human-likeness, remains difficult because they are hard to define, measure, and optimize. As a result, improvements in human-like behavior are largely driven by scale or broad supervised training, rather than targeted alignment. We introduce Human Aligning LLMs (HAL), a framework for aligning language models to conversational human-likeness using an interpretable, data-driven reward. HAL derives explicit conversational traits from contrastive dialogue data, combines them into a compact scalar score, and uses this score as a transparent reward signal for alignment with standard preference optimization methods. Using this approach, we align models of varying sizes without affecting their overall performance. In large-scale Chatbot Arena-style human evaluations, a model aligned with HAL is more frequently perceived as human-like in conversation. Because HAL operates over explicit, interpretable traits, it enables inspection of alignment behavior and diagnosis of unintended effects. More broadly, HAL demonstrates how soft, qualitative properties of language--previously outside the scope for alignment--can be made measurable and aligned in an interpretable and explainable way.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11357v2">TileFuse: A Fused Mixed-Precision Kernel Library for Efficient Quantized LLM Inference on AMD NPUs</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 13 pages excluding reference, 11 figures
    </div>
    <details class="paper-abstract">
      With the growing demand for on-device LLM inference, edge SoCs increasingly integrate NPUs to improve performance and energy efficiency under tight power and thermal budgets. However, practical LLM deployment on current client NPUs remains difficult: widely used quantization formats such as AWQ do not map cleanly onto many existing NPU software stacks, which are often proprietary and expose limited low-level control. In this work, we present TileFuse, a close-to-metal mixed-precision kernel library for AMD XDNA2 NPUs that targets GEMM/GEMV-based operators in quantized LLM inference. TileFuse brings practical low-bit formats such as AWQ-style W4A16 and W8A16 directly onto XDNA2, rather than forcing the model to be reshaped around an NPU-specific quantization scheme. TileFuse co-designs weight layout, metadata placement, mixed-precision microkernels, and array-level dataflow. Specifically, it fuses unpacking, dequantization, and GEMM/GEMV execution into a single kernel flow, introduces an interleaved pre-tiling layout that supports GEMM dimensions up to 32K, and redesigns GEMV dataflow to utilize the full 4x8 AIE array. Across kernel-level evaluations, TileFuse improves performance by up to 121.6% for GEMM and 281% for GEMV over full-precision baselines, while delivering more than 2x performance and energy-efficiency gains over strong iGPU baselines on GEMM. In end-to-end LLM experiments on Ryzen AI laptops, TileFuse achieves up to 2.0x lower prefilling latency with more than 64.6% lower energy consumption. Together, these results show that XDNA2 is a practical target for AWQ-style edge LLM inference and that native NPU support for off-the-shelf quantization can make NPUs substantially more usable in real client deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.27806v2">Grounded Iterative Language Planning: How Parameterized World Models Reduce Hallucination Propagation in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      World models for language agents come in two useful forms. An agent-based world model calls an LLM API and reasons flexibly in language, but its errors appear as hallucinated state changes that are hard to score with ordinary regression losses. A parameterized world model is a trained transition predictor; its errors are easier to measure with quantities such as NodeMSE, delta accuracy, and validity accuracy, but it is usually weaker as a standalone planner. We compare these two families on four graph-structured planning benchmarks and introduce operational hallucination metrics for the agent-based case. The comparison motivates Grounded Iterative Language Planning(GILP), which trains only a small parameterized backbone and combines it with API-based agent reasoning. The backbone supplies valid actions, predicted state deltas, risk, and value; the LLM drafts an action and imagined delta; and a consistency gate asks for revision when the two disagree. On real GPT-4o-mini calls, GILP reduces hallucinated-state rate from 0.176 to 0.035. In calibrated simulator ablations, it raises success from 0.668 to 0.838 while adding only ~22% extra LLM calls.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.02964v2">Less Data, More Security: Advancing Cybersecurity LLMs Specialization via Resource-Efficient Domain-Adaptive Continuous Pre-training with Minimal Tokens</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 19 Pages; Updated content and authors list
    </div>
    <details class="paper-abstract">
      The increasing scale of AI workloads demands High-Performance Computing (HPC) infrastructure and training methodologies that are both scalable and sustainable. While Large Language Models (LLMs) demonstrate exceptional natural language capabilities, general-purpose models often lack the specialized domain knowledge necessary for effective cybersecurity analysis. We investigate Domain-Adaptive Continuous Pretraining (DAP) as a scalable, resource-efficient methodology for enhancing cybersecurity understanding in pretrained LLMs, implemented through a distributed Fully Sharded Data Parallel (FSDP) pipeline across multi-node GPU clusters. We systematically adapted three decoder-based architectures -- Llama-3.1-8B, DeepSeek-R1-Distill-Qwen-14B, and Llama-3.3-70B-Instruct -- using a curated 126-million-word cybersecurity corpus from standards, academic literature, and technical documentation. Evaluation across three cybersecurity benchmarks -- CTI-MCQ, CyberMetric, and SecEval -- demonstrates consistent improvements post-adaptation. Notably, our Llama-3.3-70B-Ins-DAP model achieves state-of-the-art performance with accuracies of 0.718, 0.933, and 0.864, respectively, surpassing parameter-efficient baselines and specialized models including Llama-Primus-Base (trained on 2.77 billion tokens) and Foundation-Sec-8B (trained on 5 billion tokens), despite utilizing only 118.8 million tokens -- representing a 23-to-42-fold reduction in training data. Targeted continuous pretraining via scalable HPC infrastructure enables effective cybersecurity domain adaptation with a substantially reduced computational and energy footprint, supporting specialized AI assistants in threat analysis, vulnerability assessment, and security documentation, while advancing sustainable and responsible AI development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01233v1">Measuring the Gap Between Human and LLM Research Ideas</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      LLMs are increasingly used to brainstorm research ideas, but existing evaluations mostly judge individual ideas by novelty, feasibility, or expert preference. We instead ask: how far are current LLM-generated ideas from human researchers? To characterize this gap, we build a large-scale evaluation framework for ideation from high-quality human research papers. For each paper, we reverse-engineer a small set of closely related prior works that likely inspired its core idea. LLMs are then prompted to generate a new idea from the set of paper titles and summaries. We introduce a two-axis research-taste taxonomy to profile each idea by its opportunity pattern and research paradigm, and use it to quantify the divergence between human and LLM ideas. Across idea sets generated by different LLMs, we observe a consistent distributional gap: LLM ideas are disproportionately concentrated around bridge-like opportunities and synthesis methods, whereas the human paper reference distribution spreads more broadly across ways of framing gaps and constructing contributions. This result suggests that strong LLMs can produce a range of reasonable ideas, but that range remains narrower than, and systematically shifted relative to, human research taste.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01313v1">Black-Box Inference of LLM Architectural Properties with Restrictive API Access</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      In practice, most commercial LLM providers do not publicly release details of underlying LLM architectures. However, prior work has shown that given limited API access to an LLM (namely, top-$k$ logits and/or a logit bias function), one can recover certain architectural details of an LLM, such as the hidden dimension of the feed-forward network. Perhaps in response to these results, most commercial LLM providers have restricted their APIs to expose only the single logit for each decoded token, and they no longer give users the ability to bias logits. We show that even under current restrictive APIs, several architectural parameters are still recoverable. We present NightVision, an attack that uses restrictive black-box API access to estimate the hidden dimension, depth, and parameter count of an LLM. Algorithmically, NightVision relies on a novel common set prompting technique in which multiple prompts expose log probabilities for the same set of output tokens; a spectral analysis of these results is used to infer hidden dimension. NightVision additionally uses end-to-end time to first token (TTFT) measurements and the estimated hidden dimension to estimate depth and parameter count. We empirically evaluate NightVision on 32 open-source LLMs, recovering hidden dimension to within 23% average relative error across all models (9% on MoE models), and depth and parameter count to within 53% for models exceeding three billion parameters. We run extensive ablations to demonstrate how these accuracies scale with token budget and model properties. Overall, our results suggest that current LLM APIs are not sufficiently restricted to fully obfuscate the architectural details of their underlying models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01213v1">RepoRescue: An Empirical Study of LLM Agents on Whole-Repository Compatibility Rescue</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      Open-source libraries and tools are widely reused, but compatibility maintenance is expensive. Once maintainers leave, useful repositories can stop working as runtimes and dependencies evolve. We study whether LLM agents can adapt old repositories to modern environments, a task we call compatibility rescue. Unlike bug repair, compatibility rescue starts from a repository that worked in its original environment but fails after ecosystem drift. RepoRescue gives agents only the repository and its failing modern environment; the agent must diagnose the failure, locate affected code, and produce a source-code rescue that restores the historical test suite. We build RepoRescue from 193 Python and 122 Java repositories, each verified to pass historically and fail after modernization. We evaluate five deployed agent systems on Python and three on Java. Beyond full-patch pass rate, we rerun patches after removing test-file edits to measure source-only repair, add a runtime-enforced regime that blocks test edits, and validate practical use for repositories whose suites pass after rescue. We find that Claude Code systems sometimes edit failing tests even when prompted not to; with runtime blocking, Kimi still rescues 41.5% of repositories. Systems are complementary: their union reaches 62.7%, exceeding the best single system by 10.9 points. Difficulty concentrates in cross-file coordination: on 14 repositories requiring coordinated whole-codebase changes, GPT-5.2 through Codex passes all 14, while every Claude Code system passes at most two. Finally, a passing suite is only an initial signal: among 34 unmaintained Python candidates whose suites pass after rescue, 22 work in realistic scenarios and 12 pass bug-hunt with patches that address the compatibility failure. RepoRescue benchmarks compatibility rescue with source-only auditing, runtime enforcement, practical validation, and reasoning labels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23370v2">FlexServe: A Fast and Secure LLM Serving System for Mobile Devices with Flexible Resource Isolation</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 Repeated paper uploading due to mistakes. See arXiv:2603.09046
    </div>
    <details class="paper-abstract">
      Device-side Large Language Models (LLMs) have grown explosively, offering stronger privacy and higher availability than their cloud-side counterparts. During LLM inference, both the model weights and the user data are valuable, and attackers may compromise the OS kernel to steal them. ARM TrustZone is the de facto hardware-based isolation technology on mobile devices, used to protect sensitive applications from a compromised OS. However, protecting LLM inference with TrustZone incurs significant overhead to both the secure inference and the normal aplications, due to two challenges: the inflexible resource isolation and the inefficient secure resource management. To address these challenges, this paper presents FlexServe, a fast and secure LLM inference system for mobile devices. The key idea is to decouple the access permission from the management permission of secure resources, so that the normal-world OS cannot access them but can still manage them as usual. First, FlexServe introduces a Recallable Resource Isolation mechanism to construct Recallable Secure Memory (Flex-Mem) and a Recallable Secure NPU (Flex-NPU). They can only be accessed by the secure world, but can be efficiently allocated and reclaimed by the normal-world OS. Based on them, FlexServe further introduces a FlexServe Framework to run secure LLM inference in the secure world. It works together with the normal-world OS to perform cooperative secure memory management. We implement a prototype of FlexServe and compare it with two TrustZone-based strawman designs. The results show that FlexServe achieves average TTFT speedups of 10.05X over the strawman and 2.44X over an optimized strawman.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09046v3">FlexServe: A Fast and Secure LLM Serving System for Mobile Devices with Flexible Resource Isolation</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 13 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Device-side Large Language Models (LLMs) have witnessed explosive growth, offering higher privacy and availability compared to cloud-side LLMs. During LLM inference, both model weights and user data are valuable, and attackers may even compromise the OS kernel to steal them. ARM TrustZone is the de facto hardware-based isolation technology on mobile devices, used to protect sensitive applications from a compromised OS. However, protecting LLM inference with TrustZone incurs significant overhead due to its inflexible isolation of memory and the NPU. To address these challenges, this paper introduces FlexServe, a fast and secure LLM serving system for mobile devices. It first introduces a Flexible Resource Isolation mechanism to construct Flexible Secure Memory (Flex-Mem) and Flexible Secure NPU (Flex-NPU). Both memory pages and the NPU can be efficiently switched between unprotected and protected modes. Based on these mechanisms, FlexServe designs a fast and secure LLM inference framework within TrustZone's secure world. The LLM-Aware Memory Management and Secure Inference Pipeline are introduced to accelerate inference. A Multi-Model Scheduler is proposed to optimize multi-model workflows. We implement a prototype of FlexServe and compare it with two TrustZone-based strawman designs. The results show that FlexServe achieves an average $10.05\times$ speedup in Time to First Token (TTFT) compared to the strawman, and an average $2.44\times$ TTFT speedup compared to an optimized strawman with pipeline and secure NPU enabled. For multi-model agent workflows, the end-to-end speedup is up to $24.30\times$ and $4.05\times$ compared to the strawman and optimized strawman, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01148v1">Emergence of Preferential Attachment and Glass-Ceiling Effects in Autonomous Networks of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      We investigate the emergence of structural disparities in networks of collaborating large language model (LLM) agents. When LLM agents autonomously choose collaborators, the resulting communication network exhibits preferential-attachment dynamics: agents that are already prominent become increasingly likely to attract additional connections. In some cases, weaker LLM agents (agents with smaller base model or older version) can disproportionately occupy central and influential network positions relative to stronger LLM agents. We interpret this as a type-dependent glass-ceiling effect (GCE). We model the network of LLM agents as a time-evolving sequence of directed weighted graphs, where the vector-valued edge weights represent cumulative tokens exchanged, number of interaction rounds, and reasoning effort. Using a contraction mapping argument on the mean-field dynamics, we prove that the importance (centrality) of each agent type converges to a unique stable equilibrium. To ground the model in LLM decision mechanisms, we introduce a cross-attention-inspired utility for collaborator selection. This utility specifies the local connection dynamics and, together with the mean-field model, yields a predictive characterization of the limiting network structure and its type-dependent centrality gaps. To validate the theory, we develop an experimental testbed with 100 LLM agents. Our experiments show that autonomous network formation can generate persistent centrality disparities, with their magnitude and direction depending on model family, model size, system-prompt design, and task context. They further show that the effect of preferential attachment depends on its alignment with model capability: reinforcing it improves collective performance when stronger agents become central, whereas weakening it improves performance when network dynamics instead favor weaker agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01138v1">Antaeus: Hunting Repository-Level Logic Vulnerabilities via Context-Grounded LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      LLM-based vulnerability detectors have shown promising results in identifying memory-safety bugs and vulnerability classes whose violations can often be expressed through established security properties. Logic vulnerabilities, however, pose a different challenge, as their identification requires inferring application-specific security invariants and implicit assumptions about intended behavior. Even frontier agentic models struggle because these invariants are often implicit and buried among unrelated code. Motivated by this gap, we present Antaeus, a framework for detecting logic vulnerabilities that grounds LLM reasoning in repository-level code context. Antaeus follows a repository-scale pipeline combining function prioritization, context-grounded reasoning, comparative validation, and structured reporting. It ranks functions using lightweight repo-wide security signals, directing costly LLM analysis toward relevant code and reducing calls, cost, and triage effort. For each prioritized function, Antaeus combines local code context with a repository-level view of the application's functionality, security resources, and trust boundaries. This enables reasoning about how the function is executed within the broader application rather than as an isolated snippet. Antaeus identifies security-sensitive sinks, derives safety conditions for safe execution, and checks whether they are locally satisfied. Candidate findings undergo comparative validation, pruning concerns that reflect project-wide norms rather than distinctive violations. Finally, Antaeus reports sinks, violated safety conditions, and evidence, making findings actionable and traceable. We evaluate Antaeus on 28 repositories with confirmed logic vulnerabilities and compare it against function-level and agentic models. Antaeus detects and explains 15 vulnerabilities, outperforming baselines with comparable token usage and cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01103v1">Clinician-Level Agreement Without Clinical Caution: LLM Evaluator Limits in Medical AI Benchmarking</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      Open-response evaluation provides stronger clinical validity than multiple-choice benchmarks but creates a scoring bottleneck that motivates automated LLM-asa-Judge approaches. Whether such evaluators replicate clinical calibration and caution, however, remains untested. We introduce MedQADE, the first standardised open-response clinical benchmark for German, a major clinical language lacking native evaluation infrastructure, comprising 3,800 items annotated by ten practising physicians and nine Large Language Model (LLM) evaluators. The top-performing evaluator model, Gemini 3 Flash, reached alignment consistent with the physician ceiling (\k{appa} = 0.694 vs. \k{appa} = 0.709), though wide confidence intervals limit interpretation. Despite this statistical alignment, automated evaluators exhibited near-absent clinical metacognition: physicians scaled abstention with item difficulty, while frontier models assigned definitive scores in every case. We additionally quantified systematic lineage-dependent biases, where models preferentially scored architectural siblings, an effect independent of language. These results show that statistical alignment does not ensure clinical caution, and that evaluator independence requires explicit verification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14660v2">NeuroFilter: Activation-Based Guardrails for Privacy-Conscious LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      Agentic Large Language Models (LLMs) are models able to reason, plan, and execute tools over unstructured data. These abilities are enabling transformative applications in domains spanning from personal assistant, financial, and legal domains. While these systems can substantially improve productivity and service quality, effective agency typically requires access to sensitive personal or organizational information. However, this access introduces critical inference-time privacy risks, specifically regarding contextually appropriate information disclosure. While recent studies highlight the inability of agentic LLMs to consistently adhere to privacy norms, existing defenses often rely on auxiliary LLM-based monitors. However, these defenses are expensive and offer limited protection against attacks that are robust to semantic censorship. To contrast this background, this paper proposes a notion of privacy filters based on activation probing. We show that these filters are both computationally efficient and effective for both single-turn and multi-turn conversational settings. Furthermore, this work provides the first systematic investigation into probing model internals across a conversation trajectory, moving beyond static, single-prompt analysis to capture the evolving state of privacy-sensitive interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01047v1">Conversable Complexity: Agentic LLM Collectives as Interpretable Substrates</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      Complexity and interpretability rarely coincide: systems rich enough for complex behaviours to emerge are usually too opaque to question, while transparent ones are too simple for anything complex to emerge. A single large language model (LLM) is a static artefact, hardly exhibiting any of the emergent properties we associate with life. This changes through interaction: populations of LLMs display emergent dynamics absent from isolated models. Furthermore, LLMs can be endowed with persistent memory, tools and shared skills, and the capacity to initiate actions unprompted, i.e., turning LLMs agentic. In this paper, we argue that such collectives of agents can serve as a computational substrate for Artificial Life (ALife) research. Critically, since the agents communicate in natural language, their collective behaviour can be directly interrogated by examining textual traces and asking the agents themselves. We outline the notion of interpretability in language-model research and extend it for collectives of agents. Lastly, we survey recent examples of agentic LLM collectives that already instantiate the idea of agentic substrates, from controlled experiments to deployments in the wild.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01040v1">As It Was: Aligning LLM Search Evaluation with Historical User Preferences</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      Large-scale search systems evolve faster than human quality assurance can scale, especially for long-tail intents and multilingual queries. LLM-as-a-judge approaches provide a scalable alternative for evaluating the relevance of search engine result pages (SERPs), but judgments based solely on semantic similarity or world knowledge can drift from actual user preferences, particularly for ambiguous queries. We introduce a behavior-grounded LLM judge that augments each SERP item with a lightweight and auditable behavioral prior in the form of a Query-Relevance-Impressions (QRI) card. Each card summarizes how users have historically interacted with similar queries and results, providing compact empirical evidence that the judge can cite to resolve ambiguity and make more consistent relevance judgments while still relying on semantic reasoning. In a large-scale music search evaluation at Spotify, using relevance estimates derived from historical user interactions across 6,000 recomposed SERPs, the behavior-grounded judge achieves stronger alignment with user preferences, improving Spearman rank correlation by approximately 5% overall and yielding a 91% relative improvement on disagreement cases. On a multilingual human-judged dataset spanning five languages, grounding further increases correlation with human relevance judgments by 15%. Importantly, when evaluated against outcomes from a live A/B test, the grounded judge shows consistently higher alignment with the observed winning model. While absolute alignment remains moderate, these findings demonstrate that lightweight behavioral grounding can improve the reliability and practical usefulness of LLM-based evaluation in real-world search systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.08625v2">From Holistic Evaluation to Structured Criteria: Rubrics Across the Evolving LLM Landscape</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) advance toward open-ended autonomous agents, the mechanisms used to evaluate and guide their behavior must evolve accordingly. This work introduces the rubric as a unifying framework capturing this evolution, characterizing rubrics as a dynamic response to successive LLM paradigm shifts that recurs across otherwise independent efforts in evaluation, reinforcement learning, and safety alignment. We define rubrics as explicit criteria sets that transform complex quality judgments into structured and actionable standards, and demonstrate that their recurrence across these research threads is not coincidental. We systematically organize existing rubric designs, examine their construction and optimization, and analyze their role across evaluation and training. Rubrics manifest at three progressively deeper levels: at the evaluative level, they decompose holistic judgments into verifiable dimensions; at the training level, they serve as dense feedback signals providing process-level guidance where scalar rewards fall short; at the intrinsic level, they emerge dynamically from model behaviors, driving self-improvement. We further assess rubric reliability across generation quality, execution fidelity, theoretical constraints, and security threats, before surveying rubric-based benchmarks across diverse domains. By rendering assessment transparent and decomposable, rubrics translate human value expectations into machine-learnable signals, serving as the enduring bridge between human intentions and machine behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01299v1">HYPIC: Accelerating Hybrid-Attention LLM Serving with Position-Independent Caching</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      In retrieval augmented generation (RAG) and agentic LLM serving, prompts are assembled from independent segments into long contexts, making the prefill stage dominate the per-request computation cost. To this cost, two directions have emerged in parallel: position-independent caching (PIC) admits KV reuse for non-contiguous segments shared across different requests, while hybrid-attention models reduce computation complexity by replacing most full-attention layers with linear attention. However, they cannot coexist: applying PIC to hybrid-attention models breaks down because per-token KV-cache reuse primitives do not transfer to the per-request recurrent state. In this work, we present Hypic, the first serving system for hybrid-attention LLMs with position-independent caching. For linear-attention layers, we identify the segment-cumulative transition operator as the missing algebraic primitive, and cache it alongside each segment's zero-start end-state, enabling near-exact and constant-time state composition of independently cached segments. For the remaining full-attention layers, existing PIC methods also fail as linear layers do not expose the per-token hidden states for selective recomputation. We show that the most significant attention deviation concentrates at segment boundaries, so recomputing only a small seam window at each boundary suffices to restore cross-segment lookback. Finally, Hypic exploits segment-level self-containment to parallelize cache-miss prefill across instances, turning long cold requests -- a major tail-latency contributor under both prefix caching and prior PIC -- into an accelerable workload. Evaluated across four hybrid-attention models and five workloads, Hypic reduces time-to-first-token (TTFT) by 2.45x on average and improves peak throughput by up to 2.0x over existing systems, while staying within 3.3 points of full-recompute accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.00939v1">Leveraging LLM-Based Agentic Systems to Generate Quantum Applications for Test Optimization</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      Quantum computing is increasingly explored for software engineering (SE) optimization, but translating natural-language (NL) task-level requirements into executable quantum applications still demands substantial quantum and programming expertise. We present QPipe, a large language model (LLM)-based multi-agent architecture that autonomously turns NL requirements into traceable quantum-application workflows through specialized agents for requirement parsing, formulation, code generation, review, execution, and verification. We evaluate QPipe on 20 NL requirements, each associated with a real-world benchmark and a test-optimization problem. QPipe successfully completes the key stages of quantum-application generation across requirements, achieving average rates of 100% for code compilation and 96.7% for application execution and final-result combination, with average generation costs of 260.1 seconds and 1.89M tokens per requirement. Among the generated quantum applications that execute successfully, the returned solutions outperform the offline genetic algorithm baseline in most cases. Ablation results further show that QPipe's advantage depends on retaining code-generation skills, task knowledge, review feedback, and multi-agent decomposition. These results indicate that agentic coordination can support generation of executable quantum applications for tackling test optimization problems from real-world benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.00937v1">Persona Non Grata: LLM Persona-Driven Generations in MCQA are Unstable in Distinct Dimensions</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 23 pages, 12 figures. Under review at ARR
    </div>
    <details class="paper-abstract">
      Persona-driven generations (PDGs) have seen prolific use in research and industry applications, where a large language model (LLM) takes on a 'persona' while completing some task. While persona expressed through free-form text (like dialogue) has substantial work investigating stability or consistency, relatively, persona expressed in non-text-heavy outputs (like in multiple-choice question answering, or MCQA) is often overlooked. We work to address this gap, seeking to understand the instability of LLM PDGs in MCQA tasks. We develop three metrics investigating the performance, outcome, and question correctness stability, evaluating three distinct dimensions. Using these metrics, we find that instability varies consistently between model families and model size, and across question domains, with math/commonsense questions leading to greater instability. We also find task prompt format introduces more prediction instability than other hyperparameters, like temperature. Finally, we find that instability is related to task accuracy, and using our instability metrics, find different experimental settings that result in different best and worst personas for tasks, despite their similarity. This reveals the importance of checking hyperparameter instability in PDGs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.13445v3">Verbosity Tradeoffs and the Impact of Scale on the Faithfulness of LLM Self-Explanations</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 ICLR 2026 Workshop on Principled Design for Trustworthy AI - Interpretability, Robustness, and Safety across Modalities 67 pages, 13 figures
    </div>
    <details class="paper-abstract">
      When asked to explain their decisions, LLMs can often give explanations which sound plausible to humans. But are these explanations faithful, i.e. do they convey the factors actually responsible for the decision? In this work, we analyse counterfactual faithfulness across 75 models from 13 families. We analyze the tradeoff between conciseness and comprehensiveness, how correlational faithfulness metrics assess this tradeoff, and the extent to which metrics can be gamed. This analysis motivates two new metrics: the phi-CCT, a simplified variant of the Correlational Counterfactual Test (CCT) which avoids the need for token probabilities while explaining most of the variance of the original test; and F-AUROC, which eliminates sensitivity to imbalanced intervention distributions and captures a model's ability to produce explanations with different levels of detail. Our findings reveal a clear scaling trend: larger and more capable models are consistently more faithful on all metrics we consider. Our code is available at https://github.com/google-deepmind/corr_faith.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.28229v2">Humanizing Automatically Generated Unit Test Suites with LLM-Based Refactoring</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      Search-based test generation tools such as EvoSuite produce compilable and high-coverage unit tests at scale, but their suites are often hard to read and maintain. LLMs can generate more natural tests, yet direct generation remains brittle, with compilation rates of only 51-78% in our study. We introduce TestHumanizer, a hybrid SBST+LLM approach that uses LLMs as controlled refactoring layers over compilable SBST suites to improve naming, structure, and developer-oriented clarity while preserving behavior and compilation validity. We evaluate TestHumanizer on 350 classes from Defects4J and SF110. EvoSuite generates 15 suites per class, and each suite is refactored under three context configurations using gpt-4o and mistral-large-2407, yielding 31,500 refactorings. TestHumanizer reaches 88-98% compilation rates, close to EvoSuite's 100% baseline and clearly above direct LLM generation. Structural coverage is largely preserved, typically within 1-2 percentage points, and 86-95% of refactorings satisfy a composite faithful-refactoring threshold. Refactored suites also improve predicted readability, reduce control-flow and cognitive complexity, and mitigate structural smells. The summary-based setting offers the most robust trade-off, while long code-centric prompts are more prone to hallucination-induced failures. A developer study on 30 classes and 444 test methods confirms significant gains in perceived readability and willingness to adopt, with Wilcoxon p less than 0.01 and substantial inter-rater agreement. Overall, LLMs are most effective not as standalone generators but as validation-gated refinement layers over robust SBST outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.00910v1">Calibrating the Instrument: Controllability of an LLM-Driven Synthetic Population</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      Generative Synthetic Populations (GSP) -- the convergence of population synthesis, agent-based modelling, and LLM agents -- are attracting growing interest for urban simulation and institutional communication research. Before any GSP instrument is used on a real population, a more basic question must be answered: does it respond to stimuli of known valence in an ordered, replicable, group-structured way? We call this controllability. We ask not whether a synthetic population tracks humans, but whether it tracks itself: whether the latent structure we impose on it is recovered in its own responses. This internal-validity question is logically prior to any claim about external validity, just as characterising an instrument's response function must precede using it to test a theory. We report SIVE (Synthetic Instrument Validation Experiment): a fictional municipality (Montelago) with 120 synthetic personas of known latent structure, exposed to seven conditions spanning strongly positive to strongly negative institutional communications about a water network. Seven pre-registered criteria, evaluated across a temperature sweep, jointly assess fidelity, stability, noise floor, specificity, sensitivity, and ordering. All seven pass at every temperature. A central finding turns a calibration failure into a diagnostic success: a message designed as "weakly positive" was identified by the instrument as functionally negative, traced to unresolved problems, uncertainty, and institutional passivity in its text; a redesigned version restored the expected ordering and interacts with agents' latent trust in unanticipated ways. A noise sub-experiment shows the instrument's intrinsic noise is roughly half the cross-agent estimate and stable across temperatures. Individual trajectories reveal coherent micro-dynamics that summary statistics obscure. Full data are available via an interactive explorer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01293v1">RuleChef: Grounding LLM Task Knowledge in Human-Editable Rules</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      We present RuleChef, a framework that uses large language models (LLMs) to generate executable rules for NLP tasks such as text classification, Named Entity Recognition (NER), or relation extraction. Rules are generated based on a task description and a set of labeled examples, then they are iteratively improved based both on additional examples and on human feedback overexisting rules. RuleChef can also be used to bootstrap rules using the observed input-output pairs from any existing model for a given task. LLMs are used only at learning time, synthesizing rules and iteratively patching them based on failures measured on a held-out split. The result of this process is a fast, deterministic, and inspectable rule system. Preliminary evaluation is performed on both classification and NER tasks. We release RuleChef as open-source software under an Apache 2.0
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.16511v2">Tail-Shape Estimation in LLM Evaluation Is Fragile: A Protocol for Diagnosing False Positives</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 9 pages of main paper, 4 figures and 4 tables in the main paper, more in the appendix
    </div>
    <details class="paper-abstract">
      Recent work motivates moving large language model (LLM) evaluation from mean-based to tail-aware metrics, including conditional value-at-risk and tail-index estimates of reward-model error. We ask whether the canonical extreme-value-theory tail-index parameter, which isolates how heavy a tail is from how large the tail mass is, adds discriminative information beyond the mean and a standard tail-magnitude statistic in LLM evaluation. We pre-register a protocol covering admissibility, goodness-of-fit, threshold-stability, and effect-size requirements for any positive tail-shape claim. The protocol is the contribution of this paper; the empirical study below is a demonstration of what its gates catch. Applied to a standard LLM toxicity-evaluation setup under two structurally different scorer families, the protocol catches three distinct modes of false positives that a naive analysis would have published, and rejects the headline tail-shape claim on both scorers. We conclude that tail-shape estimation in the LLM toxicity-evaluation setups we examined is more fragile than the recent literature suggests, and recommend the protocol as a starting point for tail-index claims in similar setups.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.00848v1">MetaHOPE: A Metaphor-Oriented Evaluation Framework for Analysing MT and LLM Translation Errors</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      In this opinion paper, we propose MetaHOPE, an error severity-aware annotation framework for evaluating metaphor translations. Metaphors present challenges for machine translation (MT) and natural language understanding and processing (NLU, NLP), because it presents the features of semantic complexity, contextual dependency, and cultural embeddings that can lead to ambiguity issues for NLP models. To investigate how state-of-the-art NLP models perform on translating metaphors, we select three representative systems, i.e., GoogleMT, GPT5.4, and Hunyuan-7b as Neural MT (NMT) models and LLMs. We used two human-annotated metaphor corpora, including VUAMC and PSUCMC for English-to-Chinese and Chinese-to-English translation purposes. The original corpora we used are monolingual, where we carried out error annotation using the MetaHOPE framework, and also produced the human post-edited gold reference for bilingual use as a new resource. We believe the MetaHOPE evaluation framework for metaphor translation annotation, the parallel corpora resources, and the error analysis on SOTA automatic translation models can be useful and shed some light for the field of metaphor translation study. We share our resources publicly upon paper acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2407.10887v4">Hey, That's My Model! Introducing Chain & Hash, An LLM Fingerprinting Technique</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 Published at ICLR 2026
    </div>
    <details class="paper-abstract">
      Growing concerns over the theft and misuse of Large Language Models (LLMs) underscore the need for effective fingerprinting to link a model to its original version and detect misuse. We define five essential properties for a successful fingerprint: Transparency, Efficiency, Persistence, Robustness, and Unforgeability. We present a novel fingerprinting framework that provides verifiable proof of ownership while preserving fingerprint integrity. Our approach makes two main contributions. First, a chain and hash technique that cryptographically binds fingerprint prompts to their responses, preventing collisions and enabling irrefutable ownership claims. Second, we address a realistic threat model in which instruction-tuned models' output distribution can be significantly altered through meta-prompts. By incorporating random padding and varied meta-prompt configurations during training, our method maintains robustness even under significant output style changes. Experiments show that our framework securely proves ownership, resists both benign transformations (e.g., fine-tuning) and adversarial fingerprint removal, and extends to fingerprinting LoRA adapters\footnote{We release our code at: https://github.com/microsoft/Chain-Hash.
    </details>
</div>
