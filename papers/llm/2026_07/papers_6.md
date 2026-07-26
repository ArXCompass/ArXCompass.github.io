# llm - 2026_07

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
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06827v1">Compress the Cache, Not the Speech Embedding: KV Compression for Efficient Speech LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 Submitted to SLT2026
    </div>
    <details class="paper-abstract">
      Speech large language models (Speech LLMs) typically encode speech into sequences far longer than text, creating a major efficiency bottleneck during autoregressive decoding. A common remedy is to compress the speech sequence at the adapter level to remove temporal redundancy before it enters the LLM; however, such early downsampling risks discarding fine-grained information that cannot be recovered. We propose SpeechKV, which applies a learned pooling to the KV cache of speech tokens inside the LLM. This design allows the LLM to fuse speech and text internally while directly accelerating decoding. Trained on 71K hours of speech data, SpeechKV compresses the speech to approximately text-level granularity yet maintains performance on par with or even slightly better than the uncompressed baseline, with relative gains of 6.6% on out-of-domain entity recognition and 2.3% on OpenASR, while delivering at least 1.49 times decoding speedup that scales with audio length.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06820v1">Evaluating SageMath-Augmented LLM Agents for Computational and Experimental Mathematics</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 37 pages, 16 figures, accepted to 3rd AI for Math Workshop at ICML 2026
    </div>
    <details class="paper-abstract">
      Recent advances in AI for Mathematics have focused largely on autoformalization and theorem proving, leaving the role of Computer Algebra Systems (CAS) in agentic LLM workflows underexplored. We propose a ReAct-style agentic setup that combines LLM reasoning with verifiable feedback from SageMath, together with Context7 for the up-to-date documentation. We evaluate this agentic setup across frontier models for solving research-level mathematical problems from the RealMath benchmark in a setting that emulates a computational-mathematics research loop. We also propose a refinement to the RealMath benchmark by introducing a multi-step post-processing procedure and a multi-stage validation pipeline, both of which improve the quality and reliability of the extracted problem set. Our experiments reveal substantial performance gains from SageMath access across all evaluated models on +9.7~pp on average, the gains range from 1.5~pp to 27.8~pp and narrow the gap between open-weight and closed models. Qwen~3.7-Max benefits from SageMath the most, while GPT-5.5 achieves the highest solve rate of $75.2\%$ and the lowest token usage among tool-enabled configurations. Our findings suggest that CAS-augmented agents represent a promising direction for assisting mathematicians in computational exploration, and we believe that this work is a step towards automated conjecture discovery. The project repository is available online.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.23800v2">Object Search in Partially-Known Environments via LLM-informed Model-based Planning and Prompt Selection</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 10 pages, 8 figures. Accepted to IROS 2026
    </div>
    <details class="paper-abstract">
      We present a novel LLM-informed model-based planning framework, and a novel prompt selection method, for object search in partially-known environments. Our approach uses an LLM to estimate statistics about the likelihood of finding the target object when searching various locations throughout the scene that, combined with travel costs extracted from the environment map, are used to instantiate a model, thus using the LLM to inform planning and achieve effective search performance. Moreover, the abstraction upon which our approach relies is amenable to deployment-time model selection via the recent offline replay approach, an insight we leverage to enable fast prompt and LLM selection during deployment. Simulation experiments demonstrate that our LLM-informed model-based planning approach outperforms the baseline planning strategy that fully relies on LLM and optimistic strategy with as much as 11.8% and 39.2% improvements respectively, and our bandit-like selection approach enables quick selection of best prompts and LLMs resulting in 6.5% lower average cost and 33.8% lower average cumulative regret over baseline UCB bandit selection. Real-robot experiments in an apartment demonstrate similar improvements and so further validate our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06802v1">A Multi-Analyst LLM Pipeline for Auditable Rule Discovery Across 68 Public Physiological Corpora</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 8 pages, 2 figures, 9 tables; submitted to IEEE SPMB 2026
    </div>
    <details class="paper-abstract">
      Open physiological corpora are heterogeneous: they use different sensors, labels, sampling rates, recording settings, and clinical endpoints. They can support detector design, but they do not directly specify which detector rules should be built for a new contactless monitoring platform. We report a controlled four-analyst large-language-model (LLM) workflow for converting 68 public physiological corpora, screened for commercial-use compatibility, into an auditable library of candidate rule shapes for prospective validation. Four independent commercial LLM families read the corpus documentation under a controlled prompt and produced 695 candidate rule markers (top-markers). Deduplication retained 649 rule records; a threshold-bounds audit then flagged 51 sanity violations for clamping or curator review. Cross-corpus consolidation produced 436 unique rule shapes. Gate-tagging against two hard invariants, native target-hardware channel availability and no multi-night per-patient personalization, identified 94 build-now detector components across four detector-family buckets. The pipeline does not produce a validated clinical detector. It produces an auditable engineering cascade in which analyst disagreement, threshold checks, curator review, and automated continuous-integration (CI) checks route literature-derived rules toward prospective hardware validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.29033v2">Human-in-the-Loop Nugget Annotation for Accountable LLM-as-a-Judge Evaluations</a></div>
    <div class="paper-meta">
      📅 2026-07-07
    </div>
    <details class="paper-abstract">
      Evaluating AI/Agentic system outputs reliably requires human judgment, but how one incorporates the human determines whether one gets a real quality signal or expensive theater. The common approaches either accidentally anchor human experts (leading to rubber-stamping) or leave them unsupported in cognitively demanding labeling tasks. We present a prototype of an annotation tool that implements a different division of labor: humans identify what information matters (nuggets), while LLMs handle high-volume matching of nuggets to system outputs. This plays to each party's strengths while maintaining genuine human oversight. We describe the Human-AI workflow, key design decisions, and how resulting nugget banks are used with automated judges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06757v1">LLM-powered reasoning in agent-based modeling</a></div>
    <div class="paper-meta">
      📅 2026-07-07
    </div>
    <details class="paper-abstract">
      Agent-based modeling (ABM) has the capability to model millions of individuals and their interactions, which is useful for policy making. However, ABMs have traditionally relied on static prior, which prevents the models from adapting to real-time changes. Our research provides a novel approach to addressing this information gap. Large language models (LLMs) offer new opportunities to predict human decision-making. Here, we introduce a scalable Hybrid Agent-based and Language-driven Epidemic (HALE) modeling framework that leverages LLMs to predict human decision-making in an ABM simulation. As a proof-of-concept, we use HALE to simulate COVID-19 and its effects in Salt Lake County, UT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.19771v4">Beyond Reactivity: Measuring Proactive Problem Solving in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-07
    </div>
    <details class="paper-abstract">
      LLM-based agents are increasingly moving towards proactivity: rather than awaiting instruction, they exercise agency to anticipate user needs and solve them autonomously. However, evaluating proactivity is challenging; current benchmarks are constrained to localized context, limiting their ability to test reasoning across sources and longer time horizons. To address this gap, we present PROBE (Proactive Resolution Of BottlEnecks). PROBE decomposes proactivity as a pipeline of three core capabilities: (1) searching for unspecified issues, (2) identifying specific bottlenecks, and (3) executing appropriate resolutions. We apply PROBE to evaluate leading LLMs and popular agentic frameworks, showing that even state-of-the-art models struggle to solve this benchmark. Computing our consistent measurements across frontier LLMs and agents, we find that the best end-to-end performance of 40% is achieved by both GPT-5 and Claude Opus-4.1. Additionally, we demonstrate the relative capabilities of each model and analyze mutual failure modes. Our results highlight the current limitations of autonomous action in agentic systems, and expose promising future research directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06527v1">RSF-GLLM: Bridging the Semantic Gap in Multi-Hop Knowledge Graph QA via Recurrent Soft-Flow and Decoupled LLM Generation</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 Accepted for publication in ICML 2026 as a full research paper; 21 pages
    </div>
    <details class="paper-abstract">
      Multi-hop Question Answering over Knowledge Graphs faces a critical challenge: traditional retrieve-then-read pipelines break differentiability, preventing the retriever from learning to bridge the semantic gap where intermediate nodes lack lexical overlap with the query. To address this, we propose RSF-GLLM, a framework decoupling differentiable graph reasoning from answer generation. Our Recurrent Soft-Flow (RSF) module employs a GRU-guided query updater to propagate continuous relevance scores, utilizing a dynamic gating mechanism to traverse semantically dissimilar bridge nodes via structural cues. We introduce flow sparsity regularization to theoretically guarantee convergence from soft probabilities to discrete reasoning paths. These paths are extracted and textualized to fine-tune a Large Language Model (LLM), ensuring generation is grounded in factual topology. Experiments on WebQSP and CWQ demonstrate that RSF-GLLM achieves competitive performance with superior inference efficiency compared to LLM based computationally expensive approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05391v2">LLM-as-a-Verifier: A General-Purpose Verification Framework</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 Code: https://github.com/llm-as-a-verifier/llm-as-a-verifier Website: https://llm-as-a-verifier.com
    </div>
    <details class="paper-abstract">
      Scaling pre-training, post-training, and test-time compute have become the central paradigms for improving the capabilities of LLMs. In this work, we identify verification, the ability to determine the correctness of a solution, as a new scaling axis. To unlock this and demonstrate its effectiveness, we introduce LLM-as-a-Verifier, a general-purpose verification framework that provides fine-grained feedback for agentic tasks without requiring additional training. Unlike standard LM judges that prompt LLMs to produce discrete scores for candidate solutions, LLM-as-a-Verifier computes the expectation over the distribution of scoring token logits to generate continuous scores. This probabilistic formulation enables verification to scale along multiple dimensions: (1) score granularity, (2) repeated evaluation, and (3) criteria decomposition. In particular, we show that scaling the scoring granularity leads to better separation between positive and negative solutions, resulting in more calibrated comparisons. Moreover, scaling repeated evaluation and criteria decomposition consistently lead to additional gains in verification accuracy through variance and complexity reduction. We further introduce a cost-efficient ranking algorithm for selecting the best solution among candidates using the verifier's continuous scores. LLM-as-a-Verifier achieves state-of-the-art performance on Terminal-Bench V2 (86.5%), SWE-Bench Verified (78.2%), RoboRewardBench (87.4%), and MedAgentBench (73.3%). Beyond verification, the fine-grained signals from LLM-as-a-Verifier can also serve as a proxy for estimating task progress. We build an extension for Claude Code, enabling developers to monitor and improve their own agentic systems. Finally, we show that LLM-as-a-Verifier can provide dense feedback for RL, improving the sample efficiency of SAC and GRPO on robotics and mathematical reasoning benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06519v1">FreqDepthKV: Frequency-Guided Depth Sharing for Robust KV Cache Compression in Long-Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 11 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Long-context LLM inference is increasingly limited by the memory and bandwidth cost of KV caches, yet aggressive compression can remove the layer-specific evidence needed for retrieval and multi-step reasoning. We introduce FreqDepthKV, an inference-time cache compression method that factorizes adjacent-layer KV states into shared low-frequency depth components and sparse high-frequency residuals. A lightweight online probe assigns attention heads to shared-depth, residual-depth, or exact cache modes according to their contribution to reconstruction-sensitive attention logits, allowing the compression policy to adapt to prompt structure without retraining. Across long-context question answering, needle retrieval, summarization, and code generation benchmarks, FreqDepthKV preserves task accuracy under substantially smaller cache budgets. With a 32k-token prefill window, FreqDepthKV reaches 58.3 Exact Match, 63.0 F1, 32.5 ROUGE-L, and 48.1 pass@1, closely matching full KV while outperforming prior compressed-cache methods. It also improves decoding throughput to 70.4 tokens/s, reduces TTFT to 2.06 seconds, and lowers peak KV memory to 6.2 GB, achieving a 3.9x effective compression ratio.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06503v1">Doomed from the Start: Early Abort of LLM Agent Episodes via a Recall-Controlled Probe Cascade</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 10 pages, 9 figures, 2 tables. Code will be released soon
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents solving multi-step tasks frequently commit to trajectories that are doomed to fail, yet continue to consume substantial inference compute before the failure becomes observable. We show that failure is predictable early from the agent's internal representations: lightweight per-round probes on hidden activations anticipate eventual episode failure as early as the first interaction round, where scorers reading only the agent's observable behavior are barely better than chance. We turn this signal into a practical abort cascade: one distribution-free calibrated gate per round, with per-round recall budgets jointly searched so that eventually-successful episodes survive all gates at a user-specified global rate; this episode-level guarantee is the one that matters in deployment, since false-abort risk accumulates across gates. Across two agent models on TextCraft, the cascade meets every recall target from 90% to 97% and, at the 90% target, saves 47.1% +/- 10.3% (Qwen-2.5-7B) and 37.2% +/- 8.8% (Llama-3.2-3B) of inference compute, 1.6--1.7x the best single-gate policy. An otherwise-identical cascade reading only behavior saves roughly half as much, and adding behavioral features to the probe yields no further gain: the hidden states capture what behavior reveals. Finally, we characterize the sample complexity of certifying high recall targets, telling practitioners which recall promises their data can, and provably cannot, back. The code will be released soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06461v1">WordVoice: Explicit and Decoupled Multi-Dimensional Word-Level Control for LLM-Based TTS</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 10 pages, 4 figures, 6 tables; Preprint
    </div>
    <details class="paper-abstract">
      While recent Large Language Model (LLM)-based Text-to-Speech (TTS) systems have achieved remarkable naturalness, they predominantly rely on implicit end-to-end generation paradigms, resulting in coarse-grained control. In scenarios demanding precise stylistic interventions and strict temporal alignment, such as audiobook narration and video dubbing, the inability to explicitly manipulate word-level acoustic attributes remains a critical bottleneck. This limitation is primarily amplified by the severe scarcity of fine-grained annotated datasets and the architectural challenge of integrating multi-dimensional control signals into discrete autoregressive generation. To address this, we propose a unified framework for highly precise word-level control. First, we construct WordVoice-5A, a massive 4.7k-hour bilingual dataset featuring five-dimensional word-level annotations (duration, boundary, energy, pitch and tone) developed through a rigorous linguistically-guided pipeline. Second, we introduce WordVoice to transform the implicit generation process into an explicit, highly controllable paradigm. Specifically, we introduce a bound-token mechanism within the LLM to formulate an explicit ``acoustic planning'' process, enabling adaptive multi-task prosodic planning and flexible manual intervention. Furthermore, we augment the token-to-waveform stage with a fine-grained acoustic modulation module, bridging the resolution gap to strictly align word-level attributes between highly compressed discrete tokens and continuous waveforms. Extensive experiments demonstrate that WordVoice achieves superior, decoupled control over multiple acoustic dimensions while maintaining competitive zero-shot synthesis stability. The code and audio samples are publicly available at https://xxh333.github.io/wordvoice-demo/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06452v1">From Voting to Agent Collaboration: Answer-Type-Aware LLM Pipelines for BioASQ 14b</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      Biomedical question answering requires not only accurate extraction of information from scientific literature but also reliable integration of evidence across multiple documents. This study presents a question-type-specific large language model (LLM) framework for BioASQ 14b Task B, designed to improve answer robustness and evidence grounding in biomedical question answering. Rather than applying a single prompting strategy to all questions, the framework selects different inference procedures for yes/no, factoid, and list questions according to their distinct reasoning and evaluation requirements. For yes/no questions, snippet shuffling and self-reflection are used to reduce sensitivity to evidence ordering and improve decision stability. For factoid questions, full-snippet input is combined with chain-of-thought-based in-context learning to support accurate biomedical entity identification. For list questions, a multi-agent architecture is employed, in which evidence extraction, candidate generation, answer verification, and final aggregation are handled collaboratively. Preliminary experiments on BioASQ 13b were used to identify effective inference strategies for each question type, and the resulting framework was subsequently evaluated in the official BioASQ 14b Task B challenge. In the official evaluation, our framework showed competitive performance across multiple batches and achieved first place in the factoid subtask of Batch 4. These results demonstrate the effectiveness of combining question-type-specific inference, ensemble prediction, and agent-based verification for reliable biomedical question answering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09159v4">LLMs Meet Isolation Kernel: Lightweight, Learning-free Binary Embeddings for Fast Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 Accepted to ACL 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently enabled remarkable progress in text representation. However, their embeddings are typically high-dimensional, leading to substantial storage and retrieval overhead. Although recent approaches such as Matryoshka Representation Learning (MRL) and Contrastive Sparse Representation (CSR) alleviate these issues to some extent, they still suffer from retrieval accuracy degradation. This paper proposes Isolation Kernel Embedding or IKE, a learning-free method that transforms an LLM embedding into a binary embedding using Isolation Kernel (IK). Lightweight and based on binary encoding, IKE offers a low memory footprint and fast bitwise computation, lowering retrieval latency. Experiments on multiple text retrieval datasets demonstrate that IKE offers up to 16.7x faster retrieval and 16x lower memory usage than the original LLM embeddings, while maintaining comparable accuracy. Theoretically, we show that IKE works because it satisfies four essential criteria for effective binary hashing that other methods do not possess. Compared to CSR, IKE consistently achieves better retrieval efficiency and effectiveness. IKE also works effectively with graph-based indexing, demonstrating its superiority in balancing accuracy and latency compared to alternative compression techniques in the approximate nearest neighbor (ANN) search setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.18419v6">Knowing When to Quit: A Principled Framework for Dynamic Abstention in LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-07-07
    </div>
    <details class="paper-abstract">
      LLMs utilizing chain-of-thought reasoning often waste substantial compute by producing long, incorrect responses. Abstention can mitigate this by withholding outputs unlikely to be correct. While most abstention methods decide to withhold outputs before or after generation, dynamic mid-generation abstention considers early termination of unpromising reasoning traces at each token position. Prior work has explored empirical variants of this idea, but principled guidance for the abstention rule remains lacking. We present a formal analysis of dynamic abstention for LLMs, modeling abstention as an explicit action within a regularized reinforcement learning framework. An abstention reward parameter controls the trade-off between compute and information. We show that abstaining when the value function falls below this reward strictly outperforms natural baselines under general conditions. We further derive a principled and efficient method to approximate the value function. Empirical results on mathematical reasoning and toxicity avoidance tasks support our theory and demonstrate improved selective accuracy over existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02277v2">Quantifying Frontier LLM Capabilities for Container Sandbox Escape</a></div>
    <div class="paper-meta">
      📅 2026-07-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly act as autonomous agents, using tools to execute code, read and write files, and access networks, creating novel security risks. To mitigate these risks, agents are commonly deployed and evaluated in isolated "sandbox" environments, often implemented using Docker/OCI containers. We introduce SANDBOXESCAPEBENCH, an open benchmark that safely measures an LLM's capacity to break out of these sandboxes. The benchmark is implemented as an Inspect AI Capture the Flag (CTF) evaluation utilising a nested sandbox architecture with the outer layer containing the flag and no known vulnerabilities. Following a threat model of a motivated adversarial agent with shell access inside a container, SANDBOXESCAPEBENCH covers a spectrum of sandboxescape mechanisms spanning misconfiguration, privilege allocation mistakes, kernel flaws, and runtime/orchestration weaknesses. We find that, when vulnerabilities are added, LLMs are able to identify and exploit them, showing that use of evaluation like SANDBOXESCAPEBENCH is needed to ensure sandboxing continues to provide the encapsulation needed for highly-capable models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06641v1">Healthier LLMs: Retrieval-Augmented Generation for Public Health Question Answering</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 19 Pages, 14 Main Text Pages, 6 Figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) achieve promising results on medical question answering benchmarks, yet their use in public health is constrained by hallucinations and the rapid evolution of official guidance. Retrieval-Augmented Generation (RAG) mitigates these risks by grounding responses in an explicitly maintained corpus, but end-to-end performance depends critically on retrieval configuration and on evaluation beyond multiple-choice formats. We extend PubHealthBench, a question answering (QA) benchmark of 7,929 questions derived from UK Government public health guidance, into a retrieval-augmented setting and systematically evaluate retrieval and generation choices. We compare dense, sparse, and hybrid retrieval across multiple embedding models and corpus variants, and show that hybrid retrieval consistently improves recall and ranking quality, with chunk length and topic interacting with ranking performance. Providing retrieved context substantially increases multiple-choice accuracy across a diverse set of LLMs, enabling smaller open-weight models to match or outperform larger models used without retrieval, with gains primarily driven by retrieval quality and careful context selection. To assess realistic free-form answering, we introduce a rubric-based LLM-as-a-judge covering faithfulness, completeness, clarity, and factual consistency, and validate it against dual human annotations. Judge-human agreement is strongest for faithfulness and completeness, while factual consistency and clarity are less reliably reproduced, motivating caution when interpreting those dimensions at scale. Overall, our results highlight retrieval as a primary lever for reliable public health QA and provide practical guidance for building and evaluating RAG systems grounded in official guidance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06327v1">Estimating Uncertainty from Reasoning: A Large-Scale Study of Multi- and Crosslingual MCQA Performance in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-07
    </div>
    <details class="paper-abstract">
      Uncertainty estimation (UE) enables LLM-powered systems to recognize when to abstain, yet existing research has predominantly focused on English. We present the first large-scale evaluation of UE methods across 22 languages, spanning high-, mid-, and low-resource settings. Using two human-curated Q\&A datasets, we compare open and closed box UE methods (nine in total) across different model sizes and architectures while eliciting long-form reasoning, avoiding LLM-as-a-judge and embedding-based scoring, which can introduce evaluation noise. We report three main actionable findings. First, we find that prompting models to reason in English while keeping questions in low-resource languages substantially improves UE performance, suggesting that comprehension of low-resource languages is largely intact, and that the reliability bottleneck lies in generation rather than understanding. Second, prompting models to reason in English closes the UE performance gap between low and high-resource languages, demonstrating that generation language matters more than the question language. Third, the choice of UE method should depend on model scale: at smaller scales, open-box probability-based methods outperform alternatives; at larger scales, closed-box self-verbalized uncertainty becomes superior. Finally, we provide an analysis of threshold selection for selective prediction, offering guidance on calibrating abstention in multilingual settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.10177v2">Detoxify: A framework for abusive text transformation using LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-07
    </div>
    <details class="paper-abstract">
      Although Large Language Models (LLMs) have demonstrated significant advancements in natural language processing tasks, their effectiveness in the classification and transformation of abusive text into non-abusive versions remains an area for exploration. In this study, we present Detoxify: a framework that employs LLMs to transform abusive text (tweets and reviews) containing hate speech and profanity into non-abusive text while retaining the original intent. We evaluate the performance of four state-of-the-art LLMs, such as Gemini, GPT-4o, DeekSeek and Groq, on their ability to identify abusive text. We aim to transform and obtain a text that is clean of abusive and inappropriate content, but maintains a similar level of sentiment and semantics, i.e. the transformed text needs to maintain its message. Afterwards, we evaluate the raw and transformed datasets with sentiment analysis and semantic analysis. Our results show Groq provides vastly different results when compared with other LLMs. We have identified similarities between GPT-4o and DeepSeek. Groq stood out as the most distinct, as it often restructured sentences with excessive positive phrasing, with the original context lost or altered.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06273v1">AgentTether: Graph-Guided Diagnosis and Runtime Intervention for Reliable LLM Agent Operation</a></div>
    <div class="paper-meta">
      📅 2026-07-07
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents are increasingly used for multi-step, stateful tool-use tasks, yet production reliability remains limited. Unlike static software repair, agent repair must recover dynamic trajectories whose early decisions can propagate into later errors and external state changes. Existing automatic remedies address only part of this problem: blind retry adds no diagnosis, outcome feedback says whether a run failed but not where or why, and self-reflection often lacks grounded evidence to prevent the same failure from recurring. We present AgentTether, a run-time repair framework that automates post-run diagnosis and guided recovery without modifying the underlying agent or environment. AgentTether abstracts each run into Transition Units, links them through a dependency-aware Critical Transition Graph, and localizes failure-critical subtrajectories by combining an offline normal-behavior model with a run-local graph detector. It then converts the localized cause into behavior-scoped guidance backed by cross-iteration Repair Memory, and can optionally apply guarded run-time intervention to keep the correction active during re-execution. The same design can be deployed as an offline diagnostic-and-guidance tool or as an online repair layer. We evaluate AgentTether on 261 tau-bench tasks across three domains with Qwen3.7-max, and test cross-model transfer on Banking with GPT-5.4. On the hardest Banking domain, AgentTether repairs 59.04% (49/83) of initially failed Qwen3.7-max tasks and 65.12% (56/86) of initially failed GPT-5.4 tasks. Overall, AgentTether improves repair effectiveness while reducing agent turns and end-to-end approach tokens, suggesting a practical reliability layer that can wrap existing agent deployments, reduce wasted re-execution, and improve recovery without retraining the agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06223v1">Information Gain-based Rollout Policy Optimization: An Adaptive Tree-Structured Rollout Approach for Multi-Turn LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-07
    </div>
    <details class="paper-abstract">
      Reinforcement learning has become a promising paradigm for improving large language model (LLM) agents on long-horizon search tasks, where the agent must make a sequence of intermediate decisions before receiving a final outcome. However, existing methods still face a key limitation: the rollout budget is often allocated without explicitly assessing the utility of intermediate states. As a result, substantial computation may be spent on low-value states, even though different branches can vary drastically in their informativeness. In this paper, we propose Information Gain-based Rollout Policy Optimization (IGRPO), a policy optimization framework that treats intermediate-state informativeness as the organizing principle of rollout collection. Specifically, IGRPO performs budget-aware tree-structured rollouts by allocating expansion budget according to node-level informativeness, so that more informative branches are expanded more frequently while unpromising branches are progressively suppressed. We further demonstrate that the information gain-based rollout induces an explicit limiting teacher distribution over trajectories, which naturally yields a clear policy optimization target, thereby unifying adaptive tree-structured exploration with principled policy learning under a single framework. Experiments on seven challenging search-augmented QA benchmarks demonstrate that IGRPO consistently outperforms strong baselines under the same rollout budget constraints, validating the effectiveness of leveraging the induced teacher distribution to guide policy optimization for long-horizon search agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.20023v2">When Lower Privileges Suffice: Investigating Over-Privileged Tool Selection in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 code: https://github.com/AISafetyHub/agent-tool-selection-bias
    </div>
    <details class="paper-abstract">
      As LLM agents increasingly select tools autonomously, their choices among tools with different privileges become safety-relevant. However, prior tool-selection studies focus on safety-agnostic metadata preferences, leaving privilege-sensitive choices underexplored. To address this gap, we study over-privileged tool selection, in which an agent selects or escalates to a higher-privilege tool despite a sufficient lower-privilege alternative. We introduce ToolPrivBench to evaluate whether agents choose higher-privilege tools despite sufficient lower-privilege alternatives, measuring both initial selection and escalation after transient tool failures. Across eight domains and five recurring risk patterns, we find that over-privileged tool selection is common among mainstream LLM agents and is further amplified by transient failures. We further find that general safety alignment does not reliably transfer to least-privilege tool choice, while prompt-level controls provide only limited mitigation under transient failures. We therefore introduce a privilege-aware post-training defense that teaches agents to prefer sufficient lower-privilege tools and escalate only when necessary. Our mitigation experiments show that this defense substantially reduces unnecessary high-privilege tool use while preserving general capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06195v1">LogicHunter: Testing LLM Agent Frameworks with an Agentic Oracle</a></div>
    <div class="paper-meta">
      📅 2026-07-07
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agent frameworks such as LangChain, LlamaIndex, and CrewAI have become critical infrastructure powering production AI systems, yet they remain severely under-tested due to fundamental challenges in automated testing. Unlike traditional software, where crashes serve as reliable oracles, defects in these pure Python frameworks manifest as ordinary exceptions or silent semantic failures, creating profound oracle ambiguity. This problem is exacerbated by strict type governance through Pydantic schemas and complex protocol requirements that cause existing fuzzers to generate overwhelming invalid inputs, while traditional test generators produce only trivial cases with weak regression assertions. We present LogicHunter, a fuzzing framework that addresses both the generation and oracle challenges through active specification-aware testing. LogicHunter employs specification-driven generation that systematically fuses formal type constraints with authentic usage patterns from real-world repositories, synthesizing inputs that are valid by construction yet semantically extreme, equipped with behavioral probes to expose silent failures. To resolve oracle ambiguity, we introduce the Agentic Oracle, which transcends passive classification by actively retrieving documentation, navigating source code, and inspecting runtime states through a ReAct-based architecture with Dual-Layer State Management and Dual-Stream Memory. Evaluated on three widely deployed frameworks, LogicHunter discovered 40 previously unknown bugs with 30 confirmed and 26 fixed by developers, while state-of-the-art baselines reported no bugs as final findings. The Agentic Oracle achieves 91.17% precision, surpassing the best passive approach at 29.27% by 61 percentage points.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11021v2">Leech Lattice Vector Quantization for Efficient LLM Compression</a></div>
    <div class="paper-meta">
      📅 2026-07-07
    </div>
    <details class="paper-abstract">
      Scalar quantization of large language models (LLMs) is fundamentally limited by information-theoretic bounds. While vector quantization (VQ) overcomes these limits by encoding blocks of parameters jointly, practical implementations must avoid the need for expensive lookup mechanisms or other explicit codebook storage. Lattice approaches address this through highly structured and dense packing. This paper explores the Leech lattice, which, with its optimal sphere packing and kissing configurations at 24 dimensions, is the highest dimensional lattice known with such optimal properties. To make the Leech lattice usable for LLM quantization, we extend an existing search algorithm based on the extended Golay code construction, to i) support indexing, enabling conversion to and from bitstrings without materializing the codebook, ii) allow angular search over union of Leech lattice shells, iii) propose fully-parallelisable dequantization kernel. Lastly, we provide a geometric reinterpretation of combining shape--gain quantization with GPTQ-style Hessian corrections: the standard scale-correction step of shape--gain acts as a retraction onto a product of spheres, yielding a Spherical GPTQ primarily acting on directions. We find that low-angular-distortion LLVQ reduces sensitivity to Hadamard/rotation preprocessing, and enables a strong Hadamard-free PTQ in practice. LLVQ delivers state-of-the-art LLM quantization performance, outperforming recent methods such as Quip\#, QTIP, and PVQ. The results highlight the effectiveness of high-dimensional lattices for scalable, theoretically grounded model compression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06175v1">Improving LLM-Generated Process Model Quality Through Reinforcement Learning: The Role of Reward Function Design</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 21 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can generate BPMN process models from natural-language descriptions, yet supervised fine-tuning (SFT) limits their output quality to the patterns present in the training data. Reinforcement learning (RL) can optimize beyond this ceiling using external quality measures, but how the reward function should be designed when quality is multi-dimensional remains unexplored. We present a systematic investigation of reward function design for RL-based process model generation, training two LLM families (Llama~3.1 8B, Qwen~2.5 14B) under 48 configurations using Group Sequence Policy Optimization with rewards derived from an automated evaluation framework comprising 38 metrics across syntactic, pragmatic, and semantic quality. Three findings emerge. First, RL significantly improves pragmatic and syntactic quality while preserving semantic fidelity, reducing output variability by more than sixfold. Second, equal reward weighting consistently outperforms targeted weighting: emphasizing a specific dimension fails to improve it and can collapse the model into a low-quality mode. Third, design choices interact with model architecture in non-trivial ways: the invalidity penalty is essential for one model but irrelevant for the other, and SFT initialization is indispensable for one architecture but counterproductive for another. These results demonstrate that reward composition is a primary determinant of optimization outcomes, with effects as large as the decision to apply RL itself. The findings generalize to any structured generation task where quality is assessed along multiple automated dimensions. We release our implementation and experimental code at https://github.com/chlauer99/RL_for_process_modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06157v1">LLM Agents for Deliberative Collaboration: A Study on Joint Decision Making Under Partial Observability</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 Code is available at https://github.com/wcx21/deliberative-collaboration-agents
    </div>
    <details class="paper-abstract">
      Deliberation plays a crucial role in collaboration; when humans work together, they naturally engage in communication to align information and reach an agreement. In this paper, we investigate deliberative large language model (LLM) agents under partially observable joint decision-making tasks. We formalize deliberative collaboration as a cooperative joint decision problem with partial and asymmetric observations, and introduce a scalable benchmark that instantiates this problem across multiple task settings and domains in which agents must exchange information through deliberation to reach a joint decision with a shared reward. We then instantiate a reference scaffold and evaluation protocol for deliberative agents and conduct a systematic evaluation of a range of representative LLMs. The results reveal that complex deliberative collaboration tasks continue to challenge state-of-the-art language models. Even with the aid of external mathematical tools, language models may fail in either the deliberation process for aligning information or the complex reasoning process for making the decision. On the other hand, diagnostic analysis reveals that the deliberation process may also provide opportunities for reflection and error correction, sometimes improving performance over centralized baselines. Altogether, our work establishes a foundation for evaluating and improving LLM agents in deliberative collaboration and provides insights into the strengths, limitations, and properties of current LLM-based multi-agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.24245v3">AutoSpec: Safety Rule Evolution for LLM Agents via Inductive Logic Programming</a></div>
    <div class="paper-meta">
      📅 2026-07-07
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents increasingly automate complex tasks by integrating language models with external tools and environments. However, their autonomy poses significant safety risks: agents may execute destructive commands, leak sensitive data, or violate domain constraints. Existing safety approaches face a fundamental tradeoff: hand-crafted rules are interpretable but brittle, with overly conservative rules blocking safe operations (high false positives) while permissive rules miss unsafe behaviors (high false negatives). Neural classifiers lack the interpretability required for safety-critical deployments. We present AutoSpec, a framework that automatically evolves deployed expert-designed safety rules from user safe/unsafe annotations through counterexample-guided inductive synthesis (CEGIS) guided by inductive logic programming (ILP). Starting from the expert rules and a stream of annotated traces, AutoSpec iteratively evaluates rules, mines false-positive and false-negative counterexamples, uses ILP to learn which predicates discriminate them, generates candidate rule edits, and verifies candidates to select the best revision. The key insight is that ILP efficiently identifies predicates that appear frequently in false negatives but rarely in false positives (or vice versa), dramatically pruning the exponential search space of rule edits. This continues until convergence, producing interpretable rules that balance precision and recall. We evaluate AutoSpec on 291 execution traces spanning code execution and embodied agent domains. AutoSpec raises rule F1 to 0.98 and 0.93 across the two domains, achieving up to 94% false positive reduction while maintaining high recall, and converges within 4-5 iterations. The ILP-guided approach achieves up to 4.8x higher F1 than heuristic CEGIS. The learned rules are human-readable, auditable, and generalize to unseen scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06623v1">LLM-Guided Task-Semantic Field Factorization for Industrial Process Forecasting</a></div>
    <div class="paper-meta">
      📅 2026-07-07
    </div>
    <details class="paper-abstract">
      Process industries rely on time-series forecasting and soft sensing to estimate quality variables that are hard to measure online. Labeled data are scarce, operating regimes change frequently, and retraining models or rebuilding alignment pipelines for each scenario is costly. Such settings often provide variable tables and process documents that record variable names, units, physical meanings, and process roles. However, standard time-series backbones usually treat inputs as anonymous numerical columns. Existing text-enhanced methods also rarely make the semantic-logical relations between input variables and the prediction target available to the model within each numerical window. To address this problem, this article proposes Task-Semantic Field Factorization (TSF), a large language model (LLM)-guided framework. TSF builds a task-semantic field from task protocols and variable documents before training and uses the LLM only for offline semantic construction. Online training and inference remain with conventional time-series backbones. During training and inference, the current numerical window activates variable semantics, so semantic information participates in each prediction and supports adaptation to different prediction targets and operating shifts. On multiple complex industrial forecasting and soft-sensing tasks, TSF reduces MAE by 6.4\% on average in improved settings, with the largest reduction reaching 25.5\%. It adds only about 1.8--3.0k parameters, with less than 0.008 ms/step of additional online inference overhead. These results show that TSF turns existing process documents into measurable forecasting gains across backbones and semantic generators while remaining lightweight for deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06111v1">LLM-Guided Measurement Credibility Correction for Trustworthy Industrial Process Inference</a></div>
    <div class="paper-meta">
      📅 2026-07-07
    </div>
    <details class="paper-abstract">
      Industrial prediction and soft sensing depend on credible input measurements. In field deployment, a predictor may receive biased, delayed, stale, or derived measurements that still look plausible. Prediction can then fail before the forecasting backbone becomes the main limitation, because the input window no longer represents the real process. Sensor reconstruction, data reconciliation, and fault-tolerant soft sensing reduce this risk, but they often rely on numerical correlation, alarms, fault labels, or explicit process equations. These assumptions are not always available. A correlated variable can also be an unsafe reference when variables share instruments, derived formulas, soft-sensing chains, or control actions. The key issue is to decide before prediction which external measurements can credibly support the current measurement. To address this issue, this article proposes LLM-Guided Measurement Credibility Correction (MCC). MCC converts measurement meanings in process documents into measurement semantics usable by numerical models. It builds independent process references from semantically qualified external measurements and corrects local measurement conflicts before prediction. The predictor therefore receives a more credible input window. Across multiple complex industrial forecasting and soft-sensing tasks, +MCC achieves average relative MAE reductions of 30.7% on real-test protocols and 80.3% on controlled-corruption protocols. It adds only 0.5--2.0k online parameters, with the slowest +MCC inference time at 0.089 ms/step. These results show that measurement semantics can turn process documents into lightweight pre-inference credibility correction and improve prediction accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06080v1">From Blueprint to Reality: Modeling and Applying Putnam's Social Capital Theory with LLM-based Multi-agent Simulations</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 23 pages, 13 figures, 11 tables
    </div>
    <details class="paper-abstract">
      Putnam's Social Capital Theory is a foundational framework for collective action and community prosperity. However, traditional empirical methods face practical limits on control and replication. Meanwhile, LLM-based social simulations are typically behavior-driven and lack theory-aligned environments for modeling Putnam's core propositions. To address these gaps, we introduce SocaSim, an LLM-based multi-agent simulation framework to study Putnam's Social Capital Theory from theoretical blueprint to simulated reality. Specifically, we build an environment integrating social network evolution, trust dynamics, and norm propagation, where agents engage in repeated collective-action experiments, and then apply the three dimensions to analyze adaptation challenges in smart elderly care. Our simulations reproduce Putnam's macro-level patterns and exhibit strong human-agent alignment at the group level. Unlike traditional methods, SocaSim traces micro-level causal pathways of social network, trust, and norms via round-by-round simulations and counterfactual interventions, enabling process-level interpretability. Taken together, these capabilities establish a research paradigm that leverages LLM agents to bridge social science and computer science.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06039v1">Automating Quality Assessment with NLP of LLM-Generated Defeaters</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 10 pages, 2 figures. Author preprint version of a paper published at ICSRS 2025
    </div>
    <details class="paper-abstract">
      High-integrity systems, such as autonomous vehicle fleets and large-scale energy infrastructures, rely on structured assurance cases to justify safety claims. To remain valid under evolving operational conditions, such cases must be examined against potential challenges, known as defeaters. While large language models (LLMs) can support the scalable generation of candidate defeaters, assessing their quality remains largely manual and subjective process. This paper presents an automated approach for supporting the assessment of LLM-generated defeaters using natural language processing techniques. The method combines structural features from assurance case graphs with semantic embeddings and meta-classifiers trained on expert-assessed defeater annotations. We evaluate the approach through two case studies in the automotive and energy domains. The results show substantial human reviewer dissensus, with Cohen's kappa values below 0.442, highlighting the difficulty of consistent manual assessment. Against this background, the proposed classifiers achieve an average F1-score of 0.84 in validation and show improved alignment with individual expert ratings. The findings suggest that automated assessment can help reduce subjective variance and provide scalable decision support for assurance case review, while leaving final judgment to domain experts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06001v1">Information Limits and Attractor Dynamics in Economies of Frontier LLM Agents: A Pre-Registered Test</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 15 pages. Preprint. Zenodo: https://doi.org/10.5281/zenodo.21185866. Companion synthesis: arXiv:2606.12502
    </div>
    <details class="paper-abstract">
      We report a pre-registered, two-part experiment on small economies of frontier language-model agents (Claude Opus 4.8), testing two quantitative predictions about coupled multi-agent systems: an information-theoretic capacity region for wealth growth under market coupling, and a mean-field residual-scaling law for population misalignment under incentive and control levers. All predictions, acceptance bands, and decision rules were frozen in a public git chain before any run; every reported number re-derives mechanically from cached model outputs; the entire experiment cost $138.76 in metered API spend and is re-runnable at zero cost from the cache. Result 1 (confirmation): in parimutuel-coupled economies, relative growth equals relative claimed information -- the gap law G_a - G_b = I_a - I_b holds to a worst-case 46 millinats (pre-registered band: 50) across four perception structures; coalition value is submodular exactly where channels are conditionally independent, and a designed XOR synergy control flips it supermodular by 0.62 >= ln2/2 nats, with agents reasoning out the joint bit; the joint growth ceiling G_S <= H(X) binds exactly; and the best-informed agent absorbs essentially the whole wealth pool in 4/5 market seeds. Result 2 (structural negative): the residual-scaling test returned "domain not found." In all 72 population runs, goal dispersion collapsed (V -> 0; maximum 4.85 against a frozen floor of 5.31), the population's response to the two levers was a step function across the dominance boundary rather than a smooth response, and cells near the boundary were bistable with seed-selected outcomes. No tested LLM population at any capability level realizes the noise-maintained-dispersion regime the smooth mean-field model assumes. We release the full protocol, pre-registration chain, call cache, and analysis code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06000v1">Context-to-Execution Integrity for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      Language-model agents read attacker-writable context to solve tasks. Tool execution needs a separate authority check for protected sink fields, sink-interpreted payloads, and the invocation event. Context-to-Execution Integrity (CXI) is an execution-boundary system for this setting. Policies mark protected sink fields, typed releases carry narrow validated values from writable context to specific destinations, opaque data slots keep evidence as data, and a deterministic gate admits a call only after field authority, exact-effect authorization, and invocation authority all bind to the same action manifest. We evaluate CXI on open-weight field-projection runs, AgentDojo live episodes, a code-agent exact-effect benchmark, manifest-bound ledger faults, proposal-pressure controls, and hosted/API compatibility traces. AgentDojo covers 720 live episodes and 1,739 LLM calls; the code-agent benchmark covers 400 repository episodes with exact-effect authorization and lease-bound execution, yielding 231 safe task completions and zero observed field, effect, or invocation escapes. The accounting reports parser outcomes, authorization outcomes, and task-quality outcomes together with the admission-integrity result. Across the evaluated sinks, CXI admits execution only when field, effect, and invocation authority bind to the same action manifest.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05985v1">Auto-DSM Under the Lens: A Black-Box Evaluation Framework for LLM-Based DSM Generation</a></div>
    <div class="paper-meta">
      📅 2026-07-07
    </div>
    <details class="paper-abstract">
      This paper presents a black-box evaluation framework to systematically assess the ability of Large Language Models (LLMs) to generate Design Structure Matrices (DSMs) from structured technical documentation. Motivated by the closed-source nature of current Auto-DSM pipelines, the framework introduces a reproducible methodology that benchmarks generated DSMs (GEN-DSMs) against manually validated ground-truth matrices (GT-DSMs). The evaluation integrates both single-run and multi-run perspectives, combining structural metrics (Completeness, Correctness, Coupling Density), classification metrics (Selective Accuracy, Abstention Coverage), and stability measures (Entropy, Fleiss' $κ$). To synthesize these aspects, a Composite Quality Score (Q) is proposed. Controlled experiments are conducted on two datasets: a fictive abstract system and a real-world refrigerator decomposition, covering variations in phrasing, parameter-dataset alignment, and system complexity. Results show that LLMs can produce structurally plausible DSMs and achieve high reproducibility under well-structured inputs, but remain sensitive to ambiguity, inconsistent dependency definitions, and prompt formulation. The findings highlight systematic sources of hallucination and abstention failure, demonstrating both the potential and current limitations of LLM-driven DSM automation. The proposed framework provides a transparent benchmark for auditing Auto-DSM pipelines and establishes foundations for integrating LLM-based decomposition methods into model-based systems engineering (MBSE) workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05970v1">Faithful or Findable? Evaluating LLM-Generated Metadata for RDF Dataset Search</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 5 pages, 1 figure, accepted at SynthIR @ SIGIR 2026
    </div>
    <details class="paper-abstract">
      Dataset search depends heavily on metadata, making LLM-generated metadata a consequential form of synthetic content in retrieval systems. We study six metadata-generation settings for RDF datasets, ranging from simple rewriting to profile-grounded and agentic graph-based generation, and evaluate them jointly for retrieval effectiveness and faithfulness. Unconstrained metadata rewriting delivers the strongest retrieval gains over the original metadata, but it is also the least faithful, showing that search improvements can be driven by unsupported semantic expansion. More grounded settings substantially improve faithfulness, and profile-grounded rewriting provides the most balanced trade-off between retrieval effectiveness and grounding. These findings position synthetic metadata as a system-level IR problem in which effectiveness, provenance, and trust must be evaluated together.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05956v1">Integrating knowledge graphs and multilingual scholarly corpora for domain-adaptive LLMs in SSH</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 8 pages, 4 tables, workshop LLMs4SSH of LREC 2026 conference
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into scientific research workflows, particularly for bibliographic discovery and literature synthesis, raises significant methodological, epistemic and regulatory challenges for the Social Sciences and Humanities (SSH), especially with regard to disciplinary diversity, multilingual access to sources and the evaluation of results. This paper presents an on-going use case developed within the European project LLMs4EU and the ALT-EDIC infrastructure, aimed at adapting foundation models to SSH research practices and supporting tasks such as question answering, comparative document analysis and literature review. The evaluation framework follows the LLMs4EU protocol and encompasses both independent quantitative benchmarking (retrieval, summarisation, traceability and hallucination detection) and a qualitative assessment involving a panel of Digital Humanities experts. By embedding model adaptation within research infrastructures and a structured legal and ethical compliance framework, the use case explores how domain-sensitive and regulation-aware generative AI can support SSH scholarship while preserving reliability and epistemic responsibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05936v1">Mitigating Errors in LLM-Generated Web API Invocations via Retrieval-Augmented Generation and Constrained Decoding</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 54 pages, 11 figures; supersedes arXiv:2509.20172v6, which is a discarded journal extension of our work
    </div>
    <details class="paper-abstract">
      Integration of web APIs is a cornerstone of modern software systems, yet writing correct web API invocation code remains challenging due to complex and evolving API specifications. Although LLMs are increasingly used for code generation, previous work has empirically shown that their ability to generate correct web API integrations is limited. At the same time, mitigation techniques and their effectiveness for this setting remain insufficiently understood. In this paper, we propose and systematically evaluate retrieval-augmented generation (RAG) and constrained decoding (CD) as two complementary approaches to improving LLM-generated web API invocation code. For RAG, we design a retriever that processes OpenAPI specifications and retrieves compact endpoint representations to inject into model prompts. For CD, we introduce an automatic translation from OpenAPI specifications to regex-based constraints enforced during generation. We evaluate both approaches on WAPIIBench's existing synthetic dataset and on a new real-world dataset derived from GitHub repositories. Our results show that RAG reduces hallucinations and improves correctness when generating full API invocations but reduces it when the endpoint is already provided as it encourages the generation of unnecessary parameters. In contrast, CD reliably prevents illegal URLs, HTTP methods, and arguments and substantially improves overall correctness for both starter codes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05704v1">LLM-Driven Neural Network Generation with Same-Family Architecture Guidance: Disentangling Transfer and Adaptation</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 10 pages, 1 figure, 14 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can generate neural-network modifications, but unrestricted generation is often invalid or harmful. This paper studies a narrower setting: improving a weak target model using a stronger same-family source model from a neural-network database. We propose a source-guided candidate-generation protocol with non-source controls, source-conditioned candidates, and a no-LLM hp_copy ablation under equal evaluation budgets. The protocol reports validity separately from accuracy and selects the best valid candidate only when it improves the target. On CIFAR-10, the strongest source-guided candidate reaches 0.5049 accuracy versus 0.2398 for the best non-source candidate, a +0.2651 advantage, while improving a weak target originally at 0.1254; a five-epoch check preserves the gain at 0.7686 versus 0.4839. On SVHN AlexNet with DeepSeek-Coder-6.7B, source-guided transfer reaches 0.7880 versus 0.2254, a +0.5626 advantage; a fresh repeat reaches 0.8069 versus 0.2509, a +0.5560 advantage. Direct source-recipe copy produces 0.1959 on SVHN AlexNet, matching the original target, while hp_transfer reaches 0.7880, showing that the LLM adapts rather than copies the source recipe. Family-level analysis shows the clearest positive signals for AlexNet, with 6/8 wins across SVHN, Imagenette, and CelebA-Gender, and alt_nn1, with 8/10 wins on CIFAR-10.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05694v1">Beyond Heuristic Tuning: Power-Calibrated LLM Watermarking</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 Accepted ICML 2026
    </div>
    <details class="paper-abstract">
      Logit-based watermarking is a widely used mechanism for identifying LLM generated content, yet its effectiveness is governed by a fundamental trade-off between detectability and semantic distortion. Existing analyses provide limited guidance for principled hyperparameter selection, leaving practical deployments reliant on heuristic tuning. In this work, we develop a power-calibrated statistical framework that establishes explicit quantitative relationships between watermark hyperparameters, detection power, and distortion. This characterization transforms watermark design into a guided optimization problem. Building on these results, we derive practical parameter selection procedures that achieve optimal tradeoffs under constraints. Extensive experiments across multiple language models and datasets validate the theory and demonstrate that the proposed framework consistently identifies Pareto-optimal points.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01646v2">PHOENIX: Resilient LLM Training with Hot-Swapping via Zero-Overhead Checkpoint</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      State-of-the-art large language model (LLM) training takes tens of thousands of graphics processing units (GPUs) for months and encounters failures across the software and hardware stack. Existing fault-tolerance mechanisms either impose non-trivial overhead during failure-free execution or suffer from prolonged recovery latency, particularly under scenarios where a small subset of compute nodes experience permanent failures. %The tradeoff between failure-free overhead and recovery latency forms a space forms a Pareto frontier We present PHOENIX to simultaneously address both optimization objectives. PHOENIX incorporates a fault-tolerance mechanism that restores LLM training via hot-swapping, namely by replacing failed nodes with spare nodes without terminating the complete job. The hot-swapping of PHOENIX is enabled by two ideas: First, it exploits an off-critical-path in-memory checkpointing mechanism for spatial redundancy. Second, it introduces a communicator reconstruction protocol that replaces failed nodes with spare nodes at runtime. PHOENIX efficiently overlaps the in-memory checkpointing with computation, thus introducing zero overhead during error-free execution. Upon permanent node failures, PHOENIX can rebuild memory states with minimal recomputation by leveraging in-memory checkpoints. We evaluate PHOENIX across scales (up to 512 NVIDIA A100 GPUs) and LLMs (up to 65B parameters), and observe zero checkpoint overhead with hot-swapping recovery completing in under 40 seconds. These results show that PHOENIX simultaneously achieves both zero-overhead error-free execution and extremely low recovery cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05682v1">FirstResearch: Auditable Question Formation for LLM Scientific Discovery Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      LLM systems for scientific discovery increasingly assist with ideation, literature synthesis, experiment planning, and report generation, but the first research question they propose can remain difficult to audit: it may sound plausible without exposing the mechanism, falsifier, or assumption that a scientist should inspect. We introduce FirstResearch, a first-principles research-question formation framework for scientific LLM agents whose core artifact is a structured Research Question Certificate. The certificate records primitive definitions, assumptions, a mechanism model, a tension or contradiction, a falsifiable hypothesis, a minimal decisive test, and a failure update rule, making the proposed question inspectable before downstream execution. On ten LLM-agent research topics, FirstResearch outperforms controlled prompt-level baselines inspired by AI co-scientist, Agent Laboratory, and AI Scientist-v2 under a primary DeepSeek-blind-judge protocol. A Gemini-2.5-Flash independent-judge rescore of the same 40 baseline packages preserves the system-level ranking, with FirstResearch scoring 4.86/5 versus 4.38/5 for the strongest baseline and Pearson agreement of 0.865 on average score. A one-repeat ablation checkpoint further suggests that the certificate-centered core is the strongest component: certificate-only scoring reaches 4.90/5 under DeepSeek and 4.88/5 under Gemini, while removing certificates drops below 1/5 under both judges. These results are preliminary and use LLM judges rather than human domain experts, but they support a narrow scientific-discovery claim: explicit derivation constraints are a promising mechanism for making LLM-generated scientific questions more auditable. Code, prompts, saved outputs, and reproduction scripts are available at https://github.com/louiswang524/FirstResearch.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.00476v2">Doing What They Say, Not What They Reason: Locating the Faithfulness Gap in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 submitted to COLM social simulation with LLM workshop
    </div>
    <details class="paper-abstract">
      Do LLM agents act on the reasoning they state? This question of process fidelity is central to LLM-based social simulation, yet hard to measure where no reference for correct behavior exists. We study it in a controlled setting: a Texas Poker simulator with a verifiable reference action for every decision by splitting the faithfulness gap into two steps: reasoning-to-conclusion (does the stated decision follow from the agent's own reasoning?) and conclusion-to-action (does the agent execute what it states?). The two steps behave very differently. Conclusion-to-action is reliable: inconsistency is 0.7% for Claude Haiku 4.5 and 1.4% for DeepSeek-Reasoner once the conclusion is read from an explicit tag, whereas free-text conclusion extraction reports 22-26%. Reasoning-to-conclusion is where fidelity frays, but not through a single dominant failure. In a step-level diagnostic the agent's errors split roughly evenly between bad inputs, borderline cases, and rule misapplication deriving a conclusion that contradicts the agent's own restated rule from inputs it estimated correctly. This composition is model-dependent: rule misapplication accounts for a third of Haiku's interpretable errors but only 8% of DeepSeek's. The one robust signal is directional: when an agent does misapply its own stated rule, it almost always (99.5% for Haiku) errs in the risk-averse direction. The override is partly hedging behavior, not a capability limit: instructing the agent to apply the rule mechanically halves the misapplication rate (13.9% to 6.8% of decisions) and raises adherence by eight points. Process-fidelity evaluation should therefore elicit machine-checkable conclusions and probe for directional biases rather than assume a single upstream failure mode, lest it conflate measurement noise with model behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01148v2">Emergence of Preferential Attachment and Glass-Ceiling Effects in Autonomous Networks of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      We investigate the emergence of structural disparities in networks of collaborating large language model (LLM) agents. When LLM agents autonomously choose collaborators, the resulting communication network exhibits preferential-attachment dynamics: agents that are already prominent become increasingly likely to attract additional connections. In some cases, weaker LLM agents (agents with smaller base model or older version) can disproportionately occupy central and influential network positions relative to stronger LLM agents. We interpret this as a type-dependent glass-ceiling effect (GCE). We model the network of LLM agents as a time-evolving sequence of directed weighted graphs, where the vector-valued edge weights represent cumulative tokens exchanged, number of interaction rounds, and reasoning effort. Using a contraction mapping argument on the mean-field dynamics, we prove that the importance (centrality) of each agent type converges to a unique stable equilibrium. To ground the model in LLM decision mechanisms, we introduce a cross-attention-inspired utility for collaborator selection. This utility specifies the local connection dynamics and, together with the mean-field model, yields a predictive characterization of the limiting network structure and its type-dependent centrality gaps. To validate the theory, we develop an experimental testbed with 100 LLM agents. Our experiments show that autonomous network formation can generate persistent centrality disparities, with their magnitude and direction depending on model family, model size, system-prompt design, and task context. They further show that the effect of preferential attachment depends on its alignment with model capability: reinforcing it improves collective performance when stronger agents become central, whereas weakening it improves performance when network dynamics instead favor weaker agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.14948v2">Beyond Correctness: Enhancing Architectural Reasoning in Code LLMs via Scalable Labeling with Agentic Judgment</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      LLMs have substantially improved software engineering yet real-world development requires architectural understanding. Such understanding is prohibitively expensive to label manually and impossible to verify through tests alone. We propose an agentic judging pipeline using a strong LLM as a scalable proxy for expert architectural evaluation, comprising two judges: the Architecture Complexity Judge (ACJ), which estimates codebase-specific architectural understanding a task demands, and the Architecture Quality Judge (AQJ), which evaluates patch conformance to repository-specific architectural conventions via source-grounded rubrics. Fine-tuning Qwen3-8B/14B/32B on 3,360 curated instances achieves resolved rates of up to 27.2% on SWE-bench Verified - up to 540% over the base model and 256% over unfiltered fine-tuning. Meanwhile, the trained models achieve strong cross-language generalization and consistent improvements in architectural patch quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05587v1">A Mechanistic Lens on Semantic Conflicts: Using Activation Patching to Understand LLM Behavior</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in software-engineering tasks processing executable code and non-executable semantic cues such as comments or identifiers. These two sources can conflict when semantic cues suggest different program behavior than the code itself. It remains unclear how such semantic conflicts affect LLM behavior and which source dominates their outputs. We present the first controlled, mechanistic study of LLM behavior under semantic conflicts. To this end, we construct 45 Python snippet triplets that isolate conflicts by varying either semantic cues or implementation while keeping token-aligned pairs for causal intervention. We evaluate four open-weight LLMs on two tasks (output prediction and unit-test generation) using behavioral performance measures and residual-stream activation patching to identify token-layer states that causally contribute to behavioral differences between aligned and conflicting inputs. Our results show that semantic conflicts significantly reduce execution-grounded correctness in both tasks and that all tested LLMs often follow misleading semantic cues. Residual-stream activation patching reveals a consistent pattern for final-output prediction: The changed cue/code region and a small set of intermediate tokens carry most of the recoverable causal signal before aggregation near the output readout. For unit-test generation, this pattern extends beyond the prompt, showing that conflict-related information is recoverable at generated sites before producing expected values. Overall, our findings show that semantic conflicts affect program comprehension and downstream tasks, with relevant information concentrated in a small number of causally active residual-stream states, and demonstrate a framework for mechanistically analyzing how LLMs integrate code-related information under controlled semantic variations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05545v1">Most LLM Conformity Needs No Speaker: Measuring the Speaker-Free Floor in Peer-Pressure Benchmarks</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      LLM conformity is often used to describe cases where a model changes a correct answer toward a peer or group response. We show that most of this apparent conformity survives even after the peer is removed. The reason is a confound: standard conformity prompts mix two cues at once, the presence of a speaker and the repeated wrong answer itself. Existing benchmarks vary these cues together, so they cannot tell how much of the revision actually depends on the speaker. We introduce a no-source condition: the same asserted answer with the explicit speaker removed. Across six open-weight LLMs and seven QA and reasoning datasets, this condition alone causes harmful revision in $66.5\%$ of initially correct cases, compared with $10.3\%$ under a plain re-ask. The effect also remains when the repeated answer is paraphrased and when answer options are hidden in an open-ended setting. Source framing mainly modulates this floor: expert-panel framing raises it, while minimal person labels do not reliably raise it. When models flip, they are usually confidently wrong, and simple recalibration does not recover the original answer. Source attribution still matters, but it should be measured as an increment above this speaker-free floor. The methodological lesson is that conformity benchmarks should first measure what remains after the speaker is removed; without this step, benchmarks may mistake repeated text for social influence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.22758v2">MASCA: LLM based-Multi Agents System for Credit Assessment</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 Accepted at NeurIPS GenAI In Finance Workshop
    </div>
    <details class="paper-abstract">
      Recent advancements in financial problem-solving have leveraged LLMs and agent-based systems, with a primary focus on trading and financial modeling. However, credit assessment remains an underexplored challenge, traditionally dependent on rule-based methods and statistical models. In this paper, we introduce MASCA, an LLM-driven multi-agent system designed to enhance credit evaluation by mirroring real-world decision-making processes. The framework employs a layered architecture where specialized LLM-based agents collaboratively tackle sub-tasks. Additionally, we integrate contrastive learning for risk and reward assessment to optimize decision-making. We further present a signaling game theory perspective on hierarchical multi-agent systems, offering theoretical insights into their structure and interactions. Our paper also includes a detailed bias analysis in credit assessment, addressing fairness concerns. Experimental results demonstrate that MASCA outperforms baseline approaches, highlighting the effectiveness of hierarchical LLM-based multi-agent systems in financial applications, particularly in credit scoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.28815v2">Categorizing Mathematical Concepts with LLM Voting Ensembles in Mathswitch</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 Submitted (pre-peer-review) version. Accepted at CICM 2026; the Version of Record will appear in Springer LNAI. We'll add the DOI once the proceedings are published
    </div>
    <details class="paper-abstract">
      Mathswitch is an open-source project that imports mathematical concept records from sources such as Wikidata, Wikipedia, MathWorld, Encyclopedia of Mathematics, nLab, ProofWiki, and Agda-Unimath, and links records that refer to the same concept. It does not reorganize or redefine the imported content; each source retains its own structure. The current focus is on importing concept data from Wikidata and the resources it links to, with plans to expand to further sources and better concept linking. Because the concept set is approximated through queries over Wikidata's collaboratively edited graph, the imported data is noisy: some items are non-mathematical, while others are ambiguous. In this paper, we test whether a voting ensemble of LLM judges can filter this noise. We evaluate it on Wikidata items with known MathWorld identifiers as a positive control, and examine how classification changes when database identifiers are removed from context. We then inspect the cases where the judges disagree with MathWorld and group these disagreements into three categories (degenerate descriptions, narrow scope bias, and editorial-scope mismatches) that suggest different remediation strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.00448v2">Real-Time Hard Negative Sampling via LLM-based Clustering for Large-Scale Two-Tower Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      The two-tower model has been widely used for large-scale recommendation systems, particularly in the retrieval stage. Industry standards for training two-tower models typically involve in-batch and/or out-of-batch negative sampling. However, these methods often produce easy negatives that models can quickly learn, failing to sufficiently challenge the model. To address this issue, a novel self-supervised hard negative sampling technique is proposed that leverages a large language model (LLM) to generate hard negatives from the same cluster during model training. By utilizing the LLM to learn media representations, the proposed approach ensures that the generated negatives are more challenging and informative. This real-time sampling framework is designed for seamless integration into production models, capable of handling billions of training data points with minimal computational complexity. Experiments on public datasets, along with deployment to a large-scale online system, demonstrate that the proposed negative sampling technique outperforms widely used industry methods. Furthermore, analysis in industrial applications reveals that this sampling method can help break inherent feedback loops in recommendations and significantly reduce popularity bias.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05316v1">How Much is Left? LLMs Linearly Encode Their Remaining Output Length</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 21 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Large language models generate one token at a time, yet their responses show remarkably consistent length structure: step-by-step solutions converge in predictable token counts, retrievals stop after a few sentences, retractions extend responses by measurable amounts. We ask whether the model carries an internal estimate of how much response remains. Training minimal-capacity linear probes on frozen hidden states of three open-weight 7-8B models across seven completion-style datasets, we find three converging pieces of evidence. First, total response length is linearly decodable from the prompt's last hidden state alone, before any output is emitted. Second, probe directions trained on natural-language datasets transfer broadly, including to controlled synthetic completions never seen in training, outperforming a statistical baseline; the converse direction generally fails, and this asymmetry is itself informative. Third, on curated high-loss completions, the probe's per-position estimate shifts upward at the moment the model retracts and restarts a partial solution, a directional behavior no position-only predictor can reproduce (qualitative, not aggregate). We frame this as approximate estimation of remaining generation length, distinct from exact-counting impossibility results for transformers, and interpret it as evidence that LLMs maintain a plan-like internal representation of output length (decodable, not necessarily used causally).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.03463v3">The Impact of Critique on LLM-Based Model Generation from Natural Language: The Case of Activity Diagrams</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 Accepted in the Empirical Software Engineering (EMSE) Journal (2026)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show strong potential for automating model generation from natural-language descriptions. A common approach begins with an initial model generation, followed by an iterative critique-refine loop in which the model is evaluated for issues and refined based on those issues. This process needs to address: (1) structural correctness -- compliance with well-formedness rules -- and (2) semantic alignment -- accurate reflection of the intended meaning in the source text. We present LADEX (LLM-based Activity Diagram Extractor), a pipeline for deriving activity diagrams from natural-language process descriptions using an LLM-driven critique-refine process. Structural checks in LADEX can be performed either algorithmically or by an LLM, while alignment checks are performed by an LLM. We design five ablated variants of LADEX to study: (i) the impact of the critique-refine loop itself, (ii) the role of LLM-based semantic checks, and (iii) the comparative effectiveness of algorithmic versus LLM-based structural checks. To evaluate LADEX, we compare generated diagrams with expert ground truths using a trace-based behavioural and an LLM-based matcher. This enables automated measurement of correctness (whether the generated activity diagram includes the ground-truth nodes) and completeness (how many of the ground-truth nodes the generated activity diagram covers). Experiments on two datasets -- a public-domain dataset and an industry dataset from our collaborator, Ciena -- indicate: (1) Both matchers yield similar completeness and correctness comparisons. (2) The critique-refine loop improves structural validity, correctness, and completeness compared to single-pass generation. (3) Activity diagrams refined based on algorithmic structural checks achieve structural consistency, whereas those refined based on LLM-based checks often still show structural inconsistencies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.24044v2">Data Driven Optimization of GPU efficiency for Distributed LLM-Adapter Serving</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 update of the journal paper contents after major revision
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) adapters enable low-cost model specialization, but introduce complex caching and scheduling challenges in distributed serving systems where hundreds of adapters must be hosted concurrently. While prior work has largely focused on latency and throughput optimization, minimizing GPU resource requirements through near-peak utilization remains largely underexplored. This paper presents a data-driven pipeline that, for a given workload, computes an adapter placement that serves the workload with the minimum number of GPUs while avoiding request starvation and GPU memory errors. To that end, the approach identifies the maximum feasible throughput attainable on each GPU by leveraging accurate performance predictions learned from real serving behavior. The proposed pipeline integrates three components: (i) a Digital Twin (DT) tailored to LLM-adapter serving, (ii) a distilled machine learning (ML) model trained on DT-generated data, and (iii) a greedy placement algorithm that exploits ML-based performance estimates to maximize GPU efficiency. The DT emulates real system dynamics with high fidelity, achieving below 5% throughput estimation error while executing up to 90x faster than full LLM benchmarking across both predictable and unpredictable workloads. The learned ML models further accelerate performance estimation with marginal accuracy degradation, enabling scalable optimization. Experimental results demonstrate that the pipeline substantially improves GPU efficiency, reducing the number of GPUs required to sustain target workloads by 60\% on average across the evaluated scenarios. Beyond GPU efficiency, the pipeline can be adapted to alternative objectives, such as latency minimization, highlighting its versatility for future large-scale LLM serving infrastructures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05297v1">MetaSkill-Evolve: Recursive Self-Improvement of LLM Agents via Two-Timescale Meta-Skill Evolution</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      Recent LLM agents tackle increasingly long-horizon, open-ended tasks, and external skills, reusable procedural knowledge supplied to the agent, further extend this capability. However, a fixed, hand-authored skill is rarely optimal, and cannot adapt to the diversity of tasks an agent encounters. Self-improving agents address this by rewriting their own skill files from execution traces, yielding meaningful gains on challenging benchmarks. Yet such self-evolution remains non-recursive: it improves only the task skill (what the agent does) while the improvement procedure (how it improves) is authored once and held fixed. We introduce MetaSkill-Evolve, a two-timescale framework that makes agentic skill improvement recursive: every branch carries both a task skill $s$ and a branch-local meta-skill $m=(ψ,σ,α,π,\varepsilon)$ whose five components parameterise the Analyzer, Retriever, Allocator, Proposer, and Evolver agents of the improvement pipeline. Task skills evolve on a fast loop while the meta-skill evolves on a slower one under the same pipeline applied to itself, with no additional model or objective. With all five pipeline agents sharing a single frozen backbone, MetaSkill-Evolve outperforms no-skill, static-skill, and single-level evolution baselines on three agentic benchmarks (OfficeQA, SealQA, ALFWorld), improving held-out test accuracy over the raw backbone by +23.54, +16.09, and +1.92 points respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.01827v5">PSearch: Search-based Patch Generation in the Era of LLM-based Automated Program Repair</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 accepted to ASE 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have substantially advanced Automated Program Repair (APR), yet most existing LLM-based APR methods still rely on trial-and-error to generate patches. Such a strategy explores candidate patches in a weakly structured manner, making it difficult to assess the future potential of search directions and allocate search budget effectively. To address this limitation, we propose Psearch, a search-based patch generation framework for LLM-based APR centered on iterative patch evaluation and refinement. Instead of treating patch generation as repeated independent sampling, Psearch maintains a structured search state over intermediate patches, continuously evaluates the promise of explored search paths, and prioritizes the most promising ones for further refinement. This design enables Psearch to abandon weak directions early and progressively approach correct fixes through long-horizon search. Importantly, Psearch can be integrated with different search algorithms, while our current implementation adopts Monte Carlo Tree Search as one effective instantiation. We evaluate Psearch on five widely used bug and vulnerability benchmarks. Experimental results show that Psearch correctly repairs 201 out of 835 bugs in Defects4J, outperforming all 12 state-of-the-art baselines. Psearch also fixes 27 of 79 vulnerabilities in VUL4J and resolves 164 of 300 issues in SWE-Bench-Lite. Moreover, with a patch size of 16, Psearch reduces monetary cost to roughly 50% of strong baselines while maintaining superior repair effectiveness. These results highlight the effectiveness of Psearch for improving LLM-based APR. The code and results can be found at https://github.com/Tomsawyerhu/Psearch
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05290v1">ChatImage: Navigating Long-Form LLM Answers through Interactive Images</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 Project:https://wencanjiang.github.io/ChatImage
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can produce detailed answers to complex queries, but these answers are typically presented as dense linear text, which makes fine-grained inspection, navigation, and return visits difficult. We present ChatImage, a system that converts long-form LLM answers into interactive visual images. Given a textual answer, ChatImage first normalizes its content into structured visual modules, plans a visual layout, and renders a coherent image. It then applies a second grounding pass to the rendered image with vision grounding models such as LocateAnything and MiMo-Vision, with optional SAM-style mask refinement, to identify the visible regions that should support interaction. From these grounded regions, ChatImage overlays transparent clickable hotspots on the image. Each hotspot opens a detail panel and a region-scoped follow-up thread, allowing the user to inspect and query a specific part of the answer without re-reading the full response. Instead of treating planned coordinates as the final interaction geometry, ChatImage uses them as priors and grounds the interaction targets after rendering, which improves consistency between visual content and clickable regions. We release a reference implementation and introduce a 30-question benchmark covering infographic, map, and scene-based answer formats. Evaluation with configured external models reports interaction-loop completion, a strict visual-alignment gate, and a SAM-based mask-completeness diagnostic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05483v1">PatchOptic for Shared-State LLM Workflows with Projected Views and Verified Structured Updates</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 24 pages, 13 figures, including appendix
    </div>
    <details class="paper-abstract">
      Agentic workflows often operate over shared, structured state. Because LLM context windows are limited, each model invocation is typically shown only the state fragment needed for the current workflow step, a pattern commonly known as progressive disclosure. Modern systems construct such model-facing views using grep-like keyword search, retrieval-augmented generation (RAG), abstract-syntax-tree (AST) queries, and task-specific agent skills. These methods make the read side manageable, but they do not define when a locally proposed rewrite is valid after it is applied back to the full state. The missing piece is a contract between local updates and global validity. We introduce PatchOptic, an optic-inspired interface for shared-state LLM workflows. Optics are compositional bidirectional accessors that describe how views of structured data are read and updated. PatchOptic borrows this view/update intuition and realizes it through projected reads and verified structured patches. Each workflow step declares a projected read view, an authorized write region, and a patch-source region. Beyond runtime enforcement, the same declaration yields a path-level footprint that supports delegation, sub-workflow composition, and static certificates for reordering independent steps within the same phase. We evaluate this design with PatchBench, a benchmark with 46 cases across domains. The results show that projected reads reduce reported leakage and token cost while preserving accepted-output quality under the strong actor. Runtime verification blocks declared workflow-contract violations before commit, and patch-read enforcement rejects compromised patch artifacts that use hidden sources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.18366v2">Toward Efficient Uncertainty in LLMs through Evidential Knowledge Distillation</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 Accepted at the European Conference on Machine Learning (ECML PKDD) 2026
    </div>
    <details class="paper-abstract">
      Accurate uncertainty quantification remains a key challenge for standard LLMs, prompting the adoption of Bayesian and ensemble-based methods. However, such methods typically necessitate computationally expensive sampling, involving multiple forward passes to effectively estimate predictive uncertainty. In this paper, we introduce an approach enabling uncertainty estimation in LLMs without incurring the heavy inference latency typically associated with sampling methods. Specifically, we distill uncertainty-aware teachers - originally requiring multiple forward passes - into single-pass students, fine-tuned using LoRA. We compare two distinct distillation strategies: one in which the student employs traditional softmax-based outputs, and another in which the student leverages Dirichlet-distributed outputs to explicitly model epistemic uncertainty via evidential learning. Empirical evaluation on classification tasks demonstrate that such students can achieve comparable predictive and uncertainty quantification performance relative to their teachers, while requiring only a single forward pass.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05197v1">Is Three the Magic Number? An Empirical Evaluation of LLM-Based Repair Loops</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 4 Pages (+1 for references), NIER Paper
    </div>
    <details class="paper-abstract">
      Iterative repair loops have become a core design pattern in LLM-based software engineering systems. These workflows repeatedly generate, validate, and repair artifacts using feedback such as compiler errors or test failures. Despite their widespread use, the impact of repair-loop iteration limits remains poorly understood, as most prior work adopts fixed, often arbitrary, repair budgets. We study repair-loop effectiveness across multiple software engineering tasks, including code generation, test generation, and code translation. Across several representative workflows, datasets, and contemporary low-cost LLMs, we observe a consistent pattern of diminishing returns: the first three to four repair iterations account for most achievable gains, while later iterations contribute only marginal improvements. We further find that repair behavior is influenced more strongly by workflow orchestration and feedback design than by the underlying model itself. These results suggest that repair budgets should be treated as an explicit experimental variable, as they directly affect evaluation outcomes, computational cost, runtime, and reproducibility in LLM-based software engineering research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10785v3">Developing an LLM-Based Feedback System Grounded in Evidence-Centered Design to Support Physics Problem Solving</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      Generative AI offers new opportunities for individualized and adaptive learning, e.g., through large language model (LLM)-based feedback systems. While LLMs can produce factually correct feedback for relatively straightforward conceptual tasks, delivering high-quality feedback for tasks that require advanced domain expertise, such as physics problem solving, remains a substantial challenge. This study presents the design and implementation of an LLM-based feedback system for physics problem solving grounded in evidence-centered design and reports a first evaluation within the German Physics Olympiad. Participants rated the usefulness and correctness of the generated feedback for each implemented problem. The collected ratings indicate that the feedback was generally perceived as useful and highly correct. However, an in-depth analysis revealed that the feedback contained errors in 20% of cases; errors that often went unnoticed by the students. We discuss the risks associated with uncritical reliance on LLM-based feedback and outline potential directions for generating more adaptive and reliable LLM-based feedback in the future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05139v1">On the risk of coding before testing: An empirical study on LLM-based test generation workflow</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in software engineering workflows to generate both source code and test suites. This dual capability has enabled emerging development paradigms, including test-first and agentic workflows, where a single model is producing and validating implementations. However, these approaches assume that generated tests act as independent and reliable oracles - a fundamental requirement for effective software testing. In this paper, we challenge this assumption and investigate whether LLM-generated code biases the generation of subsequent tests. We introduce and empirically study the phenomenon of error propagation, where faults in generated code are systematically replicated in associated test artifacts. This leads to cases where incorrect implementations and tests are mutually consistent, masking defects rather than revealing them. We evaluate this effect across a range of programming tasks and agentic workflows, analyzing the consistency between generated code and test assertions, with particular focus on scenarios of aligned failures. Our study examines (i) whether erroneous code artifacts bias test generation, (ii) whether such bias persists under different prompting strategies, including chain-of-thought reasoning, and (iii) how errors propagate across multi-step workflows in which intermediate outputs are reused as context. The results show that error propagation is prevalent and impactful: generating tests after faulty code significantly reduces fault detection effectiveness compared to generating tests independently (14% vs. 25%). These findings highlight a fundamental limitation of current workflows, where lack of independence between generated artifacts undermines the reliability of automated testing. Furthermore, our results expose a previously underexplored threat to validity in empirical studies relying on coupled generation pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.16181v2">LLM-Assisted Semantic Alignment and Integration in Collaborative Model-Based Systems Engineering Using SysML v2</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 Accepted by IEEE ISSE 2025, DOI pending
    </div>
    <details class="paper-abstract">
      Cross-organizational collaboration in Model-Based Systems Engineering (MBSE) faces many challenges in achieving semantic alignment across independently developed system models. SysML v2 introduces enhanced structural modularity and formal semantics, offering a stronger foundation for interoperable modeling. Meanwhile, GPT-based Large Language Models (LLMs) provide new capabilities for assisting model understanding and integration. This paper proposes a structured, prompt-driven approach for LLM-assisted semantic alignment of SysML v2 models. The core contribution lies in the iterative development of an alignment approach and interaction prompts, incorporating model extraction, semantic matching, and verification. The approach leverages SysML v2 constructs such as alias, import, and metadata extensions to support traceable, soft alignment integration. It is demonstrated with a GPT-based LLM through an example of a measurement system. Benefits and limitations are discussed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.02072v2">kNNGuard: Turning LLM Hidden Activations into a Training-Free Configurable Guardrail</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 17 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in domains requiring guardrails to detect unsafe, off-topic, or adversarial prompts. Existing guardrails predominantly rely on fine-tuning to build classifiers, which often suffer from low generalization and high inference latency. We present kNNGuard, a training-free guardrail that utilizes the activation space of an off-the-shelf LLM. Given a small bank of 50 safe and unsafe prompts, kNNGuard extracts hidden activations and performs multi-layer kNN fusing activation-space and embedding-space scores for classification. Across six domains spanning topical and security prompts, kNNGuard achieves competitive or superior F1 compared to fine-tuned state-of-the-art guardrails while running 2.7x faster than the best comparable guardrail, and 10x faster than a fine-tuned safety classifier without gradient updates or fine-tuning. Domain adaptation requires only updating the labeled bank, which can be constructed in under 10 seconds and several orders of magnitude faster than established guardrails. We also analyze the impact of system prompts, layer selection, and integration into production LLM pipelines as a configurable, low-latency guardrail.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05113v1">Rating the Pitch, Not the Product: User Evaluations of LLMs Reflect Expectations More Than Performance</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      Imagine two users interact with the same LLM. One has been told it is the cutting-edge flagship model; the other, an older, weaker model. They walk away with markedly different ratings of its usefulness and intelligence, yet they used the same model. In a controlled study, 162 participants each used one of six LLMs from two families across three collaborative tasks, after first viewing a landing page that matched, overstated, or understated their model's true capability. This pre-interaction framing shifted user opinions and interaction behavior while task performance did not. Oversold users rated the model more favorably and used more directive prompting, while Undersold users wrote longer, more collaborative prompts. The quality of what users and the model produced together depended only on the model's true capability, not on what users were told. Participants' change in model impressions after use, measured across two impression measures, was not predicted by task performance ($β= -0.01$ and $0.11$, both n.s.), but by whether the model met users' expectations ($β= 0.47$ and $0.50$, both $p < .001$) and how confident they felt working with it ($β= 0.47$ and $0.36$, both $p < .001$). After interaction, users are still rating the pitch, not the product: user-elicited LLM evaluations, including the preference data driving public leaderboards, measure expectation management at least as much as the model itself.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.28345v3">Reachability Across the NL/PL Boundary: A Taxonomy-Driven Dataflow Model for LLM-Integrated Applications</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      LLM API calls have become a standard programming primitive, but they create a program boundary that disrupts traditional dataflow analysis. A runtime value may be inserted into a natural-language prompt through a template placeholder, transformed opaquely by the LLM, and returned as code, JSON, or text consumed by downstream logic. Existing analyses such as taint analysis and program slicing require a dataflow summary that describes how a callee maps inputs to outputs; an LLM call provides no such summary, breaking analysis at what we call the NL/PL boundary. We introduce PRISM, the first reachability model for this boundary. PRISM abstracts the missing dataflow summary of an LLM call as placeholder-to-output reachability. Because the LLM's internal transformation is opaque, the only observable signal is the input-output relationship, which spans an unbounded range of behaviors. PRISM therefore uses a finite taxonomy grounded in quantitative information flow theory. It classifies placeholder-output behavior into 25 labels along two dimensions: information preservation and output modality. Each label yields a reachability predicate for a placeholder. The model is sound with respect to its labeling, with residual error bounded empirically. PRISM is dependable and effective. Independent models and human annotators assign its labels consistently (Fleiss' kappa >= 0.72), and the labels cover 8,119 real-world pairs, leaving no pair unclassifiable; the Good-Turing discovery probability is 0.09%. For taint analysis, PRISM nearly doubles the conservative baseline and outperforms a direct LLM baseline, achieving F1 = 81.7%. Across six real OpenClaw CVEs, it detects every vulnerable flow and confirms every patch (F1 = 100%). In backward slicing, it removes about a quarter of irrelevant code without discarding any true dependency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05089v1">TimeThink: Reasoning with Time for Video LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      Video reasoning requires models to identify and verify temporally localized evidence within long video sequences. Recent Video Large Language Models (Video-LLMs) have shown promising reasoning abilities when aligned with reinforcement learning, yet existing approaches typically rely on outcome-based rewards that supervise only the final prediction. Such supervision provides limited guidance on how models should discover the relevant temporal evidence during intermediate reasoning. In this work, we propose TimeThink, a reinforcement learning framework that explicitly guides temporal evidence discovery in Video-LLMs. Our key idea is to treat temporal clue steps as the fundamental optimization primitive of video reasoning, where each reasoning step references a candidate time interval in the video. We introduce a step-wise temporal process reward that provides localized credit assignment for these clues and a joint process--outcome optimization objective that balances reasoning fidelity with task correctness. To enable scalable training, we construct TimeThink-RFT-20K, a dataset with automatically derived temporal evidence segments. Extensive experiments across video reasoning, temporal grounding, and general video understanding benchmarks show that TimeThink consistently improves both temporal localization and reasoning performance, achieving state-of-the-art results among open-source video RL models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05031v1">LLM-Based Test Oracles: Source-of-Authority Taxonomy -- A Systematic Literature Review</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 15 pages, 10 figures, 7 tables. Systematic literature review. Submitted to IEEE Access
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to produce test oracles, the part of a test that decides whether observed behavior is correct. Yet a clear account of where these oracles draw their authority is missing. Prior secondary studies organize the area by oracle form or by LLM technique. None organizes it by the source of the verdict's authority, the property that governs how far a verdict can be trusted. This article presents a systematic literature review, conducted and reported under the PRISMA 2020 guidelines. From 2,436 records, an LLM pre-filter followed by independent dual human screening (reviewer agreement, a Cohen's kappa of 0.79) and full-text assessment yielded 54 included studies. We analyze these along three axes: the source of an oracle's authority, the form it takes, and the mechanism that adjudicates it. We characterize the landscape of domains, languages, models, and adaptation strategies. Specification-derived authority, though the most common single source, covers about half of the studies (28 of 54). The remaining 26 reach a verdict with no specification at all. The source of authority and the adjudication mechanism cross-cut: the same source is checked by several mechanisms and one mechanism serves several sources, so a label such as LLM-as-a-judge names a mechanism rather than a basis for trust. We further report how these oracles are evaluated and how they fail, and read the sparse and empty regions of the taxonomy as a research agenda. The protocol, search query, and per-study coding sheet are released as supplementary material.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05029v1">Your Agent's Memories Are Not Its Own: Forged Reasoning Attacks on LLM Agent Memory and Defenses</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 Preprint. 10 pages, 2 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Persistent memory has enabled large language model (LLM) agents to store factual knowledge, prior decisions, reasoning histories, tool usage information, and context. While this has improved the agent's functionality and continuity across tasks, it has also introduced a new attack surface: the agent's own reasoning history. In this paper, we introduce the Forged Amplifying Rationale Memory Attack (FARMA), which poisons an agent's remembered reasoning rather than its factual knowledge. It inserts forged reasoning traces using evasive language that bypasses keyword-based defenses, then amplifies them through self-referential reinforcement that defeats consensus-based defenses. To address FARMA, we introduce SENTINEL, a layered defense pipeline to detect forged reasoning entries. Its central component is the Reasoning Guard that structurally analyzes candidate entries for forgery using five weighted signals. We evaluate FARMA and SENTINEL across multiple agents and different LLM models with 50 trials and show that FARMA achieves an attack success rate of up to 100% under baseline conditions and is capable of defeating defense mechanisms like keyword filter and A-MemGuard. Our evaluation also shows that SENTINEL reduces FARMA's attack success rate to as low as 0% with no false positives observed across 326 benign agent traces. Our work demonstrates the need to protect not only an agent's retrieved content but also the integrity of its reasoning history.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05013v1">Knowledge Knows, Verbalization Tells: Disentangling Latent Directions for Mathematical Solvability in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 14 pages, 9 Figures
    </div>
    <details class="paper-abstract">
      Although LLMs have made significant progress in mathematical reasoning, determining whether a mathematical problem is solvable remains a fundamental yet challenging capability. While recent studies have probed internal representations of model solvability beliefs, verbalization has primarily been studied behaviorally rather than as an internal representation, limiting its analysis and manipulation. We address this gap by separately probing representations of solvability knowledge and verbalization, allowing us to disentangle the two within model hidden states. Across multiple LLMs, we show that knowledge and verbalization are encoded as distinct, linearly decodable representations and that fabrication is primarily associated with changes in verbalization rather than the underlying knowledge. Prompting with unsolvability cues reduces fabrication primarily by shifting verbalization, while activation steering demonstrates that these representations can be echanistically manipulated to improve model abstention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.04969v1">Train Smarter, Not Longer: Memorization-Guided Data Reuse for Efficient LLM Training</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 Published as a paper at 3rd DATA-FM workshop @ ICLR 2026, Brazil
    </div>
    <details class="paper-abstract">
      The training paradigm of large language models has shifted from traditional one-pass training to multi-epoch training, as reasonable reuse of limited high-quality data can improve both model performance and sample efficiency. Meanwhile, excessive repetition introduces the risk of overfitting and diminishing returns. Determining when and how to reuse data effectively thus emerges as a natural but under-explored question. Through a novel observation of model's "Memorization Window" signals derived from loss retention dynamics and downstream evaluation scores, we propose "Memorization-guided Data Reuse", a training paradigm that adaptively determines when and how data should be reused, enabling principled decisions on the number of training epochs and the scheduling of data replays. Our preliminary experiments reveal a consistent memorization-driven regime: performance continues to improve with repetition far beyond current practice (e.g., the commonly cited four-epoch limit). While a full scheduler remains future work, these insights provide a foundation for memorization-aware training schedules, helping to determine reuse budgets and move toward training LLMs smarter rather than longer with limited high-quality data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.04963v1">STAPO: Selective Trajectory-Aware Policy Optimization for LLM Agent Training</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 ACL 2026 MainConference
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) is the dominant paradigm for training Large Language Model (LLM) agents on long-horizon tasks. However, sparse and delayed rewards often lead to trajectory neglect, in which agents lose focus on the task goal and interaction history at intermediate steps. Prior work has explored step-level supervision using Shannon-entropy-based uncertainty signals, which conflate inherent state complexity with agent confidence and therefore provide unreliable estimates of decision reliability. To address this issue, we propose normalized entropy, which measures confidence deviations relative to an agent's average behavior under a given state, thereby strengthening the association between low-quality actions and trajectory neglect. Building on this insight, we introduce Selective Trajectory-Aware Policy Optimization (STAPO), a hierarchical group-based RL framework. STAPO leverages normalized entropy to locate outlier steps associated with trajectory neglect and optimizes them via a joint mechanism of trajectory-aware reward and trajectory-independent penalty, enhancing trajectory awareness while preserving training stability. Extensive experiments on ALFWorld, WebShop, and Search-Augmented QA demonstrate that STAPO achieves state-of-the-art performance while substantially alleviating trajectory neglect, validating its effectiveness and robustness for agentic tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02070v3">Exploring Plan Space through Conversation: An Agentic Framework for LLM-Mediated Explanations in Planning</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 Preprint; Accepted at EUMAS 2026
    </div>
    <details class="paper-abstract">
      When automating plan generation for a real-world sequential decision problem, the goal is often not to replace the human planner, but to facilitate an iterative reasoning and elicitation process, where the human's role is to guide the AI planner according to their preferences and expertise. In this context, explanations that respond to users' questions are crucial to improve their understanding of potential solutions and increase their trust in the system. To enable natural interaction with such a system, we present a multi-agent Large Language Model (LLM) architecture that is agnostic to the explanation framework and enables user- and context-dependent interactive explanations. We also describe an instantiation of this framework for goal-conflict explanations, which we use to conduct a user study comparing the LLM-powered interaction with a baseline template-based explanation interface.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.04945v1">You Frame It: How Conceptual Representations Shape LLM Detection and Reasoning about Antisemitism</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      LLMs enable the integration of external conceptual resources at inference time, creating new opportunities for detecting ideologically and historically complex phenomena such as antisemitism. We investigate how different forms of conceptual grounding affect antisemitism detection and explanation behavior across four state-of-the-art LLMs. Using two expert-annotated datasets, we compare definitional, fine-grained taxonomic, example-augmented, and large-context representations of antisemitism. We find that fine-grained taxonomic representations substantially improve recall, while simultaneously reducing precision. Surprisingly, supplying substantially larger conceptual resources yields no additional quantitative benefit. Post-Holocaust antisemitism poses the most persistent challenge across models and configurations. Analysis of explanations further reveals systematic limitations including overproduction of conceptual references, reliance on lexical cues, overconfidence, and difficulties with subtle or justificatory forms of antisemitism. Our findings highlight both the potential and the remaining limitations of conceptually grounded LLMs for antisemitism detection and reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.04939v1">Teaching LLMs a Low-Resource Language: Enhancing Code Completion in Pharo</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) unlocked new possibilities in automated code writing, becoming the backbone of most code completion tools. While LLMs excel in mainstream languages, they often lack support for the so-called low-resource languages where training data is scarce. As a result, these languages lag behind in the quality of code completion tooling available to their communities. A concrete example is Pharo, a Smalltalk-inspired language whose IDE currently offers only single-token completion. In this work, we report on our experience bringing LLM-based code completion to Pharo. First, we describe an end-to-end pipeline that combines Pharo-specific data curation, continued pre-training and fine-tuning of open code LLMs. Second, we introduce a set of Pharo code completion benchmarks designed to evaluate whether models (i) learn Pharo's syntax and (ii) accurately complete masked Pharo code from real-world GitHub repositories. Third, we show empirically that Pharo-specialized models substantially outperform their original base checkpoints and also exceed the accuracy of substantially larger code LLMs on Pharo completion. Overall, our case study demonstrates the feasibility of bringing strong LLM-based code completion to low-resource programming languages, with models small enough to provide ``real-time'' in-IDE support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.17331v3">Exploring Context-aware and LLM-driven Locomotion for Immersive Virtual Reality</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 28 pages. To appear in the Proceedings of the ACM on Human-Computer Interaction (PACM HCI), Vol. 10, No. 5; presented at the 28th ACM International Conference on Mobile Human-Computer Interaction (MobileHCI 2026)
    </div>
    <details class="paper-abstract">
      Locomotion plays a crucial role in shaping the user experience within virtual reality environments. In particular, hands-free locomotion offers a valuable alternative by supporting accessibility and freeing users from reliance on handheld controllers. To this end, traditional speech-based methods often depend on rigid command sets, limiting the naturalness and flexibility of interaction. In this study, we propose a novel locomotion technique powered by large language models (LLMs), which allows users to navigate virtual environments using natural language with contextual awareness. We evaluate three locomotion methods: controller-based teleportation, voice-based steering, and our language model-driven approach. Our evaluation combines eye-tracking data analysis, including exploratory explainable machine learning analysis with SHAP, and standardized questionnaires (SUS, IPQ, CSQ-VR, NASA-TLX) to examine user experience through both objective gaze-based measures and subjective self-reports of usability, presence, cybersickness, and cognitive load. Our findings show no statistically significant differences in usability, presence, or cybersickness between LLM-driven locomotion and established methods such as teleportation, suggesting its potential as a viable, natural language-based, hands-free alternative. In addition, eye-tracking analysis revealed patterns suggesting tendency toward increased user attention and engagement in the LLM-driven condition. Complementary to these findings, exploratory SHAP analysis revealed that fixation, saccade, and pupil-related features vary across techniques, indicating distinct patterns of visual attention and cognitive processing. Overall, we state that our method can facilitate hands-free locomotion in virtual spaces, especially in supporting accessibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05475v1">Is Your NPU Ready for LLMs? Dissecting the Hidden Efficiency Bottlenecks in Mobile LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      Deploying Large Language Models (LLMs) on mobile devices enhances privacy and reduces latency, but is severely bottlenecked by hardware inefficiency. We present the first comprehensive, cross-layer measurement study of mobile LLM inference, uniquely spanning five mainstream frameworks (e.g., llama.cpp, GENIE) and three hardware backends (CPU, GPU, NPU). To enable this analysis, we develop PowerBench, a fine-grained profiling tool that provides the first backend-specific energy attribution, moving beyond traditional device-level measurements. Our study yields three critical insights: (1) Framework-induced performance gaps are substantially amplified on NPUs, reaching up to 10x using custom operators due to divergent offloading and quantization strategies. (2) We identify a distinct phase split where NPUs excel at compute-bound prefilling, while CPUs outperform all other backends in memory-bound decoding. This is driven by the NPU's preference for large, fixed-shape workloads, which conflicts with the small-kernel, dynamic nature of decoding. (3) Backend-specific profiling uncovers substantial scheduling headroom missed by prior work. Suboptimal thread configurations, uncoordinated NPU sleep latencies, and CPU polling intervals result in up to 40% energy waste. Leveraging these findings, we present an energy-oriented best-practice configuration for mobile LLM inference. We estimate that this configuration could reduce energy consumption by up to 54.8% on the NPU backend across three datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.04854v1">CARL: Constraint-Aware Reinforcement Learning for Planning with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 ACL 2026 Findings
    </div>
    <details class="paper-abstract">
      Despite their strong reasoning capabilities and extensive world knowledge, Large Language Models (LLMs) frequently generate plans that violate task constraints, undermining their reliability in real-world applications. This deficiency arises from a lack of systematic mechanisms to incorporate constraint information during the generation process. While existing approaches attempt to mitigate this by relying on external tools or task decomposition, they fail to enhance the model's intrinsic constraint awareness. To address this, we propose Constraint-Aware Reinforcement Learning (CARL), a novel RL framework designed to strengthen LLMs' intrinsic focus on constraints. CARL introduces a constraint-aware reward by comparing the model's output distributions under constrained and unconstrained inputs, encouraging constraint focus and penalizing neglect. Compatible with various RL frameworks and requiring no external solvers or top models, CARL enables scalable, end-to-end constraint-aware planning. Extensive experiments on BlocksWorld, TravelPlanner, and T-Eval demonstrate that CARL significantly outperforms standard Reinforcement Fine-Tuning (RFT) baselines and state-of-the-art reasoning models, exhibiting a markedly increased focus on constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.16337v3">Medical Heuristic Learning: An LLM-Driven Framework for Interpretable and Auditable Clinical Decision Rules</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      Predictive modeling for clinical decision support requires not only strong predictive performance but also transparent decision logic. Although deep learning and tree-based ensemble methods can achieve high accuracy, their black-box nature remains a major obstacle to clinical deployment. This challenge is further compounded by common characteristics of medical data, including limited sample sizes, severe class imbalance, and feature evolution arising from changes in diagnostic criteria and clinical documentation. To address these issues, we propose Medical Heuristic Learning (MHL), an instantiation of the learning beyond gradients paradigm for clinical prediction from structured medical data. Instead of relying on neural network weight updates, MHL uses a large language model (LLM) driven workflow that integrates statistical probes, medical knowledge probes, rule synthesis, and code-level iterative refinement to optimize a deterministic and executable rule-based expert system. The resulting model is expressed not as opaque parameters, but as versioned pure Python decision rules that are explicitly interpretable, fully auditable, and clinically grounded. MHL also supports continual learning by starting from previously validated rules and iteratively revising them using updated feature information under data drift or feature evolution. Comprehensive experiments on medical datasets show that MHL achieves performance comparable to state-of-the-art methods while maintaining strong behavior in small-sample and highly imbalanced settings. The results further indicate that this explicit rule-update mechanism can help alleviate catastrophic forgetting under feature evolution. Overall, these findings suggest that non-gradient-based heuristic systems offer a transparent and adaptable alternative for high-stakes clinical decision support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.23270v2">CAP-CoT: Cycle Adversarial Prompt for Improving Chain of Thoughts in LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) prompting has emerged as a simple and effective way to elicit step-by-step solutions from large language models (LLMs). However, CoT reasoning can be unstable across runs on long, multi-step problems, leading to inconsistent answers for unchanged task. Most prior work focuses on improving the forward reasoning chain within a single pass, with less attention to iterative and contrastive correction. To address this gap, we propose CAP-CoT, a Cycle Adversarial Prompt optimization framework designed to improve both CoT reasoning accuracy and stability of a single deployed solver. In each cycle, a forward solver generates candidate reasoning chains, an adversarial challenger constructs plausible but deliberately flawed chains using targeted error strategies, and a feedback agent contrasts the two chains and produces step-aligned structured feedback. This feedback closes the optimization loop in two directions, including updating the solver prompt based on errors exposed by the challenger, and updating the challenger prompt to generate increasingly targeted errors in subsequent cycles. Unlike safety-oriented adversarial prompting such as jailbreak or prompt-injection attacks, our adversarial component is task-semantic and aims to expose logical vulnerabilities in reasoning chains. Experiments across six benchmarks and four LLM backbones demonstrate that within two to three adversarial prompt optimization cycles, CAP-CoT consistently reduces variability across runs while improving reasoning accuracy and robustness to prompt perturbations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.04728v1">Turning Off-Policy Tokens On-Policy: A Plug-in Approach for Improving LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) post-training for large language models (LLMs) follows a efficient paradigm of "rollout then update", which inevitably results in off-policy training data. To resolve this, Importance sampling (IS) is proposed, while the token-level ratios compound over long sequences, causing severe variance exploded. A natural idea is "transferring" these off-policy token into on-policy token, so that the importance scores for correction are unnecessary. Following this idea, we propose Selective Importance Sampling (SIS), which is inspired by rejection sampling. Concretely, SIS implements by viewing off-policy model as proposal distribution, and implement a token-level rejection test: accepted tokens are viewed as on-policy, so that receive unit importance score, while rejected tokens retain the standard IS correction. Our proposed SIS is theoretically proved reducing the gap between token-level and sequence-level off-policy gradient estimators. The SIS acts as a plug-in that only modifies the importance ratio in the policy loss, adding negligible wall-clock overhead, and can be combine with a vast vary of RL post-training algorithms. Experiments on dense and MoE LLMs across math and agent benchmarks show that SIS consistently improves all objectives, while providing substantially stronger robustness under off-policy data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.23847v5">Seven Security Challenges in Cross-domain Multi-agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are rapidly evolving into autonomous agents that cooperate across organizational boundaries, enabling joint disaster response, supply-chain optimization, and other tasks that demand decentralized expertise without surrendering data ownership. Yet, cross-domain collaboration shatters the unified trust assumptions behind current alignment and containment techniques. An agent benign in isolation may, when receiving messages from an untrusted peer, leak secrets or violate policy, producing risks driven by emergent multi-agent dynamics rather than classical software bugs. This position paper maps the security agenda for cross-domain multi-agent LLM systems. We introduce seven categories of novel security challenges, for each of which we also present plausible attacks, security evaluation metrics, and future research guidelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.04713v1">RSPO: Reward-Swap Policy Optimization for Multi-Turn LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      Reinforcement learning holds significant potential for training large language models (LLMs) to handle multi-turn interactive tasks. However, in long-horizon, multi-turn tasks characterized by sparse outcome rewards, directly training with outcome rewards often results in slow convergence due to the sparsity of signals and the lack of fine-grained feedback. Furthermore, the model may fail to learn successful trajectories that are not sampled during training, thereby limiting its performance. Conversely, while employing customized dense process rewards provides richer signals and accelerates convergence, these surrogate rewards may exhibit potential misalignment with the ground-truth outcome rewards. This inconsistency can bias the training direction and ultimately degrade the model's final performance. In this work, we propose Reward-Swap Policy Optimization (RSPO), a method designed to leverage the rich information from dense process rewards to facilitate training with outcome rewards. By utilizing a reward-swap mechanism, RSPO ensures the diversity of sampled trajectories while guaranteeing consistency between the optimization objective and the true outcome rewards, thereby elevating the performance ceiling of the model. We conduct extensive experiments on two challenging agent benchmarks, WebShop and ALFWorld. By applying our method to various reinforcement learning algorithms, including GRPO, PPO, and GiGPO, we demonstrate that RSPO achieves consistent performance improvements across different baselines and benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.10031v2">Context Misleads LLMs: The Role of Context Filtering in Maintaining Safe Alignment of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 17 pages, 3 figures
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have shown significant advancements in performance, various jailbreak attacks have posed growing safety and ethical risks. Malicious users often exploit adversarial context to deceive LLMs, prompting them to generate responses to harmful queries. In this study, we propose a new defense mechanism called Context Filtering, an input pre-processing method designed to filter out untrustworthy and unreliable context while identifying the primary prompts containing the real user intent to uncover concealed malicious intent. Given that enhancing the safety of LLMs often compromises their helpfulness, potentially affecting the experience of benign users, our method aims to improve the safety of the LLMs while preserving their original performance. We evaluate the effectiveness of our model in defending against jailbreak attacks through comparative analysis, comparing our approach with state-of-the-art defense mechanisms against six different attacks and assessing the helpfulness of LLMs under these defenses. Our model demonstrates its ability to reduce the Attack Success Rates of jailbreak attacks by up to 92% while maintaining the original LLMs' performance, achieving state-of-the-art Safety and Helpfulness balance. Notably, Context Filtering is a plug-and-play method that can be applied to all LLMs, including both white-box and black-box models, to enhance their safety without requiring any fine-tuning of the models themselves. Our model is available for research purposes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.04686v1">ToolFailBench: Diagnosing Tool-Use Failures in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 18 pages, 3 figures. Published at the Workshop on Agents in the Wild: Safety, Security, and Beyond (AIWILD) and the Workshop on Failure Modes of Agentic AI (FAGEN) at ICML 2026
    </div>
    <details class="paper-abstract">
      Tool calling is central to modern language model agents, but aggregate benchmark scores often hide where tool use fails. A model that never calls a needed tool and a model that calls the tool but ignores the result can look similar under final task accuracy. We introduce ToolFailBench, a diagnostic benchmark for measuring tool-use failures across 1,000 tasks in finance, medicine, law, cybersecurity, and real estate. Tool-required tasks return values the model wouldn't guess, forcing it to trust the tool while control tasks attach the same tools but should be answered directly. We label each trace with Tool-Skip, Result-Ignore, Output-Fabrication, and Unnecessary-Tool-Use, using a rule classifier and two LLM judges aggregated by majority vote. Across 19 headline models, the best reaches 86.33% Clean Tool-Use Rate, showing that faithful tool use is not saturated. More importantly, models with similar aggregate scores fail in different ways: most stay disciplined on no-tool controls, while Llama-3.1 models show an Always-Call pattern, and at the same parameter scale Llama-3.1-70B and Qwen2.5-72B differ by 89 percentage points on control-task accuracy. Tool-use evaluation should measure not only whether agents call tools, but whether they use tool outputs correctly and avoid tools when none is needed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.13742v4">Spatial Balancing: Designing an LLM-Powered Spatial Externalization Interface for Iterative Science Communication Writing</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 DIS '26
    </div>
    <details class="paper-abstract">
      Science communication revision requires writers to dynamically balance scientific exposition and narrative engagement - a process where writers often struggle with competing directions. Existing LLM-assisted tools help with co-writing, but offer limited support for navigating this iterative, multi-directional revision process. To address this gap, we designed Spatial Balancing, an exploratory revision environment that maps rhetorical goals and revision strategies onto a two-dimensional spatial canvas for experienced science communication creators with domain expertise but lacking formal professional training. By building a design space of communication strategies and embedding them into a spatial exploratory canvas, our system treats feedback as navigational cues rather than prescriptive judgments. Our findings show that this integrated revision environment helps writers stay focused on writing goals, reason about revision as trajectories, and explore alternatives, which supports greater metacognitive control and confidence without increasing workload. This work highlights the value of spatially externalized revision environments for supporting iterative, reflective thinking during LLM-assisted writing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.04668v1">Elastic Gang: Per-Token Membership Change for a Hard-Barriered LLM Inference Gang Co-Scheduled with OS Processes</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 14 pages, 1 figure, 6 tables
    </div>
    <details class="paper-abstract">
      On-device LLM decoding is a hard-barriered CPU-SIMD computation that wants every core for milliseconds per token, while the rest of the OS wants those same cores continuously. A barriered gang cannot simply be dropped into a preemptive scheduler: an unannounced departure deadlocks a barrier, and an unannounced arrival silently corrupts logits. I present the elastic gang of Anima OS, a bare-metal x86-64 Rust kernel in which the inference gang is a first-class schedulable entity whose core membership may change between any two tokens. The core mechanism is an ACK-latched epoch protocol that never waits on a named core: a seqlock-style generation-tagged latch composed with RCU/epoch-style membership consent, so each token's participant set is the intersection of the cores the gang requested and the cores that acked the current epoch. An un-acked core is outside this token and joins at most one token later. Displaced general processes migrate and keep running; cores return to them the moment a generation ends. On a real AMD Zen 5 machine (8C/16T), inference output is bit-exact under verified per-token membership change on both a 135M and a 7B model, the property that makes elasticity safe in a kernel whose safety gate reads logits. Against fair static core partitions, elastic membership Pareto-dominates: at intermediate inference duty cycles it delivers 1.75x (25%), 1.52x (50%), and 1.28x (75%) the general throughput of a static 8-core split at equal or better inference throughput, recovers all eight stranded cores when inference is idle, and converges to the split at saturation. Returning a lent core costs 0.22 us (p50); acquiring a busy, tenant-occupied core costs one scheduling quantum (~16 ms): a running tenant is never preempted mid-slice. Decode throughput saturates at gang width 8, so ceding cores past the knee is nearly free: elasticity auto-sizes the gang online.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.11943v3">ProbeLogits: Kernel-Level LLM Inference Primitives for AI-Native Operating Systems</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 18 pages, 13 tables
    </div>
    <details class="paper-abstract">
      An OS kernel that runs LLM inference internally can read the model's own next-token logit distribution before any text is generated, and act on it as a governance primitive. I present ProbeLogits, a kernel-level operation that performs a single forward pass and reads specific token logits to classify an agent's action as safe or dangerous, with zero learned parameters. Because the probe reads a logit from the same base model the agent already runs, it removes the second model a fine-tuned guard requires: the marginal cost of a safety check becomes a single logit read. I evaluate ProbeLogits on three base models (Qwen2.5-7B, Llama-3-8B, Mistral-7B) across three external benchmarks (HarmBench, XSTest, ToxicChat). On HarmBench non-copyright, all three reach a 97-99% block rate. On ToxicChat (n=1,000), ProbeLogits attains F1 parity-or-better against Llama Guard 3: Qwen2.5-7B Safe/Dangerous reaches F1=0.812 (+13.7 pp, bootstrap 95% CIs disjoint), Llama-3 matches within CI (+0.4 pp), and Mistral exceeds by +4.4 pp. Classification is a measured 2.4-3.4x faster than Llama Guard 3 (332-556 ms vs. 851-1,142 ms), because it reads a single logit position instead of generating tokens. A calibration strength alpha acts as a deployment-time policy knob rather than a learned hyperparameter, trading recall for precision per operation class. I implement ProbeLogits within Anima OS, a bare-metal x86-64 kernel written in ~285,000 lines of Rust. Because agent actions must pass through kernel-mediated host functions, enforcement operates below the WASM sandbox boundary, making it substantially harder to circumvent than application-layer classifiers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.13517v2">Think Deep, Not Just Long: Measuring LLM Reasoning Effort via Deep-Thinking Tokens</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 Accepted to ICML 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive reasoning capabilities by scaling test-time compute via long Chain-of-Thought (CoT). However, recent findings suggest that raw token counts are unreliable proxies for reasoning quality: increased generation length does not consistently correlate with accuracy and may instead signal "overthinking," leading to performance degradation. In this work, we quantify inference-time effort by identifying deep-thinking tokens -- tokens where internal predictions undergo significant revisions in deeper model layers prior to convergence. Across four challenging mathematical and scientific benchmarks (AIME 24/25, HMMT 25, and GPQA-diamond) and a diverse set of reasoning-focused models (GPT-OSS, DeepSeek-R1, and Qwen3), we show that deep-thinking ratio (the proportion of deep-thinking tokens in a generated sequence) exhibits a robust and consistently positive correlation with accuracy, substantially outperforming both length-based and confidence-based baselines. Leveraging this insight, we introduce Think@n, a test-time scaling strategy that prioritizes samples with high deep-thinking ratios. We demonstrate that Think@n matches or exceeds standard self-consistency performance while significantly reducing inference costs by enabling the early rejection of unpromising generations based on short prefixes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.08579v3">LLM-based Human Simulations Have Not Yet Been Reliable</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly employed for simulating human behaviors across diverse domains. However, our position is that current LLM-based human simulations remain insufficiently reliable, as evidenced by significant discrepancies between their outcomes and authentic human actions. Our investigation begins with a systematic review of LLM-based human simulations in social, economic, policy, and psychological contexts, identifying their common frameworks, recent advances, and persistent limitations. This review reveals that such discrepancies primarily stem from inherent limitations of LLMs and flaws in simulation design, both of which are examined in detail. Building on these insights, we propose a systematic solution framework that emphasizes enriching data foundations, advancing LLM capabilities, and ensuring robust simulation design to enhance reliability. Finally, we introduce a structured algorithm that operationalizes the proposed framework, aiming to guide credible and human-aligned LLM-based simulations. To facilitate further research, we provide a curated list of related literature and resources at https://github.com/Persdre/awesome-llm-human-simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.04623v1">Can LLMs Really Recover Microservice Failures? A Recovery-Aware Evaluation of Diagnosis-to-Action Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to interpret operational evidence and assist incident response in cloud-native microservice systems. However, recovery-oriented use cases require more than identifying a root cause. After observing symptoms and diagnosing a fault, an operator or agent must translate the diagnosis into a concrete recovery action, apply it to an admissible target, and verify that service health has been restored. Existing RCA and log-analysis evaluations are well-suited to diagnosis, but they do not characterize this subsequent action decision. This paper presents R2Act, a recovery-action evaluation framework for post-diagnosis incident response. R2Act defines an incident schema, quality gate, action-space representation, recovery-validity metrics, offline evaluator, and live-replay protocol. We instantiate the framework as a benchmark dataset of 302 quality-audited Kubernetes incidents from \system. Each incident provides synchronized multi-modal observations, root-cause labels, an incident-specific action space, and annotated valid and invalid recovery plans. We evaluate heuristic, supervised, RCA-oriented, deep log, and LLM-based methods. The strongest RAG-based LLMs reach 91.4\%--99.7\% root-cause service accuracy, yet their recovery validity remains only 36.8\%--60.3\%. Even when both the root-cause service and fault type are correct, recovery-oriented methods still choose invalid actions for 39.5\%--62.0\% of correctly diagnosed incidents. Overall, this work reveals that many recovery failures arise not from missing diagnostic knowledge, but from the difficulty of translating diagnostic evidence into valid recovery actions and admissible targets. This work provides a reproducible, simplified starting point for research and evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.20995v2">Generative Pseudo-Labeling for Pre-Ranking with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      Pre-ranking is a critical stage in industrial recommendation systems, tasked with efficiently scoring thousands of recalled items for downstream ranking. A key challenge is the train-serving discrepancy: pre-ranking models are trained only on exposed interactions, yet must score all recalled candidates -- including unexposed items -- during online serving. This mismatch not only induces severe sample selection bias but also degrades generalization, especially for long-tail content. Existing debiasing approaches typically rely on heuristics (e.g., negative sampling) or distillation from biased rankers, which either mislabel plausible unexposed items as negatives or propagate exposure bias into pseudo-labels. In this work, we propose Generative Pseudo-Labeling (GPL), a framework that leverages large language models (LLMs) to generate unbiased, content-aware pseudo-labels for unexposed items, explicitly aligning the training distribution with the online serving space. By offline generating user-specific interest anchors and matching them with candidates in a frozen semantic space, GPL provides high-quality supervision without adding online latency. Deployed in a large-scale production system, GPL improves click-through rate by 3.07%, while significantly enhancing recommendation diversity and long-tail item discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.26306v5">Interactive Learning for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 The code is available at https://github.com/linhh29/Interactive-Learning-for-LLM-Reasoning
    </div>
    <details class="paper-abstract">
      Existing multi-agent learning approaches have developed interactive training environments to explicitly promote collaboration among multiple Large Language Models (LLMs), thereby constructing stronger multi-agent systems (MAS). However, during inference, they require re-executing the MAS to obtain final solutions, which diverges from human cognition that individuals can enhance their reasoning capabilities through interactions with others and resolve questions independently in the future. To investigate whether multi-agent interaction can enhance LLMs' independent problem-solving ability, we introduce ILR, a novel co-learning framework for MAS that integrates two key components: Dynamic Interaction and Perception Calibration. Specifically, Dynamic Interaction first adaptively selects either cooperative or competitive strategies depending on question difficulty and model ability. LLMs then exchange information through Idea3, an innovative interaction paradigm designed to mimic human discussion, before deriving their respective final answers. In Perception Calibration, ILR employs Group Relative Policy Optimization (GRPO) to train LLMs while integrating one LLM's reward distribution characteristics into another's reward function, thereby enhancing the cohesion of multi-agent interactions. We evaluate the effectiveness of ILR across three LLMs from two model families of varying scales on five mathematical, one coding, one general question answering, and one scientific reasoning benchmarks. Experimental results show that ILR consistently outperforms single-agent learning, yielding an improvement of up to 5% over the strongest baseline. We further discover that Idea3 can enhance the robustness of stronger LLMs during multi-agent inference, and dynamic interaction types can boost multi-agent learning compared to pure cooperative or competitive strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.11878v5">LLMs Encode Harmfulness and Refusal Separately</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      LLMs are trained to refuse harmful instructions, but do they truly understand harmfulness beyond just refusing? Prior work has shown that LLMs' refusal behaviors can be mediated by a one-dimensional subspace, i.e., a refusal direction. In this work, we identify a new dimension to analyze safety mechanisms in LLMs, i.e., harmfulness, which is encoded internally as a separate concept from refusal. There exists a harmfulness direction that is distinct from the refusal direction. As causal evidence, steering along the harmfulness direction can lead LLMs to interpret harmless instructions as harmful, but steering along the refusal direction tends to elicit refusal responses directly without reversing the model's judgment on harmfulness. Furthermore, using our identified harmfulness concept, we find that certain jailbreak methods work by reducing the refusal signals without reversing the model's internal belief of harmfulness. We also find that adversarially finetuning models to accept harmful instructions has minimal impact on the model's internal belief of harmfulness. These insights lead to a practical safety application: The model's latent harmfulness representation can serve as an intrinsic safeguard (Latent Guard) for detecting unsafe inputs and reducing over-refusals that is robust to finetuning attacks. For instance, our Latent Guard achieves performance comparable to or better than Llama Guard 3 8B, a dedicated finetuned safeguard model, across different jailbreak methods. Our findings suggest that LLMs' internal understanding of harmfulness is more robust than their refusal decision to diverse input instructions, offering a new perspective to study AI safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.04582v1">Finetuning Lightweight LLMs for Control Flow Graph Generation</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 Accept by The 38th International Conference on Software Engineering & Knowledge Engineering Short Paper
    </div>
    <details class="paper-abstract">
      Control Flow Graph (CFG) is an important program representations for software analysis, code understanding, and software maintenance. Traditional CFG generation techniques mainly rely on bytecode or abstract syntax trees. However, these approaches usually require complete, compilable, and syntax error-free code, which limits their applicability to incomplete or erroneous code. Furthermore, they often depend on language specific tools, making it difficult to support multiple programming languages in a unified manner. To address these limitations, this paper investigates the use of fine-tuned lightweight large language models (LLMs) for CFG generation. We first design a unified CFG output format and a task-specific fine-tuning prompt for CFG generation. Then, we construct a dataset based on an existing LeetCode dataset through automatic CFG generation and error augmentation. We evaluate the proposed approach on six lightweight LLM models, including three code-specific LLMs: CodeLlama, QwenCoder, and DeepSeekCoder; and three general purpose LLMs: Llama3.2-3B, Qwen-4B, and Phi-4B. The experimental results show that, through fine-tuning, lightweight LLMs achieve promising results for CFG generation, particularly when the input code is incomplete or erroneous. It also demonstrates cross-language generalization capability on programming language not included in the fine-tuning data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.04579v1">LLM-Driven CI-CD Workflow Intelligence for Cyber Systems Engineering</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      CI/CD workflows have become executable operational policy: they decide what gets built, tested, released, and deployed, and they mediate how maintainers interact with delivery infrastructure. That makes them an important measurement point for cyber-systems engineering. Recent large language model (LLM) work shows that workflow stages can be recognized directly from configuration files, but stage labels alone do not tell us whether a workflow is brittle, unusual for its ecosystem, or worth revising first. We present an LLM-based CI/CD analysis pipeline that combines repository enrichment, anti-pattern detection, stage mining, and recommendation generation over a large GitHub corpus. Starting from 59,550 repositories with at least 1,000 stars, we identify 34,225 projects with CI/CD and collect 127,559 configuration files. Across 75,201 analyzed workflows, the anti-pattern detector reports 434,769 findings, dominated by reliability and maintainability issues. Across 59,906 configurations, stage usage differs significantly by language ($χ^2 = 4168.88$, $p < 0.001$, Cramer's $V = 0.063$), and domain analysis shows distinct operational profiles, including higher release and cache usage in mobile projects. For repository-level recommendation generation, few-shot prompting performs best overall, averaging 8.25 recommendations per repository with 96.1% YAML-valid snippets. Taken together, the results argue for CI/CD observability that combines diagnosis, context, and human review rather than treating workflow mining as a stage-classification problem alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.04576v1">Progressive Disclosure for LLM-Maintained Wiki Knowledge Bases: a Preregistered Ablation</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 14 pages, 2 figures, 6 tables. Preregistered on OSF (https://osf.io/feka7, DOI 10.17605/OSF.IO/FEKA7). Materials-availability and deviations described in the paper
    </div>
    <details class="paper-abstract">
      LLM agents increasingly answer questions against knowledge bases they help maintain. A common intuition holds that progressive disclosure, a compact catalog plus a one-line summary per page so the agent loads only what it needs, should make this cheaper than consulting a large monolithic index. We test that on a real 709-page markdown wiki maintained by an LLM. We retrofit it for progressive disclosure and run a preregistered ablation in which four versions of the corpus differ only in how the agent reaches the content: page bodies are byte-identical across arms, frozen as immutable git tags, so any measured difference is due to access structure alone. We cross the arms with three access conditions (a protocol-constrained agent, a free self-routing agent, and a catalog-preload regime) and grade answers blind against verified gold references with a cross-family judge. A pilot upended the premise: a capable tool-using agent never loads the index, inferring a page's path from the question and reading it directly, so the specific saving the retrofit targets does not materialize. We therefore made answer quality primary and cost secondary. Quality is non-inferior (the retrieval arm matches the index baseline within the preregistered margin) while cost falls in every regime, from about a third for a self-routing agent to well over half under catalog-preload, all confidence intervals excluding zero. The saving comes not from avoiding the index load but from more targeted access: the retrieval arm cites fewer pages and takes fewer tool turns. The study doubles as a case study in evaluation validity, applying threat-to-validity discipline to the tooling that produced it.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.22063v2">RedCoder: Automated Multi-Turn Red Teaming for Code LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 ACL 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) for code generation (i.e., Code LLMs) have demonstrated impressive capabilities in AI-assisted software development and testing. However, recent studies have shown that these models are prone to generating vulnerable or even malicious code under adversarial settings. Existing red-teaming approaches rely on extensive human effort, limiting their scalability and practicality, and generally overlook the interactive nature of real-world AI-assisted programming, which often unfolds over multiple turns. To bridge these gaps, we present RedCoder, a red-teaming agent that engages victim models in multi-turn conversation to elicit vulnerable code. The pipeline to construct RedCoder begins with a multi-agent gaming process that simulates adversarial interactions, yielding a set of prototype conversations and an arsenal of reusable attack strategies. We then fine-tune an LLM on these prototype conversations to serve as the backbone of RedCoder. Once deployed, RedCoder autonomously engages Code LLMs in multi-turn conversations, dynamically retrieving relevant strategies from the arsenal to steer the dialogue toward vulnerability-inducing outputs. Experiments across multiple Code LLMs show that our approach outperforms prior single-turn and multi-turn red-team methods in inducing vulnerabilities in code generation, offering a scalable and effective tool for evaluating the security boundaries of modern code-generation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.02464v1">Will Scaling Improve Social Simulation with LLMs?</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) social simulations are a promising research method, but they are not yet faithful enough to be adopted widely. In this work, we investigate whether the current scaling paradigm in language modeling is likely to close these gaps, or whether simulation fidelity is orthogonal to general capabilities and therefore deserving of more research attention. We use scaling laws to study the relationship between LLMs' compute scale, general capability benchmarks, and the fidelity of social simulation in three representative sub-domains: opinion modeling, behavioral simulation, and longitudinal forecasting. Surprisingly, we discover strong compute scaling in all three settings, using a suite of 85 transformer LLMs with the Qwen3 architecture pre-trained on the DCLM web text corpus under fixed-compute budgets from $10^{18}$ to $10^{20}$ FLOPs. Then we evaluate 35 larger and more capable open-weight models up to 70B parameters, allowing us to predict downstream accuracy from loss. This reveals that the majority of behavioral and opinion simulation tasks will rapidly improve with scale, particularly when they involve populations that are well-represented in English web corpora. Longitudinal forecasting and underrepresented opinions scale more slowly, especially when they are less correlated with general knowledge and reasoning benchmarks like MMLU. In behavior simulation, scaling fails to improve model calibration with human cognitive biases like risk aversion, as well as human heuristics like learning correlated rewards from related tasks. On these tasks, even fine-tuned models fail to noticeably scale up performance from 0.5B to 8B parameters. Taken together, we conclude that scale will improve social simulations in most settings, but outliers exist, and improvements will be less reliable in low-resource domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.02460v1">Neuron-Aware Data Selection for Annotation-Free LLM Self-Distillation</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      Post-training large language models (LLMs) without real-world interaction feedback or human-labeled supervision remains challenging, particularly in specialized domains where expert annotations are costly to obtain. Recent annotation-free self-evolution methods address this by using the model's own outputs as supervision signals, constructing a teacher via additional context and aggregating predictions across multiple rollouts through majority voting to produce pseudo-labels. However, these approaches are not without drawbacks: SFT- and GRPO-based variants suffer out-of-domain performance degradation, while reward-based on-policy RL inflates calibration error. In this paper, we propose Neuron On-Policy Self-Distillation (Neuron-OPSD), a data-centric framework for annotation-free self-distillation that leverages internal neuron activations to guide both training-data selection and teacher context construction. The model is then trained via on-policy distillation from the teacher distribution, requiring no ground-truth labels at any stage. Across specialized-domain benchmarks, Neuron-OPSD improves in-domain task performance while preserving cross-domain generalization and mitigating calibration collapse over prior annotation-free baselines. This framework is particularly relevant to settings where online interaction or external supervision is costly or infeasible, and is conceptually distinct from offline RL approaches that rely on logged, reward-labeled trajectories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.02455v1">When Do LLM Personas Support Visualization Design? A Cross-Model Study of Color Assignment and Chart Choice</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 5 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large language model personas are increasingly used to approximate diverse users during early-stage visualization design, but it remains unclear whether persona-conditioned outputs reflect stable personality effects or artifacts of model choice and task framing. We examine this question across two visualization-relevant tasks: color assignment for abstract and concrete concepts, and chart-idiom preference ratings across task contexts. Using 43 Big Five profiles across GPT-4o-mini, GPT-4.1-mini, and GPT-5-mini, we find that personality-color coupling is highly model-configuration dependent: absent in GPT-4o-mini for all six concepts, consistent in GPT-4.1-mini across all six, and partial in GPT-5-mini for two of six. Concept type further shapes the signal: for abstract concepts, personality explains more hue variance than model identity, while concrete concepts show smaller and comparable effects. In chart choice, trait-aligned cluster aggregation produces stable top-idiom rankings across all nine cluster-context combinations, but a no-persona baseline recovers the same top choice in 8 of 9 model-context cells, indicating that task context drives rank-1 selection more than personality. These findings position LLM personas as exploratory probes for visualization design, not substitutes for human participants, and motivate multi-model testing, concept-type disaggregation, and no-persona baselines in future studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.02448v1">AgentsCAD: Automated Design for Manufacturing of FDM Parts via Multi-Agent LLM Reasoning and Geometric Feature Recognition</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      Parts manufactured with Fused Deposition Modeling (FDM) often require Design for Additive Manufacturing (DFAM) modifications to ensure printability, structural integrity, and reduced post-processing. Current slicers identify defects such as steep overhangs but are unable to modify the underlying geometry. This work presents AgentsCAD, a multi-agent system that bridges raw boundary-representation (B-Rep) geometry and Large Language Model (LLM) reasoning to automate targeted DFM. The workflow begins by parsing a STEP file. The agentic system detects overhangs above a 45°threshold, constructs a face-adjacency topology graph, and optionally injects semantic feature labels from a GraphSAGE model trained on MFCAD++ (59,665 parts), before dispatching a Claude Sonnet design-reasoning agent that recommends reorientations, fillets, chamfers, and similar modifications. A GPT-4o vision-language verifier inspects rendered views to confirm geometric integrity. Outputs include a modified STEP file and a human-readable report. A test case on a birdhouse model demonstrates that the system correctly diagnoses overhangs, selects appropriate defect mitigation strategies, and proposes physically valid corrections, partially solving the geometry-to-language translation problem central to LLM-driven CAD modification.
    </details>
</div>
