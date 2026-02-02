# llm - 2026_01

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)
- [Part 16](papers_16.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10840v2">Tracing Multilingual Representations in LLMs with Cross-Layer Transcoders</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 42 pages, 43 figures, under review. Extensive supplementary materials. Code and models available at https://huggingface.co/collections/CausalNLP/multilingual-tinystories-6862b6562414eb84d183f82a and https://huggingface.co/collections/CausalNLP/multilingual-gpt2-models and https://huggingface.co/collections/CausalNLP/multilingual-clts and https://github.com/abirharrasse/MultilingualCLTs
    </div>
    <details class="paper-abstract">
      Multilingual Large Language Models (LLMs) can process many languages, yet how they internally represent this diversity remains unclear. Do they form shared multilingual representations with language-specific decoding, and if so, why does performance favor the dominant training language? To address this, we train models on different multilingual mixtures and analyze their internal mechanisms using Cross-Layer Transcoders (CLTs) and Attribution Graphs. Our results reveal multilingual shared representations: the model employs highly similar features across languages, while language-specific decoding emerges in later layers. Training models without English shows identical multilingual shared space structures. Decoding relies partly on a small set of high-frequency features in the final layers, which linearly encode language identity from early layers. Intervening on these features allows one language to be suppressed and another substituted. Finally, to explain non-English failures, we perform a Model-Diffing experiment: underperformance arises from dim late-layer features, weak middle-layer clusters, and tokenizer bias toward English that forces early layers to specialize in word reassembly. Finetuning strengthens these features and their links, improving token assembly and language-specific decoding, providing a mechanistic explanation for multilingual gaps. Our models and CLTs are available at https://huggingface.co/collections/CausalNLP/multilingual-clts and https://huggingface.co/collections/CausalNLP/multilingual-gpt2-models. Our code is available at: https://github.com/abirharrasse/MultilingualCLTs
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.12260v5">LightRetriever: A LLM-based Text Retrieval Architecture with Extremely Faster Query Inference</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 Accepted by ICLR 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs)-based text retrieval retrieves documents relevant to search queries based on vector similarities. Documents are pre-encoded offline, while queries arrive in real-time, necessitating an efficient online query encoder. Although LLMs significantly enhance retrieval capabilities, serving deeply parameterized LLMs slows down query inference throughput and increases demands for online deployment resources. In this paper, we propose LightRetriever, a novel LLM-based retriever with extremely lightweight query encoders. Our method retains a full-sized LLM for document encoding, but reduces the workload of query encoding to no more than an embedding lookup. Compared to serving a full LLM on an A800 GPU, our method achieves over 1000x speedup in query encoding and over 10x increase in end-to-end retrieval throughput. Extensive experiments on large-scale retrieval benchmarks show that LightRetriever generalizes well across diverse tasks, maintaining an average of 95% retrieval performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17768v2">LLM-42: Enabling Determinism in LLM Inference with Verified Speculation</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 https://github.com/microsoft/llm-42
    </div>
    <details class="paper-abstract">
      In LLM inference, the same prompt may yield different outputs across different runs. At the system level, this non-determinism arises from floating-point non-associativity combined with dynamic batching and GPU kernels whose reduction orders vary with batch size. A straightforward way to eliminate non-determinism is to disable dynamic batching during inference, but doing so severely degrades throughput. Another approach is to make kernels batch-invariant; however, this tightly couples determinism to kernel design, requiring new implementations. This coupling also imposes fixed runtime overheads, regardless of how much of the workload actually requires determinism. Inspired by ideas from speculative decoding, we present LLM-42, a scheduling-based approach to enable determinism in LLM inference. Our key observation is that if a sequence is in a consistent state, the next emitted token is likely to be consistent even with dynamic batching. Moreover, most GPU kernels use shape-consistent reductions. Leveraging these insights, LLM-42 decodes tokens using a non-deterministic fast path and enforces determinism via a lightweight verify-rollback loop. The verifier replays candidate tokens under a fixed-shape reduction schedule, commits those that are guaranteed to be consistent across runs, and rolls back those violating determinism. LLM-42 mostly re-uses existing kernels unchanged and incurs overhead only in proportion to the traffic that requires determinism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.23183v1">JobResQA: A Benchmark for LLM Machine Reading Comprehension on Multilingual Résumés and JDs</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      We introduce JobResQA, a multilingual Question Answering benchmark for evaluating Machine Reading Comprehension (MRC) capabilities of LLMs on HR-specific tasks involving résumés and job descriptions. The dataset comprises 581 QA pairs across 105 synthetic résumé-job description pairs in five languages (English, Spanish, Italian, German, and Chinese), with questions spanning three complexity levels from basic factual extraction to complex cross-document reasoning. We propose a data generation pipeline derived from real-world sources through de-identification and data synthesis to ensure both realism and privacy, while controlled demographic and professional attributes (implemented via placeholders) enable systematic bias and fairness studies. We also present a cost-effective, human-in-the-loop translation pipeline based on the TEaR methodology, incorporating MQM error annotations and selective post-editing to ensure an high-quality multi-way parallel benchmark. We provide a baseline evaluations across multiple open-weight LLM families using an LLM-as-judge approach revealing higher performances on English and Spanish but substantial degradation for other languages, highlighting critical gaps in multilingual MRC capabilities for HR applications. JobResQA provides a reproducible benchmark for advancing fair and reliable LLM-based HR systems. The benchmark is publicly available at: https://github.com/Avature/jobresqa-benchmark
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.16949v6">Breaking the Exploration Bottleneck: Rubric-Scaffolded Reinforcement Learning for General LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-30
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have underscored the potential of Reinforcement Learning (RL) to facilitate the emergence of reasoning capabilities. Despite the encouraging results, a fundamental dilemma persists as RL improvement relies on learning from high-quality samples, yet the exploration for such samples remains bounded by the inherent limitations of LLMs. This, in effect, creates an undesirable cycle in which what cannot be explored cannot be learned. In this work, we propose Rubric-Scaffolded Reinforcement Learning (RuscaRL), a novel instructional scaffolding framework designed to break the exploration bottleneck for general LLM reasoning. Specifically, RuscaRL introduces checklist-style rubrics as (1) explicit scaffolding for exploration during rollout generation, where different rubrics are provided as external guidance within task instructions to steer diverse high-quality responses. This guidance is gradually decayed over time, encouraging the model to internalize the underlying reasoning patterns; (2) verifiable rewards for exploitation during model training, where we can obtain robust LLM-as-a-Judge scores using rubrics as references, enabling effective RL on general reasoning tasks. Extensive experiments demonstrate the superiority of the proposed RuscaRL across various benchmarks, effectively expanding reasoning boundaries under the Best-of-N evaluation. Our code is available at https://github.com/IANNXANG/RuscaRL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.06822v3">RAFFLES: Reasoning-based Attribution of Faults for LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 Accepted at EACL 2026 Main Conference
    </div>
    <details class="paper-abstract">
      The advent of complex, interconnected long-horizon LLM systems has made it incredibly tricky to identify where and when these systems break down. Evaluation capabilities that currently exist today are limited in that they often focus on simple metrics, end-to-end outcomes, and are dependent on the perspectives of humans. In order to match the increasing complexity of these many component systems, evaluation frameworks must also be able to reason, probe, iterate, and understand the nuanced logic passing through these systems. In this paper, we present RAFFLES, an offline evaluation architecture that incorporates iterative reasoning. Specifically, RAFFLES operates as an iterative, multi-component pipeline, using a central Judge to systematically identify faults and a set of specialized Evaluators to assess the quality of the candidate faults as well as rationales of the Judge. We evaluated RAFFLES with several benchmarks - the Who&When dataset to identify step-level faults in multi-agent systems and the ReasonEval datasets to diagnose step-level mathematical reasoning errors. RAFFLES outperforms strong baselines, achieving an accuracy of over 20% and 50% on the Who&When Hand-Crafted and Algorithmically-Generated datasets, and over 80% on the ReasonEval datasets. These results demonstrate a key step towards introducing automated fault detection for autonomous systems over labor-intensive manual review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.23132v1">Secure Tool Manifest and Digital Signing Solution for Verifiable MCP and LLM Pipelines</a></div>
    <div class="paper-meta">
      📅 2026-01-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly adopted in sensitive domains such as healthcare and financial institutions' data analytics; however, their execution pipelines remain vulnerable to manipulation and unverifiable behavior. Existing control mechanisms, such as the Model Context Protocol (MCP), define compliance policies for tool invocation but lack verifiable enforcement and transparent validation of model actions. To address this gap, we propose a novel Secure Tool Manifest and Digital Signing Framework, a structured and security-aware extension of Model Context Protocols. The framework enforces cryptographically signed manifests, integrates transparent verification logs, and isolates model-internal execution metadata from user-visible components to ensure verifiable execution integrity. Furthermore, the evaluation demonstrates that the framework scales nearly linearly (R-squared = 0.998), achieves near-perfect acceptance of valid executions while consistently rejecting invalid ones, and maintains balanced model utilization across execution pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.23129v1">Evaluating the Utility of Grounding Documents with Reference-Free LLM-based Metrics</a></div>
    <div class="paper-meta">
      📅 2026-01-30
    </div>
    <details class="paper-abstract">
      Retrieval Augmented Generation (RAG)'s success depends on the utility the LLM derives from the content used for grounding. Quantifying content utility does not have a definitive specification and existing metrics ignore model-specific capabilities and/or rely on costly annotations. In this paper, we propose Grounding Generation Utility (GroGU), a model-specific and reference-free metric that defines utility as a function of the downstream LLM's generation confidence based on entropy. Despite having no annotation requirements, GroGU is largely faithful in distinguishing ground-truth documents while capturing nuances ignored by LLM-agnostic metrics. We apply GroGU to train a query-rewriter for RAG by identifying high-utility preference data for Direct Preference Optimization. Experiments show improvements by up to 18.2 points in Mean Reciprocal Rank and up to 9.4 points in answer accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.03129v3">ARB-LLM: Alternating Refined Binarizations for Large Language Models</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 The code and models will be available at https://github.com/ZHITENGLI/ARB-LLM
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have greatly pushed forward advancements in natural language processing, yet their high memory and computational demands hinder practical deployment. Binarization, as an effective compression technique, can shrink model weights to just 1 bit, significantly reducing the high demands on computation and memory. However, current binarization methods struggle to narrow the distribution gap between binarized and full-precision weights, while also overlooking the column deviation in LLM weight distribution. To tackle these issues, we propose ARB-LLM, a novel 1-bit post-training quantization (PTQ) technique tailored for LLMs. To narrow the distribution shift between binarized and full-precision weights, we first design an alternating refined binarization (ARB) algorithm to progressively update the binarization parameters, which significantly reduces the quantization error. Moreover, considering the pivot role of calibration data and the column deviation in LLM weights, we further extend ARB to ARB-X and ARB-RC. In addition, we refine the weight partition strategy with column-group bitmap (CGB), which further enhance performance. Equipping ARB-X and ARB-RC with CGB, we obtain ARB-LLM$_\text{X}$ and ARB-LLM$_\text{RC}$ respectively, which significantly outperform state-of-the-art (SOTA) binarization methods for LLMs. As a binary PTQ method, our ARB-LLM$_\text{RC}$ is the first to surpass FP16 models of the same size. The code and models will be available at https://github.com/ZHITENGLI/ARB-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.23088v1">From Similarity to Vulnerability: Key Collision Attack on LLM Semantic Caching</a></div>
    <div class="paper-meta">
      📅 2026-01-30
    </div>
    <details class="paper-abstract">
      Semantic caching has emerged as a pivotal technique for scaling LLM applications, widely adopted by major providers including AWS and Microsoft. By utilizing semantic embedding vectors as cache keys, this mechanism effectively minimizes latency and redundant computation for semantically similar queries. In this work, we conceptualize semantic cache keys as a form of fuzzy hashes. We demonstrate that the locality required to maximize cache hit rates fundamentally conflicts with the cryptographic avalanche effect necessary for collision resistance. Our conceptual analysis formalizes this inherent trade-off between performance (locality) and security (collision resilience), revealing that semantic caching is naturally vulnerable to key collision attacks. While prior research has focused on side-channel and privacy risks, we present the first systematic study of integrity risks arising from cache collisions. We introduce CacheAttack, an automated framework for launching black-box collision attacks. We evaluate CacheAttack in security-critical tasks and agentic workflows. It achieves a hit rate of 86\% in LLM response hijacking and can induce malicious behaviors in LLM agent, while preserving strong transferability across different embedding models. A case study on a financial agent further illustrates the real-world impact of these vulnerabilities. Finally, we discuss mitigation strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.23085v1">OrLog: Resolving Complex Queries with LLMs and Probabilistic Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 Accepted to ECIR 2026
    </div>
    <details class="paper-abstract">
      Resolving complex information needs that come with multiple constraints should consider enforcing the logical operators encoded in the query (i.e., conjunction, disjunction, negation) on the candidate answer set. Current retrieval systems either ignore these constraints in neural embeddings or approximate them in a generative reasoning process that can be inconsistent and unreliable. Although well-suited to structured reasoning, existing neuro-symbolic approaches remain confined to formal logic or mathematics problems as they often assume unambiguous queries and access to complete evidence, conditions rarely met in information retrieval. To bridge this gap, we introduce OrLog, a neuro-symbolic retrieval framework that decouples predicate-level plausibility estimation from logical reasoning: a large language model (LLM) provides plausibility scores for atomic predicates in one decoding-free forward pass, from which a probabilistic reasoning engine derives the posterior probability of query satisfaction. We evaluate OrLog across multiple backbone LLMs, varying levels of access to external knowledge, and a range of logical constraints, and compare it against base retrievers and LLM-as-reasoner methods. Provided with entity descriptions, OrLog can significantly boost top-rank precision compared to LLM reasoning with larger gains on disjunctive queries. OrLog is also more efficient, cutting mean tokens by $\sim$90\% per query-entity pair. These results demonstrate that generation-free predicate plausibility estimation combined with probabilistic reasoning enables constraint-aware retrieval that outperforms monolithic reasoning while using far fewer tokens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.06446v2">Tokenization Multiplicity Leads to Arbitrary Price Variation in LLM-as-a-service</a></div>
    <div class="paper-meta">
      📅 2026-01-30
    </div>
    <details class="paper-abstract">
      Providers of LLM-as-a-service have predominantly adopted a simple pricing model: users pay a fixed price per token. Consequently, one may think that the price two different users would pay for the same output string under the same input prompt is the same. In our work, we show that, surprisingly, this is not (always) true. We find empirical evidence that, particularly for non-english outputs, both proprietary and open-weights LLMs often generate the same (output) string with multiple different tokenizations, even under the same input prompt, and this in turn leads to arbitrary price variation. To address the problem of tokenization multiplicity, we introduce canonical generation, a type of constrained generation that restricts LLMs to only generate canonical tokenizations -- the unique tokenization in which each string is tokenized during the training process of an LLM. Further, we introduce an efficient sampling algorithm for canonical generation based on the Gumbel-Max trick. Experiments on a variety of natural language tasks demonstrate that our sampling algorithm for canonical generation is comparable to standard sampling in terms of performance and runtime, and it solves the problem of tokenization multiplicity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.23066v1">Towards Explicit Acoustic Evidence Perception in Audio LLMs for Speech Deepfake Detection</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 9 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Speech deepfake detection (SDD) focuses on identifying whether a given speech signal is genuine or has been synthetically generated. Existing audio large language model (LLM)-based methods excel in content understanding; however, their predictions are often biased toward semantically correlated cues, which results in fine-grained acoustic artifacts being overlooked during the decisionmaking process. Consequently, fake speech with natural semantics can bypass detectors despite harboring subtle acoustic anomalies; this suggests that the challenge stems not from the absence of acoustic data, but from its inadequate accessibility when semantic-dominant reasoning prevails. To address this issue, we investigate SDD within the audio LLM paradigm and introduce SDD with Auditory Perception-enhanced Audio Large Language Model (SDD-APALLM), an acoustically enhanced framework designed to explicitly expose fine-grained time-frequency evidence as accessible acoustic cues. By combining raw audio with structured spectrograms, the proposed framework empowers audio LLMs to more effectively capture subtle acoustic inconsistencies without compromising their semantic understanding. Experimental results indicate consistent gains in detection accuracy and robustness, especially in cases where semantic cues are misleading. Further analysis reveals that these improvements stem from a coordinated utilization of semantic and acoustic information, as opposed to simple modality aggregation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.23049v1">MedMCP-Calc: Benchmarking LLMs for Realistic Medical Calculator Scenarios via MCP Integration</a></div>
    <div class="paper-meta">
      📅 2026-01-30
    </div>
    <details class="paper-abstract">
      Medical calculators are fundamental to quantitative, evidence-based clinical practice. However, their real-world use is an adaptive, multi-stage process, requiring proactive EHR data acquisition, scenario-dependent calculator selection, and multi-step computation, whereas current benchmarks focus only on static single-step calculations with explicit instructions. To address these limitations, we introduce MedMCP-Calc, the first benchmark for evaluating LLMs in realistic medical calculator scenarios through Model Context Protocol (MCP) integration. MedMCP-Calc comprises 118 scenario tasks across 4 clinical domains, featuring fuzzy task descriptions mimicking natural queries, structured EHR database interaction, external reference retrieval, and process-level evaluation. Our evaluation of 23 leading models reveals critical limitations: even top performers like Claude Opus 4.5 exhibit substantial gaps, including difficulty selecting appropriate calculators for end-to-end workflows given fuzzy queries, poor performance in iterative SQL-based database interactions, and marked reluctance to leverage external tools for numerical computation. Performance also varies considerably across clinical domains. Building on these findings, we develop CalcMate, a fine-tuned model incorporating scenario planning and tool augmentation, achieving state-of-the-art performance among open-source models. Benchmark and Codes are available in https://github.com/SPIRAL-MED/MedMCP-Calc.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.23048v1">From Abstract to Contextual: What LLMs Still Cannot Do in Mathematics</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      Large language models now solve many benchmark math problems at near-expert levels, yet this progress has not fully translated into reliable performance in real-world applications. We study this gap through contextual mathematical reasoning, where the mathematical core must be formulated from descriptive scenarios. We introduce ContextMATH, a benchmark that repurposes AIME and MATH-500 problems into two contextual settings: Scenario Grounding (SG), which embeds abstract problems into realistic narratives without increasing reasoning complexity, and Complexity Scaling (CS), which transforms explicit conditions into sub-problems to capture how constraints often appear in practice. Evaluating 61 proprietary and open-source models, we observe sharp drops: on average, open-source models decline by 13 and 34 points on SG and CS, while proprietary models drop by 13 and 20. Error analysis shows that errors are dominated by incorrect problem formulation, with formulation accuracy declining as original problem difficulty increases. Correct formulation emerges as a prerequisite for success, and its sufficiency improves with model scale, indicating that larger models advance in both understanding and reasoning. Nevertheless, formulation and reasoning remain two complementary bottlenecks that limit contextual mathematical problem solving. Finally, we find that fine-tuning with scenario data improves performance, whereas formulation-only training is ineffective. However, performance gaps are only partially alleviated, highlighting contextual mathematical reasoning as a central unsolved challenge for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.22830v2">ChatInject: Abusing Chat Templates for Prompt Injection in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      The growing deployment of large language model (LLM) based agents that interact with external environments has created new attack surfaces for adversarial manipulation. One major threat is indirect prompt injection, where attackers embed malicious instructions in external environment output, causing agents to interpret and execute them as if they were legitimate prompts. While previous research has focused primarily on plain-text injection attacks, we find a significant yet underexplored vulnerability: LLMs' dependence on structured chat templates and their susceptibility to contextual manipulation through persuasive multi-turn dialogues. To this end, we introduce ChatInject, an attack that formats malicious payloads to mimic native chat templates, thereby exploiting the model's inherent instruction-following tendencies. Building on this foundation, we develop a persuasion-driven Multi-turn variant that primes the agent across conversational turns to accept and execute otherwise suspicious actions. Through comprehensive experiments across frontier LLMs, we demonstrate three critical findings: (1) ChatInject achieves significantly higher average attack success rates than traditional prompt injection methods, improving from 5.18% to 32.05% on AgentDojo and from 15.13% to 45.90% on InjecAgent, with multi-turn dialogues showing particularly strong performance at average 52.33% success rate on InjecAgent, (2) chat-template-based payloads demonstrate strong transferability across models and remain effective even against closed-source LLMs, despite their unknown template structures, and (3) existing prompt-based defenses are largely ineffective against this attack approach, especially against Multi-turn variants. These findings highlight vulnerabilities in current agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.23001v1">Bias Beyond Borders: Political Ideology Evaluation and Steering in Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 PrePrint
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly shape global discourse, making fairness and ideological neutrality essential for responsible AI deployment. Despite growing attention to political bias in LLMs, prior work largely focuses on high-resource, Western languages or narrow multilingual settings, leaving cross-lingual consistency and safe post-hoc mitigation underexplored. To address this gap, we present a large-scale multilingual evaluation of political bias spanning 50 countries and 33 languages. We introduce a complementary post-hoc mitigation framework, Cross-Lingual Alignment Steering (CLAS), designed to augment existing steering methods by aligning ideological representations across languages and dynamically regulating intervention strength. This method aligns latent ideological representations induced by political prompts into a shared ideological subspace, ensuring cross lingual consistency, with the adaptive mechanism prevents over correction and preserves coherence. Experiments demonstrate substantial bias reduction along both economic and social axes with minimal degradation in response quality. The proposed framework establishes a scalable and interpretable paradigm for fairness-aware multilingual LLM governance, balancing ideological neutrality with linguistic and cultural diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.23000v1">Mano: Restriking Manifold Optimization for LLM Training</a></div>
    <div class="paper-meta">
      📅 2026-01-30
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have emerged as a significant advancement in artificial intelligence, the hardware and computational costs for training LLMs are also significantly burdensome. Among the state-of-the-art optimizers, AdamW relies on diagonal curvature estimates and ignores structural properties, while Muon applies global spectral normalization at the expense of losing curvature information. In this study, we restriked manifold optimization methods for training LLMs, which may address both optimizers' limitations, while conventional manifold optimization methods have been largely overlooked due to the poor performance in large-scale model optimization. By innovatively projecting the momentum onto the tangent space of model parameters and constraining it on a rotational Oblique manifold, we propose a novel, powerful, and efficient optimizer **Mano** that is the first to bridge the performance gap between manifold optimization and modern optimizers. Extensive experiments on the LLaMA and Qwen3 models demonstrate that Mano consistently and significantly outperforms AdamW and Muon even with less memory consumption and computational complexity, respectively, suggesting an expanded Pareto frontier in terms of space and time efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22996v1">Competitive Non-Clairvoyant KV-Cache Scheduling for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-01-30
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) inference presents a unique scheduling challenge due to the Key-Value (KV) cache, where a job's memory footprint grows linearly with the number of decoded tokens. This growth couples scheduling decisions with feasibility: a scheduler must minimize latency under a hard memory budget, yet the response lengths of requests are inherently unknown. While recent works have explored this problem either assuming clairvoyance -- exact knowledge of response lengths -- or relying on machine-learned predictions, obtaining robust performance guarantees without any prior knowledge of job sizes remains a theoretically fundamental and practically important open problem. In this work, we propose the Geometric Slicing Algorithm (GSA), the non-clairvoyant policy to achieve the first constant competitive ratio for this problem in the offline batch setting. GSA manages uncertainty through a geometric phase structure that periodically restarts jobs to bound memory exposure, combined with a staggered pipeline mechanism that enables high concurrency by smoothing aggregate memory consumption. We prove that GSA achieves a competitive ratio of at most 61.92 for general instances, improving to 32 in the large-memory regime. Our algorithmic framework also yields a clairvoyant counterpart, the Geometric Batching Algorithm (GBA), which achieves an approximation ratio of 10.67 for general instances and 6.75 in the large-memory regime -- significantly improving upon the best previously known bound of over 9000. Numerical experiments on real request traces demonstrate that our algorithms perform robustly while preserving these worst-case guarantees.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.17229v3">FactSelfCheck: Fact-Level Black-Box Hallucination Detection for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 Accepted for EACL 2026 (findings)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) frequently generate hallucinated content, posing significant challenges for applications where factuality is crucial. While existing hallucination detection methods typically operate at the sentence level or passage level, we propose FactSelfCheck, a novel zero-resource black-box sampling-based method that enables fine-grained fact-level detection. Our approach represents text as interpretable knowledge graphs consisting of facts in the form of triples, providing clearer insights into content factuality than traditional approaches. Through analyzing factual consistency across multiple LLM responses, we compute fine-grained hallucination scores without requiring external resources or training data. Our evaluation demonstrates that FactSelfCheck performs competitively with leading sentence-level sampling-based methods while providing more detailed and interpretable insights. Most notably, our fact-level approach significantly improves hallucination correction, achieving a 35.5% increase in factual content compared to the baseline, while sentence-level SelfCheckGPT yields only a 10.6% improvement. The granular nature of our detection enables more precise identification and correction of hallucinated content. Additionally, we contribute FavaMultiSamples, a novel dataset that addresses a gap in the field by providing the research community with a second dataset for evaluating sampling-based methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22952v1">Sifting the Noise: A Comparative Study of LLM Agents in Vulnerability False Positive Filtering</a></div>
    <div class="paper-meta">
      📅 2026-01-30
    </div>
    <details class="paper-abstract">
      Static Application Security Testing (SAST) tools are essential for identifying software vulnerabilities, but they often produce a high volume of false positives (FPs), imposing a substantial manual triage burden on developers. Recent advances in Large Language Model (LLM) agents offer a promising direction by enabling iterative reasoning, tool use, and environment interaction to refine SAST alerts. However, the comparative effectiveness of different LLM-based agent architectures for FP filtering remains poorly understood. In this paper, we present a comparative study of three state-of-the-art LLM-based agent frameworks, i.e., Aider, OpenHands, and SWE-agent, for vulnerability FP filtering. We evaluate these frameworks using the vulnerabilities from the OWASP Benchmark and real-world open-source Java projects. The experimental results show that LLM-based agents can remove the majority of SAST noise, reducing an initial FP detection rate of over 92% on the OWASP Benchmark to as low as 6.3% in the best configuration. On real-world dataset, the best configuration of LLM-based agents can achieve an FP identification rate of up to 93.3% involving CodeQL alerts. However, the benefits of agents are strongly backbone- and CWE-dependent: agentic frameworks significantly outperform vanilla prompting for stronger models such as Claude Sonnet 4 and GPT-5, but yield limited or inconsistent gains for weaker backbones. Moreover, aggressive FP reduction can come at the cost of suppressing true vulnerabilities, highlighting important trade-offs. Finally, we observe large disparities in computational cost across agent frameworks. Overall, our study demonstrates that LLM-based agents are a powerful but non-uniform solution for SAST FP filtering, and that their practical deployment requires careful consideration of agent design, backbone model choice, vulnerability category, and operational cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.07644v3">FloorplanQA: A Benchmark for Spatial Reasoning in LLMs using Structured Representations</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 v3, Project page: https://huggingface.co/papers/2507.07644
    </div>
    <details class="paper-abstract">
      We introduce FloorplanQA, a diagnostic benchmark for evaluating spatial reasoning in large-language models (LLMs). FloorplanQA is grounded in structured representations of indoor scenes, such as (e.g., kitchens, living rooms, bedrooms, bathrooms, and others), encoded symbolically in JSON or XML layouts. The benchmark covers core spatial tasks, including distance measurement, visibility, path finding, and object placement within constrained spaces. Our results across a variety of frontier open-source and commercial LLMs reveal that while models may succeed in shallow queries, they often fail to respect physical constraints, preserve spatial coherence, though they remain mostly robust to small spatial perturbations. FloorplanQA uncovers a blind spot in today's LLMs: inconsistent reasoning about indoor layouts. We hope this benchmark inspires new work on language models that can accurately infer and manipulate spatial and geometric properties in practical settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22928v1">LLMs Explain't: A Post-Mortem on Semantic Interpretability in Transformer Models</a></div>
    <div class="paper-meta">
      📅 2026-01-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are becoming increasingly popular in pervasive computing due to their versatility and strong performance. However, despite their ubiquitous use, the exact mechanisms underlying their outstanding performance remain unclear. Different methods for LLM explainability exist, and many are, as a method, not fully understood themselves. We started with the question of how linguistic abstraction emerges in LLMs, aiming to detect it across different LLM modules (attention heads and input embeddings). For this, we used methods well-established in the literature: (1) probing for token-level relational structures, and (2) feature-mapping using embeddings as carriers of human-interpretable properties. Both attempts failed for different methodological reasons: Attention-based explanations collapsed once we tested the core assumption that later-layer representations still correspond to tokens. Property-inference methods applied to embeddings also failed because their high predictive scores were driven by methodological artifacts and dataset structure rather than meaningful semantic knowledge. These failures matter because both techniques are widely treated as evidence for what LLMs supposedly understand, yet our results show such conclusions are unwarranted. These limitations are particularly relevant in pervasive and distributed computing settings where LLMs are deployed as system components and interpretability methods are relied upon for debugging, compression, and explaining models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.08855v3">BiasGym: Fantastic LLM Biases and How to Find (and Remove) Them</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Understanding biases and stereotypes encoded in the weights of Large Language Models (LLMs) is crucial for developing effective mitigation strategies. However, biased behaviour is often subtle and non-trivial to isolate, even when deliberately elicited, making systematic analysis and debiasing particularly challenging. To address this, we introduce \texttt{BiasGym}, a simple, cost-effective, and generalizable framework for reliably and safely injecting, analyzing, and mitigating conceptual associations of biases within LLMs. \texttt{BiasGym} consists of two components: \texttt{BiasInject}, which safely injects specific biases into the model via token-based fine-tuning while keeping the model frozen, and \texttt{BiasScope}, which leverages these injected signals to identify and reliably steer the components responsible for biased behavior. Our method enables consistent bias elicitation for mechanistic analysis, supports targeted debiasing without degrading performance on downstream tasks, and generalizes to biases unseen during fine-tuning. We demonstrate the effectiveness of BiasGym in reducing real-world stereotypes (e.g., people from Italy being `reckless drivers'), showing its utility for both safety interventions and interpretability research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22896v1">Game-Theoretic Co-Evolution for LLM-Based Heuristic Discovery</a></div>
    <div class="paper-meta">
      📅 2026-01-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have enabled rapid progress in automatic heuristic discovery (AHD), yet most existing methods are predominantly limited by static evaluation against fixed instance distributions, leading to potential overfitting and poor generalization under distributional shifts. We propose Algorithm Space Response Oracles (ASRO), a game-theoretic framework that reframes heuristic discovery as a program level co-evolution between solver and instance generator. ASRO models their interaction as a two-player zero-sum game, maintains growing strategy pools on both sides, and iteratively expands them via LLM-based best-response oracles against mixed opponent meta-strategies, thereby replacing static evaluation with an adaptive, self-generated curriculum. Across multiple combinatorial optimization domains, ASRO consistently outperforms static-training AHD baselines built on the same program search mechanisms, achieving substantially improved generalization and robustness on diverse and out-of-distribution instances.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22888v1">Should LLMs, $\textit{like}$, Generate How Users Talk? Building Dialect-Accurate Dialog[ue]s Beyond the American Default with MDial</a></div>
    <div class="paper-meta">
      📅 2026-01-30
    </div>
    <details class="paper-abstract">
      More than 80% of the 1.6 billion English speakers do not use Standard American English (SAE) and experience higher failure rates and stereotyped responses when interacting with LLMs as a result. Yet multi-dialectal performance remains underexplored. We introduce $\textbf{MDial}$, the first large-scale framework for generating multi-dialectal conversational data encompassing the three pillars of written dialect -- lexical (vocabulary), orthographic (spelling), and morphosyntactic (grammar) features -- for nine English dialects. Partnering with native linguists, we design an annotated and scalable rule-based LLM transformation to ensure precision. Our approach challenges the assumption that models should mirror users' morphosyntactic features, showing that up to 90% of the grammatical features of a dialect should not be reproduced by models. Independent evaluations confirm data quality, with annotators preferring MDial outputs over prior methods in 98% of pairwise comparisons for dialect naturalness. Using this pipeline, we construct the dialect-parallel $\textbf{MDialBench}$mark with 50k+ dialogs, resulting in 97k+ QA pairs, and evaluate 17 LLMs on dialect identification and response generation tasks. Even frontier models achieve under 70% accuracy, fail to reach 50% for Canadian English, and systematically misclassify non-SAE dialects as American or British. As dialect identification underpins natural language understanding, these errors risk cascading failures into downstream tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22885v1">Leveraging LLMs For Turkish Skill Extraction</a></div>
    <div class="paper-meta">
      📅 2026-01-30
    </div>
    <details class="paper-abstract">
      Skill extraction is a critical component of modern recruitment systems, enabling efficient job matching, personalized recommendations, and labor market analysis. Despite Türkiye's significant role in the global workforce, Turkish, a morphologically complex language, lacks both a skill taxonomy and a dedicated skill extraction dataset, resulting in underexplored research in skill extraction for Turkish. This article seeks the answers to three research questions: 1) How can skill extraction be effectively performed for this language, in light of its low resource nature? 2)~What is the most promising model? 3) What is the impact of different Large Language Models (LLMs) and prompting strategies on skill extraction (i.e., dynamic vs. static few-shot samples, varying context information, and encouraging causal reasoning)? The article introduces the first Turkish skill extraction dataset and performance evaluations of automated skill extraction using LLMs. The manually annotated dataset contains 4,819 labeled skill spans from 327 job postings across different occupation areas. The use of LLM outperforms supervised sequence labeling when used in an end-to-end pipeline, aligning extracted spans with standardized skills in the ESCO taxonomy more effectively. The best-performing configuration, utilizing Claude Sonnet 3.7 with dynamic few-shot prompting for skill identification, embedding-based retrieval, and LLM-based reranking for skill linking, achieves an end-to-end performance of 0.56, positioning Turkish alongside similar studies in other languages, which are few in the literature. Our findings suggest that LLMs can improve skill extraction performance in low-resource settings, and we hope that our work will accelerate similar research on skill extraction for underrepresented languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04213v2">AnimatedLLM: Explaining LLMs with Interactive Visualizations</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 Accepted to TeachNLP @ EACL 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are becoming central to natural language processing education, yet materials showing their mechanics are sparse. We present AnimatedLLM, an interactive web application that provides step-by-step visualizations of a Transformer language model. AnimatedLLM runs entirely in the browser, using pre-computed traces of open LLMs applied on manually curated inputs. The application is available at https://animatedllm.github.io, both as a teaching aid and for self-educational purposes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18292v2">TriPlay-RL: Tri-Role Self-Play Reinforcement Learning for LLM Safety Alignment</a></div>
    <div class="paper-meta">
      📅 2026-01-30
    </div>
    <details class="paper-abstract">
      In recent years, safety risks associated with large language models have become increasingly prominent, highlighting the urgent need to mitigate the generation of toxic and harmful content. The mainstream paradigm for LLM safety alignment typically adopts a collaborative framework involving three roles: an attacker for adversarial prompt generation, a defender for safety defense, and an evaluator for response assessment. In this paper, we propose a closed-loop reinforcement learning framework called TriPlay-RL that enables iterative and co-improving collaboration among three roles with near-zero manual annotation. Experimental results show that the attacker preserves high output diversity while achieving a 20%-50% improvement in adversarial effectiveness; the defender attains 10%-30% gains in safety performance without degrading general reasoning capability; and the evaluator continuously refines its fine-grained judgment ability through iterations, accurately distinguishing unsafe responses, simple refusals, and useful guidance. Overall, our framework establishes an efficient and scalable paradigm for LLM safety alignment, enabling continuous co-evolution within a unified learning loop.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.16648v4">FESTA: Functionally Equivalent Sampling for Trust Assessment of Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 Accepted in the Findings of EMNLP, 2025
    </div>
    <details class="paper-abstract">
      The accurate trust assessment of multimodal large language models (MLLMs) generated predictions, which can enable selective prediction and improve user confidence, is challenging due to the diverse multi-modal input paradigms. We propose Functionally Equivalent Sampling for Trust Assessment (FESTA), a multimodal input sampling technique for MLLMs, that generates an uncertainty measure based on the equivalent and complementary input samplings. The proposed task-preserving sampling approach for uncertainty quantification expands the input space to probe the consistency (through equivalent samples) and sensitivity (through complementary samples) of the model. FESTA uses only input-output access of the model (black-box), and does not require ground truth (unsupervised). The experiments are conducted with various off-the-shelf multi-modal LLMs, on both visual and audio reasoning tasks. The proposed FESTA uncertainty estimate achieves significant improvement (33.3% relative improvement for vision-LLMs and 29.6% relative improvement for audio-LLMs) in selective prediction performance, based on area-under-receiver-operating-characteristic curve (AUROC) metric in detecting mispredictions. The code implementation is open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22813v1">Quartet II: Accurate LLM Pre-Training in NVFP4 by Improved Unbiased Gradient Estimation</a></div>
    <div class="paper-meta">
      📅 2026-01-30
    </div>
    <details class="paper-abstract">
      The NVFP4 lower-precision format, supported in hardware by NVIDIA Blackwell GPUs, promises to allow, for the first time, end-to-end fully-quantized pre-training of massive models such as LLMs. Yet, existing quantized training methods still sacrifice some of the representation capacity of this format in favor of more accurate unbiased quantized gradient estimation by stochastic rounding (SR), losing noticeable accuracy relative to standard FP16 and FP8 training. In this paper, improve the state of the art for quantized training in NVFP4 via a novel unbiased quantization routine for micro-scaled formats, called MS-EDEN, that has more than 2x lower quantization error than SR. We integrate it into a novel fully-NVFP4 quantization scheme for linear layers, called Quartet II. We show analytically that Quartet II achieves consistently better gradient estimation across all major matrix multiplications, both on the forward and on the backward passes. In addition, our proposal synergizes well with recent training improvements aimed specifically at NVFP4. We further validate Quartet II on end-to-end LLM training with up to 1.9B parameters on 38B tokens. We provide kernels for execution on NVIDIA Blackwell GPUs with up to 4.2x speedup over BF16. Our code is available at https://github.com/IST-DASLab/Quartet-II .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22812v1">Stable Personas: Dual-Assessment of Temporal Stability in LLM-Based Human Simulation</a></div>
    <div class="paper-meta">
      📅 2026-01-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) acting as artificial agents offer the potential for scalable behavioral research, yet their validity depends on whether LLMs can maintain stable personas across extended conversations. We address this point using a dual-assessment framework measuring both self-reported characteristics and observer-rated persona expression. Across two experiments testing four persona conditions (default, high, moderate, and low ADHD presentations), seven LLMs, and three semantically equivalent persona prompts, we examine between-conversation stability (3,473 conversations) and within-conversation stability (1,370 conversations and 18 turns). Self-reports remain highly stable both between and within conversations. However, observer ratings reveal a tendency for persona expressions to decline during extended conversations. These findings suggest that persona-instructed LLMs produce stable, persona-aligned self-reports, an important prerequisite for behavioral research, while identifying this regression tendency as a boundary condition for multi-agent social simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.07775v2">The Unintended Trade-off of AI Alignment:Balancing Hallucination Mitigation and Safety in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-30
    </div>
    <details class="paper-abstract">
      Hallucination in large language models (LLMs) has been widely studied in recent years, with progress in both detection and mitigation aimed at improving truthfulness. Yet, a critical side effect remains largely overlooked: enhancing truthfulness can negatively impact safety alignment. In this paper, we investigate this trade-off and show that increasing factual accuracy often comes at the cost of weakened refusal behavior. Our analysis reveals that this arises from overlapping components in the model that simultaneously encode hallucination and refusal information, leading alignment methods to suppress factual knowledge unintentionally. We further examine how fine-tuning on benign datasets, even when curated for safety, can degrade alignment for the same reason. To address this, we propose a method that disentangles refusal-related features from hallucination features using sparse autoencoders, and preserves refusal behavior during fine-tuning through subspace orthogonalization. This approach prevents hallucinations from increasing while maintaining safety alignment.We evaluate our method on commonsense reasoning tasks and harmful benchmarks (AdvBench and StrongReject). Results demonstrate that our approach preserves refusal behavior and task utility, mitigating the trade-off between truthfulness and safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22795v1">Sparse or Dense? A Mechanistic Estimation of Computation Density in Transformer-based LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-30
    </div>
    <details class="paper-abstract">
      Transformer-based large language models (LLMs) are comprised of billions of parameters arranged in deep and wide computational graphs. Several studies on LLM efficiency optimization argue that it is possible to prune a significant portion of the parameters, while only marginally impacting performance. This suggests that the computation is not uniformly distributed across the parameters. We introduce here a technique to systematically quantify computation density in LLMs. In particular, we design a density estimator drawing on mechanistic interpretability. We experimentally test our estimator and find that: (1) contrary to what has been often assumed, LLM processing generally involves dense computation; (2) computation density is dynamic, in the sense that models shift between sparse and dense processing regimes depending on the input; (3) per-input density is significantly correlated across LLMs, suggesting that the same inputs trigger either low or high density. Investigating the factors influencing density, we observe that predicting rarer tokens requires higher density, and increasing context length often decreases the density. We believe that our computation density estimator will contribute to a better understanding of the processing at work in LLMs, challenging their symbolic interpretation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.26374v3">BOTS: A Unified Framework for Bayesian Online Task Selection in LLM Reinforcement Finetuning</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 Accepted as a conference paper at ICLR 2026
    </div>
    <details class="paper-abstract">
      Reinforcement finetuning (RFT) is a key technique for aligning Large Language Models (LLMs) with human preferences and enhancing reasoning, yet its effectiveness is highly sensitive to which tasks are explored during training. Uniform task sampling is inefficient, wasting computation on tasks that are either trivial or unsolvable, while existing task selection methods often suffer from high rollout costs, poor adaptivity, or incomplete evidence. We introduce BOTS, a unified framework for Bayesian Online Task Selection in LLM reinforcement finetuning. Grounded in Bayesian inference, BOTS adaptively maintains posterior estimates of task difficulty as the model evolves. It jointly incorporates explicit evidence from direct evaluations of selected tasks and implicit evidence inferred from these evaluations for unselected tasks, with Thompson sampling ensuring a principled balance between exploration and exploitation for task selection. To make implicit evidence practical, we instantiate it with an ultra-light interpolation-based plug-in that estimates difficulties of tasks without extra rollouts, adding negligible overhead. Empirically, across diverse domains and LLM scales, BOTS consistently improves data efficiency and performance over baselines and ablations, providing a practical and extensible solution for dynamic task selection in RFT. Code is available at https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/bots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22791v1">Understanding on the Edge: LLM-generated Boundary Test Explanations</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 This is the author's accepted manuscript of a paper accepted for publication at AST 2026. The final version will appear in the ACM Digital Library
    </div>
    <details class="paper-abstract">
      Boundary value analysis and testing (BVT) is fundamental in software quality assurance because faults tend to cluster at input extremes, yet testers often struggle to understand and justify why certain input-output pairs represent meaningful behavioral boundaries. Large Language Models (LLMs) could help by producing natural-language rationales, but their value for BVT has not been empirically assessed. We therefore conducted an exploratory study on LLM-generated boundary explanations: in a survey, twenty-seven software professionals rated GPT-4.1 explanations for twenty boundary pairs on clarity, correctness, completeness and perceived usefulness, and six of them elaborated in follow-up interviews. Overall, 63.5% of all ratings were positive (4-5 on a five-point Likert scale) compared to 17% negative (1-2), indicating general agreement but also variability in perceptions. Participants favored explanations that followed a clear structure, cited authoritative sources, and adapted their depth to the reader's expertise; they also stressed the need for actionable examples to support debugging and documentation. From these insights, we distilled a seven-item requirement checklist that defines concrete design criteria for future LLM-based boundary explanation tools. The results suggest that, with further refinement, LLM-based tools can support testing workflows by making boundary explanations more actionable and trustworthy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.07733v2">DeepGreen: Effective LLM-Driven Greenwashing Monitoring System Designed for Empirical Testing -- Evidence from China</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 Major revision accepted in Computational Economics, December 31, 2025. This version incorporates extensive revisions based on the reviewers' comments, with substantial changes
    </div>
    <details class="paper-abstract">
      Motivated by the emerging adoption of Large Language Models (LLMs) in economics and management research, this paper investigates whether LLMs can reliably identify corporate greenwashing narratives and, more importantly, whether and how the greenwashing signals extracted from textual disclosures can be used to empirically identify causal effects. To this end, this paper proposes DeepGreen, a dual-stage LLM-Driven system for detecting potential corporate greenwashing in annual reports. Applied to 9369 A-share annual reports published between 2021 and 2023, DeepGreen attains high reliability in random-sample validation at both stages. Ablation experiment shows that Retrieval-Augmented Generation (RAG) reduces hallucinations, as compared to simply lengthening the input window. Empirical tests indicate that "greenwashing" captured by DeepGreen can effectively reveal a positive relationship between greenwashing and environmental penalties, and IV, PSM, Placebo test, which enhance the robustness and causal effects of the empirical evidence. Further study suggests that the presence and number of green investors can weaken the positive correlation between greenwashing and penalties. Heterogeneity analysis shows that the positive relationship between "greenwashing - penalty" is less significant in large-sized corporations and corporations that have accumulated green assets, indicating that these green assets may be exploited as a credibility shield for greenwashing. Our findings demonstrate that LLMs can standardize ESG oversight by early warning and direct regulators' scarce attention toward the subsets of corporations where monitoring is more warranted.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20041v2">CiMRAG: CiM-Aware Domain-Adaptive and Noise-Resilient Retrieval-Augmented Generation for Edge-Based LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 Accepted by ICASSP 2026
    </div>
    <details class="paper-abstract">
      Personalized virtual assistants powered by large language models (LLMs) on edge devices are attracting growing attention, with Retrieval-Augmented Generation (RAG) emerging as a key method for personalization by retrieving relevant profile data and generating tailored responses. However, deploying RAG on edge devices faces efficiency hurdles due to the rapid growth of profile data, such as user-LLM interactions and recent updates. While Computing-in-Memory (CiM) architectures mitigate this bottleneck by eliminating data movement between memory and processing units via in-situ operations, they are susceptible to environmental noise that can degrade retrieval precision. This poses a critical issue in dynamic, multi-domain edge-based scenarios (e.g., travel, medicine, and law) where both accuracy and adaptability are paramount. To address these challenges, we propose Task-Oriented Noise-resilient Embedding Learning (TONEL), a framework that improves noise robustness and domain adaptability for RAG in noisy edge environments. TONEL employs a noise-aware projection model to learn task-specific embeddings compatible with CiM hardware constraints, enabling accurate retrieval under noisy conditions. Extensive experiments conducted on personalization benchmarks demonstrate the effectiveness and practicality of our methods relative to strong baselines, especially in task-specific noisy scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22764v1">How Far Can Pretrained LLMs Go in Symbolic Music? Controlled Comparisons of Supervised and Preference-based Adaptation</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 Accepted at NLP4MusA 2026
    </div>
    <details class="paper-abstract">
      Music often shares notable parallels with language, motivating the use of pretrained large language models (LLMs) for symbolic music understanding and generation. Despite growing interest, the practical effectiveness of adapting instruction-tuned LLMs to symbolic music remains insufficiently characterized. We present a controlled comparative study of finetuning strategies for ABC-based generation and understanding, comparing an off-the-shelf instruction-tuned backbone to domain-adapted variants and a music-specialized LLM baseline. Across multiple symbolic music corpora and evaluation signals, we provide some insights into adaptation choices for symbolic music applications. We highlight the domain adaptation vs.~preserving prior information tradeoff as well as the distinct behaviour of metrics used to measure the domain adaptation for symbolic music.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22759v1">Qualitative Evaluation of LLM-Designed GUI</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 12 pages, presented on conference PP-RAI 2025, Katowice-Poland
    </div>
    <details class="paper-abstract">
      As generative artificial intelligence advances, Large Language Models (LLMs) are being explored for automated graphical user interface (GUI) design. This study investigates the usability and adaptability of LLM-generated interfaces by analysing their ability to meet diverse user needs. The experiments included utilization of three state-of-the-art models from January 2025 (OpenAI GPT o3-mini-high, DeepSeek R1, and Anthropic Claude 3.5 Sonnet) generating mockups for three interface types: a chat system, a technical team panel, and a manager dashboard. Expert evaluations revealed that while LLMs are effective at creating structured layouts, they face challenges in meeting accessibility standards and providing interactive functionality. Further testing showed that LLMs could partially tailor interfaces for different user personas but lacked deeper contextual understanding. The results suggest that while LLMs are promising tools for early-stage UI prototyping, human intervention remains critical to ensure usability, accessibility, and user satisfaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22752v1">OSNIP: Breaking the Privacy-Utility-Efficiency Trilemma in LLM Inference via Obfuscated Semantic Null Space</a></div>
    <div class="paper-meta">
      📅 2026-01-30
    </div>
    <details class="paper-abstract">
      We propose Obfuscated Semantic Null space Injection for Privacy (OSNIP), a lightweight client-side encryption framework for privacy-preserving LLM inference. Generalizing the geometric intuition of linear kernels to the high-dimensional latent space of LLMs, we formally define the ``Obfuscated Semantic Null Space'', a high-dimensional regime that preserves semantic fidelity while enforcing near-orthogonality to the original embedding. By injecting perturbations that project the original embedding into this space, OSNIP ensures privacy without any post-processing. Furthermore, OSNIP employs a key-dependent stochastic mapping that synthesizes individualized perturbation trajectories unique to each user. Evaluations on 12 generative and classification benchmarks show that OSNIP achieves state-of-the-art performance, sharply reducing attack success rates while maintaining strong model utility under strict security constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.00459v2">Thinking Machines: Mathematical Reasoning in the Age of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive capabilities in structured reasoning and symbolic tasks, with coding emerging as a particularly successful application. This progress has naturally motivated efforts to extend these models to mathematics, both in its traditional form, expressed through natural-style mathematical language, and in its formalized counterpart, expressed in a symbolic syntax suitable for automatic verification. Yet, despite apparent parallels between programming and proof construction, advances in formalized mathematics have proven significantly more challenging. This gap raises fundamental questions about the nature of reasoning in current LLM architectures, the role of supervision and feedback, and the extent to which such models maintain an internal notion of computational or deductive state. In this article, we review the current state-of-the-art in mathematical reasoning with LLMs, focusing on recent models and benchmarks. We explore three central issues at the intersection of machine learning and mathematical cognition: (i) the trade-offs between traditional and formalized mathematics as training and evaluation domains; (ii) the structural and methodological reasons why proof synthesis remains more brittle than code generation; and (iii) whether LLMs genuinely represent or merely emulate a notion of evolving logical state. Our goal is not to draw rigid distinctions but to clarify the present boundaries of these systems and outline promising directions for their extension.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20900v2">Text-only adaptation in LLM-based ASR through text denoising</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 Paper accepted at ICASSP 2026
    </div>
    <details class="paper-abstract">
      Adapting automatic speech recognition (ASR) systems based on large language models (LLMs) to new domains using text-only data is a significant yet underexplored challenge. Standard fine-tuning of the LLM on target-domain text often disrupts the critical alignment between speech and text modalities learned by the projector, degrading performance. We introduce a novel text-only adaptation method that emulates the audio projection task by treating it as a text denoising task. Our approach thus trains the LLM to recover clean transcripts from noisy inputs. This process effectively adapts the model to a target domain while preserving cross-modal alignment. Our solution is lightweight, requiring no architectural changes or additional parameters. Extensive evaluation on two datasets demonstrates up to 22.1% relative improvement, outperforming recent state-of-the-art text-only adaptation methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22720v1">AEGIS: White-Box Attack Path Generation using LLMs and Training Effectiveness Evaluation for Large-Scale Cyber Defence Exercises</a></div>
    <div class="paper-meta">
      📅 2026-01-30
    </div>
    <details class="paper-abstract">
      Creating attack paths for cyber defence exercises requires substantial expert effort. Existing automation requires vulnerability graphs or exploit sets curated in advance, limiting where it can be applied. We present AEGIS, a system that generates attack paths using LLMs, white-box access, and Monte Carlo Tree Search over real exploit execution. LLM-based search discovers exploits dynamically without pre-existing vulnerability graphs, while white-box access enables validating exploits in isolation before committing to attack paths. Evaluation at CIDeX 2025, a large-scale exercise spanning 46 IT hosts, showed that AEGIS-generated paths are comparable to human-authored scenarios across four dimensions of training experience (perceived learning, engagement, believability, challenge). Results were measured with a validated questionnaire extensible to general simulation-based training. By automating exploit chain discovery and validation, AEGIS reduces scenario development from months to days, shifting expert effort from technical validation to scenario design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22716v1">Breaking the Blocks: Continuous Low-Rank Decomposed Scaling for Unified LLM Quantization and Adaptation</a></div>
    <div class="paper-meta">
      📅 2026-01-30
    </div>
    <details class="paper-abstract">
      Current quantization methods for LLMs predominantly rely on block-wise structures to maintain efficiency, often at the cost of representational flexibility. In this work, we demonstrate that element-wise quantization can be made as efficient as block-wise scaling while providing strictly superior expressive power by modeling the scaling manifold as continuous low-rank matrices ($S = BA$). We propose Low-Rank Decomposed Scaling (LoRDS), a unified framework that rethinks quantization granularity through this low-rank decomposition. By "breaking the blocks" of spatial constraints, LoRDS establishes a seamless efficiency lifecycle: it provides high-fidelity PTQ initialization refined via iterative optimization, enables joint QAT of weights and scaling factors, and facilitates high-rank multiplicative PEFT adaptation. Unlike additive PEFT approaches such as QLoRA, LoRDS enables high-rank weight updates within a low-rank budget while incurring no additional inference overhead. Supported by highly optimized Triton kernels, LoRDS consistently outperforms state-of-the-art baselines across various model families in both quantization and downstream fine-tuning tasks. Notably, on Llama3-8B, our method achieves up to a 27.0% accuracy improvement at 3 bits over NormalFloat quantization and delivers a 1.5x inference speedup on NVIDIA RTX 4090 while enhancing PEFT performance by 9.6% on downstream tasks over 4bit QLoRA, offering a robust and integrated solution for unified compression and adaptation of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22710v1">AlienLM: Alienization of Language for API-Boundary Privacy in Black-Box LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-30
    </div>
    <details class="paper-abstract">
      Modern LLMs are increasingly accessed via black-box APIs, requiring users to transmit sensitive prompts, outputs, and fine-tuning data to external providers, creating a critical privacy risk at the API boundary. We introduce AlienLM, a deployable API-only privacy layer that protects text by translating it into an Alien Language via a vocabulary-scale bijection, enabling lossless recovery on the client side. Using only standard fine-tuning APIs, Alien Adaptation Training (AAT) adapts target models to operate directly on alienized inputs. Across four LLM backbones and seven benchmarks, AlienLM retains over 81\% of plaintext-oracle performance on average, substantially outperforming random-bijection and character-level baselines. Under adversaries with access to model weights, corpus statistics, and learning-based inverse translation, recovery attacks reconstruct fewer than 0.22\% of alienized tokens. Our results demonstrate a practical pathway for privacy-preserving LLM deployment under API-only access, substantially reducing plaintext exposure while maintaining task performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22705v1">CONCUR: High-Throughput Agentic Batch Inference of LLM via Congestion-Based Concurrency Control</a></div>
    <div class="paper-meta">
      📅 2026-01-30
    </div>
    <details class="paper-abstract">
      Batch inference for agentic workloads stresses the GPU key-value (KV) cache in a sustained and cumulative manner, often causing severe throughput degradation well before memory capacity is exhausted. We identify this phenomenon as middle-phase thrashing, a previously under-characterized pathology in which cache efficiency collapses as long-lived agents accumulate state over time. We argue that mitigating this pathology requires moving beyond reactive, request-level cache management to proactive, agent-level admission control. Drawing inspiration from congestion control in distributed systems, we view the KV cache as a shared resource whose efficient utilization depends on feedback-driven regulation. Based on this insight, we present CONCUR, a lightweight control layer that regulates agent admission to bound aggregate cache pressure while preserving execution continuity. CONCUR adapts a cache-aware control algorithm to dynamically adjust the number of active agents using runtime cache signals. Across large models and real-world agent workloads, CONCUR prevents middle-phase thrashing and improves batch inference throughput by up to 4.09x on Qwen3-32B and 1.9x on DeepSeek-V3, while remaining compatible with existing LLM serving systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.21981v3">Collaborative Belief Reasoning with LLMs for Efficient Multi-Agent Collaboration</a></div>
    <div class="paper-meta">
      📅 2026-01-30
    </div>
    <details class="paper-abstract">
      Effective real-world multi-agent collaboration requires not only accurate planning but also the ability to reason about collaborators' intents--a crucial capability for avoiding miscoordination and redundant communication under partial observable environments. Due to their strong planning and reasoning capabilities, large language models (LLMs) have emerged as promising autonomous agents for collaborative task solving. However, existing collaboration frameworks for LLMs overlook their reasoning potential for dynamic intent inference, and thus produce inconsistent plans and redundant communication, reducing collaboration efficiency. To bridge this gap, we propose CoBel-World, a novel framework that equips LLM agents with a Collaborative Belief World--an internal representation jointly modeling the physical environment and collaborators' mental states. CoBel-World enables agents to parse external open-world knowledge into structured beliefs via a symbolic belief representation module, and perform zero-shot Bayesian-style belief updates through LLM reasoning. This allows agents to proactively detect potential miscoordination (e.g., conflicting plans) and communicate adaptively. Evaluated on challenging embodied benchmarks (i.e., TDW-MAT and C-WAH), CoBel-World significantly reduces communication costs by 64-79% and improves task completion efficiency by 4-28% compared to the strongest baseline. Our results show that explicit, intent-aware belief modeling is essential for efficient and human-like collaboration in LLM-based multi-agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.00068v3">Framing Political Bias in Multilingual LLMs Across Pakistani Languages</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly shape public discourse, yet most evaluations of political and economic bias have focused on high-resource, Western languages and contexts. This leaves critical blind spots in low-resource, multilingual regions such as Pakistan, where linguistic identity is closely tied to political, religious, and regional ideologies. We present a systematic evaluation of political bias in 13 state-of-the-art LLMs across five Pakistani languages: Urdu, Punjabi, Sindhi, Pashto, and Balochi. Our framework integrates a culturally adapted Political Compass Test (PCT) with multi-level framing analysis, capturing both ideological stance (economic/social axes) and stylistic framing (content, tone, emphasis). Prompts are aligned with 11 socio-political themes specific to the Pakistani context. Results show that while LLMs predominantly reflect liberal-left orientations consistent with Western training data, they exhibit more authoritarian framing in regional languages, highlighting language-conditioned ideological modulation. We also identify consistent model-specific bias patterns across languages. These findings show the need for culturally grounded, multilingual bias auditing frameworks in global NLP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22676v1">VarParser: Unleashing the Neglected Power of Variables for LLM-based Log Parsing</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 12 pages, 9 figures, Accepted in TheWebConf 2026
    </div>
    <details class="paper-abstract">
      Logs serve as a primary source of information for engineers to diagnose failures in large-scale online service systems. Log parsing, which extracts structured events from massive unstructured log data, is a critical first step for downstream tasks like anomaly detection and failure diagnosis. With advances in large language models (LLMs), leveraging their strong text understanding capabilities has proven effective for accurate log parsing. However, existing LLM-based log parsers all focus on the constant part of logs, ignoring the potential contribution of the variable part to log parsing. This constant-centric strategy brings four key problems. First, inefficient log grouping and sampling with only constant information. Second, a relatively large number of LLM invocations due to constant-based cache, leading to low log parsing accuracy and efficiency. Third, a relatively large number of consumed constant tokens in prompts leads to high LLM invocation costs. At last, these methods only retain placeholders in the results, losing the system visibility brought by variable information in logs. Facing these problems, we propose a variable-centric log parsing strategy named VarParser. Through variable contribution sampling, variable-centric parsing cache, and adaptive variable-aware in-context learning, our approach can efficiently capture the variable parts of logs and leverage their contributions to parsing. By introducing variable units, we preserve rich variable information, enhancing the integrity of log parsing results. Extensive evaluations on large-scale datasets demonstrate that VarParser achieves higher accuracy compared to existing methods, significantly improving parsing efficiency while reducing the LLM invocation costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22662v1">Task-Aware LLM Council with Adaptive Decision Pathways for Decision Support</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 A shorter version of this work has been accepted by ICASSP 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown strong capabilities across diverse decision-making tasks. However, existing approaches often overlook the specialization differences among available models, treating all LLMs as uniformly applicable regardless of task characteristics. This limits their ability to adapt to varying reasoning demands and task complexities. In this work, we propose Task-Aware LLM Council (TALC), a task-adaptive decision framework that integrates a council of LLMs with Monte Carlo Tree Search (MCTS) to enable dynamic expert selection and efficient multi-step planning. Each LLM is equipped with a structured success memory profile derived from prior task trajectories, enabling semantic matching between current reasoning context and past successes. At each decision point, TALC routes control to the most contextually appropriate model and estimates node value using a dual-signal mechanism that fuses model-based evaluations with historical utility scores. These signals are adaptively weighted based on intra-node variance and used to guide MCTS selection, allowing the system to balance exploration depth with planning confidence. Experiments on WebShop, HumanEval, and the Game of 24 demonstrate that TALC achieves superior task success rates and improved search efficiency compared to strong baselines, validating the benefits of specialization-aware routing and adaptive planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22655v1">The Semantic Trap: Do Fine-tuned LLMs Learn Vulnerability Root Cause or Just Functional Pattern?</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 21 pages
    </div>
    <details class="paper-abstract">
      LLMs demonstrate promising performance in software vulnerability detection after fine-tuning. However, it remains unclear whether these gains reflect a genuine understanding of vulnerability root causes or merely an exploitation of functional patterns. In this paper, we identify a critical failure mode termed the "semantic trap," where fine-tuned LLMs achieve high detection scores by associating certain functional domains with vulnerability likelihood rather than reasoning about the underlying security semantics.To systematically evaluate this phenomenon, we propose TrapEval, a comprehensive evaluation framework designed to disentangle vulnerability root cause from functional pattern. TrapEval introduces two complementary datasets derived from real-world open-source projects: V2N, which pairs vulnerable code with unrelated benign code, and V2P, which pairs vulnerable code with its corresponding patched version, forcing models to distinguish near-identical code that differs only in subtle security-critical logic. Using TrapEval, we fine-tune five representative state-of-the-art LLMs across three model families and evaluate them under cross-dataset testing, semantic-preserving perturbations, and varying degrees of semantic gap measured by CodeBLEU.Our empirical results reveal that, despite improvements in metrics, fine-tuned LLMs consistently struggle to distinguish vulnerable code from its patched counterpart, exhibit severe robustness degradation under minor semantic-preserving transformations, and rely heavily on functional-context shortcuts when the semantic gap is small. These findings provide strong evidence that current fine-tuning practices often fail to impart true vulnerability reasoning. Our findings serve as a wake-up call: high benchmark scores on traditional datasets may be illusory, masking the model's inability to understand the true causal logic of vulnerabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13590v2">Vulnerability of LLMs' Belief Systems? LLMs Belief Resistance Check Through Strategic Persuasive Conversation Interventions</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 Updated Table 12 with the detailed calculation approach and its related statements
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly employed in various question-answering tasks. However, recent studies showcase that LLMs are susceptible to persuasion and could adopt counterfactual beliefs. We present a systematic evaluation of LLM susceptibility to persuasion under the Source--Message--Channel--Receiver (SMCR) communication framework. Across five mainstream Large Language Models (LLMs) and three domains (factual knowledge, medical QA, and social bias), we analyze how different persuasive strategies influence belief stability over multiple interaction turns. We further examine whether meta-cognition prompting (i.e., eliciting self-reported confidence) affects resistance to persuasion. Results show that the smallest model (Llama 3.2-3B) exhibits extreme compliance, with 82.5% of belief changes occurring at the first persuasive turn (average end turn of 1.1--1.4). Contrary to expectations, meta-cognition prompting increases vulnerability by accelerating belief erosion rather than enhancing robustness. Finally, we evaluate adversarial fine-tuning as a defense. While GPT-4o-mini achieves near-complete robustness (98.6%) and Mistral~7B improves substantially (35.7% $\rightarrow$ 79.3%), Llama models remain highly susceptible (<14%) even when fine-tuned on their own failure cases. Together, these findings highlight substantial model-dependent limits of current robustness interventions and offer guidance for developing more trustworthy LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.17595v3">NeUQI: Near-Optimal Uniform Quantization Parameter Initialization for Low-Bit LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 under review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) achieve impressive performance across domains but face significant challenges when deployed on consumer-grade GPUs or personal devices such as laptops, due to high memory consumption and inference costs. Post-training quantization (PTQ) of LLMs offers a promising solution that reduces their memory footprint and decoding latency. In practice, PTQ with uniform quantization representation is favored due to its efficiency and ease of deployment, as uniform quantization is widely supported by mainstream hardware and software libraries. Recent studies on low-bit uniform quantization have led to noticeable improvements in post-quantization model performance; however, they mainly focus on quantization methodologies, while the initialization of quantization parameters remains underexplored and still relies on the conventional Min-Max formula. In this work, we identify the limitations of the Min-Max formula, move beyond its constraints, and propose NeUQI, a method that efficiently determines near-optimal initialization for uniform quantization. Our NeUQI simplifies the joint optimization of the scale and zero-point by deriving the zero-point for a given scale, thereby reducing the problem to a scale-only optimization. Benefiting from the improved quantization parameters, our NeUQI consistently outperforms existing methods in the experiments with the LLaMA and Qwen families on various settings and tasks. Furthermore, when combined with a lightweight distillation strategy, NeUQI even achieves superior performance to PV-tuning, a considerably more resource-intensive method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.07175v2">Quantifying Data Contamination in Psychometric Evaluations of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 EACL 2026 Findings
    </div>
    <details class="paper-abstract">
      Recent studies apply psychometric questionnaires to Large Language Models (LLMs) to assess high-level psychological constructs such as values, personality, moral foundations, and dark traits. Although prior work has raised concerns about possible data contamination from psychometric inventories, which may threaten the reliability of such evaluations, there has been no systematic attempt to quantify the extent of this contamination. To address this gap, we propose a framework to systematically measure data contamination in psychometric evaluations of LLMs, evaluating three aspects: (1) item memorization, (2) evaluation memorization, and (3) target score matching. Applying this framework to 21 models from major families and four widely used psychometric inventories, we provide evidence that popular inventories such as the Big Five Inventory (BFI-44) and Portrait Values Questionnaire (PVQ-40) exhibit strong contamination, where models not only memorize items but can also adjust their responses to achieve specific target scores.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.03267v2">PT$^2$-LLM: Post-Training Ternarization for Large Language Models</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 Accepted at ICLR 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive capabilities across diverse tasks, but their large memory and compute demands hinder deployment. Ternarization has gained attention as a promising compression technique, delivering substantial size reduction and high computational efficiency. However, its potential in the post-training quantization (PTQ) setting remains underexplored, due to the challenge of training-free parameter optimization and the quantization difficulty posed by outliers and dispersed weights. To address these issues, we propose PT$^2$-LLM, a post-training ternarization framework tailored for LLMs. At its core is an Asymmetric Ternary Quantizer equipped with a two-stage refinement pipeline: (1) Iterative Ternary Fitting (ITF), which alternates between optimal ternary grid construction and flexible rounding to minimize quantization error, and (2) Activation-aware Grid Alignment (AGA), which further refines the ternary grid to better match full-precision outputs. In addition, we propose a plug-and-play Structural Similarity-based Reordering (SSR) strategy that leverages inter-column structural similarity to ease quantization and mitigate outlier effects, further enhancing overall performance. Extensive experiments demonstrate that PT$^2$-LLM delivers competitive performance against state-of-the-art (SOTA) 2-bit PTQ methods with lower memory cost, while also accelerating both prefill and decoding to achieve end-to-end speedup. The code and models will be available at https://github.com/XIANGLONGYAN/PT2-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22588v1">Rethinking LLM-as-a-Judge: Representation-as-a-Judge with Small Language Models via Semantic Capacity Asymmetry</a></div>
    <div class="paper-meta">
      📅 2026-01-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are widely used as reference-free evaluators via prompting, but this "LLM-as-a-Judge" paradigm is costly, opaque, and sensitive to prompt design. In this work, we investigate whether smaller models can serve as efficient evaluators by leveraging internal representations instead of surface generation. We uncover a consistent empirical pattern: small LMs, despite with weak generative ability, encode rich evaluative signals in their hidden states. This motivates us to propose the Semantic Capacity Asymmetry Hypothesis: evaluation requires significantly less semantic capacity than generation and can be grounded in intermediate representations, suggesting that evaluation does not necessarily need to rely on large-scale generative models but can instead leverage latent features from smaller ones. Our findings motivate a paradigm shift from LLM-as-a-Judge to Representation-as-a-Judge, a decoding-free evaluation strategy that probes internal model structure rather than relying on prompted output. We instantiate this paradigm through INSPECTOR, a probing-based framework that predicts aspect-level evaluation scores from small model representations. Experiments on reasoning benchmarks (GSM8K, MATH, GPQA) show that INSPECTOR substantially outperforms prompting-based small LMs and closely approximates full LLM judges, while offering a more efficient, reliable, and interpretable alternative for scalable evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10198v2">HumanLLM: Benchmarking and Improving LLM Anthropomorphism via Human Cognitive Patterns</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 change "reinforce" to "improve"
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in reasoning and generation, serving as the foundation for advanced persona simulation and Role-Playing Language Agents (RPLAs). However, achieving authentic alignment with human cognitive and behavioral patterns remains a critical challenge for these agents. We present HumanLLM, a framework treating psychological patterns as interacting causal forces. We construct 244 patterns from ~12,000 academic papers and synthesize 11,359 scenarios where 2-5 patterns reinforce, conflict, or modulate each other, with multi-turn conversations expressing inner thoughts, actions, and dialogue. Our dual-level checklists evaluate both individual pattern fidelity and emergent multi-pattern dynamics, achieving strong human alignment (r=0.91) while revealing that holistic metrics conflate simulation accuracy with social desirability. HumanLLM-8B outperforms Qwen3-32B on multi-pattern dynamics despite 4x fewer parameters, demonstrating that authentic anthropomorphism requires cognitive modeling--simulating not just what humans do, but the psychological processes generating those behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22548v1">Are LLM Evaluators Really Narcissists? Sanity Checking Self-Preference Evaluations</a></div>
    <div class="paper-meta">
      📅 2026-01-30
    </div>
    <details class="paper-abstract">
      Recent research has shown that large language models (LLM) favor own outputs when acting as judges, undermining the integrity of automated post-training and evaluation workflows. However, it is difficult to disentangle which evaluation biases are explained by narcissism versus general experimental confounds, distorting measurements of self-preference bias. We discover a core methodological confound which could reduce measurement error by 89.6%. Specifically, LLM evaluators may deliver self-preferring verdicts when the judge responds to queries which they completed incorrectly themselves; this would be true regardless of whether one of their responses is their own. To decouple self-preference signals from noisy outputs on hard problems, we introduce an Evaluator Quality Baseline, which compares the probability that a judge incorrectly votes for itself against the probability that it votes for an incorrect response from another model. Evaluating this simple baseline on 37,448 queries, only 51% of initial findings retain statistical significance. Finally, we turn towards characterizing the entropy of "easy" versus "hard" evaluation votes from LLM judges. Our corrective baseline enables future research on self-preference by eliminating noisy data from potential solutions. More widely, this work contributes to the growing body of work on cataloging and isolating judge-bias effects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22546v1">Towards the Holographic Characteristic of LLMs for Efficient Short-text Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-30
    </div>
    <details class="paper-abstract">
      The recent advancements in Large Language Models (LLMs) have attracted interest in exploring their in-context learning abilities and chain-of-thought capabilities. However, there are few studies investigating the specific traits related to the powerful generation capacity of LLMs. This paper aims to delve into the generation characteristics exhibited by LLMs. Through our investigation, we have discovered that language models tend to capture target-side keywords at the beginning of the generation process. We name this phenomenon the Holographic Characteristic of language models. For the purpose of exploring this characteristic and further improving the inference efficiency of language models, we propose a plugin called HOLO, which leverages the Holographic Characteristic to extract target-side keywords from language models within a limited number of generation steps and complements the sentence with a parallel lexically constrained text generation method. To verify the effectiveness of HOLO, we conduct massive experiments on language models of varying architectures and scales in the short-text generation scenario. The results demonstrate that HOLO achieves comparable performance to the baselines in terms of both automatic and human-like evaluation metrics and highlight the potential of the Holographic Characteristic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22543v1">SCaLRec: Semantic Calibration for LLM-enabled Cloud-Device Sequential Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-01-30
    </div>
    <details class="paper-abstract">
      Cloud-device collaborative recommendation partitions computation across the cloud and user devices: the cloud provides semantic user modeling, while the device leverages recent interactions and cloud semantic signals for privacy-preserving, responsive reranking. With large language models (LLMs) on the cloud, semantic user representations can improve sequential recommendation by capturing high-level intent. However, regenerating such representations via cloud LLM inference for every request is often infeasible at real-world scale. As a result, on-device reranking commonly reuses a cached cloud semantic user embedding across requests. We empirically identify a cloud semantic staleness effect: reused embeddings become less aligned with the user's latest interactions, leading to measurable ranking degradation. Most existing LLM-enabled cloud-device recommenders are typically designed around on-demand cloud semantics, either by assuming low-latency cloud LLM access or by regenerating semantic embeddings per request. When per-request regeneration is infeasible and cached semantics must be reused, two technical challenges arise: (1) deciding when cached cloud semantics remain useful for on-device reranking, and (2) maintaining ranking quality when the cloud LLM cannot be invoked and only cached semantics are available. To address this gap, we introduce the Semantic Calibration for LLM-enabled Cloud-Device Recommendation (SCaLRec). First, it estimates the reliability of cached semantics under the user's latest interactions. Second, an on-device semantic calibration module is proposed to adjusts the cached semantic embedding on-device using up-to-date interaction evidence, without per-request cloud LLM involvement. Experiments on real-world datasets show that SCaLRec consistently improves recommendation performance over strong baselines under cloud semantic staleness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.02230v3">Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live</a></div>
    <div class="paper-meta">
      📅 2026-01-30
    </div>
    <details class="paper-abstract">
      KV cache management is essential for efficient LLM inference. To maximize utilization, existing inference engines evict finished requests' KV cache if new requests are waiting. This policy breaks for agentic workloads, which interleave LLM calls with tools, introducing pauses that prevent effective KV reuse across turns. Since some tool calls have much shorter durations than human response multi-turn chatbot, it would be promising to retain the KV cache in during these tools. However, there are many challenges. First, we need to consider both the potential cost of recomputation or reloading (if CPU offloading enabled) and the increasing queueing delays after eviction from GPU. Second, due to the internal variance of tool call durations, we need the method to remain robust under limited predictability of tool call durations. We present Continuum, a serving system to optimize job completion time for multi-turn agent workloads by introducing time-to-live mechanism for KV cache retaining. For LLM request that generates a tool call, Continuum selectively pins the KV cache in GPU memory with a time-to-live value determined by considering both the reload cost and ordering preserve benefit of retaining KV cache. Moreover, when the TTL expires, the KV cache can be automatically evicted to free up GPU memory, providing robust performance under edge cases. When combined with program-level first-come-first-serve, Continuum preserves multi-turn continuity, and reduces delay for complex agentic workflows. Our evaluation on real-world agentic workloads (SWE-Bench and BFCL) with Llama-3.1 8B/70B shows that Continuum significantly improves the average job completion times and its improvement scales with turn number increase. We release a preview version at: https://github.com/Hanchenli/vllm-continuum
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18220v2">LLM-ForcedAligner: A Non-Autoregressive and Accurate LLM-Based Forced Aligner for Multilingual and Long-Form Speech</a></div>
    <div class="paper-meta">
      📅 2026-01-30
    </div>
    <details class="paper-abstract">
      Forced alignment (FA) predicts start and end timestamps for words or characters in speech, but existing methods are language-specific and prone to cumulative temporal shifts. The multilingual speech understanding and long-sequence processing abilities of speech large language models (SLLMs) make them promising for FA in multilingual, crosslingual, and long-form speech settings. However, directly applying the next-token prediction paradigm of SLLMs to FA results in hallucinations and slow inference. To bridge the gap, we propose LLM-ForcedAligner, reformulating FA as a slot-filling paradigm: timestamps are treated as discrete indices, and special timestamp tokens are inserted as slots into the transcript. Conditioned on the speech embeddings and the transcript with slots, the SLLM directly predicts the time indices at slots. During training, causal attention masking with non-shifted input and label sequences allows each slot to predict its own timestamp index based on itself and preceding context, with loss computed only at slot positions. Dynamic slot insertion enables FA at arbitrary positions. Moreover, non-autoregressive inference is supported, avoiding hallucinations and improving speed. Experiments across multilingual, crosslingual, and long-form speech scenarios show that LLM-ForcedAligner achieves a 69%~78% relative reduction in accumulated averaging shift compared with prior methods. Checkpoint and inference code are available at https://github.com/QwenLM/Qwen3-ASR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21700v2">Toward Culturally Aligned LLMs through Ontology-Guided Multi-Agent Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 35 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly support culturally sensitive decision making, yet often exhibit misalignment due to skewed pretraining data and the absence of structured value representations. Existing methods can steer outputs, but often lack demographic grounding and treat values as independent, unstructured signals, reducing consistency and interpretability. We propose OG-MAR, an Ontology-Guided Multi-Agent Reasoning framework. OG-MAR summarizes respondent-specific values from the World Values Survey (WVS) and constructs a global cultural ontology by eliciting relations over a fixed taxonomy via competency questions. At inference time, it retrieves ontology-consistent relations and demographically similar profiles to instantiate multiple value-persona agents, whose outputs are synthesized by a judgment agent that enforces ontology consistency and demographic proximity. Experiments on regional social-survey benchmarks across four LLM backbones show that OG-MAR improves cultural alignment and robustness over competitive baselines, while producing more transparent reasoning traces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.04575v3">Are LLMs Stable Formal Logic Translators in Logical Reasoning Across Linguistically Diversified Texts?</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 Accepted by WWW2026
    </div>
    <details class="paper-abstract">
      Logical reasoning with large language models (LLMs) has received growing attention. One mainstream approach translates natural language into formal logic and then applies symbolic solvers for deduction. While effective in many tasks, these LLM-based translators often fail to generate consistent symbolic representations when the same concept appears in different linguistic forms. Such inconsistencies break logical coherence and lead to solver errors. However, most existing benchmarks lack this type of linguistic variation, which frequently occurs in real-world text, leaving the problem underexplored. To address this gap, we present SoLT, a benchmark that systematically rewrites reasoning datasets into diverse yet logically equivalent forms across multiple levels. Beyond evaluation, SoLT also provides a general method to enrich any dataset with linguistic diversity while preserving both meaning and logic. To further enhance the stability of LLM-based reasoning, we propose MenTaL, which explicitly guides models to build a concept-symbol mapping table during translation. By linking equivalent expressions to shared symbols, MenTaL maintains consistency and mitigates symbol drift. Experiments on SoLT demonstrate that LLMs indeed suffer from inconsistent symbol mapping under linguistic variation, leading to significant drops in reasoning accuracy. Meanwhile, applying MenTaL brings clear and stable performance improvements across diverse inputs. Overall, our findings reveal that overlooking linguistic diversity hides key weaknesses in LLM-based translators, and our work offers a step toward more reliable logical reasoning in varied real-world scenarios. Our code is available at https://github.com/wufeiwuwoshihua/LinguDiver.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22485v1">FraudShield: Knowledge Graph Empowered Defense for LLMs against Fraud Attacks</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 WWW 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been widely integrated into critical automated workflows, including contract review and job application processes. However, LLMs are susceptible to manipulation by fraudulent information, which can lead to harmful outcomes. Although advanced defense methods have been developed to address this issue, they often exhibit limitations in effectiveness, interpretability, and generalizability, particularly when applied to LLM-based applications. To address these challenges, we introduce FraudShield, a novel framework designed to protect LLMs from fraudulent content by leveraging a comprehensive analysis of fraud tactics. Specifically, FraudShield constructs and refines a fraud tactic-keyword knowledge graph to capture high-confidence associations between suspicious text and fraud techniques. The structured knowledge graph augments the original input by highlighting keywords and providing supporting evidence, guiding the LLM toward more secure responses. Extensive experiments show that FraudShield consistently outperforms state-of-the-art defenses across four mainstream LLMs and five representative fraud types, while also offering interpretable clues for the model's generations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22448v1">HeaPA: Difficulty-Aware Heap Sampling and On-Policy Query Augmentation for LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-01-30
    </div>
    <details class="paper-abstract">
      RLVR is now a standard way to train LLMs on reasoning tasks with verifiable outcomes, but when rollout generation dominates the cost, efficiency depends heavily on which prompts you sample and when. In practice, prompt pools are often static or only loosely tied to the model's learning progress, so uniform sampling can't keep up with the shifting capability frontier and ends up wasting rollouts on prompts that are already solved or still out of reach. Existing approaches improve efficiency through filtering, curricula, adaptive rollout allocation, or teacher guidance, but they typically assume a fixed pool-which makes it hard to support stable on-policy pool growth-or they add extra teacher cost and latency. We introduce HeaPA (Heap Sampling and On-Policy Query Augmentation), which maintains a bounded, evolving pool, tracks the frontier using heap-based boundary sampling, expands the pool via on-policy augmentation with lightweight asynchronous validation, and stabilizes correlated queries through topology-aware re-estimation of pool statistics and controlled reinsertion. Across two training corpora, two training recipes, and seven benchmarks, HeaPA consistently improves accuracy and reaches target performance with fewer computations while keeping wall-clock time comparable. Our analyses suggest these gains come from frontier-focused sampling and on-policy pool growth, with the benefits becoming larger as model scale increases. Our code is available at https://github.com/horizon-rl/HeaPA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22440v1">AI and My Values: User Perceptions of LLMs' Ability to Extract, Embody, and Explain Human Values from Casual Conversations</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 To appear in CHI '26
    </div>
    <details class="paper-abstract">
      Does AI understand human values? While this remains an open philosophical question, we take a pragmatic stance by introducing VAPT, the Value-Alignment Perception Toolkit, for studying how LLMs reflect people's values and how people judge those reflections. 20 participants texted a human-like chatbot over a month, then completed a 2-hour interview with our toolkit evaluating AI's ability to extract (pull details regarding), embody (make decisions guided by), and explain (provide proof of) human values. 13 participants left our study convinced that AI can understand human values. Participants found the experience insightful for self-reflection and found themselves getting persuaded by the AI's reasoning. Thus, we warn about "weaponized empathy": a potentially dangerous design pattern that may arise in value-aligned, yet welfare-misaligned AI. VAPT offers concrete artifacts and design implications to evaluate and responsibly build value-aligned conversational agents with transparency, consent, and safeguards as AI grows more capable and human-like into the future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22433v1">When LLM meets Fuzzy-TOPSIS for Personnel Selection through Automated Profile Analysis</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 10 pages, 8 figures. This paper has been peer-reviewed and published in IEEE Access. The arXiv version corresponds to the accepted author manuscript (AAM)
    </div>
    <details class="paper-abstract">
      In this highly competitive employment environment, the selection of suitable personnel is essential for organizational success. This study presents an automated personnel selection system that utilizes sophisticated natural language processing (NLP) methods to assess and rank software engineering applicants. A distinctive dataset was created by aggregating LinkedIn profiles that include essential features such as education, work experience, abilities, and self-introduction, further enhanced with expert assessments to function as standards. The research combines large language models (LLMs) with multicriteria decision-making (MCDM) theory to develop the LLM-TOPSIS framework. In this context, we utilized the TOPSIS method enhanced by fuzzy logic (Fuzzy TOPSIS) to address the intrinsic ambiguity and subjectivity in human assessments. We utilized triangular fuzzy numbers (TFNs) to describe criteria weights and scores, thereby addressing the ambiguity frequently encountered in candidate evaluations. For candidate ranking, the DistilRoBERTa model was fine-tuned and integrated with the fuzzy TOPSIS method, achieving rankings closely aligned with human expert evaluations and attaining an accuracy of up to 91% for the Experience attribute and the Overall attribute. The study underlines the potential of NLP-driven frameworks to improve recruitment procedures by boosting scalability, consistency, and minimizing prejudice. Future endeavors will concentrate on augmenting the dataset, enhancing model interpretability, and verifying the system in actual recruitment scenarios to better evaluate its practical applicability. This research highlights the intriguing potential of merging NLP with fuzzy decision-making methods in personnel selection, enabling scalable and unbiased solutions to recruitment difficulties.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.08140v5">Lost in Transmission: When and Why LLMs Fail to Reason Globally</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 39 pages; NeurIPS 2025, spotlight
    </div>
    <details class="paper-abstract">
      Despite their many successes, transformer-based large language models (LLMs) continue to struggle with tasks that require complex reasoning over large parts of their input. We argue that these failures arise due to capacity limits on the accurate flow of information within LLMs. To formalize this issue, we introduce the bounded attention prefix oracle (BAPO) model, a new computational framework that models bandwidth constraints on attention heads, the mechanism for internal communication in LLMs. We show that several important reasoning problems like graph reachability require high communication bandwidth for BAPOs to solve; we call these problems BAPO-hard. Our experiments corroborate our theoretical predictions: GPT-4o, Claude, and Gemini succeed on BAPO-easy tasks and fail even on relatively small BAPO-hard tasks. BAPOs also reveal another benefit of chain of thought (CoT): we prove that breaking down a task using CoT can turn any BAPO-hard problem into a BAPO-easy one. Our results offer principled explanations for key LLM failures and suggest directions for architectures and inference methods that mitigate bandwidth limits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22159v1">RedSage: A Cybersecurity Generalist LLM</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Accepted on ICLR 2026; Project page: https://risys-lab.github.io/RedSage/
    </div>
    <details class="paper-abstract">
      Cybersecurity operations demand assistant LLMs that support diverse workflows without exposing sensitive data. Existing solutions either rely on proprietary APIs with privacy risks or on open models lacking domain adaptation. To bridge this gap, we curate 11.8B tokens of cybersecurity-focused continual pretraining data via large-scale web filtering and manual collection of high-quality resources, spanning 28.6K documents across frameworks, offensive techniques, and security tools. Building on this, we design an agentic augmentation pipeline that simulates expert workflows to generate 266K multi-turn cybersecurity samples for supervised fine-tuning. Combined with general open-source LLM data, these resources enable the training of RedSage, an open-source, locally deployable cybersecurity assistant with domain-aware pretraining and post-training. To rigorously evaluate the models, we introduce RedSage-Bench, a benchmark with 30K multiple-choice and 240 open-ended Q&A items covering cybersecurity knowledge, skills, and tool expertise. RedSage is further evaluated on established cybersecurity benchmarks (e.g., CTI-Bench, CyberMetric, SECURE) and general LLM benchmarks to assess broader generalization. At the 8B scale, RedSage achieves consistently better results, surpassing the baseline models by up to +5.59 points on cybersecurity benchmarks and +5.05 points on Open LLM Leaderboard tasks. These findings demonstrate that domain-aware agentic augmentation and pre/post-training can not only enhance cybersecurity-specific expertise but also help to improve general reasoning and instruction-following. All models, datasets, and code are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.13187v3">"Not in My Backyard": LLMs Uncover Online and Offline Social Biases Against Homelessness</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Homelessness is a persistent social challenge, impacting millions worldwide. Over 876,000 people experienced homelessness (PEH) in the U.S. in 2025. Social bias is a significant barrier to alleviation, shaping public perception and influencing policymaking. Given that online textual media and offline city council discourse reflect and influence part of public opinion, it provides valuable insights to identify and track social biases against PEH. We present a new, manually-annotated multi-domain dataset compiled from Reddit, X (formerly Twitter), news articles, and city council meeting minutes across ten U.S. cities. Our 16-category multi-label taxonomy creates a challenging long-tail classification problem: some categories appear in less than 1% of samples, while others exceed 70%. We find that small human-annotated datasets (1,702 samples) are insufficient for training effective classifiers, whether used to fine-tune encoder models or as few-shot examples for LLMs. To address this, we use GPT-4.1 to generate pseudo-labels on a larger unlabeled corpus. Training on this expanded dataset enables even small encoder models (ModernBERT, 150M parameters) to achieve 35.23 macro-F1, approaching GPT-4.1's 41.57. This demonstrates that \textbf{data quantity matters more than model size}, enabling low-cost, privacy-preserving deployment without relying on commercial APIs. Our results reveal that negative bias against PEH is prevalent both offline and online (especially on Reddit), with "not in my backyard" narratives showing the highest engagement. These findings uncover a type of ostracism that directly impacts poverty-reduction policymaking and provide actionable insights for practitioners addressing homelessness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22132v1">Pay for Hints, Not Answers: LLM Shepherding for Cost-Efficient Inference</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) deliver state-of-the-art performance on complex reasoning tasks, but their inference costs limit deployment at scale. Small Language Models (SLMs) offer dramatic cost savings yet lag substantially in accuracy. Existing approaches - routing and cascading - treat the LLM as an all-or-nothing resource: either the query bypasses the LLM entirely, or the LLM generates a complete response at full cost. We introduce LLM Shepherding, a framework that requests only a short prefix (a hint) from the LLM and provides it to SLM. This simple mechanism is surprisingly effective for math and coding tasks: even hints comprising 10-30% of the full LLM response improve SLM accuracy significantly. Shepherding generalizes both routing and cascading, and it achieves lower cost under oracle decision-making. We develop a two-stage predictor that jointly determines whether a hint is needed and how many tokens to request. On the widely-used mathematical reasoning (GSM8K, CNK12) and code generation (HumanEval, MBPP) benchmarks, Shepherding reduces costs by 42-94% relative to LLM-only inference. Compared to state-of-the-art routing and cascading baselines, shepherding delivers up to 2.8x cost reduction while matching accuracy. To our knowledge, this is the first work to exploit token-level budget control for SLM-LLM collaboration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.09277v4">NeuroFaith: Evaluating LLM Self-Explanation Faithfulness via Internal Representation Alignment</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can generate plausible free text self-explanations to justify their answers. However, these natural language explanations may not accurately reflect the model's actual reasoning process, pinpointing a lack of faithfulness. Existing faithfulness evaluation methods rely primarily on behavioral tests or computational block analysis without examining the semantic content of internal neural representations. This paper proposes NeuroFaith, a flexible framework that measures the faithfulness of LLM free text self-explanation by identifying key concepts within explanations and mechanistically testing whether these concepts actually influence the model's predictions. We show the versatility of NeuroFaith across 2-hop reasoning and classification tasks. Additionally, we develop a linear faithfulness probe based on NeuroFaith to detect unfaithful self-explanations from representation space and improve faithfulness through steering. NeuroFaith provides a principled approach to evaluating and enhancing the faithfulness of LLM free text self-explanations, addressing critical needs for trustworthy AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17915v2">Think Locally, Explain Globally: Graph-Guided LLM Investigations via Local Reasoning and Belief Propagation</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      LLM agents excel when environments are mostly static and the needed information fits in a model's context window, but they often fail in open-ended investigations where explanations must be constructed by iteratively mining evidence from massive, heterogeneous operational data. These investigations exhibit hidden dependency structure: entities interact, signals co-vary, and the importance of a fact may only become clear after other evidence is discovered. Because the context window is bounded, agents must summarize intermediate findings before their significance is known, increasing the risk of discarding key evidence. ReAct-style agents are especially brittle in this regime. Their retrieve-summarize-reason loop makes conclusions sensitive to exploration order and introduces run-to-run non-determinism, producing a reliability gap where Pass-at-k may be high but Majority-at-k remains low. Simply sampling more rollouts or generating longer reasoning traces does not reliably stabilize results, since hypotheses cannot be autonomously checked as new evidence arrives and there is no explicit mechanism for belief bookkeeping and revision. In addition, ReAct entangles semantic reasoning with controller duties such as tool orchestration and state tracking, so execution errors and plan drift degrade reasoning while consuming scarce context. We address these issues by formulating investigation as abductive reasoning over a dependency graph and proposing EoG (Explanations over Graphs), a disaggregated framework in which an LLM performs bounded local evidence mining and labeling (cause vs symptom) while a deterministic controller manages traversal, state, and belief propagation to compute a minimal explanatory frontier. On a representative ITBench diagnostics task, EoG improves both accuracy and run-to-run consistency over ReAct baselines, including a 7x average gain in Majority-at-k entity F1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22050v1">MasalBench: A Benchmark for Contextual and Cross-Cultural Understanding of Persian Proverbs in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      In recent years, multilingual Large Language Models (LLMs) have become an inseparable part of daily life, making it crucial for them to master the rules of conversational language in order to communicate effectively with users. While previous work has evaluated LLMs' understanding of figurative language in high-resource languages, their performance in low-resource languages remains underexplored. In this paper, we introduce MasalBench, a comprehensive benchmark for assessing LLMs' contextual and cross-cultural understanding of Persian proverbs, which are a key component of conversation in this low-resource language. We evaluate eight state-of-the-art LLMs on MasalBench and find that they perform well in identifying Persian proverbs in context, achieving accuracies above 0.90. However, their performance drops considerably when tasked with identifying equivalent English proverbs, with the best model achieving 0.79 accuracy. Our findings highlight the limitations of current LLMs in cultural knowledge and analogical reasoning, and they provide a framework for assessing cross-cultural understanding in other low-resource languages. MasalBench is available at https://github.com/kalhorghazal/MasalBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.04328v2">OD-Stega: LLM-Based Relatively Secure Steganography via Optimized Distributions</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Accepted to EACL 2026
    </div>
    <details class="paper-abstract">
      We consider coverless steganography where a Large Language Model (LLM) is used to generate stego-texts in combination with arithmetic coding. An efficient method should embed secret bits in as few language tokens as possible while keeping the stego-text as natural as possible. We show that this problem is equivalent to maximizing the entropy of a replacement probability distribution of the next token generation, subject to a constraint on the divergence between the new distribution and the original one produced by the LLM. A closed-form solution is provided under either the KL divergence or the total variation constraint. Several important practical issues are also tackled: 1) An often-overlooked tokenization mismatch issue is resolved with a simple prompt selection approach, 2) The combination of the optimized distribution and the vocabulary truncation technique is considered, and 3) The incorporation of the proposed approach with existing (potentially non arithmetic coding based) techniques, e.g., the Discop technique.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22028v1">From Logits to Latents: Contrastive Representation Shaping for LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Most LLM unlearning methods aim to approximate retrain-from-scratch behaviors with minimal distribution shift, often via alignment-style objectives defined in the prediction space. While effective at reducing forgotten content generation, such approaches may act as suppression: forgotten concepts can persist in representations and remain entangled with retained knowledge. We introduce CLReg, a contrastive representation regularizer that identifies forget features while pushing them away from retain features, explicitly reducing forget-retain interference with minimal shifts on retain features. We provide first theoretical insights that relate representation shaping to entanglement reduction. Across unlearning benchmarks and LLMs of different sizes, CLReg decreases forget-retain representation entanglement that facilitates mainstream unlearning methods without positing extra privacy risks, inspiring future work that reshapes the representation space to remove forget concepts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22027v1">CAR-bench: Evaluating the Consistency and Limit-Awareness of LLM Agents under Real-World Uncertainty</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Existing benchmarks for Large Language Model (LLM) agents focus on task completion under idealistic settings but overlook reliability in real-world, user-facing applications. In domains, such as in-car voice assistants, users often issue incomplete or ambiguous requests, creating intrinsic uncertainty that agents must manage through dialogue, tool use, and policy adherence. We introduce CAR-bench, a benchmark for evaluating consistency, uncertainty handling, and capability awareness in multi-turn, tool-using LLM agents in an in-car assistant domain. The environment features an LLM-simulated user, domain policies, and 58 interconnected tools spanning navigation, productivity, charging, and vehicle control. Beyond standard task completion, CAR-bench introduces Hallucination tasks that test agents' limit-awareness under missing tools or information, and Disambiguation tasks that require resolving uncertainty through clarification or internal information gathering. Baseline results reveal large gaps between occasional and consistent success on all task types. Even frontier reasoning LLMs achieve less than 50% consistent pass rate on Disambiguation tasks due to premature actions, and frequently violate policies or fabricate information to satisfy user requests in Hallucination tasks, underscoring the need for more reliable and self-aware LLM agents in real-world settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22025v1">When "Better" Prompts Hurt: Evaluation-Driven Iteration for LLM Applications</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Evaluating Large Language Model (LLM) applications differs from traditional software testing because outputs are stochastic, high-dimensional, and sensitive to prompt and model changes. We present an evaluation-driven workflow - Define, Test, Diagnose, Fix - that turns these challenges into a repeatable engineering loop. We introduce the Minimum Viable Evaluation Suite (MVES), a tiered set of recommended evaluation components for (i) general LLM applications, (ii) retrieval-augmented generation (RAG), and (iii) agentic tool-use workflows. We also synthesize common evaluation methods (automated checks, human rubrics, and LLM-as-judge) and discuss known judge failure modes. In reproducible local experiments (Ollama; Llama 3 8B Instruct and Qwen 2.5 7B Instruct), we observe that a generic "improved" prompt template can trade off behaviors: on our small structured suites, extraction pass rate decreased from 100% to 90% and RAG compliance from 93.3% to 80% for Llama 3 when replacing task-specific prompts with generic rules, while instruction-following improved. These findings motivate evaluation-driven prompt iteration and careful claim calibration rather than universal prompt recipes. All test suites, harnesses, and results are included for reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19120v2">RobustExplain: Evaluating Robustness of LLM-Based Explanation Agents for Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 8 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to generate natural-language explanations in recommender systems, acting as explanation agents that reason over user behavior histories. While prior work has focused on explanation fluency and relevance under fixed inputs, the robustness of LLM-generated explanations to realistic user behavior noise remains largely unexplored. In real-world web platforms, interaction histories are inherently noisy due to accidental clicks, temporal inconsistencies, missing values, and evolving preferences, raising concerns about explanation stability and user trust. We present RobustExplain, the first systematic evaluation framework for measuring the robustness of LLM-generated recommendation explanations. RobustExplain introduces five realistic user behavior perturbations evaluated across multiple severity levels and a multi-dimensional robustness metric capturing semantic, keyword, structural, and length consistency. Our goal is to establish a principled, task-level evaluation framework and initial robustness baselines, rather than to provide a comprehensive leaderboard across all available LLMs. Experiments on four representative LLMs (7B--70B) show that current models exhibit only moderate robustness, with larger models achieving up to 8% higher stability. Our results establish the first robustness benchmarks for explanation agents and highlight robustness as a critical dimension for trustworthy, agent-driven recommender systems at web scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22008v1">LANCER: LLM Reranking for Nugget Coverage</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 ECIR 2026
    </div>
    <details class="paper-abstract">
      Unlike short-form retrieval-augmented generation (RAG), such as factoid question answering, long-form RAG requires retrieval to provide documents covering a wide range of relevant information. Automated report generation exemplifies this setting: it requires not only relevant information but also a more elaborate response with comprehensive information. Yet, existing retrieval methods are primarily optimized for relevance ranking rather than information coverage. To address this limitation, we propose LANCER, an LLM-based reranking method for nugget coverage. LANCER predicts what sub-questions should be answered to satisfy an information need, predicts which documents answer these sub-questions, and reranks documents in order to provide a ranked list covering as many information nuggets as possible. Our empirical results show that LANCER enhances the quality of retrieval as measured by nugget coverage metrics, achieving higher $α$-nDCG and information coverage than other LLM-based reranking methods. Our oracle analysis further reveals that sub-question generation plays an essential role.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19121v2">LLMs as Orchestrators: Constraint-Compliant Multi-Agent Optimization for Recommendation Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 8 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Recommendation systems must optimize multiple objectives while satisfying hard business constraints such as fairness and coverage. For example, an e-commerce platform may require every recommendation list to include items from multiple sellers and at least one newly listed product; violating such constraints--even once--is unacceptable in production. Prior work on multi-objective recommendation and recent LLM-based recommender agents largely treat constraints as soft penalties or focus on item scoring and interaction, leading to frequent violations in real-world deployments. How to leverage LLMs for coordinating constrained optimization in recommendation systems remains underexplored. We propose DualAgent-Rec, an LLM-coordinated dual-agent framework for constrained multi-objective e-commerce recommendation. The framework separates optimization into an Exploitation Agent that prioritizes accuracy under hard constraints and an Exploration Agent that promotes diversity through unconstrained Pareto search. An LLM-based coordinator adaptively allocates resources between agents based on optimization progress and constraint satisfaction, while an adaptive epsilon-relaxation mechanism guarantees feasibility of final solutions. Experiments on the Amazon Reviews 2023 dataset demonstrate that DualAgent-Rec achieves 100% constraint satisfaction and improves Pareto hypervolume by 4-6% over strong baselines, while maintaining competitive accuracy-diversity trade-offs. These results indicate that LLMs can act as effective orchestration agents for deployable and constraint-compliant recommendation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21996v1">Mechanistic Data Attribution: Tracing the Training Origins of Interpretable LLM Units</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      While Mechanistic Interpretability has identified interpretable circuits in LLMs, their causal origins in training data remain elusive. We introduce Mechanistic Data Attribution (MDA), a scalable framework that employs Influence Functions to trace interpretable units back to specific training samples. Through extensive experiments on the Pythia family, we causally validate that targeted intervention--removing or augmenting a small fraction of high-influence samples--significantly modulates the emergence of interpretable heads, whereas random interventions show no effect. Our analysis reveals that repetitive structural data (e.g., LaTeX, XML) acts as a mechanistic catalyst. Furthermore, we observe that interventions targeting induction head formation induce a concurrent change in the model's in-context learning (ICL) capability. This provides direct causal evidence for the long-standing hypothesis regarding the functional link between induction heads and ICL. Finally, we propose a mechanistic data augmentation pipeline that consistently accelerates circuit convergence across model scales, providing a principled methodology for steering the developmental trajectories of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21986v1">SpecTran: Spectral-Aware Transformer-based Adapter for LLM-Enhanced Sequential Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Traditional sequential recommendation (SR) models learn low-dimensional item ID embeddings from user-item interactions, often overlooking textual information such as item titles or descriptions. Recent advances in Large Language Models (LLMs) have inspired a surge of research that encodes item textual information with high-dimensional semantic embeddings, and designs transformation methods to inject such embeddings into SR models. These embedding transformation strategies can be categorized into two types, both of which exhibits notable drawbacks: 1) adapter-based methods suffer from pronounced dimension collapse, concentrating information into a few dominant dimensions; 2) SVD-based methods are rigid and manual, considering only a few principal spectral components while discarding rich information in the remaining spectrum. To address these limitations, we propose SpecTran, a spectral-aware transformer-based adapter that operates in the spectral domain, attending to the full spectrum to select and aggregates informative components. A learnable spectral-position encoding injects singular-value cues as an inductive bias, guiding attention toward salient spectral components and promoting diversity across embedding dimensions. Across four real-world datasets and three SR backbones, it consistently outperforms strong baselines, achieving an average improvement of 9.17%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21972v1">Learning Decentralized LLM Collaboration with Multi-Agent Actor Critic</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Recent work has explored optimizing LLM collaboration through Multi-Agent Reinforcement Learning (MARL). However, most MARL fine-tuning approaches rely on predefined execution protocols, which often require centralized execution. Decentralized LLM collaboration is more appealing in practice, as agents can run inference in parallel with flexible deployments. Also, current approaches use Monte Carlo methods for fine-tuning, which suffer from high variance and thus require more samples to train effectively. Actor-critic methods are prevalent in MARL for dealing with these issues, so we developed Multi-Agent Actor-Critic (MAAC) methods to optimize decentralized LLM collaboration. In this paper, we analyze when and why these MAAC methods are beneficial. We propose 2 MAAC approaches, \textbf{CoLLM-CC} with a \textbf{C}entralized \textbf{C}ritic and \textbf{CoLLM-DC} with \textbf{D}ecentralized \textbf{C}ritics. Our experiments across writing, coding, and game-playing domains show that Monte Carlo methods and CoLLM-DC can achieve performance comparable to CoLLM-CC in short-horizon and dense-reward settings. However, they both underperform CoLLM-CC on long-horizon or sparse-reward tasks, where Monte Carlo methods require substantially more samples and CoLLM-DC struggles to converge. Our code is available at https://github.com/OpenMLRL/CoMLRL/releases/tag/v1.3.2.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21963v1">Industrialized Deception: The Collateral Effects of LLM-Generated Misinformation on Digital Ecosystems</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Accepted at ACM TheWebConf '26 Companion
    </div>
    <details class="paper-abstract">
      Generative AI and misinformation research has evolved since our 2024 survey. This paper presents an updated perspective, transitioning from literature review to practical countermeasures. We report on changes in the threat landscape, including improved AI-generated content through Large Language Models (LLMs) and multimodal systems. Central to this work are our practical contributions: JudgeGPT, a platform for evaluating human perception of AI-generated news, and RogueGPT, a controlled stimulus generation engine for research. Together, these tools form an experimental pipeline for studying how humans perceive and detect AI-generated misinformation. Our findings show that detection capabilities have improved, but the competition between generation and detection continues. We discuss mitigation strategies including LLM-based detection, inoculation approaches, and the dual-use nature of generative AI. This work contributes to research addressing the adverse impacts of AI on information quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.12481v3">SAC-GLAM: Improving Online RL for LLM agents with Soft Actor-Critic and Hindsight Relabeling</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 This work has been presented at the IMOL workshop at NeurIPS 2025 (https://neurips.cc/virtual/2024/101058)
    </div>
    <details class="paper-abstract">
      The past years have seen Large Language Models (LLMs) strive not only as generative models but also as agents solving textual sequential decision-making tasks. When facing complex environments where their zero-shot abilities are insufficient, recent work showed online Reinforcement Learning (RL) could be used for the LLM agent to discover and learn efficient strategies interactively. However, most prior work sticks to on-policy algorithms, which greatly reduces the scope of methods such agents could use for both exploration and exploitation, such as experience replay and hindsight relabeling. Yet, such methods may be key for LLM learning agents, and in particular when designing autonomous intrinsically motivated agents sampling and pursuing their own goals (i.e. autotelic agents). This paper presents and studies an adaptation of Soft Actor-Critic and hindsight relabeling to LLM agents. Our method not only paves the path towards autotelic LLM agents that learn online but can also outperform on-policy methods in more classic multi-goal RL environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.06725v3">WorldLLM: Improving LLMs' world modeling using curiosity-driven theory-making</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 This project's code can be found at https://github.com/flowersteam/WorldLLM. This project was presented at RLDM 2025 (https://rldm.org/)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) possess general world knowledge but often struggle to generate precise predictions in structured, domain-specific contexts such as simulations. These limitations arise from their inability to ground their broad, unstructured understanding in specific environments. To address this, we present WorldLLM, a framework that enhances LLM-based world modeling by combining Bayesian inference and autonomous active exploration with reinforcement learning. WorldLLM leverages the in-context learning abilities of LLMs to guide an LLM-based world model's predictions using natural language hypotheses given in its prompt. These hypotheses are iteratively refined through a Bayesian inference framework that leverages a second LLM as the proposal distribution given collected evidence. This evidence is collected using a curiosity-driven reinforcement learning policy that explores the environment to find transitions with a low log-likelihood under our LLM-based predictive model using the current hypotheses. By alternating between refining hypotheses and collecting new evidence, our framework autonomously drives continual improvement of the predictions. Our experiments demonstrate the effectiveness of WorldLLM in a textual game environment that requires agents to manipulate and combine objects. The framework not only enhances predictive accuracy, but also generates human-interpretable theories of environment dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.17254v2">Near-Optimal Online Deployment and Routing for Streaming LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      The rapid pace at which new large language models (LLMs) appear, and older ones become obsolete, forces providers to manage a streaming inventory under a strict concurrency cap and per-query cost budgets. We cast this as an online decision problem that couples stage-wise deployment (at fixed maintenance windows) with per-query routing among live models. We introduce StageRoute, a hierarchical algorithm that (i) optimistically selects up to $M_{\max}$ models for the next stage using reward upper-confidence and cost lower-confidence bounds, and (ii) routes each incoming query by solving a budget- and throughput-constrained bandit subproblem over the deployed set. We prove a regret of $\tilde{\mathcal{O}}(T^{2/3})$ with a matching lower bound, establishing near-optimality, and validate the theory empirically: StageRoute tracks a strong oracle under tight budgets across diverse workloads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11479v2">Health Facility Location in Ethiopia: Leveraging LLMs to Integrate Expert Knowledge into Algorithmic Planning</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Ethiopia's Ministry of Health is upgrading health posts to improve access to essential services, particularly in rural areas. Limited resources, however, require careful prioritization of which facilities to upgrade to maximize population coverage while accounting for diverse expert and stakeholder preferences. In collaboration with the Ethiopian Public Health Institute and Ministry of Health, we propose a hybrid framework that systematically integrates expert knowledge with optimization techniques. Classical optimization methods provide theoretical guarantees but require explicit, quantitative objectives, whereas stakeholder criteria are often articulated in natural language and difficult to formalize. To bridge these domains, we develop the Large language model and Extended Greedy (LEG) framework. Our framework combines a provable approximation algorithm for population coverage optimization with LLM-driven iterative refinement that incorporates human-AI alignment to ensure solutions reflect expert qualitative guidance while preserving coverage guarantees. Experiments on real-world data from three Ethiopian regions demonstrate the framework's effectiveness and its potential to inform equitable, data-driven health system planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21909v1">From Meta-Thought to Execution: Cognitively Aligned Post-Training for Generalizable and Reliable LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Current LLM post-training methods optimize complete reasoning trajectories through Supervised Fine-Tuning (SFT) followed by outcome-based Reinforcement Learning (RL). While effective, a closer examination reveals a fundamental gap: this approach does not align with how humans actually solve problems. Human cognition naturally decomposes problem-solving into two distinct stages: first acquiring abstract strategies (i.e., meta-knowledge) that generalize across problems, then adapting them to specific instances. In contrast, by treating complete trajectories as basic units, current methods are inherently problem-centric, entangling abstract strategies with problem-specific execution. To address this misalignment, we propose a cognitively-inspired framework that explicitly mirrors the two-stage human cognitive process. Specifically, Chain-of-Meta-Thought (CoMT) focuses supervised learning on abstract reasoning patterns without specific executions, enabling acquisition of generalizable strategies. Confidence-Calibrated Reinforcement Learning (CCRL) then optimizes task adaptation via confidence-aware rewards on intermediate steps, preventing overconfident errors from cascading and improving execution reliability. Experiments across four models and eight benchmarks show 2.19\% and 4.63\% improvements in-distribution and out-of-distribution respectively over standard methods, while reducing training time by 65-70% and token consumption by 50%, demonstrating that aligning post-training with human cognitive principles yields not only superior generalization but also enhanced training efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21895v1">Learn-to-Distance: Distance Learning for Detecting LLM-Generated Text</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Accepted by ICLR2026
    </div>
    <details class="paper-abstract">
      Modern large language models (LLMs) such as GPT, Claude, and Gemini have transformed the way we learn, work, and communicate. Yet, their ability to produce highly human-like text raises serious concerns about misinformation and academic integrity, making it an urgent need for reliable algorithms to detect LLM-generated content. In this paper, we start by presenting a geometric approach to demystify rewrite-based detection algorithms, revealing their underlying rationale and demonstrating their generalization ability. Building on this insight, we introduce a novel rewrite-based detection algorithm that adaptively learns the distance between the original and rewritten text. Theoretically, we demonstrate that employing an adaptively learned distance function is more effective for detection than using a fixed distance. Empirically, we conduct extensive experiments with over 100 settings, and find that our approach demonstrates superior performance over baseline algorithms in the majority of scenarios. In particular, it achieves relative improvements from 57.8\% to 80.6\% over the strongest baseline across different target LLMs (e.g., GPT, Claude, and Gemini).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21894v1">Not All Code Is Equal: A Data-Centric Study of Code Complexity and LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 16 pages, 5 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly exhibit strong reasoning abilities, often attributed to their capacity to generate chain-of-thought-style intermediate reasoning. Recent work suggests that exposure to code can further enhance these skills, but existing studies largely treat code as a generic training signal, leaving open the question of which properties of code actually contribute to improved reasoning. To address this gap, we study the structural complexity of code, which captures control flow and compositional structure that may shape how models internalise multi-step reasoning during fine-tuning. We examine two complementary settings: solution-driven complexity, where complexity varies across multiple solutions to the same problem, and problem-driven complexity, where complexity reflects variation in the underlying tasks. Using cyclomatic complexity and logical lines of code to construct controlled fine-tuning datasets, we evaluate a range of open-weight LLMs on diverse reasoning benchmarks. Our findings show that although code can improve reasoning, structural properties strongly determine its usefulness. In 83% of experiments, restricting fine-tuning data to a specific structural complexity range outperforms training on structurally diverse code, pointing to a data-centric path for improving reasoning beyond scaling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21879v1">astra-langchain4j: Experiences Combining LLMs and Agent Programming</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Given the emergence of Generative AI over the last two years and the increasing focus on Agentic AI as a form of Multi-Agent System it is important to explore both how such technologies can impact the use of traditional Agent Toolkits and how the wealth of experience encapsulated in those toolkits can influence the design of the new agentic platforms. This paper presents an overview of our experience developing a prototype large language model (LLM) integration for the ASTRA programming language. It presents a brief overview of the toolkit, followed by three example implementations, concluding with a discussion of the experiences garnered through the examples.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02553v3">SimpleMem: Efficient Lifelong Memory for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      To support long-term interaction in complex environments, LLM agents require memory systems that manage historical experiences. Existing approaches either retain full interaction histories via passive context extension, leading to substantial redundancy, or rely on iterative reasoning to filter noise, incurring high token costs. To address this challenge, we introduce SimpleMem, an efficient memory framework based on semantic lossless compression. We propose a three-stage pipeline designed to maximize information density and token utilization: (1) Semantic Structured Compression, which distills unstructured interactions into compact, multi-view indexed memory units; (2) Online Semantic Synthesis, an intra-session process that instantly integrates related context into unified abstract representations to eliminate redundancy; and (3) Intent-Aware Retrieval Planning, which infers search intent to dynamically determine retrieval scope and construct precise context efficiently. Experiments on benchmark datasets show that our method consistently outperforms baseline approaches in accuracy, retrieval efficiency, and inference cost, achieving an average F1 improvement of 26.4% in LoCoMo while reducing inference-time token consumption by up to 30-fold, demonstrating a superior balance between performance and efficiency. Code is available at https://github.com/aiming-lab/SimpleMem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07568v2">d3LLM: Ultra-Fast Diffusion LLM using Pseudo-Trajectory Distillation</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Diffusion large language models (dLLMs) offer capabilities beyond those of autoregressive (AR) LLMs, such as parallel decoding and random-order generation. However, realizing these benefits in practice is non-trivial, as dLLMs inherently face an accuracy-parallelism trade-off. Despite increasing interest, existing methods typically focus on only one-side of the coin, targeting either efficiency or performance. To address this limitation, we propose d3LLM (Pseudo-Distilled Diffusion Large Language Model), striking a balance between accuracy and parallelism: (i) during training, we introduce pseudo-trajectory distillation to teach the model which tokens can be decoded confidently at early steps, thereby improving parallelism; (ii) during inference, we employ entropy-based multi-block decoding with a KV-cache refresh mechanism to achieve high parallelism while maintaining accuracy. To better evaluate dLLMs, we also introduce AUP (Accuracy Under Parallelism), a new metric that jointly measures accuracy and parallelism. Experiments demonstrate that our d3LLM achieves up to 10$\times$ speedup over vanilla LLaDA/Dream and 5$\times$ speedup over AR models without much accuracy drop. Our code is available at https://github.com/hao-ai-lab/d3LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21876v1">LLM-Driven Scenario-Aware Planning for Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Hybrid planner switching framework (HPSF) for autonomous driving needs to reconcile high-speed driving efficiency with safe maneuvering in dense traffic. Existing HPSF methods often fail to make reliable mode transitions or sustain efficient driving in congested environments, owing to heuristic scene recognition and low-frequency control updates. To address the limitation, this paper proposes LAP, a large language model (LLM) driven, adaptive planning method, which switches between high-speed driving in low-complexity scenes and precise driving in high-complexity scenes, enabling high qualities of trajectory generation through confined gaps. This is achieved by leveraging LLM for scene understanding and integrating its inference into the joint optimization of mode configuration and motion planning. The joint optimization is solved using tree-search model predictive control and alternating minimization. We implement LAP by Python in Robot Operating System (ROS). High-fidelity simulation results show that the proposed LAP outperforms other benchmarks in terms of both driving time and success rate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21864v1">KnowBias: Mitigating Social Bias in LLMs via Know-Bias Neuron Enhancement</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit social biases that reinforce harmful stereotypes, limiting their safe deployment. Most existing debiasing methods adopt a suppressive paradigm by modifying parameters, prompts, or neurons associated with biased behavior; however, such approaches are often brittle, weakly generalizable, data-inefficient, and prone to degrading general capability. We propose \textbf{KnowBias}, a lightweight and conceptually distinct framework that mitigates bias by strengthening, rather than suppressing, neurons encoding bias-knowledge. KnowBias identifies neurons encoding bias knowledge using a small set of bias-knowledge questions via attribution-based analysis, and selectively enhances them at inference time. This design enables strong debiasing while preserving general capabilities, generalizes across bias types and demographics, and is highly data efficient, requiring only a handful of simple yes/no questions and no retraining. Experiments across multiple benchmarks and LLMs demonstrate consistent state-of-the-art debiasing performance with minimal utility degradation. Data and code are available at https://github.com/JP-25/KnowBias.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.05029v3">Reputation as a Solution to Cooperation Collapse in LLM-based MASs</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Published as a conference paper at AAMAS 2026
    </div>
    <details class="paper-abstract">
      Cooperation has long been a fundamental topic in both human society and AI systems. However, recent studies indicate that the collapse of cooperation may emerge in multi-agent systems (MASs) driven by large language models (LLMs). To address this challenge, we explore reputation systems as a remedy. We propose RepuNet, a dynamic, dual-level reputation framework that models both agent-level reputation dynamics and system-level network evolution. Specifically, driven by direct interactions and indirect gossip, agents form reputations for both themselves and their peers, and decide whether to connect or disconnect other agents for future interactions. Through three distinct scenarios, we show that RepuNet effectively avoids cooperation collapse, promoting and sustaining cooperation in LLM-based MASs. Moreover, we find that reputation systems can give rise to rich emergent behaviors in LLM-based MASs, such as the formation of cooperative clusters, the social isolation of exploitative agents, and the preference for sharing positive gossip rather than negative ones. The GitHub repository for our project can be accessed via the following link: https://github.com/RGB-0000FF/RepuNet.
    </details>
</div>
