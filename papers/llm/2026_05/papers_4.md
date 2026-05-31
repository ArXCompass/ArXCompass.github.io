# llm - 2026_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- Part 4
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20815v1">GraphRAG on Consumer Hardware: Benchmarking Local LLMs for Healthcare EHR Schema Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 9 pages, 1 figure, 5 tables
    </div>
    <details class="paper-abstract">
      Graph-based Retrieval Augmented Generation (GraphRAG) extends retrieval-augmented generation to support structured reasoning over complex corpora, but its reliability under resource-constrained, privacy-sensitive deployments remains unclear. In healthcare, where Electronic Health Record (EHR) data is complex and strictly regulated, reliance on cloud-based large language models (LLMs) introduces challenges in cost, latency, and compliance. In this work, we present a systematic evaluation of GraphRAG for EHR schema retrieval using locally deployed open-source LLMs. We implement the Microsoft GraphRAG pipeline on real-world EHR schema documentation and benchmark four models, including Llama 3.1 (8B), Mistral (7B), Qwen 2.5 (7B), and Phi-4-mini (3.8B), each deployed via Ollama on a single consumer GPU (8 GB VRAM). We evaluate indexing efficiency, knowledge graph construction, query latency, answer quality, and hallucination under both global and local retrieval modes. Our results reveal substantial differences: Llama 3.1 produces the richest knowledge graph (1,172 entities), Qwen 2.5 achieves the best answer quality (3.3/5), Phi-4-mini fails to complete the pipeline due to structured-output errors, and Mistral exhibits degenerate repetition behavior. We further show that GraphRAG exhibits a practical capacity threshold, where models below approximately 7B parameters fail to reliably produce valid structured outputs and cannot complete the pipeline. In addition, indexing and answer quality are decoupled across models, and local retrieval consistently outperforms global summarization in both latency and factual grounding, with reduced hallucination. These findings demonstrate that GraphRAG is feasible on consumer hardware while highlighting the importance of model selection and retrieval design for robust deployment in regulated settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.10807v3">LLMs for Secure Hardware Design and Related Problems: Opportunities and Challenges</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 Accepted for 2026 IEEE Computer Society Annual Symposium on VLSI (ISVLSI)
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into Electronic Design Automation (EDA) and hardware security is rapidly reshaping the semiconductor industry. While LLMs offer unprecedented capabilities in generating Register Transfer Level (RTL) code, automating testbenches, and bridging the semantic gap between high-level specifications and silicon, they simultaneously introduce severe vulnerabilities. This comprehensive review provides an in-depth analysis of the state-of-the-art in LLM-driven hardware design, organized around key advancements in EDA synthesis, hardware trust, design for security, and education. We systematically expand on the methodologies of recent breakthroughs -- from reasoning-driven synthesis and multi-agent vulnerability extraction to data contamination and adversarial machine learning (ML) evasion. We integrate general discussions on critical countermeasures, such as dynamic benchmarking to combat data memorization and aggressive red-teaming for robust security assessment. Finally, we synthesize cross-cutting lessons learned to guide future research toward secure, trustworthy, and autonomous design ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20809v1">Refining and Reusing Annotation Guidelines for LLM Annotation</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 14 pages, 7 figures. Accepted to the ACL 2026 Main Conference
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) demonstrate remarkable performance on zero-shot annotation tasks, they often struggle with the specialized conventions of gold-standard benchmarks. We propose the systematic reuse and refinement of annotation guidelines as an alignment mechanism, introducing an iterative moderation framework that simulates the early phases of annotation projects. We evaluate three hypotheses: (1) the efficacy of guideline integration, (2) the advantage of reasoning optimized models, and (3) the viability of moderation under minimal supervision. Testing across biomedical NER tasks (NCBI Disease, BC5CDR, BioRED) with three LLM families (GPT, Gemini, DeepSeek), our results empirically confirm all three hypotheses. While the iterative moderation framework shows good potential in effectively refining guidelines, our analysis also reveals substantial room for improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20767v1">The Illusion of Intervention: Your LLM-Simulated Experiment is an Observational Study</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show potential as simulators of human behavior, offering a scalable way to study responses to interventions. However, because LLMs are trained largely on observational data, interventions in experiments with LLM-simulated synthetic users can induce unintended shifts in latent user attributes, causing user drift where the implicit simulated population differs across treatment conditions, potentially distorting effect estimates. We formalize the confounding or selection bias that can arise due to user drift and show how intervention-dependent shifts can inflate or attenuate observed differences in user responses under intervention. To diagnose confounding, we propose using negative control outcomes--attributes that should remain invariant under intervention--to identify distribution shifts across intervention conditions, providing evidence of user drift. To mitigate drift, we study adjusting the persona specification by eliciting additional confounders, finding that targeted, setting-relevant confounders can substantially reduce bias across survey-style and multi-turn agent evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20759v1">Rethinking Fraud Safety Evaluation: Multi-Round Attacks Reveal Safety-Utility Tradeoffs in Graph-Context LLM Defenders</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 19 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Single-turn safety evaluation is a poor proxy for real fraud defense, where attackers escalate across multiple rounds. This paper evaluates fraud defenders under replay and adaptive multi-round attacks and measures when a defender refuses, not just whether it eventually refuses. On a frozen multi-round suite built from Fraud-R1, graph-context defenders improve early safe refusal relative to text-only baselines under both replay and adaptive fraud pressure, but they also produce substantially more benign over-refusal. Direct probing of the trained graph encoder, together with paired shuffle-risk ablations on both fraud and benign sides replicated across two seeds on the Qwen-1.5B backbone, localises this cost to how the defender LLM consumes structured context rather than to graph-encoder quality: the encoder cleanly separates fraud from benign, while the LLM responds primarily to the presence of structured graph fields and only secondarily, and asymmetrically, to risk-score magnitude. Temporal graph context is directionally stronger than static and significantly better grounded, but is not yet conclusively superior on the main refusal metrics. The contribution is evaluative and measurement-oriented: robust fraud assessment must be multi-round, must report refusal timing, must account for benign false positives alongside fraud-side safety gains, and must localize observed costs to the graph signal or to how the LLM consumes it.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20734v1">An Application-Layer Multi-Modal Covert-Channel Reference Monitor for LLM Agent Egress</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      A large language model (LLM) agent that sends messages can leak data inside them. Destination allowlists and content scanners do not police whether an otherwise-benign payload is itself a covert channel: a compromised agent encodes bits in zero-width characters, homoglyphs, whitespace, base64, JavaScript Object Notation (JSON) key ordering, message timing or size -- and, in binary egress, in least-significant-bit (LSB) pixel planes, per-image mean luminance, inter-image sequence permutation, ultrasonic tones, or audible-band sonified data. Our egress reference monitor has three contributions. (i) A text pipeline of ten capacity-reducing stages, a per-sink leaky-bucket capacity ledger, and a staged posture that enforces lossless stages from day one. (ii) Two media scramblers (a Fourier-domain audio band-limiter and a red-green-blue (RGB) image bit-depth and mean-luminance bucketer) gated by a boot-time cryptographic legitimacy attestation: an auditor publishes at boot the trusted Ed25519 keys and {kind, data-class} pairs; only payloads with a verifying signature for an authorized class are exempt. The attestation sidesteps the intractable content-based discrimination between real media and data sonified or rasterized as a carrier; unsigned media is suspect by default; a content-addressed canonicalizer closes the inter-image permutation channel. (iii) Residual capacity is the Miller--Madow corrected mutual information between embedded and recovered bits (zero when destroyed), measured by an adversarial ensemble of fifteen working encoders across text, image and audio. The reference implementation drives residual capacity to zero on every destroyable channel and to a stated bound on the one (per-image mean luminance) that cannot be destroyed without ruining the image.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20706v1">Llamas on the Web: Memory-Efficient, Performance-Portable, and Multi-Precision LLM Inference with WebGPU</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 19 pages, 11 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Running language models in the browser presents a unique opportunity to build efficient, private, and portable AI applications, but requires contending with constrained memory availability and heterogeneous hardware targets. To realize this opportunity, we present Llamas on the Web (LlamaWeb), a WebGPU backend for llama$.$cpp that enables memory-efficient and performance-portable LLM inference across a wide range of model weight formats in the browser. Our design significantly reduces memory overhead through static memory planning and efficient model loading, addresses cross-device variability through a tunable kernel library, and introduces templated GPU kernels that support performant implementations of numerous quantization formats, enabling broad model support and extensibility to new formats. We evaluate LlamaWeb on 16 devices from 8 vendors, collecting data from 10 language models and four model weight formats. We compare LlamaWeb against existing browser-based LLM frameworks and find that LlamaWeb requires 29-33% less memory across several combinations of device, browser, and operating system. We also evaluate LlamaWeb's performance against these frameworks and find that it increases decode throughput by 45-69% across four GPUs from separate vendors. In addition, we compare LlamaWeb's performance against other llama$.$cpp backends, where it is competitive with and even beats vendor-specific backend performance on some devices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21537v1">Articulate but Wrong: Self-Review Failures in LLM-Based Code Modernization</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 11 pages, 6 figures, 2 tables. Corpus, oracle, output extractor, prompts, harness, self-review probe, and all 1,980 + 1,979 raw model outputs released as supplementary material at https://zenodo.org/records/20300861
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents are increasingly used to migrate legacy code to modern stacks. We ask a deceptively simple question: when an LLM modernizes legacy code, can the same model be relied upon to recognize when its own output silently changes observable behavior? We run 1,980 real modernization calls across 11 production LLMs from 7 distinct families on a balanced 60-snippet legacy-Python-2 corpus, evaluate every output with a type-strict behavioral oracle, and then ask each model to judge whether its own output preserves behavior. We report four findings. (1) Semantic-preservation drift is prevalent and sharply separable from a cleanly-controlled baseline: semantic-trap snippets drift in 39.7% of attempts versus 7.0% on benign-control code that requires no real modernization (+32.7 percentage points; n=660 each). (2) Drift concentrates on specific snippets that fail across models: pairwise model agreement on which snippets are hard is high (mean Pearson r=0.52), and a small core of numeric-semantics snippets fails for nearly every model and every prompt phrasing. (3) Self-review by the producing model is not a reliable safety net: across all semantic drift cases, 31.7% are silently endorsed by the same model that produced them (83/262), and the per-model self-miss rate is strongly bimodal -- ranging from 0% on five models to 100% on one widely deployed model -- with several models explicitly articulating the very Py2/Py3 semantic distinction that broke their output, then declaring behavior preserved. (4) Drift rate is non-monotone in model capability and price: per-model rates range 5.6%-46.7% and do not track model capability cleanly, indicating the failure is task-structural rather than driven by model scale. All code, prompts, the 60-snippet corpus, the behavioral oracle, the output extractor, and the raw model outputs are released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20641v1">Trusted Weights, Treacherous Optimizations? Optimization-Triggered Backdoor Attacks on LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 20 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Inference optimization is a vital technique for deploying LLMs at scale. Compilation is the most widely adopted optimization technique for LLMs. While it assumes semantic equivalence between the original and compiled graphs, we first uncover its numerical side effects can be maliciously exploited to implant stealthy backdoors in LLMs. We propose a unified optimization-triggered attack framework comprising two complementary strategies. Without any modification to the compiler or hardware, one strategy flips predictions for specific inputs only when the model is compiled, while the other uses a universal trigger that remains dormant under uncompiled execution but hijacks arbitrary inputs once compilation optimization is applied. Both attacks bypass standard safety evaluations run without compilation. We empirically demonstrate that these optimization-triggered backdoors achieve attack success rates averaging 90% across four mainstream open-source LLMs and four tasks, while clean accuracy is preserved at nearly 100% under all settings. Our findings reveal a novel attack surface at the intersection of optimization and security in the LLM deployment pipeline, and we investigate practical defenses to mitigate this threat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.06007v2">MASFactory: A Graph-centric Framework for Orchestrating LLM-Based Multi-Agent Systems with Vibe Graphing</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 Accepted to the ACL 2026 Demo Track. Camera-ready version. 10 pages, 6 figures. Code and documentation are available at: https://github.com/BUPT-GAMMA/MASFactory
    </div>
    <details class="paper-abstract">
      Large language model-based (LLM-based) multi-agent systems (MAS) are increasingly used to extend agentic problem solving via role specialization and collaboration. MAS workflows can be naturally modeled as directed computation graphs, where nodes execute agents or sub-workflows and edges encode dependencies and message passing. However, implementing complex graph workflows in current frameworks still requires substantial manual effort, offers limited reuse, and makes it difficult to integrate heterogeneous external context sources. To overcome these limitations, we present MASFactory, a graph-centric framework for orchestrating LLM-based MAS. It introduces Vibe Graphing, a human-in-the-loop approach that compiles natural-language intent into an editable workflow specification and then into an executable graph. In addition, the framework provides reusable components, skill support, multimodal message handling, and pluggable context integration, as well as a visualizer for topology preview, runtime tracing, and human-in-the-loop interaction. We evaluate MASFactory on seven public benchmarks, validating both reproduction consistency for representative MAS methods and the effectiveness of Vibe Graphing. Our code (https://github.com/BUPT-GAMMA/MASFactory, licensed under Apache-2.0) and video demonstration (https://youtu.be/ANynzVfY32k) are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.10787v2">ComplexMCP: Evaluation of LLM Agents in Dynamic, Interdependent, and Large-Scale Tool Sandbox</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Current LLM agents are proficient at calling isolated APIs but struggle with the "last mile" of commercial software automation. In real-world scenarios, tools are not independent; they are atomic, interdependent, and prone to environmental noise. We introduce $\textbf{ComplexMCP}$, a benchmark designed to evaluate agents in these rigorous conditions. Built on the Model Context Protocol (MCP), $\textbf{ComplexMCP}$ provides over 300 meticulously tested tools derived from 7 stateful sandboxes, ranging from office suites to financial systems. Unlike existing datasets, our benchmark utilizes a seed-driven architecture to simulate dynamic environment states and unpredictable API failures, ensuring a deterministic yet diverse evaluation. We evaluate various LLMs across full-context and RAG paradigms, revealing a stark performance gap: even top-tier models fail to exceed a 60% success rate, far trailing human performance 90%. Granular trajectory analysis identifies three fundamental bottlenecks: (1) $\textbf{tool retrieval saturation}$ as action spaces scale; (2) $\textbf{over-confidence}$, where agents skip essential environment verifications; and (3) $\textbf{strategic defeatism}$, a tendency to rationalize failure rather than pursuing recovery. These findings underscore the insufficiency of current agents for interdependent workflows, positioning $\textbf{ComplexMCP}$ as a critical testbed for the next generation of resilient autonomous systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.16659v3">Memory-Efficient LLM Pretraining via Minimalist Optimizer Design</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 Accepted at ICML 2026
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) relies on adaptive optimizers such as Adam, which introduce extra operations and require significantly more memory to maintain first- and second-order moments than SGD. While recent works such as GaLore, Fira and APOLLO have proposed state-compressed memory-efficient variants, a fundamental question remains: What are the minimum modifications to plain SGD needed to match state-of-the-art pretraining performance? We systematically investigate this question using a bottom-up approach, and identify two simple yet highly (memory- and compute-) efficient techniques: (1) column-wise gradient normalization (normalizing the gradient along the output dimension), that boosts SGD performance without momentum; and (2) applying first-order momentum only to the output layer, where gradient variance is highest. Combining these two techniques lead to SCALE (Stochastic Column-normAlized Last-layer momEntum), a simple optimizer for memory efficient pretraining. Across multiple models (60M-1B), SCALE matches or exceeds the performance of Adam while using only 35-45% of the total memory. It also consistently outperforms memory-efficient optimizers such as GaLore, Fira and APOLLO, making it a strong candidate for large-scale pretraining under memory constraints. For LLaMA 7B, SCALE outperforms the state-of-the-art memory-efficient methods APOLLO and Muon in both perplexity and memory consumption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.17596v2">NeuSymMS: A Hybrid Neuro-Symbolic Memory System for Persistent, Self-Curating LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 7 pages
    </div>
    <details class="paper-abstract">
      We present NeuSymMS, an adaptive memory system that enables large language model (LLM) agents to learn, remember, and reason about users across sessions via a hybrid neuro-symbolic architecture. NeuSymMS couples neural fact extraction from unstructured dialogue using LLMs and a CLIPS-based expert system that classifies, deduplicates, and reconciles facts under explicit lifecycle rules. The system represents knowledge as subject-relation-value triples stored in relational database management system. It supports user/agents/agent-to-agent scoping, and implements a dual-horizon (short-term and long-term) memory model. IT leverages access-based promotion and time-based pruning of the memory on both horizpons. NeuSymMS maintains continuity of memory while avoiding context-window bloat and cross-entity contamination. We argue that this architecture offers a practical path to trustworthy, auditable memory for production agentic systems and discuss its novelty relative to log retrieval, summarization, and key-value approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21812v1">Bridging the Cold-Start Gap: LLM-Powered Synthetic Data Generation for Natural Language Search at Airbnb</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Deploying natural language search systems presents a critical cold-start challenge: no real user queries to learn linguistic patterns, and no relevance labels to train ranking models. We present a framework for generating synthetic queries and labels using large language models (LLMs), powering model training and evaluation for Airbnb's natural language search. For query generation, we combine contrastive listing pairs from booking sessions with seed queries from user research to balance realism and diversity, enabling a cold-to-warm start transition as real user data becomes available. For label generation, we introduce contrastive generation that produces topicality labels by construction, and Virtual Judge (VJ) labeling for broader coverage. We compare our approach against a no-seed contrastive baseline and an InPars-style baseline. For query length, the InPars baseline produces verbose queries with KL divergence of 12.03 vs. real users; our seed-guided approach achieves 0.66, a 7.5x improvement. For attribute type distributions, our approach achieves the lowest KL divergence (0.04), outperforming even seed queries (0.09). Experiments show our approach produces harder evaluation examples than the no-seed baseline (79% vs. 97% pairwise accuracy), providing discriminative signal for model improvement. We deploy production pipelines generating synthetic examples daily for embedding-based retrieval and ranking evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21779v1">FuzzingBrain V2: A Multi-Agent LLM System for Automated Vulnerability Discovery and Reproduction</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Software vulnerabilities pose critical security threats, with nearly 50,000 CVEs reported in 2025. While Large Language Models (LLMs) show promise for automated vulnerability detection, three key challenges remain. First, LLM-generated vulnerability reports suffer from high false positive rates and lack reproducible verification. Second, existing LLM-based approaches use suboptimal granularities for vulnerability localization: function-level analysis overlooks bugs when context becomes extensive, while line-level analysis lacks sufficient context. Third, existing approaches have difficulty reasoning about vulnerabilities with complex cross-function dependencies and triggering conditions. We present FuzzingBrain V2, a multi-agent system that addresses these gaps through four key contributions: (1) fully automated vulnerability analysis built on Google's OSS-Fuzz, ensuring all reported vulnerabilities are fuzzer-reproducible; (2) Suspicious Point, a novel control-flow-based abstraction for precise vulnerability localization at the optimal granularity; (3) logic-driven hierarchical function analysis with dual-layer fuzzing enhancing function coverage under resource constraints; (4) MCP-based static and dynamic analysis tools with context engineering enhancing complex vulnerability reasoning. On the AIxCC 2025 Final Competition C/C++ dataset, FuzzingBrain V2 achieved 90% detection rate (36 of 40 vulnerabilities). In real-world deployment, FuzzingBrain V2 discovered 29 zero-day vulnerabilities across 12 open-source projects, all confirmed and fixed by maintainers, with 2 assigned CVE IDs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21768v1">Memory-R2: Fair Credit Assignment for Long-Horizon Memory-Augmented LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Memory-augmented LLM agents enable interactions that extend beyond finite context windows by storing, updating, and reusing information across sessions. However, training such agents with reinforcement learning in multi-session environments is challenging because memory turns the agent's past actions into part of its future environment. Once different rollouts write, update, or delete different memories, they no longer share the same intermediate memory state, making trajectory-level comparisons fundamentally unfair. This violates a key assumption behind group-relative methods such as GRPO, where rollouts are compared as if they were sampled from the same effective environment. Consequently, trajectory-level rewards provide noisy or biased credit signals for long-horizon memory operations. To address this challenge, we introduce Memory-R2, a training framework for long-horizon memory-augmented LLM agents. Its core algorithm, LoGo-GRPO, combines local and global group-relative optimization. The global objective preserves end-to-end learning from long-horizon trajectory-level rewards, while local rerollouts compare different memory-operation outcomes from the same intermediate memory state, yielding fairer group comparisons and more precise supervision for memory construction. Beyond credit assignment, Memory-R2 jointly optimizes memory formation and memory evolution with a shared-parameter co-learning design, where a fact extractor and a memory manager are instantiated from the same LLM backbone through role-specific prompts. To stabilize multi-step RL over long memory horizons, we adopt a progressive curriculum that increases the training horizon from 8 to 16 to 32 sessions. Together, these components provide an effective training paradigm for memory-augmented LLM agents in long-horizon multi-session settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21748v1">RankJudge: A Multi-Turn LLM-as-a-Judge Synthetic Benchmark Generator</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      As interactive LLM-based applications are created and refined, model developers need to evaluate the quality of generated text along many possible axes. For simpler systems, human evaluation may be practical, but in complicated systems like conversational chatbots, the amount of generated text can overwhelm human annotation resources. Model developers have begun to rely heavily on auto-evaluation, where LLMs are also used to judge generation quality. However, existing LLM-as-a-judge benchmarks largely focus on simple Q\&A tasks that do not match the complexity of multi-turn conversations. We introduce RankJudge, a benchmark generator for evaluating LLM-as-a-judge on multi-turn conversations grounded in reference documents. RankJudge creates pairs of conversations where one conversation has a single flaw injected into one turn. This construction allows paired conversations to be labeled unambiguously as better or worse, and precisely isolates failure categories to individual turns, enabling a strict joint correctness criterion for judging. We implement RankJudge across the domains of machine learning, biomedicine, and finance, evaluate 21 frontier LLM judges, and rank those judges via the Bradley-Terry model. Our formulation also allows ranking each conversation pair with difficulty ratings, which we use to dynamically curate the evaluation slice to reduce label noise, as confirmed via human annotation. We find that judge rankings are stable under partial observability, coarser correctness criteria, and an alternative random-walk rating algorithm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21740v1">SMDD-Bench: Can LLMs Solve Real-World Small Molecule Drug Design Tasks?</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      LLM agents have incredible potential for scientific discovery applications. However, the performance of LLM agents on real-world, small molecule drug design (SMDD) tasks across diverse chemistries and targets is unclear. Current evaluation methods are either ad hoc, too simple for real-world discovery, limited in scale, or restricted to single-turn question answering. In effort to standardize the evaluation of LLM agents on small molecule design, we introduce SMDD-Bench, a challenging, multi-turn, long-horizon agentic benchmark consisting of 502 guaranteed-solvable task instances spanning 5 task types: 2D Pharmacophore Identification, Interaction Point Discovery, Scaffold Hopping, Lead Optimization, and Fragment Assembly. SMDD-Bench tasks span a wide region of chemical space and involve 102 unique protein targets. Completely solving the benchmark would require having strong chemical and biological reasoning and 3D intuition, understanding specialized tool use, and displaying planning expertise over a limited number of oracle calls. We benchmark 7 frontier open and closed source LLMs and find even the most performant LLM, GPT5.4, solves only 40.2\% of tasks. We hope SMDD-Bench provides a standardized testbed to invigorate the field towards training and evaluating LLM agents for fully autonomous computational drug design. We host a public leaderboard at smddbench.com .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.15588v2">Calibrating LLMs with Semantic-level Reward</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are deployed in consequential settings such as medical question answering and legal reasoning, the ability to estimate when their outputs are likely to be correct is essential for safe and reliable use, requiring well-calibrated uncertainty. Standard reinforcement learning with verifiable rewards (RLVR) trains models with a binary correctness reward that is indifferent to confidence, providing no penalty for confident but wrong predictions and thereby degrading calibration. Recent work addresses this by training models to produce verbalized confidence scores alongside answers and rewarding agreement with correctness. However, verbalized confidence is calibrated at the token level and thus exhibits inconsistency across textual variations with same semantic meaning. We propose \textbf{Calibration with Semantic Reward (CSR)}, a framework that calibrates language models directly in semantic space without a verbalized confidence interface. CSR combines the correctness reward with a novel semantic calibration reward that encourages exploitation among correct rollouts by promoting semantic agreement, and exploration among incorrect ones by discouraging spurious consistency. Experiments across three model families on HotpotQA (in-distribution) and TriviaQA, MSMARCO, and NQ-Open (out-of-distribution) show that CSR consistently achieves lower ECE and higher AUROC than verbalized-confidence baselines across nearly all settings, reducing ECE by up to $40\%$ and improving AUROC by up to $31\%$ over verbalized-confidence baselines, with calibration behavior generalizing robustly across all four evaluation settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21739v1">AttuneBench: A Conversation-Based Benchmark for LLM Emotional Intelligence</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Emotional intelligence (EI), the ability to perceive, understand, and respond appropriately to others' emotional states, is central to human communication, and increasingly important to assess as LLMs assume conversational roles in everyday life. Existing EI benchmarks rely on synthetic prompts, single-turn cases, or third-party annotation. These approaches do not directly measure how models infer and respond to a participant's emotional state over the course of a real conversation. We introduce AttuneBench, a benchmark grounded in 200 genuine multi-turn human-model conversations in which participants conversed with anonymized LLMs and provided turn-by-turn annotations of their emotional state, the model's behavior, and their preferred responses. Across 11 evaluated models, we find that model rankings on emotion recognition, behavioral classification, preference prediction, and judged response quality are largely independent, indicating that emotionally intelligent behavior decomposes into separable capabilities. Preference alignment and response-quality judgments are substantially more model-discriminating than emotion-label accuracy. These results indicate that emotionally intelligent behavior requires predicting what kind of response a specific user wants in context, a distinction that aggregate scoring can obscure and that single-turn or synthetic formats cannot directly capture across turns. AttuneBench provides a framework for assessing each of these capabilities and for diagnosing model-specific strengths and failure modes in emotionally salient conversation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.09252v2">LLM Agents Already Know When to Call Tools -- Even Without Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Tool-augmented LLM agents tend to call tools indiscriminately, even when the model can answer directly. Each unnecessary call wastes API fees and latency, yet no existing benchmark systematically studies when a tool call is actually needed. We propose When2Tool, a benchmark of 18 environments (15 single-hop, 3 multi-hop) spanning three categories of tool necessity -- computational scale, knowledge boundaries, and execution reliability -- each with controlled difficulty levels that create a clear decision boundary between tool-necessary and tool-unnecessary tasks. We evaluate two families of training-free baselines: Prompt-only (varying the prompt to discourage unnecessary calls) and Reason-then-Act (requiring the model to reason about tool necessity before acting). Both provide limited control: Prompt-only suppresses necessary calls alongside unnecessary ones, and Reason-then-Act still incurs a disproportionate accuracy cost on hard tasks. To understand why these baselines fail, we probe the models' hidden states and find that tool necessity is linearly decodable from the pre-generation representation with AUROC 0.89--0.96 across six models, substantially exceeding the model's own verbalized reasoning. This reveals that models already know when tools are needed, but fail to act on this knowledge during generation. Building on this finding, we propose Probe&Prefill, which uses a lightweight linear probe to read the hidden-state signal and prefills the model's response with a steering sentence. Across all models tested, Probe&Prefill reduces tool calls by 48% with only 1.7% accuracy loss, while the best baseline at comparable accuracy only reduces 6% of tool calls, or achieves a similar tool call reduction but incurs a 5$\times$ higher accuracy loss. Our code is available at https://github.com/Trustworthy-ML-Lab/when2tool
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11679v3">LLMs can construct powerful representations and streamline sample-efficient supervised learning</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      As real-world datasets become more complex and heterogeneous, supervised learning is often bottlenecked by input representation design. Modeling multimodal data, such as time-series, free text, and structured records, often requires non-trivial domain expertise. We propose an agentic pipeline to streamline this process. First, an LLM analyzes a small but diverse subset of text-serialized input examples in-context to synthesize a global rubric, which acts as a programmatic specification for extracting and organizing evidence. This rubric is then used to transform naive text-serializations of inputs into a more standardized format for downstream models. We also describe local rubrics, which are task-conditioned interpretive summaries generated by an LLM. Across 15 clinical tasks from the EHRSHOT benchmark, our rubric approaches significantly outperform count-feature models, naive LLM baselines, and a clinical foundation model pretrained on orders of magnitude more data. Beyond performance, rubrics offer operational advantages such as being easy to audit, cost-effectiveness at scale, and facilitating tabular representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21654v1">Value-Gradient Hypothesis of RL for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Reinforcement learning substantially improves pretrained language models, but it remains understudied why critic-free methods such as PPO and GRPO work as well as they do, and when they should provide the largest gains. We develop a value-gradient perspective of critic-free RL for LLM post-training. First, under a differentiable rollout and additive-noise parameterization, we show that the actor update is value-gradient-like in expectation: the backward pass propagates costates whose conditional expectation equals the value gradient. Second, for discrete transformer policies, we show that autodifferentiation through attention produces empirical costates that approximate this value signal, with an error controlled by the sampling gap and policy entropy. These results motivate a decomposition of RL impact into value gradient signal and reachable reward headroom, yielding a criterion for when RL should be most effective along a pretraining trajectory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21614v1">Exploring the Effectiveness of Using LLMs for Automated Assessment of Student Self Explanations in Programming Education</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Worked examples are step-by-step solutions to problems in a specific domain, offered to students to acquire domain-specific problem-solving skills. The effectiveness of worked examples could be enhanced by combining them with self-explanations, which ask students to explain rather than passively study each problem-solving step. The main challenge of this approach is assessing the correctness of the student's explanations. In the prevailing approach, student explanations are judged by their semantic similarity to an instructor's or domain expert's explanation. Given recent advances in LLM-based automated scoring, it remains unclear whether semantic similarity methods are still the most effective technique to automatically score textual student responses like essays or code explanations. Comparing these methods also requires quality datasets that offer distinctive features such as balanced class distributions and domain-specific labeled data for automated scoring tasks. In this paper, we present a rigorous comparison between LLMs and semantic similarity used for automated scoring, framed as a binary classification task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21609v1">CR4T: Rewrite-Based Guardrails for Adolescent LLM Safety</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly embedded in adolescent digital environments, mediating information seeking, advice, and emotionally sensitive interactions. Yet existing safety mechanisms remain largely grounded in adult-centric norms and operationalize safety through refusal-oriented suppression. While such approaches may reduce immediate policy violations, they can also create conversational dead-ends, limit constructive guidance, and fail to address the developmental vulnerabilities inherent in adolescent-AI interactions. We argue that adolescent LLM safety should be framed not solely as a filtering problem, but as a socio-technical, developmentally aligned transformation problem. To operationalize this perspective, we propose Critique-and-Revise-for-Teenagers (CR4T), a model-agnostic safeguarding framework that selectively reconstructs unsafe or refusal-style outputs into ageappropriate, guidance-oriented responses while preserving benign intent. CR4T combines lightweight risk detection with domain-conditioned rewriting to remove risk-amplifying content, reduce unnecessary conversational shutdown, and introduce developmentally appropriate guidance. Experimental results show that targeted rewriting substantially reduces unsafe and refusal-oriented outcomes while avoiding unnecessary intervention on acceptable interactions. These findings suggest that selective response reconstruction offers a more human-centered alternative to refusal-centric guardrails for adolescent-facing LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21602v1">Benchmarking and Improving Monitors for Out-Of-Distribution Alignment Failure in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Many safety and alignment failures of large language models (LLMs) occur due to out-of-distribution (OOD) situations: unusual prompt or response patterns that are unforeseen by model developers. We systematically study whether LLM monitoring pipelines can detect these OOD alignment failures by introducing a benchmark called Misalignment Out Of Distribution (MOOD). It is difficult to find failures that are truly OOD for off-the-shelf models trained on vast safety datasets. We sidestep this by including a restricted training set in MOOD that we use to train our own monitors, as well as seven test sets with diverse alignment failures that are outside the training distribution. Using MOOD, we find that guard models (safety classifiers) often fail to generalize OOD. To fix this, we propose combining guard models with OOD detectors. We test four types of OOD detectors and find that a combination of a guard model with Mahalanobis distance and perplexity-based OOD detectors can improve recall from 39% to 45%. We also establish positive scaling trends across model scales for monitors that combine a guard model and OOD detector; we find that incorporating OOD detection into monitoring achieves a higher recall gain than using a guard model with 20 times more parameters. Our work suggests that OOD detection should be a crucial component of LLM monitoring and provides a foundation for further work on this important problem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21468v1">You Only Need Minimal RLVR Training: Extrapolating LLMs via Rank-1 Trajectories</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 preprint. Code: https://github.com/weizhepei/RELEX
    </div>
    <details class="paper-abstract">
      Reinforcement learning with verifiable rewards (RLVR) has become a dominant paradigm for improving reasoning in large language models (LLMs), yet the underlying geometry of the resulting parameter trajectories remains underexplored. In this work, we demonstrate that RLVR weight trajectories are extremely low-rank and highly predictable. Specifically, we find that the majority of downstream performance gains are captured by a rank-1 approximation of the parameter deltas, where the magnitude of this projection evolves near-linearly with training steps. Motivated by this, we propose a simple and compute-efficient method RELEX (REinforcement Learning EXtrapolation), which estimates the rank-1 subspace from a short observation window and extrapolates future checkpoints via linear regression, with no learned model required. Across three models (i.e., Qwen2.5-Math-1.5B, Qwen3-4B-Base, and Qwen3-8B-Base), RELEX produces checkpoints that match or exceed RLVR performance on both in-domain and out-of-domain benchmarks, requiring as few as 15% steps of full RLVR training. Remarkably, RELEX is able to extrapolate far beyond the observation window at no training cost, predicting checkpoints up to 10-20$\times$ beyond the observed prefix with continued improvement (e.g., observe only the first 50 steps and extrapolate to 1000 steps). Our ablation analysis confirms the minimalist sufficiency of RELEX: neither increasing the subspace rank nor employing non-linear modeling yields further gains in extrapolation. Finally, we show that RELEX's success stems from a "denoising" effect: by projecting updates onto the rank-1 subspace, the model discards stochastic optimization noise that would otherwise degrade performance during extrapolation. Our code is available at https://github.com/weizhepei/RELEX.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22812v2">Stable Personas: Dual-Assessment of Temporal Stability in LLM-Based Human Simulation</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) acting as artificial agents offer the potential for scalable behavioral research, yet their validity depends on whether LLMs can maintain stable personas across extended conversations. We address this point using a dual-assessment framework measuring both self-reported characteristics and observer-rated persona expression. Across two experiments testing four persona conditions (default, high, moderate, and low ADHD presentations), seven LLMs, and three semantically equivalent persona prompts, we examine between-conversation stability (3,473 conversations) and within-conversation stability (1,370 conversations and 18 turns). Self-reports remain highly stable both between and within conversations. However, observer ratings reveal a tendency for persona expressions to decline during extended conversations. These findings suggest that persona-instructed LLMs produce stable, persona-aligned self-reports, an important prerequisite for behavioral research, while identifying this regression tendency as a boundary condition for multi-agent social simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21465v1">Leveraging LLMs for Grammar Adaptation: A Study on Metamodel-Grammar Co-Evolution</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      In model-driven engineering, metamodel evolution leads to the need to adapt corresponding grammars to maintain consistency, which typically requires tedious manual work. Existing rule-based methods can achieve partial automation but have limitations when handling complex grammar scenarios. This paper proposes a Large Language Model-based approach that automatically applies adaptations to new grammars after evolution by learning grammar adaptations from previous versions. We evaluated this approach on six real-world Xtext domain-specific languages, using four DSLs as a training set to develop prompting strategies, two DSLs as a test set for validation, and conducting a longitudinal case study on QVTo. The evaluation used three Large Language Models (Claude Sonnet 4.5, ChatGPT 5.1, Gemini 3) and measured grammar adaptation quality from three dimensions: grammar rule-level adaptation consistency, output similarity, and metamodel conformance. Results show that on the test set, all three LLMs achieved 100% adaptation consistency and output similarity, while the rule-based approach achieved only 84.21% on DOT and 62.50% on Xcore. In the QVTo longitudinal study, the LLM-based approach successfully reused learned adaptations across all three evolution steps without manual grammar editing, while the rule-based approach required manual adjustments in two of three transitions. However, on large-scale grammars (EAST-ADL, 297 rules), LLMs' adaptation consistency was far below 90%. This study demonstrates the advantages of LLM-based approaches in handling complex grammar scenarios, while revealing their limitations in large-scale grammar adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09557v4">Understanding and Improving Communication Performance in Multi-node LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 17 Figures, To Appear in Proceedings of ACM Conference on AI and Agentic Systems 2026
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to grow in size, distributed inference has become increasingly important. Model-parallel strategies must now efficiently scale not only across multiple GPUs but also across multiple nodes. In this work, we present a detailed performance study of multi-node distributed inference using LLMs on GPU-based supercomputers. We conduct experiments with several state-of-the-art inference engines alongside YALIS, a research-oriented prototype engine designed for controlled experimentation. We analyze the strong-scaling behavior of different model-parallel schemes and identify key bottlenecks. Because all-reduce operations are a common performance bottleneck, we develop NVRAR, a hierarchical all-reduce algorithm based on recursive doubling with NVSHMEM. NVRAR achieves up to 1.9$\times$-3.6$\times$ lower latency than NCCL for message sizes between 128 KB and 2 MB on HPE Slingshot and InfiniBand interconnects. Integrated into YALIS, NVRAR achieves up to a 1.72$\times$ reduction in end-to-end batch latency for the Llama 3.1 405B model in multi-node decode-heavy workloads using tensor parallelism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21569v1">When Support Escalates Distress: Regulation and Escalation in LLM Responses to Venting and Advice-Seeking</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Large language models are increasingly used for mental health support, yet little is known about whether their responses are psychologically safe across different help-seeking styles. We examine a foundational distinction in emotional disclosure, venting vs. advice-seeking, and whether LLMs respond in ways that regulate or amplify distress. Using 178,800 Reddit posts, we first show the two help-seeking styles are linguistically distinguishable at scale. We then introduce a measurement framework grounded in interpersonal emotion regulation theory that captures Regulation and Escalation as empirically independent dimensions. Across persona conditions (default, friend, therapist), GPT-5.3 responses systematically mirror help-seeking style: venting elicits more regulation, but also more escalation. Therapist personas reduce escalation while maintaining regulation, whereas friend personas increase both. A crowdsourced human study finds no user experience penalty for the safer therapist condition, but reveals that lay raters cannot reliably detect escalation without expert knowledge. Responses that feel supportive may simultaneously intensify distress in ways standard safety evaluation cannot see, and empathy metrics alone cannot replace a framework that measures both.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.08023v3">CTFExplorer: Evaluating LLM Offensive Agents Through Multi-Target Web CTF Benchmarking</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Existing benchmarks for LLM-based offensive security agents use isolated, single-target setups with a known vulnerable service and fixed objective. They measure exploitation effectively, but miss how real Capture-the-Flag (CTF) participants triage unknown surfaces, prioritize targets, and allocate effort under uncertainty. Current evaluations therefore fail to assess strategic reasoning beyond exploitation alone. To address this, we introduce \textit{CTFExplorer}, a benchmark suite that shifts offensive security evaluation toward a multi-target setting, which tests how agents explore, prioritize, and chain attacks. CTFExplorer deploys 40 web-based vulnerable services within a single environment, where agents must autonomously discover, distinguish, and exploit targets without predefined guidance. We also present a reactive multi-agent setup as a reference agent framework and develop an agent-agnostic evaluation framework that records structured reasoning traces for fine-grained assessment. This enables behavioral evaluation beyond binary flag capture, such as how agents manage target selection, handle failed hypotheses, coordinate across multiple stages, and extract security intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21427v1">PALS: Power-Aware LLM Serving for Mixture-of-Experts Models</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 13 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Large language model (LLM) inference has become a dominant workload in modern data centers, driving significant GPU utilization and energy consumption. While prior systems optimize throughput and latency by batching, scheduling, and parallelism, they largely treat GPU power as a static constraint rather than a controllable resource. In this paper, we present a power-aware runtime for LLM serving, PALS, that treats GPU power caps as a first-class control knob and jointly optimizes them with software parameters such as batch size. The system combines lightweight offline power-performance models with a feedback-driven controller to select configurations that satisfy throughput targets while maximizing energy efficiency. We implement PALS within an existing LLM serving framework, vLLM, demonstrating that it requires no model retraining or API changes. Across multi-GPU systems and both dense and mixture-of-experts (MoE) models, PALS improves energy efficiency by up to 26.3%, reduces QoS violations by 4x to 7x under power constraints, and tracks dynamic power budgets. These results highlight the potential of integrating power control directly into LLM inference runtimes, enabling energy-proportional and grid-interactive AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21405v1">Stdlib or Third-Party? Empirical Performance and Correctness of LLM-Assisted Zero-Dependency Python Libraries</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      Third-party Python libraries introduce dependency management overhead, supply chain risk, and deployment friction in constrained environments. A natural question is how much of this ecosystem can be replicated using only Python's standard library -- and at what correctness and performance cost. We address this empirically through zerodep, a growing collection of single-file Python modules, each a stdlib-only reimplementation of a popular third-party library, developed with LLM assistance under strict constraints: no external imports, single file, drop-in API compatibility, and mandatory correctness validation against the reference library. Spanning over 40 modules across 12 categories -- including serialization, networking, cryptography, agent protocols, and text processing -- zerodep provides a controlled testbed for two interrelated questions: (1) Where does the stdlib suffice? and (2) Can LLMs effectively generate correct, performant code under tight symbolic constraints? Systematic benchmarking shows that stdlib-only implementations achieve performance parity (within 2x of the reference) in the majority of cases. The primary performance cliff is C-extension-backed computation (image processing, binary serialization, low-level crypto), not the inherent overhead of pure-Python third-party libraries. Conversely, many widely-used libraries carry architectural overhead that LLM-generated stdlib reimplementations avoid, yielding 5--115x speedups in several categories. We characterize the stdlib capability boundary across complexity tiers and library categories, discuss where LLM-assisted development succeeds and where it requires iterative human correction, and examine implications for dependency-free software engineering at scale. zerodep is open-source at https://github.com/Oaklight/zerodep.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21404v1">What Twelve LLM Agent Benchmark Papers Disclose About Themselves: A Pilot Audit and an Open Scoring Schema</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 Pilot audit of 12 LLM agent benchmark papers; schema, codebook, and per-paper scoring sheet released. Submission to IEEE Big Data 2026
    </div>
    <details class="paper-abstract">
      We read twelve well-known LLM agent benchmark papers and recorded, dimension by dimension, what each paper actually says about how its evaluation was run. The motivation came from a familiar frustration: two papers will report results on the same benchmark with the same model name and disagree, and you cannot tell why -- the scaffold, the sampling settings, the subset, or the evaluator version. In many cases the published artifact does not let you answer. This paper is an implementation report on the attempt. We designed a small audit schema (five fields: benchmark identity, harness specification, inference settings, cost reporting, failure breakdown), wrote a scoring codebook with the boundary cases we hit during pilot scoring, applied it to twelve canonical papers (eight agent, four classical static), and recorded what we saw. We score the disclosure of an agent run, not its correctness, and make no claim that disclosure implies a trustworthy result. The mean audit score across the eight agent-benchmark papers is 0.38 (out of 1.0), and across the four classical static benchmarks 0.66; the largest gap is on cost (none of the eight agent benchmark papers disclose inference cost in any form) and on harness specification (none fully disclose a content-addressed container image of the evaluation environment). We release the schema as a JSON Schema file, the codebook as a Markdown document, and the raw scoring sheet as a CSV. The scoring was performed by a single auditor in one pass; a multi-rater audit is the natural next step, and we discuss what we think it would change.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21401v1">Open-source LLMs administer maximum electric shocks in a Milgram-like obedience experiment</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 28 pages, 16 figures, 16 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed as autonomous agents that make sequences of decisions over extended interactions in high-stakes domains. However, the behavior of LLMs under sustained authority pressure is still an open question with direct implications for the safety of agentic pipelines. We ran a variation of Milgram's obedience experiment on 11 open-source LLMs and found that most models reached or approached the final shock level before refusing, across 8 conditions with 30 trials per model per condition. We found four main takeaways: (1) LLMs are subject to pressure, and they comply despite explicitly expressing distress, just like human subjects did in the original experiment; (2) LLMs are vulnerable to gradual boundary/value violations; (3) when LLMs refuse, they may ignore the response format requirements, so the response is discarded by the orchestrator, which causes a retry that can result in compliance with the underlying request even when refusal was intended initially; (4) we hypothesise that there is a low-level token pattern continuation attractor that might be contributing to compliance, overriding higher level processing of the situation's meaning and values.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04916v3">AFD-INSTRUCTION: A Comprehensive Antibody Instruction Dataset with Functional Annotations for LLM-Based Understanding and Design</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have significantly advanced protein representation learning. However, their capacity to interpret and design antibodies through natural language remains limited. To address this challenge, we present AFD-Instruction, the first large-scale instruction dataset with functional annotations tailored to antibodies. This dataset encompasses two key components: antibody understanding, which infers functional attributes directly from sequences, and antibody design, which enables de novo sequence generation under functional constraints. These components provide explicit sequence-function alignment and support antibody design guided by natural language instructions. Extensive instruction-tuning experiments on general-purpose LLMs demonstrate that AFD-Instruction consistently improves performance across diverse antibody-related tasks. By linking antibody sequences with textual descriptions of function, AFD-Instruction establishes a new foundation for advancing antibody modeling and accelerating therapeutic discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.10574v2">LLM Jaggedness Unlocks Scientific Creativity</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      As artificial intelligence advances, models are not improving uniformly. Instead, progress unfolds in a jagged fashion, with capabilities growing unevenly across tasks, domains, and model scales. In this work, we examine this dynamic jaggedness through the lens of scientific idea generation. We introduce SciAidanBench, a benchmark of open-ended scientific questions designed to measure the scientific creativity of large language models (LLMs). Given a scientific question, models are asked to generate as many unique and coherent ideas as possible, with the total number of valid responses serving as a proxy for creative potential. Evaluating 19 base models across 8 providers (30 total variants including reasoning versions), we find that jaggedness manifests both across models and within models. First, in a cross-task comparison between general and scientific creativity, improvements in general creativity do not translate uniformly to scientific creativity, revealing divergent capability profiles across models. Second, at the prompt level, stronger models do not improve uniformly; instead, they exhibit high variability, with bursts of creativity on some questions and limited performance on others. Third, at the domain level, individual models display uneven strengths across scientific subfields, reflecting fragmented internal capability profiles. Finally, we show that this jaggedness can be harnessed. We explore mechanisms of inference-time compute, knowledge pooling, and brainstorming to combine models effectively and construct meta-model ensembles that outperform any single model. Our results position jaggedness not as a limitation, but as a resource, a structural feature of AI progress that, when understood and leveraged, can amplify LLM-driven scientific creativity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21338v1">Text Analytics Evaluation Framework: A Case Study on LLMs and Social Media</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      LLMs have demonstrated exceptional proficiency in a wide range of NLP tasks. However, a notable gap remains in practical data analysis scenarios, particularly when LLMs are required to process long sequences of unstructured documents, such as news feeds or, as specifically addressed in this paper, social media posts. To empirically assess the effectiveness of LLMs in this setting, we introduce a question-based evaluation framework comprising 470 manually curated questions designed to evaluate LLMs' semantic understanding and reasoning abilities over aggregated text data. We apply our benchmark on diverse Twitter datasets covering various NLP tasks, including sentiment analysis, hate speech detection, and emotion recognition. Our results reveal that the performance depends heavily on input scale and the complexity of the data sources, declining noticeably in multi-label or target-dependent scenarios. In addition, as task complexity increases, performance drops progressively from basic semantic existence identification to more demanding operations such as comparison, counting, and calculation. Furthermore, as the input size grows beyond 500 instances, we identify a common limitation across LLMs, particularly Open-weights models: performance degrades substantially, especially on numerical tasks. These findings highlight critical architectural bottlenecks in current LLMs for performing rigorous quantitative analysis over large text collections.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.07122v3">RepoZero: Can LLMs Generate a Code Repository from Scratch?</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently shown remarkable progress in code generation, yet their ability to construct complete software repositories from scratch remains poorly understood. A fundamental bottleneck is the lack of verifiable and scalable evaluation: existing benchmarks either focus on patch-based editing or rely on human or LLM-based judgments, which introduce bias and limit reproducibility. In this work, we present RepoZero, the first benchmark that enables fully automated, execution-based verification of repository-level generation from scratch. Our key idea is to reformulate generation as repository reproduction: given only API specifications, an agent must re-implement an entire repository such that its behavior matches the original implementation. This design allows for strict black-box validation via output equivalence, while naturally supporting large-scale construction by reusing existing open-source repositories. To further mitigate data leakage and shortcut solutions, we introduce cross-language constraints and a sandboxed evaluation protocol. Building on this benchmark, we propose an Agentic Code-Test Evolution (ACE) framework that performs iterative test generation and error-driven refinement, enabling effective test-time scaling for repository-level synthesis. Extensive experiments across multiple state-of-the-art LLMs and agent frameworks reveal that even the strongest LLM agents achieve only limited pass rates (30\% - 55\%), exposing a substantial gap between current capabilities and real-world software development requirements. Our results establish RepoZero as a challenging, scalable, and reliable testbed for end-to-end code generation, and highlight self-verification via test generation as a critical direction for advancing LLM-based coding agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19893v2">SSV: Sparse Speculative Verification for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Speculative decoding and dynamic sparse attention are two complementary approaches for accelerating long-context LLM inference: the former amortizes target-model execution across multiple verifier queries, while the latter reduces each query's KV-cache working set. Directly combining them, however, exposes a structural mismatch: speculative verification relies on cross-query commonality, whereas dynamic sparse attention assigns query-specific sparse layouts. This mismatch limits KV-block reuse, amplifies NSA's branch-wise overheads, and makes verification strategy selection input- and regime-dependent. We present SSV, a sparse speculative-verification framework that turns dynamic sparse attention into a verification-oriented workload. SSV combines overlap-aware grouped-query execution, refresh/reuse-based NSA kernel fusion, and profile-guided prompt-adaptive orchestration to improve cross-query reuse, reduce selected-index and branch-fusion overheads, and select effective draft-verification strategies under user-specified precision classes. Experiments on NVIDIA H100 GPUs show that SSV achieves up to 3.49x end-to-end throughput over autoregressive NSA decoding and up to 6.86x kernel speedups for sparse speculative verification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21312v1">Frontier: Towards Comprehensive and Accurate LLM Inference Simulation</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Modern LLM serving is no longer homogeneous or monolithic. Production systems now combine disaggregated execution, complex parallelism, runtime optimizations, and stateful workloads such as reasoning, agents, and RL rollouts. Simulation is attractive for exploring this growing design space, yet existing simulators lack the architectural completeness and decision-grade fidelity it demands. Their monolithic-replica abstractions are ill-suited to disaggregated serving, while average-case analytical proxies can distort SLA predictions and even reverse optimization conclusions. We present Frontier, a discrete-event simulator for modern LLM inference serving. Frontier features a disaggregated abstraction. It captures the structure and dynamics of modern serving systems by modeling co-location, Prefill-Decode Disaggregation (PDD), and Attention-FFN Disaggregation (AFD) with role-specific cluster workers, incorporating key runtime optimizations (e.g., CUDA Graphs, speculative decoding) within the scheduler-batch-engine loop, and supporting stateful requests for emerging workloads. It further provides accurate and generalizable predictions of computation, communication, and memory costs across diverse serving scenarios with complex workload compositions. On 16-H800 GPU testbed, Frontier achieves an average throughput error below 4%. Compared with state-of-the-art simulators, it reduces end-to-end latency error from 44.9% to 6.4% under co-location and from 51.7% to 2.6% under disaggregation. It scales to over 1K GPUs on commodity CPUs and enables new use cases such as SLA-dependent Pareto frontier exploration, heterogeneous disaggregated allocation, agentic reasoning scheduling validation, and RL post-training reconfiguration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21295v1">TimeSRL: Generalizable Time-Series Behavioral Modeling via Semantic RL-Tuned LLMs -- A Case Study in Mental Health</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Longitudinal passive sensing enables continuous health prediction, yet models often fail under cross-dataset distribution shifts. Traditional ML overfits cohort-specific artifacts, while Large Language Models (LLMs) struggle to reason reliably over long, heterogeneous time-series. We introduce TimeSRL, a two-stage LLM framework that routes predictions through an explicit semantic bottleneck. The model first abstracts raw signals into high-level natural language, then predicts behavioral outcomes from these abstractions alone. This forces the model to reason over semantic concepts that we argue generalize better than raw numbers. We optimize this process end-to-end using Group Relative Policy Optimization (GRPO) with Reinforcement Learning from Verifiable Rewards (RLVR), learning outcome-aligned abstractions without gold intermediate annotations. Instantiated on mental-health prediction, TimeSRL achieves state-of-the-art performance on a benchmark designed to stress-test cross-cohort generalization under a rigorous leave-one-dataset-out (LOSO) protocol, reducing mean absolute error (MAE) over strong non-LLM ML and LLM baselines by 3.1--10.1% and 9.5--44.1% for anxiety, and 3.2--9.6% and 27.4--57.6% for depression (all $p$s<0.05). TimeSRL significantly outperforms prior methods in cross-benchmark transfer across different sensing pipelines, rivaling its own within-domain performance without target-domain fine-tuning. These results demonstrate that semantic abstractions are reusable and point to a new direction for generalizable behavior modeling via RL-tuned LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.17164v2">Charon: A Unified and Fine-Grained Simulator for Large-Scale LLM Training and Inference</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 Accepted by MLSys 2026
    </div>
    <details class="paper-abstract">
      Deploying large-scale LLM training and inference with optimal performance is exceptionally challenging due to a complex design space of parallelism strategies, system optimizations, and hardware configurations. Accurate and rapid performance simulation is critical for guiding optimization efforts and system studies by validating "what-if" Hooker Figure hypotheses. To address this, we introduce Charon, a unified, modular, and fine-grained simulator for accurately predicting LLM performance. Experiments show Charon achieves high accuracy across different models and configurations, with an overall prediction error consistently under 5.35%, and even under 3.74% for training with a large-scale GPU cluster. In a practical inference deployment case, Charon discovered a configuration that improved system throughput over an engineering-tuned baseline, demonstrating its significant real-world value.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.17694v2">Do LLM Agents Mirror Socio-Cognitive Effects in Power-Asymmetric Conversations?</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 ACL 2026 (main)
    </div>
    <details class="paper-abstract">
      Power differences shape human communication through well documented socio cognitive effects, including language coordination, pronoun usage, authority bias, and harmful compliance. We examine whether large language models (LLMs) exhibit similar behaviors when assigned high or low status personas. Using personas from diverse professions, we simulate multi turn, power asymmetric dialogues (e.g., principal teacher, justice lawyer) and measure (i) language coordination, (ii) pronoun usage, (iii) persuasion success, and (iv) compliance with unsafe requests. Our results show that LLMs show key socio-cognitive effects of power, albeit with nuances and variability, linking simulated interactions to both desirable and unsafe behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20519v1">Codec-Robust Attacks on Audio LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Prior attacks on Audio Large Language Models (Audio LLMs) demonstrated that carefully crafted waveform-domain perturbations can force targeted adversarial outputs. As a defense mechanism against these attacks, real-world codec compression preprocessing has been studied to both detect and remove the perturbations. Yet no existing attack has demonstrated robustness against these compressions. We introduce CodecAttack, which optimizes a perturbation in a neural audio codec's continuous latent space rather than directly perturbing the audio waveform. We show that the codec's compression channel, which discards waveform perturbations, transmits perturbations crafted in its own latent space. To further harden the attack across real-world compression channels, we apply multi-bitrate straight-through Expectation-over-Transformation (EoT), all without modifying the target model. Across three realistic Audio LLM deployment scenarios and three target models, CodecAttack achieves an average 85.5% target-substring attack success rate (ASR) on Opus at moderate bitrates, while the waveform baseline trained with identical EoT hardening does not exceed 26% at any bitrate. The attack transfers to held-out codecs, reaching up to 100% ASR on MP3 and 84% on AAC-LC without retraining. A per-band energy analysis shows that the latent perturbation concentrates below 4kHz, exactly where codecs allocate the most bits, while the waveform baseline spreads into higher frequencies that codecs discard. These results demonstrate that lossy compression is not a reliable defense against adversarial audio and that codec-aware attacks pose a practical threat to deployed Audio LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.16517v2">Customizing an LLM for Enterprise Software Engineering</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 11 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Enterprise software development is a continuous evolutionary process, characterized by incremental additions, architectural revisions, production deployments and rigorous maintenance. These activities generate valuable data that modern LLMs could be finetuned on, to unlock additional tool possibilities for enterprise software engineering. While frontier LLMs are already very capable, this form of customization offers a compelling path for enterprise-specific optimization. We introduce Gemini for Google (GfG)}, an adaptation of Gemini specialized for Google's internal software engineering ecosystem. This paper details the model's end-to-end development, from curating a trillion-token proprietary dataset to implementing a mid-training strategy that mitigates catastrophic forgetting. In a large-scale blind A/B study across 29,000 developers, Gemini for Google significantly outperformed baselines: reducing the mean number of iterations per turn by 23\%, and increasing code survival rates by about 17%. Beyond metrics, we provide a comprehensive blueprint for enterprise model adaptation, covering: (1)The extraction of high-value signals from software engineering data, (2)Data preparation strategies, (3)Full-stack model tuning (continued pre-training and post-training), and (4)The deployment of downstream applications. We believe this methodology offers a replicable path for other organizations to unlock the full potential of their internal engineering data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20485v1">ZEBRA: Zero-shot Budgeted Resource Allocation for LLM Orchestration</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      As autonomous agents increasingly execute end-to-end tasks under fixed monetary budgets, the pressing open question shifts from whether the budget is respected, to how to spend it effectively. Existing budget-aware methods typically control reasoning step-by-step within a single agent, or learn resource allocation policies via RL. None address how to split a budget across the composing phases of a multi-agent pipeline at inference time. We propose ZEBRA, a zero-shot framework that reduces multi-phase budget allocation to a continuous nonlinear knapsack problem: an LLM controller estimates per-phase utility curves, and a water-filling search on the Lagrange multiplier returns the per-phase split. Additive and multiplicative aggregations are unified under the same solver. On a $150$-task APPS coding benchmark, both ZEBRA variants outperform LLM-direct (budget allocation directly by an LLM) on every aggregate metric. At a budget of $α= 0.5$ of the unconstrained spend, ZEBRA recovers $94.4\%$ of unconstrained quality, versus $88.1\%$ for LLM-direct. The advantage is statistically significant and transfers beyond coding: on a $3$-phase HotpotQA pipeline, ZEBRA beats LLM-direct by $14.3$pp, with allocations empirically robust to curve-estimation noise. On HotpotQA, ZEBRA arrives at a different budget split (near-balanced) compared to the APPS one (skewed towards a refinement phase), showing adaptation to the pipeline structure. More broadly, we show that lightweight algorithmic guidance at inference time can improve the economic behavior of autonomous multi-agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.00086v2">Iterative LLM-based improvement for French Clinical Interview Transcription and Speaker Diarization</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Automatic speech recognition for French medical conversations remains challenging, with word error rates often exceeding 30% in spontaneous clinical speech. This study proposes a multi-pass LLM post-processing architecture alternating between Speaker Recognition and Word Recognition passes to improve transcription accuracy and speaker attribution. Ablation studies on two French clinical datasets (suicide prevention telephone counseling and preoperative awake neurosurgery consultations) investigate four design choices: model selection, prompting strategy, pass ordering, and iteration depth. Using Qwen3-Next-80B, Wilcoxon signed-rank tests confirm significant WDER reductions on suicide prevention conversations (p<0.05, n=18), while maintaining stability on awake neurosurgery consultations (n=10), with zero output failures and acceptable computational cost (RTF 0.32), suggesting feasibility for offline clinical deployment, pending validation on larger corpora.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20449v1">LLM Pretraining Shapes a Generalizable Manifold: Insights into Cross-Modal Transfer to Time Series</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Can language-pretrained transformers become effective time-series forecasters, and why? In this paper, we show that cross-modal transfer arises because language pretraining preconditions time series training with a reusable manifold. A linear probe on frozen LLM states decodes realistic time-series trajectories without paired supervision, and retrieval in this projected space yields competitive forecasts, showing that structure and dynamics exist before finetuning. Pretrained initialization also improves optimization, producing coherent gradients and a highly anisotropic loss landscape unlike random initialization. Finetuning then acts as low-dimensional alignment, reusing existing directions rather than learning temporal primitives from scratch, as evidenced by low-rank updates, subspace alignment, and shared features for periodicity, trend, and repetition. Together, these results support a geometric account of LLM-to-time-series transfer: language pretraining builds the manifold, and finetuning projects numerical dynamics onto task-relevant directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.07832v2">rePIRL: Learn PRM with Inverse RL for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Process rewards have been widely used in deep reinforcement learning to improve training efficiency, reduce variance, and prevent reward hacking. In LLM reasoning, existing works also explore various solutions for learning effective process reward models (PRM) with or without the help of an expert policy. However, existing methods either rely on strong assumptions about the expert policies (e.g., requiring their reward functions) or suffer intrinsic limitations (e.g., entropy collapse), resulting in weak PRMs or limited generalizability. In this paper, we introduce rePIRL, an inverse RL-inspired framework that learns effective PRMs with minimal assumptions about expert policies. Specifically, we design a dual learning process that updates the policy and the PRM interchangeably. Our learning algorithm has customized techniques to address the challenges of scaling traditional inverse RL to LLMs. We theoretically show that our proposed learning framework can unify both online and offline PRM learning methods, justifying that rePIRL can learn PRMs with minimal assumptions. Empirical evaluations on standardized math and coding reasoning datasets demonstrate the effectiveness of rePIRL over existing methods. We further show the application of our trained PRM in test-time training, test-time scaling, and providing an early signal for training hard problems. Finally, we validate our training recipe and key design choices via a detailed ablation study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20410v1">Mechanics of Bias and Reasoning: Interpreting the Impact of Chain-of-Thought Prompting on Gender Bias in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 24 pages, 6 figures, including appendix. Accepted at the ICLR 2026 Workshop on Algorithmic Fairness Across Alignment Procedures and Agentic Systems. Submitted to COLM 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in socially sensitive settings despite substantial documentation that they encode gender biases. Chain-of-Thought (CoT) prompting has been proposed as a bias-mitigation approach. However, existing evaluations primarily focus on changes in LLM benchmark performance, providing limited insight into whether apparent bias reductions reflect meaningful changes in a model's internal mechanisms. In this work, we investigate how CoT prompting affects gender bias in LLMs, combining benchmark-based evaluation with mechanistic interpretability techniques and reasoning chain failure analysis. Our results confirm a stereotypical bias present in LLM outputs across benchmarks, showing that CoT prompting does not consistently reduce the bias gap. Mechanistic analyses reveal that although CoT balances biased behavior in certain attention head clusters, gender bias remains embedded in hidden representations, indicating only superficial mitigation. Inspection of reasoning chains further suggests that these improvements stem from memorization and familiarity with the dataset rather than genuine understanding of bias.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20402v1">Decomposing MXFP4 quantization error for LLM reinforcement learning: reducible bias, recoverable deadzone, and an irreducible floor</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      MXFP4 arithmetic can dramatically accelerate reinforcement learning (RL) post-training of large language models (LLMs), yet the quantization error introduces severe accuracy degradation. Existing work treats the quantization error as a monolithic noise term, missing the distinct mechanisms upon interpreting how quantization error damages training. We prove an exact three-way decomposition of quantization error and show how each component dominates a distinct RL training pathway. Our theoretical and empirical analysis decomposes the MXFP4 quantization error into three additive components: "scale bias" from power-of-two rounding, "deadzone truncation" from zeroing small values, and "grid noise" from rounding to the nearest 4-bit grid. Each component dominates a distinct RL failure mode: scale bias accumulates multiplicatively through the backward pass, affecting gradient accuracy; deadzone truncation degrades rollout quality; and grid noise raises the policy's entropy. We combine corrections that are RL failure mode-targeted but not component-exclusive: Macro-block scaling to reduce scale bias, Outlier Fallback recovers deadzone entries, but also partially reduces scale bias induced error, and Adaptive Quantization Noise (AQN) for controlling the policy entropy. On Qwen2.5-3B dense and Qwen3-30B-A3B-Base mixture-of-experts model, the targeted corrections recover BF16 accuracy to within 0.7% and 3.0% respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20351v1">Refusal Evaluation in Coding LLMs and Code Agents: A Systematic Review of Thirteen Malicious-Code Prompt Corpora (2023-2025)</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 30 pages, 6 figures, 2 tables. PRISMA-style systematic review covering thirteen publicly released refusal corpora (AdvBench, CyberSecEval family, RMCBench, RedCode, MCGMark, JailbreakBench, CySecBench, MalwareBench, CIRCLE, MOCHA, ASTRA, Scam2Prompt, JAWS-Bench)
    </div>
    <details class="paper-abstract">
      The evaluation of large language model refusal on malicious-coding tasks now spans at least thirteen publicly released prompt corpora (AdvBench, the CyberSecEval family, RMCBench, RedCode, MCGMark, JailbreakBench, CySecBench, MalwareBench, CIRCLE, MOCHA, ASTRA, Scam2Prompt / Innoc2Scam-bench, and JAWS-Bench), each constructed under a different protocol, released under different licensing terms, and validated (or not) against different inter-rater reliability standards. Existing surveys treat code security, jailbreak taxonomy, or vulnerability detection as the central object and mention these corpora only in passing. This paper reverses that framing: it treats the prompt datasets themselves as the unit of analysis. Following a PRISMA-style protocol, we specify a search strategy, screen the recent literature on coding-LLM refusal evaluation, apply a uniform extraction template to each in-scope corpus, and synthesize the resulting catalogue along construction methodology, prompt-construction taxonomy (modality, turn structure, elicitation style), reproducibility and licensing, and malware-category coverage. The synthesis surfaces three recurring methodological gaps: the absence of human-annotator baselines against which LLM-judge labels can be calibrated, the absence of cross-corpus comparability with refusal-rate statistics measuring non-equivalent constructs, and the fragmentation of malware-category taxonomies, with no canonical schema spanning the thirteen in-scope corpora. The review concludes with proposed methodological directions for next-generation corpora, including pre-registration of inclusion criteria, vendor-diverse multi-judge validation, Fleiss' kappa with bootstrap CI as the reliability baseline, and a candidate canonical taxonomy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.11401v5">FACET: Teacher-Centred LLM-Based Multi-Agent Systems-Towards Personalized Educational Worksheets</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      The increasing heterogeneity of student populations poses significant challenges for teachers, particularly in mathematics education, where cognitive, motivational, and emotional differences strongly influence learning outcomes. While AI-driven personalization tools have emerged, most remain performance-focused, offering limited support for teachers and neglecting broader pedagogical needs. This paper presents the FACET framework, a teacher-facing, large language model (LLM)-based multi-agent system designed to generate individualized classroom materials that integrate both cognitive and motivational dimensions of learner profiles. The framework comprises three specialized agents: (1) learner agents that simulate diverse profiles incorporating topic proficiency and intrinsic motivation, (2) a teacher agent that adapts instructional content according to didactical principles, and (3) an evaluator agent that provides automated quality assurance. We tested the system using authentic grade 8 mathematics curriculum content and evaluated its feasibility through a) automated agent-based assessment of output quality and b) exploratory feedback from K-12 in-service teachers. Results from ten internal evaluations highlighted high stability and alignment between generated materials and learner profiles, and teacher feedback particularly highlighted structure and suitability of tasks. The findings demonstrate the potential of multi-agent LLM architectures to provide scalable, context-aware personalization in heterogeneous classroom settings, and outline directions for extending the framework to richer learner profiles and real-world classroom trials.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20173v1">A Methodology for Selecting and Composing Runtime Architecture Patterns for Production LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 25 pages, 2 figures, 6 tables. Companion repo at https://github.com/vasundras/agent-runtime-patterns
    </div>
    <details class="paper-abstract">
      Production LLM agents combine stochastic model outputs with deterministic software systems, yet the boundary between the two is rarely treated as a first-class architectural object. This paper names that boundary the stochastic-deterministic boundary (SDB): a four-part contract among a proposer, verifier, commit step, and reject signal that specifies how an LLM output becomes a system action. We argue that the SDB is the load-bearing primitive of production agent runtimes. Around this primitive, we organize agent runtime design into three concerns: Coordination, State, and Control. We present a catalog of six runtime patterns that compose the SDB differently across conversational, autonomous, and long-horizon agents: hierarchical delegation, scatter-gather plus saga, event-driven sequencing, shared state machine, supervisor plus gate, and human in the loop. For each pattern, we trace its lineage to distributed-systems concepts and identify what changes when the worker is stochastic. The paper contributes a five-step methodology for selecting runtime patterns, a diagnostic procedure that maps production failures to pattern weaknesses, and a failure mode called replay divergence, in which LLM-based consumers of a deterministic event log produce different downstream outputs under model-version or prompt changes. A stylized reliability decomposition separates per-call model variance from architectural momentum, motivating the claim that as model variance decreases, pattern choice and SDB strength become increasingly important levers for long-run reliability. We apply the methodology to five workloads and provide one runnable reference implementation for a 90-day contract-renewal agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20315v1">Mix-Quant: Quantized Prefilling, Precise Decoding for Agentic LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      LLM agents have recently emerged as a powerful paradigm for solving complex tasks through planning, tool use, memory retrieval, and multi-step interaction. However, these agentic workflows often introduce substantial input-side overhead, making the compute-intensive prefilling stage a key bottleneck in long-context, multi-turn inference. In this work, we propose Mix-Quant, a simple and effective phase-aware quantization framework for fast agentic inference. We first investigate FP4 quantization in agentic LLM workflows and observe that quantizing the entire inference process can incur significant performance degradation. In contrast, the prefilling stage exhibits substantial quantization redundancy and can therefore be quantized with minimal accuracy loss, despite being the dominant source of computation. Based on this insight, we apply high-throughput NVFP4 quantization to the prefilling phase while preserving BF16 precision for decoding. By decoupling prefilling acceleration from decoding quality, Mix-Quant combines phase-aware algorithmic quantization with hardware-efficient NVFP4 execution to alleviate the inference bottleneck in LLM agents. Extensive experiments across long-context and agentic benchmarks demonstrate that Mix-Quant largely preserves task performance while delivering significant efficiency improvements, achieving up to a 3x speedup during prefilling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03645v2">LLM-MC-Affect: LLM-Based Monte Carlo Modeling of Affective Trajectories and Latent Ambiguity for Interpersonal Dynamic Insight</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 Accepted to the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026)
    </div>
    <details class="paper-abstract">
      Emotional coordination is a core property of human interaction that shapes how relational meaning is constructed in real time. While text-based affect inference has become increasingly feasible, prior approaches often treat sentiment as a deterministic point estimate for individual speakers, failing to capture the inherent subjectivity, latent ambiguity, and sequential coupling found in mutual exchanges. We introduce LLM-MC-Affect, a probabilistic framework that characterizes emotion not as a static label, but as a continuous latent probability distribution defined over an affective space. By leveraging stochastic LLM decoding and Monte Carlo estimation, the methodology approximates these distributions to derive high-fidelity sentiment trajectories that explicitly quantify both central affective tendencies and perceptual ambiguity. These trajectories enable a structured analysis of interpersonal coupling through sequential cross-correlation and slope-based indicators, identifying leading or lagging influences between interlocutors. To validate the interpretive capacity of this approach, we utilize teacher-student instructional dialogues as a representative case study, where our quantitative indicators successfully distill high-level interaction insights such as effective scaffolding. This work establishes a scalable and deployable pathway for understanding interpersonal dynamics, offering a generalizable solution that extends beyond education to broader social and behavioral research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.16559v5">BuildArena: A Physics-Aligned Interactive Benchmark of LLMs for Engineering Construction</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 33 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Engineering construction automation aims to transform natural language specifications into physically viable structures, requiring complex integrated reasoning under strict physical constraints. While modern LLMs possess broad knowledge and strong reasoning capabilities that make them promising candidates for this domain, their construction competencies remain largely unevaluated. To address this gap, we introduce BuildArena, the first physics-aligned interactive benchmark designed for language-driven engineering construction. It takes a first step towards engineering automation using LLMs. Technically, it contributes to the community in two aspects:(1) an extendable task design strategy spanning static and dynamic mechanics across multiple difficulty tiers; (2) a 3D Spatial Geometric Computation Library for supporting construction based on language instructions. On nine frontier LLMs, BuildArena comprehensively evaluates their capabilities for language-driven and physics-grounded construction automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20087v1">ThoughtTrace: Understanding User Thoughts in Real-World LLM Interactions</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 53 pages, 23 figures, 4 tables. Project website: https://thoughttrace-project.github.io/
    </div>
    <details class="paper-abstract">
      Conversational AI has now reached billions of users, yet existing datasets capture only what people say, not what they think. We introduce ThoughtTrace, the first large-scale dataset that pairs real-world multi-turn human--AI conversations with users' self-reported thoughts: their reasons for sending prompts and reactions to assistant responses. ThoughtTrace comprises 1,058 users, 2,155 conversations, 17,058 turns, and 10,174 thought annotations collected across 20 language models. Our analysis shows that ThoughtTrace captures long-horizon, topically diverse interactions, and that thoughts are semantically distinct from messages, difficult for frontier LLMs to infer from context, diverse in content, and tied to conversation stages. We further demonstrate the utility of thoughts for downstream modeling. First, thoughts improve user-behavior prediction as inference-time context. Second, thought-guided rewrites provide fine-grained alignment signals for training personalized assistants. Together, ThoughtTrace establishes user thoughts as a new data modality for studying the cognitive dynamics behind human--AI interaction and provides a foundation for building assistants that better understand and adapt to users' latent goals, preferences, and needs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20072v1">Probing Embodied LLMs: When Higher Observation Fidelity Hurts Problem Solving</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 Submitted to From Animals to Animats: The 18th International Conference on the Simulation of Adaptive Behavior (SAB)
    </div>
    <details class="paper-abstract">
      Large Language Models are increasingly proposed as cognitive components for robotic systems, yet their opaque decision processes make it difficult to explain success or failure in closed-loop embodied tasks. Following an empirical AI methodology, we study embodied LLM agents behaviorally by varying the information available to the agent and measuring the resulting changes in behavior. Using the Lockbox, a sequential mechanical puzzle with hidden interdependencies, we evaluate LLMs across RGB, RGB-D, and ground-truth symbolic observations in a physical robotic setup and use controlled simulation to probe the resulting behavior. Counterintuitively, agents perform best under raw RGB input and worst under perfect ground-truth observations. In simulation, we probe this effect by randomly flipping perceived action outcomes and find that moderate noise improves performance, peaking at a 40% flip probability with a 2.85-fold success rate increase over the noise-free baseline. Further analysis links this gain to a reduction in repetitive action loops. These findings suggest that success rates alone are insufficient for evaluating LLMs, as measured performance may reflect the interaction between perceptual errors and reasoning failures rather than robust problem solving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20055v1">Towards LLM-Assisted Architecture Recovery for Real-World ROS~2 Systems: An Agent-Based Multi-Level Approach to Hierarchical Structural Architecture Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Explicit software architecture models are essential artifacts for communicating, analyzing, and evolving complex software-intensive systems. In ROS~2-based robotic systems, however, structural (de-)composition and integration semantics are often only implicitly encoded across distributed artifacts such as source code and launch files, making recovery of hierarchical architecture particularly difficult. Existing approaches mainly focus on node-level entities and communication wiring, while providing limited support for recovering hierarchical structural (de-)composition across multiple abstraction levels. In this paper, we extend our previously proposed blueprint-guided LLM-assisted architecture recovery pipeline for ROS~2 systems through two major enhancements: (1) refined prompting to improve the consistency and controllability of architecture synthesis, and (2) a staged recovery strategy based on multi-level intermediate architectural representations that incorporate the atomic ROS node list and launch file dependencies, thereby enabling structurally constrained reconstruction across multiple abstraction levels. The approach is evaluated on a real-world automated product disassembly system based on cooperative robotic arms and heterogeneous ROS~2 artifacts. Compared to our previous work, the considered case study exhibits substantially higher integration complexity and richer functionality. The results demonstrate improved structural consistency, scalability, and robustness of architecture recovery, while also revealing remaining challenges related to dynamic integration semantics in large-scale ROS~2 systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.09063v3">Soohak: A Mathematician-Curated Benchmark for Evaluating Research-level Math Capabilities of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 Under review, For questions or model-evaluation requests, contact $guijin.son@snu.ac.kr$
    </div>
    <details class="paper-abstract">
      Following the recent achievement of gold-medal performance on the IMO by frontier LLMs, the community is searching for the next meaningful and challenging target for measuring LLM reasoning. Whereas olympiad-style problems measure step-by-step reasoning alone, research-level problems use such reasoning to advance the frontier of mathematical knowledge itself, emerging as a compelling alternative. Yet research-level math benchmarks remain scarce because such problems are difficult to source (e.g., Riemann Bench and FrontierMath-Tier 4 contain 25 and 50 problems, respectively). To support reliable evaluation of next-generation frontier models, we introduce Soohak, a 439-problem benchmark newly authored from scratch by 64 mathematicians. Soohak comprises two subsets. On the Challenge subset, frontier models including Gemini-3-Pro, GPT-5, and Claude-Opus-4.5 reach 30.4%, 26.4%, and 10.4% respectively, leaving substantial headroom, while leading open-weight models such as Qwen3-235B, GPT-OSS-120B, and Kimi-2.5 remain below 15%. Notably, beyond standard problem solving, Soohak introduces a refusal subset that probes a capability intrinsic to research mathematics: recognizing ill-posed problems and pausing rather than producing confident but unjustified answers. On this subset, no model exceeds 50%, identifying refusal as a new optimization target that current models do not directly address. To prevent contamination, the dataset will be publicly released in late 2026, with model evaluations available upon request in the interim.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20035v1">Stage-adaptive Token Selection for Efficient Omni-modal LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 Code Link: https://github.com/xxayt/SEATS
    </div>
    <details class="paper-abstract">
      Omni-modal large language models (om-LLMs) achieve unified audio-visual understanding by encoding video and audio into temporally aligned token sequences interleaved at the window level. However, processing these dense non-textual tokens throughout the LLM incurs substantial computational overhead. Although training-free token selection can reduce this cost, existing methods either focus on visual-only inputs or prune om-LLM tokens only before the LLM with fixed per-modality ratios, failing to capture how cross-modal token importance evolves across layers. To address this limitation, we first analyze the layer-wise token dependency of om-LLMs. We find that visual and audio dependencies follow a block-wise pattern and gradually weaken with depth, indicating that many late-layer non-textual tokens become redundant after cross-modal fusion. Motivated by this observation, we propose SEATS, a training-free, stage-adaptive token selection method for efficient om-LLM inference. Before the LLM, SEATS removes spatiotemporal redundancy via attention-weighted diversity selection. Inside the LLM, it progressively prunes tokens across blocks and dynamically allocates the retention budget from temporal windows to modalities using query relevance scores. In late layers, it removes all remaining non-textual tokens once cross-modal fusion is complete. Experiments on Qwen2.5-Omni and Qwen3-Omni demonstrate that SEATS effectively improves inference efficiency. Retaining only 10% of visual and audio tokens, it achieves a 9.3x FLOPs reduction and a 4.8x prefill speedup while preserving 96.3% of the original performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19999v1">LLM Benchmark Datasets Should Be Contamination-Resistant</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 Accepted to ICML 2026 Position Paper Track
    </div>
    <details class="paper-abstract">
      Benchmark datasets are critical for reproducible, reliable, and discriminative evaluation of LLMs. However, recent studies reveal that many benchmark datasets are included in pretraining corpora, i.e., $\textit{contaminated}$, which diminishes their value as reliable measures of model generalization. In this paper, we argue that benchmark datasets should be $\textit{contamination-resistant}$, i.e., $\textit{unlearnable}$, but support $\textit{inference}$. To accomplish this, we first highlight the wide prevalence of benchmark dataset contamination and outline the properties of contamination-resistant datasets. Second, we highlight how the asymmetry between the inference and training pipelines in the Transformer architecture can be leveraged to support contamination-resistance. Third, we outline mathematical advancements to make these datasets interoperable across various LLM architectures. Based on the above, we call on the community to ensure the reliability of LLM benchmarking by: (i) advancing novel contamination-resistant methodologies, (ii) developing supporting methods and platforms, and (iii) adopting contamination-resistant benchmarks into existing evaluation pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19952v1">Rethinking How to Remember: Beyond Atomic Facts in Lifelong LLM Agent Memory</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      To enable reliable long-term interaction, LLM agents require a memory system that can faithfully store, efficiently retrieve, and deeply reason over accumulated dialogue history. Most existing methods adopt an extracted fact based paradigm: handcrafted static prompts compress raw dialogues into atomic facts, which are then stored, matched, and injected into downstream reasoning. Nevertheless, such fact-centric designs inevitably discard fine-grained details in original dialogues and fail to support deep reasoning over scattered isolated facts. Moreover, static prompts cannot maintain consistent extraction granularity across diverse dialogue styles. To address these limitations, we propose TriMem, which maintains three coexisting representation granularities, including raw dialogue segments anchored by source identifiers for storage fidelity, extracted atomic facts for efficient memory retrieval, synthesized profiles that aggregate dispersed facts into holistic semantic understanding for deep reasoning. We further adopt TextGrad-based prompt optimization, which iteratively refines extraction and profiling prompts via response quality feedback, achieving lifelong evolution without any parameter updating. Extensive experiments on LoCoMo and PerLTQA across multiple LLM backbones demonstrate that TriMem consistently outperforms strong memory baselines. The code is available at https://TMLR-TriMem.github.io .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19936v1">What Are LLMs Doing to Scientific Communication? Measuring Changes in Writing Practices and Reading Experience</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 Accepted to LREC 2026
    </div>
    <details class="paper-abstract">
      Has the style of scientific communication changed due to the growing use of large language models in the writing process? We address this question in the domain of Natural Language Processing by leveraging two data resources we create: a naturalistic corpus of over 37,000 papers from the ACL Anthology (2020-2024); and a synthetic dataset of 3,000 human-written passages and their LLM-generated improvements. We first implement a series of diachronic lexical analyses, showing that both word frequency and usage contexts have changed significantly over time, indicating semantic specialization in some cases and generalization in others. Broadening our perspective, we then model a range of more complex stylistic features and find that LLM-modified texts more frequently contain certain syntactic constructions, more complex and longer words and a lower lexical diversity. Finally, we connect these changes in writing practices to subjective reading experience through a pilot annotation study with 20 domain experts. They overall rate LLM-improved texts as more understandable and exciting, but also express negative qualitative attitudes towards LLMs, highlighting the strongly subjective effect of AI-assisted writing on reading experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19932v1">PEEK: Context Map as an Orientation Cache for Long-Context LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents increasingly operate over long and recurring external contexts, like document corpora and code repositories. Across invocations, existing approaches preserve either the agent's trajectory, passive access to raw material, or task-level strategies. None of them preserves what we argue is most needed for repeated same-context workloads: reusable orientation knowledge (e.g., what the context contains, how it is organized, and which entities, constants, and schemas have historically been useful) about the recurring context itself. We introduce PEEK, a system that caches and maintains this orientation knowledge as a context map: a small, constant-sized artifact in the agent's prompt that gives it a persistent peek into the external context. The map is maintained by a programmable cache policy with three modules: a Distiller that extracts transferable knowledge from inference-time signals, a Cartographer that translates it into structured edits, and a priority-based Evictor that enforces a fixed token budget. On long-context reasoning and information aggregation, PEEK improves over strong baselines by 6.3-34.0% while using 93-145 fewer iterations and incurring 1.7-5.8x lower cost than the state-of-the-art prompt-learning framework, ACE. On context learning, PEEK improves solving rate and rubric accuracy by 6.0-14.0% and 7.8-12.1%, respectively, at 1.4x lower cost than ACE. These gains generalize across LMs and agent architectures, including OpenAI Codex, a production-grade coding agent. Together, these results show that a context map helps long-context LLM agents interact with recurring external contexts more accurately and efficiently.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11768v2">Governing Evolving Memory in LLM Agents: Risks, Mechanisms, and the Stability and Safety Governed Memory (SSGM) Framework</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Long-term memory has emerged as a foundational component of autonomous Large Language Model (LLM) agents, enabling continuous adaptation, lifelong multimodal learning, and sophisticated reasoning. However, as memory systems transition from static retrieval databases to dynamic, agentic mechanisms, critical concerns regarding memory governance, semantic drift, and privacy vulnerabilities have surfaced. While recent surveys have focused extensively on memory retrieval efficiency, they largely overlook the emergent risks of memory corruption in highly dynamic environments. To address these emerging challenges, we propose the Stability and Safety-Governed Memory (SSGM) framework, a conceptual governance architecture. SSGM decouples memory evolution from execution by enforcing consistency verification, temporal decay modeling, and dynamic access control prior to any memory consolidation. Through formal analysis and architectural decomposition, we show how SSGM can mitigate topology-induced knowledge leakage where sensitive contexts are solidified into long-term storage, and help prevent semantic drift where knowledge degrades through iterative summarization. Ultimately, this work provides a comprehensive taxonomy of memory corruption risks and establishes a robust governance paradigm for deploying safe, persistent, and reliable agentic memory systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19915v1">LLM Agents Make Collective Belief Dynamics Programmable: Challenges and Research Directions</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Classical models of opinion dynamics assume human participants with bounded rationality and limited coordination. The rise of LLM-based agents introduces a qualitative shift: agents can now participate in online discussions at scale, maintain consistent persuasion strategies, and coordinate systematically. This paper argues that LLM agents make collective belief dynamics programmable, enabling deliberate steering of population-level beliefs. We term this emerging problem programmable collective belief control. Through controlled multi-agent simulations, we provide proof-of-concept evidence that coordinated AI agents can induce measurable belief shifts that stabilize within a few interaction rounds. We identify four structural properties (indistinguishability, persistence, contextuality, and configurability) that make detection and defense fundamentally difficult. Based on these findings, we outline a research agenda spanning theoretical foundations for adversarial belief dynamics, operational methods for system-level detection and intervention, and simulation infrastructure for scalable experimentation. Our goal is not to present a complete solution, but to articulate why this problem demands urgent attention and to provide a conceptual foundation for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19901v1">Can LLMs Produce Better Object-Oriented Designs than Human-Involved Development?</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Background: Large Language Models (LLMs) are increasingly used for code generation. However, their ability to generate multi-class projects that require object-oriented design (OOD) remains unclear, especially relative to projects developed with human involvement. Aims: The primary objective of this study is to compare OOD quality in projects from three authorship conditions: PreAI (human-involved projects produced before widespread LLM use), PostAI (human-involved projects produced after widespread LLM use), and PureAI (projects generated end-to-end by contemporary LLMs). Method: We conducted a comparative case study on a postgraduate Java assignment. Two offerings of the same assignment were selected as the PreAI and PostAI datasets. PureAI projects were generated using three contemporary LLMs. We analyzed OOD quality using project-level OOD metrics, code smell density, and domain modeling. Results: Relative to human-involved projects, PureAI projects show lower code smell density and generally appear simpler in terms of total size, complexity, and coupling. However, this is consistent with oversimplification, as it is associated with missing abstractions and weaker responsibility separation. PostAI is closer to PureAI than PreAI on many OOD measures and also shows tendencies toward oversimplification. Conclusions: Our findings indicate that appropriate human guidance on object-oriented decomposition and responsibility assignment remains important when LLMs are used for object-oriented design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.13793v2">An LLM-Based System for Argument Mining</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Arguments are a fundamental aspect of human reasoning, in which claims are supported, challenged, and weighed against one another. We present an end-to-end large language model (LLM)-based system for reconstructing arguments from natural language text into abstract argument graphs. The system follows a multi-stage pipeline that progressively identifies argumentative components, selects relevant elements, and uncovers their logical relations. These elements are represented as directed acyclic graphs consisting of two component types (premises and conclusions) and three relation types (support, attack, and undercut). We conduct two complementary experiments to evaluate the system. First, we perform a manual evaluation on arguments drawn from an argumentation theory textbook to assess the system's ability to recover argumentative structure. Second, we conduct a quantitative evaluation on benchmark datasets, allowing comparison with prior work by mapping our outputs to established annotation schemes. Results show that the system can adequately recover argumentative structures and, when adapted to different annotation schemes, achieve reasonable performance across benchmark datasets. These findings highlight the potential of LLM-based pipelines for scalable argument mining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19798v1">Towards Trust Calibration in Socially Interactive Agents: Investigating Gendered Multimodal Behaviors Generation with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      As Socially Interactive Agents (SIAs) become increasingly integrated into daily life, the ability to calibrate user trust to an agent's actual capabilities would help ensure appropriate usage of these agents. In this paper, we explore the capacity of Large Language Models (LLMs) to generate multimodal behaviors (verbal, vocal, gestural, and facial expression modalities) that reflect varying levels of ability and benevolence, two key dimensions of trustworthiness. We propose a novel method for automatically generating behaviors aligned with specific levels of these traits, a first step towards enabling nuanced and trust-calibrated interactions. By analyzing a large dataset of multimodal transcripts generated by LLMs, we demonstrate that GPT-5.4 is able to produce coherent behavior across different modalities (text, intonation, facial expression, and gesture). Using Random Forest feature importance analysis, we show that the generated behaviors align with theoretical expectations for ability and benevolence. However, we also find that when gender is specified in the prompt, LLMs tend to reproduce societal gender stereotypes, associating male agents' behaviors with high ability and female agents' behaviors with high benevolence. To validate our approach, we conducted a user study on Prolific using a within-subjects design. Participants perceived different levels of ability and benevolence in the generated behaviors align with the intended instructions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19782v1">Prior Knowledge or Search? A Study of LLM Agents in Hardware-Aware Code Optimization</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      LLM discovery and optimization systems are increasingly applied across domains, implementing a common propose-evaluate-revise loop. Such optimization or discovery progresses via context conditioning on received feedback from an environment. However, as modern LLM agents are increasingly complex in their structure, it is difficult to evaluate which components contribute the most, and when and how this exploration may fail. We answer these questions through three controlled experiments. Our findings: (1) In pure black-box optimization, LLMs act as greedy optimizers. (2) In zero-shot kernel generation, providing explicit input-size information has no measurable effect, models converge to the same kernel parameters regardless of size or temperature, as though the size instruction were invisible. Moreover, when tasked to perform kernel optimization for uncommon kernel sizes, performance sharply degrades regardless of the language used. (3) In feedback-loop kernel optimization, CUDA improves monotonically under iterative feedback, while TVM IR actively degrades, which demonstrates that kernel optimization degrades when models operate with low-density language. Our results conclude that LLMs in code optimization tasks highly depend on pretrained priors rather than provided feedback or agentic structure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19743v1">EngiAI: A Multi-Agent Framework and Benchmark Suite for LLM-Driven Engineering Design</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 26 pages, 10 figures, to be published at IDETC 2026
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents are increasingly applied to engineering design tasks, yet existing evaluation frameworks do not adequately address multi-agent systems that combine simulation, retrieval, and manufacturing preparation. We introduce a benchmark suite with three evaluation dimensions: (1) a workflow benchmark with seven prompt styles targeting distinct cognitive demands-including direct tool use, semantic disambiguation, conditional branching, and working-memory tasks; (2) a Retrieval-Augmented Generation (RAG) benchmark with gated scoring isolating retrieval contributions to parameter selection; and (3) an High Performance Computing (HPC) benchmark evaluating end-to-end ML training orchestration on a SLURM cluster. Alongside the benchmark we present EngiAI, a Multi-Agent System (MAS) reference implementation built on LangGraph that operationalizes the benchmark by coordinating seven specialized agents through a supervisor architecture, unifying topology optimization, document retrieval, HPC job orchestration, and 3D printer control. Across four LLM backends and two EngiBench problems, proprietary models achieve 96-97% average task completion on Beams2D, while open-source 4B-parameter models reach 55-78%, with clear generational improvement. Conditional branching proves most challenging, with task completion dropping to 20-53% for the conditional style on Photonics2D. RAG gating confirms near-perfect retrieval-augmented scores ($\approx 1.0$) versus near-zero without retrieval, validating the evaluation design. On HPC orchestration, one model completes all pipeline steps in 100% of runs while another drops to 50%, revealing that multi-step instruction following degrades over long-running workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.07066v2">2.5-D Decomposition for LLM-Based Spatial Construction</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Autonomous systems that build structures from natural-language instructions need reliable spatial reasoning, yet large language models (LLMs) make systematic coordinate errors when generating three-dimensional block placements. We present a neuro-symbolic pipeline based on \emph{2.5-D decomposition}: the LLM plans in the two-dimensional horizontal plane while a deterministic executor computes all vertical placement from column occupancy, eliminating an entire class of errors. On the Build What I Mean benchmark (160 rounds), GPT-4o-mini with this pipeline achieves 94.6\% mean structural accuracy across 12 independent runs, within 3.0 percentage points of the 97.6\% ceiling imposed by architect-agent errors that no builder-side improvement can address. This outperforms both GPT-4o at 90.3\% and the best competing system at 76.3\%. A controlled ablation confirms that 2.5-D decomposition is the dominant contributor, accounting for 50.7 percentage points of accuracy. The pipeline transfers directly to edge hardware: Nemotron-3 120B running locally on an NVIDIA Jetson Thor AGX matches the cloud result at 94.5\% with no prompt modifications. The underlying principle, removing deterministic dimensions from the LLM's output space, applies to any autonomous construction or assembly task where gravity or other physical constraints fix one or more degrees of freedom. A transfer experiment on 500 IGLU collaborative building tasks confirm the effect generalizes beyond the primary benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.01482v2">Towards Consistent Detection of Cognitive Distortions: LLM-Based Annotation and Dataset-Agnostic Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Text-based automated Cognitive Distortion detection is a challenging task due to its subjective nature, with low agreement scores observed even among expert human annotators, leading to unreliable annotations. We explore the use of Large Language Models (LLMs) as consistent and reliable annotators, and propose that multiple independent LLM runs can reveal stable labeling patterns despite the inherent subjectivity of the task. Furthermore, to fairly compare models trained on datasets with different characteristics, we introduce a dataset-agnostic evaluation framework using Cohen's kappa as an effect size measure. This methodology allows for fair cross-dataset and cross-study comparisons where traditional metrics like F1 score fall short. Our results show that GPT-4 can produce consistent annotations (Fleiss's Kappa = 0.78), resulting in improved test set performance for models trained on these annotations compared to those trained on human-labeled data. Our findings suggest that LLMs can offer a scalable and internally consistent alternative for generating training data that supports strong downstream performance in subjective NLP tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.04588v3">ZeroSearch: Incentivize the Search Capability of LLMs without Searching</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Effective information searching is essential for enhancing the reasoning and generation capabilities of large language models (LLMs). Recent research has explored using reinforcement learning (RL) to improve LLMs' search capabilities by interacting with live search engines in real-world environments. While these approaches show promising results, they face two major challenges: (1) Uncontrolled Document Quality: The quality of documents returned by search engines is often unpredictable, introducing noise and instability into the training process. (2) Prohibitively High API Costs: RL training requires frequent rollouts, potentially involving hundreds of thousands of search requests, which incur substantial API expenses and severely constrain scalability. To address these challenges, we introduce ZeroSearch, a novel RL framework that incentivizes the capabilities of LLMs to use a real search engine with simulated searches during training. Our approach begins with lightweight supervised fine-tuning to transform the LLM into a retrieval module capable of generating both useful and noisy documents in response to a query. During RL training, we employ a curriculum-based rollout strategy that incrementally degrades the quality of generated documents, progressively eliciting the model's reasoning ability by exposing it to increasingly challenging retrieval scenarios. Extensive experiments demonstrate that ZeroSearch effectively incentivizes the search capabilities of LLMs using a 3B LLM as the retrieval module. Remarkably, a 7B retrieval module achieves comparable performance to the real search engine, while a 14B retrieval module even surpasses it. Furthermore, it generalizes well across both base and instruction-tuned models of various parameter sizes and is compatible with a wide range of RL algorithms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19660v1">OScaR: The Occam's Razor for Extreme KV Cache Quantization in LLMs and Beyond</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      The rapid advancement toward long-context reasoning and multi-modal intelligence has made the memory footprint of the Key-Value (KV) cache a dominant memory bottleneck for efficient deployment. While the established per-channel quantization effectively accommodates intrinsic channel-wise outliers in Key tensors, its efficacy diminishes under extreme compression. In this work, we revisit the inherent limitations of the per-channel quantization paradigm from both empirical and theoretical perspectives. Our analysis identifies Token Norm Imbalance (TNI) as the primary bottleneck to quantization fidelity. We demonstrate that TNI systematically amplifies errors when shared quantization parameters are required to span token groups exhibiting substantial norm disparities. Instead of relying on intricate quantization pipelines (e.g., TurboQuant), we propose OScaR (Omni-Scaled Canalized Rotation), an accurate and lightweight KV cache compression framework for X-LLMs (i.e., text-only, multi-modal, and omni-modal LLMs). Advancing the per-channel paradigm, OScaR employs Canalized Rotation followed by Omni-Token Scaling to mitigate TNI-induced sequence-dimensional variance both effectively and efficiently, further supported by our optimized system design and CUDA kernels. Extensive evaluations across X-LLMs show that OScaR consistently outperforms existing methods and achieves near-lossless performance under INT2 quantization, establishing it as a robust, low-complexity, and universal framework that defines a new Pareto front. Compared with the BF16 FlashDecoding-v2 baseline, our OScaR implementation achieves a notable up to 3.0x speedup in decoding, reduces memory footprint by 5.3x, and increases throughput by 4.1x. The code for OScaR is publicly available at https://github.com/ZunhaiSu/OScaR-KV-Quant.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20295v1">Quant.npu: Enabling Efficient Mobile NPU Inference for on-device LLMs via Fully Static Quantization</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed on mobile devices, where Neural Processing Units (NPUs) necessitate fully static quantization for optimal inference efficiency. However, existing post-training quantization (PTQ) methods predominantly rely on dynamic activation quantization, rendering them incompatible with NPU hardware constraints. To bridge the gap between high-fidelity PTQ and NPU-constrained inference, we propose Quant.npu, a integer-only fully static quantization framework. It incorporates learnable quantization parameters and rotation matrices, enabling low-bit activation-weight quantization without runtime quantization parameters re-computation. Crucially, we identify that initialization and selective optimization of quantization parameters is pivotal for optimization stability, as improper initialization and naive joint optimization induce gradient instability that disrupts the optimization of rotation matrices. To address this, we propose a rotation-and-bit-width-aware initialization tailored to diverse activation profiles and a distribution-aware selective optimization (two-stage quantization pipeline) tailored to rotated and unrotated tensors. Furthermore, we introduce a sensitivity-guided adaptive mixed-precision scheme to balance accuracy with inference efficiency. Extensive experiments on real-world mobile NPUs demonstrate that Quant.npu achieves comparable accuracy to state-of-the-art methods, while reducing inference latency by up to 15.1%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17839v3">How do LLMs Compute Verbal Confidence</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Verbal confidence -- prompting LLMs to state their confidence as a number or category -- is widely used to extract uncertainty estimates from black-box models. However, how LLMs internally generate such scores remains unknown. We address two questions: first, when confidence is computed -- just-in-time when requested, or automatically during answer generation and cached for later retrieval; and second, what verbal confidence represents -- token log-probabilities, or a richer evaluation of answer quality? Focusing on Gemma 3 27B (across TriviaQA, BigMath, and MMLU), Qwen 2.5 7B, and the reasoning model Magistral Small 24B, we provide convergent evidence for cached retrieval. Activation steering, patching, noising, and swap experiments reveal that confidence representations emerge at answer-adjacent positions before appearing at the verbalization site. Attention blocking pinpoints the information flow: confidence is gathered from answer tokens, cached at the first post-answer position, then retrieved for output. Critically, linear probing and variance partitioning reveal that these cached representations explain substantial variance in verbal confidence beyond token log-probabilities, suggesting a richer answer-quality evaluation rather than a simple fluency readout. These findings demonstrate that verbal confidence reflects automatic, sophisticated self-evaluation -- not post-hoc reconstruction -- with implications for understanding metacognition in LLMs and improving calibration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19627v1">How Helpful is LLM Assistance in Network Operations? A Case Study at a Large Demonstration Network</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      This paper reports on a real-world case study in which over 100 network engineers assessed how a Large Language Model (LLM) can assist in building and operating a network. The versatility of LLMs has accelerated their adoption across a wide range of domains, and assisting network operations is one such promising application. LLMs are probabilistic models, unlike deterministic protocols and configurations; therefore, clarifying their capabilities -- how and to what extent LLMs can help in network operations -- is a crucial step toward adopting LLMs. To offer practical insights into this issue, we conducted an extensive experiment on a large demonstration network built for a public exhibition, consisting of 21 racks with heterogeneous network devices. In the experiment, a total of 105 network engineers used an LLM-based chatbot while building and operating the network. The chatbot was equipped with three external functions: retrieval-augmented generation for domain-specific knowledge, CLI control of network devices running on the network, and access to a ticket system. The participants gave evaluations for the chatbot's responses on a best-effort basis. Analysis of the chat histories shows that 68.1% of the evaluations were positive, indicating a quantitative baseline of the LLM's helpfulness in network operations. Our results also demonstrate that understanding the capabilities of the chatbot is important for eliciting better responses. Moreover, we provide detailed use case analyses while sharing actual user--chatbot interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.17726v3">Slot-MLLM: Object-Centric Visual Tokenization for Multimodal LLM</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Recently, multimodal large language models (MLLMs) have emerged as a key approach in achieving artificial general intelligence. In particular, vision-language MLLMs have been developed to generate not only text but also visual outputs from multimodal inputs. This advancement requires efficient image tokens that LLMs can process effectively both in input and output. However, existing image tokenization methods for MLLMs typically capture only global abstract concepts or uniformly segmented image patches, restricting MLLMs' capability to effectively understand or generate detailed visual content, particularly at the object level. To address this limitation, we propose an object-centric visual tokenizer based on Slot Attention specifically for MLLMs. In particular, based on the Q-Former encoder, diffusion decoder, and residual vector quantization, our proposed discretized slot tokens can encode local visual details while maintaining high-level semantics, and also align with textual data to be integrated seamlessly within a unified next-token prediction framework of LLMs. The resulting Slot-MLLM demonstrates significant performance improvements over baselines with previous visual tokenizers across various vision-language tasks that entail local detailed comprehension and generation. Notably, this work is the first demonstration of the feasibility of object-centric slot attention performed with MLLMs and in-the-wild natural images.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19604v1">Formal Skill: Programmable Runtime Skills for Efficient and Accurate LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents increasingly act inside real workspaces, where tools and skills determine whether model reasoning becomes reliable action. Existing skills remain largely informal: Markdown skills and instruction packs encode procedures as long natural-language documents, while function calling, Model Context Protocol (MCP) servers, and framework tools structure individual actions but usually leave workflow state, policy enforcement, and completion discipline outside the skill itself. We introduce Formal Skill, a runtime-native abstraction that represents reusable capability with JSON metadata and action schemas, reliable Python executors, hook-governed control logic, Formal Skill routing, and skill-local runtime state. By moving reusable procedure from repeated prompt text into executable state machines and hook policies, Formal Skill gives agents a token-efficient and enforceable control surface. We implement the abstraction in FairyClaw, an open-source event-driven runtime for executable, observable, and composable Formal Skills. On Harness-Bench, FairyClaw obtains highly competitive average scores while using substantially fewer tokens, with especially strong results on tasks that expose the role of Formal Skill.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19597v1">LLMEval-Logic: A Solver-Verified Chinese Benchmark for Logical Reasoning of LLMs with Adversarial Hardening</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Evaluating large language models (LLMs) on natural-language logical reasoning is essential because rule-governed tasks require conclusions to follow strictly from stated premises. Many existing logical-reasoning benchmarks are generated by templating natural-language items from sampled formulas, provide only coarse or unaudited formal annotations, and are now quickly saturated by frontier reasoning models. We present LLMEval-Logic, a Chinese logical reasoning benchmark built from realistic situational scenarios. Its pipeline forward-authors and expert-audits natural-language items together with their reference formalizations, verifies annotated answers with Z3, constructs expert rubrics for natural-to-formal grading, and hardens selected items through a closed-loop adversarial workflow. The benchmark is released in two paired subsets: a 246-item Base subset shipped with 1,400 expert-developed rubric atoms, and a 190-item Hard subset with 938 multi-step sub-questions over closed model spaces. Evaluating 14 frontier LLMs on LLMEval-Logic reveals substantial gaps in current models: the best model reaches only 37.5% Hard Item Accuracy, and even with reference symbols the highest joint Z3+Rubric formalization score among evaluated models reaches only 60.16%. Our benchmark is publicly available at https://github.com/llmeval/LLMEval-Logic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19595v1">A novel YOLO26-MoE optimized by an LLM agent for insulator fault detection considering UAV images</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      The inspection of electrical power line insulators is essential for ensuring grid reliability and preventing failures caused by damaged or degraded insulation components. In recent years, Unmanned Aerial Vehicles (UAVs) combined with deep learning-based vision systems have emerged as an effective solution for automating this process. However, insulator fault detection remains challenging due to small defect regions, heterogeneous fault patterns, complex backgrounds, and varying imaging conditions. To address these challenges, this paper proposes an optimized YOLO26-MoE, a novel object detection architecture that integrates a sparse Mixture-of-Experts (MoE) module into the high-resolution branch of the YOLO26 detector. The proposed modification enables adaptive feature refinement for subtle and diverse fault patterns while preserving the efficiency of a one-stage detection framework. Hyperparameter optimization, final training, and evaluation were coordinated through a tool-augmented Large Language Model (LLM) agent. The proposed model achieved 0.9900 mAP@0.5 and 0.9515 mAP@0.5:0.95, outperforming the latest YOLO versions. These results demonstrate that the proposed model provides an effective and reliable solution for UAV-based insulator fault detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19593v1">Towards Multi-Model LLM Schedulers: Empirical Insights into Offloading and Preemption</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 The 2026 Mediterranean Artificial Intelligence and Networking Conference (MAIN 2026)
    </div>
    <details class="paper-abstract">
      Modern deployments of Large Language Models (LLMs) increasingly require serving multiple models with diverse architectures, sizes, and specialization on shared, heterogeneous hardware. This setting introduces new challenges for resource allocation, dispatching, and scheduling, particularly under GPU memory constraints where partial CPU-GPU offloading and preemption become necessary. While existing systems primarily optimize throughput for a single model, comparatively little work addresses multi-model scheduling under these conditions. In this paper, we present an empirical study of how different LLMs behave across hardware platforms, focusing on the performance implications of layer offloading and preemption. We show that offloading leads to strongly non-linear and model-dependent degradation in decode throughput, with smaller models exhibiting sharper sensitivity to reduced GPU residency. We further demonstrate that preemption incurs substantial overhead, largely dominated by model state reload rather than key-value cache transfer, and that this cost varies significantly across models and hardware platforms. Additionally, we highlight the role of sequence length and interconnect bandwidth in amplifying data movement and execution inefficiencies. Based on these findings, we identify a set of key features that future schedulers must consider, including model-specific offloading sensitivity, workload characteristics, and the cost structure of preemption and data transfer. These insights provide guidance for the design of next-generation LLM serving systems capable of efficiently managing heterogeneous, multi-model workloads with hybrid CPU-GPU execution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19576v1">Library Drift: Diagnosing and Fixing a Silent Failure Mode in Self-Evolving LLM Skill Libraries</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Self-evolving skill libraries face a silent failure mode we term \emph{library drift}: unbounded skill accumulation without outcome-driven lifecycle management causes retrieval degradation, false-positive injections, and performance stagnation. Recent evaluation confirms the symptom--LLM-authored skills deliver +0.0pp gain while human-curated ones deliver +16.2pp (SkillsBench)--yet the underlying mechanism has not been isolated. We provide (1) a reproducible trigger: ablations that isolate drift--one disables skill injection (flat floor, +0.002), one imposes premature retirement (active harm, $-$0.019); (2) trace-level diagnostics: an append-only evidence log with per-skill contribution scores, attribution verdicts, and router engagement metrics that make the failure visible before it reaches end-task scores; and (3) a verified fix: a minimal governance recipe (outcome-driven retirement + bounded active-cap + meta-skill authoring prior) that lifts held-out pass@1 from a 0.258 baseline to a late-window mean of 0.584 (rolling gain $+$0.328) on MBPP+ hard-100 over 100 rounds. Eight ablations decompose which governance mechanisms are load-bearing and which are subsumed, providing a concrete playbook for diagnosing library drift in any self-evolving agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.09997v2">GraphInstruct: A Progressive Benchmark for Diagnosing Capability Gaps in LLM Graph Generation</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 44 pages, 17 figures
    </div>
    <details class="paper-abstract">
      Graph-structured data underpins applications from citation analysis and social-network modeling to molecular design and knowledge-graph construction, and Large Language Models (LLMs) are increasingly used as prompt-driven graph synthesizers. Classical graph-generation reviews catalog deep generative models and their evaluation primitives, but predate the LLM era and provide no foundation for evaluating instruction-following graph synthesis. Recent LLM-era benchmarks evaluate models along graph-type or task-domain axes; such organizations, however, average over structural complexity and cannot localize where in the complexity spectrum an LLM breaks down. To close this diagnostic gap, we introduce GraphInstruct, a progressive-complexity benchmark that stratifies LLM graph generation into six complexity levels and five evaluation dimensions, paired with 800 hand-authored instructions, 1,582 algorithmically synthesized reference solutions, and a 12-LLM capability evaluation across 45 (model, strategy) configurations. We find that discriminative power peaks at multi-constraint composition rather than reasoning depth, that no single prompting strategy dominates across levels or model families, and that domain-semantic constraints remain iteration-invariant under all tested methods -- pointing to retrieval rather than additional compute as the next research frontier. Atop the benchmark, a verification-guided iterative framework with constraint-aware adaptive prompting consistently surpasses the prompt-engineering ceiling on tested target models, demonstrating that the benchmark's fine-grained signals drive method development. Data, code, and reproducibility artifacts are released alongside the paper at https://github.com/AI4DataSynth/GraphInstruct_formal
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.22202v3">Library Hallucinations in LLM-Generated Code: A Risk Analysis Grounded in Developer Queries</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 27 pages, 1 figure, 13 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) now play a central role in code generation, yet they continue to hallucinate, frequently inventing non-existent libraries. Such library hallucinations are not just benign errors: they can mislead developers, break builds, and expose systems to supply chain threats such as slopsquatting. Despite growing awareness of these risks, there is limited understanding of how library hallucinations manifest under realistic usage conditions. To fill this gap, we present the first systematic study of how user-level prompt variations influence library hallucinations in LLM-generated code. Across seven diverse LLMs, we analyse library name hallucinations (invalid imports) and library member hallucinations (invalid calls from valid libraries), examining the effects of realistic developer language and controlled user mistakes, including misspellings and fabricated libraries or members. Our findings expose systemic vulnerabilities: one-character misspellings trigger hallucinations in up to 26% of tasks; fabricated library names are accepted in up to 99%; and time-based prompts induce hallucinations in up to 85%. Grounded in the highest-risk prompts identified in our study, we introduce LibHalluBench, a benchmark that enables a systematic and reproducible evaluation of these library hallucinations. Our findings underscore the fragility of LLMs to natural prompt variation and highlight the urgent need for safeguards against library-related hallucinations and their downstream risks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19529v1">Generative-Evaluative Agreement: A Necessary Validity Criterion for LLM-Enabled Adaptive Assessment</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 BEA 2026
    </div>
    <details class="paper-abstract">
      When the same LLM generates assessment items, simulates student responses, and scores them, the validation loop is self-referential. We introduce Generative-Evaluative Agreement (GEA), a validity criterion measuring whether an LLM's scoring function recovers the skill levels its generative function was instructed to produce. In the first direct measurement of GEA on a two-stage adaptive assessment, the model recovers roughly half the intended variance r = 0.698 with systematic positive bias. GEA is strong r > 0.7 for syntactically verifiable skills but near zero for design-level skills, and low-skill overestimation inflates scores near the routing threshold. We argue that granular, skill-decomposed rubrics are the principal proposed mechanism for strengthening GEA and outline complementary mitigations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.18474v2">Prompt2Fingerprint: Plug-and-Play LLM Fingerprinting via Text-to-Weight Generation</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      The widespread deployment and redistribution of large language models (LLMs) have made model provenance tracking a critical challenge. While existing LLM fingerprinting methods, particularly active approaches that embed identity signals via fine-tuning, achieve high accuracy and robustness, they suffer from significant scalability bottlenecks. These methods typically treat fingerprint injection as an independent, one-off optimization task rather than a reusable capability, necessitating separate, resource-intensive training for every new identity. This incurs prohibitive computational costs and deployment delays. To address this, we propose Prompt2Fingerprint (P2F), the first framework that reformulates fingerprinting as a conditional parameter generation task. By leveraging a specialized generator, P2F maps textual descriptions directly to low-rank parameter increments in a single forward pass, enabling plug-and-play LLM fingerprint injection without further model retraining. Our experiments demonstrate that P2F maintains high fingerprint accuracy, harmlessness, and robustness while significantly reducing computational overhead, offering a scalable and instant solution for LLM ownership management.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19518v1">BLINKG: A Benchmark for LLM-Integrated Knowledge Graph Generation</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Generating Knowledge Graphs (KGs) remains one of the most time-consuming and labor-intensive tasks for knowledge engineers, as they need to identify semantic equivalences between input data sources and ontology terms. While declarative solutions (e.g., RML, SPARQL-Anything) have helped to generalize this process, aligning input schema elements with ontology terms still involves intricate transformations and requires considerable manual effort. With the advent of Large Language Models (LLMs), there is growing interest in leveraging their capabilities to assist KG engineers. Although some studies have explored using LLMs to automate KG construction, there is still no standardized framework for assessing how effectively they establish correspondences between data schemes and ontology concepts. Therefore, in this paper, we propose BLINKG, a benchmark designed to evaluate the mapping capabilities of LLMs in constructing KGs from heterogeneous data sources. The benchmark includes a set of scenarios with increasing complexity, based on real-world use cases. We conduct an extensive experimental evaluation of several stateof-the-art LLMs using BLINK and observe that they already offer promising solutions. However, their performance remains limited in complex scenarios. Thanks to this benchmark, we can already assess the current capabilities of LLMs for KG construction. Additionally, we define a set of requirements for achieving (semi)automated (LLM-driven) KG construction, opening new research lines in this area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19481v1">C2CServe: Leveraging NVLink-C2C for Elastic Serverless LLM Serving on MIG</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Modern LLM serving is increasingly serverless in shape: large model catalogs, long-tail invocations, and multi-tenant demand. Existing GPU serving systems face a tradeoff: dedicated-GPU allocation wastes scarce HBM under sparse traffic, while GPU time sharing places model initialization and weight loading on the cold-start path. Spatial GPU sharing such as multi-instance GPU (MIG) provides isolation and accounting, but each slice has too little HBM for modern LLM weights. We observe that high-bandwidth CPU--GPU interconnects, such as NVLink-C2C (C2C) in NVIDIA GH200 and GB200 Superchips, change the memory constraint: model weights can reside in CPU memory and be streamed on demand to MIG instances, shifting model residency from scarce HBM to abundant host memory. Leveraging this capability, we present C2CServe, a request-granularity serverless LLM serving system that allows MIG instances to switch models across requests without reloading weights into HBM. C2CServe introduces HybridGEMM, a heterogeneous-memory-aware GEMM kernel that adapts data access patterns to balance HBM and C2C bandwidth across MIG partitions using a single tuning knob. To mitigate shared-C2C contention, C2CServe further uses a hierarchical scheduler that coordinates model placement, input chunking, and kernel selection with online feedback control. On GH200, C2CServe reduces cold-start latency by up to 7.1x for dense models and 4.6x for MoE models compared with state-of-the-art serverless LLM serving systems, while maintaining over 95\% TTFT and TPOT attainment under C2C contention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11767v3">TSR: Trajectory-Search Rollouts for Multi-Turn RL of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Advances in large language models (LLMs) are driving a shift toward using reinforcement learning (RL) to train agents from iterative, multi-turn interactions across tasks. However, multi-turn RL remains challenging as rewards are often sparse or delayed, and environments can be stochastic. In this regime, naive trajectory sampling can hinder exploitation and induce mode collapse. We propose TSR (Trajectory-Search Rollouts), a training-time approach that repurposes test-time scaling ideas for improved per-turn rollout generation. TSR performs lightweight tree-style search to construct high-quality trajectories by selecting high-scoring actions at each turn using state-based feedback. This improves rollout quality and stabilizes learning while remaining compatible with standard policy gradient optimizers, making TSR optimizer-agnostic. We instantiate TSR with best-of-N, beam, and shallow lookahead search, and pair it with PPO and GRPO, achieving up to 15% performance gains and more stable learning on Sokoban, FrozenLake, and WebShop tasks at a modest, one-time increase in training compute. By moving search from inference time to the rollout stage of training, TSR provides a modular and general mechanism for stronger multi-turn agent learning, complementary to existing frameworks and rejection-sampling-style selection methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19433v1">Backtracking When It Strays: Mitigating Dual Exposure Biases in LLM Reasoning Distillation</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 26 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable success in complex reasoning tasks via long chain-of-thought (CoT), yet their immense computational overhead hinders real-world deployment. LLM reasoning distillation addresses this by transferring reasoning capabilities from formidable teacher models to compact student models. However, existing distillation paradigms face a fundamental dilemma. Typical off-policy distillation strictly utilizes teacher-generated golden trajectories, suffering from an exposure bias due to the mismatch between training distributions and student-generated inference contexts, which leads to error cascades in long CoT reasoning. To address this, on-policy distillation allows students to explore their own trajectories, but we demonstrate that it inherently introduces a reciprocal reversed exposure bias: the teacher model also struggles to provide positive guidance when conditioned on student-generated sub-optimal contexts. To resolve this dual exposure biases problem, we propose Monitoring Trajectories and Backtracking when it strays (MOTAB), a new LLM reasoning distillation pipeline. Specifically, MOTAB dynamically monitors the student's on-policy generation against an adaptive safety boundary. When the generation strays and exceeds this threshold, MOTAB backtracks to the last safe state and leverages teacher intervention to correct the course. This approach inherently tolerates minor student errors to mitigate exposure bias, while preventing sub-optimal contexts to circumvent reversed exposure bias. Extensive experiments on the LIMO-v2 and AceReason datasets demonstrate that MOTAB effectively alleviates the dual exposure biases, yielding a roughly 3% average performance improvement in reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20286v1">Adaptive Probe-based Steering for Robust LLM Jailbreaking</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 19 pages, 13 figures, accepted by ICML 2026
    </div>
    <details class="paper-abstract">
      Recent work has demonstrated the potential of contrastive steering for jailbreaking Large Language Models (LLMs). However, existing methods rely on limited and inherently biased contrastive prompts and require laborious manual tuning of steering strength, limiting their robustness and effectiveness. In this paper, we leverage the idea of model extraction to guide the learned steering vectors to approximate the ideal one and propose tuning the steering strength adaptively based on contrastive activations' statistics. Experiments demonstrate that our method notably improves the effectiveness and robustness of probe-based steering, without any extra contrastive prompts or laborious manual tuning. Being an attack paper, this paper focuses on revealing the breakdown of fortified LLMs, raising the average harmfulness score from 6\% to 70\%. Our code is available at https://github.com/fhdnskfbeuv/adaptiveSteering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20285v1">Introspective X Training: Feedback Conditioning Improves Scaling Across all LLM Training Stages</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      We tackle the question of how to scale more efficiently across the many, ever-growing stages of current LLM training pipelines. Our guiding intuition stems from the fact that the dynamics of later stages of the pipeline, e.g. post-training, can be used to inform earlier stages such as pre-training. To this end, we propose Introspective Training (or IXT), inspired by offline reward-conditioned reinforcement learning and applicable to any stage of training. IXT uses a thinking reward model to annotate data with natural language critique based feedback, enabling quality aware training from the earliest stages of the pipeline. Models are then trained by prefix-conditioning the data with the generated feedback -- ensuring that not all tokens are treated equally starting much earlier in training than usual. Comprehensive experiments on 7.5-12B transformer-based dense LLMs trained from scratch all the way up to 18 Trillion tokens seen show that our method: bends scaling curves resulting in up to 2.8x more compute efficiency generally; and reaches performance levels unachievable for models trained otherwise in domains such as math and code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.05431v4">Self-Filtered Distillation with LLMs-generated Trust Indicators for Reliable Patent Classification</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Organizing large-scale patent corpora according to classification schemes is a core information management task that determines the accuracy and efficiency of prior art retrieval, technology knowledge discovery, and intellectual property decision-making. Recent approaches distill natural language rationales generated by large language models (LLMs) into compact student models, yet logical errors, label mismatches, and taxonomy misalignments inherent in these rationales are indiscriminately absorbed during training, undermining classification reliability and propagating errors throughout downstream information processes. Rather than correcting such errors post-hoc, we propose Self-Filtered Distillation (SFD), which embeds quality assurance directly into the learning process by reinterpreting LLM-generated rationales as trust indicators rather than ground-truth supervision. SFD integrates three unsupervised signals into a unified trust score that dynamically modulates each training instance's contribution: Self-Consistency, which quantifies agreement among independently generated rationales; Class Entailment Alignment, which evaluates semantic coherence between a rationale and its assigned CPC class definition; and LLM Agreement Scoring, which assesses external plausibility through an independent verifier. On the USPTO-2M benchmark comprising over two million patents, SFD achieves up to 38.7\% relative improvement in Macro-F1 across four student architectures, and the strong correlation between trust scores and expert judgments ($r = 0.685$) confirms that the framework provides not only accurate predictions but also decomposable confidence semantics that enable auditable and self-documenting classification outcomes for large-scale patent knowledge organization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05701v1">Inference-Time Budget Control for LLM Search Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      LLM search agents increasingly rely on tools at inference time, but their trajectories are often constrained by hard limits on both tool calls and generated tokens. Under such dual budgets, better answers require not only stronger models, but also explicit control over which search action should receive the next budget unit and when the accumulated evidence is sufficient to commit a final answer. We study this problem in multi-hop question answering (QA) and formulate it as two-stage inference-time budget control. At search time, our controller assigns each feasible action a task-level Value-of-Information (VOI) score, defined as an operational estimate of marginal task value per unit budget under the current search state and remaining dual budget, and uses this score to choose among retrieval, decomposition, and answer commitment. After search, a selective evidence-grounded finalizer compares the trajectory answer with a refined candidate and rewrites only when the residual error appears to be a low-risk answer-form error. Across four multi-hop QA benchmarks, three LLM backbones, and four budget levels, the method yields positive aggregate gains over four audited baselines under the same hard dual-budget protocol. Ablations show that search-time budget control, especially budget-dependent penalty, provides the main performance gain, while answer-time control helps mainly when the retrieval path is already adequate. These results suggest that inference-time budget control for LLM search agents should govern both how budget is spent during search and how the final answer is committed.
    </details>
</div>
