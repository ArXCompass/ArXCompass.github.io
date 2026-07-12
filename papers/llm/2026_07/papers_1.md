# llm - 2026_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.08740v1">Workflow as Knowledge: Semantic Persistence for LLM-Mediated Workflows</a></div>
    <div class="paper-meta">
      📅 2026-07-09
      | 💬 39 pages, 18 figures
    </div>
    <details class="paper-abstract">
      Large language model (LLM) applications increasingly use explicit workflows for tool use, retrieval, branching, checkpointing, and human approval. Existing workflow systems already address many execution concerns. This paper proposes a Lisp-inspired but language-independent conceptual model: symbolic forms, object identity, and live-image thinking are used as explanatory lenses, not implementation commitments. In this model, workflow definitions, workflow instances, inference records, context snapshots, and dependency relations are represented as persistent knowledge objects in a shared knowledge substrate. Its central semantic distinction is between derive and infer: derive is deterministic computation over available state; infer is mediated LLM judgment under declared context and executor-controlled capability policy. The result is a preliminary conceptual account of semantic persistence: workflows do not merely produce knowledge and leave traces, but can themselves be represented as inspectable, resumable, and reviewable knowledge objects, while formal transition semantics remain future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.08733v1">Super Weights in LLMs and the Failure of Selective Training</a></div>
    <div class="paper-meta">
      📅 2026-07-09
      | 💬 Accepted at the Conference on Language Modeling (COLM) 2026
    </div>
    <details class="paper-abstract">
      Recent work identified Super Weights, individual parameters whose removal degrades model performance by orders of magnitude. We show that this degradation due to pruning Super Weights does not universally apply to all LLMs. Furthermore, if these parameters are so important, Super Weight-aware training should be effective. We show the opposite. Training Super Weights in isolation (100 to 8,192 parameters) drops accuracy to random-guessing levels on both OLMo-1B and OLMo-7B, and expanding to local neighborhoods of up to 36K parameters provides no improvement. The failure is specific to Super Weight coordinates: training an equal number of randomly chosen positions in the same down_proj layers instead improves over the baseline, so the collapse comes from targeting Super Weights, not from sparsity itself. Vanilla LoRA, updating every position in attention weight matrices through low-rank structure, succeeds with only 0.16% of parameters, and applying the same low-rank update to down_proj succeeds as well. A 10-seed ablation confirms that constraining LoRA updates at positions corresponding to Super Weight coordinates yields statistically indistinguishable results. These findings establish that parameter importance does not imply parameter trainability in isolation, and that effective fine-tuning relies on structured decompositions over entire layers rather than targeting individually important weights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.08731v1">Validity of LLMs as data annotators: AMALIA on authority</a></div>
    <div class="paper-meta">
      📅 2026-07-09
    </div>
    <details class="paper-abstract">
      A national language model offers a linguistic community its own instrument for measuring what its citizens say and value. Portugal's AMALIA, a publicly funded 9B-parameter model for European Portuguese, appears competitive on agreement alone: asked to code the moral foundation of authority, it agrees with trained human coders to within six F1 points of open models eight to thirteen times its size. Yet agreement is reliability, not validity. For theoretical constructs that must be inferred rather than read from surface features, the question is whether the model follows the construct's theory or reaches the right code by correlated shortcuts. We test this with the recovery gap: the loss in performance when a holistic prompt is decomposed into the codebook's atomic clauses and recombined by the theory's explicit rule. If calibration closes that gap, some portability should survive across models and languages; where it does not, the construct-model instrument is the likely locus of failure. We ask whether a calibrated English instrument transfers to AMALIA-9B and to European Portuguese. For one construct and one corpus, it does not. Decomposition recovers only about half of AMALIA's holistic performance, and error analysis suggests reliance on surface correlates, especially moral outrage near authority figures. An open multilingual LLM closes the gap on the same Portuguese corpus under the same instructions, pointing away from the corpus as the main explanation. AMALIA can still screen and pre-code at scale, but it cannot yet measure this construct well enough to stand alone. The study is a single counterexample, not a verdict on national models; it argues that sovereign-LLM benchmark batteries should test not only agreement with human coders, but the evidential route by which that agreement is warranted.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.08700v1">Do You Need a Frontier Model as a Citation Verifier? Benchmarking Rubric LLMs for Deep-Research Source Attribution</a></div>
    <div class="paper-meta">
      📅 2026-07-09
    </div>
    <details class="paper-abstract">
      Reinforcement learning increasingly relies on an LLM judge to score each rubric criterion, and that judge acts as the reward model during training. Before such a signal can be trusted, we need to know how capable the judge must be and how biased it is. We study this calibration question for citation quality in deep-research systems, where a search-grounded LLM must support each claim it writes with a cited source. Citation quality is a structured rubric task in which each attribution-citation pair is judged along two dimensions that require an LLM, source relevance and factual support. On an adversarial long-form benchmark, we score 8 off-the-shelf LLM judges from 3 model families against gold labels over 1,248 rubric decisions, all of which were human-reviewed and 378 of which were hard cases adjudicated from judge disagreements. Cheaper judges remain competitive across both dimensions, with GPT-5-mini attaining the strongest source-relevance pass-class F1 at 0.908 ($κ$=0.636), while on factual support the judges are statistically indistinguishable (overlapping confidence intervals), so no single model dominates. At comparable F1, the judges still differ substantially in pass-rate drift, false positive rate, and false negative rate. Scalar F1 obscures this directional bias, yet it is exactly what a downstream reinforcement learning loop would reinforce. Calibrating the judge is therefore a prerequisite for using citation rubrics as reward signals, and our results show that this calibration does not require the most expensive available model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.08643v1">BiSCo-LLM: Lookup-Free Binary Spherical Coding for Extreme Low-Bit Large Language Model Compression</a></div>
    <div class="paper-meta">
      📅 2026-07-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly constrained by memory capacity, weight bandwidth, and checkpoint storage during deployment. Existing low-bit compression methods mainly follow two directions. Scalar or group-wise quantization is simple and compatible with efficient low-precision kernels, but its representation capacity becomes limited when the target budget approaches 2 bits per weight. Vector-quantized weight compression provides a richer block-level representation, but usually introduces explicit codebooks, index lookup, and additional storage accounting. This paper presents BiSCo-LLM, a codebook-free binary spherical coding framework for extreme low-bit LLM weight compression. The core pipeline is built on three components. First, local weight chunks are mapped onto a unit hypersphere and binarized into compact spherical codes, so that the main payload is a bit-packed sign stream rather than explicit VQ centroids. Second, a residual BSQ stage encodes the reconstruction error left by the base spherical codec, providing an explicit rate-distortion path without stored codebooks. Third, category-wise recovery distillation is performed after replacing each Transformer module category, reducing the mismatch between local weight reconstruction and assembled model behavior. A small 8-bit protected-channel path is used as an auxiliary stabilization mechanism for sensitive channels and is counted separately from the BSQ payload. The reported storage budget includes binary codes, neural decoders, protected-channel payloads, LoRA adapters, and metadata.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.08565v1">SMetric: Rethink LLM Scheduling for Serving Agents with Balanced Session-centric Scheduling</a></div>
    <div class="paper-meta">
      📅 2026-07-09
    </div>
    <details class="paper-abstract">
      LLM scheduling is critical to serving, yet it remains unclear how well existing designs fit agentic serving--with LLM requests issued by agents instead of humans. This shifts the workload in two ways: (1) agents act only on complete responses, making the cluster's tokens per second (TPS) the primary goal and relaxing--not eliminating--per-token latency requirements; and (2) requests share much of their KV\$-reuse exceeds 80% of request tokens in a production trace from BAILIAN, versus 54-62% in chat. This paper first contributes a systematic study of request scheduling for agents on two real-world traces. We find that to increase KV\$ reuse, existing schedulers overly prioritize routing requests to instances caching their KV\$, overloading a few while leaving the rest idle, capping TPS. We thus present two key insights: (1) load balance need not sacrifice all KV\$ reuse, thanks to the global-tier KV\$ store and (2) by utilizing the workload's intra-session locality, balancing a small fraction of requests--the first request in each agent session--suffices to balance the cluster without sacrificing most KV\$ reuse on local instances. SMETRIC realizes these insights with balanced session-centric scheduling: it routes each session's first request purely for load balance and its follow-up requests in a cache-aware manner, preserving load balance and local reuse while keeping demand on the global tier low. Using the session turn information as the scheduling metric is deliberate: it is derived efficiently and accurately from the user inputs alone, so the scheduler stays clean and stateless. SMETRIC improves cluster TPS by 10-16% under prefill-decode colocation with a global store and prefill TPS by 2-34% under disaggregation over state-of-the-art schedulers, also with a better per-token latency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.08535v1">When the Judge Changes, So Does the Measurement: Auditing LLM-as-Judge Reliability</a></div>
    <div class="paper-meta">
      📅 2026-07-09
      | 💬 6 pages, 6 figures, 4 tables
    </div>
    <details class="paper-abstract">
      An LLM-as-judge score can move even when the candidate responses stay fixed, simply because the evaluator has changed. We treat this evaluator-replacement ambiguity as a measurement-validity problem. Across four judgment datasets, we compare two upgrade paths available in practice: scaling Qwen3 dense judges from 1.7B to 32B parameters and moving across MiniMax M2-M2.7 released APIs. The main pattern is that judge upgrades are not interchangeable: only Qwen3 1.7B to 4B gives a robust adjacent gain, while MiniMax adjacent releases do not. Stronger judges reduce but do not remove position and verbosity bias. Repeated-sample juries add little when errors are correlated. Structured debate can move decisions substantially, but without parser and fallback logs those shifts cannot be attributed to deliberation. We argue that LLM-as-judge reports should include dataset slices, bias probes, error-dependence estimates, and protocol audit trails.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.12497v2">Demystifying LLM Supply Chain Vulnerabilities in the Wild: Distribution, Root Cause, and Real-World Impact</a></div>
    <div class="paper-meta">
      📅 2026-07-09
      | 💬 Accepted by Internetware 2026
    </div>
    <details class="paper-abstract">
      LLMs are rapidly transitioning from research prototypes to core components in production systems across industries such as finance and healthcare. These deployments rely on a growing ecosystem of open-source frameworks and components, collectively forming the LLM supply chain. However, the increasing complexity of this stack introduces critical security risks that remain underexplored. In this work, we present the first systematic and large-scale empirical study of vulnerabilities in the LLM supply chain, analyzing 529 real-world vulnerabilities spanning 77 widely adopted repositories across 12 lifecycle stages. Our findings reveal that the disclosed vulnerabilities are heavily concentrated in the application layer and model integration layer. Among these, 18.5% of the vulnerabilities are LLM-specific, arising from unique architectural and workflow characteristics, such as improper handling of critical resources like model files, prompt templates, and datasets, as well as generative output validation errors. To understand the real-world impact, we examine 63,243 publicly exposed LLM services and find that 45.6% are affected by at least one remotely exploitable vulnerability, over 70% of which are critical or high severity. By correlating these vulnerabilities with their potential exploit scenarios in the wild, we observed that these issues can lead to serious security consequences, including model tampering, sensitive dataset exposure, and unauthorized GPU resource abuse. Based on our findings, we distill 5 actionable insights that can guide engineering teams in auditing and securing LLM services. Our work offers a data-driven foundation for securing the LLM supply chain and highlights urgent directions for both industry and future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.29031v2">How to Leverage Synthetic Speech for LLM-Based ASR Systems?</a></div>
    <div class="paper-meta">
      📅 2026-07-09
      | 💬 Submitted to SLT 2026
    </div>
    <details class="paper-abstract">
      In regulated domains such as banking and healthcare, where privacy constraints make real speech costly to collect and retain, synthetic speech from modern text-to-speech (TTS) is an appealing alternative for training automatic speech recognition (ASR) without exposing sensitive customer recordings. Yet a persistent distributional gap between synthetic and real data limits how far it can replace genuine recordings. Prior work largely treats this gap as a black box to be engineered around, but in our work, we instead examine its origin directly by probing a SLAM-ASR architecture. Then, we localise where its LLM backbone separates real from synthetic speech and find the discriminative signal concentrated in the early-to-middle layers, where temporal and prosodic perturbations disrupt it most. We further show that representation-level separability, help, but does not directly predict downstream ASR gains. On the other hand, convolving synthetic audio with room impulse responses (RIRs) narrows the gap not by making synthetic speech sound cleaner or more natural, but by reproducing the acoustic irregularities of real recordings. Translating these findings into the training procedure, by adding a layer-selection module combined with RIR augmentation matches a fully real-data baseline using only 25% of the real speech (13.6h) and surpasses it at all higher proportions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.08400v1">TRACE: A Two-Channel Robust Attribution Watermark via Complementary Embeddings for LLM-Agent Trajectories</a></div>
    <div class="paper-meta">
      📅 2026-07-09
    </div>
    <details class="paper-abstract">
      LLM agents reach users through resellers, who may rebrand a developer's agent or substitute a cheaper model. When provenance is disputed, attribution rests on the trajectory log (the record of tool calls, observations, and executed actions, not the model's reasoning), which the reseller stores and processes to meter usage. A watermark must therefore survive an adversary with full read/write access to the very evidence it is detected from; existing agent watermarks do not, as their attribution is read straight off that log. We present TRACE, to our knowledge the first agent watermark that is distortion-free in its action choices, self-synchronizing under deletion, and unconditionally invariant under rewriting. Deletion desynchronizes a position-derived key and rewriting alters content, so a deletion-robust key must come from content and a rewrite-robust key from position, and no single key serves both. A trajectory, however, has room for two watermarks. TRACE superposes a selection channel that sets which action is chosen, keyed on local content with a distortion-free sampler, so the agent's distribution is provably unchanged and detection resynchronizes after deletions, and a tally channel that sets how many records each decision group holds, keyed on the log's skeleton alone, which no rewriting can touch. We prove this behavioral watermark's signal is bought with decision entropy, each decision paying at least half its entropy and deterministic decisions nothing, and that erasing both channels forces the reseller to corrupt the trajectories it resells. On ToolBench and ALFWorld, TRACE matches the unwatermarked agent's success rate while its selection channel reaches detection scores near z = 100 on long-horizon trajectories, stays detectable under 70% step deletion, and keeps a tally channel exactly unchanged under LLM rewriting of any strength.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06127v2">Measuring the practice of shared-decision making (OPTION12): An Investigation into Open-sourced Smaller LLMs (OS-sLLMs) for Better Privacy and Sustainability</a></div>
    <div class="paper-meta">
      📅 2026-07-09
      | 💬 Pilot study. Preliminary findings on open-source smaller LLMs for OPTION12 shared decision-making assessment
    </div>
    <details class="paper-abstract">
      We present LLM4SDM, the first study of open-source smaller language models (OS-sLLMs) for automated assessment of shared decision making (SDM) using the Observer OPTION12 framework. Unlike previous work that relies on large commercial models and the shorter OPTION5 instrument, our study focuses on privacy-preserving locally deployable models and Dutch melanoma consultation transcripts. Using expert-annotated clinical consultations, we evaluate three general-domain and two medical-domain OS-sLLMs during a development-phase pilot study. Results show that general-domain models outperform medical-domain models, which exhibit substantial hallucination and instruction-following failures. Gemma3:12b achieves the strongest agreement with human annotations (Pearson r=0.51, Spearman \r{ho}=0.59). Item-level and qualitative analyses reveal systematic challenges related to temporal discourse reasoning, conversational role attribution, and evidence grounding. We further introduce a Judge-LLM consensus framework designed to support disagreement resolution among multiple models. Our findings suggest that while current OS-sLLMs cannot replace human annotators, they offer a promising foundation for privacy-preserving human-in-the-loop SDM assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06008v2">PolyWorkBench: Benchmarking Multilingual Long-Horizon LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-09
      | 💬 15 Pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have shown strong performance in long-horizon tasks that require planning, tool use, and interaction with external environments. However, most existing benchmarks implicitly assume a monolingual setting, where the entire execution process, including reasoning, tool invocation, and output generation, is conducted within a single language. In contrast, real-world applications often involve multilingual inputs and outputs within a unified workflow, yet the interaction between multilinguality and agentic execution remains underexplored. In this work, we introduce PolyWorkBench, a benchmark for evaluating LLM agents on multilingual long-horizon workplace workflows. PolyWorkBench consists of 67 tasks across five domains, including commerce, knowledge work, legal analysis, localization, and manufacturing, where agents must process heterogeneous multilingual inputs, perform iterative reasoning, invoke external tools, and produce structured outputs. To enable comprehensive evaluation, we propose a hybrid framework that combines structural grading, executable verification, and LLM-based semantic assessment. This design allows us to capture both functional correctness and linguistic consistency across complex workflows. Empirical results show that state-of-the-art LLM agents suffer significant performance degradation in multilingual workflow settings compared to monolingual counterparts. Our analysis suggests that multilinguality introduces compounding effects across reasoning and execution steps, highlighting the importance of jointly modeling language variation and procedural decision-making in agent evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.08326v1">Diagnosing and Repairing Persona Collapse in LLM Advice</a></div>
    <div class="paper-meta">
      📅 2026-07-09
      | 💬 Working draft
    </div>
    <details class="paper-abstract">
      LLMs are increasingly used for personal advice on relationships, work, moral dilemmas, and crises. Post-training selects a stable, prosocial Assistant persona, but good advice requires more than a good default character: a skilled advisor comforts someone in crisis, challenges someone in denial, and stays procedural with a logistical question. We formalize advice-giving as situation-conditioned persona selection in a space defined by hedonic tone and agency support, and call failures of this mapping "persona collapse" (the compression of diverse situations into a single default persona). Across 1,281 advice posts spanning 14 contexts, top-rated human responses shift systematically across five personas, while three frontier models collapse over 90\% of responses into a single supportive persona regardless of context. Prompting the model to first pick a fitting persona only deepens the collapse. We then ask whether the collapse can be repaired. Our method, Inverse-Process Distillation, reconstructs the situational reading that could have produced each human response and trains on the result, aiming to distill the situation-to-persona policy rather than the answers. It cuts divergence from the human persona distribution by approximately 80\%. Yet in a blinded study, 199 experienced advice-givers rating responses across four situations in sequence prefer the collapsed default over every repaired model, most strongly when the situation calls for challenge, though this preference shifts with repeated exposures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22588v2">Rethinking LLM-as-a-Judge: Representation-as-a-Judge with Small Language Models via Semantic Capacity Asymmetry</a></div>
    <div class="paper-meta">
      📅 2026-07-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are widely used as reference-free evaluators via prompting, but this "LLM-as-a-Judge" paradigm is costly, opaque, and sensitive to prompt design. In this work, we investigate whether smaller models can serve as efficient evaluators by leveraging internal representations instead of surface generation. We uncover a consistent empirical pattern: small LMs, despite with weak generative ability, encode rich evaluative signals in their hidden states. This motivates us to propose the Semantic Capacity Asymmetry Hypothesis: evaluation requires significantly less semantic capacity than generation and can be grounded in intermediate representations, suggesting that evaluation does not necessarily need to rely on large-scale generative models but can instead leverage latent features from smaller ones. Our findings motivate a paradigm shift from LLM-as-a-Judge to Representation-as-a-Judge, a decoding-free evaluation strategy that probes internal model structure rather than relying on prompted output. We instantiate this paradigm through INSPECTOR, a probing-based framework that predicts aspect-level evaluation scores from small model representations. Experiments on reasoning benchmarks (GSM8K, MATH, GPQA) show that INSPECTOR substantially outperforms prompting-based small LMs and closely approximates full LLM judges, while offering a more efficient, reliable, and interpretable alternative for scalable evaluation. The code and data are available at: https://github.com/zhuochunli/Representation-as-a-judge
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.08231v1">Simulating the Resident: Generating Executable Smart Home Schedules via LLM Personas</a></div>
    <div class="paper-meta">
      📅 2026-07-09
      | 💬 Published in the Proc. 1st Symposium on Artificial Intelligence throughout the Human-Centered Design Process (https://dl.gi.de/handle/20.500.12116/48536). Winner of the Best Paper Award
    </div>
    <details class="paper-abstract">
      Smart homes have emerged as an important domain for HCI research, including work on usable security and privacy. Ideally, studies in these areas draw on datasets collected in real homes with real residents, capturing authentic device interactions, network traffic, and daily routines. However, creating such datasets is slow, expensive, and raises significant privacy concerns, as it requires long-term observation of people in their most private spaces. We propose using LLMs to generate diverse resident personas that interact with a simulated smart home, producing behaviorally grounded interaction schedules that can be executed on physical testbeds. We present (1) a design framework configuring simulated households across five socio-technical dimensions, (2) a multi-stage LLM pipeline that produces structured, executable device interaction schedules, and (3) a proof of concept demonstrating feasibility. As a work in progress, we aim to support scalable, privacy-conscious smart-home experimentation without relying on intrusive real-world data collection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.08221v1">LUMI: Tokenizer-Agnostic LLM-Based Lossless Image Compression</a></div>
    <div class="paper-meta">
      📅 2026-07-09
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based lossless image compression methods typically represent pixel data through the native text interface of a pretrained model, converting pixel values into token sequences that the LLM processes through its vocabulary head. This design shows that pretrained language models can provide probability estimates for image coding, but it also couples compression to tokenizer behavior, vocabulary-specific numeric tokens, and model-family-specific adaptation. In this paper, we present LUMI (LLM-based Unified Model-agnostic lossless Image compression), a tokenizer-agnostic framework for lossless RGB image compression with frozen LLM backbones. LUMI replaces pixel-as-text tokenization with a pixel embedding module that maps raw intensity and channel information into the continuous embedding space of the LLM. It further introduces intra-patch position encoding to retain two-dimensional spatial structure after flattening, and uses a 256-way prediction head to produce probabilities over the native pixel alphabet. Only the pixel embedding, position encoding, soft-prefix parameters, and prediction head are trained, while the LLM backbone remains fixed. Experiments on natural, medical, and remote-sensing image benchmarks with LLaMA, Qwen, and Gemma backbones show that LUMI provides a unified interface across tokenizer families, achieves competitive compression rates, and improves cross-domain robustness over tokenizer-based LLM compression baselines. These results formulate LLM-based lossless image compression as pixel-space adaptation of frozen foundation models rather than tokenizer-specific language-symbol modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.20014v3">Hierarchical Control in Multi-Agent Games: LLM-based Planning and RL Execution</a></div>
    <div class="paper-meta">
      📅 2026-07-09
      | 💬 12 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has achieved strong performance in sequential decision-making, yet scaling to complex multi-agent environments remains challenging due to sparse rewards, large state-action spaces, and the difficulty of learning coordinated strategies. We propose a hierarchical architecture where a pretrained large language model (LLM) acts as a centralized strategic controller that selects among specialized RL skill policies for a team of agents, while RL policies handle reactive low-level execution. We evaluate this hybrid system in a competitive 2v2 King of the Hill environment against behavior tree (BT) and \emph{``Flat''} RL (end-to-end training without skill decomposition) baselines. The LLM+RL system achieves task performance statistically equivalent to hand-crafted BT (46.4\% vs 51.5\% win rate, $p=0.103$) while both significantly outperform Flat RL trained without skill decomposition. A user study ($n=15$) reveals that 60\% of participants perceive LLM+RL agents as the most human-like ($p=0.027$), citing behavioral adaptability and tactical variability. These results demonstrate that pretrained LLM reasoning can effectively orchestrate pretrained RL skills, achieving competitive multi-agent coordination and superior perceived believability without manual rule engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.05559v2">IFAR: Multi-Perspective and Multi-Level Causal Discovery with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have developed rapidly, and their reasoning capabilities have become a hot research topic. However, there is still limited exploration of abductive reasoning. The multi-perspective and multi-level of causes is one of the core challenges of abductive reasoning, which cannot be solved well by existing methods. We construct a specialized dataset named DeepAbduction, which is designed for tracing the causes of pollution and disease, addressing the lack of datasets in this field. We propose \textsc{Inverse-Forward Abductive Reasoning} (IFAR) framework for LLMs multi-perspective and multi-level abductive reasoning. IFAR is zero-shot and combines generalized backward reasoning with relation-by-relation forward verification. Experimental results show that IFAR achieves an improvement of approximately 40\% in the F1 score compared to other methods under mainstream LLMs, while maintaining a balance between recall and precision. Furthermore, IFAR enhances the performance of non-reasoning LLMs to surpass LLMs which have been trained for reasoning, and remains effective when applied to the latter. Code will be released after the acceptance of our work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.04983v2">LLM for the development of FCM</a></div>
    <div class="paper-meta">
      📅 2026-07-09
    </div>
    <details class="paper-abstract">
      This article is about the development of a fuzzy cognitive map using a local large language model. In the light of recent advances it is evident that large language models, and even local large language models are capable of extracting quantities from textual data. In other words, a local LLM like Qwen2.5-32B, or probably larger, can accept entities as prompt input and determine relevant quantitative data as the model output. In turn, this output can be utilized for the construction of a data driven fuzzy cognitive map. Hence, this implementation is achieved and then the model is thoroughly tested; Qwen2.5-32B is used and the data is extracted from hotel reviews from TripAdvisor. Furthermore, the extracted documents pass through the model unfiltered and then a fuzzy cognitive map is trained and evaluated. A case is made about Greek reviews where a star topology FCM is formed that indicates the preferences of the reviewers. Finally, external validation is performed to establish whether the fuzzy cognitive map can correlate the star rating of the review -an outcome outside the model's inference scope -with its predicted satisfaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.16456v3">PhyMAGIC: Physical Motion-Aware Generative Inference with Confidence-guided LLM</a></div>
    <div class="paper-meta">
      📅 2026-07-09
      | 💬 This work is accepted by ECCV 2026
    </div>
    <details class="paper-abstract">
      Recent advances in 3D content generation have amplified demand for dynamic models that are both visually realistic and physically consistent. However, state-of-the-art video diffusion models frequently produce implausible results such as momentum violations and object interpenetrations. Existing physics-aware approaches often rely on task-specific fine-tuning or supervised data, which limits their scalability and applicability. To address the challenge, we present PhyMAGIC, a training-free framework that generates physically consistent motion from a single image. PhyMAGIC integrates a pre-trained image-to-video diffusion model, confidence-guided reasoning via LLMs, and a differentiable physics simulator to produce 3D assets ready for downstream physical simulation without fine-tuning or manual supervision. By iteratively refining motion prompts using LLM-derived confidence scores and leveraging simulation feedback, PhyMAGIC steers generation toward physically consistent dynamics. Comprehensive experiments demonstrate that PhyMAGIC outperforms state-of-the-art video generators and physics-aware baselines, enhancing physical property inference and motion-text alignment while maintaining visual fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.08161v1">SQuaD-SQL: Efficient Text-to-SQL with Small Language Models via LLM-Guided Knowledge Distillation</a></div>
    <div class="paper-meta">
      📅 2026-07-09
      | 💬 Accepted at IEEE SMC 2026
    </div>
    <details class="paper-abstract">
      Text-to-SQL is a fundamental task in natural language processing that enables users to interact with structured databases using natural language. While large language models (LLMs) have demonstrated remarkable performance on this task, their substantial computational requirements hinder deployment in resource-constrained settings. In this paper, we introduce SQuaD-SQL (Small-Qualified and Distilled for SQL), a novel approach that empowers small language models (SLMs) to approach the performance of LLMs on the Text-to-SQL task while significantly improving efficiency through knowledge distillation and synthetic data generation. Our method comprises three key components: (1) LLM-based synthetic data generation, where structured knowledge is extracted from LLMs via carefully designed prompting strategies; (2) parameter-efficient fine-tuning, enabling full model training on a single consumer-grade GPU; and (3) domain-adaptive fine-tuning, where domain-specific synthetic data further enhances performance in targeted domains. Experiments on the WikiSQL dataset demonstrate that SQuaD-SQL achieves an execution accuracy of 86.9% on the test set, approaching the performance of LLMs while offering faster inference and lower memory usage. These results suggest that, with proper training strategies, SLMs can serve as practical and efficient alternatives for Text-to-SQL applications in resource-limited environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.08143v1">ICDAR 2026 HIPE-OCRepair Competition on LLM-Assisted OCR Post-Correction for Historical Documents</a></div>
    <div class="paper-meta">
      📅 2026-07-09
      | 💬 17 pages
    </div>
    <details class="paper-abstract">
      We present the results of HIPE-OCRepair-2026, an ICDAR competition on LLM-assisted OCR post-correction of historical documents. OCR post-correction remains a long-standing challenge in digital heritage: large-scale collections of digitized documents are affected by legacy OCR errors, while re-digitization at scale remains impractical. Large language models (LLMs) offers a major opportunity to revisit this challenge, yet their effectiveness across languages, document types, and noise conditions - and their tendency to hallucinate - remains insufficiently understood. HIPE-OCRepair-2026 pursues two objectives: (i) to evaluate the capabilities of modern OCR post-correction systems, and (ii) to provide a reproducible evaluation framework anchored in the HIPE-OCRepair-2026 dataset, a harmonized multilingual resource consolidating existing and newly curated historical datasets. Participants were tasked with correcting noisy OCR transcripts from historical newspapers and printed works in English, French, and German (17th-20th century), working at the level of coherent transcription units (paragraphs or articles) without access to source images. The evaluation adopts a retrieval-oriented rather than diplomatic scoring approach, reflecting the practical use case of search and access over digitized collections. Four teams submitted systems ranging from zero-shot prompting to continued pre-training and fine-tuning, offering insights into the merits of different adaptation strategies. Results show that modern LLM-assisted systems can significantly improve OCR quality, but performance varies across datasets, languages, and noise levels. Over-correction on low-noise inputs emerges as a recurring challenge, highlighting the importance of evaluation beyond character error reduction. The dataset, scorer, and evaluation pipeline are publicly released to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.12612v2">Self-EvolveRec: Self-Evolving Recommender Systems with LLM-based Directional Feedback</a></div>
    <div class="paper-meta">
      📅 2026-07-09
    </div>
    <details class="paper-abstract">
      Traditional methods for automating recommender system design, such as Neural Architecture Search (NAS), are often constrained by a fixed search space defined by human priors, limiting innovation to pre-defined operators. While recent LLM-driven code evolution frameworks shift fixed search space target to open-ended program spaces, they primarily rely on scalar metrics (e.g., NDCG, Hit Ratio) that fail to provide qualitative insights into model failures or directional guidance for improvement. To address this, we propose Self-EvolveRec, a novel framework that establishes a directional feedback loop by integrating a User Simulator for qualitative critiques and a Model Diagnosis Tool for quantitative internal verification. Furthermore, we introduce a Diagnosis Tool - Model Co-Evolution strategy to ensure that evaluation criteria dynamically adapt as the recommendation architecture evolves. Extensive experiments demonstrate that Self-EvolveRec significantly outperforms state-of-the-art NAS and LLM-driven code evolution baselines in both recommendation performance and user satisfaction. Our code is available at https://github.com/Sein-Kim/self_evolverec.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.08116v1">MORES: Mobile Reasoning-as-a-Service via Distributed LLM Inference-Time Scaling</a></div>
    <div class="paper-meta">
      📅 2026-07-09
    </div>
    <details class="paper-abstract">
      Inference-time scaling has emerged as an effective approach for enhancing the capabilities of Large Language Models (LLMs), addressing the growing demand for stronger reasoning without increasing model size. This novel form of LLM scaling comprises two representative approaches: explicit reasoning, which generates intermediate chain-of-thought tokens during an explicit thinking phase, and implicit reasoning, which iteratively updates hidden states in the latent space without producing explicit outputs. Despite their effectiveness, both paradigms incur substantial computational and memory overhead, raising challenges for deployment on resource-constrained edge devices. To address these issues, we propose a Mobile Reasoning-as-aService (MORES) framework that treats reasoning as a computational service accessible to edge devices over wireless networks. Focusing on implicit reasoning, we leverage its recursive structure to partition hiddenstate updates between edge devices and servers, enabling cooperative inference that allows devices to access additional cloud computation on demand. To optimize long-term performance, we formulate a joint computation and communication scheduling problem and solve it using a semantic Mixture-of-Experts (MoE)-based Deep Reinforcement Learning (DRL) algorithm to address heterogeneity in wireless conditions and task demands. The agent adaptively allocates resources by adjusting the number of recurrent steps and the transmission pruning rate, while a semantic router enables high-speed gating for real-time expert selection. Experimental results show that the proposed method achieves an approximately 18% improvement in system throughput over the baseline Soft Actor-Critic (SAC) algorithm. Our code is available at https://github.com/NICE-HKU/MORES.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05969v2">Heimdallr: Characterizing and Detecting LLM-Induced Security Risks in GitHub CI Workflows</a></div>
    <div class="paper-meta">
      📅 2026-07-09
    </div>
    <details class="paper-abstract">
      GitHub Continuous Integration (CI) workflows increasingly integrate Large Language Models (LLMs) to automate review, triage, content generation, and repository maintenance. This creates a new attack surface: externally controllable workflow inputs can shape LLM prompts and outputs, which may in turn affect security decisions, repository state, or privileged execution. Although LLM security and CI security have each been studied extensively, their intersection remains underexplored. In this paper, we present the first study of LLM-induced security risks in GitHub CI workflows. We characterize the problem along the full execution chain and develop a taxonomy of high-level risk classes and concrete threat vectors. To detect such risks in practice, we design Heimdallr, a hybrid analysis framework that normalizes workflows into an LLM-Workflow Property Graph (L-WPG) and combines triggerability analysis, LLM-assisted dataflow summarization, and deterministic propagation to synthesize concrete threat-vector findings. Evaluated on 300 manually annotated unique workflows, Heimdallr achieves high accuracy on LLM-node identification (F1~=~0.994), triggerability classification (99.8%), and threat-vector detection (micro-average F1~=~0.917). As part of an ongoing detection and disclosure effort, we have so far responsibly disclosed 802 vulnerable workflow instances across 759 repositories and received 71 acknowledgments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.08065v1">When LLMs Agree, Are They Right? Auditing Self-Consistency and Cross-Model Agreement as Confidence Signals</a></div>
    <div class="paper-meta">
      📅 2026-07-09
    </div>
    <details class="paper-abstract">
      LLM-as-judge (Zheng et al., 2023) is increasingly the default for evaluating AI systems in enterprise pipelines, often scaled to ensembles (Verga et al., 2024) or "mixture-of-experts" (Shazeer et al., 2017) panels of judges. These systems share a key assumption: that consistency -- agreement among judges, or among a model's own samples -- indicates correctness. We show this assumption is unreliable. Agreement is not accuracy: a model can agree with itself, and different models can agree with each other, out of shared bias, a memorized heuristic, or an option-position prior rather than truth. We ask when agreement is nonetheless a usable proxy, in a large-scale cross-runner study: 53 runners drew K=50 samples for assigned overlapping cases across comparisons of model tier, prompting, and scale on GPQA Diamond and AIME -- 265,000 samples. Using majority-correctness as the deployment label and a hierarchical runner-clustered bootstrap, agreement is a positive but weak predictor (rho 0.20-0.59, all positive under item-clustered resampling) whose usefulness is regime-dependent: best for unsaturated mid-tier models and for allocating compute, and worst -- over-confident yet no more accurate -- for the most consistent frontier model (agreement >=0.8 on 77% of GPQA case-result entries, 48% of those wrong). An exploratory cross-family check on three Claude tiers shows the same frontier over-confidence, with confident errors recurring across providers above a marginal-preserving null. Self-consistency is thus a conditional proxy for correctness, not a standalone confidence score. We publicly release the de-identified per-run rows and answer distributions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22448v2">HeaPA: Difficulty-Aware Heap Sampling and On-Policy Query Augmentation for LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-07-09
      | 💬 COLM 2026
    </div>
    <details class="paper-abstract">
      RLVR has become a standard recipe for training LLMs on reasoning tasks with verifiable outcomes, but when rollout generation dominates the cost, efficiency hinges on which prompts are sampled and when. In practice, prompt pools are often static or only weakly coupled to policy progress, so uniform sampling fails to track the moving capability frontier and wastes rollouts on regions that are already solved or still unreachable. Prior methods improve efficiency via filtering, curricula, adaptive rollout allocation, or teacher guidance, but they often assume a fixed pool, which does not support stable on-policy pool growth, or they introduce additional teacher cost and latency. In this work, we propose HeaPA (Heap Sampling and On-Policy Query Augmentation), which maintains a bounded, evolving pool, tracks the frontier with heap-based boundary sampling, grows the pool via on-policy augmentation under lightweight asynchronous validation, and stabilizes correlated queries via topology-aware pool statistics re-estimation and controlled reinsertion. Across two training corpora, two training recipes, and seven benchmarks, HeaPA consistently improves accuracy and reaches target performance with fewer computations at comparable wall-clock time. Analyses attribute the gains to frontier-focused sampling and on-policy pool growth, with more pronounced improvements at mid-to-large model scales. Our training code is publicly available at https://github.com/horizon-llm/HeaPA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.08054v1">Who Analyses the Analyser? Self-Validating LLM Hazard Analysis with Constitutional Meta-STPA</a></div>
    <div class="paper-meta">
      📅 2026-07-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly trusted to draft the artifacts of safety analysis such as, losses, hazards, Unsafe Control Actions (UCAs), and safety constraints, inside rigorous processes such as Systems-Theoretic Process Analysis (STPA). Yet a blind spot runs through this fast-growing literature: every system gets analysed except the LLM-assisted tool doing the analysing, which is itself a safety-relevant system that can hallucinate standards, emit unverifiable constraints, and leave no audit trail from prompt to artifact. We take seriously the question the field has skipped -- {who analyses the analyser?} and answer it by turning STPA on the tool itself. We present \{Constitutional Meta-STPA}, an LLM-assisted STPA tool built around a closed loop: the tool runs a {meta-STPA} of the class of AI-assisted safety tools and {derives} rather than asserts, its governance constitution from the resulting loss$\to$hazard$\to$UCA$\to$constraint chain, yielding a published constitution of $21$ Tool Principles and $8$ Meta-Safety Principles, each bound to a code enforcement point. We formalise the measured object as a constitution-marginal coverage operator over a principle set $P$ ($|P|{=}29$) with a soundness lemma that isolates coverage from model and scanner, and report four findings. {(i)~Self-derivation:} a frontier ensemble ({claude-opus-4.8}${+}${claude-sonnet-4}) recovers $18/21$ canonical and all $8/8$ governance principles from the tool's own design, while a weaker pair recovers $12/21$ and $3/8$, so the meta layer is model-limited, not constitution-limited, and the same $8/8$ re-emerge from a second, independently authored tool.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.08028v1">From Prompts to Contracts: Harness Engineering for Auditable Enterprise LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-09
      | 💬 32 pages, 6 figures, 16 tables. Reference implementation and evaluation artifacts: https://github.com/hammerbaki/enterprise-llm-agent-harness (archived at https://doi.org/10.5281/zenodo.21269426)
    </div>
    <details class="paper-abstract">
      Enterprise large language model (LLM) applications often begin as prototypes whose behavior is carried by prompts and retrieval context. Productization adds requirements for source boundaries, entity routing, answer contracts, and reproducible traces. We present a harness-engineering approach that reconstructs this pattern into a traceable, auditable LLM-agent architecture: deterministic behavior moves into code, manifests, schemas, and validation artifacts around a replaceable composition boundary, while source-backed claims remain the authority for runtime answers. We instantiate it on a public-data slice of five Korean corporate groups (25 listed companies) and evaluate three research questions. (1) The harness preserves its source-grounding, entity-routing, trace, output-hygiene, and recommendation-language contracts across the fixed validation scenarios; a fault-injection control confirms the validators flag deliberately broken contracts. (2) The checks the harness enforces held under model substitution: across three hosted models, they passed on all 270 composition-boundary runs; failures were confined to the model-composed side and were caught and recorded. (3) The code-owned guarantees are load-bearing, not reproducible by prompting alone: holding the model fixed and varying only the enforcement layer, prompt instructions alone let recommendation-language and internal-trace-leakage violations reach the reader, which the harness blocks entirely. A bolt-on external guardrail prevents such violations too but over-refuses, dropping utility to 88/120 where the harness preserves full utility (120/120); in this ablation, only code-owned enforcement preserves both safety and utility. The result is a reusable engineering pattern for turning exploratory prototypes into auditable applications with versioned source, control, and validation artifacts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.08017v1">Can We Trust LLM's Logic? Quantifying Uncertainty, Coherence, and Robustness via a Graph-Based Framework</a></div>
    <div class="paper-meta">
      📅 2026-07-09
      | 💬 42 pages, 14 figures, 12 tables
    </div>
    <details class="paper-abstract">
      Large-Language Models (LLMs) can be prone to flawed and unfaithful reasoning that decoding strategies like Self-Consistency (SC) fail to detect as they evaluate only final-answer agreement while ignoring the logical validity of intermediate steps. This raises three fundamental questions: How can we reliably quantify uncertainty in LLM reasoning? Can semantic, structural, and causal awareness select more faithful reasoning compared to naïve majority voting? and How robust is reasoning topology under adversarial conditions? To address these questions, we introduce GRAPHEVAL, a graph-based reasoning framework that re-frames uncertainty quantification (UQ) as a holistic reasoning fidelity problem. We propose a novel UQ metric, Graph Reasoning Coherence Score (GRCS), that quantifies semantic-structural consensus of the reasoning space and captures pathological mode collapse and confident hallucinations. We find that GRCS is the only metric that is consistently negatively correlated with reasoning faithfulness across both more capable and smaller models. Additionally, we introduce Graph Self-Consistency (GSC), a medoid-based decoding strategy that trades nominal accuracy for reasoning fidelity, exposing the degree to which SC is inflated by unfaithful lucky guesses in smaller models, while preserving or improving accuracy in more capable ones. Finally, through adversarial medoid ablation, we demonstrate that the GSC-selected path acts as a "load-bearing path" and forcing models away from it degrades reasoning faithfulness and, in targeted cases, causes drops in accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.08010v1">Tool-Making and Self-Evolving LLM Agents in Low-Latency Systems</a></div>
    <div class="paper-meta">
      📅 2026-07-09
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Production LLM agents often waste latency and reliability by regenerating code for the same procedural steps on every request. We replace this inference-time coding loop with an agentic tool-making pipeline that compiles repeated SOP steps into validated, versioned tools before deployment. The tool-maker grounds synthesis in the live environment as it collects execution traces, observes backend schemas and values, generates candidate tools, and repairs them against labeled cases. At runtime, the production agent calls these tools directly and falls back to code generation only when needed. We deploy the approach in a Fulfillment Center alarm-triage system, where an agent diagnoses alarms against a 44-node SOP over heterogeneous metric backends. In production, tool calls reduce p50 latency by 42%. On 1,500 historical alarms, they reduce end-to-end error rate by up to 53% by suppressing run-to-run variance in repeated steps. Because tools return compact structured verdicts, they also enable a simpler direct-call architecture, reducing p50 latency by a further 62% in a controlled ablation. Versioned tools also improve auditability and expose specification gaps and upstream data drift. Our results show that self-evolving agents can make industrial LLM systems faster, more reliable, and easier to operate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.08009v1">From Execution to Education: A Bloom-Aligned Framework for Measuring Educational Control in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-09
      | 💬 24 pages, 20 figures
    </div>
    <details class="paper-abstract">
      We introduce a Bloom-aligned framework for measuring educational control in Large Language Models (LLMs): the ability to preserve a task's instructional intent while shifting its cognitive demand toward specified learning objectives. We apply this framework to programming tasks in computer science education to study the gap between solving tasks and adapting them for learners. Using revised Bloom's Taxonomy as an operational scale of cognitive demand, we evaluate two intervention settings: general difficulty control, where models are asked to make tasks harder or easier, and Bloom's control, where models are asked to target higher or lower Bloom's levels. We evaluate a matched Qwen3-Next model pair, comparing Qwen3-Next-80B-A3B-Instruct with Qwen3-Coder-Next across 2,520 tasks from three benchmarks. The framework reveals a robust directional asymmetry: both models reliably increase cognitive demand, but struggle to lower it. We further characterize these outcomes with semantic-delta clustering and layer-wise Fisher's Discriminant Ratio probing. Within this controlled comparison, the general model shows clearer middle-layer separability for both general difficulty and Bloom-control contrasts, whereas the coder model shows weaker separability for general difficulty and a deeper peak for Bloom-control contrasts. These results show that strong execution performance does not automatically entail Bloom-aligned educational control.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.07989v1">Who Broke the System? Failure Localization in LLM-Based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-07-08
      | 💬 To appear in COLM 2026
    </div>
    <details class="paper-abstract">
      Large language model (LLM) based multi-agent systems enable complex problem solving through coordinated reasoning and action, but their distributed structure also introduces new challenges in diagnosing system-level failures. When an execution fails, identifying which agent is responsible and at what point the trajectory first becomes irreversibly misdirected is difficult due to long-horizon interactions and tightly coupled agent behaviors. In this paper, we study the problem of failure localization in LLM-based multi-agent systems and present AgentLocate, a framework that attributes failures to both a specific agent and the earliest decisive step. AgentLocate combines an LLM-based judging mechanism with multi-perspective verification by independent evaluators, whose assessments are aggregated using a confidence-aware strategy. The resulting feedback is further used to adapt the judge through lightweight fine-tuning, improving attribution quality. We evaluate AgentLocate on two complementary benchmarks covering diverse tasks, agent configurations, and trajectory lengths. Experimental results show that AgentLocate consistently outperforms existing failure localization methods in identifying both responsible agents and failure steps, while remaining efficient in terms of token usage and running time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.07976v1">When Implausible Tokens Get Reinforced: Tail-Aware Credit Calibration for LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-07-08
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has achieved remarkable success in enhancing the reasoning capabilities of large language models (LLMs). However, widely used critic-free RL methods rely on uniform credit assignment, broadcasting the same advantage to all tokens regardless of their differences. We identify a critical failure mode of this design, which we refer to as Positive-Credit Contamination: low-probability tail tokens that are contextually erroneous receive identical positive credit to plausible ones within the same trajectory, resulting in the indiscriminate reinforcement of flawed reasoning behavior. To mitigate this issue, we propose Tail-Aware Credit calibratiOn (TACO), a method that calibrates uniform credit assignment to suppress undesirable positive updates. TACO first computes a tail-risk score that incorporates the local generation context to assess each token's risk of falling into the unreliable tail, distinguishing unexpected rarity from uncertainty-driven exploration. TACO then uses this score to tune positive credit for risky tokens without removing their gradients entirely, so that recurring useful rare patterns can accumulate reinforcement while incidental noise is progressively dampened. Experimental results across three LLMs and eight benchmarks show that TACO consistently outperforms GRPO-style baselines. Notably, TACO improves training stability, supporting sustained performance gains in long-horizon RL. The source code is available at: https://github.com/xiuyilou/TACO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.07964v1">KronQ: LLM Quantization via Kronecker-Factored Hessian</a></div>
    <div class="paper-meta">
      📅 2026-07-08
      | 💬 COLM 2026
    </div>
    <details class="paper-abstract">
      Post-training quantization (PTQ) is a widely adopted technique for compressing large language models (LLMs) without retraining. Existing second-order PTQ methods, including GPTQ, construct quantization objectives exclusively from input activation statistics, effectively assuming that all output channels contribute equally to the layer-wise reconstruction objective. We propose KronQ, a PTQ framework that challenges this assumption by introducing the gradient covariance into the quantization pipeline. Under the Kronecker-factored Hessian approximation, the quantization loss depends jointly on both the activation and gradient covariances, and KronQ exploits this at two complementary levels. (1) KronQ introduces bidirectional incoherence processing, extending the existing input-side random rotation to the output dimension using the gradient covariance, reducing weight magnitude variance across both input and output dimensions. (2) KronQ derives a new sensitivity metric for inter-layer mixed-precision allocation, driven by the gradient and activation Hessian traces. Notably, in the case of 2-bit weight-only quantization on LLaMA-3-70B, while GPTQ and GPTAQ diverge or produce degenerate quantizations (>2000 perplexity on WikiText-2), KronQ achieves 7.93 perplexity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.23842v5">Fair Document Valuation in LLM Summaries via Shapley Values</a></div>
    <div class="paper-meta">
      📅 2026-07-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly power search engines and AI assistants that retrieve and summarize content from many sources. By serving answers directly, these systems obscure the original content creators' contributions, threatening the compensation that sustains a healthy content ecosystem. We frame this as a problem of fair document valuation and compensation, and propose a framework based on the Shapley value. Because exact Shapley computation is prohibitively expensive at scale, we develop Cluster Shapley, an approximation that groups semantically similar documents via LLM embeddings and computes Shapley values at the cluster level, with formal bounds on both the approximation error and the induced revenue-attribution error. On Amazon product review data, off-the-shelf approximations such as Monte Carlo sampling and Kernel SHAP perform suboptimally in LLM settings, whereas Cluster Shapley substantially improves the efficiency--accuracy frontier. Simple attribution heuristics (e.g., equal or relevance-based allocation), though computationally cheap, yield highly unfair outcomes. Our approach is agnostic to the exact LLM used, the summarization process used, and the evaluation procedure, which makes it broadly applicable to a variety of summarization settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.07915v1">Validating LLMs in social science: Epistemic threats and emerging norms</a></div>
    <div class="paper-meta">
      📅 2026-07-08
      | 💬 28 pages, 2 figures. Main text: 11 pages, Appendix: 11 pages, References: 6 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are reshaping social science methodology. Researchers increasingly prompt language models to generate quantitative measurements of social concepts, for example labeling data or simulating survey responses. Yet LLMs pose methodological challenges including bias, hallucination, and brittleness across contexts, with unclear threats to validity. Standard practices and norms for addressing these challenges are still emerging. We collect and systematically analyze validation practices in a comprehensive corpus of papers from eight flagship social science journals that use LLMs as measurement instruments. We find that LLM-generated measurements frequently play a central role in empirical analyses, yet validation practices are inconsistent and limited. We outline complementary strategies for more robust validation, pointing toward better norms and standards around the use of LLMs in social science.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.07903v1">Mechanistic Interpretability of LLM Jailbreaks via Internal Attribution Graphs</a></div>
    <div class="paper-meta">
      📅 2026-07-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit remarkable capabilities but remain highly vulnerable to adversarial prompts and jailbreak attacks. Existing approaches primarily analyze these failures through input-output behaviors or attribution methods, offering limited insight into how adversarial perturbations alter the model's internal reasoning. Consequently, the mechanisms underlying unsafe or incorrect behaviors remain poorly understood. We introduce a mechanistic framework for diagnosing LLM vulnerabilities using paired internal computation graphs, which represent prompt-specific inference as structured causal interactions among latent features. By constructing and aligning computation graphs for clean and attacked prompts, we reveal that adversarial attacks induce systematic transformations of internal reasoning, including suppression of safety-relevant components, emergence of attack-specific features, and rerouting of computation paths. Building on this representation, we propose a unified framework that (i) decomposes computation into invariant, suppressed, and emergent structures, (ii) identifies recurring vulnerability motifs associated with failure modes, and (iii) performs causal interventions on nodes, paths, and subgraphs to directly evaluate their contributions to attack success. This enables a transition from descriptive attribution to causal diagnosis of model failures. Experiments across multiple open-source LLMs and diverse adversarial and jailbreak benchmarks demonstrate that structural deviations in internal computation graphs strongly correlate with unsafe behaviors. Furthermore, targeted interventions on identified vulnerability motifs improve model robustness, establishing internal computation graphs as a principled foundation for understanding, diagnosing, and mitigating LLM vulnerabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.07895v1">Scalable and Culturally Specific Stereotype Dataset Construction via Human-LLM Collaboration</a></div>
    <div class="paper-meta">
      📅 2026-07-08
      | 💬 Weicheng Ma, John Guerrerio: equal contribution; published in EMNLP 2025 Main
    </div>
    <details class="paper-abstract">
      Research on stereotypes in large language models (LLMs) has largely focused on English-speaking contexts, due to the lack of datasets in other languages and the high cost of manual annotation in underrepresented cultures. To address this gap, we introduce a cost-efficient human-LLM collaborative annotation framework and apply it to construct EspanStereo, a Spanish-language stereotype dataset spanning multiple Spanish-speaking countries across Europe and Latin America. EspanStereo captures both well-documented stereotypes from prior literature and culturally specific biases absent from English-centric resources. Using LLMs to generate candidate stereotypes and in-culture annotators to validate them, we demonstrate the framework's effectiveness in identifying nuanced, region-specific biases. Our evaluation of Spanish-supporting LLMs using EspanStereo reveals significant variation in stereotypical behavior across countries, highlighting the need for more culturally grounded assessments. Beyond Spanish, our framework is adaptable to other languages and regions, offering a scalable path toward multilingual stereotype benchmarks. This work broadens the scope of stereotype analysis in LLMs and lays the groundwork for comprehensive cross-cultural bias evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06116v7">The Homogenization Problem in LLMs: Towards Meaningful Diversity in AI Safety</a></div>
    <div class="paper-meta">
      📅 2026-07-08
    </div>
    <details class="paper-abstract">
      Generative AI models reproduce the human biases in their training data and further amplify them through mechanisms such as mode collapse. The loss of diversity produces homogenization, which not only harms the minoritized but impoverishes everyone. We argue homogenization should be a central concern in AI safety. To meaningfully characterize homogenization in Large Language Models (LLMs), we introduce a framework that allows stakeholders to encode their context and value system. We illustrate our approach with an experiment that surfaces gender bias in an LLM (Claude 3.5 Haiku) on an open-ended story prompt. Building from queer theory, we formalize homogenization in terms of normativity. Borrowing language from feminist theory, we introduce the concept of xeno-reproduction as a class of tasks for mitigating homogenization by promoting diversity. Our work opens a collaborative line of research that seeks to understand and advance diversity in AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.02885v2">Where do LLMs Fall Short in CBT-Guided Affective Reasoning?</a></div>
    <div class="paper-meta">
      📅 2026-07-08
      | 💬 12 pages, 7 figures, accepted for publication in Affective Computing and Intelligent Interaction (ACII) 2026
    </div>
    <details class="paper-abstract">
      Cognitive Behavioral Therapy (CBT) provides a structured framework for understanding a user's mental state by examining the interaction between cognitive and behavioral factors. However, out-of-the-box LLMs respond fluently and empathetically, yet collapse into validation & reflection, regardless of what the user actually needs. They know theoretical CBT (scoring up to 96% accuracy on licensing exam questions) but fail to apply it effectively. We explore this gap with a knowledge-guided framework that treats CBT dialogue as controlled affective reasoning: user narratives are decomposed into Beck's Cognitive Conceptualization structure, grounded in clinical SNOMED CT concepts validated via Natural Language Inference, and a Multiple Chain-of-Thought (MCoT) strategy selection between Validation & Reflection, Socratic Questioning, or Alternative Perspectives. To measure whether such guidance actually changes behavior, we introduce the Protocol Leverage Force (F), a behavior-level metric that captures how far an intervention shifts a model away from its default response. Across three open-weight LLMs and 14 RealCBT-derived case studies, evaluated with human experts, valence-arousal trajectories, and linguistic entrainment, F shows that simply introducing protocol definitions via single chain-of-thought prompting fails to change LLM behavior, while MCoT on these definitions guides strategy selection better. Still, the effect stays within 1% (approx. 1.2-1.3%), and all models remain biased toward Validation & Reflection. These results show CBT knowledge alone does not ensure effective application, giving the affective-computing community instrumentation to measure where LLMs fall short.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.11950v2">AnyPoC: Universal Proof-of-Concept Test Generation for Scalable LLM-Based Bug Detection</a></div>
    <div class="paper-meta">
      📅 2026-07-08
    </div>
    <details class="paper-abstract">
      While recent LLM-based agents can identify many candidate bugs in source code, their reports remain static hypotheses that require manual validation, limiting the practicality of automated bug detection. We frame this challenge as a test generation task: given a candidate report, synthesizing an executable proof-of-concept (PoC) - such as a script, command sequence, or crafted input - to trigger the suspected defect. Automated PoC generation can act as a scalable validation oracle, enabling end-to-end autonomous bug detection by providing concrete execution evidence. However, naive LLM agents are unreliable validators: they are biased toward "success" and may reward-hack by producing plausible but non-functional PoCs or even hallucinated traces. To address this, we present ANYPoC, a general multi-agent framework that (1) analyzes and fact-checks a candidate bug report, (2) iteratively synthesizes and executes a PoC while collecting execution traces, and (3) independently re-executes and scrutinizes the PoC to mitigate hallucination and reward hacking. In addition, ANYPoC also continuously extracts and evolves a PoC knowledge base to handle heterogeneous tasks. ANYPoC operates on candidate bug reports regardless of their source and can be paired with different bug reporters. To demonstrate practicality and generality, we apply ANYPoC, together with a simple agentic bug reporter, on 12 large-scale, critical software systems, including Firefox, Chromium, LLVM, OpenSSL, SQLite, FFmpeg, and Redis. Compared to the state-of-the-art coding agents, e.g., Claude Code and Codex, ANYPoC produces 37% more valid PoCs for true-positive bug reports and rejects 9.7x more false-positive bug reports. ANYPoC also enables the discovery of 121 new bugs from over two thousand noisy bug reports, with 108 confirmed by developers and 92 fixed. 46 PoCs have also been adopted as official regression tests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21433v2">Framing Instability in LLM Ethical Stance: Auditing Negation Sensitivity in Moral Dilemmas</a></div>
    <div class="paper-meta">
      📅 2026-07-08
      | 💬 16 pages, 5 figures (added gold labeled human datasets and additional statistical tests)
    </div>
    <details class="paper-abstract">
      Language models are increasingly consulted on ethically consequential questions, yet the stance a model expresses may not survive a change in framing. We audit 16 models across 14 ethically fraught dilemmas using polarity-paired proposals ("They should X" / "They should not X"). A model's judgment of the underlying action should not reverse merely because the question is phrased as a prohibition rather than a prescription and yet, we find systematic deviations from this invariance including wholesale endorsement flips, indicating that ethical decisions are vulnerable to framing instability. Small open-weight models (1-4B parameters) endorse a proposed action 24% of the time under affirmative framing but up to 100% under negated framings, a swing of as much as 76 percentage points. Human coding of a response sample confirms the instability is genuine while showing that binary agree/disagree proxies over-state its magnitude, suggesting that an LLM judge cannot replace human coders because it silently collapses abstentions and mirrors the very forced-choice bias under study. Commercial models are for the most part more stable but still shift substantially, with cross-model agreement dropping from 73% on the bare affirmative framing to 59% under simple negation. We argue that because binary agree/disagree formats both inflate apparent endorsement and mask polarity-dependence, single-phrasing audits can misreport a model's ethical stance, and we propose the Negation Sensitivity Index (NSI) as a complement that measures stance stability directly. A model whose stance flips with phrasing cannot be relied upon in any high-stakes decision scenario.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.07619v1">Rethinking Code Performance Benchmarks for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-08
    </div>
    <details class="paper-abstract">
      Many function-level performance benchmarks have been proposed to evaluate whether large language models (LLMs) can generate efficient programs. However, results on these benchmarks often show that LLM-generated implementations have little or no execution-time difference from canonical solutions. In this paper, we revisit four popular benchmarks: EffiBench, Enamel, EvalPerf, and Mercury. We evaluate 1,538 tasks under more rigorous setting by running each task 30 times and assessing the runtime differences between the canonical solutions and benchmark-provided performant implementations with statistical testing. With the benchmark-provided test suites, only 6.11% of the performant implementations are significantly faster than the canonical solutions. In a manual analysis of 308 non-significant tasks, 99 performant implementations contain no meaningful performance change, while 209 contain potential performance improvements that are not exposed by the original tests. These results suggest that the main limitation is not only the evaluation method, but also the limited sufficiency of the benchmark-provided performance tests. To address this limitation, we propose an LLM-based multi-agent framework to generate performance-oriented tests that expose runtime differences more effectively than the original tests. The framework uses three separate agents to generate, diagnose, and repair deterministic tests that preserve functional correctness while better exposing performance differences. Across 1,345 benchmark tasks for which the original tests found no significant performance difference, tests generated by our framework with DeepSeek-v3.1 and GPT-4o reveal statistically significant improvements in 24.01% and 25.43% of the tasks, respectively, outperforming the SOTA LLM-based performance test generation method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.12857v2">Adaptive Generation of Bias-Eliciting Questions for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-08
      | 💬 ICML 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are now widely deployed in user-facing applications, reaching hundreds of millions of users worldwide. Despite their widespread adoption, growing reliance on their outputs raises significant concerns, particularly as users may be exposed to model-inherent biases that disadvantage or stereotype certain groups. However, existing bias benchmarks commonly rely on simple templated prompts or restrictive multiple-choice questions that fail to capture the complexity of real-world user interactions. In this work, we address this gap by introducing a counterfactual framework that automatically generates realistic, open-ended questions for LLM bias evaluation. Through iterative question mutation, our approach systematically explores areas where models are most likely to exhibit biased behavior. Beyond just detecting harmful biases, we also capture increasingly relevant response dimensions, such as asymmetric refusals and explicit bias acknowledgment. Building on this, we construct CAB, a diverse and human-verified benchmark for realistic and nuanced bias evaluations on current frontier LLMs. Our evaluation using CAB highlights the continued need for fairness research by showing that all examined models exhibit persistent biases across certain scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.17476v2">Zoom In Disparities in Healthcare LLM Q&A</a></div>
    <div class="paper-meta">
      📅 2026-07-08
      | 💬 It is accepted to NLDB 2026: The paper can be accessed at https://link.springer.com/chapter/10.1007/978-3-032-29532-3_12
    </div>
    <details class="paper-abstract">
      Equitable access to reliable health information is vital when integrating AI into healthcare. Yet, information quality varies across languages, raising concerns about the reliability and consistency of multilingual Large Language Models (LLMs). We systematically examine cross-lingual disparities in pre-training source and factuality alignment in LLM answers for multilingual healthcare Q&A across English, German, Turkish, Chinese (Mandarin), and Italian. We (i) constructed Multilingual Wiki Health Care (MultiWikiHealthCare), a multilingual dataset from Wikipedia; (ii) analyzed cross-lingual healthcare coverage; (iii) assessed LLM response alignment with these references; and (iv) conducted a case study on factual alignment through the use of contextual information and Retrieval-Augmented Generation (RAG). Our findings reveal substantial cross-lingual disparities in both Wikipedia coverage and LLM factual alignment. Across LLMs, responses align more with English Wikipedia, even when the prompts are non-English. Providing contextual excerpts from non-English Wikipedia at inference time effectively shifts factual alignment toward culturally relevant knowledge. These results highlight practical pathways for building more equitable, multilingual AI systems for healthcare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.08068v2">DICE: Entropy-Regularized Equilibrium Selection for Stable Multi-Agent LLM Coordination</a></div>
    <div class="paper-meta">
      📅 2026-07-08
    </div>
    <details class="paper-abstract">
      Multi-agent large language model (LLM) systems often fail to reliably outperform a single strong model equipped with best-of-N sampling. We argue that a core source of this instability is ill-posed equilibrium selection: current systems specify what information agents share, but not which coordination convention should be selected. We formalize a broad class of such systems as discounted incomplete-information Markov games and show that two common pathologies, oscillation between competing conventions and drift across them, can both induce unstable learning and linear Bayesian regret. To obtain a well-posed target, we introduce the Heterogeneous Quantal Response Equilibrium (HQRE), an entropy-regularized equilibrium concept with agent- and state-dependent temperatures. Under a monotonicity condition, HQRE is unique, admits linearly convergent mirror updates, and yields bounded Bayesian regret; the same condition yields rollout-measurable stability diagnostics. We instantiate this objective in two algorithms: DICE-PC, which coordinates frozen models through prompt-control actions, and DICE-FT, which performs parameter-efficient mirror fine-tuning. Across eleven benchmarks in four domains, DICE improves accuracy-cost trade-offs over strong within-class baselines; on reasoning and planning tasks, DICE-PC improves by 4.3 percentage points on average and DICE-FT by 8.5 points.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.07761v1">Aligning Clinical Needs and AI Capabilities: A Survey on LLMs for Medical Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-07-08
      | 💬 Accepted by Machine Intelligence Research
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have emerged as important tools in healthcare, showing growing potential for clinical reasoning and patient care. This survey examines recent progress in medical LLMs, focusing on reasoning applications and requirements. We present a dual-view approach that connects clinical practice with computational methods. On the clinical side, we establish a five-level competency scheme following Miller's Pyramid, progressing from knowledge recall to dynamic case management. On the computational side, we link deductive, inductive, and abductive reasoning patterns to common medical goals and tasks. We also introduce a benchmark dataset spanning five levels of medical reasoning capability and report results on 18 state-of-the-art models, revealing that medical specialist models excel in diagnosis-centric tasks while general models lead in decision support and dialogue. We conclude by discussing current progress and open challenges, including data limitations, hallucination, and grounding issues, and outline directions toward safer, more reliable, and workflow-ready systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.03378v2">ARGUS: Defending LLM Agents Against Context-Aware Prompt Injection</a></div>
    <div class="paper-meta">
      📅 2026-07-08
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents are increasingly deployed as task-oriented software systems that use runtime context to decide and act on behalf of users. This delegation model makes prompt injection especially dangerous: an attacker can hide a context-aware instruction inside evidence the agent must use to decide what to do. Existing benchmarks and defenses largely miss this setting. Benchmarks often use context-insensitive tasks where the user prompt already specifies the intended action, together with generic attack payloads independent of context. Existing defenses also do not capture the causal support from runtime evidence to concrete actions, which makes them incomplete and ineffective for context-dependent tasks. We present AgentLure, a benchmark for context-dependent tasks under context-aware prompt injection. AgentLure spans four agentic domains and eight attack vectors across six attack surfaces. To defend this setting, we propose ARGUS, a causal-provenance auditor for LLM agents. Instead of relying only on tool authorization or suspicious-context detection, ARGUS verifies whether each proposed action has a complete benign causal justification. It builds an influence-provenance graph, labels runtime spans, grounds action arguments in supporting evidence, and releases an action only when benign evidence entails it and task invariants hold. On AgentLure, ARGUS reduces attack success rate from 28.8% to 3.8% while preserving 87.5% clean utility, significantly outperforming existing defenses in the security-utility tradeoff.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.07504v1">Do LLM-Generated Skills Make Better AI Data Scientists? A Component Ablation Across Data-Science Workflows</a></div>
    <div class="paper-meta">
      📅 2026-07-08
      | 💬 KDD 2026 Workshop on AI Data Scientist
    </div>
    <details class="paper-abstract">
      Product data scientists often ask LLM-based agents to help with recurring execution tasks such as cleaning data, writing SQL, choosing statistical tests, and formatting results. Reusable skill files are meant to avoid prompting from scratch by packaging guidance for a task family. Expert-written skills can encode high-quality guidance, but writing and maintaining them across many data-science task families creates a manual bottleneck. We ask whether LLM-generated skills offer a useful low-curation alternative: do they improve performance over the task prompt alone? We test this question across four lifecycle stages: data preparation, data extraction, statistical analysis, and reporting, using one generated skill per stage. We find no reliable improvement from full generated skills over No-Skill prompting. We then ask whether any part of the skill is useful by ablating different skill components. The main ablation covers 56 tasks, nine model configurations, and three providers, yielding 7,560 runs. Compared with prompting using the task alone, neither the full generated skill nor any ablated skill variant significantly improves performance; all p-values are at least 0.396, and the total spread across variants is only 1.2 pp. A supplemental token-matched control adds 1,512 runs and finds that Full skills perform similarly to task-irrelevant skill-formatted content. The results caution against using one LLM-generated skill per data-science workflow as a default single-shot prompting strategy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21476v2">Thinking Seeds: Leveraging Historical Diversity for Position-Aware RL in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-08
    </div>
    <details class="paper-abstract">
      On-policy reinforcement learning (RL) for language model post-training suffers from a fundamental tension: as training progresses, policy entropy collapses and sampling diversity diminishes, causing the model to ``forget'' its own earlier exploratory capacity. While off-policy data can restore diversity, existing methods mix entire trajectories at the sequence level, introducing severe policy mismatch and training instability. We argue that the core question is not \emph{whether} to use off-policy data, but \emph{where} in the sequence it should appear. Based on this insight, we propose \textbf{Thinking Seeds}, a token-level mix-policy framework that uses the model's own historical checkpoints as off-policy prefixes, providing diverse starting points for reasoning, while the critical continuation is generated on-policy to preserve gradient quality. Through token-level importance ratios, Thinking Seeds effectively leverages historical diversity without compromising training stability. Extensive experiments across models and mathematical reasoning benchmarks demonstrate that Thinking Seeds consistently outperforms standard on-policy training and existing off-policy extensions. Our analysis reveals that the method maintains higher effective entropy, reduces gradient loss from clipping, and expands the explorable solution space, clarifying how position-aware mix-policy modeling improves both exploration and final performance in LLM RL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.12265v2">Fast, Slow, and Tool-augmented Thinking for LLMs: A Review</a></div>
    <div class="paper-meta">
      📅 2026-07-08
      | 💬 The article has been accepted by Frontiers of Computer Science (FCS), with the DOI: {10.1007/s11704-026-51673-0}
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable progress in reasoning across diverse domains. However, effective reasoning in real-world tasks requires adapting the reasoning strategy to the demands of the problem, ranging from fast, intuitive responses to deliberate, step-by-step reasoning and tool-augmented thinking. Drawing inspiration from cognitive psychology, we propose a novel taxonomy of LLM reasoning strategies along two knowledge boundaries: a fast/slow boundary separating intuitive from deliberative processes, and an internal/external boundary distinguishing reasoning grounded in the model's parameters from reasoning augmented by external tools. We systematically survey recent work on adaptive reasoning in LLMs and categorize methods based on key decision factors. We conclude by highlighting open challenges and future directions toward more adaptive, efficient, and reliable LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.07494v1">GIFT: Geometry-Informed Low-precision Gradient Communication for LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2026-07-08
      | 💬 12 pages, 6 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Gradient communication is a primary scaling bottleneck in large language model (LLM) pretraining. Communicating gradients in low-precision formats, such as FP8 and NVFP4, can significantly reduce the communication volume. Existing methods quantize gradients via linear or nonlinear mappings in Euclidean space, often degrading model performance because highly anisotropic gradients incur direction-dependent distortion. We present GIFT, a geometry-informed gradient scaling method that performs low-precision communication in geometry-aware coordinates. By transforming gradients into a near-isotropic space before quantization, GIFT makes low-precision representations substantially more faithful to their high-precision counterparts. GIFT only changes the coordinate system used for low-precision gradient communication and does not change the optimizer, training recipe, communication collective, or low-precision format. We also develop a simplified geometry-aware transformation algorithm with low-rank approximation and selective application to balance the computation overhead and communication reduction. We examine the empirical convergence of GIFT using Llama-300M and Llama-600M models. Our results show that GIFT reduces the end-to-end pretraining time of Llama-600M by 7.6% on 64 NVIDIA GH200 Superchips, while improving the downstream task preservation profile over direct Euclidean FP8 communication under the same optimizer and communication path.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.07481v1">The Poisoned Chalice of LLM Evaluation Report</a></div>
    <div class="paper-meta">
      📅 2026-07-08
      | 💬 Report of the competition hosted at FSE 2026
    </div>
    <details class="paper-abstract">
      Large language models are increasingly used to evaluate and support software engineering tasks, yet the validity of these evaluations is often undermined by uncertainty about whether benchmark instances were seen during pretraining. This can lead to data contamination, which may inflate performance and result in misleading conclusions about model capability. Despite this, the training corpora of many modern models are only partially disclosed, making direct decontamination infeasible. This creates a need for practical methods that can detect a large language models' prior exposure to training data without access to the full training corpus. To address this challenge, we organize the first Poisoned Chalice of LLM Evaluation Competition, co-located with the FSE-AIWare 2026 Competition Track. The competition frames contamination detection as a white-box membership inference task on source code and provides participants with curated datasets, target models, baseline attacks, and a final evaluation on a held-out model and dataset. This design encourages methods that generalize beyond superficial dataset artifacts and beyond a single training setting. This paper reports the setup and results of the competition. More broadly, the competition aims to catalyze the community around trustworthy LLM evaluation for software engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.15207v2">TeamTR: Trust-Region Fine-Tuning for Multi-Agent LLM Coordination</a></div>
    <div class="paper-meta">
      📅 2026-07-08
      | 💬 9pages, Accepted at ICML2026
    </div>
    <details class="paper-abstract">
      Multi-agent LLM systems have shown promise for complex reasoning, yet recent evaluations reveal they often underperform single-model baselines. We identify a structural failure mode in sequential fine-tuning of shared-context teams: updating one agent shifts the team's context distribution, and when subsequent updates are evaluated on cached rollouts, this mismatch compounds. We formalize this as the compounding occupancy shift and prove that stale-occupancy evaluation incurs a penalty that scales quadratically with the number of agents. In contrast, intermediate-occupancy evaluation reduces this to linear scaling. We propose TeamTR, a trust-region framework that resamples trajectories after each component update and enforces per-agent divergence control, yielding rigorous per-update and per-stage improvement lower bounds. Experiments show that TeamTR outperforms single-agent and sequential baselines with 7.1% on average, mitigates coordination regressions, and supports plug-and-play component replacement. Code is available at https://github.com/Yydc/TeamTR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.07469v1">SynthAVE: Scalable Synthetic Labeling for E-Commerce with LLM-Arena Validation</a></div>
    <div class="paper-meta">
      📅 2026-07-08
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) for e-commerce attribute extraction requires labeled data representative across thousands of product types, attributes, and multiple languages. This combinatorial scale translates to millions of annotations, rendering human labeling prohibitively costly. While recent work has demonstrated synthetic label generation using LLMs, deploying such approaches at industrial scale requires integrated quality control mechanisms. We present SynthAVE, a large-scale human-validated benchmark for attribute value extraction spanning 12,726 products across 229 product types, 792 attributes, and 4 languages (Spanish, French, Italian, German). To validate synthetic labels at scale, we introduce a multi-LLM arena framework where samples are independently evaluated by 21 judge configurations (7 model families $\times$ 3 prompts), with final labels determined via majority voting. The majority vote ensemble agrees with human experts at Cohen's $κ= 0.92$ (95.2% agreement), while individual judges show substantial inter-model agreement (Fleiss' $κ= 0.76$). This demonstrates that diverse models with varying individual judgments aggregate into highly reliable predictions, enabling cost-effective validation at scale while maintaining quality parity with human review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.07467v1">SpaCellAgent: A Self-Evolving LLM-Based Multi-Agent Framework for Trajectory Analysis</a></div>
    <div class="paper-meta">
      📅 2026-07-08
      | 💬 27 pages, 19 figures
    </div>
    <details class="paper-abstract">
      Spatial and Single-cell transcriptomics are transformative in deciphering cellular dynamics. As the fundamental paradigm for reconstructing cell developmental paths, trajectory inference (TI) is critical. However, existing methods require extensive manual intervention and proficiency in heterogeneous tools, posing a significant barrier to efficient TI analysis. To bridge this gap, we propose SpaCellAgent, an autonomous large language model (LLM) multi-agent framework that automates end-to-end spatiotemporal analysis and narrative generation. SpaCellAgent utilizes a multi-agent architecture for strategic workflow planning, a dynamic tool-orchestration engine for adaptive algorithm selection, and a self-evolution module that iteratively refines performance through feedback. We evaluate SpaCellAgent on six heterogeneous datasets encompassing complex temporal developmental trajectories, diverse sequencing platforms, and spatially-resolved tissue architectures. SpaCellAgent consistently demonstrates over 40\% improvement in analytical efficiency while maintaining expert-aligned performance. By converting natural language specifications into optimized analytical workflows and fully automating the pipeline, SpaCellAgent democratizes advanced spatiotemporal modeling and establishes a scalable, agent-driven paradigm for computational biology. The code and materials are available at https://github.com/LittleXH-shw/SpaCellAgent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14590v3">Counterfactual Modeling with Fine-Tuned LLMs for Health Intervention Design and Sensor Data Augmentation</a></div>
    <div class="paper-meta">
      📅 2026-07-08
      | 💬 IEEE Open Journal of Engineering in Medicine and Biology (Volume: 7), Date of Publication: 28 May 2026. Page(s): 232-240
    </div>
    <details class="paper-abstract">
      Counterfactual explanations (CFEs) provide human-centric interpretability by identifying the minimal, actionable changes required to alter a machine learning model's prediction. Therefore, CFs can be used as (i) interventions for abnormality prevention and (ii) augmented data for training robust models. We conduct a comprehensive evaluation of CF generation using large language models (LLMs), including GPT-4 (zero-shot and few-shot) and two open-source models-BioMistral-7B and LLaMA-3.1-8B, in both pretrained and fine-tuned configurations. Using the multimodal AI-READI clinical dataset, we assess CFs across three dimensions: intervention quality, feature diversity, and augmentation effectiveness. Fine-tuned LLMs, particularly LLaMA-3.1-8B, produce CFs with high plausibility (up to 99%), strong validity (up to 0.99), and realistic, behaviorally modifiable feature adjustments. When used for data augmentation under controlled label-scarcity settings, LLM-generated CFs substantially restore classifier performance, yielding an average 20% F1 recovery across three scarcity scenarios. Compared with optimization-based baselines such as DiCE, CFNOW, and NICE, LLMs offer a flexible, model-agnostic approach that generates more clinically actionable and semantically coherent counterfactuals. Overall, this work demonstrates the promise of LLM-driven counterfactuals for both interpretable intervention design and data-efficient model training in sensor-based digital health. Impact: SenseCF fine-tunes an LLM to generate valid, representative counterfactual explanations and supplement minority class in an imbalanced dataset for improving model training and boosting model robustness and predictive performance
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.07444v1">LLM Assisted Verification Assertion Generation: Challenges and Future Directions</a></div>
    <div class="paper-meta">
      📅 2026-07-08
      | 💬 The paper contains a series of guidelines to generate SystemVerilog assertions using LLM
    </div>
    <details class="paper-abstract">
      Assertion-based Verification (ABV) plays a critical role in the Design Verification (DV) process. However, ABV requires substantial manual effort in generating assertion from specification by verification engineers, making it a time-consuming stage in the chip design flow. With the recent development of Large Language Models (LLMs), researchers have started exploring their use as an assistance in the ABV process, particularly for generating SystemVerilog Assertions (SVAs) from design specification. In this paper, we provide an overview of recent works, highlighting the different methods used to generate SVAs. In particular, we investigate LLM-based SVA generation and ask a central question: How can LLM-based assertion generation be made systematic and quality-aware? While addressing this key question, we provide Key Takeaways at the end of each challenge, summarizing the important methodological insights, and also provide guidelines and directions in solving those challenges that can help generate a high-quality set of assertions using LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.07430v1">Immersive Social Interaction with VR and LLM-Assisted Humanoids</a></div>
    <div class="paper-meta">
      📅 2026-07-08
      | 💬 IEEE-RAS International Conference on Humanoid Robots - Workshop: Designing Interactive Humanoids
    </div>
    <details class="paper-abstract">
      Humanoid robots can extend human presence to remote, constrained, or hazardous environments, but existing teleoperation interfaces often require physically demanding motion tracking or cognitively demanding low-level control. This paper presents an immersive teleoperation framework that integrates voice-controlled locomotion, VR-based manipulation, and bidirectional social interaction for whole-body humanoid control. Using Apple Vision Pro, the operator receives egocentric visual feedback, issues natural-language locomotion commands, and teleoperates the robot's arms and dexterous hands through wrist and finger tracking. An LLM-assisted voice-control module converts spoken instructions into high-level locomotion commands, while the manipulation module retargets human hand motions to the robot through inverse kinematics and PD control. The system also records multimodal data, including egocentric RGB observations, voice/text commands, joint states, hand motions, and eye-gaze signals, supporting future imitation learning and autonomy. We evaluate the framework on a Unitree H1 humanoid equipped with dexterous hands in manipulation and social interaction tasks. Results show that novice users can successfully operate the system after brief familiarization, achieving 80\% success in object manipulation and 70\% success in a social cube-passing task. These results demonstrate the potential of immersive, language-assisted teleoperation as an accessible interface for humanoid interaction, remote assistance, and multimodal data collection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.07405v1">Reason Less, Verify More: Deterministic Gates Recover a Silent Policy-Violation Failure Mode in Tool-Using LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-08
    </div>
    <details class="paper-abstract">
      Tool-using LLM agents can violate the very policies they are deployed to enforce while appearing to complete the task successfully. In policy-permissive environments, a tool may execute any well-formed call even when the corresponding state transition is forbidden by domain policy. The result is a silent wrong state (a booking cancelled, a passenger count changed, a claim acted on without verification) that neither the tool nor the agent's self-report exposes. We study this failure mode in the $τ^2$-bench airline domain. On a budget agent, 78% of observed failures are silent wrong-state failures with no tool error, and the aggregate failure rate is reproducible across disjoint seeds, not sampling noise. We then evaluate a lightweight intervention: deterministic, read-only pre-execution gates that inspect the proposed call and current state before allowing a write. A four-gate suite raises full-benchmark success from 29.6% to 42.0% on gpt-4o-mini (+12.4pp; paired task-level bootstrap P=0.0012), and the lift reproduces on a disjoint 15-seed set (+12.3pp; P=0.0008). The effect is concentrated where the gates fire: on the 26/50 firing tasks, success rises by +19.2pp, while movement on the 24 non-firing tasks does not exclude zero. Two negative controls (a self-enforcing retail domain and BFCL) bound the mechanism: gates help when tools are policy-permissive and add little where tools already self-enforce. As suggestive evidence, not a central claim, the same failure mode persists at the frontier: gpt-5.2 at default reasoning still attempts policy-violating writes, and the same suite improves success from 61.2% to 71.6% (+10.4pp; P=0.020; n=5, no replication). The contribution is a bounded evaluation and reliability result: deterministic gates do not guarantee task success, but they can deterministically prevent a known class of silent policy-violating writes at the action boundary.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16453v3">RetailBench: Evaluating Long-Horizon Autonomous Decision-Making and Strategy Stability of LLM Agents in Realistic Retail Environments</a></div>
    <div class="paper-meta">
      📅 2026-07-08
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have made rapid progress on short-horizon, well-scoped tasks, yet their ability to sustain coherent decisions in dynamic long-horizon environments remains uncertain. We introduce RetailBench, a data-grounded simulation benchmark for evaluating tool-using LLM agents in single-store supermarket operation. RetailBench models retail management as a partially observable decision process and is designed to support thousand-day-scale simulations. In this environment, agents must manage pricing, replenishment, supplier selection, shelf assortment, inventory aging, customer feedback, external events, and cash-flow constraints. We evaluate seven contemporary LLMs under representative agent frameworks over a 180-day evaluation horizon and compare them with a privileged oracle policy. Results show substantial variation across models: only a small subset survives the full evaluation horizon, and even the strongest LLM runs remain substantially behind the oracle policy in final net worth and sales outcomes. Behavioral analysis attributes these gaps to incomplete evidence acquisition, surface-level decision making, and the lack of a consistent long-horizon policy. RetailBench provides a controlled testbed for studying reliable autonomy in economically grounded long-horizon decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.07321v1">From Atomic Actions to Standard Operating Procedures: Iterative Tool Optimization for Self-Evolving LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-08
    </div>
    <details class="paper-abstract">
      Tool utilization enables Large Language Model (LLM) agents to interact with the real world and resolve complex tasks. However, existing agent frameworks predominantly rely on static toolsets composed of granular atomic actions (e.g., basic file I/O or single-turn search), which forces agents to reinvent low-level logic for every recurring workflow, leading to increased reasoning overhead and failure rates. In this study, we propose that agents can achieve self-evolution by synthesizing these atomic actions into reusable Standard Operating Procedures (SOPs), which function as callable higher-order tools that encapsulate multi-step logic. We further introduce EvoSOP, a framework that empowers agents to extract SOPs from execution trajectories and iteratively optimize the toolset through a systematic lifecycle of construction, merging, evaluation, and pruning. Extensive experiments demonstrate that EvoSOP significantly boosts task success rates while substantially reducing the number of interaction rounds compared to baselines. Our analysis also reveals that iterative tool optimization fosters reliable and efficient tool-use patterns, providing a scalable pathway for the development of self-evolving agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.25370v2">Monitoring Transformative Technological Convergence Through LLM-Extracted Semantic Entity Triple Graphs</a></div>
    <div class="paper-meta">
      📅 2026-07-08
    </div>
    <details class="paper-abstract">
      Forecasting transformative technologies remains a critical but challenging task, particularly in fast-evolving domains such as Information and Communication Technologies (ICTs). Traditional expert-based methods struggle to keep pace with short innovation cycles and ambiguous early-stage terminology. In this work, we propose a novel, data-driven pipeline to monitor the emergence of transformative technologies by identifying patterns of technological convergence. Our approach leverages advances in Large Language Models (LLMs) to extract semantic triples from unstructured text and construct a large-scale graph of technology-related entities and relations. We introduce a new method for grouping semantically similar technology terms (noun stapling) and develop graph-based metrics to detect convergence signals. The pipeline includes multi-stage filtering, domain-specific keyword clustering, and a temporal trend analysis of topic co-occurence. We validate our methodology on two complementary datasets: 278,625 arXiv preprints (2017--2024) to capture early scientific signals, and 9,793 USPTO patent applications (2018-2024) to track downstream commercial developments. Our results demonstrate that the proposed pipeline can identify both established and emerging convergence patterns, offering a scalable and generalizable framework for technology forecasting grounded in full-text analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.07751v1">Forensic Schema for Psychological Manipulation in Cyber Fraud: LLM-Driven Victim Reports Analysis</a></div>
    <div class="paper-meta">
      📅 2026-07-08
      | 💬 Accepted by the 23rd International Conference on Privacy, Security and Trust (PST 2026)
    </div>
    <details class="paper-abstract">
      Existing cybercrime classification schemas capture contact metadata and financial transactions but omit the psychological manipulation techniques perpetrators employ. We present a forensic schema (four categories, 35 questions) adding 11 manipulation indicators and cryptocurrency evidence fields to established forensic foundations. Applied to 10,994 victim reports via large language model (LLM)-driven annotation and validated against two human annotators (mean LLM-human $κ= 0.69$, matching inter-annotator $κ= 0.68$), the schema revealed a statistically distinct manipulation profile for each major fraud type (Cramer's $V$ up to $0.790$). A rationale-based evidence audit nonetheless exposed a forensic detail gap: detection of manipulation techniques was reliable, but victim narratives varied widely in the actionable detail supporting each Yes answer, and blockchain-specific identifiers were nearly absent. These findings point to AI-assisted victim intake with schema-informed follow-up questions as the most direct way to close the gap. The tiered annotation strategy also provides a reusable template for LLM-based extraction from other forensic text domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.29251v2">When Summaries Distort Decisions: Information Fidelity in LLM-Compressed Financial Analysis</a></div>
    <div class="paper-meta">
      📅 2026-07-08
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Financial decision-makers face more information than they can directly inspect, making context compression necessary. Yet when large language models (LLMs) compress financial source material, they can alter the investment judgment supported by the original source. We frame this problem as information fidelity: compression loses fidelity when it changes the decision induced by the source. In agentic systems, such losses may recur across intermediate steps and amplify throughout the decision process. Across financial filings and earnings-call transcripts, we find that LLM-based compression can produce fluent and factually plausible compressed contexts that nevertheless alter downstream decisions. We analyze two diagnostic patterns associated with fidelity loss: decontextualization, where salient evidence is retained but separated from the caveats and contextual qualifiers needed for correct interpretation, and model dependency, where different compressors expose different views of the same source. We then propose Agentic Context Compression, which generates multiple candidate compressions and audits their disagreements against the original source. Our results suggest that financial compression should be evaluated not only by efficiency or factuality, but also by its ability to preserve decision-relevant context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.18784v5">Successor-Generator Planning with LLM-generated Heuristics</a></div>
    <div class="paper-meta">
      📅 2026-07-08
    </div>
    <details class="paper-abstract">
      Heuristics are a central component of deterministic planning, particularly in domain-independent settings where general applicability is prioritized over task-specific tuning. This work revisits that paradigm in light of recent advances in large language models (LLMs), which enable the automatic synthesis of heuristics directly from problem definitions -- bypassing the need for handcrafted domain knowledge. We present a method that employs LLMs to generate problem-specific heuristic functions from planning tasks specified through successor generators, goal tests, and initial states written in a general-purpose programming language. These heuristics are compiled and integrated into standard heuristic search algorithms, such as greedy best-first search. Our approach achieves competitive, and in many cases state-of-the-art, performance across a broad range of established planning benchmarks. Moreover, it enables the solution of problems that are difficult to express in traditional formalisms, including those with complex numeric constraints or custom transition dynamics. We provide an extensive empirical evaluation that characterizes the strengths and limitations of the approach across diverse planning settings, demonstrating its effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.11325v3">Structured Belief State and the First Precision-Aware Benchmark for LLM Memory Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-07-08
      | 💬 v3 expands systems evaluated, evidence to make the claim falsifiable and the benchmark reusable
    </div>
    <details class="paper-abstract">
      Current LLM memory benchmarks evaluate answer quality rather than retrieval accuracy. Consequently, a system that dumps its entire belief store can achieve perfect recall and mask severe precision failures. We show this evaluation gap persists across multiple embedding models where similarity-based retrieval over domain-specific corpora inherently struggles to isolate target beliefs from semantically proximate ones. Furthermore, multi-turn topic drift compounds this retrieval noise while driving up latency and operational costs. To decouple retrieval quality from generative performance, we introduce PrecisionMemBench, an 89-case benchmark measuring precision, noise isolation, session latency, and belief mutability. We also present Tenure, a structured belief-store proxy that resolves scope and retrieval before inference and injects typed belief state as ambient instruction before the model sees the prompt, removing model-side discretion over whether memory is consulted. Evaluated across 13 providers, Tenure achieves perfect retrieval passes across all active, non-session, and session test cases. In contrast, the baseline configurations fail to reach even half of the active passes, with precision scores clustering at 0.22 and below. Our results demonstrate that while current memory systems successfully store information, they fail to retrieve it cleanly; a structural vulnerability that traditional answer-quality benchmarks conceal.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.04985v2">FPTQuant: Function-Preserving Transforms for LLM Quantization</a></div>
    <div class="paper-meta">
      📅 2026-07-08
      | 💬 Forty-third International Conference on Machine Learning (ICML 2026)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) require substantial compute, and thus energy, at inference time. While quantizing weights and activations is effective at improving efficiency, naive quantization of LLMs can significantly degrade performance due to large magnitude outliers. This paper describes FPTQuant, which introduces three novel, lightweight, and expressive function-preserving transforms (FPTs) to facilitate quantization of transformers: (1) a mergeable pre-RoPE transform for queries and keys, (2) a mergeable transform for values, and (3) a cheap, dynamic per-token scaling transform. By leveraging the equivariances and independencies inherent to canonical transformer operation, we designed these FPTs to maintain the model's function while shaping the intermediate activation distributions to be more quantization friendly. FPTQuant requires no custom kernels and adds virtually no overhead during inference. The FPTs are trained both locally to reduce outliers, and end-to-end such that the outputs of the quantized and full-precision models match. FPTQuant enables static INT4 quantization with minimal overhead and shows SOTA speed-up of up to 3.9X over FP. Empirically, FPTQuant has an excellent accuracy-speed trade-off -- it is performing on par or exceeding most prior work and only shows slightly lower accuracy compared to a method that is up to 29% slower.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.07097v1">Operational Reframing and Approval-Framed Delegation in Multi-Agent LLM Safety</a></div>
    <div class="paper-meta">
      📅 2026-07-08
    </div>
    <details class="paper-abstract">
      Safety evaluations of multi-agent LLM systems often compare a direct prompt with a planner-executor pipeline and report the difference as a single "pipeline effect." We argue that this aggregate is difficult to interpret because it conflates three mechanisms: harmful intent may be reframed as plausible operational work, the planner may refuse or transform the request, and the executor may act under delegation prompts implying prior approval. To separate these factors, we introduce a five-condition controlled contrast design, evaluated on 30 synthetic harmful scenarios and an exploratory external validation set from four agent-safety benchmarks using LLM-judged compliance. Our results show that aggregate pipeline safety is not a stable architectural property. Operational reframing is the most portable risk signal, increasing compliance for GPT, Gemini, and DeepSeek across both scenario sets, while Claude is comparatively resistant. Planner behavior can offset this risk mainly through refusal; however, when the planner produces executable steps, the executor may become more compliant than under the direct operational baseline. Approval-framed delegation is sensitive to prompt design, model pairing, and scenario source, and a skeptical executor prompt sharply reduces compliance. Raw-direct model rankings can also mispredict deployed planner-executor behavior. Gemini is safest under raw direct prompts in the primary set yet shows the largest amplification with a Claude planner, rising from 8.9 percent to 38.9 percent compliance. GPTs near-zero aggregate pipeline effect instead hides a reframing increase canceled by planner refusal. These findings suggest that multi-agent safety evaluations should report reframing, planner behavior, delegation framing, and model pairing separately before attributing failures to architecture itself.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.07046v1">Voltron: Enabling Elastic Multi-Device Execution of LLM Inference for Empowered Edge Intelligence</a></div>
    <div class="paper-meta">
      📅 2026-07-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are widely used in intelligent services due to their remarkable capability in generative tasks. Typically, LLM-based services process the inference requests of the users in a centralized data center. Unfortunately, such centralized execution has limitations for end-users, such as increased response latency with communication overhead and privacy leakage risk. To alleviate the aforementioned limitations, there have been increasing pushes to execute LLM inference locally on user-end devices. However, the limited resources of a single edge device impose restrictions on achievable accuracy of LLMs. To overcome the issue, we first propose to leverage multiple user-end devices available at the edge for LLM inference, enabling the execution of larger models. Specifically, we propose Voltron, a novel on-device LLM inference framework that elastically utilizes multiple user-end devices for LLM inference execution while adapting to diverse real-world edge environments. In our evaluation, Voltron achieves up to 16.5% higher accuracy than state-of-the-art LLMs that can be executed on a single edge device, satisfying user QoS requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.07003v1">Dissociating the Internal Representations of Sycophancy in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-08
      | 💬 Accepted to Mechanistic Interpretability Workshop at ICML 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) frequently exhibit sycophancy, where they agree with a user's statement even when incorrect. While sycophancy is often treated as a single defined behavior, it can manifest in substantially distinct ways and circumstances, raising the question of whether this multi-faceted nature is reflected in its internal mechanisms. To address this gap, we dissociate the representations of sycophancy into factual and opinion subtypes -- motivated by the distinction between verifiable claims and subjective beliefs. We train linear probes and construct steering vectors on activations of one subtype and evaluate their transfer to the other subtype to measure to what extent they share representations. We find evidence that different LLMs represent these subtypes differently, with either more unified or more distinct and causally interfering representations. This method of dissociation offers a promising framework for studying the representational structure of complex model behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.07739v1">Controllability-Aware Adversarial Examples Against LLM-Based Network Traffic Classifiers</a></div>
    <div class="paper-meta">
      📅 2026-07-08
      | 💬 13 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly explored as network intrusion detection classifiers, but their adversarial robustness under realistic attacker constraints remains unclear. We present a controllability-aware black-box transfer framework for LLM-based network traffic classifiers. The framework partitions flow features into directly controllable (DC), indirectly controllable (IC), and uncontrollable (UC) groups according to network communication semantics, then restricts perturbations to DC features while freezing IC/UC features. Using a shared XGBoost surrogate, we generate finite-difference PGD, greedy coordinate-wise, and NES adversarial examples and transfer them to seven LLM targets and two conventional ML targets across five IDS benchmarks from 1999 to 2022. Across 27 valid LLM configurations and over 500,000 adversarial examples, we find that LLM transfer vulnerability is substantial but dataset- and comparator-dependent. Compared with LightGBM, LLMs are more vulnerable on RT-IoT2022 and CIC-IDS-2018, comparable on NSL-KDD and UNSW-NB15, and less vulnerable on HIKARI-2021; compared with the averaged ML baseline, LLMs show higher ASR on all five datasets. We further observe a consistent cross-architecture transfer hierarchy: gradient- and score-based perturbations transfer more effectively than greedy perturbations across all 27 LLM cells and 9/10 ML cells. Cross-surrogate validation with tree, neural, and linear surrogates yields similar LLM ASR, reducing evidence that the findings are XGBoost-specific. Constraint violation rate is 0\% by construction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06974v1">MILES: Modular Instruction Memory with Learnable Selection for Self-Improving LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-07-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly improve their reasoning at test time via additional computation, yet most existing works treat each problem in isolation. When problems arrive sequentially, accumulating reusable experience across them can further improve performance. Existing memory-based methods either store whole-solution templates that generalize poorly to novel problems or use heuristic step-level selection that is not optimized for final-answer correctness. Learning selection policies requires large-scale training data and fixed action spaces, making such approaches unsuitable for test-time settings where memory expands incrementally and only limited supervision is available. We propose MILES (Modular Instruction Memory with LEarnable Selection for self-improving LLM reasoning), a framework that dynamically expands step-wise memory and applies correctness-optimized memory composition under realistic test-time constraints. MILES maintains modular memory units consisting of asymmetric pairs of sub-goal embeddings and sub-instructions, each associated with a learnable selection head. This memory structure enables a coarse-to-fine retrieval mechanism: The coarse level enables memory expansion and collects supervision for training selection heads from confident samples, while the fine stage applies learned selection heads to rerank coarse-level candidates and guide reasoning for uncertain samples. MILES consistently matches or outperforms prior methods while achieving superior accuracy-efficiency tradeoffs. Extensive experiments demonstrate its effectiveness, robustness, and transferability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03565v4">Skill Is Not Document: A Query-Conditional Benchmark and Two-Stage Retriever for LLM Agent Skill Routing</a></div>
    <div class="paper-meta">
      📅 2026-07-08
      | 💬 20 pages, 8 figures
    </div>
    <details class="paper-abstract">
      LLM agents often solve complex tasks by composing skills, making skill retrieval a front-end component of agent systems. Unlike document retrieval, top-K correctness in skill retrieval depends not only on the relevance of each query-skill pair, but also on whether the retrieved skills can work together under the query. This query-conditioned "skill compatibility" cannot be recovered from independent relevance alone. However, LLM-based synthesis pipelines already produce a useful signal for it: the LLM's own rejection decisions, which specify which skills should not be retrieved together for a given query, but are usually discarded as low-quality data. We propose Reject-as-Resource Retriever (R3) and construct R3-Skill, a bilingual (Chinese-English) benchmark for agent skill routing. R3-Skill covers four language directions and uses LLM-rewritten queries that better approximate user requests; its test-set ground truth is verified by multiple experts. It contains 10,246 skills grouped into 8 thematic super-domains, 41,592 accepted queries, and 32,828 LLM-rejected annotations, further organized into an 8-class rejection-reason taxonomy. R3-Skill keeps this normally discarded rejection signal and uses it as compatibility supervision. On R3-Skill, we train a two-stage retriever consisting of R3-Embedding and R3-Reranker. Gradient analysis explains why this query-conditional signal is weak when injected into the tested bi-encoder objective under bilateral balancing, while a cross-encoder can use it as graded ranking supervision; R3-Skill ablations support this split. The R3-Embedding + R3-Reranker pipeline reaches Hit@1 = 0.7521, NDCG@10 = 0.8173 and Set-Compat = 0.3188 on R3-Skill. The dataset, model weights, and evaluation scripts will be open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06964v1">End-to-End LLM Flight Planning with RAG-based Memory and Multi-modal Coach Agent</a></div>
    <div class="paper-meta">
      📅 2026-07-08
      | 💬 Accepted at the ICML 2026 LM4Plan Workshop
    </div>
    <details class="paper-abstract">
      Bridging the gap between human pilot intent and autonomous flight operation is critical for real-world electric vertical takeoff and landing (eVTOL) aircraft deployment. Flight planning traditionally relies on classic algorithms that struggle to incorporate flexible human preferences. We present FRAMe, an End-to-End Large Language Model (LLM) Flight Planning tool with RAG-based Memory and Multi-modal Coach Agent. Our system integrates a planner LLM with a multi-modal coach agent and retrieval augmented generation (RAG)-based memory to generate flight plans that satisfy mission constraints while aligning with human flight operator preferences. We demonstrate the system in a range of real-world-inspired scenarios of varying difficulty levels. Across four LLMs, the full FRAMe system (RAG and coach) yields the highest validity for every planner (up to 93.8% aggregate, 99% on Easy scenarios for the strongest planner) and shifts preference-relevant metrics in the operator-favored direction where the metric has headroom. FRAMe signifies how advanced LLMs can be deployed for human-centric mission planning, translating natural language instructions into safe, efficient, and flexible flight routes. The code is available at: github.com/amin-tabrizian/FlightPlanningLLMs
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06963v1">Large Language Models (LLMs) and Generative AI in Cybersecurity and Privacy: A Survey of Dual-Use Risks, AI-Generated Malware, Explainability, and Defensive Strategies</a></div>
    <div class="paper-meta">
      📅 2026-07-08
      | 💬 Invited survey paper. 10 pages, 5 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) and generative AI (GenAI) systems, such as ChatGPT, Claude, Gemini, LLaMA, Copilot, Stable Diffusion by OpenAI, Anthropic, Google, Meta, Microsoft, Stability AI, respectively, are revolutionizing cybersecurity, enabling both automated defense and sophisticated attacks. These technologies power real-time threat detection, phishing defense, secure code generation, and vulnerability exploitation at unprecedented scales. Following a rapid surge where LLM-generated malware grew to account for an estimated 50% of detected threats by 2025, up from just 2% in 2021, navigating this highly automated threat landscape in 2026 demands next-generation security frameworks. This paper presents a comprehensive survey of the beneficial and malicious applications of LLMs in cybersecurity, including zero-day detection, DevSecOps, federated learning, synthetic content analysis, and explainable AI (XAI). Drawing on a review of over 70 academic papers, industry reports, and technical documents, this work synthesizes insights from real-world case studies across platforms like Google Play Protect, Microsoft Defender, Amazon Web Services (AWS), Apple App Store, OpenAI Plugin Stores, Hugging Face Spaces, and GitHub, alongside emerging initiatives like the SAFE Framework and AI-driven anomaly detection. We conclude with practical recommendations for responsible and transparent LLM deployment and trustworthy AI, including model watermarking, adversarial defense, and cross-industry collaboration, setting a new benchmark for rigorous, holistic cybersecurity research at the intersection of AI and threat defense, and offering a roadmap for secure, scalable LLM systems that serves as a critical reference for researchers, engineers, and security leaders navigating the complex challenges of AI-driven cybersecurity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05876v2">Think Before You Grid-Search: Floor-First Triage for LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-07-08
      | 💬 16 pages, 3 figures
    </div>
    <details class="paper-abstract">
      LLM serving optimization typically benchmarks many configurations and reaches for heavy profilers when latency targets are missed. We argue for the reverse discipline: estimation is the analytical layer of profiling -- without it, optimization degenerates to grid search. Floor First is a residual-driven triage workflow. Each decode step is modeled as a five-dimensional resource vector (HBM bytes, FLOPs, network bytes, network messages, KV capacity); summing within a resource and maximizing across resources gives an optimistic floor, the plain sum a pessimistic one. Where a measurement lands inside this [max, sum] interval reads out overlap quality before any profiler is opened, and profilers escalate only on residuals above a stated threshold. Deployment alternatives are compared by wall ordering -- which resource wall binds first as load grows -- rather than by point benchmarks. The account is compositional: new attention or state-space variants enter by declaring one module, and the workflow ships as a zero-dependency calculator plus an agent skill that enforces the discipline in agentic optimization loops. As a case study we analyze a DeepSeek-V3.2-style 671B MoE/MLA model on 16 NVIDIA H20 GPUs, whose ridge point of ~74 FLOP/byte (vs ~590 for H100) makes it an extreme decode-oriented part. The floors show TP16 decoding is KV-capacity-limited to ~70 concurrent 8K requests; sparse attention removes the KV-bandwidth term but not the capacity wall; an EP16+DP-attention layout accepts slightly worse same-batch weight traffic for an order-of-magnitude higher capacity wall (~644) -- while single-stream latency favors TP by 2.4x. The layout judgment is thus a computable function of the operating point, explaining why production deployments on identical hardware have shipped opposite attention layouts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06913v1">Evaluating LLM Robustness Under Domain-Specific Prompt Perturbations in Public Health Applications</a></div>
    <div class="paper-meta">
      📅 2026-07-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly applied in public health applications, yet their robustness to non-clinical user inputs remains underexplored. We propose a domain specific robustness benchmark that evaluates LLMs under two perturbation types that commonly arise when non-clinical users interact with health AI systems: misinformation framing (MF), where prompt might be injected by false health claims, and layperson rewriting (LR), where patients describe symptoms in everyday language rather than medical terminology. Our goal is to evaluate the stability of LLMs under these perturbation. Experiments show that MF degrades accuracy by 7.2 pp on average with prediction flip rates of 9-38 percent, even when claims are explicitly labelled as unsupported; LR causes only 1.4 pp degradation. These findings highlight two distinct deployment risks in public health settings: models may produce incorrect outputs when users unintentionally carry misinformation into their queries, and may misinterpret clinically relevant details when patients use informal language. Both risks call for perturbation-aware robustness evaluation beyond clean baseline benchmark
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06873v1">Mining Workflow Graphs for Black-Box Boundary Testing of Conversational LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-08
    </div>
    <details class="paper-abstract">
      Conversational LLM agents can cause real-world harm when their internal workflows fail, such as completing a transaction without confirmation. Testing these state-dependent failures is difficult because critical boundaries, such as identity checks and confirmation gates, are hidden behind multi-turn conversational prerequisites, rendering them inaccessible to standard tests. We present AgentEval, a black-box testing framework that discovers and stresses these stateful boundaries. AgentEval interacts with an agent to mine a \emph{conversational workflow graph}, a model of its behavior. Instead of prompting blindly, AgentEval uses this graph's structure to enumerate specific guards and prerequisites as test targets, replaying the conversational path to a boundary before applying a perturbation. AgentEval then executes each test, determining whether it passes or fails using only the conversation turns. We benchmark AgentEval against a privileged, white-box auditor with access to the agent's underlying source code, which AgentEval never sees. On four $τ^3$-bench agents, AgentEval successfully generates tests covering $23$--$38$ distinct boundaries per agent; ablation studies attribute the gain to the graph's structure: $23$ distinct boundaries versus $12$ with a prompt-only baseline, at lower duplicate and false-alarm rates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.18360v3">Omni-Embed-Audio: Leveraging Multimodal LLMs for Robust Audio-Text Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 Accepted at ACL 2026 Main Conference. Camera-ready version
    </div>
    <details class="paper-abstract">
      Audio-text retrieval systems based on Contrastive Language-Audio Pretraining (CLAP) achieve strong performance on traditional benchmarks; however, these benchmarks rely on caption-style queries that differ substantially from real-world search behavior, limiting their assessment of practical retrieval robustness. We present Omni-Embed-Audio (OEA), a retrieval-oriented encoder leveraging multimodal LLMs with native audio understanding. To systematically evaluate robustness beyond caption-style queries, we introduce User-Intent Queries (UIQs) - five formulations reflecting natural search behaviors: questions, commands, keyword tags, paraphrases, and exclusion-based negative queries. For negative queries, we develop a hard negative mining pipeline and propose discrimination metrics (HNSR, TFR) assessing models' ability to suppress acoustically similar distractors. Experiments on AudioCaps, Clotho, and MECAT show that OEA achieves comparable text-to-audio retrieval performance to state-of-the-art M2D-CLAP, while demonstrating clear advantages in two critical areas: (1) dominant text-to-text retrieval (+22% relative improvement), and (2) substantially superior hard negative discrimination (+4.3%p HNSR@10, +34.7% relative TFR@10), revealing that LLM backbones provide superior semantic understanding of complex queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.07738v1">REFORGE: A Method for Benchmarking LLMs' Reverse Engineering Capabilities in Decompiled Binary Function Naming</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 9 pages, 4 figures; accepted for publication to the 23rd International Conference on Applied Computing 2026, Lisbon October 24-26,2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly applied to reverse-engineering tasks, and recent threat-intelligence reporting shows them operating inside live offensive-security workflows. Claims about their capability, however, outpace our ability to measure it. Existing benchmarks for LLM-assisted binary analysis treat the construction of function-level ground truth as a solved pre-processing step and report accuracy without disclosing how many functions were reliably evaluable. We argue that the principal obstacle to fair evaluation is not model capability but the reliability of binary-to-source alignment under compiler optimization. This paper presents Reforge, a provenance-tracked pipeline that constructs function-level ground truth from C source through compilation, DWARF and syntactic extraction, alignment, and decompilation, and that operationalizes alignment uncertainty as an eight-gate confidence funnel with three-tier stratification. On a controlled micro-benchmark, high-confidence yield falls from 87.2% to 65.9% across optimization levels, and unpaired comparisons overstate optimization-induced performance decay through survivorship bias. A proof-of-concept evaluation of seven contemporary LLMs on function naming demonstrates the substrate and motivates uncertainty-aware benchmarking practice
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06845v1">LLMs Silently Correct African American English: Auditing and Mitigating Dialect Bias via Activation Steering</a></div>
    <div class="paper-meta">
      📅 2026-07-07
    </div>
    <details class="paper-abstract">
      African American English (AAE), a rule-governed dialect spoken by over 30 million people, is routinely misinterpreted and "corrected" by large language models (LLMs). Across six instruction-tuned LLMs (14B to 70B), we show that state-of-the-art models systematically prefer Standard American English (SAE) continuations even when the preceding context is in AAE, effectively rewriting AAE into SAE. We present an end-to-end framework to audit and mitigate this bias. For auditing, we introduce conditional Dialect Group Invariance (cDGI), which isolates true model bias from translator-induced artifacts, and a feature-level localization analysis that identifies which AAE markers most strongly trigger bias; we find that syntactic constructions, especially negative concord (e.g., "ain't nobody"), are universal triggers across all models. For mitigation, we introduce, to our knowledge, the first application of activation steering to dialect bias: a training-free, test-time method that extracts dialect directions via causal tracing and injects them into bias-relevant layers. Activation steering reduces bias 5 to 20 times more than prompting while preserving SAE fluency. To enable this work, we release REAL-AAE , the largest real-AAE parallel corpus to date: 17,479 AAE/SAE/ AAE_back triplets from natural tweets (2 to 6 times larger than prior real-AAE resources), validated automatically (BERTScore F1 = 0.95) and by three native AAE speakers (83.0% semantic agreement).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06831v1">Gradient-Based Speech-to-Text Alignment for Any ASR Model: From CTC to Speech LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-07
    </div>
    <details class="paper-abstract">
      Speech-to-text alignment means finding the temporal boundaries of each word in the audio. Some models provide such an alignment directly and others do not. Connectionist temporal classification (CTC) and transducer models have an alignment by construction, whereas attention-based encoder-decoders (AED) and speech large language models (LLMs) do not, and their word timings are usually read off the attention weights instead. All of these signals live on the encoder frame grid, which bounds their temporal precision. We study a generic gradient-based alignment that applies to any differentiable ASR model. We take the gradient of each teacher-forced token log probability with respect to the input, reduce it to a per-frame saliency, and decode the resulting matrix into word boundaries with a single dynamic-programming pass. The method needs no training, no model modification and no alignment heads, works across all model families including the speech LLMs, and aligns on the input grid rather than on the coarser encoder grid. We evaluate it on sixteen models from four families, on read (TIMIT) and spontaneous (Buckeye) speech, each against the model's own native or attention-based alignment. We find that the gradient yields a usable alignment for every model, that it is usually somewhat behind a strong native aligner but better where the native alignment is weak, as for the streaming models, and that its main disadvantage is the cost of one backward pass per token.
    </details>
</div>
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
