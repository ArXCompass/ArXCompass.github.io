# llm - 2026_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)

## Papers

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
