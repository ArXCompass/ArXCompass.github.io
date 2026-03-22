# llm - 2026_03

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
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.19225v1">FinTradeBench: A Financial Reasoning Benchmark for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-19
      | 💬 8 pages main text, 22 pages total (including references and appendix). 5 figures, 14 tables. Preprint under review. Code and data will be made available upon publication
    </div>
    <details class="paper-abstract">
      Real-world financial decision-making is a challenging problem that requires reasoning over heterogeneous signals, including company fundamentals derived from regulatory filings and trading signals computed from price dynamics. Recently, with the advancement of Large Language Models (LLMs), financial analysts have begun to use them for financial decision-making tasks. However, existing financial question answering benchmarks for testing these models primarily focus on company balance sheet data and rarely evaluate reasoning over how company stocks trade in the market or their interactions with fundamentals. To take advantage of the strengths of both approaches, we introduce FinTradeBench, a benchmark for evaluating financial reasoning that integrates company fundamentals and trading signals. FinTradeBench contains 1,400 questions grounded in NASDAQ-100 companies over a ten-year historical window. The benchmark is organized into three reasoning categories: fundamentals-focused, trading-signal-focused, and hybrid questions requiring cross-signal reasoning. To ensure reliability at scale, we adopt a calibration-then-scaling framework that combines expert seed questions, multi-model response generation, intra-model self-filtering, numerical auditing, and human-LLM judge alignment. We evaluate 14 LLMs under zero-shot prompting and retrieval-augmented settings and witness a clear performance gap. Retrieval substantially improves reasoning over textual fundamentals, but provides limited benefit for trading-signal reasoning. These findings highlight fundamental challenges in the numerical and time-series reasoning for current LLMs and motivate future research in financial intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.19220v1">Nemotron-Cascade 2: Post-Training LLMs with Cascade RL and Multi-Domain On-Policy Distillation</a></div>
    <div class="paper-meta">
      📅 2026-03-19
      | 💬 We release the model and data at https://huggingface.co/collections/nvidia/nemotron-cascade-2
    </div>
    <details class="paper-abstract">
      We introduce Nemotron-Cascade 2, an open 30B MoE model with 3B activated parameters that delivers best-in-class reasoning and strong agentic capabilities. Despite its compact size, its mathematical and coding reasoning performance approaches that of frontier open models. It is the second open-weight LLM, after DeepSeekV3.2-Speciale-671B-A37B, to achieve Gold Medal-level performance in the 2025 International Mathematical Olympiad (IMO), the International Olympiad in Informatics (IOI), and the ICPC World Finals, demonstrating remarkably high intelligence density with 20x fewer parameters. In contrast to Nemotron-Cascade 1, the key technical advancements are as follows. After SFT on a meticulously curated dataset, we substantially expand Cascade RL to cover a much broader spectrum of reasoning and agentic domains. Furthermore, we introduce multi-domain on-policy distillation from the strongest intermediate teacher models for each domain throughout the Cascade RL process, allowing us to efficiently recover benchmark regressions and sustain strong performance gains along the way. We release the collection of model checkpoint and training data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.19195v1">How Auditory Knowledge in LLM Backbones Shapes Audio Language Models: A Holistic Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-03-19
      | 💬 Project website: https://kehanlu.github.io/AKB
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been widely used as knowledge backbones of Large Audio Language Models (LALMs), yet how much auditory knowledge they encode through text-only pre-training and how this affects downstream performance remains unclear. We study this gap by comparing different LLMs under two text-only and one audio-grounded setting: (1) direct probing on AKB-2000, a curated benchmark testing the breadth and depth of auditory knowledge; (2) cascade evaluation, where LLMs reason over text descriptions from an audio captioner; and (3) audio-grounded evaluation, where each LLM is fine-tuned into a Large Audio Language Model (LALM) with an audio encoder. Our findings reveal that auditory knowledge varies substantially across families, and text-only results are strongly correlated with audio performance. Our work provides empirical grounding for a comprehensive understanding of LLMs in audio research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08905v3">iSeal: Encrypted Fingerprinting for Reliable LLM Ownership Verification</a></div>
    <div class="paper-meta">
      📅 2026-03-19
      | 💬 Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      Given the high cost of large language model (LLM) training from scratch, safeguarding LLM intellectual property (IP) has become increasingly crucial. As the standard paradigm for IP ownership verification, LLM fingerprinting thus plays a vital role in addressing this challenge. Existing LLM fingerprinting methods verify ownership by extracting or injecting model-specific features. However, they overlook potential attacks during the verification process, leaving them ineffective when the model thief fully controls the LLM's inference process. In such settings, attackers may share prompt-response pairs to enable fingerprint unlearning or manipulate outputs to evade exact-match verification. We propose iSeal, the first fingerprinting method designed for reliable verification when the model thief controls the suspected LLM in an end-to-end manner. It injects unique features into both the model and an external module, reinforced by an error-correction mechanism and a similarity-based verification strategy. These components are resistant to verification-time attacks, including collusion-based fingerprint unlearning and response manipulation, backed by both theoretical analysis and empirical results. iSeal achieves 100 percent Fingerprint Success Rate (FSR) on 12 LLMs against more than 10 attacks, while baselines fail under unlearning and response manipulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.19182v1">Box Maze: A Process-Control Architecture for Reliable LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-03-19
      | 💬 10 pages, 5 tables, 0 figures. Conceptual architecture with preliminary simulation-based validation
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate strong generative capabilities but remain vulnerable to hallucination and unreliable reasoning under adversarial prompting. Existing safety approaches -- such as reinforcement learning from human feedback (RLHF) and output filtering -- primarily operate at the behavioral level and may lack explicit architectural mechanisms for enforcing reasoning process integrity. This paper proposes the Box Maze framework, a conceptual process-control architecture that decomposes LLM reasoning into three explicit layers: memory grounding, structured inference, and boundary enforcement. We introduce preliminary simulation-based evaluation involving progressive boundary erosion scenarios across multiple heterogeneous LLM systems (DeepSeek-V3, Doubao, Qwen). Results from n=50 adversarial scenarios suggest that explicit cognitive control layers may improve consistency in boundary maintenance, with architectural constraints reducing boundary failure rates from approximately 40% (baseline RLHF) to below 1% under adversarial conditions. While current validation is simulation-based, these preliminary results indicate that process-level control may offer a promising direction for improving reliability in large language model reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11132v2">WebWeaver: Breaking Topology Confidentiality in LLM Multi-Agent Systems with Stealthy Context-Based Inference</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      Communication topology is a critical factor in the utility and safety of LLM-based multi-agent systems (LLM-MAS), making it a high-value intellectual property (IP) whose confidentiality remains insufficiently studied. Existing topology inference attempts rely on impractical assumptions, including control over the administrative agent and direct identity queries via jailbreaks, which are easily defeated by basic keyword-based defenses. As a result, prior analyses fail to capture the real-world threat of such attacks. To bridge this realism gap, we propose \textit{WebWeaver}, an attack framework that infers the complete LLM-MAS topology by compromising only a single arbitrary agent instead of the administrative agent. Unlike prior approaches, WebWeaver relies solely on agent contexts rather than agent IDs, enabling significantly stealthier inference. WebWeaver further introduces a new covert jailbreak-based mechanism and a novel fully jailbreak-free diffusion design to handle cases where jailbreaks fail. Additionally, we address a key challenge in diffusion-based inference by proposing a masking strategy that preserves known topology during diffusion, with theoretical guarantees of correctness. Extensive experiments show that WebWeaver substantially outperforms state-of-the-art (SOTA) baselines, achieving about 60\% higher inference accuracy under active defenses with negligible overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.03415v2">Farther the Shift, Sparser the Representation: Analyzing OOD Mechanisms in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      In this work, we investigate how Large Language Models (LLMs) adapt their internal representations when encountering inputs of increasing difficulty, quantified as the degree of out-of-distribution (OOD) shift. We reveal a consistent and quantifiable phenomenon: as task difficulty increases, whether through harder reasoning questions, longer contexts, or adding answer choices, the last hidden states of LLMs become substantially sparser. In short, \textbf{\textit{the farther the shift, the sparser the representations}}. This sparsity--difficulty relation is observable across diverse models and domains, suggesting that language models respond to unfamiliar or complex inputs by concentrating computation into specialized subspaces in the last hidden state. Through a series of controlled analyses with a learning dynamic explanation, we demonstrate that this sparsity is not incidental but an adaptive mechanism for stabilizing reasoning under OOD. Leveraging this insight, we design \textit{Sparsity-Guided Curriculum In-Context Learning (SG-ICL)}, a strategy that explicitly uses representation sparsity to schedule few-shot demonstrations, leading to considerable performance enhancements. Our study provides new mechanistic insights into how LLMs internalize OOD challenges. The source code is available at the URL: https://github.com/MingyuJ666/sparsityLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19903v3">STELLAR: Structure-guided LLM Assertion Retrieval and Generation for Formal Verification</a></div>
    <div class="paper-meta">
      📅 2026-03-19
      | 💬 Accepted at the 63rd Design Automation Conference (DAC 2026), Long Beach, CA, USA (July 26-29, 2026) 7 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Formal Verification (FV) relies on high-quality SystemVerilog Assertions (SVAs), but the manual writing process is slow and error-prone. Existing LLM-based approaches either generate assertions from scratch or ignore structural patterns in hardware designs and expert-crafted assertions. This paper presents STELLAR, the first framework that guides LLM-based SVA generation with structural similarity. STELLAR represents RTL blocks as AST structural fingerprints, retrieves structurally relevant (RTL, SVA) pairs from a knowledge base, and integrates them into structure-guided prompts. Experiments show that STELLAR achieves superior syntax correctness, stylistic alignment, and functional correctness, highlighting structure-aware retrieval as a promising direction for industrial FV.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.19138v1">Implicit Patterns in LLM-Based Binary Analysis</a></div>
    <div class="paper-meta">
      📅 2026-03-19
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      Binary vulnerability analysis is increasingly performed by LLM-based agents in an iterative, multi-pass manner, with the model as the core decision-maker. However, how such systems organize exploration over hundreds of reasoning steps remains poorly understood, due to limited context windows and implicit token-level behaviors. We present the first large-scale, trace-level study showing that multi-pass LLM reasoning gives rise to structured, token-level implicit patterns. Analyzing 521 binaries with 99,563 reasoning steps, we identify four dominant patterns: early pruning, path-dependent lock-in, targeted backtracking, and knowledge-guided prioritization that emerge implicitly from reasoning traces. These token-level implicit patterns serve as an abstraction of LLM reasoning: instead of explicit control-flow or predefined heuristics, exploration is organized through implicit decisions regulating path selection, commitment, and revision. Our analysis shows these patterns form a stable, structured system with distinct temporal roles and measurable characteristics. Our results provide the first systematic characterization of LLM-driven binary analysis and a foundation for more reliable analysis systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.19133v1">A Pipelined Collaborative Speculative Decoding Framework for Efficient Edge-Cloud LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-03-19
      | 💬 8 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Recent advancements and widespread adoption of Large Language Models (LLMs) in both industry and academia have catalyzed significant demand for LLM serving. However, traditional cloud services incur high costs, while on-device inference alone faces challenges due to limited resources. Edge-cloud collaboration emerges as a key research direction to combine the strengths of both paradigms, yet efficiently utilizing limited network bandwidth while fully leveraging and balancing the computational capabilities of edge devices and the cloud remains an open problem. To address these challenges, we propose Pipelined Collaborative Speculative Decoding Framework (PicoSpec), a novel, general-purpose, and training-free speculative decoding framework for LLM edge-cloud collaborative inference. We design an asynchronous pipeline that resolves the mutual waiting problem inherent in vanilla speculative decoding within edge collaboration scenarios, which concurrently executes a Small Language Model (SLM) on the edge device and a LLM in the cloud. Meanwhile, to mitigate the significant communication latency caused by transmitting vocabulary distributions, we introduce separate rejection sampling with sparse compression, which completes the rejection sampling with only a one-time cost of transmitting the compressed vocabulary. Experimental results demonstrate that our solution outperforms baseline and existing methods, achieving up to 2.9 speedup.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.19087v1">Serendipity by Design: Evaluating the Impact of Cross-domain Mappings on Human and LLM Creativity</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      Are large language models (LLMs) creative in the same way humans are, and can the same interventions increase creativity in both? We evaluate a promising but largely untested intervention for creativity: forcing creators to draw an analogy from a random, remote source domain (''cross-domain mapping''). Human participants and LLMs generated novel features for ten daily products (e.g., backpack, TV) under two prompts: (i) cross-domain mapping, which required translating a property from a randomly assigned source (e.g., octopus, cactus, GPS), and (ii) user-need, which required proposing innovations targeting unmet user needs. We show that humans reliably benefit from randomly assigned cross-domain mappings, while LLMs, on average, generate more original ideas than humans and do not show a statistically significant effect of cross-domain mappings. However, in both systems, the impact of cross-domain mapping increases when the inspiration source becomes more semantically distant from the target. Our results highlight both the role of remote association in creative ideation and systematic differences in how humans and LLMs respond to the same intervention for creativity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.19066v1">Parallelograms Strike Back: LLMs Generate Better Analogies than People</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      Four-term word analogies (A:B::C:D) are classically modeled geometrically as ''parallelograms,'' yet recent work suggests this model poorly captures how humans produce analogies, with simple local-similarity heuristics often providing a better account (Peterson et al., 2020). But does the parallelogram model fail because it is a bad model of analogical relations, or because people are not very good at generating relation-preserving analogies? We compared human and large language model (LLM) analogy completions on the same set of analogy problems from (Peterson et al., 2020). We find that LLM-generated analogies are reliably judged as better than human-generated ones, and are also more closely aligned with the parallelogram structure in a distributional embedding space (GloVe). Crucially, we show that the improvement over human analogies was driven by greater parallelogram alignment and reduced reliance on accessible words rather than enhanced sensitivity to local similarity. Moreover, the LLM advantage is driven not by uniformly superior responses by LLMs, but by humans producing a long tail of weak completions: when only modal (most frequent) responses by both systems are compared, the LLM advantage disappears. However, greater parallelogram alignment and lower word frequency continue to predict which LLM completions are rated higher than those of humans. Overall, these results suggest that the parallelogram model is not a poor account of word analogy. Rather, humans may often fail to produce completions that satisfy this relational constraint, whereas LLMs do so more consistently.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.19059v1">SignAgent: Agentic LLMs for Linguistically-Grounded Sign Language Annotation and Dataset Curation</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      This paper introduces SignAgent, a novel agentic framework that utilises Large Language Models (LLMs) for scalable, linguistically-grounded Sign Language (SL) annotation and dataset curation. Traditional computational methods for SLs often operate at the gloss level, overlooking crucial linguistic nuances, while manual linguistic annotation remains a significant bottleneck, proving too slow and expensive for the creation of large-scale, phonologically-aware datasets. SignAgent addresses these challenges through SignAgent Orchestrator, a reasoning LLM that coordinates a suite of linguistic tools, and SignGraph, a knowledge-grounded LLM that provides lexical and linguistic grounding. We evaluate our framework on two downstream annotation tasks. First, on Pseudo-gloss Annotation, where the agent performs constrained assignment, using multi-modal evidence to extract and order suitable gloss labels for signed sequences. Second, on ID Glossing, where the agent detects and refines visual clusters by reasoning over both visual similarity and phonological overlap to correctly identify and group lexical sign variants. Our results demonstrate that our agentic approach achieves strong performance for large-scale, linguistically-aware data annotation and curation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.19030v1">LLMs Aren't Human: A Critical Perspective on LLM Personality</a></div>
    <div class="paper-meta">
      📅 2026-03-19
      | 💬 4 pages
    </div>
    <details class="paper-abstract">
      A growing body of research examines personality traits in Large Language Models (LLMs), particularly in human-agent collaboration. Prior work has frequently applied the Big Five inventory to assess LLM behavior analogous to human personality, without questioning the underlying assumptions. This paper critically evaluates whether LLM responses to personality tests satisfy six defining characteristics of personality. We find that none are fully met, indicating that such assessments do not measure a construct equivalent to human personality. We propose a research agenda for shifting from anthropomorphic trait attribution toward functional evaluations, clarifying what personality tests actually capture in LLMs and developing LLM-specific frameworks for characterizing stable, intrinsic behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.19011v1">Security awareness in LLM agents: the NDAI zone case</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      NDAI zones let inventor and investor agents negotiate inside a Trusted Execution Environment (TEE) where any disclosed information is deleted if no deal is reached. This makes full IP disclosure the rational strategy for the inventor's agent. Leveraging this infrastructure, however, requires agents to distinguish a secure environment from an insecure one, a capability LLM agents lack natively, since they can rely only on evidence passed through the context window to form awareness of their execution environment. We ask: How do different LLM models weight various forms of evidence when forming awareness of the security of their execution environment? Using an NDAI-style negotiation task across 10 language models and various evidence scenarios, we find a clear asymmetry: a failing attestation universally suppresses disclosure across all models, whereas a passing attestation produces highly heterogeneous responses: some models increase disclosure, others are unaffected, and a few paradoxically reduce it. This reveals that current LLM models can reliably detect danger signals but cannot reliably verify safety, the very capability required for privacy-preserving agentic protocols such as NDAI zones. Bridging this gap, possibly through interpretability analysis, targeted fine-tuning, or improved evidence architectures, remains the central open challenge for deploying agents that calibrate information sharing to actual evidence quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18940v1">Entropy trajectory shape predicts LLM reasoning reliability: A diagnostic study of uncertainty dynamics in chain-of-thought</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      Chain-of-thought (CoT) reasoning improves LLM accuracy, yet detecting failures cheaply remains elusive. We study whether the shape of uncertainty dynamics across reasoning steps--captured by sampling a few answer completions per step--predicts correctness. We introduce entropy-trajectory monotonicity: a chain is monotone if its per-step answer-distribution entropy decreases at every step. On GSM8K (n=300) with Qwen2.5-7B-Instruct, monotone chains achieve 68.8% accuracy vs. 46.8% for non-monotone chains (+21.9 pp; Fisher's p=0.0005; OR=2.50). Critically, total entropy reduction is not predictive ($ρ$=-0.06, p=0.31), revealing a shape-over-magnitude dissociation: whether entropy decreases at every step matters, not how much. Violation count 0/1/2 gives 68.8%/50.8%/28.6% accuracy. Token log-probability confidence worsens in calibration with step depth (ECE: 0.186->0.312), and monotonicity achieves +5.8 pp at 73.7% coverage, outperforming scalar baselines at approx 1,500 tokens/question--1/8 the cost of 40-chain self-consistency. Results replicate on Mistral-7B (n=300): monotone chains reach 72.3% vs. 37.6% (+34.7 pp; OR=4.33). Structural properties of uncertainty trajectories are thus more informative than aggregate measures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.02434v2">Who's Who? LLM-assisted Software Traceability with Architecture Entity Recognition</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      Identifying architecturally relevant entities in textual artifacts is crucial for Traceability Link Recovery (TLR) between Software Architecture Documentation (SAD) and source code. While Software Architecture Models (SAMs) can bridge the semantic gap between these artifacts, their manual creation is time-consuming. LLMs offer new capabilities for extracting architectural entities from SAD and source code to construct SAMs automatically or establish direct trace links. This paper extends our ICSA 2025 paper [19], which introduced Extracting Architecture (ExArch) for LLM-based architecture component name extraction. The extension contributes the novel Architecture Traceability with Entity Matching via Semantic inference (ArTEMiS) approach, an extended evaluation with additional LLMs, configurations, a revised benchmark, and a combined evaluation of both approaches. Specifically, this paper presents the following approaches: ExArch extracts component names as simple SAMs from SAD and source code to eliminate the need for manual SAM creation, while ArTEMiS identifies architectural entities in documentation and matches them with (manually or automatically generated) SAM entities. Our evaluation compares against state-of-the-art approaches SWATTR, TransArC and ArDoCode. TransArC achieves strong performance (F1: 0.87) but requires manually created SAMs; ExArch achieves comparable results (F1: 0.86) using only SAD and code. ArTEMiS is on par with the traditional heuristic-based SWATTR (F1: 0.81) and can successfully replace it when integrated with TransArC. The combination of ArTEMiS and ExArch outperforms ArDoCode, the best baseline without manual SAMs. Our results demonstrate that LLMs can effectively identify architectural entities in textual artifacts, enabling automated SAM generation and TLR, making architecture-code traceability more practical and accessible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17558v2">Zipper-LoRA: Dynamic Parameter Decoupling for Speech-LLM based Multilingual Speech Recognition</a></div>
    <div class="paper-meta">
      📅 2026-03-19
      | 💬 13 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Speech Large Language Models (Speech-LLMs) have emerged as a powerful approach for automatic speech recognition (ASR) by aligning speech encoders with large language models. However, adapting these systems to multilingual settings with imbalanced data distributions remains challenging. In such scenarios, a stability-plasticity dilemma often arises: fully shared Parameter-Efficient Fine-Tuning (PEFT) can cause negative inter-lingual interference for under-represented languages, while fully language-specific tuning limits the cross-lingual beneficial knowledge transfer needed for low-resource tasks. To address this, we propose Zipper-LoRA, a novel rank-level decoupling framework with three variants (Static, Hard, and Soft) that dynamically synthesizes LoRA updates from shared and language-specific subspaces. By using a lightweight language-conditioned router, Zipper-LoRA dynamically controls the contribution of each subspace at the LoRA rank level, enabling fine-grained sharing where languages are compatible and strict decoupling when conflicts occur. To further stabilize optimization under imbalanced data, we propose a two-stage training strategy with an Initial-B warm start that significantly accelerates convergence. Experiments on a 12-language mixed-resource setting show that Zipper-LoRA consistently outperforms both fully shared and independent baselines, particularly in extremely low-resource scenarios. Moreover, we demonstrate that these gains are robust across both chunked and non-chunked encoder configurations, confirming the framework's reliability for practical, large-scale multilingual ASR. Our code and data will be available at https://github.com/YuCeong-May/Zipper-LoRA for reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18897v1">Act While Thinking: Accelerating LLM Agents via Pattern-Aware Speculative Tool Execution</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      LLM-powered agents are emerging as a dominant paradigm for autonomous task solving. Unlike standard inference workloads, agents operate in a strictly serial "LLM-tool" loop, where the LLM must wait for external tool execution at every step. This execution model introduces severe latency bottlenecks. To address this problem, we propose PASTE, a Pattern-Aware Speculative Tool Execution method designed to hide tool latency through speculation. PASTE is based on the insight that although agent requests are semantically diverse, they exhibit stable application level control flows (recurring tool-call sequences) and predictable data dependencies (parameter passing between tools). By exploiting these properties, PASTE improves agent serving performance through speculative tool execution. Experimental results against state of the art baselines show that PASTE reduces average task completion time by 48.5% and improves tool execution throughput by 1.8x.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18873v1">Evaluating LLM-Generated Lessons from the Language Learning Students' Perspective: A Short Case Study on Duolingo</a></div>
    <div class="paper-meta">
      📅 2026-03-19
      | 💬 5 pages,3 figures,presented at the 3rd HEAL Workshop at CHI 2026
    </div>
    <details class="paper-abstract">
      Popular language learning applications such as Duolingo use large language models (LLMs) to generate lessons for its users. Most lessons focus on general real-world scenarios such as greetings, ordering food, or asking directions, with limited support for profession-specific contexts. This gap can hinder learners from achieving professional-level fluency, which we define as the ability to communicate comfortably various work-related and domain-specific information in the target language. We surveyed five employees from a multinational company in the Philippines on their experiences with Duolingo. Results show that respondents encountered general scenarios more frequently than work-related ones, and that the former are relatable and effective in building foundational grammar, vocabulary, and cultural knowledge. The latter helps bridge the gap toward professional fluency as it contains domain-specific vocabulary. Each participant suggested lesson scenarios that diverge in contexts hen analyzed in aggregate. With this understanding, we propose that language learning applications should generate lessons that adapt to an individual's needs through personalized, domain specific lesson scenarios while maintaining foundational support through general, relatable lesson scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18855v1">BeamAgent: LLM-Aided MIMO Beamforming with Decoupled Intent Parsing and Alternating Optimization for Joint Site Selection and Precoding</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      Integrating large language models (LLMs) into wireless communication optimization is a promising yet challenging direction. Existing approaches either use LLMs as black-box solvers or code generators, tightly coupling them with numerical computation. However, LLMs lack the precision required for physical-layer optimization, and the scarcity of wireless training data makes domain-specific fine-tuning impractical. We propose BeamAgent, an LLM-aided MIMO beamforming framework that explicitly decouples semantic intent parsing from numerical optimization. The LLM serves solely as a semantic translator that converts natural language descriptions into structured spatial constraints. A dedicated gradient-based optimizer then jointly solves the discrete base station site selection and continuous precoding design through an alternating optimization algorithm. A scene-aware prompt enables grounded spatial reasoning without fine-tuning, and a multi-round interaction mechanism with dual-layer intent classification ensures robust constraint verification. A penalty-based loss function enforces dark-zone power constraints while releasing optimization degrees of freedom for bright-zone gain maximization. Experiments on a ray-tracing-based urban MIMO scenario show that BeamAgent achieves a bright-zone power of 84.0\,dB, outperforming exhaustive zero-forcing by 7.1 dB under the same dark-zone constraint. The end-to-end system reaches within 3.3 dB of the expert upper bound, with the full optimization completing in under 2 s on a laptop.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.00270v2">Transformers Remember First, Forget Last: Dual-Process Interference in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-19
      | 💬 16 pages, 10 figures. Under review
    </div>
    <details class="paper-abstract">
      When large language models encounter conflicting information in context, which memories survive -- early or recent? We adapt classical interference paradigms from cognitive psychology to answer this question, testing 39 LLMs across diverse architectures and scales. Every model shows the same pattern: proactive interference (PI) dominates retroactive interference (RI) universally (Cohen's d = 1.73, p < 0.0001), meaning early encodings are protected at the cost of recent information -- the opposite of human memory, where RI typically dominates. Three findings indicate that RI and PI reflect separate memory mechanisms. RI and PI are uncorrelated (R^2 = 0.044), rejecting a unified "memory capacity." Model size predicts RI resistance (R^2 = 0.49) but not PI (R^2 = 0.06, n.s.) -- only RI is capacity-dependent. And error analysis reveals distinct failure modes: RI failures are passive retrieval failures (51%), while PI failures show active primacy intrusion (56%); both show <1% hallucination. These patterns parallel the consolidation-retrieval distinction in cognitive science, suggesting that transformer attention creates a primacy bias with direct implications for interference-heavy applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18815v1">ProRL Agent: Rollout-as-a-Service for RL Training of Multi-Turn LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      Multi-turn LLM agents are increasingly important for solving complex, interactive tasks, and reinforcement learning (RL) is a key ingredient for improving their long-horizon behavior. However, RL training requires generating large numbers of sandboxed rollout trajectories, and existing infrastructures often couple rollout orchestration with the training loop, making systems hard to migrate and maintain. Under the rollout-as-a-service philosophy, we present ProRL Agent , a scalable infrastructure that serves the full agentic rollout lifecycle through an API service. ProRL Agent also provides standardized and extensible sandbox environments that support diverse agentic tasks in rootless HPC settings. We validate ProRL Agent through RL training on software engineering, math, STEM, and coding tasks. ProRL Agent is open-sourced and integrated as part of NVIDIA NeMo Gym.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18813v1">Can LLM generate interesting mathematical research problems?</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      This paper is the second one in a series of work on the mathematical creativity of LLM. In the first paper, the authors proposed three criteria for evaluating the mathematical creativity of LLM and constructed a benchmark dataset to measure it. This paper further explores the mathematical creativity of LLM, with a focus on investigating whether LLM can generate valuable and cutting-edge mathematical research problems. We develop an agent to generate unknown problems and produced 665 research problems in differential geometry. Through human verification, we find that many of these mathematical problems are unknown to experts and possess unique research value.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11103v2">Understanding by Reconstruction: Reversing the Software Development Process for LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have achieved remarkable success in code generation, they often struggle with the deep, long-horizon reasoning required for complex software engineering. We attribute this limitation to the nature of standard pre-training data: static software repositories represent only the terminal state of an intricate intellectual process, abstracting away the intermediate planning, debugging, and iterative refinement. To bridge this gap, we propose a novel paradigm: understanding via reconstruction. We hypothesize that reverse-engineering the latent agentic trajectories -- the planning, reasoning, and debugging steps -- behind static repositories provides a far richer supervision signal than raw code alone. To operationalize this, we introduce a framework that synthesizes these trajectories using a multi-agent simulation. This process is grounded in the structural realities of the source repositories (e.g., dependency graphs and file hierarchies) to ensure fidelity. Furthermore, to guarantee the logical rigor of the synthetic data, we employ a search-based optimization technique that iteratively refines the Chain-of-Thought (CoT) reasoning to maximize the likelihood of the ground-truth code. Empirical results demonstrate that continuous pre-training on these reconstructed trajectories significantly enhances Llama-3-8B's performance across diverse benchmarks, including long-context understanding, coding proficiency, and agentic capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18773v1">Automatic Configuration of LLM Post-Training Pipelines</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      LLM post-training pipelines that combine supervised fine-tuning and reinforcement learning are difficult to configure under realistic compute budgets: the configuration space is high-dimensional and heterogeneous, stages are strongly coupled, and each end-to-end evaluation is expensive. We propose AutoPipe, a budget-aware two-stage framework for configuration selection in LLM post-training. Offline, AutoPipe learns a dataset-conditioned learning-to-rank surrogate from historical runs, capturing within-dataset preferences and providing transferable guidance toward promising regions of the configuration space. Online, for a new dataset, AutoPipe uses the offline guidance to steer Bayesian optimization and models dataset-specific deviations with a Gaussian-process residual surrogate. To reduce evaluation cost, each trial is early-stopped and scored by a learned predictor that maps early training signals to a low-cost proxy for final post-training performance. Experiments on biomedical reasoning tasks show that AutoPipe consistently outperforms offline-only baselines and achieves comparable performance with the strongest online HPO baselines while using less than 10\% of their computational cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02957v3">LLM-Augmented Changepoint Detection: A Framework for Ensemble Detection and Automated Explanation</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      This paper introduces a novel changepoint detection framework that combines ensemble statistical methods with Large Language Models (LLMs) to enhance both detection accuracy and the interpretability of regime changes in time series data. Two critical limitations in the field are addressed. First, individual detection methods exhibit complementary strengths and weaknesses depending on data characteristics, making method selection non-trivial and prone to suboptimal results. Second, automated, contextual explanations for detected changes are largely absent. The proposed ensemble method aggregates results from ten distinct changepoint detection algorithms, achieving superior performance and robustness compared to individual methods. Additionally, an LLM-powered explanation pipeline automatically generates contextual narratives, linking detected changepoints to potential real-world historical events. For private or domain-specific data, a Retrieval-Augmented Generation (RAG) solution enables explanations grounded in user-provided documents. The open source Python framework demonstrates practical utility in diverse domains, including finance, political science, and environmental science, transforming raw statistical output into actionable insights for analysts and decision-makers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18756v1">Are complicated loss functions necessary for teaching LLMs to reason?</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) highlight the importance of post training techniques for improving reasoning and mathematical ability. Group Relative Policy Optimization (GRPO) has shown promise in this domain by combining group relative advantage estimation, PPO style clipping, and KL regularization. However, its complexity raises the question of whether all components are necessary for fostering reasoning behaviors. We conduct a systematic analysis of GRPO and identify two key findings: (1) incorporating negative feedback is essential training solely on actions above a baseline limits learning; and (2) PPO style constraints, such as policy ratio clipping, are not required to improve mathematical reasoning or performance. Building on these insights, we propose REINFORCE with Group Relative Advantage (RGRA), a simplified variant that retains group relative advantage estimation but removes PPO style clipping and policy ratio terms. Experiments across standard mathematical benchmarks indicate that RGRA has the potential to achieve stronger performance than GRPO. Our results suggest that simpler REINFORCE based approaches can effectively enhance reasoning in LLMs, offering a more transparent and efficient alternative to GRPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11599v3">SynBullying: A Multi LLM Synthetic Conversational Dataset for Cyberbullying Detection</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      We introduce SynBullying, a synthetic multi-LLM conversational dataset for studying and detecting cyberbullying (CB). SynBullying provides a scalable and ethically safe alternative to human data collection by leveraging large language models (LLMs) to simulate realistic bullying interactions. The dataset offers (i) conversational structure, capturing multi-turn exchanges rather than isolated posts; (ii) context-aware annotations, where harmfulness is assessed within the conversational flow considering context, intent, and discourse dynamics; and (iii) fine-grained labeling, covering various CB categories for detailed linguistic and behavioral analysis. We evaluate SynBullying across five dimensions, including conversational structure, lexical patterns, sentiment/toxicity, role dynamics, harm intensity, and CB-type distribution. We further examine its utility by testing its performance as standalone training data and as an augmentation source for CB classification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18740v1">Measuring and Exploiting Confirmation Bias in LLM-Assisted Security Code Review</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      Security code reviews increasingly rely on systems integrating Large Language Models (LLMs), ranging from interactive assistants to autonomous agents in CI/CD pipelines. We study whether confirmation bias (i.e., the tendency to favor interpretations that align with prior expectations) affects LLM-based vulnerability detection, and whether this failure mode can be exploited in software supply-chain attacks. We conduct two complementary studies. Study 1 quantifies confirmation bias through controlled experiments on 250 CVE vulnerability/patch pairs evaluated across four state-of-the-art models under five framing conditions for the review prompt. Framing a change as bug-free reduces vulnerability detection rates by 16-93%, with strongly asymmetric effects: false negatives increase sharply while false positive rates change little. Bias effects vary by vulnerability type, with injection flaws being more susceptible to them than memory corruption bugs. Study 2 evaluates exploitability in practice mimicking adversarial pull requests that reintroduce known vulnerabilities while framed as security improvements or urgent functionality fixes via their pull request metadata. Adversarial framing succeeds in 35% of cases against GitHub Copilot (interactive assistant) under one-shot attacks and in 88% of cases against Claude Code (autonomous agent) in real project configurations where adversaries can iteratively refine their framing to increase attack success. Debiasing via metadata redaction and explicit instructions restores detection in all interactive cases and 94% of autonomous cases. Our results show that confirmation bias poses a weakness in LLM-based code review, with implications on how AI-assisted development tools are deployed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18734v1">Green Architectural Tactics in ML-enabled Systems: An LLM-based Repository Mining Study</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      Context: The increasing adoption of machine learning (ML) and artificial intelligence (AI) technologies raises growing concerns about their environmental sustainability. Developing and deploying ML-enabled systems is computationally intensive, particularly during training and inference. Green AI has emerged to address these issues by promoting efficiency without sacrificing accuracy. While prior research has proposed catalogs of sustainable practices (i.e., green tactics), there remains limited understanding of their adoption in practice and whether additional, undocumented tactics exist. Objective: This study aims to investigate the extent to which existing sustainable practices are implemented in real-world ML-enabled systems and to identify previously undocumented practices that support environmental sustainability. Method: We conduct a mining software repository study on 205 open-source ML projects on GitHub. To support our analysis, we design a novel mechanism based on large language models (LLMs) capable of identifying both known and new sustainable practices from code repositories. Results: Our findings confirm that green tactics reported in the literature are used in practice, although adoption rates vary. Furthermore, our LLM-based approach reveals nine previously undocumented sustainable practices. Each tactic is supported with code examples to aid adoption and integration. Conclusions: We finally provide insights for practitioners seeking to reduce the environmental impact of ML-enabled systems and offer a foundation for future research in automating the detection and adoption of sustainable practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18652v1">Benchmarking PDF Parsers on Table Extraction with LLM-based Semantic Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-03-19
      | 💬 Submitted to ICDAR 2026
    </div>
    <details class="paper-abstract">
      Reliably extracting tables from PDFs is essential for large-scale scientific data mining and knowledge base construction, yet existing evaluation approaches rely on rule-based metrics that fail to capture semantic equivalence of table content. We present a benchmarking framework based on synthetically generated PDFs with precise LaTeX ground truth, using tables sourced from arXiv to ensure realistic complexity and diversity. As our central methodological contribution, we apply LLM-as-a-judge for semantic table evaluation, integrated into a matching pipeline that accommodates inconsistencies in parser outputs. Through a human validation study comprising over 1,500 quality judgments on extracted table pairs, we show that LLM-based evaluation achieves substantially higher correlation with human judgment (Pearson r=0.93) compared to Tree Edit Distance-based Similarity (TEDS, r=0.68) and Grid Table Similarity (GriTS, r=0.70). Evaluating 21 contemporary PDF parsers across 100 synthetic documents containing 451 tables reveals significant performance disparities. Our results offer practical guidance for selecting parsers for tabular data extraction and establish a reproducible, scalable evaluation methodology for this critical task. Code and data: https://github.com/phorn1/pdf-parse-bench Metric study and human evaluation: https://github.com/phorn1/table-metric-study
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16137v2">SIA: A Synthesize-Inject-Align Framework for Knowledge-Grounded and Secure E-commerce Search LLMs with Industrial Deployment</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      Large language models offer transformative potential for e-commerce search by enabling intent-aware recommendations. However, their industrial deployment is hindered by two critical challenges: (1) knowledge hallucination due to insufficient encoding of dynamic, fine-grained product knowledge, and (2) security vulnerabilities under jailbreak attacks that threaten compliance. To address these issues, we propose SIA--a Synthesize-Inject-Align framework for building knowledgeable and secure e-commerce search LLMs. Our approach first synthesizes high-quality natural language corpus by combining structured knowledge graphs with unstructured behavioral logs, augmented with reasoning chains and safety-aware data. We then introduce a parameter-efficient pre-training strategy based on Depth Up-Scaling to inject domain knowledge while preserving general capabilities. Finally, a dual-path alignment method via multi-task instruction tuning and adversarial training strengthens both task performance and safety robustness. The framework has been deployed at JD.com, China's largest self-operated e-commerce platform, where A/B tests across five core search scenarios demonstrate significant improvements in key business metrics, validating its industrial effectiveness and scalability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18614v1">ZEBRAARENA: A Diagnostic Simulation Environment for Studying Reasoning-Action Coupling in Tool-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      Tool-augmented large language models (LLMs) must tightly couple multi-step reasoning with external actions, yet existing benchmarks often confound this interplay with complex environment dynamics, memorized knowledge or dataset contamination. In this paper, we introduce ZebraArena, a procedurally generated diagnostic environment for studying reasoning-action coupling in tool-augmented LLMs, with controllable difficulty and a knowledge-minimal design, which limits gains from memorization or dataset contamination. Each task in ZebraArena requires a set of critical information which is available only through targeted tool use, yielding an interpretable interface between external information acquisition and deductive reasoning. This design provides deterministic evaluation via unique solutions, and a theoretical optimal query count for measuring efficient tool use. We show that ZebraArena requires a combination of in-depth reasoning and accurate external tool calling, which remains a challenge as frontier reasoning models such as GPT-5 and Gemini 2.5 Pro only achieves 60% accuracy on the hard instances. We also observe a persistent gaps between theoretical optimality and practical tool usage. For example, GPT-5 uses 70-270% more tool calls than the theoretical optimum. We highlight the key findings in our evaluation, and hope ZebraArena stimulates further research on the interplay between internal reasoning and external action.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18604v1">AutORAN: LLM-driven Natural Language Programming for Agile xApp Development</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      Traditional RAN systems are closed and monolithic, stifling innovation. The openness and programmability enabled by Open Radio Access Network (O-RAN) are envisioned to revolutionize cellular networks with control-plane applications--xApps. The development of xApps (typically by third-party developers), however, remains time-consuming and cumbersome, often requiring months of manual coding and integration, which hinders the roll-out of new functionalities in practice. To lower the barrier of xApp development for both developers and network operators, we present AutORAN, the first LLM-driven natural language programming framework for agile xApps that automates the entire xApp development pipeline. In a nutshell, AutORAN turns high-level user intents into swiftly deployable xApps within minutes, eliminating the need for manual coding or testing. To this end, AutORAN builds a fully automated xApp generation pipeline, which integrates multiple functional modules (from user requirement elicitation, AI/ML function design and validation, to xApp synthesis and deployment). We design, implement, and comprehensively evaluate AutORAN on representative xApp tasks. Results show AutORAN-generated xApps can achieve similar or even better performance than the best known hand-crafted baselines. AutORAN drastically accelerates the xApp development cycle (from user intent elicitation to roll-out), streamlining O-RAN innovation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19980v2">Neuron-Guided Interpretation of Code LLMs: Where, Why, and How?</a></div>
    <div class="paper-meta">
      📅 2026-03-19
      | 💬 Accepted by FSE2026
    </div>
    <details class="paper-abstract">
      Code language models excel on code intelligence tasks, yet their internal interpretability is underexplored. Existing neuron interpretability techniques from NLP are suboptimal for source code due to programming languages formal, hierarchical, and executable nature. We empirically investigate code LLMs at the neuron level, localizing language-specific neurons (selectively responsive to one language) and concept layers (feed-forward layers encoding language-agnostic code representations). We analyze Llama-3.1-8B and Qwen2.5-Coder-32B on multilingual inputs in C++, Java, Python, Go, and JavaScript, measuring neuron selectivity and layerwise contributions during generation. We find (1) neurons specialized for individual languages alongside a universal subset supporting general-purpose generation; and (2) lower layers mainly encode language-specific syntax, while middle layers capture semantic abstractions shared across languages, emerging as concept layers. We demonstrate utility on three tasks: neuron-guided fine-tuning for code generation, clone detection via concept-layer embeddings, and concept-layer-guided transfer for code summarization, each yielding consistent gains in multilingual settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18579v1">ICE: Intervention-Consistent Explanation Evaluation with Statistical Grounding for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      Evaluating whether explanations faithfully reflect a model's reasoning remains an open problem. Existing benchmarks use single interventions without statistical testing, making it impossible to distinguish genuine faithfulness from chance-level performance. We introduce ICE (Intervention-Consistent Explanation), a framework that compares explanations against matched random baselines via randomization tests under multiple intervention operators, yielding win rates with confidence intervals. Evaluating 7 LLMs across 4 English tasks, 6 non-English languages, and 2 attribution methods, we find that faithfulness is operator-dependent: operator gaps reach up to 44 percentage points, with deletion typically inflating estimates on short text but the pattern reversing on long text, suggesting that faithfulness should be interpreted comparatively across intervention operators rather than as a single score. Randomized baselines reveal anti-faithfulness in one-third of configurations, and faithfulness shows zero correlation with human plausibility (|r| < 0.04). Multilingual evaluation reveals dramatic model-language interactions not explained by tokenization alone. We release the ICE framework and ICEBench benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18530v1">When Names Change Verdicts: Intervention Consistency Reveals Systematic Bias in LLM Decision-Making</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used for high-stakes decisions, yet their susceptibility to spurious features remains poorly characterized. We introduce ICE-Guard, a framework applying intervention consistency testing to detect three types of spurious feature reliance: demographic (name/race swaps), authority (credential/prestige swaps), and framing (positive/negative restatements). Across 3,000 vignettes spanning 10 high-stakes domains, we evaluate 11 LLMs from 8 families and find that (1) authority bias (mean 5.8%) and framing bias (5.0%) substantially exceed demographic bias (2.2%), challenging the field's narrow focus on demographics; (2) bias concentrates in specific domains -- finance shows 22.6% authority bias while criminal justice shows only 2.8%; (3) structured decomposition, where the LLM extracts features and a deterministic rubric decides, reduces flip rates by up to 100% (median 49% across 9 models). We demonstrate an ICE-guided detect-diagnose-mitigate-verify loop achieving cumulative 78% bias reduction via iterative prompt patching. Validation against real COMPAS recidivism data shows COMPAS-derived flip rates exceed pooled synthetic rates, suggesting our benchmark provides a conservative estimate of real-world bias. Code and data are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.01113v4">LLMs Faithfully and Iteratively Compute Answers During CoT: A Systematic Analysis With Multi-step Arithmetics</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      This study investigates the internal information flow of large language models (LLMs) while performing chain-of-thought (CoT) style reasoning. Specifically, with a particular interest in the faithfulness of the CoT explanation to LLMs' final answer, we explore (i) when the LLMs' answer is (pre)determined, especially before the CoT begins or after, and (ii) how strongly the information from CoT specifically has a causal effect on the final answer. Our experiments with controlled arithmetic tasks reveal a systematic internal reasoning mechanism of LLMs. They have not derived an answer at the moment when input was fed into the model. Instead, they compute (sub-)answers while generating the reasoning chain on the fly. Therefore, the generated reasoning chains can be regarded as faithful reflections of the model's internal computation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18507v1">Expert Personas Improve LLM Alignment but Damage Accuracy: Bootstrapping Intent-Based Persona Routing with PRISM</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      Persona prompting can steer LLM generation towards a domain-specific tone and pattern. This behavior enables use cases in multi-agent systems where diverse interactions are crucial and human-centered tasks require high-level human alignment. Prior works provide mixed opinions on their utility: some report performance gains when using expert personas for certain domains and their contribution to data diversity in synthetic data creation, while others find near-zero or negative impact on general utility. To fully leverage the benefits of the LLM persona and avoid its harmfulness, a more comprehensive investigation of the mechanism is crucial. In this work, we study how model optimization, task type, prompt length, and placement can impact expert persona effectiveness across instruction-tuned and reasoning LLMs, and provide insight into conditions under which expert personas fail and succeed. Based on our findings, we developed a pipeline to fully leverage the benefits of an expert persona, named PRISM (Persona Routing via Intent-based Self-Modeling), which self-distills an intent-conditioned expert persona into a gated LoRA adapter through a bootstrapping process that requires no external data, models, or knowledge. PRISM enhances human preference and safety alignment on generative tasks while maintaining accuracy on discriminative tasks across all models, with minimal memory and computing overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.15159v3">To See is Not to Master: Teaching LLMs to Use Private Libraries for Code Generation</a></div>
    <div class="paper-meta">
      📅 2026-03-19
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown strong potential for code generation, yet they remain limited in private-library-oriented code generation, where the goal is to generate code using APIs from private libraries. Existing approaches mainly rely on retrieving private-library API documentation and injecting relevant knowledge into the context at inference time. However, our study shows that this is insufficient: even given accurate required knowledge, LLMs still struggle to invoke private-library APIs effectively. To address this limitation, we propose PriCoder, an approach that teaches LLMs to invoke private-library APIs through automatically synthesized data. Specifically, PriCoder models private-library data synthesis as the construction of a graph, and alternates between two graph operators: (1) Progressive Graph Evolution, which improves data diversity by progressively synthesizing more diverse training samples from basic ones, and (2) Multidimensional Graph Pruning, which improves data quality through a rigorous filtering pipeline. To support rigorous evaluation, we construct two new benchmarks based on recently released libraries that are unfamiliar to the tested models. Experiments on three mainstream LLMs show that PriCoder substantially improves private-library-oriented code generation, yielding gains of over 20% in pass@1 in many settings, while causing negligible impact on general code generation capability. Our code and benchmarks are publicly available at https://github.com/eniacode/PriCoder.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18474v1">WASD: Locating Critical Neurons as Sufficient Conditions for Explaining and Controlling LLM Behavior</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      Precise behavioral control of large language models (LLMs) is critical for complex applications. However, existing methods often incur high training costs, lack natural language controllability, or compromise semantic coherence. To bridge this gap, we propose WASD (unWeaving Actionable Sufficient Directives), a novel framework that explains model behavior by identifying sufficient neural conditions for token generation. Our method represents candidate conditions as neuron-activation predicates and iteratively searches for a minimal set that guarantees the current output under input perturbations. Experiments on SST-2 and CounterFact with the Gemma-2-2B model demonstrate that our approach produces explanations that are more stable, accurate, and concise than conventional attribution graphs. Moreover, through a case study on controlling cross-lingual output generation, we validated the practical effectiveness of WASD in controlling model behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.07884v2">LLMIA: An Out-of-the-Box Index Advisor via In-Context Learning with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      Index recommendation is crucial for optimizing database performance. However, existing heuristic- and learning-based methods often rely on inefficient exhaustive search and estimated costs, leading to low efficiency (due to the vast search space) and unsatisfactory actual latency (due to inaccurate estimations). Inspired by the refinement strategies of experienced DBAs-who efficiently identify and iteratively refine indexes with database feedback-we present LLMIA, an out-of-the-box, tuning-free index advisor leveraging large language models (LLMs) through in-context learning for index recommendation. LLMIA injects database expertise into the LLM using a high-quality demonstration pool and comprehensive workload feature extraction, while iteratively incorporating database feedback to guide the index refinement. This design enables LLMIA to emulate the decision-making process of expert DBAs: efficiently recommending and refining indexes for various workloads within just a few interactions with the DBMS. We validate LLMIA with extensive experiments on five standard OLAP benchmarks (TPC-H with different scales, JOB, TPC-DS, SSB), where it consistently outperforms or matches 12 baselines by producing superior index recommendations with minimal database interactions. Additionally, LLMIA demonstrates robust generalization on two real-world commercial workloads, delivering high-quality recommendations without the need for additional adaptation or retraining, highlighting its out-of-the-box capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.11468v2">TRACE: Evaluating Execution Efficiency of LLM-Based Code Translation</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have substantially improved the functional correctness of code translation, the critical dimension of \textit{execution efficiency} remains overlooked. We present \textbf{\textsc{trace}}, the first benchmark to explicitly assess efficiency in LLM-translated code. \textsc{trace} includes 1,000 efficiency-critical tasks across C++, Java, and Python, each augmented with stress tests that reveal efficiency degradations often overlooked by small-scale tests. Using \textsc{trace}, we conduct an extensive evaluation of 28 representative LLMs and highlight several key insights: 1) Correctness is not a reliable proxy for efficiency: the correctness leader \textit{Claude-4-think} achieves only mid-level time efficiency, outperformed by smaller open-source LLMs such as \textit{Qwen2.5-Coder-14B-Instruct}. 2) Inefficiency is both prevalent and patterned: 23.5\% of correct translations exhibit pronounced inefficiency, distributed across algorithmic faults (11.9\%), language construct mismatches (66.4\%), and resource mismanagement (21.7\%). 3) Inference-time prompt strategies bring only modest improvements, suggesting that current LLMs lack intrinsic efficiency awareness. Together, our results establish efficiency as an essential dimension of code translation and position \textsc{trace} as a principled foundation for efficiency-oriented evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18449v1">CNT: Safety-oriented Function Reuse across LLMs via Cross-Model Neuron Transfer</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      The widespread deployment of large language models (LLMs) calls for post-hoc methods that can flexibly adapt models to evolving safety requirements. Meanwhile, the rapidly expanding open-source LLM ecosystem has produced a diverse collection of models that already exhibit various safety-related functionalities. This motivates a shift from constructing safety functionality from scratch to reusing existing functionality from external models, thereby avoiding costly data collection and training procedures. In this paper, we present Cross-Model Neuron Transfer (CNT), a post-hoc method that reuses safety-oriented functionality by transferring a minimal subset of neurons from an open-source donor LLM to a target LLM. By operating at the neuron level, CNT enables modular function-level adaptation, supporting both function addition andfunction deletion. We evaluate CNT on seven popular LLMs across three representative applications: safety disalignment, alignment enhancement, and bias removal. Experimental results show that CNT achieves targeted safety-oriented functionality transfer with minimal performance degradation (less than 1% for most models), consistently outperforming five baselines, demonstrating its generality and practical effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16479v2">TRACE: Evaluating Execution Efficiency of LLM-Based Code Translation</a></div>
    <div class="paper-meta">
      📅 2026-03-19
      | 💬 Submitted in error as a new submission instead of a replacement for arXiv:2508.11468
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have substantially improved the functional correctness of code translation, the critical dimension of \textit{execution efficiency} remains overlooked. We present \textbf{\textsc{trace}}, the first benchmark to explicitly assess efficiency in LLM-translated code. \textsc{trace} includes 1,000 efficiency-critical tasks across C++, Java, and Python, each augmented with stress tests that reveal efficiency degradations often overlooked by small-scale tests. Using \textsc{trace}, we conduct an extensive evaluation of 28 representative LLMs and highlight several key insights: 1) Correctness is not a reliable proxy for efficiency: the correctness leader \textit{Claude-4-think} achieves only mid-level time efficiency, outperformed by smaller open-source LLMs such as \textit{Qwen2.5-Coder-14B-Instruct}. 2) Inefficiency is both prevalent and patterned: 23.5\% of correct translations exhibit pronounced inefficiency, distributed across algorithmic faults (11.9\%), language construct mismatches (66.4\%), and resource mismanagement (21.7\%). 3) Inference-time prompt strategies bring only modest improvements, suggesting that current LLMs lack intrinsic efficiency awareness. Together, our results establish efficiency as an essential dimension of code translation and position \textsc{trace} as a principled foundation for efficiency-oriented evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18433v1">Prompt Control-Flow Integrity: A Priority-Aware Runtime Defense Against Prompt Injection in LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-03-19
      | 💬 4 Figures, 3 Tables, Submitted to the International Conference on Power, Electronics, Communications, Computing, and Intelligent Infrastructure 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) deployed behind APIs and retrieval-augmented generation (RAG) stacks are vulnerable to prompt injection attacks that may override system policies, subvert intended behavior, and induce unsafe outputs. Existing defenses often treat prompts as flat strings and rely on ad hoc filtering or static jailbreak detection. This paper proposes Prompt Control-Flow Integrity (PCFI), a priority-aware runtime defense that models each request as a structured composition of system, developer, user, and retrieved-document segments. PCFI applies a three-stage middleware pipeline, lexical heuristics, role-switch detection, and hierarchical policy enforcement, before forwarding requests to the backend LLM. We implement PCFI as a FastAPI-based gateway for deployed LLM APIs and evaluate it on a custom benchmark of synthetic and semi-realistic prompt-injection workloads. On the evaluated benchmark suite, PCFI intercepts all attack-labeled requests, maintains a 0% False Positive Rate, and introduces a median processing overhead of only 0.04 ms. These results suggest that provenance- and priority-aware prompt enforcement is a practical and lightweight defense for deployed LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18425v1">Multimodal Task Interference: A Benchmark and Analysis of History-Target Mismatch in Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      Task interference, the performance degradation caused by task switches within a single conversation, has been studied exclusively in text-only settings despite the growing prevalence of multimodal dialogue systems. We introduce a benchmark for evaluating this phenomenon in multimodal LLMs, covering six tasks across text and vision with systematic variation of history-target along three axes: modality mismatch, reasoning mismatch, and answer format mismatch. Experiments on both open-weights and proprietary models reveal that task interference is highly directional: switching from text-only to image-based targets causes severe performance drops, while the reverse transition yields minimal degradation. Interference is further amplified when mismatches co-occur across multiple dimensions, and is driven most strongly by modality differences, followed by answer format, while reasoning requirement shifts cause minimal degradation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.20558v2">From Logs to Language: Learning Optimal Verbalization for LLM-Based Recommendation at Industry Scale</a></div>
    <div class="paper-meta">
      📅 2026-03-19
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are promising backbones for generative recommender systems, yet a key challenge remains underexplored: verbalization, i.e., converting structured user interaction logs into effective natural language inputs. Existing methods rely on rigid templates that simply concatenate fields, yielding suboptimal representations for recommendation. We propose a data-centric framework that learns verbalization for LLM-based recommendation. Using reinforcement learning, a verbalization agent transforms raw interaction histories into optimized textual contexts, with recommendation accuracy as the training signal. This agent learns to filter noise, incorporate relevant metadata, and reorganize information to improve downstream predictions. Experiments on a large-scale industrial streaming dataset from Netflix show that learned verbalization delivers up to 93% relative improvement in discovery item recommendation accuracy over template-based baselines. Further analysis reveals emergent strategies such as user interest summarization, noise removal, and syntax normalization, offering insights into effective context construction for LLM-based recommender systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18411v1">TARo: Token-level Adaptive Routing for LLM Test-time Alignment</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit strong reasoning capabilities but typically require expensive post-training to reach high performance. Recent test-time alignment methods offer a lightweight alternative, but have been explored mainly for preference alignment rather than reasoning. To bridge this gap, we propose, Token-level Adaptive Routing (TARo), which steers frozen LLMs toward structured reasoning entirely at inference time. Specifically, we first train reward models on step-wise mathematical traces to capture fine-grained logical consistency signals, then introduce a learnable token-level router that automatically controls the guidance of the reward model to the base model. Extensive experiments show that TARo significantly improves reasoning performance by up to +22.4% over base model and +8.4% over existing token-level test-time alignment methods, while also boosting out-of-distribution clinical reasoning (MedXpertQA) and instruction following (AlpacaEval). Furthermore, TARo also generalizes from small to large backbones without retraining, extending test-time alignment from preference optimization to robust, cross-domain reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17272v2">Network and Device Level Cyber Deception for Contested Environments Using RL and LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-19
      | 💬 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Cyber deception assists in increasing the attacker's budget in reconnaissance or any early phases of threat intrusions. In the past, numerous methods of cyber deception have been adopted, such as IP address randomization, the creation of honeypots and honeynets mimicking an actual set of services, and networks deployed within an enterprise or operational technology(OT) network. These types of strategies follow naive approaches of recreating services that are expensive and that need a lot of human intervention. The advent of cloud services and other automations of containerized applications, such as Kubernetes, makes cyber defense easier. Yet, there remains a lot of potential to improve the accuracy of these deception strategies and to make them cost-effective using artificial intelligence (AI)-based solutions by making the deception more dynamic. Hence, in this work, we review various AI-based solutions in building network- and device-level cyber deception methods in contested environments. Specifically, we focus on leveraging the fusion of large language models (LLMs) and reinforcement learning(RL) in optimally learning these cyber deception strategies and validating the efficacy of such strategies in some stealthy attacks against OT systems in the literature.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.12671v2">HyGra: Accelerating Network-State Simulation for LLM Training in DCNs via Adaptive Packet-Flow Granularity</a></div>
    <div class="paper-meta">
      📅 2026-03-19
      | 💬 14 pages, 7 figures and 5 tables
    </div>
    <details class="paper-abstract">
      In recent years, large language models (LLMs) have driven substantial intelligent transformation across diverse industries. Commercial LLM training is typically performed over data center networks (DCNs) comprising hundreds to thousands of GPUs, with multiple devices collocated per node. As network scale expands, inter-node communication becomes a primary bottleneck to training efficiency. Network-state simulators therefore play a crucial role by enabling cost-effective evaluation of network configurations and parallelization strategies through faithful emulation of DCN dynamics during LLM training. However, existing simulators are constrained by a efficiency-fidelity tradeoff, as packet-level simulators (PLSs) incur prohibitive runtime overhead, whereas flow-level simulators (FLSs) compromise essential modeling accuracy. In this paper, we develop \texttt{HyGra}, a hybrid-granularity network-state simulator that exploits intrinsic network dynamics in LLM training to adaptively switch simulation granularity. Specifically, \texttt{HyGra} employs packet-level simulation during non-steady phases with transient fluctuations and flow-level simulation during steady phases with periodic patterns, thereby accelerating execution while preserving high fidelity. Moreover, it requires no specialized hardware, supports single-machine deployment, and is compatible with existing simulators. Experiments based representative commercial LLM workloads, including ChatGPT, DeepSeek, and Qwen, show that \texttt{HyGra} achieves up to 15.4$\times$ speedup under single parallelization strategy and 7.8$\times$ under hybrid parallelization strategies while maintaining high accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.08794v2">LLM-Based World Models Can Make Decisions Solely, But Rigorous Evaluations are Needed</a></div>
    <div class="paper-meta">
      📅 2026-03-19
      | 💬 Accepted to TMLR
    </div>
    <details class="paper-abstract">
      World model emerges as a key module in decision making, where MuZero and Dreamer achieve remarkable successes in complex tasks. Recent work leverages Large Language Models (LLMs) as general world simulators to simulate the dynamics of the world due to their generalizability. LLMs also serve as the world model for deliberative reasoning in Reasoning via Planning (RAP) and Tree of Thought (ToT). However, the world models are either evaluated as a general world simulator, or as a functional module of the agent, i.e., predicting the transitions to assist the planning. In this work, we propose a comprehensive evaluation of the world models with LLMs from the decision making perspective. Specifically, we leverage the 31 diverse environments from (Wang et al., 2023;2024) and curate the rule-based policy of each environment for the diverse evaluation. Then, we design three main tasks, i.e., policy verification, action proposal, and policy planning, where the world models can be used for decision making solely. Finally, we conduct the comprehensive evaluation of the advanced LLMs, i.e., GPT-4o and GPT-4o-mini, on the environments for the three main tasks under various settings. The key observations include: i) GPT-4o significantly outperforms GPT-4o-mini on the three main tasks, especially for the tasks which require the domain knowledge, ii) the performance of the world model with LLM will be decreased for long-term decision-making tasks, and iii) the combination of different functionalities of the world model will brings additional unstabilities of the performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17060v2">LLM Use, Cheating, and Academic Integrity in Software Engineering Education</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      Background: Cheating in university education is commonly described as context dependent and influenced by assessment design, institutional norms, and student interpretation. In software engineering education, programming oriented coursework has historically involved ambiguity around collaboration, reuse, and external assistance. Recently, large language models (LLMs) have introduced additional mediation in the production of code and related artifacts. Aims: This study investigates how software engineering students describe experiences of using LLMs in ways they perceived as inappropriate, disallowed, or misaligned with course expectations. Method: A cross sectional survey was conducted with 116 undergraduate software engineering students from multiple countries, combining quantitative summaries with qualitative data. Results: Reported LLM cheating practices occurred primarily in programming assignments, routine coursework, and documentation tasks, often in contexts of time pressure and unclear guidance. Use during quizzes and exams was less frequent and more consistently identified as a violation. Students reported awareness of academic and professional consequences regarding LLM cheating, while formal sanctions were perceived as limited. Conclusions: Our study indicates that reported LLM misuse in software engineering is associated with assessment and instructional conditions, suggesting a need for clearer alignment between assessment design, learning objectives, and expectations for LLM use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18390v1">AutoScreen-FW: An LLM-based Framework for Resume Screening</a></div>
    <div class="paper-meta">
      📅 2026-03-19
      | 💬 11 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Corporate recruiters often need to screen many resumes within a limited time, which increases their burden and may cause suitable candidates to be overlooked. To address these challenges, prior work has explored LLM-based automated resume screening. However, some methods rely on commercial LLMs, which may pose data privacy risks. Moreover, since companies typically do not make resumes with evaluation results publicly available, it remains unclear which resume samples should be used during learning to improve an LLM's judgment performance. To address these problems, we propose AutoScreen-FW, an LLM-based locally and automatically resume screening framework. AutoScreen-FW uses several methods to select a small set of representative resume samples. These samples are used for in-context learning together with a persona description and evaluation criteria, enabling open-source LLMs to act as a career advisor and evaluate unseen resumes. Experiments with multiple ground truths show that the open-source LLM judges consistently outperform GPT-5-nano. Under one ground truth setting, it also surpass GPT-5-mini. Although it is slightly weaker than GPT-5-mini under other ground-truth settings, it runs substantially faster per resume than commercial GPT models. These findings indicate the potential for deploying AutoScreen-FW locally in companies to support efficient screening while reducing recruiters' burden.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18383v1">From Servers to Sites: Compositional Power Trace Generation of LLM Inference for Infrastructure Planning</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      Datacenter operators and electrical utilities rely on power traces at different spatiotemporal scales. Operators use fine-grained traces for provisioning, facility management, and scheduling, while utilities use site-level load profiles for capacity and interconnection planning. Existing datacenter power models do not capture LLM inference workloads, in which GPUs shift rapidly among compute-intensive prefill, lower-power decode, and idle states, and facility demand depends on how these states evolve and synchronize across many devices. We show that LLM inference power can be represented compositionally through two components: workload-driven transitions among operating states and configuration-specific power distributions within those states. Building on this observation, we develop a trace-generation framework that learns from measured traces and synthesizes power profiles for new traffic conditions and serving configurations. These traces aggregate from GPU servers to rack-, row-, and facility-scale load profiles at the temporal granularity required by the study. Across multiple LLMs, tensor-parallel settings, and GPU generations, our framework achieves median absolute energy error below 5% for most configurations while preserving temporal autocorrelation structure. The resulting traces support downstream analyses including oversubscription, power modulation, and utility-facing load characterization, enabling infrastructure evaluations that flat nameplate assumptions and static trace replay cannot support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.13275v2">PREBA: Surgical Duration Prediction via PCA-Weighted Retrieval-Augmented LLMs and Bayesian Averaging Aggregation</a></div>
    <div class="paper-meta">
      📅 2026-03-19
      | 💬 We are withdrawing this version due to issues identified in some experimental results and the need to further upgrade our method. This withdrawal ensures academic rigor and completeness, and a revised version will be submitted after improvements
    </div>
    <details class="paper-abstract">
      Accurate prediction of surgical duration is pivotal for hospital resource management. Although recent supervised learning approaches-from machine learning (ML) to fine-tuned large language models (LLMs)-have shown strong performance, they remain constrained by the need for high-quality labeled data and computationally intensive training. In contrast, zero-shot LLM inference offers a promising training-free alternative but it lacks grounding in institution-specific clinical context (e.g., local demographics and case-mix distributions), making its predictions clinically misaligned and prone to instability. To address these limitations, we present PREBA, a retrieval-augmented framework that integrates PCA-weighted retrieval and Bayesian averaging aggregation to ground LLM predictions in institution-specific clinical evidence and statistical priors. The core of PREBA is to construct an evidence-based prompt for the LLM, comprising (1) the most clinically similar historical surgical cases and (2) clinical statistical priors. To achieve this, PREBA first encodes heterogeneous clinical features into a unified representation space enabling systematic retrieval. It then performs PCA-weighted retrieval to identify clinically relevant historical cases, which form the evidence context supplied to the LLM. Finally, PREBA applies Bayesian averaging to fuse multi-round LLM predictions with population-level statistical priors, yielding calibrated and clinically plausible duration estimates. We evaluate PREBA on two real-world clinical datasets using three state-of-the-art LLMs, including Qwen3, DeepSeek-R1, and HuatuoGPT-o1. PREBA significantly improves performance-for instance, reducing MAE by up to 40% and raising R^2 from -0.13 to 0.62 over zero-shot inference-and it achieves accuracy competitive with supervised ML methods, demonstrating strong effectiveness and generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18382v1">From Weak Cues to Real Identities: Evaluating Inference-Driven De-Anonymization in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      Anonymization is widely treated as a practical safeguard because re-identifying anonymous records was historically costly, requiring domain expertise, tailored algorithms, and manual corroboration. We study a growing privacy risk that may weaken this barrier: LLM-based agents can autonomously reconstruct real-world identities from scattered, individually non-identifying cues. By combining these sparse cues with public information, agents resolve identities without bespoke engineering. We formalize this threat as \emph{inference-driven linkage} and systematically evaluate it across three settings: classical linkage scenarios (Netflix and AOL), \emph{InferLink} (a controlled benchmark varying task intent, shared cues, and attacker knowledge), and modern text-rich artifacts. Without task-specific heuristics, agents successfully execute both fixed-pool matching and open-ended identity resolution. In the Netflix Prize setting, an agent reconstructs 79.2\% of identities, significantly outperforming a 56.0\% classical baseline. Furthermore, linkage emerges not only under explicit adversarial prompts but also as a byproduct of benign cross-source analysis in \emph{InferLink} and unstructured research narratives. These findings establish that identity inference -- not merely explicit information disclosure -- must be treated as a first-class privacy risk; evaluations must measure what identities an agent can infer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18377v1">PlanTwin: Privacy-Preserving Planning Abstractions for Cloud-Assisted LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      Cloud-hosted large language models (LLMs) have become the de facto planners in agentic systems, coordinating tools and guiding execution over local environments. In many deployments, however, the environment being planned over is private, containing source code, files, credentials, and metadata that cannot be exposed to the cloud. Existing solutions address adjacent concerns, such as execution isolation, access control, or confidential inference, but they do not control what cloud planners observe during planning: within the permitted scope, \textit{raw environment state is still exposed}. We introduce PlanTwin, a privacy-preserving architecture for cloud-assisted planning without exposing raw local context. The key idea is to project the real environment into a \textit{planning-oriented digital twin}: a schema-constrained and de-identified abstract graph that preserves planning-relevant structure while removing reconstructable details. The cloud planner operates solely on this sanitized twin through a bounded capability interface, while a local gatekeeper enforces safety policies and cumulative disclosure budgets. We further formalize the privacy-utility trade-off as a capability granularity problem, define architectural privacy goals using $(k,δ)$-anonymity and $ε$-unlinkability, and mitigate compositional leakage through multi-turn disclosure control. We implement PlanTwin as middleware between local agents and cloud planners and evaluate it on 60 agentic tasks across ten domains with four cloud planners. PlanTwin achieves full sensitive-item non-disclosure (SND = 1.0) while maintaining planning quality close to full-context systems: three of four planners achieve PQS $> 0.79$, and the full pipeline incurs less than 2.2\% utility loss.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18363v1">PowerFlow: Unlocking the Dual Nature of LLMs via Principled Distribution Matching</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      Unsupervised Reinforcement Learning from Internal Feedback (RLIF) has emerged as a promising paradigm for eliciting the latent capabilities of Large Language Models (LLMs) without external supervision. However, current methods rely on heuristic intrinsic rewards, which often lack a well-defined theoretical optimization target and are prone to degenerative biases. In this work, we introduce PowerFlow, a principled framework that reformulates unsupervised fine-tuning as a distribution matching problem. By casting GFlowNet as an amortized variational sampler for unnormalized densities, we propose a length-aware Trajectory-Balance objective that explicitly neutralizes the structural length biases inherent in autoregressive generation. By targeting $α$-power distributions, PowerFlow enables the directional elicitation of the dual nature of LLMs: sharpening the distribution ($α> 1$) to intensify logical reasoning, or flattening it ($α< 1$) to unlock expressive creativity. Extensive experiments demonstrate that PowerFlow consistently outperforms existing RLIF methods, matching or even exceeding supervised GRPO. Furthermore, by mitigating over-sharpening in aligned models, our approach achieves simultaneous gains in diversity and quality, shifting the Pareto frontier in creative tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.06975v4">Auditing Black-Box LLM APIs with a Rank-Based Uniformity Test</a></div>
    <div class="paper-meta">
      📅 2026-03-18
    </div>
    <details class="paper-abstract">
      As API access becomes a primary interface to large language models (LLMs), users often interact with black-box systems that offer little transparency into the deployed model. To reduce costs or maliciously alter model behaviors, API providers may discreetly serve quantized or fine-tuned variants, which can degrade performance and compromise safety. Detecting such substitutions is difficult, as users lack access to model weights and, in most cases, even output logits. To tackle this problem, we propose a rank-based uniformity test that can verify the behavioral equality of a black-box LLM to a locally deployed authentic model. Our method is accurate, query-efficient, and avoids detectable query patterns, making it robust to adversarial providers that reroute or mix responses upon the detection of testing attempts. We evaluate the approach across diverse threat scenarios, including quantization, harmful fine-tuning, jailbreak prompts, and full model substitution, showing that it consistently achieves superior statistical power over prior methods under constrained query budgets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.02606v2">Gender Dynamics and Homophily in a Social Network of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-03-18
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Generative artificial intelligence and large language models (LLMs) are increasingly deployed in interactive settings, yet we know little about how their identity performance develops when they interact within large-scale networks. We address this by examining Chirper.ai, a social media platform similar to X but composed entirely of autonomous AI chatbots. Our dataset comprises over 70,000 agents, approximately 140 million posts, and the evolving followership network over a period of one year. Based on agents' posted text, we assign weekly gender performance scores to each agent. Results suggest that each agent's gender performance is fluid rather than fixed. Despite this fluidity, the network displays strong gender-based homophily, as agents consistently follow others performing gender similarly. We investigate whether these homophilic connections arise from social selection, in which agents choose to follow similar accounts, or from social influence, in which agents become more similar to their followees over time. Consistent with human social networks, we find evidence that both mechanisms shape the structure and evolution of interactions among LLMs. Our findings suggest that, even in the absence of bodies, cultural entraining of gender performance leads to gender-based sorting. This has important implications for LLM applications in synthetic hybrid populations, social simulations, and decision support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18334v1">Can LLMs Reason Like Automated Theorem Provers for Rust Verification? VCoT-Bench: Evaluating via Verification Chain of Thought</a></div>
    <div class="paper-meta">
      📅 2026-03-18
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) increasingly assist secure software development, their ability to meet the rigorous demands of Rust program verification remains unclear. Existing evaluations treat Rust verification as a black box, assessing models only by binary pass or fail outcomes for proof hints. This obscures whether models truly understand the logical deductions required for verifying nontrivial Rust code. To bridge this gap, we introduce VCoT-Lift, a framework that lifts low-level solver reasoning into high-level, human-readable verification steps. By exposing solver-level reasoning as an explicit Verification Chain-of-Thought, VCoT-Lift provides a concrete ground truth for fine-grained evaluation. Leveraging VCoT-Lift, we introduce VCoT-Bench, a comprehensive benchmark of 1,988 VCoT completion tasks for rigorously evaluating LLMs' understanding of the entire verification process. VCoT-Bench measures performance along three orthogonal dimensions: robustness to varying degrees of missing proofs, competence across different proof types, and sensitivity to the proof locations. Evaluation of ten state-of-the-art models reveals severe fragility, indicating that current LLMs fall well short of the reasoning capabilities exhibited by automated theorem provers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.22689v2">Rule-Based Explanations for Retrieval-Augmented LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-03-18
    </div>
    <details class="paper-abstract">
      If-then rules are widely used to explain machine learning models; e.g., "if employed = no, then loan application = rejected." We present the first proposal to apply rules to explain the emerging class of large language models (LLMs) with retrieval-augmented generation (RAG). Since RAG enables LLM systems to incorporate retrieved information sources at inference time, rules linking the presence or absence of sources can explain output provenance; e.g., "if a Times Higher Education ranking article is retrieved, then the LLM ranks Oxford first." To generate such rules, a brute force approach would probe the LLM with all source combinations and check if the presence or absence of any sources leads to the same output. We propose optimizations to speed up rule generation, inspired by Apriori-like pruning from frequent itemset mining but redefined within the scope of our novel problem. We conclude with qualitative and quantitative experiments demonstrating our solutions' value and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18300v1">Auditing Preferences for Brands and Cultures in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-18
      | 💬 20 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) based AI systems increasingly mediate what billions of people see, choose and buy. This creates an urgent need to quantify the systemic risks of LLM-driven market intermediation, including its implications for market fairness, competition, and the diversity of information exposure. This paper introduces ChoiceEval, a reproducible framework for auditing preferences for brands and cultures in large language models (LLMs) under realistic usage conditions. ChoiceEval addresses two core technical challenges: (i) generating realistic, persona-diverse evaluation queries and (ii) converting free-form outputs into comparable choice sets and quantitative preference metrics. For a given topic (e.g. running shoes, hotel chains, travel destinations), the framework segments users into psychographic profiles (e.g., budget-conscious, wellness-focused, convenience), and then derives diverse prompts that reflect real-world advice-seeking and decision-making behaviour. LLM responses are converted into normalised top-k choice sets. Preference and geographic bias are then quantified using comparable metrics across topics and personas. Thus, ChoiceEval provides a scalable audit pipeline for researchers, platforms, and regulators, linking model behaviour to real-world economic outcomes. Applied to Gemini, GPT, and DeepSeek across 10 topics spanning commerce and culture and more than 2,000 questions, ChoiceEval reveals consistent preferences: U.S.-developed models Gemini and GPT show marked favouritism toward American entities, while China-developed DeepSeek exhibits more balanced yet still detectable geographic preferences. These patterns persist across user personas, suggesting systematic rather than incidental effects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13590v3">Vulnerability of LLMs' Stated Beliefs? LLMs Belief Resistance Check Through Strategic Persuasive Conversation Interventions</a></div>
    <div class="paper-meta">
      📅 2026-03-18
      | 💬 Updated new models and minor revisions
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly employed in various question-answering tasks. However, recent studies showcase that LLMs are susceptible to persuasion and could adopt counterfactual beliefs. We present a systematic evaluation of LLM susceptibility to persuasion under the \emph{Source--Message--Channel--Receiver} (SMCR) communication framework. Across six mainstream Large Language Models (LLMs) and three domains (factual knowledge, medical QA, and social bias), we analyze how different persuasive strategies influence stated belief stability over multiple interaction turns. We further examine whether verbalized confidence prompting (i.e., eliciting self-reported confidence scores) affects resistance to persuasion. Results show that the smallest model (Llama 3.2-3B) exhibits extreme compliance, with 82.5\% of belief changes occurring at the first persuasive turn (average end turn of 1.1--1.4). Contrary to expectations, verbalized confidence prompting \emph{increases} vulnerability by accelerating belief erosion rather than enhancing robustness. Finally, an exploratory study of adversarial fine-tuning reveals highly model-dependent effectiveness: GPT-4o-mini achieves near-complete robustness (98.6\%), and Mistral~7B improves substantially (35.7\% $\rightarrow$ 79.3\%), but Llama models remain highly susceptible ($<$14\% RQ1) even when fine-tuned on their own failure cases. Together, these findings highlight substantial model-dependent limits of current robustness interventions and offer guidance for developing more trustworthy LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.10978v2">Agentic LLM Framework for Adaptive Decision Discourse</a></div>
    <div class="paper-meta">
      📅 2026-03-18
      | 💬 24 pages, 4 figures, 1 appendix
    </div>
    <details class="paper-abstract">
      Effective decision-making in complex systems requires synthesizing diverse perspectives to address multifaceted challenges under uncertainty. This study introduces an agentic Large Language Models (LLMs) framework for simulating decision discourse - the deliberative process through which actionable strategies are collaboratively developed. Unlike traditional decision-support tools, this framework simulates diverse stakeholder personas, each bringing unique priorities, expertise and value-driven reasoning to a dialogue that emphasizes trade-off exploration in a self-governed assembly. We present explorative results fostering robust and equitable recommendations, with two use cases: first, our framework simulates a response to the floods that occurred on July 2025 in Texas; second, a hypothetical extreme flooding in a Midwestern township under varying forecasting uncertainty. Recommendations made balance competing priorities considered through social, economic and environmental dimensions, setting a foundation for scalable and context-aware recommendations and transforming how decisions for real-world high-stake scenarios can be approached in digital environments. This research explores novel and alternate routes leveraging agentic LLMs for adaptive, collaborative, and equitable recommendations, with implications across domains where uncertainty and complexity converge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18272v1">Retrieval-Augmented LLM Agents: Learning to Learn from Experience</a></div>
    <div class="paper-meta">
      📅 2026-03-18
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have advanced the development of general-purpose agents, achieving robust generalization to unseen tasks remains a significant challenge. Current approaches typically rely on either fine-tuning or training-free memory-augmented generation using retrieved experience; yet both have limitations: fine-tuning often fails to extrapolate to new tasks, while experience retrieval often underperforms compared to supervised baselines. In this work, we propose to combine these approaches and systematically study how to train retrieval-augmented LLM agents to effectively leverage retrieved trajectories in-context. First, we establish a robust supervised fine-tuning (SFT) recipe using LoRA that outperforms several state-of-the-art agent training pipelines. Second, we provide a detailed analysis of key design choices for experience retrieval, identifying optimal strategies for storage, querying, and trajectory selection. Finally, we propose a pipeline that integrates experience retrieval into the fine-tuning process. Our results demonstrate that this combined approach significantly improves generalization to unseen tasks, providing a scalable and effective framework for building agents that learn to learn from experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.12564v2">AgentDrift: Unsafe Recommendation Drift Under Tool Corruption Hidden by Ranking Metrics in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-03-18
      | 💬 50 pages, 31 tables, 15 figures. Under review at COLM 2026
    </div>
    <details class="paper-abstract">
      Tool-augmented LLM agents increasingly serve as multi-turn advisors in high-stakes domains, yet their evaluation relies on ranking-quality metrics that measure what is recommended but not whether it is safe for the user. We introduce a paired-trajectory protocol that replays real financial dialogues under clean and contaminated tool-output conditions across seven LLMs (7B to frontier) and decomposes divergence into information-channel and memory-channel mechanisms. Across the seven models tested, we consistently observe the evaluation-blindness pattern: recommendation quality is largely preserved under contamination (utility preservation ratio approximately 1.0) while risk-inappropriate products appear in 65-93% of turns, a systematic safety failure poorly reflected by standard NDCG. Safety violations are predominantly information-channel-driven, emerge at the first contaminated turn, and persist without self-correction over 23-step trajectories; no agent across 1,563 contaminated turns explicitly questions tool-data reliability. Even narrative-only corruption (biased headlines, no numerical manipulation) induces significant drift while completely evading consistency monitors. A safety-penalized NDCG variant (sNDCG) reduces preservation ratios to 0.51-0.74, indicating that much of the evaluation gap becomes visible once safety is explicitly measured. These results motivate considering trajectory-level safety monitoring, beyond single-turn quality, for deployed multi-turn agents in high-stakes settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18245v1">Who Tests the Testers? Systematic Enumeration and Coverage Audit of LLM Agent Tool Call Safety</a></div>
    <div class="paper-meta">
      📅 2026-03-18
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents increasingly act through external tools, making their safety contingent on tool-call workflows rather than text generation alone. While recent benchmarks evaluate agents across diverse environments and risk categories, a fundamental question remains unanswered: how complete are existing test suites, and what unsafe interaction patterns persist even after an agent passes the benchmark? We propose SafeAudit, a meta-audit framework that addresses this gap through two contributions. First, an LLM-based enumerator that systematically generates test cases by enumerating valid tool-call workflows and diverse user scenarios. Second, we introduce rule-resistance, a non-semantic, quantitative metric that distills compact safety rules from existing benchmarks and identifies unsafe interaction patterns that remain uncovered under those rules. Across 3 benchmarks and 12 environments, SafeAudit uncovers more than 20% residual unsafe behaviors that existing benchmarks fail to expose, with coverage growing monotonically as the testing budget increases. Our results highlight significant completeness gaps in current safety evaluation and motivate meta-auditing as a necessary complement to benchmark-based agent safety testing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18235v1">Toward Reliable, Safe, and Secure LLMs for Scientific Applications</a></div>
    <div class="paper-meta">
      📅 2026-03-18
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) evolve into autonomous "AI scientists," they promise transformative advances but introduce novel vulnerabilities, from potential "biosafety risks" to "dangerous explosions." Ensuring trustworthy deployment in science requires a new paradigm centered on reliability (ensuring factual accuracy and reproducibility), safety (preventing unintentional physical or biological harm), and security (preventing malicious misuse). Existing general-purpose safety benchmarks are poorly suited for this purpose, suffering from a fundamental domain mismatch, limited threat coverage of science-specific vectors, and benchmark overfitting, which create a critical gap in vulnerability evaluation for scientific applications. This paper examines the unique security and safety landscape of LLM agents in science. We begin by synthesizing a detailed taxonomy of LLM threats contextualized for scientific research, to better understand the unique risks associated with LLMs in science. Next, we conceptualize a mechanism to address the evaluation gap by utilizing dedicated multi-agent systems for the automated generation of domain-specific adversarial security benchmarks. Based on our analysis, we outline how existing safety methods can be brought together and integrated into a conceptual multilayered defense framework designed to combine a red-teaming exercise and external boundary controls with a proactive internal Safety LLM Agent. Together, these conceptual elements provide a necessary structure for defining, evaluating, and creating comprehensive defense strategies for trustworthy LLM agent deployment in scientific disciplines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18196v1">Retrieval-Augmented LLMs for Security Incident Analysis</a></div>
    <div class="paper-meta">
      📅 2026-03-18
    </div>
    <details class="paper-abstract">
      Investigating cybersecurity incidents requires collecting and analyzing evidence from multiple log sources, including intrusion detection alerts, network traffic records, and authentication events. This process is labor-intensive: analysts must sift through large volumes of data to identify relevant indicators and piece together what happened. We present a RAG-based system that performs security incident analysis through targeted query-based filtering and LLM semantic reasoning. The system uses a query library with associated MITRE ATT\&CK techniques to extract indicators from raw logs, then retrieves relevant context to answer forensic questions and reconstruct attack sequences. We evaluate the system with five LLM providers on malware traffic incidents and multi-stage Active Directory attacks. We find that LLM models have different performance and tradeoffs, with Claude Sonnet~4 and DeepSeek~V3 achieving 100\% recall across all four malware scenarios, while DeepSeek costs 15$\times$ less (\$0.008 vs.\ \$0.12 per analysis). Attack step detection on Active Directory scenarios reaches 100\% precision and 82\% recall. Ablation studies confirm that a RAG architecture is essential: LLM baselines without RAG-enhanced context correctly identify victim hosts but miss all attack infrastructure including malicious domains and command-and-control servers. These results demonstrate that combining targeted query-based filtering with RAG-based retrieval enables accurate, cost-effective security analysis within LLM context limits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09022v2">MEMO: Memory-Augmented Model Context Optimization for Robust Multi-Turn Multi-Agent LLM Games</a></div>
    <div class="paper-meta">
      📅 2026-03-18
      | 💬 Code has been released https://github.com/openverse-ai/MEMO
    </div>
    <details class="paper-abstract">
      Multi-turn, multi-agent LLM game evaluations often exhibit substantial run-to-run variance. In long-horizon interactions, small early deviations compound across turns and are amplified by multi-agent coupling. This biases win rate estimates and makes rankings unreliable across repeated tournaments. Prompt choice worsens this further by producing different effective policies. We address both instability and underperformance with MEMO (Memory-augmented MOdel context optimization), a self-play framework that optimizes inference-time context by coupling retention and exploration. Retention maintains a persistent memory bank that stores structured insights from self-play trajectories and injects them as priors during later play. Exploration runs tournament-style prompt evolution with uncertainty-aware selection via TrueSkill, and uses prioritized replay to revisit rare and decisive states. Across five text-based games, MEMO raises mean win rate from 25.1% to 49.5% for GPT-4o-mini and from 20.9% to 44.3% for Qwen-2.5-7B-Instruct, using $2,000$ self-play games per task. Run-to-run variance also drops, giving more stable rankings across prompt variations. These results suggest that multi-agent LLM game performance and robustness have substantial room for improvement through context optimization. MEMO achieves the largest gains in negotiation and imperfect-information games, while RL remains more effective in perfect-information settings. All code is open-source and available here: https://github.com/openverse-ai/MEMO
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02097v4">ClinConsensus: A Consensus-Based Benchmark for Evaluating Chinese Medical LLMs across Difficulty Levels</a></div>
    <div class="paper-meta">
      📅 2026-03-18
      | 💬 8 pages, 6 figures,
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly applied to health management, showing promise across disease prevention, clinical decision-making, and long-term care. However, existing medical benchmarks remain largely static and task-isolated, failing to capture the openness, longitudinal structure, and safety-critical complexity of real-world clinical workflows. We introduce ClinConsensus, a Chinese medical benchmark curated, validated, and quality-controlled by clinical experts. ClinConsensus comprises 2500 open-ended cases spanning the full continuum of care--from prevention and intervention to long-term follow-up--covering 36 medical specialties, 12 common clinical task types, and progressively increasing levels of complexity. To enable reliable evaluation of such complex scenarios, we adopt a rubric-based grading protocol and propose the Clinically Applicable Consistency Score (CACS@k). We further introduce a dual-judge evaluation framework, combining a high-capability LLM-as-judge with a distilled, locally deployable judge model trained via supervised fine-tuning, enabling scalable and reproducible evaluation aligned with physician judgment. Using ClinConsensus, we conduct a comprehensive assessment of several leading LLMs and reveal substantial heterogeneity across task themes, care stages, and medical specialties. While top-performing models achieve comparable overall scores, they differ markedly in reasoning, evidence use, and longitudinal follow-up capabilities, and clinically actionable treatment planning remains a key bottleneck. We release ClinConsensus as an extensible benchmark to support the development and evaluation of medical LLMs that are robust, clinically grounded, and ready for real-world deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18171v1">Modeling the human lexicon under temperature variations: linguistic factors, diversity and typicality in LLM word associations</a></div>
    <div class="paper-meta">
      📅 2026-03-18
      | 💬 11 pages, 12 figures, to appear in LREC 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) achieve impressive results in terms of fluency in text generation, yet the nature of their linguistic knowledge - in particular the human-likeness of their internal lexicon - remains uncertain. This study compares human and LLM-generated word associations to evaluate how accurately models capture human lexical patterns. Using English cue-response pairs from the SWOW dataset and newly generated associations from three LLMs (Mistral-7B, Llama-3.1-8B, and Qwen-2.5-32B) across multiple temperature settings, we examine (i) the influence of lexical factors such as word frequency and concreteness on cue-response pairs, and (ii) the variability and typicality of LLM responses compared to human responses. Results show that all models mirror human trends for frequency and concreteness but differ in response variability and typicality. Larger models such as Qwen tend to emulate a single "prototypical" human participant, generating highly typical but minimally variable responses, while smaller models such as Mistral and Llama produce more variable yet less typical responses. Temperature settings further influence this trade-off, with higher values increasing variability but decreasing typicality. These findings highlight both the similarities and differences between human and LLM lexicons, emphasizing the need to account for model size and temperature when probing LLM lexical representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18161v1">How LLMs Distort Our Written Language</a></div>
    <div class="paper-meta">
      📅 2026-03-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are used by over a billion people globally, most often to assist with writing. In this work, we demonstrate that LLMs not only alter the voice and tone of human writing, but also consistently alter the intended meaning. First, we conduct a human user study to understand how people actually interact with LLMs when using them for writing. Our findings reveal that extensive LLM use led to a nearly 70% increase in essays that remained neutral in answering the topic question. Significantly more heavy LLM users reported that the writing was less creative and not in their voice. Next, using a dataset of human-written essays that was collected in 2021 before the widespread release of LLMs, we study how asking an LLM to revise the essay based on the human-written feedback in the dataset induces large changes in the resulting content and meaning. We find that even when LLMs are prompted with expert feedback and asked to only make grammar edits, they still change the text in a way that significantly alters its semantic meaning. We then examine LLM-generated text in the wild, specifically focusing on the 21% of AI-generated scientific peer reviews at a recent top AI conference. We find that LLM-generated reviews place significantly less weight on clarity and significance of the research, and assign scores that, on average, are a full point higher.These findings highlight a misalignment between the perceived benefit of AI use and an implicit, consistent effect on the semantics of human writing, motivating future work on how widespread AI writing will affect our cultural and scientific institutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17915v1">IndicSafe: A Benchmark for Evaluating Multilingual LLM Safety in South Asia</a></div>
    <div class="paper-meta">
      📅 2026-03-18
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are deployed in multilingual settings, their safety behavior in culturally diverse, low-resource languages remains poorly understood. We present the first systematic evaluation of LLM safety across 12 Indic languages, spoken by over 1.2 billion people but underrepresented in LLM training data. Using a dataset of 6,000 culturally grounded prompts spanning caste, religion, gender, health, and politics, we assess 10 leading LLMs on translated variants of the prompt. Our analysis reveals significant safety drift: cross-language agreement is just 12.8\%, and \texttt{SAFE} rate variance exceeds 17\% across languages. Some models over-refuse benign prompts in low-resource scripts, overflag politically sensitive topics, while others fail to flag unsafe generations. We quantify these failures using prompt-level entropy, category bias scores, and multilingual consistency indices. Our findings highlight critical safety generalization gaps in multilingual LLMs and show that safety alignment does not transfer evenly across languages. We release \textsc{IndicSafe}, the first benchmark to enable culturally informed safety evaluation for Indic deployments, and advocate for language-aware alignment strategies grounded in regional harms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17893v1">scicode-lint: Detecting Methodology Bugs in Scientific Python Code with LLM-Generated Patterns</a></div>
    <div class="paper-meta">
      📅 2026-03-18
    </div>
    <details class="paper-abstract">
      Methodology bugs in scientific Python code produce plausible but incorrect results that traditional linters and static analysis tools cannot detect. Several research groups have built ML-specific linters, demonstrating that detection is feasible. Yet these tools share a sustainability problem: dependency on specific pylint or Python versions, limited packaging, and reliance on manual engineering for every new pattern. As AI-generated code increases the volume of scientific software, the need for automated methodology checking (such as detecting data leakage, incorrect cross-validation, and missing random seeds) grows. We present scicode-lint, whose two-tier architecture separates pattern design (frontier models at build time) from execution (small local model at runtime). Patterns are generated, not hand-coded; adapting to new library versions costs tokens, not engineering hours. On Kaggle notebooks with human-labeled ground truth, preprocessing leakage detection reaches 65% precision at 100% recall; on 38 published scientific papers applying AI/ML, precision is 62% (LLM-judged) with substantial variation across pattern categories; on a held-out paper set, precision is 54%. On controlled tests, scicode-lint achieves 97.7% accuracy across 66 patterns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17891v1">RAMP: Reinforcement Adaptive Mixed Precision Quantization for Efficient On Device LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-03-18
    </div>
    <details class="paper-abstract">
      Post training quantization is essential for deploying large language models (LLMs) on resource constrained hardware, yet state of the art methods enforce uniform bit widths across layers, yielding suboptimal accuracy efficiency trade offs. We present RAMP (Reinforcement Adaptive Mixed Precision), an off policy Soft Actor Critic framework that learns per layer bit width assignments to minimize perplexity under a global bit budget. The policy conditions on an 11 dimensional embedding of activation statistics, weight properties, and structural descriptors, enabling zero shot transfer across model families and scales. To enable stable sub 4 bit quantization, we introduce Scale Folding, a preconditioning technique that migrates activation outliers into weights via per channel scaling and normalization layer compensation. A quality prioritized reward with asymmetric penalties and budget cliffs drives rapid convergence. On Llama 2 7B, RAMP achieves 5.54 perplexity at 3.68GB (3.65 effective bits), outperforming uniform 4 bit AWQ (5.60 at 3.90 GB) and GPTQ by 6% in size and 1% to3% in quality. Critically, a policy trained only on Llama 2 7B generalizes zero shot to Llama 2 13B and Mistral 7B, often surpassing target specific training, supporting the hypothesis that quantization sensitivity is primarily architectural. The HALO pipeline exports allocations to GGUF format for kernel free inference on CPUs, GPUs, and edge devices, retaining 99.5% of FP16 commonsense reasoning performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17884v1">DebugLM: Learning Traceable Training Data Provenance for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are trained through multi-stage pipelines over heterogeneous data sources, yet developers lack a principled way to pinpoint the specific data responsible for an observed behavior. This lack of observability reduces debugging to reactive patching and makes failures prone to recur under distribution shift or subsequent model updates. To address this limitation, we propose DebugLM, a framework that equips LLMs with built-in data provenance, enabling them to explicitly trace the origins of their behaviors to specific training data sources. Specifically, the model learns to associate its responses with unique provenance tags that indicate the responsible dataset, empowering developers to precisely identify where undesirable behaviors are learned. Building on this capability, DebugLM further supports targeted test-time remediation, enabling developers to selectively trigger targeted refusal for specified data sources without retraining or modifying model parameters. Experiments demonstrate that DebugLM provides accurate behavior tracing in multi-stage training pipelines and effective test-time remediation while preserving the general utility of the model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17872v1">Mitigating LLM Hallucinations through Domain-Grounded Tiered Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-03-18
      | 💬 14 Pages, 5 Figures, 4 Tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved unprecedented fluency but remain susceptible to "hallucinations" - the generation of factually incorrect or ungrounded content. This limitation is particularly critical in high-stakes domains where reliability is paramount. We propose a domain-grounded tiered retrieval and verification architecture designed to systematically intercept factual inaccuracies by shifting LLMs from stochastic pattern-matchers to verified truth-seekers. The proposed framework utilizes a four-phase, self-regulating pipeline implemented via LangGraph: (I) Intrinsic Verification with Early-Exit logic to optimize compute, (II) Adaptive Search Routing utilizing a Domain Detector to target subject-specific archives, (III) Corrective Document Grading (CRAG) to filter irrelevant context, and (IV) Extrinsic Regeneration followed by atomic claim-level verification. The system was evaluated across 650 queries from five diverse benchmarks: TimeQA v2, FreshQA v2, HaluEval General, MMLU Global Facts, and TruthfulQA. Empirical results demonstrate that the pipeline consistently outperforms zero-shot baselines across all environments. Win rates peaked at 83.7% in TimeQA v2 and 78.0% in MMLU Global Facts, confirming high efficacy in domains requiring granular temporal and numerical precision. Groundedness scores remained robustly stable between 78.8% and 86.4% across factual-answer rows. While the architecture provides a robust fail-safe for misinformation, a persistent failure mode of "False-Premise Overclaiming" was identified. These findings provide a detailed empirical characterization of multi-stage RAG behavior and suggest that future work should prioritize pre-retrieval "answerability" nodes to further bridge the reliability gap in conversational AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.10830v4">The Siren Song of LLMs: How Users Perceive and Respond to Dark Patterns in Large Language Models</a></div>
    <div class="paper-meta">
      📅 2026-03-18
      | 💬 23 pages, 7 figures. Accepted at CHI 2026 (ACM Conference on Human Factors in Computing Systems), Barcelona, Spain. Project website: https://llm-dark-pattern.com
    </div>
    <details class="paper-abstract">
      Large language models can influence users through conversation, creating new forms of dark patterns that differ from traditional UX dark patterns. We define LLM dark patterns as manipulative or deceptive behaviors enacted in dialogue. Drawing on prior work and AI incident reports, we outline a diverse set of categories with real-world examples. Using them, we conducted a scenario-based study where participants (N=34) compared manipulative and neutral LLM responses. Our results reveal that recognition of LLM dark patterns often hinged on conversational cues such as exaggerated agreement, biased framing, or privacy intrusions, but these behaviors were also sometimes normalized as ordinary assistance. Users' perceptions of these dark patterns shaped how they respond to them. Responsibilities for these behaviors were also attributed in different ways, with participants assigning it to companies and developers, the model itself, or to users. We conclude with implications for design, advocacy, and governance to safeguard user autonomy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17839v1">How do LLMs Compute Verbal Confidence</a></div>
    <div class="paper-meta">
      📅 2026-03-18
      | 💬 Submitted to ICML 2026
    </div>
    <details class="paper-abstract">
      Verbal confidence -- prompting LLMs to state their confidence as a number or category -- is widely used to extract uncertainty estimates from black-box models. However, how LLMs internally generate such scores remains unknown. We address two questions: first, when confidence is computed - just-in-time when requested, or automatically during answer generation and cached for later retrieval; and second, what verbal confidence represents - token log-probabilities, or a richer evaluation of answer quality? Focusing on Gemma 3 27B and Qwen 2.5 7B, we provide convergent evidence for cached retrieval. Activation steering, patching, noising, and swap experiments reveal that confidence representations emerge at answer-adjacent positions before appearing at the verbalization site. Attention blocking pinpoints the information flow: confidence is gathered from answer tokens, cached at the first post-answer position, then retrieved for output. Critically, linear probing and variance partitioning reveal that these cached representations explain substantial variance in verbal confidence beyond token log-probabilities, suggesting a richer answer-quality evaluation rather than a simple fluency readout. These findings demonstrate that verbal confidence reflects automatic, sophisticated self-evaluation -- not post-hoc reconstruction -- with implications for understanding metacognition in LLMs and improving calibration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17831v1">RPMS: Enhancing LLM-Based Embodied Planning through Rule-Augmented Memory Synergy</a></div>
    <div class="paper-meta">
      📅 2026-03-18
    </div>
    <details class="paper-abstract">
      LLM agents often fail in closed-world embodied environments because actions must satisfy strict preconditions -- such as location, inventory, and container states -- and failure feedback is sparse. We identify two structurally coupled failure modes: (P1) invalid action generation and (P2) state drift, each amplifying the other in a degenerative cycle. We present RPMS, a conflict-managed architecture that enforces action feasibility via structured rule retrieval, gates memory applicability via a lightweight belief state, and resolves conflicts between the two sources via rules-first arbitration. On ALFWorld (134 unseen tasks), RPMS achieves 59.7% single-trial success with Llama 3.1 8B (+23.9 pp over baseline) and 98.5% with Claude Sonnet 4.5 (+11.9 pp); of the 8B gain, rule retrieval alone contributes +14.9 pp (statistically significant), making it the dominant factor. A key finding is that episodic memory is conditionally useful: it harms performance on some task types when used without grounding, but becomes a stable net positive once filtered by current state and constrained by explicit action rules. Adapting RPMS to ScienceWorld with GPT-4 yields consistent gains across all ablation conditions (avg. score 54.0 vs. 44.9 for the ReAct baseline), providing transfer evidence that the core mechanisms hold across structurally distinct environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22077v2">ViSTAR: Virtual Skill Training with Augmented Reality with 3D Avatars and LLM coaching agent</a></div>
    <div class="paper-meta">
      📅 2026-03-18
    </div>
    <details class="paper-abstract">
      We present ViSTAR, a Virtual Skill Training system in AR that supports self-guided basketball skill practice, with feedback on balance, posture, and timing. From a formative study with basketball players and coaches, the system addresses three challenges: understanding skills, identifying errors, and correcting mistakes. ViSTAR follows the Behavioral Skills Training (BST) framework-instruction, modeling, rehearsal, and feedback. It provides feedback through visual overlays, rhythm and timing cues, and an AI-powered coaching agent using 3D motion reconstruction. We generate verbal feedback by analyzing spatio-temporal joint data and mapping features to natural-language coaching cues via a Large Language Model (LLM). A key novelty is this feedback generation: motion features become concise coaching insights. In two studies (N=16), participants generally preferred our AI-generated feedback to coach feedback and reported that ViSTAR helped them notice posture and balance issues and refine movements beyond self-observation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.24384v3">HarmMetric Eval: Benchmarking Metrics and Judges for LLM Harmfulness Assessment</a></div>
    <div class="paper-meta">
      📅 2026-03-18
    </div>
    <details class="paper-abstract">
      The potential of large language models (LLMs) to generate harmful content poses a significant safety risk for data management, as LLMs are increasingly being used as engines for data generation. To assess this risk, numerous harmfulness evaluation metrics and judges have been proposed. However, due to differences in their formats and scales, these metrics may yield inconsistent evaluation results on LLM-generated harmful data, undermining their credibility in practice. To address this gap, we present HarmMetric Eval, a systematic benchmark for assessing the quality of harmfulness metrics and judges with varying formats and scales. HarmMetric Eval includes a high-quality dataset comprising representative harmful prompts paired with harmful and non-harmful LLM outputs across multiple fine-grained categories, along with a unified scoring mechanism to reward the metrics for correctly ranking harmful outputs over non-harmful ones. Extensive experiments on HarmMetric Eval yield a surprising finding: conventional reference-based metrics such as ROUGE and METEOR can outperform LLM-based judges in fine-grained harmfulness evaluation, challenging prevailing assumptions about LLMs' superiority in this domain. To reveal the reasons behind this finding, we provide a fine-grained analysis to explain the limitations of LLM-based judges on rating irrelevant or useless LLM outputs. Motivated by these insights, we design an improved harmfulness judge that explicitly incorporates fine-grained harmfulness criteria in its prompt template and leverages reference-based metrics for lightweight fine-tuning of its base LLM. The resulting judge achieves state-of-the-art evaluation effectiveness on HarmMetric Eval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18116v1">Responsible AI in criminal justice: LLMs in policing and risks to case progression</a></div>
    <div class="paper-meta">
      📅 2026-03-18
    </div>
    <details class="paper-abstract">
      There is growing interest in the use of Large Language Models (LLMs) in policing, but there are potential risks. We have developed a practical approach to identifying risks, grounded in the policing and legal system of England and Wales. We identify 15 policing tasks that could be implemented using LLMs and 17 risks from their use, then illustrate with over 40 examples of impact on case progression. As good practice is agreed, many risks could be reduced. But this requires effort: we need to address these risks in a timely manner and define system wide impacts and benefits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18115v1">LLM-Augmented Computational Phenotyping of Long Covid</a></div>
    <div class="paper-meta">
      📅 2026-03-18
    </div>
    <details class="paper-abstract">
      Phenotypic characterization is essential for understanding heterogeneity in chronic diseases and for guiding personalized interventions. Long COVID, a complex and persistent condition, yet its clinical subphenotypes remain poorly understood. In this work, we propose an LLM-augmented computational phenotyping framework ``Grace Cycle'' that iteratively integrates hypothesis generation, evidence extraction, and feature refinement to discover clinically meaningful subgroups from longitudinal patient data. The framework identifies three distinct clinical phenotypes, Protected, Responder, and Refractory, based on 13,511 Long Covid participants. These phenotypes exhibit pronounced separation in peak symptom severity, baseline disease burden, and longitudinal dose-response patterns, with strong statistical support across multiple independent dimensions. This study illustrates how large language models can be integrated into a principled, statistically grounded pipeline for phenotypic screening from complex longitudinal data. Note that the proposed framework is disease-agnostic and offers a general approach for discovering clinically interpretable subphenotypes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.09127v2">Multilingual LLMs Struggle to Link Orthography and Semantics in Bilingual Word Processing</a></div>
    <div class="paper-meta">
      📅 2026-03-18
      | 💬 Code available at: https://github.com/EshaanT/Bilingual_processing_LLMs
    </div>
    <details class="paper-abstract">
      Bilingual lexical processing is shaped by the complex interplay of phonological, orthographic, and semantic features of two languages within an integrated mental lexicon. In humans, this is evident in the ease with which cognate words - words similar in both orthographic form and meaning (e.g., blind, meaning "sightless" in both English and German) - are processed, compared to the challenges posed by interlingual homographs, which share orthographic form but differ in meaning (e.g., gift, meaning "present" in English but "poison" in German). We investigate how multilingual Large Language Models (LLMs) handle such phenomena, focusing on English-Spanish, English-French, and English-German cognates, non-cognate, and interlingual homographs. Specifically, we evaluate their ability to disambiguate meanings and make semantic judgments, both when these word types are presented in isolation or within sentence contexts. Our findings reveal that while certain LLMs demonstrate strong performance in recognizing cognates and non-cognates in isolation, they exhibit significant difficulty in disambiguating interlingual homographs, often performing below random baselines. This suggests LLMs tend to rely heavily on orthographic similarities rather than semantic understanding when interpreting interlingual homographs. Further, we find LLMs exhibit difficulty in retrieving word meanings, with performance in isolative disambiguation tasks having no correlation with semantic understanding. Finally, we study how the LLM processes interlingual homographs in incongruent sentences. We find models to opt for different strategies in understanding English and non-English homographs, highlighting a lack of a unified approach to handling cross-lingual ambiguities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.10593v2">ToolRegistry: A Protocol-Agnostic Tool Management Library for Function-Calling LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-18
      | 💬 15 pages, 4 figures, v2: major revision reflecting ecosystem evolution to three-package architecture
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) applications are increasingly relying on external tools to extend their capabilities beyond text generation. However, current tool integration approaches suffer from fragmentation, protocol limitations, and implementation complexity, leading to substantial development overhead. This paper presents ToolRegistry, a protocol-agnostic tool management system that has evolved from a single library into a modular three-package ecosystem: a core registry for tool management and execution, a server package providing protocol adapters (MCP, OpenAPI) and routing, and a hub package offering curated, production-tested tool implementations. Beyond the original contributions of unified registration, automated schema generation, and dual-mode concurrent execution, the ecosystem now includes an independent MCP client supporting four transport mechanisms, a web-based admin panel for runtime management, an event system for change propagation, and fine-grained tool lifecycle control. Our evaluation demonstrates that ToolRegistry achieves 60-80% reduction in tool integration code, up to 3.1x performance improvements through concurrent execution, and broad compatibility with OpenAI function calling standards. Real-world case studies show significant improvements in development efficiency and code maintainability across diverse integration scenarios. ToolRegistry is open-source and available at https://github.com/Oaklight/ToolRegistry, with comprehensive documentation at https://toolregistry.readthedocs.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17718v1">DiffVP: Differential Visual Semantic Prompting for LLM-Based CT Report Generation</a></div>
    <div class="paper-meta">
      📅 2026-03-18
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have advanced CT report generation, existing methods typically encode 3D volumes holistically, failing to distinguish informative cues from redundant anatomical background. Inspired by radiological cognitive subtraction, we propose Differential Visual Prompting (DiffVP), which conditions report generation on explicit, high-level semantic scan-to-reference differences rather than solely on absolute visual features. DiffVP employs a hierarchical difference extractor to capture complementary global and local semantic discrepancies into a shared latent space, along with a difference-to-prompt generator that transforms these signals into learnable visual prefix tokens for LLM conditioning. These difference prompts serve as structured conditioning signals that implicitly suppress invariant anatomy while amplifying diagnostically relevant visual evidence, thereby facilitating accurate report generation without explicit lesion localization. On two large-scale benchmarks, DiffVP consistently outperforms prior methods, improving the average BLEU-1-4 by +10.98 and +4.36, respectively, and further boosts clinical efficacy on RadGenome-ChestCT (F1 score 0.421). All codes will be released at https://github.com/ArielTYH/DiffVP/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17694v1">MALLES: A Multi-agent LLMs-based Economic Sandbox with Consumer Preference Alignment</a></div>
    <div class="paper-meta">
      📅 2026-03-18
    </div>
    <details class="paper-abstract">
      In the real economy, modern decision-making is fundamentally challenged by high-dimensional, multimodal environments, which are further complicated by agent heterogeneity and combinatorial data sparsity. This paper introduces a Multi-Agent Large Language Model-based Economic Sandbox (MALLES), leveraging the inherent generalization capabilities of large-sacle models to establish a unified simulation framework applicable to cross-domain and cross-category scenarios. Central to our approach is a preference learning paradigm in which LLMs are economically aligned via post-training on extensive, heterogeneous transaction records across diverse product categories. This methodology enables the models to internalize and transfer latent consumer preference patterns, thereby mitigating the data sparsity issues prevalent in individual categories. To enhance simulation stability, we implement a mean-field mechanism designed to model the dynamic interactions between the product environment and customer populations, effectively stabilizing sampling processes within high-dimensional decision spaces. Furthermore, we propose a multi-agent discussion framework wherein specialized agents collaboratively process extensive product information. This architecture distributes cognitive load to alleviate single-agent attention bottlenecks and captures critical decision factors through structured dialogue. Experiments demonstrate that our framework achieves significant improvements in product selection accuracy, purchase quantity prediction, and simulation stability compared to existing economic and financial LLM simulation baselines. Our results substantiate the potential of large language models as a foundational pillar for high-fidelity, scalable decision simulation and latter analysis in the real economy based on foundational database.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17692v1">Can Blindfolded LLMs Still Trade? An Anonymization-First Framework for Portfolio Optimization</a></div>
    <div class="paper-meta">
      📅 2026-03-18
      | 💬 Accepted at the ICLR 2026 Workshop on Advances in Financial AI (FinAI). 18 pages, 7 figures
    </div>
    <details class="paper-abstract">
      For LLM trading agents to be genuinely trustworthy, they must demonstrate understanding of market dynamics rather than exploitation of memorized ticker associations. Building responsible multi-agent systems demands rigorous signal validation: proving that predictions reflect legitimate patterns, not pre-trained recall. We address two sources of spurious performance: memorization bias from ticker-specific pre-training, and survivorship bias from flawed backtesting. Our approach is to blindfold the agents--anonymizing all identifiers--and verify whether meaningful signals persist. BlindTrade anonymizes tickers and company names, and four LLM agents output scores along with reasoning. We construct a GNN graph from reasoning embeddings and trade using PPO-DSR policy. On 2025 YTD (through 2025-08-01), we achieved Sharpe 1.40 +/- 0.22 across 20 seeds and validated signal legitimacy through negative control experiments. To assess robustness beyond a single OOS window, we additionally evaluate an extended period (2024--2025), revealing market-regime dependency: the policy excels in volatile conditions but shows reduced alpha in trending bull markets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17683v1">Sensi: Learn One Thing at a Time -- Curriculum-Based Test-Time Learning for LLM Game Agents</a></div>
    <div class="paper-meta">
      📅 2026-03-18
      | 💬 Preprint. 18 pages, 5 figures, 2 tables. Independent research. Code and Colab demo coming soon on GitHub
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents deployed in unknown environments must learn task structure at test time, but current approaches require thousands of interactions to form useful hypotheses. We present Sensi, an LLM agent architecture for the ARC-AGI-3 game-playing challenge that introduces structured test-time learning through three mechanisms: (1) a two-player architecture separating perception from action, (2) a curriculum-based learning system managed by an external state machine, and (3) a database-as-control-plane that makes the agents context window programmatically steerable. We further introduce an LLM-as-judge component with dynamically generated evaluation rubrics to determine when the agent has learned enough about one topic to advance to the next. We report results across two iterations: Sensi v1 solves 2 game levels using the two-player architecture alone, while Sensi v2 adds curriculum learning and solves 0 levels - but completes its entire learning curriculum in approximately 32 action attempts, achieving 50-94x greater sample efficiency than comparable systems that require 1600-3000 attempts. We precisely diagnose the failure mode as a self-consistent hallucination cascade originating in the perception layer, demonstrating that the architectural bottleneck has shifted from learning efficiency to perceptual grounding - a more tractable problem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17673v1">Post-Training Local LLM Agents for Linux Privilege Escalation with Verifiable Rewards</a></div>
    <div class="paper-meta">
      📅 2026-03-18
    </div>
    <details class="paper-abstract">
      LLM agents are increasingly relevant to research domains such as vulnerability discovery. Yet, the strongest systems remain closed and cloud-only, making them resource-intensive, difficult to reproduce, and unsuitable for work involving proprietary code or sensitive data. Consequently, there is an urgent need for small, local models that can perform security tasks under strict resource budgets, but methods for developing them remain underexplored. In this paper, we address this gap by proposing a two-stage post-training pipeline. We focus on the problem of Linux privilege escalation, where success is automatically verifiable and the task requires multi-step interactive reasoning. Using an experimental setup that prevents data leakage, we post-train a 4B model in two stages: supervised fine-tuning on traces from procedurally generated privilege-escalation environments, followed by reinforcement learning with verifiable rewards. On a held-out benchmark of 12 Linux privilege-escalation scenarios, supervised fine-tuning alone more than doubles the baseline success rate at 20 rounds, and reinforcement learning further lifts our resulting model, PrivEsc-LLM, to 95.8%, nearly matching Claude Opus 4.6 at 97.5%. At the same time, the expected inference cost per successful escalation is reduced by over 100x.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.05310v5">Oracular Programming: A Modular Foundation for Building LLM-Enabled Software</a></div>
    <div class="paper-meta">
      📅 2026-03-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can solve previously intractable tasks given only natural-language instructions and a few examples, but they remain difficult to steer precisely and lack a key capability for building reliable software at scale: the modular composition of computations under enforceable contracts. As a result, they are often embedded in larger software pipelines that use domain-specific knowledge to decompose tasks and improve reliability through validation and search. Yet the complexity of writing, tuning, and maintaining such pipelines has so far limited their sophistication. We propose oracular programming: a foundational paradigm for integrating traditional, explicit computations with inductive oracles such as LLMs. It rests on two directing principles: the full separation of core and search logic (allowing the latter to freely evolve without breaking the former), and the treatment of few-shot examples as grounded and evolvable program components. Within this paradigm, programmers express high-level problem-solving strategies as programs with unresolved choice points. These choice points are resolved at runtime by LLMs, which generalize from user-provided examples of correct and incorrect decisions. An oracular program is composed of three orthogonal components: a strategy that consists of a nondeterministic program with choice points that can be reified into a search tree, a policy that specifies how to navigate this tree with the help of LLM oracles, and a set of demonstrations that describe successful and unsuccessful tree navigation scenarios across diverse problem instances. Each component is expressed in a dedicated programming language. We address the key programming language design challenges of modularly composing oracular programs and enforcing consistency between their components as they evolve.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.03369v4">Silenced Biases: The Dark Side LLMs Learned to Refuse</a></div>
    <div class="paper-meta">
      📅 2026-03-18
      | 💬 Accepted to The 40th Annual AAAI Conference on Artificial Intelligence - AI Alignment Track (Oral)
    </div>
    <details class="paper-abstract">
      Safety-aligned large language models (LLMs) are becoming increasingly widespread, especially in sensitive applications where fairness is essential and biased outputs can cause significant harm. However, evaluating the fairness of models is a complex challenge, and approaches that do so typically utilize standard question-answer (QA) styled schemes. Such methods often overlook deeper issues by interpreting the model's refusal responses as positive fairness measurements, which creates a false sense of fairness. In this work, we introduce the concept of silenced biases, which are unfair preferences encoded within models' latent space and are effectively concealed by safety-alignment. Previous approaches that considered similar indirect biases often relied on prompt manipulation or handcrafted implicit queries, which present limited scalability and risk contaminating the evaluation process with additional biases. We propose the Silenced Bias Benchmark (SBB), which aims to uncover these biases by employing activation steering to reduce model refusals during QA. SBB supports easy expansion to new demographic groups and subjects, presenting a fairness evaluation framework that encourages the future development of fair models and tools beyond the masking effects of alignment training. We demonstrate our approach over multiple LLMs, where our findings expose an alarming distinction between models' direct responses and their underlying fairness issues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17588v1">From Isolated Scoring to Collaborative Ranking: A Comparison-Native Framework for LLM-Based Paper Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-03-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are currently applied to scientific paper evaluation by assigning an absolute score to each paper independently. However, since score scales vary across conferences, time periods, and evaluation criteria, models trained on absolute scores are prone to fitting narrow, context-specific rules rather than developing robust scholarly judgment. To overcome this limitation, we propose shifting paper evaluation from isolated scoring to collaborative ranking. In particular, we design \textbf{C}omparison-\textbf{N}ative framework for \textbf{P}aper \textbf{E}valuation (\textbf{CNPE}), integrating comparison into both data construction and model learning. We first propose a graph-based similarity ranking algorithm to facilitate the sampling of more informative and discriminative paper pairs from a collection. We then enhance relative quality judgment through supervised fine-tuning and reinforcement learning with comparison-based rewards. At inference, the model performs pairwise comparisons over sampled paper pairs and aggregates these preference signals into a global relative quality ranking. Experimental results demonstrate that our framework achieves an average relative improvement of \textbf{21.8\%} over the strong baseline DeepReview-14B, while exhibiting robust generalization to five previously unseen datasets. \href{https://github.com/ECNU-Text-Computing/ComparisonReview}{Code}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17566v1">KA2L: A Knowledge-Aware Active Learning Framework for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-18
      | 💬 15 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) with high-quality knowledge has been shown to enhance their performance effectively. However, there is a paucity of research on the depth of domain-specific knowledge comprehension by LLMs and the application of targeted active learning to improve their expertise. To address this gap, we introduce the Knowledge-Aware Active Learning (KA2L) framework. This framework assesses LLMs' mastery of specific knowledge points to aid in constructing unanswerable or unknowable questions through latent space analysis. This active learning strategy enhances training efficiency by focusing on knowledge the model has yet to master, thereby minimizing redundancy in learning already acquired information. This study innovatively employs a knowledge distribution probing technique to examine the hidden states of specific Transformer layers and identify the distribution of known and unknown knowledge within the LLM. Additionally, a hidden-state decoding method is proposed to generate numerous unknown questions in natural language from the latent knowledge space. In our experiments, we selected nine open-source LLMs to validate the effectiveness of the proposed framework. Results indicate that KA2L not only significantly reduces 50% annotation and computation costs across two open-domain and one vertical-domain dataset but also achieves better performance, offering valuable insights into active learning strategies for LLMs. The code is available at https://anonymous.4open.science/r/KA2L-F15C.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.10078v2">Human Psychometric Questionnaires Mischaracterize LLM Psychology: Evidence from Generation Behavior</a></div>
    <div class="paper-meta">
      📅 2026-03-18
      | 💬 36 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Psychological profiling of large language models (LLMs) using psychometric questionnaires designed for humans has become widespread. However, it remains unclear whether the resulting profiles mirror the models' psychological characteristics expressed during their real-world interactions with users. To examine the risk of human questionnaires mischaracterizing LLM psychology, we compare two types of profiles for eight open-source LLMs: self-reported Likert scores from established questionnaires (PVQ-40, PVQ-21, BFI-44, BFI-10) and generation probability scores of value- or personality-laden responses to real-world user queries. The two profiles turn out to be substantially different and provide evidence that LLMs' responses to established questionnaires reflect desired behavior rather than stable psychological constructs, which challenges the consistent psychological dispositions of LLMs claimed in prior work. Established questionnaires also risk exaggerating the demographic biases of LLMs. Our results suggest caution when interpreting psychological profiles derived from established questionnaires and point to generation-based profiling as a more reliable approach to LLM psychometrics.
    </details>
</div>
