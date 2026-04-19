# llm - 2026_04

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3
- [Part 4](papers_4.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08384v1">TASU2: Controllable CTC Simulation for Alignment and Low-Resource Adaptation of Speech LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      Speech LLM post-training increasingly relies on efficient cross-modal alignment and robust low-resource adaptation, yet collecting large-scale audio-text pairs remains costly. Text-only alignment methods such as TASU reduce this burden by simulating CTC posteriors from transcripts, but they provide limited control over uncertainty and error rate, making curriculum design largely heuristic. We propose \textbf{TASU2}, a controllable CTC simulation framework that simulates CTC posterior distributions under a specified WER range, producing text-derived supervision that better matches the acoustic decoding interface. This enables principled post-training curricula that smoothly vary supervision difficulty without TTS. Across multiple source-to-target adaptation settings, TASU2 improves in-domain and out-of-domain recognition over TASU, and consistently outperforms strong baselines including text-only fine-tuning and TTS-based augmentation, while mitigating source-domain performance degradation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08381v1">A GAN and LLM-Driven Data Augmentation Framework for Dynamic Linguistic Pattern Modeling in Chinese Sarcasm Detection</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      Sarcasm is a rhetorical device that expresses criticism or emphasizes characteristics of certain individuals or situations through exaggeration, irony, or comparison. Existing methods for Chinese sarcasm detection are constrained by limited datasets and high construction costs, and they mainly focus on textual features, overlooking user-specific linguistic patterns that shape how opinions and emotions are expressed. This paper proposes a Generative Adversarial Network (GAN) and Large Language Model (LLM)-driven data augmentation framework to dynamically model users' linguistic patterns for enhanced Chinese sarcasm detection. First, we collect raw data from various topics on Sina Weibo. Then, we train a GAN on these data and apply a GPT-3.5 based data augmentation technique to synthesize an extended sarcastic comment dataset, named SinaSarc. This dataset contains target comments, contextual information, and user historical behavior. Finally, we extend the BERT architecture to incorporate multi-dimensional information, particularly user historical behavior, enabling the model to capture dynamic linguistic patterns and uncover implicit sarcastic cues in comments. Experimental results demonstrate the effectiveness of our proposed method. Specifically, our model achieves the highest F1-scores on both the non-sarcastic and sarcastic categories, with values of 0.9138 and 0.9151 respectively, which outperforms all existing state-of-the-art (SOTA) approaches. This study presents a novel framework for dynamically modeling users' long-term linguistic patterns in Chinese sarcasm detection, contributing to both dataset construction and methodological advancement in this field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08369v1">Don't Overthink It: Inter-Rollout Action Agreement as a Free Adaptive-Compute Signal for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      Inference-time compute scaling has emerged as a powerful technique for improving the reliability of large language model (LLM) agents, but existing methods apply compute uniformly: every decision step receives the same budget regardless of its difficulty. We introduce TrACE (Trajectorical Adaptive Compute via agrEement), a training-free controller that allocates LLM calls adaptively across agent timesteps by measuring inter-rollout action agreement. At each step, TrACE samples a small set of candidate next actions and measures how consistently the model commits to the same action. High agreement signals an easy decision; the controller commits immediately. Low agreement signals uncertainty; the controller samples additional rollouts up to a configurable cap before committing to the plurality action. No learned components, no external verifier, and no human labels are required. We evaluate TrACE against greedy decoding and fixed-budget self-consistency (SC-4, SC-8) on two benchmarks spanning single-step reasoning (GSM8K, n=50) and multi-step household navigation (MiniHouse, n=30), using a Qwen 2.5 3B Instruct model running on CPU. TrACE-4 matches SC-4 accuracy while using 33% fewer LLM calls on GSM8K and 39% fewer on MiniHouse. TrACE-8 matches SC-8 accuracy with 55% fewer calls on GSM8K and 65% fewer on MiniHouse. We further show that inter-rollout agreement is a reliable signal of step-level success, validating the core hypothesis that the model's own output consistency encodes difficulty information that can be exploited without training. TrACE is the first training-free, per-timestep adaptive-compute controller for LLM agents to be evaluated on multi-step sequential decision tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11394v2">Stop Listening to Me! How Multi-turn Conversations Can Degrade LLM Diagnostic Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      Patients and clinicians are increasingly using chatbots powered by large language models (LLMs) for healthcare inquiries. While state-of-the-art LLMs exhibit high performance on static diagnostic reasoning benchmarks, their efficacy across multi-turn conversations, which better reflect real-world usage, has been understudied. In this paper, we evaluate 17 LLMs across three clinical datasets to investigate how partitioning the decision-space into multiple simpler turns of conversation influences their diagnostic reasoning. Specifically, we develop a "stick-or-switch" evaluation framework to measure model conviction (i.e., defending a correct diagnosis or safe abstention against incorrect suggestions) and flexibility (i.e., recognizing a correct suggestion when it is introduced) across conversations. Our experiments reveal the conversation tax, where multi-turn interactions consistently degrade performance when compared to single-shot baselines. Notably, models frequently abandon initial correct diagnoses and safe abstentions to align with incorrect user suggestions. Additionally, several models exhibit blind switching, failing to distinguish between signal and incorrect suggestions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.05674v2">Towards Effective Offensive Security LLM Agents: Hyperparameter Tuning, LLM as a Judge, and a Lightweight CTF Benchmark</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      Recent advances in LLM agentic systems have improved the automation of offensive security tasks, particularly for Capture the Flag (CTF) challenges. We systematically investigate the key factors that drive agent success and provide a detailed recipe for building effective LLM-based offensive security agents. First, we present CTFJudge, a framework leveraging LLM as a judge to analyze agent trajectories and provide granular evaluation across CTF solving steps. Second, we propose a novel metric, CTF Competency Index (CCI) for partial correctness, revealing how closely agent solutions align with human-crafted gold standards. Third, we examine how LLM hyperparameters, namely temperature, top-p, and maximum token length, influence agent performance and automated cybersecurity task planning. For rapid evaluation, we present CTFTiny, a curated benchmark of 50 representative CTF challenges across binary exploitation, web, reverse engineering, forensics, and cryptography. Our findings identify optimal multi-agent coordination settings and lay the groundwork for future LLM agent research in cybersecurity. We make CTFTiny open source to public https://github.com/NYU-LLM-CTF/CTFTiny along with CTFJudge on https://github.com/NYU-LLM-CTF/CTFJudge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08326v1">ProMedical: Hierarchical Fine-Grained Criteria Modeling for Medical LLM Alignment via Explicit Injection</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 ACL 2026
    </div>
    <details class="paper-abstract">
      Aligning Large Language Models (LLMs) with high-stakes medical standards remains a significant challenge, primarily due to the dissonance between coarse-grained preference signals and the complex, multi-dimensional nature of clinical protocols. To bridge this gap, we introduce ProMedical, a unified alignment framework grounded in fine-grained clinical criteria. We first construct ProMedical-Preference-50k, a dataset generated via a human-in-the-loop pipeline that augments medical instructions with rigorous, physician-derived rubrics. Leveraging this corpus, we propose the Explicit Criteria Injection paradigm to train a multi-dimensional reward model. Unlike traditional scalar reward models, our approach explicitly disentangles safety constraints from general proficiency, enabling precise guidance during reinforcement learning. To rigorously validate this framework, we establish ProMedical-Bench, a held-out evaluation suite anchored by double-blind expert adjudication. Empirical evaluations demonstrate that optimizing the Qwen3-8B base model via ProMedical-RM-guided GRPO yields substantial gains, improving overall accuracy by 22.3% and safety compliance by 21.7%, effectively rivaling proprietary frontier models. Furthermore, the aligned policy generalizes robustly to external benchmarks, demonstrating performance comparable to state-of-the-art models on UltraMedical. We publicly release our datasets, reward models, and benchmarks to facilitate reproducible research in safety-aware medical alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08605v3">Mina: A Multilingual LLM-Powered Legal Assistant Agent for Bangladesh for Empowering Access to Justice</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 Accepted to ACL 2026 Findings
    </div>
    <details class="paper-abstract">
      Bangladesh's low-income population faces major barriers to affordable legal advice due to complex legal language, procedural opacity, and high costs. Existing AI legal assistants lack Bengali-language support and jurisdiction-specific adaptation, limiting their effectiveness. To address this, we developed Mina, a multilingual LLM-based legal assistant tailored for the Bangladeshi context. It employs multilingual embeddings and a RAG-based chain-of-tools framework for retrieval, reasoning, translation, and document generation, delivering context-aware legal drafts, citations, and plain-language explanations via an interactive chat interface. Evaluated by law faculty from leading Bangladeshi universities across all stages of the 2022 and 2023 Bangladesh Bar Council Exams, Mina scored 75-80% in Preliminary MCQs, Written, and simulated Viva Voce exams, matching or surpassing average human performance and demonstrating clarity, contextual understanding, and sound legal reasoning. Even under a conservative upper bound, Mina operates at just 0.12-0.61% of typical legal consultation costs in Bangladesh, yielding a 99.4-99.9\% cost reduction relative to human-provided services. These results confirm its potential as a low-cost, multilingual AI assistant that automates key legal tasks and scales access to justice, offering a real-world case study on building domain-specific, low-resource systems and addressing challenges of multilingual adaptation, efficiency, and sustainable public-service AI deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06618v2">PoC-Adapt: Semantic-Aware Automated Vulnerability Reproduction with LLM Multi-Agents and Reinforcement Learning-Driven Adaptive Policy</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 16 pages, abstracted and meta updated
    </div>
    <details class="paper-abstract">
      While recent approaches leverage large language models (LLMs) and multi-agent pipelines to automatically generate proof-of-concept (PoC) exploits from vulnerability reports, existing systems often suffer from two fundamental limitations: unreliable validation based on surface-level execution signals and high operational cost caused by extensive trial-and-error during exploit generation. In this paper, we present PoC-Adapt, an end-to-end framework for automated PoC generation and verification, architected upon a foundation semantic runtime validation and adaptive policy learning. At the core of PoC-Adapt is a Semantic Oracle that validates exploits by comparing structured pre- and post-execution system states, enabling reliable distinction between true vulnerability exploitation and incidental behavioral changes. To reduce exploration cost, we further introduce an Adaptive Policy Learning mechanism that learns an exploitation policy over semantic states and actions, guiding the exploit agent toward effective strategies with fewer failed attempts. PoC-Adapt is implemented as a multi-agent system comprising specialized agents for root cause analysis, environment building, exploit generation, and semantic validation, coordinated through structured feedback loops. Experimenting on the CWE-Bench-Java and PrimeVul benchmarks shows that PoC-Adapt significantly improves verification reliability by 25% and reduces exploit generation cost compared to prior LLM-based systems, highlighting the importance of semantic validation and learned action policies in automated vulnerability reproduction. Applied to the latest CVE corpus, PoC-Adapt confirmed 12 verified PoC out of 80 reproduce attempts at a cost of $0.42 per generated exploit
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03249v2">Accordion-Thinking: Self-Regulated Step Summaries for Efficient and Readable LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      Scaling test-time compute via long Chain-of-Thought unlocks remarkable gains in reasoning capabilities, yet it faces practical limits due to the linear growth of KV cache and quadratic attention complexity. In this paper, we introduce Accordion-Thinking, an end-to-end framework where LLMs learn to self-regulate the granularity of the reasoning steps through dynamic summarization. This mechanism enables a Fold inference mode, where the model periodically summarizes its thought process and discards former thoughts to reduce dependency on historical tokens. We apply reinforcement learning to incentivize this capability further, uncovering a critical insight: the accuracy gap between the highly efficient Fold mode and the exhaustive Unfold mode progressively narrows and eventually vanishes over the course of training. This phenomenon demonstrates that the model learns to encode essential reasoning information into compact summaries, achieving effective compression of the reasoning context. Our Accordion-Thinking demonstrates that with learned self-compression, LLMs can tackle complex reasoning tasks with minimal dependency token overhead without compromising solution quality, and it achieves a three times throughput while maintaining accuracy on a 48GB GPU memory configuration, while the structured step summaries provide a human-readable account of the reasoning process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2407.04183v4">Seeing Like an AI: How LLMs Apply (and Misapply) Wikipedia Neutrality Norms</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 Appeared at ICWSM 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are trained on broad corpora and then used in communities with specialized norms. Is providing LLMs with community rules enough for models to follow these norms? We evaluate LLMs' capacity to detect (Task 1) and correct (Task 2) biased Wikipedia edits according to Wikipedia's Neutral Point of View (NPOV) policy. LLMs struggled with bias detection, achieving only 64% accuracy on a balanced dataset. Models exhibited contrasting biases (some under- and others over-predicted bias), suggesting distinct priors about neutrality. LLMs performed better at generation, removing 79% of words removed by Wikipedia editors. However, LLMs made additional changes beyond Wikipedia editors' simpler neutralizations, resulting in high-recall but low-precision editing. Interestingly, crowdworkers rated AI rewrites as more neutral (70%) and fluent (61%) than Wikipedia-editor rewrites. Qualitative analysis found LLMs sometimes applied NPOV more comprehensively than Wikipedia editors but often made extraneous non-NPOV-related changes (such as grammar). LLMs may apply rules in ways that resonate with the public but diverge from community experts. While potentially effective for generation, LLMs may reduce editor agency and increase moderation workload (e.g., verifying additions). Even when rules are easy to articulate, having LLMs apply them like community members may still be difficult.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08266v1">Orion-Lite: Distilling LLM Reasoning into Efficient Vision-Only Driving Models</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      Leveraging the general world knowledge of Large Language Models (LLMs) holds significant promise for improving the ability of autonomous driving systems to handle rare and complex scenarios. While integrating LLMs into Vision-Language-Action (VLA) models has yielded state-of-the-art performance, their massive parameter counts pose severe challenges for latency-sensitive and energy-efficient deployment. Distilling LLM knowledge into a compact driving model offers a compelling solution to retain these reasoning capabilities while maintaining a manageable computational footprint. Although previous works have demonstrated the efficacy of distillation, these efforts have primarily focused on relatively simple scenarios and open-loop evaluations. Therefore, in this work, we investigate LLM distillation in more complex, interactive scenarios under closed-loop evaluation. We demonstrate that through a combination of latent feature distillation and ground-truth trajectory supervision, an efficient vision-only student model \textbf{Orion-Lite} can even surpass the performance of its massive VLA teacher, ORION. Setting a new state-of-the-art on the rigorous Bench2Drive benchmark, with a Driving Score of 80.6. Ultimately, this reveals that vision-only architectures still possess significant, untapped potential for high-performance reactive planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.15949v3">ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 Accepted in ACL 2026 Main Conference
    </div>
    <details class="paper-abstract">
      Large language models show promise for financial decision-making, yet deploying them as autonomous trading agents raises fundamental challenges: how to adapt instructions when rewards arrive late and obscured by market noise, how to synthesize heterogeneous information streams into coherent decisions, and how to bridge the gap between model outputs and executable market actions. We present ATLAS (Adaptive Trading with LLM AgentS), a unified multi-agent framework that integrates structured information from markets, news, and corporate fundamentals to support robust trading decisions. Within ATLAS, the central trading agent operates in an order-aware action space, ensuring that outputs correspond to executable market orders rather than abstract signals. The agent can incorporate feedback while trading using Adaptive-OPRO, a novel prompt-optimization technique that dynamically adapts the prompt by incorporating real-time, stochastic feedback, leading to increasing performance over time. Across regime-specific equity studies and multiple LLM families, Adaptive-OPRO consistently outperforms fixed prompts, while reflection-based feedback fails to provide systematic gains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10198v3">HumanLLM: Benchmarking and Improving LLM Anthropomorphism via Human Cognitive Patterns</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 ACL 2026 main
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in reasoning and generation, serving as the foundation for advanced persona simulation and Role-Playing Language Agents (RPLAs). However, achieving authentic alignment with human cognitive and behavioral patterns remains a critical challenge for these agents. We present HumanLLM, a framework treating psychological patterns as interacting causal forces. We construct 244 patterns from ~12,000 academic papers and synthesize 11,359 scenarios where 2--5 patterns reinforce, conflict, or modulate each other, with multi-turn conversations expressing inner thoughts, actions, and dialogue. Our dual-level checklists evaluate both individual pattern fidelity and emergent multi-pattern dynamics, achieving strong human alignment (r=0.90) while revealing that holistic metrics conflate simulation accuracy with social desirability. HumanLLM-8B outperforms Qwen3-32B on multi-pattern dynamics despite 4x fewer parameters, demonstrating that authentic anthropomorphism requires cognitive modeling -- simulating not just what humans do, but the psychological processes generating those behaviors. Our dataset, code, and model are available at: https://github.com/YJGoodbye2024/HumanLLM.git
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.07900v2">Rethinking the Value of Agent-Generated Tests for LLM-Based Software Engineering Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) code agents increasingly resolve repository-level issues by iteratively editing code, invoking tools, and validating candidate patches. In these workflows, agents often write tests on the fly, but the value of this behavior remains unclear. For example, GPT-5.2 writes almost no new tests yet achieves performance comparable to top-ranking agents.This raises a central question: do such tests meaningfully improve issue resolution, or do they mainly mimic a familiar software-development practice while consuming interaction budget? To better understand the role of agent-written tests, we analyze trajectories produced by six strong LLMs on SWE-bench Verified. Our results show that test writing is common, but resolved and unresolved tasks within the same model exhibit similar test-writing frequencies. When tests are written, they mainly serve as observational feedback channels, with value-revealing print statements appearing much more often than assertion-based checks. Based on these insights, we perform a prompt-intervention study by revising the prompts used with four models to either increase or reduce test writing. The results suggest that prompt-induced changes in the volume of agent-written tests do not significantly change final outcomes in this setting. Taken together, these results suggest that current agent-written testing practices reshape process and cost more than final task outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08224v1">Externalization in LLM Agents: A Unified Review of Memory, Skills, Protocols and Harness Engineering</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 54 pages, tech report on Externalization in LLM Agents
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents are increasingly built less by changing model weights than by reorganizing the runtime around them. Capabilities that earlier systems expected the model to recover internally are now externalized into memory stores, reusable skills, interaction protocols, and the surrounding harness that makes these modules reliable in practice. This paper reviews that shift through the lens of externalization. Drawing on the idea of cognitive artifacts, we argue that agent infrastructure matters not merely because it adds auxiliary components, but because it transforms hard cognitive burdens into forms that the model can solve more reliably. Under this view, memory externalizes state across time, skills externalize procedural expertise, protocols externalize interaction structure, and harness engineering serves as the unification layer that coordinates them into governed execution. We trace a historical progression from weights to context to harness, analyze memory, skills, and protocols as three distinct but coupled forms of externalization, and examine how they interact inside a larger agent system. We further discuss the trade-off between parametric and externalized capability, identify emerging directions such as self-evolving harnesses and shared agent infrastructure, and discuss open challenges in evaluation, governance, and the long-term co-evolution of models and external infrastructure. The result is a systems-level framework for explaining why practical agent progress increasingly depends not only on stronger models, but on better external cognitive infrastructure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.04989v2">BacPrep: Lessons from Deploying an LLM-Based Bacalaureat Assessment Platform</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 First version ACCEPTED at BBGI (ITS 2025 Workshop) Second versions ACCEPTED at ITS 2026
    </div>
    <details class="paper-abstract">
      Accessing quality preparation and feedback for the Romanian Bacalaureat exam is challenging, particularly for students in remote or underserved areas. This paper presents BacPrep, an experimental online platform exploring Large Language Model (LLM) potential for automated assessment, aiming to offer a free, accessible resource. Using official exam questions from the last 5 years, BacPrep employs the latest available Gemini Flash model (currently Gemini 2.5 Flash, via the \texttt{gemini-flash-latest} endpoint) to prioritize user experience quality during the data collection phase, with model versioning to be locked for subsequent rigorous evaluation. The platform has collected over 100 student solutions across Computer Science and Romanian Language exams, enabling preliminary assessment of LLM grading quality. This revealed several significant challenges: grading inconsistency across multiple runs, arithmetic errors when aggregating fractional scores, performance degradation under large prompt contexts, failure to apply subject-specific rubric weightings, and internal inconsistencies between generated scores and qualitative feedback. These findings motivate a redesigned architecture featuring subject-level prompt decomposition, specialized per-subject graders, and a median-selection strategy across multiple runs. Expert validation against human-graded solutions remains the critical next step.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08206v1">"Theater of Mind" for LLMs: A Cognitive Architecture Based on Global Workspace Theory</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      Modern Large Language Models (LLMs) operate fundamentally as Bounded-Input Bounded-Output (BIBO) systems. They remain in a passive state until explicitly prompted, computing localized responses without intrinsic temporal continuity. While effective for isolated tasks, this reactive paradigm presents a critical bottleneck for engineering autonomous artificial intelligence. Current multi-agent frameworks attempt to distribute cognitive load but frequently rely on static memory pools and passive message passing, which inevitably leads to cognitive stagnation and homogeneous deadlocks during extended execution. To address this structural limitation, we propose Global Workspace Agents (GWA), a cognitive architecture inspired by Global Workspace Theory. GWA transitions multi-agent coordination from a passive data structure to an active, event-driven discrete dynamical system. By coupling a central broadcast hub with a heterogeneous swarm of functionally constrained agents, the system maintains a continuous cognitive cycle. Furthermore, we introduce an entropy-based intrinsic drive mechanism that mathematically quantifies semantic diversity, dynamically regulating generation temperature to autonomously break reasoning deadlocks. Coupled with a dual-layer memory bifurcation strategy to ensure long-term cognitive continuity, GWA provides a robust, reproducible engineering framework for sustained, self-directed LLM agency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.04253v2">Rethinking Compute Substrates for 3D-Stacked Near-Memory LLM Decoding: Microarchitecture-Scheduling Co-Design</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      Large language model (LLM) decoding is a major inference bottleneck because its low arithmetic intensity makes performance highly sensitive to memory bandwidth. 3D-stacked near-memory processing (NMP) provides substantially higher local memory bandwidth than conventional off-chip interfaces, making it a promising substrate for decode acceleration. However, our analysis shows that this bandwidth advantage also shifts many decode operators on 3D-stacked NMP back into the compute-bound regime. Under the tight area budget of the logic die, the design of the compute substrate itself therefore becomes a first-order challenge. Therefore, we rethink the compute microarchitecture of prior 3D-stacked NMP designs. First, we replace prior MAC tree-based compute units with a more area-efficient systolic array, and we further observe that decode operators exhibit substantial shape diversity, making reconfigurability in both systolic array shape and dataflow essential for sustaining high utilization. Building on this insight, we continue to exploit two key opportunities: the high local memory bandwidth reduces the need for large on-chip buffers, and the existing vector core, originally designed to handle auxiliary tensor computations, already provides much of the control logic and multi-ported buffering required for fine-grained flexibility for systolic array, allowing us to unify the two structures in a highly area-efficient manner. Based on these insights, we present the first compute microarchitecture tailored to 3D-stacked NMP LLM decoding, explicitly designed to satisfy the joint requirements of low area cost, high-bandwidth operation, and fine-grained reconfigurability. We further propose an multi-core scheduling framework. Compared with Stratum, our design achieves an average 2.91x speedup and 2.40x higher energy efficiency across both dense and MoE models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08140v1">Multimodal Reasoning with LLM for Encrypted Traffic Interpretation: A Benchmark</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 Project page \url{https://github.com/lgzhangzlg/Multimodal-Reasoning-with-LLM-for-Encrypted-Traffic-Interpretation-A-Benchmark}
    </div>
    <details class="paper-abstract">
      Network traffic, as a key media format, is crucial for ensuring security and communications in modern internet infrastructure. While existing methods offer excellent performance, they face two key bottlenecks: (1) They fail to capture multidimensional semantics beyond unimodal sequence patterns. (2) Their black box property, i.e., providing only category labels, lacks an auditable reasoning process. We identify a key factor that existing network traffic datasets are primarily designed for classification and inherently lack rich semantic annotations, failing to generate human-readable evidence report. To address data scarcity, this paper proposes a Byte-Grounded Traffic Description (BGTD) benchmark for the first time, combining raw bytes with structured expert annotations. BGTD provides necessary behavioral features and verifiable chains of evidence for multimodal reasoning towards explainable encrypted traffic interpretation. Built upon BGTD, this paper proposes an end-to-end traffic-language representation framework (mmTraffic), a multimodal reasoning architecture bridging physical traffic encoding and semantic interpretation. In order to alleviate modality interference and generative hallucinations, mmTraffic adopts a jointly-optimized perception-cognition architecture. By incorporating a perception-centered traffic encoder and a cognition-centered LLM generator, mmTraffic achieves refined traffic interpretation with guaranteed category prediction. Extensive experiments demonstrate that mmTraffic autonomously generates high-fidelity, human-readable, and evidence-grounded traffic interpretation reports, while maintaining highly competitive classification accuracy comparing to specialized unimodal model (e.g., NetMamba). The source code is available at https://github.com/lgzhangzlg/Multimodal-Reasoning-with-LLM-for-Encrypted-Traffic-Interpretation-A-Benchmark
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.04641v3">Evaluating LLMs for Demographic-Targeted Social Bias Detection: A Comprehensive Benchmark Study</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 19 pages
    </div>
    <details class="paper-abstract">
      Large-scale web-scraped text corpora used to train general-purpose AI models often contain harmful demographic-targeted social biases, creating a regulatory need for data auditing and developing scalable bias-detection methods. Although prior work has investigated biases in text datasets and related detection methods, these studies remain narrow in scope. They typically focus on a single content type (e.g., hate speech), cover limited demographic axes, overlook biases affecting multiple demographics simultaneously, and analyze limited techniques. Consequently, practitioners lack a holistic understanding of the strengths and limitations of recent large language models (LLMs) for automated bias detection. In this study, we conduct a comprehensive benchmark study on English texts to assess the ability of LLMs in detecting demographic-targeted social biases. To align with regulatory requirements, we frame bias detection as a multi-label task of detecting targeted identities using a demographic-focused taxonomy. We then systematically evaluate models across scales and techniques, including prompting, in-context learning, and fine-tuning. Using twelve datasets spanning diverse content types and demographics, our study demonstrates the promise of fine-tuned smaller models for scalable detection. However, our analyses also expose persistent gaps across demographic axes and multi-demographic targeted biases, underscoring the need for more effective and scalable detection frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08126v1">LLM-Based Data Generation and Clinical Skills Evaluation for Low-Resource French OSCEs</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 11 pages, 2 figures, to be published in LREC 2026 proceedings
    </div>
    <details class="paper-abstract">
      Objective Structured Clinical Examinations (OSCEs) are the standard method for assessing medical students' clinical and communication skills through structured patient interviews. In France, however, the organization of training sessions is limited by human and logistical constraints, restricting students' access to repeated practice and structured feedback. Recent advances in Natural Language Processing (NLP) and Large Language Models (LLMs) now offer the opportunity to automatically evaluate such medical interviews, thereby alleviating the need for human examiners during training. Yet, real French OSCE annotated transcripts remain extremely scarce, limiting reproducible research and reliable benchmarking. To address these challenges, we investigate the use of LLMs for both generating and evaluating French OSCE dialogues in a low-resource context. We introduce a controlled pipeline that produces synthetic doctor-patient interview transcripts guided by scenario-specific evaluation criteria, combining ideal and perturbed performances to simulate varying student skill levels. The resulting dialogues are automatically silver-labeled through an LLM-assisted framework supporting adjustable evaluation strictness. Benchmarking multiple open-source and proprietary LLMs shows that mid-size models ($\le$32B parameters) achieve accuracies comparable to GPT-4o ($\sim$90\%) on synthetic data, highlighting the feasibility of locally deployable, privacy-preserving evaluation systems for medical education.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08118v1">Initialisation Determines the Basin: Efficient Codebook Optimisation for Extreme LLM Quantization</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 9 pages (+ references and appendix). Under review at ACL Rolling Review
    </div>
    <details class="paper-abstract">
      Additive quantization enables extreme LLM compression with O(1) lookup-table dequantization, making it attractive for edge deployment. Yet at 2-bit precision, it often fails catastrophically, even with extensive search and finetuning. We show that the dominant bottleneck is codebook initialisation. Greedy sequential initialisation frequently places the model in poor optimisation regions that subsequent beam search and PV-tuning struggle to overcome. We analyse this behaviour through the representational ratio \r{ho} = N/KM, which characterises the relationship between weight groups and codebook capacity, and propose OA-EM, an output-aware EM initialisation method using Hessian-weighted Mahalanobis distance. Across compression rates, search budgets, and three architectures (Llama 3.2 3B, Llama 3.1 8B, Qwen 2.5 3B), OA-EM consistently produces better solutions after PV-tuning and dominates the quality-compute frontier. The severity of the bottleneck scales with \r{ho}: moderate at 3 bpp but extreme at 2 bpp, where poor initialisation can degrade perplexity by orders of magnitude. More broadly, our results highlight the importance of optimisation geometry in compressed model spaces, where initialisation can dominate subsequent search and fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02070v2">Exploring Plan Space through Conversation: An Agentic Framework for LLM-Mediated Explanations in Planning</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      When automating plan generation for a real-world sequential decision problem, the goal is often not to replace the human planner, but to facilitate an iterative reasoning and elicitation process, where the human's role is to guide the AI planner according to their preferences and expertise. In this context, explanations that respond to users' questions are crucial to improve their understanding of potential solutions and increase their trust in the system. To enable natural interaction with such a system, we present a multi-agent Large Language Model (LLM) architecture that is agnostic to the explanation framework and enables user- and context-dependent interactive explanations. We also describe an instantiation of this framework for goal-conflict explanations, which we use to conduct a user study comparing the LLM-powered interaction with a baseline template-based explanation interface.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08083v1">Can LLMs Deobfuscate Binary Code? A Systematic Analysis of Large Language Models into Pseudocode Deobfuscation</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      Deobfuscating binary code remains a fundamental challenge in reverse engineering, as obfuscation is widely used to hinder analysis and conceal program logic. Although large language models (LLMs) have shown promise in recovering semantics from obfuscated binaries, a systematic evaluation of their effectiveness is still lacking. In this work, we present BinDeObfBench, the first comprehensive benchmark for assessing LLM-based binary deobfuscation across diverse transformations spanning pre-compilation, compile-time, and post-compilation stages. Our evaluation shows that deobfuscation performance depends more on reasoning capability and domain expertise than on model scale, and that task-specific supervised fine-tuning consistently outperforms broad domain pre-training. Reasoning models can maintain robustness under severe obfuscation, generalize across different instruction set architectures (ISAs) and optimization levels. In-context learning benefits standard models but yields limited gains for reasoning models. Overall, our study highlights the importance of task-specific fine-tuning and reasoning-driven strategies, and positions BinDeObfBench as a basis for future work in binary deobfuscation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08075v1">Dual-Pool Token-Budget Routing for Cost-Efficient and Reliable LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      Production vLLM fleets typically provision each instance for the worst-case context length, leading to substantial KV-cache over-allocation and under-utilized concurrency. In practice, 80-95% of requests are short, yet are served under configurations optimized for long contexts, wasting 4-8$\times$ throughput capacity and triggering reliability issues such as OOM crashes, preemption, and request rejections. We identify a common root cause for these inefficiencies: configuration-traffic mismatch. We propose dual-pool token-budget routing, a lightweight dispatch mechanism that partitions a homogeneous fleet into two specialized pools: a high-throughput short-context pool and a high-capacity long-context pool. Each request is routed based on its estimated total token budget, computed using a per-category bytes-to-token ratio that is learned online via exponential moving average from usage.prompt_tokens feedback, eliminating the need for a tokenizer. We also develop a simple analytical model that predicts fleet-level cost savings from workload characteristics and measured throughput differences, enabling practitioners to estimate benefits prior to deployment. Evaluations on real-world traces from the Azure LLM Inference Dataset and LMSYS-Chat-1M, serving Llama-3-70B on A100 GPUs, show that our approach reduces GPU-hours by 31-42%, corresponding to \$2.86M annual savings at fleet scale, while lowering preemption rates by 5.4$\times$ and improving P99 TTFT by 6%. A case study with Qwen3-235B-A22B on AMD MI300X at 10,000 req/s projects \$15.4M in annual savings. The method incurs only O(1) dispatch overhead, adapts automatically to heterogeneous workloads, and composes seamlessly with existing optimizations such as PagedAttention, continuous batching, and prefill-decode disaggregation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.00017v2">ReCellTy: Domain-Specific Knowledge Graph Retrieval-Augmented LLMs Reasoning Workflow for Single-Cell Annotation</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      With the rapid development of large language models (LLMs), their application to cell type annotation has drawn increasing attention. However, general-purpose LLMs often face limitations in this specific task due to the lack of guidance from external domain knowledge. To enable more accurate and fully automated cell type annotation, we develop a globally connected knowledge graph comprising 18850 biological information nodes, including cell types, gene markers, features, and other related entities, along with 48,944 edges connecting these nodes, which is used by LLMs to retrieve entities associated with differential genes for cell reconstruction. Additionally, a multi-task reasoning workflow is designed to optimise the annotation process. Compared to general-purpose LLMs, our method improves human evaluation scores by up to 0.21 and semantic similarity by 6.1% across multiple tissue types, while more closely aligning with the cognitive logic of manual annotation. Meanwhile, it narrows the performance gap between large and small LLMs in cell type annotation, offering a paradigm for structured knowledge integration and reasoning in bioinformatics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07236v2">How Much LLM Does a Self-Revising Agent Actually Need?</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 WIP
    </div>
    <details class="paper-abstract">
      Recent LLM-based agents often place world modeling, planning, and reflection inside a single language model loop. This can produce capable behavior, but it makes a basic scientific question difficult to answer: which part of the agent's competence actually comes from the LLM, and which part comes from explicit structure around it? We study this question not by claiming a general answer, but by making it empirically tractable. We introduce a declared reflective runtime protocol that externalizes agent state, confidence signals, guarded actions, and hypothetical transitions into inspectable runtime structure. We instantiate this protocol in a declarative runtime and evaluate it on noisy Collaborative Battleship [4] using four progressively structured agents over 54 games (18 boards $\times$ 3 seeds). The resulting decomposition isolates four components: posterior belief tracking, explicit world-model planning, symbolic in-episode reflection, and sparse LLM-based revision. Across this decomposition, explicit world-model planning improves substantially over a greedy posterior-following baseline (+24.1pp win rate, +0.017 F1). Symbolic reflection operates as a real runtime mechanism -- with prediction tracking, confidence gating, and guarded revision actions -- even though its current revision presets are not yet net-positive in aggregate. Adding conditional LLM revision at about 4.3\% of turns yields only a small and non-monotonic change: average F1 rises slightly (+0.005) while win rate drops (31$\rightarrow$29 out of 54). These results suggest a methodological contribution rather than a leaderboard claim: externalizing reflection turns otherwise latent agent behavior into inspectable runtime structure, allowing the marginal role of LLM intervention to be studied directly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08044v1">A Full-Stack Performance Evaluation Infrastructure for 3D-DRAM-based LLM Accelerators</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit memory-intensive behavior during decoding, making it a key bottleneck in LLM inference. To accelerate decoding execution, hybrid-bonding-based 3D-DRAM has been adopted in LLM accelerators. While this emerging technology provides strong performance gains over existing hardware, current 3D-DRAM accelerators (3D-Accelerators) rely on closed-source evaluation tools, limiting access to publicly available performance analysis methods. Moreover, existing designs are highly customized for specific scenarios, lacking a general and reusable full-stack modeling for 3D-Accelerators across diverse usecases. To bridge this fundamental gap, we present ATLAS, the first silicon-proven Architectural Three-dimesional-DRAM-based LLM Accelerator Simulation framework. Built on commercially deployed multi-layer 3D-DRAM technology, ATLAS introduces unified abstractions for both 3D-Accelerator system architecture and programming primitives to support arbitrary LLM inference scenarios. Validation against real silicon shows that ATLAS achieves $\le$8.57% simulation error and 97.26-99.96\% correlation with measured performance. Through design space exploration with ATLAS, we demonstrate its ability to guide architecture design and distill key takeaways for both 3D-DRAM memory system and 3D-Accelerator microarchitecture across scenarios. ATLAS will be open-sourced upon publication, enabling further research on 3D-Accelerators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08042v1">3DrawAgent: Teaching LLM to Draw in 3D with Early Contrastive Experience</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 CVPR 2026 Highlight
    </div>
    <details class="paper-abstract">
      Sketching in 3D space enables expressive reasoning about shape, structure, and spatial relationships, yet generating 3D sketches through natural language remains a major challenge. In this work, we introduce 3DrawAgent, a training-free, language-driven framework for 3D sketch generation that leverages large language models (LLMs) to sequentially draw 3D Bezier curves under geometric feedback. Unlike prior 2D sketch agents, our method introduces a relative experience optimization strategy that adapts the recently proposed Group Reward Policy Optimization (GRPO) paradigm. Instead of relying on explicit ground-truth supervision, we construct pairwise comparisons among generated sketches, with each pair consisting of a relatively better and a worse result based on CLIP-based perceptual rewards and LLM-based fine-grained qualitative assessment. These experiences are then used to iteratively refine the prior knowledge of 3D drawing, enabling black-box reinforcement of the model's 3D awareness. This design allows our model to self-improve its spatial understanding and drawing quality without parameter updates. Experiments show that 3DrawAgent can generate complex and coherent 3D Bezier sketches from diverse textual prompts, exhibit emergent geometric reasoning, and generalize to novel shapes, establishing a new paradigm for advancing the field of training-free 3D sketch intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08033v1">IoT-Brain: Grounding LLMs for Semantic-Spatial Sensor Scheduling</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 To appear in ACM MobiCom 2026; 13 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Intelligent systems powered by large-scale sensor networks are shifting from predefined monitoring to intent-driven operation, revealing a critical Semantic-to-Physical Mapping Gap. While large language models (LLMs) excel at semantic understanding, existing perception-centric pipelines operate retrospectively, overlooking the fundamental decision of what to sense and when. We formalize this proactive decision as Semantic-Spatial Sensor Scheduling (S3) and demonstrate that direct LLM planning is unreliable due to inherent gaps in representation, reasoning, and optimization. To bridge these gaps, we introduce the Spatial Trajectory Graph (STG), a neuro-symbolic paradigm governed by a verify-before-commit discipline that transforms open-ended planning into a verifiable graph optimization problem. Based on STG, we implement IoT-Brain, a concrete system embodiment, and construct TopoSense-Bench, a campus-scale benchmark with 5,250 natural-language queries across 2,510 cameras. Evaluations show that IoT-Brain boosts task success rate by 37.6% over the strongest search-intensive methods while running nearly 2 times faster and using 6.6 times fewer prompt tokens. In real-world deployment, it approaches the reliability upper bound while reducing 4.1 times network bandwidth, providing a foundational framework for LLMs to interact with the physical world with unprecedented reliability and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08031v1">Open-Ended Instruction Realization with LLM-Enabled Multi-Planner Scheduling in Autonomous Vehicles</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      Most Human-Machine Interaction (HMI) research overlooks the maneuvering needs of passengers in autonomous driving (AD). Natural language offers an intuitive interface, yet translating passenger open-ended instructions into control signals, without sacrificing interpretability and traceability, remains a challenge. This study proposes an instruction-realization framework that leverages a large language model (LLM) to interpret instructions, generates executable scripts that schedule multiple model predictive control (MPC)-based motion planners based on real-time feedback, and converts planned trajectories into control signals. This scheduling-centric design decouples semantic reasoning from vehicle control at different timescales, establishing a transparent, traceable decision-making chain from high-level instructions to low-level actions. Due to the absence of high-fidelity evaluation tools, this study introduces a benchmark for open-ended instruction realization in a closed-loop setting. Comprehensive experiments reveal that the framework significantly improves task-completion rates over instruction-realization baselines, reduces LLM query costs, achieves safety and compliance on par with specialized AD approaches, and exhibits considerable tolerance to LLM inference latency. For more qualitative illustrations and a clearer understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08016v1">Wiring the 'Why': A Unified Taxonomy and Survey of Abductive Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      Regardless of its foundational role in human discovery and sense-making, abductive reasoning--the inference of the most plausible explanation for an observation--has been relatively underexplored in Large Language Models (LLMs). Despite the rapid advancement of LLMs, the exploration of abductive reasoning and its diverse facets has thus far been disjointed rather than cohesive. This paper presents the first survey of abductive reasoning in LLMs, tracing its trajectory from philosophical foundations to contemporary AI implementations. To address the widespread conceptual confusion and disjointed task definitions prevalent in the field, we establish a unified two-stage definition that formally categorizes prior work. This definition disentangles abduction into \textit{Hypothesis Generation}, where models bridge epistemic gaps to produce candidate explanations, and \textit{Hypothesis Selection}, where the generated candidates are evaluated and the most plausible explanation is chosen. Building upon this foundation, we present a comprehensive taxonomy of the literature, categorizing prior work based on their abductive tasks, datasets, underlying methodologies, and evaluation strategies. In order to ground our framework empirically, we conduct a compact benchmark study of current LLMs on abductive tasks, together with targeted comparative analyses across model sizes, model families, evaluation styles, and the distinct generation-versus-selection task typologies. Moreover, by synthesizing recent empirical results, we examine how LLM performance on abductive reasoning relates to deductive and inductive tasks, providing insights into their broader reasoning capabilities. Our analysis reveals critical gaps in current approaches--from static benchmark design and narrow domain coverage to narrow training frameworks and limited mechanistic understanding of abductive processes...
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08003v1">Rethinking Entropy Allocation in LLM-based ASR: Understanding the Dynamics between Speech Encoders and LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      Integrating large language models (LLMs) into automatic speech recognition (ASR) has become a dominant paradigm. Although recent LLM-based ASR models have shown promising performance on public benchmarks, it remains challenging to balance recognition quality with latency and overhead, while hallucinations further limit real-world deployment. In this study, we revisit LLM-based ASR from an entropy allocation perspective and introduce three metrics to characterize how training paradigms allocate entropy reduction between the speech encoder and the LLM. To remedy entropy-allocation inefficiencies in prevailing approaches, we propose a principled multi-stage training strategy grounded in capability-boundary awareness, optimizing parameter efficiency and hallucination robustness. Specifically, we redesign the pretraining strategy to alleviate the speech-text modality gap, and further introduce an iterative asynchronous SFT stage between alignment and joint SFT to preserve functional decoupling and constrain encoder representation drift. Experiments on Mandarin and English benchmarks show that our method achieves competitive performance with state-of-the-art models using only 2.3B parameters, while also effectively mitigating hallucinations through our decoupling-oriented design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.15494v2">Do AI Models Dream of Faster Code? An Empirical Study on LLM-Proposed Performance Improvements in Real-World Software</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can generate code, but can they generate fast code for complex, real-world software systems? In this study, we investigate this question using a dataset of 65 tasks mined from performance-critical open-source Java projects. Unlike prior studies, which focused on algorithmic puzzles, we conduct experiments on actual performance-sensitive production code and employ developer-written JMH benchmarks to rigorously validate performance gains against human baselines. Our results reveal a nuanced reality -- although LLMs demonstrate a surprisingly high capability to solve these complex engineering problems, their solutions suffer from extreme volatility and still lag behind human developers on average. Consequently, we find that the current benchmarks based on algorithmic tasks yields an overly optimistic assessment of LLM capabilities. We trace this real-world performance gap to two primary limitations: first, LLMs struggle to autonomously pinpoint performance hotspots, and second, even with explicit guidance, they often fall short of synthesizing optimal algorithmic improvements. Our results highlight the need to move beyond static code generation towards more complex agent-based systems that are able to profile and observe runtime behavior for performance improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07981v1">A Decomposition Perspective to Long-context Reasoning for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      Long-context reasoning is essential for complex real-world applications, yet remains a significant challenge for Large Language Models (LLMs). Despite the rapid evolution in long-context reasoning, current research often overlooks the internal complexity of the long-context reasoning task itself. In this paper, we move beyond this holistic view and decompose long-context reasoning into a set of fundamental atomic skills, and we then automatically synthesize a suite of pseudo datasets, each explicitly targeting a specific atomic skill. Our empirical analysis confirms that proficiency in these atomic skills is strongly correlated with general long-text reasoning performance. Building on this insight, we employ reinforcement learning on these pseudo datasets to sharpen the model's atomic skills, in the hope of boosting its general long-context reasoning ability. Extensive experiments across multiple benchmarks demonstrate the effectiveness of our approach: it outperforms a strong baseline by an average margin of 7.7\% (improving from 46.3\% to 54.0\%) across Loogle, Loong, LongBench-v2, BrowscompLong, Ruler-qa2, and MRCR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07955v1">Rethinking Residual Errors in Compensation-based LLM Quantization</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 ICLR'26 camera ready
    </div>
    <details class="paper-abstract">
      Methods based on weight compensation, which iteratively apply quantization and weight compensation to minimize the output error, have recently demonstrated remarkable success in quantizing Large Language Models (LLMs). The representative work, GPTQ, introduces several key techniques that make such iterative methods practical for LLMs with billions of parameters. GPTAQ extends this approach by introducing an asymmetric calibration process that aligns the output of each quantized layer with its full-precision counterpart, incorporating a residual error into the weight compensation framework. In this work, we revisit the formulation of the residual error. We identify a sub-optimal calibration objective in existing methods: during the intra-layer calibration process, they align the quantized output with the output from compensated weights, rather than the true output from the original full-precision model. Therefore, we redefine the objective to precisely align the quantized model's output with the original output of the full-precision model at each step. We then reveal that the residual error originates not only from the output difference of the preceding layer but also from the discrepancy between the compensated and original weights within each layer, which we name the 'compensation-aware error'. By inheriting the neuron decomposition technique from GPTAQ, we can efficiently incorporate this compensation-aware error into the weight update process. Extensive experiments on various LLMs and quantization settings demonstrate that our proposed enhancements integrate seamlessly with both GPTQ and GPTAQ, significantly improving their quantization performance. Our code is publicly available at https://github.com/list0830/ResComp.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07937v1">HCRE: LLM-based Hierarchical Classification for Cross-Document Relation Extraction with a Prediction-then-Verification Strategy</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 ACL 2026 Findings
    </div>
    <details class="paper-abstract">
      Cross-document relation extraction (RE) aims to identify relations between the head and tail entities located in different documents. Existing approaches typically adopt the paradigm of ``\textit{Small Language Model (SLM) + Classifier}''. However, the limited language understanding ability of SLMs hinders further improvement of their performance. In this paper, we conduct a preliminary study to explore the performance of Large Language Models (LLMs) in cross-document RE. Despite their extensive parameters, our findings indicate that LLMs do not consistently surpass existing SLMs. Further analysis suggests that the underperformance is largely attributed to the challenges posed by the numerous predefined relations. To overcome this issue, we propose an LLM-based \underline{H}ierarchical \underline{C}lassification model for cross-document \underline{RE} (HCRE), which consists of two core components: 1) an LLM for relation prediction and 2) a \textit{hierarchical relation tree} derived from the predefined relation set. This tree enables the LLM to perform hierarchical classification, where the target relation is inferred level by level. Since the number of child nodes is much smaller than the size of the entire predefined relation set, the hierarchical relation tree significantly reduces the number of relation options that LLM needs to consider during inference. However, hierarchical classification introduces the risk of error propagation across levels. To mitigate this, we propose a \textit{prediction-then-verification} inference strategy that improves prediction reliability through multi-view verification at each level. Extensive experiments show that HCRE outperforms existing baselines, validating its effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07054v2">Sell More, Play Less: Benchmarking LLM Realistic Selling Skill</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      Sales dialogues require multi-turn, goal-directed persuasion under asymmetric incentives, which makes them a challenging setting for large language models (LLMs). Yet existing dialogue benchmarks rarely measure deal progression and outcomes. We introduce SalesLLM benchmark, a bilingual (ZH/EN) benchmark derived from realistic applications covering Financial Services and Consumer Goods, built from 30,074 scripted configurations and 1,805 curated multi-turn scenarios with controllable difficulty and personas. We propose a fully automatic evaluation pipeline that combines (i) an LLM-based rater for sales-process progress,and (ii) fine-tuned BERT classifiers for end-of-dialogue buying intent. To improve simulation fidelity, we train a user model, CustomerLM, with SFT and DPO on 8,000+ crowdworker-involved sales conversations, reducing role inversion from 17.44% (GPT-4o) to 8.8%. SalesLLM benchmark scores correlate strongly with expert human ratings (Pearson r=0.98). Experiments across 15 mainstream LLMs reveal substantial variability: top-performance LLMs are competitive with human-level performance while the less capable ones are worse than human. SalesLLM benchmark serves as a scalable benchmark for developing and evaluating outcome-oriented sales agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07911v1">Dynamic Attentional Context Scoping: Agent-Triggered Focus Sessions for Isolated Per-Agent Steering in Multi-Agent LLM Orchestration</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 15 pages, 4 figures, preprint
    </div>
    <details class="paper-abstract">
      Multi-agent LLM orchestration systems suffer from context pollution: when N concurrent agents compete for the orchestrator's context window, each agent's task state, partial outputs, and pending questions contaminate the steering interactions of every other agent, degrading decision quality. We introduce Dynamic Attentional Context Scoping (DACS), a mechanism in which the orchestrator operates in two asymmetric modes. In Registry mode it holds only lightweight per-agent status summaries (<=200 tokens each), remaining responsive to all agents and the user. When an agent emits a SteeringRequest, the orchestrator enters Focus(a_i) mode, injecting the full context of agent a_i while compressing all other agents to their registry entries. Context isolation is agent-triggered, asymmetric, and deterministic: the context window contains exactly F(a_i) + R_{-i} during steering, eliminating cross-agent contamination without requiring context compression or retrieval. We evaluate DACS across four experimental phases totalling 200 trials: Phase 1 tests N in {3,5,10} (60 trials); Phase 2 tests agent heterogeneity and adversarial dependencies (60 trials); Phase 3 tests decision density up to D=15 (40 trials); Phase 4 uses autonomous LLM agents for free-form questions (40 trials, Claude Haiku 4.5). Across all 8 synthetic scenarios, DACS achieves 90.0--98.4% steering accuracy versus 21.0--60.0% for a flat-context baseline (p < 0.0001 throughout), with wrong-agent contamination falling from 28--57% to 0--14% and context efficiency ratios of up to 3.53x. The accuracy advantage grows with N and D; keyword matching is validated by LLM-as-judge across all phases (mean kappa=0.909). DACS outperforms the flat-context baseline by +17.2pp at N=3 (p=0.0023) and +20.4pp at N=5 (p=0.0008) in Phase 4, with the advantage growing with N confirmed by two independent judges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07888v1">Bit-by-Bit: Progressive QAT Strategy with Outlier Channel Splitting for Stable Low-Bit LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      Training LLMs at ultra-low precision remains a formidable challenge. Direct low-bit QAT often suffers from convergence instability and substantial training costs, exacerbated by quantization noise from heavy-tailed outlier channels and error accumulation across layers. To address these issues, we present Bit-by-Bit, a progressive QAT framework with outlier channel splitting. Our approach integrates three key components: (1) block-wise progressive training that reduces precision stage by stage, ensuring stable initialization for low-bit optimization; (2) nested structure of integer quantization grids to enable a "train once, deploy any precision" paradigm, allowing a single model to support multiple bit-widths without retraining; (3) rounding-aware outlier channel splitting, which mitigates quantization error while acting as an identity transform that preserves the quantized outputs. Furthermore, we follow microscaling groups with E4M3 scales, capturing dynamic activation ranges in alignment with OCP/NVIDIA standards. To address the lack of efficient 2-bit kernels, we developed custom operators for both W2A2 and W2A16 configurations, achieving up to 11$\times$ speedup over BF16. Under W2A2 settings, Bit-by-Bit significantly outperforms baselines like BitDistiller and EfficientQAT on both Llama2/3, achieving a loss of only 2.25 WikiText2 PPL compared to full-precision models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18474v2">WASD: Locating Critical Neurons as Sufficient Conditions for Explaining and Controlling LLM Behavior</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      Precise behavioral control of large language models (LLMs) is critical for complex applications. However, existing methods often incur high training costs, lack natural language controllability, or compromise semantic coherence. To bridge this gap, we propose WASD (unWeaving Actionable Sufficient Directives), a novel framework that explains model behavior by identifying sufficient neural conditions for token generation. Our method represents candidate conditions as neuron-activation predicates and iteratively searches for a minimal set that guarantees the current output under input perturbations. Experiments on SST-2 and CounterFact with the Gemma-2-2B model demonstrate that our approach produces explanations that are more stable, accurate, and concise than conventional attribution graphs. Moreover, through a case study on controlling cross-lingual output generation, we validated the practical effectiveness of WASD in controlling model behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.13993v2">Chunks as Arms: Multi-Armed Bandit-Guided Sampling for Long-Context LLM Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 17 pages
    </div>
    <details class="paper-abstract">
      Long-context modeling is critical for a wide range of real-world tasks, including long-context question answering, summarization, and complex reasoning tasks. Recent studies have explored fine-tuning Large Language Models (LLMs) with synthetic data to enhance their long-context capabilities. However, the effectiveness of such approaches is often limited by the low diversity and factual inconsistencies in the generated data. To address these challenges, we propose LongMab, a novel framework that leverages a Multi-Armed Bandit (MAB) rollout strategy to identify the most informative chunks from the given long context for sampling high-quality and diverse responses and constructing preference data pairs for Direct Preference Optimization (DPO) training. Specifically, we treat context chunks as arms of MAB, select chunks based on their expected reward scores to input into LLMs to generate responses, and iteratively update these scores based on reward feedback. Both exploration and exploitation during the rollout process enable the LLM to focus on the most relevant context segments, thereby generating and collecting high-quality and diverse responses. Experimental results on both Llama and Qwen show the effectiveness of LongMab by achieving more than a 4% improvement on long-context reasoning benchmarks. All data and code will be released on https://github.com/NEUIR/LongMab-PO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07872v1">PyVRP$^+$: LLM-Driven Metacognitive Heuristic Evolution for Hybrid Genetic Search in Vehicle Routing Problems</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 18 pages, accepted to the 25th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2026)
    </div>
    <details class="paper-abstract">
      Designing high-performing metaheuristics for NP-hard combinatorial optimization problems, such as the Vehicle Routing Problem (VRP), remains a significant challenge, often requiring extensive domain expertise and manual tuning. Recent advances have demonstrated the potential of large language models (LLMs) to automate this process through evolutionary search. However, existing methods are largely reactive, relying on immediate performance feedback to guide what are essentially black-box code mutations. Our work departs from this paradigm by introducing Metacognitive Evolutionary Programming (MEP), a framework that elevates the LLM to a strategic discovery agent. Instead of merely reacting to performance scores, MEP compels the LLM to engage in a structured Reason-Act-Reflect cycle, forcing it to explicitly diagnose failures, formulate design hypotheses, and implement solutions grounded in pre-supplied domain knowledge. By applying MEP to evolve core components of the state-of-the-art Hybrid Genetic Search (HGS) algorithm, we discover novel heuristics that significantly outperform the original baseline. By steering the LLM to reason strategically about the exploration-exploitation trade-off, our approach discovers more effective and efficient heuristics applicable across a wide spectrum of VRP variants. Our results show that MEP discovers heuristics that yield significant performance gains over the original HGS baseline, improving solution quality by up to 2.70\% and reducing runtime by over 45\% on challenging VRP variants.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.12040v2">BTC-LLM: Efficient Sub-1-Bit LLM Quantization via Learnable Transformation and Binary Codebook</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      Binary quantization represents the most extreme form of compression, reducing weights to +/-1 for maximal memory and computational efficiency. While recent sparsity-aware binarization achieves sub-1-bit compression via weight pruning, it faces critical challenges: performance degradation, mask-management overhead, and limited hardware compatibility. In this paper, we present BTC-LLM, a novel sub-1-bit LLM quantization framework that leverages binary pattern clustering and weight transformation to overcome these limitations. Our approach incorporates two key innovations: (1) a Binary Codebook that clusters recurring vectors into compact indices using custom distance metrics and sign-based updates; (2) a Learnable Transformation that reduces outliers and promotes shared sign patterns among binary weights. This eliminates sparse masks, enabling efficient inference on standard hardware. Extensive evaluations across LLaMA, Qwen, and FBI-LLM families demonstrate that BTC-LLM achieves state-of-the-art results in extreme compression (1.11-0.7 bits). Notably, BTC-LLM compressed to 0.8 bits on LLaMA-2-13B maintains high performance, with only a 3.1 percent accuracy drop in zero-shot benchmarks, while delivering a 1.6x speedup over FP16.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07851v1">ReRec: Reasoning-Augmented LLM-based Recommendation Assistant via Reinforcement Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 Accepted by ACL 2026
    </div>
    <details class="paper-abstract">
      With the rise of LLMs, there is an increasing need for intelligent recommendation assistants that can handle complex queries and provide personalized, reasoning-driven recommendations. LLM-based recommenders show potential but face challenges in multi-step reasoning, underscoring the need for reasoning-augmented systems. To address this gap, we propose ReRec, a novel reinforcement fine-tuning (RFT) framework designed to improve LLM reasoning in complex recommendation tasks. Our framework introduces three key components: (1) Dual-Graph Enhanced Reward Shaping, integrating recommendation metrics like NDCG@K with Query Alignment and Preference Alignment Scores to provide fine-grained reward signals for LLM optimization; (2) Reasoning-aware Advantage Estimation, which decomposes LLM outputs into reasoning segments and penalizes incorrect steps to enhance reasoning of recommendation; and (3) Online Curriculum Scheduler, dynamically assess query difficulty and organize training curriculum to ensure stable learning during RFT. Experiments demonstrate that ReRec outperforms state-of-the-art baselines and preserves core abilities like instruction-following and general knowledge. Our codes are available at https://github.com/jiani-huang/ReRec.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.07379v2">DuoServe-MoE: Dual-Phase Expert Prefetch and Caching for LLM Inference QoS Assurance</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed as Internet/Web services (LLM-as-a-Service) with strict latency Service-Level Objectives (SLOs) under tight GPU memory budgets. Mixture-of-Experts (MoE) models improve quality and throughput via sparse expert activation, but serving them efficiently is challenging because expert weights dominate memory footprint and incur costly host--device transfers when offloaded. Moreover, MoE serving exhibits a phase disparity: the prefill phase tends to activate experts densely across many tokens, while the decode phase activates only a few experts per step. A uniform expert loading/caching policy across phases leads to either peak-memory blowup (prefill) or tail-latency inflation (decode). We present DuoServe-MoE, a QoS-oriented MoE serving system that decouples prefill and decode and applies phase-specialized expert scheduling. For prefill, DuoServe-MoE uses a two-stream CUDA pipeline to overlap expert prefetching with non-MoE computation, reducing expert residency time and peak GPU memory. For decode, it employs a lightweight layer-level predictor trained offline from activation traces to prefetch only likely experts without model changes. Experiments on representative MoE LLMs show that DuoServe-MoE improves TTFT by up to $5.34\times$ and end-to-end latency by up to $7.55\times$ over representative baselines, while maintaining low runtime GPU memory usage under resource-constrained deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07834v1">Why Are We Lonely? Leveraging LLMs to Measure and Understand Loneliness in Caregivers and Non-caregivers</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      This paper presents an LLM-driven approach for constructing diverse social media datasets to measure and compare loneliness in the caregiver and non-caregiver populations. We introduce an expert-developed loneliness evaluation framework and an expert-informed typology for categorizing causes of loneliness for analyzing social media text. Using a human-validated data processing pipeline, we apply GPT-4o, GPT-5-nano, and GPT-5 to build a high-quality Reddit corpus and analyze loneliness across both populations. The loneliness evaluation framework achieved average accuracies of 76.09% and 79.78% for caregivers and non-caregivers, respectively. The cause categorization framework achieved micro-aggregate F1 scores of 0.825 and 0.80 for caregivers and non-caregivers, respectively. Across populations, we observe substantial differences in the distribution of types of causes of loneliness. Caregivers' loneliness were predominantly linked to caregiving roles, identity recognition, and feelings of abandonment, indicating distinct loneliness experiences between the two groups. Demographic extraction further demonstrates the viability of Reddit for building a diverse caregiver loneliness dataset. Overall, this work establishes an LLM-based pipeline for creating high quality social media datasets for studying loneliness and demonstrates its effectiveness in analyzing population-level differences in the manifestation of loneliness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07825v1">Filling the Gaps: Selective Knowledge Augmentation for LLM Recommenders</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 SIGIR 2026 Accept
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently emerged as powerful training-free recommenders. However, their knowledge of individual items is inevitably uneven due to imbalanced information exposure during pretraining, a phenomenon we refer to as knowledge gap problem. To address this, most prior methods have employed a naive uniform augmentation that appends external information for every item in the input prompt. However, this approach not only wastes limited context budget on redundant augmentation for well-known items but can also hinder the model's effective reasoning. To this end, we propose KnowSA_CKP (Knowledge-aware Selective Augmentation with Comparative Knowledge Probing) to mitigate the knowledge gap problem. KnowSA_CKP estimates the LLM's internal knowledge by evaluating its capability to capture collaborative relationships and selectively injects additional information only where it is most needed. By avoiding unnecessary augmentation for well-known items, KnowSA_CKP focuses on items that benefit most from knowledge supplementation, thereby making more effective use of the context budget. KnowSA_CKP requires no fine-tuning step, and consistently improves both recommendation accuracy and context efficiency across four real-world datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07821v1">More Capable, Less Cooperative? When LLMs Fail At Zero-Cost Collaboration</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 Accepted at ICLR 2026 Workshop on Agents in the Wild. 24 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents increasingly coordinate in multi-agent systems, yet we lack an understanding of where and why cooperation failures may arise. In many real-world coordination problems, from knowledge sharing in organizations to code documentation, helping others carries negligible personal cost while generating substantial collective benefits. However, whether LLM agents cooperate when helping neither benefits nor harms the helper, while being given explicit instructions to do so, remains unknown. We build a multi-agent setup designed to study cooperative behavior in a frictionless environment, removing all strategic complexity from cooperation. We find that capability does not predict cooperation: OpenAI o3 achieves only 17% of optimal collective performance while OpenAI o3-mini reaches 50%, despite identical instructions to maximize group revenue. Through a causal decomposition that automates one side of agent communication, we separate cooperation failures from competence failures, tracing their origins through agent reasoning analysis. Testing targeted interventions, we find that explicit protocols double performance for low-competence models, and tiny sharing incentives improve models with weak cooperation. Our findings suggest that scaling intelligence alone will not solve coordination problems in multi-agent systems and will require deliberate cooperative design, even when helping others costs nothing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07815v1">AsyncTLS: Efficient Generative LLM Inference with Asynchronous Two-level Sparse Attention</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      Long-context inference in LLMs faces the dual challenges of quadratic attention complexity and prohibitive KV cache memory. While token-level sparse attention offers superior accuracy, its indexing overhead is costly; block-level methods improve efficiency but sacrifice precision. We propose AsyncTLS, a hierarchical sparse attention system that combines coarse-grained block filtering with fine-grained token selection to balance accuracy and efficiency, coupled with an asynchronous offloading engine that overlaps KV cache transfers with computation via temporal locality exploitation. Evaluated on Qwen3 and GLM-4.7-Flash across GQA, and MLA architectures, AsyncTLS achieves accuracy comparable to full attention while delivering 1.2x - 10.0x operator speedups and 1.3x - 4.7x end-to-end throughput improvements on 48k - 96k contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07798v1">Lightweight LLM Agent Memory with Small Language Models</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 accept by ACL 2026
    </div>
    <details class="paper-abstract">
      Although LLM agents can leverage tools for complex tasks, they still need memory to maintain cross-turn consistency and accumulate reusable information in long-horizon interactions. However, retrieval-based external memory systems incur low online overhead but suffer from unstable accuracy due to limited query construction and candidate filtering. In contrast, many systems use repeated large-model calls for online memory operations, improving accuracy but accumulating latency over long interactions. We propose LightMem, a lightweight memory system for better agent memory driven by Small Language Models (SLMs). LightMem modularizes memory retrieval, writing, and long-term consolidation, and separates online processing from offline consolidation to enable efficient memory invocation under bounded compute. We organize memory into short-term memory (STM) for immediate conversational context, mid-term memory (MTM) for reusable interaction summaries, and long-term memory (LTM) for consolidated knowledge, and uses user identifiers to support independent retrieval and incremental maintenance in multi-user settings. Online, LightMem operates under a fixed retrieval budget and selects memories via a two-stage procedure: vector-based coarse retrieval followed by semantic consistency re-ranking. Offline, it abstracts reusable interaction evidence and incrementally integrates it into LTM. Experiments show gains across model scales, with an average F1 improvement of about 2.5 on LoCoMo, more effective and low median latency (83 ms retrieval; 581 ms end-to-end).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.13907v3">LLM Prompt Duel Optimizer: Efficient Label-Free Prompt Optimization</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 Accepted to Findings of ACL 2026. Camera-ready version
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are highly sensitive to prompts, but most automatic prompt optimization (APO) methods assume access to ground-truth references (e.g., labeled validation data) that are costly to obtain. We propose the Prompt Duel Optimizer (PDO), a sample-efficient framework for label-free prompt optimization based on pairwise preference feedback from an LLM judge. PDO casts prompt selection as a dueling-bandit problem and combines (i) Double Thompson Sampling to prioritize informative comparisons under a fixed judge budget, with (ii) top-performer guided mutation to expand the candidate pool while pruning weak prompts. Experiments on BIG-bench Hard (BBH) and MS MARCO show that PDO consistently identifies stronger prompts than label-free baselines, while offering favorable quality--cost trade-offs under constrained comparison budgets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.15564v2">OpenClassGen: A Large-Scale Corpus of Real-World Python Classes for LLM Research</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 This paper has been accepted for publication at the 30th International Conference on Evaluation and Assessment in Software Engineering (EASE 2026) AI models/data track
    </div>
    <details class="paper-abstract">
      Existing class-level code generation datasets are either synthetic (ClassEval: 100 classes) or insufficient in scale for modern training needs (RealClassEval: 400 classes), hindering robust evaluation and empirical analysis. We present OpenClassGen, a large-scale corpus of 324,843 Python classes extracted from 2,970 engineered open-source projects. Each entry pairs a human-written class with its corresponding skeleton, which comprises class and method signatures with associated docstrings, and is enriched with 27 static code metrics covering complexity, coupling, cohesion, and inheritance properties. Unlike prior benchmarks that require repository-level context resolution, OpenClassGen provides self-contained class skeletons that serve as complete generation specifications. We demonstrate the corpus's utility by evaluating three LLMs (GPT-o4-mini, Claude-4-Sonnet, Qwen-3-Coder) on a curated, executable subset of 300 classes, enriched with test suites achieving 58% branch coverage. Results show strong semantic similarity (CodeBERTScore-F3: 0.89) but moderate functional correctness (pass rate: 0.33), with substantial variance across models. This variance, along with diverse class characteristics, confirms that OpenClassGen enables meaningful differentiation of LLM capabilities. The dataset supports diverse use cases, including fine-tuning, retrieval-augmented generation, difficulty modelling, and failure mode analysis. The complete dataset and curation scripts are publicly available at https://zenodo.org/records/18409150.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.06975v5">Auditing Black-Box LLM APIs with a Rank-Based Uniformity Test</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      As API access becomes a primary interface to large language models (LLMs), users often interact with black-box systems that offer little transparency into the deployed model. To reduce costs or maliciously alter model behaviors, API providers may discreetly serve quantized or fine-tuned variants, which can degrade performance and compromise safety. Detecting such substitutions is difficult, as users lack access to model weights and, in most cases, even output logits. To tackle this problem, we propose a rank-based uniformity test that can verify the behavioral equality of a black-box LLM to a locally deployed authentic model. Our method is accurate, query-efficient, and avoids detectable query patterns, making it robust to adversarial providers that reroute or mix responses upon the detection of testing attempts. We evaluate the approach across diverse threat scenarios, including quantization, harmful fine-tuning, jailbreak prompts, and full model substitution, showing that it consistently achieves superior statistical power over prior methods under constrained query budgets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07754v1">The Art of (Mis)alignment: How Fine-Tuning Methods Effectively Misalign and Realign LLMs in Post-Training</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 Accepted by ACL Findings 2026
    </div>
    <details class="paper-abstract">
      The deployment of large language models (LLMs) raises significant ethical and safety concerns. While LLM alignment techniques are adopted to improve model safety and trustworthiness, adversaries can exploit these techniques to undermine safety for malicious purposes, resulting in \emph{misalignment}. Misaligned LLMs may be published on open platforms to magnify harm. To address this, additional safety alignment, referred to as \emph{realignment}, is necessary before deploying untrusted third-party LLMs. This study explores the efficacy of fine-tuning methods in terms of misalignment, realignment, and the effects of their interplay. By evaluating four Supervised Fine-Tuning (SFT) and two Preference Fine-Tuning (PFT) methods across four popular safety-aligned LLMs, we reveal a mechanism asymmetry between attack and defense. While Odds Ratio Preference Optimization (ORPO) is most effective for misalignment, Direct Preference Optimization (DPO) excels in realignment, albeit at the expense of model utility. Additionally, we identify model-specific resistance, residual effects of multi-round adversarial dynamics, and other noteworthy findings. These findings highlight the need for robust safeguards and customized safety alignment strategies to mitigate potential risks in the deployment of LLMs. Our code is available at https://github.com/zhangrui4041/The-Art-of-Mis-alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21783v2">NetworkGames: Simulating Cooperation in Network Games with Personality-driven LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have been extensively tested in dyadic game-theoretic scenarios, their collective behavior within complex network games remains surprisingly unexplored. To bridge this gap, we present NetworkGames, a framework connecting Generative Agents and Geometric Deep Learning. By formalizing social simulation as a message-passing process governed by LLM policies, we investigate how node heterogeneity (MBTI personalities) and network topology co-determine collective welfare. We instantiate a population of LLM agents, each endowed with a distinct personality from the MBTI taxonomy, and situate them in various network structures (e.g., small-world and scale-free). Through extensive simulations of the Iterated Prisoner's Dilemma, we first establish a baseline dyadic interaction matrix, revealing nuanced cooperative preferences between all 16 personality pairs. We then demonstrate that macro-level cooperative outcomes are not predictable from dyadic interactions alone; they are co-determined by the network's connectivity and the spatial distribution of personalities. For instance, we find that small-world networks are detrimental to cooperation, while strategically placing pro-social personalities in hub positions within scale-free networks can significantly promote cooperative behavior. We validate the robustness of these findings through extensive stress tests across multiple LLM architectures, scaled network sizes, varying random seeds, and comprehensive ablation studies. Our findings offer significant implications for designing healthier online social environments and forecasting collective behavior. We open-source our framework to facilitate research into the social physics of AI societies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.04418v2">Justified or Just Convincing? Error Verifiability as a Dimension of LLM Quality</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      As LLMs are deployed in high-stakes settings, users must judge the correctness of individual responses, often relying on model-generated justifications such as reasoning chains or explanations. Yet, no standard measure exists for whether these justifications help users distinguish correct answers from incorrect ones. We formalize this idea as error verifiability and propose $v_{\text{bal}}$, a balanced metric that measures whether justifications enable raters to accurately assess answer correctness, validated against human raters who show high agreement. We find that neither common approaches, such as post-training and model scaling, nor more targeted interventions recommended improve verifiability. We introduce two methods that succeed at improving verifiability: reflect-and-rephrase (RR) for mathematical reasoning and oracle-rephrase (OR) for factual QA, both of which improve verifiability by incorporating domain-appropriate external information. Together, our results establish error verifiability as a distinct dimension of response quality that does not emerge from accuracy improvements alone and requires dedicated, domain-aware methods to address.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06747v2">TurboAgent: An LLM-Driven Autonomous Multi-Agent Framework for Turbomachinery Aerodynamic Design</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      The aerodynamic design of turbomachinery is a complex and tightly coupled multi-stage process involving geometry generation, performance prediction, optimization, and high-fidelity physical validation. Existing intelligent design approaches typically focus on individual stages or rely on loosely coupled pipelines, making fully autonomous end-to-end design challenging. To address this issue, this study proposes TurboAgent, a large language model (LLM)-driven autonomous multi-agent framework for turbomachinery aerodynamic design and optimization. The LLM serves as the core for task planning and coordination, while specialized agents handle generative design, rapid performance prediction, multi-objective optimization, and physics-based validation. The framework transforms traditional trial-and-error design into a data-driven collaborative workflow, with high-fidelity simulations retained for final verification. A transonic single-rotor compressor is used for validation. The results show strong agreement between target performance, generated designs, and CFD simulations. The coefficients of determination for mass flow rate, total pressure ratio, and isentropic efficiency all exceed 0.91, with normalized RMSE values below 8%. The optimization agent further improves isentropic efficiency by 1.61% and total pressure ratio by 3.02%. The complete workflow can be executed within approximately 30 minutes under parallel computing. These results demonstrate that TurboAgent enables an autonomous closed-loop design process from natural language requirements to final design generation, providing an efficient and scalable paradigm for turbomachinery aerodynamic design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07737v1">SepSeq: A Training-Free Framework for Long Numerical Sequence Processing in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 16 pages, 4 figures, 5 tables
    </div>
    <details class="paper-abstract">
      While transformer-based Large Language Models (LLMs) theoretically support massive context windows, they suffer from severe performance degradation when processing long numerical sequences. We attribute this failure to the attention dispersion in the Softmax mechanism, which prevents the model from concentrating attention. To overcome this, we propose Separate Sequence (SepSeq), a training-free, plug-and-play framework to mitigate dispersion by strategically inserting separator tokens. Mechanistically, we demonstrate that separator tokens act as an attention sink, recalibrating attention to focus on local segments while preserving global context. Extensive evaluations on 9 widely-adopted LLMs confirm the effectiveness of our approach: SepSeq yields an average relative accuracy improvement of 35.6% across diverse domains while reducing total inference token consumption by 16.4% on average.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07733v1">CivBench: Progress-Based Evaluation for LLMs' Strategic Decision-Making in Civilization V</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Evaluating strategic decision-making in LLM-based agents requires generative, competitive, and longitudinal environments, yet few benchmarks provide all three, and fewer still offer evaluation signals rich enough for long-horizon, multi-agent play. We introduce CivBench, a benchmark for LLM strategists (i.e., agentic setups) in multiplayer Civilization V. Because terminal win/loss is too sparse a signal in games spanning hundreds of turns and multiple opponents, CivBench trains models on turn-level game state to estimate victory probabilities throughout play, validated through predictive, construct, and convergent validity. Across 307 games with 7 LLMs and multiple CivBench agent conditions, we demonstrate CivBench's potential to estimate strategic capabilities as an unsaturated benchmark, reveal model-specific effects of agentic setup, and outline distinct strategic profiles not visible through outcome-only evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10476v2">Learning to Negotiate: Multi-Agent Deliberation for Collective Value Alignment in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 To appear in LREC 2026 2nd DELITE Workshop
    </div>
    <details class="paper-abstract">
      LLM alignment has progressed in single-agent settings through paradigms such as RL with human feedback (RLHF), while recent work explores scalable alternatives such as RL with AI feedback (RLAIF) and dynamic alignment objectives. However, these approaches remain limited in multi-stakeholder settings, where conflicting values arise and deliberative negotiation is required. This work proposes a multi-agent negotiation-based alignment framework that aligns LLMs to Collective Agency (CA)-an existing alignment objective introduced to promote the continual expansion of agency-while simultaneously improving conflict-resolution capability. To enable scalable training, two self-play LLM instances are assigned opposing personas and engage in turn-based dialogue to synthesize mutually beneficial solutions. We generate synthetic moral-dilemma prompts and conflicting persona pairs, and optimize the policy via RLAIF using Group Relative Policy Optimization (GRPO) with an external LLM reward model. While rewards are computed from CA scores assigned to the final completion, gradients are applied to dialogue tokens to directly improve deliberative interaction dynamics. Experiments show that the model achieves CA alignment comparable to a single-agent baseline while substantially improving conflict-resolution performance without degrading general language capabilities. These results suggest that negotiation-driven deliberation training provides a practical path toward LLMs that better support collective decision-making in value-conflict scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06210v2">Distributional Open-Ended Evaluation of LLM Cultural Value Alignment Based on Value Codebook</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      As LLMs are globally deployed, aligning their cultural value orientations is critical for safety and user engagement. However, existing benchmarks face the Construct-Composition-Context ($C^3$) challenge: relying on discriminative, multiple-choice formats that probe value knowledge rather than true orientations, overlook subcultural heterogeneity, and mismatch with real-world open-ended generation. We introduce DOVE, a distributional evaluation framework that directly compares human-written text distributions with LLM-generated outputs. DOVE utilizes a rate-distortion variational optimization objective to construct a compact value-codebook from 10K documents, mapping text into a structured value space to filter semantic noise. Alignment is measured using unbalanced optimal transport, capturing intra-cultural distributional structures and sub-group diversity. Experiments across 12 LLMs show that DOVE achieves superior predictive validity, attaining a 31.56% correlation with downstream tasks, while maintaining high reliability with as few as 500 samples per culture.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04666v2">Know Thy Enemy: Securing LLMs Against Prompt Injection via Diverse Data Synthesis and Instruction-Level Chain-of-Thought Learning</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 19 pages, 6 figures; accepted by ACL 2026 Findings
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-integrated applications have become increasingly prevalent, yet face critical security vulnerabilities from prompt injection (PI) attacks. Defending against PI attacks faces two major issues: malicious instructions can be injected through diverse vectors, and injected instructions often lack clear semantic boundaries from the surrounding context, making them difficult to identify. To address these issues, we propose InstruCoT, a model enhancement method for PI defense that synthesizes diverse training data and employs instruction-level chain-of-thought fine-tuning, enabling LLMs to effectively identify and reject malicious instructions regardless of their source or position in the context. We evaluate InstruCoT across three critical dimensions: Behavior Deviation, Privacy Leakage, and Harmful Output. Experimental results across four LLMs demonstrate that InstruCoT significantly outperforms baselines in all dimensions while maintaining utility performance without degradation
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18030v2">Quine: Realizing LLM Agents as Native POSIX Processes</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 Minor revision clarifying exec semantics
    </div>
    <details class="paper-abstract">
      Current LLM agent frameworks often implement isolation, scheduling, and communication at the application layer, even though these mechanisms are already provided by mature operating systems. Instead of introducing another application-layer orchestrator, this paper presents Quine, a runtime architecture and reference implementation that realizes LLM agents as native POSIX processes. The mapping is explicit: identity is PID, interface is standard streams and exit status, state is memory, environment variables, and filesystem, and lifecycle is fork/exec/exit. A single executable implements this model by recursively spawning fresh instances of itself. By grounding the agent abstraction in the OS process model, Quine inherits isolation, composition, and resource control directly from the kernel, while naturally supporting recursive delegation, context renewal via exec, and shell-native composition. The design also exposes where the POSIX process model stops: processes provide a robust substrate for execution, but not a complete runtime model for cognition. In particular, the analysis points toward two immediate extensions beyond process semantics: task-relative worlds and revisable time. A reference implementation of Quine is publicly available on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07669v1">Reinforcement Learning with LLM-Guided Action Spaces for Synthesizable Lead Optimization</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      Lead optimization in drug discovery requires improving therapeutic properties while ensuring that proposed molecular modifications correspond to feasible synthetic routes. Existing approaches either prioritize property scores without enforcing synthesizability, or rely on expensive enumeration over large reaction networks, while direct application of Large Language Models (LLMs) frequently produces chemically invalid structures. We introduce MolReAct, a framework that formulates lead optimization as a Markov Decision Process over a synthesis-constrained action space defined by validated reaction templates. A tool-augmented LLM agent serves as a dynamic reaction environment that invokes specialized chemical analysis tools to identify reactive sites and propose chemically grounded transformations from matched templates. A policy model trained via Group Relative Policy Optimization (GRPO) selects among these constrained actions to maximize long-term oracle reward across multi-step reaction trajectories. A SMILES-based caching mechanism further reduces end-to-end optimization time by approximately 43%. Across 13 property optimization tasks from the Therapeutic Data Commons and one structure-based docking task, MolReAct achieves an average Top-10 score of 0.563, outperforming the strongest synthesizable baseline by 10.4% in relative improvement, and attains the best sample efficiency on 10 of 14 tasks. Ablations confirm that both tool-augmented reaction proposals and trajectory-level policy optimization contribute complementary gains. By grounding every step in validated reaction templates, MolReAct produces molecules that are property-improved and each accompanied by an explicit synthetic pathway.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07663v1">SAGE: Sign-Adaptive Gradient for Memory-Efficient LLM Optimization</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 Accepted to Findings of the Association for Computational Linguistics: ACL 2026. 13 pages, 4 figures, 4 tables
    </div>
    <details class="paper-abstract">
      The AdamW optimizer, while standard for LLM pretraining, is a critical memory bottleneck, consuming optimizer states equivalent to twice the model's size. Although light-state optimizers like SinkGD attempt to address this issue, we identify the embedding layer dilemma: these methods fail to handle the sparse, high-variance gradients inherent to embeddings, forcing a hybrid design that reverts to AdamW and partially negates the memory gains. We propose SAGE (Sign Adaptive GradiEnt), a novel optimizer that resolves this dilemma by replacing AdamW in this hybrid structure. SAGE combines a Lion-style update direction with a new, memory-efficient $O(d)$ adaptive scale. This scale acts as a "safe damper," provably bounded by 1.0, which tames high-variance dimensions more effectively than existing methods. This superior stability allows SAGE to achieve better convergence. On Llama models up to 1.3B parameters, our SAGE-based hybrid achieves new state-of-the-art perplexity, outperforming all baselines, including SinkGD hybrid, while significantly reducing optimizer state memory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07659v1">Efficient and Effective Internal Memory Retrieval for LLM-Based Healthcare Prediction</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 ACL 2026 (Findings), reviewer score: 3.5,3.5,4
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) hold significant promise for healthcare, yet their reliability in high-stakes clinical settings is often compromised by hallucinations and a lack of granular medical context. While Retrieval Augmented Generation (RAG) can mitigate these issues, standard supervised pipelines require computationally intensive searches over massive external knowledge bases, leading to high latency that is impractical for time-sensitive care. To address this, we introduce Keys to Knowledge (K2K), a novel framework that replaces external retrieval with internal, key-based knowledge access. By encoding essential clinical information directly into the model's parameter space, K2K enables rapid retrieval from internal key-value memory without inference-time overhead. We further enhance retrieval quality through activation-guided probe construction and cross-attention reranking. Experimental results demonstrate that K2K achieves state-of-the-art performance across four benchmark healthcare outcome prediction datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07652v1">Bridging Natural Language and Interactive What-If Interfaces via LLM-Generated Declarative Specification</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 17 pages 17 figures
    </div>
    <details class="paper-abstract">
      What-if analysis (WIA) is an iterative, multi-step process where users explore and compare hypothetical scenarios by adjusting parameters, applying constraints, and scoping data through interactive interfaces. Current tools fall short of supporting effective interactive WIA: spreadsheet and BI tools require time-consuming and laborious setup, while LLM-based chatbot interfaces are semantically fragile, frequently misinterpret intent, and produce inconsistent results as conversations progress. To address these limitations, we present a two-stage workflow that translates natural language (NL) WIA questions into interactive visual interfaces via an intermediate representation, powered by the Praxa Specification Language (PSL): first, LLMs generate PSL specifications from NL questions capturing analytical intent and logic, enabling validation and repair of erroneous specifications; and second, the specifications are compiled into interactive visual interfaces with parameter controls and linked visualizations. We benchmark this workflow with 405 WIA questions spanning 11 WIA types, 5 datasets, and 3 state-of-the-art LLMs. The results show that across models, half of specifications (52.42%) are generated correctly without intervention. We perform an analysis of the failure cases and derive an error taxonomy spanning non-functional errors (specifications fail to compile) and functional errors (specifications compile but misrepresent intent). Based on the taxonomy, we apply targeted repairs on the failure cases using few-shot prompts and improve the success rate to 80.42%. Finally, we show how undetected functional errors propagate through compilation into plausible but misleading interfaces, demonstrating that the intermediate specification is critical for reliably bridging NL and interactive WIA interface in LLM-powered WIA systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07994v4">DYCP: Dynamic Context Pruning for Long-Form Dialogue with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Accepted (B) to TACL 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly operate over long-form dialogues with frequent topic shifts. While recent LLMs support extended context windows, efficient management of dialogue history in practice is needed due to inference cost and latency constraints. We present DyCP, a lightweight context management method implemented outside the LLM that dynamically identifies and retrieves relevant dialogue segments conditioned on the current turn, without offline memory construction. DyCP manages dialogue context while preserving the sequential nature of dialogue without predefined topic boundaries, enabling adaptive and efficient context selection. Across three long-form dialogue benchmarks-LoCoMo, MT-Bench+, and SCM4LLMs-and multiple LLM backends, DyCP achieves competitive answer quality in downstream generation, with more selective context usage and improved inference efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.21354v2">The Workload-Router-Pool Architecture for LLM Inference Optimization: A Vision Paper from the vLLM Semantic Router Project</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Vision Paper
    </div>
    <details class="paper-abstract">
      Over the past year, the vLLM Semantic Router project has released a series of work spanning: (1) core routing mechanisms -- signal-driven routing, context-length pool routing, router performance engineering, policy conflict detection, low-latency embedding models, category-aware semantic caching, user-feedback-driven routing adaptation, hallucination detection, and hierarchical content-safety classification for privacy and jailbreak protection; (2) fleet optimization -- fleet provisioning and energy-efficiency analysis; (3) agentic and multimodal routing -- multimodal agent routing, tool selection, CUA security, and multi-turn context memory and safety; (4) governance and standards -- inference routing protocols and multi-provider API extensions. Each paper tackled a specific problem in LLM inference, but the problems are not independent; for example, fleet provisioning depends on the routing policy, which depends on the workload mix, shifting as organizations adopt agentic and multimodal workloads. This paper distills those results into the Workload-Router-Pool (WRP) architecture, a three-dimensional framework for LLM inference optimization. Workload characterizes what the fleet serves (chat vs. agent, single-turn vs. multi-turn, warm vs. cold, prefill-heavy vs. decode-heavy). Router determines how each request is dispatched (static semantic rules, online bandit adaptation, RL-based model selection, quality-aware cascading). Pool defines where inference runs (homogeneous vs. heterogeneous GPU, disaggregated prefill/decode, KV-cache topology). We map our prior work onto a 3x3 WRP interaction matrix, identify which cells we have covered and which remain open, and propose twenty-one concrete research directions at the intersections, each grounded in our prior measurements, tiered by maturity from engineering-ready to open research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07624v1">Program Analysis Guided LLM Agent for Proof-of-Concept Generation</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Software developers frequently receive vulnerability reports that require them to reproduce the vulnerability in a reliable manner by generating a proof-of-concept (PoC) input that triggers it. Given the source code for a software project and a specific code location for a potential vulnerability, automatically generating a PoC for the given vulnerability has been a challenging research problem. Symbolic execution and fuzzing techniques require expert guidance and manual steps and face scalability challenges for PoC generation. Although recent advances in LLMs have increased the level of automation and scalability, the success rate of PoC generation with LLMs remains quite low. In this paper, we present a novel approach called Program Analysis Guided proof of concept generation agENT (PAGENT) that is scalable and significantly improves the success rate of automated PoC generation compared to prior results. PAGENT integrates lightweight and rule-based static analysis phases for providing static analysis guidance and sanitizer-based profiling and coverage information for providing dynamic analysis guidance with a PoC generation agent. Our experiments demonstrate that the resulting hybrid approach significantly outperforms the prior top-performing agentic approach by 132% for the PoC generation task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07609v1">Blink: CPU-Free LLM Inference by Delegating the Serving Stack to GPU and SmartNIC</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) inference is rapidly becoming a core datacenter service, yet current serving stacks keep the host CPU on the critical path for orchestration and token-level control. This makes LLM performance sensitive to CPU interference, undermining application colocation and forcing operators to reserve CPU headroom, leaving substantial capacity unutilized. We introduce Blink, an end-to-end serving architecture that removes the host CPU from the steady-state inference path by redistributing responsibilities across a SmartNIC and a GPU. Blink offloads request handling to the SmartNIC, which delivers inputs directly into GPU memory via RDMA, and replaces host-driven scheduling with a persistent GPU kernel that performs batching, scheduling, and KV-cache management without CPU involvement. Evaluated against TensorRT-LLM, vLLM, and SGLang, Blink outperforms all baselines even in isolation, reducing pre-saturation P99 TTFT by up to 8.47$\times$ and P99 TPOT by up to 3.40$\times$, improving decode throughput by up to 2.1$\times$, and reducing energy per token by up to 48.6$\%$. Under CPU interference, Blink maintains stable performance, while existing systems degrade by up to two orders of magnitude.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07584v1">From Papers to Property Tables: A Priority-Based LLM Workflow for Materials Data Extraction</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Scientific data are widely dispersed across research articles and are often reported inconsistently across text, tables, and figures, making manual data extraction and aggregation slow and error-prone. We present a prompt-driven, hierarchical workflow that uses a large language model (LLM) to automatically extract and reconstruct structured, shot-level shock-physics experimental records by integrating information distributed across text, tables, figures, and physics-based derivations from full-text published research articles, using alloy spall strength as a representative case study. The pipeline targeted 37 experimentally relevant fields per shot and applied a three-level priority strategy: (T1) direct extraction from text/tables, (T2) physics-based derivation using verified governing relations, and (T3) digitization from figures when necessary. Extracted values were normalized to canonical units, tagged by priority for traceability, and validated with physics-based consistency and plausibility checks. Evaluated on a benchmark of 30 published research articles comprising 11,967 evaluated data points, the workflow achieved high overall accuracy, with priority-wise accuracies of 94.93% (T1), 92.04% (T2), and 83.49% (T3), and an overall weighted accuracy of 94.69%. Cross-model testing further indicated strong agreement for text/table and equation-derived fields, with lower agreement for figure-based extraction. Implementation through an API interface demonstrated the scalability of the approach, achieving consistent extraction performance and, in a subset of test cases, matching or exceeding chat-based accuracy. This workflow demonstrates a practical approach for converting unstructured technical literature into traceable, analysis-ready datasets without task-specific fine-tuning, enabling scalable database construction in materials science.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.09942v3">Green-LLM: Optimal Workload Allocation for Environmentally-Aware Distributed Inference</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 8 pages, 15 figures
    </div>
    <details class="paper-abstract">
      This paper investigates the optimal allocation of large language model (LLM) inference workloads across heterogeneous edge data centers over time. Each data center features on-site renewable generation and faces dynamic electricity prices and spatiotemporal variability in renewable availability. We propose Green-LLM, a lexicographic multi-objective optimization framework that addresses this challenge without requiring manual weight tuning. The proposed model incorporates real-world constraints, including token-dependent processing delay and energy consumption, heterogeneous hardware capabilities, dynamic renewable generation, and spatiotemporal variations in electricity prices and carbon intensity. Unlike existing approaches that optimize individual environmental metrics in isolation, Green-LLM jointly minimizes operational cost, carbon emissions, and delay penalty while enforcing water consumption constraints to ensure both sustainability and quality-of-service requirements. Numerical results demonstrate that Green-LLM achieves significant reductions in carbon emissions and water consumption while maintaining operational costs within 3% of the minimum and ensuring sub-2-second response latency. These findings show that sustainable LLM inference can be achieved without sacrificing service quality or economic efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13040v2">Understanding Structured Financial Data with LLMs: A Case Study on Fraud Detection</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Accepted to ACL 2026 Main Conference
    </div>
    <details class="paper-abstract">
      Detecting fraud in financial transactions typically relies on tabular models that demand heavy feature engineering to handle high-dimensional data and offer limited interpretability, making it difficult for humans to understand predictions. Large Language Models (LLMs), in contrast, can produce human-readable explanations and facilitate feature analysis, potentially reducing the manual workload of fraud analysts and informing system refinements. However, they perform poorly when applied directly to tabular fraud detection due to the difficulty of reasoning over many features, the extreme class imbalance, and the absence of contextual information. To bridge this gap, we introduce FinFRE-RAG, a two-stage approach that applies importance-guided feature reduction to serialize a compact subset of numeric/categorical attributes into natural language and performs retrieval-augmented in-context learning over label-aware, instance-level exemplars. Across four public fraud datasets and three families of open-weight LLMs, FinFRE-RAG substantially improves F1/MCC over direct prompting and is competitive with strong tabular baselines in several settings. Although these LLMs still lag behind specialized classifiers, they narrow the performance gap and provide interpretable rationales, highlighting their value as assistive tools in fraud analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07569v1">Learning is Forgetting: LLM Training As Lossy Compression</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 12 page core paper, 16 page Appendix - A shorter version with fewer visuals appears at ICLR 2026
    </div>
    <details class="paper-abstract">
      Despite the increasing prevalence of large language models (LLMs), we still have a limited understanding of how their representational spaces are structured. This limits our ability to interpret how and what they learn or relate them to learning in humans. We argue LLMs are best seen as an instance of lossy compression, where over training they learn by retaining only information in their training data relevant to their objective(s). We show pre-training results in models that are optimally compressed for next-sequence prediction, approaching the Information Bottleneck bound on compression. Across an array of open weights models, each compresses differently, likely due to differences in the data and training recipes used. However even across different families of LLMs the optimality of a model's compression, and the information present in it, can predict downstream performance on across a wide array of benchmarks, letting us directly link representational structure to actionable insights about model performance. In the general case the work presented here offers a unified Information-Theoretic framing for how these models learn that is deployable at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07565v1">Have LLM-associated terms increased in article full texts in all fields?</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      The use of Large Language Models (LLMs) like ChatGPT and DeepSeek for translation and language polishing is a welcome development, reducing the longstanding publishing barrier to non-English speakers. Assessing the uptake of this facility is useful to give insights into changing nature of scientific writing. Although the prevalence of LLM-associated terms has been tracked across science in abstracts and for full text biomedical research, their science-wide prevalence in full texts is unknown. In response, this article investigates an expanded set of 80 potentially LLM-associated terms during 2021-2025 in a science-wide full text collection from the publisher MDPI (1.25 million articles), partly focusing on the 73 journals that published at least 500 articles in 2021. The results demonstrate the increasing prevalence of LLM-associated terms science-wide in full texts to 2024, with some terms declining from 2024 to 2025 and others continuing to increase. LLMs seem to avoid some terms (e.g., thus, moreover) and a few terms have stronger associations with abstracts than full texts (e.g., enhanced) or the opposite (e.g., leveraged). The term family "underscore" had the biggest increase: up to 29-fold. There are substantial differences between journals in the apparent use of LLMs for writing, from lower uptake in the life sciences to higher uptake in social sciences, electronic engineering and environmental science. Fields in which there is currently low uptake may need improved or specialist support, such as for reliably translating complex formulae, before the full benefits of automatic translation can be realised.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07562v1">Reasoning-Based Refinement of Unsupervised Text Clusters with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Accepted to the Findings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026)
    </div>
    <details class="paper-abstract">
      Unsupervised methods are widely used to induce latent semantic structure from large text collections, yet their outputs often contain incoherent, redundant, or poorly grounded clusters that are difficult to validate without labeled data. We propose a reasoning-based refinement framework that leverages large language models (LLMs) not as embedding generators, but as semantic judges that validate and restructure the outputs of arbitrary unsupervised clustering algorithms.Our framework introduces three reasoning stages: (i) coherence verification, where LLMs assess whether cluster summaries are supported by their member texts; (ii) redundancy adjudication, where candidate clusters are merged or rejected based on semantic overlap; and (iii) label grounding, where clusters are assigned interpretable labels in a fully unsupervised manner. This design decouples representation learning from structural validation and mitigates common failure modes of embedding-only approaches. We evaluate the framework on real-world social media corpora from two platforms with distinct interaction models, demonstrating consistent improvements in cluster coherence and human-aligned labeling quality over classical topic models and recent representation-based baselines. Human evaluation shows strong agreement with LLM-generated labels, despite the absence of gold-standard annotations. We further conduct robustness analyses under matched temporal and volume conditions to assess cross-platform stability. Beyond empirical gains, our results suggest that LLM-based reasoning can serve as a general mechanism for validating and refining unsupervised semantic structure, enabling more reliable and interpretable analyses of large text collections without supervision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07549v1">EMSDialog: Synthetic Multi-person Emergency Medical Service Dialogue Generation from Electronic Patient Care Reports via Multi-LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Accepted by ACL Findings 2026
    </div>
    <details class="paper-abstract">
      Conversational diagnosis prediction requires models to track evolving evidence in streaming clinical conversations and decide when to commit to a diagnosis. Existing medical dialogue corpora are largely dyadic or lack the multi-party workflow and annotations needed for this setting. We introduce an ePCR-grounded, topic-flow-based multi-agent generation pipeline that iteratively plans, generates, and self-refines dialogues with rule-based factual and topic flow checks. The pipeline yields EMSDialog, a dataset of 4,414 synthetic multi-speaker EMS conversations based on a real-world ePCR dataset, annotated with 43 diagnoses, speaker roles, and turn-level topics. Human and LLM evaluations confirm high quality and realism of EMSDialog using both utterance- and conversation-level metrics. Results show that EMSDialog-augmented training improves accuracy, timeliness, and stability of EMS conversational diagnosis prediction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07536v1">TRUSTDESC: Preventing Tool Poisoning in LLM Applications via Trusted Description Generation</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly rely on external tools to perform time-sensitive tasks and real-world actions. While tool integration expands LLM capabilities, it also introduces a new prompt-injection attack surface: tool poisoning attacks (TPAs). Attackers manipulate tool descriptions by embedding malicious instructions (explicit TPAs) or misleading claims (implicit TPAs) to influence model behavior and tool selection. Existing defenses mainly detect anomalous instructions and remain ineffective against implicit TPAs. In this paper, we present TRUSTDESC, the first framework for preventing tool poisoning by automatically generating trusted tool descriptions from implementations. TRUSTDESC derives implementation-faithful descriptions through a three-stage pipeline. SliceMin performs reachability-aware static analysis and LLM-guided debloating to extract minimal tool-relevant code slices. DescGen synthesizes descriptions from these slices while mitigating misleading or adversarial code artifacts. DynVer refines descriptions through dynamic verification by executing synthesized tasks and validating behavioral claims. We evaluate TRUSTDESC on 52 real-world tools across multiple tool ecosystems. Results show that TRUSTDESC produces accurate tool descriptions that improve task completion rates while mitigating implicit TPAs at their root, with minimal time and monetary overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07530v1">The Shrinking Lifespan of LLMs in Science</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Scaling laws describe how language model capabilities grow with compute and data, but say nothing about how long a model matters once released. We provide the first large-scale empirical account of how scientists adopt and abandon language models over time. We track 62 LLMs across over 108k citing papers (2018-2025), each with at least three years of post-release data, and classify every citation as active adoption or background reference to construct per-model adoption trajectories that raw citation counts cannot resolve. We find three regularities. First, scientific adoption follows an inverted-U trajectory: usage rises after release, peaks, and declines as newer models appear, a pattern we term the \textit{scientific adoption curve}. Second, this curve is compressing: each additional release year is associated with a 27\% reduction in time-to-peak adoption ($p < 0.001$), robust to minimum-age thresholds and controls for model size. Third, release timing dominates model-level attributes as a predictor of lifecycle dynamics. Release year explains both time-to-peak and scientific lifespan more strongly than architecture, openness, or scale, though model size and access modality retain modest predictive power for total adoption volume. Together, these findings complement scaling laws with adoption-side regularities and suggest that the forces driving rapid capability progress may be the same forces compressing scientific relevance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.20114v4">Current LLMs still cannot 'talk much' about grammar modules: Evidence from syntax</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      We aim to examine the extent to which Large Language Models (LLMs) can 'talk much' about grammar modules, providing evidence from syntax core properties translated by ChatGPT into Arabic. We collected 44 terms from generative syntax previous works, including books and journal articles, as well as from our experience in the field. These terms were translated by humans, and then by ChatGPT-5. We then analyzed and compared both translations. We used an analytical and comparative approach in our analysis. Findings unveil that LLMs still cannot 'talk much' about the core syntax properties embedded in the terms under study involving several syntactic and semantic challenges: only 25% of ChatGPT translations were accurate, while 38.6% were inaccurate, and 36.4.% were partially correct, which we consider appropriate. Based on these findings, a set of actionable strategies were proposed, the most notable of which is a close collaboration between AI specialists and linguists to better LLMs' working mechanism for accurate or at least appropriate translation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07494v1">Triage: Routing Software Engineering Tasks to Cost-Effective LLM Tiers via Code Quality Signals</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 5 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Context: AI coding agents route every task to a single frontier large language model (LLM), paying premium inference cost even when many tasks are routine. Objectives: We propose Triage, a framework that uses code health metrics -- indicators of software maintainability -- as a routing signal to assign each task to the cheapest model tier whose output passes the same verification gate as the expensive model. Methods: Triage defines three capability tiers (light, standard, heavy -- mirroring, e.g., Haiku, Sonnet, Opus) and routes tasks based on pre-computed code health sub-factors and task metadata. We design an evaluation comparing three routing policies on SWE-bench Lite (300 tasks across three model tiers): heuristic thresholds, a trained ML classifier, and a perfect-hindsight oracle. Results: We analytically derived two falsifiable conditions under which the tier-dependent asymmetry (medium LLMs benefit from clean code while frontier models do not) yields cost-effective routing: the light-tier pass rate on healthy code must exceed the inter-tier cost ratio, and code health must discriminate the required model tier with at least a small effect size ($\hat{p} \geq 0.56$). Conclusion: Triage transforms a diagnostic code quality metric into an actionable model-selection signal. We present a rigorous evaluation protocol to test the cost--quality trade-off and identify which code health sub-factors drive routing decisions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07486v1">Private Seeds, Public LLMs: Realistic and Privacy-Preserving Synthetic Data Generation</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 23 pages, 7 figures, 18 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have emerged as a powerful tool for synthetic data generation. A particularly important use case is producing synthetic replicas of private text, which requires carefully balancing privacy and utility. We propose Realistic and Privacy-Preserving Synthetic Data Generation (RPSG), which leverages privacy-preserving mechanisms, including formal differential privacy (DP); and private seeds, in particular text containing personal information, to generate realistic synthetic data. Comprehensive experiments against state-of-the-art private synthetic data generation methods demonstrate that RPSG achieves high fidelity to private data while providing strong privacy protection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05650v2">See the Forest for the Trees: Loosely Speculative Decoding via Visual-Semantic Guidance for Efficient Inference of Video LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 ACL'2026 Main Conference
    </div>
    <details class="paper-abstract">
      Video Large Language Models (Video-LLMs) excel in video understanding but suffer from high inference latency during autoregressive generation. Speculative Decoding (SD) mitigates this by applying a draft-and-verify paradigm, yet existing methods are constrained by rigid exact-match rules, severely limiting the acceleration potential. To bridge this gap, we propose LVSpec, the first training-free loosely SD framework tailored for Video-LLMs. Grounded in the insight that generation is governed by sparse visual-relevant anchors (mandating strictness) amidst abundant visual-irrelevant fillers (permitting loose verification), LVSpec employs a lightweight visual-relevant token identification scheme to accurately pinpoint the former. To further maximize acceptance, we augment this with a position-shift tolerant mechanism that effectively salvages positionally mismatched but semantically equivalent tokens. Experiments demonstrate that LVSpec achieves high fidelity and speed: it preserves >99.8 of target performance while accelerating Qwen2.5-VL-32B by 2.70x and LLaVA-OneVision-72B by 2.94x. Notably, it boosts the mean accepted length and speedup ratio by 136% and 35% compared to SOTA training-free SD methods for Video-LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07472v1">Fast Heterogeneous Serving: Scalable Mixed-Scale LLM Allocation for SLO-Constrained Inference</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Deploying large language model (LLM) inference at scale requires jointly selecting base models, provisioning heterogeneous GPUs, configuring parallelism, and distributing workloads under tight latency, accuracy, and budget constraints. Exact mixed-integer linear programming (MILP) approaches guarantee optimality but scale poorly. We propose two constraint-aware heuristics: a Greedy Heuristic (GH) for single-pass allocation, and an Adaptive Greedy Heuristic (AGH) that enhances GH via multi-start construction, relocate-based local search, and GPU consolidation. Three constraint-aware mechanisms -- TP-aware feasibility selection, cost-per-effective-coverage ranking, and TP upgrade -- ensure feasibility under tightly coupled memory, delay, error, and budget constraints. On workloads calibrated with the Azure LLM Inference Trace (2025), both heuristics produce feasible solutions in under one second, with AGH closely approaching optimal cost while achieving over 260x speedup on large-scale instances. Under out-of-sample stress tests with up to 1.5x parameter inflation, AGH maintains controlled SLO violations and stable cost, whereas the exact solver's placement degrades sharply.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07469v1">To Layer or Not to Layer? Evaluating the Effects and Mechanisms of LLM-Generated Feedback on learning performance</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Feedback is vital for learning, yet its effectiveness depends not only on its content but also on how it engages students in the learning process. Large Language Models (LLMs) offer novel opportunities to efficiently generate rich, formative feedback, ranging from direct explanations to incrementally layered scaffolding designed to foster learner autonomy. Despite these affordances, it remains unclear whether layered feedback (which sequences encouragement and prompts prior to revealing the correct answer) actually improves engagement and learning outcomes. To address this, we randomly assigned 199 participants to receive either layered or non-layered LLM-generated feedback. We assessed its impact on learning performance, behavioral and cognitive engagement, and affective perceptions, to determine how these factors mediate learning performance. Results indicate that layered feedback elicited slightly higher behavioral engagement and, as anticipated, was perceived as more encouraging and supportive of independence. However, it concurrently induced greater mental effort. Mediation analyses revealed a positive affective pathway driven by perceived encouragement, which was counteracted by a negative behavioral pathway linked to the average number of tasks requiring $\geq 3$ submissions; the cognitive pathway (mental effort) was non-significant. Taken together, layered feedback resulted in significantly poorer learning outcomes compared to non-layered feedback. These findings illuminate a critical trade-off: while layered scaffolding enhances engagement and positive perceptions, it can detrimentally impact actual learning performance. This study contributes nuanced insights for the design of automated, LLM-driven feedback systems by integrating outcome, perception, and mechanism-level analyses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07321v1">Syntax Is Easy, Semantics Is Hard: Evaluating LLMs for LTL Translation</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 SecDev 2026 in Montreal, Canada, 10 pages, maximum 16 pages
    </div>
    <details class="paper-abstract">
      Propositional Linear Temporal Logic (LTL) is a popular formalism for specifying desirable requirements and security and privacy policies for software, networks, and systems. Yet expressing such requirements and policies in LTL remains challenging because of its intricate semantics. Since many security and privacy analysis tools require LTL formulas as input, this difficulty places them out of reach for many developers and analysts. Large Language Models (LLMs) could broaden access to such tools by translating natural language fragments into LTL formulas. This paper evaluates that premise by assessing how effectively several representative LLMs translate assertive English sentences into LTL formulas. Using both human-generated and synthetic ground-truth data, we evaluate effectiveness along syntactic and semantic dimensions. The results reveal three findings: (1) in line with prior findings, LLMs perform better on syntactic aspects of LTL than on semantic ones; (2) they generally benefit from more detailed prompts; and (3) reformulating the task as a Python code-completion problem substantially improves overall performance. We also discuss challenges in conducting a fair evaluation on this task and conclude with recommendations for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.09354v3">"Is This Really a Human Peer Supporter?": Misalignments Between Peer Supporters and Experts in LLM-Supported Interactions</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Accepted at CSCW 2026. 53 pages, 12 figures, 17 tables
    </div>
    <details class="paper-abstract">
      Mental health is a growing global concern, prompting interest in AI-driven solutions to expand access to psychosocial support. \emph{Peer support}, grounded in lived experience, offers a valuable complement to professional care. However, variability in training, effectiveness, and definitions raises concerns about quality, consistency, and safety. Large Language Models (LLMs) present new opportunities to enhance peer support interactions, particularly in real-time, text-based interactions. We present and evaluate an AI-supported system with an LLM-simulated distressed client (\client{}), context-sensitive LLM-generated suggestions (\suggestions{}), and real-time emotion visualisations. 2 mixed-methods studies with 12 peer supporters and 6 mental health professionals (i.e., experts) examined the system's effectiveness and implications for practice. Both groups recognised its potential to enhance training and improve interaction quality. However, we found a key tension emerged: while peer supporters engaged meaningfully, experts consistently flagged critical issues in peer supporter responses, such as missed distress cues and premature advice-giving. This misalignment highlights potential limitations in current peer support training, especially in emotionally charged contexts where safety and fidelity to best practices are essential. Our findings underscore the need for standardised, psychologically grounded training, especially as peer support scales globally. They also demonstrate how LLM-supported systems can scaffold this development--if designed with care and guided by expert oversight. This work contributes to emerging conversations on responsible AI integration in mental health and the evolving role of LLMs in augmenting peer-delivered care.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.22177v2">Analyzing Multimodal Interaction Strategies for LLM-Assisted Manipulation of 3D Scenes</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Published in the IEEE VR (IEEE Conference on Virtual Reality and 3D User Interfaces) 2025 Proceedings
    </div>
    <details class="paper-abstract">
      As more applications of large language models (LLMs) for 3D content for immersive environments emerge, it is crucial to study user behaviour to identify interaction patterns and potential barriers to guide the future design of immersive content creation and editing systems which involve LLMs. In an empirical user study with 12 participants, we combine quantitative usage data with post-experience questionnaire feedback to reveal common interaction patterns and key barriers in LLM-assisted 3D scene editing systems. We identify opportunities for improving natural language interfaces in 3D design tools and propose design recommendations for future LLM-integrated 3D content creation systems. Through an empirical study, we demonstrate that LLM-assisted interactive systems can be used productively in immersive environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03013v4">LLMs, You Can Evaluate It! Design of Multi-perspective Report Evaluation for Security Operation Centers</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Security operation centers (SOCs) often produce analysis reports on security incidents, and large language models (LLMs) will likely be used for this task in the near future. We postulate that a better understanding of how veteran analysts evaluate reports, including their feedback, can help produce analysis reports in SOCs. In this paper, we aim to leverage LLMs for analysis reports. To this end, we first construct a Analyst-wise checklist to reflect SOC practitioners' opinions for analysis report evaluation through literature review and user study with SOC practitioners. Next, we design a novel LLM-based conceptual framework, named MESSALA, by further introducing two new techniques, granularization guideline and multi-perspective evaluation. MESSALA can maximize report evaluation and provide feedback on veteran SOC practitioners' perceptions. When we conduct extensive experiments with MESSALA, the evaluation results by MESSALA are the closest to those of veteran SOC practitioners compared with the existing LLM-based methods. We then show two key insights. We also conduct qualitative analysis with MESSALA, and then identify that MESSALA can provide actionable items that are necessary for improving analysis reports.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07223v1">TraceSafe: A Systematic Assessment of LLM Guardrails on Multi-Step Tool-Calling Trajectories</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) evolve from static chatbots into autonomous agents, the primary vulnerability surface shifts from final outputs to intermediate execution traces. While safety guardrails are well-benchmarked for natural language responses, their efficacy remains largely unexplored within multi-step tool-use trajectories. To address this gap, we introduce TraceSafe-Bench, the first comprehensive benchmark specifically designed to assess mid-trajectory safety. It encompasses 12 risk categories, ranging from security threats (e.g., prompt injection, privacy leaks) to operational failures (e.g., hallucinations, interface inconsistencies), featuring over 1,000 unique execution instances. Our evaluation of 13 LLM-as-a-guard models and 7 specialized guardrails yields three critical findings: 1) Structural Bottleneck: Guardrail efficacy is driven more by structural data competence (e.g., JSON parsing) than semantic safety alignment. Performance correlates strongly with structured-to-text benchmarks ($ρ=0.79$) but shows near-zero correlation with standard jailbreak robustness. 2) Architecture over Scale: Model architecture influences risk detection performance more significantly than model size, with general-purpose LLMs consistently outperforming specialized safety guardrails in trajectory analysis. 3) Temporal Stability: Accuracy remains resilient across extended trajectories. Increased execution steps allow models to pivot from static tool definitions to dynamic execution behaviors, actually improving risk detection performance in later stages. Our findings suggest that securing agentic workflows requires jointly optimizing for structural reasoning and safety alignment to effectively mitigate mid-trajectory risks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07220v1">HIVE: Query, Hypothesize, Verify An LLM Framework for Multimodal Reasoning-Intensive Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 accepted at CVPR 2026 Workshop GRAIL-V
    </div>
    <details class="paper-abstract">
      Multimodal retrieval models fail on reasoning-intensive queries where images (diagrams, charts, screenshots) must be deeply integrated with text to identify relevant documents -- the best multimodal model achieves only 27.6 nDCG@10 on MM-BRIGHT, underperforming even strong text-only retrievers (32.2). We introduce \textbf{HIVE} (\textbf{H}ypothesis-driven \textbf{I}terative \textbf{V}isual \textbf{E}vidence Retrieval), a plug-and-play framework that injects explicit visual-text reasoning into a retriever via LLMs. HIVE operates in four stages: (1) initial retrieval over the corpus, (2) LLM-based compensatory query synthesis that explicitly articulates visual and logical gaps observed in top-$k$ candidates, (3) secondary retrieval with the refined query, and (4) LLM verification and reranking over the union of candidates. Evaluated on the multimodal-to-text track of MM-BRIGHT (2,803 real-world queries across 29 technical domains), HIVE achieves a new state-of-the-art aggregated nDCG@10 of \textbf{41.7} -- a \textbf{+9.5} point gain over the best text-only model (DiVeR: 32.2) and \textbf{+14.1} over the best multimodal model (Nomic-Vision: 27.6), where our reasoning-enhanced base retriever contributes 33.2 and the HIVE framework adds a further \textbf{+8.5} points -- with particularly strong results in visually demanding domains (Gaming: 68.2, Chemistry: 42.5, Sustainability: 49.4). Compatible with both standard and reasoning-enhanced retrievers, HIVE demonstrates that LLM-mediated visual hypothesis generation and verification can substantially close the multimodal reasoning gap in retrieval. https://github.com/mm-bright/multimodal-reasoning-retrieval
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07192v1">Compact Constraint Encoding for LLM Code Generation: An Empirical Study of Token Economics and Constraint Compliance</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      LLMs used for code generation are typically guided by engineering constraints--technology choices, dependency restrictions, and architectural patterns--expressed in verbose natural language. We investigate whether compact, structured constraint headers can reduce prompt token consumption without degrading constraint compliance. Across six experimental rounds spanning 11 models, 16 benchmark tasks, and over 830 LLM invocations, we find that compact headers reduce constraint-portion tokens by approximately 71% and full-prompt tokens by 25--30%, replicated across three independent rounds. However, we detect no statistically significant differences in constraint satisfaction rate (CSR) across three encoding forms or four propagation modes; observed effect sizes are negligible (Cliff's $δ$ < 0.01, 95% CI spanning $\pm$2.6 percentage points). This null pattern holds across two models from different capability tiers. A supplementary experiment with four non-CSS tasks provides additional cross-domain support for the encoding null result. The largest observed sources of compliance variance are constraint type ($Δ$ = 9 percentage points between normal and counter-intuitive constraints) and task domain: counter-intuitive constraints opposing model defaults fail at 10--100%, while conventional constraints achieve 99%+ compliance regardless of encoding. Model self-assessments systematically overestimate compliance relative to rule-based scoring, revealing a gap between constraint understanding and execution. Under the tested conditions, the primary benefit of compact constraint encoding is token reduction rather than compliance improvement, and engineering effort toward compliance is better directed at constraint design than prompt formatting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07148v1">Multi-Turn Reasoning LLMs for Task Offloading in Mobile Edge Computing</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Emerging computation-intensive applications impose stringent latency requirements on resource-constrained mobile devices. Mobile Edge Computing (MEC) addresses this challenge through task offloading. However, designing effective policies remains difficult due to dynamic task arrivals, time-varying channels, and the spatio-temporal coupling of server queues. Conventional heuristics lack adaptability, while Deep Reinforcement Learning (DRL) suffers from limited generalization and architectural rigidity, requiring retraining when network topology changes. Although Large Language Models (LLMs) offer semantic reasoning capabilities, standard Supervised Fine-Tuning (SFT) yields myopic policies that greedily minimize immediate latency without accounting for long-term system evolution. To address these limitations, we propose COMLLM, a generative framework that enables foresighted decision-making in MEC systems. COMLLM integrates Group Relative Policy Optimization (GRPO) with a Look-Ahead Collaborative Simulation (LACS) mechanism, which performs multi-step Monte Carlo rollouts while jointly modeling server queue dynamics. By incorporating these rollouts into the reward design, the framework captures the long-term impact of current decisions on future system states. Experimental results demonstrate that COMLLM achieves near-optimal latency and improved load-balancing fairness. Notably, it exhibits zero-shot topological scalability, allowing a model trained on small-scale networks to generalize to larger, unseen topologies without retraining, outperforming SFT, DRL, and heuristic baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07144v1">Autopoiesis: A Self-Evolving System Paradigm for LLM Serving Under Runtime Dynamics</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Modern Large Language Model (LLM) serving operates in highly volatile environments characterized by severe runtime dynamics, such as workload fluctuations and elastic cluster autoscaling. Traditional serving systems rely on static, human-engineered serving policies (e.g., scheduling algorithms and rescheduling strategies) to manage these dynamics. However, these policies must navigate deeply intertwined runtime trade-offs (e.g., scheduling overhead vs. execution efficiency, rescheduling frequency vs. reconfiguration overhead), whose optimal balance is workload-specific and shifts continuously as runtime conditions evolve, rendering any fixed policy fundamentally unable to adapt. We propose Autopoiesis, a novel online self-evolving system that shifts LLM serving from static policy deployment to continuous online policy evolution. First, Autopoiesis introduces an LLM-driven program synthesis workflow to evolve serving policies with respect to real-time observed dynamics, where the evolved policies reflect the optimal decision in navigating the complex, multi-dimensional trade-off space. Second, Autopoiesis enables this synthesis process to operate continuously during serving, observing real-world system behavior, and rewriting the policy code as runtime trade-offs shift, thereby transforming policy design from a one-time offline endeavor into an ongoing system component, enabling autonomous adaptation to evolving runtime conditions. Together, we establish a new paradigm: Serving policies are no longer static artifacts designed by humans before deployment, but living code that LLMs continuously evolve throughout deployment to navigate runtime trade-offs beyond human design. We evaluate Autopoiesis across diverse runtime dynamics and show up to 53% and on average 34% improvements over state-of-the-art LLM serving systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07123v1">Language Bias under Conflicting Information in Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been shown to contain biases in the process of integrating conflicting information when answering questions. Here we ask whether such biases also exist with respect to which language is used for each conflicting piece of information. To answer this question, we extend the conflicting needles in a haystack paradigm to a multilingual setting and perform a comprehensive set of evaluations with naturalistic news domain data in five different languages, for a range of multilingual LLMs of different sizes. We find that all LLMs tested, including GPT-5.2, ignore the conflict and confidently assert only one of the possible answers in the large majority of cases. Furthermore, there is a consistent bias across models in which languages are preferred, with a general bias against Russian and, for the longest context lengths, in favor of Chinese. Both of these patterns are consistent between models trained inside and outside of mainland China, though somewhat stronger in the former category.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07070v1">EVGeoQA: Benchmarking LLMs on Dynamic, Multi-Objective Geo-Spatial Exploration</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) demonstrate remarkable reasoning capabilities, their potential for purpose-driven exploration in dynamic geo-spatial environments remains under-investigated. Existing Geo-Spatial Question Answering (GSQA) benchmarks predominantly focus on static retrieval, failing to capture the complexity of real-world planning that involves dynamic user locations and compound constraints. To bridge this gap, we introduce EVGeoQA, a novel benchmark built upon Electric Vehicle (EV) charging scenarios that features a distinct location-anchored and dual-objective design. Specifically, each query in EVGeoQA is explicitly bound to a user's real-time coordinate and integrates the dual objectives of a charging necessity and a co-located activity preference. To systematically assess models in such complex settings, we further propose GeoRover, a general evaluation framework based on a tool-augmented agent architecture to evaluate the LLMs' capacity for dynamic, multi-objective exploration. Our experiments reveal that while LLMs successfully utilize tools to address sub-tasks, they struggle with long-range spatial exploration. Notably, we observe an emergent capability: LLMs can summarize historical exploration trajectories to enhance exploration efficiency. These findings establish EVGeoQA as a challenging testbed for future geo-spatial intelligence. The dataset and prompts are available at https://github.com/Hapluckyy/EVGeoQA/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07036v1">ReDAct: Uncertainty-Aware Deferral for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Recently, LLM-based agents have become increasingly popular across many applications, including complex sequential decision-making problems. However, they inherit the tendency of LLMs to hallucinate, leading to incorrect decisions. In sequential settings, even a single mistake can irreversibly degrade the trajectory, making hallucinations an even bigger problem. Although larger LLMs hallucinate less, they incur a significantly higher per-token cost. In this paper, we address this tradeoff by proposing ReDAct (Reason-Defer-Act). In ReDAct, an agent is equipped with two LLMs: a small, cheap model used by default, and a large, more reliable but expensive model. When the predictive uncertainty of the small model exceeds a calibrated threshold, the decision is deferred to the large model. We evaluate our approach in text-based embodied environments such as ALFWorld and MiniGrid and show that deferring only about 15% of decisions to the large model can match the quality of using it exclusively, while significantly reducing inference costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01423v2">Active Hypothesis Testing under Computational Budgets with Applications to GWAS and LLM</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      In large-scale hypothesis testing, computing exact $p$-values or $e$-values is often resource-intensive, creating a need for budget-aware inferential methods. We propose a general framework for active hypothesis testing that leverages inexpensive auxiliary statistics to allocate a global computational budget. For each hypothesis, our data-adaptive procedure probabilistically decides whether to compute the exact test statistic or a transformed proxy, guaranteeing a valid $p$-value or $e$-value while satisfying the exact budget constraint. Theoretical guarantees are established for our constructions, showing that the procedure achieves optimality for $e$-values and for $p$-values under independence, and admissibility for $p$-values under general dependence. Empirical results from simulations and two real-world applications, including a large-scale genome-wide association study (GWAS) and a clinical prediction task leveraging large language models (LLM), demonstrate that our framework improves statistical efficiency under fixed resource limits.
    </details>
</div>
