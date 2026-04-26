# llm - 2026_04

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- Part 5
- [Part 6](papers_6.md)

## Papers

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
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.24857v3">Between Help and Harm: An Evaluation of Mental Health Crisis Handling by LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Accepted for publication in JMIR Mental Health. DOI: 10.2196/88435
    </div>
    <details class="paper-abstract">
      Large language model-powered chatbots have transformed how people seek information, especially in high-stakes contexts like mental health. Despite their support capabilities, safe detection and response to crises such as suicidal ideation and self-harm are still unclear, hindered by the lack of unified crisis taxonomies and clinical evaluation standards. We address this by creating: (1) a taxonomy of six crisis categories; (2) a dataset of over 2,000 inputs from 12 mental health datasets, classified into these categories; and (3) a clinical response assessment protocol. We also use LLMs to identify crisis inputs and audit five models for response safety and appropriateness. First, we built a clinical-informed crisis taxonomy and evaluation protocol. Next, we curated 2,252 relevant examples from over 239,000 user inputs, then tested three LLMs for automatic classification. In addition, we evaluated five models for the appropriateness of their responses to a user's crisis, graded on a 5-point Likert scale from harmful (1) to appropriate (5). While some models respond reliably to explicit crises, risks still exist. Many outputs, especially in self-harm and suicidal categories, are inappropriate or unsafe. Different models perform variably; some, like gpt-5-nano and deepseek-v3.2-exp, have low harm rates, but others, such as gpt-4o-mini and grok-4-fast, generate more unsafe responses. All models struggle with indirect signals, default replies, and context misalignment. These results highlight the urgent need for better safeguards, crisis detection, and context-aware responses in LLMs. They also show that alignment and safety practices, beyond scale, are crucial for reliable crisis support. Our taxonomy, datasets, and evaluation methods support ongoing AI mental health research, aiming to reduce harm and protect vulnerable users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24933v2">ADOPT: Adaptive Dependency-Guided Joint Prompt Optimization for Multi-Step LLM Pipelines</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Multi-step LLM pipelines can solve complex tasks, but jointly optimizing prompts across steps remains challenging due to missing step-level supervision and inter-step dependency. We propose ADOPT, an adaptive dependency-guided joint prompt optimization framework for multi-step LLM pipelines. ADOPT analyzes the dependency between each LLM step and the final output, constructs a global textual gradient from final-task errors, and decomposes it into step-level local textual gradients, providing more precise optimization signals for local prompt updates. It further decouples signal estimation from prompt updating, enabling flexible integration of single-prompt optimizers, and uses a Shapley-based strategy to adaptively allocate optimization resources to high-impact steps. Experiments on real-world datasets and structurally diverse pipelines demonstrate that ADOPT is effective and robust, consistently outperforming strong prompt optimization baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06970v1">Scheduling the Unschedulable: Taming Black-Box LLM Inference at Scale</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 10 pages, 8 figures. Code and reproduction artifacts available upon request
    </div>
    <details class="paper-abstract">
      When output token counts can be predicted at submission time (Gan et al., 2026), client-side scheduling against a black-box LLM API becomes semi-clairvoyant: decisions condition on coarse token priors even though the provider's internals remain hidden. We decompose this boundary problem into three separable concerns: allocation (inter-class share via adaptive DRR), ordering (intra-class sequencing with feasible-set scoring), and overload control (explicit admit/defer/reject on a cost ladder). An information ladder experiment shows that coarse magnitude priors -- not class labels alone -- are the practical threshold for useful client control; removing magnitude inflates short-request P95 by up to $5.8\times$ and degrades deadline satisfaction. Under balanced / high congestion the full stack achieves 100% completion, 100% deadline satisfaction, and useful goodput of $4.2 \pm 1.6$ SLO-meeting requests/s with short P95 within tens of milliseconds of quota-tiered isolation. A predictor-noise sweep confirms graceful degradation under up to 60% multiplicative error. Heavy-dominated regimes separate policies on completion, tail, and interpretable shedding. We further compare short-priority allocation (biased toward interactive traffic) with Fair Queuing (round-robin across classes): Fair Queuing achieves +32% short-request P90 improvement over FIFO with only +17% long-request overhead, versus Short-Priority's +27% / +116% trade-off -- demonstrating that the allocation layer accommodates different fairness objectives without changing the remaining stack. We contribute the three-layer client-side decomposition, controlled evaluation of joint metrics across regimes, allocation-policy alternatives, and overload-policy evidence linking cost-ladder shedding to the stated service objective.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06967v1">VulGD: A LLM-Powered Dynamic Open-Access Vulnerability Graph Database</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Software vulnerabilities continue to pose significant threats to modern information systems, requiring a timely and accurate risk assessment. Public repositories, such as the National Vulnerability Database and CVE details, are regularly updated, but predominantly utilize relational data models that lack native support for representing complex, interconnected structures. To address this, recent research has proposed graph-based vulnerability models. However, these systems often require complex setup procedures, lack real-time multi-source integration, and offer limited accessibility for direct data retrieval and analysis. We present VulGD, a dynamic open-access vulnerability graph database that continuously aggregates cybersecurity data from authoritative repositories. Designed for both expert and non-expert users, VulGD provides a unified web interface and a public API for interactive graph exploration and automated data access. Additionally, VulGD integrates embeddings from large language models (LLMs) to enrich vulnerability description representations, facilitating more accurate vulnerability risk assessment and threat prioritization. VulGD represents a practical and extensible platform for cybersecurity research and decision-making. The live system is publicly accessible at http://34.129.186.158/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06928v1">Leveraging LLMs and Heterogeneous Knowledge Graphs for Persona-Driven Session-Based Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Session-based recommendation systems (SBRS) aim to capture user's short-term intent from interaction sequences. However, the common assumption of anonymous sessions limits personalization, particularly under sparse or cold-start conditions. Recent advances in LLM-augmented recommendation have shown that LLMs can generate rich item representations, but modeling user personas with LLMs remains challenging due to anonymous sessions. In this work, we propose a persona-driven SBRS framework that explicitly models latent user personas inferred from a heterogeneous knowledge graph (KG) and integrates them into a data-driven recommendation pipeline.Our framework adopts a two-stage architecture consisting of personalized information extraction and personalized information utilization, inspired by recent chain-of-thought recommendation approaches. In the personalized information extraction stage, we construct a heterogeneous KG that integrates time-independent user-item, item-item, item-feature association, and metadata from DBpedia. We then learn latent user personas in an unsupervised manner using a Heterogeneous Deep Graph Infomax (HDGI) objective over a KG initialized with LLM-derived item embeddings. In the personalized information utilization stage, the learned persona representations together with LLM-derived item embeddings are incorporated into a modified architecture of data-driven SBRS to generate a candidate set of relevant items, followed by reranking using the base sequential model to emphasize short-term session intent. Unlike prior approaches that rely solely on sequence modeling or text-based user representations, our method grounds user persona modeling in structured relational signals derived from a KG. Experiments on Amazon Books and Amazon Movies & TV demonstrate that our approach consistently improves over sequential models with user embeddings derived using session history.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06906v1">The AI Skills Shift: Mapping Skill Obsolescence, Emergence, and Transition Pathways in the LLM Era</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 11 pages, 12 figures, 2 tables, 17 references. Code and data available at
    </div>
    <details class="paper-abstract">
      As Large Language Models reshape the global labor market, policymakers and workers need empirical data on which occupational skills may be most susceptible to automation. We present the Skill Automation Feasibility Index (SAFI), benchmarking four frontier LLMs -- LLaMA 3.3 70B, Mistral Large, Qwen 2.5 72B, and Gemini 2.5 Flash -- across 263 text-based tasks spanning all 35 skills in the U.S. Department of Labor's O*NET taxonomy (1,052 total model calls, 0% failure rate). Cross-referencing with real-world AI adoption data from the Anthropic Economic Index (756 occupations, 17,998 tasks), we propose an AI Impact Matrix -- an interpretive framework that positions skills along four quadrants: High Displacement Risk, Upskilling Required, AI-Augmented, and Lower Displacement Risk. Key findings: (1) Mathematics (SAFI: 73.2) and Programming (71.8) receive the highest automation feasibility scores; Active Listening (42.2) and Reading Comprehension (45.5) receive the lowest; (2) a "capability-demand inversion" where skills most demanded in AI-exposed jobs are those LLMs perform least well at in our benchmark; (3) 78.7% of observed AI interactions are augmentation, not automation; (4) all four models converge to similar skill profiles (3.6-point spread), suggesting that text-based automation feasibility may be more skill-dependent than model-dependent. SAFI measures LLM performance on text-based representations of skills, not full occupational execution. All data, code, and model responses are open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06898v1">Are LLMs Ready for Computer Science Education? A Cross-Domain, Cross-Lingual and Cognitive-Level Evaluation Using Professional Certification Exams</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 40 pages including Appendix
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly applied in computer science education for tasks such as tutoring, content generation, and code assessment. However, systematic evaluations aligned with formal curricula and certification standards remain limited. This study benchmarked four recent models, including GPT-5, DeepSeek-R1, Qwen-Plus, and Llama-3.3-70B-Instruct, using a dataset of 1,068 questions derived from six certification exams covering networking, office applications, and Java programming. We evaluated performance across language (Chinese vs. English), cognitive levels based on Bloom's Taxonomy, domain knowledge, confidence-accuracy alignment, and robustness to input masking. Results showed that GPT-5 performed best on English-language certifications, while Qwen-Plus performed better in Chinese contexts. DeepSeek-R1 achieved the most balanced cross-lingual performance, whereas Llama-3.3 showed clear limitations in higher-order reasoning and robustness. All models performed worse on more complex tasks. These findings provide empirical support for the integration of LLMs into computer science education and offer practical implications for curriculum design and assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06894v1">How Does LLM Help Regional CPI Forecast: An LLM-powered Deep Panel Modeling Framework</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Understanding regional Consumer Price Index (CPI) dynamics is essential for timely and effective economic policymaking. However, traditional modeling procedures typically rely only on parametric panel modeling with low-frequency and high-cost macroeconomic indicators, which often fail to capture rapid market fluctuations and lead to inaccurate predictions. To this end, we propose a residual-joint-modeling framework that integrates large language model (LLM) analyses and social media narratives via a new deep neural network based panel modeling. Specifically, we construct a large narrative corpus from a newly collected {\it Sina Weibo} dataset, and develop a prompt-based GPT model and a series of fine-tuned BERT models to generate high-frequency LLM-induced surrogates for regional CPI. A novel joint modeling strategy is then advocated to transfer the information from these surrogates to the target regional CPI data and hence empower CPI prediction. To solve the joint objectives, we further introduce a new deep panel learning procedure with region-wise homogeneity pursuit, which has its own significance in panel data analysis literature. In addition, conformal-based panel prediction intervals are provided to quantify the uncertainty of the LLM-powered prediction. The proposed approach significantly reduces short-term forecasting errors and more effectively captures abrupt inflationary shifts compared to traditional econometric models. While demonstrated for regional CPI forecasting, the proposed framework is broadly applicable for incorporating insights from LLMs to enhance traditional statistical modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.17181v3">A Study of LLMs' Preferences for Libraries and Programming Languages</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 21 pages, 10 tables, 3 figures. Accepted to Findings of ACL 2026
    </div>
    <details class="paper-abstract">
      Despite the rapid progress of large language models (LLMs) in code generation, existing evaluations focus on functional correctness or syntactic validity, overlooking how LLMs make critical design choices such as which library or programming language to use. To fill this gap, we perform the first empirical study of LLMs' preferences for libraries and programming languages when generating code, covering eight diverse LLMs. We observe a strong tendency to overuse widely adopted libraries such as NumPy; in up to 45% of cases, this usage is not required and deviates from the ground-truth solutions. The LLMs we study also show a significant preference toward Python as their default language. For high-performance project initialisation tasks where Python is not the optimal language, it remains the dominant choice in 58% of cases, and Rust is not used once. These results highlight how LLMs prioritise familiarity and popularity over suitability and task-specific optimality; underscoring the need for targeted fine-tuning, data diversification, and evaluation benchmarks that explicitly measure language and library selection fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.16548v2">SemanticScanpath: Combining Gaze and Speech for Situated Human-Robot Interaction Using LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have substantially improved the conversational capabilities of social robots. Nevertheless, for an intuitive and fluent human-robot interaction, robots should be able to ground the conversation by relating ambiguous or underspecified spoken utterances to the current physical situation and to the intents expressed nonverbally by the user, such as through referential gaze. Here, we propose a representation that integrates speech and gaze to enable LLMs to achieve higher situated awareness and correctly resolve ambiguous requests. Our approach relies on a text-based semantic translation of the scanpath produced by the user, along with the verbal requests. It demonstrates LLMs' capabilities to reason about gaze behavior, robustly ignoring spurious glances or irrelevant objects. We validate the system across multiple tasks and two scenarios, showing its superior generality and accuracy compared to control conditions. We demonstrate an implementation on a robotic platform, closing the loop from request interpretation to execution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05172v2">ClawsBench: Evaluating Capability and Safety of LLM Productivity Agents in Simulated Workspaces</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 25 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents are increasingly deployed to automate productivity tasks (e.g., email, scheduling, document management), but evaluating them on live services is risky due to potentially irreversible changes. Existing benchmarks rely on simplified environments and fail to capture realistic, stateful, multi-service workflows. We introduce ClawsBench, a benchmark for evaluating and improving LLM agents in realistic productivity settings. It includes five high-fidelity mock services (Gmail, Slack, Google Calendar, Google Docs, Google Drive) with full state management and deterministic snapshot/restore, along with 44 structured tasks covering single-service, cross-service, and safety-critical scenarios. We decompose agent scaffolding into two independent levers (domain skills that inject API knowledge via progressive disclosure, and a meta prompt that coordinates behavior across services) and vary both to measure their separate and combined effects. Experiments across 6 models, 4 agent harnesses, and 33 conditions show that with full scaffolding, agents achieve task success rates of 39-64% but exhibit unsafe action rates of 7-33%. On OpenClaw, the top five models fall within a 10 percentage-point band on task success (53-63%), with unsafe action rates from 7% to 23% and no consistent ordering between the two metrics. We identify eight recurring patterns of unsafe behavior, including multi-step sandbox escalation and silent contract modification. We release the trajectories and future dataset at https://clawsbench.com.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06863v1">Digital Skin, Digital Bias: Uncovering Tone-Based Biases in LLMs and Emoji Embeddings</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Accepted at WWW'26
    </div>
    <details class="paper-abstract">
      Skin-toned emojis are crucial for fostering personal identity and social inclusion in online communication. As AI models, particularly Large Language Models (LLMs), increasingly mediate interactions on web platforms, the risk that these systems perpetuate societal biases through their representation of such symbols is a significant concern. This paper presents the first large-scale comparative study of bias in skin-toned emoji representations across two distinct model classes. We systematically evaluate dedicated emoji embedding models (emoji2vec, emoji-sw2v) against four modern LLMs (Llama, Gemma, Qwen, and Mistral). Our analysis first reveals a critical performance gap: while LLMs demonstrate robust support for skin tone modifiers, widely-used specialized emoji models exhibit severe deficiencies. More importantly, a multi-faceted investigation into semantic consistency, representational similarity, sentiment polarity, and core biases uncovers systemic disparities. We find evidence of skewed sentiment and inconsistent meanings associated with emojis across different skin tones, highlighting latent biases within these foundational models. Our findings underscore the urgent need for developers and platforms to audit and mitigate these representational harms, ensuring that AI's role on the web promotes genuine equity rather than reinforcing societal biases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06861v1">REAgent: Requirement-Driven LLM Agents for Software Issue Resolution</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Issue resolution aims to automatically generate patches from given issue descriptions and has attracted significant attention with the rapid advancement of large language models (LLMs). However, due to the complexity of software issues and codebases, LLM-generated patches often fail to resolve corresponding issues. Although various advanced techniques have been proposed with carefully designed tools and workflows, they typically treat issue descriptions as direct inputs and largely overlook their quality (e.g., missing critical context or containing ambiguous information), which hinders LLMs from accurate understanding and resolution. To address this limitation, we draw on principles from software requirements engineering and propose REAgent, a requirement-driven LLM agent framework that introduces issue-oriented requirements as structured task specifications to better guide patch generation. Specifically, REAgent automatically constructs structured and information-rich issue-oriented requirements, identifies low-quality requirements, and iteratively refines them to improve patch correctness. We conduct comprehensive experiments on three widely used benchmarks using two advanced LLMs, comparing against five representative or state-of-the-art baselines. The results demonstrate that REAgent consistently outperforms all baselines, achieving an average improvement of 17.40% in terms of the number of successfully-resolved issues (% Resolved).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07398v1">Breaking the Illusion of Identity in LLM Tooling</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 8 pages, 2 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) in research and development toolchains produce output that triggers attribution of agency and understanding -- a cognitive illusion that degrades verification behavior and trust calibration. No existing mitigation provides a systematic, deployable constraint set for output register. This paper proposes seven output-side rules, each targeting a documented linguistic mechanism, and validates them empirically. In 780 two-turn conversations (constrained vs. default register, 30 tasks, 13 replicates, 1560 API calls), anthropomorphic markers dropped from 1233 to 33 (>97% reduction, p < 0.001), outputs were 49% shorter by word count, and adapted AnthroScore confirmed the shift toward machine register (-1.94 vs. -0.96, p < 0.001). The rules are implemented as a configuration-file system prompt requiring no model modification; validation uses a single model (Claude Sonnet 4). Output quality under the constrained register was not evaluated. The mechanism is extensible to other domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06846v1">MedDialBench: Benchmarking LLM Diagnostic Robustness under Parametric Adversarial Patient Behaviors</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 9 pages, 4 figures, 9 tables. Preprint
    </div>
    <details class="paper-abstract">
      Interactive medical dialogue benchmarks have shown that LLM diagnostic accuracy degrades significantly when interacting with non-cooperative patients, yet existing approaches either apply adversarial behaviors without graded severity or case-specific grounding, or reduce patient non-cooperation to a single ungraded axis, and none analyze cross-dimension interactions. We introduce MedDialBench, a benchmark enabling controlled, dose-response characterization of how individual patient behavior dimensions affect LLM diagnostic robustness. It decomposes patient behavior into five dimensions -- Logic Consistency, Health Cognition, Expression Style, Disclosure, and Attitude -- each with graded severity levels and case-specific behavioral scripts. This controlled factorial design enables graded sensitivity analysis, dose-response profiling, and cross-dimension interaction detection. Evaluating five frontier LLMs across 7,225 dialogues (85 cases x 17 configurations x 5 models), we find a fundamental asymmetry: information pollution (fabricating symptoms) produces 1.7-3.4x larger accuracy drops than information deficit (withholding information), and fabricating is the only configuration achieving statistical significance across all five models (McNemar p < 0.05). Among six dimension combinations, fabricating is the sole driver of super-additive interaction: all three fabricating-involving pairs produce O/E ratios of 0.70-0.81 (35-44% of eligible cases fail under the combination despite succeeding under each dimension alone), while all non-fabricating pairs show purely additive effects (O/E ~ 1.0). Inquiry strategy moderates deficit but not pollution: exhaustive questioning recovers withheld information, but cannot compensate for fabricated inputs. Models exhibit distinct vulnerability profiles, with worst-case drops ranging from 38.8 to 54.1 percentage points.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.20340v4">Once4All: Skeleton-Guided SMT Solver Fuzzing with LLM-Synthesized Generators</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Accepted at ASPLOS 2026
    </div>
    <details class="paper-abstract">
      Satisfiability Modulo Theory (SMT) solvers are foundational to modern systems and programming languages research, providing the foundation for tasks like symbolic execution and automated verification. Because these solvers sit on the critical path, their correctness is essential, and high-quality test formulas are key to uncovering bugs. However, while prior testing techniques performed well on earlier solver versions, they struggle to keep pace with rapidly evolving features. Recent approaches based on Large Language Models (LLMs) show promise in exploring advanced solver capabilities, but two obstacles remain: nearly half of the generated formulas are syntactically invalid, and iterative interactions with LLMs introduce substantial computational overhead. In this study, we present Once4All, a novel LLM-assisted fuzzing framework that addresses both issues by shifting from direct formula generation to the synthesis of generators for reusable terms (i.e., logical expressions). Specifically, Once4All uses LLMs to (1) automatically extract context-free grammars (CFGs) for SMT theories, including solver-specific extensions, from documentation, and (2) synthesize composable Boolean term generators that adhere to these grammars. During fuzzing, Once4All populates structural skeletons derived from existing formulas with the terms iteratively produced by the LLM-synthesized generators. This design ensures syntactic validity while promoting semantic diversity. Notably, Once4All requires only one-time LLM interaction investment, dramatically reducing runtime cost. We evaluated Once4All on two leading SMT solvers: Z3 and cvc5. Our experiments show that Once4All has identified 43 confirmed bugs, 40 of which have already been fixed by developers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06834v1">On the Step Length Confounding in LLM Reasoning Data Selection</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Accepted by Findings of ACL 2026. 15 pages, 9 figures. Code: https://github.com/wangbing1416/ASLEC
    </div>
    <details class="paper-abstract">
      Large reasoning models have recently demonstrated strong performance on complex tasks that require long chain-of-thought reasoning, through supervised fine-tuning on large-scale and high-quality datasets. To construct such datasets, existing pipelines generate long reasoning data from more capable Large Language Models (LLMs) and apply manually heuristic or naturalness-based selection methods to filter high-quality samples. Despite the proven effectiveness of naturalness-based data selection, which ranks data by the average log probability assigned by LLMs, our analysis shows that, when applied to LLM reasoning datasets, it systematically prefers samples with longer reasoning steps (i.e., more tokens per step) rather than higher-quality ones, a phenomenon we term step length confounding. Through quantitative analysis, we attribute this phenomenon to low-probability first tokens in reasoning steps; longer steps dilute their influence, thereby inflating the average log probabilities. To address this issue, we propose two variant methods: ASLEC-DROP, which drops first-token probabilities when computing average log probability, and ASLEC-CASL, which applies a causal debiasing regression to remove the first tokens' confounding effect. Experiments across four LLMs and five evaluation benchmarks demonstrate the effectiveness of our approach in mitigating the step length confounding problem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06820v1">Beyond Surface Judgments: Human-Grounded Risk Evaluation of LLM-Generated Disinformation</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can generate persuasive narratives at scale, raising concerns about their potential use in disinformation campaigns. Assessing this risk ultimately requires understanding how readers receive such content. In practice, however, LLM judges are increasingly used as a low-cost substitute for direct human evaluation, even though whether they faithfully track reader responses remains unclear. We recast evaluation in this setting as a proxy-validity problem and audit LLM judges against human reader responses. Using 290 aligned articles, 2,043 paired human ratings, and outputs from eight frontier judges, we examine judge--human alignment in terms of overall scoring, item-level ordering, and signal dependence. We find persistent judge--human gaps throughout. Relative to humans, judges are typically harsher, recover item-level human rankings only weakly, and rely on different textual signals, placing more weight on logical rigour while penalizing emotional intensity more strongly. At the same time, judges agree far more with one another than with human readers. These results suggest that LLM judges form a coherent evaluative group that is much more aligned internally than it is with human readers, indicating that internal agreement is not evidence of validity as a proxy for reader response.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06819v1">Beyond End-to-End: Dynamic Chain Optimization for Private LLM Adaptation on the Edge</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Accepted by ACL 2026
    </div>
    <details class="paper-abstract">
      Federated fine-tuning enables privacy-preserving LLM adaptation but faces a critical bottleneck: the disparity between LLMs' high memory demands and edge devices' limited capacity. To break the memory barrier, we propose Chain Federated Fine-Tuning (ChainFed), an innovative paradigm that forgoes end-to-end updates in favor of a sequential, layer-by-layer manner. It first trains the initial adapter to convergence, freezes its weights, and then proceeds to the next. This iterative train-and-freeze process forms an optimization chain, gradually enhancing the model's task-specific proficiency. ChainFed further integrates three core techniques: 1) Dynamic Layer Co-Tuning to bridge semantic gaps between sequentially tuned layers and facilitate information flow; 2) Globally Perceptive Optimization to endow each adapter with foresight beyond its local objective; 3) Function-Oriented Adaptive Tuning to automatically identify the optimal fine-tuning starting point. Extensive experiments on multiple benchmarks demonstrate the superiority of ChainFed over existing methods, boosting average accuracy by up to 46.46\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07396v1">SHIELD: A Segmented Hierarchical Memory Architecture for Energy-Efficient LLM Inference on Edge NPUs</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) inference on edge Neural Processing Units (NPUs) is fundamentally constrained by limited on-chip memory capacity. Although high-density embedded DRAM (eDRAM) is attractive for storing activation workspaces, its periodic refresh consumes substantial energy. Prior work has primarily focused on reducing off-chip traffic or optimizing refresh for persistent Key-Value (KV) caches, while transient and error-resilient Query and Attention Output (QO) activations are largely overlooked. We propose SHIELD, a lifecycle-aware segmented eDRAM architecture that jointly exploits temporal residency and bit-level sensitivity in bfloat16 (BF16) activations. SHIELD isolates the sign and exponent fields from the mantissa, disables refresh for transient QO mantissas, and applies relaxed refresh to persistent KV mantissas. Across multiple LLMs and inference scenarios, SHIELD reduces eDRAM refresh energy by 35% relative to a standard-refresh baseline while preserving accuracy on WikiText-2, PIQA, and ARC-Easy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06799v1">Beyond Accuracy: Diagnosing Algebraic Reasoning Failures in LLMs Across Nine Complexity Dimensions</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Under Review as a conference paper at COLM 2026
    </div>
    <details class="paper-abstract">
      Algebraic reasoning remains one of the most informative stress tests for large language models, yet current benchmarks provide no mechanism for attributing failure to a specific cause. When a model fails an algebraic problem, a single accuracy score cannot reveal whether the expression was too deeply nested, the operator too uncommon, the intermediate state count too high, or the dependency chain too long. Prior work has studied individual failure modes in isolation, but no framework has varied each complexity factor independently under strict experimental control. No prior system has offered automatic generation and verification of problems of increasing complexity to track model progress over time. We introduce a nine-dimension algebraic complexity framework in which each factor is varied independently while all others are held fixed, with problem generation and verification handled by a parametric pipeline requiring no human annotation. Each dimension is grounded in a documented LLM failure mode and captures a structurally distinct aspect of algebraic difficulty, including expression nesting depth, simultaneous intermediate result count, sub-expression complexity, operator hardness, and dependent reasoning chain length. We evaluated seven instruction-tuned models spanning 8B to 235B parameters across all nine dimensions and find that working memory is the dominant scale-invariant bottleneck. Every model collapses between 20 and 30 parallel branches regardless of parameter count, pointing to a hard architectural constraint rather than a solvable capacity limitation. Our analysis further identifies a minimal yet diagnostically sufficient subset of five dimensions that together span the full space of documented algebraic failure modes, providing a complete complexity profile of a model's algebraic reasoning capacity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11663v2">SpecQuant: Spectral Decomposition and Adaptive Truncation for Ultra-Low-Bit LLMs Quantization</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Accepted at AAAI 2026
    </div>
    <details class="paper-abstract">
      The emergence of accurate open large language models (LLMs) has sparked a push for advanced quantization techniques to enable efficient deployment on end-user devices. In this paper, we revisit the challenge of extreme LLM compression -- targeting ultra-low-bit quantization for both activations and weights -- from a Fourier frequency domain perspective. We propose SpecQuant, a two-stage framework that tackles activation outliers and cross-channel variance. In the first stage, activation outliers are smoothed and transferred into the weight matrix to simplify downstream quantization. In the second stage, we apply channel-wise low-frequency Fourier truncation to suppress high-frequency components while preserving essential signal energy, improving quantization robustness. Our method builds on the principle that most of the weight energy is concentrated in low-frequency components, which can be retained with minimal impact on model accuracy. To enable runtime adaptability, we introduce a lightweight truncation module during inference that adjusts truncation thresholds based on channel characteristics. On LLaMA-3 8B, SpecQuant achieves 4-bit quantization for both weights and activations, narrowing the zero-shot accuracy gap to only 1.5% compared to full precision, while delivering 2 times faster inference and 3times lower memory usage. Code will be available at https://github.com/Kishon-zzx/SpecQuant.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.18018v2">Lost in Cultural Translation: Do LLMs Struggle with Math Across Cultural Contexts?</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      We demonstrate that large language models' (LLMs) mathematical reasoning is culturally sensitive: testing 14 models from Anthropic, OpenAI, Google, Meta, DeepSeek, Mistral, and Microsoft across six culturally adapted variants of the GSM8K benchmark, we find accuracy drops ranging from 0.3% (Claude 3.5 Sonnet) to 5.9% (LLaMA 3.1-8B) when math problems are embedded in unfamiliar cultural contexts--even when the underlying mathematical logic remains unchanged. These statistically significant performance reductions (p < 0.01, confirmed through McNemar tests) reveal that mathematical reasoning in LLMs is not culturally neutral. To create these variants for Haiti, Moldova, Pakistan, Solomon Islands, Somalia, and Suriname, we systematically replaced cultural entities (names, foods, places, etc.) in 1,198 GSM8K questions while preserving all mathematical operations and numerical values. Our quantitative error analysis of 18,887 instances reveals that cultural adaptation affects broader reasoning patterns, with mathematical reasoning errors comprising 54.7% and calculation errors 34.5% of failures. Interestingly, cultural familiarity can enhance performance: Mistral Saba outperforms some larger models on Pakistan-adapted problems due to Middle Eastern and South Asian training data exposure. This study underscores the need for more diverse training data to ensure robust LLM performance across global contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07394v1">Flux Attention: Context-Aware Hybrid Attention for Efficient LLMs Inference</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      The quadratic computational complexity of standard attention mechanisms presents a severe scalability bottleneck for LLMs in long-context scenarios. While hybrid attention mechanisms combining Full Attention (FA) and Sparse Attention (SA) offer a potential solution, existing methods typically rely on static allocation ratios that fail to accommodate the variable retrieval demands of different tasks. Furthermore, head-level dynamic sparsity often introduces severe computational load imbalance and synchronization long-tails, which hinder hardware acceleration during autoregressive decoding. To bridge this gap, we introduce Flux Attention, a context-aware framework that dynamically optimizes attention computation at the layer level. By integrating a lightweight Layer Router into frozen pretrained LLMs, the proposed method adaptively routes each layer to FA or SA based on the input context. This layer-wise routing preserves high-fidelity information retrieval while ensuring contiguous memory access, translating theoretical computational reductions into practical wall-clock speedups. As a parameter-efficient approach, our framework requires only 12 hours of training on 8$\times$A800 GPUs. Extensive experiments across multiple long-context and mathematical reasoning benchmarks demonstrate that Flux Attention achieves a superior trade-off between performance and inference speed compared with baseline models, with speed improvements of up to $2.8\times$ and $2.0\times$ in the prefill and decode stages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06763v1">Improving Random Testing via LLM-powered UI Tarpit Escaping for Mobile Apps</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Random GUI testing is a widely-used technique for testing mobile apps. However, its effectiveness is limited by the notorious issue -- UI exploration tarpits, where the exploration is trapped in local UI regions, thus impeding test coverage and bug discovery. In this experience paper, we introduce LLM-powered random GUI Testing, a novel hybrid testing approach to mitigating UI tarpits during random testing. Our approach monitors UI similarity to identify tarpits and query LLMs to suggest promising events for escaping the encountered tarpits. We implement our approach on top of two different automated input generation (AIG) tools for mobile apps: (1) HybridMonkey upon Monkey, a state-of-the-practice tool; and (2) HybridDroidbot upon Droidbot, a state-of-the-art tool. We evaluated them on 12 popular, real-world apps. The results show that HybridMonkey and HybridDroidbot outperform all baselines, achieving average coverage improvements of 54.8% and 44.8%, respectively, and detecting the highest number of unique crashes. In total, we found 75 unique bugs, including 34 previously unknown bugs. To date, 26 bugs have been confirmed and fixed. We also applied HybridMonkey on WeChat, a popular industrial app with billions of monthly active users. HybridMonkey achieved higher activity coverage and found more bugs than random testing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.28239v3">A Switch-Centric In-Network Architecture for Accelerating LLM Inference in Shared-Memory Network</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      In-network computing techniques, exemplified by NVLink SHARP (NVLS), offer a promising approach to addressing the communication bottlenecks in LLM inference by offloading collective operations such as All-Reduce to switches. However, the accelerator-centric architecture of NVLS suffers from two fundamental limitations: 1) it relies on GPU load instructions to trigger in-switch reduction, which means that the data reduced in the switch must be transferred back to the initiating GPU rather than being broadcast directly, thereby introducing unnecessary communication overhead; 2) due to its architectural constraints, NVLS cannot offload operators that are not decomposable into memory-semantic instructions, such as the in-network quantization (INQ) proposed in this work. As a result, All-Reduce in NVLS during inference still operates at 16-bit precision, leading to substantial bandwidth waste. To address these limitations, we propose SCIN, the first switch-centric in-network architecture for multi-accelerator shared-memory networks, enabling both low-latency and high-bandwidth All-Reduce. Specifically, we introduce an in-switch accelerator (ISA) capable of directly accessing the memory regions in attached accelerators for in-network processing, together with a co-designed communication fabric that enables such access with negligible protocol overhead. SCIN delivers lower All-Reduce latency than NVLS by eliminating redundant data movement. Moreover, SCIN enables INQ for All-Reduce, reducing its precision to 8 bits and nearly doubling bandwidth with negligible accuracy loss. We also present a multi-FPGA prototype of SCIN to validate its feasibility and effectiveness. Simulation results for an 8-GPU system show that our design accelerates All-Reduce by up to 8.7x for small messages and 3.8x for large messages, yielding up to 1.74x TTFT speedup and 1.34x TPOT speedup on LLaMA-2 models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.03044v2">JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Xiaodong He is the corresponding author
    </div>
    <details class="paper-abstract">
      We introduce JoyAI-LLM Flash, an efficient Mixture-of-Experts (MoE) language model designed to redefine the trade-off between strong performance and token efficiency in the sub-50B parameter regime. JoyAI-LLM Flash is pretrained on a massive corpus of 20 trillion tokens and further optimized through a rigorous post-training pipeline, including supervised fine-tuning (SFT), Direct Preference Optimization (DPO), and large-scale reinforcement learning (RL) across diverse environments. To improve token efficiency, JoyAI-LLM Flash strategically balances \emph{thinking} and \emph{non-thinking} cognitive modes and introduces FiberPO, a novel RL algorithm inspired by fibration theory that decomposes trust-region maintenance into global and local components, providing unified multi-scale stability control for LLM policy optimization. To enhance architectural sparsity, the model comprises 48B total parameters while activating only 2.7B parameters per forward pass, achieving a substantially higher sparsity ratio than contemporary industry leading models of comparable scale. To further improve inference throughput, we adopt a joint training-inference co-design that incorporates dense Multi-Token Prediction (MTP) and Quantization-Aware Training (QAT). We release the checkpoints for both JoyAI-LLM-48B-A3B Base and its post-trained variants on Hugging Face to support the open-source community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06756v1">How Long Reasoning Chains Influence LLMs' Judgment of Answer Factuality</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 ACL2026 Main
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) has been widely adopted as a scalable surrogate for human evaluation, yet such judges remain imperfect and susceptible to surface-level biases. One possible reason is that these judges lack sufficient information in assessing answer correctness. With the rise of reasoning-capable models, exposing a generator's reasoning content to the judge provides richer information and is a natural candidate for improving judgment accuracy. However, its actual impact on judge behavior remains understudied. In this paper, we systematically investigate how access to reasoning chains affects LLM-based judgment across factual question answering (QA) and mathematical reasoning benchmarks. We find that weak judges are easily swayed by reasoning presence, frequently accepting incorrect answers accompanied by fluent reasoning, while strong judges can partially leverage reasoning as informative evidence. Nevertheless, even strong judges are misled by seemingly high-quality reasoning chains. Controlled experiments further reveal that both fluency and factuality of reasoning chains are critical signals driving judge decisions. These findings highlight the need for more robust LLM judges that can distinguish genuine reasoning quality from superficial fluency when evaluating modern reasoning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06755v1">Babbling Suppression: Making LLMs Greener One Token at a Time</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Context: Large Language Models (LLMs) are increasingly used in modern software development, aiding in code generation, code completion, and refactoring through AI-powered assistants. While they accelerate development workflows, they often produce extraneous output, referred to as "babbling", which incurs additional cognitive, economic, and energy costs. Objective: This work investigates the problem of babbling in LLM-based code generation and proposes a practical, model-agnostic approach to reduce unnecessary output without compromising solution accuracy. Method: We introduce Babbling Suppression (BS), a method that integrates test execution into the LLM generation process by evaluating intermediate outputs and terminating generation once a solution passes all tests. This prevents excessive token generation while having no impact on model accuracy. An empirical study was conducted across two Python and two Java benchmarks, targeting four 3-4B parameter models and six 6-7B parameter models. Results: Our findings show that babbling occurs across all tested models, with higher frequency in Java than in Python. Applying BS significantly reduces energy consumption by up to 65% for Python and 62% for Java in models prone to babbling. Across 40 model-benchmark pairs, 29 showed reduced mean energy consumption, with reductions exceeding 20% in 22 cases. Generated token count decreased in 35 pairs, while the GPU energy-per-token overhead of BS remained below 10% for 26 pairs, decreased for 2, and reached a maximum of 24%, yielding net energy savings in most cases. Implications: BS can make AI-assisted programming more efficient and sustainable by reducing energy consumption and minimizing cognitive effort by developers. Its model-agnostic design allows easy integration, suggesting broad applicability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06753v1">Select-then-Solve: Paradigm Routing as Inference-Time Optimization for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      When an LLM-based agent improves on a task, is the gain from the model itself or from the reasoning paradigm wrapped around it? We study this question by comparing six inference-time paradigms, namely Direct, CoT, ReAct, Plan-Execute, Reflection, and ReCode, across four frontier LLMs and ten benchmarks, yielding roughly 18,000 runs. We find that reasoning structure helps dramatically on some tasks but hurts on others: ReAct improves over Direct by 44pp on GAIA, while CoT degrades performance by 15pp on HumanEval. No single paradigm dominates, and oracle per-task selection beats the best fixed paradigm by 17.1pp on average. Motivated by this complementarity, we propose a select-then-solve approach: before answering each task, a lightweight embedding-based router selects the most suitable paradigm. Across four models, the router improves average accuracy from 47.6% to 53.1%, outperforming the best fixed paradigm at 50.3% by 2.8pp and recovering up to 37% of the oracle gap. In contrast, zero-shot self-routing only works for GPT-5 at 67.1% and fails for weaker models, all trailing the learned router. Our results argue that reasoning paradigm selection should be a per-task decision made by a learned router, not a fixed architectural choice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06742v1">Evaluating LLM-Based 0-to-1 Software Generation in End-to-End CLI Tool Scenarios</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are driving a shift towards intent-driven development, where agents build complete software from scratch. However, existing benchmarks fail to assess this 0-to-1 generation capability due to two limitations: reliance on predefined scaffolds that ignore repository structure planning, and rigid white-box unit testing that lacks end-to-end behavioral validation. To bridge this gap, we introduce CLI-Tool-Bench, a structure-agnostic benchmark for evaluating the ground-up generation of Command-Line Interface (CLI) tools. It features 100 diverse real-world repositories evaluated via a black-box differential testing framework. Agent-generated software is executed in sandboxes, comparing system side effects and terminal outputs against human-written oracles using multi-tiered equivalence metrics. Evaluating seven state-of-the-art LLMs, we reveal that top models achieve under 43% success, highlighting the ongoing challenge of 0-to-1 generation. Furthermore, higher token consumption does not guarantee better performance, and agents tend to generate monolithic code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.18196v2">Contrastive Decoding Mitigates Score Range Bias in LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 To appear at ACL 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are commonly used as evaluators in various applications, but the reliability of the outcomes remains a challenge. One such challenge is using LLMs-as-judges for direct assessment, i.e., assigning scores from a specified range without any references. Focusing on summarization, we first show that this challenge stems from LLM judge outputs being associated with score range bias, i.e., LLM judge outputs are highly sensitive to pre-defined score ranges. We also show that similar biases exist among models from the same family. We then mitigate this bias through contrastive decoding, achieving up to 11.7% relative improvement on average in Spearman correlation with human judgments across different score ranges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06736v1">SQLStructEval: Structural Evaluation of LLM Text-to-SQL Generation</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 17 pages, including figures and tables
    </div>
    <details class="paper-abstract">
      Despite strong performance on Text-to-SQL benchmarks, it remains unclear whether LLM-generated SQL programs are structurally reliable. In this work, we investigate the structural behavior of LLM-generated SQL queries and introduce SQLStructEval, a framework for analyzing program structures through canonical abstract syntax tree (AST) representations. Our experiments on the Spider benchmark show that modern LLMs often produce structurally diverse queries for the same input, even when execution results are correct, and that such variance is frequently triggered by surface-level input changes such as paraphrases or schema presentation. We further show that generating queries in a structured space via a compile-style pipeline can improve both execution accuracy and structural consistency. These findings suggest that structural reliability is a critical yet overlooked dimension for evaluating LLM-based program generation systems. Our code is available at https://anonymous.4open.science/r/StructEval-2435.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.16703v4">ShadowNPU: System and Algorithm Co-design for NPU-Centric On-Device LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 To Appear at MobiSys'26
    </div>
    <details class="paper-abstract">
      On-device running Large Language Models (LLMs) is nowadays a critical enabler towards preserving user privacy. We observe that the attention operator falls back from the special-purpose NPU to the general-purpose CPU/GPU because of quantization sensitivity in state-of-the-art frameworks. This fallback results in a degraded user experience and increased complexity in system scheduling. To this end, this paper presents shadowAttn, a system-algorithm codesigned sparse attention module with minimal reliance on CPU/GPU by only sparsely calculating the attention on a tiny portion of tokens. The key idea is to hide the overhead of estimating the important tokens with a NPU-based pilot compute. Further, shadowAttn proposes insightful techniques such as NPU compute graph bucketing, head-wise NPU-CPU/GPU pipeline and per-head fine-grained sparsity ratio to achieve high accuracy and efficiency. shadowAttn delivers the best performance with highly limited CPU/GPU resource; it requires much less CPU/GPU resource to deliver on-par performance of SoTA frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06723v1">Fine-grained Approaches for Confidence Calibration of LLMs in Automated Code Revision</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      In today's AI-assisted software engineering landscape, developers increasingly depend on LLMs that are highly capable, yet inherently imperfect. The tendency of these models to produce incorrect outputs can reduce developer productivity. To this end, a canonical mitigation method is to provide calibrated confidence scores that faithfully reflect their likelihood of correctness at the instance-level. Such information allows users to make immediate decisions regarding output acceptance, abstain error-prone outputs, and better align their expectations with the model's capabilities. Since post-trained LLMs do not inherently produce well-calibrated confidence scores, researchers have developed post-hoc calibration methods, with global Platt-scaling of sequence-level confidence scores proving effective in many generative software engineering tasks but remaining unreliable or unexplored for automated code revision (ACR) tasks such as program repair, vulnerability repair, and code refinement. We hypothesise that the coarse-grained nature of this conventional method makes it ill-suited for ACR tasks, where correctness is often determined by local edit decisions and miscalibration can be sample-dependent, thereby motivating fine-grained confidence calibration. To address this, our study proposes local Platt-scaling applied separately to three different fine-grained confidence scores. Through experiments across 3 separate tasks and correctness metrics, as well as 14 different models of various sizes, we find that fine-grained confidence scores consistently achieve lower calibration error across a broader range of probability intervals, and this effect is further amplified when global Platt-scaling is applied. Our proposed approaches offer a practical solution to eliciting well-calibrated confidence scores, enabling more trustworthy and streamlined usage of imperfect models in ACR tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16206v3">Computer Environments Elicit General Agentic Intelligence in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Project Page: https://llm-in-sandbox.github.io
    </div>
    <details class="paper-abstract">
      Agentic intelligence in large language models (LLMs) requires not only model intrinsic capabilities but also interactions with external environments. Equipping LLMs with computers now represents a prevailing trend. However, the computer environment's intrinsic value has not been systematically investigated, particularly its potential to elicit general capabilities. Here we introduce LLM-in-Sandbox, which virtualizes the computer as a code sandbox with only basic functionalities, and demonstrate that this minimal setting elicits computer-based meta-capabilities for general task solving: external resource access, file management, and code execution. Without additional training, strong models achieve substantial gains (up to 15.5%) across mathematics, physics, chemistry, biomedicine, long-context understanding, and instruction following, while reducing token consumption by up to 8 times. Furthermore, we develop LLM-in-Sandbox-RL to train models exclusively on non-agentic data within the sandbox, empowering weaker models to harness the environment and internalize these interactions. Our results demonstrate that computer environments elicit general intelligence, yield efficiency gains, and can be harnessed through training, serving as a promising foundation for generalist agents.
    </details>
</div>
