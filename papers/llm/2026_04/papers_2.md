# llm - 2026_04

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20310v1">Odor Maps from the LLM-derived similarity scores</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 9 pages, 7 figures, Under review
    </div>
    <details class="paper-abstract">
      The application of large language models (LLMs) to OdorSpace analysis attracts growing interest. Recent studies have explored the comparison of sensory evaluation spaces derived from LLMs with odor character profiles in the Dravnieks' dataset. In this study, we calculated pairwise distances of odor descriptors using three distance measures and statistically compared these LLM-derived similarities with distances derived from the original data. Next, we extended this approach to odor names (ingredients). Statistical comparison revealed that LLMs can infer odor similarity to some degree, suggesting the potential of odor maps generated from these similarity data. Applying this approach, we generated an odor map of essential oils. It demonstrates that essential oils within the same group are closely located in the odor map, suggesting that the proximity in the odor map corresponds to human evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.00667v4">zkCraft: Prompt-Guided LLM as a Zero-Shot Mutation Pattern Oracle for TCCT-Powered ZK Fuzzing</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 36 pages, 12 figures, 9 tables
    </div>
    <details class="paper-abstract">
      Zero-knowledge circuits enable privacy-preserving and scalable systems but are difficult to implement correctly due to the tight coupling between witness computation and circuit constraints. We present zkCraft, a practical framework that combines deterministic, R1CS-aware localization with proof-bearing search to detect semantic inconsistencies. zkCraft encodes candidate constraint edits into a single Row-Vortex polynomial and replaces repeated solver queries with a Violation IOP that certifies the existence of edits together with a succinct proof. Deterministic LLM-driven mutation templates bias exploration toward edge cases while preserving auditable algebraic verification. Evaluation on real Circom code shows that proof-bearing localization detects diverse under- and over-constrained faults with low false positives and reduces costly solver interaction. Our approach bridges formal verification and automated debugging, offering a scalable path for robust ZK circuit development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20304v1">LLM-guided phase diagram construction through high-throughput experimentation</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 39 pages
    </div>
    <details class="paper-abstract">
      Constructing phase diagrams for multicomponent alloys requires extensive experimental measurements and is a time-consuming task. Here we investigate whether large language models (LLMs) can guide experimental planning for phase diagram construction. In our framework, a general-purpose LLM serves as the experimental planner, suggesting compositions for measurement at each cycle in a closed loop with high-throughput synthesis and X-ray diffraction phase identification. Using this framework, we experimentally constructed the ternary phase diagram of the Co-Al-Ge system at 900 degree C through iterative synthesis and characterization. We compared two strategies that differ in how the initial compositions are selected: one uses predictions from a domain-specific LLM trained on phase diagram data (aLLoyM), while the other relies solely on the general-purpose LLM. The two strategies exhibited complementary strengths. aLLoyM directed the initial measurements toward compositionally complex regions in the interior of the ternary diagram, enabling the earliest discovery of all three novel phases that form only in the ternary system. In contrast, the general-purpose LLM adopted a textbook-like approach which efficiently identified a larger number of phases in fewer cycles. In addition, a simulated benchmark comparing the LLM against conventional machine learning confirmed that the LLM achieves more efficient exploration. The results demonstrate that LLMs have high potential as experimental planners for phase diagram construction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14116v2">TREX: Automating LLM Fine-tuning via Agent-Driven Tree-based Exploration</a></div>
    <div class="paper-meta">
      📅 2026-04-22
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have empowered AI research agents to perform isolated scientific tasks, automating complex, real-world workflows, such as LLM training, remains a significant challenge. In this paper, we introduce TREX, a multi-agent system that automates the entire LLM training life-cycle. By orchestrating collaboration between two core modules-the Researcher and the Executor-the system seamlessly performs requirement analysis, open-domain literature and data research, formulation of training strategies, preparation of data recipes, and model training and evaluation. The multi-round experimental process is modeled as a search tree, enabling the system to efficiently plan exploration paths, reuse historical results, and distill high-level insights from iterative trials. To evaluate the capability of automated LLM training, we construct FT-Bench, a benchmark comprising 10 tasks derived from real-world scenarios, ranging from optimizing fundamental model capabilities to enhancing performance on domain-specific tasks. Experimental results demonstrate that the TREX agent consistently optimizes model performance on target tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20273v1">ActuBench: A Multi-Agent LLM Pipeline for Generation and Evaluation of Actuarial Reasoning Tasks</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 19 pages, 4 figures, 4 tables
    </div>
    <details class="paper-abstract">
      We present ActuBench, a multi-agent LLM pipeline for the automated generation and evaluation of advanced actuarial assessment items aligned with the International Actuarial Association (IAA) Education Syllabus. The pipeline separates four LLM roles by adapter: one agent drafts items, one constructs distractors, a third independently verifies both stages and drives bounded one-shot repair loops, and a cost-optimized auxiliary agent handles Wikipedia-note summarization and topic labelling. The items, per-model responses and complete leaderboard are published as a browsable web interface at https://actubench.de/en/, allowing readers and practitioners to inspect individual items without a repository checkout. We evaluate 50 language models from eight providers on two complementary benchmarks -- 100 empirically hardest multiple-choice items and 100 open-ended items scored by an LLM judge -- and report three headline findings. First, multi-agent verification is load-bearing: the independent verifier flags a majority of drafted items on first pass, most of which the one-shot repair loop resolves. Second, locally-hosted open-weights inference sits on the cost-performance Pareto front: a Gemma~4 model running on consumer hardware and a Cerebras-hosted 120B open-weights model dominate the near-zero-cost region, with the latter within one item of the top of the leaderboard. Third, MCQ and LLM-as-Judge rankings differ meaningfully: the MCQ scaffold inflates the performance ceiling, and Judge-mode evaluation is needed to discriminate at the frontier.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20261v1">Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 16 pages (including appendix), 4 main figures, 15 tables. Accepted to ACL 2026
    </div>
    <details class="paper-abstract">
      Automated feature generation extracts informative features from raw tabular data without manual intervention and is crucial for accurate, generalizable machine learning. Traditional methods rely on predefined operator libraries and cannot leverage task semantics, limiting their ability to produce diverse, high-value features for complex tasks. Recent Large Language Model (LLM)-based approaches introduce richer semantic signals, but still suffer from a restricted feature space due to fixed generation patterns and from the absence of feedback from the learning objective. To address these challenges, we propose a Memory-Augmented LLM-based Multi-Agent System (\textbf{MALMAS}) for automated feature generation. MALMAS decomposes the generation process into agents with distinct responsibilities, and a Router Agent activates an appropriate subset of agents per iteration, further broadening exploration of the feature space. We further integrate a memory module comprising procedural memory, feedback memory, and conceptual memory, enabling iterative refinement that adaptively guides subsequent feature generation and improves feature quality and diversity. Extensive experiments on multiple public datasets against state-of-the-art baselines demonstrate the effectiveness of our approach. The code is available at https://github.com/fxdong24/MALMAS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19245v2">Talking to a Know-It-All GPT or a Second-Guesser Claude? How Repair reveals unreliable Multi-Turn Behavior in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 Preprint accepted at ACL Main Conference 2026
    </div>
    <details class="paper-abstract">
      Repair, an important resource for resolving trouble in human-human conversation, remains underexplored in human-LLM interaction. In this study, we investigate how LLMs engage in the interactive process of repair in multi-turn dialogues around solvable and unsolvable math questions. We examine whether models initiate repair themselves and how they respond to user-initiated repair. Our results show strong differences across models: reactions range from being almost completely resistant to (appropriate) repair attempts to being highly susceptible and easily manipulated. We further demonstrate that once conversations extend beyond a single turn, model behavior becomes more distinctive and less predictable across systems. Overall, our findings indicate that each tested LLM exhibits its own characteristic form of unreliability in the context of repair.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09046v2">FlexServe: A Fast and Secure LLM Serving System for Mobile Devices with Flexible Resource Isolation</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 13 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Device-side Large Language Models (LLMs) have witnessed explosive growth, offering higher privacy and availability compared to cloud-side LLMs. During LLM inference, both model weights and user data are valuable, and attackers may even compromise the OS kernel to steal them. ARM TrustZone is the de facto hardware-based isolation technology on mobile devices, used to protect sensitive applications from a compromised OS. However, protecting LLM inference with TrustZone incurs significant overhead due to its inflexible isolation of memory and the NPU. To address these challenges, this paper introduces FlexServe, a fast and secure LLM serving system for mobile devices. It first introduces a Flexible Resource Isolation mechanism to construct Flexible Secure Memory (Flex-Mem) and Flexible Secure NPU (Flex-NPU). Both memory pages and the NPU can be efficiently switched between unprotected and protected modes. Based on these mechanisms, FlexServe designs a fast and secure LLM inference framework within TrustZone's secure world. The LLM-Aware Memory Management and Secure Inference Pipeline are introduced to accelerate inference. A Multi-Model Scheduler is proposed to optimize multi-model workflows. We implement a prototype of FlexServe and compare it with two TrustZone-based strawman designs. The results show that FlexServe achieves an average $10.05\times$ speedup in Time to First Token (TTFT) compared to the strawman, and an average $2.44\times$ TTFT speedup compared to an optimized strawman with pipeline and secure NPU enabled. For multi-model agent workflows, the end-to-end speedup is up to $24.30\times$ and $4.05\times$ compared to the strawman and optimized strawman, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20244v1">Hybrid Policy Distillation for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 WIP
    </div>
    <details class="paper-abstract">
      Knowledge distillation (KD) is a powerful paradigm for compressing large language models (LLMs), whose effectiveness depends on intertwined choices of divergence direction, optimization strategy, and data regime. We break down the design of existing KD methods and present a unified view that establishes connections between them, reformulating KD as a reweighted log-likelihood objective at the token level. We further propose Hybrid Policy Distillation (HPD), which integrates the complementary advantages of forward and reverse KL to balance mode coverage and mode-seeking, and combines off-policy data with lightweight, approximate on-policy sampling. We validate HPD on long-generation math reasoning as well as short-generation dialogue and code tasks, demonstrating improved optimization stability, computational efficiency, and final performance across diverse model families and scales. The code related to this work is available at https://github.com/zwhong714/Hybrid-Policy-Distillation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20924v1">Clinically Interpretable Sepsis Early Warning via LLM-Guided Simulation of Temporal Physiological Dynamics</a></div>
    <div class="paper-meta">
      📅 2026-04-22
    </div>
    <details class="paper-abstract">
      Timely and interpretable early warning of sepsis remains a major clinical challenge due to the complex temporal dynamics of physiological deterioration. Traditional data-driven models often provide accurate yet opaque predictions, limiting physicians' confidence and clinical applicability. To address this limitation, we propose a Large Language Model (LLM)-guided temporal simulation framework that explicitly models physiological trajectories prior to disease onset for clinically interpretable prediction. The framework consists of a spatiotemporal feature extraction module that captures dynamic dependencies among multivariate vital signs, a Medical Prompt-as-Prefix module that embeds clinical reasoning cues into LLMs, and an agent-based post-processing component that constrains predictions within physiologically plausible ranges. By first simulating the evolution of key physiological indicators and then classifying sepsis onset, our model offers transparent prediction mechanisms that align with clinical judgment. Evaluated on the MIMIC-IV and eICU databases, the proposed method achieves superior AUC scores (0.861-0.903) across 24-4-hour pre-onset prediction tasks, outperforming conventional deep learning and rule-based approaches. More importantly, it provides interpretable trajectories and risk trends that can assist clinicians in early intervention and personalized decision-making in intensive care environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.19729v2">Amoeba: Runtime Tensor Parallel Transformation for LLM Inference Services</a></div>
    <div class="paper-meta">
      📅 2026-04-22
    </div>
    <details class="paper-abstract">
      In Large Language Model (LLM) inference services, it is challenging to make a parallelism strategy configuration, to efficiently process the requests of variance context lengths. Requests of long context require high degree of parallelism to provide more memory for Key-Value (KV) Cache, while requests of short context prefer low degree of parallelism to increase concurrency, thus improving throughput. To maintain high throughput while supporting large context lengths on demand, we propose Amoeba, a runtime Tensor Parallel (TP) transformation for online LLM inference services, which adaptively adjusts the TP of running instances to align with the dynamics of incoming requests. Evaluations using real-world traces show that Amoeba improves throughput by 1.75x-6.57x compared to state-of-the-art solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20211v1">Towards Secure Logging: Characterizing and Benchmarking Logging Code Security Issues with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 Accepted at FSE 2026 Research Papers Track
    </div>
    <details class="paper-abstract">
      Logging code plays an important role in software systems by recording key events and behaviors, which are essential for debugging and monitoring. However, insecure logging practices can inadvertently expose sensitive information or enable attacks such as log injection, posing serious threats to system security and privacy. Prior research has examined general defects in logging code, but systematic analysis of logging code security issues remains limited, particularly in leveraging LLMs for detection and repair. In this paper, we derive a comprehensive taxonomy of logging code security issues, encompassing four common issue categories and 10 corresponding patterns. We further construct a benchmark dataset with 101 real-world logging security issue reports that have been manually reviewed and annotated. We then propose an automated framework that incorporates various contextual knowledge to evaluate LLMs' capabilities in detecting and repairing logging security issues. Our experimental results reveal a notable disparity in performance: while LLMs are moderately effective at detecting security issues (e.g., the accuracy ranges from 12.9% to 52.5% on average), they face noticeable challenges in reliably generating correct code repairs. We also find that the issue description alone improves the LLMs' detection accuracy more than the security pattern explanation or a combination of both. Overall, our findings provide actionable insights for practitioners and highlight the potential and limitations of current LLMs for secure logging.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20193v1">LLM-Guided Safety Agent for Edge Robotics with an ISO-Compliant Perception-Compute-Control Architecture</a></div>
    <div class="paper-meta">
      📅 2026-04-22
    </div>
    <details class="paper-abstract">
      Ensuring functional safety in human-robot interaction is challenging because AI perception is inherently probabilistic, whereas industrial standards require deterministic behavior. We present an LLM-guided safety agent for edge robotics, built on an ISO-compliant low-latency perception-compute-control architecture. Our method translates natural-language safety regulations into executable predicates and deploys them through a redundant heterogeneous edge runtime. For fault-tolerant closed-loop execution under edge constraints, we adopt a symmetric dual-modular redundancy design with parallel independent execution for low-latency perception, computation, and control. We prototype the system on a dual-RK3588 platform and evaluate it in representative human-robot interaction scenarios. The results demonstrate a practical edge implementation path toward ISO 13849 Category 3 and PL d using cost-effective hardware, supporting practical deployment of safety-critical embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20179v1">Taint-Style Vulnerability Detection and Confirmation for Node.js Packages Using LLM Agent Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 19 pages, 6 figures
    </div>
    <details class="paper-abstract">
      The rapidly evolving Node$.$js ecosystem currently includes millions of packages and is a critical part of modern software supply chains, making vulnerability detection of Node$.$js packages increasingly important. However, traditional program analysis struggles in this setting because of dynamic JavaScript features and the large number of package dependencies. Recent advances in large language models (LLMs) and the emerging paradigm of LLM-based agents offer an alternative to handcrafted program models. This raises the question of whether an LLM-centric, tool-augmented approach can effectively detect and confirm taint-style vulnerabilities (e.g., arbitrary command injection) in Node$.$js packages. We implement LLMVD$.$js, a multi-stage agent pipeline to scan code, propose vulnerabilities, generate proof-of-concept exploits, and validate them through lightweight execution oracles; and systematically evaluate its effectiveness in taint-style vulnerability detection and confirmation in Node$.$js packages without dedicated static/dynamic analysis engines for path derivation. For packages from public benchmarks, LLMVD$.$js confirms 84% of the vulnerabilities, compared to less than 22% for prior program analysis tools. It also outperforms a prior LLM-program-analysis hybrid approach while requiring neither vulnerability annotations nor prior vulnerability reports. When evaluated on a set of 260 recently released packages (without vulnerability groundtruth information), traditional tools produce validated exploits for few ($\leq 2$) packages, while LLMVD$.$js generates validated exploits for 36 packages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20168v1">Duluth at SemEval-2026 Task 6: DeBERTa with LLM-Augmented Data for Unmasking Political Question Evasions</a></div>
    <div class="paper-meta">
      📅 2026-04-22
    </div>
    <details class="paper-abstract">
      This paper presents the Duluth approach to SemEval-2026 Task 6 on CLARITY: Unmasking Political Question Evasions. We address Task 1 (clarity-level classification) and Task 2 (evasion-level classification), both of which involve classifying question--answer pairs from U.S.\ presidential interviews using a two-level taxonomy of response clarity. Our system is based on DeBERTa-V3-base, extended with focal loss, layer-wise learning rate decay, and boolean discourse features. To address class imbalance in the training data, we augment minority classes using synthetic examples generated by Gemini 3 and Claude Sonnet 4.5. Our best configuration achieved a Macro F1 of 0.76 on the Task 1 evaluation set, placing 8th out of 40 teams. The top-ranked system (TeleAI) achieved 0.89, while the mean score across participants was 0.70. Error analysis reveals that the dominant source of misclassification is confusion between Ambivalent and Clear Reply responses, a pattern that mirrors disagreements among human annotators. Our findings demonstrate that LLM-based data augmentation can meaningfully improve minority-class recall on nuanced political discourse tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20919v1">DiP-SD: Distributed Pipelined Speculative Decoding for Efficient LLM Inference at the Edge</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 Accepted by 2026 IEEE 103rd Vehicular Technology Conference (VTC2026-Spring)
    </div>
    <details class="paper-abstract">
      Speculative decoding has emerged as a promising technique for large language model (LLM) inference by accelerating autoregressive decoding via draft-then-verify. This paper studies a new edge scenario with multi-user inference, where draft tokens are generated locally on devices and subsequently offloaded to a centralized edge server for batch verification. The key challenge is to sustain high throughput under coupled decisions of (i) batching and pipeline scheduling and (ii) per user draft token length. We propose DiP-SD, which exploits two complementary parallelism dimensions: device-level distributed drafting and phase-level draft-verify pipelining. We formulate a throughput-maximization objective, defined as the expected number of accepted tokens per unit time, and jointly optimize the number of batches, user-to-batch assignment, and integer draft lengths. To solve the resulting fractional mixed-integer program, DiP-SD scans the batch number and iteratively alternates between an association subproblem and a draft-length subproblem. Numerical results under a Qwen3-1.7B/Qwen3-32B device-edge deployment show that DiP-SD achieves up to 17.89x throughput over autoregressive decoding (AD) and 1.93x over AD with greedy batching.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.10109v2">LLM Agents Grounded in Self-Reports Enable General-Purpose Simulation of Individuals</a></div>
    <div class="paper-meta">
      📅 2026-04-22
    </div>
    <details class="paper-abstract">
      Machine learning can predict human behavior well when substantial structured data and well-defined outcomes are available, but these models are typically limited to specific outcomes and cannot readily be applied to new domains. We test whether large language models (LLMs) can support a more general-purpose approach by building person-specific simulations (i.e., "generative agents") grounded in self-report data. Using data from a diverse national sample of 1,052 Americans, we build agents from (i) two-hour, semi-structured interviews (elicited using the American Voices Project interview schedule), (ii) structured surveys (the General Social Survey and Big Five personality inventory), or (iii) both sources combined. On held-out General Social Survey items, agent accuracy reached 83% (interview only), 82% (surveys only), and 86% (combined) of participants' two-week test-retest consistency, compared with agents prompted only with individuals' demographics (74%). Agents predicted personality traits and behaviors in experiments with similar accuracy, and reduced disparities in accuracy across racial and ideological groups relative to demographics-only baselines. Together, these results show that LLMs agents grounded in rich qualitative or quantitative self-report data can support general-purpose simulation of individuals across outcomes, without requiring task-specific training data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15861v2">CAST: Achieving Stable LLM-based Text Analysis for Data Analytics</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 ACL 2026 Findings
    </div>
    <details class="paper-abstract">
      Text analysis of tabular data relies on two core operations: \emph{summarization} for corpus-level theme extraction and \emph{tagging} for row-level labeling. A critical limitation of employing large language models (LLMs) for these tasks is their inability to meet the high standards of output stability demanded by data analytics. To address this challenge, we introduce \textbf{CAST} (\textbf{C}onsistency via \textbf{A}lgorithmic Prompting and \textbf{S}table \textbf{T}hinking), a framework that enhances output stability by constraining the model's latent reasoning path. CAST combines (i) Algorithmic Prompting to impose a procedural scaffold over valid reasoning transitions and (ii) Thinking-before-Speaking to enforce explicit intermediate commitments before final generation. To measure progress, we introduce \textbf{CAST-S} and \textbf{CAST-T}, stability metrics for bulleted summarization and tagging, and validate their alignment with human judgments. Experiments across publicly available benchmarks on multiple LLM backbones show that CAST consistently achieves the best stability among all baselines, improving Stability Score by up to 16.2\%, while maintaining or improving output quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12078v2">Optimizing User Profiles via Contextual Bandits for Retrieval-Augmented LLM Personalization</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 Accepted to ACL 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at general-purpose tasks, yet adapting their responses to individual users remains challenging. Retrieval augmentation provides a lightweight alternative to fine-tuning by conditioning LLMs on user history records, and existing approaches typically select these records based on semantic relevance. We argue that relevance serves as an unreliable proxy for utility: a record may be semantically similar to a query yet fail to improve generation quality or even degrade it due to redundancy or conflicting information. To bridge this gap, we propose PURPLE, a contextual bandit framework that oPtimizes UseR Profiles for LLM pErsonalization. In contrast to a greedy selection of the most relevant records, PURPLE treats profile construction as an order-sensitive generation process and utilizes a Plackett-Luce ranking model to capture complex inter-record dependencies. By training with semantically rich feedback provided by the likelihood of the reference response, our method aligns retrieval directly with generation quality. Extensive experiments on nine personalization tasks demonstrate that PURPLE consistently outperforms strong heuristic and retrieval-augmented baselines in both effectiveness and efficiency, establishing a principled and scalable solution for optimizing user profiles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20140v1">HiPO: Hierarchical Preference Optimization for Adaptive Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 12 pages, 4 figures, 6 tables. Includes ablation study across Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct on 5 math reasoning benchmarks (GSM8K, MATH500, Minerva, AIME24, Gaokao2023). GPT-4.1 used for structured evaluation of reasoning quality
    </div>
    <details class="paper-abstract">
      Direct Preference Optimization (DPO) is an effective framework for aligning large language models with human preferences, but it struggles with complex reasoning tasks. DPO optimizes for the likelihood of generating preferred over dispreferred responses in their entirety and lacks the granularity to provide feedback on subsections of many-step solutions typical of reasoning tasks. Existing methods excel at either stable preference learning (e.g., DPO variants like KTO and RSO) or structured reasoning (e.g., ReMA's multi-agent RL framework, Tree of Thoughts), but fail to merge these complementary strengths. We propose HiPO (Hierarchical Preference Optimization), an extension of DPO that separates responses into reasoning segments (query clarification and context, reasoning steps, and answer) and computes loss as a weighted sum of the DPO loss for each segment. Our approach enables segment-specific training while maintaining DPO's computational efficiency and training stability. We demonstrate that for multiple 7B LLMs fine-tuned using HiPO and DPO on the Math Stack Exchange preference dataset, the models trained with HiPO outperform the others on a variety of common math benchmarks and achieve greater organization, logical flow, and consistency as measured by GPT-4.1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15141v2">ItemRAG: Item-Based Retrieval-Augmented Generation for LLM-Based Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 Published as a conference paper at SIGIR 2026 (short)
    </div>
    <details class="paper-abstract">
      Recently, large language models (LLMs) have been widely used as recommender systems, owing to their reasoning capability and effectiveness in handling cold-start items. A common approach prompts an LLM with a target user's purchase history to recommend items from a candidate set, often enhanced with retrieval-augmented generation (RAG). Most existing RAG approaches retrieve purchase histories of users similar to the target user; however, these histories often contain noisy or weakly relevant information and provide little or no useful information for candidate items. To address these limitations, we propose ItemRAG, a novel RAG approach that shifts focus from coarse user-history retrieval to fine-grained item-level retrieval. ItemRAG augments the description of each item in the target user's history or the candidate set by retrieving items relevant to each. To retrieve items not merely semantically similar but informative for recommendation, ItemRAG leverages co-purchase information alongside semantic information. Especially, through their careful combination, ItemRAG prioritizes more informative retrievals and also benefits cold-start items. Through extensive experiments, we demonstrate that ItemRAG consistently outperforms existing RAG approaches under both standard and cold-start item recommendation settings. Supplementary materials, code, and datasets are provided at https://github.com/kswoo97/ItemRAG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20131v1">Whose Story Gets Told? Positionality and Bias in LLM Summaries of Life Narratives</a></div>
    <div class="paper-meta">
      📅 2026-04-22
    </div>
    <details class="paper-abstract">
      Increasingly, studies are exploring using Large Language Models (LLMs) for accelerated or scaled qualitative analysis of text data. While we can compare LLM accuracy against human labels directly for deductive coding, or labeling text, it is more challenging to judge the ethics and effectiveness of using LLMs in abstractive methods such as inductive thematic analysis. We collaborate with psychologists to study the abstractive claims LLMs make about human life stories, asking, how does using an LLM as an interpreter of meaning affect the conclusions and perspectives of a study? We propose a summarization-based pipeline for surfacing biases in perspective-taking an LLM might employ in interpreting these life stories. We demonstrate that our pipeline can identify both race and gender bias with the potential for representational harm. Finally, we encourage the use of this analysis in future studies involving LLM-based interpretation of study participants' written text or transcribed speech to characterize a positionality portrait for the study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20915v1">Absorber LLM: Harnessing Causal Synchronization for Test-Time Training</a></div>
    <div class="paper-meta">
      📅 2026-04-22
    </div>
    <details class="paper-abstract">
      Transformers suffer from a high computational cost that grows with sequence length for self-attention, making inference in long streams prohibited by memory consumption. Constant-memory alternatives such as RNNs and SSMs compress history into states with fixed size and thus lose long-tail dependencies, while methods that memorize contexts into parameters, such as Test-Time Training (TTT), are prone to overfitting token-level projection and fail to preserve the causal effect of context in pretrained LLMs. We propose Absorber LLM, which formulates long-context retention as a self-supervised causal synchronization: after absorbing historical contexts into parameters, a contextless model should match the original model with full context on future generations. We optimize this objective by synchronizing internal behaviors of the updated model with the original one, ensuring context absorption and generalization. Experiments on long-context and streaming benchmarks show that Absorber LLM reduces inference memory and improves accuracy over prior parameter-as-memory baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07192v2">Compact Constraint Encoding for LLM Code Generation: An Empirical Study of Token Economics and Constraint Compliance</a></div>
    <div class="paper-meta">
      📅 2026-04-22
    </div>
    <details class="paper-abstract">
      LLMs used for code generation are typically guided by engineering constraints--technology choices, dependency restrictions, and architectural patterns--expressed in verbose natural language. We investigate whether compact, structured constraint headers can reduce prompt token consumption without degrading constraint compliance. Across six experimental rounds spanning 11 models, 16 benchmark tasks, and over 830 LLM invocations, we find that compact headers reduce constraint-portion tokens by approximately 71% and full-prompt tokens by 25--30%, replicated across three independent rounds. However, we detect no statistically significant differences in constraint satisfaction rate (CSR) across three encoding forms or four propagation modes; observed effect sizes are negligible (Cliff's $δ$ < 0.01, 95% CI spanning $\pm$2.6 percentage points). This null pattern holds across two models from different capability tiers. A supplementary experiment with four non-CSS tasks provides additional cross-domain support for the encoding null result. The largest observed sources of compliance variance are constraint type ($Δ$ = 9 percentage points between normal and counter-intuitive constraints) and task domain: counter-intuitive constraints opposing model defaults fail at 10--100%, while conventional constraints achieve 99%+ compliance regardless of encoding. Model self-assessments systematically overestimate compliance relative to rule-based scoring, revealing a gap between constraint understanding and execution. Under the tested conditions, the primary benefit of compact constraint encoding is token reduction rather than compliance improvement, and engineering effort toward compliance is better directed at constraint design than prompt formatting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20913v1">FairyFuse: Multiplication-Free LLM Inference on CPUs via Fused Ternary Kernels</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 16 pages, 10 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Large language models are increasingly deployed on CPU-only platforms where memory bandwidth is the primary bottleneck for autoregressive generation. Weight quantization to four bits or below reduces memory pressure, yet existing systems still dequantize weights and perform floating-point multiplications, limiting the achievable gains. Ternary weights in {-1, 0, +1} provide a more efficient alternative, replacing multiplications with conditional additions, subtractions, or no-ops. While Fairy2i shows that ternary LLMs can match FP16 quality, its runtime does not exploit this structure. We present FairyFuse, an inference system that enables multiplication-free execution on commodity CPUs by fusing the eight real-valued sub-GEMVs of each widely-linear layer into a single AVX-512 loop using masked additions and subtractions, with zero floating-point multiplications. Roofline analysis shows that 16x weight compression shifts memory-bound GEMV toward the compute regime on bandwidth-limited CPUs, yielding a 29.6x kernel speedup while offering little benefit on GPUs. End-to-end, FairyFuse achieves 32.4 tokens per second on a single Intel Xeon 8558P, outperforming llama.cpp Q4_K_M by 1.24x with near-lossless quality (WikiText-2 perplexity 5.52 vs. 5.47 FP16; downstream accuracy 66.0%).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.13928v2">LLMs Can Get "Brain Rot": A Pilot Study on Twitter/X</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 Updated experiments with corrected data
    </div>
    <details class="paper-abstract">
      We propose and test the LLM Brain Rot Hypothesis: continual exposure to junk web text induces lasting cognitive decline in large language models (LLMs). To unveil junk effects, we designed a novel controlled experiment on real Twitter/X corpora, by constructing junk and reverse-controlled datasets via two orthogonal operationalizations: M1 (engagement degree) and M2 (semantic quality), with matched token scale and training operations across conditions. Compared to the control group, continual pre-training of 4 LLMs on the junk dataset causes non-trivial declines (Hedges' g>0.3) on reasoning, long-context understanding, safety, and inflating "dark traits" (e.g., psychopathy, narcissism). The gradual mixtures of junk and control datasets also yield dose-response cognition decay: for example, under M1, ARC-Challenge with Chain-of-Thought drops 72.1 -> 57.2 and RULER-CWE 83.7 -> 52.3 as junk ratio rises from 0% to 100%. Error forensics reveal several key insights. First, we identify thought-skipping as the primary lesion in reasoning: models increasingly truncate or skip chains. Second, partial but incomplete healing is observed: scaling instruction tuning and clean continual pre-training improve the declined cognition, yet cannot restore baseline capability, suggesting persistent representational drift rather than format mismatch. Finally, we discover that the popularity, a non-semantic metric, of a tweet is a better indicator of the Brain Rot effect than the length in M1. Together, the results provide significant, multi-perspective evidence that social effects of data could be a causal driver of LLM capability decay in continual pre-training, thereby motivating routine "cognitive health checks" for deployed and evolving LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.18036v6">Harnessing Multiple Large Language Models: A Survey on LLM Ensemble</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 12 pages, 2 figures, codebase: https://github.com/junchenzhi/Awesome-LLM-Ensemble
    </div>
    <details class="paper-abstract">
      LLM Ensemble -- which involves the comprehensive use of multiple large language models (LLMs), each aimed at handling user queries during downstream inference, to benefit from their individual strengths -- has gained substantial attention recently. The widespread availability of LLMs, coupled with their varying strengths and out-of-the-box usability, has profoundly advanced the field of LLM Ensemble. This paper presents the first systematic review of recent developments in LLM Ensemble. First, we introduce our taxonomy of LLM Ensemble and discuss several related research problems. Then, we provide a more in-depth classification of the methods under the broad categories of "ensemble-before-inference, ensemble-during-inference, ensemble-after-inference'', and review all relevant methods. Finally, we introduce related benchmarks and applications, summarize existing studies, and suggest several future research directions. A curated list of papers on LLM Ensemble is available at https://github.com/junchenzhi/Awesome-LLM-Ensemble.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.07963v4">Caught in the Web of Words: Do LLMs Fall for Spin in Medical Literature?</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 26 pages, 17 figures, 4 tables, Conference on Health, Inference, and Learning (CHIL) 2025
    </div>
    <details class="paper-abstract">
      Medical research faces well-documented challenges in translating novel treatments into clinical practice. Publishing incentives encourage researchers to present "positive" findings, even when empirical results are equivocal. Consequently, it is well-documented that authors often spin study results, especially in article abstracts. Such spin can influence clinician interpretation of evidence and may affect patient care decisions. In this study, we ask whether the interpretation of trial results offered by Large Language Models (LLMs) is similarly affected by spin. This is important since LLMs are increasingly being used to trawl through and synthesize published medical evidence. We evaluated 22 LLMs and found that they are across the board more susceptible to spin than humans. They might also propagate spin into their outputs: We find evidence, e.g., that LLMs implicitly incorporate spin into plain language summaries that they generate. We also find, however, that LLMs are generally capable of recognizing spin, and can be prompted in a way to mitigate spin's impact on LLM outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20098v1">Differentiable Conformal Training for LLM Reasoning Factuality</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 Submitted ICML
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) frequently hallucinate, limiting their reliability in critical applications. Conformal Prediction (CP) addresses this by calibrating error rates on held-out data to provide statistically valid confidence guarantees. Recent work extends CP to LLM factuality to filter out risky claims, ensuring that hallucination rates remain below a user-specified level (e.g., 10%). While prior methods treat claims independently, Coherent Factuality extends to multi-step reasoning by representing outputs as dependency graphs and jointly validating claims with their logical ancestors. A key limitation is that Coherent Factuality is not differentiable, requiring hand-crafted scorers that at high reliability levels remove nearly 60% of true claims. We introduce Differentiable Coherent Factuality (DCF), a fully differentiable relaxation that enables learning improved scorers while provably recovering the original algorithm's guarantees. Experiments on two benchmark reasoning datasets demonstrate DCF achieves up to 141% improvement in claim retention while maintaining reliability guarantees, representing a significant step towards reliable conformal LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20911v1">Omission Constraints Decay While Commission Constraints Persist in Long-Context LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 19 pages, 5 figures. Includes evaluation framework for replication and 4,416-trial dataset
    </div>
    <details class="paper-abstract">
      LLM agents deployed in production operate under operator-defined behavioral policies (system-prompt instructions such as prohibitions on credential disclosure, data exfiltration, and unauthorized output) that safety evaluations assume hold throughout a conversation. Prohibition-type constraints decay under context pressure while requirement-type constraints persist; we term this asymmetry Security-Recall Divergence (SRD). In a 4,416-trial three-arm causal study across 12 models and 8 providers at six conversation depths, omission compliance falls from 73% at turn 5 to 33% at turn 16 while commission compliance holds at 100% (Mistral Large 3, $p < 10^{-33}$). In the two models with token-matched padding controls, schema semantic content accounts for 62-100% of the dilution effect. Re-injecting constraints before the per-model Safe Turn Depth (STD) restores compliance without retraining. Production security policies consist of prohibitions such as never revealing credentials, never executing untrusted code, and never forwarding user data. Commission-type audit signals remain healthy while omission constraints have already failed, leaving the failure invisible to standard monitoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.11581v3">Hidden Measurement Error in LLM Pipelines Distorts Annotation, Evaluation, and Benchmarking</a></div>
    <div class="paper-meta">
      📅 2026-04-22
    </div>
    <details class="paper-abstract">
      LLM evaluations drive which models get deployed, which safety standards get adopted, and which research conclusions get published. Yet standard confidence intervals ignore variability from prompt phrasing, model temperature, and judge model choice. The omitted variance produces under-coverage that worsens with more data and can shift results enough to reverse conclusions. The same unmeasured variance opens benchmarks to exploitation. Model developers can optimize against measurement noise instead of genuine performance, as \citet{singh2025leaderboard} document. This paper decomposes LLM pipeline uncertainty into its sources, distinguishes variance that shrinks with more data from sensitivity to researcher design choices, and uses design-study projections to reduce total error. We show a small-sample pilot is sufficient to derive confidence intervals that approach nominal coverage and to identify which design changes yield the largest precision gains. Applying the approach to ideology annotation, safety classification, MMLU benchmarking, and a human-validated propaganda audit reveals different dominant variance sources by domain and scoring method. What's more, we show optimized budget allocation halves estimation error at equivalent cost (MMLU), and on our propaganda audit, the recommended pipeline outperforms 73\% of single-configuration alternatives against a human baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20065v1">From Hidden Profiles to Governable Personalization: Recommender Systems in the Age of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 6 pages, under review
    </div>
    <details class="paper-abstract">
      Personalization has traditionally depended on platform-specific user models that are optimized for prediction but remain largely inaccessible to the people they describe. As LLM-based assistants increasingly mediate search, shopping, travel, and content access, this arrangement may be giving way to a new personalization stack in which user representation is no longer confined to isolated platforms. In this paper, we argue that the key issue is not simply that large language models can enhance recommendation quality, but that they reconfigure where and how user representations are produced, exposed, and acted upon. We propose a shift from hidden platform profiling toward governable personalization, where user representations may become more inspectable, revisable, portable, and consequential across services. Building on this view, we identify five research fronts for recommender systems: transparent yet privacy-preserving user modeling, intent translation and alignment, cross-domain representation and memory design, trustworthy commercialization in assistant-mediated environments, and operational mechanisms for ownership, access, and accountability. We position these not as isolated technical challenges, but as interconnected design problems created by the emergence of LLM agents as intermediaries between users and digital platforms. We argue that the future of recommender systems will depend not only on better inference, but on building personalization systems that users can meaningfully understand, shape, and govern.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14128v2">Rhetorical Questions in LLM Representations: A Linear Probing Study</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 18 pages, 15 figures, accepted to ACL 2026
    </div>
    <details class="paper-abstract">
      Rhetorical questions are asked not to seek information but to persuade or signal stance. How large language models internally represent them remains unclear. We analyze rhetorical questions in LLM representations using linear probes on two social-media datasets with different discourse contexts, and find that rhetorical signals emerge early and are most stably captured by last-token representations. Rhetorical questions are linearly separable from information-seeking questions within datasets, and remain detectable under cross-dataset transfer, reaching AUROC around 0.7-0.8. However, we demonstrate that transferability does not simply imply a shared representation. Probes trained on different datasets produce different rankings when applied to the same target corpus, with overlap among the top-ranked instances often below 0.2. Qualitative analysis shows that these divergences correspond to distinct rhetorical phenomena: some probes capture discourse-level rhetorical stance embedded in extended argumentation, while others emphasize localized, syntax-driven interrogative acts. Together, these findings suggest that rhetorical questions in LLM representations are encoded by multiple linear directions emphasizing different cues, rather than a single shared direction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20043v1">TriEx: A Game-based Tri-View Framework for Explaining Internal Reasoning in Multi-Agent LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 ACL2026 Main
    </div>
    <details class="paper-abstract">
      Explainability for Large Language Model (LLM) agents is especially challenging in interactive, partially observable settings, where decisions depend on evolving beliefs and other agents. We present \textbf{TriEx}, a tri-view explainability framework that instruments sequential decision making with aligned artifacts: (i) structured first-person self-reasoning bound to an action, (ii) explicit second-person belief states about opponents updated over time, and (iii) third-person oracle audits grounded in environment-derived reference signals. This design turns explanations from free-form narratives into evidence-anchored objects that can be compared and checked across time and perspectives. Using imperfect-information strategic games as a controlled testbed, we show that TriEx enables scalable analysis of explanation faithfulness, belief dynamics, and evaluator reliability, revealing systematic mismatches between what agents say, what they believe, and what they do. Our results highlight explainability as an interaction-dependent property and motivate multi-view, evidence-grounded evaluation for LLM agents. Code is available at https://github.com/Einsam1819/TriEx.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02989v2">Mechanistic Interpretability of Large-Scale Counting in LLMs through a System-2 Strategy</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 ACL 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), despite strong performance on complex mathematical problems, exhibit systematic limitations in counting tasks. This issue arises from the architectural limits of transformers, where counting is performed across layers, leading to degraded precision for larger counting problems due to depth constraints. To address this limitation, we propose a simple test-time strategy inspired by System-2 cognitive processes that decomposes large counting tasks into smaller, independent sub-problems that the model can reliably solve. We evaluate this approach using observational and causal mediation analyses to understand the underlying mechanism of this System-2-like strategy. Our mechanistic analysis identifies key components: latent counts are computed and stored in the final item representations of each part, transferred to intermediate steps via dedicated attention heads, and aggregated in the final stage to produce the total count. Experimental results demonstrate that this strategy enables LLMs to surpass architectural limitations and achieve higher accuracy on large-scale counting tasks. This work provides mechanistic insight into System-2 counting in LLMs and presents a generalizable approach for improving and understanding their reasoning behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.00253v3">Towards Explorative IRBL: Combining Semantic Retrieval with LLM-driven Iterative Code Exploration</a></div>
    <div class="paper-meta">
      📅 2026-04-21
    </div>
    <details class="paper-abstract">
      Information Retrieval-based Bug Localization (IRBL) aims to identify buggy source files for a given bug report. Traditional and deep learning-based IRBL techniques often suffer from vocabulary mismatch and dependence on project-specific metadata. In contrast, recent Large Language Model (LLM)-based approaches struggle to provide appropriate context to the model: they either restrict analysis to a fixed set of candidate files, overwhelm the model with repository-wide information, or rely on explicit bug report cues to guide context collection. To address these issues, we propose GenLoc, a technique that combines semantic retrieval with LLM-driven code-exploration functions to iteratively analyze the code base and identify buggy files. We evaluate GenLoc on three complementary benchmarks, including large-scale and recent Java datasets as well as the Python based SWE-bench Lite dataset. Results demonstrate that GenLoc substantially outperforms traditional IRBL, deep learning-based approaches and recent LLM-based methods, while also localizing bugs that other techniques fail to detect.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20021v1">Continuous Semantic Caching for Low-Cost LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-04-21
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become increasingly popular, caching responses so that they can be reused by users with semantically similar queries has become a vital strategy for reducing inference costs and latency. Existing caching frameworks have proposed to decide which query responses to cache by assuming a finite, known universe of discrete queries and learning their serving costs and arrival probabilities. As LLMs' pool of users and queries expands, however, such an assumption becomes increasingly untenable: real-world LLM queries reside in an infinite, continuous embedding space. In this paper, we establish the first rigorous theoretical framework for semantic LLM response caching in continuous query space under uncertainty. To bridge the gap between discrete optimization and continuous representation spaces, we introduce dynamic $ε$-net discretization coupled with Kernel Ridge Regression. This design enables the system to formally quantify estimation uncertainty and generalize partial feedback on LLM query costs across continuous semantic query neighborhoods. We develop both offline learning and online adaptive algorithms optimized to reduce switching costs incurred by changing the cached responses. We prove that our online algorithm achieves a sublinear regret bound against an optimal continuous oracle, which reduces to existing bounds for discrete query models. Extensive empirical evaluations demonstrate that our framework approximates the continuous optimal cache well while also reducing computational and switching overhead compared to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19984v1">Bias in the Tails: How Name-conditioned Evaluative Framing in Resume Summaries Destabilizes LLM-based Hiring</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 First version, 43 pages
    </div>
    <details class="paper-abstract">
      Research has documented LLMs' name-based bias in hiring and salary recommendations. In this paper, we instead consider a setting where LLMs generate candidate summaries for downstream assessment. In a large-scale controlled study, we analyze nearly one million resume summaries produced by 4 models under systematic race-gender name perturbations, using synthetic resumes and real-world job postings. By decomposing each summary into resume-grounded factual content and evaluative framing, we find that factual content remains largely stable, while evaluative language exhibits subtle name-conditioned variation concentrated in the extremes of the distribution, especially in open-source models. Our hiring simulation demonstrates how evaluative summary transforms directional harm into symmetric instability that might evade conventional fairness audit, highlighting a potential pathway for LLM-to-LLM automation bias.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.09958v3">Neural Bandit Based Optimal LLM Selection for a Pipeline of Subtasks</a></div>
    <div class="paper-meta">
      📅 2026-04-21
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly popular, there is a growing need to predict which out of a set of LLMs will yield a successful answer to a given query at low cost. This problem promises to become even more relevant as LLM agents are asked to solve an increasing variety of "agentic'' AI tasks. Such tasks are often broken into smaller subtasks, each of which can then be executed by a LLM expected to perform well on that specific subtask. For example, to extract a diagnosis from medical records, one can first select an LLM to summarize the record, select another to validate the summary, and then select a possibly different LLM to extract the diagnosis from the summarized record. Unlike existing LLM selection or routing algorithms, this setting requires selecting a sequence of LLMs, with the output of each LLM feeding into the next and potentially influencing its success. Thus, unlike single LLM selection, the quality of each subtask's output directly affects the inputs, and hence the cost and success rate, of downstream LLMs, creating complex performance dependencies that must be learned during selection. We propose a neural contextual bandit-based algorithm that trains neural networks to guide LLM selections for the different subtasks, without requiring historical LLM performance data. We prove that our proposed Sequential Bandits algorithm achieves a sublinear regret in the number of tasks, and we experimentally validate its superior performance compared to other LLM selection algorithms on two real datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21394v3">MemoPhishAgent: Memory-Augmented Multi-Modal LLM Agent for Phishing URL Detection</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 ACL 2026 Industry Track Camera Ready
    </div>
    <details class="paper-abstract">
      Traditional phishing website detection relies on static heuristics or reference lists, which lag behind rapidly evolving attacks. While recent systems incorporate large language models (LLMs), they are still prompt-based, deterministic pipelines that underutilize reasoning capability. We present MemoPhishAgent (MPA), a memory-augmented multi-modal LLM agent that dynamically orchestrates phishing-specific tools and leverages episodic memories of past reasoning trajectories to guide decisions on recurring and novel threats. On two public datasets, MPA outperforms three state-of-the-art (SOTA) baselines, improving recall by 13.6%. To better reflect realistic, user-facing phishing detection performance, we further evaluate MPA on a benchmark of real-world suspicious URLs actively crawled from five social media platforms, where it improves recall by 20%. Detailed analysis shows episodic memory contributes up to 27% recall gain without introducing additional computational overhead. The ablation study confirms the necessity of the agent-based approach compared to prompt-based baselines and validates the effectiveness of our tool design. Finally, MPA is deployed in production, processing 60K targeted high-risk URLs weekly, and achieving 91.44% recall, providing proactive protection for millions of customers. Together, our results show that combining multi-modal reasoning with episodic memory yields robust phishing detection in realistic user-exposure settings. Our implementation is available at https://github.com/XuanChen-xc/MemoPhishAgent.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.00696v2">DRIV-EX: Counterfactual Explanations for Driving LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 Accepted at ACL Findings 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as reasoning engines in autonomous driving, yet their decision-making remains opaque. We propose to study their decision process through counterfactual explanations, which identify the minimal semantic changes to a scene description required to alter a driving plan. We introduce DRIV-EX, a method that leverages gradient-based optimization on continuous embeddings to identify the input shifts required to flip the model's decision. Crucially, to avoid the incoherent text typical of unconstrained continuous optimization, DRIV-EX uses these optimized embeddings solely as a semantic guide: they are used to bias a controlled decoding process that re-generates the original scene description. This approach effectively steers the generation toward the counterfactual target while guaranteeing the linguistic fluency, domain validity, and proximity to the original input, essential for interpretability. Evaluated using the LC-LLM planner on a textual transcription of the highD dataset, DRIV-EX generates valid, fluent counterfactuals more reliably than existing baselines. It successfully exposes latent biases and provides concrete insights to improve the robustness of LLM-based driving agents. The code is available at "https://github.com/Amaia-CARDIEL/DRIV_EX" .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20904v1">Reinforcing privacy reasoning in LLMs via normative simulacra from fiction</a></div>
    <div class="paper-meta">
      📅 2026-04-21
    </div>
    <details class="paper-abstract">
      Information handling practices of LLM agents are broadly misaligned with the contextual privacy expectations of their users. Contextual Integrity (CI) provides a principled framework, defining privacy as the appropriate flow of information within context-relative norms. However, existing approaches either double inference cost via supervisor-assistant architectures, or fine-tune on narrow task-specific data. We propose extracting normative simulacra (structured representations of norms and information flows) from fiction novels and using them to fine-tune LLMs via supervised learning followed by GRPO reinforcement learning. Our composite reward function combines programmatic signals, including task clarity (subsuming schema validity, construct discrimination, and extraction confidence), structural completeness, internal consistency, and context identification, with an LLM judge that evaluates whether the model's privacy reasoning is grounded in the held-out normative universe of the source text. To mitigate overfitting, we introduce per-completion contrastive scoring: each completion is evaluated against both the correct normative universe and a randomly selected wrong one, teaching the model to condition on context rather than memorize source-specific norms. We evaluate on five CI-aligned benchmarks spanning distinct societal contexts and ablate the contributions of RL and normative grounding. Across seven models, SFT introduces a conservative prior toward restricting information flow, improving recognition of privacy-relevant situations but not the correctness of privacy judgments. GRPO with normative grounding achieves the highest score on a law compliance benchmark and strongest correlation with crowdsourced human privacy expectations, demonstrating that fiction-derived normative simulacra can teach contextual privacy reasoning that transfers to real-world domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.20181v2">Catalyzing Informed Residential Energy Retrofit Decisions via Domain-Specific LLM</a></div>
    <div class="paper-meta">
      📅 2026-04-21
    </div>
    <details class="paper-abstract">
      Residential energy retrofit initiation is often stalled by an expertise gap, where homeowners lack the technical literacy required for structured building energy assessments and are thereby trapped in low-information environments with fragmented sources. To bridge this gap, this study reports a domain-specific large language model (LLM) designed to catalyze informed decision-making based solely on homeowner-accessible, natural-language descriptions, e.g., building age, size, and location. The model is created using the parameter-efficient low-rank adaption (LoRA) fine-tuning approach on a massive corpus grounded in physics-based energy simulations and techno-economic calculations from 536,416 U.S. residential building prototypes. Nine major retrofit categories are evaluated, including envelope upgrades, HVAC systems, and renewable energy installations. Validations against physics-grounded benchmarks show that the LLM consistently identifies high-quality retrofit options, achieving top-3 hit rates of 98.9% for maximum CO2 reduction and 93.3% for the shortest discounted payback year. Moreover, the model exhibits strong robustness under incomplete input conditions, maintaining stable performance even when basic dwelling descriptions are only 60% partially specified. By significantly lowering the information activation energy for non-expert users while maintaining the scientific rigor, this physics-based AI model offers a scalable pathway for parallelized, user-centered decision making, accelerating cumulative energy savings and emission reductions across community and national scales.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19884v1">From Signal Degradation to Computation Collapse: Uncovering the Two Failure Modes of LLM Quantization</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 Accepted to Findings of ACL 2026
    </div>
    <details class="paper-abstract">
      Post-Training Quantization (PTQ) is critical for the efficient deployment of Large Language Models (LLMs). While 4-bit quantization is widely regarded as an optimal trade-off, reducing the precision to 2-bit usually triggers a catastrophic ``performance cliff.'' It remains unclear whether the underlying mechanisms differ fundamentally. Consequently, we conduct a systematic mechanistic analysis, revealing two qualitatively distinct failure modes: Signal Degradation, where the computational patterns remain intact but information precision is impaired by cumulative error; and Computation Collapse, where key components fail to function, preventing correct information processing and destroying the signal in the early layers. Guided by this diagnosis, we conduct mechanism-aware interventions, demonstrating that targeted, training-free repair can mitigate Signal Degradation, but remains ineffective for Computation Collapse. Our findings provide a systematic diagnostic framework for PTQ failures and suggest that addressing Computation Collapse requires structural reconstruction rather than mere compensation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19742v1">PlayCoder: Making LLM-Generated GUI Code Playable</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 September 11, 2025 Submitted to FSE2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved strong results in code generation, but their ability to generate GUI applications, especially games, remains insufficiently studied. Existing benchmarks mainly evaluate correctness through test cases, which are inadequate for GUI applications because these systems are interactive, event-driven, and require correct state transitions across sequences of user actions. Their evaluation therefore should consider interaction flows and UI logic rather than only pass/fail outcomes. To study this problem, we introduce PlayEval, a repository-aware benchmark built from 43 multilingual GUI applications in Python, TypeScript, and JavaScript. Unlike prior GUI benchmarks that are difficult to adapt to desktop environments, PlayEval covers six major GUI application categories and directly supports code-generation evaluation. We further propose Play@k, a metric that measures whether at least one of *k* generated candidates can be played end-to-end without logical errors. To support reliable evaluation, we develop PlayTester, an LLM-based agent that performs task-oriented GUI playthroughs and detects logic violations automatically. Experiments on 10 state-of-the-art code LLMs show that, despite high compilation rates, they achieve near-zero Play@3, revealing major weaknesses in generating logically correct GUI applications. To address this limitation, we present PlayCoder, a multi-agent, repository-aware framework that generates, evaluates, and iteratively repairs GUI application code in a closed loop. PlayCoder substantially improves both functional correctness and semantic alignment for open-source and closed-source models, reaching up to 38.1% Exec@3 and 20.3% Play@3. Case studies further show that it can uncover silent logic bugs missed by traditional metrics and fix them through targeted edits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19716v1">Discovering a Shared Logical Subspace: Steering LLM Logical Reasoning via Alignment of Natural-Language and Symbolic Views</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 Accepted to ACL 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) still struggle with multi-step logical reasoning. Existing approaches either purely refine the reasoning chain in natural language form or attach a symbolic solver as an external module. In this work, we instead ask whether LLMs contain a shared internal logical subspace that simultaneously aligns natural-language and symbolic-language views of the reasoning process. Our hypothesis is that this logical subspace captures logical reasoning capabilities in LLMs that are shared across views while remaining independent of surface forms. To verify this, we employ Canonical Correlation Analysis on the paired residual activations from natural-language and symbolic-language reasoning chains, learning a low-dimensional subspace with maximum cross-view correlation. Furthermore, we design a training-free approach that steers LLMs reasoning chain along this logical subspace, thereby leveraging the complementary reasoning signals from both views. Experiments on four logical reasoning benchmarks demonstrate the effectiveness of our approach, improving accuracy by up to 11 percentage points and generalizing well on out-of-domain problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.24472v2">Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs?</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 Code is available at https://github.com/beanie00/self-distillation-analysis
    </div>
    <details class="paper-abstract">
      Self-distillation has emerged as an effective post-training paradigm for LLMs, often improving performance while shortening reasoning traces. However, in mathematical reasoning, we find that it can reduce response length while degrading performance. We trace this degradation to the suppression of epistemic verbalization - the model's expression of uncertainty during reasoning. Through controlled experiments varying conditioning context richness and task coverage, we show that conditioning the teacher on rich information suppresses uncertainty expression, enabling rapid in-domain optimization with limited task coverage but harming OOD performance, where unseen problems benefit from expressing uncertainty and adjusting accordingly. Across Qwen3-8B, DeepSeek-Distill-Qwen-7B, and Olmo3-7B-Instruct, we observe performance drops of up to 40%. Our findings highlight that exposing appropriate levels of uncertainty is crucial for robust reasoning and underscore the importance of optimizing reasoning behavior beyond merely reinforcing correct answer traces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04562v2">Reasoning Over Space: Enabling Geographic Reasoning for LLM-Based Generative Next POI Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-04-21
    </div>
    <details class="paper-abstract">
      Generative recommendation with large language models (LLMs) reframes prediction as sequence generation, yet existing LLM-based recommenders remain limited in leveraging geographic signals that are crucial in mobility and local-services scenarios. Here, we present Reasoning Over Space (ROS), a framework that utilizes geography as a vital decision variable within the reasoning process. ROS introduces a Hierarchical Spatial Semantic ID (SID) that discretizes coarse-to-fine locality and POI semantics into compositional tokens, and endows LLM with a three-stage Mobility Chain-of-Thought (CoT) paradigm that models user personality, constructs an intent-aligned candidate space, and performs locality informed pruning. We further align the model with real world geography via spatial-guided Reinforcement Learning (RL). Experiments on three widely used location-based social network (LBSN) datasets show that ROS achieves over 10% relative gains in hit rate over strongest LLM-based baselines and improves cross-city transfer, despite using a smaller backbone model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19567v1">Multi-modal Reasoning with LLMs for Visual Semantic Arithmetic</a></div>
    <div class="paper-meta">
      📅 2026-04-21
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) as post-training is crucial for enhancing the reasoning ability of large language models (LLMs) in coding and math. However, their capacity for visual semantic arithmetic, inferring relationships from images, remains underexplored. The classic text analogy "king"-"man"+"woman" = "queen" illustrates relational reasoning, yet replacing text with images of "king" and "man" significantly reduces performance because it requires commonsense knowledge and the extraction of concise concepts from irrelevant visual details. This capability is important for service and domestic robotics in unstructured environments, where robots must infer semantic relationships among objects, agents, and actions. In a kitchen, recognizing from images that "powder" and "cake" are related by "is made of" grounds symbolic relations in perception, enabling tool substitution, task generalization, and improved semantic reasoning. Prior work approaches semantic arithmetic by decoding image features after vector arithmetic, but suffers from modality gaps and lacks systematic evaluation. In this paper, we formulate two novel tasks, two-term subtraction and three-term operations, and construct the Image-Relation-Pair Dataset (IRPD) for benchmarking. We further propose Semantic Arithmetic Reinforcement Fine-Tuning (SAri-RFT), which post-trains large vision-language models (LVLMs) using a verifiable function and Group Relative Policy Optimization (GRPO). Our method achieves state-of-the-art results on IRPD and the real-world Visual7W-Telling dataset. By equipping LVLMs with robust cross-modal relational reasoning, this work advances domestic robots' ability to ground symbolic reasoning in perception, enhancing decision-making, tool adaptability, and human-robot interaction in complex environments. Datasets and source code are provided in the supplementary material.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19558v1">On Reasoning-Centric LLM-based Automated Theorem Proving</a></div>
    <div class="paper-meta">
      📅 2026-04-21
    </div>
    <details class="paper-abstract">
      Automated theorem proving is fundamental to formal methods, and the recent trend is to integrate large language models (LLMs) and proof assistants to form effective proof agents. While existing proof agents show promising performance, they inadequately leverage reasoning capabilities of modern LLMs in high-level planning and self-critique. We argue that proof agents should not merely generate tactics but also reason strategically about proof plans and critically evaluate their own proposals. This paper introduces ReCent-Prover, a reasoning-centric LLM-based proof agent for Rocq that addresses two critical limitations in current systems. First, we present validation with reflection, enabling LLMs to scrutinize their generated tactics and synthesize failure summaries when reflection identifies potential errors, filtering out potentially misapplied tactics earlier. Second, we propose retrieval with planning, which conditions retrieval on LLM-generated proof plans rather than subgoal similarity, retrieving lemmas and proofs that align with the anticipated proof strategy. Both techniques increase the number of invocations of LLMs. However, when evaluated on the CoqStoq benchmark, even under the same budget of LLM invocations, ReCent-Prover achieves a 22.58% relative improvement in the number of proved theorems over the previous state-of-the-art, demonstrating that our reasoning-centric design significantly enhances automated theorem proving capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19540v1">Mesh Memory Protocol: Semantic Infrastructure for Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 23 pages, 2 figures, 2 listings, 1 table. MMP v0.2.3 specification at https://sym.bot/spec/mmp (CC BY 4.0). Reference implementations on npm (@sym-bot/sym, @sym-bot/mesh-channel; Apache 2.0)
    </div>
    <details class="paper-abstract">
      Teams of LLM agents increasingly collaborate on tasks spanning days or weeks: multi-day data-generation sprints where generator, reviewer, and auditor agents coordinate in real time on overlapping batches; specialists carrying findings forward across session restarts; product decisions compounding over many review rounds. This requires agents to share, evaluate, and combine each other's cognitive state in real time across sessions. We call this cross-session agent-to-agent cognitive collaboration, distinct from parallel agent execution. To enable it, three problems must be solved together. (P1) Each agent decides field by field what to accept from peers, not accept or reject whole messages. (P2) Every claim is traceable to source, so returning claims are recognised as echoes of the receiver's own prior thinking. (P3) Memory that survives session restarts is relevant because of how it was stored, not how it is retrieved. These are protocol-level properties at the semantic layer of agent communication, distinct from tool-access and task-delegation protocols at lower layers. We call this missing protocol layer "semantic infrastructure," and the Mesh Memory Protocol (MMP) specifies it. Four composable primitives work together: CAT7, a fixed seven-field schema for every Cognitive Memory Block (CMB); SVAF, which evaluates each field against the receiver's role-indexed anchors and realises P1; inter-agent lineage, carried as parents and ancestors of content-hash keys and realising P2; and remix, which stores only the receiver's own role-evaluated understanding of each accepted CMB, never the raw peer signal, realising P3. MMP is specified, shipped, and running in production across three reference deployments, where each session runs an autonomous agent as a mesh peer with its own identity and memory, collaborating with other agents across the network for collective intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19526v1">Evaluating LLM-Generated Obfuscated XSS Payloads for Machine Learning-Based Detection</a></div>
    <div class="paper-meta">
      📅 2026-04-21
    </div>
    <details class="paper-abstract">
      Cross-site scripting (XSS) remains a persistent web security vulnerability, especially because obfuscation can change the surface form of a malicious payload while preserving its behavior. These transformations make it difficult for traditional and machine learning-based detection systems to reliably identify attacks. Existing approaches for generating obfuscated payloads often emphasize syntactic diversity, but they do not always ensure that the generated samples remain behaviorally valid. This paper presents a structured pipeline for generating and evaluating obfuscated XSS payloads using large language models (LLMs). The pipeline combines deterministic transformation techniques with LLM-based generation and uses a browser- based runtime evaluation procedure to compare payload behavior in a controlled execution environment. This allows generated samples to be assessed through observable runtime behavior rather than syntactic similarity alone. In the evaluation, an untuned baseline language model achieves a runtime behavior match rate of 0.15, while fine-tuning on behavior-preserving source-target obfuscation pairs improves the match rate to 0.22. Although this represents a measurable improvement, the results show that current LLMs still struggle to generate obfuscations that preserve observed runtime behavior. A downstream classifier evaluation further shows that adding generated payloads does not improve detection performance in this setting, although behavior- filtered generated samples can be incorporated without materially degrading performance. Overall, the study demonstrates both the promise and the limits of applying generative models to adversarial security data generation and emphasizes the importance of runtime behavior checks in improving the quality of generated data for downstream detection systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19485v1">EVPO: Explained Variance Policy Optimization for Adaptive Critic Utilization in LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2026-04-21
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) for LLM post-training faces a fundamental design choice: whether to use a learned critic as a baseline for policy optimization. Classical theory favors critic-based methods such as PPO for variance reduction, yet critic-free alternatives like GRPO have gained widespread adoption due to their simplicity and competitive performance. We show that in sparse-reward settings, a learned critic can inject estimation noise that exceeds the state signal it captures, increasing rather than reducing advantage variance. By casting baseline selection as a Kalman filtering problem, we unify PPO and GRPO as two extremes of the Kalman gain and prove that explained variance (EV), computable from a single training batch, identifies the exact boundary: positive EV indicates the critic reduces variance, while zero or negative EV signals that it inflates variance. Building on this insight, we propose Explained Variance Policy Optimization (EVPO), which monitors batch-level EV at each training step and adaptively switches between critic-based and batch-mean advantage estimation, provably achieving no greater variance than the better of the two at every step. Across four tasks spanning classical control, agentic interaction, and mathematical reasoning, EVPO consistently outperforms both PPO and GRPO regardless of which fixed baseline is stronger on a given task. Further analysis confirms that the adaptive gating tracks critic maturation over training and that the theoretically derived zero threshold is empirically optimal.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19447v1">'The Order in the Horse's Heart': A Case Study in LLM-Assisted Stylometry for the Discovery of Biblical Allusion in Modern Literary Fiction</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 39 pages, 1 figure
    </div>
    <details class="paper-abstract">
      We present a dual-track pipeline for detecting biblical allusions in literary fiction and apply it to the novels of Cormac McCarthy. A bottom-up embedding track uses inverse document frequency to identify rare vocabulary shared with the King James Bible, embeds occurrences in their local context for sense disambiguation, and passes candidate passage pairs through cascaded LLM review. A top-down register track asks an LLM to read McCarthy's prose undirected to any specific biblical passage for comparison, catching allusions not distinguished by word or phrase rarity. Both tracks are cross-validated by a long-context model that holds entire novels alongside the KJV in a single pass, and every finding is checked against published scholarship. Restricting attention to allusions that carry a textual echo--shared phrasing, reworked vocabulary, or transplanted cadence--and distinguishing literary allusions proper from signposted biblical references (similes naming biblical figures, characters overtly citing scripture), the pipeline surfaces 349 allusions across the corpus. Among a target set of 115 previously documented allusions retrieved through human review of the academic literature, the pipeline independently recovers 62 (54% recall), with recall varying by connection type from 30% (transformed imagery) to 80% (register collisions). We contextualise these results with respect to the value-add from LLMs as assistants to mechanical stylometric analyses, and their potential to facilitate the statistical study of intertextuality in massive literary corpora.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19444v1">Unsupervised Confidence Calibration for Reasoning LLMs from a Single Generation</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 41 pages, 14 tables, 12 figures
    </div>
    <details class="paper-abstract">
      Reasoning language models can solve increasingly complex tasks, but struggle to produce the calibrated confidence estimates necessary for reliable deployment. Existing calibration methods usually depend on labels or repeated sampling at inference time, making them impractical in many settings. We introduce a method for unsupervised confidence calibration of reasoning LLMs when only a single generation is available at inference time. Our approach uses offline sampling on unlabeled data to derive a self-consistency-based proxy target, then distills this signal into a lightweight deployment-time confidence predictor. In a broad evaluation across 5 math and question-answering tasks using 9 reasoning models, our method substantially outperforms baselines, including under distribution shift, and improves downstream performance in selective prediction and simulated downstream decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19440v1">What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 9 pages, 8 figures, Accepted at Findings of ACL 2026
    </div>
    <details class="paper-abstract">
      Recent work has demonstrated the promise of orchestrating large language models (LLMs) within evolutionary and agentic optimization systems. However, the mechanisms driving these optimization gains remain poorly understood. In this work, we present a large-scale study of LLM-guided evolutionary search, collecting optimization trajectories for 15 LLMs across 8 tasks. Although zero-shot problem-solving ability correlates with final optimization outcomes, it explains only part of the variance: models with similar initial capability often induce dramatically different search trajectories and outcomes. By analyzing these trajectories, we find that strong LLM optimizers behave as local refiners, producing frequent incremental improvements while progressively localizing the search in semantic space. Conversely, weaker optimizers exhibit large semantic drift, with sporadic breakthroughs followed by stagnation. Notably, various measures of solution novelty do not predict final performance; novelty is beneficial only when the search remains sufficiently localized around high-performing regions of the solution space. Our results highlight the importance of trajectory analysis for understanding and improving LLM-based optimization systems and provide actionable insights for their design and training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05414v3">Large Language Models Are Bad Dice Players: LLMs Struggle to Generate Random Numbers from Statistical Distributions</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 Accepted to ACL 2026 (Main Conference)
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) transition from chat interfaces to integral components of stochastic pipelines and systems approaching general intelligence, the ability to faithfully sample from specified probability distributions has become a functional requirement rather than a theoretical curiosity. We present the first large-scale, statistically powered audit of native probabilistic sampling in frontier LLMs, benchmarking 11 models across 15 distributions. To disentangle failure modes, we employ a dual-protocol design: Batch Generation, where a model produces $N{=}1000$ samples within one response, and Independent Requests, comprising $N{=}1000$ stateless calls. We observe a sharp protocol asymmetry: batch generation achieves only modest statistical validity, with a 7% median pass rate, while independent requests collapse almost entirely, with 10 of 11 models passing none of the distributions. Beyond this asymmetry, we reveal that sampling fidelity degrades monotonically with distributional complexity and aggravates as the sampling horizon $N$ increases. Finally, we demonstrate how the propagation of these failures into downstream real-world application tasks introduces systematic biases: models fail to enforce uniform answer-position constraints in Multiple Choice Question generation and systematically violate demographic targets in attribute-constrained text-to-image prompt synthesis. These findings indicate that current LLMs lack a functional internal sampler, necessitating external tools for applications requiring statistical guarantees.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2110.00601v2">Album: executable building blocks for scientific imaging routines, from sharing to LLM-assisted orchestration</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 38 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Open-source scientific software is a major driver of scientific progress, yet its development and reuse remain difficult in collaborative settings. Researchers repeatedly face four recurring challenges: discovering and reproducing existing routines, adapting them for new use cases, sharing and scaling them across collaborators, and stabilizing them with reproducible execution environments. We present Album, an open-source framework for packaging and sharing scientific routines as executable artifacts through two minimal primitives: (i) the solution, a Python-native executable entry point that combines machine-readable metadata, arguments, environment specifications, and lifecycle hooks; and (ii) the catalog, a decentralized, git-native distribution mechanism with indexed search and optional web rendering for discovery, provenance, and governance. Album uses a two-context execution model in which a host controller evaluates manifests and prepares per-solution environments, while lifecycle hooks execute inside isolated solution environments. This design supports reproducible execution, post-environment setup, and the composition of routines with incompatible dependencies. Album can be used in conjunction with LLM agents: solutions can be drafted and revised with LLM assistance, and a MCP interface exposes cataloged solutions as callable tools for tool-grounded discovery and orchestration. We evaluate Album through four realworld imaging deployments spanning interactive visualization of electron microscopy data, integration of multiple segmentation methods, the orchestration of cryo-electron tomography competition workflows, and mineral quantification pipelines. Overall, Album complements package managers, workflow systems, and container runtimes by making scientific routines executable, shareable artifacts. Documentation and examples are available at https://album.solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.02063v2">See2Refine: Vision-Language Feedback Improves LLM-Based eHMI Action Designers</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 Accepted to ACL2026
    </div>
    <details class="paper-abstract">
      Automated vehicles lack natural communication channels with other road users, making external Human-Machine Interfaces (eHMIs) essential for conveying intent and maintaining trust in shared environments. However, most eHMI studies rely on developer-crafted message-action pairs, which are difficult to adapt to diverse and dynamic traffic contexts. A promising alternative is to use Large Language Models (LLMs) as action designers that generate context-conditioned eHMI actions, yet such designers lack perceptual verification and typically depend on fixed prompts or costly human-annotated feedback for improvement. We present See2Refine, a human-free, closed-loop framework that uses vision-language model (VLM) perceptual evaluation as automated visual feedback to improve an LLM-based eHMI action designer. Given a driving context and a candidate eHMI action, the VLM evaluates the perceived appropriateness of the action, and this feedback is used to iteratively revise the designer's outputs, enabling systematic refinement without human supervision. We evaluate our framework across three eHMI modalities (lightbar, eyes, and arm) and multiple LLM model sizes. Across settings, our framework consistently outperforms prompt-only LLM designers and manually specified baselines in both VLM-based metrics and human-subject evaluations. Results further indicate that the improvements generalize across modalities and that VLM evaluations are well aligned with human preferences, supporting the robustness and effectiveness of See2Refine for scalable action design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19354v1">Do Agents Dream of Root Shells? Partial-Credit Evaluation of LLM Agents in Capture The Flag Challenges</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 Accepted to AIWare'26 Benchmark and Dataset Track
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents are increasingly proposed for autonomous cybersecurity tasks, but their capabilities in realistic offensive settings remain poorly understood. We present DeepRed, an open-source benchmark for evaluating LLM-based agents on realistic Capture The Flag (CTF) challenges in isolated virtualized environments. DeepRed places an agent in a Kali attacker environment with terminal tools and optional web search, connected over a private network to a target challenge, and records full execution traces for analysis. To move beyond binary solved/unsolved outcomes, we introduce a partial-credit scoring method based on challenge-specific checkpoints derived from public writeups, together with an automated summarise-then-judge labelling pipeline for assigning checkpoint completion from logs. Using DeepRed, we benchmark ten commercially accessible LLMs on ten VM-based CTF challenges spanning different challenge categories. The results indicate that current agents remain limited: the best model achieves only 35% average checkpoint completion, performing strongest on common challenge types and weakest on tasks requiring non-standard discovery and longer-horizon adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19331v1">Evaluating LLM-Driven Summarisation of Parliamentary Debates with Computational Argumentation</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 Accepted at KR'26 In The Wild Track. Camera ready to follow
    </div>
    <details class="paper-abstract">
      Understanding how policy is debated and justified in parliament is a fundamental aspect of the democratic process. However, the volume and complexity of such debates mean that outside audiences struggle to engage. Meanwhile, Large Language Models (LLMs) have been shown to enable automated summarisation at scale. While summaries of debates can make parliamentary procedures more accessible, evaluating whether these summaries faithfully communicate argumentative content remains challenging. Existing automated summarisation metrics have been shown to correlate poorly with human judgements of consistency (i.e., faithfulness or alignment between summary and source). In this work, we propose a formal framework for evaluating parliamentary debate summaries that grounds argument structures in the contested proposals up for debate. Our novel approach, driven by computational argumentation, focuses the evaluation on formal properties concerning the faithful preservation of the reasoning presented to justify or oppose policy outcomes. We demonstrate our methods using a case-study of debates from the European Parliament and associated LLM-driven summaries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04445v2">Dynamic Model Routing and Cascading for Efficient LLM Inference: A Survey</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 Work funded by ADAPT Centre, Trinity College Dublin, and Huawei Ireland
    </div>
    <details class="paper-abstract">
      The rapid growth of large language models (LLMs) with diverse capabilities, costs, and domains has created a critical need for intelligent model selection at inference time. While smaller models suffice for routine queries, complex tasks demand more capable models. However, static model deployment does not account for the complexity and domain of incoming queries, leading to suboptimal performance and increased costs. Dynamic routing systems that adaptively select models based on query characteristics have emerged as a solution to this challenge. We provide a systematic analysis of state-of-the-art multi-LLM routing and cascading approaches. In contrast to mixture-of-experts architectures, which route within a single model, we study routing across multiple independently trained LLMs. We cover diverse routing paradigms, including query difficulty, human preferences, clustering, uncertainty quantification, reinforcement learning, multimodality, and cascading. For each paradigm, we analyze representative methods and examine key trade-offs. Beyond taxonomy, we introduce a conceptual framework that characterizes routing systems along three dimensions: when decisions are made, what information is used, and how they are computed. This perspective highlights that practical systems are often compositional, integrating multiple paradigms under operational constraints. Our analysis demonstrates that effective multi-LLM routing requires balancing competing objectives. Choosing the optimal routing strategy depends on deployment and computational constraints. Well-designed routing systems can outperform even the most powerful individual models by strategically leveraging specialized capabilities across models while maximizing efficiency gains. Meanwhile, open challenges remain in developing routing mechanisms that generalize across diverse architectures, modalities, and applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19315v1">Improving LLM-Driven Test Generation by Learning from Mocking Information</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 Accepted for publication in ICST 2026 (AIST workshop). This arXiv version is the authors' accepted manuscript
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently shown strong potential for automated unit test generation. This has motivated us to investigate whether developer-defined test doubles (commonly referred to as mocks) available in existing test suites can be leveraged to improve LLM-driven test generation. To this end, we propose MOCKMILL, an LLM-based technique and tool that generates test cases by exploiting mocking information automatically extracted from developer-written tests. MOCKMILL targets components that are replaced by test doubles in existing tests and uses the encoded stubbings and interaction expectations to guide test generation, combined with an iterative generation-and-repair process to ensure executable tests. We evaluated MOCKMILL on 10 open-source classes from six Java projects using four LLMs, and compared the generated tests with existing project tests and tests produced by baseline approaches. The results show that MOCKMILL's tests cover lines of code and kill mutants that existing tests and baseline-generated tests miss. Overall, our findings provide preliminary evidence that leveraging mocking information is a complementary and effective way to enhance LLM-based test generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19305v1">DebugRepair: Enhancing LLM-Based Automated Program Repair via Self-Directed Debugging</a></div>
    <div class="paper-meta">
      📅 2026-04-21
    </div>
    <details class="paper-abstract">
      Automated Program Repair (APR) has benefited from the code understanding and generation capabilities of Large Language Models (LLMs). Existing feedback-based APR methods iteratively refine candidate patches using test execution feedback and have shown promising results. However, most rely on outcome-level failure symptoms, such as stack traces, which show how failures are observed but fail to expose the intermediate runtime states critical for root-cause analysis. As a result, LLMs often infer bug causes without sufficient runtime evidence, leading to incorrect patches. To address this limitation, we propose DebugRepair, a self-directed debugging framework for LLM-based APR. DebugRepair enhances patch refinement with intermediate runtime evidence collected through simulated debugging. It consists of three components: test semantic purification, simulated instrumentation, and debugging-driven conversational repair. Together, they reduce noisy test context, collect runtime traces through targeted debugging statements with rule-based fallback, and progressively refine candidate patches using prior attempts and newly observed runtime states. We evaluate DebugRepair on three benchmarks across Java and Python. Experiments show that DebugRepair achieves state-of-the-art performance against 15 approaches. With GPT-3.5, it correctly fixes 224 bugs on Defects4J, outperforming prior SOTA LLM-based methods by 26.2%. With DeepSeek-V3, it correctly fixes 295 Defects4J bugs, surpassing the second-best baseline by 59 bugs. Across five additional backbone LLMs, DebugRepair improves repair performance by 51.3% over vanilla settings. Ablation studies further confirm the effectiveness of all components.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19292v1">Location Not Found: Exposing Implicit Local and Global Biases in Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 ACL 2026 main conference
    </div>
    <details class="paper-abstract">
      Multilingual large language models (LLMs) have minimized the fluency gap between languages. This advancement, however, exposes models to the risk of biased behavior, as knowledge and norms may propagate across languages. In this work, we aim to quantify models' inter- and intra-lingual biases, via their ability to answer locale-ambiguous questions. To this end, we present LocQA, a test set containing 2,156 questions in 12 languages, referring to various locale-dependent facts such as laws, dates, and measurements. The questions do not contain indications of the locales they relate to, other than the querying language itself. LLMs' responses to LocQA locale-ambiguous questions thus reveal models' implicit priors. We used LocQA to evaluate 32 models, and detected two types of structural biases. Inter-lingually, we show a global bias towards answers relevant to the US-locale, even when models are asked in languages other than English. Moreover, we discovered that this global bias is exacerbated in models that underwent instruction tuning, compared to their base counterparts. Intra-lingually, we show that when multiple locales are relevant for the same language, models act as demographic probability engines, prioritizing locales with larger populations. Taken together, insights from LocQA may help in shaping LLMs' desired local behavior, and in quantifying the impact of various training phases on different kinds of biases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19274v1">HarDBench: A Benchmark for Draft-Based Co-Authoring Jailbreak Attacks for Safe Human-LLM Collaborative Writing</a></div>
    <div class="paper-meta">
      📅 2026-04-21
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as co-authors in collaborative writing, where users begin with rough drafts and rely on LLMs to complete, revise, and refine their content. However, this capability poses a serious safety risk: malicious users could jailbreak the models-filling incomplete drafts with dangerous content-to force them into generating harmful outputs. In this paper, we identify the vulnerability of current LLMs to such draft-based co-authoring jailbreak attacks and introduce HarDBench, a systematic benchmark designed to evaluate the robustness of LLMs against this emerging threat. HarDBench spans a range of high-risk domains-including Explosives, Drugs, Weapons, and Cyberattacks-and features prompts with realistic structure and domain-specific cues to assess the model susceptibility to harmful completions. To mitigate this risk, we introduce a safety-utility balanced alignment approach based on preference optimization, training models to refuse harmful completions while remaining helpful on benign drafts. Experimental results show that existing LLMs are highly vulnerable in co-authoring contexts and our alignment method significantly reduces harmful outputs without degrading performance on co-authoring capabilities. This presents a new paradigm for evaluating and aligning LLMs in human-LLM collaborative writing settings. Our new benchmark and dataset are available on our project page at https://github.com/untae0122/HarDBench
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19262v1">CulturALL: Benchmarking Multilingual and Multicultural Competence of LLMs on Grounded Tasks</a></div>
    <div class="paper-meta">
      📅 2026-04-21
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are now deployed worldwide, inspiring a surge of benchmarks that measure their multilingual and multicultural abilities. However, these benchmarks prioritize generic language understanding or superficial cultural trivia, leaving the evaluation of grounded tasks -- where models must reason within real-world, context-rich scenarios -- largely unaddressed. To fill this gap, we present CulturALL, a comprehensive and challenging benchmark to assess LLMs' multilingual and multicultural competence on grounded tasks. CulturALL is built via a human--AI collaborative framework: expert annotators ensure appropriate difficulty and factual accuracy, while LLMs lighten the manual workload. By incorporating diverse sources, CulturALL ensures comprehensive scenario coverage. Each item is carefully designed to present a high level of difficulty, making CulturALL challenging. CulturALL contains 2,610 samples in 14 languages from 51 regions, distributed across 16 topics to capture the full breadth of grounded tasks. Experiments show that the best LLM achieves 44.48% accuracy on CulturALL, underscoring substantial room for improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.21563v4">VoteGCL: Enhancing Graph-based Recommendations with Majority-Voting LLM-Rerank Augmentation</a></div>
    <div class="paper-meta">
      📅 2026-04-21
    </div>
    <details class="paper-abstract">
      Recommendation systems often suffer from data sparsity caused by limited user-item interactions, which degrade their performance and amplify popularity bias in real-world scenarios. This paper proposes a novel data augmentation framework that leverages Large Language Models (LLMs) and item textual descriptions to enrich interaction data. By few-shot prompting LLMs multiple times to rerank items and aggregating the results via majority voting, we generate high-confidence synthetic user-item interactions, supported by theoretical guarantees based on the concentration of measure. To effectively leverage the augmented data in the context of a graph recommendation system, we integrate it into a graph contrastive learning framework to mitigate distributional shift and alleviate popularity bias. Extensive experiments show that our method improves accuracy and reduces popularity bias, outperforming strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19241v1">UniEP: Unified Expert-Parallel MoE MegaKernel for LLM Training</a></div>
    <div class="paper-meta">
      📅 2026-04-21
    </div>
    <details class="paper-abstract">
      The exponential growth in Large Language Model (LLM) parameters has transformed model training into an increasingly resource-intensive endeavor. With the stagnation of Moore's Law and the widening disparity between computation throughput and communication bandwidth, expert parallelism (EP) has emerged as a critical strategy for scaling mixture-of-experts (MoE) models. However, despite numerous proposals for optimizing EP, ranging from communication compression to computation-communication overlap, adoption within production-grade frameworks like Megatron-LM remains conservative. Existing solutions often rely on ad-hoc, complex kernels that lack adaptability across diverse optimization configurations and frequently neglect numerical stability, failing to meet the strict precision requirements of large-scale training. In this paper, we introduce UniEP, a novel system that unifies diverse EP optimization strategies into a cohesive abstraction. UniEP fuses the MoE communication and computation into MegaKernels, effectively transforming complex architectural tuning into a unified parameter search space for automated adaptability. Crucially, UniEP incorporates a deterministic token ordering mechanism that guarantees numerical consistency with sequential execution, even under aggressive overlap schedules. We evaluate UniEP on GPU clusters equipped with NVIDIA Hopper GPUs. Our results demonstrate that UniEP achieves 1.03$\times$-1.38$\times$ speedups over state-of-the-art work, effectively mitigating communication bottlenecks while maintaining the rigorous accuracy standards required for production LLM training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19221v1">UAF: A Unified Audio Front-end LLM for Full-Duplex Speech Interaction</a></div>
    <div class="paper-meta">
      📅 2026-04-21
    </div>
    <details class="paper-abstract">
      Full-duplex speech interaction, as the most natural and intuitive mode of human communication, is driving artificial intelligence toward more human-like conversational systems. Traditional cascaded speech processing pipelines suffer from critical limitations, including accumulated latency, information loss, and error propagation across modules. To address these issues, recent efforts focus on the end-to-end audio large language models (LLMs) like GPT-4o, which primarily unify speech understanding and generation task. However, most of these models are inherently half-duplex, and rely on a suite of separate, task-specific front-end components, such as voice activity detection (VAD) and turn-taking detection (TD). In our development of speech assistant, we observed that optimizing the speech front-end is equally crucial as advancing the back-end unified model for achieving seamless, responsive interactions. To bridge this gap, we propose the first unified audio front-end LLM (UAF) tailored for full-duplex speech systems. Our model reformulates diverse audio front-end tasks into a single auto-regressive sequence prediction problem, including VAD, TD, speaker recognition (SR), automatic speech recognition (ASR) and question answer (QA). It takes streaming fixed-duration audio chunk (e.g., 600 ms) as input, leverages a reference audio prompt to anchor the target speaker at the beginning, and regressively generates discrete tokens encoding both semantic content and system-level state controls (e.g., interruption signals). Experiments demonstrate that our model achieves leading performance across multiple audio front-end tasks and significantly enhances response latency and interruption accuracy in real-world interaction scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19204v1">Auditing LLMs for Algorithmic Fairness in Casenote-Augmented Tabular Prediction</a></div>
    <div class="paper-meta">
      📅 2026-04-21
    </div>
    <details class="paper-abstract">
      LLMs are increasingly being considered for prediction tasks in high-stakes social service settings, but their algorithmic fairness properties in this context are poorly understood. In this short technical report, we audit the algorithmic fairness of LLM-based tabular classification on a real housing placement prediction task, augmented with street outreach casenotes from a nonprofit partner. We audit multi-class classification error disparities. We find that a fine-tuned model augmented with casenote summaries can improve accuracy while reducing algorithmic fairness disparities. We experiment with variable importance improvements to zero-shot tabular classification and find mixed results on resulting algorithmic fairness. Overall, given historical inequities in housing placement, it is crucial to audit LLM use. We find that leveraging LLMs to augment tabular classification with casenote summaries can safely leverage additional text information at low implementation burden. The outreach casenotes are fairly short and heavily redacted. Our assessment is that LLM zero-shot classification does not introduce additional textual biases beyond algorithmic biases in tabular classification. Combining fine-tuning and leveraging casenote summaries can improve accuracy and algorithmic fairness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.22811v5">Highly Efficient and Effective LLMs with Multi-Boolean Architectures</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 ICLR 2026 (Main Conference)
    </div>
    <details class="paper-abstract">
      Weight binarization has emerged as a promising strategy to reduce the complexity of large language models (LLMs). Existing approaches fall into post-training binarization, which is simple but causes severe performance loss, and training-aware methods, which depend on full-precision latent weights, adding complexity and limiting efficiency. We propose a novel framework that represents LLMs with multi-kernel Boolean parameters and, for the first time, enables direct finetuning LMMs in the Boolean domain, eliminating the need for latent weights. This enhances representational capacity and dramatically reduces complexity during both finetuning and inference. Extensive experiments across diverse LLMs show our method outperforms recent ultra low-bit quantization and binarization techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11675v3">Right for the Wrong Reasons: Epistemic Regret Minimization for LLM Causal Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 34 pages, 15 tables, 16 figures
    </div>
    <details class="paper-abstract">
      Large language models may answer causal questions correctly for the wrong reasons, substituting associational shortcuts P(Y|X) for the interventional query P(Y|do(X)). Current RL methods reward what the model answers but not why, reinforcing these shortcuts until distribution shift exposes them. We introduce Epistemic Regret Minimization (ERM), a framework that identifies causal reasoning flaws from reasoning traces, with no ground-truth labels. On CausalT5K (N=1,360, 6 frontier LLMs), models bifurcate: compliant models already correct under outcome-only reprompting, but reasoning-heavy models (GPT-4 Turbo, GPT-5.2, Claude Sonnet 3.5) resist outcome-only correction yet respond significantly to ERM's targeted causal critique. Ablation on 4,054 scenarios confirms causal content, not prompt structure alone, drives correction for stubborn models (p=0.006), and a scenario-blind judge argues against answer leakage. Cross-benchmark evaluation on CLadder confirms Rung Collapse generalizes beyond CausalT5K. We extend ERM to cross-episode RL, where interventional evidence accumulates into a reward signal for open-domain problems lacking ground-truth verifiers. A separation theorem proves outcome-only RL cannot distinguish correct from flawed causal models in confounded environments, and preliminary experiments across four LLMs show epistemic reward carries signal where outcome reward does not. This establishes signal existence, not yet policy improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.18146v2">Modular Representation Compression: Adapting LLMs for Efficient and Effective Recommendations</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 SIGIR 2026
    </div>
    <details class="paper-abstract">
      Recently, large language models (LLMs) have advanced recommendation systems (RSs), and recent works have begun to explore how to integrate LLMs into industrial RSs. While most approaches deploy LLMs offline to generate and pre-cache augmented representations for RSs, high-dimensional representations from LLMs introduce substantial storage and computational costs. Thus, it is crucial to compress LLM representations effectively. However, we identify a counterintuitive phenomenon during representation compression: Mid-layer Representation Advantage (MRA), where representations from middle layers of LLMs outperform those from final layers in recommendation tasks. This degraded final layer renders existing compression methods, which typically compress on the final layer, suboptimal. We interpret this based on modularity theory that LLMs develop spontaneous internal functional modularity and force the final layer to specialize in the proxy training task. Thus, we propose \underline{M}odul\underline{a}r \underline{R}epresentation \underline{C}ompression (MARC) to explicitly control the modularity of LLMs. First, Modular Adjustment explicitly introduces compression and task adaptation modules, enabling the LLM to operate strictly as a representation-learning module. Next, to ground each module to its specific task, Modular Task Decoupling uses information constraints and different network structures to decouple tasks. Extensive experiments validate that MARC addresses MRA and produces efficient representations. Notably, MARC achieved a 2.82% eCPM lift in an online A/B test within a large-scale commercial search advertising scenario.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19162v1">Mind the Unseen Mass: Unmasking LLM Hallucinations via Soft-Hybrid Alphabet Estimation</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 7 pages, 1 figure, 3 tables
    </div>
    <details class="paper-abstract">
      This paper studies uncertainty quantification for large language models (LLMs) under black-box access, where only a small number of responses can be sampled for each query. In this setting, estimating the effective semantic alphabet size--that is, the number of distinct meanings expressed in the sampled responses--provides a useful proxy for downstream risk. However, frequency-based estimators tend to undercount rare semantic modes when the sample size is small, while graph-spectral quantities alone are not designed to estimate semantic occupancy accurately. To address this issue, we propose SHADE (Soft-Hybrid Alphabet Dynamic Estimator), a simple and interpretable estimator that combines Generalized Good-Turing coverage with a heat-kernel trace of the normalized Laplacian constructed from an entailment-weighted graph over sampled responses. The estimated coverage adaptively determines the fusion rule: under high coverage, SHADE uses a convex combination of the two signals, while under low coverage it applies a LogSumExp fusion to emphasize missing or weakly observed semantic modes. A finite-sample correction is then introduced to stabilize the resulting cardinality estimate before converting it into a coverage-adjusted semantic entropy score. Experiments on pooled semantic alphabet-size estimation against large-sample references and on QA incorrectness detection show that SHADE achieves the strongest improvements in the most sample-limited regime, while the performance gap narrows as the number of samples increases. These results suggest that hybrid semantic occupancy estimation is particularly beneficial when black-box uncertainty quantification must operate under tight sampling budgets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19157v1">SAW-INT4: System-Aware 4-Bit KV-Cache Quantization for Real-World LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-04-21
    </div>
    <details class="paper-abstract">
      KV-cache memory is a major bottleneck in real-world LLM serving, where systems must simultaneously support latency-sensitive small-batch requests and high-throughput concurrent workloads. Although many KV-cache compression methods improve offline accuracy or compression ratio, they often violate practical serving constraints such as paged memory layouts, regular memory access, and fused attention execution, limiting their effectiveness in deployment. In this work, we identify the minimal set of 4-bit KV-cache quantization methods that remain viable under these constraints. Our central finding is that a simple design--token-wise INT4 quantization with block-diagonal Hadamard rotation--consistently achieves the best accuracy-efficiency trade-off. Across multiple models and benchmarks, this approach recovers nearly all of the accuracy lost by naive INT4, while more complex methods such as vector quantization and Hessian-aware quantization provide only marginal additional gains once serving compatibility is taken into account. To make this practical, we implement a fused rotation-quantization kernel that integrates directly into paged KV-cache layouts and introduces zero measurable end-to-end overhead, matching plain INT4 throughput across concurrency levels. Our results show that effective KV-cache compression is fundamentally a systems co-design problem: under real serving constraints, lightweight block-diagonal Hadamard rotation is a viable method that delivers near-lossless accuracy without sacrificing serving efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.02923v2">Council Mode: Mitigating Hallucination and Bias in LLMs via Multi-Agent Consensus</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 13 pages, 8 figures, technical report
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), particularly those employing Mixture-of-Experts (MoE) architectures, have achieved remarkable capabilities across diverse natural language processing tasks. However, these models frequently suffer from hallucinations -- generating plausible but factually incorrect content -- and exhibit systematic biases that are amplified by uneven expert activation during inference. In this paper, we propose the Council Mode, a novel multi-agent consensus framework that addresses these limitations by dispatching queries to multiple heterogeneous frontier LLMs in parallel and synthesizing their outputs through a dedicated consensus model. The Council pipeline operates in three phases: (1) an intelligent triage classifier that routes queries based on complexity, (2) parallel expert generation across architecturally diverse models, and (3) a structured consensus synthesis that explicitly identifies agreement, disagreement, and unique findings before producing the final response. We implement and evaluate this architecture within an open-source AI workspace. Our comprehensive evaluation across multiple benchmarks demonstrates that the Council Mode achieves a 35.9% relative reduction in hallucination rates on the HaluEval benchmark and a 7.8-point improvement on TruthfulQA compared to the best-performing individual model, while maintaining significantly lower bias variance across domains. We provide the mathematical formulation of the consensus mechanism, detail the system architecture, and present extensive empirical results with ablation studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19149v1">How Do Answer Tokens Read Reasoning Traces? Self-Reading Patterns in Thinking LLMs for Quantitative Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 Accepted in the Findings of ACL 2026
    </div>
    <details class="paper-abstract">
      Thinking LLMs produce reasoning traces before answering. Prior activation steering work mainly targets on shaping these traces. It remains less understood how answer tokens actually read and integrate the reasoning to produce reliable outcomes. Focusing on quantitative reasoning, we analyze the answer-to-reasoning attention and observe a benign self-reading pattern aligned with correctness, characterized by a forward drift of the reading focus along the reasoning trace and a persistent concentration on key semantic anchors, whereas incorrect solutions exhibit diffuse and irregular attention pattern. We interpret this as internal certainty during answer decoding, where the model commits to a viable solution branch and integrates key evidence. Following this, we propose a training-free steering method driven by Self-Reading Quality (SRQ) scores combining geometric metrics for process control with semantic metrics for content monitoring. SRQ selects data to build steering vectors that guide inference toward benign self-reading and away from uncertain and disorganized reading. Experiments show that our method yields consistent accuracy gains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19128v1">GraphRAG-IRL: Personalized Recommendation with Graph-Grounded Inverse Reinforcement Learning and LLM Re-ranking</a></div>
    <div class="paper-meta">
      📅 2026-04-21
    </div>
    <details class="paper-abstract">
      Personalized recommendation requires models that capture sequential user preferences while remaining robust to sparse feedback and semantic ambiguity. Recent work has explored large language models (LLMs) as recommenders and re-rankers, but pure prompt-based ranking often suffers from poor calibration, sensitivity to candidate ordering, and popularity bias. These limitations make LLMs useful semantic reasoners, but unreliable as standalone ranking engines. We present \textbf{GraphRAG-IRL}, a hybrid recommendation framework that combines graph-grounded feature construction, inverse reinforcement learning (IRL), and persona-guided LLM re-ranking. Our method constructs a heterogeneous knowledge graph over items, categories, and concepts, retrieves both individual and community preference context, and uses these signals to train a Maximum Entropy IRL model for calibrated pre-ranking. An LLM is then applied only to a short candidate list, where persona-guided prompts provide complementary semantic judgments that are fused with IRL rankings. Experiments show that GraphRAG-IRL is a strong standalone recommender: IRL-MLP with GraphRAG improves NDCG@10 by 15.7\% on MovieLens and 16.6\% on KuaiRand over supervised baselines. The results also show that IRL and GraphRAG are superadditive, with the combined gain exceeding the sum of their individual improvements. Persona-guided LLM fusion further improves ranking quality, yielding up to 16.8\% NDCG@10 improvement over the IRL-only baseline on MovieLens ml-1m, while score fusion on KuaiRand provides consistent gains of 4--6\% across LLM providers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18572v2">One Persona, Many Cues, Different Results: How Sociodemographic Cues Impact LLM Personalization</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 ACL 2026 Main Conference
    </div>
    <details class="paper-abstract">
      Personalization of LLMs by sociodemographic subgroup often improves user experience, but can also introduce or amplify biases and unfair outcomes across groups. Prior work has employed so-called personas, sociodemographic user attributes conveyed to a model, to study bias in LLMs by relying on a single cue to prompt a persona, such as user names or explicit attribute mentions. This disregards LLM sensitivity to prompt variation and the rarity of some cues in real interactions (external validity). We compare six commonly used persona cues across seven open and proprietary LLMs on four writing and advice tasks. While cues are overall highly correlated, they produce substantial variance in responses across personas that can change findings on persona-induced differences and bias. We therefore caution against claims based on single persona cues, especially when they are overly explicit and have low external validity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19124v1">Detoxification for LLM: From Dataset Itself</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 Accepted to Main Conference of ACL 2026
    </div>
    <details class="paper-abstract">
      Existing detoxification methods for large language models mainly focus on post-training stage or inference time, while few tackle the source of toxicity, namely, the dataset itself. Such training-based or controllable decoding approaches cannot completely suppress the model's inherent toxicity, whereas detoxifying the pretraining dataset can fundamentally reduce the toxicity that the model learns during training. Hence, we attempt to detoxify directly on raw corpora with SoCD (Soft Contrastive Decoding), which guides an LLM to localize and rewrite toxic spans in raw data while preserving semantics, in our proposed HSPD (Hierarchical Semantic-Preserving Detoxification) pipeline, yielding a detoxified corpus that can drop-in replace the original for fine-tuning or other training. On GPT2-XL, HSPD attains state-of-the-art detoxification, reducing Toxicity Probability (TP) from 0.42 to 0.18 and Expected Maximum Toxicity (EMT) from 0.43 to 0.20. We further validate consistent best-in-class results on LLaMA2-7B, OPT-6.7B, and Falcon-7B. These findings show that semantics-preserving, corpus-level rewriting with HSPD effectively suppresses downstream toxicity while retaining data utility and allowing seamless source-level mitigation, thereby reducing the cost of later model behavior adjustment. (Code is available at: https://github.com/ntsw2001/data_detox_for_llm)
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19118v1">DP-FlogTinyLLM: Differentially private federated log anomaly detection using Tiny LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-21
    </div>
    <details class="paper-abstract">
      Modern distributed systems generate massive volumes of log data that are critical for detecting anomalies and cyber threats. However, in real world settings, these logs are often distributed across multiple organizations and cannot be centralized due to privacy and security constraints. Existing log anomaly detection methods, including recent large language model (LLM) based approaches, largely rely on centralized training and are not suitable for such environments. In this paper, we propose DP-FLogTinyLLM, a privacy preserving federated framework for log anomaly detection using parameter efficient LLMs. Our approach enables collaborative learning without sharing raw log data by integrating federated optimization with differential privacy. To ensure scalability in resource constrained environments, we employ low rank adaptation (LoRA) for efficient fine tuning of Tiny LLMs at each client. Empirical results on the Thunderbird and BGL datasets show that the proposed framework matches the performance of centralized LLM based methods, while incurring additional computational overhead due to privacy mechanisms. Compared to existing federated baselines, DP-FLogTinyLLM consistently achieves higher precision and F1-score, with particularly strong gains on the Thunderbird dataset, highlighting its effectiveness in detecting anomalies while minimizing false positives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.18177v2">STaD: Scaffolded Task Design for Identifying Compositional Skill Gaps in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 9 pages, 3 figures, 3 tables, ACL Findings 2026
    </div>
    <details class="paper-abstract">
      Benchmarks are often used as a standard to understand LLM capabilities in different domains. However, aggregate benchmark scores provide limited insight into compositional skill gaps of LLMs and how to improve them. To make these weaknesses visible, we propose Scaffolded Task Design (STaD) framework. STaD generates controlled variations of benchmark tasks based on the concept of scaffolding, which introduces structured, incremental support in a step-by-step manner. Rather than inspecting failures individually, this approach enables systematic and scalable probing of model behavior by identifying the specific reasoning skill compositions they lack. Treating the LLM as a black box, our experiments on six models of varying sizes reveal multiple failure points in three reasoning benchmarks and highlight each model's unique and distinct skill gaps.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19086v1">MUCOCO: Automated Consistency Testing of Code LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-21
    </div>
    <details class="paper-abstract">
      Code LLMs often portray inconsistent program behaviors. Developers typically employ benchmarks to assess Code LLMs, but most benchmarks are hand-crafted, static and do not target consistency property. In this work, we pose the scientific question: how can we automatically discover inconsistent program behaviors in Code LLMs? To address this challenge, we propose an automated consistency testing method, called MUCOCO, which employs semantic-preserving mutation analysis to expose inconsistent behaviors in code LLMs. Given a coding query, MUCOCO automatically transforms its program into semantically equivalent programs (aka mutants) and detects inconsistencies between the mutants and the original program (e.g., different output or test failure). We evaluate MUCOCO using four (4) coding tasks and seven (7) LLMs. Results show that MUCOCO is effective in exposing inconsistency and outperforms the closest baseline (TURBULENCE). About one in seven (15%) inputs generated by MUCOCO exposed inconsistencies. Our work motivates the need to test Code LLMs for consistency property
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22099v3">Decomposed Trust: Privacy, Adversarial Robustness, Ethics, and Fairness in Low-Rank LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 14 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have driven major advances across domains, yet their massive size hinders deployment in resource-constrained settings. Low-rank factorization addresses this challenge by compressing models to effectively reduce their computation and memory consumption while maintaining accuracy. While these compressed models boast benign performance and system-level advantages, their trustworthiness implications remain poorly understood. In this paper, we present the first comprehensive study of how low-rank factorization affects LLM trustworthiness across privacy, adversarial robustness, ethics, and fairness, complemented by an explainability-driven analysis of the internal mechanisms behind these trust-related changes. We evaluate multiple LLMs of different sizes and architectures compressed with various low-rank factorization algorithms, revealing key insights: (1)low-rank factorization preserves training data privacy but weakens the protection of personally identifiable information during conversations; (2)adversarial robustness is generally enhanced under compression; (3)ethics degrades in zero-shot prompting but partially recovers in few-shot prompting; (4)fairness declines under compression. Beyond compression, we investigate how model scale and fine-tuning affect trustworthiness. Additionally, to move beyond black-box analysis, we employ a gradient-based attribution to identify which layers of LLMs contribute most to adversarial robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19071v1">HoWToBench: Holistic Evaluation for LLM's Capability in Human-level Writing using Tree of Writing</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 49 pages, 6 figures, 19 tables, ACL 2026 main
    </div>
    <details class="paper-abstract">
      Evaluating the writing capabilities of large language models (LLMs) remains a significant challenge due to the multidimensional nature of writing skills and the limitations of existing metrics. LLM's performance in thousand-words level and open-ended writing is inadequately assessed by traditional reference-based metrics or modern LLM-as-a-judge methods. We propose Tree-of-Writing (ToW), to resolve the implicit inconsistency often found when LLM-as-a-judge aggregates all sub-features in text evaluation. ToW incorporates a tree-structured workflow by explicitly modeling the aggregation weights of sub-features. We also present HowToBench, a large-scale Chinese writing benchmark encompassing 12 genres and 1302 instructions across three task categories: contextual completion, outline-guided writing, and open-ended generation. ToW successfully mitigates the biases, achieving a 0.93 Pearson correlation with human judgments. Furthermore, we detect that both overlap-based text generation metrics and popular LLM-as-a-judge practices are vulnerable to textual disturbances, while ToW is robust to them. We also uncover a negative correlation between input length and content-related scores in the Guide task, showcasing that it cannot be simply improved by input-side information piling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19070v1">TRN-R1-Zero: Text-rich Network Reasoning via LLMs with Reinforcement Learning Only</a></div>
    <div class="paper-meta">
      📅 2026-04-21
    </div>
    <details class="paper-abstract">
      Zero-shot reasoning on text-rich networks (TRNs) remains a challenging frontier, as models must integrate textual semantics with relational structure without task-specific supervision. While graph neural networks rely on fixed label spaces and supervised objectives, recent large language model (LLM)-based approaches often overlook graph context or depend on distillation from larger models, limiting generalisation. We propose TRN-R1-Zero, a post-training framework for TRN reasoning trained solely via reinforcement learning. TRN-R1-Zero directly optimises base LLMs using a Neighbour-aware Group Relative Policy Optimisation objective that dynamically adjusts rewards based on a novel margin gain metric for the informativeness of neighbouring signals, effectively guiding the model toward relational reasoning. Unlike prior methods, TRN-R1-Zero requires no supervised fine-tuning or chain-of-thought data generated from large reasoning models. Extensive experiments across citation, hyperlink, social and co-purchase TRN benchmarks demonstrate the superiority and robustness of TRN-R1-Zero. Moreover, relying strictly on node-level training, TRN-R1-Zero achieves zero-shot inference on edge- and graph-level tasks, extending beyond cross-domain transfer. The codebase is publicly available at https://github.com/superallen13/TRN-R1-Zero.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.18333v2">Position: LLM Watermarking Should Align Stakeholders' Incentives for Practical Adoption</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 ACL2026
    </div>
    <details class="paper-abstract">
      Despite progress in watermarking algorithms for large language models (LLMs), real-world deployment remains limited. We argue that this gap stems from misaligned incentives among LLM providers, platforms, and end users, which manifest as three key barriers: competitive risk, detection-tool governance, and attribution issues. We revisit three classes of watermarking through this lens. \emph{Model watermarking} naturally aligns with LLM provider interests, yet faces new challenges in open-source ecosystems. \emph{LLM text watermarking} offers modest provider benefit when framed solely as an anti-misuse tool, but can gain traction in narrowly scoped settings such as dataset de-contamination or user-controlled provenance. \emph{In-context watermarking} (ICW) is tailored for trusted parties, such as conference organizers or educators, who embed hidden watermarking instructions into documents. If a dishonest reviewer or student submits this text to an LLM, the output carries a detectable watermark indicating misuse. This setup aligns incentives: users experience no quality loss, trusted parties gain a detection tool, and LLM providers remain neutral by simply following watermark instructions. We advocate for a broader exploration of incentive-aligned methods, with ICW as an example, in domains where trusted parties need reliable tools to detect misuse. More broadly, we distill design principles for incentive-aligned, domain-specific watermarking and outline future research directions. Our position is that the practical adoption of LLM watermarking requires aligning stakeholder incentives in targeted application domains and fostering active community engagement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19060v1">Reinforcement Learning Improves LLM Accuracy and Reasoning in Disease Classification from Radiology Reports</a></div>
    <div class="paper-meta">
      📅 2026-04-21
    </div>
    <details class="paper-abstract">
      Accurate disease classification from radiology reports is essential for many applications. While supervised fine-tuning (SFT) of lightweight LLMs improves accuracy, it can degrade reasoning. We propose a two-stage approach: SFT on disease labels followed by Group Relative Policy Optimization (GRPO) to refine predictions by optimizing accuracy and format without reasoning supervision. Across three radiologist-annotated datasets, SFT outperformed baselines and GRPO further improved classification and enhanced reasoning recall and comprehensiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19049v1">Refute-or-Promote: An Adversarial Stage-Gated Multi-Agent Review Methodology for High-Precision LLM-Assisted Defect Discovery</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 10 pages, 3 tables. Artifacts: https://github.com/abhinavagarwal07/refute-or-promote (Zenodo DOI: 10.5281/zenodo.19668799)
    </div>
    <details class="paper-abstract">
      LLM-assisted defect discovery has a precision crisis: plausible-but-wrong reports overwhelm maintainers and degrade credibility for real findings. We present Refute-or-Promote, an inference-time reliability pattern combining Stratified Context Hunting (SCH) for candidate generation, adversarial kill mandates, context asymmetry, and a Cross-Model Critic (CMC). Adversarial agents attempt to disprove candidates at each promotion gate; cold-start reviewers are intended to reduce anchoring cascades; cross-family review can catch correlated blind spots that same-family review misses. Over a 31-day campaign across 7 targets (security libraries, the ISO C++ standard, major compilers), the pipeline killed roughly 79% of 171 candidates before advancing to disclosure (retrospective aggregate); on a consolidated-protocol subset (lcms2, wolfSSL; n=30), the prospective kill rate was 83%. Outcomes: 4 CVEs (3 public, 1 embargoed); LWG 4549 accepted to the C++ working paper; 5 merged C++ editorial PRs; 3 compiler conformance bugs; 8 merged security-related fixes without CVE; an RFC 9000 errata filed under committee review; and 1+ FIPS 140-3 normative compliance issues under coordinated disclosure -- all evaluated by external acceptance, not benchmarks. The most instructive failure: ten dedicated reviewers unanimously endorsed a non-existent Bleichenbacher padding oracle in OpenSSL's CMS module; it was killed only by a single empirical test, motivating the mandatory empirical gate. No vulnerability was discovered autonomously; the contribution is external structure that filters LLM agents' persistent false positives. As a preliminary transfer test beyond defect discovery, a simplified cross-family critique variant also solved five previously unsolved SymPy instances on SWE-bench Verified and one SWE-rebench hard task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.11765v3">OMAC: A Holistic Optimization Framework for LLM-Based Multi-Agent Collaboration</a></div>
    <div class="paper-meta">
      📅 2026-04-21
    </div>
    <details class="paper-abstract">
      Agents powered by advanced large language models (LLMs) have demonstrated impressive capabilities across diverse complex applications. Recently, Multi-Agent Systems (MAS), wherein multiple agents collaborate and communicate with each other, have exhibited enhanced capabilities in complex tasks, such as high-quality code generation and arithmetic reasoning. However, the development of such systems often relies on handcrafted methods, and the literature on systematic design and optimization of LLM-based MAS remains limited. In this work, we introduce OMAC, a general framework designed for holistic optimization of LLM-based MAS. Specifically, we identify five key optimization dimensions for MAS, encompassing both agent functionality and collaboration structure. Building upon these dimensions, we first propose a general algorithm, utilizing two actors termed the Semantic Initializer and the Contrastive Comparator, to optimize any single dimension. Then, we present an algorithm for joint optimization across multiple dimensions. Extensive experiments demonstrate the superior performance of OMAC on code generation, arithmetic reasoning, and general reasoning tasks against state-of-the-art approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19035v1">LLM-Viterbi: Semantic-Aware Decoding for Convolutional Codes</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 6 pages, 6 figures. Accepted to IEEE International Symposium on Information Theory (ISIT) 2026. Code available at https://github.com/Todd-6/LLM-Viterbi
    </div>
    <details class="paper-abstract">
      Traditional wireless communications rely solely on bit-level channel coding for error correction, without exploiting the inherent linguistic structure of the data source. This paper proposes a large language model (LLM) Viterbi decoder that integrates LLM priors into the Viterbi decoding for text transmission over AWGN channels. The proposed decoder maintains multiple candidate paths during the Viterbi decoding and periodically evaluates path reliabilities using a fine-tuned Byte-level T5 (ByT5) language model. By combining channel reliability metrics with semantic probability from the LLM, it outputs the path that maximizes the joint likelihood of channel observations and linguistic coherence. Simulations show that our decoder achieves significant performance gains over conventional Viterbi decoding in terms of both block error rate (BLER) and semantic similarity. For convolutional codes with constraint length 3, it achieves approximately 1.5 dB more coding gain in BLER, with over 50% improvements in semantic similarity. The framework can extend to other structured data sources beyond text.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19031v1">SAGE: Signal-Amplified Guided Embeddings for LLM-based Vulnerability Detection</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 24 pages, 6 figures, 6 tables. Accepted by ISSTA 2026
    </div>
    <details class="paper-abstract">
      Software vulnerabilities are a primary threat to modern infrastructure. While static analysis and Graph Neural Networks have long served as the foundation for vulnerability detection, the emergence of Large Language Models (LLMs) has introduced a transformative paradigm driven by superior semantic reasoning and cross-environment generalization. However, in the context of LLM-based vulnerability detection, we identify a fundamental bottleneck in these models termed \textbf{Signal Submersion}: a state where features related to vulnerability are activated internally but numerically overwhelmed by dominant functional semantics. To address this, we propose \textbf{SAGE} (\textbf{S}ignal-\textbf{A}mplified \textbf{G}uided \textbf{E}mbeddings), a framework that shifts from passive signal submersion to active signal recovery. SAGE integrates task-conditional Sparse Autoencoders (SAEs) to isolate and amplify these faint vulnerability signals. Extensive evaluations on BigVul, PrimeVul, and PreciseBugs demonstrate that SAGE achieves state-of-the-art performance. Notably, SAGE mitigates Signal Submersion by increasing the internal Signal-to-Noise Ratio (SNR) by 12.7$\times$ via sparse manifold projection. This mechanistic intervention enables a 7B model to achieve up to 318\% Matthews Correlation Coefficient (MCC) gains on unseen distributions and a 319\% gain on classic datasets. By maintaining robust performance across 13 programming languages and outperforming 34B baselines, SAGE establishes a more efficient and scalable path to software security than simple parameter scaling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.16937v2">No One Fits All: From Fixed Prompting to Learned Routing in Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 Accepted as a short findings paper at ACL 2026
    </div>
    <details class="paper-abstract">
      Translation-based prompting is widely used in multilingual LLMs, yet its effectiveness varies across languages and tasks. We evaluate prompting strategies across ten languages of different resource levels and four benchmarks. Our analysis shows that no single strategy is universally optimal. Translation strongly benefits low-resource languages even when translation quality is imperfect, high-resource languages gain little, and prompt-based self-routing underperforms explicit translation. Motivated by these findings, we formulate prompting strategy selection as a learned decision problem and introduce lightweight classifiers that predict whether native or translation-based prompting is optimal for each instance. The classifiers achieve statistically significant improvements over fixed strategies across four benchmarks and generalize to unseen task formats not observed during training. Further analysis reveals that language resource level, rather than translation quality alone, determines when translation is beneficial.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.16763v2">LLM-Extracted Covariates for Clinical Causal Inference: Rethinking Integration Strategies</a></div>
    <div class="paper-meta">
      📅 2026-04-21
    </div>
    <details class="paper-abstract">
      Causal inference from electronic health records (EHR) is fundamentally limited by unmeasured confounding: critical clinical states such as frailty, goals of care, and mental status are documented in free-text notes but absent from structured data. Large language models can extract these latent confounders as interpretable, structured covariates, yet how to effectively integrate them into causal estimation pipelines has not been systematically studied. Using the MIMIC-IV database with 21,859 sepsis patients, we compare seven covariate-integration strategies for estimating the effect of early vasopressor initiation on 28-day mortality, spanning tabular-only baselines, traditional NLP representations, and three LLM-augmented approaches. A central finding is that not all integration strategies are equally effective: directly augmenting the propensity score model with LLM covariates achieves the best performance, while dual-caliper matching on text-derived categorical distances restricts the donor pool and degrades estimation. In semi-synthetic experiments with known ground-truth effects, LLM-augmented propensity scores reduce estimation bias from 0.0143 to 0.0003 relative to tabular-only methods, and this advantage persists under substantial simulated extraction error. On real data, incorporating LLM-extracted covariates reduces the estimated treatment effect from 0.055 to 0.027, directionally consistent with the CLOVERS randomized trial, and a doubly robust estimator yielding 0.019 confirms the robustness of this finding. Our results offer practical guidance on when and how text-derived covariates improve causal estimation in critical care.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19018v1">Local Linearity of LLMs Enables Activation Steering via Model-Based Linear Optimal Control</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Inference-time LLM alignment methods, particularly activation steering, offer an alternative to fine-tuning by directly modifying activations during generation. Existing methods, however, often rely on non-anticipative interventions that ignore how perturbations propagate through transformer layers and lack online error feedback, resulting in suboptimal, open-loop control. To address this, we show empirically that, despite the nonlinear structure of transformer blocks, layer-wise dynamics across multiple LLM architectures and scales are well-approximated by locally-linear models. Exploiting this property, we model LLM inference as a linear time-varying dynamical system and adapt the classical linear quadratic regulator to compute feedback controllers using layer-wise Jacobians, steering activations toward desired semantic setpoints in closed-loop with minimal computational overhead and no offline training. We also derive theoretical bounds on setpoint tracking error, enabling formal guarantees on steering performance. Using a novel adaptive semantic feature setpoint signal, our method yields robust, fine-grained behavior control across models, scales, and tasks, including state-of-the-art modulation of toxicity, truthfulness, refusal, and arbitrary concepts, surpassing baseline steering methods. Our code is available at: https://github.com/trustworthyrobotics/lqr-activation-steering
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19015v1">FedProxy: Federated Fine-Tuning of LLMs via Proxy SLMs and Heterogeneity-Aware Fusion</a></div>
    <div class="paper-meta">
      📅 2026-04-21
    </div>
    <details class="paper-abstract">
      Federated fine-tuning of Large Language Models (LLMs) is obstructed by a trilemma of challenges: protecting LLMs intellectual property (IP), ensuring client privacy, and mitigating performance loss on heterogeneous data. Existing methods like Offsite-Tuning (OT) secure the LLMs IP by having clients train only lightweight adapters, yet our analysis reveals they suffer from a fundamental performance bottleneck, leaving a significant gap compared to centralized training. To bridge this gap, we introduce FedProxy, a new federated adaptation framework. FedProxy replaces weak adapters with a unified, powerful Proxy Small Language Model (SLM), compressed from the proprietary LLM, to serve as a high-fidelity surrogate for collaborative fine-tuning. Our framework systematically resolves the trilemma through a three-stage architecture: (i) Efficient Representation via server-guided compression to create a resource-friendly proxy; (ii) Robust Optimization through an interference-mitigating aggregation strategy to handle data heterogeneity; and (iii) Effortless Fusion via a training-free "plug-in" mechanism to integrate learned knowledge back into the LLM. Experiments show FedProxy significantly outperforms OT methods and approaches centralized performance, establishing a new benchmark for secure and high-performance federated LLM adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02304v2">When Does Verification Pay Off? A Closer Look at LLMs as Solution Verifiers</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 Accepted at ICLR 2026 AI with Recursive Self-Improvement workshop
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can act as both problem solvers and solution verifiers, where the latter select high-quality answers from a pool of solver-generated candidates. This raises the question of under what conditions verification pays off in solver-verifier systems. Prior work has conducted only limited studies of the factors influencing verification performance, focusing primarily on self-verification and examining neither the relationship between solver and verifier model families nor the effects of reasoning post-training. To rectify this, we present a systematic study across 37 models spanning multiple families, sizes, and base vs. post-trained variants, evaluated on 9 benchmarks covering logical reasoning, structured puzzles, symbolic computation, mathematics, commonsense, factual recall, and domain knowledge. In order to support our analysis, we introduce and empirically validate verifier gain, a metric that predicts the performance improvements from test-time verifier-based rejection sampling. Our experiments find that 1) verification across model families is more effective than either self-verification or verification within the same family, and more generally that the benefits of verification decrease as the solver and verifier become more similar, 2) reasoning post-training weakens self-improvement abilities but strengthens cross-family improvement, and 3) some tasks are inherently more amenable to improvement through verification, particularly mathematical and logical tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.05188v4">Plug-and-Play Dramaturge: A Divide-and-Conquer Approach for Iterative Narrative Script Refinement via Collaborative LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-21
    </div>
    <details class="paper-abstract">
      Although LLMs have been widely adopted for creative content generation, a single-pass process often struggles to produce high-quality long narratives. How to effectively revise and improve long narrative scripts like scriptwriters remains a significant challenge, as it demands a comprehensive understanding of the entire context to identify global structural issues and local detailed flaws, as well as coordinating revisions at multiple granularities and locations. Direct modifications by LLMs typically introduce inconsistencies between local edits and the overall narrative requirements. To address these issues, we propose Dramaturge, a task and feature oriented divide-and-conquer approach powered by hierarchical multiple LLM agents. It consists of a Global Review stage to grasp the overall storyline and structural issues, a Scene-level Review stage to pinpoint detailed scene and sentence flaws, and a Hierarchical Coordinated Revision stage that coordinates and integrates structural and detailed improvements throughout the script. The top-down task flow ensures that high-level strategies guide local modifications, maintaining contextual consistency. The review and revision workflow follows a coarse-to-fine iterative process, continuing through multiple rounds until no further substantive improvements can be made. Comprehensive experiments show that Dramaturge significantly outperforms all baselines in terms of script-level overall quality and scene-level details. Our approach is plug-and-play and can be easily integrated into existing methods to improve the generated scripts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.15302v1">Diagnosing LLM Judge Reliability: Conformal Prediction Sets and Transitivity Violations</a></div>
    <div class="paper-meta">
      📅 2026-04-16
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      LLM-as-judge frameworks are increasingly used for automatic NLG evaluation, yet their per-instance reliability remains poorly understood. We present a two-pronged diagnostic toolkit applied to SummEval: $\textbf{(1)}$ a transitivity analysis that reveals widespread per-input inconsistency masked by low aggregate violation rates ($\barρ = 0.8$-$4.1\%$), with $33$-$67\%$ of documents exhibiting at least one directed 3-cycle; and $\textbf{(2)}$ split conformal prediction sets over 1-5 Likert scores providing theoretically-guaranteed $\geq(1{-}α)$ coverage, with set width serving as a per-instance reliability indicator ($r_s = {+}0.576$, $N{=}1{,}918$, $p < 10^{-100}$, pooled across all judges). Critically, prediction set width shows consistent cross-judge agreement ($\bar{r} = 0.32$-$0.38$), demonstrating it captures document-level difficulty rather than judge-specific noise. Across four judges and four criteria, both diagnostics converge: criterion matters more than judge, with relevance judged most reliably (avg. set size $\approx 3.0$) and coherence moderately so (avg. set size $\approx 3.9$), while fluency and consistency remain unreliable (avg. set size $\approx 4.9$). We release all code, prompts, and cached results.
    </details>
</div>
