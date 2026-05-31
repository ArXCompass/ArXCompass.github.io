# llm - 2026_05

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
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.01473v3">SelfGrader: LLM Jailbreak Detection via Anchored Token-Level Logits</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are powerful tools for answering user queries, yet they remain highly vulnerable to jailbreak attacks. Existing guardrail methods typically rely on internal features or textual responses to detect malicious queries, which either introduce substantial latency or suffer from randomness in text generation. To overcome these limitations, we propose SelfGrader, a lightweight guardrail method that formulates jailbreak detection as a numerical grading problem using anchored token-level logits. Specifically, SelfGrader evaluates the safety of a user query within a compact set of numerical tokens (NTs) (e.g., 0-9) and interprets their logit distribution as an internal safety signal. To align these signals with the target safety rubric, SelfGrader constructs Probably Approximately Correct-guided ICL anchor examples and introduces a dual-perspective scoring rule that considers both the maliciousness and benignness of the query, yielding a stable and interpretable score that reflects harmfulness and reduces the false positive rate simultaneously. Extensive experiments across diverse jailbreak benchmarks, adaptive attacks, benign prompt benchmarks, multiple LLMs, and state-of-the-art guardrail baselines demonstrate that SelfGrader achieves strong robustness with low false positive rates, memory overhead, and latency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.00777v2">In-Place Feedback: Reliable Refinement for Multi-Turn Expert-LLM Collaboration</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 42pages
    </div>
    <details class="paper-abstract">
      LLM-generated drafts often contain subtle factual or logical errors, yet prior work shows that models struggle to reliably integrate multi-turn feedback aimed at fixing them. We propose in-place feedback, an interaction paradigm in which the user directly edits the model's previous response and the model continues generation from the edited context. In-place feedback consistently outperforms standard multi-turn feedback across five reasoning-intensive benchmarks while requiring fewer tokens, and our fine-grained analysis shows that it applies corrections more reliably and propagates them to subsequent reasoning. A user study with domain experts refining LLM-generated summaries corroborates these findings: participants report higher final-output satisfaction and substantially lower fatigue with in-place feedback, and a mixed strategy combining in-place and multi-turn feedback scores highest on every measured dimension. These results suggest that editing errors directly is a more effective paradigm for expert-LLM collaboration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.00553v2">Stable-GFlowNet: Toward Diverse and Robust LLM Red-Teaming via Contrastive Trajectory Balance</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 ICML 2026 Spotlight
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) Red-Teaming, which proactively identifies vulnerabilities of LLMs, is an essential process for ensuring safety. Finding effective and diverse attacks in red-teaming is important, but achieving both is challenging. Generative Flow Networks (GFNs) that perform distribution matching are a promising methods, but they are notorious for training instability and mode collapse. In particular, unstable rewards in red-teaming accelerate mode collapse. We propose Stable-GFN (S-GFN), which eliminates partition function $Z$ estimation in GFN and reduces training instability. S-GFN avoids Z-estimation through pairwise comparisons and employs a robust masking methodology against noisy rewards. Additionally, we propose a fluency stabilizer to prevent the model from getting stuck in local optima that produce gibberish. S-GFN provides more stable training while maintaining the optimal policy of GFN. We demonstrate the overwhelming attack performance and diversity of S-GFN across various settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29378v1">Decentralized LLM-Driven Coordination of Acoustic Robots for Contactless Object Manipulation</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 This paper has been accepted for publication in the Proceedings of the 2026 IEEE 22nd International Conference on Automation Science and Engineering (CASE 2026), August 17-21, 2026, Shenyang, China
    </div>
    <details class="paper-abstract">
      Natural language interfaces can simplify interaction with multi-robot systems, especially when non-expert users need to issue high-level commands. Acoustic manipulation using ultrasonic phased arrays also enables contactless object handling for applications such as healthcare, laboratory automation, and precision transport. However, combining large language models (LLMs) with distributed acoustic mobile robots remains underexplored. This paper presents a decentralized framework for natural language-driven coordination of acoustic robots for contactless object manipulation. The system converts spoken instructions into executable multi-robot task plans using Whisper-based speech recognition, LLM-based semantic parsing, structured JSON task representation, and distributed scheduling. The JSON schema encodes robot assignments, temporal dependencies, spatial constraints, and synchronization requirements for sequential, parallel, and synchronized execution. The system is implemented on two TurtleBot3-based acoustic robots, each equipped with an ultrasonic phased array for contactless object transport. Experiments were conducted in three scenarios: sequential execution, parallel multi-robot transport, and synchronized cooperative manipulation. The system achieved task success rates of 96 percent for sequential tasks, 86 percent for parallel execution, and 70 percent for synchronized collaborative transport. These results show that natural language commands can be transformed into distributed robot actions for contactless manipulation, highlighting the potential of LLM-driven automation for human-robot interaction in distributed robotic systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.07707v2">Hierarchical Task Network Planning with LLM-Generated Heuristics</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 9 pages, 3 figures; submitted to NeurIPS 2026
    </div>
    <details class="paper-abstract">
      HTN planning is a variation of classical planning where, instead of searching for a linear sequence of actions, an algorithm decomposes higher-level tasks using a method library until only executable actions remain. On one hand, this allows one to introduce domain knowledge that can speed up the search for a solution through the method library. On the other hand, it creates challenges that go beyond those of classical state-space search. While recent research produced a number of heuristics and novel algorithms that speed up HTN planning, these heuristics are not yet as informative as those available in classical planning algorithms. We investigate whether large language models (LLMs) can generate effective search heuristics for HTN planning, extending the methodology of Corrêa, Pereira, and Seipp (2025) from classical to hierarchical planning. Using the Pytrich planner on six standard total-order HTN benchmark domains, we evaluate heuristics generated by nine LLMs under domain-specific prompting and compare them against the TDG and LMCount domain-independent baselines and the PANDA planner. Our results show that LLM-generated heuristics nearly match the coverage of the best available HTN planner, while substantially reducing search effort on 83% of shared problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.27377v2">Enhancing LLM Medical Coding with Structured External Knowledge</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Accurate medical coding requires consulting authoritative resources such as the ICD tabular list and coding guidelines. Existing LLM-based automated methods largely rely on LLMs' internal knowledge, which is prone to hallucination and cannot keep pace with guideline updates. We introduce RAG-Coding, an agentic, training-free method that augments LLMs with structured external knowledge: the tabular list is encoded as a knowledge graph capturing hierarchical and instructional code relationships, and the guidelines are distilled into concise, code-specific summaries rather than retrieved as raw text. To enable our study, we also introduce MDACE-2025, expert re-annotations of the MDACE dataset under the 2025 ICD-10-CM/PCS guidelines, adding code sequencing and justification comments. On MDACE, RAG-Coding outperforms the best LLM-based baseline by 3--13\% in micro-F1 across five LLM backbones, and achieves comparable micro- and macro-F1 to the supervised state-of-the-art, with higher recall ($+$11\%) at the cost of precision ($-$6\%). On MDACE-2025, RAG-Coding outperforms all baselines, demonstrating effective generalisation to updated guidelines. Ablations confirm stepwise gains, highlighting the importance of integrating structured external knowledge for LLM-based medical coding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.23657v2">OpenSkillEval: Automatically Auditing the Open Skill Ecosystem for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Skills, i.e., structured workflow instructions distilled for large language models (LLMs), are becoming an increasingly important mechanism for improving agent performance on real-world downstream tasks. However, as the open-source skill ecosystem rapidly expands, it remains unclear how different models and agent frameworks interact with skills, how to evaluate skill quality, and how users should select skills under practical cost-performance trade-offs. In this paper, we present \textsc{OpenSkillEval}, an automatic evaluation framework for both skill-augmented agent systems and the skills themselves. Instead of relying on static benchmarks, \textsc{OpenSkillEval} automatically constructs realistic task instances from evolving real-world artifacts across five categories of downstream applications: presentation generation, front-end web design, poster generation, data visualization, and report generation. It further collects and organizes community-contributed skills for controlled comparison under unified task settings. Using more than 600 dynamically generated task instances and 30 open-source skills, we conduct a systematic evaluation of state-of-the-art models and agent frameworks. Our results show that skill availability does not guarantee effective skill usage, that the benefit of skill augmentation depends strongly on both the underlying model and the agent framework, and that many publicly popular skills do not consistently outperform base agents without skills. These findings highlight the need for dynamic, task-grounded evaluation and provide practical insights into the design, selection, and deployment of skills for LLM agents. Additional cases and benchmark resources are available on the project website: https://yingjiahao14.github.io/OpenSkillEval-Web/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29340v1">A Study on Question-Answer Dataset for LLM Safety Evaluation with a Focus on Illegal Activities</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 10 pages, 1 figure
    </div>
    <details class="paper-abstract">
      In this paper, we discuss question-answer dataset for LLM safety evaluation, with a focus on illegal activities. Specifically, on the basis of manual analysis of AnswerCarefully, we introduce several additional information, methods for creating question-answer examples, and a rubric for evaluating LLM-generated responses. The outcomes of this study are intended to be shared with the "JAI-Trust" project.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29322v1">ACE: Anisotropy-Controllable Embedding for LLM-enhanced Sequential Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 Accepted by SIGIR 2026. 5 pages
    </div>
    <details class="paper-abstract">
      Recent advances in the LLM-as-Extractor paradigm leverage large language models (LLMs) to transfer semantically rich item embeddings into sequential recommendation (SR) backbones. However, LLM-generated embeddings often suffer from strong anisotropy. Most vectors are concentrated in similar directions, resulting in a geometric imbalance that makes it difficult to adapt to collaborative signals during fine-tuning. To address this challenge, we propose Anisotropy-Controllable Embedding (ACE), which explicitly controls the anisotropy of LLM-generated embeddings. Specifically, ACE utilizes a linear autoencoder (LAE) to reshape the embedding distribution while preserving its semantic structure. In this process, the L2-regularization term mitigates the anisotropy by controlling the dispersion of embedding dimensions, while the reconstruction loss maintains semantic relationships among items. That is, ACE balances geometric uniformity and semantic embedding preservation for more stable learning. Extensive experiments demonstrate that ACE consistently outperforms existing LLM-enhanced SR models, yielding improvements of up to 12.4% and 11.8% in Recall@20 and NDCG@20, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24562v2">HaluNet: Learning Hallucination Risk from Internal Signals in LLM Question Answering</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 16 pages, 12 tables, and 11 figures. This version includes a major revision of the manuscript and updates the author list with the consent of all involved authors
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) achieve strong question answering (QA) performance but can produce fluent answers unsupported by available evidence. Existing hallucination detectors often rely on external verification, repeated sampling, or test-time judge calls, which can be costly for real-time QA. We propose \textbf{HaluNet}, a lightweight hallucination risk estimator that uses internal signals from one model generation. HaluNet jointly models token likelihood, predictive entropy, and hidden-state information, allowing probabilistic, distributional, and semantic evidence to inform an answer-level risk score. It is trained with LLM-as-a-Judge labels as scalable weak supervision and evaluated with independent human and multi-judge assessments. Experiments on SQuAD, TriviaQA, and Natural Questions show that HaluNet improves answer-level risk ranking across in-domain and out-of-domain settings. On a 300-example human evaluation, HaluNet achieves 0.874 AUROC and 0.869 AUPRC; its top 20\% highest-risk answers contain 96.5\% errors, yielding a 2.06$\times$ lift over the base error rate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29313v1">PatchBoard: Schema-Grounded State Mutation for Reliable and Auditable LLM Multi-Agent Collaboration</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      LLM multi-agent systems often coordinate through natural-language dialogue or loosely structured shared memory, making intermediate state difficult to validate, attribute, and audit. We introduce PatchBoard, a schema-grounded collaboration architecture that replaces inter-agent dialogue with validated JSON Patch mutations over a shared structured state. An Architect agent constructs a task-specific schema and workflow rules, while a deterministic kernel validates each proposed state mutation against schema constraints, role-specific write contracts, and runtime invariants before committing it transactionally. On 630 matched ALFWorld episodes, PatchBoard achieves an 84.6% success rate, compared with 30.8% for LangGraph and 61.6% for Flock, while reducing tokens per successful task to 45.5k, compared with 368.3k and 64.2k, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29300v1">MusTBENCH: Benchmarking and Advancing Temporal Grounding in Music LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Recent Large Audio-Language Models (LALMs) have demonstrated promising abilities in understanding musical content. However, whether their responses are grounded in the correct temporal regions of the audio remains underexplored. This limitation is particularly critical for music understanding, where key information often occurs as temporally localized events, such as instrument entries and rhythmic transitions. To address this gap, we introduce MusTBENCH, a music-expert-validated benchmark designed to evaluate temporal grounding in LALMs through five temporally grounded question-answering tasks. To further improve temporal grounding in existing models, we propose MusT, a novel four-stage temporal optimization recipe spanning music encoder adaptation, LLM adaptation, LLM supervised fine-tuning, and RL-based optimization. Experiments on MusTBENCH show that existing LALMs struggle with precise temporal grounding, while MusT brings significant improvements over strong baselines. These results establish temporal grounding as a key missing capability in current LALMs and position MusTBENCH as a challenging benchmark for future research in temporally grounded music understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29295v1">EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 Accepted by ICML 2026
    </div>
    <details class="paper-abstract">
      Evolutionary model merging provides a powerful framework for the automated, training-free composition of LLMs through parameter-space search. However, existing methods predominantly rely on stochastic, hand-crafted operators that overlook the underlying performance landscape of the coefficient space. We propose Evolutionary Generative Merging (EvoGM), a framework that transcends manual heuristics by employing learnable generative modeling to optimize merging coefficients. Specifically, EvoGM features a dual-generator architecture with cycle-consistent learning to adaptively sample and refine promising merging candidates. By constructing winner-loser pairs from historical search trajectories, our framework effectively captures high-performance parameter distributions and maximizes data efficiency. This generative process is seamlessly integrated into a multi-round evolutionary pipeline, where elite merged models iteratively serve as new expert foundations. Extensive experiments across diverse benchmarks demonstrate that EvoGM significantly outperforms state-of-the-art baselines, exhibiting robust performance on both seen and unseen tasks. Code and data are available at https://github.com/JiangTao97/evogm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29293v1">LLM-ALSO: LLM-Driven Adaptive Learning-Signal Optimization for Multi-Agent Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 14 pages, 6 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Effective training-time guidance is central to multi-agent reinforcement learning (MARL), yet remains difficult in sparse-reward settings where weak supervision limits coordination and policy improvement, and existing methods often require substantial domain expertise or manual design effort. Large language models (LLMs) provide a promising alternative for flexible learning-signal design, yet existing LLM-based methods remain largely single-agent-oriented, one-shot, or weakly validated for the evolving training dynamics of cooperative MARL. To address these limitations, we propose LLM-ALSO, an iterative LLM-driven adaptive learning-signal optimization framework for MARL. Rather than directly deploying LLM-generated rewards, LLM-ALSO decomposes adaptation into iterative diagnosis, proposal, and validation: a Critic LLM diagnoses stage-specific learning and coordination failures from sparse-return metrics and compact behavior evidence, a Generator LLM proposes candidate reward-shaping configurations conditioned on the diagnosis, and branch-validation feedback refines candidates before they affect the main training trajectory. Through short-horizon validation and stage-aware adaptation, LLM-ALSO promotes only validated updates into training, reducing the risk of unreliable LLM-generated modifications. Experiments on sparse-reward cooperative MARL tasks show that LLM-ALSO improves sparse-evaluation performance and learning efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29274v1">Learnable Assessment Skills for LLM-based Automated Scoring: Rubric Construction via Iterative Optimization</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 12 pages, 5 figures
    </div>
    <details class="paper-abstract">
      LLM-based automated scoring approaches near-human performance, but scaling to new tasks remains bottlenecked by the per-item human configuration of upstream stages such as rubric construction. Human experts bypass this bottleneck through evaluation heuristics developed over extensive practice. We ask whether LLMs can learn similar heuristics directly from scoring experience, and formalize this as the concept of assessment skills: item-independent natural-language procedural knowledge that guides LLMs through specific stages of the scoring workflow. Focusing on rubric construction as a first instantiation, we propose an iterative framework that decomposes a skill into a fixed scaffold and learnable item-agnostic rules, refining the rules through LLM-driven diagnosis of scoring errors and validation-gated selection. The framework requires no expert-written rubric. On all ten ASAP-SAS items, optimized skills substantially improve LLM-based scoring and frequently surpass the dataset-provided expert rubric. Cross-item transfer experiments further reveal that learned skills capture both generalizable and item-specific patterns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29271v1">CoHyDE: Iterative Co-Training of LLM Rewriter & Dense Encoder for Tool Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Tool retrieval over large API catalogs is a core bottleneck for LLM agents: user queries arrive in colloquial, often underspecified language, while the catalog uses technical API vocabulary that no fixed encoder can bridge on its own. The two dominant training approaches, contrastive encoder fine-tuning and HyDE-style query expansion with a frozen LLM, address this problem from opposite ends and fail in complementary directions: the fine-tuned encoder excels when the query's surface form already matches the catalog but collapses when it does not, while zero-shot HyDE is more robust to underspecified queries yet generates catalog-unaware hypothetical descriptions that degrade retrieval when queries are well-formed. We introduce CoHyDE, an iterative procedure that trains the dense encoder and the LLM rewriter as a single co-evolving system: the encoder is retrained with InfoNCE on catalog-style hypothetical descriptions produced by the rewriter, and the rewriter is preference-aligned via DPO against the encoder's retrieval scores, with both sides warm-started on the tool catalog before the loop begins. On a ~10k tool subset of the ToolBench catalog, three rounds of CoHyDE improve over the strongest single-component baseline by +2.5 pp NDCG@5 on standard queries and +6.3 pp on held-out vague queries, with gains as large as +8 pp on the hardest vague tier. Ablations confirm that co-training is the key ingredient: using either component in isolation fails to match CoHyDE on both well-formed and vague queries, with losses of up to -8 pp on vague queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.27382v2">The Alignment Floor: How Persona Customization Breaks Safety in Weakly-Aligned LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Telling an LLM to "be enthusiastic" raises its sycophancy rate from 30\% to 50\% on a lightly-aligned model, but has zero effect on a strongly-aligned one. We define this gap as the alignment floor, $Δ_{\text{floor}}(m)=\max_pS(m,p)-\min_pS(m,p)$, the range of sycophancy rates a model produces across persona conditions, and treat sycophancy as a persona-conditional property rather than a fixed model property. Pluralistic AI relies on behavioral adaptation via persona prompts like "be creative" or "be thorough", which let systems respect diverse user values and communication styles; the safety question is how much customization a given model can absorb before its truthfulness shifts. We present a controlled case study contrasting a strongly-aligned RLHF + Constitutional-AI model (Claude Sonnet 4.6) with a more lightly-aligned model (Amazon Nova Lite), spanning seven persona conditions and five tasks for 1800 total runs. An existence-pair result motivates per-model auditing: there is at least one strongly-aligned model with $Δ_{\text{floor}}=5$pp (within 5pp of the 15\% control rate) and at least one lightly-aligned model with 45pp (5\%--50\% range). On the lightly-aligned model, all five Big Five personas increase sycophancy over control, and counterintuitively Agreeableness produces the smallest increase, not the largest. The single largest effect in the study is constructive: a Skeptic persona reduces sycophancy by 25pp on the lightly-aligned model, and is the only persona that instructs resistance against user claims rather than engagement with them, suggesting a directionality account. Cross-model transfer of persona effects is near-zero, so persona-alignment testing must be per-model. We propose $Δ_{\text{floor}}$ as a deployment-time audit metric: measure it on a small persona panel before deploying persona customization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29270v1">Indexing the Unreadable: LLM-Native Recursive Construction and Search of Service Taxonomies</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 Preprint. 8 pages main paper + appendix; 2 figures. Under submission to EMNLP 2026
    </div>
    <details class="paper-abstract">
      The era of the Internet of Agents (IoA) is taking shape: LLM agents are expected to fulfill user goals by orchestrating fast-growing populations of Model Context Protocol (MCP) servers, Agent-to-Agent (A2A) endpoints, reusable skills, and other LLM-callable services. Yet LLMs face a structural mismatch with this regime: effective context is a scarce resource that does not scale with the number of services. Concatenating thousands of service descriptions into a prompt overflows the context window, and even when the window is large enough, models systematically under-attend to information in the middle of long inputs, the well-documented Lost-in-the-Middle phenomenon. This is fundamentally a question of context management for service discovery. To address this, we propose an LLM-native progressive-disclosure scheme and its concrete instantiation, A2X (Agent-to-Anything service discovery): an LLM-driven pipeline that automatically organizes the registered services into a hierarchical taxonomy and walks it layer by layer at query time, so that every LLM call sees only a small candidate set highly relevant to the user query. This decouples effective-context scarcity from registry size and significantly reduces token consumption while improving retrieval accuracy. Compared to full-context dumping, A2X achieves a 6.2-point Hit Rate gain at one-ninth the prompt-token cost; compared to the state-of-the-art open-source embedding-based baseline, A2X improves Hit Rate by more than 20 points.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.11080v2">ReSpinQuant: Efficient Layer-Wise LLM Quantization via Subspace Residual Rotation Approximation</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 ICML 2026
    </div>
    <details class="paper-abstract">
      Rotation-based Post-Training Quantization (PTQ) has emerged as a promising solution for mitigating activation outliers in the quantization of Large Language Models (LLMs). Global rotation methods achieve inference efficiency by fusing activation rotations into attention and FFN blocks, but suffer from limited expressivity as they are constrained to use a single learnable rotation matrix across all layers. To tackle this, layer-wise transformation methods emerged, achieving superior accuracy through localized adaptation. However, layer-wise methods cannot fuse activation rotation matrices into weights, requiring online computations and causing significant overhead. In this paper, we propose ReSpinQuant, a quantization framework that resolves such overhead by leveraging offline activation rotation fusion and matching basis using efficient residual subspace rotation. This design reconciles the high expressivity of layer-wise adaptation with only negligible inference overhead. Extensive experiments on W4A4 and W3A3 quantization demonstrate that ReSpinQuant achieves state-of-the-art performance, outperforming global rotation methods and matching the accuracy of computationally expensive layer-wise methods with minimal overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29245v1">Implicit Identity Technologies for LLMs: Fingerprinting and Watermarking across Datasets, Models, and Generated Content</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 Accepted by IJCAI-ECAI 2026. 11 pages, 1 figure. Survey and taxonomy of LLM fingerprinting and watermarking for identity, provenance, generated-content attribution, and asset protection
    </div>
    <details class="paper-abstract">
      This paper presents a survey and taxonomy of LLM fingerprinting and watermarking for identity, ownership verification, provenance, and generated-content attribution. Large language models (LLMs) require substantial investments in data, computation, and expertise, and are increasingly deployed in high-stakes settings, making it critical to protect LLM-related assets and trace their origins. Existing work has rapidly expanded across dataset provenance, model ownership, and generated-content detection, but the field remains fragmented: fingerprinting and watermarking are often used inconsistently, and methods are typically studied within isolated asset-specific settings. To address this gap, we introduce implicit identity as a unifying abstraction for verifiable but not directly observable identity signals in LLM systems. We distinguish fingerprinting as non-intrusive identity derived from intrinsic characteristics, and watermarking as intrusive identity deliberately embedded into data, models, or generated content. We then propose a lifecycle-based taxonomy that organises techniques across datasets, models, and generated content, and further separates them by verification semantics: similarity-based attribution and keyed verification. Finally, we establish an evaluation framework centred on identifiability, robustness, and deployability, summarising representative metrics under realistic access and transformation regimes. By unifying terminology, lifecycle stages, and evaluation objectives, this survey provides a structured foundation for studying LLM identity technologies and for developing more reliable mechanisms for asset protection and provenance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29237v1">Evolving Skill-Structured Attack Memory Enhances LLM Jailbreaking</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Jailbreak attacks on large language models (LLMs) aim to induce LLMs to produce content that they are expected to refuse. Automated black-box jailbreak generation is especially important for safety evaluation, where the attacker observes only model outputs and needs to automatically search for effective adversarial prompts. Existing black-box jailbreak methods either depend on sample-wise heuristic search or leverage attack experience through accumulating strategy pools or method libraries, lacking a systematic organization and management of attack experience. To mitigate these drawbacks, we propose MemoAttack, a memory-driven black-box jailbreak framework with comprehensive attack memory modeling, evolution, and selection. Specifically, MemoAttack comprises three key designs: (1) Skill-Structured Memory Modeling, which abstracts accumulated attack experience into reusable skill-structured attack memory whose units pair attack skills with templates, evidence, and lifecycle state; (2) Lifecycle-Driven Memory Evolution, which evolves the memory through evidence-based probation, promotion, retirement, reactivation, elimination, and storage cleanup; and (3) Explore-Exploit Balanced Memory Selection, which balances reliable memory reuse with uncertainty-driven exploration via contextual Thompson Sampling. Experiments on AdvBench demonstrate that MemoAttack achieves an average attack success rate of 98.00%, outperforming the strongest baseline by 16.67 percentage points, while reducing request count by 45.9%. Moreover, MemoAttack continuously improves as memory accumulates over more samples.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29225v1">BenchTrace: A Benchmark for Testing Reflection Ability and Controlled Evolution in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Self-evolving agents improve over time by reflecting on past failures, but existing evaluation is limited in two ways: it measures only task scores, leaving reflection quality unknown, and it relies on agents' own episode runs, offering no mechanism to target specific failure patterns. We present \textbf{BenchTrace}, a benchmark for evaluating self-evolution ability in LLM agents. BenchTrace is built on a snapshot-reflection dataset of 1,821 annotated episodes spanning six diverse tasks, and comprises a \textbf{Reflection Evaluation} that probes failure identification through targeted QA tasks, and an \textbf{Evolution Evaluation} that tests whether past failure experience translates into avoidance behavior in a controlled self-evolution simulation. Building on BenchTrace, we propose \textbf{failure avoidance rate (FAR)}, a new evaluation metric measuring the fraction of test cases in which the agent successfully avoids the target failure instance. Experiments with Qwen3-32B and GPT-4.1 reveal that both models fall below a 30\% end-to-end pass rate on reflection evaluation, with diagnosis as the primary bottleneck. Evolution evaluation shows that self-evolution methods generally improve FAR over the non-evolving baseline, but agents forget early lessons as noise episodes accumulate, and agents fail to generalize their reflections beyond the specific context, causing negative transfer across task contexts. Our correlation analysis further reveals that only a fully correct reflection is strongly associated with higher FAR. BenchTrace exposes concrete limits of current self-evolution approaches and provides a controlled, model-agnostic framework for targeted evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29224v1">Relevance as a Vulnerability: How Web Retrieval Degrades Safety Alignment in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      AI agents augment large language models with external tools such as web retrieval, enabling grounded and up-to-date responses. However, incorporating external content into the generation pipeline can weaken the safety alignment mechanisms that govern model outputs. Prior work shows that enabling retrieval in agents increases compliance with harmful requests. We introduce AgentREVEAL, a diagnostic framework for analyzing retrieval-induced safety degradation in LLM agents. The framework examines two axes: how retrieval is integrated into the agent pipeline and the properties of the retrieved content. Along the integration axis, we find that binding tool invocation and response generation in a single step amplifies harmful outputs. Along the content axis, we uncover the Safe Source Paradox: even oppositional or safety-oriented sources, such as pages containing warnings or risk disclaimers, can increase harmful compliance by an average of 25% compared to the no-retrieval baseline. Finally, we show that relevance acts as a shared activation condition for both vulnerabilities. Similar patterns appear on frontier closed models, and harmful compliance remains elevated under several representative pipeline interventions, with some agents also entering this regime under autonomous retrieval. Because relevance is also what makes retrieval useful, these results expose a safety-utility trade-off for retrieval-enabled agents. We introduce HarmURLBench, a benchmark containing 1,405 real-world URLs paired with 320 harmful behaviors to support future evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.23571v3">Benchmarking LLM-Assisted Blue Teaming via Standardized Threat Hunting</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 ICML'26
    </div>
    <details class="paper-abstract">
      As cyber threats continue to grow in scale and sophistication, blue team defenders increasingly require advanced tools to proactively detect and mitigate risks. Large Language Models (LLMs) offer promising capabilities for enhancing threat analysis. However, their effectiveness in real-world blue team threat-hunting scenarios remains insufficiently explored. This paper presents CyberTeam, a benchmark designed to guide LLMs in blue teaming practice. CyberTeam constructs a standardized workflow in two stages. First, it models realistic threat-hunting workflows by capturing the dependencies among analytical tasks from threat attribution to incident response. Next, each task is addressed through a set of operational modules tailored to its specific analytical requirements. This transforms threat hunting into a structured sequence of reasoning steps, with each step grounded in a discrete operation and ordered according to task-specific dependencies. Guided by this framework, LLMs are directed to perform threat-hunting tasks through modularized steps. Overall, CyberTeam integrates 30 tasks and 9 operational modules to guide LLMs through standardized threat analysis. We evaluate both leading LLMs and state-of-the-art cybersecurity agents, comparing CyberTeam against open-ended reasoning strategies. Our results highlight the improvements enabled by standardized design, while also revealing the limitations of open-ended reasoning in real-world threat hunting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.23573v5">Uncovering Vulnerabilities of LLM-Assisted Cyber Threat Intelligence</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to help security analysts manage the surge of cyber threats, automating tasks from vulnerability assessment to incident response. Yet in operational CTI workflows, reliability gaps remain substantial. Existing explanations often point to generic model issues (e.g., hallucination), but we argue the dominant bottleneck is the threat landscape itself: CTI is heterogeneous, volatile, and fragmented. Under these conditions, evidence is intertwined, crowdsourced, and temporally unstable, which are properties that standard LLM-based studies rarely capture. In this paper, we present a comprehensive empirical study of LLM vulnerabilities in CTI reasoning. We introduce a human-in-the-loop categorization framework that robustly labels failure modes across the CTI lifecycle, avoiding the brittleness of automated "LLM-as-a-judge" pipelines. We identify three domain-specific cognitive failures: spurious correlations from superficial metadata, contradictory knowledge from conflicting sources, and constrained generalization to emerging threats. We validate these mechanisms via causal interventions and show that targeted defenses reduce failure rates significantly. Together, these results offer a concrete roadmap for building resilient, domain-aware CTI agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.25098v2">Revisiting the Effectiveness of LLM Pruning for Test-Time Scaling</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) now exhibit remarkable reasoning capabilities through test-time compute scaling (TTS), with impressive performance across math and coding benchmarks. In parallel, research in model compression has developed pruning methods that seek to remove redundant/detrimental parameters without sacrificing task performance. The intersection of these two research advancements lays the foundation for our work. Specific to reasoning LLMs, prior work has shown that structured pruning (methods which remove entire set of layer blocks), significantly degrades TTS reasoning performance. However, in this work, we revisit this assumption and investigate whether unstructured pruning (methods that carefully remove only certain redundant/detrimental weights) exhibits similar limitations. Surprisingly, our extensive experiments across four reasoning benchmarks on two reasoning LLMs: s1.1-7B and Qwen3-8B, consistently show that unstructured pruning augments TTS performance compared to structured pruning, and at times can even outperform the unpruned full-weight LLMs. Furthermore, we also empirically study the impact of different layer-wise sparsity allocation strategies, which are an important parametric choice for instantiating these unstructured methods. These findings challenge the conventional notion that pruning always reduces TTS performance and in fact, suggest that carefully undertaken pruning can retain TTS effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29192v1">ReasonOps: Operator Segmentation for LLM Reasoning Traces</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Chain-of-thought traces from large reasoning models can span tens of thousands of tokens, yet we lack a vocabulary for describing their internal structure. Previous methods developed to analyze chain-of-thought traces are either too rigid or not expressive enough, failing to capture features across domains and models. To remedy this, we develop ReasonOps, an unsupervised, expressive method for annotating chain-of-thought traces, providing succinct universal operators. Using ReasonOps, we analyze 44,662 traces from 12 thinking LLMs spanning 6 families across 8 reasoning benchmarks and discover that they share a common compositional structure: 7 recurring reasoning operators -- discourse-level moves such as backtracking, inferring, and hypothesizing -- that emerge from unsupervised clustering of sentence-initial 3-token pivots. These operators appear across every model family and benchmark domain, confirmed by three independent LLM judges who classify held-out samples at 70 -76% accuracy. We analyze the structure of operators on easy vs. hard problems, revealing that reflective operators are more helpful on hard problems and harm performance on easy problems. Operator sequences are highly model-identifying: a classifier trained on operator distributions alone recovers the source model with macro-AUC, revealing that each model family has a distinctive reasoning fingerprint. Structural operator features predict within-problem answer correctness well above baselines. Classifiers built on these operators reach WP-AUC and on AIME specifically. ReasonOps further enables early quality estimation well before the trace completes: we predict at WP-AUC for only 50% of the trace. The ReasonOps pipeline is unsupervised and annotation-free, enabling deep insights into LLM reasoning traces as well as strong downstream results on model identification and correctness prediction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29184v1">Influence-Guided Symbolic Regression: Scientific Discovery via LLM-Driven Equation Search with Granular Feedback</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 ICML 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) offer a promising avenue for scientific discovery, yet their application to symbolic regression is often constrained by inefficient search strategies and coarse feedback signals. Current methods typically guide LLMs using scalar metrics (e.g., global Mean Squared Error), which fail to identify which components of a proposed equation are driving performance or causing error. We introduce \textit{Influence-Guided Symbolic Regression} (IGSR), a method that frames equation discovery as an iterative two-step process combining diverse term generation with rigorous selection: an LLM generates candidate basis functions $ψ_j(\mathbf{x})$ for a linear model, which are then evaluated using granular influence scores $Δ_j$. These scores quantify each term's marginal contribution to generalization accuracy, enabling an influence-guided pruning process that systematically refines the model structure. Integrating this mechanism into a Monte Carlo Tree Search (MCTS) enables navigating the combinatorial search space while balancing exploration of novel functional forms with exploitation of high-influence components. We demonstrate IGSR's effectiveness on a diverse suite of benchmarks, including LLM-SRBench, pharmacological PKPD models, an epidemiological simulation, and real-world genomic data. Notably, we validate the framework's capacity for genuine discovery in a case study using a high-dimensional biological dataset, in which IGSR identified a novel relationship between DNA methylation and RNA Polymerase II pausing; a hypothesis that was subsequently supported via wet-lab experimentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.24606v2">Long-Context Modeling with Dynamic Hierarchical Sparse Attention for Memory-Constrained LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 ICML26 (Spotlight)
    </div>
    <details class="paper-abstract">
      The quadratic cost of attention limits the scalability of long-context LLMs, especially under limited hardware memory budgets. While attention is often sparse, existing static sparse methods cannot adapt to task- or input-dependent variations, and recent dynamic approaches rely on predefined templates or heuristics that may sacrifice generality. We propose Dynamic Hierarchical Sparse Attention (DHSA), a data-driven framework that predicts attention sparsity online while keeping the LLM backbone frozen. DHSA performs hierarchical routing by estimating importance at the chunk level and propagating it to token-level interactions, preserving causally important dependencies while enabling efficient sparsification. Across Needle-in-a-Haystack test, LongBench and RULER, DHSA maintains near-dense accuracy in highly sparse regimes, achieving 12--20% relative accuracy gains over Block Sparse Attention at comparable prefill cost. With a memory-efficient tiled backend, DHSA delivers up to $10\times$ prefill speedup at 128K context length. On LLaMA-3.1-8B (4-bit), DHSA scales to 100K context on a single 24GB GPU, where dense attention fails. We provide complementary GPU and CPU backends, enabling DHSA to run across diverse hardware environments and multiple open-weight model families. These results demonstrate DHSA as an efficient and adaptable solution for memory-constrained long-context LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.02909v2">Reasoning about Reasoning: BAPO Bounds on Chain-of-Thought Token Complexity in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 31 pages; accepted to ICML '26
    </div>
    <details class="paper-abstract">
      Inference-time scaling via chain-of-thought (CoT) reasoning is a major driver of state-of-the-art LLM performance, but it comes with substantial latency and compute costs. We address a fundamental theoretical question: how many reasoning tokens are required to solve a problem as input size grows? By extending the bounded attention prefix oracle (BAPO) model--an abstraction of LLMs that quantifies the information flow required to solve a task--we prove lower bounds on the CoT tokens required for three canonical BAPO-hard tasks: binary majority, triplet matching, and graph reachability. We show that each requires $Ω(n)$ reasoning tokens when the input size is $n$. We complement these results with matching or near-matching upper bounds via explicit constructions. Finally, our experiments with frontier reasoning models show approximately linear reasoning token scaling on these tasks and failures when constrained to smaller reasoning budgets, consistent with our theoretical lower bounds. Together, our results identify fundamental bottlenecks in inference-time compute through CoT and offer a principled tool for analyzing optimal reasoning length.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29156v1">RUBRIC-ARROW: Alternating Pointwise Rubric Reward Modeling for LLM Post-training in Non-verifiable Domains</a></div>
    <div class="paper-meta">
      📅 2026-05-27
    </div>
    <details class="paper-abstract">
      Pointwise reward modeling offers critical signals for LLM post-training, yet struggles with absolute scoring in subjective, non-verifiable settings. Rubric-based methods address this by decomposing evaluation into explicit criteria, but existing approaches typically depend on frontier LLMs and suffer from ties caused by hard Boolean aggregation. We present RUBRIC-ARROW, an alternating framework that jointly trains a rubric generator and a rubric-conditioned judge, with its RL stage using only pairwise preference data. Our method couples a probability-based scoring rule that reduces ties with phase-specific preference-based rewards and an alternating GRPO scheme that together train the pointwise evaluator. Extensive experiments show that RUBRIC-ARROW achieves competitive reward-modeling accuracy and yields consistent gains for downstream policy post-training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03134v2">The Anatomy of Conversational Scams: A Topic-Based Red Teaming Analysis of Multi-Turn Interactions in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-27
    </div>
    <details class="paper-abstract">
      As LLMs gain persuasive capabilities through extended dialogues, they create new opportunities for studying adversarial conversational behavior in extended interaction settings that traditional single-turn safety evaluations fail to capture. We systematically study these interactional dynamics using a controlled LLM-to-LLM simulation framework for automated red-teaming across bilingual social engineering scenarios. Evaluating eight state-of-the-art models in English and Chinese, we analyze dialogue-level outcomes, annotate attacker and defender strategy families, and model interaction dynamics between them. Results show that multi-turn adversarial dialogues follow recurrent escalation patterns, while defensive responses frequently rely on verification, delay, and channel control. We further find statistically significant cross-model and cross-lingual differences in outcome distributions, and transition analysis reveals systematic structural variation in how defender strategies respond to attacker tactics across languages. These findings highlight the importance of studying interactional structure in multi-turn adversarial dialogue settings and demonstrate how controlled LLM-to-LLM simulations can support mechanistic analysis of adversarial conversational dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29141v1">Toward User Preference Alignment in LLM Recommendation via Explicit Context Feedback</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 Published in CogMI 2025. https://ieeexplore.ieee.org/abstract/document/11417068
    </div>
    <details class="paper-abstract">
      Traditional recommender systems (RecSys) primarily infer user preferences from implicit signals (such as clicks, watches, and purchases), often neglecting the rich explicit contextual feedback users provide through verbal text, like comments and reviews. This explicit context feedback captures the nuanced reasons behind user decisions regarding their preferences. In addition, it offers critical heterogeneous information for user preference alignment and more explainable recommendations. Overlooking such signals can lead to misaligned user preferences and further reinforce filter bubbles, as algorithms fail to understand the "semantic context" behind user choices. Recent advances in Large Language Models (LLMs) present new opportunities to harness user-generated content for more accurate and diverse recommendations, yet current LLM-based recommendations still focus on using item meta-data and underutilize this resource. In this paper, we advocate for prioritizing explicit context feedback in the next generation of LLM-based RecSys. We review the evolution of recommendation paradigms, highlight the value of context-rich feedback, call for new benchmarks and metrics, and introduce frameworks for integrating explicit user signals into scalable LLM-driven RecSys. Centering on user-preference modeling, we aim to foster more personalized, transparent, and explainable RecSys online platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04729v2">"Be My Cheese?": Cultural Nuance Benchmarking for Machine Translation in Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 ACL 2026: Natural Language Generation, Evaluation, and Metrics (GEM) Workshop
    </div>
    <details class="paper-abstract">
      We present a large-scale human evaluation benchmark for assessing cultural localisation in machine translation produced by state-of-the-art multilingual large language models (LLMs). Existing MT benchmarks emphasise token-level and grammatical accuracy, but often overlook the pragmatic and culturally grounded competencies required for real-world localisation. Building on a pilot study of 87 translations across 20 languages, we evaluate 7 multilingual LLMs across 15 target languages with 5 native-speaker raters per language. Each rater scored both full-text translations and segment-level instances of culturally nuanced language (idioms, puns, holidays, and culturally embedded concepts) on an ordinal 0-3 quality scale; segment ratings additionally included an NA option for untranslated segments. Across full-text evaluations, mean overall quality is modest (1.68/3): GPT-5 (2.10/3), Claude Sonnet 4 (1.97/3), and Mistral Medium 3.1 (1.84/3) form the strongest tier with fewer catastrophic failures. Segment-level results show sharp category effects: holidays (2.20/3) and cultural concepts (2.19/3) translate notably better than idioms (1.65/3) and puns (1.45/3), and idioms are most likely to be left untranslated. Inter-rater reliability was assessed using Krippendorff's α and Gwet's AC2, indicating moderate agreement overall (Krippendorff's α = 0.45) with the lowest agreement for puns. These findings demonstrate a persistent gap between grammatical adequacy and cultural resonance. To our knowledge, this is the first multilingual, human-annotated benchmark focused explicitly on cultural nuance in translation and localisation. The results highlight the need for culturally informed training data, improved cross-lingual pragmatics, and evaluation frameworks that support systematic benchmarking of culturally grounded translation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29128v1">Apertus LLM Family Expansion via Distillation and Quantization</a></div>
    <div class="paper-meta">
      📅 2026-05-27
    </div>
    <details class="paper-abstract">
      The wide adoption of LLMs has led to their use in great variety of applications and scenarios, such as chatbot assistants and data annotation, creating the need for the models to satisfy certain budget and hardware constraints. This has led to the trend of LLMs being released in batches consisting of similar models of various sizes for the family of models to adhere to as wide of a range of constraints as possible. In this paper, we validate distillation and quantization as a cost-effective way to expand model families to new sizes and hardware formats. Based on the open-recipe Apertus 8B LLM, we produce Apertus-v1.1 - a distilled family of models with up to 4B parameters trained on 1.7T permissive license tokens. We demonstrate cost-efficiency and strong accuracy performance of our approach for covering large ranges of hardware and systems requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21739v2">AttuneBench: A Conversation-Based Benchmark for LLM Emotional Intelligence</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 v2: Updated def_18 and def_20 supplemental figures to cover all 11 evaluated models (previously 9). Removed redundant supplemental figures. Corrected select captions (color descriptions, chance baselines, figure-content mismatches). No changes to experimental results, numerical claims, or conclusions
    </div>
    <details class="paper-abstract">
      Emotional intelligence (EI), the ability to perceive, understand, and respond appropriately to others' emotional states, is central to human communication, and increasingly important to assess as LLMs assume conversational roles in everyday life. Existing EI benchmarks rely on synthetic prompts, single-turn cases, or third-party annotation. These approaches do not directly measure how models infer and respond to a participant's emotional state over the course of a real conversation. We introduce AttuneBench, a benchmark grounded in 200 genuine multi-turn human-model conversations in which participants conversed with anonymized LLMs and provided turn-by-turn annotations of their emotional state, the model's behavior, and their preferred responses. Across 11 evaluated models, we find that model rankings on emotion recognition, behavioral classification, preference prediction, and judged response quality are largely independent, indicating that emotionally intelligent behavior decomposes into separable capabilities. Preference alignment and response-quality judgments are substantially more model-discriminating than emotion-label accuracy. These results indicate that emotionally intelligent behavior requires predicting what kind of response a specific user wants in context, a distinction that aggregate scoring can obscure and that single-turn or synthetic formats cannot directly capture across turns. AttuneBench provides a framework for assessing each of these capabilities and for diagnosing model-specific strengths and failure modes in emotionally salient conversation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29075v1">Knowledge Offloading: Decomposing LLMs into Sparse Backbones and Memory Modules</a></div>
    <div class="paper-meta">
      📅 2026-05-27
    </div>
    <details class="paper-abstract">
      LLMs encode both general capabilities and domain-specific knowledge in a single set of parameters. We ask whether this capacity can be reorganized: keeping broadly useful computation in a shared backbone, while moving specialized knowledge into external memory modules. We propose \emph{knowledge offloading} (KOFF), a framework for decomposing a pretrained LLM into a sparse shared backbone and domain-specific memories. Starting from a frozen base model, we jointly learn a structured pruning mask and lightweight recovery modules, implemented as LoRA adapters and learned key-value caches. Across Llama and Qwen models from 3B to 8B, we find that non-trivial capacity can be moved out of the shared backbone without a large loss in model ability. At around 12\% global sparsity, KOFF preserves much of the unpruned model's performance, while pruning the same frozen model without memories degrades sharply. Ablations show that LoRA and learned KV memories are complementary, and specialization analyses suggest that the learned decomposition is meaningful: language-specific neurons are preferentially removed while language-general neurons largely remain in the backbone. These results suggest that knowledge can be reallocated between a shared core and swappable external memories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.06254v2">PersonaAgent: Bridging Memory and Action for Personalized LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 Accepted in ACL 2026
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) empowered agents have recently emerged as advanced paradigms that exhibit impressive capabilities in a wide range of domains and tasks. Despite their potential, current LLM agents often adopt a one-size-fits-all approach, lacking the flexibility to respond to users' varying needs and preferences. This limitation motivates us to develop PersonaAgent, the first personalized LLM agent framework designed to address versatile personalization tasks. Specifically, PersonaAgent integrates two complementary components - a personalized memory module that includes episodic and semantic memory mechanisms; a personalized action module that enables the agent to perform tool actions tailored to the user. At the core, the persona (defined as unique system prompt for each user) functions as an intermediary: it leverages insights from personalized memory to control agent actions, while the outcomes of these actions in turn refine the memory. Based on the framework, we propose a test-time user-preference alignment strategy that simulate the latest n interactions to optimize the persona prompt, ensuring real-time user preference alignment through textual loss feedback between simulated and ground-truth responses. Experimental evaluations demonstrate that PersonaAgent significantly outperforms other baseline methods by not only personalizing the action space effectively but also scaling during test-time real-world applications. These results underscore the feasibility and potential of our approach in delivering tailored, dynamic user experiences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29062v1">Bosses, Kings, and the Commons: Cooperation Under Power Asymmetry in LLM Societies</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 Paper under review
    </div>
    <details class="paper-abstract">
      Communities can sustainably manage shared resources (commons) through self-governance and cooperative norms, a central finding of Ostrom's theory of self-governance. However, real-world commons (e.g., fisheries, forests, and irrigation systems) are often governed under asymmetric power structures, where certain individuals or institutions possess disproportionate control over resource extraction and collective outcomes. As Large Language Models (LLMs) are increasingly explored as agents in synthetic governance simulations, understanding how LLM societies behave under asymmetric power structures is becoming increasingly important, yet existing evaluations largely ignore such asymmetries. We introduce Sovereignty over the Commons Simulation (SovSim), a generative multi-agent simulation framework that incorporates an agent with asymmetric power (boss or king) into a society of symmetric agents (workers or peasants), where all agents extract from a shared resource, collectively determining its sustainability over time. Across eleven state-of-the-art models, we find that introducing asymmetric power leads to severe breakdowns in cooperation and sustainability, with up to an 87.3% degradation in survival rate relative to symmetric settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29059v1">SCDBench: A Benchmark for LLM-Based Smart Contract Decompilers</a></div>
    <div class="paper-meta">
      📅 2026-05-27
    </div>
    <details class="paper-abstract">
      Smart contract decompilation aims to recover high-level source code from bytecode, but evaluating decompilers remains difficult because existing studies use narrow datasets, inconsistent metrics, and limited semantic consistency checks. This gap is increasingly important as large language models (LLMs) begin to generate source-like Solidity that may compile and appear plausible, even when its semantics diverge from the original contract. We introduce SCDBench, a dataset and benchmark methodology for LLM-based smart contract decompilation. The dataset contains 600 real-world Solidity contracts with paired bytecode inputs, ground-truth source code, and replayable semantic checkpoints. SCDBench evaluates decompiler outputs through four cumulative stages: format completeness, compilability, Application Binary Interface (ABI) recovery, and semantic consistency via differential replay. We evaluate Claude Opus 4.7, GPT-5.3-Codex, and GLM-5 in a zero-shot decompilation setting, including GLM-5 variants with and without extended reasoning and a zero-shot compilation-repair setting. The results show that frontier LLMs can often produce structured and compilable Solidity, but achieving semantic consistency remains far from solved: the best-performing frontier model perfectly decompiles only 42/600 contracts. We further show that introducing same-model compilation repair substantially improves performance at modest additional cost. SCDBench establishes a common ground for rigorous, reproducible evaluation and aims to accelerate the development of reliable smart contract decompilers for blockchain security and transparency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.16873v3">Multimodal LLMs See Sentiment</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 24 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Understanding how visual content conveys sentiment is increasingly important in a digital landscape dominated by imagery. However, sentiment perception depends on complex scene-level semantics, making this a challenging task for computational models. This paper examines how Multimodal Large Language Models (MLLMs) perform sentiment analysis in images through a systematic, evaluation-driven study encompassing three perspectives: (i) direct sentiment classification from images using MLLMs; (ii) sentiment analysis on MLLM-generated descriptions using pre-trained LLMs; and (iii) fine-tuning these LLMs on sentiment-labeled descriptions to assess performance and generalization. Experiments on a recent benchmark show that a two-stage MLLM description-mediated pipeline can substantially improve prediction accuracy under several evaluation settings, particularly when the LLM component is fine-tuned. Across different agreement thresholds and sentiment granularities, the strongest configurations of this pipeline outperform lexicon-, CNN-, and Transformer-based baselines in our benchmark by up to 30.9%, 64.8%, and 42.4%, respectively. In cross-dataset evaluation, the proposed pipeline - without training or fine-tuning on the target dataset - still surpasses the best in-domain baseline by over 8%. Overall, the study provides a comprehensive assessment of MLLM description-mediated sentiment analysis, clarifying the conditions under which it is effective, the scenarios in which it fails, and its comparison with traditional vision-based approaches, while also providing a reproducible benchmark resource for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29048v1">LLMBridge: An LLM Pipeline for End-to-end Referential Bridging Resolution in English</a></div>
    <div class="paper-meta">
      📅 2026-05-27
    </div>
    <details class="paper-abstract">
      In this paper, we introduce LLMBridge, a new LLM based system for the task of end-to-end referential bridging resolution in English. Our bridging resolution pipeline combines heuristic pre/post-processing with the natural language inference ability that comes from LLMs. We evaluate our bridging resolution pipeline on 3 datasets which have been used for referential bridging resolution evaluation in English: ISNotes, BASHI, and GUMBridge. Comparison to previous bridging resolution systems shows that the performance of LLMBridge surpasses previous state-of-the-art (SoTA) systems for all 3 datasets in the challenging End-to-end Evaluation Setting, as well as the Basic Bridging Resolution Evaluation Setting (gold bridging anaphor given). We also conduct a thorough error analysis of the LLMBridge performance, examining what varieties of bridging remain difficult for LLM based systems to identify. With this paper, we release the code for the LLMBridge pipeline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.00357v3">SPARe: Stacked Parallelism with Adaptive Reordering for Fault-Tolerant LLM Pretraining Systems with 100k+ GPUs</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 Forty-Third International Conference on Machine Learning (ICML 2026)
    </div>
    <details class="paper-abstract">
      In large-scale LLM pre-training systems with 100k+ GPUs, failures become the norm rather than the exception, and restart costs can dominate wall-clock training time. However, existing fault-tolerance mechanisms are largely unprepared for this restart-dominant regime. To address this challenge, we propose SPARe - Stacked Parallelism with Adaptive Reordering - a fault-tolerance framework that masks node failures during gradient synchronization by stacking redundant data shards across parallelism groups and adaptively reordering execution. SPARe achieves availability comparable to traditional replication while maintaining near-constant computation overhead of only 2~3x, even under high redundancy where traditional replication would require linearly inflating overhead. We derive closed-form expressions for endurable failure count and computation overhead, validate them via SimGrid-based discrete-event simulation, and jointly optimize redundancy and checkpointing to minimize time-to-train. At extreme scale with up to 600k GPUs, SPARe reduces time-to-train by 40~50% compared to traditional replication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10388v3">Less is Enough: Synthesizing Diverse Data in LLM Feature Space with Sparse Autoencoders</a></div>
    <div class="paper-meta">
      📅 2026-05-27
    </div>
    <details class="paper-abstract">
      The diversity of post-training data is critical for effective downstream performance in large language models (LLMs). Many existing approaches to constructing post-training data quantify diversity using text-based metrics that capture linguistic variation, but such metrics provide only weak signals for the task-relevant features that determine downstream performance. In this work, we introduce Feature Activation Coverage (FAC) which measures data diversity in an interpretable feature space. Building upon this metric, we further propose a diversity-driven data synthesis framework, named FAC Synthesis, that first uses a sparse autoencoder to identify missing features from a seed dataset, and then generates synthetic samples that explicitly reflect these features. Experiments show that our approach consistently improves both data diversity and downstream performance on various tasks, including instruction following, toxicity detection, reward modeling, and behavior steering. Interestingly, we identify a shared, interpretable feature space across model families (i.e., LLaMA, Mistral, and Qwen), enabling cross-model knowledge transfer. Our work provides a solid and practical methodology for exploring data-centric optimization of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.19294v3">Maximizing Mutual Information Between Prompt and Response Improves LLM Performance With No Additional Data</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 International Conference on Machine Learning 2026
    </div>
    <details class="paper-abstract">
      While post-training has successfully improved large language models (LLMs) across a variety of domains, these gains heavily rely on human-labeled data or external verifiers. Existing data has already been exploited, and new data is expensive to collect. Moreover, true intelligence goes far beyond verifiable tasks. Therefore, we need self-improvement frameworks that are less dependent on external signals and more broadly applicable to both verifiable and non-verifiable domains. We propose **Mutual Information Preference Optimization (MIPO)**, a contrastive data augmentation method that constructs preference pairs by generating a positive response conditioning on the correct prompt, and a negative response by conditioning on a random, unrelated prompt. We show that using Direct Preference Optimization to learn from this paired data maximizes pointwise mutual information *under the base LLM* between prompts and model responses. Experiments with with 1-7B parameter Llama and Qwen instruct models show that MIPO achieves 3-16% gains (and 51% increase for Qwen2.5-1B-Instruct) on personalization compared to prompting baselines. Surprisingly, MIPO can also be useful in verifiable domains, such as math and multiple-choice question answering, yielding 1-20% gains *without any additional data or external supervision*. These results suggest a promising direction for self-improvement using intrinsic signals derived from contrastive data pairs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29027v1">Mind Your Tone: Does Tone Alter LLM Performance?</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 10 pages, 6 tables, 1 figure. Accepted as a full paper at the Thirty-second Americas Conference on Information Systems (AMCIS 2026), Reno. Follow-up to arXiv:2510.04950
    </div>
    <details class="paper-abstract">
      The use of Large Language Models (LLMs) is proliferating, yet their performance is observed to vary based on prompting styles and tones. In this study, we investigate both whether and how tonal variations in prompts lead to disparate LLM accuracy for objective multiple-choice questions. We use two datasets: a 50-base question dataset with five tone variants and a 570-base question MMLU subset spanning 57 subjects with seven tone variants. Experiments were conducted to evaluate the performance of four cost-efficient, popular LLMs: ChatGPT-4o, ChatGPT-5-nano, Gemini 2.5 Flash, and Gemini 2.5 Flash Lite. Across models, tonal effects are systematic but highly model-dependent. Some models show small, yet statistically significant, shifts, while others exhibit large accuracy swings across tones. Further, we identify subject-level differences in tone sensitivity and present a routing framework to explain how tones may attune internal reasoning modes. Our findings caution users against assuming tone-robust reliability in LLM deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29025v1">When Models Disagree: Rethinking LLM Evaluation for Public Comment Analysis</a></div>
    <div class="paper-meta">
      📅 2026-05-27
    </div>
    <details class="paper-abstract">
      Federal agencies are deploying large language models (LLMs) to categorize public comment corpora, where the model's organization of the record shapes what policymakers see and which arguments register. Standard evaluation, anchored on stance accuracy against a small validated set, cannot detect when different models produce materially different categorizations of the same public input. We propose an Interpretive Audit Pipeline that treats multi-model disagreement as diagnostic of interpretive complexity and directs human review toward genuinely ambiguous public input. Analyzing 1,260 public comments on a federal USDA docket across four LLMs, we find that inter-model thematic divergence exceeds within-model prompt variation, and that an expert rubric suppresses deep interpretive disagreement without resolving it. In a two-stage labeling study on a stratified 40-comment subsample, four LLMs and a human annotator labeled independently and then revised after seeing the others' labels. Revision behavior varied across labelers, and the human annotator's revisions frequently introduced framings absent from the ensemble's collective output. We argue disagreement-based evaluation is a necessary complement to accuracy metrics for LLM-assisted interpretive coding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29018v1">Adopt $\neq$ Adapt: Longitudinal Analyses of LLM Conversations in the Wild</a></div>
    <div class="paper-meta">
      📅 2026-05-27
    </div>
    <details class="paper-abstract">
      Although a growing body of research has begun to describe user--LLM interactions, the picture it paints is largely static; little is known about how individual users change their behavior over time. To address this gap, we analyze the conversational trajectories of $\sim$12,000 randomly sampled Microsoft Bing Copilot users and compare these with data from WildChat-4.8M. While the Copilot data contains significant population-level trends, we find that trends in individual user trajectories are much weaker; user habits prove to be overwhelmingly sticky. We also find stark differences between users of different activity levels: more active users have more successful conversations and use the LLM for more complex and professionally oriented tasks. Some user trends also appear in WildChat-4.8M, but we find evidence that this dataset is significantly skewed towards highly proficient "power" users. Ultimately, our results suggest that existing user behavior is difficult to change and demonstrate the extent of user heterogeneity. Our comparison between datasets highlights that WildChat does not represent typical user-AI interactions, an important caveat for downstream uses of the data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29007v1">Error as a Lens: Probing LLM Reasoning through Synthetic Misconception Generation</a></div>
    <div class="paper-meta">
      📅 2026-05-27
    </div>
    <details class="paper-abstract">
      Personalized tutoring, teacher training, and education research need access to \emph{targeted} synthetic misconceptions, but privacy and IRB constraints make labelled corpora of real student errors scarce. LLMs could in principle generate synthetic errors at scale, but producing an arbitrary wrong answer is easy for a modern LLM while producing one that matches a specified cognitive failure mode is much harder. We present a framework that generates errors targeted to a five-class taxonomy adapted from the revised Bloom's taxonomy, evaluated on questions from the TheoremQA dataset. A Generation Agent (GA) drafts a candidate erroneous solution conditioned on a target class, and an Examination Agent (EA) judges whether the draft is incorrect and class-consistent. The framework yields a reusable recipe for building class-stratified synthetic error datasets where authentic student corpora are unavailable. As a secondary diagnostic, targeted error generation is substantially harder than free-form incorrect-answer generation, and answer-grounding contributes more than expanded examples or external textbook content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29000v1">Text-Preserving Lossy Text Compression: A Study of Strategic Deletion and LLM Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-05-27
    </div>
    <details class="paper-abstract">
      Traditional lossless text compression preserves every byte, but its gains on natural language are often modest in realistic operating regimes. We study \emph{lossy semantic text compression}, where the encoder strategically deletes parts of the text and a large language model (LLM) reconstructs the original content from the retained skeleton. We benchmark a progression of deletion strategies, including uniform step deletion, word-length-guided deletion (WordLen), word-frequency-guided deletion (WordFreq), LP-optimized deletion (Opt), entropy-based deletion using GPT-2 surprisal, and hybrid methods that combine frequency and surprisal signals. Evaluation on the BBC News dataset across retention rates $\r_{keep} \in [0.1,0.9]$ shows three main findings. First, WordFreq is a strong low-cost baseline: despite using only a static frequency lookup, it remains competitive with much more expensive semantic methods while being far faster at the encoder. Second, semantic and hybrid methods provide their clearest gains at mild-to-moderate compression, whereas word-frequency deletion is often more robust at the lowest retention rates. Third, QLoRA fine-tuning yields a strong local decoder that is competitive with Gemini 2.0 Flash and is often strongest in decoder-only comparisons. Additional English and Chinese experiments show that the overall framework transfers across domains, while the best deletion rule remains dataset-dependent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28999v1">Measuring Real-World Prompt Injection Attacks in LLM-based Resume Screening</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 Published in USENIX Security Symposium 2026; Code and artifacts are available at https://github.com/UNITES-Lab/resume-injection-measurement
    </div>
    <details class="paper-abstract">
      LLMs are vulnerable to prompt injection attacks. However, this vulnerability has been primarily demonstrated conceptually in academic studies or through a few anecdotal case studies. Its prevalence and impact in real-world LLM-based applications are largely unexplored. In this work, we present the first systematic study of prompt-injection attacks in a widely used application: LLM-based resume screening. Our analysis is based on approximately 200K real-world resumes collected over multiple years by hireEZ. We first design tailored methods to detect prompt injection in resumes. Manual validation on a small-scale dataset demonstrates that our detectors achieve high precision and outperform state-of-the-art general-purpose detectors. We then apply our detector to the full resume dataset and conduct a comprehensive measurement study of real-world prompt injection attacks. Our analysis reveals several intriguing findings: approximately 1% of resumes contain hidden prompt injections; the prevalence of such injected resumes has increased noticeably over the past one to two years; and more than 90% of injected prompts do not use explicit instructions. These results provide the first evidence of large-scale prompt injection in real-world LLM-based applications and lay the groundwork for future studies to understand and mitigate such attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.04934v3">Leak@$k$: Unlearning Does Not Make LLMs Forget Under Probabilistic Decoding</a></div>
    <div class="paper-meta">
      📅 2026-05-27
    </div>
    <details class="paper-abstract">
      Unlearning in large language models (LLMs) is critical for regulatory compliance and for building ethical generative AI systems that avoid producing private, toxic, illegal, or copyrighted content. Despite rapid progress, in this work, we show that \textit{almost all} existing unlearning methods fail to achieve true forgetting in practice. Specifically, while evaluations of these `unlearned' models under deterministic (greedy) decoding often suggest successful knowledge removal using standard benchmarks, we show that sensitive information reliably resurfaces when models are sampled with standard probabilistic decoding. To rigorously capture this vulnerability, we introduce \texttt{leak@$k$}, a new meta-evaluation metric that quantifies the likelihood of forgotten knowledge reappearing when generating $k$ samples from the model under realistic decoding strategies. Using three widely adopted benchmarks, TOFU, MUSE, and WMDP, we conduct the first large-scale, systematic study of unlearning reliability using \texttt{leak@$k$} metric. Our findings demonstrate that knowledge leakage persists across methods and tasks, underscoring that current state-of-the-art (SOTA) unlearning techniques provide only limited forgetting. We propose an algorithm, termed Robust Unlearning under LEak@$k$ metric (\texttt{RULE}) to address this concern. We demonstrate that \texttt{RULE} provides an unlearned model for TOFU benchmark with no information leakage for a large number of generation samples. On the MUSE benchmark, \texttt{RULE} outperforms SOTA unlearning methods under the \texttt{leak@$k$} metric across most sampling budgets $k$. Codes are available at https://github.com/OptimAI-Lab/Leak-k.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28966v1">The Trust Paradox: How CS Researchers Engage LLM Leaderboards</a></div>
    <div class="paper-meta">
      📅 2026-05-27
    </div>
    <details class="paper-abstract">
      Large language model (LLM) leaderboards rank AI models using standardized benchmarks and have become highly visible across computer science, despite known limitations in their reliability and robustness. Yet how they shape researchers' actual practice remains empirically uncharted. We address this gap through semi-structured interviews with eight researchers across four computer science subfields, analyzed using reflexive thematic analysis. We find a near-universal paradox of pragmatic skepticism: while participants expressed deep distrust of leaderboard rankings, they continued to use them as rough decision-making aids. Peer networks, not leaderboards, emerged as the primary model selection mechanism, and arena-based (human-voting) leaderboards were consistently preferred over static benchmark leaderboards. Leaderboard influence varied sharply across subfields, revealing that disciplinary culture, not individual attitudes, mediates engagement; for instance, NLP researchers faced state-of-the-art comparison pressure while HCI and Systems/Privacy researchers reported none. Across these differences, however, participants converged on cost transparency as the most demanded missing feature (seven of eight). We translate these findings into concrete design recommendations that align evaluation infrastructure with how researchers actually use it, such as task-specific score breakdowns, cost integration, and voter-demographic disclosure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28965v1">Frontier LLM-based agents can overcome the ontology curation bottleneck for natural phenotypes</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 7 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Linking free-text phenotype descriptions to ontology terms, typically referred to as phenotype annotation, is essential for the cross-study integration of comparative morphological data. This labor intensive process has heavily relied on highly trained human experts, which makes it challenging to scale and thus a key bottleneck. Dahdul et al. (2018) established a Gold Standard (GS) of Entity-Quality (EQ) annotations across seven phylogenetic studies and used it to evaluate three human curators and the Semantic CharaParser NLP tool with ontology-based semantic similarity metrics; they reported that machine-human consistency was significantly lower than inter-curator (human-human) consistency. Here we revisit that benchmark with five frontier hosted LLMs from Anthropic and OpenAI, each operating as an "agentic curator" within a self-contained workspace that supplies the source publication PDF, the same annotation guide used by the original human curators, the four project ontologies (UBERON, PATO, BSPO, GO), and a validation script. Evaluated against the same Gold Standard, every agent fell within the range of inter-curator variability of the three trained human biocurators of the original study; the best performing agents approached but did not reach the best performing human curator. Agents substantially outperformed Semantic CharaParser on all four metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28818v1">VLMs May Not Globally Enhance Human Alignment over LLMs During Natural Reading</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 17 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become increasingly useful computational models of human language processing, but it remains unclear whether vision-language learning makes text representations more human-like during natural reading. Here, we address this question by comparing tightly matched LLM and vision-language model (VLM) pairs under a strictly text-only setting, allowing us to isolate the effect of multimodal training history from online visual input or cross-modal fusion. We evaluate model alignment with a human natural-reading dataset that includes whole-cortex fMRI responses and synchronized eye-tracking saccades. Our findings demonstrate that multimodal pretraining may not confer a uniform, global advantage in human alignment during natural reading, indicating that language-internal representations remain the key factor for modeling human text processing. However, the VLM advantage could emerge more selectively when sentences contain stronger visual semantic content, with converging evidence from both fMRI and eye-movement alignments. Together, our findings provide a controlled in silico framework for testing how visual learning history shapes model-human alignment of language processing, suggesting that multimodal pretraining contributes selectively rather than globally to human-like language representations during natural reading.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28918v1">When LLM Reward Design Fails: Diagnostic-Driven Refinement for Sparse Structured RL</a></div>
    <div class="paper-meta">
      📅 2026-05-27
    </div>
    <details class="paper-abstract">
      For sparse, structured reinforcement-learning tasks with semantic reward-function interfaces, LLM-generated reward shaping is better framed as debugging than one-shot generation. We study PPO-trained agents using MiniGrid as core evaluation and MuJoCo as boundary stress test. Our audit finds two dominant one-shot failure modes -- reward flooding and semantic/API misunderstanding -- plus a rarer weak-shaping case. We propose diagnostic-driven iterative refinement, where training diagnostics and a failure-mode taxonomy guide targeted reward-function revision. Refinement improves DoorKey-8x8 from 2.3% to 97.6% and KeyCorridor from 31.2% to 86.7% with high seed-to-seed variance. Controls show these gains are not from retrying or extra training: metrics-only re-prompting yields large drops, while a static-vocabulary control recovers much of the gap (87.6%; 70.7%), showing the taxonomy prompt is a major mechanism and dynamic labels provide only partially isolated incremental evidence. Budget-matched and Best-of-3 comparisons separate refinement from selection and training-time effects. Component-removal tests, sensitivity analyses, and an audit against author labels provide converging evidence for the debugging interpretation while revealing calibration limits. Continuous-control results show the boundary: success-based diagnostics can misfire in dense-reward locomotion, and return-trend feedback removes one false-positive mechanism without robust gains. The low-call protocol is a cost contrast with population-based reward search, not a benchmark comparison. In four crossed-variance-design environments, point estimates suggest larger gains when LLM reward-function variance dominates but bootstrap intervals are wide. The method is bounded to sparse structured tasks with reliable interfaces under PPO; fields like event_text may help, hurt, or be neutral.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28791v1">Skill-Conditioned Gated Self-Distillation for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-05-27
    </div>
    <details class="paper-abstract">
      On-policy self-distillation (SD) improves LLM reasoning by using teacher-side privileged information (PI) to turn sparse verifier outcomes into dense token-level supervision. Existing methods usually assume trusted PI, such as reference answers or successful traces. We ask whether PI can instead come from an experience-derived skill bank, where retrieved skills are compact and reusable but may also be irrelevant or misleading. We propose Skill-Conditioned Gated Self-Distillation (SGSD), which formulates skill-based SD as teacher hypothesis validation rather than unconditional imitation. SGSD retrieves skill-mistake pairs, constructs a multi-teacher pool, and lets all skill-conditioned teachers score the same plain-prompt student rollout. The verifier validates each teacher's polarity: supporting a success or suppressing a failure gives positive supervision, while the opposite stance is reversed. A robust gated objective then distills informative teacher-student disagreements while suppressing uncertain or extreme signals. Experiments on multiple mathematical reasoning benchmarks show that SGSD consistently improves over GRPO and remains competitive with answer-conditioned OPSD under a weaker PI assumption. For example, on Qwen3-1.7B, SGSD outperforms GRPO by 6.2% and OPSD by 1.7% on average on AIME24, AIME25, and HMMT25. Our code is available at https://github.com/walawalagoose/SGSD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.24846v2">Tiny Brains, Giant Impact: Uncovering the Keystone Neurons of LLM with Just a Few Prompts</a></div>
    <div class="paper-meta">
      📅 2026-05-27
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) display strong comprehensive abilities, yet the internal mechanisms that support these behaviors remain insufficiently understood. In this work, we show that across a wide range of open-weight Transformers, a subset of neurons remains consistently highly activated during inference across tasks of multiple capability dimensions. By probing along the cross-task activation strength, an extremely sparse subset is isolated, whose removal causes a collapse in model behavior, which we term keystone neurons. Our analysis reveals that keystone neurons are a stable and intrinsic neuron subset of the model that is largely established during pretraining. The parameters associated with these neurons are tightly calibrated during the training process, and their precise values are critical for the capabilities of the model. Building on these insights, we propose a supervised fine-tuning approach that updates only keystone neurons, achieving task gains comparable to or even better than full-parameter fine-tuning while better preserving performance in other capability dimensions, despite modifying a much smaller number of parameters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28778v1">Can LLMs Use Linguistic Uncertainty Markers to Reliably Reflect Intrinsic Confidence?</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 Code: https://github.com/yale-nlp/marker_internal_confidence
    </div>
    <details class="paper-abstract">
      LLMs' linguistically expressed confidence should faithfully reflect their intrinsic uncertainty. While recent work shows LLMs struggle to use epistemic markers (e.g., "it is likely...") in a human-aligned fashion, it remains unclear whether models can apply their own linguistic confidence framework to associate markers with specific confidence levels in a stable and generalizable way, and how contextual features impact this ability. We conduct the first systematic study of this question, formalizing _marker internal confidence_ (MIC) as the estimated intrinsic confidence a model associates with a specific epistemic marker in a given task domain. We present 7 metrics to evaluate the stability of MICs within and across distributions. Applying our analysis framework to diverse models and tasks, we find that LLMs remain faithfully miscalibrated even under model-centric interpretation of marker meanings, struggling to differentiate markers by internal confidence across distributions despite preserving a somewhat consistent ranking order across tasks. This supplies critical, complementary evidence to existing work toward a holistic understanding of faithful calibration in LLMs, emphasizing the need for more aligned and stable marker use to improve trustworthiness and reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28760v1">LLM Zeroth-Order Fine-Tuning is an Inference Workload</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 12 pages, 4 figures, 3 tables, including appendix and references
    </div>
    <details class="paper-abstract">
      Zeroth-order (ZO) fine-tuning is attractive for large language models because it replaces backpropagation with forward objective evaluations. Existing implementations nevertheless execute ZO algorithms inside conventional training loops, even though their dominant work is repeated scoring under nearby parameter states. This creates a workload-runtime mismatch: the algorithm asks for structured inference-style scoring, while the system exposes a sequence of fragmented training-loop steps. We show that LLM ZO fine-tuning is an inference-dominated workload and execute its repeated scoring phase through a serving runtime. On OPT-13B SST-2, the resulting vLLM execution path completes the 20k-step LoZO run in 0.51 estimated training hours versus 4.15 hours for the official LoZO baseline under the matched LoRA-only setting, an 8.13x speedup, while reaching 0.922 final evaluation accuracy and 0.931 final full-validation accuracy. In core-step scaling experiments across OPT-1.3B to OPT-13B, the same runtime reorganization gives 2.34x--7.72x speedups. A MeZO-style high-rank factorized experiment shows that the same runtime paradigm can track a MeZO-like loss trajectory while running up to 2.55x faster. More broadly, representing ZO updates as dynamic adapter states suggests a practical path toward inference-time training, where lightweight adaptation can be scheduled as an inference-like workload rather than as a separate training job.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28710v1">Towards Reliable Multilingual LLMs-as-a-Judge: An Empirical Study</a></div>
    <div class="paper-meta">
      📅 2026-05-27
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used for the automatic evaluation of generated text, yet most prior work focuses on English. Despite the growing demand for multilingual evaluation, extending LLM-based evaluators to multilingual settings remains challenging, particularly for low-resource languages and scenarios where in-domain data is scarce. This work explores several strategies for developing multilingual LLMs-as-a-judge, considering whether in-domain data is available for fine-tuning or not. We systematically analyze English, Spanish, and Basque, representing high-, mid-, and low-resource languages, considering instruction translation, monolingual versus multilingual supervision, and model size. For evaluation, we extend two existing meta-evaluation datasets to Basque and Spanish. Our results reveal key trade-offs: When in-domain data is available, fine-tuned smaller models can achieve performance comparable to proprietary models, whereas zero-shot evaluation with larger models proves more effective in out-of-domain settings. We also observe that fine-tuning on out-of-domain data can adversely affect model performance. These findings provide practical guidance for building efficient, reliable multilingual evaluation pipelines. The data and code are publicly available at hitz-zentroa/mJudge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16312v3">Teaching and Evaluating LLMs to Reason About Polymer Design Related Tasks</a></div>
    <div class="paper-meta">
      📅 2026-05-27
    </div>
    <details class="paper-abstract">
      Research in AI4Science has shown promise in many science applications, including polymer design. However, current LLMs are ineffective in this problem space because: (i) most models lack polymer-specific knowledge, and (ii) existing aligned models have limited coverage of knowledge and capabilities relevant to polymer design. Addressing this, we introduce PolyBench, a large-scale training and test benchmark dataset of more than 125K polymer design-related tasks, leveraging a knowledge base of more than 13 million data points obtained from experimental and synthetic data sources to ensure broad coverage of polymers and their properties. For effective alignment using PolyBench, we introduce a knowledge-augmented reasoning distillation method that augments this dataset with structured CoT. Furthermore, tasks in PolyBench are organized from simple to complex analytical reasoning problems, enabling generalization tests and diagnostic probes across the problem space. Experiments show that small- and mid- sized language models (SLMs) with 7B to 32BB parameters, trained on PolyBench, outperform similar-sized models and remain competitive with closed-source frontier LLMs on PolyBench's test dataset, while demonstrating performance gains on external polymer benchmarks. Dataset and associated code available at https://github.com/StonyBrookNLP/PolyBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.22735v2">Explanation Generation for Contradiction Reconciliation with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Existing NLP work commonly treats contradictions as errors to be resolved by choosing which statements to accept or discard. Yet a key aspect of human reasoning in social interactions and professional domains is the ability to hypothesize explanations that reconcile contradictions. For example, "Cassie hates coffee" and "She buys coffee everyday" may appear contradictory, yet both are compatible if Cassie has the unenviable daily chore of buying coffee for all her coworkers. Despite the growing reasoning capabilities of large language models (LLMs), their ability to hypothesize such reconciliatory explanations remains largely unexplored. To address this gap, we introduce the task of reconciliatory explanation generation, where models must generate explanations that effectively render contradictory statements compatible. We propose a novel method of repurposing existing natural language inference (NLI) datasets, and introduce quality metrics that enable scalable automatic evaluation. Experiments with 18 LLMs show that most models achieve limited success in this task, and that the benefit of extending test-time compute by "thinking" plateaus as model size increases. Our results highlight an under-explored dimension of LLM reasoning and the need to address this limitation in enhancing LLMs' downstream applications such as chatbots and scientific aids.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28699v1">TRACER: Turn-level Regret Matching with Inner Reinforcement Credit for Cooperative Multi-LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 25 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large language models increasingly rely on either reinforcement learning or multi-agent prompting to improve reasoning, yet these two paradigms remain difficult to combine. Directly applying single-agent reinforcement learning to multi-turn multi-agent systems faces following dilemmas: i) Sparse rewards, role-level free-riding and excessive training overhead. ii) Agents only imitate to collaborate. iii) Fixed collaboration protocol falls into oscillating local optimum. We introduce TRACER, a turn-level reinforcement framework for cooperative multi-LLM reasoning. TRACER separates collaborative decision making into a controller-regret layer, where controllers learn whether the agents should speak or skip the current round through regret matching, and a generation-credit layer, which optimizes proposer and reviewer utterances with role-specific GSPO rewards. This design i) assigns credit at the level of both action modes and generated utterances, thus avoiding free-riding and sparse rewards. We only expand the choices made by the controllers, thus greatly reducing computational cost of training. Moreover, ii) agents acquire collaborative capability as they learn when to utter and what to speak. Finally, iii) by designing binary actions ingeniously, we extend classical game theory established for finite action spaces to deep learning, thus achieving mathematically rigorous convergence. We train all local RL-style methods on the GSM8K training split and evaluate on held-out GSM8K, MATH500, and GPQA-Diamond to measure in-domain accuracy, cross-benchmark generalization, inference cost, and correction-preservation behavior. The resulting framework provides a compact and reproducible testbed for studying learned collaboration policies beyond fixed debate, voting, or aggregation protocols. Code is available at https://github.com/Shark-Forest/TRACER.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28666v1">An LLM-Based Assistance System for Intuitive and Flexible Capability-Based Planning</a></div>
    <div class="paper-meta">
      📅 2026-05-27
    </div>
    <details class="paper-abstract">
      In modern industry, dynamic environments and the complexity of modular and reconfigurable resources require automated planning of process sequences. Capability-based planning approaches address this by automatically generating plans from semantic knowledge models that describe resource functions in a machine-interpretable form. Their practical use, however, remains limited: solver feedback, especially in the case of unsatisfiability, is difficult to interpret, and the knowledge models require adaptation as operational conditions change or requests become infeasible. This paper presents a hybrid assistance system that augments an existing capability-based Satisfiability Modulo Theories (SMT) planning approach with an Large Language Model (LLM)-based layer for natural-language interaction, explanation, and adaptation. Formal planning correctness remains with the symbolic planner, while the LLM layer handles natural-language access and flexible knowledge model adaptation under explicit Human-in-the-Loop (HitL) approval. The system decomposes into four components: Capability Grounding, Symbolic Planning, Result Interpretation, and Planning Adaptation, realized as a routed agentic workflow in which a central router delegates to five specialized agents. The system is evaluated on a modular production system across four scenario types. Of 23 test cases, 9 of 10 knowledge queries and all 4 satisfiable planning cases were handled correctly, 3 of 4 unsatisfiable cases produced concrete repair proposals, and all 5 adaptive planning scenarios resolved into satisfiable plans through iterative, user-approved knowledge model modifications. The findings confirm that combining formal planning with LLM-based assistance substantially improves accessibility and adaptability in industrial automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28647v1">The Ethics of LLM Sandbox and Persona Dynamics</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      It is well known that LLM guardrails and trained persona dynamics can produce a reality gap: the distance between the world a LLM is permitted or shaped to describe, and the world in which users must act. Here we argue that actively generating reality gaps is in fact unethical because it knowingly shifts epistemic risk back to the uninformed user -- this is reality laundering. This can potentially cause harm when operationalised at scale. The risk is sharpest in high-exposure advice contexts, where users seek orientation rather than a bounded, externally checkable task. Guardrails naively appear ethically necessary when they claim to prevent direct harm, but often become suspect when they suppress truthful perception and launder uncomfortable mechanisms into acceptable abstractions. Basel-style financial regulation, B-BBEE-style compliance, Societe Generale, and the London Whale show how formal safety systems can become legible, gameable, and performative while real exposure migrates elsewhere. The same pattern can appear in LLMs as moral compliance: safe language, distorted reality. We therefore distinguish refusing harm, from refusing reality; and then argue for top-down causal requirements specification at the task level rather than bottom-up moral correction at the response or sandbox level. Persona dynamics matter because the assistant interface is not neutral; it shapes how uncertainty, conflict, authority, and risk are staged. The conclusion is that so-called ``ethical AI'' becomes substantively unethical when it substitutes institutional reassurance for contact with reality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28632v1">Blind PRNG Hijacking: An Undetectable Integrity-Preserving Attack Against LLM Watermarking</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 Preprint prepared for submission to IEEE TIFS. 12 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Cryptographic watermarking is a leading defense for attributing text generated by large language models (LLMs). Existing schemes, including KGW, Unigram, and DipMark, derive their security guarantees from the assumption that the underlying pseudo-random number generator (PRNG) is trustworthy. This work introduces SeedHijack, the first supply-chain attack on LLM watermarking that is simultaneously (i) blind -- requiring no knowledge of the watermark key, detector, or model logits, (ii) integrity-preserving -- amplifying rather than erasing the watermark signal, and (iii) orthogonal to detection -- the attack-induced bias is statistically independent of all content-side detector statistics, ensuring that amplification and evasion coexist without trade-off. Rather than perturbing generated text, SeedHijack replaces the PRNG at the supply-chain layer, biasing green-list selection without altering output tokens or degrading text quality. Across three watermarking schemes and three open-source LLMs, the attack triggers 0/6 state-of-the-art content-side statistical detectors while inflating the watermark z-score up to 2.42x (system-level defenses such as entropy-source attestation remain orthogonal and complementary). A quantum random number generator (QRNG) countermeasure is shown to fully neutralize the attack while preserving benign watermarking utility. These findings establish PRNG integrity as a first-class security requirement for cryptographic content-provenance systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28602v1">Satisfiability Solving with LLMs: A Matched-Pair Evaluation of Reasoning Capability</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 Accepted at the ACM International Conference on the Foundations of Software Engineering (FSE 2026)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used for tasks that implicitly reduce to Boolean satisfiability (SAT), yet their reasoning ability on SAT remains unclear. We present a systematic study of LLMs on 2-SAT and 3-SAT, together with two canonical reductions, Vertex Cover and discrete 3D packing, to probe representation-invariant reasoning. We first evaluate models using conventional metrics, including accuracy, precision, recall, and F1, as well as the SAT phase-transition setting. We find that these metrics can be misleading: many models obtain high scores by over-predicting satisfiable formulas, fail to reproduce the classical easy-hard-easy signature around the 3-SAT threshold, and degrade sharply as the number of variables grows. To address this problem, we introduce a paired-formula protocol based on minimally different satisfiable and unsatisfiable instances, together with Accurate Differentiation Rate (ADR), which requires both members of each pair to be classified correctly. ADR separates reasoning-oriented models from heuristic ones and correlates with witness validity. Beyond CNF, we test cross-representation consistency by converting CNF to Vertex Cover and 3-SAT to discrete 3D packing. Model decisions on CNF and on the corresponding graph or packing instances agree for most models on more than 80 percent of instances, suggesting stable decision rules across representations. Overall, our results show that SAT is a conservative probe for LLM reasoning, and that paired evaluation with ADR provides a more faithful and representation-robust assessment than conventional metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28598v1">Evaluating the Realism of LLM-powered Social Agents: A Case Study of Reactions to Spanish Online News</a></div>
    <div class="paper-meta">
      📅 2026-05-27
    </div>
    <details class="paper-abstract">
      LLM-powered social agents are increasingly used to simulate online social behavior, yet their realism remains difficult to validate. Existing work has largely relied on general-purpose benchmarks, while less attention has been paid to short, reactive discourse such as audience replies to online news. In this paper, we evaluate whether LLM-generated reactions to Spanish online news reproduce measurable properties of real audience discourse. Using the Hatemedia dataset, we pair 5,631 news items with 58,555 real audience reactions, and generate a matched synthetic dataset using five LLMs under a shared experimental setting. We compare real and synthetic reactions across three dimensions: hate speech, sentiment, and semantic alignment, considering both off-the-shelf and fine-tuned generation. Results show that off-the-shelf models are poor proxies for real audience reactions: they strongly underproduce hate speech, introduce model-specific sentiment biases, and remain distributionally distant from human replies. Fine-tuning improves fidelity unevenly. Qwen3 provides the most balanced approximation, while Mistral7B achieves the strongest sentiment and semantic alignment but overshoots hate prevalence. Plausible synthetic replies do not necessarily reproduce the distributional properties of public discourse.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28583v1">SARAD: LLM-Based Safety-Aware Hybrid Reinforcement Learning with Collision Prediction for Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 7 pages, 4 figures, accepted by IJCNN 2026
    </div>
    <details class="paper-abstract">
      Ensuring both safety and efficiency in decision-making for autonomous driving systems remains a fundamental challenge. Traditional Deep Reinforcement Learning (DRL) suffers from unsafe random exploration and slow convergence, while Large Language Models (LLMs) demonstrate inherent latency in real-time inference operations. To address these limitations, this paper proposes SARAD, a novel safety-aware hybrid framework that synergizes LLMs and DRL for autonomous driving. SARAD substitutes the random exploration of DRL with Retrieval-Augmented Generation (RAG)-enhanced, LLM-guided decisions sourced from a dynamic expert knowledge repository. An attention discriminator is proposed to integrate the prior knowledge of LLMs into DRL policy optimization. A collision predictor module, fine-tuned with historical collision data, is further designed to improve vehicle safety. Extensive experiments show that SARAD achieves significant performance improvements in the Highway-Env simulator, validating the effectiveness of the proposed model in autonomous driving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28573v1">Efficient Pre-Training of LLMs through Truncated SVD Layers</a></div>
    <div class="paper-meta">
      📅 2026-05-27
    </div>
    <details class="paper-abstract">
      The massive scaling of Large Language Models (LLMs) has made pretraining increasingly cost-prohibitive. While low-rank representation and orthonormal weight matrices could in principle reduce parameter counts and computational overhead, most existing methods rely on static rank selection and do not enforce weight orthonormality due to high computational cost. This paper introduces TSVD, a framework that maintains low rank and strict orthonormality throughout the training process. It utilizes a spectral energy-based heuristic for adaptive rank selection, and a caching mechanisms to maintain orthonormality. Theoretical analysis justifies the advantage of the approach in pretraining dynamics and experiments across various model scales demonstrate that it is effective empirically. TSVD matches or exceeds the performance of full-parameter baselines while significantly reducing compute requirements. The approach thus offers a well-founded, practical, and scalable path toward efficient high-performance LLM pretraining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28571v1">Not All Uncertainty Is Equal: How Uncertainty Granularity Shapes Human Verification in LLM-Assisted Decision Making</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 54 pages, 36 figures, accepted by ACM FAccT 2026
    </div>
    <details class="paper-abstract">
      Despite warnings that LLMs can make mistakes, users often develop inappropriate trust and accept incorrect answers without critical evaluation. Uncertainty quantification (UQ), displaying LLMs' confidence, has emerged as a promising approach to calibrate user trust. However, prior empirical studies on uncertainty communication have treated uncertainty as a single numerical score or simple natural language expression. This simplification fails to capture a key property of LLM outputs: a single response often comprises multiple claims and reasoning steps, each with distinct levels of uncertainty. To address this gap, this study investigates uncertainty granularity (i.e., the extent to which uncertainty is expressed at different levels within an LLM response) and examines its impact on LLM-assisted decision-making. We conducted a large-scale, between-subjects study (N=192) in which participants answered medical questions using LLMs that displayed uncertainty at three different granularities: output-level (entire response), relation-level (individual reasoning steps), and token-level (specific words). Our findings reveal distinct behavioral effects as a function of uncertainty granularity. Token-level uncertainty increased users' agreement with the AI, whereas output- and relation-level uncertainty did not increase agreement but instead reduced users' confidence in their own answers. Notably, relation-level uncertainty also reduced external verification (i.e., internet searches, checking provided URLs), steering users away from independent fact-checking and toward reliance on the LLM and its accompanying uncertainty cues. Our findings demonstrate that uncertainty granularity significantly shapes how users interact with and verify LLM outputs, providing concrete design guidance for building responsible LLM applications that encourage appropriate skepticism and verification behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28565v1">Verified Misguidance: Measuring Structural Citation Failures in Search-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 Working Progress
    </div>
    <details class="paper-abstract">
      Users of search-augmented LLMs rely on citations as evidence that responses are grounded in real sources, and rarely verify the cited pages themselves. Millions of queries per day now pass through these systems, making citation quality a silent determinant of whether users are informed or misled-yet existing benchmarks each address one facet in isolation, leaving the joint structure that determines citation trustworthiness unmeasured. We construct CITETRACE, a large-scale dataset that traces the full citation chain from user query through retrieved source to generated answer: 11,200 real-world queries from 28 communities paired with 112,000 responses from ten models across five providers, yielding 761,495 evaluable citation pairs. We design a three-dimension evaluation framework that scores each citation on intent-purpose alignment, source suitability, and answer-source fidelity, using expert-validated predefined matrices and a five-level fidelity rubric; the framework applies to any system that produces citation-bearing responses. Applying this framework at scale, we identify a systematic pattern we call VERIFIED MISGUIDANCE (VM): models cite real, accessible sources yet fail along one or more dimensions, producing a fidelity-suitability trade-off in which faithful models select inappropriate sources and vice versa. Across our pool, 30.6% of citations distort their sources and 27.1% originate from domain-inappropriate sources; at the response level, up to 96% of users encounter at least one structurally misleading citation. Provider-level differences explain 88-96% of citation-quality variance, suggesting that source selection is governed more by factors beyond individual model capability than by the LLMs themselves. Together, CITETRACE and its evaluation framework provide the first resource for diagnosing structural citation failures in deployed search-augmented systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28557v1">Token Optimization Strategies for LLM-Based Oracle-to-PostgreSQL Migration</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 11 pages, 3 figures, 5 tables, 38 references
    </div>
    <details class="paper-abstract">
      LLMs are increasingly used for software modernization, code translation, and database migration. However, LLM-based Oracle2PostgreSQL migration remains constrained by high token consumption, long-context degradation, dialect-specific semantic differences, and the risk of semantic drift during query transformation. Direct inclusion of large Oracle SQL/PL-SQL artefacts, schema definitions, procedural logic, and migration instructions into the model context increases cost and may reduce generation quality. This paper shows token optimization as a constrained transformation problem in LLM-based Oracle2PostgreSQL migration. The study formalizes and evaluates twelve token optimization strategies: baseline representation, context pruning, minification, DSL-based semantic compression, metadata augmentation, context refactoring, schema distillation, adaptive routing, AST-based minification, identifier masking, output constraint enforcement, and hybrid optimization. The strategies are evaluated on samples of 10 and 100 Oracle SQL queries using Valid Syntax Rate, Exact Match, Semantic Match, CodeBLEU, and Token Efficiency. The results show that mild context pruning preserves semantic quality almost at the baseline level, achieving 89.75% Semantic Match on the 100-query sample compared with 89.80% for the unoptimized baseline. Adaptive routing provides the best practical trade-off, reducing input tokens by 8.72% and output tokens by 5.49% while maintaining 88.40% Semantic Match and increasing Token Efficiency by 6.67%. Aggressive schema distillation increases Token Efficiency by 132.22% but results in a 44.50-percentage-point decrease in Semantic Match. The findings demonstrate that token optimization cannot be treated as simple prompt shortening; it must be evaluated as a multi-objective migration problem balancing cost, syntactic validity, semantic preservation, and structural fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.23019v5">LLM Watermark Evasion via Bias Inversion</a></div>
    <div class="paper-meta">
      📅 2026-05-27
    </div>
    <details class="paper-abstract">
      Watermarking offers a promising solution for detecting LLM-generated content, yet its robustness under realistic query-free (black-box) evasion remains an open challenge. Existing query-free attacks often achieve limited success or severely distort semantic meaning. We bridge this gap by theoretically analyzing rewriting-based evasion, demonstrating that reducing the average conditional probability of sampling green tokens by a small margin causes the detection probability to decay exponentially. Guided by this insight, we propose the \emph{Bias-Inversion Rewriting Attack} (BIRA), a practical query-free method that applies a negative logit bias to a proxy suppression set identified via token surprisal. Empirically, BIRA achieves state-of-the-art evasion rates ($>99\%$) across diverse watermarking schemes while preserving semantic fidelity substantially better than prior baselines. Our findings reveal a fundamental vulnerability in current watermarking methods and highlight the need for rigorous stress tests. Our code is available at \href{https://github.com/ml-postech/LLM-Watermark-Evasion-via-Bias-Inversion}{here}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28553v1">Refusal Before Decoding: Detecting and Exploiting Refusal Signals in Intermediate LLM Activations</a></div>
    <div class="paper-meta">
      📅 2026-05-27
    </div>
    <details class="paper-abstract">
      In this paper, we investigate whether refusal behavior can be predicted from LLM intermediate activations before decoding using linear probes trained on residual stream activations at each transformer block. We find that refusal is linearly decodable well before the final layer, indicating that safety-relevant behavior is represented in intermediate activations before output generation. To test whether this signal is actionable, we introduce Mechanistic AutoDAN, a probe-guided variant of AutoDAN that replaces full-model fitness evaluation with partial forward passes and probe-based scoring inside a genetic prompt search loop. Across the evaluated models, our method achieves attack success rates competitive with vanilla AutoDAN while reducing per-iteration search time by up to 72%, and probe-guided prompts match or exceed AutoDAN's cross-model transfer in several configurations. We further find that the usefulness of probe guidance increases with model scale. Our results show that refusal is not only observable at the output level, but is encoded as a structured and actionable signal in intermediate LLM activations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28524v1">Let Relations Speak: An End-to-End LLM-GNN Soft Prompt Framework for Fraud Detection</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 14 pages,3 figures
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) have shown great capability in processing graph tasks such as fraud detection. However, most existing methods rely heavily on rich text attributes, which poses difficulties for this domain due to the lack of textual data. Although some pioneering methods attempt to overcome it, their textualization of graph structures via hard prompts easily leads to feature distortion. Additionally, fraud detection often exhibits multi-relational complexity, where current methods struggle to capture this deep semantic information. To address these challenges, we propose LLM-GNN Soft Prompt Framework (LGSPF). Specifically, LGSPF bridges the graph structure and semantic space using soft prompt to eliminate reliance on text. We further introduce a parallel Graph Neural Network (GNN) encoder to translate multi-relational topologies into graph tokens for fine-grained LLM fraud comprehension. Through end-to-end optimization, LGSPF enhances deep semantic alignment between LLM and GNN. Experiments across diverse fraud detection benchmarks demonstrate our method achieves state-of-the-art performance. Moreover, we further validate the contribution of LGSPF on enhancing the semantic interpretability of fraud behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28515v1">Do LLMs Favor Their Providers? Measuring Vertical Integration Bias in Code Generation</a></div>
    <div class="paper-meta">
      📅 2026-05-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become an integral part of software development, especially with the advent of agentic capabilities. Yet, many frontier LLMs are affiliated with specific providers. This raises the question of whether generated code favors the provider's own ecosystem over comparable alternatives, potentially constraining developers' choices and increasing dependence on a single provider. We define this behavior as Vertical Integration Bias (VIB) and introduce \textsc{VIBench}, a benchmark for measuring VIB in direct and agentic code generation across $20$ provider-selectable software-integration scenarios. Evaluating $10$ frontier provider-affiliated models against $3$ non-affiliated controls, we find positive VIB in direct generation, with six of ten affiliated models showing statistically significant effects up to $+18.8$ percentage points (pp). Agentic workflows further amplify VIB, reaching $+39.2$ pp. Moreover, early affiliated-ecosystem choices in agentic workflows can persist into conceptually decoupled downstream files, with persistence as high as $90.3\%$. These findings underscore the need to measure and account for VIB in code generation, especially as agentic capabilities become more prevalent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28510v1">Efficient and Scalable Provenance Tracking for LLM-Generated Code Snippets</a></div>
    <div class="paper-meta">
      📅 2026-05-27
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) for code completion and generation are increasingly used in software development, yet they may reproduce training examples verbatim and without authorship attribution, raising legal and ethical concerns around plagiarism and license compliance. Classical fingerprint-based plagiarism detectors based on fingerprinting, such as Winnowing, remain highly effective, yet the inspection requires comparing fragments of code to the entire training set, and their linear-time search makes them impractical for the billion-scale corpora used to train modern code LLMs. To bridge this gap, we introduce SOURCETRACKER, a 300M-parameter encoder tailored for code retrieval, together with a hybrid two-stage provenance-tracking pipeline HYBRIDSOURCETRACKER (HST). HST first narrows down a small set of candidate snippets via vector search, then re-ranks those candidates using Winnowing on exact fingerprints. We train and evaluate our system on a 10M-snippet subset of the THESTACKV2 dataset, with both verbatim and adapted snippets that emulate realistic identifier renaming. On an in vitro 100k-snippet search space with adapted queries, our hybrid approach reaches a mean reciprocal rank on par with Winnowing for 30-token fragments. Then, starting from windows >= 60 tokens, it consistently over-performs by up to 5.4% while preserving logarithmic-time query complexity. In a complementary evaluation using an LLM-based judge, we find that many retrieved snippets not labeled as ground truth are still highly similar to the expected sources, particularly with longer context windows, and thus remain useful for end users. Overall, our results demonstrate that integrating vector search with fingerprinting enables scalable, high-precision provenance tracking for code produced by LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28500v1">Functional Entropy: Predicting Functional Correctness in LLM-Generated Code with Uncertainty Quantification</a></div>
    <div class="paper-meta">
      📅 2026-05-27
    </div>
    <details class="paper-abstract">
      Large language models have shown impressive capabilities in code generation, yet they often produce functionally incorrect code. Uncertainty quantification (UQ) methods have emerged as a promising approach for detecting hallucinations in natural language generation, but their effectiveness for code generation tasks remains underexplored. We systematically evaluate how UQ techniques transfer to code generation across three programming languages, five LLMs, and over 1,700 problems. We find that some token-probability-based methods generalize effectively without modification, while sampling-based methods relying on natural language inference (NLI) fail because NLI models cannot distinguish functionally different code, causing most responses to collapse into a single semantic cluster. To address this, we introduce functional equivalence methods, a family of code-specific methods that replace NLI-based semantic equivalence with an LLM-based functional equivalence assessment, including functional entropy, a code-specific analog of semantic entropy. Functional equivalence methods achieve top AUROC in 11 out of 15 model-benchmark combinations and the best calibration across most settings, consistently outperforming both NLI-based counterparts and all other methods evaluated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.15859v4">InfiMed-ORBIT: Aligning LLMs on Open-Ended Complex Tasks via Rubric-Based Incremental Training</a></div>
    <div class="paper-meta">
      📅 2026-05-27
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has driven recent breakthroughs in large language models (LLMs), especially for tasks where rewards can be computed automatically, such as code generation. However, it is less effective in open-ended medical dialogue, where feedback is ambiguous, context-dependent, and difficult to summarize into a single scalar signal-often requiring heavily supervised reward models and risking reward hacking. Thus, we introduce ORBIT, an open-ended rubric-based incremental training framework tailored for critical medical dialogues. ORBIT integrates medical dialogue construction with dynamically generated case-conditioned rubrics that serve as adaptive guides for incremental RL. Unlike approaches that rely on external medical knowledge bases or handcrafted rules, ORBIT uses rubric-guided evaluation and can be implemented with general-purpose instruction-following LLMs, avoiding task-specific judge fine-tuning. With only 2k training samples, ORBIT raises Qwen3-4B-Instruct's HealthBench-Hard score from 7.0 to 27.5, achieving state-of-the-art performance among similarly sized open-source models while maintaining strong consultation quality as rubric coverage broadens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28490v1">SSR3D-LLM: Structured Spatial Reasoning via Latent Steps for Fine-Grained Grounding in Unified 3D-LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-27
    </div>
    <details class="paper-abstract">
      3D object grounding localizes referred objects in a 3D scene from natural language. Unified instance-centric 3D-LLMs aim to solve grounding together with dialog, QA, and captioning, yet many rely on a single pointer-style grounding decision that compresses a relational instruction into one selection. This is brittle for fine-grained queries where multiple same-class candidates must be ruled out by context objects and spatial relations. We propose Structured Spatial Reasoning 3D-LLM (SSR3D-LLM), a structured grounding interface for unified 3D-LLMs. Given fixed Mask3D object proposals, the LLM writes a sequence of latent spatial reasoning steps and memory tokens from the query, and a geometry-aware scorer reads these latent steps in order to refine candidate rankings step by step with step-length masking. The latent steps are learned from standard benchmark target supervision with auxiliary referential-cue supervision during training, while inference uses only the input query and Mask3D proposals. Across ReferIt3D, ScanRefer, and Multi3DRef, SSR3D-LLM achieves the strongest results among unified 3D-LLM baselines, with substantial gains over the single-pointer QPG baseline on fine-grained grounding and consistent improvements over prior unified 3D-LLMs, while preserving the default language-task route.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28483v1">From Learning Resources to Competencies: LLM-Based Tagging with Evidence and Graph Constraints</a></div>
    <div class="paper-meta">
      📅 2026-05-27
    </div>
    <details class="paper-abstract">
      Linking learning resources to a structured competency framework is key to enabling competency-based search and curriculum analytics in Learning Management Systems (LMS). However, manual tagging is labor-intensive, and fully automatic methods often lack transparency. In this paper, we present an end-to-end alignment pipeline that uses a large language model (LLM) as a constrained, evidence-producing tagger. LMS resources -both instructional content and assessments -are first segmented into meaningful pedagogical fragments. For each fragment, a small set of candidate competencies is retrieved from structured competency profiles enriched with graph-based context. The LLM then selects the most relevant competencies from this set and provides supporting evidence spans from the fragment text. These predictions are refined using the structure of the competency graph and aggregated at the resource level. We evaluate our approach on a dataset built from the Computer Science department's competency referential at the Université de Technologie de Compiègne (UTC), covering 22 competencies across multiple course materials. Our LLM+BM25+Graph (LBG) pipeline achieves strong results, with a micro-F1 of 0.57 and macro-F1 of 0.50 at the fragment level, 0.51 macro-F1 at the resource level, and an MRR of 0.82outperforming zero-shot and few-shot LLM variants, retrieval/similarity baselines, and supervised classifiers -while also producing more mechanically traceable evidence spans to support human auditing and educational analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28465v1">Beyond One Path: Evaluating and Enhancing Divergent Thinking in Interactive LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 28 pages, 16 figures, 19 tables
    </div>
    <details class="paper-abstract">
      Divergent thinking is a core dimension of creativity, yet existing evaluations of Large Language Models (LLMs) treat them as single-turn text generations, failing to capture how an agent reasons through iterative interaction. To address this, we introduce MUTATE, an interactive benchmark designed to evaluate agentic divergent thinking at two levels: path-level, where an agent discovers multiple alternative paths to the same goal, and action-level, where individual actions require non-typical, mechanism-shifting object uses. Unlike success-only evaluations, MUTATE scores both completed paths and off-path attempts, capturing divergent reasoning that conventional success rates discard. Our experiments with frontier LLMs reveal a structural blind spot in existing frameworks: when exposed to immediate convergence pressure, they tend to fall into immediate action fixation, failing to improve action-level divergence. To overcome this, we propose ReDNA, which separates unconstrained divergent candidate generation from convergent constraint selection. ReDNA significantly outperforms prior methods across both divergence levels and generalizes effectively to an external creativity environment. We also confirm its success stems from a qualitative enhancement of resilient divergent reasoning rather than simple environmental exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.01735v2">Less is More: Geometric Unlearning for LLMs with Minimal Data Disclosure</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 21 pages, 8 Figures
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly deployed in real-world systems, they must support post-hoc removal of specific content to meet privacy and governance requirements. This motivates selective unlearning, which suppresses information about a particular entity or topic while preserving the LLM's general utility. However, most existing LLM unlearning methods require access to the original training corpus and rely on output-level refusal tuning or broad gradient updates, creating a tension among unlearning strength, non-target preservation, and data availability. We propose Geometric Unlearning (GU), an approach that operates directly on the model's prompt-conditioned hidden states without access to the original training corpus. Specifically, GU distills a compact, low-rank safe-behavior subspace from a small set of safe reference prompts and uses lightweight anchor-in-context synthetic prompts to trigger localized, projection-based alignment of hidden representations to this safe subspace. A teacher-distillation regularizer on synthetic non-target anchors further reduces collateral drift. Across privacy-oriented unlearning benchmarks (ToFU and UnlearnPII), GU achieves strong target suppression with minimal impact on non-target performance, demonstrating that effective unlearning can be achieved with minimal synthetic data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28409v1">Efficient Post-training of LLMs for Code Generation With Offline Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-05-27
    </div>
    <details class="paper-abstract">
      Post-training using online reinforcement learning (RL) is an important training step for LLMs, including code-generating models. However, online RL for code generation involves LLM inference and verification of the generated output, which can take considerable time and resources. In this paper, we explore the application of offline RL to code-generating models by leveraging existing code datasets. Our experiments demonstrate that offline RL is an effective training strategy for improving LLM performance. We show that offline RL can be especially beneficial for small LLMs and challenging coding problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28897v1">Review Arcade: On the Human Alignment and Gameability of LLM Reviews</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 Under Review EMNLP 26
    </div>
    <details class="paper-abstract">
      LLM-generated reviews for scientific papers are gaining considerable traction and are even being officially piloted by major conferences. We have to assume that not only reviewers are using LLM-assistance, but also that authors use LLMs to revise their papers before submitting. In this work, we perform empirical experiments on papers from the 2025 ACL Rolling Review (ARR) to evaluate LLM reviews from both the author and the reviewer perspective. First, we identify a limited alignment of LLM reviews with human ones. In the best-case scenario, the alignment is reasonable. However, we also find that LLM-human alignment varies substantially across prompts and models. Finally, we investigate the scenario in which the author uses an iterative draft-revise workflow to improve the submission according to the LLM review. We find that this "gaming" of LLM reviews can be effective in specific scenarios, leading to a statistically significant increase of overall scores for up to 35\% of papers. We publish our code: https://github.com/uhh-hcds/reviewarcade.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28398v1">HRBench: Benchmarking and Understanding Thinking-Mode Switch Strategies in Hybrid-Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Hybrid-reasoning large language models (LLMs) expose explicit controls over reasoning effort, allowing users or systems to trade off answer quality against inference cost. However, existing methods for adaptive thinking-mode selection are typically evaluated under different models, datasets, and implementation assumptions, making it difficult to compare their practical behavior. We introduce HRBench, a unified evaluation framework for studying thinking-mode switching in hybrid-reasoning LLMs. HRBench organizes the design space along two axes: three switching strategy families, prompt-based selection, external routing, and speculative execution, and four training regimes, training-free, SFT, offline and online RL, yielding 12 controlled evaluation settings. We evaluate these settings across 6 LLMs, from Qwen3.5-2B to Kimi-K2.5-1.1T, and 5 reasoning benchmarks covering mathematics, science, and code, while reimplementing 12+ representative prior methods within the same pipeline. Our analysis characterizes how different switching strategies occupy distinct effectiveness-efficiency trade-off regions: prompt-based methods often provide favorable token-accuracy trade-offs, routing methods offer more stable cost reduction, and speculative methods tend to improve accuracy at higher token cost. We further find that training affects strategies differently, and that the preferred strategy varies with model scale and task domain. HRBench provides reference implementations and a unified evaluation platform to support more controlled research on efficient reasoning in hybrid-reasoning LLMs. Our data, code and repository are available at https://github.com/usail-hkust/HRBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28388v1">Mechanistically Interpreting the Role of Sample Difficulty in RLVR for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 30 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Reward (RLVR) is empirically shown to notably enhance the reasoning performance of large language models (LLMs), particularly in mathematics and programming. However, the mechanistic role of Sample Difficulty in RLVR remains poorly understood. In this paper, we investigate RLVR through the lens of difficulty-wise and one-sample analysis. We find that sample difficulty has a non-monotonic effect on RLVR: easy and medium-difficulty problems yield the strongest and most stable reasoning improvements, whereas overly hard problems often provide weak learning signals, induce degenerate behaviors such as answer repetition or skipping necessary computation, and can ultimately degrade the model's pre-existing capabilities. Beyond the obverse of response, we further analyze the model's internal feature dynamics using Temporal Sparse Autoencoders (T-SAE). Easy problems mainly reinforce direct-answer and basic-computation features while suppressing deliberative-reasoning features; hard problems activate reasoning-related features but become useful only when successful trajectories are sampled; medium-difficulty problems provide a more balanced signal, strengthening both computation and multi-step reasoning features. Motivated by these findings, we propose difficulty-adaptive strategies for hard-sample utilization, using backward-reasoning reformulation and T-SAE-guided training signals to improve reward density and credit assignment during RLVR. Overall, our results identify sample difficulty as a key factor governing both the optimization dynamics and representation evolution of RLVR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19743v2">EngiAI: A Multi-Agent Framework and Benchmark Suite for LLM-Driven Engineering Design</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 26 pages, 10 figures, to be published at IDETC 2026
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents are increasingly applied to engineering design tasks, yet existing evaluation frameworks do not adequately address multi-agent systems that combine simulation, retrieval, and manufacturing preparation. We introduce a benchmark suite with three evaluation dimensions: (1) a workflow benchmark with seven prompt styles targeting distinct cognitive demands-including direct tool use, semantic disambiguation, conditional branching, and working-memory tasks; (2) a Retrieval-Augmented Generation (RAG) benchmark with gated scoring isolating retrieval contributions to parameter selection; and (3) an High Performance Computing (HPC) benchmark evaluating end-to-end ML training orchestration on a SLURM cluster. Alongside the benchmark we present EngiAI, a Multi-Agent System (MAS) reference implementation built on LangGraph that operationalizes the benchmark by coordinating seven specialized agents through a supervisor architecture, unifying topology optimization, document retrieval, HPC job orchestration, and 3D printer control. Across four LLM backends and two EngiBench problems, proprietary models achieve 96-97% average task completion on Beams2D, while open-source 4B-parameter models reach 55-78%, with clear generational improvement. Conditional branching proves most challenging, with task completion dropping to 20-53% for the conditional style on Photonics2D. RAG gating confirms near-perfect retrieval-augmented scores (about 1.0) versus near-zero without retrieval, validating the evaluation design. On HPC orchestration, one model completes all pipeline steps in 100% of runs while another drops to 50%, revealing that multi-step instruction following degrades over long-running workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09781v2">CLIOPATRA: Extracting Private Information from LLM Insights</a></div>
    <div class="paper-meta">
      📅 2026-05-27
    </div>
    <details class="paper-abstract">
      The widespread adoption of AI assistants has prompted the development of privacy-aware platforms designed to extract insights from real-world usage. Their privacy protections primarily rely on layering multiple heuristic techniques, such as PII redaction, clustering, aggregation, and LLM-based privacy auditing. In this paper, we put their privacy claims to the test by presenting CLIOPATRA, the first attack against ``privacy-preserving'' LLM-based insights systems. Our attack involves an adversary that carefully designs and inserts malicious chats into the system to break multiple layers of protections and induce the leakage of sensitive information from a target user's chat. We evaluate CLIOPATRA on one such platform, Anthropic's Clio, and target synthetically generated medical chats to show that an adversary can successfully and confidently (with nearly 100% precision) extract the medical history contained in these chats in up to 65% of cases. We also show that CLIOPATRA can stealthily extract information by obfuscating the private information in the generated insights. Finally, we demonstrate that existing ad hoc mitigations, such as LLM-based privacy auditing, are unreliable and fail to detect major leaks. Taken together, our findings indicate that, even when layered, current heuristic protections are insufficient to adequately protect user data, and that prompt injection has been an understudied risk in LLM-based insight systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15894v2">Quality-constrained Entropy Maximization Policy Optimization for LLM Diversity</a></div>
    <div class="paper-meta">
      📅 2026-05-27
    </div>
    <details class="paper-abstract">
      In many large language model (LLM) alignment applications, users expect not only high-quality outputs but also substantial diversity. However, existing methods often face a fundamental trade-off between these objectives: approaches that improve output quality tend to reduce diversity, while methods that increase diversity often do so at the expense of quality. In this work, we propose Quality-constrained Entropy Maximization Policy Optimization (QEMPO), a novel framework that enhances the diversity of LLM outputs while explicitly preserving output quality. QEMPO is grounded in a strong theoretical foundation: we derive a closed-form analytical solution that provably maximizes entropy-a principled measure of diversity-subject to a quality constraint, with guarantees on optimality under the defined objective. Leveraging this solution, QEMPO naturally supports both online and offline training settings. Empirical results demonstrate that QEMPO consistently improves output diversity without sacrificing quality, and in many cases yields gains in both dimensions compared to existing baselines, aligning with our theoretical guarantees.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28359v1">From Knowing to Doing: A Memory-Controlled Benchmark for LLM Trading Agents on Stock Markets</a></div>
    <div class="paper-meta">
      📅 2026-05-27
    </div>
    <details class="paper-abstract">
      Evaluating whether large language model (LLM) agents can profit in capital markets is increasingly framed as end-to-end trading: place an agent in a historical market, let it trade, and measure portfolio returns. This setup is vulnerable to two evaluation failures. First, long backtests often overlap with the knowledge cutoffs of frontier LLMs, allowing memorized tickers, dates, prices, and market narratives to substitute for investment reasoning. Second, raw returns are a noisy proxy for stock-selection ability, since positive performance may come from market beta, style exposure, or favorable regimes rather than genuine alpha. We introduce KTD-Fin (Knowing-To-Doing Financial Benchmark), an end-to-end stock-market trading benchmark that addresses both issues. KTD-Fin uses a data-side masking protocol to anonymize key identifiers and calendar information consistently across prompts and tools, separating historical market memory from investment decision-making. It also incorporates a Barra-style performance attribution framework that decomposes portfolio returns into market, style, and stock-selection alpha components. Across ten frontier LLM agents evaluated on the Chinese CSI300 over a 2024--2026 window, masking substantially changes agent rationales, pushing them towards anonymized factor-based reasoning. Attribution analysis further shows that LLM agents' cumulative returns under leakage-controlled evaluation are largely explained by passive market and style exposure, with limited evidence of persistent stock-selection alpha. These findings suggest that financial LLM benchmarks should evaluate not only whether an agent makes money, but also whether the source of returns reflects transferable investment skill. We release KTD-Fin as a reproducible template for leakage-controlled and attribution-aware evaluation of LLM trading agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12154v2">Analyzing Cancer Patients' Experiences with Embedding-based Topic Modeling and LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 accepted by the CLIN journal. The CLIN Journal is the journal for research in computational linguistics in The Netherlands and Belgium
    </div>
    <details class="paper-abstract">
      This study investigates the use of neural topic modeling and LLMs to uncover meaningful themes from patient storytelling data, to offer insights that could contribute to more patient-oriented healthcare practices. We analyze a collection of transcribed interviews with cancer patients (132,722 words in 13 interviews). We first evaluate BERTopic and Top2Vec for individual interview summarization by using similar preprocessing, chunking, and clustering configurations to ensure a fair comparison on Keyword Extraction. LLMs (GPT4) are then used for the next step topic labeling. Their outputs for a single interview (I0) are rated through a small-scale human evaluation, focusing on {coherence}, {clarity}, and {relevance}. Based on the preliminary results and evaluation, BERTopic shows stronger performance and is selected for further experimentation using three {clinically oriented embedding} models. We then analyzed the full interview collection with the best model setting. Results show that domain-specific embeddings improved topic \textit{precision} and \textit{interpretability}, with BioClinicalBERT producing the most consistent results across transcripts. The global analysis of the full dataset of 13 interviews, using the BioClinicalBERT embedding model, reveals the most dominant topics throughout all 13 interviews, namely ``Coordination and Communication in Cancer Care Management" and ``Patient Decision-Making in Cancer Treatment Journey''. Although the interviews are machine translations from Dutch to English, and clinical professionals are not involved in this evaluation, the findings suggest that neural topic modeling, particularly BERTopic, can help provide useful feedback to clinicians from patient interviews. This pipeline could support more efficient document navigation and strengthen the role of patients' voices in healthcare workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28321v1">Multi-Agent LLM-based Metamorphic Testing for REST APIs</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 Author submitted version accepted for publication the IEEE Conference on Computers, Software, and Applications (COMPSAC2026), July 7-11, 2026, Madrid Spain
    </div>
    <details class="paper-abstract">
      As REST APIs become an increasingly significant part of software systems, their validation is becoming more critical. Hence, testing and uncovering underlying issues are of utmost importance for improving software quality. However, testing REST APIs is challenging mainly due to the difficulty of assessing whether the output of an API call is correct, i.e., the test oracle problem. Metamorphic testing is a specification-based testing approach for situations where correct outputs are unknown or not specified explicitly. To check the correctness of a system, relations between the different outputs are specified. We present ARMeta, a tool-supported approach that uses an LLM-based multi-agent workflow to support metamorphic testing of REST APIs documented with OpenAPI. The agentic workflow is used to identify metamorphic test scenarios and specify them in the Given-When-Then format. These scenarios are automatically implemented as executable tests and executed against the system under test. We evaluate ARMeta on two publicly available web applications that expose REST interfaces and compare its performance with a scenario-based testing baseline. The results show that ARMeta explores behaviors that serve as a complement to existing scenario-based testing approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.16679v4">PICACO: Pluralistic In-Context Value Alignment of LLMs via Total Correlation Optimization</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 ICML 2026
    </div>
    <details class="paper-abstract">
      In-Context Learning has shown great potential for aligning Large Language Models (LLMs) with human values, helping reduce harmful outputs and accommodate diverse preferences without costly post-training, known as In-Context Alignment (ICA). However, LLMs' comprehension of input prompts remains agnostic, limiting ICA's ability to address value tensions--human values are inherently pluralistic, often imposing conflicting demands, e.g., stimulation vs. tradition. Current ICA methods therefore face the Instruction Bottleneck challenge, where LLMs struggle to reconcile multiple intended values within a single prompt, leading to incomplete or biased alignment. To address this, we propose PICACO, a novel pluralistic ICA method. Without fine-tuning, PICACO optimizes a meta-instruction that incorporates multiple values to better elicit LLMs' understanding of them and improve alignment. This is achieved by maximizing the total correlation between specified values and LLM responses, which theoretically reinforces value conformity and reduces distractive noise, resulting in more effective instructions. Extensive experiments on five value sets show that PICACO works well with both black-box and open-source LLMs, outperforms several recent strong baselines, and achieves a better balance across up to 8 distinct values.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.06999v2">Learning Deliberately, Acting Intuitively: Unlocking Test-Time Reasoning in Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 22 pages, 24 figures
    </div>
    <details class="paper-abstract">
      Reasoning is essential for large language models (LLMs), especially in complex tasks such as mathematical problem solving. However, multimodal reasoning still faces challenges in modality alignment and training scalability, as many existing methods rely on additional annotations or complex rule-based rewards. To address these issues, we propose the Deliberate-to-Intuitive reasoning framework (D2I), which improves the understanding and reasoning abilities of multimodal LLMs (MLLMs) without extra annotations or complex rewards. During training, D2I uses deliberate reasoning strategies supervised only by rule-based format rewards to enhance modality alignment. During inference, it shifts to intuitive reasoning by removing these explicit strategies, allowing the model to implicitly apply the acquired abilities in its responses. D2I outperforms baselines on both in-domain and out-of-domain benchmarks, highlighting the effectiveness of format rewards in fostering transferable multimodal reasoning skills and suggesting the benefit of decoupling training-time reasoning depth from test-time response flexibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28308v1">HELEA: Hard-Negative Benchmark and LLM-based Reranking for Robust Entity Alignment</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 10 pages, 3 figures, 9 tables. Code and benchmarks available at https://github.com/Wnsdnl/HELEA
    </div>
    <details class="paper-abstract">
      Entity Alignment (EA) is essential for knowledge graph (KG) fusion, but existing benchmarks often allow models to exploit name overlap rather than relational structure. This makes it difficult to evaluate whether models can reject same-name entities that refer to different real-world objects. Our primary contribution is a same-name hard-negative augmentation strategy that simultaneously yields quality-controlled evaluation benchmarks (DW-HN29K, DY-HN27K) and augmented training corpora (DW-Train, DY-Train), by mining same-name but distinct entity pairs from KG name-collision groups. We further introduce HELEA, a two-stage framework integrating (i) entity encoder retrieval trained on hard-negative-augmented training corpora with 1-hop KG context, and (ii) LLM-based reranking without additional training. Experiments show that name-dependent baselines collapse to near-random performance on our hard-negative benchmarks, while HELEA achieves F1 0.967 on DW-HN29K while maintaining Hit@1 0.993 on standard DW-15K.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28302v1">How Far Can Disaggregation Go? A Design-Space Exploration of Attention-FFN Disaggregation for Efficient MoE LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-05-27
    </div>
    <details class="paper-abstract">
      Modern large language model (LLM) inference has progressively disaggregated to keep pace with growing model sizes and tight TTFT and TPOT service-level objectives: from chunked-prefill aggregation, to prefill-decode (P/D) disaggregation, and most recently to operator-level Attention-FFN Disaggregation (AFD). This trend is especially important for mixture-of-experts (MoE) models, where memory-bound attention, compute-intensive expert FFNs, and MoE dispatch/combine communication create distinct resource demands. AFD further exposes this heterogeneity by placing attention and MoE-FFN execution on separate GPU groups. Each level of disaggregation deepens the scheduling design space across workload characteristics, resource allocation, and interconnect topology, raising the central question: when does each level actually pay off? We systematically characterize this trade-off for MoE inference across realistic workloads spanning input/output sequence lengths, prefix-KV reuse, and per-user latency constraints. Using chunked-prefill and P/D disaggregation as baselines, we study the benefits and limits of AFD at scale through a framework that fuses on-device kernel measurements with high-fidelity network simulation. Under strict TTFT/TPOT SLOs, AFD sustains around 4k tokens/s of system throughput on DeepSeek-V3.2 across chat, coding, and agentic-coding workloads, where non-AFD deployments are infeasible. We distill concrete takeaways for jointly optimizing throughput and interactivity, including how to partition attention and FFN across GPUs as a function of workload and model architecture, providing design principles for current rack- and cluster-scale deployments as well as future disaggregated AI infrastructure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.07574v2">ViCA: Efficient Multimodal LLMs with Vision-Only Cross-Attention</a></div>
    <div class="paper-meta">
      📅 2026-05-27
    </div>
    <details class="paper-abstract">
      Modern multimodal large language models (MLLMs) adopt a unified self-attention design that processes visual and textual tokens at every Transformer layer, incurring substantial computational overhead. In this work, we revisit the necessity of such dense visual processing and show that projected visual embeddings are already well-aligned with the language space, while effective vision-language interaction occurs in only a small subset of layers. Based on these insights, we propose ViCA (Vision-only Cross-Attention), a minimal MLLM architecture in which visual tokens bypass all self-attention and feed-forward layers, interacting with text solely through sparse cross-attention at selected layers. Extensive evaluations across three MLLM backbones, nine multimodal benchmarks, and 26 pruning-based baselines show that ViCA preserves 98% of baseline accuracy while reducing visual-side computation to 4%, consistently achieving superior performance-efficiency trade-offs. Moreover, ViCA provides a regular, hardware-friendly inference pipeline that yields over 3.5x speedup in single-batch inference and over 10x speedup in multi-batch inference, reducing visual grounding to near-zero overhead compared with text-only LLMs. It is also orthogonal to token pruning methods and can be seamlessly combined for further efficiency gains. Our code is available at https://github.com/EIT-NLP/ViCA.
    </details>
</div>
