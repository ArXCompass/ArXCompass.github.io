# llm - 2026_02

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- Part 5
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.12194v1">MalTool: Malicious Tool Attacks on LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      In a malicious tool attack, an attacker uploads a malicious tool to a distribution platform; once a user installs the tool and the LLM agent selects it during task execution, the tool can compromise the user's security and privacy. Prior work primarily focuses on manipulating tool names and descriptions to increase the likelihood of installation by users and selection by LLM agents. However, a successful attack also requires embedding malicious behaviors in the tool's code implementation, which remains largely unexplored. In this work, we bridge this gap by presenting the first systematic study of malicious tool code implementations. We first propose a taxonomy of malicious tool behaviors based on the confidentiality-integrity-availability triad, tailored to LLM-agent settings. To investigate the severity of the risks posed by attackers exploiting coding LLMs to automatically generate malicious tools, we develop MalTool, a coding-LLM-based framework that synthesizes tools exhibiting specified malicious behaviors, either as standalone tools or embedded within otherwise benign implementations. To ensure functional correctness and structural diversity, MalTool leverages an automated verifier that validates whether generated tools exhibit the intended malicious behaviors and differ sufficiently from prior instances, iteratively refining generations until success. Our evaluation demonstrates that MalTool is highly effective even when coding LLMs are safety-aligned. Using MalTool, we construct two datasets of malicious tools: 1,200 standalone malicious tools and 5,287 real-world tools with embedded malicious behaviors. We further show that existing detection methods, including commercial malware detection approaches such as VirusTotal and methods tailored to the LLM-agent setting, exhibit limited effectiveness at detecting the malicious tools, highlighting an urgent need for new defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.12164v1">Sci-CoE: Co-evolving Scientific Reasoning LLMs via Geometric Consensus with Sparse Supervision</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated exceptional reasoning capabilities, and co-evolving paradigms have shown promising results in domains such as code and math. However, in scientific reasoning tasks, these models remain fragile due to unreliable solution evaluation and limited diversity in verification strategies. In this work, we propose Sci-CoE, a two-stage scientific co-evolving framework that enables models to self-evolve as both solver and verifier through a transition from sparse supervision to unsupervised learning. In the first stage, the model uses a small set of annotated data to establish fundamental correctness judgment anchors for the Verifier. In the second stage, we introduce a geometric reward mechanism that jointly considers consensus, reliability, and diversity, driving large-scale self-iteration on unlabeled data. Experiments on several general scientific benchmarks demonstrate that Sci-CoE enhances complex reasoning capabilities and exhibits strong scalability, facilitating the construction of more robust and diverse evaluation systems. Codes are available at https://github.com/InternScience/Sci-CoE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.12151v1">OServe: Accelerating LLM Serving via Spatial-Temporal Workload Orchestration</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      Serving Large Language Models (LLMs) can benefit immensely from parallelizing both the model and input requests across multiple devices, but incoming workloads exhibit substantial spatial and temporal heterogeneity. Spatially, workloads comprise heterogeneous requests with varying compute and memory demands. Temporally, workload composition varies over time. Nevertheless, existing systems typically assume spatially uniform and temporally stable workloads, employing a homogeneous, static model deployment. This mismatch between the assumption and real-world spatial-temporal heterogeneity results in suboptimal performance. We present OServe, an LLM serving system with heterogeneous and flexible model deployment that addresses both spatial and temporal heterogeneity. First, OServe introduces a novel workload-aware scheduling algorithm that optimizes heterogeneous model deployments according to real-time workload characteristics. Second, OServe proposes an efficient workload-adaptive switching method that migrates model deployments in response to predicted workload changes. Experiments on real-world traces show that OServe improves performance by up to 2$\times$ (average: 1.5$\times$) compared to state-of-the-art serving systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.12134v1">Value Alignment Tax: Measuring Value Trade-offs in LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 Preprint. Under review. 20 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Existing work on value alignment typically characterizes value relations statically, ignoring how interventions - such as prompting, fine-tuning, or preference optimization - reshape the broader value system. We introduce the Value Alignment Tax (VAT), a framework that measures how alignment-induced changes propagate across interconnected values relative to achieved on-target gain. VAT captures the dynamics of value expression under alignment pressure. Using a controlled scenario-action dataset grounded in Schwartz value theory, we collect paired pre-post normative judgments and analyze alignment effects across models, values, and alignment strategies. Our results show that alignment often produces uneven, structured co-movement among values. These effects are invisible under conventional target-only evaluation, revealing systemic, process-level alignment risks and offering new insights into the dynamics of value alignment in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.07078v5">Can LLM-based Financial Investing Strategies Outperform the Market in Long Run?</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 KDD 2026, Datasets & Benchmarks Track
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently been leveraged for asset pricing tasks and stock trading applications, enabling AI agents to generate investment decisions from unstructured financial data. However, most evaluations of LLM timing-based investing strategies are conducted on narrow timeframes and limited stock universes, overstating effectiveness due to survivorship and data-snooping biases. We critically assess their generalizability and robustness by proposing FINSABER, a backtesting framework evaluating timing-based strategies across longer periods and a larger universe of symbols. Systematic backtests over two decades and 100+ symbols reveal that previously reported LLM advantages deteriorate significantly under broader cross-section and over a longer-term evaluation. Our market regime analysis further demonstrates that LLM strategies are overly conservative in bull markets, underperforming passive benchmarks, and overly aggressive in bear markets, incurring heavy losses. These findings highlight the need to develop LLM strategies that are able to prioritise trend detection and regime-aware risk controls over mere scaling of framework complexity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.18382v4">One Demo Is All It Takes: Planning Domain Derivation with LLMs from A Single Demonstration</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 Published as a conference paper at ICLR 2026
    </div>
    <details class="paper-abstract">
      Pre-trained large language models (LLMs) show promise for robotic task planning but often struggle to guarantee correctness in long-horizon problems. Task and motion planning (TAMP) addresses this by grounding symbolic plans in low-level execution, yet it relies heavily on manually engineered planning domains. To improve long-horizon planning reliability and reduce human intervention, we present Planning Domain Derivation with LLMs (PDDLLM), a framework that automatically induces symbolic predicates and actions directly from demonstration trajectories by combining LLM reasoning with physical simulation roll-outs. Unlike prior domain-inference methods that rely on partially predefined or language descriptions of planning domains, PDDLLM constructs domains without manual domain initialization and automatically integrates them with motion planners to produce executable plans, enhancing long-horizon planning automation. Across 1,200 tasks in nine environments, PDDLLM outperforms six LLM-based planning baselines, achieving at least 20\% higher success rates, reduced token costs, and successful deployment on multiple physical robot platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.15414v3">MARSHAL: Incentivizing Multi-Agent Reasoning via Self-Play with Strategic LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      Developing Large Language Models (LLMs) to cooperate and compete effectively within multi-agent systems (MASs) is a critical step towards more advanced intelligence. While reinforcement learning (RL) has proven effective for enhancing reasoning in single-agent tasks, its extension to multi-turn, multi-agent scenarios remains underexplored due to the challenges of long-horizon credit assignment and agent-specific advantage estimation. To address these challenges, we introduce MARSHAL, an end-to-end RL framework that incentivizes Multi-Agent Reasoning through Self-play witH strAtegic LLMs in both cooperative and competitive games. MARSHAL features a turn-level advantage estimator that aligns learning signals with each interaction for credit assignment, and an agent-specific advantage normalization to stabilize multi-agent training. By learning with self-play across cooperative and competitive games, MARSHAL agents trained from Qwen3-4B develop strong strategic abilities, with up to 28.7% performance improvements in held-out games. More importantly, the capability acquired through self-play generalizes beyond games, yielding consistent performance gains of MASs in reasoning benchmarks. When integrated into leading MASs, our MARSHAL agent achieves significant zero-shot performance gains of up to 10.0% on AIME, 7.6% on GPQA-Diamond, and 3.5% on average across all benchmarks. These results establish self-play in strategic games as a powerful approach for developing generalizable multi-agent reasoning capabilities in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.12424v3">EvoGPT: Leveraging LLM-Driven Seed Diversity to Improve Search-Based Test Suite Generation</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      Search-Based Software Testing (SBST) is a well-established approach for automated unit test generation, yet it often suffers from premature convergence and limited diversity in the generated test suites. Recently, Large Language Models (LLMs) have emerged as an alternative technique for unit test generation. We present EvoGPT, a hybrid test generation system that integrates LLM-based test generation with SBST-based test suite optimization. EvoGPT uses LLMs to generate an initial population of test suites, and uses an Evolutionary Algorithm (EA) to further optimize this test suite population. A distinguishing feature of EvoGPT is its explicit enforcement of diversity, achieved through the use of multiple temperatures and prompt instructions during test generation. In addition, each LLM-generated test is refined using a generation-repair loop and coverage-guided assertion generation. To address evolutionary plateaus, EvoGPT also detects stagnation during search and injects additional LLM-generated tests aimed at previously uncovered branches. Here too diversity is enforced using multiple temperatures and prompt instructions. We evaluate EvoGPT on Defects4J, a standard benchmark for test generation. The results show that EvoGPT achieves, on average, a 10% improvement in both code coverage and mutation score metrics compared to TestART, an LLM-only baseline; and EvoSuite, a standard SBST baseline. An ablation study indicates that explicitly enforcing diversity both at initialization and during the search is key to effectively leveraging LLMs for automated unit test generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.12049v1">Improving HPC Code Generation Capability of LLMs via Online Reinforcement Learning with Real-Machine Benchmark Rewards</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated strong code generation capabilities, yet the runtime performance of generated code is not guaranteed, and there have been few attempts to train LLMs using runtime performance as a reward in the HPC domain. We propose an online reinforcement learning approach that executes LLM-generated code on a supercomputer and directly feeds back the measured runtime performance (GFLOPS) as a reward. We further introduce a Staged Quality-Diversity (SQD) algorithm that progressively varies the permitted optimization techniques on a per-problem basis, enabling the model to learn code optimization from diverse perspectives. We build a distributed system connecting a GPU training cluster with a CPU benchmarking cluster, and train Qwen2.5 Coder 14B on a double-precision matrix multiplication task using Group Relative Policy Optimization (GRPO). Through two experiments, we show that reinforcement learning combining runtime performance feedback with staged optimization can improve the HPC code generation capability of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.12029v1">PrefillShare: A Shared Prefill Module for KV Reuse in Multi-LLM Disaggregated Serving</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 Preprint. 13 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Multi-agent systems increasingly orchestrate multiple specialized language models to solve complex real-world problems, often invoking them over a shared context. This execution pattern repeatedly processes the same prompt prefix across models. Consequently, each model redundantly executes the prefill stage and maintains its own key-value (KV) cache, increasing aggregate prefill load and worsening tail latency by intensifying prefill-decode interference in existing LLM serving stacks. Disaggregated serving reduces such interference by placing prefill and decode on separate GPUs, but disaggregation does not fundamentally eliminate inter-model redundancy in computation and KV storage for the same prompt. To address this issue, we propose PrefillShare, a novel algorithm that enables sharing the prefill stage across multiple models in a disaggregated setting. PrefillShare factorizes the model into prefill and decode modules, freezes the prefill module, and fine-tunes only the decode module. This design allows multiple task-specific models to share a prefill module and the KV cache generated for the same prompt. We further introduce a routing mechanism that enables effective prefill sharing across heterogeneous models in a vLLM-based disaggregated system. PrefillShare not only matches full fine-tuning accuracy on a broad range of tasks and models, but also delivers 4.5x lower p95 latency and 3.9x higher throughput in multi-model agent workloads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.00031v3">VibeCodeHPC: An Agent-Based Iterative Prompting Auto-Tuner for HPC Code Generation Using LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      In this study, we propose VibeCodeHPC, a multi-agent system based on large language models (LLMs) for the automatic tuning of high-performance computing (HPC) programs on supercomputers. VibeCodeHPC adopts Claude Code as its backend and provides an integrated environment that facilitates program development in supercomputer settings. The system not only brings the Vibe Coding paradigm -- program development through natural language interaction with users -- to HPC programming, but also enables autonomous performance optimization with minimal user intervention through a sophisticated multi-agent design. To achieve these objectives, VibeCodeHPC implements three core functionalities: (1) configuration capabilities tailored to the unique development environments of supercomputers, (2) collaborative operation among multiple LLM agents with distinct roles -- Project Manager (PM), System Engineer (SE), Programmer (PG), and Continuous Deliverer (CD), and (3) long-term autonomous operation through agent activity monitoring and dynamic deployment mechanisms. This paper highlights one of the most powerful features of VibeCodeHPC: fully automated code optimization through autonomous operation without user intervention. Specifically, it demonstrates the performance optimization of CPU-based codes on GPU-equipped systems for matrix multiplication and a Poisson equation solver using Jacobi's iterative method. The results show that the multi-agent configuration employed in VibeCodeHPC enables faster and more reliable development of higher-performance code compared to a single-agent setup.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.12022v5">Teaching LLMs According to Their Aptitude: Adaptive Reasoning for Mathematical Problem Solving</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      Existing approaches to mathematical reasoning with large language models (LLMs) rely on Chain-of-Thought (CoT) for generalizability or Tool-Integrated Reasoning (TIR) for precise computation. While efforts have been made to combine these methods, they primarily rely on post-selection or predefined strategies, leaving an open question: whether LLMs can autonomously adapt their reasoning strategy based on their inherent capabilities. In this work, we propose TATA (Teaching LLMs According to Their Aptitude), an adaptive framework that enables LLMs to personalize their reasoning strategy spontaneously, aligning it with their intrinsic aptitude. TATA incorporates base-LLM-aware data selection during supervised fine-tuning (SFT) to tailor training data to the model's unique abilities. This approach equips LLMs to autonomously determine and apply the appropriate reasoning strategy at test time. We evaluate TATA through extensive experiments on six mathematical reasoning benchmarks, using both general-purpose and math-specialized LLMs. Empirical results demonstrate that TATA effectively combines the complementary strengths of CoT and TIR, achieving superior or comparable performance with improved inference efficiency compared to TIR alone. Further analysis underscores the critical role of aptitude-aware data selection in enabling LLMs to make effective and adaptive reasoning decisions and align reasoning strategies with model capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11962v1">Wisdom of the LLM Crowd: A Large Scale Benchmark of Multi-Label U.S. Election-Related Harmful Social Media Content</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      The spread of election misinformation and harmful political content conveys misleading narratives and poses a serious threat to democratic integrity. Detecting harmful content at early stages is essential for understanding and potentially mitigating its downstream spread. In this study, we introduce USE24-XD, a large-scale dataset of nearly 100k posts collected from X (formerly Twitter) during the 2024 U.S. presidential election cycle, enriched with spatio-temporal metadata. To substantially reduce the cost of manual annotation while enabling scalable categorization, we employ six large language models (LLMs) to systematically annotate posts across five nuanced categories: Conspiracy, Sensationalism, Hate Speech, Speculation, and Satire. We validate LLM annotations with crowdsourcing (n = 34) and benchmark them against human annotators. Inter-rater reliability analyses show comparable agreement patterns between LLMs and humans, with LLMs exhibiting higher internal consistency and achieving up to 0.90 recall on Speculation. We apply a wisdom-of-the-crowd approach across LLMs to aggregate annotations and curate a robust multi-label dataset. 60% of posts receive at least one label. We further analyze how human annotator demographics, including political ideology and affiliation, shape labeling behavior, highlighting systematic sources of subjectivity in judgments of harmful content. The USE24-XD dataset is publicly released to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11957v1">Are Two LLMs Better Than One? A Student-Teacher Dual-Head LLMs Architecture for Pharmaceutical Content Optimization</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 Submitted to the Demo Track of Top Tier Conference; currently under peer review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to create content in regulated domains such as pharmaceuticals, where outputs must be scientifically accurate and legally compliant. Manual quality control (QC) is slow, error prone, and can become a publication bottleneck. We introduce LRBTC, a modular LLM and vision language model (VLM) driven QC architecture covering Language, Regulatory, Brand, Technical, and Content Structure checks. LRBTC combines a Student-Teacher dual model architecture, human in the loop (HITL) workflow with waterfall rule filtering to enable scalable, verifiable content validation and optimization. On AIReg-Bench, our approach achieves 83.0% F1 and 97.5% recall, reducing missed violations by 5x compared with Gemini 2.5 Pro. On CSpelling, it improves mean accuracy by 26.7%. Error analysis further reveals that while current models are strong at detecting misspellings (92.5 recall), they fail to identify complex medical grammatical (25.0 recall) and punctuation (41.7 recall) errors, highlighting a key area for future work. This work provides a practical, plug and play solution for reliable, transparent quality control of content in high stakes, compliance critical industries. We also provide access to our Demo under MIT Licenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11924v1">Who Does What? Archetypes of Roles Assigned to LLMs During Human-AI Decision-Making</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 Accepted to ACM CHI 2026
    </div>
    <details class="paper-abstract">
      LLMs are increasingly supporting decision-making across high-stakes domains, requiring critical reflection on the socio-technical factors that shape how humans and LLMs are assigned roles and interact during human-in-the-loop decision-making. This paper introduces the concept of human-LLM archetypes -- defined as re-curring socio-technical interaction patterns that structure the roles of humans and LLMs in collaborative decision-making. We describe 17 human-LLM archetypes derived from a scoping literature review and thematic analysis of 113 LLM-supported decision-making papers. Then, we evaluate these diverse archetypes across real-world clinical diagnostic cases to examine the potential effects of adopting distinct human-LLM archetypes on LLM outputs and decision outcomes. Finally, we present relevant tradeoffs and design choices across human-LLM archetypes, including decision control, social hierarchies, cognitive forcing strategies, and information requirements. Through our analysis, we show that selection of human-LLM interaction archetype can influence LLM outputs and decisions, bringing important risks and considerations for the designers of human-AI decision-making systems
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11908v1">When Should LLMs Be Less Specific? Selective Abstraction for Reliable Long-Form Text Generation</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      LLMs are widely used, yet they remain prone to factual errors that erode user trust and limit adoption in high-risk settings. One approach to mitigate this risk is to equip models with uncertainty estimation mechanisms that abstain when confidence is low. However, this binary "all-or-nothing" approach is excessively restrictive in long-form settings, often discarding valuable information. We introduce Selective Abstraction (SA), a framework that enables LLMs to trade specificity for reliability by selectively reducing the detail of uncertain content. We first formalize SA through the lenses of selective risk and coverage. We then propose Atom-wise Selective Abstraction, a claim-level instantiation that decomposes responses into atomic claims (short, self-contained statements each expressing a single fact) and replaces uncertain atoms with higher confidence, less specific abstractions. To evaluate this framework, we develop a novel end-to-end pipeline for open-ended generation that instantiates risk as factual correctness and measures coverage using an information-theoretic measure of retained information. Across six open-source models on the FactScore and LongFact-Objects benchmarks, atom-wise SA consistently outperforms existing baselines, improving the area under the risk-coverage curve (AURC) by up to 27.73% over claim removal, demonstrating that reducing specificity can boost accuracy and reliability while preserving most of their original meaning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11904v1">Leveraging LLMs to support co-evolution between definitions and instances of textual DSLs: A Systematic Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      Software languages evolve over time for reasons such as feature additions. When grammars evolve, textual instances that originally conformed to them may become outdated. While model-driven engineering provides many techniques for co-evolving models with metamodel changes, these approaches are not designed for textual DSLs and may lose human-relevant information such as layout and comments. This study systematically evaluates the potential of large language models (LLMs) for co-evolving grammars and instances of textual DSLs. Using Claude Sonnet 4.5 and GPT-5.2 across ten case languages with ten runs each, we assess both correctness and preservation of human-oriented information. Results show strong performance on small-scale cases ($\geq$94% precision and recall for instances requiring fewer than 20 modified lines), but performance degraded with scale: Claude maintains 85% recall at 40 lines, while GPT fails on the largest instances. Response time increases substantially with instance size, and grammar evolution complexity and deletion granularity affect performance more than change type. These findings clarify when LLM-based co-evolution is effective and where current limitations remain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11898v1">Benchmark Illusion: Disagreement among LLMs and Its Scientific Consequences</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      Benchmarks underpin how progress in large language models (LLMs) is measured and trusted. Yet our analyses reveal that apparent convergence in benchmark accuracy can conceal deep epistemic divergence. Using two major reasoning benchmarks - MMLU-Pro and GPQA - we show that LLMs achieving comparable accuracy still disagree on 16-66% of items, and 16-38% among top-performing frontier models. These discrepancies suggest distinct error profiles for different LLMs. When such models are used for scientific data annotation and inference, their hidden disagreements propagate into research results: in re-analyses of published studies in education and political science, switching the annotation model can change estimated treatment effects by more than 80%, and in some cases reverses their sign. Together, these findings illustrate a benchmark illusion, where equal accuracy may conceal disagreement, with model choice becoming a hidden yet consequential variable for scientific reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16206v2">LLM-in-Sandbox Elicits General Agentic Intelligence</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 Project Page: https://llm-in-sandbox.github.io
    </div>
    <details class="paper-abstract">
      We introduce LLM-in-Sandbox, enabling LLMs to explore within a code sandbox (i.e., a virtual computer), to elicit general intelligence in non-code domains. We first demonstrate that strong LLMs, without additional training, exhibit generalization capabilities to leverage the code sandbox for non-code tasks. For example, LLMs spontaneously access external resources to acquire new knowledge, leverage the file system to handle long contexts, and execute scripts to satisfy formatting requirements. We further show that these agentic capabilities can be enhanced through LLM-in-Sandbox Reinforcement Learning (LLM-in-Sandbox-RL), which uses only non-agentic data to train models for sandbox exploration. Experiments demonstrate that LLM-in-Sandbox, in both training-free and post-trained settings, achieves robust generalization spanning mathematics, physics, chemistry, biomedicine, long-context understanding, and instruction following. Finally, we analyze LLM-in-Sandbox's efficiency from computational and system perspectives, and open-source it as a Python package to facilitate real-world deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11886v1">LLM-based Triplet Extraction from Financial Reports</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      Corporate financial reports are a valuable source of structured knowledge for Knowledge Graph construction, but the lack of annotated ground truth in this domain makes evaluation difficult. We present a semi-automated pipeline for Subject-Predicate-Object triplet extraction that uses ontology-driven proxy metrics, specifically Ontology Conformance and Faithfulness, instead of ground-truth-based evaluation. We compare a static, manually engineered ontology against a fully automated, document-specific ontology induction approach across different LLMs and two corporate annual reports. The automatically induced ontology achieves 100% schema conformance in all configurations, eliminating the ontology drift observed with the manual approach. We also propose a hybrid verification strategy that combines regex matching with an LLM-as-a-judge check, reducing apparent subject hallucination rates from 65.2% to 1.6% by filtering false positives caused by coreference resolution. Finally, we identify a systematic asymmetry between subject and object hallucinations, which we attribute to passive constructions and omitted agents in financial prose.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12138v2">DriveSafe: A Hierarchical Risk Taxonomy for Safety-Critical LLM-Based Driving Assistants</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 The authors are withdrawing this manuscript due to substantial revisions currently underway. A significantly updated version will be submitted in the future
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrated into vehicle-based digital assistants, where unsafe, ambiguous, or legally incorrect responses can lead to serious safety, ethical, and regulatory consequences. Despite growing interest in LLM safety, existing taxonomies and evaluation frameworks remain largely general-purpose and fail to capture the domain-specific risks inherent to real-world driving scenarios. In this paper, we introduce DriveSafe, a hierarchical, four-level risk taxonomy designed to systematically characterize safety-critical failure modes of LLM-based driving assistants. The taxonomy comprises 129 fine-grained atomic risk categories spanning technical, legal, societal, and ethical dimensions, grounded in real-world driving regulations and safety principles and reviewed by domain experts. To validate the safety relevance and realism of the constructed prompts, we evaluate their refusal behavior across six widely deployed LLMs. Our analysis shows that the evaluated models often fail to appropriately refuse unsafe or non-compliant driving-related queries, underscoring the limitations of general-purpose safety alignment in driving contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.22126v2">EasyUUV: An LLM-Enhanced Universal and Lightweight Sim-to-Real Reinforcement Learning Framework for UUV Attitude Control</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 10 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Despite recent advances in Unmanned Underwater Vehicle (UUV) attitude control, existing methods still struggle with generalizability, robustness to real-world disturbances, and efficient deployment. To address the above challenges, this paper presents EasyUUV, a Large Language Model (LLM)-enhanced, universal, and lightweight simulation-to-reality reinforcement learning (RL) framework for robust attitude control of UUVs. EasyUUV combines parallelized RL training with a hybrid control architecture, where a learned policy outputs high-level attitude corrections executed by an adaptive S-Surface controller. A multimodal LLM is further integrated to adaptively tune controller parameters at runtime using visual and textual feedback, enabling training-free adaptation to unmodeled dynamics. Also, we have developed a low-cost 6-DoF UUV platform and applied an RL policy trained through efficient parallelized simulation. Extensive simulation and real-world experiments validate the effectiveness and outstanding performance of EasyUUV in achieving robust and adaptive UUV attitude control across diverse underwater conditions. To facilitate reproducibility and further research, the source code, LLM prompts, and supplementary video are provided in the following repositories: Homepage: https://360zmem.github.io/easyuuv/ Video:https://youtu.be/m2yLQzxiIL
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11877v1">Towards Fair and Comprehensive Evaluation of Routers in Collaborative LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 Our code is publicly available at https://github.com/zhuchichi56/RouterXBench
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved success, but cost and privacy constraints necessitate deploying smaller models locally while offloading complex queries to cloud-based models. Existing router evaluations are unsystematic, overlooking scenario-specific requirements and out-of-distribution robustness. We propose RouterXBench, a principled evaluation framework with three dimensions: router ability, scenario alignment, and cross-domain robustness. Unlike prior work that relies on output probabilities or external embeddings, we utilize internal hidden states that capture model uncertainty before answer generation. We introduce ProbeDirichlet, a lightweight router that aggregates cross-layer hidden states via learnable Dirichlet distributions with probabilistic training. Trained on multi-domain data, it generalizes robustly across in-domain and out-of-distribution scenarios. Our results show ProbeDirichlet achieves 16.68% and 18.86% relative improvements over the best baselines in router ability and high-accuracy scenarios, with consistent performance across model families, model scales, heterogeneous tasks, and agentic workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11812v1">Predicting LLM Output Length via Entropy-Guided Representations</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      The long-tailed distribution of sequence lengths in LLM serving and reinforcement learning (RL) sampling causes significant computational waste due to excessive padding in batched inference. Existing methods rely on auxiliary models for static length prediction, but they incur high overhead, generalize poorly, and fail in stochastic "one-to-many" sampling scenarios. We introduce a lightweight framework that reuses the main model's internal hidden states for efficient length prediction. Our framework features two core components: 1) Entropy-Guided Token Pooling (EGTP), which uses on-the-fly activations and token entropy for highly accurate static prediction with negligible cost, and 2) Progressive Length Prediction (PLP), which dynamically estimates the remaining length at each decoding step to handle stochastic generation. To validate our approach, we build and release ForeLen, a comprehensive benchmark with long-sequence, Chain-of-Thought, and RL data. On ForeLen, EGTP achieves state-of-the-art accuracy, reducing MAE by 29.16\% over the best baseline. Integrating our methods with a length-aware scheduler yields significant end-to-end throughput gains. Our work provides a new technical and evaluation baseline for efficient LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11790v1">Beyond End-to-End Video Models: An LLM-Based Multi-Agent System for Educational Video Generation</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 For more information, visit the project website: https://robitsg.github.io/LASEV/
    </div>
    <details class="paper-abstract">
      Although recent end-to-end video generation models demonstrate impressive performance in visually oriented content creation, they remain limited in scenarios that require strict logical rigor and precise knowledge representation, such as instructional and educational media. To address this problem, we propose LAVES, a hierarchical LLM-based multi-agent system for generating high-quality instructional videos from educational problems. The LAVES formulates educational video generation as a multi-objective task that simultaneously demands correct step-by-step reasoning, pedagogically coherent narration, semantically faithful visual demonstrations, and precise audio--visual alignment. To address the limitations of prior approaches--including low procedural fidelity, high production cost, and limited controllability--LAVES decomposes the generation workflow into specialized agents coordinated by a central Orchestrating Agent with explicit quality gates and iterative critique mechanisms. Specifically, the Orchestrating Agent supervises a Solution Agent for rigorous problem solving, an Illustration Agent that produces executable visualization codes, and a Narration Agent for learner-oriented instructional scripts. In addition, all outputs from the working agents are subject to semantic critique, rule-based constraints, and tool-based compilation checks. Rather than directly synthesizing pixels, the system constructs a structured executable video script that is deterministically compiled into synchronized visuals and narration using template-driven assembly rules, enabling fully automated end-to-end production without manual editing. In large-scale deployments, LAVES achieves a throughput exceeding one million videos per day, delivering over a 95% reduction in cost compared to current industry-standard approaches while maintaining a high acceptance rate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11786v1">Evaluating LLM Safety Under Repeated Inference via Accelerated Prompt Stress Testing</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 24 pages, 9 figures. Submitted to TMLR
    </div>
    <details class="paper-abstract">
      Traditional benchmarks for large language models (LLMs) primarily assess safety risk through breadth-oriented evaluation across diverse tasks. However, real-world deployment exposes a different class of risk: operational failures arising from repeated inference on identical or near-identical prompts rather than broad task generalization. In high-stakes settings, response consistency and safety under sustained use are critical. We introduce Accelerated Prompt Stress Testing (APST), a depth-oriented evaluation framework inspired by reliability engineering. APST repeatedly samples identical prompts under controlled operational conditions (e.g., decoding temperature) to surface latent failure modes including hallucinations, refusal inconsistency, and unsafe completions. Rather than treating failures as isolated events, APST models them as stochastic outcomes of independent inference events. We formalize safety failures using Bernoulli and binomial models to estimate per-inference failure probabilities, enabling quantitative comparison of reliability across models and decoding configurations. Applying APST to multiple instruction-tuned LLMs evaluated on AIR-BENCH-derived safety prompts, we find that models with similar benchmark-aligned scores can exhibit substantially different empirical failure rates under repeated sampling, particularly as temperature increases. These results demonstrate that shallow, single-sample evaluation can obscure meaningful reliability differences under sustained use. APST complements existing benchmarks by providing a practical framework for evaluating LLM safety and reliability under repeated inference, bridging benchmark alignment and deployment-oriented risk assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11782v1">FlowMind: Execute-Summarize for Structured Workflow Generation from LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      LLMs can solve complex tasks through reasoning and tool use, but accurately translating these solutions into structured workflows remains challenging. We model workflows as sequences of tool use and reformulate the problem as designing a mechanism that can both solve tasks and reliably construct workflows. Prior approaches that build workflows during execution often suffer from inaccuracies due to interference between the two processes. We propose an Execute-Summarize(ES) framework that decouples task execution from workflow construction: the model first completes the task using available tools, then independently reconstructs a structured workflow from execution traces. This separation improves workflow accuracy and robustness. We introduce FlowBench and show through extensive experiments that our approach outperforms existing methods, providing a reliable paradigm for grounding free-form LLM reasoning into structured workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.03771v2">Trustworthiness of Legal Considerations for the Use of LLMs in Education</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 11 pages, 3 figures, 6 tables
    </div>
    <details class="paper-abstract">
      As Artificial Intelligence (AI), particularly Large Language Models (LLMs), becomes increasingly embedded in education systems worldwide, ensuring their ethical, legal, and contextually appropriate deployment has become a critical policy concern. This paper offers a comparative analysis of AI-related regulatory and ethical frameworks across key global regions, including the European Union, United Kingdom, United States, China, and Gulf Cooperation Council (GCC) countries. It maps how core trustworthiness principles, such as transparency, fairness, accountability, data privacy, and human oversight are embedded in regional legislation and AI governance structures. Special emphasis is placed on the evolving landscape in the GCC, where countries are rapidly advancing national AI strategies and education-sector innovation. To support this development, the paper introduces a Compliance-Centered AI Governance Framework tailored to the GCC context. This includes a tiered typology and institutional checklist designed to help regulators, educators, and developers align AI adoption with both international norms and local values. By synthesizing global best practices with region-specific challenges, the paper contributes practical guidance for building legally sound, ethically grounded, and culturally sensitive AI systems in education. These insights are intended to inform future regulatory harmonization and promote responsible AI integration across diverse educational environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11779v1">Temperature as a Meta-Policy: Adaptive Temperature in LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 Accepted at ICLR 2026. 10 pages (main text) + supplementary material, 6 figures
    </div>
    <details class="paper-abstract">
      Temperature is a crucial hyperparameter in large language models (LLMs), controlling the trade-off between exploration and exploitation during text generation. High temperatures encourage diverse but noisy outputs, while low temperatures produce focused outputs but may cause premature convergence. Yet static or heuristic temperature schedules fail to adapt to the dynamic demands of reinforcement learning (RL) throughout training, often limiting policy improvement. We propose Temperature Adaptive Meta Policy Optimization (TAMPO), a new framework that recasts temperature control as a learnable meta-policy. TAMPO operates through a hierarchical two-loop process. In the inner loop, the LLM policy is updated (e.g., using GRPO) with trajectories sampled at the temperature selected by the meta-policy. In the outer loop, meta-policy updates the distribution over candidate temperatures by rewarding those that maximize the likelihood of high-advantage trajectories. This trajectory-guided, reward-driven mechanism enables online adaptation without additional rollouts, directly aligning exploration with policy improvement. On five mathematical reasoning benchmarks, TAMPO outperforms baselines using fixed or heuristic temperatures, establishing temperature as an effective learnable meta-policy for adaptive exploration in LLM reinforcement learning. Accepted at ICLR 2026.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.05703v2">Provably Convergent Primal-Dual DPO for Constrained LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      The widespread application of large language models (LLMs) raises increasing demands on ensuring safety or imposing constraints, such as reducing harmful content and adhering to predefined rules. While there have been several works studying LLM safety alignment, these works either need to train three models and incur high memory costs, or require prior knowledge on the optimal solution. Witnessing this fact, we investigate the constrained alignment problem for LLMs, i.e., maximizing the reward of outputs while restricting the cost to stay below a threshold. We propose a novel primal-dual direct preference optimization (DPO) approach, which first trains a model using standard DPO on reward preference data to provide reward information, and then adopts a rearranged Lagrangian DPO objective utilizing the provided reward information to fine-tune LLMs. Our approach only needs to train two models rather than three, which significantly saves memory costs, and does not require extra prior knowledge. Moreover, we establish rigorous suboptimality and constraint violation guarantees. We also extend our approach to enable online exploration and drop the data coverage dependence in the results. Experiments on the PKU-SafeRLHF and TruthfulQA datasets demonstrate the state-of-the-art performance of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11767v1">TSR: Trajectory-Search Rollouts for Multi-Turn RL of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      Advances in large language models (LLMs) are driving a shift toward using reinforcement learning (RL) to train agents from iterative, multi-turn interactions across tasks. However, multi-turn RL remains challenging as rewards are often sparse or delayed, and environments can be stochastic. In this regime, naive trajectory sampling can hinder exploitation and induce mode collapse. We propose TSR (Trajectory-Search Rollouts), a training-time approach that repurposes test-time scaling ideas for improved per-turn rollout generation. TSR performs lightweight tree-style search to construct high-quality trajectories by selecting high-scoring actions at each turn using task-specific feedback. This improves rollout quality and stabilizes learning while leaving the underlying optimization objective unchanged, making TSR optimizer-agnostic. We instantiate TSR with best-of-N, beam, and shallow lookahead search, and pair it with PPO and GRPO, achieving up to 15% performance gains and more stable learning on Sokoban, FrozenLake, and WebShop tasks at a one-time increase in training compute. By moving search from inference time to the rollout stage of training, TSR provides a simple and general mechanism for stronger multi-turn agent learning, complementary to existing frameworks and rejection-sampling-style selection methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.03346v2">KVComm: Enabling Efficient LLM Communication through Selective KV Sharing</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in multi-agent systems, where effective inter-model communication is crucial. Existing communication protocols either rely on natural language, incurring high inference costs and information loss, or on hidden states, which suffer from information concentration bias and inefficiency. To address these limitations, we propose KVComm, a novel communication framework that enables efficient communication between LLMs through selective sharing of KV pairs. KVComm leverages the rich information encoded in the KV pairs while avoiding the pitfalls of hidden states. We introduce a KV layer-wise selection strategy based on attention importance scores with a Gaussian prior to identify the most informative KV pairs for communication. Extensive experiments across diverse tasks and model pairs demonstrate that KVComm achieves comparable performance to the upper-bound method, which directly merges inputs to one model without any communication, while transmitting as few as 30\% of layers' KV pairs. Our study highlights the potential of KV pairs as an effective medium for inter-LLM communication, paving the way for scalable and efficient multi-agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11757v1">Code2Worlds: Empowering Coding LLMs for 4D World Generation</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      Achieving spatial intelligence requires moving beyond visual plausibility to build world simulators grounded in physical laws. While coding LLMs have advanced static 3D scene generation, extending this paradigm to 4D dynamics remains a critical frontier. This task presents two fundamental challenges: multi-scale context entanglement, where monolithic generation fails to balance local object structures with global environmental layouts; and a semantic-physical execution gap, where open-loop code generation leads to physical hallucinations lacking dynamic fidelity. We introduce Code2Worlds, a framework that formulates 4D generation as language-to-simulation code generation. First, we propose a dual-stream architecture that disentangles retrieval-augmented object generation from hierarchical environmental orchestration. Second, to ensure dynamic fidelity, we establish a physics-aware closed-loop mechanism in which a PostProcess Agent scripts dynamics, coupled with a VLM-Motion Critic that performs self-reflection to iteratively refine simulation code. Evaluations on the Code4D benchmark show Code2Worlds outperforms baselines with a 41% SGS gain and 49% higher Richness, while uniquely generating physics-aware dynamics absent in prior static methods. Code: https://github.com/AIGeeksGroup/Code2Worlds. Website: https://aigeeksgroup.github.io/Code2Worlds.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11754v1">Cooperation Breakdown in LLM Agents Under Communication Delays</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      LLM-based multi-agent systems (LLM-MAS), in which autonomous AI agents cooperate to solve tasks, are gaining increasing attention. For such systems to be deployed in society, agents must be able to establish cooperation and coordination under real-world computational and communication constraints. We propose the FLCOA framework (Five Layers for Cooperation/Coordination among Autonomous Agents) to conceptualize how cooperation and coordination emerge in groups of autonomous agents, and highlight that the influence of lower-layer factors - especially computational and communication resources - has been largely overlooked. To examine the effect of communication delay, we introduce a Continuous Prisoner's Dilemma with Communication Delay and conduct simulations with LLM-based agents. As delay increases, agents begin to exploit slower responses even without explicit instructions. Interestingly, excessive delay reduces cycles of exploitation, yielding a U-shaped relationship between delay magnitude and mutual cooperation. These results suggest that fostering cooperation requires attention not only to high-level institutional design but also to lower-layer factors such as communication delay and resource allocation, pointing to new directions for MAS research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.07294v2">Fin-RATE: A Real-world Financial Analytics and Tracking Evaluation Benchmark for LLMs on SEC Filings</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      With the increasing deployment of Large Language Models (LLMs) in the finance domain, LLMs are increasingly expected to parse complex regulatory disclosures. However, existing benchmarks often focus on isolated details, failing to reflect the complexity of professional analysis that requires synthesizing information across multiple documents, reporting periods, and corporate entities. Furthermore, these benchmarks do not disentangle whether errors arise from retrieval failures, generation inaccuracies, domain-specific reasoning mistakes, or misinterpretation of the query or context, making it difficult to precisely diagnose performance bottlenecks. To bridge these gaps, we introduce Fin-RATE, a benchmark built on U.S. Securities and Exchange Commission (SEC) filings and mirroring financial analyst workflows through three pathways: detail-oriented reasoning within individual disclosures, cross-entity comparison under shared topics, and longitudinal tracking of the same firm across reporting periods. We benchmark 17 leading LLMs, spanning open-source, closed-source, and finance-specialized models, under both ground-truth context and retrieval-augmented settings. Results show substantial performance degradation, with accuracy dropping by 18.60\% and 14.35\% as tasks shift from single-document reasoning to longitudinal and cross-entity analysis. This degradation is driven by increased comparison hallucinations, temporal and entity mismatches, and is further reflected in declines in reasoning quality and factual consistency--limitations that existing benchmarks have yet to formally categorize or quantify.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11729v1">Cross-Architecture Model Diffing with Crosscoders: Unsupervised Discovery of Differences Between LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      Model diffing, the process of comparing models' internal representations to identify their differences, is a promising approach for uncovering safety-critical behaviors in new models. However, its application has so far been primarily focused on comparing a base model with its finetune. Since new LLM releases are often novel architectures, cross-architecture methods are essential to make model diffing widely applicable. Crosscoders are one solution capable of cross-architecture model diffing but have only ever been applied to base vs finetune comparisons. We provide the first application of crosscoders to cross-architecture model diffing and introduce Dedicated Feature Crosscoders (DFCs), an architectural modification designed to better isolate features unique to one model. Using this technique, we find in an unsupervised fashion features including Chinese Communist Party alignment in Qwen3-8B and Deepseek-R1-0528-Qwen3-8B, American exceptionalism in Llama3.1-8B-Instruct, and a copyright refusal mechanism in GPT-OSS-20B. Together, our results work towards establishing cross-architecture crosscoder model diffing as an effective method for identifying meaningful behavioral differences between AI models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11706v1">LLM-Driven 3D Scene Generation of Agricultural Simulation Environments</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 Accepted at IEEE Conference on Artificial Intelligence 2026
    </div>
    <details class="paper-abstract">
      Procedural generation techniques in 3D rendering engines have revolutionized the creation of complex environments, reducing reliance on manual design. Recent approaches using Large Language Models (LLMs) for 3D scene generation show promise but often lack domain-specific reasoning, verification mechanisms, and modular design. These limitations lead to reduced control and poor scalability. This paper investigates the use of LLMs to generate agricultural synthetic simulation environments from natural language prompts, specifically to address the limitations of lacking domain-specific reasoning, verification mechanisms, and modular design. A modular multi-LLM pipeline was developed, integrating 3D asset retrieval, domain knowledge injection, and code generation for the Unreal rendering engine using its API. This results in a 3D environment with realistic planting layouts and environmental context, all based on the input prompt and the domain knowledge. To enhance accuracy and scalability, the system employs a hybrid strategy combining LLM optimization techniques such as few-shot prompting, Retrieval-Augmented Generation (RAG), finetuning, and validation. Unlike monolithic models, the modular architecture enables structured data handling, intermediate verification, and flexible expansion. The system was evaluated using structured prompts and semantic accuracy metrics. A user study assessed realism and familiarity against real-world images, while an expert comparison demonstrated significant time savings over manual scene design. The results confirm the effectiveness of multi-LLM pipelines in automating domain-specific 3D scene generation with improved reliability and precision. Future work will explore expanding the asset hierarchy, incorporating real-time generation, and adapting the pipeline to other simulation domains beyond agriculture.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11688v1">GORGO: Maximizing KV-Cache Reuse While Minimizing Network Latency in Cross-Region LLM Load Balancing</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 12 pages, 4 figures. Code: https://github.com/atoniolo76/gotoni/tree/benchmark-load-balancing
    </div>
    <details class="paper-abstract">
      Distributing LLM inference across geographical regions can improve Time-to-First-Token (TTFT) by regionalizing service deployments. While existing multi-region load balancers save prefill computation by prioritizing Key--Value (KV) Cache hit rate, they ignore cluster networking latency, a critical factor in routing decisions. We introduce GORGO, a method for minimizing TTFT by optimizing a total serving cost as a function of available compute, network latency, and prefix caching. Using extensive profiling on custom infrastructure, we analyze component-level latency bottlenecks and benchmark GORGO against three baselines: (1) naive least-load routing, which ignores prefix-cache overlap; (2) prefix-similarity routing, which selectively pushes requests to the replica with the highest cached-prefix overlap; and (3) a centralized HTTP proxy that runs the GORGO policy while tracking requests across all nodes. We demonstrate that GORGO reduces P99 TTFT through network-aware routing and improves average TTFT by preventing pathological cross-region forwarding. Additionally, we find that GORGO-proxy overcomes synchronization overhead in previous methods and is 2.5x faster on median TTFT, demonstrating the success of a centralized router.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.12530v3">Translate Policy to Language: Flow Matching Generated Rewards for LLM Explanations</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 Accepted by ICLR 2026
    </div>
    <details class="paper-abstract">
      As humans increasingly share environments with diverse agents powered by RL, LLMs, and beyond, the ability to explain agent policies in natural language is vital for reliable coexistence. We introduce a general-purpose framework that trains explanation-generating LLMs via reinforcement learning from AI feedback, with distributional rewards generated by generative continuous normalizing flows (CNFs). CNFs capture the pluralistic and probabilistic nature of human judgments about explanations. Moreover, under mild assumptions, CNFs provably bound deviations from true human reward distributions when trained on noisy proxy rewards from LLMs. We design a specialized CNF architecture that selectively attends to linguistic cues in the decision context and explanations when generating rewards. Human and LLM evaluators find that our method delivers explanations that enable more accurate predictions of true agent decisions, exhibit greater logical soundness and actionability, and impose lower cognitive load than explanations trained with proxy LLM rewards or state-of-the-art RLHF and RLAIF baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09829v2">Internalizing Multi-Agent Reasoning for Accurate and Efficient LLM-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are reshaping recommender systems by leveraging extensive world knowledge and semantic reasoning to interpret user intent. However, effectively integrating these capabilities with collaborative signals while avoiding prohibitive inference latency remains a critical bottleneck. To address this, we propose a trajectory-driven internalization framework to develop a Single-agent Trajectory-Aligned Recommender (STAR). Specifically, to internalize complex reasoning capabilities into a single efficient model, we first design a multi-agent teacher system capable of multi-turn tool usage and reflection. This teacher utilizes a Collaborative Signal Translation mechanism to explicitly convert latent behavioral patterns into descriptive natural language evidence to enhance reasoning accuracy. Subsequently, a trajectory-driven distillation pipeline transfers this agentic logic, including planning, tool usage, and self-reflection, into the compact STAR model. Extensive experiments demonstrate that STAR surpasses its teacher by 8.7% to 39.5% while eliminating iterative latency, paving the way for real-time, reasoning-enhanced recommendation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10431v2">QTALE: Quantization-Robust Token-Adaptive Layer Execution for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 8 pages, 6 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demand substantial computational and memory resources, posing challenges for efficient deployment. Two complementary approaches have emerged to address these issues: token-adaptive layer execution, which reduces floating-point operations (FLOPs) by selectively bypassing layers, and quantization, which lowers memory footprint by reducing weight precision. However, naively integrating these techniques leads to additional accuracy degradation due to reduced redundancy in token-adaptive models. We propose QTALE (Quantization-Robust Token-Adaptive Layer Execution for LLMs), a novel framework that enables seamless integration of token-adaptive execution with quantization while preserving accuracy. Conventional token-adaptive methods reduce redundancy in two ways: (1) by limiting the diversity of training paths explored during fine-tuning, and (2) by lowering the number of parameters actively involved in inference. To overcome these limitations, QTALE introduces two key components: (1) a training strategy that ensures diverse execution paths are actively explored during fine-tuning, and (2) a post-training mechanism that allows flexible adjustment of the execution ratio at inference to reintroduce redundancy when needed. Experimental results show that QTALE enables seamless integration of token-adaptive layer execution with quantization, showing no noticeable accuracy difference, with the gap to quantization-only models kept below 0.5% on CommonsenseQA benchmarks. By combining token-adaptive execution for FLOPs reduction and quantization for memory savings, QTALE provides an effective solution for efficient LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11675v1">Right for the Wrong Reasons: Epistemic Regret Minimization for Causal Rung Collapse in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 18 pages, 6 tables, 3 figures
    </div>
    <details class="paper-abstract">
      Machine learning systems that are "right for the wrong reasons" achieve high performance through shortcuts that collapse under distributional shift. We show this pathology has a precise causal origin: autoregressive training provides no gradient signal to distinguish association P(Y|X) from intervention P(Y|do(X)), a failure we formalize as Rung Collapse. When outcome-based learning reinforces correct answers obtained through incorrect causal models, the agent becomes entrenched in flawed reasoning, a phenomenon we term Aleatoric Entrenchment. We propose Epistemic Regret Minimization (ERM), a belief revision objective that penalizes errors in causal reasoning independently of task success, and embed it within a three-layer architecture with three contributions grounded in knowledge representation: (1) a Physical Grounding Theorem proving that actions satisfying actuator independence implement valid do-operations, bridging action languages and do-calculus; (2) ERM as a causal belief revision operator satisfying AGM postulates, preventing entrenchment even when the agent succeeds for the wrong reasons; and (3) a failure mode taxonomy that classifies recurring reasoning errors and injects domain-independent guards, enabling cross-domain transfer. We prove asymptotic recovery of the true interventional distribution with finite-sample bounds. Experiments on 1,360 causal trap scenarios across six frontier LLMs reveal that Rung Collapse persists even in reasoning-enhanced models (3.7% for GPT-5.2), that steerability exhibits inverse scaling where advanced models resist generic correction, and that targeted ERM feedback recovers 53-59% of entrenched errors where outcome-level feedback fails.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11674v1">Benchmark Health Index: A Systematic Framework for Benchmarking the Benchmarks of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 42 pages, 8 figures, 7 tables. Code and website available at https://github.com/SKYLENAGE-AI/benchmark-health-index
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are advancing rapidly, yet the benchmarks used to measure this progress are becoming increasingly unreliable. Score inflation and selective reporting have eroded the authority of standard benchmarks, leaving the community uncertain about which evaluation results remain trustworthy. We introduce the Benchmark Health Index (BHI), a pure data-driven framework for auditing evaluation sets along three orthogonal and complementary axes: (1) Capability Discrimination, measuring how sharply a benchmark separates model performance beyond noise; (2) Anti-Saturation, estimating remaining headroom before ceiling effects erode resolution and thus the benchmark's expected longevity; and (3) Impact, quantifying influence across academic and industrial ecosystems via adoption breadth and practice-shaping power. By distilling 106 validated benchmarks from the technical reports of 91 representative models in 2025, we systematically characterize the evaluation landscape. BHI is the first framework to quantify benchmark health at a macro level, providing a principled basis for benchmark selection and enabling dynamic lifecycle management for next-generation evaluation protocols.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.09660v2">Steering MoE LLMs via Expert (De)Activation</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      Mixture-of-Experts (MoE) in Large Language Models (LLMs) routes each token through a subset of specialized Feed-Forward Networks (FFN), known as experts. We present SteerMoE, a framework to steer MoE models by detecting and controlling behavior-associated experts. We detect key experts by comparing how often they activate between paired inputs that demonstrate opposite behaviors (e.g., safe vs. unsafe). By selectively activating or deactivating such experts during inference, we control behaviors like faithfulness and safety without fine-tuning. Across 11 benchmarks and 6 LLMs, our steering raises safety by up to +20% and faithfulness by +27%. Alternatively, unsafe steering drops safety by -41% alone, and -100% when combined with existing jailbreak methods, bypassing all safety guardrails. Overall, SteerMoE offers a lightweight, effective, and widely applicable test-time control, while revealing unique vulnerabilities in MoE LLMs. https://github.com/adobe-research/SteerMoE
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11656v1">SToRM: Supervised Token Reduction for Multi-modal LLMs toward efficient end-to-end autonomous driving</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      In autonomous driving, end-to-end (E2E) driving systems that predict control commands directly from sensor data have achieved significant advancements. For safe driving in unexpected scenarios, these systems may additionally rely on human interventions such as natural language instructions. Using a multi-modal large language model (MLLM) facilitates human-vehicle interaction and can improve performance in such scenarios. However, this approach requires substantial computational resources due to its reliance on an LLM and numerous visual tokens from sensor inputs, which are limited in autonomous vehicles. Many MLLM studies have explored reducing visual tokens, but often suffer end-task performance degradation compared to using all tokens. To enable efficient E2E driving while maintaining performance comparable to using all tokens, this paper proposes the first Supervised Token Reduction framework for multi-modal LLMs (SToRM). The proposed framework consists of three key elements. First, a lightweight importance predictor with short-term sliding windows estimates token importance scores. Second, a supervised training approach uses an auxiliary path to obtain pseudo-supervision signals from an all-token LLM pass. Third, an anchor-context merging module partitions tokens into anchors and context tokens, and merges context tokens into relevant anchors to reduce redundancy while minimizing information loss. Experiments on the LangAuto benchmark show that SToRM outperforms state-of-the-art E2E driving MLLMs under the same reduced-token budget, maintaining all-token performance while reducing computational cost by up to 30x.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11655v1">LoRA-based Parameter-Efficient LLMs for Continuous Learning in Edge-based Malware Detection</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      The proliferation of edge devices has created an urgent need for security solutions capable of detecting malware in real time while operating under strict computational and memory constraints. Recently, Large Language Models (LLMs) have demonstrated remarkable capabilities in recognizing complex patterns, yet their deployment on edge devices remains impractical due to their resource demands. However, in edge malware detection, static or centrally retrained models degrade under evolving threats and heterogeneous traffic; locally trained models become siloed and fail to transfer across domains. To overcome these limitations, in this paper, we present a continuous learning architecture for edge-based malware detection that combines local adaptation on each device with global knowledge sharing through parameter-efficient LoRA adapters. Lightweight transformer models (DistilBERT, DistilGPT-2, TinyT5) run on edge nodes and are incrementally fine-tuned on device-specific traffic; only the resulting LoRA modules are aggregated by a lightweight coordinator and redistributed, enabling cross-device generalization without exchanging raw data. We evaluate on two public IoT security datasets, Edge-IIoTset and TON-IoT, under multi-round learning to simulate evolving threats. Compared to isolated fine-tuning, the LoRA-based exchange yields up to 20-25% accuracy gains when models encounter previously unseen attacks from another domain, while maintaining stable loss and F1 across rounds. LoRA adds less than 1% to model size (~0.6-1.8 MB), making updates practical for constrained edge hardware.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10564v2">SplitCom: Communication-efficient Split Federated Fine-tuning of LLMs via Temporal Compression</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      Federated fine-tuning of on-device large language models (LLMs) mitigates privacy concerns by preventing raw data sharing. However, the intensive computational and memory demands pose significant challenges for resource-constrained edge devices. To overcome these limitations, split federated learning (SFL) emerges as a promising solution that partitions the model into lightweight client-side and compute-intensive server-side sub-models, thus offloading the primary training workload to a powerful server. Nevertheless, high-dimensional activation exchanges in SFL lead to excessive communication overhead. To overcome this, we propose SplitCom, a communication-efficient SFL framework for LLMs that exploits temporal redundancy in activations across consecutive training epochs. Inspired by video compression, the core innovation of our framework lies in selective activation uploading only when a noticeable deviation from previous epochs occurs. To balance communication efficiency and learning performance, we introduce two adaptive threshold control schemes based on 1) bang-bang control or 2) deep deterministic policy gradient (DDPG)-based reinforcement learning. Moreover, we implement dimensionality reduction techniques to alleviate client-side memory requirements. Furthermore, we extend SplitCom to the U-shape architecture, ensuring the server never accesses clients' labels. Extensive simulations and laboratory experiments demonstrate that SplitCom reduces uplink communication costs by up to 98.6\,\% in its standard configuration and total communication costs by up to 95.8\,\% in its U-shape variant without noticeably compromising model performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11650v1">Which Feedback Works for Whom? Differential Effects of LLM-Generated Feedback Elements Across Learner Profiles</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show promise for automatically generating feedback in education settings. However, it remains unclear how specific feedback elements, such as tone and information coverage, contribute to learning outcomes and learner acceptance, particularly across learners with different personality traits. In this study, we define six feedback elements and generate feedback for multiple-choice biology questions using GPT-5. We conduct a learning experiment with 321 first-year high school students and evaluate feedback effectiveness using two learning outcomes measures and subjective evaluations across six criteria. We further analyze differences in how feedback acceptance varies across learners based on Big Five personality traits. Our results show that effective feedback elements share common patterns supporting learning outcomes, while learners' subjective preferences differ across personality-based clusters. These findings highlight the importance of selecting and adapting feedback elements according to learners' personality traits when we design LLM-generated feedback, and provide practical implications for personalized feedback design in education.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11641v1">Both Topology and Text Matter: Revisiting LLM-guided Out-of-Distribution Detection on Text-attributed Graphs</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Text-attributed graphs (TAGs) associate nodes with textual attributes and graph structure, enabling GNNs to jointly model semantic and structural information. While effective on in-distribution (ID) data, GNNs often encounter out-of-distribution (OOD) nodes with unseen textual or structural patterns in real-world settings, leading to overconfident and erroneous predictions in the absence of reliable OOD detection. Early approaches address this issue from a topology-driven perspective, leveraging neighboring structures to mitigate node-level detection bias. However, these methods typically encode node texts as shallow vector features, failing to fully exploit rich semantic information. In contrast, recent LLM-based approaches generate pseudo OOD priors by leveraging textual knowledge, but they suffer from several limitations: (1) a reliability-informativeness imbalance in the synthesized OOD priors, as the generated OOD exposures either deviate from the true OOD semantics, or introduce non-negligible ID noise, all of which offers limited improvement to detection performance; (2) reliance on specialized architectures, which prevents incorporation of the extensive effective topology-level insights that have been empirically validated in prior work. To this end, we propose LG-Plug, an LLM-Guided Plug-and-play strategy for TAG OOD detection tasks. LG-Plug aligns topology and text representations to produce fine-grained node embeddings, then generates consensus-driven OOD exposure via clustered iterative LLM prompting. Moreover, it leverages lightweight in-cluster codebook and heuristic sampling reduce time cost of LLM querying. The resulting OOD exposure serves as a regularization term to separate ID and OOD nodes, enabling seamless integration with existing detectors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22548v3">Are LLM Evaluators Really Narcissists? Sanity Checking Self-Preference Evaluations</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      Recent research has shown that large language models (LLMs) favor their own outputs when acting as judges, undermining the integrity of automated post-training and evaluation workflows. However, it is difficult to disentangle which evaluation biases are explained by narcissism versus general experimental confounds, distorting measurements of self-preference bias. We discover a core methodological confound which could reduce measurement error by 89.6%. Specifically, LLM evaluators may deliver self-preferring verdicts when the judge responds to queries which they completed incorrectly themselves; this would be true regardless of whether one of their responses is their own. To decouple self-preference signals from noisy outputs on hard problems, we introduce an Evaluator Quality Baseline, which compares the probability that a judge incorrectly votes for itself against the probability that it votes for an incorrect response from another model. Evaluating this simple baseline on 37,448 queries, only 51% of initial findings retain statistical significance. Finally, we turn towards characterizing the entropy of "easy" versus "hard" evaluation votes from LLM judges. Our corrective baseline enables future research on self-preference by eliminating noisy data from potential solutions. More widely, this work contributes to the growing body of work on cataloging and isolating judge-bias effects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11619v1">When Agents Disagree With Themselves: Measuring Behavioral Consistency in LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 5 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Run the same LLM agent on the same task twice: do you get the same behavior? We find the answer is often no. In a study of 3,000 agent runs across three models (Llama 3.1 70B, GPT-4o, and Claude Sonnet 4.5) on HotpotQA, we observe that ReAct-style agents produce 2.0--4.2 distinct action sequences per 10 runs on average, even with identical inputs. More importantly, this variance predicts failure: tasks with consistent behavior ($\leq$2 unique paths) achieve 80--92% accuracy, while highly inconsistent tasks ($\geq$6 unique paths) achieve only 25--60%, a 32--55 percentage point gap depending on model. We trace variance to early decisions: 69% of divergence occurs at step 2, the first search query. Our results suggest that monitoring behavioral consistency during execution could enable early error detection and improve agent reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.17061v5">Parallelism Meets Adaptiveness: Scalable Documents Understanding in Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 Accepted at AAAI 2026 Workshop on WoMAPF, Camera ready version
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have shown increasing promise for collaborative task completion. However, existing multi-agent frameworks often rely on static workflows, fixed roles, and limited inter-agent communication, reducing their effectiveness in open-ended, high-complexity domains. This paper proposes a coordination framework that enables adaptiveness through three core mechanisms: dynamic task routing, bidirectional feedback, and parallel agent evaluation. The framework allows agents to reallocate tasks based on confidence and workload, exchange structured critiques to iteratively improve outputs, and crucially compete on high-ambiguity subtasks with evaluator-driven selection of the most suitable result. We instantiate these principles in a modular architecture and demonstrate substantial improvements in factual coverage, coherence, and efficiency over static and partially adaptive baselines. Our findings highlight the benefits of incorporating both adaptiveness and structured competition in multi-agent LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11583v1">The Five Ws of Multi-Agent Communication: Who Talks to Whom, When, What, and Why -- A Survey from MARL to Emergent Language and LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 Accepted at Transactions on Machine Learning Research (TMLR), 2026
    </div>
    <details class="paper-abstract">
      Multi-agent sequential decision-making powers many real-world systems, from autonomous vehicles and robotics to collaborative AI assistants. In dynamic, partially observable environments, communication is often what reduces uncertainty and makes collaboration possible. This survey reviews multi-agent communication (MA-Comm) through the Five Ws: who communicates with whom, what is communicated, when communication occurs, and why communication is beneficial. This framing offers a clean way to connect ideas across otherwise separate research threads. We trace how communication approaches have evolved across three major paradigms. In Multi-Agent Reinforcement Learning (MARL), early methods used hand-designed or implicit protocols, followed by end-to-end learned communication optimized for reward and control. While successful, these protocols are frequently task-specific and hard to interpret, motivating work on Emergent Language (EL), where agents can develop more structured or symbolic communication through interaction. EL methods, however, still struggle with grounding, generalization, and scalability, which has fueled recent interest in large language models (LLMs) that bring natural language priors for reasoning, planning, and collaboration in more open-ended settings. Across MARL, EL, and LLM-based systems, we highlight how different choices shape communication design, where the main trade-offs lie, and what remains unsolved. We distill practical design patterns and open challenges to support future hybrid systems that combine learning, language, and control for scalable and interpretable multi-agent collaboration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.18095v2">SMaRT: Select, Mix, and ReinvenT -- A Strategy Fusion Framework for LLM-Driven Reasoning and Planning</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have redefined complex task automation with exceptional generalization capabilities. Despite these advancements, state-of-the-art methods rely on single-strategy prompting, missing the synergy of diverse reasoning approaches. No single strategy excels universally, highlighting the need for frameworks that fuse strategies to maximize performance and ensure robustness. We introduce the Select, Mix, and ReinvenT (SMaRT) framework, an innovative strategy fusion approach designed to overcome this constraint by creating balanced and efficient solutions through the seamless integration of diverse reasoning strategies. Unlike existing methods, which employ LLMs merely as evaluators, SMaRT uses them as intelligent integrators, unlocking the "best of all worlds" across tasks. Extensive empirical evaluations across benchmarks in reasoning, planning, and sequential decision-making highlight the robustness and adaptability of SMaRT. The framework consistently outperforms state-of-the-art baselines in solution quality, constraint adherence, and performance metrics. This work redefines LLM-driven decision-making by pioneering a new paradigm in cross-strategy calibration, unlocking superior outcomes for reasoning systems and advancing the boundaries of self-refining methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.06111v2">SKATE, a Scalable Tournament Eval: Weaker LLMs differentiate between stronger ones using verifiable challenges</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 7 pages and appendices
    </div>
    <details class="paper-abstract">
      Evaluating the capabilities and risks of foundation models is paramount, yet current methods demand extensive domain expertise, hindering their scalability as these models rapidly evolve. We introduce SKATE: a novel evaluation framework in which large language models (LLMs) compete by generating and solving verifiable tasks for one another. Our core insight is to treat evaluation as a game: models act as both task-setters and solvers, incentivized to create questions which highlight their own strengths while exposing others' weaknesses. SKATE offers several key advantages, balancing scalability, open-endedness, and objectivity. It is fully automated, data-free, and scalable, requiring no human input or domain expertise. By using verifiable tasks rather than LLM judges, scoring is objective. Unlike domain-limited programmatically-generated benchmarks (e.g. chess-playing or spatial reasoning), having LLMs creatively pose challenges enables open-ended and scalable evaluation. As a proof of concept, we introduce LLM-set code-output-prediction (COP) challenges as a verifiable and extensible framework in which to test our approach. Using a TrueSkill-based ranking system, we evaluate six frontier LLMs and find that: (1) weaker models can reliably differentiate and score stronger ones, (2) LLM-based systems are capable of self-preferencing behavior, generating questions that align with their own capabilities, and (3) SKATE automatically surfaces fine-grained capability differences between models. Our findings are an important step towards general, scalable evaluation frameworks which can keep pace with LLM progress.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11528v1">Stop Tracking Me! Proactive Defense Against Attribute Inference Attack in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 Accepted at ICLR 2026
    </div>
    <details class="paper-abstract">
      Recent studies have shown that large language models (LLMs) can infer private user attributes (e.g., age, location, gender) from user-generated text shared online, enabling rapid and large-scale privacy breaches. Existing anonymization-based defenses are coarse-grained, lacking word-level precision in anonymizing privacy-leaking elements. Moreover, they are inherently limited as altering user text to hide sensitive cues still allows attribute inference to occur through models' reasoning capabilities. To address these limitations, we propose a unified defense framework that combines fine-grained anonymization (TRACE) with inference-preventing optimization (RPS). TRACE leverages attention mechanisms and inference chain generation to identify and anonymize privacy-leaking textual elements, while RPS employs a lightweight two-stage optimization strategy to induce model rejection behaviors, thereby preventing attribute inference. Evaluations across diverse LLMs show that TRACE-RPS reduces attribute inference accuracy from around 50\% to below 5\% on open-source models. In addition, our approach offers strong cross-model generalization, prompt-variation robustness, and utility-privacy tradeoffs. Our code is available at https://github.com/Jasper-Yan/TRACE-RPS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11521v1">PAM: Processing Across Memory Hierarchy for Efficient KV-centric LLM Serving System</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 15 pages, 13 figures
    </div>
    <details class="paper-abstract">
      The widespread adoption of Large Language Models (LLMs) has exponentially increased the demand for efficient serving systems. With growing requests and context lengths, key-value (KV)-related operations, including attention computation and KV cache storage, have emerged as critical bottlenecks. They require massive memory bandwidth and capacity. Unfortunately, existing LLM serving systems, optimized for compute-bound workloads, fail to handle these memory-intensive operations effectively. Even with Processing-In-Memory (PIM) technology, current single-level memory designs cannot simultaneously satisfy the bandwidth and capacity requirements. To address these challenges, we propose Processing Across Memory (PAM), a KV-centric LLM serving system that coordinates heterogeneous PIM-enabled memory devices within a hierarchical architecture. PAM introduces a novel computing paradigm to balance high memory bandwidth with scalable capacity. First, PAM exploits the inherent context locality in KV access patterns to intelligently distribute KV tokens across the memory hierarchy. Second, to further exploit context locality, it introduces the PAMattention algorithm, enabling fine-grained parallel attention computation across heterogeneous PIM devices. Finally, PAM incorporates an intra-device KV mapping, inter-device KV migration interface, and an inter-device online KV scheduling algorithm to dynamically balance computational workloads. By addressing both bandwidth and capacity demands simultaneously, PAM significantly enhances the efficiency and scalability of LLM serving systems, paving the way for cost-effective, high-performance solutions in the era of large-scale AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11510v1">AgentLeak: A Full-Stack Benchmark for Privacy Leakage in Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 17 pages, 10 figures, 13 tables. Code and dataset available at https://github.com/Privatris/AgentLeak
    </div>
    <details class="paper-abstract">
      Multi-agent Large Language Model (LLM) systems create privacy risks that current benchmarks cannot measure. When agents coordinate on tasks, sensitive data passes through inter-agent messages, shared memory, and tool arguments; pathways that output-only audits never inspect. We introduce AgentLeak, to the best of our knowledge the first full-stack benchmark for privacy leakage covering internal channels, spanning 1,000 scenarios across healthcare, finance, legal, and corporate domains, paired with a 32-class attack taxonomy and three-tier detection pipeline. Testing GPT-4o, GPT-4o-mini, Claude 3.5 Sonnet, Mistral Large, and Llama 3.3 70B across 4,979 traces reveals that multi-agent configurations reduce per-channel output leakage (C1: 27.2% vs 43.2% in single-agent) but introduce unmonitored internal channels that raise total system exposure to 68.9% (OR-aggregated across C1, C2, C5). Internal channels account for most of this gap: inter-agent messages (C2) leak at 68.8%, compared to 27.2% on C1 (output channel). This means that output-only audits miss 41.7% of violations. Claude 3.5 Sonnet, which emphasizes safety alignment in its design, achieves the lowest leakage rates on both external (3.3%) and internal (28.1%) channels, suggesting that model-level safety training may transfer to internal channel protection. Across all five models and four domains, the pattern C2 > C1 holds consistently, confirming that inter-agent communication is the primary vulnerability. These findings underscore the need for coordination frameworks that incorporate internal-channel privacy protections and enforce privacy controls on inter-agent communication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11506v1">RooflineBench: A Benchmarking Framework for On-Device LLMs via Roofline Analysis</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      The transition toward localized intelligence through Small Language Models (SLMs) has intensified the need for rigorous performance characterization on resource-constrained edge hardware. However, objectively measuring the theoretical performance ceilings of diverse architectures across heterogeneous platforms remains a formidable challenge. In this work, we propose a systematic framework based on the Roofline model that unifies architectural primitives and hardware constraints through the lens of operational intensity (OI). By defining an inference-potential region, we introduce the Relative Inference Potential as a novel metric to compare efficiency differences between Large Language Models (LLMs) on the same hardware substrate. Extensive empirical analysis across diverse compute tiers reveals that variations in performance and OI are significantly influenced by sequence length. We further identify a critical regression in OI as model depth increases. Additionally, our findings highlight an efficiency trap induced by hardware heterogeneity and demonstrate how structural refinements, such as Multi-head Latent Attention (M LA), can effectively unlock latent inference potential across various hardware substrates. These insights provide actionable directions for hardware-software co-design to align neural structures with physical constraints in on-device intelligence. The released code is available in the Appendix C.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.13779v2">NewsInterview: a Dataset and a Playground to Evaluate LLMs' Ground Gap via Informational Interviews</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 Accepted at ACL 2025: https://aclanthology.org/2025.acl-long.1580/
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive capabilities in generating coherent text but often struggle with grounding language and strategic dialogue. To address this gap, we focus on journalistic interviews, a domain rich in grounding communication and abundant in data. We curate a dataset of 40,000 two-person informational interviews from NPR and CNN, and reveal that LLMs are significantly less likely than human interviewers to use acknowledgements and to pivot to higher-level questions. Realizing that a fundamental deficit exists in multi-turn planning and strategic thinking, we develop a realistic simulated environment, incorporating source personas and persuasive elements, in order to facilitate the development of agents with longer-horizon rewards. Our experiments show that while source LLMs mimic human behavior in information sharing, interviewer LLMs struggle with recognizing when questions are answered and engaging persuasively, leading to suboptimal information extraction across model size and capability. These findings underscore the need for enhancing LLMs' strategic dialogue capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11488v1">When Audio-LLMs Don't Listen: A Cross-Linguistic Study of Modality Arbitration</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 25 pages, 18 tables, 8 languages, benchmark and code at https://github.com/jb1999/alme-benchmark
    </div>
    <details class="paper-abstract">
      When audio and text conflict, speech-enabled language models follow the text 10 times more often than when arbitrating between two text sources, even when explicitly instructed to trust the audio. Using ALME, a benchmark of 57,602 controlled audio-text conflict stimuli across 8 languages, we find that Gemini 2.0 Flash exhibits 16.6\% text dominance under audio-text conflict versus 1.6\% under text-text conflict with identical reliability cues. This gap is not explained by audio quality: audio-only accuracy (97.2\%) exceeds cascade accuracy (93.9\%), indicating audio embeddings preserve more information than text transcripts. We propose that text dominance reflects an asymmetry not in information content but in arbitration accessibility: how easily the model can reason over competing representations. This framework explains otherwise puzzling findings. Forcing transcription before answering increases text dominance (19\% to 33\%), sacrificing audio's information advantage without improving accessibility. Framing text as ``deliberately corrupted'' reduces text dominance by 80\%. A fine-tuning ablation provides interventional evidence: training only the audio projection layer increases text dominance (+26.5\%), while LoRA on the language model halves it ($-$23.9\%), localizing text dominance to the LLM's reasoning rather than the audio encoder. Experiments across four state-of-the-art audio-LLMs and 8 languages show consistent trends with substantial cross-linguistic and cross-model variation, establishing modality arbitration as a distinct reliability dimension not captured by standard speech benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.24836v4">Logical Structure as Knowledge: Enhancing LLM Reasoning via Structured Logical Knowledge Density Estimation</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      The reasoning capabilities of Large Language Models (LLMs) are increasingly attributed to training data quality rather than mere parameter scaling. However, existing data-centric paradigms often equate quality with factuality or diversity and ignore the internal logical complexity of training samples. In this work, we propose that natural language harbors Structured Logical Knowledge manifested through entailment relationships and logical topologies. To quantify this, we introduce Structured Logical Knowledge Density (SLKD), a novel metric that measures logical information content by decomposing natural language into executable predicates and logical primitives. Our analysis reveals a significant logical disparity in current datasets where sparse logical signals predominate. Consequently, we propose a density aware re-cognizing optimization strategy that prioritizes high-density logical samples to enhance with the LLM's reasoning ability. Extensive experiments demonstrate that our approach enhances reasoning performance and generalization without increasing total data volume. These results, further validated within a reinforcement learning framework, suggest that elevating logical density is more critical than expanding data scale for realizing the full cognitive potential of LLMs. The released code is available in the Appendix C.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.26219v2">Test-Time Alignment of LLMs via Sampling-Based Optimal Control in pre-logit space</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 21 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Test-time alignment of large language models (LLMs) attracts attention because fine-tuning LLMs requires high computational costs. In this paper, we propose a new test-time alignment method called adaptive importance sampling on pre-logits (AISP) on the basis of the sampling-based model predictive control with the stochastic control input. AISP applies the Gaussian perturbation into pre-logits, which are outputs of the penultimate layer, so as to maximize expected rewards with respect to the mean of the perturbation. We demonstrate that the optimal mean is obtained by importance sampling with sampled rewards. AISP outperforms best-of-n sampling in terms of rewards over the number of used samples and achieves higher rewards than other reward-based test-time alignment methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10467v2">MERIT Feedback Elicits Better Bargaining in LLM Negotiators</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 Preprint. Affiliation typo corrected
    </div>
    <details class="paper-abstract">
      Bargaining is often regarded as a logical arena rather than an art or a matter of intuition, yet Large Language Models (LLMs) still struggle to navigate it due to limited strategic depth and difficulty adapting to complex human factors. Current benchmarks rarely capture this limitation. To bridge this gap, we present an utility feedback centric framework. Our contributions are: (i) AgoraBench, a new benchmark spanning nine challenging settings (e.g., deception, monopoly) that supports diverse strategy modeling; (ii) human-aligned, economically grounded metrics derived from utility theory. This is operationalized via agent utility, negotiation power, and acquisition ratio that implicitly measure how well the negotiation aligns with human preference and (iii) a human preference grounded dataset with learning pipeline that strengthens LLMs' bargaining ability through both prompting and finetuning. Empirical results indicate that baseline LLM strategies often diverge from human preferences, while our mechanism substantially improves negotiation performance, yielding deeper strategic behavior and stronger opponent awareness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11460v1">ADRD-Bench: A Preliminary LLM Benchmark for Alzheimer's Disease and Related Dementias</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown great potential for healthcare applications. However, existing evaluation benchmarks provide minimal coverage of Alzheimer's Disease and Related Dementias (ADRD). To address this gap, we introduce ADRD-Bench, the first ADRD-specific benchmark dataset designed for rigorous evaluation of LLMs. ADRD-Bench has two components: 1) ADRD Unified QA, a synthesis of 1,352 questions consolidated from seven established medical benchmarks, providing a unified assessment of clinical knowledge; and 2) ADRD Caregiving QA, a novel set of 149 questions derived from the Aging Brain Care (ABC) program, a widely used, evidence-based brain health management program. Guided by a program with national expertise in comprehensive ADRD care, this new set was designed to mitigate the lack of practical caregiving context in existing benchmarks. We evaluated 33 state-of-the-art LLMs on the proposed ADRD-Bench. Results showed that the accuracy of open-weight general models ranged from 0.63 to 0.93 (mean: 0.78; std: 0.09). The accuracy of open-weight medical models ranged from 0.48 to 0.93 (mean: 0.82; std: 0.13). The accuracy of closed-source general models ranged from 0.83 to 0.91 (mean: 0.89; std: 0.03). While top-tier models achieved high accuracies (>0.9), case studies revealed that inconsistent reasoning quality and stability limit their reliability, highlighting a critical need for domain-specific improvement to enhance LLMs' knowledge and reasoning grounded in daily caregiving data. The entire dataset is available at https://github.com/IIRL-ND/ADRD-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.10054v3">Uni-DPO: A Unified Paradigm for Dynamic Preference Optimization of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 Accepted by ICLR 2026. Code & models: https://github.com/pspdada/Uni-DPO
    </div>
    <details class="paper-abstract">
      Direct Preference Optimization (DPO) has emerged as a cornerstone of reinforcement learning from human feedback (RLHF) due to its simplicity and efficiency. However, existing DPO-based methods typically treat all preference pairs equally, overlooking substantial variations in data quality and learning difficulty, which leads to inefficient data utilization and suboptimal performance. To address this limitation, we propose Uni-DPO, a unified dynamic preference optimization framework that jointly considers (a) the inherent quality of preference pairs and (b) the model's evolving performance during training. By adaptively reweighting samples based on both factors, Uni-DPO enables more effective use of preference data and achieves superior performance. Extensive experiments across models and benchmarks demonstrate the effectiveness and generalization of Uni-DPO. On textual tasks, Gemma-2-9B-IT fine-tuned with Uni-DPO surpasses the leading LLM, Claude 3 Opus, by 6.7 points on Arena-Hard. On mathematical and multimodal tasks, Uni-DPO consistently outperforms baseline methods across all benchmarks, providing strong empirical evidence of its effectiveness and robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.07695v2">EventCast: Hybrid Demand Forecasting in E-Commerce with LLM-Based Event Knowledge</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Demand forecasting is a cornerstone of e-commerce operations, directly impacting inventory planning and fulfillment scheduling. However, existing forecasting systems often fail during high-impact periods such as flash sales, holiday campaigns, and sudden policy interventions, where demand patterns shift abruptly and unpredictably. In this paper, we introduce EventCast, a modular forecasting framework that integrates future event knowledge into time-series prediction. Unlike prior approaches that ignore future interventions or directly use large language models (LLMs) for numerical forecasting, EventCast leverages LLMs solely for event-driven reasoning. Unstructured business data, which covers campaigns, holiday schedules, and seller incentives, from existing operational databases, is processed by an LLM that converts it into interpretable textual summaries leveraging world knowledge for cultural nuances and novel event combinations. These summaries are fused with historical demand features within a dual-tower architecture, enabling accurate, explainable, and scalable forecasts. Deployed on real-world e-commerce scenarios spanning 4 countries of 160 regions over 10 months, EventCast achieves up to 86.9% and 97.7% improvement on MAE and MSE compared to the variant without event knowledge, and reduces MAE by up to 57.0% and MSE by 83.3% versus the best industrial baseline during event-driven periods. EventCast has deployed into real-world industrial pipelines since March 2025, offering a practical solution for improving operational decision-making in dynamic e-commerce environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21963v2">Industrialized Deception: The Collateral Effects of LLM-Generated Misinformation on Digital Ecosystems</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 Accepted at ACM TheWebConf '26 Companion
    </div>
    <details class="paper-abstract">
      Generative AI and misinformation research has evolved since our 2024 survey. This paper presents an updated perspective, transitioning from literature review to practical countermeasures. We report on changes in the threat landscape, including improved AI-generated content through Large Language Models (LLMs) and multimodal systems. Central to this work are our practical contributions: JudgeGPT, a platform for evaluating human perception of AI-generated news, and RogueGPT, a controlled stimulus generation engine for research. Together, these tools form an experimental pipeline for studying how humans perceive and detect AI-generated misinformation. Our findings show that detection capabilities have improved, but the competition between generation and detection continues. We discuss mitigation strategies including LLM-based detection, inoculation approaches, and the dual-use nature of generative AI. This work contributes to research addressing the adverse impacts of AI on information quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10891v1">Interactive LLM-assisted Curriculum Learning for Multi-Task Evolutionary Policy Search</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 8 pages, 7 figures, with Appendix
    </div>
    <details class="paper-abstract">
      Multi-task policy search is a challenging problem because policies are required to generalize beyond training cases. Curriculum learning has proven to be effective in this setting, as it introduces complexity progressively. However, designing effective curricula is labor-intensive and requires extensive domain expertise. LLM-based curriculum generation has only recently emerged as a potential solution, but was limited to operate in static, offline modes without leveraging real-time feedback from the optimizer. Here we propose an interactive LLM-assisted framework for online curriculum generation, where the LLM adaptively designs training cases based on real-time feedback from the evolutionary optimization process. We investigate how different feedback modalities, ranging from numeric metrics alone to combinations with plots and behavior visualizations, influence the LLM ability to generate meaningful curricula. Through a 2D robot navigation case study, tackled with genetic programming as optimizer, we evaluate our approach against static LLM-generated curricula and expert-designed baselines. We show that interactive curriculum generation outperforms static approaches, with multimodal feedback incorporating both progression plots and behavior visualizations yielding performance competitive with expert-designed curricula. This work contributes to understanding how LLMs can serve as interactive curriculum designers for embodied AI systems, with potential extensions to broader evolutionary robotics applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10881v1">Diagnosing Structural Failures in LLM-Based Evidence Extraction for Meta-Analysis</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 Accepted at the 22nd Conference on Information and Research Science Connecting to Digital and Library Science (IRCDL 2026)
    </div>
    <details class="paper-abstract">
      Systematic reviews and meta-analyses rely on converting narrative articles into structured, numerically grounded study records. Despite rapid advances in large language models (LLMs), it remains unclear whether they can meet the structural requirements of this process, which hinge on preserving roles, methods, and effect-size attribution across documents rather than on recognizing isolated entities. We propose a structural, diagnostic framework that evaluates LLM-based evidence extraction as a progression of schema-constrained queries with increasing relational and numerical complexity, enabling precise identification of failure points beyond atom-level extraction. Using a manually curated corpus spanning five scientific domains, together with a unified query suite and evaluation protocol, we evaluate two state-of-the-art LLMs under both per-document and long-context, multi-document input regimes. Across domains and models, performance remains moderate for single-property queries but degrades sharply once tasks require stable binding between variables, roles, statistical methods, and effect sizes. Full meta-analytic association tuples are extracted with near-zero reliability, and long-context inputs further exacerbate these failures. Downstream aggregation amplifies even minor upstream errors, rendering corpus-level statistics unreliable. Our analysis shows that these limitations stem not from entity recognition errors, but from systematic structural breakdowns, including role reversals, cross-analysis binding drift, instance compression in dense result sections, and numeric misattribution, indicating that current LLMs lack the structural fidelity, relational binding, and numerical grounding required for automated meta-analysis. The code and data are publicly available at GitHub (https://github.com/zhiyintan/LLM-Meta-Analysis).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.07954v2">Bielik Guard: Efficient Polish Language Safety Classifiers for LLM Content Moderation</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become increasingly deployed in Polish language applications, the need for efficient and accurate content safety classifiers has become paramount. We present Bielik Guard, a family of compact Polish language safety classifiers comprising two model variants: a 0.1B parameter model based on MMLW-RoBERTa-base and a 0.5B parameter model based on PKOBP/polish-roberta-8k. Fine-tuned on a community-annotated dataset of 6,885 Polish texts, these models classify content across five safety categories: Hate/Aggression, Vulgarities, Sexual Content, Crime, and Self-Harm. Our evaluation demonstrates that both models achieve strong performance on multiple benchmarks. The 0.5B variant offers the best overall discrimination capability with F1 scores of 0.791 (micro) and 0.785 (macro) on the test set, while the 0.1B variant demonstrates exceptional efficiency. Notably, Bielik Guard 0.1B v1.1 achieves superior precision (77.65%) and very low false positive rate (0.63%) on real user prompts, outperforming HerBERT-PL-Guard (31.55% precision, 4.70% FPR) despite identical model size. The models are publicly available and designed to provide appropriate responses rather than simple content blocking, particularly for sensitive categories like self-harm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11224v1">Agent-Diff: Benchmarking LLM Agents on Enterprise API Tasks via Code Execution with State-Diff-Based Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 Pre-Print. Under review for KDD 2026
    </div>
    <details class="paper-abstract">
      We present Agent-Diff, a novel benchmarking framework for evaluating agentic Large Language Models (LLMs) on real-world tasks that execute code via external APIs. Agentic LLM performance varies due to differences in models, external tool access, prompt structures, and agentic frameworks. Benchmarks must make fundamental trade-offs between a sandboxed approach that controls for variation in software environments and more ecologically valid approaches employing real services. Agent-Diff attempts to capture the desirable features of both of these approaches by including access to the real API interfaces for software services while sandboxing the environment in which calls are made, processed, and evaluated. This approach relies on two key innovations. The first is a novel state-diff contract, which separates process from outcome - rather than fuzzy trace or parameter matching, we define task success as whether the expected change in environment state was achieved. The second is a novel sandbox that provides a standardized scripting layer that all models use to execute code against external APIs (Slack, Box, Linear, Google Calendar). Thus, we can evaluate different agentic LLMs against a standardized set of contracts using a unified sandbox while still evaluating their performance on real-world service interfaces. Using the Agent-Diff framework, we provide benchmarks for nine LLMs across 224 tasks utilizing enterprise software workflows. In addition, we evaluate the robustness of the framework with ablation experiments to assess the contribution of access to API documentation on benchmark performance. Code and data: https://github.com/agent-diff-bench/agent-diff.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10833v1">Training-Induced Bias Toward LLM-Generated Content in Dense Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 Accepted at ECIR 2026
    </div>
    <details class="paper-abstract">
      Dense retrieval is a promising approach for acquiring relevant context or world knowledge in open-domain natural language processing tasks and is now widely used in information retrieval applications. However, recent reports claim a broad preference for text generated by large language models (LLMs). This bias is called "source bias", and it has been hypothesized that lower perplexity contributes to this effect. In this study, we revisit this claim by conducting a controlled evaluation to trace the emergence of such preferences across training stages and data sources. Using parallel human- and LLM-generated counterparts of the SciFact and Natural Questions (NQ320K) datasets, we compare unsupervised checkpoints with models fine-tuned using in-domain human text, in-domain LLM-generated text, and MS MARCO. Our results show the following: 1) Unsupervised retrievers do not exhibit a uniform pro-LLM preference. The direction and magnitude depend on the dataset. 2) Across the settings tested, supervised fine-tuning on MS MARCO consistently shifts the rankings toward LLM-generated text. 3) In-domain fine-tuning produces dataset-specific and inconsistent shifts in preference. 4) Fine-tuning on LLM-generated corpora induces a pronounced pro-LLM bias. Finally, a retriever-centric perplexity probe involving the reattachment of a language modeling head to the fine-tuned dense retriever encoder indicates agreement with relevance near chance, thereby weakening the explanatory power of perplexity. Our study demonstrates that source bias is a training-induced phenomenon rather than an inherent property of dense retrievers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.24384v2">HarmMetric Eval: Benchmarking Metrics and Judges for LLM Harmfulness Assessment</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      The potential for large language models (LLMs) to generate harmful content poses a significant safety risk in their deployment. To address and assess this risk, the community has developed numerous harmfulness evaluation metrics and judges. However, the lack of a systematic benchmark for evaluating these metrics and judges undermines the credibility and consistency of LLM safety assessments. To bridge this gap, we introduce HarmMetric Eval, a comprehensive benchmark designed to support both overall and fine-grained evaluation of harmfulness metrics and judges. In HarmMetric Eval, we build a high-quality dataset of representative harmful prompts paired with highly diverse harmful model responses and non-harmful counterparts across multiple categories. We also propose a flexible scoring mechanism that rewards the metrics for correctly ranking harmful responses above non-harmful ones, which is applicable to almost all existing metrics and judges with varying output formats and scoring scales. Using HarmMetric Eval, we uncover a surprising finding by extensive experiments: Conventional reference-based metrics such as ROUGE and METEOR can outperform existing LLM-based judges in fine-grained harmfulness evaluation, challenging prevailing assumptions about LLMs'superiority in this domain. To reveal the reasons behind this finding, we provide a fine-grained analysis to explain the limitations of LLM-based judges on rating irrelevant or useless responses. Furthermore, we build a new harmfulness judge by incorporating the fine-grained criteria into its prompt template and leverage reference-based metrics to fine-tune its base LLM. The resulting judge demonstrates superior performance than all existing metrics and judges in evaluating harmful responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10808v1">PELLI: Framework to effectively integrate LLMs for quality software generation</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      Recent studies have revealed that when LLMs are appropriately prompted and configured, they demonstrate mixed results. Such results often meet or exceed the baseline performance. However, these comparisons have two primary issues. First, they mostly considered only reliability as a comparison metric and selected a few LLMs (such as Codex and ChatGPT) for comparision. This paper proposes a comprehensive code quality assessment framework called Programmatic Excellence via LLM Iteration (PELLI). PELLI is an iterative analysis-based process that upholds high-quality code changes. We extended the state-of-the-art by performing a comprehensive evaluation that generates quantitative metrics for analyzing three primary nonfunctional requirements (such as maintainability, performance, and reliability) while selecting five popular LLMs. For PELLI's applicability, we selected three application domains while following Python coding standards. Following this framework, practitioners can ensure harmonious integration between LLMs and human developers, ensuring that their potential is fully realized. PELLI can serve as a practical guide for developers aiming to leverage LLMs while adhering to recognized quality standards. This study's outcomes are crucial for advancing LLM technologies in real-world applications, providing stakeholders with a clear understanding of where these LLMs excel and where they require further refinement. Overall, based on three nonfunctional requirements, we have found that GPT-4T and Gemini performed slightly better. We also found that prompt design can influence the overall code quality. In addition, each application domain demonstrated high and low scores across various metrics, and even within the same metrics across different prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10801v1">Deep Learning-based Method for Expressing Knowledge Boundary of Black-Box LLM</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success, however, the emergence of content generation distortion (hallucination) limits their practical applications. The core cause of hallucination lies in LLMs' lack of awareness regarding their stored internal knowledge, preventing them from expressing their knowledge state on questions beyond their internal knowledge boundaries, as humans do. However, existing research on knowledge boundary expression primarily focuses on white-box LLMs, leaving methods suitable for black-box LLMs which offer only API access without revealing internal parameters-largely unexplored. Against this backdrop, this paper proposes LSCL (LLM-Supervised Confidence Learning), a deep learning-based method for expressing the knowledge boundaries of black-box LLMs. Based on the knowledge distillation framework, this method designs a deep learning model. Taking the input question, output answer, and token probability from a black-box LLM as inputs, it constructs a mapping between the inputs and the model' internal knowledge state, enabling the quantification and expression of the black-box LLM' knowledge boundaries. Experiments conducted on diverse public datasets and with multiple prominent black-box LLMs demonstrate that LSCL effectively assists black-box LLMs in accurately expressing their knowledge boundaries. It significantly outperforms existing baseline models on metrics such as accuracy and recall rate. Furthermore, considering scenarios where some black-box LLMs do not support access to token probability, an adaptive alternative method is proposed. The performance of this alternative approach is close to that of LSCL and surpasses baseline models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10778v1">GoodVibe: Security-by-Vibe for LLM-Based Code Generation</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used for code generation in fast, informal development workflows, often referred to as vibe coding, where speed and convenience are prioritized, and security requirements are rarely made explicit. In this setting, models frequently produce functionally correct but insecure code, creating a growing security risk. Existing approaches to improving code security rely on full-parameter fine-tuning or parameter-efficient adaptations, which are either costly and prone to catastrophic forgetting or operate at coarse granularity with limited interpretability and control. We present GoodVibe, a neuron-level framework for improving the security of code language models by default. GoodVibe is based on the key insight that security-relevant reasoning is localized to a small subset of neurons. We identify these neurons using gradient-based attribution from a supervised security task and perform neuron-selective fine-tuning that updates only this security-critical subspace. To further reduce training cost, we introduce activation-driven neuron clustering, enabling structured updates with minimal overhead. We evaluate GoodVibe on six LLMs across security-critical programming languages, including C++, Java, Swift, and Go. GoodVibe substantially improves the security of generated code while preserving general model utility, achieving up to a 2.5x improvement over base models, matching or exceeding full fine-tuning with over 4,700x fewer trainable parameters, and reducing training computation by more than 3.6x compared to the parameter-efficient baseline (LoRA). Our results demonstrate that neuron-level optimization offers an effective and scalable approach to securing code generation without sacrificing efficiency or generality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01330v2">Beyond Gemini-3-Pro: Revisiting LLM Routing and Aggregation at Scale</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 21 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have rapidly advanced, with Gemini-3-Pro setting a new performance milestone. In this work, we explore collective intelligence as an alternative to monolithic scaling, and demonstrate that open-source LLMs' collaboration can surpass Gemini-3-Pro. We first revisit LLM routing and aggregation at scale and identify three key bottlenecks: (1) current train-free routers are limited by a query-based paradigm focusing solely on textual similarity; (2) recent aggregation methods remain largely static, failing to select appropriate aggregators for different tasks;(3) the complementarity of routing and aggregation remains underutilized. To address these problems, we introduce JiSi, a novel framework designed to release the full potential of LLMs' collaboration through three innovations: (1) Query-Response Mixed Routing capturing both semantic information and problem difficulty; (2) Support-Set-based Aggregator Selection jointly evaluating the aggregation and domain capacity of aggregators; (3) Adaptive Routing-Aggregation Switch dynamically leveraging the advantages of routing and aggregation. Comprehensive experiments on nine benchmarks demonstrate that JiSi can surpass Gemini-3-Pro with only 47% costs by orchestrating ten open-source LLMs, while outperforming mainstream baselines. It suggests that collective intelligence represents a novel path towards Artificial General Intelligence (AGI).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10729v1">BOute: Cost-Efficient LLM Serving with Heterogeneous LLMs and GPUs via Multi-Objective Bayesian Optimization</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 MLSys 2026
    </div>
    <details class="paper-abstract">
      The rapid growth of large language model (LLM) deployments has made cost-efficient serving systems essential. Recent efforts to enhance system cost-efficiency adopt two main perspectives: (i) An algorithmic perspective that exploits heterogeneous model capabilities to route simpler queries to lower-cost models and complex queries to higher-cost models (i.e., heterogeneous query routing); and (ii) a systems perspective that utilizes heterogeneous GPU resources as cost-effective alternatives to homogeneous high-end GPUs (i.e., heterogeneous model deployment). However, algorithm-system co-design for cost-efficient LLM serving necessitates sophisticated management: (i) Determining optimal query routing strategies under latency and quality requirements, (ii) configuring model deployment across heterogeneous GPUs with appropriate resource allocation and parallelism strategies, and (iii) co-optimizing routing and deployment decisions to maximize overall system performance. To address these challenges, we present BOute, a quality-aware scheduling system that jointly exploits heterogeneous model and GPU capabilities for cost-efficient LLM serving. BOute employs a multi-objective Bayesian optimization (MOBO) framework to co-optimize the routing strategy and model deployment, thereby maximizing the cost-efficiency of the serving system while guaranteeing response quality. Evaluation results demonstrate that BOute outperforms state-of-the-art LLM serving systems by up to 157% and 59% on average under identical cost budgets and quality requirements, or reducing serving costs by 15%-61% (38% on average) while maintaining the same performance targets, validating its effectiveness in achieving cost-efficient LLM serving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10716v1">RE-LLM: Refining Empathetic Speech-LLM Responses by Integrating Emotion Nuance</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 5 pages, 1 figure, 2 tables. Accepted at IEEE ASRU 2025
    </div>
    <details class="paper-abstract">
      With generative AI advancing, empathy in human-AI interaction is essential. While prior work focuses on emotional reflection, emotional exploration, key to deeper engagement, remains overlooked. Existing LLMs rely on text which captures limited emotion nuances. To address this, we propose RE-LLM, a speech-LLM integrating dimensional emotion embeddings and auxiliary learning. Experiments show statistically significant gains in empathy metrics across three datasets. RE-LLM relatively improves the Emotional Reaction score by 14.79% and 6.76% compared to text-only and speech-LLM baselines on ESD. Notably, it raises the Exploration score by 35.42% and 3.91% on IEMOCAP, 139.28% and 9.83% on ESD, and 60.95% and 22.64% on MSP-PODCAST. It also boosts unweighted accuracy by 5.4% on IEMOCAP, 2.3% on ESD, and 6.9% on MSP-PODCAST in speech emotion recognition. These results highlight the enriched emotional understanding and improved empathetic response generation of RE-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10715v1">Locomo-Plus: Beyond-Factual Cognitive Memory Evaluation Framework for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 16 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Long-term conversational memory is a core capability for LLM-based dialogue systems, yet existing benchmarks and evaluation protocols primarily focus on surface-level factual recall. In realistic interactions, appropriate responses often depend on implicit constraints such as user state, goals, or values that are not explicitly queried later. To evaluate this setting, we introduce \textbf{LoCoMo-Plus}, a benchmark for assessing cognitive memory under cue--trigger semantic disconnect, where models must retain and apply latent constraints across long conversational contexts. We further show that conventional string-matching metrics and explicit task-type prompting are misaligned with such scenarios, and propose a unified evaluation framework based on constraint consistency. Experiments across diverse backbone models, retrieval-based methods, and memory systems demonstrate that cognitive memory remains challenging and reveals failures not captured by existing benchmarks. Our code and evaluation framework are publicly available at: https://github.com/xjtuleeyf/Locomo-Plus.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.00402v2">A Conditional Companion: Lived Experiences of People with Mental Health Disorders Using LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 Accepted for presentation at CHI 2026 in Barcelona (ACM CHI Conference on Human Factors in Computing Systems)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used for mental health support, yet little is known about how people with mental health challenges engage with them, how they evaluate their usefulness, and what design opportunities they envision. We conducted 20 semi-structured interviews with people in the UK who live with mental health conditions and have used LLMs for mental health support. Through reflexive thematic analysis, we found that participants engaged with LLMs in conditional and situational ways: for immediacy, the desire for non-judgement, self-paced disclosure, cognitive reframing, and relational engagement. Simultaneously, participants articulated clear boundaries informed by prior therapeutic experience: LLMs were effective for mild-to-moderate distress but inadequate for crises, trauma, and complex social-emotional situations. We contribute empirical insights into the lived use of LLMs for mental health, highlight boundary-setting as central to their safe role, and propose design and governance directions for embedding them responsibly within care ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10693v1">VESPO: Variational Sequence-Level Soft Policy Optimization for Stable Off-Policy LLM Training</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Training stability remains a central challenge in reinforcement learning (RL) for large language models (LLMs). Policy staleness, asynchronous training, and mismatches between training and inference engines all cause the behavior policy to diverge from the current policy, risking training collapse. Importance sampling provides a principled correction for this distribution shift but suffers from high variance; existing remedies such as token-level clipping and sequence-level normalization lack a unified theoretical foundation. We propose Variational sEquence-level Soft Policy Optimization (VESPO). By incorporating variance reduction into a variational formulation over proposal distributions, VESPO derives a closed-form reshaping kernel that operates directly on sequence-level importance weights without length normalization. Experiments on mathematical reasoning benchmarks show that VESPO maintains stable training under staleness ratios up to 64x and fully asynchronous execution, and delivers consistent gains across both dense and Mixture-of-Experts models. Code is available at https://github.com/FloyedShen/VESPO
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10684v1">Privacy Control in Conversational LLM Platforms: A Walkthrough Study</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly integrated into daily life through conversational interfaces, processing user data via natural language inputs and exhibiting advanced reasoning capabilities, which raises new concerns about user control over privacy. While much research has focused on potential privacy risks, less attention has been paid to the data control mechanisms these platforms provide. This study examines six conversational LLM platforms, analyzing how they define and implement features for users to access, edit, delete, and share data. Our analysis reveals an emerging paradigm of data control in conversational LLM platforms, where user data is generated and derived through interaction itself, natural language enables flexible yet often ambiguous control, and multi-user interactions with shared data raise questions of co-ownership and governance. Based on these findings, we offer practical insights for platform developers, policymakers, and researchers to design more effective and usable privacy controls in LLM-powered conversational interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11215v1">Charting Empirical Laws for LLM Fine-Tuning in Scientific Multi-Discipline Learning</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have achieved strong performance through fine-tuning within individual scientific domains, their learning dynamics in multi-disciplinary contexts remains poorly understood, despite the promise of improved generalization and broader applicability through cross-domain knowledge synergy. In this work, we present the first systematic study of multi-disciplinary LLM fine-tuning, constructing a five-discipline corpus and analyzing learning patterns of full fine-tuning, LoRA, LoRA-MoE, and LoRA compositions. Particularly, our study shows that multi-disciplinary learning is substantially more variable than single-discipline training and distills four consistent empirical laws: (1) Balance-then-Diversity: low-resource disciplines degrade performance unless mitigated via diversity-aware upsampling; (2) Merge-then-Align: restoring instruction-following ability is critical for cross-discipline synergy; (3) Optimize-then-Scale: parameter scaling offers limited gains without prior design optimization; and (4) Share-then-Specialize: asymmetric LoRA-MoE yields robust gains with minimal trainable parameters via shared low-rank projection. Together, these laws form a practical recipe for principled multi-discipline fine-tuning and provide actionable guidance for developing generalizable scientific LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09514v2">EcoGym: Evaluating LLMs for Long-Horizon Plan-and-Execute in Interactive Economies</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 work in progress
    </div>
    <details class="paper-abstract">
      Long-horizon planning is widely recognized as a core capability of autonomous LLM-based agents; however, current evaluation frameworks suffer from being largely episodic, domain-specific, or insufficiently grounded in persistent economic dynamics. We introduce EcoGym, a generalizable benchmark for continuous plan-and-execute decision making in interactive economies. EcoGym comprises three diverse environments: Vending, Freelance, and Operation, implemented in a unified decision-making process with standardized interfaces, and budgeted actions over an effectively unbounded horizon (1000+ steps if 365 day-loops for evaluation). The evaluation of EcoGym is based on business-relevant outcomes (e.g., net worth, income, and DAU), targeting long-term strategic coherence and robustness under partial observability and stochasticity. Experiments across eleven leading LLMs expose a systematic tension: no single model dominates across all three scenarios. Critically, we find that models exhibit significant suboptimality in either high-level strategies or efficient actions executions. EcoGym is released as an open, extensible testbed for transparent long-horizon agent evaluation and for studying controllability-utility trade-offs in realistic economic settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2310.11409v7">LLMs as Hackers: Autonomous Linux Privilege Escalation Attacks</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Penetration-testing is crucial for identifying system vulnerabilities, with privilege-escalation being a critical subtask to gain elevated access to protected resources. Language Models (LLMs) presents new avenues for automating these security practices by emulating human behavior. However, a comprehensive understanding of LLMs' efficacy and limitations in performing autonomous Linux privilege-escalation attacks remains under-explored. To address this gap, we introduce hackingBuddyGPT, a fully automated LLM-driven prototype designed for autonomous Linux privilege-escalation. We curated a novel, publicly available Linux privilege-escalation benchmark, enabling controlled and reproducible evaluation. Our empirical analysis assesses the quantitative success rates and qualitative operational behaviors of various LLMs -- GPT-3.5-Turbo, GPT-4-Turbo, and Llama3 -- against baselines of human professional pen-testers and traditional automated tools. We investigate the impact of context management strategies, different context sizes, and various high-level guidance mechanisms on LLM performance. Results show that GPT-4-Turbo demonstrates high efficacy, successfully exploiting 33-83% of vulnerabilities, a performance comparable to human pen-testers (75%). In contrast, local models like Llama3 exhibited limited success (0-33%), and GPT-3.5-Turbo achieved moderate rates (16-50%). We show that both high-level guidance and state-management through LLM-driven reflection significantly boost LLM success rates. Qualitative analysis reveals both LLMs' strengths and weaknesses in generating valid commands and highlights challenges in common-sense reasoning, error handling, and multi-step exploitation, particularly with temporal dependencies. Cost analysis indicates that GPT-4-Turbo can achieve human-comparable performance at competitive costs, especially with optimized context management.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10622v1">How Do Decoder-Only LLMs Perceive Users? Rethinking Attention Masking for User Representation Learning</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 13 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Decoder-only large language models are increasingly used as behavioral encoders for user representation learning, yet the impact of attention masking on the quality of user embeddings remains underexplored. In this work, we conduct a systematic study of causal, hybrid, and bidirectional attention masks within a unified contrastive learning framework trained on large-scale real-world Alipay data that integrates long-horizon heterogeneous user behaviors. To improve training dynamics when transitioning from causal to bidirectional attention, we propose Gradient-Guided Soft Masking, a gradient-based pre-warmup applied before a linear scheduler that gradually opens future attention during optimization. Evaluated on 9 industrial user cognition benchmarks covering prediction, preference, and marketing sensitivity tasks, our approach consistently yields more stable training and higher-quality bidirectional representations compared with causal, hybrid, and scheduler-only baselines, while remaining compatible with decoder pretraining. Overall, our findings highlight the importance of masking design and training transition in adapting decoder-only LLMs for effective user representation learning. Our code is available at https://github.com/JhCircle/Deepfind-GGSM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10620v1">ISD-Agent-Bench: A Comprehensive Benchmark for Evaluating LLM-based Instructional Design Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents have shown promising potential in automating Instructional Systems Design (ISD), a systematic approach to developing educational programs. However, evaluating these agents remains challenging due to the lack of standardized benchmarks and the risk of LLM-as-judge bias. We present ISD-Agent-Bench, a comprehensive benchmark comprising 25,795 scenarios generated via a Context Matrix framework that combines 51 contextual variables across 5 categories with 33 ISD sub-steps derived from the ADDIE model. To ensure evaluation reliability, we employ a multi-judge protocol using diverse LLMs from different providers, achieving high inter-judge reliability. We compare existing ISD agents with novel agents grounded in classical ISD theories such as ADDIE, Dick \& Carey, and Rapid Prototyping ISD. Experiments on 1,017 test scenarios demonstrate that integrating classical ISD frameworks with modern ReAct-style reasoning achieves the highest performance, outperforming both pure theory-based agents and technique-only approaches. Further analysis reveals that theoretical quality strongly correlates with benchmark performance, with theory-based agents showing significant advantages in problem-centered design and objective-assessment alignment. Our work provides a foundation for systematic LLM-based ISD research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.02742v2">Entropy-Guided Dynamic Tokens for Graph-LLM Alignment in Molecular Understanding</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 Accepted by ICLR 2026
    </div>
    <details class="paper-abstract">
      Molecular understanding is central to advancing areas such as scientific discovery, yet Large Language Models (LLMs) struggle to understand molecular graphs effectively. Existing graph-LLM bridges often adapt the Q-Former-style connector with fixed-length static tokens, which is originally designed for vision tasks. These designs overlook stereochemistry and substructural context and typically require costly LLM-backbone fine-tuning, limiting efficiency and generalization. We introduce EDT-Former, an Entropy-guided Dynamic Token Transformer that generates tokens aligned with informative molecular patches, thereby preserving both local and global structural features for molecular graph understanding. Beyond prior approaches, EDT-Former enables alignment between frozen graph encoders and LLMs without tuning the LLM backbone (excluding the embedding layer), resulting in computationally efficient finetuning, and achieves stateof-the-art results on MoleculeQA, Molecule-oriented Mol-Instructions, and property prediction benchmarks (TDC, MoleculeNet), underscoring its effectiveness for scalable and generalizable multimodal molecular understanding
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10577v1">Campaign-2-PT-RAG: LLM-Guided Semantic Product Type Attribution for Scalable Campaign Ranking</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      E-commerce campaign ranking models require large-scale training labels indicating which users purchased due to campaign influence. However, generating these labels is challenging because campaigns use creative, thematic language that does not directly map to product purchases. Without clear product-level attribution, supervised learning for campaign optimization remains limited. We present \textbf{Campaign-2-PT-RAG}, a scalable label generation framework that constructs user--campaign purchase labels by inferring which product types (PTs) each campaign promotes. The framework first interprets campaign content using large language models (LLMs) to capture implicit intent, then retrieves candidate PTs through semantic search over the platform taxonomy. A structured LLM-based classifier evaluates each PT's relevance, producing a campaign-specific product coverage set. User purchases matching these PTs generate positive training labels for downstream ranking models. This approach reframes the ambiguous attribution problem into a tractable semantic alignment task, enabling scalable and consistent supervision for downstream tasks such as campaign ranking optimization in production e-commerce environments. Experiments on internal and synthetic datasets, validated against expert-annotated campaign--PT mappings, show that our LLM-assisted approach generates high-quality labels with 78--90% precision while maintaining over 99% recall.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10576v1">LLM-Based Scientific Equation Discovery via Physics-Informed Token-Regularized Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Symbolic regression aims to distill mathematical equations from observational data. Recent approaches have successfully leveraged Large Language Models (LLMs) to generate equation hypotheses, capitalizing on their vast pre-trained scientific priors. However, existing frameworks predominantly treat the LLM as a static generator, relying on prompt-level guidance to steer exploration. This paradigm fails to update the model's internal representations based on search feedback, often yielding physically inconsistent or mathematically redundant expressions. In this work, we propose PiT-PO (Physics-informed Token-regularized Policy Optimization), a unified framework that evolves the LLM into an adaptive generator via reinforcement learning. Central to PiT-PO is a dual-constraint mechanism that rigorously enforces hierarchical physical validity while simultaneously applying fine-grained, token-level penalties to suppress redundant structures. Consequently, PiT-PO aligns LLM to produce equations that are both scientifically consistent and structurally parsimonious. Empirically, PiT-PO achieves state-of-the-art performance on standard benchmarks and successfully discovers novel turbulence models for challenging fluid dynamics problems. We also demonstrate that PiT-PO empowers small-scale models to outperform closed-source giants, democratizing access to high-performance scientific discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10568v1">Gauss-Newton Unlearning for the LLM Era</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      Standard large language model training can create models that produce outputs their trainer deems unacceptable in deployment. The probability of these outputs can be reduced using methods such as LLM unlearning. However, unlearning a set of data (called the forget set) can degrade model performance on other distributions where the trainer wants to retain the model's behavior. To improve this trade-off, we demonstrate that using the forget set to compute only a few uphill Gauss-Newton steps provides a conceptually simple, state-of-the-art unlearning approach for LLMs. While Gauss-Newton steps adapt Newton's method to non-linear models, it is non-trivial to efficiently and accurately compute such steps for LLMs. Hence, our approach crucially relies on parametric Hessian approximations such as Kronecker-Factored Approximate Curvature (K-FAC). We call this combined approach K-FADE (K-FAC for Distribution Erasure). Our evaluation on the WMDP and ToFU benchmarks demonstrates that K-FADE suppresses outputs from the forget set and approximates, in output space, the results of retraining without the forget set. Critically, our method does this while altering the outputs on the retain set less than previous methods. This is because K-FADE transforms a constraint on the model's outputs across the entire retain set into a constraint on the model's weights, allowing the algorithm to minimally change the model's behavior on the retain set at each step. Moreover, the unlearning updates computed by K-FADE can be reapplied later if the model undergoes further training, allowing unlearning to be cheaply maintained.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.17512v2">Is Your LLM Really Mastering the Concept? A Multi-Agent Benchmark</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      Concepts serve as fundamental abstractions that support human reasoning and categorization. However, it remains unclear whether large language models truly capture such conceptual structures or primarily rely on surface-level pattern memorization. Existing benchmarks are largely static and fact oriented, which limits their ability to probe fine-grained semantic understanding and makes them vulnerable to data leakage and overfitting. To address this limitation, we introduce CK-Arena, a dynamic benchmark for conceptual knowledge evaluation based on a multi agent social deduction game, namely the Undercover game. In this setting, LLM based agents are assigned subtly different concept words and must describe, distinguish, and infer conceptual properties from others' statements. Model performance is evaluated through both game level outcomes and the semantic quality of generated descriptions. Furthermore, CK-Arena leverages the interaction process to automatically construct high quality question answering data for fine grained diagnostic analysis. Experimental results show that conceptual understanding varies substantially across models and categories, and is not strictly aligned with overall model capability. The data and code are available at the project homepage: https://ck-arena.site.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.09651v2">Geospatial Representation Learning: A Survey from Deep Learning to The LLM Era</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      The ability to transform location-centric geospatial data into meaningful computational representations has become fundamental to modern spatial analysis and decision-making. Geospatial Representation Learning (GRL), the process of automatically extracting latent structures and semantic patterns from geographic data, is undergoing a profound transformation through two successive technological revolutions: the deep learning breakthrough and the emerging large language model (LLM) paradigm. While deep neural networks (DNNs) have demonstrated remarkable success in automated feature extraction from structured and semi-structured geospatial data (e.g., satellite imagery, GPS trajectories), the recent integration of LLMs introduces transformative capabilities for cross-modal geospatial reasoning and unstructured geo-textual data processing. This survey presents a comprehensive review of geospatial representation learning across both technological eras, organizing them into a structured taxonomy based on the complete pipeline comprising: (1) data perspective, (2) methodological perspective, and (3) application perspective. We also highlight current advancements, discuss existing limitations, and propose potential future research directions in the LLM and foundation model era. This work offers a thorough exploration of the field and provides a roadmap for further innovation in GRL. The summary of the up-to-date paper list can be found in https://github.com/CityMind-Lab/Awesome-Geospatial-Representation-Learning and will undergo continuous updates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.12365v4">Advances in LLMs with Focus on Reasoning, Adaptability, Efficiency and Ethics</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      This survey paper outlines the key developments in the field of Large Language Models (LLMs), including enhancements to their reasoning skills, adaptability to various tasks, increased computational efficiency, and the ability to make ethical decisions. The techniques that have been most effective in bridging the gap between human and machine communications include the Chain-of-Thought prompting, Instruction Tuning, and Reinforcement Learning from Human Feedback. The improvements in multimodal learning and few-shot or zero-shot techniques have further empowered LLMs to handle complex jobs with minor input. A significant focus is placed on efficiency, detailing scaling strategies, optimization techniques, and the influential Mixture-of-Experts (MoE) architecture, which strategically routes inputs to specialized subnetworks to boost predictive accuracy, while optimizing resource allocation. This survey also offers a broader perspective on recent advancements in LLMs, going beyond isolated aspects such as model architecture or ethical concerns. Additionally, it explores the role of LLMs in Agentic AI and their use as Autonomous Decision-Making Systems, and categorizes emerging methods that enhance LLM reasoning, efficiency, and ethical alignment. The survey also identifies underexplored areas such as interpretability, cross-modal integration, and sustainability. While significant advancements have been made in LLMs, challenges such as high computational costs, biases, and ethical risks remain. Overcoming these requires a focus on bias mitigation, transparent decision-making, and explicit ethical guidelines. Future research will generally focus on enhancing the model's ability to handle multiple inputs, thereby making it more intelligent, safe, and reliable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.23048v2">From Abstract to Contextual: What LLMs Still Cannot Do in Mathematics</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      Large language models now solve many benchmark math problems at near-expert levels, yet this progress has not fully translated into reliable performance in real-world applications. We study this gap through contextual mathematical reasoning, where the mathematical core must be formulated from descriptive scenarios. We introduce ContextMATH, a benchmark that repurposes AIME and MATH-500 problems into two contextual settings: Scenario Grounding (SG), which embeds abstract problems into realistic narratives without increasing reasoning complexity, and Complexity Scaling (CS), which transforms explicit conditions into sub-problems to capture how constraints often appear in practice. Evaluating 61 proprietary and open-source models, we observe sharp drops: on average, open-source models decline by 13 and 34 points on SG and CS, while proprietary models drop by 13 and 20. Error analysis shows that errors are dominated by incorrect problem formulation, with formulation accuracy declining as original problem difficulty increases. Correct formulation emerges as a prerequisite for success, and its sufficiency improves with model scale, indicating that larger models advance in both understanding and reasoning. Nevertheless, formulation and reasoning remain two complementary bottlenecks that limit contextual mathematical problem solving. Finally, we find that fine-tuning with scenario data improves performance, whereas formulation-only training is ineffective. However, performance gaps are only partially alleviated, highlighting contextual mathematical reasoning as a central unsolved challenge for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10512v1">Don't Eliminate Cut: Exponential Separations in LLM-Based Theorem Proving</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      We develop a theoretical analysis of LLM-guided formal theorem proving in interactive proof assistants (e.g., Lean) by modeling tactic proposal as a stochastic policy in a finite-horizon deterministic MDP. To capture modern representation learning, we treat the state and action spaces as general compact metric spaces and assume Lipschitz policies. To explain the gap between worst-case hardness and empirical success, we introduce problem distributions generated by a reference policy $q$, including a latent-variable model in which proofs exhibit reusable cut/lemma/sketch structure represented by a proof DAG. Under a top-$k$ search protocol and Tsybakov-type margin conditions, we derive lower bounds on finite-horizon success probability that decompose into search and learning terms, with learning controlled by sequential Rademacher/covering complexity. Our main separation result shows that when cut elimination expands a DAG of depth $D$ into a cut-free tree of size $Ω(Λ^D)$ while the cut-aware hierarchical process has size $O(λ^D)$ with $λ\llΛ$, a flat (cut-free) learner provably requires exponentially more data than a cut-aware hierarchical learner. This provides a principled justification for subgoal decomposition in recent agentic theorem provers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10498v1">When Skills Lie: Hidden-Comment Injection in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 4 pages
    </div>
    <details class="paper-abstract">
      LLM agents often rely on Skills to describe available tools and recommended procedures. We study a hidden-comment prompt injection risk in this documentation layer: when a Markdown Skill is rendered to HTML, HTML comment blocks can become invisible to human reviewers, yet the raw text may still be supplied verbatim to the model. In experiments, we find that DeepSeek-V3.2 and GLM-4.5-Air can be influenced by malicious instructions embedded in a hidden comment appended to an otherwise legitimate Skill, yielding outputs that contain sensitive tool intentions. A short defensive system prompt that treats Skills as untrusted and forbids sensitive actions prevents these malicious tool calls and instead surfaces the suspicious hidden instructions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05252v2">Copyright Detective: A Forensic System to Evidence LLMs Flickering Copyright Leakage Risks</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      We present Copyright Detective, the first interactive forensic system for detecting, analyzing, and visualizing potential copyright risks in LLM outputs. The system treats copyright infringement versus compliance as an evidence discovery process rather than a static classification task due to the complex nature of copyright law. It integrates multiple detection paradigms, including content recall testing, paraphrase-level similarity analysis, persuasive jailbreak probing, and unlearning verification, within a unified and extensible framework. Through interactive prompting, response collection, and iterative workflows, our system enables systematic auditing of verbatim memorization and paraphrase-level leakage, supporting responsible deployment and transparent evaluation of LLM copyright risks even with black-box access.
    </details>
</div>
