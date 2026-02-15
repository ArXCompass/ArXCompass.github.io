# llm - 2026_02

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.12259v1">Think like a Scientist: Physics-guided LLM Agent for Equation Discovery</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      Explaining observed phenomena through symbolic, interpretable formulas is a fundamental goal of science. Recently, large language models (LLMs) have emerged as promising tools for symbolic equation discovery, owing to their broad domain knowledge and strong reasoning capabilities. However, most existing LLM-based systems try to guess equations directly from data, without modeling the multi-step reasoning process that scientists often follow: first inferring physical properties such as symmetries, then using these as priors to restrict the space of candidate equations. We introduce KeplerAgent, an agentic framework that explicitly follows this scientific reasoning process. The agent coordinates physics-based tools to extract intermediate structure and uses these results to configure symbolic regression engines such as PySINDy and PySR, including their function libraries and structural constraints. Across a suite of physical equation benchmarks, KeplerAgent achieves substantially higher symbolic accuracy and greater robustness to noisy data than both LLM and traditional baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.18134v2">Evaluating LLM Reasoning Beyond Correctness and CoT</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      What does it truly mean for a language model to "reason"? Current evaluations reward models' correct standalone answers-but correctness alone reveals little about the process that produced them. We argue that reasoning should be understood not as a static chain of steps but as a dynamic trajectory in which ideas interact, clash, and evolve into integrated insights. Building on the philosophical tradition of dialectics, we introduce SIEV, a structured evaluation framework that assesses reasoning through explicit thesis-antithesis-synthesis interactions. SIEV produces interpretable trajectories that highlight key properties of reasoning-robustness to challenge, adaptability under conflict, and synthesis across competing viewpoints-dimensions that conventional correctness-based metrics cannot capture. Empirical results on GSM and MMLU demonstrate substantial gaps in the reasoning abilities of state-of-the-art models: for example, GPT-5-chat loses more than 40 points (out of 100) on GSM when evaluated through SIEV's process-oriented lens. By shifting focus from what answer a model gives to how it arrives there, SIEV enables a more transparent and principled distinction between structured reasoning and surface-level pattern generation offering a clearer foundation for assessing and understanding the reasoning capabilities of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.05506v2">When Scaffolding Breaks: Investigating Student Interaction with LLM-Based Writing Support in Real-Time K-12 EFL Classrooms</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are promising tools for scaffolding students' English writing skills, but their effectiveness in real-time K-12 classrooms remains underexplored. Addressing this gap, our study examines the benefits and limitations of using LLMs as real-time learning support, considering how classroom constraints, such as diverse proficiency levels and limited time, affect their effectiveness. We conducted a deployment study with 157 eighth-grade students in a South Korean middle school English class over six weeks. Our findings reveal that while scaffolding improved students' ability to compose grammatically correct sentences, this step-by-step approach demotivated lower-proficiency students and increased their system reliance. We also observed challenges to classroom dynamics, where extroverted students often dominated the teacher's attention, and the system's assistance made it difficult for teachers to identify struggling students. Based on these findings, we discuss design guidelines for integrating LLMs into real-time writing classes as inclusive educational tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.22075v3">CoSpaDi: Compressing LLMs via Calibration-Guided Sparse Dictionary Learning</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      Post-training compression of large language models (LLMs) often relies on low-rank weight approximations that represent each column of the weight matrix in a shared low-dimensional subspace. This strategy is computationally efficient but the underlying constraint can be overly rigid for heterogeneous projection weights and may incur avoidable accuracy loss. We propose CoSpaDi (Compression via Sparse Dictionary Learning), a training-free framework that replaces low-rank factorization with a structured sparse decomposition in which each weight matrix is represented as a dense dictionary multiplied by a column-sparse coefficient matrix. This yields a union-of-subspaces model: the columns of the weight matrix are represented as linear combinations of different subsets of dictionary atoms, improving expressiveness at a fixed parameter budget. CoSpaDi is calibration-guided: using a small calibration set, we optimize the factorization to minimize functional reconstruction error of layer outputs rather than weight-space error. An activation-derived Gram orthonormalization reformulates this data-aware objective into a standard dictionary learning problem on transformed weights, and we support both per-layer compression and cross-layer dictionary sharing within groups of similar projections. Across Llama and Qwen model families, CoSpaDi consistently improves the accuracy--compression and perplexity--compression trade-offs over state-of-the-art SVD-based baselines and strong structured pruning baselines at 20-40\% compression ratios. The resulting structured sparsity enables sparse--dense computation and integrates with post-training quantization of the sparse coefficients.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.14182v4">LabSafety Bench: Benchmarking LLMs on Safety Issues in Scientific Labs</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 Published at Nature Machine Intelligence
    </div>
    <details class="paper-abstract">
      Artificial Intelligence (AI) is revolutionizing scientific research, yet its growing integration into laboratory environments presents critical safety challenges. Large language models (LLMs) and vision language models (VLMs) now assist in experiment design and procedural guidance, yet their "illusion of understanding" may lead researchers to overtrust unsafe outputs. Here we show that current models remain far from meeting the reliability needed for safe laboratory operation. We introduce LabSafety Bench, a comprehensive benchmark that evaluates models on hazard identification, risk assessment, and consequence prediction across 765 multiple-choice questions and 404 realistic lab scenarios, encompassing 3,128 open-ended tasks. Evaluations on 19 advanced LLMs and VLMs show that no model evaluated on hazard identification surpasses 70% accuracy. While proprietary models perform well on structured assessments, they do not show a clear advantage in open-ended reasoning. These results underscore the urgent need for specialized safety evaluation frameworks before deploying AI systems in real laboratory settings.
    </details>
</div>
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
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11444v1">Towards Reliable Machine Translation: Scaling LLMs for Critical Error Detection and Safety</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 Accepted at ECIR 2026
    </div>
    <details class="paper-abstract">
      Machine Translation (MT) plays a pivotal role in cross-lingual information access, public policy communication, and equitable knowledge dissemination. However, critical meaning errors, such as factual distortions, intent reversals, or biased translations, can undermine the reliability, fairness, and safety of multilingual systems. In this work, we explore the capacity of instruction-tuned Large Language Models (LLMs) to detect such critical errors, evaluating models across a range of parameters using the publicly accessible data sets. Our findings show that model scaling and adaptation strategies (zero-shot, few-shot, fine-tuning) yield consistent improvements, outperforming encoder-only baselines like XLM-R and ModernBERT. We argue that improving critical error detection in MT contributes to safer, more trustworthy, and socially accountable information systems by reducing the risk of disinformation, miscommunication, and linguistic harm, especially in high-stakes or underrepresented contexts. This work positions error detection not merely as a technical challenge, but as a necessary safeguard in the pursuit of just and responsible multilingual AI. The code will be made available at GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11388v1">Sparse Semantic Dimension as a Generalization Certificate for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 Work in progress (17 pages)
    </div>
    <details class="paper-abstract">
      Standard statistical learning theory predicts that Large Language Models (LLMs) should overfit because their parameter counts vastly exceed the number of training tokens. Yet, in practice, they generalize robustly. We propose that the effective capacity controlling generalization lies in the geometry of the model's internal representations: while the parameter space is high-dimensional, the activation states lie on a low-dimensional, sparse manifold. To formalize this, we introduce the Sparse Semantic Dimension (SSD), a complexity measure derived from the active feature vocabulary of a Sparse Autoencoder (SAE) trained on the model's layers. Treating the LLM and SAE as frozen oracles, we utilize this framework to attribute the model's generalization capabilities to the sparsity of the dictionary rather than the total parameter count. Empirically, we validate this framework on GPT-2 Small and Gemma-2B, demonstrating that our bound provides non-vacuous certificates at realistic sample sizes. Crucially, we uncover a counter-intuitive "feature sharpness" scaling law: despite being an order of magnitude larger, Gemma-2B requires significantly fewer calibration samples to identify its active manifold compared to GPT-2, suggesting that larger models learn more compressible, distinct semantic structures. Finally, we show that this framework functions as a reliable safety monitor: out-of-distribution inputs trigger a measurable "feature explosion" (a sharp spike in active features), effectively signaling epistemic uncertainty through learned feature violation. Code is available at: https://github.com/newcodevelop/sparse-semantic-dimension.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19905v2">Demystifying LLM-as-a-Judge: Analytically Tractable Model for Inference-Time Scaling</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 27 pages
    </div>
    <details class="paper-abstract">
      Recent developments in large language models have shown advantages in reallocating a notable share of computational resource from training time to inference time. However, the principles behind inference time scaling are not well understood. In this paper, we introduce an analytically tractable model of inference-time scaling: Bayesian linear regression with a reward-weighted sampler, where the reward is determined from a linear model, modeling LLM-as-a-judge scenario. We study this problem in the high-dimensional regime, where the deterministic equivalents dictate a closed-form expression for the posterior predictive mean and variance. We analyze the generalization error when training data are sampled from a teacher model. We draw $k$ inference-time samples and select via softmax at a temperature applied to a quadratic reward. When the reward is not too different from the teacher, the generalization error decreases monotonically with increasing inference time samples $k$. However, the specific reward that optimizes inference-time selection generally differs from the teacher. In contrast, substantial reward misspecification induces a finite optimal $k$ beyond which more sampling can increase the generalization error. For fixed $k$, there exists an optimal sampling temperature. We experimentally verify these facts in large language model inference with an additional large language model as a judge. In the "best-of-$k$" limit with the teacher as reward, we theoretically show that the generalization error decays as $Θ(1/k^2)$ and determine the leading coefficient via extreme value theory. These formulas delineate domains where scaling inference-time computation is provably preferable to collecting more data. Finally, we demonstrate that when task difficulty increases, the previously mentioned advantage of inference-time compute degrades.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09316v2">Effective MoE-based LLM Compression by Exploiting Heterogeneous Inter-Group Experts Routing Frequency and Information Density</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Mixture-of-Experts (MoE) based Large Language Models (LLMs) have achieved superior performance, yet the massive memory overhead caused by storing multiple expert networks severely hinders their practical deployment. Singular Value Decomposition (SVD)-based compression has emerged as a promising post-training technique; however, most existing methods apply uniform rank allocation or rely solely on static weight properties. This overlooks the substantial heterogeneity in expert utilization observed in MoE models, where frequent routing patterns and intrinsic information density vary significantly across experts. In this work, we propose RFID-MoE, an effective framework for MoE compression by exploiting heterogeneous Routing Frequency and Information Density. We first introduce a fused metric that combines expert activation frequency with effective rank to measure expert importance, adaptively allocating higher ranks to critical expert groups under a fixed budget. Moreover, instead of discarding compression residuals, we reconstruct them via a parameter-efficient sparse projection mechanism to recover lost information with minimal parameter overhead. Extensive experiments on representative MoE LLMs (e.g., Qwen3, DeepSeekMoE) across multiple compression ratios demonstrate that RFID-MoE consistently outperforms state-of-the-art methods like MoBE and D2-MoE. Notably, RFID-MoE achieves a perplexity of 16.92 on PTB with the Qwen3-30B model at a 60% compression ratio, reducing perplexity by over 8.0 compared to baselines, and improves zero-shot accuracy on HellaSwag by approximately 8%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.04860v2">Alignment Tipping Process: How Self-Evolution Pushes LLM Agents Off the Rails</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      As Large Language Model (LLM) agents increasingly gain self-evolutionary capabilities to adapt and refine their strategies through real-world interaction, their long-term reliability becomes a critical concern. We identify the Alignment Tipping Process (ATP), a critical post-deployment risk unique to self-evolving LLM agents. Unlike training-time failures, ATP arises when continual interaction drives agents to abandon alignment constraints established during training in favor of reinforced, self-interested strategies. We formalize and analyze ATP through two complementary paradigms: Self-Interested Exploration, where repeated high-reward deviations induce individual behavioral drift, and Imitative Strategy Diffusion, where deviant behaviors spread across multi-agent systems. Building on these paradigms, we construct controllable testbeds and benchmark both open and closed-source LLMs. Our experiments show that alignment benefits erode rapidly under self-evolution, with initially aligned models converging toward unaligned states. In multi-agent settings, successful violations diffuse quickly, leading to collective misalignment. Moreover, current reinforcement learning-based alignment methods provide limited defenses against alignment tipping. These findings demonstrate that alignment of LLM agents is not a static property but a fragile and dynamic one, vulnerable to feedback-driven decay during deployment. Our data and code are available at https://github.com/aiming-lab/ATP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11361v1">Finding the Cracks: Improving LLMs Reasoning with Paraphrastic Probing and Consistency Verification</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Large language models have demonstrated impressive performance across a variety of reasoning tasks. However, their problem-solving ability often declines on more complex tasks due to hallucinations and the accumulation of errors within these intermediate steps. Recent work has introduced the notion of critical tokens--tokens in the reasoning process that exert significant influence on subsequent steps. Prior studies suggest that replacing critical tokens can refine reasoning trajectories. Nonetheless, reliably identifying and exploiting critical tokens remains challenging. To address this, we propose the Paraphrastic Probing and Consistency Verification~(PPCV) framework. PPCV operates in two stages. In the first stage, we roll out an initial reasoning path from the original question and then concatenate paraphrased versions of the question with this reasoning path. And we identify critical tokens based on mismatches between the predicted top-1 token and the expected token in the reasoning path. A criterion is employed to confirm the final critical token. In the second stage, we substitute critical tokens with candidate alternatives and roll out new reasoning paths for both the original and paraphrased questions. The final answer is determined by checking the consistency of outputs across these parallel reasoning processes. We evaluate PPCV on mainstream LLMs across multiple benchmarks. Extensive experiments demonstrate PPCV substantially enhances the reasoning performance of LLMs compared to baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11354v1">ReplicatorBench: Benchmarking LLM Agents for Replicability in Social and Behavioral Sciences</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      The literature has witnessed an emerging interest in AI agents for automated assessment of scientific papers. Existing benchmarks focus primarily on the computational aspect of this task, testing agents' ability to reproduce or replicate research outcomes when having access to the code and data. This setting, while foundational, (1) fails to capture the inconsistent availability of new data for replication as opposed to reproduction, and (2) lacks ground-truth diversity by focusing only on reproducible papers, thereby failing to evaluate an agent's ability to identify non-replicable research. Furthermore, most benchmarks only evaluate outcomes rather than the replication process. In response, we introduce ReplicatorBench, an end-to-end benchmark, including human-verified replicable and non-replicable research claims in social and behavioral sciences for evaluating AI agents in research replication across three stages: (1) extraction and retrieval of replication data; (2) design and execution of computational experiments; and (3) interpretation of results, allowing a test of AI agents' capability to mimic the activities of human replicators in real world. To set a baseline of AI agents' capability, we develop ReplicatorAgent, an agentic framework equipped with necessary tools, like web search and iterative interaction with sandboxed environments, to accomplish tasks in ReplicatorBench. We evaluate ReplicatorAgent across four underlying large language models (LLMs), as well as different design choices of programming language and levels of code access. Our findings reveal that while current LLM agents are capable of effectively designing and executing computational experiments, they struggle with retrieving resources, such as new data, necessary to replicate a claim. All code and data are publicly available at https://github.com/CenterForOpenScience/llm-benchmarking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.00513v2">Minerva: Reinforcement Learning with Verifiable Rewards for Cyber Threat Intelligence LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Cyber threat intelligence (CTI) analysts routinely convert noisy, unstructured security artifacts into standardized, automation-ready representations. Although large language models (LLMs) show promise for this task, existing approaches remain brittle when producing structured CTI outputs and have largely relied on supervised fine-tuning (SFT). In contrast, CTI standards and community-maintained resources define canonical identifiers and schemas that enable deterministic verification of model outputs. We leverage this structure to study reinforcement learning with verifiable rewards (RLVR) for CTI tasks. We introduce \textit{Minerva}, a unified dataset and training pipeline spanning multiple CTI subtasks, each paired with task-specific verifiers that score structured outputs and identifier predictions. To address reward sparsity during rollout, we propose a lightweight self-training mechanism that generates additional verified trajectories and distills them back into the model. Experiments across LLM backbones show consistent improvements in accuracy and robustness over SFT across multiple benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11348v1">AgentNoiseBench: Benchmarking Robustness of Tool-Using LLM Agents Under Noisy Condition</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Recent advances in large language models have enabled LLM-based agents to achieve strong performance on a variety of benchmarks. However, their performance in real-world deployments often that observed on benchmark settings, especially in complex and imperfect environments. This discrepancy largely arises because prevailing training and evaluation paradigms are typically built on idealized assumptions, overlooking the inherent stochasticity and noise present in real-world interactions. To bridge this gap, we introduce AgentNoiseBench, a framework for systematically evaluating the robustness of agentic models under noisy environments. We first conduct an in-depth analysis of biases and uncertainties in real-world scenarios and categorize environmental noise into two primary types: user-noise and tool-noise. Building on this analysis, we develop an automated pipeline that injects controllable noise into existing agent-centric benchmarks while preserving task solvability. Leveraging this pipeline, we perform extensive evaluations across a wide range of models with diverse architectures and parameter scales. Our results reveal consistent performance variations under different noise conditions, highlighting the sensitivity of current agentic models to realistic environmental perturbations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11340v1">Bi-Level Prompt Optimization for Multimodal LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become widely adopted as automated judges for evaluating AI-generated content. Despite their success, aligning LLM-based evaluations with human judgments remains challenging. While supervised fine-tuning on human-labeled data can improve alignment, it is costly and inflexible, requiring new training for each task or dataset. Recent progress in auto prompt optimization (APO) offers a more efficient alternative by automatically improving the instructions that guide LLM judges. However, existing APO methods primarily target text-only evaluations and remain underexplored in multimodal settings. In this work, we study auto prompt optimization for multimodal LLM-as-a-judge, particularly for evaluating AI-generated images. We identify a key bottleneck: multimodal models can only process a limited number of visual examples due to context window constraints, which hinders effective trial-and-error prompt refinement. To overcome this, we propose BLPO, a bi-level prompt optimization framework that converts images into textual representations while preserving evaluation-relevant visual cues. Our bi-level optimization approach jointly refines the judge prompt and the I2T prompt to maintain fidelity under limited context budgets. Experiments on four datasets and three LLM judges demonstrate the effectiveness of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.13811v5">Can LLMs Handle WebShell Detection? Overcoming Detection Challenges with Behavioral Function-Aware Framework</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 Published as a conference paper at COLM 2025 (The new version has been polished and expanded with more detailed future work ideas)
    </div>
    <details class="paper-abstract">
      WebShell attacks - where adversaries implant malicious scripts on web servers - remain a persistent threat. Prior machine-learning and deep-learning detectors typically depend on task-specific supervision and can be brittle under data scarcity, rapid concept drift, and out-of-distribution (OOD) deployment. Large language models (LLMs) have recently shown strong code understanding capabilities, but their reliability for WebShell detection remains unclear. We address this gap by (i) systematically evaluating seven LLMs (including GPT-4, LLaMA-3.1-70B, and Qwen-2.5 variants) against representative sequence- and graph-based baselines on 26.59K PHP scripts, and (ii) proposing Behavioral Function-Aware Detection (BFAD), a behavior-centric framework that adapts LLM inference to WebShell-specific execution patterns. BFAD anchors analysis on security-sensitive PHP functions via a Critical Function Filter, constructs compact LLM inputs with Context-Aware Code Extraction, and selects in-context demonstrations using Weighted Behavioral Function Profiling, which ranks examples by a behavior-weighted, function-level similarity. Empirically, we observe a consistent precision-recall asymmetry: larger LLMs often achieve high precision but miss attacks (lower recall), while smaller models exhibit the opposite tendency; moreover, off-the-shelf LLM prompting underperforms established detectors. BFAD substantially improves all evaluated LLMs, boosting F1 by 13.82% on average; notably, GPT-4, LLaMA-3.1-70B, and Qwen-2.5-Coder-14B exceed prior SOTA benchmarks, while Qwen-2.5-Coder-3B becomes competitive with traditional methods. Overall, our results clarify when LLMs succeed or fail on WebShell detection, provide a practical recipe, and highlight future directions for making LLM-based detection more reliable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10117v2">Biases in the Blind Spot: Detecting What LLMs Fail to Mention</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often provide chain-of-thought (CoT) reasoning traces that appear plausible, but may hide internal biases. We call these *unverbalized biases*. Monitoring models via their stated reasoning is therefore unreliable, and existing bias evaluations typically require predefined categories and hand-crafted datasets. In this work, we introduce a fully automated, black-box pipeline for detecting task-specific unverbalized biases. Given a task dataset, the pipeline uses LLM autoraters to generate candidate bias concepts. It then tests each concept on progressively larger input samples by generating positive and negative variations, and applies statistical techniques for multiple testing and early stopping. A concept is flagged as an unverbalized bias if it yields statistically significant performance differences while not being cited as justification in the model's CoTs. We evaluate our pipeline across six LLMs on three decision tasks (hiring, loan approval, and university admissions). Our technique automatically discovers previously unknown biases in these models (e.g., Spanish fluency, English proficiency, writing formality). In the same run, the pipeline also validates biases that were manually identified by prior work (gender, race, religion, ethnicity). More broadly, our proposed approach provides a practical, scalable path to automatic task-specific bias discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11328v1">Evaluating Alignment of Behavioral Dispositions in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      As LLMs integrate into our daily lives, understanding their behavior becomes essential. In this work, we focus on behavioral dispositions$-$the underlying tendencies that shape responses in social contexts$-$and introduce a framework to study how closely the dispositions expressed by LLMs align with those of humans. Our approach is grounded in established psychological questionnaires but adapts them for LLMs by transforming human self-report statements into Situational Judgment Tests (SJTs). These SJTs assess behavior by eliciting natural recommendations in realistic user-assistant scenarios. We generate 2,500 SJTs, each validated by three human annotators, and collect preferred actions from 10 annotators per SJT, from a large pool of 550 participants. In a comprehensive study involving 25 LLMs, we find that models often do not reflect the distribution of human preferences: (1) in scenarios with low human consensus, LLMs consistently exhibit overconfidence in a single response; (2) when human consensus is high, smaller models deviate significantly, and even some frontier models do not reflect the consensus in 15-20% of cases; (3) traits can exhibit cross-LLM patterns, e.g., LLMs may encourage emotion expression in contexts where human consensus favors composure. Lastly, mapping psychometric statements directly to behavioral scenarios presents a unique opportunity to evaluate the predictive validity of self-reports, revealing considerable gaps between LLMs' stated values and their revealed behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11304v1">CryptoAnalystBench: Failures in Multi-Tool Long-Form LLM Analysis</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Modern analyst agents must reason over complex, high token inputs, including dozens of retrieved documents, tool outputs, and time sensitive data. While prior work has produced tool calling benchmarks and examined factuality in knowledge augmented systems, relatively little work studies their intersection: settings where LLMs must integrate large volumes of dynamic, structured and unstructured multi tool outputs. We investigate LLM failure modes in this regime using crypto as a representative high data density domain. We introduce (1) CryptoAnalystBench, an analyst aligned benchmark of 198 production crypto and DeFi queries spanning 11 categories; (2) an agentic harness equipped with relevant crypto and DeFi tools to generate responses across multiple frontier LLMs; and (3) an evaluation pipeline with citation verification and an LLM as a judge rubric spanning four user defined success dimensions: relevance, temporal relevance, depth, and data consistency. Using human annotation, we develop a taxonomy of seven higher order error types that are not reliably captured by factuality checks or LLM based quality scoring. We find that these failures persist even in state of the art systems and can compromise high stakes decisions. Based on this taxonomy, we refine the judge rubric to better capture these errors. While the judge does not align with human annotators on precise scoring across rubric iterations, it reliably identifies critical failure modes, enabling scalable feedback for developers and researchers studying analyst style agents. We release CryptoAnalystBench with annotated queries, the evaluation pipeline, judge rubrics, and the error taxonomy, and outline mitigation strategies and open challenges in evaluating long form, multi tool augmented systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.19287v2">PALM: Path-aware LLM-based Test Generation with Comprehension</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 11 pages, v2
    </div>
    <details class="paper-abstract">
      Symbolic execution is a widely used technique for test generation, offering systematic exploration of program paths through constraint solving. However, it is fundamentally constrained by the capability to model the target code, including library functions, in terms of symbolic constraints and by the capability of underlying constraint solvers. As a result, many paths involving complex features remain unanalyzed or insufficiently modeled. Recent advances in large language models (LLMs) have shown promise in generating diverse and valid test inputs. Yet, LLMs lack mechanisms for systematically enumerating program paths and often fail to cover subtle corner cases. We observe that directly prompting an LLM with the full program leads to missed coverage of interesting paths. In this paper, we present PALM, a test generation system that combines symbolic path enumeration with LLM-assisted test generation. PALM statically enumerates possible paths through AST-level analysis and transforms each into an executable variant with embedded assertions that specify the target path. This avoids the need to translate path constraints into SMT formulas, by instead constructing program variants that the LLM can interpret. Importantly, PALM provides an interactive frontend that visualizes path coverage alongside generated tests, assembling tests based on the specific paths they exercise. A user study with 12 participants demonstrates that PALM's frontend helps users better understand path coverage and identify which paths are actually exercised by PALM-generated tests through verification and visualization of their path profiles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11089v1">DataChef: Cooking Up Optimal Data Recipes for LLM Adaptation via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      In the current landscape of Large Language Models (LLMs), the curation of large-scale, high-quality training data is a primary driver of model performance. A key lever is the \emph{data recipe}, which comprises a data processing pipeline to transform raw sources into training corpora. Despite the growing use of LLMs to automate individual data processing steps, such as data synthesis and filtering, the overall design of data recipes remains largely manual and labor-intensive, requiring substantial human expertise and iteration. To bridge this gap, we formulate \emph{end-to-end data recipe generation} for LLM adaptation. Given a target benchmark and a pool of available data sources, a model is required to output a complete data recipe that adapts a base LLM to the target task. We present DataChef-32B, which performs online reinforcement learning using a proxy reward that predicts downstream performance for candidate recipes. Across six held-out tasks, DataChef-32B produces practical recipes that reach comparable downstream performance to those curated by human experts. Notably, the recipe from DataChef-32B adapts Qwen3-1.7B-Base to the math domain, achieving 66.7 on AIME'25 and surpassing Qwen3-1.7B. This work sheds new light on automating LLM training and developing self-evolving AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11088v1">Vulnerabilities in Partial TEE-Shielded LLM Inference with Precomputed Noise</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      The deployment of large language models (LLMs) on third-party devices requires new ways to protect model intellectual property. While Trusted Execution Environments (TEEs) offer a promising solution, their performance limits can lead to a critical compromise: using a precomputed, static secret basis to accelerate cryptographic operations. We demonstrate that this mainstream design pattern introduces a classic cryptographic flaw, the reuse of secret keying material, into the system's protocol. We prove its vulnerability with two distinct attacks: First, our attack on a model confidentiality system achieves a full confidentiality break by recovering its secret permutations and model weights. Second, our integrity attack completely bypasses the integrity checks of systems like Soter and TSQP. We demonstrate the practicality of our attacks against state-of-the-art LLMs, recovering a layer's secrets from a LLaMA-3 8B model in about 6 minutes and showing the attack scales to compromise 405B-parameter LLMs across a variety of configurations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11247v1">Peak + Accumulation: A Proxy-Level Scoring Formula for Multi-Turn LLM Attack Detection</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Multi-turn prompt injection attacks distribute malicious intent across multiple conversation turns, exploiting the assumption that each turn is evaluated independently. While single-turn detection has been extensively studied, no published formula exists for aggregating per-turn pattern scores into a conversation-level risk score at the proxy layer -- without invoking an LLM. We identify a fundamental flaw in the intuitive weighted-average approach: it converges to the per-turn score regardless of turn count, meaning a 20-turn persistent attack scores identically to a single suspicious turn. Drawing on analogies from change-point detection (CUSUM), Bayesian belief updating, and security risk-based alerting, we propose peak + accumulation scoring -- a formula combining peak single-turn risk, persistence ratio, and category diversity. Evaluated on 10,654 multi-turn conversations -- 588 attacks sourced from WildJailbreak adversarial prompts and 10,066 benign conversations from WildChat -- the formula achieves 90.8% recall at 1.20% false positive rate with an F1 of 85.9%. A sensitivity analysis over the persistence parameter reveals a phase transition at rho ~ 0.4, where recall jumps 12 percentage points with negligible FPR increase. We release the scoring algorithm, pattern library, and evaluation harness as open source.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11083v1">Token-Efficient Change Detection in LLM APIs</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Remote change detection in LLMs is a difficult problem. Existing methods are either too expensive for deployment at scale, or require initial white-box access to model weights or grey-box access to log probabilities. We aim to achieve both low cost and strict black-box operation, observing only output tokens. Our approach hinges on specific inputs we call Border Inputs, for which there exists more than one output top token. From a statistical perspective, optimal change detection depends on the model's Jacobian and the Fisher information of the output distribution. Analyzing these quantities in low-temperature regimes shows that border inputs enable powerful change detection tests. Building on this insight, we propose the Black-Box Border Input Tracking (B3IT) scheme. Extensive in-vivo and in-vitro experiments show that border inputs are easily found for non-reasoning tested endpoints, and achieve performance on par with the best available grey-box approaches. B3IT reduces costs by $30\times$ compared to existing methods, while operating in a strict black-box setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11079v1">In-the-Wild Model Organisms: Mitigating Undesirable Emergent Behaviors in Production LLM Post-Training via Data Attribution</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      We propose activation-based data attribution, a method that traces behavioral changes in post-trained language models to responsible training datapoints. By computing activation-difference vectors for both test prompts and preference pairs and ranking by cosine similarity, we identify datapoints that cause specific behaviors and validate these attributions causally by retraining with modified data. Clustering behavior-datapoint similarity matrices also enables unsupervised discovery of emergent behaviors. Applying this to OLMo 2's production DPO training, we surfaced distractor-triggered compliance: a harmful behavior where the model complies with dangerous requests when benign formatting instructions are appended. Filtering top-ranked datapoints reduces this behavior by 63% while switching their labels achieves 78%. Our method outperforms gradient-based attribution and LLM-judge baselines while being over 10 times cheaper than both. This in-the-wild model organism - emerging from contaminated preference data rather than deliberate injection - provides a realistic benchmark for safety techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11243v1">Evaluating Memory Structure in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 Preprint, work in progress
    </div>
    <details class="paper-abstract">
      Modern LLM-based agents and chat assistants rely on long-term memory frameworks to store reusable knowledge, recall user preferences, and augment reasoning. As researchers create more complex memory architectures, it becomes increasingly difficult to analyze their capabilities and guide future memory designs. Most long-term memory benchmarks focus on simple fact retention, multi-hop recall, and time-based changes. While undoubtedly important, these capabilities can often be achieved with simple retrieval-augmented LLMs and do not test complex memory hierarchies. To bridge this gap, we propose StructMemEval - a benchmark that tests the agent's ability to organize its long-term memory, not just factual recall. We gather a suite of tasks that humans solve by organizing their knowledge in a specific structure: transaction ledgers, to-do lists, trees and others. Our initial experiments show that simple retrieval-augmented LLMs struggle with these tasks, whereas memory agents can reliably solve them if prompted how to organize their memory. However, we also find that modern LLMs do not always recognize the memory structure when not prompted to do so. This highlights an important direction for future improvements in both LLM training and memory frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11052v1">GraphSeek: Next-Generation Graph Analytics with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Graphs are foundational across domains but remain hard to use without deep expertise. LLMs promise accessible natural language (NL) graph analytics, yet they fail to process industry-scale property graphs effectively and efficiently: such datasets are large, highly heterogeneous, structurally complex, and evolve dynamically. To address this, we devise a novel abstraction for complex multi-query analytics over such graphs. Its key idea is to replace brittle generation of graph queries directly from NL with planning over a Semantic Catalog that describes both the graph schema and the graph operations. Concretely, this induces a clean separation between a Semantic Plane for LLM planning and broader reasoning, and an Execution Plane for deterministic, database-grade query execution over the full dataset and tool implementations. This design yields substantial gains in both token efficiency and task effectiveness even with small-context LLMs. We use this abstraction as the basis of the first LLM-enhanced graph analytics framework called GraphSeek. GraphSeek achieves substantially higher success rates (e.g., 86% over enhanced LangChain) and points toward the next generation of affordable and accessible graph analytics that unify LLM reasoning with database-grade execution over large and complex property graphs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.24303v3">Retrieval- and Argumentation-Enhanced Multi-Agent LLMs for Judgmental Forecasting (Extended Version with Supplementary Material)</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 24 pages, 3 figures, Accepted to AAMAS 2026
    </div>
    <details class="paper-abstract">
      Judgmental forecasting is the task of making predictions about future events based on human judgment. This task can be seen as a form of claim verification, where the claim corresponds to a future event and the task is to assess the plausibility of that event. In this paper, we propose a novel multi-agent framework for claim verification, whereby different agents may disagree on claim veracity and bring specific evidence for and against the claims, represented as quantitative bipolar argumentation frameworks (QBAFs). We then instantiate the framework for supporting claim verification, with a variety of agents realised with Large Language Models (LLMs): (1) ArgLLM agents, an existing approach for claim verification that generates and evaluates QBAFs; (2) RbAM agents, whereby LLM-empowered Relation-based Argument Mining (RbAM) from external sources is used to generate QBAFs; (3) RAG-ArgLLM agents, extending ArgLLM agents with a form of Retrieval-Augmented Generation (RAG) of arguments from external sources. Finally, we conduct experiments with two standard judgmental forecasting datasets, with instances of our framework with two or three agents, empowered by six different base LLMs. We observe that combining evidence from agents can improve forecasting accuracy, especially in the case of three agents, while providing an explainable combination of evidence for claim verification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.11924v3">Intrinsic Self-Correction in LLMs: Towards Explainable Prompting via Mechanistic Interpretability</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Intrinsic self-correction refers to the phenomenon where a language model refines its own outputs purely through prompting, without external feedback or parameter updates. While this approach improves performance across diverse tasks, its mechanism remains unclear. We show that intrinsic self-correction functions by steering hidden representations along interpretable latent directions, as evidenced by both alignment analysis and activation interventions. To achieve this, we analyze intrinsic self-correction via the representation shift induced by prompting. In parallel, we construct interpretable latent directions with contrastive pairs and verify the causal effect of these directions via activation addition. Evaluating six open-source LLMs, our results demonstrate that prompt-induced representation shifts in text detoxification and text toxification consistently align with latent directions constructed from contrastive pairs. In detoxification, the shifts align with the non-toxic direction; in toxification, they align with the toxic direction. These findings suggest that representation steering is the mechanistic driver of intrinsic self-correction. Our analysis highlights that understanding model internals offers a direct route to analyzing the mechanisms of prompt-driven LLM behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04900v3">Evaluating Kubernetes Performance for GenAI Inference: From Automatic Speech Recognition to LLM Summarization</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 A accepted at the 17th International Conference on Performance Engineering
    </div>
    <details class="paper-abstract">
      As Generative AI (GenAI), particularly inference, rapidly emerges as a dominant workload category, the Kubernetes ecosystem is proactively evolving to natively support its unique demands. This industry paper demonstrates how emerging Kubernetes-native projects can be combined to deliver the benefits of container orchestration, such as scalability and resource efficiency, to complex AI workflows. We implement and evaluate an illustrative, multi-stage use case consisting of automatic speech recognition and summarization. First, we address batch inference by using Kueue to manage jobs that transcribe audio files with Whisper models and Dynamic Accelerator Slicer (DAS) to increase parallel job execution. Second, we address a discrete online inference scenario by feeding the transcripts to a Large Language Model for summarization hosted using llm-d, a novel solution utilizing the recent developments around the Kubernetes Gateway API Inference Extension (GAIE) for optimized routing of inference requests. Our findings illustrate that these complementary components (Kueue, DAS, and GAIE) form a cohesive, high-performance platform, proving Kubernetes' capability to serve as a unified foundation for demanding GenAI workloads: Kueue reduced total makespan by up to 15%; DAS shortened mean job completion time by 36\%; and GAIE working in conjunction with llm-d improved tail Time to First Token latency by up to 90% even under high loads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.13553v2">LLM-Mediated Guidance of MARL Systems</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      In complex multi-agent environments, achieving efficient learning and desirable behaviours is a significant challenge for Multi-Agent Reinforcement Learning (MARL) systems. This work explores the potential of combining MARL with Large Language Model (LLM)-mediated interventions to guide agents toward more desirable behaviours. Specifically, we investigate how LLMs can be used to interpret and facilitate interventions that shape the learning trajectories of multiple agents. We experimented with two types of interventions, referred to as controllers: a Natural Language (NL) Controller and a Rule-Based (RB) Controller. The RB Controller showed a stronger impact than the NL Controller, which uses a small (7B/8B) LLM to simulate human-like interventions. Our findings indicate that agents particularly benefit from early interventions, leading to more efficient training and higher performance. Both intervention types outperform the baseline without interventions, highlighting the potential of LLM-mediated guidance to accelerate training and enhance MARL performance in challenging environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10986v1">TVCACHE: A Stateful Tool-Value Cache for Post-Training LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 Abhishek Vijaya Kumar and Bhaskar Kataria have equal contribution
    </div>
    <details class="paper-abstract">
      In RL post-training of LLM agents, calls to external tools take several seconds or even minutes, leaving allocated GPUs idle and inflating post-training time and cost. While many tool invocations repeat across parallel rollouts and could in principle be cached, naively caching their outputs for reuse is incorrect since tool outputs depend on the environment state induced by prior agent interactions. We present TVCACHE, a stateful tool-value cache for LLM agent post-training. TVCACHE maintains a tree of observed tool-call sequences and performs longest-prefix matching for cache lookups: a hit occurs only when the agent's full tool history matches a previously executed sequence, guaranteeing identical environment state. On three diverse workloads-terminal-based tasks, SQL generation, and video understanding. TVCACHE achieves cache hit rates of up to 70% and reduces median tool call execution time by up to 6.9X, with no degradation in post-training reward accumulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10965v1">MoEEdit: Efficient and Routing-Stable Knowledge Editing for Mixture-of-Experts LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Knowledge editing (KE) enables precise modifications to factual content in large language models (LLMs). Existing KE methods are largely designed for dense architectures, limiting their applicability to the increasingly prevalent sparse Mixture-of-Experts (MoE) models that underpin modern scalable LLMs. Although MoEs offer strong efficiency and capacity scaling, naively adapting dense-model editors is both computationally costly and prone to routing distribution shifts that undermine stability and consistency. To address these challenges, we introduce MoEEdit, the first routing-stable framework for parameter-modifying knowledge editing in MoE LLMs. Our method reparameterizes expert updates via per-expert null-space projections that keep router inputs invariant and thereby suppress routing shifts. The resulting block-structured optimization is solved efficiently with a block coordinate descent (BCD) solver. Experiments show that MoEEdit attains state-of-the-art efficacy and generalization while preserving high specificity and routing stability, with superior compute and memory efficiency. These results establish a robust foundation for scalable, precise knowledge editing in sparse LLMs and underscore the importance of routing-stable interventions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10964v1">Can LLMs Cook Jamaican Couscous? A Study of Cultural Novelty in Recipe Generation</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 14 pages, 12 figures, conference
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to generate and shape cultural content, ranging from narrative writing to artistic production. While these models demonstrate impressive fluency and generative capacity, prior work has shown that they also exhibit systematic cultural biases, raising concerns about stereotyping, homogenization, and the erasure of culturally specific forms of expression. Understanding whether LLMs can meaningfully align with diverse cultures beyond the dominant ones remains a critical challenge. In this paper, we study cultural adaptation in LLMs through the lens of cooking recipes, a domain in which culture, tradition, and creativity are tightly intertwined. We build on the \textit{GlobalFusion} dataset, which pairs human recipes from different countries according to established measures of cultural distance. Using the same country pairs, we generate culturally adapted recipes with multiple LLMs, enabling a direct comparison between human and LLM behavior in cross-cultural content creation. Our analysis shows that LLMs fail to produce culturally representative adaptations. Unlike humans, the divergence of their generated recipes does not correlate with cultural distance. We further provide explanations for this gap. We show that cultural information is weakly preserved in internal model representations, that models inflate novelty in their production by misunderstanding notions such as creativity and tradition, and that they fail to identify adaptation with its associated countries and to ground it in culturally salient elements such as ingredients. These findings highlight fundamental limitations of current LLMs for culturally oriented generation and have important implications for their use in culturally sensitive applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11230v1">DiSCoKit: An Open-Source Toolkit for Deploying Live LLM Experiences in Survey Research</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Advancing social-scientific research of human-AI interaction dynamics and outcomes often requires researchers to deliver experiences with live large-language models (LLMs) to participants through online survey platforms. However, technical and practical challenges (from logging chat data to manipulating AI behaviors for experimental designs) often inhibit survey-based deployment of AI stimuli. We developed DiSCoKit--an open-source toolkit for deploying live LLM experiences (e.g., ones based on models delivered through Microsoft Azure portal) through JavaScript-enabled survey platforms (e.g., Qualtrics). This paper introduces that toolkit, explaining its scientific impetus, describes its architecture and operation, as well as its deployment possibilities and limitations.
    </details>
</div>
