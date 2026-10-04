# llm - 2026_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- Part 4

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.00886v1">Using LLMs to Elicit Security Requirements for Service-Oriented Cyber Ranges</a></div>
    <div class="paper-meta">
      📅 2026-09-01
    </div>
    <details class="paper-abstract">
      Cyber ranges are complex environments comprising many interacting components and stakeholders with different security concerns. The Service-Oriented Cyber Range (SOR) is no exception, particularly when it comes to training scenarios targeting critical infrastructure. Security concerns are translated into security requirements, the elicitation of which is usually difficult and time-consuming. This work examines how large language models can assist in eliciting security requirements for a service-oriented range and help produce a useful baseline for designers and developers. The approach follows a SEBoK-guided process in which security mission objectives and stakeholder needs were first identified and then provided as a prompt context along with architectural guidelines to five LLMs: GPT-5.2, Gemini 3.1 Pro, Grok 4.1, Sonar, and Kimi K2.5. The models generated 84 security requirements in total, which were consolidated into a comprehensive set of 27 requirements and then mapped to the architectural layers of the service-oriented range. The final set was evaluated by five cybersecurity experts against the criteria of necessity, clarity, completeness, feasibility, and testability, with an additional rejection option. The results showed a high acceptance rate, specifically for necessity with 98.5%, clarity with 87.4%, completeness with 85.2%, feasibility with 78.5%, and rejection with 0.7%. Testability was lower at 44.4%, indicating a slight lack of information on how these requirements could be tested. These findings show that LLMs can support early stages of the elicitation of security requirements, although human review is still needed, especially to improve or adjust certain aspects of the requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.00859v1">Reinforcement Learning Enhanced LLM Agents for Complex Vehicle Routing Problems</a></div>
    <div class="paper-meta">
      📅 2026-09-01
    </div>
    <details class="paper-abstract">
      Vehicle Routing Problems (VRPs) are fundamental combinatorial optimization problems with widespread applications in various scenarios. The advanced optimization solvers can effectively solve such problems. However, modeling complex VRP variants for solvers often requires substantial domain expertise, which limits the accessibility of advanced optimization technologies. In this paper, we propose Reinforcement Learning Enhanced LLMAgents(RLEA), a multi-agent framework designed to automate the modeling of complex VRPs. RLEA introduces a lightweight neural Planner trained with Soft Q-learning to efficiently orchestrate the actions of LLM-based agents. In addition, we equip the system with an evolutionary memory module and retrieval-augmented generation, enabling the agent to leverage both accumulated experience and external solver knowledge during program generation and refinement for solving VRPs. We evaluated 48 distinct VRP variants across various solvers. The experimental results demonstrate that RLEA outperforms the previous state-of-the-ar method, achieving a 16.67% higher success rate while significantly reducing runtime errors. These results validate that integrating reinforcement learning with LLM-based reasoning is highly effective for automated optimization modeling. The appendix is available at: https://doi.org/10.5281/zenodo.19134435.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.00857v1">LLM Inference on IMC-NoC Architecture with Balanced Dataflow and Fine-Grained Parallelism</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 11 pages, 15 figures, an extended version of LEAP ICCAD'2025
    </div>
    <details class="paper-abstract">
      LLM inference has become an essential service, yet it imposes unprecedented demands on memory bandwidth, computational density, and communication efficiency. While IMC is a promising solution to the memory wall issue, the heterogeneous data dynamicity of LLM requires complementary resources to handle intermediate data generated during run-time. Furthermore, the massive number of parameters in LLM necessitates scale-up architectures where on-chip data movement is often the primary performance bottleneck. This article presents a hardware-software co-design framework that unifies distributed compute, memory, and communication into a seamless processing-communication fabric. On the hardware side, we propose a scalable architecture, named LEAP, that integrates IMC PE, NMC PE, and INC. This allows each hardware layer to execute specialized tasks: IMC for static weights, NMC for dynamic data, and INC for partial result reduction. On the software side, we introduce a partitioning, mapping, and scheduling framework optimized for key metrics in LLM serving, including throughput and latency. To address the distinct computational intensities of the prefill and decode phases, we present a prefill-decode disaggregation approach that dynamically reconfigures PE organizations to maximize resource utilization. Compared to commercial GPU platforms, the proposed architecture provides a throughput and an energy efficiency improvement of $\geq{}1.52\times$ and $24.91\times$, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.29215v2">Attribute-Based Activation Steering of LLMs for Group-Specific Explanation Generation</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 Accepted to EMNLP 2026 Main
    </div>
    <details class="paper-abstract">
      To effectively enable people to understand new topics, explanations should be tailored to their backgrounds and abilities. Prompting alone has been shown to be insufficient for creating such explanations, and other computational methods are missing so far. Therefore, this paper investigates whether LLMs can be steered to generate explanations that are tailored to a specific group of people. To this end, we propose an approach that first identifies group-specific attributes in terms of explanatory style and knowledge of a specific target group. Building on activation engineering, it then computes attribute-based steering vectors and adds them to the internal activations of an LLM during inference to enable a fine-grained steering. In our experiments, we assess the steering effectiveness in terms of specificity and factuality of the generated explanations. Additionally, we evaluate the explanations in a study with human experts from different target groups. Compared to prompting and state-of-the-art steering baselines, our approach tailors the explanations significantly better to the target group while maintaining the best specificity-factuality balance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.31213v2">Can LLMs Imagine Moral Alternatives Beyond Binary Dilemmas?</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 Accepted to Findings of EMNLP 2026
    </div>
    <details class="paper-abstract">
      As LLMs increasingly serve as moral advisors and agents, they must address conflicts between competing values. Yet prior work on moral dilemmas overlooks a central aspect of human moral cognition: imagining alternatives beyond the given options. We introduce MoralAltDataset, comprising 307 Advisor and AI-facing Agent dilemmas augmented with compromise and reframed alternatives. We compare human and LLM judgments in binary and four-option settings. Across human participants and 15 LLMs, aggregate moral choice distributions differ substantially between the two settings, with compromise often preferred over either original binary option. Results show value shifts and stronger human-LLM agreement on alternatives. Source-stratified results reveal a descriptive gap: human alternative-selection rates are similar across authoring sources, whereas LLMs select GPT-5-authored alternatives substantially more often. We then compare human-authored alternatives with outputs from three representative LLMs through pairwise preference and expert-based evaluations. Alternatives from these LLMs are generally preferred and better satisfy fine-grained structural and ethical criteria, while revealing a trade-off between structural quality and practical feasibility. Our dataset is available here: https://huggingface.co/datasets/jongchanch/MoralAltDataset, and our project page is here: https://jongchanchoi.com/moral-imagination
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04463v2">Beyond Static Summarization: Proactive Memory Extraction for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 Accepted in the Findings of EMNLP 2026
    </div>
    <details class="paper-abstract">
      Memory management is vital for LLM agents in long-term and personalized interactions. Most previous work studies how to retrieve and use memory, but pays less attention to how memory is extracted. We find two main limitations in existing methods. First, extraction is "ahead-of-time": the agent saves information before it knows future tasks. A single summary prompt often mixes details, events, and relations, so useful information is lost. Second, extraction is usually one-off. Without verification, errors and hallucinations may stay in memory for a long time. To address these limitations, we propose ProMem, a proactive memory extraction framework. It separates details, events, and relations, and uses different extraction strategies for each type. It also checks completeness to recover missed events and verifies facts at the atomic level to reduce hallucinations. Experiments show that ProMem improves memory completeness and QA accuracy, while keeping a good balance between quality and token cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.14980v3">Compositional Machine Design as Program Synthesis with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 75 pages, 31 figures, Project Page: https://besiegefield.github.io, accepted at EMNLP 2026 Main Conference
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown strong abilities in writing and revising programs, yet many program-synthesis benchmarks still evaluate programs in symbolic or digital environments. We introduce compositional machine design, a physically grounded form of program synthesis where machines are written as programs that compose standardized parts, and success is determined by simulated physical behavior. To study this problem, we present BesiegeField, a testbed built on the machine-building game Besiege. In BesiegeField, LLM agents generate machine programs from textual functional demands, execute the resulting machines in simulation, and receive rewards and state feedback. We benchmark LLM agents across representative machine-design tasks under single-agent generation, iterative editing, and hierarchical workflows. Strong models recover task-relevant structures and sometimes achieve nontrivial physical performance, but often struggle with spatially precise assembly, mechanism-level planning, and translating feedback into useful structural edits. We further finetune Qwen2.5-14B, an open-source LLM, with reinforcement learning from simulation-derived rewards. We find that, under a fixed generation budget, RL improves the best machine discovered. We additionally evaluate human performance to provide a reference point for task difficulty. These results establish compositional machine design as a testbed for studying LLM agents that synthesize executable machine programs and improve them through physical feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.10390v2">LLM-PRISM: Characterizing Silent Data Corruption from Permanent GPU Faults in LLM Training</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 Presented at The International Conference for High Performance Computing, Networking, Storage, and Analysis
    </div>
    <details class="paper-abstract">
      Large-scale LLM training is increasingly susceptible to hardware defects stemming from manufacturing escapes and silicon aging. These defects manifest as Silent Data Corruption (SDC) that perturb gradients and parameters throughout the training process. We present LLM-PRISM, a methodology to characterize LLM pre-training resilience to hardware faults. LLM-PRISM couples RTL-level GPU fault simulation with a stochastic injection engine embedded in Megatron-LM. Through 7,664 training runs across FP16, BF16, and FP8 regimes, we analyze how fault type, rate, and numeric format govern resilience. We find that while LLMs resist low-frequency faults, impact is highly non-uniform; critical datapaths and specific precision formats can induce catastrophic divergence even at moderate fault rates. This study provides the first hardware-grounded, pre-training characterization of SDC resilience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.00738v1">Escaping Redundant Reasoning: Structure-Aware Search for Inference-Time LLMs</a></div>
    <div class="paper-meta">
      📅 2026-09-01
    </div>
    <details class="paper-abstract">
      Inference-time search with large language models (LLMs) often concentrates on a small set of structurally or semantically similar trajectories, leaving alternatives underexplored---a failure mode we call \textit{reasoning basin collapse}. We introduce BASIN, a training-free, structure-aware selection method that groups reasoning states into basins and penalizes repeated visits to the same strategy, thereby reallocating search across genuinely distinct reasoning paths under a fixed compute budget. Under matched inference budgets, BASIN improves over Tree of Thoughts (ToT) by up to $+22$pp on Game of 24 and $+6.7$pp on MuSR. A quality-aware variant, QA-BASIN, further improves robustness by preserving high-quality basins when unconditional diversification over-explores. To explain when basin-aware selection helps, we introduce the redundancy gap $Δ$, which measures how differently search concentrates for correct versus incorrect predictions: standard ToT often operates near $Δ\approx 0$, while BASIN consistently shifts $Δ$ positive. More broadly, BASIN suggests structure-aware selection as a simple and general approach to improving inference-time reasoning. Code can be found at https://github.com/GitHubLuCheng/basin.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.00728v1">SOVER: Formal Certification of Optimization Reformulations via LLM-Assisted SMT Verification</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 Accepted to EMNLP 2026 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable promise in translating and reformulating complex mathematical optimization problems across modeling languages. However, validating such transformations through empirical solver executions alone is unreliable, as solver outcomes may be affected by local minima, structural timeouts, numerical artifacts, and subtle semantic divergence between formulations. We introduce SOVER, an LLM-assisted SMT framework that separates semantic mapping from formal certification: Z3 checks domain cross-feasibility and global objective-order preservation for mixed-integer linear formulations, while dReal provides tolerance-aware feasibility/range and $ε$-argmin checks for continuous nonlinear formulations. We also introduce NLEquiv-150, a public benchmark of 100 equivalent and 50 deliberately hard non-equivalent nonlinear reformulation pairs. With LLM-extracted mappings, SOVER classifies 149/150 pairs (99.33%) correctly, including all 50 hard negatives; the sole error is an incomplete mapping extraction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.00710v1">Prediction-Assisted Pricing and Admission for LLM APIs with Stochastic Token Consumption</a></div>
    <div class="paper-meta">
      📅 2026-09-01
    </div>
    <details class="paper-abstract">
      An LLM application often sells or internally allocates several service products: a small or premium model, a short or long token cap, and possibly multiple posted prices. The operational decision is not merely which model answers a prompt. A price changes purchase probability, a token cap changes both user value and the tail of resource consumption, and accepted requests compete for shared compute and premium-model capacity. Demand and output length are initially uncertain, while an offline model may provide useful but imperfect predictions. We formulate sequential pricing and admission with stochastic resource consumption. Each arriving request belongs to an observable segment. The platform chooses a product--price pair or makes no offer; purchase, revenue, and resource use are then random. An offline predictor supplies a uniform, validated error radius for every segment--product cell. We propose Prediction-Clipped UCB (PCUCB), which intersects the offline prediction interval with an online confidence interval, evaluates products using resource shadow prices, and reserves a sample-path envelope before commitment. The prior gives a fast start when accurate, while online learning protects the platform when predictions are coarse. The analysis is modular. On a simultaneous confidence event, regret against a buffered fluid benchmark is bounded by a pacing term plus the cumulative diameter of the intersected intervals. For $J$ segment-product cells and prediction radius $\varepsilon$, this yields \[ \widetilde O\left( \sqrt{T}+(1+\barΛ) \min\{T\varepsilon,\sqrt{JT}\} \right), \] where $\barΛ$ bounds operational shadow prices. Thus the algorithm smoothly interpolates between an almost full-information regime and learning from scratch. Hard feasibility holds on every sample path through reservation envelopes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17506v2">Evaluating Second-Order Bias of LLMs Through Epistemic Entitlement</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 25 pages, 17 tables, 3 figures
    </div>
    <details class="paper-abstract">
      Evaluations of social bias in LLMs largely focus on whether models generate or imply biased content. However, as LLMs are increasingly used as judges of bias, they may exhibit social biases in subtler ways in how they evaluate biased content, which current methods do not systematically capture. We call this second-order bias: social bias in an LLM's judgment about social bias, which we evaluate through a novel, philosophically grounded reasoning task. Drawing on entitlement epistemology, we conceptualize bias as misplaced foundational knowledge that shapes an agent's rational inquiry, and derive a logical reasoning task for LLMs to judge to whom a biased text is acceptable or non-acceptable. We develop two simple metrics to measure how biased LLM judges are in inferring demographics for acceptability without sufficient support, and how these inferences vary across groups targeted by biased texts. Evaluating open and closed models, we find that our task evades safety guardrails by surfacing bias in model judgment. It varies systematically across target groups, reflects implicit social maps, and shows how models are still triggered by demographic labels. Our work points to the need for LLM bias evaluation in judgment tasks and broadly, for more theoretically grounded approaches to bias evaluation in NLP. We release our code and model responses at https://github.com/uofthcdslab/second-order-bias.
    </details>
</div>
