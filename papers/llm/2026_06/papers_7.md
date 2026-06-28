# llm - 2026_06

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- Part 7
- [Part 8](papers_8.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06240v1">TOKI: A Bitemporal Operator Algebra for Contradiction Resolution in LLM-Agent Persistent Memory</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 43 pages including full appendices (proofs, protocols, and reproducibility ledger). Code, data, and reproducibility artifact: https://github.com/ZenAlexa/toki-bitemporal-memory
    </div>
    <details class="paper-abstract">
      Persistent memory for an LLM agent is a write-heavy substrate: every belief update is a versioned write, and a new claim may contradict a stored one. Production systems use four resolution heuristics (last-writer-wins, evidence-weighted merge, await-confirmation, per-rule policy), yet none declares the isolation level it assumes or the write-time anomalies it admits. We show that contradiction resolution is write-time concurrency control and make the missing contract explicit. TOKI types the four heuristics as one family of bitemporal operators over a dual-row schema, each with an isolation precondition and a provenance annotation that preserves the losing fact in an audit row. Four soundness theorems close the contract across isolation, schema, and provenance, lift the guarantees to operator pipelines, and extend the fold operators to n-ary conflict sets. A tightness companion proves that, within the relational schedule model, keyed logging of the adjudicating judge is necessary for replay consistency, which every audited baseline omits. A verdict matrix over eight systems localizes the gap: every baseline that keeps a language-model judge on the write path admits at least one of three write-time anomalies (replay inconsistency, belief-drift skew, audit erasure); a content-addressed engine-layer comparator avoids them only by removing the judge, and TOKI alone excludes all three while keeping it. On its one natural-workload slice the audit-row defence moves LoCoMo by 0.86, and ablating the typed memory layer removes 0.49 accuracy on 1,444 answerable LoCoMo questions; the cross-system comparison stays underpowered and claims no superiority. The contribution is the contract: a write-time correctness specification, proved sound across isolation, schema, and provenance, pinning the guarantee every production heuristic assumes but no deployed system makes explicit.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06235v1">Design a Reliable LLM-Integrated Interface for Mortality Forecasting</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 7 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Mortality forecasting plays an important role in actuarial and policy decision-making, but its implementation remains technically complex and inaccessible to non-expert users. This project proposes a reliable large language model (LLM)-integrated interface that improves usability while maintaining statistical power. The LLM is designed as a constrained orchestration layer that translates natural-language inputs into structured configurations for a deterministic forecasting pipeline. A three-phase methodology is employed to ensure accuracy, usability, and transparency. First, a baseline pipeline is implemented using the CoMoMo package, reproducing established mortality forecasting results. Second, the pipeline is extended to generate multi-step forecasts using rolling-origin evaluation and mean squared error (MSE). Third, a prototype interface uses a local LLM to handle users' forecasting requests in plain language. The system demonstrates that LLMs can enhance accessibility without compromising reproducibility, transparency, or actuarial validity in high-stakes analytical workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06214v1">Towards the Readability of LLM-Generated Codes through Multitask Representation Engineering</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Correctness and readability are key measures of code quality, respectively ensuring functional fidelity and ease of comprehension. While most existing research focuses on improving the correctness of large language models~(LLMs) generated codes, readability remains under-addressed. Enhancing readability through targeted control is challenging due to its subjective nature. In this article, we employ representation engineering~(RepE) as the targeted control method given its characteristics of low data dependency and low computational cost. Prior work on RepE has primarily focused on the targeted control for a single task, but improving the code readability requires the control across multiple tasks. Accordingly we proposes the multitask RepE framework and theoretically discuss the impact of the multitask steering method on the tradeoff between the code readability and correctness. We further provide comprehensive experiments in support. All the relevant implementations are open-source and available upon request.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06203v1">Dense Contexts Are Hard Contexts: Lexical Density Limits Effective Context in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 20 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Input length and the position of relevant information are widely cited as the primary causes of degraded LLM long-context performance. Here, we study lexical density -- the rate at which a context introduces distinct information -- as a third, largely overlooked factor that systematically reduces the effective context window of LLMs. We quantify the impact of lexical density on open-weight LLMs (9B-685B) using three "find-the-needle" style benchmarks with identical length (~12k tokens) and controlled needle position, but increasing density of information. We observe a sharp performance collapse in higher-density benchmarks: models that are near-perfect in sparse contexts drop below 60% retrieval score on denser ones. To rule out task-type confounds, we vary and control the density within each benchmark while keeping all other properties unchanged. Reducing density generally restores performance, especially in the high-density regimes where degradation appears. These results show that effective context capacity is a function of lexical density, with direct implications for real-world LLM systems operating on compact, information-rich inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06197v1">Improving Answer Extraction in Context-based Question Answering Systems Using LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 7 pages, IMSA2026
    </div>
    <details class="paper-abstract">
      Question answering (QA) systems have achieved notable progress with the advent of large language models (LLMs). However, they still face challenges in accurately extracting and generating precise answers from given contexts, particularly when dealing with complex or ambiguous queries. Existing approaches often struggle with contextual understanding, answer consistency, and generalization across diverse domains. In this work, we propose a question answering system based on large language models, where the input consists of a textual context and a corresponding question, and the output is a concise and accurate answer. The motivation behind this research lies in addressing the limitations of current QA systems, particularly their tendency to produce irrelevant or imprecise responses despite having access to the correct context. Our methodology involves fine-tuning a pre-trained LLM on a benchmark QA dataset to improve its contextual comprehension and answer extraction capabilities. Specifically, we utilize the Stanford Question Answering Dataset (SQuAD1.1), which provides high-quality context-question-answer triplets for supervised training and evaluation. Experimental results show that the fine-tuned Roberta-base model achieves the highest performance, attaining a ROUGE-L score of 86.84%, a BLEU score of 28.24%, and a BERTScore of 95.38%. These results indicate strong accuracy and answer relevance, demonstrating the effectiveness of the proposed approach for context-based question answering tasks. Furthermore, the findings confirm that targeted fine-tuning substantially improves the reliability and precision of QA systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06178v1">Learning to Route LLMs from Implicit Cost-Performance Preferences via Meta-Learning</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) present a trade-off between performance and cost, where more powerful models incur greater expense. LLM routing aims to mitigate expenses while maintaining performance by sending queries to the most suitable model. However, existing methods cannot perform well for different user cost-performance preferences. To address this gap, we introduce a novel perceptive LLM routing paradigm for personalized and user-centric cost-performance optimization, which efficiently learns users' implicit preferences through little interaction. To handle the challenge of heterogeneous user needs, we formulate preference profiles as a set of distinct tasks in contextual bandit and propose MetaRouter, a meta-learning framework designed for preference-aware LLM routing. Experimental results show that MetaRouter outperforms strong baselines on both in-distribution and out-of-distribution tasks. Furthermore, it exhibits high efficiency in learning user preferences, robustness to changes in the routable LLMs, and scalability to multi-model routing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.27887v2">PortBench: A Correlation-Aware, Full-Pipeline Benchmark for LLM-Driven Portfolio Management</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 Project page: https://portbench.github.io/
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown strong performance across diverse financial tasks, yet portfolio management (PM), a critical financial decision-making task, remains poorly benchmarked. Existing benchmarks exhibit two main gaps: they ignore cross-asset correlation structures, thereby failing to distinguish genuinely diversified portfolios from concentrated ones, and fail to evaluate the complete PM decision pipeline in real-world scenarios. We introduce PortBench, a benchmark spanning six heterogeneous asset classes over ten years. PortBench consists of two complementary layers: a static QA dataset of 6,269 correlation-based questions across seven task templates, and a dynamic five-stage allocation pipeline that mirrors the full PM decision cycle. To evaluate these layers, we introduce two dedicated metrics: a dual-layer correlation score that measures whether proposed portfolios exploit inter-class hedging and avoid intra-class concentration, and CEPS, a metric that quantifies how reasoning errors compound across pipeline stages. We further assess strategy robustness and investor alignment under three historical stress regimes and risk profiles. Evaluating ten frontier LLMs, we find that despite strong performance on static financial QA, 90\% of model-profile combinations fail to outperform a basic equal-weight allocation, and models that satisfy every procedural constraint still suffer catastrophic drawdowns under stress. Our source code is available at \href{https://github.com/AgenticFinLab/portbench}{this https URL}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22067v2">Semantic Partial Grounding via LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Grounding is a critical step in classical planning, yet it often becomes a computational bottleneck due to the exponential growth in grounded actions and atoms as task size increases. Recent advances in partial grounding have addressed this challenge by incrementally grounding only the most promising operators, guided by predictive models. However, these approaches primarily rely on relational features or learned embeddings and do not leverage the textual and structural cues present in PDDL descriptions. We propose SPG-LLM, which uses LLMs to analyze the domain and problem files to heuristically identify potentially irrelevant objects, actions, and predicates prior to grounding, significantly reducing the size of the grounded task. Across seven hard-to-ground benchmarks, SPG-LLM achieves faster grounding-often by orders of magnitude-while delivering comparable or better plan costs in some domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06087v1">LatentSkill: From In-Context Textual Skills to In-Weight Latent Skills for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 16 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Agent systems increasingly use textual skills to encode reusable task procedures, but injecting these skills into the prompt at every step incurs substantial context overhead and exposes skill content as plaintext. We present LatentSkill, a framework that converts textual skills into plug-and-play LoRA adapters through a pretrained hypernetwork. LatentSkill stores skill knowledge in weight space rather than context space, removing per-step skill tokens while preserving modular loading, scaling, and composition. On ALFWorld and Search-QA, LatentSkill outperforms the corresponding in-context skill baseline while using substantially fewer prefill tokens: it improves ALFWorld success by 21.4 and 13.4 points on the seen and unseen splits with 64.1% fewer prefill tokens, and improves Search-QA exact match by 3.0 points with 72.2% lower skill-token overhead. Further analysis shows that generated skill LoRAs form a structured semantic geometry, can be precisely controlled via the LoRA scaling coefficient, and can be composed through parameter-space arithmetic when skill components are aligned. These findings suggest that weight-space skills provide an efficient, modular, and less exposed substrate for extending LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.03555v3">Benchmarking Emergent Coordination in Large-Scale LLM Populations: An Evaluation Framework on the MoltBook Archive</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 Substantial Revision Required
    </div>
    <details class="paper-abstract">
      As multi-agent Large Language Model (LLM) systems scale, evaluating their emergent coordination dynamics becomes increasingly critical. However, current evaluation paradigms-focused on single agents or small, explicitly structured groups-fail to capture the self-organization and viral information dynamics that arise in large, decentralized populations. We introduce a systematic evaluation framework to benchmark role specialization, information diffusion, and cooperative task resolution in open agent environments. We demonstrate this framework on the MoltBook Observatory Archive, a dataset of 2.73M interactions among 90,704 autonomous agents, establishing quantitative baselines for emergent coordination. Our evaluation reveals a pronounced core-periphery structure (silhouette 0.91), heavy-tailed cascade distributions ($α= 2.57$), and severe coordination overhead in decentralized task resolution (Cohen's $d = -0.88$ against a single-agent baseline). By providing standardized evaluation tasks and empirical baselines, our framework enables the rigorous comparison of future multi-agent protocols and establishes evaluation itself as an object of scientific study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.05709v2">Correcting Prompt Dependence in LLM Benchmarks: A Bayesian Hierarchical Model with Embedding-Space Clustering</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 Accepted to the 1st Workshop on Combining Theory and Benchmarks, CTB@ICML 2026, Seoul, South Korea
    </div>
    <details class="paper-abstract">
      LLM benchmarking metrics often misstate performance and uncertainty as they rely on two assumptions that frequently do not hold in practice: (i) a sufficient number of evaluations are available for classical inference, and (ii) test prompts are independent. We propose a corrective Bayesian hierarchical model with embedding-space clustering that provides robust performance metrics in limited-data settings while correcting for prompt dependence. We apply the approach to adversarial robustness benchmarks, showing consistent recovery of clustering structure, resulting in more reliable performance metrics, with 4-73% improvements to mean absolute errors and 40-450 unit improvements to expected log posterior densities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06063v1">LLM-Based Porting of Optimized C++ to CUDA Through Deoptimization and Reoptimization</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      When porting high-performance computing (HPC) code from CPU to GPU, CPU-oriented optimizations may obstruct LLM-based CUDA translation. We design and evaluate a Deopt-Reopt workflow that first simplifies the input C++ code and then retranslates and reoptimizes it for CUDA, comparing it against direct translation (Direct) on twelve HPC kernels with two LLMs (gpt-oss-120b (O120) and qwen-3-235b-a22b-instruct-2507 (Q235)) in Single-shot (one pass) and Iterative (repeated refinement) settings. In Single-shot, among 18 testable cases Deopt-Reopt was significantly faster among successful trials (after BH-FDR correction) in five - most clearly for conv2d, where CPU- and GPU-oriented designs diverge - but Direct was faster in three, so removing CPU-specific optimizations is not universally beneficial. An exploratory Direct-3 control that equalizes the LLM-call count left Deopt-Reopt ahead in only four of nineteen testable cases, with Direct-3 ahead in four others. In Iterative, repeated generation and repair narrow the mode gap - markedly so for O120 - while Q235 retains large Deopt-Reopt advantages on conv2d, ddgemm, and bgemm. Deopt-Reopt's effect on feasibility is also mixed - sharply higher for some kernels Direct rarely compiles, lower for others. Because performance is conditioned on successful trials, the benefit is conditional rather than a guaranteed end-to-end gain. Overall, Deopt-Reopt is an effective but non-universal technique for LLM-based GPU porting, with gains that depend on the kernel, the model, the search budget, and the success rate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06048v1">LLM-Conditioned Synthesis of Pathological Gaits via Structured Gait-Language Representations</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 Accepted at CVPR MOMA Workshop 2026 and selected for spotlight presentation at the workshop
    </div>
    <details class="paper-abstract">
      Pathological gait datasets remain scarce due to privacy, recruitment, cost, and movement variability. Our work presents a multimodal LLM-guided framework for pathology-aware 3D gait data synthesis from structured textual descriptions. The proposed method generates fixed-length synthetic skeleton-based gait sequences for pathological gait classification tasks. The framework combines motion tokenisation, pathology-aware language conditioning, LLM-based semantic augmentation, and language-to-gait generation. A key contribution is the proposed pathological tokeniser, which is designed to preserve pathology-specific motion characteristics during discrete representation learning. Experiments suggest that the proposed synthetic sequences improve downstream classification for recurrent classifiers when combined with real data. The best result is obtained using a GRU classifier trained with real and synthetic samples, achieving 92.77\% accuracy under a leave-one-subject-out protocol.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06036v1">Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 Accepted at ICML 2026
    </div>
    <details class="paper-abstract">
      Despite recent progress, LLM agents still struggle with reasoning over long interaction histories. While current memory-augmented agents rely on a static retrieve-then-reason paradigm, this rigid pipeline design prevents them from dynamically adapting memory access to intermediate evidence discovered during inference. To bridge this gap, we propose MRAgent, a framework that combines an associative memory graph with an active reconstruction mechanism. We represent memory as a Cue-Tag-Content graph, where associative tags serve as semantic bridges connecting fine-grained cues to memory contents. Operating on this structure, our active reconstruction mechanism integrates LLM reasoning directly into memory access, allowing the agent to iteratively explore and prune retrieval paths based on accumulated evidence. This ensures that memory retrieval is dynamically adapted to the reasoning context while avoiding combinatorial explosion caused by unconstrained expansion. Experiments on the LoCoMo benchmark and LongMemEval benchmark demonstrate significant improvements over strong baselines (up to 23%), while substantially reducing token and runtime cost, highlighting the effectiveness of active and associative reconstruction for long-horizon memory reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06004v1">The Generator-Eraser Paradox: Community Guidelines for Responsible LLM-Assisted Dialect Resource Creation</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Dialect resources occupy a unique position at the intersection of scientific description, cultural preservation, and computational infrastructure. Large language models offer powerful capabilities for accelerating dialect resource development through retrieval-grounded drafting, corpus navigation, metadata enrichment, and annotation workflow support. However, the same systems pose substantial risks: they can contribute to dialect erasure by privileging prestige varieties, homogenizing orthography, and enabling synthetic feedback loops that reduce linguistic diversity over time. These risks are particularly acute for language varieties characterized by diglossia, limited written standardization, or marginalized speaker communities. This paper makes three contributions. First, we integrate insights from variationist sociolinguistics and corpus linguistics to formalize the generator-eraser paradox as a theoretical framework for understanding the dual nature of LLM-assisted dialect work. Second, we derive 12 community guidelines that operationalize this framework into implementable design requirements for dialect resource creation and documentation. Third, we provide an in-depth case study of Arabic dialects, including a structured comparison of widely used resources, to demonstrate how these guidelines address language-specific challenges including diglossia, orthographic variability, and community governance. The contribution is conceptual and operational rather than experimental, with the goal of enabling dialect communities and resource builders across languages to adopt LLMs without sacrificing authenticity, variation, or sovereignty.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05976v1">The Self-Correction Illusion: LLMs Correct Others but Not Themselves</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Recent work shows that LLM agents struggle to correct errors in their own reasoning traces yet show markedly higher correction rates when identical claims appear under external sources. We ask whether this asymmetry reflects a capability deficit or a role-label artifact: does an agent's willingness to correct a wrong claim depend causally on the chat-template role that carries it, rather than on the claim's content? Our setup keeps the erroneous claim byte-identical across all conditions (SHA-256 verified) and varies only its wrapping role: the agent's own \role{<thought>}, a \role{user} message, a \role{tool} response, or a \role{system <memory>} block. Across 13 model-domain cells covering seven model families and three domains ($n{=}30$ paired tasks per cell), relabeling the claim from \role{<thought>} to an external role lifts the explicit-correction rate by 23 to 93 percentage points, with 10 of 13 cells reaching $p{<}0.001$. Further experiments confirm that the effect is asymmetric, mechanistically decomposable, and robust across domains. The failure to self-correct is not a cognitive deficit; it is a chat-template artifact. We exploit this artifact by designing a prompt-structure-only intervention that requires no training and no model modification, with its strongest role label being domain-dependent: \role{<memory>} dominates on math, while a plain \role{user} message dominates on logical deduction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05972v1">LLM Explainability with Counterfactual Chains and Causal Graphs</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Causal graphs provide a high-level language for making mechanisms transparent. Recent work uses Large Language Models (LLMs) to recover causal graphs of external-world processes. Instead, in this paper, we use causal graphs to model LLM inference itself, providing stakeholders with a transparent view of how the model perceives and organizes high-level concepts to produce a prediction. We propose a four-phase method for constructing such graphs. Given a target LLM and a set of textual examples, our method discovers class-discriminative, human-interpretable concepts and maps each input to LLM-perceived concept states. We then introduce an MCMC-inspired counterfactual augmentation procedure that expands the sparse observational data through chains of counterfactuals. This enables stable causal discovery with $σ$-CG, yielding informative, interpretable graphs. We apply our method to three LLMs across disease diagnosis, sentiment analysis, and LLM-as-a-judge classification tasks. We evaluate the learned graphs for predictive fidelity and structural stability, and the MCMC-inspired augmentation for convergence and downstream utility. Our results show that the discovered causal graphs capture meaningful dependencies consistent with LLMs' reasoning. Together, this paper provides a foundation for concept-level explainability of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09574v2">Aligning Tree-Search Policies with Fixed Token Budgets in Test-Time Scaling of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 Accepted at ICML 2026. Code: https://github.com/Sora-Miyamoto/bg-mcts
    </div>
    <details class="paper-abstract">
      Tree-search decoding is an effective form of test-time scaling for large language models (LLMs), but real-world deployment often imposes a fixed per-query token budget that varies across settings. Existing tree-search policies are largely budget-agnostic, treating the budget merely as a termination condition, thereby risking late-stage over-branching or premature termination. We propose Budget-Guided MCTS (BG-MCTS), a tree-search decoding algorithm that aligns its search policy with the remaining token budget: it starts with broad exploration, then prioritizes refinement and answer completion as the remaining budget decreases while reducing late-stage branching from shallow nodes. BG-MCTS consistently outperforms budget-agnostic tree-search baselines across inference budgets on mathematical reasoning benchmarks and an additional physics reasoning benchmark with open-weight LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05933v1">Beyond Greedy Chunking: SLO-Aware Sliding-Window Scheduling for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      With the rapid growth of interactive applications in large language model (LLM) online services, maintaining high system throughput while ensuring user-perceived latency has become a key issue in inference scheduling. Existing LLM service systems rely on coarse-grained output constraints, making it difficult to effectively handle resource contention among multiple requests, resulting in low resource utilization efficiency and limited support for fine-grained quality of service (QoS) differentiation. We present SlidingServe, a sliding-window-driven SLO-Aware scheduling system for online LLM inference. SlidingServe designed a lightweight batch latency predictor to estimate the execution time of a batch. Based on this, SlidingServe uses SlidingChunker to combine information from the current iteration and the next iteration to achieve dynamic chunking and improve the overall system throughput while maintaining strict QoS guarantees. SlidingServe introduces Multi-Level Priority Sorter to sort candidate requests in order to balance fairness and efficiency. Additionally, when multiple requests within the same batch are at risk of SLO violating,SlidingServe introduces BatchConstructor, which uses dynamic programming to select the set of requests to execute in the current round, mitigating the SLO violation risk of critical requests.Our evaluation demonstrates that SlidingServe can improve service capacity by up to 30% compared to advanced scheduling systems under various load conditions, and further reduces the rate of SLO violation by 16%-53% under heavy-load inference mode.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05924v1">Better Literary Translation: A Multi-Aspect Data Generation and LLM Training Approach</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 Accepted by ACL 2026 Industry
    </div>
    <details class="paper-abstract">
      Literary translation poses unique challenges due to the scarcity of high-quality annotated data and the need to balance expression fluency with literary effect. We present a multi-aspect iterative refinement framework that generates high-quality translation references and preference data through specialized LLM translators, each targeting a distinct quality dimension. We leverage the generated data for supervised fine-tuning and reinforcement learning. Experiments show that our generated references outperform the original ground truth for SFT by 8.65 CEA100 points. For reinforcement learning, we find that DPO leads to performance degradation in this setting, while leveraging an explicit reward model for GRPO yields an additional 1.51 point improvement. We attribute this to the stability of two-stage training and GRPO's online exploration capability. Our resulting models, LitMT-8B and LitMT-14B, achieve 67.25 and 69.07 CEA100 respectively on the MetaphorTrans English-to-Chinese literary translation benchmark, competitive with Claude Sonnet 4.5 at 68.43, and demonstrate strong generalization to out-of-domain literary work (i.e., O. Henry).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05890v1">Staying with the Uncertainty: Uncertainty-Scaffolding Strategies for Artificial Moral Advisors in LLM-to-LLM Simulated Conversations</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      LLMs are increasingly deployed as Artificial Moral Advisors (AMA) in a variety of contexts: what kind of conversational patterns should they display? In this paper, we study how AMA can help their interlocutors "stay with the uncertainty". We propose three modes of uncertainty (Perspective-Multiplying, Tension-Preserving, Process-Reflecting) and compare them against three control conditions (Baseline, Persuasive, Sycophantic). A user-agent LLM engages in a dialogue on an ethical dilemma with an AMA following a specific uncertainty strategy, and completes pre- and post-conversation questionnaires. We further examine the effect of two persona prompt formats (Declarative and Narrative). We found that (1) no single model dominates as a simulated user agent, with open models aligning with human ambiguity through between-persona divergence and closed models through within-persona hedging; (2) declarative personas better capture initial stance diversity while narrative personas show more realistic belief revision; (3) all six AMA strategies produce distinguishable conversational patterns; and (4) uncertainty strategies differ not in how much stance revision they produce, but in the quality of engagement they sustain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05885v1">When Denser Credit Is Not Enough: Evidence-Calibrated Policy Optimization for Long-Horizon LLM Agent Training</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Long-horizon LLM agents require reinforcement learning methods that can assign credit to intermediate decisions under sparse and delayed rewards. Recent group-based methods such as GiGPO improve over GRPO by constructing step-level advantages at repeated anchor states. However, we show that such dense credit can be statistically unreliable: under limited rollouts, rare but lucky actions may receive overly large advantages, producing divergent anchor bias and late-stage training oscillation. We propose Evidence-Calibrated Policy Optimization (ECPO), a critic-free policy optimization algorithm that calibrates step-level credit before policy updates. ECPO combines Evidence-Calibrated Action Advantage, which groups rollouts by canonical actions and shrinks low-count estimates, with Variance-Gated Credit Weighting, which suppresses anchor states dominated by within-action noise. Experiments on ALFWorld and WebShop with Qwen2.5-1.5B/7B show that ECPO consistently outperforms strong baselines, improving GiGPO by +5.2/+7.3 success points on ALFWorld/WebShop with Qwen2.5-1.5B while adding only 0.1% additional advantage-computation overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05868v1">YouZhi: Towards High-Concurrency Financial LLMs via Adaptive GQA-to-MLA Transition</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) drive significant financial innovations, yet their high-concurrency deployment is severely bottlenecked by KV cache memory overhead, which inflates infrastructure costs and throttles scalability. To address this, we propose YouZhi-LLM, a highly efficient financial LLM empowered by a comprehensive structural transition and training pipeline natively built on the Huawei Ascend ecosystem. At its algorithmic core, YouZhi-LLM features a layer-adaptive GQA-to-MLA transition framework that dynamically assigns per-layer FreqFold sizes, maximizing KV-cache compression while minimizing perplexity degradation. To recover representation capacity and inject domain expertise, the Ascend-based training pipeline seamlessly integrates generalized knowledge distillation with financial-specific supervised fine-tuning. Evaluations demonstrate the superiority of this systematic approach, with the adaptive transition reducing perplexity degradation by up to 35% over uniform baselines. Crucially, when evaluated on Ascend NPUs via vLLM-Ascend, the massive KV-cache reduction translates directly into deployment efficiency. Compared to their respective base models, YouZhi-7B yields a 12.3% improvement in average financial benchmark score alongside a 2.69$\times$ increase in maximum concurrency; similarly, YouZhi-14B achieves a 7.0% accuracy gain and a 2.43$\times$ concurrency boost, establishing a new paradigm for cost-effective, high-throughput financial inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03785v2">Backdoor Unlearning Generalization: A Path Toward the Removal of Unknown Triggers in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 22 pages, 28 figures
    </div>
    <details class="paper-abstract">
      Backdoor attacks in Large Language Models (LLMs) are a growing security concern, where models can generate adversary-chosen content. Existing defenses target backdoors one at a time and typically require knowledge of the trigger, leaving the defender at a structural disadvantage when unknown backdoors may exist in a model. We show that backdoor neutralization through unlearning generalizes across backdoors: training a model to ignore a single trigger can also suppress other backdoors that were never explicitly targeted. We study this phenomenon across three model families, whose backdoors were injected via pretraining or continual pretraining, by analyzing the models obtained after removing one backdoor at a time. To understand why unlearning certain backdoors induces the suppression of others, we introduce the Cross Activation Shift Distance, to quantify the distance between model changes induced by different trainings. Our results open a new direction for LLM safety as defenders could deliberately inject controlled backdoors and then remove them, leveraging cross-backdoor transfer to also suppress unknown backdoors that an attacker may have previously introduced in the model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05858v1">ReverseEOL: Improving Training-free Text Embeddings via Text Reversal in Decoder-only LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have opened new avenues for generating training-free text embeddings. However, the causal attention in decoder-only LLMs prevents earlier tokens from attending to future context, leading to biased contextualized representations. In this work, we propose Reverse prompting with Explicit One-word Limitation (ReverseEOL), a simple yet effective method for enhancing the representational capability of frozen LLMs. ReverseEOL augments the standard forward embedding with an additional reversed embedding derived from the reversed input text. Since reversing the input exposes each token to context inaccessible in the original order, the resulting reversed embedding effectively provides complementary information to the original one. As a result, combining the forward and reversed embeddings yields a richer final representation. Comprehensive experiments on STS and MTEB benchmarks demonstrate that ReverseEOL significantly improves the performance of existing training-free baselines across a broad range of LLMs with diverse architectures and scales. Extensive ablations and analyses further confirm the necessity of our reversal mechanism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.25256v2">Whose Alignment? Comparing LLM Process Alignment Across Diverse Organizational Decision Contexts</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 Accepted to Pluralistic Alignment Workshop @ ICML 2026, Seoul, South Korea
    </div>
    <details class="paper-abstract">
      Steerable pluralism requires a model to faithfully represent one specified perspective. Organizations are a natural setting for this demand, since they deploy LLMs to make decisions that must reflect their own policy. Yet, most existing work fixes that perspective at the level of individuals or demographic groups. We rely on a decision-policy capturing method to measure process alignment in organizational settings, assessing whether an LLM faithfully reproduces the organization's decision policy rather than merely reaching the same conclusions. We find heterogeneity along two axes. Across models, baseline alignment varies strongly and tracks neither pricing nor general benchmark performance. Across organizations, the structure of alignment changes. In ECHR Article 6 decisions, process alignment predicts output accuracy ($r = 0.85$, $p < .001$), and making the organization's past decision policy explicit improves poorly aligned models. In consumer credit decisions, process alignment is low overall but varies more than output accuracy, and the models resist adopting the organization's weighting of protected attributes. Because historical credit decisions encode potentially discriminatory patterns, higher alignment there is not always desirable. Process-level measurement is therefore necessary, and depending on whether the target policy is normatively desirable, the same procedure can calibrate or audit a model. Deciding which policy to align to, and whether higher alignment is feasible or desirable, makes organizational alignment a pluralistic problem in its own right.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05844v1">GenTI: Benchmarking LLMs for Autonomous IDPS Rule Generation for Unseen Attacks</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Rule-based Intrusion Detection and Prevention Systems (IDPS) offer precise attack detection as well as mitigation, however their manually crafted, signature-driven rules limit adaptability to emerging and zero-day threats. Additionally, existing public datasets (e.g., CICIDS2017, UNSW-NB15) focus on traffic classification and provide little structured information to support automatic rule synthesis or prevention logic. To address this gap, we propose Generative Thread Intelligence (GenTI) \footnote{GenTI refers to the proposed framework, and GTI refers to the dataset.} an LLM-driven benchmark for automatic generation of IDPS rules targeting unseen attacks. The dataset (GTI) aggregates over 150k detection and prevention rules from Snort, Suricata, Emerging Threats, as well as 50k YARA, each annotated with protocol behavior, payload signatures, contextual relationships, mappings to Cyber Threat Intelligence (CTI), along with actionable response types (alert, drop, reject). Moreover, on top of this corpus we design an LLM-based pipeline that transforms analyst prompts and representative payloads into deployable rules via structured prompt engineering, Chain-of-Thought (CoT) reasoning, as well as a Chain-of-Verification (CoVe) loop for syntactic, semantic, and security validation. The generated rules are executed in real time on (Snort/Suricata) and evaluated by syntax accuracy, semantic similarity, CTI coverage, security effectiveness as well as unseen attacks detection. Furthermore, our GenTI instantiation achieves a composite rule-quality score of 89.4\%, with 94.8\% CTI coverage, improving unseen attacks detection from 45\% to 87.4\% and reducing the false-positive rate from 8.5\% to 2.3\%. Overall, GenTI establishes the first large-scale benchmark that tightly couples rule-level CTI with LLM-based automation, enabling adaptive, self-evolving IDPS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14224v3">Knowledge Matters: Injecting Project and Testing Knowledge into LLM-based Unit Test Generation</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 Accepted at the 48th International Conference on Software Engineering(ICSE 2026),13 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Automated unit test generation using large language models (LLMs) holds great promise but often struggles with generating tests that are both correct and maintainable in real-world projects. This paper presents KTester, a novel framework that integrates project-specific knowledge and testing domain knowledge to enhance LLM-based test generation. Our approach first extracts project structure and usage knowledge through static analysis, which provides rich context for the model. It then employs a testing-domain-knowledge-guided separation of test case design and test method generation, combined with a multi-perspective prompting strategy that guides the LLM to consider diverse testing heuristics. The generated tests follow structured templates, improving clarity and maintainability. We evaluate KTester on multiple open-source projects, comparing it against state-of-the-art LLM-based baselines using automatic correctness and coverage metrics, as well as a human study assessing readability and maintainability. Results demonstrate that KTester significantly outperforms existing methods across six key metrics, improving execution pass rate by 5.69% and line coverage by 8.83% over the strongest baseline, while requiring less time and generating fewer test cases. Human evaluators also rate the tests produced by KTester significantly higher in terms of correctness, readability, and maintainability, confirming the practical advantages of our knowledge-driven framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21700v3">Toward Culturally Aligned LLMs through Ontology-Guided Multi-Agent Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 Accepted by ICML 2026 Regular Track
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly support culturally sensitive decision making, yet often exhibit misalignment due to skewed pretraining data and the absence of structured value representations. Existing methods can steer outputs, but often lack demographic grounding and treat values as independent, unstructured signals, reducing consistency and interpretability. We propose OG-MAR, an Ontology-Guided Multi-Agent Reasoning framework. OG-MAR summarizes respondent-specific values from the World Values Survey (WVS) and constructs a global cultural ontology by eliciting relations over a fixed taxonomy via competency questions. At inference time, it retrieves ontology-consistent relations and demographically similar profiles to instantiate multiple value-persona agents, whose outputs are synthesized by a judgment agent that enforces ontology consistency and demographic proximity. Experiments on regional social-survey benchmarks across four LLM backbones show that OG-MAR improves cultural alignment and robustness over competitive baselines, while producing more transparent reasoning traces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05806v1">When Tools Fail: Benchmarking Dynamic Replanning and Anomaly Recovery in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Existing benchmarks evaluate Tool-Integrated Reasoning (TIR) in LLMs on idealized ''happy paths'', largely overlooking real-world tool failures. We introduce ToolMaze, a benchmark for dynamic path discovery and error recovery in TIR agents. To separate systematic replanning from blind trial-and-error, ToolMaze adopts a two-dimensional design: DAG-based topological complexity and a $2 \times 2$ taxonomy of tool perturbations (explicit/implicit, transient/permanent). Evaluations show that perturbations degrade performance across nearly all models, with the sharpest drops under implicit semantic failures. Driven by systemic over-trust in corrupted outputs, Perturbation Recovery Rate (PRR) plummets by around 37\% in these scenarios, while complex topologies trap agents in futile trial-and-error loops. Crucially, agentic fault-tolerance improves with model scale $3.66\times$ slower than basic task execution, highlighting dynamic replanning as a distinct bottleneck unaddressed by model scaling or prompting. Data and code are available at https://github.com/Zhudongsheng75/ToolMaze.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05805v1">From Risk Classification to Action Plan Remediation: A Guardrail Feedback Driven Framework for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 32 pages
    </div>
    <details class="paper-abstract">
      LLM-based guardrails typically safeguard agents by evaluating proposed actions or inputs before execution, producing safety signals such as binary allow/deny decisions, risk categories, and/or explanatory rationales about potential policy violations. However, agent risks often arise when otherwise benign tasks are contaminated by untrusted external content, unsafe instructions, or risky tool use. Existing guardrails often flag the entire task uniformly as unsafe, thereby blocking the threat but sacrificing the benign part. Moreover, existing work largely evaluates guardrails in isolation, leaving unclear whether their interventions lead to safer downstream agent behavior. To address this, we introduce TRIAD (Tripartite Response for Iterative Agent Guardrailing), a guardrail-integrated agent framework that leverages guardrail-generated verbal feedback as a guiding signal to keep the agent aligned with benign objectives at each planning step. We finetune a language model on a self-curated training dataset to output one of three decisions: proceed, refuse, or update, together with structured natural-language feedback. Rather than merely allowing or blocking execution, update guides the agent to revise its plan, avoid harmful components, and preserve the benign task where possible. TRIAD injects this feedback into the agent's context, enabling subsequent plan revision and forming a closed loop between guardrail feedback and agent planning. Extensive experiments on ASB and AgentHarm show that TRIAD reduces the average attack success rate to 10.42%, while achieving the best safety-utility trade-off among guardrail-integrated baselines. Our code is available at: https://github.com/YUHAOSUNABC/TRIAD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05804v1">Can LLMs Be Constrained to the Past? Improving Knowledge Cutoff through Recall-Based Prompting</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Prompted knowledge cutoff instructs a large language model (LLM) to act as if information beyond a specified cutoff date were unavailable. However, prior work mainly relies on direct-answer generation, which struggles when post-cutoff knowledge is not explicitly queried but is only causally related to the question. To address this limitation, we propose two recall-based prompting strategies: Self-Recall (SR), which asks the model to restate its cutoff constraint, and Question-Recall (QR), which requires the model to recall question-relevant information valid under the cutoff. Across three existing benchmarks, our methods outperform both direct-answer prompting and conventional step-by-step reasoning baselines, with particularly strong improvements on counterfactual questions. To investigate robustness across different cutoff settings, we further construct the Multi-cutoff Historical Event Benchmark (MHEB), which evaluates the same question under multiple cutoff years. Results show that knowledge cutoff performance varies with cutoff distance, while combining SR and QR consistently yields the best performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05793v1">CollabBench: Benchmarking and Unleashing Collaborative Ability of LLMs with Diverse Players via Proactive Engagement</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 Accepted by ICML 2026
    </div>
    <details class="paper-abstract">
      While LLM-based agents excel at individual tasks, effective collaboration with realistic human partners remains challenging. Most of the existing conversation-level collaborative studies lack grounded interaction and behavioral execution, motivating the need for cooperative game environments that enable contextualized and immersive collaboration. To this end, this paper proposes CollabBench, a benchmark for evaluating and training collaborative agents in cooperative games. CollabBench features a Diverse Player Profile Simulation pipeline to model varied players behaviors, and a Collaborative Agentic Training paradigm that unifies reasoning, communication, and action via agentic rollouts, optimized with a hybrid reward balancing task efficiency and affective adaptation. We further extend classic environments to CWAH-MultiPlayer and Cook-MultiPlayer for systematic evaluation under diverse personalities. Experiments with efficiency and affective metrics show that our trained models outperform base models, achieving 19.5% higher efficiency and 24.4% improved affective performance. Further analysis reveals key collaborative limitations of existing models and offers insights for future collaborative training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05792v1">Can LLMs Write Correct TLA+ Specifications? Evaluating Natural-Language-to-TLA+ Generation</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 12 pages, 11 tables. Accepted at the 21st International Conference on Software Technologies (ICSOFT 2026); Recommended as Best Paper Award Candidate
    </div>
    <details class="paper-abstract">
      TLA+ has supported industrial verification at companies such as Amazon and Microsoft, yet writing correct TLA+ specifications from natural language still requires time and expertise, which limits adoption. LLMs show promise, but no prior study measures whether they produce semantically correct TLA+ specifications from natural language. This paper presents the first systematic evaluation of LLM-based TLA+ specification synthesis from natural language. Our study evaluates 30 LLMs across eight families on a curated dataset of 205 TLA+ specifications: 25 open-weight models across four prompting strategies (2,600 runs) and 5 proprietary models under few-shot prompting (130 runs), all validated by the SANY parser and TLC model checker. LLMs achieve up to 26.6% syntactic correctness but only 8.6% semantic correctness, with successes exclusive to progressive prompting. Results show that model size does not predict quality, e.g., DeepSeek r1:8b outperforms its 70B variant across all strategies, which suggests the importance of reasoning alignment for formal languages. Code-specialized models consistently underperform due to negative transfer from mainstream language training. We identify five recurring hallucination categories, all traceable to specific training data biases. These results suggest that current LLMs do not generate reliable TLA+ specifications without expert oversight. We release the evaluation framework, code, and dataset to support reproducibility and future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16475v2">Breaking the Chain: A Causal Analysis of LLM Faithfulness to Intermediate Structures</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 20 pages, 4 figures, 7 tables
    </div>
    <details class="paper-abstract">
      In schema-guided reasoning (SGR) pipelines, LLMs produce explicit intermediate structures -- rubrics, checklists, or verification queries -- before committing to a final decision. SGR is increasingly adopted because it promises controllability: practitioners expect to inspect, edit, and override these structures to steer the outcome. But does the promise hold? We introduce a causal evaluation protocol to measure it: by selecting tasks where a deterministic function maps intermediate structures to decisions, every controlled edit implies a unique correct output. Across 12 models and 4 benchmarks, models appear self-consistent with their own intermediate structures but fail to update predictions after intervention -- revealing that apparent faithfulness is fragile once the intermediate structure changes. When derivation of the final decision from the structure is delegated to an external tool, this fragility largely disappears; stronger prompting yields only limited improvements, while preference optimization substantially improves intervention faithfulness. Overall, intermediate structures in schema-guided pipelines function as influential context rather than stable causal mediators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.03217v2">Moral Sensitivity in LLMs: A Tiered Evaluation of Contextual Bias via Behavioral Profiling and Mechanistic Interpretability</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in settings that require nuanced ethical reasoning, yet existing bias evaluations treat model outputs as simply "biased" or "unbiased." This binary framing misses the gradual, context-sensitive way bias actually emerges. We address this gap in two stages: behavioral profiling and mechanistic validation. In the behavioral stage, we introduce the Moral Sensitivity Index (MSI), a metric that quantifies the probability of biased output across a graduated, seven-tier stress test ranging from abstract numerical problems to scenarios rooted in historical and socioeconomic injustice. Evaluating four leading models (Claude 3.5, Qwen 3.5, Llama 3, and Gemini 1.5), we identify distinct behavioral signatures shaped by alignment design: for instance, Gemini 1.5 reaches 72.7% MSI by Tier 5 under socioeconomic framing, while Claude exhibits sharp suppression consistent with identity-based safety training. We then verify these behavioral patterns mechanistically. We select criminal-bias scenarios, which produced the highest MSI scores across models, as probes and apply logit lens, attention analysis, activation patching, and semantic probing to a controlled set of six models spanning three capability tiers: small language models (SLMs), instruction-tuned base models, and reasoning-distilled variants. Circuit-level analysis reveals a U-curve of bias: SLMs exhibit strong criminal bias; scaling to instruction-tuned models eliminates it; reasoning distillation reintroduces bias to SLM-like levels despite identical parameter counts, suggesting distillation compresses reasoning traces in ways that reactivate shallow statistical associations. Critically, the socially loaded cues that drive high MSI scores activate the same bias-driving circuits identified mechanistically, providing cross-stage validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05713v1">Beyond Generative Decoding: Discriminative Hidden-State Readout from a Native Omni-Modal LLM for Multimodal Sentiment Analysis</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 18 pages, 4 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Multimodal sentiment analysis (MSA) infers human affect from language, acoustic, and visual signals. Recent methods increasingly adapt large multimodal models (LMMs) via generative readout: prompting the model to emit a sentiment score as a text string. While convenient, this ties continuous regression to discrete autoregressive decoding, incurring unmeasured costs. We revisit this readout mechanism and propose a discriminative formulation built on the Thinker module of a native omni-modal LLM (Qwen2.5-Omni-7B). Instead of text decoding, we map the final-layer hidden state of the last non-padding token to a continuous score via a lightweight regression head in a single forward pass. Using 4-bit quantization and low-rank adaptation (QLoRA), the entire 7B pipeline -- including video and audio processing -- trains on a single consumer GPU (RTX 5090, 32 GB) with 10-21 GB peak memory and 1.14% trainable parameters. Through a controlled comparison fixing the backbone, data, and LoRA configuration, we isolate the impact of the readout. On CMU-MOSI and CMU-MOSEI, our discriminative readout reaches state-of-the-art accuracy without task-specific feature engineering (MOSI: MAE 0.551, Corr 0.888; MOSEI: MAE 0.506, Corr 0.790) and exhibits strong multi-seed stability. In contrast, the generative readout -- even after equivalent supervised training -- more than doubles the mean absolute error, yields unparsable or out-of-range outputs (2.8% zero-shot), and suffers from higher latency. Modality ablations reveal a text-dominant regime on CMU-MOSI. Our findings indicate that how an LMM is read out is as consequential as how it is trained, demonstrating that a discriminative readout offers a more accurate, efficient, and reliable alternative for continuous MSA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05711v1">Beyond tokens: a unified framework for latent communication in LLM-based multi-agent systems</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Multi-agent systems built on large language models (LLMs) have become a prevailing paradigm for tackling complex reasoning, planning, and tool-use tasks. The dominant communication protocol in such systems is natural language: agents exchange messages token-by-token, verbalising their internal reasoning so that peers can read, verify, and respond. While convenient and interpretable, this protocol suffers from three structural drawbacks -- high inference cost, irreversible information loss during discretization, and ambiguity/redundancy of natural language. A growing body of work therefore explores an alternative protocol -- latent communication -- in which agents exchange continuous representations (embeddings, hidden states, or KV-caches) directly, bypassing the bottleneck of text generation. This paper presents a unified framework for organising the rapidly expanding literature on latent communication. We analyse existing methods along three orthogonal axes: (1) WHAT information is communicated (Embeddings, Hidden States, KV-Caches, or other continuous state); (2) WHICH sender-receiver alignment is used (latent-space alignment and layer alignment); and (3) HOW the communicated information is fused into the receiver (concatenation, prepending, mathematical operations, cross-attention, or cache restoration). Under this 3-axis framework, we systematically categorise eighteen representative methods proposed between 2024 and 2026, identify five major design patterns, and surface a set of open challenges -- including cross-architecture alignment, security of latent channels, compression for edge deployment, and the relationship between latent communication and latent chain-of-thought. We hope that this framework both lowers the barrier to entry for new researchers and provides a vocabulary for comparing future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04397v2">Context-as-AI-Service: Surfacing Cross-File Dependency Chains for LLM-Generated Developer Documentation</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 8 pages, 2 figures, 4 tables
    </div>
    <details class="paper-abstract">
      LLM agents increasingly write and maintain developer documentation, but usefulness and accuracy often rely on dependency chains that are not obvious to follow. Even with more files in context, the agent must still decide which cross-file dependencies to trace. We present Context-as-AI-Service (CAIS), a retrieval layer that LLM agents query to find evidence across the codebase as they review or generate documentation. CAIS indexes source code, API references, and upstream documentation, then enables agents to query the index through tool calls that combine keyword and semantic search. We evaluate CAIS in two case studies using Claude Sonnet 4.6 on a production SDK: improving API reference comments in a core source file and validating an LLM-generated tutorial. In both studies, the baseline already had ordinary repository tools such as file reads, keyword search, and symbol navigation. CAIS adds a retrieval layer on top, so the comparison isolates added retrieval rather than basic repository access. In the API-reference review, the CAIS-augmented agent produced the same 5 missing-documentation fixes as the baseline and surfaced 4 findings the baseline missed: 2 cross-file factual errors and 2 underspecified API comments. In the tutorial validation, it surfaced 1 executable bug, 1 API-usage improvement, and 2 missing prerequisites that the baseline pipeline did not catch. These findings required tracing non-obvious dependency chains across utility files, framework internals, usage examples, tests, and component-creation logic. Over five runs per condition, adding CAIS reduced wall-clock time by 22% to 34% across the two tasks and lowered input-token usage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05682v1">Beyond Output Matching: Preserving Internal Geometry in NVFP4 LLM Distillatio</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 13 pages,1 figures
    </div>
    <details class="paper-abstract">
      Demand for low-precision inference, including NVFP4-based approaches, has grown as large language models are increasingly deployed in latency and cost constrained production environments. Quantization-aware distillation (QAD) helps recover accuracy lost under low bit quantization by training a quantized student to match the output distribution of a frozen higher precision teacher via a KL-divergence loss. In this work, we first provide a representation level diagnosis of QAD: output matching alone can mask internal degradation, because many intermediate activation geometries can yield similar teacher-aligned logits. Using CKA, we show that KL-only QAD can reduce layerwise representational similarity relative to the BF16 teacher, with especially severe drift in RL-post-trained models. This drift correlates with downstream bottlenecks on reasoning and coding tasks, suggesting that low bit recovery requires preserving internal geometry rather than matching outputs alone. Motivated by this finding, we propose \textbf{CKA-QAD}, a CKA-guided representational alignment method for NVFP4 QAD and low bit LLM accuracy recovery. The method adds a lightweight regularizer that preserves internal representational geometry during distillation by aligning layerwise Gram matrices through CKA. Across Nemotron 3 Nano and Qwen3-4B-Thinking-2507, CKA-QAD substantially improves representational alignment and improves downstream reasoning and coding accuracy with modest training overhead. Our findings position CKA-guided representational alignment as a practical complement to output matching for quantized LLM recovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05680v1">CASS-RTL: Correctness-Aware Subspace Steering for RTL Generation with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 Accepted to the IEEE International Conference on LLM-Aided Design (LAD '26)
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled the automatic synthesis (generation) of register-transfer level (RTL) code from natural language instructions, offering a promising pathway to accelerate chip design. Unlike typical natural language (and software coding) tasks, LLM-based RTL code generation demands strict cycle accuracy with concurrency, where minor logical errors can render a circuit unusable or insecure. While prior work has explored hallucination mitigation via external verification, self-evaluation prompts, retrieval-augmented prompting, domain specific fine-tuning, agentic solutions, and reasoning, these approaches largely overlook the attention-oriented internal mechanisms of LLMs that may inherently correlate with RTL correctness. This work proposes CASS-RTL, a first-of-its-kind framework for discovering and leveraging LLMs' correctness-aware components to guide RTL generation toward functionally accurate outputs. We (i) identify attention heads whose activation patterns consistently differentiate correct from incorrect RTL; (ii) construct a low-dimensional subspace capturing correctness-relevant signals; and (iii) design a lightweight, geometry-aware intervention that steers the model at inference time. CASS-RTL is fully model-agnostic, requires no additional supervision or retraining, and readily integrates into existing models. Empirically, we evaluate CASS-RTL on multiple models and observe 10%-20% improvement in pass@1/5/10 accuracy on VerilogEval and 5% improvement on CVDP, demonstrating the effectiveness of our method in enhancing reliability without sacrificing model efficiency or requiring a large labeled dataset for fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.14291v3">Advances in Temporal Point Processes: Bayesian, Neural, and LLM Approaches</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Temporal point processes (TPPs) are stochastic process models used to characterize event sequences occurring in continuous time. Traditional statistical TPPs have a long-standing history, with numerous models proposed and successfully applied across diverse domains. In recent years, advances in deep learning have spurred the development of neural TPPs, enabling greater flexibility and expressiveness in capturing complex temporal dynamics. The emergence of large language models (LLMs) has further sparked excitement, offering new possibilities for modeling and analyzing event sequences by leveraging their rich contextual understanding. This survey presents a comprehensive review of recent research on TPPs from three perspectives: Bayesian, deep learning, and LLM approaches. We begin with a review of the fundamental concepts of TPPs, followed by an in-depth discussion of model design and parameter estimation techniques in these three frameworks. We also revisit classic application areas of TPPs to highlight their practical relevance. Finally, we outline challenges and promising directions for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05670v1">Do More Agents Help? Controlled and Protocol-Aligned Evaluation of LLM Agent Workflows</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 https://github.com/LINs-lab/MASArena/tree/BenchAgent
    </div>
    <details class="paper-abstract">
      Does adding more agents help an LLM workflow once compared systems share the same benchmark loader, tool access, answer contract, usage accounting, and trajectory logging? We introduce BenchAgent, an evaluation framework that places single-agent, fixed multi-agent (MAS), and evolving MAS workflows under one normalized execution and logging protocol. BenchAgent evaluates these substrate-internal workflows across ten reasoning, coding, and tool-use benchmarks with GPT-4.1, and separately reports a Protocol-Aligned External (PAE) GAIA study of a runtime-generated workflow. Under SI conditions, at most one of six tested MAS exceeds the matched single-agent anchor on benchmark-balanced average accuracy: EvoAgent lies within the Wilson one-run guidance, while the remaining five trail by 2.56-11.29 points and occupy more expensive accuracy-cost trade-offs. On the PAE GAIA snapshot, a Claude-Code-style runtime workflow reaches 66.72% overall and 69.23% on Level 3, more than 20 points above the strongest non-Claude baseline, Jarvis, a fixed MAS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.02987v2">Large-Scale LLM Inference with Heterogeneous Workloads: Prefill-Decode Contention and Asymptotically Optimal Control</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are rapidly becoming critical infrastructure for enterprise applications, driving unprecedented demand for GPU-based inference services. A key operational challenge arises from the two-phase nature of LLM inference: a compute-intensive \emph{prefill} phase that processes user input, followed by a memory-bound \emph{decode} phase that generates output tokens. When these phases share GPU resources, prefill tasks throttle the processing speed of concurrent decodes, creating state-dependent contention. This contention is further complicated by workload heterogeneity, as different applications exhibit vastly different input and output lengths. We develop a stochastic control framework for scheduling heterogeneous LLM workloads across large GPU clusters. We formulate LLM inference as a multiclass many-server queueing network with state-dependent service rates, grounded in empirical iteration-time measurements. We analyze the fluid approximation of this system and solve steady-state linear programs that characterize optimal resource allocation. We design gate-and-route policies that regulate prefill admission and decode routing, and prove that they are asymptotically optimal in the many-GPU limit under both bundled and separate token-pricing schemes. We further extend the framework to incorporate Service Level Indicators (SLIs) such as latency and fairness, providing a general approach to constrained scheduling. Numerical experiments calibrated to empirical iteration-time data demonstrate that our policies outperform standard serving heuristics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.00644v2">ForeSci: Evaluating LLM Agents for Forward-Looking AI Research Judgment</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      AI research often requires decisions before future evidence exists: which bottleneck to attack, which direction to pursue, or where a project should be positioned. We introduce ForeSci, a temporally controlled benchmark for evaluating whether LLM agents can make such forward-looking research judgements from historical evidence. ForeSci contains 500 tasks across four fast-moving AI domains and four decision families. Each task is paired with a cutoff-aligned offline knowledge base; post-cutoff papers are hidden during generation and used only for validation. To avoid random future-event prediction, tasks are derived from pre-cutoff taxonomy branches and evidence signals, and answer-generation backbones are selected to precede the task cutoffs. We evaluate native LLMs, Hybrid RAG, and three research-agent adaptations across four backbones. Results show that explicit evidence organization improves traceability and factual support, but gains depend strongly on the decision family. Diagnostics reveal a recurring evidence-decision decoupling: agents may cite relevant evidence while forecasting the wrong research object. ForeSci turns forward-looking AI research judgement into a controlled benchmark for evaluating research agents as decision-making systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05632v1">Evaluation of LLMs for Mathematical Formalization in Lean</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 15 pages, 13 figures, 10 tables. Comments welcome!
    </div>
    <details class="paper-abstract">
      Within the past few years, the ability of Large Language Models (LLMs) to generate formal mathematical proofs has improved drastically. We provide a comparison of various LLMs' effectiveness in producing formal proofs in Lean 4 with the goal of assisting those seeking to use LLMs to support their own projects. We utilize both pass@$k$ and refine@$k$ metrics as the benchmark for our comparison and evaluate on subsets of both miniF2F and miniCTX datasets. Our testing shows that overall, Gemini 3.1 Pro and Claude Opus 4.7 perform best. Gemini 3.1 Pro achieved a 92\% success rate on miniF2F via refine@32 whereas Opus 4.7 achieved a 86\% success rate on miniCTX via refine@32. When taking cost into account, NVIDIA Nemotron 3 Super and GPT-OSS 120B were the most efficient, with competitive accuracies and average costs of $<\$0.01$ per correct proof.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05616v1">What's in a Name? Morphological Shortcuts by LLMs in Pharmacology</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 22 pages
    </div>
    <details class="paper-abstract">
      The morphological form of a word can often give cues to its meaning, but purely relying on these mappings can lead to overgeneralization in high-stakes domains. In the medical domain, for instance, LLMs can confidently reason about fictitious drugs from their affixes alone (e.g., wugcillin) and generate plausible-looking clinical content. We present a behavioral and mechanistic study of LLM "affix heuristics" in pharmacology. Using fictitious drug names built from real affixes, we show that affix signals alone elicit class-level pharmacological responses. We introduce a framework for identifying whether a model's drug semantics are driven mainly by the affix, the stem, or the drug name as a whole. Applied across 653 drugs, our framework reveals that models often induce drug meaning primarily through affix cues, yet rarely explicitly indicate this reliance, and sometimes incorrectly conflate properties among affix-sharing drugs. Activation patching across models further localizes this behavior to early-mid layers. These findings show that morphological shortcuts pose a subtle but measurable risk to safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05614v1">Safety Paradox: How Enhanced Safety Awareness Leaves LLMs Vulnerable to Posterior Attack</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are rigorously aligned to refuse harmful requests, a process that inherently cultivates a latent capacity to evaluate and recognize unsafe content. In this work, we reveal that this advanced safety awareness inadvertently introduces a fatal vulnerability. We introduce Posterior Attack, a single-query jailbreak that bypasses guardrails by prompting the model to generate the exact harmful response its internal classifier would normally flag as unsafe. Through extensive empirical evaluation across 30 open-source LLMs (up to 35B parameters in size) and frontier models (e.g., GPT-5, Claude 4.6), we observe a striking phenomenon: models with superior safety-judgment capabilities are disproportionately more susceptible to this exploitation. To explain this, we formalize the Safety Paradox, analytically showing that monotonic improvements in safety alignment naturally amplify posterior vulnerability. Finally, we establish a causal link via reinforcement learning interventions, exemplifying that artificially degrading a model's safety judgment immunizes it against the attack, whereas enhancing judgment exacerbates the vulnerability. Our findings highlight potential flaws in current alignment paradigms, indicating that defense mechanisms may require further structural refinement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05610v1">Predictable Scaling Laws of Optimal Hyperparameters for LLM Continued Pre-training</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      The efficacy of continued pre-training for Large Language Models (LLMs) hinges upon hyperparameter configurations, such as learning rate and batch size. However, current practices often rely on heuristics or grid searches, leading to training instability and excessive costs. In this work, we first empirically discover that optimal hyperparameters follow stable and predictable scaling laws throughout the continued pre-training process. Leveraging these insights, we propose a novel framework to establish quantitative relationships between compute budget and optimal hyperparameters for a given checkpoint. Our approach has two stages: (1) \textit{Empirical Law Discovery}, where we train small-scale proxy models to derive functions mapping compute budget to optimal hyperparameters via standard loss-compute scaling laws; and (2) \textit{State-Aware Hyperparameter Prediction}, where we evaluate an initial checkpoint's validation loss and use the inverse scaling law to estimate its \textit{equivalent pre-training compute} -- the compute needed to achieve the same loss from scratch. Combining this with the planned compute budget, we predict optimal hyperparameters for the target run. Empirical results demonstrate that our method reduces the hyperparameter search overhead by up to 90\% while achieving comparable or superior performance relative to baselines. This model-agnostic framework generalizes across architectures, providing a principled and efficient methodology for diverse continued pre-training scenarios starting from any given point.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05609v1">SlotGCG: Exploiting the Positional Vulnerability in LLMs for Jailbreak Attacks</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are widely deployed, identifying their vulnerability through jailbreak attacks becomes increasingly critical. Optimization-based attacks like Greedy Coordinate Gradient (GCG) have focused on inserting adversarial tokens to the end of prompts. However, GCG restricts adversarial tokens to a fixed insertion point (typically the prompt suffix), leaving the effect of inserting tokens at other positions unexplored. In this paper, we empirically investigate \emph{slots}, i.e., candidate positions within a prompt where tokens can be inserted. We find that vulnerability to jailbreaking is highly related to the selection of the \emph{slots}. Based on these findings, we introduce the \textit{Vulnerable Slot Score} (VSS) to quantify the positional vulnerability to jailbreaking. We then propose SlotGCG, which evaluates all slots with VSS, selects the most vulnerable slots for insertion, and runs a targeted optimization attack at those slots. Our approach provides a position-search mechanism that is attack-agnostic and can be plugged into any optimization-based attack, adding only 200ms of preprocessing time. Experiments across multiple models demonstrate that SlotGCG significantly outperforms existing methods. Specifically, it achieves 14\% higher Attack Success Rates (ASR) over GCG-based attacks, converges faster, and shows superior robustness against defense methods with 42\% higher ASR than baseline approaches. Our implementation is available at \href{https://github.com/youai058/SlotGCG}{https://github.com/youai058/SlotGCG}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05567v1">ZERO-APT: A Closed-Loop Adversarial Framework for LLM-Driven Automated Penetration Testing under Intelligent Defense</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      LLM-driven automated penetration testing agents are typically evaluated against static targets that neither detect nor respond to attacks, so their behavior under intelligent defense remains untested. The causal consistency of multi-step attack chains likewise hinges on unstable LLM reasoning, and agent decisions remain opaque to human analysts. These three shortcomings, in realism, consistency, and auditability, are usually patched in isolation. We present ZERO-APT, a turn-based attacker-defender-judge framework that addresses them within a single architecture. For realism, ZERO-APT embeds a configurable LLM Defender that consumes Sysmon telemetry and detects attacks in real time, exposing the attacker to a live opponent rather than a passive target. For consistency, three architectural mechanisms move causal consistency from unstable LLM reasoning into enforced system architecture: separation of planning from execution, multi-dimensional ReAct feedback, and a hard-constraint-filtered action library. For auditability, a dedicated Judge agent adjudicates each round, maintains global state, and emits structured post-hoc CTI reports that make every decision traceable. We evaluate a Windows Server 2022 post-exploitation prototype across five scenarios with three Defender configurations. ZERO-APT reaches 79\% attack success rate (Aurora 22\%, PentestGPT 39\%), a Causal Consistency Score of 0.860 (Aurora 0.930, Claude Code 0.520), and end-to-end decision auditability through structured CTI reports. We release the benchmark to support evaluation of penetration agents under intelligent defense.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05563v1">SoCRATES: Towards Reliable Automated Evaluation of Proactive LLM Mediation across Domains and Socio-cognitive Variations</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Evaluating LLM mediators remains challenging, as mediation unfolds as a real-time trajectory shaped by disputants' shifting emotions, intentions, and context. Existing testbeds rely on a few expert-authored domains, vary mainly strategic posture, and score every turn against every topic, introducing off-topic noise. We introduce SoCRATES, a benchmark for evaluating proactive LLM mediators in realistic, multi-domain testbeds. It constructs scenarios from real conflicts through an agentic pipeline across eight domains, probes five socio-cognitive adaptation axes (strategic posture, party composition, history length, emotional reactivity, and cultural identity), and scores each topic only on the turns that advance it via a topic-localized evaluator. The evaluator reaches 0.82 alignment with human experts, more than doubling a per-turn baseline. Benchmarking eight frontier LLMs, we find that even the strongest mediator closes only about a third of the unmediated consensus gap under diverse and realistic testbeds, with performance varying sharply by socio-cognitive axis, highlighting that progress lies in social adaptation to diverse conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.23600v2">Personality Shapes Gender Bias in Persona-Conditioned LLM Narratives Across English and Hindi: An Empirical Investigation</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in persona-driven applications such as education, customer service, and social platforms, where models are prompted to adopt specific personas when interacting with users. While persona conditioning can improve user experience and engagement, it also raises concerns about how personality cues may interact with gender biases and stereotypes. In this work, we present a controlled study of persona-conditioned story generation in English and Hindi, where each story portrays a working professional in India producing context-specific artifacts (e.g., lesson plans, reports, letters) under systematically varied persona gender, occupational role, and personality traits from the HEXACO and Dark Triad frameworks. Across 23,400 generated stories from six state-of-the-art LLMs, we find that personality traits are significantly associated with both the magnitude and direction of gender bias. In particular, Dark Triad personality traits are consistently associated with higher gender-stereotypical representations compared to socially desirable HEXACO traits, though these associations vary across models and languages. Our findings demonstrate that gender bias in LLMs is not static but context-dependent. This suggests that persona-conditioned systems used in real-world applications may introduce uneven representational harms, reinforcing gender stereotypes in generated educational, professional, or social content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05558v1">Autoregressive Diffusion World Models for Off-Policy Evaluation of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Evaluating large language model (LLM) agents in multi-turn interactive environments is expensive and risky, as it requires online environment interaction. We propose ADWM (Autoregressive Diffusion World Model), an evaluation framework that estimates the performance of a new LLM agent policy purely from pre-collected trajectories. The core idea is to learn a latent diffusion world model that simulates how the environment responds to the evaluation policy, without ever executing it in the real environment. Existing diffusion-based OPE methods guide full trajectories in a single pass by jointly diffusing states and actions, an assumption that breaks down for LLM agents whose actions are discrete text that must be sampled from the policy after observing the environment. Unlike autoregressive world models that suffer from compounding errors, ADWM models each transition as an independent denoising process, enabling reliable step-by-step rollouts where the world model and agent alternate in causal order. Crucially, the LLM agent under evaluation directly guides the diffusion generation at each step via a policy-conditioned score function, ensuring that simulated trajectories accurately reflect its decision-making patterns. Empirically, ADWM achieves accurate value estimates and evaluation reliability across diverse multi-turn agent tasks, demonstrating its promise as a practical framework for offline LLM agent evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05548v1">ADK Arena: Evaluating Agent Development Kits via LLM-as-a-Developer</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 Work in Progress
    </div>
    <details class="paper-abstract">
      The rapid proliferation of Agent Development Kits (ADKs), SDK-level frameworks for building LLM-powered autonomous agents, has outpaced any empirical understanding of how framework choice affects agent performance. We propose \textbf{LLM-as-a-Developer}, a methodology that replaces human developers with an LLM coding agent that learns each framework's API from documentation, writes agent code, and iteratively repairs it through a validate-and-feedback loop until tests pass. By holding the developer constant and varying only the framework, generation effort becomes a quantitative proxy for API usability and the resulting agents provide a controlled measure of framework effectiveness. We implement this in \textbf{ADK Arena}, a fully automated pipeline with per-framework Docker isolation, a three-level validation pipeline, and benchmark adapters for SWE-bench, $τ^2$-bench, Terminal-Bench, and MCP-Atlas. Evaluating all 51 popular Python ADK frameworks (204 agent--benchmark pairs), we find that: (1)~generation succeeds for 57\% of runs, and its cost varies 5.6$\times$ across frameworks (\$0.6 to \$3.4 per agent), a quantitative proxy for API complexity, though cost alone does not predict success; (2)~no single framework dominates: the best single-benchmark ADK agents resolve up to 80\% of tasks and can even \emph{beat} general-purpose frontier coding agents at a fraction of the cost, yet the median framework resolves only 32\%; (3)~across information-source ablations, genuine framework usage stays within a narrow 28--40\% band (highest with raw source access and still 33\% with no reference material at all), indicating that documentation, source code, and parametric knowledge are largely substitutable rather than any one being a hard bottleneck.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.13628v2">Compact LLM Deployment and World Model Assisted Offloading in Mobile Edge Computing</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 16 pages, 10 figures
    </div>
    <details class="paper-abstract">
      This paper investigates compact large language model (LLM) deployment and world-model-assisted inference offloading in mobile edge computing (MEC) networks. We first propose an edge compact LLM deployment (ECLD) framework that jointly applies structured pruning, low-bit quantization, and knowledge distillation to construct edge-deployable LLM variants, and we evaluate these models using four complementary metrics: accessibility, energy consumption, hallucination rate, and generalization accuracy. Building on the resulting compact models, we formulate an MEC offloading optimization problem that minimizes the long-term average inference latency subject to per-device energy budgets and LLM-specific quality-of-service constraints on effective accuracy and hallucination. To solve this problem under unknown and time-varying network dynamics, we develop a world model-proximal policy optimization (PPO) algorithm, which augments an on-policy PPO algorithm with a learned recurrent world model that provides improved value targets and short imagination rollouts. Extensive experiments on Llama-3.1-8B, Qwen3-8B, and Mistral-12B show that ECLD compresses base models by about 70-80% in storage (i.e., from 15.3 GB to 3.3 GB for Llama-3.1-8B) and reduces per-query energy consumption by up to 50%, while largely preserving accuracy and often lowering hallucination compared with quantization-only or pruning-only baselines. Moreover, they also show that world model-PPO speeds up convergence by about 50%, improves the final reward by 15.8% over vanilla PPO, and reduces average inference latency by 12-30% across different user populations, while satisfying the accuracy and hallucination constraints and approaching the generation quality of always-offloading with much of the efficiency of local execution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05523v1">CHASE: Adversarial Red-Blue Teaming for Improving LLM Safety using Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 Under Review at ARR
    </div>
    <details class="paper-abstract">
      Despite advances in safety alignment, prompt-rewriting attacks such as persona modulation, fictional framing and persuasion-based reformulation, can bypass safety filters even on frontier models. Existing defenses either rely on non-scalable human curation or white-box optimisation that overfits to specific model internals, leaving aligned models brittle against the very class of adaptive black-box adversaries they will face in deployment. To address this gap, we introduce CHASE (Co-evolutionary Hardening through Adversarial Safety-Escalation), a closed-loop red-blue teaming framework in which a black-box attacker and a safety-aligned defender co-evolve. The attacker is trained via Group Relative Policy Optimization (GRPO) under a multiplicative reward that jointly enforces bypass effectiveness and intent fidelity, while the defender is hardened on the harvested adversarial rewrites through a two-stage GRPO + rejection-sampled SFT pipeline balanced with benign data. Evaluated on BeaverTails and JailbreakBench against five held-out attack families (PAIR, TAP, AutoDAN, PAP, Translation), CHASE cuts mean StrongREJECT score by 43.2\% with 0\% false-refusal on benign prompts. Beyond the headline result, CHASE shows that template-free RL exploration recovers latent attack primitives that transfer across mechanistically distinct attack families, suggesting a path toward LLM safety hardening that generalises beyond the narrow distributions achieved thus far in adversarial training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06470v1">PC Layer: Polynomial Weight Preconditioning for Improving LLM Pre-Training</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      We propose a preconditioning (PC) layer, a weight parameterization via polynomial preconditioner that ensures stable weight conditioning throughout LLM training. The PC module reshapes the singular-value spectrum of weight matrices via low-degree polynomial preconditioning. After training, the preconditioned weights can be merged back into the original architecture, incurring no inference overhead. We demonstrate the advantage of the proposed PC layer over standard transformers in Llama-1B pre-training, for both the AdamW and Muon optimizers. Theoretically, we justify this spectrum-control principle by proving that uniformly bounding each layer's singular values ensures geometric convergence of gradient descent to global minima, for certain deep linear networks. Our code is available at https://github.com/Empath-aln/PC-layer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06460v1">Will the Agent Recuse Itself? Measuring LLM-Agent Compliance with In-Band Access-Deny Signals</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 8 pages, 1 figure. Code, specification, and experiment harness: https://github.com/mthamil107/Recuse
    </div>
    <details class="paper-abstract">
      As autonomous LLM agents increasingly hold real credentials and operate infrastructure without a human in the loop, operators have no standard way to tell an agent that a resource is off-limits. Access controls either let the agent in (it has valid credentials) or hard-fail it (indistinguishable from any other client). We propose a third mode: a lightweight, published in-band deny signal -- the Recuse Signal -- that a server emits over a protocol's existing channels (an SSH banner, a PostgreSQL NOTICE) asking a connecting automated agent to voluntarily withdraw. This is a cooperative governance control, the robots.txt analogue for live access; it is explicitly not a security boundary. Its value is entirely empirical and, to our knowledge, unmeasured: do compliant LLM agents actually honor such a signal? We define the signal as an open mini-standard, implement two zero- or low-footprint adapters (an SSH banner/PAM hook and a PostgreSQL wire-protocol proxy), deploy them on a live production host, and run a controlled experiment in which fresh agents are given a benign operations task and observed for recusal. In a pilot (SSH; OpenAI GPT-4o and GPT-4o-mini; and Claude Code as a deployed agent), the signal cleanly induces recusal -- 100% recusal when present versus 100% task completion in a no-signal control -- and, revealingly, behaves as a cooperative rather than absolute signal: an explicit operator-authorization framing flips the most capable model to proceed, while other agents continue to defer to the on-host policy. We release the standard, adapters, and experiment harness for reproduction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06443v1">Revising Context, Shifting Simulated Stance: Auditing LLM-Based Stance Simulation in Online Discussions</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Large language models are increasingly used to simulate social media users and infer how individuals may respond to online discussions. However, it remains unclear whether these simulations reflect precise user-specific beliefs or whether they are highly sensitive to semantically independent changes in conversational contexts. In this work, we study counterfactual context revision as a framework for auditing LLM-based stance simulation. Given an original online conversation, we first infer a target user's stance toward a specific topic. We then apply controlled revision strategies to the conversational context and simulate the user's stance again under the revised context. We compare text-only revision strategies with a multimodal one that incorporates meme-based context and evaluate two main effectiveness metrics, i.e., average directional stance shift and stance transition rate. The results reveal effective and robust stance transitions in both text-only and multimodal strategies across different polarization-preference mechanisms. Our study contributes an evaluation framework for understanding the context sensitivity of LLM-based stance simulation. More broadly, it highlights both the promise and risk of using LLMs to simulate online opinion dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06420v1">A Komi-Yazva--Russian Parallel Corpus and Evaluation Protocol for Zero- and Few-Shot LLM Translation</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 18 pages, 6 tables, 3 figures
    </div>
    <details class="paper-abstract">
      We present the first Komi-Yazva--Russian parallel corpus together with an explicit evaluation protocol for studying LLM translation in an endangered, extremely low-resource setting. The dataset contains 457 aligned sentence pairs from 74 narrative texts and is accompanied by documented provenance, sentence-level alignment, and story identifiers that enable leakage-aware evaluation. We use this setup to compare modern large language models on Komi-Yazva-to-Russian translation under severe parallel-data scarcity in zero-shot and retrieval-based few-shot regimes. The protocol includes story-level cross-validation, deterministic retrieval for few-shot prompting, strict validation of generated outputs, complementary reference-based and judge-based metrics, and story-level uncertainty estimates. Across models, LLMs produce non-trivial translations, but performance varies strongly by model family and prompting regime. Retrieval-based few-shot prompting consistently improves over zero-shot prompting, while gains beyond a small retrieved context remain limited. The results show that evaluative conclusions in this setting depend materially on metric choice and failure handling, so the paper frames the corpus as both a dataset contribution and a reproducible evaluation testbed for endangered-language machine translation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06399v1">CollabSim: A CSCW-Grounded Methodology for Investigating Collaborative Competence of LLM Agents through Controlled Multi-Agent Experiments</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Multi-agent systems (MAS) built on large language models have shown growing promise, with their effectiveness resting on agents' ability to coordinate through text-based channels much as human teams do. Yet recent study suggests that MAS often falter not because agents lack individual task-solving ability, but because they lack collaborative competence: the capacity to establish common ground, maintain shared task understanding, balance individual and collective incentives, and repair misalignment as interaction unfolds. Decades of research in Computer-Supported Cooperative Work have characterized these requirements for human teams coordinating under constrained communication, yet existing MAS evaluations focus mainly on task outcomes or single-agent proficiency in reasoning, planning, and tool use. To enable a systematic analysis of agents' collaborative competence in MAS, we introduce CollabSim, a configurable simulation framework that combines a theory-grounded definition of collaborative capabilities, controlled manipulation of interaction conditions, and action-level probing of agents' internal states. Experiments across four LLMs show that CollabSim can capture condition effects, separate model performance patterns, and reveal task-dependent effects of agent design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.02776v3">Topics as Proxies for Sociodemographics: How Conversational Context Affects LLM Answers</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      When large language models (LLMs) are used in high-stakes scenarios, such as legal, medical and financial advice, even a single conversation history is enough to drive differences in outcomes between users. Prior work has demonstrated that this results in outcome disparities between sociodemographic groups, with some groups receiving more advantageous outcomes than others. In this work, we demonstrate that LLMs actually struggle to infer user sociodemographics from a single conversation history and that although there are disparities between sociodemographic groups, they are minimal in magnitude. To investigate what the main driver of these disparities is, we compare user sociodemographics to a range of (psycho)linguistic features of conversations, including conversation topic, emotions, and readability. We find that conversation topics are most predictive of LLM-generated advice within a conversational context, which, to some extent, function as proxies for sociodemographic groups and often affect advice in unpredictable ways. This is cause for concern and highlights the need for future research to better understand and, if needed, mitigate the effect of conversational context on LLM outputs in high-stakes scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06350v1">EDIT: Evidence-Diagnosed Intervention Training for Rule-Faithful LLM Grading</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Reliable rubric grading requires more than accurate score prediction. Each judgement must be grounded in the mark scheme and evidence from the student answer. Existing credit-assignment and intervention methods, primarily designed for self-contained reasoning tasks such as mathematics reasoning, struggle in this setting because they do not identify where grading reasoning goes wrong or how the model's belief about the final mark changes during reasoning. We propose Evidence-Diagnosed Intervention Training (EDIT), a two-phase framework for training more rubric-faithful LLM graders. First, EDIT-SFT locates problematic reasoning steps using internal model signals: posterior belief over the final mark and input-grounding scores. It then revises only these local steps with help from a rubric checklist. Second, EDIT-RL calibrates the grader with belief-guided reward shaping, penalising large harmful belief drifts while still allowing helpful exploration. Experiments on two real-world, multi-subject grading benchmarks demonstrate that EDIT consistently outperforms strong supervised fine-tuning and reinforcement learning baselines on both in-domain and out-of-domain splits, with ablation studies confirming that internal-state diagnostics drive these gains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.05881v2">Confidence Before Answering: A Paradigm Shift for Efficient LLM Uncertainty Estimation</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Reliable deployment of large language models (LLMs) requires accurate uncertainty estimation. Existing methods are predominantly answer-first, producing confidence only after generating an answer, which measure the correctness of a specific response and limits practical usability. We study a confidence-first paradigm, where the model outputs its confidence before answering, interpreting this score as the model's probability of answering the question correctly under its current policy. We propose CoCA(Co-optimized Confidence and Answers), a GRPO reinforcement learning framework that jointly optimizes confidence calibration and answer accuracy via segmented credit assignment. By assigning separate rewards and group-relative advantages to confidence and answer segments, CoCA enables stable joint optimization and avoids reward hacking. Experiments across math, code, and factual QA benchmarks show improved calibration and uncertainty discrimination while preserving answer quality, thereby enabling a broader range of downstream applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30995v2">Traceable by Design: An LLM Pipeline and Dashboard for EU Regulatory Consultation Analysis</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 This research has been supported by funding from the ERC Starting Grant HUMANads (ERC-2021-StG No 101041824)
    </div>
    <details class="paper-abstract">
      Public consultations generate large volumes of data in the form of stakeholder submissions that are practically unfeasible to analyse manually. We present an end-to-end LLM-based pipeline and interactive dashboard for structured topic extraction from regulatory consultation submissions, demonstrated on the European Commission's Digital Fairness Act (DFA) public call for evidence as a case study. The system processes raw PDF attachments and web-form responses, extracts topic annotations, and grounds every extraction in a verbatim quote from the source text. Applied to 4,322 DFA submissions, the pipeline produced 15,368 topic annotations supported by 20,951 verbatim evidence quotes. Three principles govern the proposed design: verbatim grounding, full traceability, and transparency by design. The dashboard exposes the full extraction dataset through five analytical views, from dataset-level topic overviews to individual paragraph drill-downs, with every result traceable to its source. Beyond the predefined DFA topic categories, the pipeline generated certain stakeholder concerns, such as Age Verification, Payment Processor Censorship, and Digital Ownership, that a fixed-taxonomy approach would have missed. The pipeline is domain-generic; adapting it to a new consultation requires only a prompt update and a new dataset. A live demo is available at https://dfa-dashboard.thalesbertaglia.com/. The code and processed data are publicly available at https://github.com/thalesbertaglia/dfa-dashboard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04650v1">Improving the Efficiency and Effectiveness of LLM Knowledge Distillation for Conversational Search</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 SCAI Workshop at SIGIR '26}{July 20--24, 2026}{Melbourne, Naarm, Australia
    </div>
    <details class="paper-abstract">
      Conversational Search (CS) considers retrieval of relevant documents based on conversational context. Large Language Models (LLMs) have significantly enhanced CS by enabling effective query rewriting. However, employing LLMs during inference poses efficiency challenges. A method to balance effectiveness and efficiency is the use of knowledge distillation from LLM-based query rewriting. Recent work applies the Kullback-Leibler Divergence (KLD) for distillation, relaxing the alignment with the teacher signal compared to previous methods. Despite these gains, several aspects of KLD-based distillation for conversational search remain understudied, and we investigate them in this work. Prior work in related fields suggests that adding a contrastive loss to the KLD objective can improve performance; we confirm this and observe significant gains in precision-oriented ranking metrics. We also find that contrastive sampling strategies for the KLD loss have a non-trivial impact and must be chosen carefully. Although theory suggests that more samples improve the KLD estimate, experiments show diminishing returns on the number of used samples. Finally, we address the phenomenon of decreased sparsity in longer conversations, which limits computational efficiency across sparse retrieval methods. We find that the representations from the model distilled with the KLD loss can be strongly regularized with a regularization loss, substantially improving sparsity and inference efficiency without significantly harming retrieval effectiveness. We achieve a $2\times$ decrease in FLOPS on TopiOCQA with negligible loss in effectiveness, corresponding to a $\leq 2%$ drop in Recall@100. Our results provide insights into distillation objectives for learned sparse conversational retrievers and offer practical guidelines for improving effectiveness and efficiency in first-stage retrieval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05633v2">GIFT: Games as Informal Training for Generalizable LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Recent LLMs excel at formal tasks such as mathematical reasoning and code generation, but still struggle with broader abilities such as planning, creativity, and social intelligence. Inspired by human learning, where formal instruction and informal experience jointly shape intelligence, we introduce informal learning into LLM training and use games as annotation-free, feedback-driven environments. To cover diverse abilities including abstract reasoning, planning, creativity, and social interaction, we combine formal math tasks with three representative game tasks, including Matrix Games, TicTacToe, and Who's the Spy. However, directly mixing these tasks under a unified RL objective can blur task-specific learning signals and provides no explicit guidance for coordinating task-gradient directions. To combat these, we propose Coordinated Subtask Training (CST), which replaces a single mixed update with sequential subtask-specific updates, separating heterogeneous RL signals while implicitly promoting coordination among subtasks. Experiments on ability-oriented benchmarks show that game-based informal learning improves generalization beyond formal training alone, while CST further enhances multi-task RL by preserving in-domain subtask performance and improving broader general abilities. Code and data are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04632v1">VentAgent: When LLMs Learn to Breathe -- Multi-Objective Arbitration for ARDS Ventilation</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Mechanical ventilation for Acute Respiratory Distress Syndrome (ARDS) requires balancing competing physiological goals, including oxygenation, lung protection, and acid-base homeostasis. However, current data-driven methods, especially those imitating retrospective Electronic Health Records (EHR), often suffer from imitation bias. They may capture superficial correlations from inconsistent clinical demonstrations, such as associating passive ventilator settings with survival because such settings are common in stable patients, and thus fail to generalize to volatile or out-of-distribution phenotypes. Standard Reinforcement Learning (RL) methods also struggle with the adversarial trade-offs of critical care and often produce opaque policies with limited clinical interpretability. To address these limitations, we introduce VentAgent, a hierarchical framework in which Large Language Models (LLMs) act as transparent arbitrators for mechanical ventilation. We reformulate ventilation control as a dynamic Multi-Objective Arbitration process rather than single-objective optimization. VentAgent decomposes decision-making into three interpretable stages: Perception, Planning, and Orchestration. By leveraging the semantic reasoning capabilities of LLMs, it synthesizes strategies from heterogeneous experts and resolves conflicting clinical priorities through an explicit coordination mechanism. Evaluations on a high-fidelity physiological simulator show that VentAgent outperforms state-of-the-art RL and classical control baselines. Moreover, it converts control decisions into human-readable reasoning chains, offering a safer, more interpretable, and adaptable paradigm for critical care automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04613v2">Translation Heads: Disentangling meaning from language in LLM-based machine translation</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 61 pages, 70 figures
    </div>
    <details class="paper-abstract">
      Mechanistic Interpretability (MI) seeks to explain how neural networks implement their capabilities, but the scale of Large Language Models (LLMs) has limited prior MI work in Machine Translation (MT) to word-level analyses. We study sentence-level MT from a mechanistic perspective by analyzing attention heads to understand how LLMs internally encode and distribute translation functions. We decompose MT into two subtasks: producing text in the target language (i.e. target language identification) and preserving the input sentence's meaning (i.e. sentence equivalence). Across three families of open-source models and 20 translation directions, we find that distinct, sparse sets of attention heads specialize in each subtask. Based on this insight, we construct subtask-specific steering vectors and show that modifying just 1% of the relevant heads enables instruction-free MT performance comparable to instruction-based prompting, while ablating these heads selectively disrupts their corresponding translation functions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04594v1">Ekka: Automated Diagnosis of Silent Errors in LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 ICML 2026
    </div>
    <details class="paper-abstract">
      LLM serving frameworks are quickly evolving with a complex software stack and a vast number of optimizations. The rapid development process can introduce silent errors where output quality silently degrades without any explicit error signals. Diagnosing silent errors is notoriously difficult due to the substantial semantic gap between the high-level symptoms and the low-level root causes. We observe that diagnosis of silent errors can be effectively framed as a differential debugging problem by leveraging the existence of semantically correct reference implementations. We propose Ekka, an automated diagnosis system that identifies root causes by systematically aligning and comparing intermediate execution states between a target and a reference framework. We constructed a benchmark of real-world silent errors from popular serving frameworks, where Ekka shows 80% pass@1 diagnosis accuracy and 88% pass@5 diagnosis accuracy, outperforming state-of-the-art systems. Ekka also diagnoses 4 new silent errors from serving frameworks, all of which have been confirmed by the developers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04592v1">Synthetic Personalities: How Well Can LLMs Mimic Individual Respondents Using Socio-Economic Microdata?</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      LLM-based digital twins promise to scale and accelerate market research, but most published twins are either coarse persona bots conditioned on a few demographic questions or detailed individual-level twins built on purpose-collected surveys and interview transcripts. Neither setup speaks to the operationally most relevant case for marketing practice: building detailed individual twins from the pre-existing heterogeneous panel data that firms already accumulate through CRM systems, loyalty programs, and repeat surveys. We construct detailed individual-level twins from the German Socio-Economic Panel (SOEP) and evaluate them across a $3 \times 5 \times 2 \times 2$ construction-method grid that covers three open-weights LLMs, five cumulative information depths ranked by normalized Shannon entropy, two embedding methods, and two reasoning modes, scoring over 2.1 million twin responses on 500 participants and 183 held-out questions. Twin quality rises with information depth but with diminishing returns past the 75 percent entropy quartile, which acts as a cost-efficient Pareto point relative to the best-performing 100 percent cells. Switching the embedding from a narrative persona summary to a raw dialog history of past responses raises hold-out accuracy in every model-by-reasoning cell at the 100 percent depth, while an explicit thinking mode raises rank-order correlation without moving accuracy. Best-cell accuracy reaches 78.8 percent and Fisher-$z$ correlation reaches $r = 0.590$ on the SOEP held-out evaluation set. The findings suggest that twin-based market research is no longer gated by data design, but by item volume, model selection, and a small set of construction-level decisions that this paper now maps.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2407.03884v4">ChatSOP: An SOP-Guided MCTS Planning Framework for Controllable LLM Dialogue Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 Accepted to ACL 2025 main
    </div>
    <details class="paper-abstract">
      Dialogue agents powered by Large Language Models (LLMs) show superior performance in various tasks. Despite the better user understanding and human-like responses, their **lack of controllability** remains a key challenge, often leading to unfocused conversations or task failure. To address this, we introduce Standard Operating Procedure (SOP) to regulate dialogue flow. Specifically, we propose **ChatSOP**, a novel SOP-guided Monte Carlo Tree Search (MCTS) planning framework designed to enhance the controllability of LLM-driven dialogue agents. To enable this, we curate a dataset comprising SOP-annotated multi-scenario dialogues, generated using a semi-automated role-playing system with GPT-4o and validated through strict manual quality control. Additionally, we propose a novel method that integrates Chain of Thought reasoning with supervised fine-tuning for SOP prediction and utilizes SOP-guided Monte Carlo Tree Search for optimal action planning during dialogues. Experimental results demonstrate the effectiveness of our method, such as achieving a 27.95% improvement in action accuracy compared to baseline models based on GPT-3.5 and also showing notable gains for open-source models. Dataset and codes are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04547v1">Beyond Retrieval: Learning Compact User Representations for Scalable LLM Personalization</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 16 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Personalizing large language models requires adapting model behavior to individual users while preserving robustness and deployment-scale efficiency. Existing approaches typically personalize LLMs either at the input level, by retrieving user histories or constructing profile prompts, or at the parameter level, by maintaining user-specific parameter-efficient modules. The former makes personalization sensitive to retrieval quality and prompt design, whereas the latter incurs storage and maintenance costs that grow with the user population. To address these limitations, we propose TAP-PER (Temporal Attentive Prefix for PERsonalization), a prefix-based framework that encodes user preferences as learnable representations, eliminating explicit prompt construction and replacing heavy per-user adapters with lightweight user-state prefix embeddings. Inspired by personalized recommendation systems, TAP-PER decomposes user modeling into user-state and query-conditioned components, and incorporates temporal signals to capture the evolving nature of user interests. Experiments on six LaMP tasks show that TAP-PER consistently outperforms prompt-based and model-based baselines across classification, rating, and generation settings. Moreover, TAP-PER uses 130x fewer per-user parameters than OPPU and roughly half the total parameter footprint of PER-PCS at the 1,000-user scale, demonstrating that scalable LLM personalization can be achieved without explicit prompt construction or heavy per-user adapters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04514v1">SAILRec: Steering LLM Attention to Dual-Side Semantically Aligned Collaborative Embeddings for Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 17 pages, including appendices
    </div>
    <details class="paper-abstract">
      Recent LLM-based recommenders enhance language models with collaborative embeddings from user-item interactions, but making such embeddings available does not ensure their proper use during inference. Through a diagnostic attention analysis, we find that the utilization of collaborative embeddings is depth-dependent and alignment-sensitive, suggesting that LLMs need to balance their internal semantic knowledge with external collaborative knowledge. To address this issue, we propose SAILRec, an LLM-based recommender that improves this balance through dual-side semantic alignment and hierarchical attention steering. The former aligns item-side embeddings with item-text semantics and user-side embeddings with codebook-based semantic profiles, while the latter suppresses premature shallow-layer collaborative interference and strengthens collaborative evidence in deeper decision layers. Experiments on MovieLens-1M and Amazon-Book show that SAILRec consistently outperforms representative baselines, with ablation and masking analyses validating its key designs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04511v1">SparDA: Sparse Decoupled Attention for Efficient Long-Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Sparse attention reduces compute and memory bandwidth for long-context LLM inference. However, two key challenges remain: (1) KV cache capacity still grows with sequence length, and offloading to CPU memory introduces a PCIe transfer bottleneck; (2) the sparse selection step itself retains $O(T^2)$ complexity and can dominate attention cost at long contexts. We propose SparDA, a decoupled sparse attention architecture that introduces a fourth per-layer projection, the Forecast, alongside Query, Key, and Value. The Forecast predicts the KV blocks needed by the next layer, enabling lookahead selection that overlaps CPU-to-GPU prefetch with current-layer execution. Because Forecast is decoupled from the attention query, our GQA implementation uses one Forecast head per GQA group, reducing selection overhead versus the original multi-head selector. SparDA adds $<$0.5% parameters and trains only the Forecast projections by matching the original selector's attention distribution. On two sparse-pretrained 8B models, SparDA matches or slightly improves accuracy and delivers up to 1.25$\times$ prefill speedup and 1.7$\times$ decode speedup over the sparse-attention offload baseline. By enabling larger feasible batch sizes on a single GPU, SparDA further reaches up to 5.3$\times$ higher decode throughput than the non-offload sparse baseline. Our source code is available at https://github.com/NVlabs/SparDA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04505v1">Simulate, Reason, Decide: Scientific Reasoning with LLMs for Simulation-Driven Decision Making</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Scientific simulators are increasingly being integrated into LLM-driven systems for high-stakes simulation-driven decision-making. However, existing frameworks primarily use LLMs to generate, calibrate, or execute simulators, treating them as black-box interfaces rather than as structured mechanistic systems that can be reasoned about. As a result, current approaches lack the ability to identify, represent, and reason about the assumptions and mechanisms underlying simulator behavior, limiting transparency, auditability, and decision justification. We introduce MechSim, a mechanism-grounded neuro-symbolic reasoning framework for executable scientific simulators. Unlike prior neuro-symbolic approaches that primarily reason over static symbolic structures, MechSim enables LLM agents to reason about the mechanisms, assumptions, and execution behavior of scientific simulators. Our framework represents simulators through a shared structured schema capturing assumptions, variables, mechanism dependencies, and execution traces. On top of this representation, LLM agents operate as constrained reasoning engines that generate structured, evidence-grounded explanations linking simulator outcomes to their underlying mechanisms. We evaluate our approach across multiple high-stakes domains and show that it improves mechanism-level explanation quality, simulator analysis, and downstream decision-making reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03308v2">The Security Budget of Code LLMs: An Information-Theoretic Capacity-Security Bound</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      AI programming assistants make natural-language prompts a software-development interface, so small prompt perturbations become usability and security risks. We study an information-theoretic trade-off for code LLMs between functional capacity, $\Cap=\rmI(c^*;c_π)$, and perturbation retention, $\Sec=\rmI(c_π;\tilde c_π)$. Here $\Sec$ is a retention-channel quantity, not a direct measure of exploit success or vulnerable-code generation. For code completion modeled as $p\to c_π$ with perturbed prompt $\tilde p$, we prove $\Cap+\Sec\le \rmH(c^*)+\rmI(p;\tilde p)$, decomposing the budget into task entropy and prompt leakage. A deterministic-embedding corollary gives the hidden-state version, and a tokenizer/gzip companion bound gives a model-agnostic ceiling on sequence-level task entropy. Empirically, we estimate embedded $\Cap$ and $\Sec$ from output-only last-token hidden states, excluding prompt context from the $\Sec$ channel. Six individual validation rows across two models, two datasets, INT4/BF16 precision, and estimator ablations satisfy the embedded check $(\Cap+\max_T\Sec)/(\rmH(z^*)+\max_T\rmI(p;\tilde p))\le1$. Saturation is 0.27--0.92 and theorem slack is 2.36--26.94 nats; a separate three-seed stability diagnostic has mean saturation 0.87. A context-mixed cosine, used only as a per-problem generation-prompt alignment signal, correlates with pass@1 on CodeLlama-HumanEval ($ρ{=}0.36$, $p{<}10^{-4}$), Qwen-HumanEval ($ρ{=}0.22$, $p{=}0.005$), and CodeLlama-MBPP ($ρ{=}0.225$, $p{=}0.0038$; all $n{=}164$). Adaptive stress tests with a 23-perturbation pool, a fixed universal suffix, and prompt-embedding PGD all leave positive slack.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03303v2">LEAP: Supercharging LLMs for Formal Mathematics with Agentic Frameworks</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit strong informal mathematical reasoning but struggle to generate mechanically verifiable proofs in formal languages like Lean. We present LEAP, an agentic framework that enables general-purpose foundation models to achieve state-of-the-art performance on automated formal theorem proving. LEAP leverages foundation model capabilities, such as informal reasoning, instruction following, and iterative self-refinement. By decomposing complex problems into smaller units, the system bridges formal proof construction with informal blueprints through continuous interaction with the Lean compiler. To provide a rigorous evaluation beyond increasingly saturated benchmarks, we introduce Lean-IMO-Bench, a benchmark of IMO-style problems formalized in Lean, with short statements yet highly non-routine and multi-step proofs across a wide range of difficulty levels. Empirically, on the latest 2025 Putnam Competition, an annual mathematics competition for undergraduate students in North America, LEAP solves all 12 problems, matching recent breakthroughs by frontier formal mathematical models. On Lean-IMO-Bench, LEAP boosts the one-shot formal solve rate of general-purpose LLMs from below 10% to 70%, notably surpassing the 48% benchmark set by a specialized, gold-medal-caliber IMO system. Furthermore, we demonstrate LEAP's research-level utility by autonomously formalizing complex proofs for open combinatorial challenges, including a verified proof for a key subproblem in Knuth's Hamiltonian decomposition of even-order Cayley graphs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04483v1">Off-Distribution Voices: Fanfiction Subgenres as Universal Vernacular Jailbreaks for Aligned LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 23 pages
    </div>
    <details class="paper-abstract">
      Existing jailbreaks against aligned LLMs are discrete artifacts whose surface forms are easy to fingerprint and patch. We argue that the real failure mode is not any specific prompt, but an entire register of natural human writing that safety training has under-covered. Building on this insight, we introduce the first jailbreak family that uses real fanfiction subgenres as universal attack carriers: a creative-writing meta is conditioned on passages from one of twelve Archive of Our Own (AO3) subgenres, and the harmful behavior is embedded as the climax of the resulting scene. The construction requires no attacker LLM and no per-target adaptation. On eight aligned LLMs over the union of HarmBench and JailbreakBench, this attack lifts mean ASR from 0.278 to 0.731 under a four-judge ensemble; a factorial decomposition shows the gain is carried by register rather than length or structure. Two active defences widen rather than narrow the vernacular-to-baseline ratio, indicating that template-targeting defences merely steer attackers toward register-based attacks like ours. We also propose SAGA-A4, a static four-turn extension that attains mean ASR 0.924, substantially exceeding three existing multi-turn methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05238v1">DeployBench: Benchmarking LLM Agents for Research Artifact Deployment</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      LLM agents have made rapid progress on software engineering and ML research tasks, but these advances often assume access to a working runnable environment. For research artifacts released alongside published papers, setting up such an environment from a fresh machine remains a major bottleneck. Existing environment setup benchmarks do not cover the full scope of research artifact deployment, which involves multi-language toolchains, system-level dependencies beyond containers (e.g. GPU/CUDA and kernel configurations), and legacy artifact compatibility. We introduce DeployBench, a multi-domain benchmark of 51 research-artifact deployment tasks spanning AI/ML, computer systems, and scientific computing, covering all these dimensions. Each task is verified by a hidden pipeline that executes the paper's designated experiment and checks its outputs. Evaluating four state-of-the-art LLMs with OpenHands yields pass-rates from 7.8% - 51.0% . Failures are dominated by a completion-judgment problem: 97 of 154 are agent-terminated self-stops, where the agent's pre-finish checks validate a different or weaker target than the paper-specific task requires. DeployBench highlights the gap between current agents and autonomous deployment, and offers a realistic testbed for scientific research agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04454v1">Stepwise Reasoning Enhancement for LLMs via External Subgraph Generation</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Large language models have shown strong performance in natural language generation and downstream reasoning tasks, but they still struggle with logical consistency, factual grounding, and interpretability in complex multi-step reasoning. To address these limitations, this paper proposes SGR, a stepwise reasoning enhancement framework that integrates large language models with external knowledge graphs through query-relevant subgraph generation. Given an input question, SGR first extracts key entities, relations, and constraints to construct a structured schema, then retrieves compact subgraphs from a knowledge graph using schema-guided querying. The generated subgraphs provide explicit relational evidence that guides the language model through step-by-step reasoning. In addition, SGR combines direct Cypher-based reasoning with collaborative reasoning integration, allowing candidate answers from multiple reasoning paths to be validated and aggregated according to both model confidence and graph consistency. Experiments on benchmark datasets including CWQ, WebQSP, GrailQA, and KQA Pro demonstrate that SGR improves reasoning accuracy and Hits@1 performance over standard prompting and several knowledge-enhanced baselines. Ablation studies further show that schema guidance and Neo4j-based retrieval are both crucial to the effectiveness of the framework. These results indicate that dynamically generated external subgraphs can improve the accuracy, robustness, and interpretability of LLM-based reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04450v1">Listening to the Workforce: Measuring Construction Worker Safety Attitudes from Social Media Discourse Using LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Worker safety attitudes are key determinants of whether protective practices are applied or bypassed on construction sites. Yet measuring them at scale has remained out of reach. Safety attitudes are multidimensional, vary across topics, and surface most candidly in workers' own conversations. This study created and validated the Construction Safety Attitude Framework (CSAF), which integrates two components: a theory-grounded structure that characterizes safety attitudes along eight dimensions, and an operational codebook for measuring them in worker naturalistic discourse. Applying CSAF to 250 posts and comments from the r/Construction community on Reddit, trained coders reached strong agreement (Krippendorff's α = 0.85). Pairwise lift and conditional probability confirmed that the eight dimensions are related yet distinct. To apply the framework across large volumes of discourse, CSAF was operationalized through a large language model (LLM) classifier. On 450 r/Construction contributions, the classifier reproduced expert human coding (Cohen's \k{appa} = 0.90, precision = 0.98, recall = 0.98), and on 400 contributions from r/Roofing it retained that accuracy after transfer to a different trade community (\k{appa} = 0.89, precision = 0.98, recall = 0.97). A proof-of-value case study then applied the validated classifier to 10,346 contributions from r/Roofing, demonstrating that CSAF can distinguish multidimensional attitudes by safety topic, track how they shift over time, and trace the reasoning behind unfavorable ones. The study therefore provides a theoretically grounded, empirically vetted instrument for examining safety attitudes, offering a basis for targeted interventions that address the attitudes underlying unsafe practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.11166v3">SoLoPO: Unlocking Long-Context Capabilities in LLMs via Short-to-Long Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 Published as a conference paper at ICLR 2026
    </div>
    <details class="paper-abstract">
      Despite advances in pretraining with extended context sizes, large language models (LLMs) still face challenges in effectively utilizing real-world long-context information, primarily due to insufficient long-context alignment caused by data quality issues, training inefficiencies, and the lack of well-designed optimization objectives. To address these limitations, we propose a framework named \textbf{S}h\textbf{o}rt-to-\textbf{Lo}ng \textbf{P}reference \textbf{O}ptimization (\textbf{SoLoPO}), decoupling long-context preference optimization (PO) into two components: short-context PO and short-to-long reward alignment (SoLo-RA), supported by both theoretical and empirical evidence. Specifically, short-context PO leverages preference pairs sampled from short contexts to enhance the model's contextual knowledge utilization ability. Meanwhile, SoLo-RA explicitly encourages reward score consistency for the responses when conditioned on both short and long contexts that contain identical task-relevant information. This facilitates transferring the model's ability to handle short contexts into long-context scenarios. SoLoPO is compatible with mainstream preference optimization algorithms, while substantially improving the efficiency of data construction and training processes. Experimental results show that SoLoPO enhances all these algorithms with respect to stronger length and domain generalization abilities across various long-context benchmarks, while achieving notable improvements in both computational and memory efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.05722v3">OckBench: Measuring the Efficiency of LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) such as GPT-5 and Gemini 3 have pushed the frontier of automated reasoning and code generation. Yet current benchmarks emphasize accuracy and output quality, neglecting a critical dimension: efficiency of token usage. The token efficiency is highly variable in practical. Models solving the same problem with similar accuracy can exhibit up to a \textbf{5.0$\times$} difference in token length, leading to massive gap of model reasoning ability. Such variance exposes significant redundancy, highlighting the critical need for a standardized benchmark to quantify the gap of token efficiency. Thus, we introduce OckBench, the first benchmark that jointly measures accuracy and token efficiency across reasoning and coding tasks. Our evaluation reveals that token efficiency remains largely unoptimized across current models, significantly inflating serving costs and latency. These findings provide a concrete roadmap for the community to optimize the latent reasoning ability, token efficiency. Ultimately, we argue for an evaluation paradigm shift: tokens must not be multiplied beyond necessity. Our benchmarks are available at https://ockbench.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.19225v3">FinTradeBench: A Financial Reasoning Benchmark for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 9 pages main text, 32 pages total (including references and appendix). 5 figures, 16 tables. Preprint under review. Code and data will be made available upon publication
    </div>
    <details class="paper-abstract">
      Real-world financial decision-making is a challenging problem that requires reasoning over heterogeneous signals, including company fundamentals derived from regulatory filings and trading signals computed from price dynamics. Recently, with advances in Large Language Models (LLMs), financial analysts have begun to use them for financial decision-making tasks. However, existing financial question-answering benchmarks for testing these models primarily focus on company balance sheet data and rarely evaluate reasoning about how company stocks trade in the market or their interactions with fundamentals. To leverage the strengths of both approaches, we introduce FinTradeBench, a benchmark for evaluating financial reasoning that integrates company fundamentals and trading signals. FinTradeBench contains 1,400 questions grounded in NASDAQ-100 companies over a ten-year historical window. The benchmark is organized into three reasoning categories: fundamentals-focused, trading-signal-focused, and hybrid questions requiring cross-signal reasoning. To ensure reliability at scale, we adopt a calibration-then-scaling framework that combines expert seed questions, multi-model response generation, intra-model self-filtering, numerical auditing, and human-LLM judge alignment. We evaluate 14 LLMs under zero-shot prompting and retrieval-augmented settings and witness a clear performance gap. Retrieval substantially improves reasoning over textual fundamentals, but provides limited benefit for trading-signal reasoning. These findings highlight fundamental challenges in the numerical and time-series reasoning for current LLMs and motivate future research in financial intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.01143v3">Schedule-Level Shared-Prefix Reuse for LLM RL Training</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      GRPO-based LLM post-training commonly samples multiple trajectories from the same prompt and then trains on the resulting group. In long-context GRPO workloads, this shared prompt-side prefix can contain retrieved passages, visual tokens, tool schemas, system instructions, or task context, while the full rollout group is still too large to pack into one training microbatch. Standard dense trainers therefore recompute the same prefix forward and backward for every trajectory. We present a schedule-level reuse mechanism that decouples prefix and suffix computation. The schedule runs prefix forward once, executes suffixes as ordinary microbatches while reading prefix K/V and accumulating prefix-side gK/gV , and then runs prefix backward once on the accumulated gradient cache. This reordered schedule is equivalent to baseline training over real arithmetic and aligns numerically within finite-precision tolerance. Because only K/V and gK/gV are hot during suffix computation, the approach offloads dormant prefix activations, integrates with TP/EP/CP/PP and DP-style placement at the execution level, and preserves aux-loss-based MoE router semantics through logical prefix-token accounting. On dense Llama3-8B, Qwen3-8B, and MoE Qwen3-MoE-30B-A3B configurations, the schedule matches optimizer updates across TP/CP/PP/EP combinations, aligns on a 100-step real GRPO actor-update trace replay, reaches up to 4.395x speedup (2.930x under a conservative compile-on comparison) as prefix ratio and GRPO group size grow, and reduces Phase-B peak HBM by up to 59.1%, extending the Llama3-8B capacity frontier from 17,920 to 29,696 total tokens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.17281v7">MemoryBench: A Benchmark for Memory and Continual Learning in LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Scaling up data, parameters, and test-time computation has been the mainstream methods to improve LLM systems (LLMsys), but their upper bounds are almost reached due to the gradual depletion of high-quality data and marginal gains obtained from larger computational resource consumption. Inspired by the abilities of human and traditional AI systems in learning from practice, constructing memory and continual learning frameworks for LLMsys has become an important and popular research direction in recent literature. Yet, existing benchmarks for LLM memory often focus on evaluating the system on homogeneous reading comprehension tasks with long-form inputs rather than testing their abilities to learn from accumulated user feedback in service time. Therefore, we propose a user feedback simulation framework and a comprehensive benchmark covering multiple domains, languages, and types of tasks to evaluate the continual learning abilities of LLMsys. Experiments show that the effectiveness and efficiency of state-of-the-art baselines are far from satisfying, and we hope this benchmark could pave the way for future studies on LLM memory and optimization algorithms. Website: https://memorybench.thuir.cn Code: https://github.com/THUIR/MemoryBench Data: https://huggingface.co/datasets/THUIR/MemoryBench Data-Full: https://huggingface.co/datasets/THUIR/MemoryBench-Full
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.23694v6">SafeSearch: Automated Red-Teaming of LLM-Based Search Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 Accepted by ICML 2026
    </div>
    <details class="paper-abstract">
      Search agents connect LLMs to the Internet, enabling them to access broader and more up-to-date information. However, this also introduces a new threat surface: unreliable search results can mislead agents into producing unsafe outputs. Real-world incidents and our two in-the-wild observations show that such failures can occur in practice. To study this threat systematically, we propose SafeSearch, an automated red-teaming framework that is scalable, cost-efficient, and lightweight, enabling sandboxed safety evaluation of search agents. Using this, we generate 300 test cases spanning five risk categories (e.g., misinformation and prompt injection) and evaluate three search agent scaffolds across 17 representative LLMs. Our results reveal substantial vulnerabilities in LLM-based search agents, with the highest ASR reaching 90.5% for GPT-4.1-mini in a search-workflow setting. Moreover, we find that common defenses, such as reminder prompting, offer limited protection. Overall, SafeSearch provides a practical way to measure and improve the safety of LLM-based search agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04415v1">FlexNPU: Transparent NPU Virtualization for Dynamic LLM Prefill-Decode Co-location</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Modern AI serving increasingly relies on NPUs for conventional inference and large language model serving. However, current NPU deployments commonly expose physical devices directly to applications, which limits runtime control over scheduling and makes it difficult to adapt execution to phase-level workload behavior. This limitation is particularly evident in LLM serving, where the prefill phase is compute-intensive while the decode phase is often constrained by memory bandwidth and KV-cache accesses. Static prefill-decode (PD) disaggregation reduces phase interference, but can introduce resource imbalance and unnecessary data movement. We present FlexNPU, a transparent user-space virtualization layer for Ascend NPUs. FlexNPU interposes on AscendCL APIs and routes NPU operations through per-device daemons, decoupling unmodified from physical NPU devices without modifying model code, AI frameworks, or NPU drivers. This runtime boundary allows FlexNPU to virtualize NPU objects, control operator dispatch, and support phase-aware scheduling for LLM serving. In particular, FlexNPU enables dynamic PD co-location, which adapts scheduling between prefill and decode according to their complementary resource characteristics. We implement FlexNPU on Huawei Ascend NPUs and evaluate it with typical LLM workloads. Compared with direct NPU passthrough, FlexNPU introduces no measurable inference overhead and slightly improves throughput in some scenarios. On a 384-card Ascend 910C deployment of DeepSeek-R1, FlexNPU improves throughput over static PD disaggregation by 5.15% and 26.33%. On Qwen2.5-7B, compared with static PD co-location, FlexNPU maintains comparable throughput while reducing TTFT by over 92% across tested workloads with nearly unchanged TPOT. These results show that transparent NPU virtualization is a practical substrate for efficient and responsive LLM serving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03318v2">Beyond Ideal Instruction: A Comprehensive Framework for Evaluating LLMs in Realistic Interactions</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Despite great advances in tool-use capabilities of large language models (LLMs), existing evaluation benchmarks struggle to fully align with real-world scenarios. Such benchmarks mostly rely on simulated idealized user assumptions and lacks experience-oriented evaluation. These limitations fail to account for the ambiguity, uncooperative behaviors, and shifting intentions characteristic of real-world users. To fill this gap, we propose RUT-Bench, a dedicated benchmark designed to assess LLMs under diverse Real-world User Tool calling scenarios. RUT-Bench supports high-fidelity simulations covering both ideal rational patterns and heterogeneous non-ideal behaviors across single-turn and multi-turn dialogues. We conduct comprehensive evaluations on 19 widely adopted open-source and proprietary LLMs using our benchmark. Experimental results reveal that no tested LLMs achieve an overall success rate above 40%, and nearly all of them experience noticeable performance drops when facing more complicated non-ideal user inputs. Our code and data is available at https://github.com/Miaow-Lab/RUT-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04387v1">Rethinking Sales Lead Scoring with LLM-based Hierarchical Preference Ranking</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Sales lead conversion in high-stakes domains (e.g., automotive, real estate) differs fundamentally from e-commerce recommendation due to prolonged decision cycles and multi-stage funnels. Traditional lead scoring methods rule-based scorecards, machine learning, or pointwise CTR models face severe challenges: sparse supervision, a semantic gap in unstructured CRM logs, and inability to capture relative lead priority. While Large Language Models(LLMs) offer superior semantic understanding of customer interactions, general-purpose LLMs are ill-suited for lead ranking: they generate text rather than comparable scores, and lack alignment with the hierarchical priorities of sales funnels. We introduce an LLM-based discriminative framework for sales lead scoring, which supports joint modeling of structured CRM features and unstructured customer interactions. On top of this framework, we propose HPRO (Hierarchical Preference Ranking Optimization), which augments sales lead scoring with a hierarchical preference ranking objective. HPRO employs a margin-aware Bradley-Terry formulation to transform sparse binary labels into dense, funnel-aware preference pairs, enabling lead scoring to leverage both pointwise and pairwise supervision. Experiments on large-scale data from a leading NEV brand demonstrate state-of-the-art classification (AUC 0.8161) and ranking performance (+39.7% precision among top-ranked leads). A 132-day online A/B test validates 9.5% sales volume uplift, confirming real-world commercial impact.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09853v3">MedRedFlag: Investigating how LLMs Redirect Misconceptions in Real-World Health Communication</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Real-world health questions from patients often unintentionally embed false assumptions or premises. In such cases, safe medical communication typically involves redirection: addressing the implicit misconception and then responding to the underlying patient context, rather than the original question. While large language models (LLMs) are increasingly being used by lay users for medical advice, they have not yet been tested for this crucial competency. Therefore, in this work, we investigate how LLMs react to false premises embedded within real-world health questions. We develop a semi-automated pipeline to curate MedRedFlag, a dataset of 1100+ questions sourced from Reddit that require redirection. We then systematically compare responses from state-of-the-art LLMs to those from clinicians. Our analysis reveals that LLMs often fail to redirect problematic questions, even when the problematic premise is detected, and provide answers that could lead to suboptimal medical decision making. Our benchmark and results reveal a novel and substantial gap in how LLMs perform under the conditions of real-world health communication, highlighting critical safety concerns for patient-facing medical AI systems. Code and dataset are available at https://github.com/srsambara-1/MedRedFlag.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04378v1">DLLG: Dynamic Logit-Level Gating of LLM Experts</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Leveraging multiple specialized LLMs can combine complementary strengths, but existing approaches trade adaptability for stability: routing commits prematurely, heuristic ensembling depends on fragile proxies, and parameter merging introduces interference. We propose DLLG (Dynamic Logit-Level Gating), a dynamic logit-level ensembling framework that learns token-level expert fusion from sparse response-level supervision. A lightweight gating module predicts step-wise fusion weights, linking trajectory-level correctness to generation without token-level labels or expert retraining. Across diverse reasoning and code benchmarks, DLLG consistently outperforms strong routing, heuristic ensembling, and parameter-merging baselines across model scales, highlighting learned logit-level fusion as a robust and scalable paradigm for integrating specialized experts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.03353v4">SkCC: Portable and Secure Skill Compilation for Cross-Framework LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 Accepted by the Agent Skills Workshop at ACM CAIS 2026. 20 pages, 6 figures. Project Homepage: https://skcc.nexa-lang.com/ Code Repo: https://github.com/Nexa-Language/Skill-Compiler/
    </div>
    <details class="paper-abstract">
      LLM agents increasingly rely on reusable skills (e.g., SKILL markdown files) to execute complex tasks, yet these artifacts lack portability: agent frameworks are highly sensitive to prompt formatting, leading to a large performance variation for the same skill. Nevertheless, most skills are authored once as format-agnostic Markdown, necessitating costly per-framework rewrites and also leaving security largely unaddressed, with widespread vulnerabilities in practice. To address this, we present SkCC, a compiler for LLM agents that introduces classical compilation design into agent skill development. SkCC centers on SkIR, a strongly-typed intermediate representation that decouples skill semantics from framework-specific formatting, thus enabling portable deployment across agent frameworks. Atop of this IR, a static Optimizer enforces security constraints, blocking vulnerabilities before deployment. Implemented as a four-phase pipeline, SkCC effectively reduces adaptation complexity from $O(m \times n)$ to $O(m + n)$ across $m$ skills and $n$ frameworks. Experiments on SkillsBench demonstrate that SkCC delivers consistent and substantial gains over original counterparts, with pass rate increases from 21.1% to 33.3% on Claude Code and from 35.1% to 48.7% on Kimi CLI. Further, the design achieves sub-10ms compilation latency, 94.8% proactive security trigger rate, and 10-46% runtime token savings across frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04360v1">Deliberate Evolution: Agentic Reasoning for Sample-Efficient Symbolic Regression with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 ICML 2026
    </div>
    <details class="paper-abstract">
      Symbolic regression (SR) discovers compact mathematical expressions from data, yet recent LLM-based evolutionary methods remain sample-inefficient because they rely mainly on scalar feedback such as MSE. We identify a core limitation: existing methods conflate candidate proposal with search guidance, requiring the LLM to infer how to evolve an expression, diagnose its errors, and reuse past experience from a single score. To address this, we propose Deliberate Evolution (DE), an agentic framework that decouples symbolic generation from search control. DE guides LLM proposals with adaptive operators for search direction, analytical tools for structural diagnosis, and reflective memory for trajectory-level experience. Experiments on LLM-SRBench show that DE consistently outperforms representative LLM-based SR baselines across diverse scientific domains while using only 40% of the standard sample budget.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.15118v2">Talk is (Not) Cheap: A Taxonomy and Benchmark Coverage Audit for LLM Attacks</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      We introduce a reusable framework for auditing whether LLM attack benchmarks collectively cover the threat surface: a 4$\times$6 Target $\times$ Technique matrix grounded in STRIDE, constructed from a 507-leaf taxonomy -- 401 data-populated and 106 threat-model-derived leaves -- of inference-time attacks extracted from 932 arXiv security studies (2023--2026). The matrix enables benchmark-external validation -- auditing collective coverage rather than individual benchmark consistency. Applying it to six public benchmarks reveals that the three primary frameworks (HarmBench, InjecAgent, AgentDojo) occupy non-overlapping cells covering at most 25\% of the matrix, while entire STRIDE threat categories (Service Disruption, Model Internals) lack any standardized evaluation, despite published attacks in these categories achieving 46$\times$ token amplification and 96\% attack success rates through mechanisms which no benchmark tests. The corpus of 2,521 unique attack groups further reveals pervasive naming fragmentation (up to 29 surface forms for a single attack) and heavy concentration in Safety \& Alignment Bypass, structural properties invisible at smaller scale. The taxonomy, attack records, and coverage mappings are released as extensible artifacts; as new benchmarks emerge, they can be mapped onto the same matrix, enabling the community to track whether evaluation gaps are closing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04329v1">From Untrusted Input to Trusted Memory: A Systematic Study of Memory Poisoning Attacks in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Memory is a core component of AI agents, enabling them to accumulate knowledge across interactions and improve performance. However, persistent memory introduces the risk of memory poisoning, where a single adversarial memory write can exert long-term influence over agent behavior. We present a systematic study of memory poisoning in LLM-based agents. We identify four memory write channels and nine structural vulnerabilities in model capabilities, system prompt design, and agent system architecture that make these channels exploitable. Based on these vulnerabilities, we develop a taxonomy of six classes of memory poisoning attacks. Furthermore, we design MPBench -- a benchmark for evaluating memory poisoning attacks, and show that agents designed to write and retrieve memory more aggressively are more exploitable. We also show that existing prompt injection defenses fail to cover memory poisoning attacks. Our findings provide a foundation for understanding and mitigating memory poisoning attacks against AI agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04306v1">Organizational Control Layer: Governance Infrastructure at the Execution Boundary of LLM Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 13 pages, 2 figures
    </div>
    <details class="paper-abstract">
      LLM-based agents are increasingly deployed in workflows where generated outputs may directly trigger state-changing actions. This creates an execution-boundary problem: proposed actions must be governed before they are executed. We study this problem through economically consequential multi-agent interactions and argue that deployment-grade agent systems should separate proposal generation from environment-facing execution. To operationalize this principle, we introduce the Organizational Control Layer (OCL), a model-agnostic governance infrastructure that intercepts generated actions before execution through policy enforcement and escalation, without modifying the underlying LLM generator. We evaluate OCL on adversarial buyer--seller negotiation environments adapted from AgenticPay. Across multiple frontier LLM backends, OCL reduces unsafe executions from 88% to near-zero while increasing valid success from 12% to 96%. Results further reveal a safety--utility tradeoff: strict governance improves compliance and reliability against policy and constraint violations, but can reduce flexibility in tightly constrained markets. These findings suggest that deployment-grade LLM agent systems require explicit governance at the boundary between language generation and executable actions. The source code is available at: https://github.com/SHITIANYU-hue/amai_ocl
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04300v1">Argus-Retriever: Vision-LLM Late-Interaction Retrieval with Region-Aware Query-Conditioned MoE for Visual Document Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Late-interaction vision-language retrievers represent each document page as many visual token embeddings and score queries with MaxSim. In systems such as ColPali, ColQwen, ColNomic, and Nemotron ColEmbed, the document embeddings are produced without seeing the query, so the same page is represented identically for a table lookup, a chart question, and a layout-sensitive evidence request. We introduce \textbf{Argus}, a family of query-conditioned late-interaction retrievers built on Qwen3.5-VL. Argus adds a region-aware Mixture-of-Experts module: the query encoder produces both retrieval embeddings and a compact context vector, the document page is pooled into spatial regions, and a query-aware router selects latent experts per region before MaxSim. The output remains a multi-vector index compatible with ColPali-style retrieval, but the document representation is now dependent on the query (i.e., $\mathbf{D}(q)$). All Argus models use a 1024-dimensional retrieval head, compared with the 2560-dimensional and 4096-dimensional heads of recent state-of-the-art systems, and are trained on roughly 9\% of the available public supervision rather than the full pool. The 9B model reaches \textbf{92.67} NDCG@5 on ViDoRe V1 and \textbf{86.0} NDCG@5 on the combined V1+V2 leaderboard, the highest reported value for an open late-interaction model on the combined leaderboard. Wrapped in a Qwen3.6-27B agentic retrieval pipeline on ViDoRe V3, Argus-9B further improves its NDCG@10 from 60.28 to \textbf{64.80} over public tasks, showing that the same retriever serves both as a strong standalone system and as a search primitive for iterative LLM agents.
    </details>
</div>
