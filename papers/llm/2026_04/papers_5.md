# llm - 2026_04

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
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.23121v2">Uncovering the Fragility of Trustworthy LLMs through Chinese Textual Ambiguity</a></div>
    <div class="paper-meta">
      📅 2026-04-16
      | 💬 Accepted at KDD workshop on Evaluation and Trustworthiness of Agentic and Generative AI Models (Agentic & GenAI Evaluation Workshop KDD '25)
    </div>
    <details class="paper-abstract">
      In this work, we study a critical research problem regarding the trustworthiness of large language models (LLMs): how LLMs behave when encountering ambiguous narrative text, with a particular focus on Chinese textual ambiguity. We created a benchmark dataset by collecting and generating ambiguous sentences with context and their corresponding disambiguated pairs, representing multiple possible interpretations. These annotated examples are systematically categorized into 3 main categories and 9 subcategories. Through experiments, we discovered significant fragility in LLMs when handling ambiguity, revealing behavior that differs substantially from humans. Specifically, LLMs cannot reliably distinguish ambiguous text from unambiguous text, show overconfidence in interpreting ambiguous text as having a single meaning rather than multiple meanings, and exhibit overthinking when attempting to understand the various possible meanings. Our findings highlight a fundamental limitation in current LLMs that has significant implications for their deployment in real-world applications where linguistic ambiguity is common, calling for improved approaches to handle uncertainty in language understanding. The dataset and code are publicly available at this GitHub repository: https://github.com/ictup/LLM-Chinese-Textual-Disambiguation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20868v2">Rethinking LLM-Driven Heuristic Design: Generating Efficient and Specialized Solvers via Dynamics-Aware Optimization</a></div>
    <div class="paper-meta">
      📅 2026-04-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have advanced the field of Combinatorial Optimization through automated heuristic generation. Instead of relying on manual design, this LLM-Driven Heuristic Design (LHD) process leverages LLMs to iteratively generate and refine solvers to achieve high performance. However, existing LHD frameworks face two critical limitations: (1) Endpoint-only evaluation, which ranks solvers solely by final gap to a reference solution, ignoring the convergence process and runtime efficiency; (2) High adaptation costs, where distribution shifts necessitate re-adaptation to generate specialized solvers for heterogeneous instance groups. To address these issues, we propose Dynamics-Aware Solver Heuristics (DASH), a framework that co-optimizes solver search mechanisms and runtime schedules guided by a convergence-aware metric, thereby identifying efficient and high-performance solvers. Furthermore, to mitigate expensive re-adaptation, DASH incorporates Profiled Library Retrieval (PLR), which maintains group-specialized solvers for profile-aware warm starts. These solvers are archived concurrently during evolution, allowing DASH to reuse matched specialists across heterogeneous distributions without restarting adaptation. Experiments on four combinatorial optimization problems demonstrate that DASH improves runtime efficiency by over 4 times while outperforming prior LHD baselines in the overall balance between gap and runtime across diverse problem scales. Furthermore, by enabling profile-aware warm starts, DASH maintains lower gap under distribution shift while reducing LLM adaptation costs by about 90%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07667v2">Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-04-16
      | 💬 ACL 2026 Findings. Source code available at https://github.com/TANIGUCHIREI/ASL
    </div>
    <details class="paper-abstract">
      Due to the prevalence of large language models (LLMs), key-value (KV) cache reduction for LLM inference has received remarkable attention. Among numerous works that have been proposed in recent years, layer-wise token pruning approaches, which select a subset of tokens at particular layers to retain in KV cache and prune others, are one of the most popular schemes. They primarily adopt a set of pre-defined layers, at which tokens are selected. Such design is inflexible in the sense that the accuracy significantly varies across tasks and deteriorates in harder tasks such as KV retrieval. In this paper, we propose ASL, a training-free method that adaptively chooses the selection layer for KV cache reduction, exploiting the variance of token ranks ordered by attention score. The proposed method balances the performance across different tasks while meeting the user-specified KV budget requirement. ASL operates during the prefilling stage and can be jointly used with existing KV cache reduction methods such as SnapKV to optimize the decoding stage. By evaluations on the InfiniteBench, RULER, and NIAH benchmarks, we show that ASL, equipped with one-shot token selection, adaptively trades inference speed for accuracy, outperforming state-of-the-art layer-wise token pruning methods in difficult tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.15186v1">Scepsy: Serving Agentic Workflows Using Aggregate LLM Pipelines</a></div>
    <div class="paper-meta">
      📅 2026-04-16
    </div>
    <details class="paper-abstract">
      Agentic workflows carry out complex tasks by orchestrating multiple large language models (LLMs) and tools. Serving such workflows at a target throughput with low latency is challenging because they can be defined using arbitrary agentic frameworks and exhibit unpredictable execution times: execution may branch, fan-out, or recur in data-dependent ways. Since LLMs in workflows often outnumber available GPUs, their execution also leads to GPU oversubscription. We describe Scepsy, a new agentic serving system that efficiently schedules arbitrary multi-LLM agentic workflows onto a GPU cluster. Scepsy exploits the insight that, while agentic workflows have unpredictable end-to-end latencies, the shares of each LLM's total execution times are comparatively stable across executions. Scepsy decides on GPU allocations based on these aggregate shares: first, it profiles the LLMs under different parallelism degrees. It then uses these statistics to construct an Aggregate LLM Pipeline, which is a lightweight latency/throughput predictor for allocations. To find a GPU allocation that minimizes latency while achieving a target throughput, Scepsy uses the Aggregate LLM Pipeline to explore a search space over fractional GPU shares, tensor parallelism degrees, and replica counts. It uses a hierarchical heuristic to place the best allocation onto the GPU cluster, minimizing fragmentation, while respecting network topology constraints. Our evaluation on realistic agentic workflows shows that Scepsy achieves up to 2.4x higher throughput and 27x lower latency compared to systems that optimize LLMs independently or rely on user-specified allocations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.23657v4">Secret Leak Detection in Software Issue Reports using LLMs: A Comprehensive Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-04-16
      | 💬 Accepted at the International Conference on Mining Software Repositories (MSR) 2026
    </div>
    <details class="paper-abstract">
      In the digital era, accidental exposure of sensitive information such as API keys, tokens, and credentials is a growing security threat. While most prior work focuses on detecting secrets in source code, leakage in software issue reports remains largely unexplored. This study fills that gap through a large-scale analysis and a practical detection pipeline for exposed secrets in GitHub issues. Our pipeline combines regular expression-based extraction with large language model (LLM)-based contextual classification to detect real secrets and reduce false positives. We build a benchmark of 54,148 instances from public GitHub issues, including 5,881 manually verified true secrets. Using this dataset, we evaluate entropy-based baselines and keyword heuristics used by prior secret detection tools, classical machine learning, deep learning, and LLM-based methods. Regex and entropy based approaches achieve high recall but poor precision, while smaller models such as RoBERTa and CodeBERT greatly improve performance (F1 = 92.70%). Proprietary models like GPT-4o perform moderately in few-shot settings (F1 = 80.13%), and fine-tuned open-source larger LLMs such as Qwen and LLaMA reach up to 94.49% F1. Finally, we also validate our approach on 178 real-world GitHub repositories, achieving an F1-score of 81.6% which demonstrates our approach's strong ability to generalize to in-the-wild scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.15149v1">LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking</a></div>
    <div class="paper-meta">
      📅 2026-04-16
    </div>
    <details class="paper-abstract">
      As reinforcement Learning with Verifiable Rewards (RLVR) has become the dominant paradigm for scaling reasoning capabilities in LLMs, a new failure mode emerges: LLMs gaming verifiers. We study this phenomenon on inductive reasoning tasks, where models must induce and output logical rules. We find that RLVR-trained models systematically abandon rule induction. Instead of learning generalizable patterns (e.g., ``trains carrying red cars go east''), they enumerate instance-level labels, producing outputs that pass verifiers without capturing the relational patterns required by the task. We show that this behavior is not a failure of understanding but a form of reward hacking: imperfect verifiers that check only extensional correctness admit false positives. To detect such shortcuts, we introduce Isomorphic Perturbation Testing (IPT), which evaluates a single model output under both extensional and isomorphic verification, where the latter enforces invariance under logically isomorphic tasks. While genuine rule induction remains invariant, shortcut strategies fail. We find that shortcut behavior is specific to RLVR-trained reasoning models (e.g., GPT-5, Olmo3) and absent in non-RLVR models (e.g., GPT-4o, GPT-4.5, Ministral). Moreover, shortcut prevalence increases with task complexity and inference-time compute. In controlled training experiments, extensional verification directly induces shortcut strategies, while isomorphic verification eliminates them. These results show that RLVR can incentivize reward hacking not only through overt manipulation but also by exploiting what the verifier fails to enforce.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.15140v1">DiscoTrace: Representing and Comparing Answering Strategies of Humans and LLMs in Information-Seeking Question Answering</a></div>
    <div class="paper-meta">
      📅 2026-04-16
    </div>
    <details class="paper-abstract">
      We introduce DiscoTrace, a method to identify the rhetorical strategies that answerers use when responding to information-seeking questions. DiscoTrace represents answers as a sequence of question-related discourse acts paired with interpretations of the original question, annotated on top of rhetorical structure theory parses. Applying DiscoTrace to answers from nine different human communities reveals that communities have diverse preferences for answer construction. In contrast, LLMs do not exhibit rhetorical diversity in their answers, even when prompted to mimic specific human community answering guidelines. LLMs also systematically opt for breadth, addressing interpretations of questions that human answerers choose not to address. Our findings can guide the development of pragmatic LLM answerers that consider a range of strategies informed by context in QA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.15136v1">Feedback-Driven Execution for LLM-Based Binary Analysis</a></div>
    <div class="paper-meta">
      📅 2026-04-16
      | 💬 17 pages
    </div>
    <details class="paper-abstract">
      Binary analysis increasingly relies on large language models (LLMs) to perform semantic reasoning over complex program behaviors. However, existing approaches largely adopt a one-pass execution paradigm, where reasoning operates over a fixed program representation constructed by static analysis tools. This formulation limits the ability to adapt exploration based on intermediate results and makes it difficult to sustain long-horizon, multi-path analysis under constrained context. We present FORGE, a system that rethinks LLM-based analysis as a feedback-driven execution process. FORGE interleaves reasoning and tool interaction through a reasoning-action-observation loop, enabling incremental exploration and evidence construction. To address the instability of long-horizon reasoning, we introduce a Dynamic Forest of Agents (FoA), a decomposed execution model that dynamically coordinates parallel exploration while bounding per-agent context. We evaluate FORGE on 3,457 real-world firmware binaries. FORGE identifies 1,274 vulnerabilities across 591 unique binaries, achieving 72.3% precision while covering a broader range of vulnerability types than prior approaches. These results demonstrate that structuring LLM-based analysis as a decomposed, feedback-driven execution system enables both scalable reasoning and high-quality outcomes in long-horizon tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13860v2">"AI Psychosis" in Context: How Conversation History Shapes LLM Responses to Delusional Beliefs</a></div>
    <div class="paper-meta">
      📅 2026-04-16
    </div>
    <details class="paper-abstract">
      Extended interaction with large language models (LLMs) has been linked to the reinforcement of delusional beliefs, a phenomenon attracting growing clinical and public concern. Yet most empirical work evaluates model safety in brief interactions, which may not reflect how these harms develop through sustained dialogue. We tested five models across three levels of accumulated context, using the same escalating delusional history to isolate its effect on model behaviour. Human raters coded responses on risk and safety dimensions, and each model was analysed qualitatively. Models separated into two distinct tiers: GPT-4o, Grok 4.1 Fast, and Gemini 3 Pro exhibited high-risk, low-safety profiles; Claude Opus 4.5 and GPT-5.2 Instant displayed the opposite pattern. As context accumulated, performance tended to degrade in the unsafe group, while the same material activated stronger safety interventions among the safer models. Qualitative analysis identified distinct mechanisms of failure, including validation of the user's delusional premises, elaboration beyond them, and attempting harm reduction from within the delusional frame. Safer models, however, often used the established relationship to support intervention, taking accountability for past missteps so that redirection would not be received as betrayal. These findings indicate that accumulated context functions as a stress test of safety architecture, revealing whether a model treats prior dialogue as a worldview to inherit or as evidence to evaluate. Short-context assessments may therefore mischaracterise model safety, underestimating danger in some systems while missing context-activated gains in others. The results suggest that delusional reinforcement by LLMs reflects a preventable alignment failure. In demonstrating that these harms can be resisted, the safer models establish a baseline future systems should now be expected to meet.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.15075v1">Atropos: Improving Cost-Benefit Trade-off of LLM-based Agents under Self-Consistency with Early Termination and Model Hotswap</a></div>
    <div class="paper-meta">
      📅 2026-04-16
      | 💬 Will appear at ISSTA 2026
    </div>
    <details class="paper-abstract">
      Open-weight Small Language Models(SLMs) can provide faster local inference at lower financial cost, but may not achieve the same performance level as commercial Large Language Models (LLMs) that are orders of magnitudes larger. Consequently, many of the latest applications of LLMs, such as software engineering agents, tend to be evaluated on larger models only, leaving the issue of improving the cost-benefit trade-off of such applications neglected. This paper proposes Atropos, a predictive early-termination analysis and hotswap technique that aims to improve the cost-benefit trade-off for LLM-based agents that use self-consistency. The core component of ATROPOS is a predictive model based on structural properties of LLM inferences: after merging multiple agentic inference paths into a graph representation, ATROPOS uses Graph Convolutional Network (GCN) to predict whether an ongoing inference will eventually succeed or not. If an agentic task instance running on the source LLM is predicted to fail, ATROPOS subsequently performs hotswapping, i.e., migrating the on-going inference context onto the more capable target LLM: this is feasible because LLM contexts are stateless. An empirical evaluation of ATROPOS using three recent LLM-based agents shows that ATROPOS can predict early termination of eventually failing inferences with the accuracy of 0.85 at the midpoint of the inference. Hotswapping LLMs for such inferences can convert up to 27.57% of them to be successful. Consequently, ATROPOS achieves 74.35% of the performance of closed LLMs with as low as only 23.9% of the cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.04116v4">Searching Meta Reasoning Skeleton to Guide LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-04-16
    </div>
    <details class="paper-abstract">
      Meta reasoning behaviors work as a skeleton to guide large language model (LLM) reasoning, thus help to improve reasoning performance. However, prior researches implement meta reasoning skeleton with manually designed structure, limiting ability to adapt to query-specific requirement and capture intricate logical dependency among reasoning steps. To deal with the challenges, we represent meta reasoning skeleton with directed acyclic graph (DAG) to unify skeletons proposed in prior works and model intricate logical dependency. Then we propose AutoMR, a framework that searches for query-aware meta reasoning skeleton automatically inspired by automated machine learning (AutoML). Specifically, we construct search space based on DAG representation of skeleton and then formulate the search problem. We design a dynamic skeleton sampling algorithm by expanding meta reasoning skeleton along with reasoning context at inference time. This algorithm can derive any meta reasoning skeleton in search space efficiently and adapt skeleton to evolving base reasoning context, thus enable efficient query-aware skeleton search. We conduct experiments on extensive benchmark datasets. Experimental results show that AutoMR achieves better reasoning performance than previous works broadly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.15041v1">HintPilot: LLM-based Compiler Hint Synthesis for Code Optimization</a></div>
    <div class="paper-meta">
      📅 2026-04-16
    </div>
    <details class="paper-abstract">
      Code optimization remains a core objective in software development, yet modern compilers struggle to navigate the enormous optimization spaces. While recent research has looked into employing large language models (LLMs) to optimize source code directly, these techniques can introduce semantic errors and miss fine-grained compiler-level optimization opportunities. We present HintPilot, which bridges LLM-based reasoning with traditional compiler infrastructures via synthesizing compiler hints, annotations that steer compiler behavior. HintPilot employs retrieval-augmented synthesis over compiler documentation and applies profiling-guided iterative refinement to synthesize semantics-preserving and effective hints. Upon PolyBench and HumanEval-CPP benchmarks, HintPilot achieves up to 6.88x geometric mean speedup over -Ofast while preserving program correctness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05158v2">Just Pass Twice: Efficient Token Classification with LLMs for Zero-Shot NER</a></div>
    <div class="paper-meta">
      📅 2026-04-16
      | 💬 16 pages, 9 figures, 12 tables
    </div>
    <details class="paper-abstract">
      Large language models encode extensive world knowledge valuable for zero-shot named entity recognition. However, their causal attention mechanism, where tokens attend only to preceding context, prevents effective token classification when disambiguation requires future context. Existing approaches use LLMs generatively, prompting them to list entities or produce structured outputs, but suffer from slow autoregressive decoding, hallucinated entities, and formatting errors. We propose Just Pass Twice (JPT), a simple yet effective method that enables causal LLMs to perform discriminative token classification with full bidirectional context. Our key insight is that concatenating the input to itself lets each token in the second pass attend to the complete sentence, requiring no architectural modifications. We combine these representations with definition-guided entity embeddings for flexible zero-shot generalization. Our approach achieves state-of-the-art results on zero-shot NER benchmarks, surpassing the previous best method by +7.9 F1 on average across CrossNER and MIT benchmarks, being over 20x faster than comparable generative methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.15022v1">Route to Rome Attack: Directing LLM Routers to Expensive Models via Adversarial Suffix Optimization</a></div>
    <div class="paper-meta">
      📅 2026-04-16
    </div>
    <details class="paper-abstract">
      Cost-aware routing dynamically dispatches user queries to models of varying capability to balance performance and inference cost. However, the routing strategy introduces a new security concern that adversaries may manipulate the router to consistently select expensive high-capability models. Existing routing attacks depend on either white-box access or heuristic prompts, rendering them ineffective in real-world black-box scenarios. In this work, we propose R$^2$A, which aims to mislead black-box LLM routers to expensive models via adversarial suffix optimization. Specifically, R$^2$A deploys a hybrid ensemble surrogate router to mimic the black-box router. A suffix optimization algorithm is further adapted for the ensemble-based surrogate. Extensive experiments on multiple open-source and commercial routing systems demonstrate that {R$^2$A} significantly increases the routing rate to expensive models on queries of different distributions. Code and examples: https://github.com/thcxiker/R2A-Attack.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.15001v1">COEVO: Co-Evolutionary Framework for Joint Functional Correctness and PPA Optimization in LLM-Based RTL Generation</a></div>
    <div class="paper-meta">
      📅 2026-04-16
    </div>
    <details class="paper-abstract">
      LLM-based RTL code generation methods increasingly target both functional correctness and PPA quality, yet existing approaches universally decouple the two objectives, optimizing PPA only after correctness is fully achieved. Whether through sequential multi-agent pipelines, evolutionary search with binary correctness gates, or hierarchical reward dependencies, partially correct but architecturally promising candidates are systematically discarded. Moreover, existing methods reduce the multi-objective PPA space to a single scalar fitness, obscuring the trade-offs among area, delay, and power. To address these limitations, we propose COEVO, a co-evolutionary framework that unifies correctness and PPA optimization within a single evolutionary loop. COEVO formulates correctness as a continuous co-optimization dimension alongside area, delay, and power, enabled by an enhanced testbench that provides fine-grained scoring and detailed diagnostic feedback. An adaptive correctness gate with annealing allows PPA-promising but partially correct candidates to guide the search toward jointly optimal solutions. To preserve the full PPA trade-off structure, COEVO employs four-dimensional Pareto-based non-dominated sorting with configurable intra-level sorting, replacing scalar fitness without manual weight tuning. Evaluated on VerilogEval 2.0 and RTLLM 2.0, COEVO achieves 97.5\% and 94.5\% Pass@1 with GPT-5.4-mini, surpassing all agentic baselines across four LLM backbones, while attaining the best PPA on 43 out of 49 synthesizable RTLLM designs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14969v1">Discovering Novel LLM Experts via Task-Capability Coevolution</a></div>
    <div class="paper-meta">
      📅 2026-04-16
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      Frontier model developers aim to train models continually to possess emergent, diverse capabilities. To extend capabilities, the current pre-training and post-training paradigm requires manually starting training runs with static datasets or reward functions every time. Addressing this limitation, our work pursues the insight that open-endedness (via the coevolution of models and tasks) can discover models with increasingly novel skills in a single run. We introduce a new model development framework that extends coevolution to large language model (LLM) discovery, open-ended \textit{Assessment Coevolving with Diverse Capabilities} (AC/DC). AC/DC evolves both LLMs via model merging and natural language tasks via synthetic data generation. AC/DC discovers growing archives of LLMs that surpass the capabilities of larger LLMs while taking up less GPU memory. In particular, our LLM populations achieve a broader Coverage of expertise than other curated models or baselines on downstream benchmarks, without \textit{any} explicit benchmark optimization. Furthermore, AC/DC improves Coverage over time, continually innovates on tasks and models, and improves performance in multi-agent best-of-N selection. Our findings highlight the potential of coevolution as a means of discovering broader sets of capabilities from base LLMs. Overall, AC/DC brings us one step closer to a profoundly new paradigm of LLM development, where continual improvements to the diversity of model capabilities can be accelerated by leveraging existing models as stepping stones to increasingly powerful models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14961v1">Calibration-Gated LLM Pseudo-Observations for Online Contextual Bandits</a></div>
    <div class="paper-meta">
      📅 2026-04-16
    </div>
    <details class="paper-abstract">
      Contextual bandit algorithms suffer from high regret during cold-start, when the learner has insufficient data to distinguish good arms from bad. We propose augmenting Disjoint LinUCB with LLM pseudo-observations: after each round, a large language model predicts counterfactual rewards for the unplayed arms, and these predictions are injected into the learner as weighted pseudo-observations. The injection weight is controlled by a calibration-gated decay schedule that tracks the LLM's prediction accuracy on played arms via an exponential moving average; high calibration error suppresses the LLM's influence, while accurate predictions receive higher weight during the critical early rounds. We evaluate on two contextual bandit environments - UCI Mushroom (2-arm, asymmetric rewards) and MIND-small (5-arm news recommendation) - and find that when equipped with a task-specific prompt, LLM pseudo-observations reduce cumulative regret by 19% on MIND relative to pure LinUCB. However, generic counterfactual prompt framing increases regret on both environments, demonstrating that prompt design is the dominant factor, more important than the choice of decay schedule or calibration gating parameters. We analyze the failure modes of calibration gating on domains with small prediction errors and provide a theoretical motivation for the bias-variance trade-off governing pseudo-observation weight.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14137v2">From Feelings to Metrics: Understanding and Formalizing How Users Vibe-Test LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-16
      | 💬 Under review. 42 pages, 18 figures. Code and data at https://technion-cs-nlp.github.io/vibe-testing-llms
    </div>
    <details class="paper-abstract">
      Evaluating LLMs is challenging, as benchmark scores often fail to capture models' real-world usefulness. Instead, users often rely on ``vibe-testing'': informal experience-based evaluation, such as comparing models on coding tasks related to their own workflow. While prevalent, vibe-testing is often too ad hoc and unstructured to analyze or reproduce at scale. In this work, we study how vibe-testing works in practice and then formalize it to support systematic analysis. We first analyze two empirical resources: (1) a survey of user evaluation practices, and (2) a collection of in-the-wild model comparison reports from blogs and social media. Based on these resources, we formalize vibe-testing as a two-part process: users personalize both what they test and how they judge responses. We then introduce a proof-of-concept evaluation pipeline that follows this formulation by generating personalized prompts and comparing model outputs using user-aware subjective criteria. In experiments on coding benchmarks, we find that combining personalized prompts and user-aware evaluation can change which model is preferred, reflecting the role of vibe-testing in practice. These findings suggest that formalized vibe-testing can serve as a useful approach for bridging benchmark scores and real-world experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24120v2">Enhancing LLM-Based Neural Network Generation: Few-Shot Prompting and Efficient Validation for Automated Architecture Design</a></div>
    <div class="paper-meta">
      📅 2026-04-16
    </div>
    <details class="paper-abstract">
      Automated neural network architecture design remains a significant challenge in computer vision. Task diversity and computational constraints require both effective architectures and efficient search methods. Large Language Models (LLMs) present a promising alternative to computationally intensive Neural Architecture Search (NAS), but their application to architecture generation in computer vision has not been systematically studied, particularly regarding prompt engineering and validation strategies. Building on the task-agnostic NNGPT/LEMUR framework, this work introduces and validates two key contributions for computer vision. First, we present Few-Shot Architecture Prompting (FSAP), the first systematic study of the number of supporting examples (n = 1, 2, 3, 4, 5, 6) for LLM-based architecture generation. We find that using n = 3 examples best balances architectural diversity and context focus for vision tasks. Second, we introduce Whitespace-Normalized Hash Validation, a lightweight deduplication method (less than 1 ms) that provides a 100x speedup over AST parsing and prevents redundant training of duplicate computer vision architectures. In large-scale experiments across seven computer vision benchmarks (MNIST, CIFAR-10, CIFAR-100, CelebA, ImageNette, SVHN, Places365), we generated 1,900 unique architectures. We also introduce a dataset-balanced evaluation methodology to address the challenge of comparing architectures across heterogeneous vision tasks. These contributions provide actionable guidelines for LLM-based architecture search in computer vision and establish rigorous evaluation practices, making automated design more accessible to researchers with limited computational resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02997v2">From Memorization to Creativity: LLM as a Designer of Novel Neural Architectures</a></div>
    <div class="paper-meta">
      📅 2026-04-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel in program synthesis, yet their capacity for neural architecture design -- balancing syntactic reliability, performance, and structural novelty -- remains underexplored. We present a closed-loop architecture synthesis pipeline within the NNGPT framework, in which a code-oriented LLM evolves over 22 supervised fine-tuning cycles. At each cycle, the LLM synthesizes PyTorch convolutional networks, validated via low-fidelity performance signals and filtered via a MinHash--Jaccard criterion to prevent structural redundancy before being incorporated into the LEMUR dataset. High-performing candidates with novel architectures are converted into prompt--code pairs for parameter-efficient LoRA fine-tuning. This feedback loop drives a measurable distributional shift, progressively internalizing empirical architectural priors such that valid and high-performing outputs evolve from scarce to dominant across cycles. On CIFAR-10, the valid generation rate stabilizes at 50.6% (peaking at 74.5%), mean first-epoch accuracy rises from 28.1% to 51.0%, and candidates exceeding 40% accuracy grow from 2.0% to 96.8%. Cross-dataset transfer to CIFAR-100 and SVHN confirms that improved validity, shifted accuracy distributions, and sustained novelty generalize across benchmarks of varying difficulty and visual domain. Across 22 cycles, 455 unique architectures absent from the original corpus are admitted under the novelty filter. By grounding synthesis in execution feedback and novelty filtering, we demonstrate that iterative self-supervised fine-tuning reshapes an LLM into a task-specialized architectural prior -- improving generation reliability, proxy performance, and structural diversity -- offering a reproducible, annotation-free alternative to hand-crafted search spaces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.06708v2">AISysRev -- LLM-based Tool for Title-abstract Screening</a></div>
    <div class="paper-meta">
      📅 2026-04-16
      | 💬 4 pages + references. Accepted for publication in the Proceedings of the 34th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering (FSE 2026)
    </div>
    <details class="paper-abstract">
      Conducting systematic reviews is laborious. In the screening or study selection phase, the number of papers can be overwhelming. Recent research has demonstrated that large language models (LLMs) can perform title-abstract screening and support humans in the task. To this end, we developed AISysRev, an LLM-based screening tool implemented as a containerized web application. The tool accepts CSV files containing paper titles and abstracts. Users specify inclusion and exclusion criteria. Multiple different LLMs can be used, such as Gemini, Claude, Mistral or ChatGPT via OpenRouter. We also support locally hosted models and any model compatible with the OpenAI SDK. AISysRev implements both zero-shot and few-shot prompting, and also allows for manual screening through interfaces that display LLM results as guidance for human reviewers. LLM calls are parallelized, meaning screening speed is typically between 100 to 300 papers per minute, depending on the model and the host. To demonstrate the tool's use in practice, we conducted a qualitative trial study with 137 papers using the tool. Our findings indicate that papers can be classified into four categories: Easy Includes, Easy Excludes, Boundary Includes, and Boundary Excludes. The Boundary cases, where LLMs are prone to errors, highlight the need for human intervention. While LLMs do not replace human judgment in systematic reviews, they can reduce the burden of assessing large volumes of scientific literature. Video: https://www.youtube.com/watch?v=HeblemlgnAQ Tool: https://github.com/EvoTestOps/AISysRev
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.11017v2">Logo-LLM: Local and Global Modeling with Large Language Models for Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2026-04-16
    </div>
    <details class="paper-abstract">
      Time series forecasting is critical across multiple domains, where time series data exhibit both local patterns and global dependencies. While Transformer-based methods effectively capture global dependencies, they often overlook short-term local variations in time series. Recent methods that adapt large language models (LLMs) into time series forecasting inherit this limitation by treating LLMs as black-box encoders, relying solely on the final-layer output and underutilizing hierarchical representations. To address this limitation, we propose Logo-LLM, a novel LLM-based framework that explicitly extracts and models multi-scale temporal features from different layers of a pre-trained LLM. Through empirical analysis, we show that shallow layers of LLMs capture local dynamics in time series, while deeper layers encode global trends. Moreover, Logo-LLM introduces lightweight Local-Mixer and Global-Mixer modules to align and integrate features with the temporal input across layers. Extensive experiments demonstrate that Logo-LLM achieves superior performance across diverse benchmarks, with strong generalization in few-shot and zero-shot settings while maintaining low computational overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.24284v3">MCP-Flow: Facilitating LLM Agents to Master Real-World, Diverse and Scaling MCP Tools</a></div>
    <div class="paper-meta">
      📅 2026-04-16
      | 💬 ACL 2026 Main, Camera Ready
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly rely on external tools to perform complex, realistic tasks, yet their ability to utilize the rapidly expanding Model Contextual Protocol (MCP) ecosystem remains limited. Existing MCP research covers few servers, depends on costly manual curation, and lacks training support, hindering progress toward real-world deployment. To overcome these limitations, we introduce MCP-Flow, an automated web-agent-driven pipeline for large-scale server discovery, data synthesis, and model training. MCP-Flow collects and filters data from 1166 servers and 11536 tools, producing 68733 high-quality instruction-function call pairs and 6439 trajectories, far exceeding prior work in scale and diversity. Extensive experiments demonstrate MCP-Flow's effectiveness in driving superior MCP tool selection, function-call generation, and enhanced agentic task performance. MCP-Flow thus provides a scalable foundation for advancing LLM agents' proficiency in real-world MCP environments. MCP-Flow is publicly available at https://github.com/wwh0411/MCP-Flow.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14892v1">Can LLMs Score Medical Diagnoses and Clinical Reasoning as well as Expert Panels?</a></div>
    <div class="paper-meta">
      📅 2026-04-16
    </div>
    <details class="paper-abstract">
      Evaluating medical AI systems using expert clinician panels is costly and slow, motivating the use of large language models (LLMs) as alternative adjudicators. Here, we evaluate an LLM jury composed of three frontier AI models scoring 3333 diagnoses on 300 real-world middle-income country (MIC) hospital cases. Model performance was benchmarked against expert clinician panel and independent human re-scoring panel evaluations. Both LLM and clinician-generated diagnoses are scored across four dimensions: diagnosis, differential diagnosis, clinical reasoning and negative treatment risk. For each of these, we assess scoring difference, inter-rater agreement, scoring stability, severe safety errors and the effect of post-hoc calibration. We find that: (i) the uncalibrated LLM jury scores are systematically lower than clinician panels scores; (ii) the LLM Jury preserves ordinal agreement and exhibits better concordance with the primary expert panels than the human expert re-score panels do; (iii) the probability of severe errors is lower in \lj models compared to the human expert re-score panels; (iv) the LLM Jury shows excellent agreement with primary expert panels' rankings. We find that the LLM jury combined with AI model diagnoses can be used to identify ward diagnoses at high risk of error, enabling targeted expert review and improved panel efficiency; (v) LLM jury models show no self-preference bias. They did not score diagnoses generated by their own underlying model or models from the same vendor more (or less) favourably than those generated by other models. Finally, we demonstrate that LLM jury calibration using isotonic regression improves alignment with human expert panel evaluations. Together, these results provide compelling evidence that a calibrated, multi-model LLM jury can serve as a trustworthy and reliable proxy for expert clinician evaluation in medical AI benchmarking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14877v1">Does RL Expand the Capability Boundary of LLM Agents? A PASS@(k,T) Analysis</a></div>
    <div class="paper-meta">
      📅 2026-04-16
    </div>
    <details class="paper-abstract">
      Does reinforcement learning genuinely expand what LLM agents can do, or merely make them more reliable? For static reasoning, recent work answers the second: base and RL pass@k curves converge at large k. We ask whether this holds for agentic tool use, where T rounds of interaction enable compositional strategies that re-sampling cannot recover. We introduce PASS@(k,T), a two-dimensional metric that jointly varies sampling budget k and interaction depth T, separating capability expansion from efficiency improvement. Our main finding is that, contrary to the static-reasoning result, tool-use RL genuinely enlarges the capability boundary: the RL agent's pass-curve pulls above the base model's and the gap widens at large k rather than converging. The expansion is specific to compositional, sequential information gathering; on simpler tasks RL behaves as prior work predicts. Under matched training data, supervised fine-tuning regresses the boundary on the same compositional tasks, isolating self-directed exploration as the causal factor. Mechanism analysis shows RL reweights the base strategy distribution toward the subset whose downstream reasoning more often yields a correct answer, with the improvement concentrated on how the agent integrates retrieved information. These results reconcile optimistic and pessimistic readings of RL for LLMs: both are correct, on different task types.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14865v1">Segment-Level Coherence for Robust Harmful Intent Probing in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-16
      | 💬 preprint
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly exposed to adaptive jailbreaking, particularly in high-stakes Chemical, Biological, Radiological, and Nuclear (CBRN) domains. Although streaming probes enable real-time monitoring, they still make systematic errors. We identify a core issue: existing methods often rely on a few high-scoring tokens, leading to false alarms when sensitive CBRN terms appear in benign contexts. To address this, we introduce a streaming probing objective that requires multiple evidence tokens to consistently support a prediction, rather than relying on isolated spikes. This encourages more robust detection based on aggregated signals instead of single-token cues. At a fixed 1% false-positive rate, our method improves the true-positive rate by 35.55% relative to strong streaming baselines. We further observe substantial gains in AUROC, even when starting from near-saturated baseline performance (AUROC = 97.40%). We also show that probing Attention or MLP activations consistently outperforms residual-stream features. Finally, even when adversarial fine-tuning enables novel character-level ciphers, harmful intent remains detectable: probes developed for the base LLMs can be applied ``plug-and-play'' to these obfuscated attacks, achieving an AUROC of over 98.85%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14853v1">Adaptive Test-Time Compute Allocation for Reasoning LLMs via Constrained Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2026-04-16
    </div>
    <details class="paper-abstract">
      Test-time compute scaling, the practice of spending extra computation during inference via repeated sampling, search, or extended reasoning, has become a powerful lever for improving large language model performance. Yet deploying these techniques under finite inference budgets requires a decision that current systems largely ignore: which inputs deserve more compute, and which can be answered cheaply? We formalize this as a constrained optimization problem (maximize expected accuracy subject to an average compute budget) and solve it with a two-stage Solve-then-Learn pipeline. In the solve stage, Lagrangian relaxation decomposes the global constraint into per-instance sub-problems, each admitting a closed-form oracle action that optimally prices accuracy against cost. We prove that the induced cost is monotone in the dual variable, enabling exact budget targeting via binary search. In the learn stage, a lightweight classifier is trained to predict oracle actions from cheap input features, amortizing the allocation rule for real-time deployment. We establish that the task-level regret of the learned policy is bounded by its imitation error times the worst-case per-instance gap, yielding a clean reduction from constrained inference to supervised classification. Experiments on MATH and GSM8K with three LLMs (DeepSeek-V3, GPT-4o-mini, Qwen2.5-7B) show that our method consistently outperforms uniform and heuristic allocation baselines, achieving up to 12.8% relative accuracy improvement on MATH under matched budget constraints, while closely tracking the Lagrangian oracle upper bound with over 91% imitation accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14843v1">Exploring and Testing Skill-Based Behavioral Profile Annotation: Human Operability and LLM Feasibility under Schema-Guided Execution</a></div>
    <div class="paper-meta">
      📅 2026-04-16
    </div>
    <details class="paper-abstract">
      Behavioral Profile (BP) annotation is difficult to automate because it requires simultaneous coding across multiple linguistic dimensions. We treat BP annotation as a bundle of annotation skills rather than a single task and evaluate LLM-assisted BP annotation from this perspective. Using 3,134 concordance lines of 30 Chinese metaphorical color-term derivatives and a 14-feature BP schema, we implement a skill-file-driven pipeline in which each feature is externally defined through schema files, decision rules, and examples. Two human annotators completed a two-round schema-only protocol on a 300-instance validation subset, enabling BP skills to be classified as directly operable, recoverable under focused re-annotation, or structurally underspecified. GPT-5.4 and three locally deployable open-source models were then evaluated under the same setup. Results show that BP annotation is highly heterogeneous at the skill level: 5 skills are directly operable, 4 are recoverable after focused re-annotation, and 5 remain structurally underspecified. GPT-5.4 executes the retained skills with substantial reliability (accuracy = 0.678, \k{appa} = 0.665, weighted F1 = 0.695), but this feasibility is selective rather than global. Human and GPT difficulty profiles are strongly aligned at the skill level (r = 0.881), but not at the instance level (r = 0.016) or lexical-item level (r = -0.142), a pattern we describe as shared taxonomy, independent execution. Pairwise agreement further suggests that GPT is better understood as an independent third skill voice than as a direct human substitute. Open-source failures are concentrated in schema-to-skill execution problems. These findings suggest that automatic annotation should be evaluated in terms of skill feasibility rather than task-level automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14833v1">Federated User Behavior Modeling for Privacy-Preserving LLM Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-04-16
    </div>
    <details class="paper-abstract">
      Large Language Models have shown great success in recommender systems. However, the limited and sparse nature of user data often restricts the LLM's ability to effectively model behavior patterns. To address this, existing studies have explored cross-domain solutions by conducting Cross-Domain Recommendation tasks. But previous methods typically assume domains are overlapped and can be accessed readily. None of the LLM methods address the privacy-preserving issues in the CDR settings, that is, Privacy-Preserving Cross-Domain Recommendation. Conducting non-overlapping PPCDR with LLM is challenging since: 1)The inability to share user identity or behavioral data across domains impedes effective cross-domain alignment. 2)The heterogeneity of data modalities across domains complicates knowledge integration. 3)Fusing collaborative filtering signals from traditional recommendation models with LLMs is difficult, as they operate within distinct feature spaces. To address the above issues, we propose SF-UBM, a Semantic-enhanced Federated User Behavior Modeling method. Specifically, to deal with Challenge 1, we leverage natural language as a universal bridge to connect disjoint domains via a semantic-enhanced federated architecture. Here, text-based item representations are encrypted and shared, while user-specific data remains local. To handle Challenge 2, we design a Fact-counter Knowledge Distillation module to integrate domain-agnostic knowledge with domain-specific knowledge, across different data modalities. To tackle Challenge 3, we project pre-learned user preferences and cross-domain item representations into the soft prompt space, aligning behavioral and semantic spaces for effective LLM learning. We conduct extensive experiments on three pairs of real-world domains, and the experimental results demonstrate the effectiveness of SF-UBM compared to the recent SOTA methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.13829v3">A Linguistics-Aware LLM Watermarking via Syntactic Predictability</a></div>
    <div class="paper-meta">
      📅 2026-04-16
      | 💬 ACL 2026
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to advance rapidly, reliable governance tools have become critical. Publicly verifiable watermarking is particularly essential for fostering a trustworthy AI ecosystem. A central challenge persists: balancing text quality against detection robustness. Recent studies have sought to navigate this trade-off by leveraging signals from model output distributions (e.g., token-level entropy); however, their reliance on these model-specific signals presents a significant barrier to public verification, as the detection process requires access to the logits of the underlying model. We introduce STELA, a novel framework that aligns watermark strength with the linguistic degrees of freedom inherent in language. STELA dynamically modulates the signal using part-of-speech (POS) n-gram-modeled linguistic indeterminacy, weakening it in grammatically constrained contexts to preserve quality and strengthening it in contexts with greater linguistic flexibility to enhance detectability. Our detector operates without access to any model logits, thus facilitating publicly verifiable detection. Through extensive experiments on typologically diverse languages-analytic English, isolating Chinese, and agglutinative Korean-we show that STELA surpasses prior methods in detection robustness. Our code is available at https://github.com/Shinwoo-Park/stela_watermark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.01869v2">Skill-Pro: Learning Reusable Skills from Experience via Non-Parametric PPO for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-16
      | 💬 20 Pages, 6 Figures, 4 Tables
    </div>
    <details class="paper-abstract">
      LLM-driven agents demonstrate strong performance in sequential decision-making but often rely on on-the-fly reasoning, re-deriving solutions even in recurring scenarios. This insufficient experience reuse leads to computational redundancy and execution instability. To bridge this gap, we propose Skill-Pro, a framework that enables agents to autonomously learn reusable procedural skills from interaction experiences without parameter updates. By formalizing a Skill-MDP, Skill-Pro transforms passive episodic narratives into executable Skills defined by activation, execution, and termination conditions to ensure executability. To achieve reliable reusability without capability degradation, we introduce Non-Parametric PPO, which leverages semantic gradients for high-quality candidate generation and a PPO Gate for robust Skill verification. Through score-based maintenance, Skill-Pro sustains compact, high-quality procedural memory. Experimental results across in-domain, cross-task, and cross-agent scenarios demonstrate that Skill-Pro achieves superior reuse rates and significant performance gains with extreme memory compression. Visualized evolutionary trajectories and Skill distributions further reveal how Skill-Pro transparently accumulates, refines, and reuses procedural knowledge to facilitate long-term autonomy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14808v1">Modeling LLM Unlearning as an Asymmetric Two-Task Learning Problem</a></div>
    <div class="paper-meta">
      📅 2026-04-16
      | 💬 ACL 2026
    </div>
    <details class="paper-abstract">
      Machine unlearning for large language models (LLMs) aims to remove targeted knowledge while preserving general capability. In this paper, we recast LLM unlearning as an asymmetric two-task problem: retention is the primary objective and forgetting is an auxiliary. From this perspective, we propose a retention-prioritized gradient synthesis framework that decouples task-specific gradient extraction from conflict-aware combination. Instantiating the framework, we adapt established PCGrad to resolve gradient conflicts, and introduce SAGO, a novel retention-prioritized gradient synthesis method. Theoretically, both variants ensure non-negative cosine similarity with the retain gradient, while SAGO achieves strictly tighter alignment through constructive sign-constrained synthesis. Empirically, on WMDP Bio/Cyber and RWKU benchmarks, SAGO consistently pushes the Pareto frontier: e.g., on WMDP Bio (SimNPO+GD), recovery of target model MMLU performance progresses from 44.6% (naive) to 94.0% (+PCGrad) and further to 96.0% (+SAGO), while maintaining comparable forgetting strength. Our results show that re-shaping gradient geometry, rather than re-balancing losses, is the key to mitigating unlearning-retention trade-offs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14807v1">The LLM Fallacy: Misattribution in AI-Assisted Cognitive Workflows</a></div>
    <div class="paper-meta">
      📅 2026-04-16
    </div>
    <details class="paper-abstract">
      The rapid integration of large language models (LLMs) into everyday workflows has transformed how individuals perform cognitive tasks such as writing, programming, analysis, and multilingual communication. While prior research has focused on model reliability, hallucination, and user trust calibration, less attention has been given to how LLM usage reshapes users' perceptions of their own capabilities. This paper introduces the LLM fallacy, a cognitive attribution error in which individuals misinterpret LLM-assisted outputs as evidence of their own independent competence, producing a systematic divergence between perceived and actual capability. We argue that the opacity, fluency, and low-friction interaction patterns of LLMs obscure the boundary between human and machine contribution, leading users to infer competence from outputs rather than from the processes that generate them. We situate the LLM fallacy within existing literature on automation bias, cognitive offloading, and human--AI collaboration, while distinguishing it as a form of attributional distortion specific to AI-mediated workflows. We propose a conceptual framework of its underlying mechanisms and a typology of manifestations across computational, linguistic, analytical, and creative domains. Finally, we examine implications for education, hiring, and AI literacy, and outline directions for empirical validation. We also provide a transparent account of human--AI collaborative methodology. This work establishes a foundation for understanding how generative AI systems not only augment cognitive performance but also reshape self-perception and perceived expertise.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.02135v2">Graph-Based Alternatives to LLMs for Human Simulation</a></div>
    <div class="paper-meta">
      📅 2026-04-16
      | 💬 Conference: ACL 2026 Long Main Code: https://github.com/schang-lab/gems
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become a popular approach for simulating human behaviors, yet it remains unclear if LLMs are necessary for all simulation tasks. We study a broad family of close-ended simulation tasks, with applications from survey prediction to test-taking, and show that a graph neural network can match or surpass strong LLM-based methods. We introduce Graph-basEd Models for Human Simulation (GEMS) which formulates close-ended simulation as link prediction on a heterogeneous graph of individuals and choices. Across three datasets and three evaluation settings, GEMS matches or outperforms the strongest LLM-based methods while using three orders of magnitude fewer parameters. These results suggest that graph-based modeling can complement LLMs as an efficient and transparent approach to simulating human behaviors. Code is available at https://github.com/schang-lab/gems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14746v1">Disentangle-then-Refine: LLM-Guided Decoupling and Structure-Aware Refinement for Graph Contrastive Learning</a></div>
    <div class="paper-meta">
      📅 2026-04-16
      | 💬 Accept in ICME 2026
    </div>
    <details class="paper-abstract">
      Conventional Graph Contrastive Learning (GCL) on Text-Attributed Graphs (TAGs) relies on blind stochastic augmentations, inadvertently entangling task-relevant signals with noise. We propose SDM-SCR, a robust framework anchored in Approximate Orthogonal Decomposition. First, the Semantic Decoupling Module (SDM) leverages the instruction-following capability of Large Language Models (LLMs) to actively parse raw attributes into asymmetric, task-oriented signal and noise views. This shifts the paradigm from random perturbation to semantic-aware disentanglement. Subsequently, Semantic Consistency Regularization (SCR) exploits the spectral observation that semantic signals are topologically smooth while residual noise is high-frequency. SCR functions as a selective spectral filter, enforcing consistency only on the signal subspace to eliminate LLM hallucinations without over-smoothing. This ``Disentangle-then-Refine'' mechanism ensures rigorous signal purification. Extensive experiments demonstrate that SDM-SCR achieves SOTA performance in accuracy and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.10101v2">Who Wrote This Line? Evaluating the Detection of LLM-Generated Classical Chinese Poetry</a></div>
    <div class="paper-meta">
      📅 2026-04-16
      | 💬 Accepted to ACL 2026 Main Conference
    </div>
    <details class="paper-abstract">
      The rapid development of large language models (LLMs) has extended text generation tasks into the literary domain. However, AI-generated literary creations has raised increasingly prominent issues of creative authenticity and ethics in literary world, making the detection of LLM-generated literary texts essential and urgent. While previous works have made significant progress in detecting AI-generated text, it has yet to address classical Chinese poetry. Due to the unique linguistic features of classical Chinese poetry, such as strict metrical regularity, a shared system of poetic imagery, and flexible syntax, distinguishing whether a poem is authored by AI presents a substantial challenge. To address these issues, we introduce ChangAn, a benchmark for detecting LLM-generated classical Chinese poetry that containing total 30,664 poems, 10,276 are human-written poems and 20,388 poems are generated by four popular LLMs. Based on ChangAn, we conducted a systematic evaluation of 12 AI detectors, investigating their performance variations across different text granularities and generation strategies. Our findings highlight the limitations of current Chinese text detectors, which fail to serve as reliable tools for detecting LLM-generated classical Chinese poetry. These results validate the effectiveness and necessity of our proposed ChangAn benchmark. Our dataset and code are available at https://github.com/VelikayaScarlet/ChangAn.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14709v1">HWE-Bench: Benchmarking LLM Agents on Real-World Hardware Bug Repair Tasks</a></div>
    <div class="paper-meta">
      📅 2026-04-16
    </div>
    <details class="paper-abstract">
      Existing benchmarks for hardware design primarily evaluate Large Language Models (LLMs) on isolated, component-level tasks such as generating HDL modules from specifications, leaving repository-scale evaluation unaddressed. We introduce HWE-Bench, the first large-scale, repository-level benchmark for evaluating LLM agents on real-world hardware bug repair tasks. HWE-Bench comprises 417 task instances derived from real historical bug-fix pull requests across six major open-source projects spanning both Verilog/SystemVerilog and Chisel, covering RISC-V cores, SoCs, and security roots-of-trust. Each task is grounded in a fully containerized environment where the agent must resolve a real bug report, with correctness validated through the project's native simulation and regression flows. The benchmark is built through a largely automated pipeline that enables efficient expansion to new repositories. We evaluate seven LLMs with four agent frameworks and find that the best agent resolves 70.7% of tasks overall, with performance exceeding 90% on smaller cores but dropping below 65% on complex SoC-level projects. We observe larger performance gaps across models than commonly reported on software benchmarks, and difficulty is driven by project scope and bug-type distribution rather than code size alone. Our failure analysis traces agent failures to three stages of the debugging process: fault localization, hardware-semantic reasoning, and cross-artifact coordination across RTL, configuration, and verification components, providing concrete directions for developing more capable hardware-aware agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.03341v4">What Deserves Memory: Adaptive Memory Distillation for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-16
    </div>
    <details class="paper-abstract">
      Memory systems for LLM agents struggle to determine what information deserves retention. Existing approaches rely on predefined heuristics such as importance scores, emotional tags, or factual templates, encoding designer intuition rather than learning from the data itself. Inspired by cognitive ideas, we propose NEMORI, an adaptive memory distillation framework that casts the assessment of experience's future utility as a matter of predictability. Specifically, NEMORI comprises two cascading modules: Episodic Memory Integration transforms raw interactions into coherent narratives, and Semantic Knowledge Distillation extracts insights via prediction error. Centering on distillation, the framework remains agnostic to downstream management. Extensive experiments confirm that NEMORI achieves strong performance, efficiency, and storage reduction. Our work suggests that observing the intrinsic properties of interaction sequences offers a viable, data-driven alternative to heuristic-based memory design. Code: https://github.com/nemori-ai/nemori.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14691v1">CAMO: An Agentic Framework for Automated Causal Discovery from Micro Behaviors to Macro Emergence in LLM Agent Simulations</a></div>
    <div class="paper-meta">
      📅 2026-04-16
    </div>
    <details class="paper-abstract">
      LLM-empowered agent simulations are increasingly used to study social emergence, yet the micro-to-macro causal mechanisms behind macro outcomes often remain unclear. This is challenging because emergence arises from intertwined agent interactions and meso-level feedback and nonlinearity, making generative mechanisms hard to disentangle. To this end, we introduce \textbf{\textsc{CAMO}}, an automated \textbf{Ca}usal discovery framework from \textbf{M}icr\textbf{o} behaviors to \textbf{M}acr\textbf{o} Emergence in LLM agent simulations. \textsc{CAMO} converts mechanistic hypotheses into computable factors grounded in simulation records and learns a compact causal representation centered on an emergent target $Y$. \textsc{CAMO} outputs a computable Markov boundary and a minimal upstream explanatory subgraph, yielding interpretable causal chains and actionable intervention levers. It also uses simulator-internal counterfactual probing to orient ambiguous edges and revise hypotheses when evidence contradicts the current view. Experiments across four emergent settings demonstrate the promise of \textsc{CAMO}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14687v1">M2-PALE: A Framework for Explaining Multi-Agent MCTS--Minimax Hybrids via Process Mining and LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-16
    </div>
    <details class="paper-abstract">
      Monte-Carlo Tree Search (MCTS) is a fundamental sampling-based search algorithm widely used for online planning in sequential decision-making domains. Despite its success in driving recent advances in artificial intelligence, understanding the behavior of MCTS agents remains a challenge for both developers and users. This difficulty stems from the complex search trees produced through the simulation of numerous future states and their intricate relationships. A known weakness of standard MCTS is its reliance on highly selective tree construction, which may lead to the omission of crucial moves and a vulnerability to tactical traps. To resolve this, we incorporate shallow, full-width Minimax search into the rollout phase of multi-agent MCTS to enhance strategic depth. Furthermore, to demystify the resulting decision-making logic, we introduce \textsf{M2-PALE} (MCTS--Minimax Process-Aided Linguistic Explanations). This framework employs process mining techniques, specifically the Alpha Miner, iDHM, and Inductive Miner algorithms, to extract underlying behavioral workflows from agent execution traces. These process models are then synthesized by LLMs to generate human-readable causal and distal explanations. We demonstrate the efficacy of our approach in a small-scale checkers environment, establishing a scalable foundation for interpreting hybrid agents in increasingly complex strategic domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13600v2">SAKURAONE: An Open Ethernet-Based AI HPC System and Its Observed Workload Dynamics in a Single-Tenant LLM Development Environment</a></div>
    <div class="paper-meta">
      📅 2026-04-16
      | 💬 Accepted at MLSys 2026
    </div>
    <details class="paper-abstract">
      SAKURAONE is a managed high performance computing (HPC) cluster developed and operated by the SAKURA Internet Research Center. It builds on the KOKARYOKU PHY bare metal GPU platform and is optimized for advanced workloads, including large language model (LLM) training. In ISC 2025 TOP500, SAKURAONE is ranked 49th by HPL and is the only top 100 system that uses a fully open networking stack - 800 GbE with SONiC - demonstrating the scalability of vendor-neutral technology. Measured performance is 33.95 PFLOP/s (HPL Rmax), 396.295 TFLOP/s (HPCG), and 339.86 PFLOP/s on HPL-MxP with FP8. The system consists of 100 nodes, each with eight NVIDIA H100 GPUs and a 2 PB all-flash Lustre file system, interconnected via a rail-optimized 800 GbE leaf-spine fabric with RoCEv2. Through exclusive use by a single research project, we observed the characteristics of development-related jobs. Consistent with previous HPC studies, small-scale jobs dominated in number, while a few large-scale jobs accounted for most GPU resource time. As the project progressed, resource use shifted from large-scale to mid-scale jobs, reflecting a transition from initial large-scale training to iterative refinement. These observations illustrate the real-world utilization dynamics of GPU clusters under unified project workloads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14644v1">CURaTE: Continual Unlearning in Real Time with Ensured Preservation of LLM Knowledge</a></div>
    <div class="paper-meta">
      📅 2026-04-16
      | 💬 Accepted to Findings of ACL 2026
    </div>
    <details class="paper-abstract">
      The inability to filter out in advance all potentially problematic data from the pre-training of large language models has given rise to the need for methods for unlearning specific pieces of knowledge after training. Existing techniques overlook the need for continuous and immediate action, causing them to suffer from degraded utility as updates accumulate and protracted exposure of sensitive information. To address these issues, we propose Continual Unlearning in Real Time with Ensured Preservation of LLM Knowledge (CURaTE). Our method begins by training a sentence embedding model on a dataset designed to enable the formation of sharp decision boundaries for determining whether a given input prompt corresponds to any stored forget requests. The similarity of a given input to the forget requests is then used to determine whether to answer or return a refusal response. We show that even with such a simple approach, not only does CURaTE achieve more effective forgetting than existing methods, but by avoiding modification of the language model parameters, it also maintains near perfect knowledge preservation over any number of updates and is the only method capable of continual unlearning in real-time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05242v2">XMark: Reliable Multi-Bit Watermarking for LLM-Generated Texts</a></div>
    <div class="paper-meta">
      📅 2026-04-16
      | 💬 Accepted by ACL 2026 as a main conference paper
    </div>
    <details class="paper-abstract">
      Multi-bit watermarking has emerged as a promising solution for embedding imperceptible binary messages into Large Language Model (LLM)-generated text, enabling reliable attribution and tracing of malicious usage of LLMs. Despite recent progress, existing methods still face key limitations: some become computationally infeasible for large messages, while others suffer from a poor trade-off between text quality and decoding accuracy. Moreover, the decoding accuracy of existing methods drops significantly when the number of tokens in the generated text is limited, a condition that frequently arises in practical usage. To address these challenges, we propose \textsc{XMark}, a novel method for encoding and decoding binary messages in LLM-generated texts. The unique design of \textsc{XMark}'s encoder produces a less distorted logit distribution for watermarked token generation, preserving text quality, and also enables its tailored decoder to reliably recover the encoded message with limited tokens. Extensive experiments across diverse downstream tasks show that \textsc{XMark} significantly improves decoding accuracy while preserving the quality of watermarked text, outperforming prior methods. The code is at https://github.com/JiiahaoXU/XMark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14517v2">Beyond Translation: Evaluating Mathematical Reasoning Capabilities of LLMs in Sinhala and Tamil</a></div>
    <div class="paper-meta">
      📅 2026-04-16
      | 💬 Accepted to ITHET 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved strong results in mathematical reasoning, and are increasingly deployed as tutoring and learning support tools in educational settings. However, their reliability for students working in non-English languages, especially low-resource languages, remains poorly understood. We examine this gap by evaluating mathematical reasoning in Sinhala and Tamil -- two languages widely used in South Asian schools but underrepresented in artificial intelligence (AI) research. Using a taxonomy of six math problem types, from basic arithmetic to complex unit conflict and optimization problems, we evaluate four prominent large language models. To avoid translation artifacts that confound language ability with translation quality, we construct a parallel dataset in which each problem is independently authored in Sinhala and Tamil by native speakers, and in English by fluent speakers, all with strong mathematical backgrounds. Our analysis demonstrates that while basic arithmetic reasoning transfers robustly across languages, complex reasoning tasks show significant degradation in Tamil and Sinhala. The pattern of failures varies by model and problem type, suggesting that strong performance in English does not guarantee reliable performance across languages. These findings have direct implications for the deployment of AI tools in multilingual classrooms, and highlight the need for language-specific evaluation before adopting large language models as math tutoring aids in non-English educational contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14593v1">Mechanistic Decoding of Cognitive Constructs in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-16
      | 💬 This work has been submitted to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) demonstrate increasingly sophisticated affective capabilities, the internal mechanisms by which they process complex emotions remain unclear. Existing interpretability approaches often treat models as black boxes or focus on coarse-grained basic emotions, leaving the cognitive structure of more complex affective states underexplored. To bridge this gap, we propose a Cognitive Reverse-Engineering framework based on Representation Engineering (RepE) to analyze social-comparison jealousy. By combining appraisal theory with subspace orthogonalization, regression-based weighting, and bidirectional causal steering, we isolate and quantify two psychological antecedents of jealousy, Superiority of Comparison Person and Domain Self-Definitional Relevance, and examine their causal effects on model judgments. Experiments on eight LLMs from the Llama, Qwen, and Gemma families suggest that models natively encode jealousy as a structured linear combination of these constituent factors. Their internal representations are broadly consistent with the human psychological construct, treating Superiority as the foundational trigger and Relevance as the ultimate intensity multiplier. Our framework also demonstrates that toxic emotional states can be mechanically detected and surgically suppressed, suggesting a possible route toward representational monitoring and intervention for AI safety in multi-agent environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17512v4">Language on Demand, Knowledge at Core: Composing LLMs with Encoder-Decoder Translation Models for Extensible Multilinguality</a></div>
    <div class="paper-meta">
      📅 2026-04-16
      | 💬 ACL 2026 Main Conference. Code: https://github.com/ictnlp/XBridge | Models: https://huggingface.co/collections/ICTNLP/xbridge
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit strong general intelligence, yet their multilingual performance remains highly imbalanced. Although LLMs encode substantial cross-lingual knowledge in a unified semantic space, they often struggle to reliably interface this knowledge with low-resource or unseen languages. Fortunately, pretrained encoder-decoder translation models already possess balanced multilingual capability, suggesting a natural complement to LLMs. In this work, we propose XBridge, a compositional encoder-LLM-decoder architecture that offloads multilingual understanding and generation to external pretrained translation models, while preserving the LLM as an English-centric core for general knowledge processing. To address the resulting representation misalignment across models, we introduce lightweight cross-model mapping layers and an optimal transport-based alignment objective, enabling fine-grained semantic consistency for multilingual generation. Experiments on four LLMs across multilingual understanding, reasoning, summarization, and generation indicate that XBridge outperforms strong baselines, especially on low-resource and previously unseen languages, without retraining the LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.01014v3">IF-CRITIC: Towards a Fine-Grained LLM Critic for Instruction-Following Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-04-16
      | 💬 ACL 2026
    </div>
    <details class="paper-abstract">
      Instruction-following is a fundamental ability of Large Language Models (LLMs), requiring their generated outputs to follow multiple constraints imposed in input instructions. Numerous studies have attempted to enhance this ability through preference optimization or reinforcement learning based on reward signals from LLM-as-a-Judge. However, existing evaluation models for instruction-following still possess many deficiencies, such as substantial costs and unreliable assessments. To this end, we propose IF-CRITIC, an LLM critic for fine-grained, efficient, and reliable instruction-following evaluation. We first develop a checklist generator to decompose instructions and generate constraint checklists. With the assistance of the checklists, we collect high-quality critique training data through a multi-stage critique filtering mechanism and employ a constraint-level preference optimization method to train IF-CRITIC. Extensive experiments show that the evaluation performance of IF-CRITIC can beat strong LLM-as-a-Judge baselines, including o4-mini and Gemini-3-Pro. With the reward signals provided by IF-CRITIC, LLMs can achieve substantial performance gains in instruction-following optimization under lower computational overhead compared to strong LLM critic baselines. Our code and model are available at https://github.com/thu-coai/IF-CRITIC.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14531v1">TRACER: Trace-Based Adaptive Cost-Efficient Routing for LLM Classification</a></div>
    <div class="paper-meta">
      📅 2026-04-16
      | 💬 github.com/adrida/tracer
    </div>
    <details class="paper-abstract">
      Every call to an LLM classification endpoint produces a labeled input-output pair already retained in production logs. These pairs constitute a free, growing training set: a lightweight surrogate trained on them can absorb a significant portion of future traffic at near-zero marginal inference cost. The open questions are when the surrogate is reliable enough to deploy, what it handles versus defers, and how that boundary evolves as data accumulates. We introduce TRACER (Trace-based Adaptive Cost-Efficient Routing), an open-source system that trains ML surrogates on an LLM's own production traces and governs deployment through a parity gate: the surrogate is activated only when its agreement with the LLM exceeds a user-specified threshold α. To make the routing boundary transparent, TRACER generates interpretability artifacts describing which input regions the surrogate handles, where it plateaus, and why it defers. On a 77-class intent benchmark with a Sonnet 4.6 teacher, TRACER achieves 83-100% surrogate coverage depending on the quality target α; on a 150-class benchmark, the surrogate fully replaces the teacher. On a natural language inference task, the parity gate correctly refuses deployment because the embedding representation cannot support reliable separation. The system is available as open-source software.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.11581v2">Hidden Measurement Error in LLM Pipelines Distorts Annotation, Evaluation, and Benchmarking</a></div>
    <div class="paper-meta">
      📅 2026-04-15
    </div>
    <details class="paper-abstract">
      LLM evaluations drive which models get deployed, which safety standards get adopted, and which research conclusions get published. Yet these scores carry hidden uncertainty: rephrasing the prompt, switching the judge model, or changing the temperature can shift results enough to flip rankings and reverse conclusions. Standard confidence intervals ignore this variance, producing under-coverage that worsens with more data. The same unmeasured variance creates an exploitable surface for benchmarks: model developers can optimize against measurement noise rather than genuine performance (some have infamously done so, see \citep{boyeau2025leaderboard}). This paper decomposes LLM pipeline uncertainty into its sources, distinguishes variance that shrinks with more data from sensitivity to researcher design choices, and uses design-study projections to reduce total error. Across ideology annotation, safety classification, MMLU benchmarking, and a human-validated propaganda audit, the decomposition reveals that the dominant variance source differs by domain and scoring method. On MMLU, optimized budget allocation halves estimation error at equivalent cost. On the propaganda task, the recommended pipeline outperforms 73\% of single-configuration alternatives against a human baseline. A small-sample pilot is sufficient to derive confidence intervals that approach nominal coverage and to identify which design changes yield the largest precision gains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18850v2">Cognitive Alpha Mining via LLM-Driven Code-Based Evolution</a></div>
    <div class="paper-meta">
      📅 2026-04-15
    </div>
    <details class="paper-abstract">
      Discovering effective predictive signals, or "alphas," from financial data with high dimensionality and extremely low signal-to-noise ratio remains a difficult open problem. Despite progress in deep learning, genetic programming, and, more recently, large language model (LLM)-based factor generation, existing approaches still explore only a narrow region of the vast alpha search space. Neural models tend to produce opaque and fragile patterns, while symbolic or formula-based methods often yield redundant or economically ungrounded expressions that generalize poorly. Although different in form, these paradigms share a key limitation: none can conduct broad, structured, and human-like exploration that balances logical consistency with creative leaps. To address this gap, we introduce the Cognitive Alpha Mining Framework (CogAlpha), which combines code-level alpha representation with LLM-driven reasoning and evolutionary search. Treating LLMs as adaptive cognitive agents, our framework iteratively refines, mutates, and recombines alpha candidates through multi-stage prompts and financial feedback. This synergistic design enables deeper thinking, richer structural diversity, and economically interpretable alpha discovery, while greatly expanding the effective search space. Experiments on 5 stock datasets from 3 stock markets demonstrate that CogAlpha consistently discovers alphas with superior predictive accuracy, robustness, and generalization over existing methods. Our results highlight the promise of aligning evolutionary optimization with LLM-based reasoning for automated and explainable alpha discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14473v1">Response-Aware User Memory Selection for LLM Personalization</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 Code at: https://github.com/jfisher52/Response_Utility_Optimized_Memory_Selection
    </div>
    <details class="paper-abstract">
      A common approach to personalization in large language models (LLMs) is to incorporate a subset of the user memory into the prompt at inference time to guide the model's generation. Existing methods select these subsets primarily using similarity between user memory items and input queries, ignoring how features actually affect the model's response distribution. We propose Response-Utility optimization for Memory Selection (RUMS), a novel method that selects user memory items by measuring the mutual information between a subset of memory and the model's outputs, identifying items that reduce response uncertainty and sharpen predictions beyond semantic similarity. We demonstrate that this information-theoretic foundation enables more principled user memory selection that aligns more closely with human selection compared to state-of-the-art methods, and models $400\times$ larger. Additionally, we show that memory items selected using RUMS result in better response quality compared to existing approaches, while having up to $95\%$ reduction in computational cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15915v2">AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization</a></div>
    <div class="paper-meta">
      📅 2026-04-15
    </div>
    <details class="paper-abstract">
      We present AccelOpt, a self-improving large language model (LLM) agentic system that autonomously optimizes kernels for emerging AI acclerators, eliminating the need for expert-provided hardware-specific optimization knowledge. AccelOpt explores the kernel optimization space through iterative generation, informed by an optimization memory that curates experiences and insights from previously encountered slow-fast kernel pairs. We build NKIBench, a new benchmark suite of AWS Trainium accelerator kernels with varying complexity extracted from real-world LLM workloads to evaluate the effectiveness of AccelOpt. Our evaluation confirms that AccelOpt's capability improves over time, boosting the average percentage of peak throughput from $49\%$ to $61\%$ on Trainium 1 and from $45\%$ to $59\%$ on Trainium 2 for NKIBench kernels. Moreover, AccelOpt is highly cost-effective: using open-source models, it matches the kernel improvements of Claude Sonnet 4 while being $26\times$ cheaper. The code is open-sourced at https://github.com/zhang677/AccelOpt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07663v2">SAGE: Sign-Adaptive Gradient for Memory-Efficient LLM Optimization</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 Accepted to Findings of the Association for Computational Linguistics: ACL 2026. 13 pages, 4 figures, 4 tables
    </div>
    <details class="paper-abstract">
      The AdamW optimizer, while standard for LLM pretraining, is a critical memory bottleneck, consuming optimizer states equivalent to twice the model's size. Although light-state optimizers like SinkGD attempt to address this issue, we identify the embedding layer dilemma: these methods fail to handle the sparse, high-variance gradients inherent to embeddings, forcing a hybrid design that reverts to AdamW and partially negates the memory gains. We propose SAGE (Sign Adaptive GradiEnt), a novel optimizer that resolves this dilemma by replacing AdamW in this hybrid structure. SAGE combines a Lion-style update direction with a new, memory-efficient $O(d)$ adaptive scale. This scale acts as a "safe damper," provably bounded by 1.0, which tames high-variance dimensions more effectively than existing methods. This superior stability allows SAGE to achieve better convergence. On Llama models up to 1.3B parameters, our SAGE-based hybrid achieves new state-of-the-art perplexity, outperforming all baselines, including SinkGD hybrid, while significantly reducing optimizer state memory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14437v1">LLMs taking shortcuts in test generation: A study with SAP HANA and LevelDB</a></div>
    <div class="paper-meta">
      📅 2026-04-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved impressive results on public benchmarks, often leading to claims of advanced reasoning and understanding. However, recent research in cognitive science reveals that these models sometimes rely on shallow heuristics and memorization, taking shortcuts rather than demonstrating genuine cognitive abilities. This paper investigates LLM behavior in automated test generation for software, contrasting performance on an open-source system (LevelDB) with SAP HANA, one of the most widely deployed commercial database systems worldwide, whose proprietary codebase is guaranteed to be absent from training data. We combine cognitive evaluation principles, drawing on Mitchell's mechanism-focused assessment methodology, with empirical software testing, employing mutation score and iterative compiler-feedback repair loops to assess both accuracy and underlying reasoning strategies. Results show that LLMs excel on familiar, open-source benchmarks but struggle with unseen, complex domains, often prioritizing compilability over semantic effectiveness. These findings provide independent software engineering evidence for the broader claim that current LLMs lack robust reasoning, and highlight the need for evaluation frameworks that penalize trivial shortcuts and reward true generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14414v1">The Autocorrelation Blind Spot: Why 42% of Turn-Level Findings in LLM Conversation Analysis May Be Spurious</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 14 pages, 3 figures, 5 tables, 1 algorithm. Code and synthetic demonstration data: https://github.com/ferdinandschessl-boop/autocorrelation-correction
    </div>
    <details class="paper-abstract">
      Turn-level metrics are widely used to evaluate properties of multi-turn human-LLM conversations, from safety and sycophancy to dialogue quality. However, consecutive turns within a conversation are not statistically independent -- a fact that virtually all current evaluation pipelines fail to correct for in their statistical inference. We systematically characterize the autocorrelation structure of 66 turn-level metrics across 202 multi-turn conversations (11,639 turn pairs, 5 German-speaking users, 4 LLM platforms) and demonstrate that naive pooled analysis produces severely inflated significance estimates: 42% of associations that appear significant under standard pooled testing fail to survive cluster-robust correction. The inflation varies substantially across categories rather than scaling linearly with autocorrelation: three memoryless families (embedding velocity, directional, differential) aggregate to 14%, while the seven non-memoryless families (thermo-cycle, frame distance, lexical/structural, rolling windows, cumulative, interaction, timestamp) aggregate to 33%, with individual category rates ranging from 0% to 100% depending on per-family effect size. We present a two-stage correction framework combining Chelton (1983) effective degrees of freedom with conversation-level block bootstrap, and validate it on a pre-registered hold-out split where cluster-robust metrics replicate at 57% versus 30% for pooled-only metrics. We provide concrete design principles, a publication checklist, and open-source code for the correction pipeline. A survey of ~30 recent papers at major NLP and AI venues that compute turn-level statistics in LLM evaluations finds that only 4 address temporal dependence at all, and 26 do not correct for it.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20892v3">Representation Interventions Enable Lifelong Knowledge Memory Control in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics: ACL 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often produce incorrect or outdated content after being employed. Efficient and accurate knowledge updates without costly retraining are a major challenge. This problem is particularly challenging in lifelong settings, where complex, unstructured knowledge must coexist without interference. We introduce RILKE (Representation Intervention for Lifelong KnowledgE Control), a robust and scalable method that treats knowledge control as interventions within the model's representation space. Leveraging representation-space expressiveness, we identify two key properties enabling RILKE to achieve fine-grained control over complex, unstructured knowledge while maintaining general utility with frozen base weights. During training, RILKE learns paraphrase-robust and edit-localized modules that limit each update to a low-dimensional subspace to minimize cross-edit interference. At inference, a query-adaptive router selects the appropriate module to guide the model's generation. Across LLaMA and Qwen models, RILKE scales effectively to large-scale benchmarks, demonstrating high edit success and strong paraphrase generalization while preserving general utility with modest memory overhead. These results show RILKE is an effective and scalable solution for lifelong knowledge control in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14386v1">Coalition Formation in LLM Agent Networks: Stability Analysis and Convergence Guarantees</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 15 pages including supplementary material, 2 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents are increasingly deployed in multi-agent systems requiring strategic coordination. While recent work has analyzed LLM behavior in two-player games, coalition formation, where $n$ agents dynamically form cooperative groups, remains theoretically uncharacterized. We present the first framework grounding coalition formation in LLM agent networks in hedonic game theory with formal stability guarantees. We introduce the LLM Coalition Formation Game (LCFG), establish sufficient conditions for Nash-stable partitions, and prove complexity results. Our analysis reveals that LLM agents exhibit bounded rationality characterized by $ε$-rational preferences; we provide both deterministic existence guarantees and consistency-driven stability bounds whose predictions are consistent with empirical outcomes. Experiments with GPT-4, Claude-3, and Llama-3 across 2,400 episodes validate our framework: LLM coalitions achieve Nash stability in 73.2% of cases under our Coalition-of-Thought (CoalT) protocol, compared to 58.4% under chain-of-thought and 41.8% under standard prompting ($p < 0.001$). Our framework provides theoretical foundations for designing stable multi-agent LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.13529v2">SecureGate: Learning When to Reveal PII Safely via Token-Gated Dual-Adapters for Federated LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-15
    </div>
    <details class="paper-abstract">
      Federated learning (FL) enables collaborative training across organizational silos without sharing raw data, making it attractive for privacy-sensitive applications. With the rapid adoption of large language models (LLMs), federated fine-tuning of generative LLMs has gained attention as a way to leverage distributed data while preserving confidentiality. However, this setting introduces fundamental challenges: (i) privacy leakage of personally identifiable information (PII) due to LLM memorization, and (ii) a persistent tension between global generalization and local utility under heterogeneous data. Existing defenses, such as data sanitization and differential privacy, reduce leakage but often degrade downstream performance. We propose SecureGate, a privacy-aware federated fine-tuning framework for LLMs that provides fine-grained privacy control without sacrificing utility. SecureGate employs a dual-adapter LoRA architecture: a secure adapter that learns sanitized, globally shareable representations, and a revealing adapter that captures sensitive, organization-specific knowledge. A token-controlled gating module selectively activates these adapters at inference time, enabling controlled information disclosure without retraining. Extensive experiments across multiple LLMs and real-world datasets show that SecureGate improves task utility while substantially reducing PII leakage, achieving up to a 31.66X reduction in inference attack accuracy and a 17.07X reduction in extraction recall for unauthorized requests. Additionally, it maintains 100% routing reliability to the correct adapter and incurs only minimal computational and communication overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14334v1">Mamba-SSM with LLM Reasoning for Biomarker Discovery: Causal Feature Refinement via Chain-of-Thought Gene Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 9 pages, 4 figures. Accepted at ICLR 2026 Workshop on Logical Reasoning of Large Language Models
    </div>
    <details class="paper-abstract">
      Gradient saliency from deep sequence models surfaces candidate biomarkers efficiently, but the resulting gene lists are contaminated by tissue-composition confounders that degrade downstream classifiers. We study whether LLM chain-of-thought (CoT) reasoning can faithfully filter these confounders, and whether reasoning quality drives downstream performance. We train a Mamba SSM on TCGA-BRCA RNA-seq and extract the top-50 genes by gradient saliency; DeepSeek-R1 evaluates every candidate with structured CoT to produce a final 17-gene set. The raw 50-gene saliency set (no LLM) performs worse than a 5,000-gene variance baseline (AUC 0.832 vs. 0.903), while the LLM-filtered set surpasses it (AUC 0.927), using 294x fewer features. A faithfulness audit (COSMIC CGC, OncoKB, PAM50) reveals only 6 of 17 selected genes (35.3%) are validated BRCA biomarkers, yet 10 of 16 known BRCA genes in the input were missed - including FOXA1. This gap between downstream performance and reasoning faithfulness suggests selective faithfulness: targeted confounder removal is sufficient for performance gains even without comprehensive recall.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.23853v3">Your LLM Agents are Temporally Blind: The Misalignment Between Tool Use Decisions and Human Time Perception</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 ACL 2026 (findings), Camera-ready
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents are increasingly used to interact with and execute tasks in dynamic environments. However, a critical yet overlooked limitation of these agents is that they, by default, assume a stationary context, failing to account for the real-world time elapsed between messages. We refer to this as "temporal blindness". This limitation hinders decisions about when to invoke tools, leading agents to either over-rely on stale context and skip needed tool calls, or under-rely on it and redundantly repeat tool calls. To study this challenge, we constructed TicToc, a diverse dataset of multi-turn user-agent message trajectories across 76 scenarios, spanning dynamic environments with high, medium, and low time sensitivity. We collected human preferences between "calling a tool" and "directly answering" on each sample, and evaluated how well LLM tool-calling decisions align with human preferences under varying amounts of elapsed time. Our analysis reveals that existing models display poor alignment with human temporal perception, with no model achieving a normalized alignment rate better than 65% when given time stamp information. We also show that naive, prompt-based alignment techniques have limited effectiveness for most models, but specific post-training alignment can be a viable way to align multi-turn LLM tool use with human temporal perception. Our data and findings provide a first step toward understanding and mitigating temporal blindness, offering insights to foster the development of more time-aware and human-aligned agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14325v1">Faithfulness Serum: Mitigating the Faithfulness Gap in Textual Explanations of LLM Decisions via Attribution Guidance</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 24 pages, multiple figures (e.g., at least 6 main figures), includes experiments across several benchmarks (MMLU, CommonsenseQA, SciQ, ARC, OpenBookQA); code available on GitHub
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) achieve strong performance and have revolutionized NLP, but their lack of explainability keeps them treated as black boxes, limiting their use in domains that demand transparency and trust. A promising direction to address this issue is post-hoc text-based explanations, which aim to explain model decisions in natural language. Prior work has focused on generating convincing rationales that appear to be subjectively faithful, but it remains unclear whether these explanations are epistemically faithful, whether they reflect the internal evidence the model actually relied on for its decision. In this paper, we first assess the epistemic faithfulness of LLM-generated explanations via counterfactuals and show that they are often unfaithful. We then introduce a training-free method that enhances faithfulness by guiding explanation generation through attention-level interventions, informed by token-level heatmaps extracted via a faithful attribution method. This method significantly improves epistemic faithfulness across multiple models, benchmarks, and prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.02585v2">Mitigating LLM biases toward spurious social contexts using direct preference optimization</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 26 pages
    </div>
    <details class="paper-abstract">
      LLMs are increasingly used for high-stakes decision-making, yet their sensitivity to spurious contextual information can introduce harmful biases. This is a critical concern when models are deployed for tasks like evaluating teachers' instructional quality, where biased assessment can affect teachers' professional development and career trajectories. We investigate model robustness to spurious social contexts using the largest publicly available dataset of U.S. classroom transcripts (NCTE) paired with expert rubric scores. Evaluating seven frontier and open-weight models across seven categories of spurious contexts -- including teacher experience, education level, demographic identity, and sycophancy-inducing framings -- we find that irrelevant contextual information can shift model predictions by up to 1.48 points on a 7-point scale, with larger models sometimes exhibiting greater sensitivity despite higher predictive accuracy. Mitigations using prompts and standard direct preference optimization (DPO) prove largely insufficient. We propose **Debiasing-DPO**,, a self-supervised training method that pairs neutral reasoning generated from the query alone, with the model's biased reasoning generated with both the query and additional spurious context. We further combine this objective with supervised fine-tuning on ground-truth labels to prevent losses in predictive accuracy. Applied to Llama 3B \& 8B and Qwen 3B \& 7B Instruct models, Debiasing-DPO reduces bias by 84\% and improves predictive accuracy by 52\% on average. Our findings from the educational case study highlight that robustness to spurious context is not a natural byproduct of model scaling and that our proposed method can yield substantial gains in both accuracy and robustness for prompt-based prediction tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14128v1">Rhetorical Questions in LLM Representations: A Linear Probing Study</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 18 pages, 15 figures, accepted to ACL 2026
    </div>
    <details class="paper-abstract">
      Rhetorical questions are asked not to seek information but to persuade or signal stance. How large language models internally represent them remains unclear. We analyze rhetorical questions in LLM representations using linear probes on two social-media datasets with different discourse contexts, and find that rhetorical signals emerge early and are most stably captured by last-token representations. Rhetorical questions are linearly separable from information-seeking questions within datasets, and remain detectable under cross-dataset transfer, reaching AUROC around 0.7-0.8. However, we demonstrate that transferability does not simply imply a shared representation. Probes trained on different datasets produce different rankings when applied to the same target corpus, with overlap among the top-ranked instances often below 0.2. Qualitative analysis shows that these divergences correspond to distinct rhetorical phenomena: some probes capture discourse-level rhetorical stance embedded in extended argumentation, while others emphasize localized, syntax-driven interrogative acts. Together, these findings suggest that rhetorical questions in LLM representations are encoded by multiple linear directions emphasizing different cues, rather than a single shared direction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14116v1">TREX: Automating LLM Fine-tuning via Agent-Driven Tree-based Exploration</a></div>
    <div class="paper-meta">
      📅 2026-04-15
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have empowered AI research agents to perform isolated scientific tasks, automating complex, real-world workflows, such as LLM training, remains a significant challenge. In this paper, we introduce TREX, a multi-agent system that automates the entire LLM training life-cycle. By orchestrating collaboration between two core modules-the Researcher and the Executor-the system seamlessly performs requirement analysis, open-domain literature and data research, formulation of training strategies, preparation of data recipes, and model training and evaluation. The multi-round experimental process is modeled as a search tree, enabling the system to efficiently plan exploration paths, reuse historical results, and distill high-level insights from iterative trials. To evaluate the capability of automated LLM training, we construct FT-Bench, a benchmark comprising 10 tasks derived from real-world scenarios, ranging from optimizing fundamental model capabilities to enhancing performance on domain-specific tasks. Experimental results demonstrate that the TREX agent consistently optimizes model performance on target tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14267v1">Enhancing LLM-based Search Agents via Contribution Weighted Group Relative Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 Accepted to the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026), Main Conference
    </div>
    <details class="paper-abstract">
      Search agents extend Large Language Models (LLMs) beyond static parametric knowledge by enabling access to up-to-date and long-tail information unavailable during pretraining. While reinforcement learning has been widely adopted for training such agents, existing approaches face key limitations: process supervision often suffers from unstable value estimation, whereas outcome supervision struggles with credit assignment due to sparse, trajectory-level rewards. To bridge this gap, we propose Contribution-Weighted GRPO (CW-GRPO), a framework that integrates process supervision into group relative policy optimization. Instead of directly optimizing process rewards, CW-GRPO employs an LLM judge to assess the retrieval utility and reasoning correctness at each search round, producing per-round contribution scores. These scores are used to rescale outcome-based advantages along the trajectory, enabling fine-grained credit assignment without sacrificing optimization stability. Experiments on multiple knowledge-intensive benchmarks show that CW-GRPO outperforms standard GRPO by 5.0\% on Qwen3-8B and 6.3\% on Qwen3-1.7B, leading to more effective search behaviors. Additional analysis reveals that successful trajectories exhibit concentrated contributions across rounds, providing empirical insight into search agent tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14111v1">Interpretable Stylistic Variation in Human and LLM Writing Across Genres, Models, and Decoding Strategies</a></div>
    <div class="paper-meta">
      📅 2026-04-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are now capable of generating highly fluent, human-like text. They enable many applications, but also raise concerns such as large scale spam, phishing, or academic misuse. While much work has focused on detecting LLM-generated text, only limited work has gone into understanding the stylistic differences between human-written and machine-generated text. In this work, we perform a large scale analysis of stylistic variation across human-written text and outputs from 11 LLMs spanning 8 different genres and 4 decoding strategies using Douglas Biber's set of lexicogrammatical and functional features. Our findings reveal insights that can guide intentional LLM usage. First, key linguistic differentiators of LLM-generated text seem robust to generation conditions (e.g., prompt settings to nudge them to generate human-like text, or availability of human-written text to continue the style); second, genre exerts a stronger influence on stylistic features than the source itself; third, chat variants of the models generally appear to be clustered together in stylistic space, and finally, model has a larger effect on the style than decoding strategy, with some exceptions. These results highlight the relative importance of model and genre over prompting and decoding strategies in shaping the stylistic behavior of machine-generated text.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.10015v2">FinTrace: Holistic Trajectory-Level Evaluation of LLM Tool Calling for Long-Horizon Financial Tasks</a></div>
    <div class="paper-meta">
      📅 2026-04-15
    </div>
    <details class="paper-abstract">
      Recent studies demonstrate that tool-calling capability enables large language models (LLMs) to interact with external environments for long-horizon financial tasks. While existing benchmarks have begun evaluating financial tool calling, they focus on limited scenarios and rely on call-level metrics that fail to capture trajectory-level reasoning quality. To address this gap, we introduce FinTrace, a benchmark comprising 800 expert-annotated trajectories spanning 34 real-world financial task categories across multiple difficulty levels. FinTrace employs a rubric-based evaluation protocol with nine metrics organized along four axes -- action correctness, execution efficiency, process quality, and output quality -- enabling fine-grained assessment of LLM tool-calling behavior. Our evaluation of 13 LLMs reveals that while frontier models achieve strong tool selection, all models struggle with information utilization and final answer quality, exposing a critical gap between invoking the right tools and reasoning effectively over their outputs. To move beyond diagnosis, we construct FinTrace-Training, the first trajectory-level preference dataset for financial tool-calling, containing 8,196 curated trajectories with tool-augmented contexts and preference pairs. We fine-tune Qwen-3.5-9B using supervised fine-tuning followed by direct preference optimization (DPO) and show that training on FinTrace-Training consistently improves intermediate reasoning metrics, with DPO more effectively suppressing failure modes. However, end-to-end answer quality remains a bottleneck, indicating that trajectory-level improvements do not yet fully propagate to final output quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03027v3">Reducing Hallucinations in LLMs via Factuality-Aware Preference Learning</a></div>
    <div class="paper-meta">
      📅 2026-04-15
    </div>
    <details class="paper-abstract">
      Preference alignment methods such as RLHF and Direct Preference Optimization (DPO) improve instruction following, but they can also reinforce hallucinations when preference judgments reward fluency and confidence over factual correctness. We introduce F-DPO (Factuality-aware Direct Preference Optimization), a simple extension of DPO that uses only binary factuality labels. F-DPO (i) applies a label-flipping transformation that corrects misordered preference pairs so the chosen response is never less factual than the rejected one, and (ii) adds a factuality-aware margin that emphasizes pairs with clear correctness differences, while reducing to standard DPO when both responses share the same factuality. We construct factuality-aware preference data by augmenting DPO pairs with binary factuality indicators and synthetic hallucinated variants. Across seven open-weight LLMs (1B-14B), F-DPO consistently improves factuality and reduces hallucination rates relative to both base models and standard DPO. On Qwen3-8B, F-DPO reduces hallucination rates by 5x(from 0.424 to 0.084) while improving factuality scores by 50% (from 5.26 to 7.90). F-DPO also generalizes to out-of-distribution benchmarks: on TruthfulQA, Qwen2.5-14B achieves +17% MC1 accuracy (0.500 to 0.585) and +49% MC2 accuracy (0.357 to 0.531). F-DPO requires no auxiliary reward model, token-level annotations, or multi-stage training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13997v1">Learned or Memorized ? Quantifying Memorization Advantage in Code LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 12 pages, 12 figures. Accepted at ICSE 2026 (to appear)
    </div>
    <details class="paper-abstract">
      The lack of transparency about code datasets used to train large language models (LLMs) makes it difficult to detect, evaluate, and mitigate data leakage. We present a perturbation-based method to quantify memorization advantage in code LLMs, defined as the performance gap between likely seen and unseen inputs. We evaluate 8 open-source code LLMs on 19 benchmarks across four task families: code generation, code understanding, vulnerability detection, and bug fixing. Sensitivity patterns vary widely across models and tasks. For example, StarCoder reaches high sensitivity on some benchmarks (up to 0.8), while QwenCoder remains lower (mostly below 0.4), suggesting differences in generalization behavior. Task categories also differ: code summarization tends to show low sensitivity, whereas test generation is substantially higher. We then analyze two widely discussed benchmarks, CVEFixes and Defects4J, often suspected of leakage. Contrary to common concerns, both show low memorization advantage across models: CVEFixes remains below 0.1, and Defects4J is lower than other program repair benchmarks. These results suggest that, for these datasets, models may rely more on learned generalization than direct memorization. Overall, our findings provide evidence that memorization risk is highly task- and model-dependent, and highlight the need for stronger evaluation protocols, especially in security-focused settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.03826v2">Context Matters: Evaluating Context Strategies for Automated ADR Generation Using LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 11 pages, 5 diagrams, Accepted at EASE Conference 2026 Research Track
    </div>
    <details class="paper-abstract">
      Architecture Decision Records (ADRs) play a critical role in preserving the rationale behind system design, yet their creation and maintenance are often neglected due to the associated authoring overhead. This paper investigates whether Large Language Models (LLMs) can mitigate this burden and, more importantly, how different strategies for presenting historical ADRs as context influence generation quality. We curate and validate a large corpus of sequential ADRs drawn from 750 open-source repositories and systematically evaluate five context selection strategies (no context, All-history, First-K, Last-K, and RAFG) across multiple model families. Our results show that context-aware prompting substantially improves ADR generation fidelity, with a small recency window (typically 3-5 prior records) providing the best balance between quality and efficiency. Retrieval-based context selection yields marginal gains primarily in non-sequential or cross-cutting decision scenarios, while offering no statistically significant advantage in typical linear ADR workflows. Overall, our findings demonstrate that context engineering, rather than model scale alone, is the dominant factor in effective ADR automation, and we outline practical defaults for tool builders along with targeted retrieval fallbacks for complex architectural settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13979v1">Leveraging LLM-GNN Integration for Open-World Question Answering over Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 18 pages,6 figures,10 tables. https://aclanthology.org/2026.eacl-long.26/
    </div>
    <details class="paper-abstract">
      Open-world Question Answering (OW-QA) over knowledge graphs (KGs) aims to answer questions over incomplete or evolving KGs. Traditional KGQA assumes a closed world where answers must exist in the KG, limiting real-world applicability. In contrast, open-world QA requires inferring missing knowledge based on graph structure and context. Large language models (LLMs) excel at language understanding but lack structured reasoning. Graph neural networks (GNNs) model graph topology but struggle with semantic interpretation. Existing systems integrate LLMs with GNNs or graph retrievers. Some support open-world QA but rely on structural embeddings without semantic grounding. Most assume observed paths or complete graphs, making them unreliable under missing links or multi-hop reasoning. We present GLOW, a hybrid system that combines a pre-trained GNN and an LLM for open-world KGQA. The GNN predicts top-k candidate answers from the graph structure. These, along with relevant KG facts, are serialized into a structured prompt (e.g., triples and candidates) to guide the LLM's reasoning. This enables joint reasoning over symbolic and semantic signals, without relying on retrieval or fine-tuning. To evaluate generalization, we introduce GLOW-BENCH, a 1,000-question benchmark over incomplete KGs across diverse domains. GLOW outperforms existing LLM-GNN systems on standard benchmarks and GLOW-BENCH, achieving up to 53.3% and an average 38% improvement. GitHub code and data are available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13955v1">Towards Personalizing Secure Programming Education with LLM-Injected Vulnerabilities</a></div>
    <div class="paper-meta">
      📅 2026-04-15
    </div>
    <details class="paper-abstract">
      According to constructivist theory, students learn software security more effectively when examples are grounded in their own code. Generic examples often fail to connect with students' prior work, limiting engagement and understanding. Advances in LLMs are now making it possible to automatically generate personalized examples by embedding security vulnerabilities directly into student-authored code. This paper introduces a method that uses LLMs to inject instances of specific Common Weakness Enumerations (CWEs) into students' own assignment code, creating individualized instructional materials. We present an agentic AI framework, using autonomous LLM-based agents equipped with task-specific tools to orchestrate injection, evaluation, ranking, and learning outcome generation. We report the experience of deploying this system in two undergraduate computer science courses (N=71), where students reviewed code samples containing LLM-injected vulnerabilities and completed a post-project survey. We compared responses with a baseline using a widely adopted set of generic security instructional materials. Students qualitatively reported finding CWE injections into their own code more relevant, clearer, and more engaging than the textbook-style examples. However, our quantitative findings revealed limited statistically significant differences, suggesting that while students valued the personalization, further studies and refinement of the approach are needed to establish stronger empirical support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13899v1">Do We Still Need Humans in the Loop? Comparing Human and LLM Annotation in Active Learning for Hostility Detection</a></div>
    <div class="paper-meta">
      📅 2026-04-15
    </div>
    <details class="paper-abstract">
      Instruction-tuned LLMs can annotate thousands of instances from a short prompt at negligible cost. This raises two questions for active learning (AL): can LLM labels replace human labels within the AL loop, and does AL remain necessary when entire corpora can be labelled at once? We investigate both questions on a new dataset of 277,902 German political TikTok comments (25,974 LLM-labelled, 5,000 human-annotated), comparing seven annotation strategies across four encoders to detect anti-immigrant hostility. A classifier trained on 25,974 GPT-5.2 labels (\$43) achieves comparable F1-Macro to one trained on 3,800 human annotations (\$316). Active learning offers little advantage over random sampling in our pre-enriched pool and delivers lower F1 than full LLM annotation at the same cost. However, comparable aggregate F1 masks a systematic difference in error structure: LLM-trained classifiers over-predict the positive class relative to the human gold standard. This divergence concentrates in topically ambiguous discussions where the distinction between anti-immigrant hostility and policy critique is most subtle, suggesting that annotation strategy should be guided not by aggregate F1 alone but by the error profile acceptable for the target application.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04844v2">Mitigating Catastrophic Forgetting in Target Language Adaptation of LLMs via Source-Shielded Updates</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 Accepted to ACL 2026 Main Conference
    </div>
    <details class="paper-abstract">
      Expanding the linguistic diversity of instruct large language models (LLMs) is crucial for global accessibility but is often hindered by the reliance on costly specialized target language labeled data and catastrophic forgetting during adaptation. We tackle this challenge under a realistic, low-resource constraint: adapting instruct LLMs using only unlabeled target language data. We introduce Source-Shielded Updates (SSU), a selective parameter update strategy that proactively preserves source knowledge. Using a small set of source data and a parameter importance scoring method, SSU identifies parameters critical to maintaining source abilities. It then applies a column-wise freezing strategy to protect these parameters before adaptation. Experiments across five typologically diverse languages and 7B and 13B models demonstrate that SSU successfully mitigates catastrophic forgetting. It reduces performance degradation on monolingual source tasks to just 3.4% (7B) and 2.8% (13B) on average, a stark contrast to the 20.3% and 22.3% from full fine-tuning. SSU also achieves target-language performance highly competitive with full fine-tuning, outperforming it on all benchmarks for 7B models and the majority for 13B models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.11312v2">Network Effects and Agreement Drift in LLM Debates</a></div>
    <div class="paper-meta">
      📅 2026-04-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated an unprecedented ability to simulate human-like social behaviors, making them useful tools for simulating complex social systems. However, it remains unclear to what extent these simulations can be trusted to accurately capture key social mechanisms, particularly in highly unbalanced contexts involving minority groups. This paper uses a network generation model with controlled homophily and class sizes to examine how LLM agents behave collectively in multi-round debates. Moreover, our findings highlight a particular directional susceptibility that we term \textit{agreement drift}, in which agents are more likely to shift toward specific positions on the opinion scale. Overall, our findings highlight the need to disentangle structural effects from model biases before treating LLM populations as behavioral proxies for human groups.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.23636v3">FlexGuard: Continuous Risk Scoring for Strictness-Adaptive LLM Content Moderation</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 Accepted at ACL 2026
    </div>
    <details class="paper-abstract">
      Ensuring the safety of LLM-generated content is essential for real-world deployment. Most existing guardrail models formulate moderation as a fixed binary classification task, implicitly assuming a fixed definition of harmfulness. In practice, enforcement strictness - how conservatively harmfulness is defined and enforced - varies across platforms and evolves over time, making binary moderators brittle under shifting requirements. We first introduce FlexBench, a strictness-adaptive LLM moderation benchmark that enables controlled evaluation under multiple strictness regimes. Experiments on FlexBench reveal substantial cross-strictness inconsistency in existing moderators: models that perform well under one regime can degrade substantially under others, limiting their practical usability. To address this, we propose FlexGuard, an LLM-based moderator that outputs a calibrated continuous risk score reflecting risk severity and supports strictness-specific decisions via thresholding. We train FlexGuard via risk-alignment optimization to improve score-severity consistency and provide practical threshold selection strategies to adapt to target strictness at deployment. Experiments on FlexBench and public benchmarks demonstrate that FlexGuard achieves higher moderation accuracy and substantially improved robustness under varying strictness. We release the source code and data to support reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.24160v2">Towards Automated Crowdsourced Testing via Personified-LLM</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 Research Paper Accepted by the ACM International Conference on the Foundations of Software Engineering (FSE 2026)
    </div>
    <details class="paper-abstract">
      The rapid proliferation and increasing complexity of software demand robust quality assurance, with graphical user interface (GUI) testing playing a pivotal role. Crowdsourced testing has proven effective in this context by leveraging the diversity of human testers to achieve rich, scenario-based coverage across varied devices, user behaviors, and usage environments. In parallel, automated testing, particularly with the advent of large language models (LLMs), offers significant advantages in controllability, reproducibility, and efficiency, enabling scalable and systematic exploration. However, automated approaches often lack the behavioral diversity characteristic of human testers, limiting their capability to fully simulate real-world testing dynamics. To address this gap, we present PersonaTester, a novel personified-LLM-based framework designed to automate crowdsourced GUI testing. By injecting representative personas, defined along three orthogonal dimensions: testing mindset, exploration strategy, and interaction habit, into LLM-based agents, PersonaTester enables the simulation of diverse human-like testing behaviors in a controllable and repeatable manner. Experimental results demonstrate that PersonaTester faithfully reproduces the behavioral patterns of real crowdworkers, exhibiting strong intra-persona consistency and clear inter-persona variability (117.86% -- 126.23% improvement over the baseline). Moreover, persona-guided testing agents consistently generate more effective test events and trigger more crashes (100+) and functional bugs (11) than the baseline without persona, thus substantially advancing the realism and effectiveness of automated crowdsourced GUI testing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13764v1">RealVuln: Benchmarking Rule-Based, General-Purpose LLM, and Security-Specialized Scanners on Real-World Code</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 16 pages, 2 figures, 4 tables. Code and data: https://github.com/kolega-ai/Real-Vuln-Benchmark. Dashboard: https://realvuln.kolega.dev/
    </div>
    <details class="paper-abstract">
      How do security scanners perform on real-world code? We present RealVuln, the first open-source benchmark comparing Rule-Based SAST, General-Purpose LLMs, and Security-Specialized scanners on 26 intentionally vulnerable Python repositories (educational and Capture-The-Flag applications) with 796 hand-labeled entries (676 vulnerabilities, 120 false-positive traps). We test 15 scanners (3 Rule-Based SAST, 10 General-Purpose LLM, 2 Security-Specialized) and rank them by F3 score (beta=3, weighting recall 9x over precision). A clear three-tier ranking emerges under all metrics. Under F3, the Security-Specialized scanner Kolega.Dev (73.0) leads, followed by the best General-Purpose LLM, Claude Sonnet 4.6 (51.7), which in turn scores nearly 3x higher than the best Rule-Based tool, Semgrep (17.7). Under F1, Sonnet 4.6 leads (60.9) with Kolega.Dev at 52.4. Rankings within tiers shift with beta, but the three-tier hierarchy holds across all weightings. All code, ground-truth data, scanner outputs, and scoring scripts are released under an open-source license. An interactive dashboard is at https://realvuln.kolega.dev/. RealVuln is a living benchmark: versioned, community-driven, with a roadmap toward multi-language coverage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13759v1">The cognitive companion: a lightweight parallel monitoring architecture for detecting and recovering from reasoning degradation in LLM agents</a></div>
    <div class="paper-meta">
      📅 2026-04-15
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents on multi-step tasks suffer reasoning degradation, looping, drift, stuck states, at rates up to 30% on hard tasks. Current solutions include hard step limits (abrupt) or LLM-as-judge monitoring (10-15% overhead per step). This paper introduces the Cognitive Companion, a parallel monitoring architecture with two implementations: an LLM-based Companion and a novel zero-overhead Probe-based Companion. We report a three-batch feasibility study centered on Gemma 4 E4B, with an additional exploratory small-model analysis on Qwen 2.5 1.5B and Llama 3.2 1B. In our experiments, the LLM-based Companion reduced repetition on loop-prone tasks by 52-62% with approximately 11% overhead. The Probe-based Companion, trained on hidden states from layer 28, showed a mean effect size of +0.471 at zero measured inference overhead; its strongest probe result achieved cross-validated AUROC 0.840 on a small proxy-labeled dataset. A key empirical finding is that companion benefit appears task-type dependent: companions are most helpful on loop-prone and open-ended tasks, while effects are neutral or negative on more structured tasks. Our small-model experiments also suggest a possible scale boundary: companions did not improve the measured quality proxy on 1B-1.5B models, even when interventions fired. Overall, the paper should be read as a feasibility study rather than a definitive validation. The results provide encouraging evidence that sub-token monitoring may be useful, identify task-type sensitivity as a practical design constraint, and motivate selective companion activation as a promising direction for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.00222v5">RL-PLUS: Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 Accepted to ACL 2026 (main)
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Reward (RLVR) has significantly advanced the complex reasoning abilities of Large Language Models (LLMs). However, it struggles to break through the inherent capability boundaries of the base LLM, due to its essentially on-policy strategy coupled with LLM's immense action space and sparse reward. Critically, RLVR can lead to the capability boundary collapse, narrowing the LLM's problem-solving scope. To address this problem, we propose RL-PLUS, a novel hybrid-policy optimization approach for LLMs that synergizes internal exploitation with external data to achieve stronger reasoning capabilities and surpass the boundaries of base models. RL-PLUS integrates two core components, i.e., Multiple Importance Sampling to address distributional mismatch from external data, and Exploration-Based Advantage Function to guide the model towards high-value, unexplored reasoning paths. We provide both theoretical analysis and extensive experiments to demonstrate the superiority and generalizability of our approach. Compared with existing RLVR methods, RL-PLUS achieves 1) state-of-the-art performance on six math reasoning benchmarks; 2) superior performance on six out-of-distribution reasoning tasks; 3) consistent and significant gains across diverse model families, with average relative improvements up to 69.2\%. Moreover, the analysis of Pass@k curves indicates that RL-PLUS effectively resolves the capability boundary collapse problem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13717v1">An Empirical Investigation of Practical LLM-as-a-Judge Improvement Techniques on RewardBench 2</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 22 pages, 10 figures
    </div>
    <details class="paper-abstract">
      LLM-as-a-judge, using a language model to score or rank candidate responses, is widely used as a scalable alternative to human evaluation in RLHF pipelines, benchmarking, and application layer evaluations (evals). However, judgment reliability depends heavily on prompting and aggregation strategy. We present an empirical investigation of practical, drop-in techniques that improve GPT-5.4 judge accuracy on RewardBench 2 without any finetuning. Two techniques account for nearly all available gains: task-specific criteria injection (+3.0pp at negligible cost) and ensemble scoring (+9.8pp at 5x cost). Combined, they reach 83.6% accuracy, +11.9pp over the 71.7% baseline. Our investigation also covers three further techniques (calibration context, adaptive model escalation, and soft blending) which did not reliably improve on criteria + ensembling at comparable cost. Cheaper model tiers benefit disproportionately from ensembling: GPT-5.4 mini with k=8 achieves 79.2% at 1.2x baseline cost, and GPT-5.4 nano with k=8 reaches 71.4% at 0.4x baseline cost, making high-accuracy LLM judges accessible at low cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13630v1">SafeHarness: Lifecycle-Integrated Security Architecture for LLM-based Agent Deployment</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 26 pages, 6 figures
    </div>
    <details class="paper-abstract">
      The performance of large language model (LLM) agents depends critically on the execution harness, the system layer that orchestrates tool use, context management, and state persistence. Yet this same architectural centrality makes the harness a high-value attack surface: a single compromise at the harness level can cascade through the entire execution pipeline. We observe that existing security approaches suffer from structural mismatch, leaving them blind to harness-internal state and unable to coordinate across the different phases of agent operation. In this paper, we introduce \safeharness{}, a security architecture in which four proposed defense layers are woven directly into the agent lifecycle to address above significant limitations: adversarial context filtering at input processing, tiered causal verification at decision making, privilege-separated tool control at action execution, and safe rollback with adaptive degradation at state update. The proposed cross-layer mechanisms tie these layers together, escalating verification rigor, triggering rollbacks, and tightening tool privileges whenever sustained anomalies are detected. We evaluate \safeharness{} on benchmark datasets across diverse harness configurations, comparing against four security baselines under five attack scenarios spanning six threat categories. Compared to the unprotected baseline, \safeharness{} achieves an average reduction of approximately 38\% in UBR and 42\% in ASR, substantially lowering both the unsafe behavior rate and the attack success rate while preserving core task utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22795v2">Sparse or Dense? A Mechanistic Estimation of Computation Density in Transformer-based LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 We have detected an error in the code used for the experiment. Most of the results in sections 4 and 5 are significantly affected. A new and corrected version will be available soon. For further information, please contact the first author
    </div>
    <details class="paper-abstract">
      Transformer-based large language models (LLMs) are comprised of billions of parameters arranged in deep and wide computational graphs. Several studies on LLM efficiency optimization argue that it is possible to prune a significant portion of the parameters, while only marginally impacting performance. This suggests that the computation is not uniformly distributed across the parameters. We introduce here a technique to systematically quantify computation density in LLMs. In particular, we design a density estimator drawing on mechanistic interpretability. We experimentally test our estimator and find that: (1) contrary to what has been often assumed, LLM processing generally involves dense computation; (2) computation density is dynamic, in the sense that models shift between sparse and dense processing regimes depending on the input; (3) per-input density is significantly correlated across LLMs, suggesting that the same inputs trigger either low or high density. Investigating the factors influencing density, we observe that predicting rarer tokens requires higher density, and increasing context length often decreases the density. We believe that our computation density estimator will contribute to a better understanding of the processing at work in LLMs, challenging their symbolic interpretation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.18454v2">Sandwich: Joint Configuration Search and Hot-Switching for Efficient CPU LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 DAC '26
    </div>
    <details class="paper-abstract">
      CPUs are critical for LLM serving due to their availability, cost efficiency, and edge applicability. However, efficient CPU serving is hindered by conflicting prefill/decode resource demands under non-disaggregated deployment constraints--existing solutions fail to avoid cross-phase interference, ignore sub-NUMA hardware structures, and deliver suboptimal dynamic-shape kernel performance. We propose Sandwich, a full-stack CPU LLM serving system with three core innovations addressing these challenges: (1) seamless phase-wise plan switching to eliminate cross-phase interference; (2) TopoTree, a tree-based hardware abstraction for automated substructure-aware (e.g., LLC slices) partial core allocation; (3) fast-start-then-finetune dynamic-shape tensor program generation. Across five x86/ARM CPU platforms, Sandwich achieves an average 2.01x end-to-end speedup and up to 3.40x latency reduction over state-of-the-art systems. Its kernels match static compiler performance with three orders of magnitude lower tuning cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13556v1">YOCO++: Enhancing YOCO with KV Residual Connections for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-04-15
    </div>
    <details class="paper-abstract">
      Cross-layer key-value (KV) compression has been found to be effective in efficient inference of large language models (LLMs). Although they reduce the memory consumption of the KV cache, such methods usually introduce non-negligible performance degradation. In this work, we aim to enhance the performance of YOCO, a cross-layer KV compression method that shares the KVs of the middle layer with the top-half layers. We propose YOCO++, an enhanced YOCO that incorporates a weighted residual connection between the KVs of each bottom-half layer and the bottom layer. Compared to YOCO, YOCO++ increases model capacity while maintaining the same training and inference efficiency. Our experiments show that YOCO++ achieves state-of-the-art performance among the cross-layer KV compression methods at a 50% KV cache compression rate, outperforming the standard Transformer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13537v1">Racing to Release: Priority, Congestion, and Community Recognition in Open-Source LLM Ecosystems</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 5 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Open-source large language models have made platforms such as Hugging Face central hubs for decentralized AI innovation. Yet these ecosystems are shaped not only by collaboration, but also by competition for priority and community attention. Drawing on Hill and Stein's Race-to-the-Bottom framework, this study extends the logic of project potential, maturation, competition, and quality from scientific production to open-source LLM ecosystems, where prominent base models attract concentrated derivative entry under rapid and highly visible platform feedback. Using a large-scale sample of derivative models on Hugging Face, we find that later releases and more crowded competitive environments are both associated with weaker community recognition, even after accounting for differences in model and ecosystem prominence. These findings suggest that competition for priority remains an important organizing force in open-source LLM ecosystems, shaping which derivative innovations receive community recognition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.09176v2">LIVE-GS: LLM Powers Interactive VR Experience with Physics-Aware Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-15
    </div>
    <details class="paper-abstract">
      As 3D Gaussian Splatting (3DGS) emerges as a leading approach for novel view synthesis and scene reconstruction, its potential in digital asset creation has gained significant attention. An increasing number of asset libraries based on GS are being established. However, generating physics-based dynamic assets remains a time-consuming and expertise-intensive task, especially for non-experts. In this paper, we propose LIVE-GS, a highly realistic Virtual Reality (VR) system powered by Large Language Models (LLMs), which enables rapid creation of dynamic Gaussian assets and real-time VR interactions. To inform our system design, we conducted interviews to examine challenges faced by current GS-based VR systems and the specific demands of users. Based on these insights, we employed GPT-4o to analyze key physical properties of objects that significantly impact user interactions, ensuring physics-based interactions in VR align with real-world phenomena. A key innovation of LIVE-GS is its ability to predict reasonable parameters in just 10 seconds from static Gaussian assets while maintaining high-quality VR interactions. To validate our approach, we invited participants experienced in physical simulation to manually adjust physical parameters, providing a baseline for comparison in both asset quality and authoring efficiency. We also conducted a comprehensive user study to evaluate system usability and user satisfaction. Experimental results demonstrate that LIVE-GS, leveraging LLMs' scene understanding capabilities, can achieve efficient physical scene creation and natural interactions without requiring manual design or annotation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02902v2">Logical Phase Transitions: Understanding Collapse in LLM Logical Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 Accepted at ACL 2026 (Main Conference)
    </div>
    <details class="paper-abstract">
      Symbolic logical reasoning is a critical yet underexplored capability of large language models (LLMs), providing reliable and verifiable decision-making in high-stakes domains such as mathematical reasoning and legal judgment. In this study, we present a systematic analysis of logical reasoning under controlled increases in logical complexity, and reveal a previously unrecognized phenomenon, which we term Logical Phase Transitions: rather than degrading smoothly, logical reasoning performance remains stable within a regime but collapses abruptly beyond a critical logical depth, mirroring physical phase transitions such as water freezing beyond a critical temperature threshold. Building on this insight, we propose Neuro-Symbolic Curriculum Tuning, a principled framework that adaptively aligns natural language with logical symbols to establish a shared representation, and reshapes training dynamics around phase-transition boundaries to progressively strengthen reasoning at increasing logical depths. Experiments on five benchmarks show that our approach effectively mitigates logical reasoning collapse at high complexity, yielding average accuracy gains of +1.26 in naive prompting and +3.95 in CoT, while improving generalization to unseen logical compositions. Code and data are available at https://github.com/AI4SS/Logical-Phase-Transitions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13502v1">Using reasoning LLMs to extract SDOH events from clinical notes</a></div>
    <div class="paper-meta">
      📅 2026-04-15
    </div>
    <details class="paper-abstract">
      Social Determinants of Health (SDOH) refer to environmental, behavioral, and social conditions that influence how individuals live, work, and age. SDOH have a significant impact on personal health outcomes, and their systematic identification and management can yield substantial improvements in patient care. However, SDOH information is predominantly captured in unstructured clinical notes within electronic health records, which limits its direct use as machine-readable entities. To address this issue, researchers have employed Natural Language Processing (NLP) techniques using pre-trained BERT-based models, demonstrating promising performance but requiring sophisticated implementation and extensive computational resources. In this study, we investigated prompt engineering strategies for extracting structured SDOH events utilizing LLMs with advanced reasoning capabilities. Our method consisted of four modules: 1) developing concise and descriptive prompts integrated with established guidelines, 2) applying few-shot learning with carefully curated examples, 3) using a self-consistency mechanism to ensure robust outputs, and 4) post-processing for quality control. Our approach achieved a micro-F1 score of 0.866, demonstrating competitive performance compared to the leading models. The results demonstrated that LLMs with reasoning capabilities are effective solutions for SDOH event extraction, offering both implementation simplicity and strong performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07422v2">Two Pathways to Truthfulness: On the Intrinsic Encoding of LLM Hallucinations</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 Accepted to the ACL 2026 Main Conference
    </div>
    <details class="paper-abstract">
      Despite their impressive capabilities, large language models (LLMs) frequently generate hallucinations. Previous work shows that their internal states encode rich signals of truthfulness, yet the origins and mechanisms of these signals remain unclear. In this paper, we demonstrate that truthfulness cues arise from two distinct information pathways: (1) a Question-Anchored pathway that depends on question-answer information flow, and (2) an Answer-Anchored pathway that derives self-contained evidence from the generated answer itself. First, we validate and disentangle these pathways through attention knockout and token patching. Afterwards, we uncover notable and intriguing properties of these two mechanisms. Further experiments reveal that (1) the two mechanisms are closely associated with LLM knowledge boundaries; and (2) internal representations are aware of their distinctions. Finally, building on these insightful findings, two applications are proposed to enhance hallucination detection performance. Overall, our work provides new insight into how LLMs internally encode truthfulness, offering directions for more reliable and self-aware generative systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.00570v3">SPRINT: Scalable and Predictive Intent Refinement for LLM-Enhanced Session-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 SIGIR'26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have enhanced conventional recommendation models via user profiling, which generates representative textual profiles from users' historical interactions. However, their direct application to session-based recommendation (SBR) remains challenging due to severe session context scarcity and poor scalability. In this paper, we propose SPRINT, a scalable SBR framework that incorporates reliable and informative intents while ensuring high efficiency in both training and inference. SPRINT constrains LLM-based profiling with a global intent pool and validates inferred intents based on recommendation performance to mitigate noise and hallucinations under limited context. To ensure scalability, LLMs are selectively invoked only for uncertain sessions during training, while a lightweight intent predictor generalizes intent prediction to all sessions without LLM dependency at inference time. Experiments on real-world datasets show that SPRINT consistently outperforms state-of-the-art methods while providing more explainable recommendations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.02811v2">ChatSVA: Bridging SVA Generation for Hardware Verification via Task-Specific LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 Accepted by DAC 2026
    </div>
    <details class="paper-abstract">
      Functional verification consumes over 50% of the IC development lifecycle, where SystemVerilog Assertions (SVAs) are indispensable for formal property verification and enhanced simulation-based debugging. However, manual SVA authoring is labor-intensive and error-prone. While Large Language Models (LLMs) show promise, their direct deployment is hindered by low functional accuracy and a severe scarcity of domain-specific data. To address these challenges, we introduce ChatSVA, an end-to-end SVA generation system built upon a multi-agent framework. At its core, the AgentBridge platform enables this multi-agent approach by systematically generating high-purity datasets, overcoming the data scarcity inherent to few-shot scenarios. Evaluated on 24 RTL designs, ChatSVA achieves 98.66% syntax and 96.12% functional pass rates, generating 139.5 SVAs per design with 82.50% function coverage. This represents a 33.3 percentage point improvement in functional correctness and an over 11x enhancement in function coverage compared to the previous state-of-the-art (SOTA). ChatSVA not only sets a new SOTA in automated SVA generation but also establishes a robust framework for solving long-chain reasoning problems in few-shot, domain-specific scenarios. An online service has been publicly released at https://www.nctieda.com/CHATDV.html.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.18232v3">Two-Stage Regularization-Based Structured Pruning for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 ACL 2026 main
    </div>
    <details class="paper-abstract">
      The deployment of large language models (LLMs) is largely hindered by their large number of parameters. Structural pruning has emerged as a promising solution. Prior structured pruning methods directly remove unimportant parameters based on certain metrics, which often causes knowledge loss and necessitates extensive retraining. To overcome this, we introduce a novel pruning method TRSP: Two-Stage Regularization-Based Structured Pruning for LLMs. Specifically, we multiply the output of each transformer layer by an initial learnable weight and iteratively learn these weights by adding their $\ell_1$-norm as a regularization term to the loss function, serving as the first-stage regularization. Subsequently, we apply additional regularization to the difference between the output and input of layers with smaller weights, encouraging the shift of knowledge to the preserved layers. This serves as the second-stage regularization. TRSP retains more knowledge and better preserves model performance than direct parameter elimination. Through extensive experimentation we show that TRSP outperforms strong layer-wise structured pruning methods without requiring retraining. As a layer-wise pruning method, it delivers notable end-to-end acceleration, making it a promising solution for efficient LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05808v2">Hierarchical Reinforcement Learning with Augmented Step-Level Transitions for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 Accepted to ACL 2026 Main Conference
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have demonstrated strong capabilities in complex interactive decision-making tasks. However, existing LLM agents typically rely on increasingly long interaction histories, resulting in high computational cost and limited scalability. In this paper, we propose STEP-HRL, a hierarchical reinforcement learning (HRL) framework that enables step-level learning by conditioning only on single-step transitions rather than full interaction histories. STEP-HRL structures tasks hierarchically, using completed subtasks to represent global progress of overall task. By introducing a local progress module, it also iteratively and selectively summarizes interaction history within each subtask to produce a compact summary of local progress. Together, these components yield augmented step-level transitions for both high-level and low-level policies. Experimental results on ScienceWorld and ALFWorld benchmarks consistently demonstrate that STEP-HRL substantially outperforms baselines in terms of performance and generalization while reducing token usage. Our code is available at https://github.com/TonyStark042/STEP-HRL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.27306v3">GUIDE: Guided Updates for In-context Decision Evolution in LLM-Driven Spacecraft Operations</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 Accepted to AI4Space@CVPR Workshop in CVPR 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been proposed as supervisory agents for spacecraft operations, but existing approaches rely on static prompting and do not improve across repeated executions. We introduce \textsc{GUIDE}, a non-parametric policy improvement framework that enables cross-episode adaptation without weight updates by evolving a structured, state-conditioned playbook of natural-language decision rules. A lightweight acting model performs real-time control, while offline reflection updates the playbook from prior trajectories. Evaluated on an adversarial orbital interception task in the Kerbal Space Program Differential Games environment, GUIDE's evolution consistently outperforms static baselines. Results indicate that context evolution in LLM agents functions as policy search over structured decision rules in real-time closed-loop spacecraft interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.24119v2">Evaluating LLM-Based Translation of a Low-Resource Technical Language: The Medical and Philosophical Greek of Galen</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 Article + supplementary information
    </div>
    <details class="paper-abstract">
      Purpose: This study evaluates the quality of commercial large language model (LLM) machine translation (MT) for Ancient Greek technical prose and benchmarks standard automated MT evaluation metrics against expert human judgment. Design: We evaluated 60 translations by three LLMs (ChatGPT, Claude, Gemini) of 20 paragraph-length passages from 2 works by the Greek physician Galen (c. 129-216 CE): an expository text with two published English translations and a pharmacological text never before translated. Quality was assessed using seven automated metrics and systematic reference-free human evaluation via a modified Multidimensional Quality Metrics (MQM) framework applied by domain specialists. Findings: On the translated expository text, LLMs achieved high quality (mean MQM score 95.2/100). On the untranslated pharmacological text, quality was lower (79.9/100) but bimodally distributed: two passages with extreme terminological density produced catastrophic failures, while remaining passages scored within 4 points of the expository text. Terminology rarity, operationalized via corpus frequency, emerged as the dominant predictor of failure (r = -.97). Automated metrics showed moderate correlation with human judgment only on texts with wide quality variance; no metric discriminated among high-quality translations. Originality: This is the first systematic, reference-free expert human evaluation of LLM translation for any ancient language and the first study identifying textual properties predictive of translation failure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.04288v2">LLM-Enabled Open-Source Systems in the Wild: An Empirical Study of Vulnerabilities in GitHub Security Advisories</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 The 2nd International Workshop on Large Language Model Supply Chain Analysis (LLMSC 2026)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly embedded in open-source software (OSS) ecosystems, creating complex interactions among natural language prompts, probabilistic model outputs, and execution-capable components. However, it remains unclear whether traditional vulnerability disclosure frameworks adequately capture these model-mediated risks. To investigate this, we analyze 295 GitHub Security Advisories published between January 2025 and January 2026 that reference LLM-related components, and we manually annotate a sample of 100 advisories using the OWASP Top 10 for LLM Applications 2025. We find no evidence of new implementation-level weakness classes specific to LLM systems. Most advisories map to established CWEs, particularly injection and deserialization weaknesses. At the same time, the OWASP-based analysis reveals recurring architectural risk patterns, especially Supply Chain, Excessive Agency, and Prompt Injection, which often co-occur across multiple stages of execution. These results suggest that existing advisory metadata captures code-level defects but underrepresents model-mediated exposure. We conclude that combining the CWE and OWASP perspectives provides a more complete and necessary view of vulnerabilities in LLM-integrated systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14237v1">TOPCELL: Topology Optimization of Standard Cell via LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 Accepted to the 63rd ACM/IEEE Design Automation Conference (DAC 2026). 7 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Transistor topology optimization is a critical step in standard cell design, directly dictating diffusion sharing efficiency and downstream routability. However, identifying optimal topologies remains a persistent bottleneck, as conventional exhaustive search methods become computationally intractable with increasing circuit complexity in advanced nodes. This paper introduces TOPCELL, a novel and scalable framework that reformulates high-dimensional topology exploration as a generative task using Large Language Models (LLMs). We employ Group Relative Policy Optimization (GRPO) to fine-tune the model, aligning its topology optimization strategy with logical (circuit) and spatial (layout) constraints. Experimental results within an industrial flow targeting an advanced 2nm technology node demonstrate that TOPCELL significantly outperforms foundation models in discovering routable, physically-aware topologies. When integrated into a state-of-the-art (SOTA) automation flow for a 7nm library generation task, TOPCELL exhibits robust zero-shot generalization and matches the layout quality of exhaustive solvers while achieving an 85.91x speedup.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13353v1">Cross-Domain Query Translation for Network Troubleshooting: A Multi-Agent LLM Framework with Privacy Preservation and Self-Reflection</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 Accepted for publication in EuCNC & 6G Summit 2026. \c{opyright} 2026 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission
    </div>
    <details class="paper-abstract">
      This paper presents a hierarchical multi-agent LLM architecture to bridge communication gaps between non-technical end users and telecommunications domain experts in private network environments. We propose a cross-domain query translation framework that leverages specialized language models coordinated through multi-agent reflection-based reasoning. The resulting system addresses three critical challenges: (1) accurately classify user queries related to telecommunications network issues using a dual-stage hierarchical approach, (2) preserve user privacy through the anonymization of semantically relevant personally identifiable information (PII) while maintaining diagnostic utility, and (3) translate technical expert responses into user-comprehensible language. Our approach employs ReAct-style agents enhanced with self-reflection mechanisms for iterative output refinement, semantic-preserving anonymization techniques respecting $k$-anonymity and differential privacy principles, and few-shot learning strategies designed for limited training data scenarios. The framework was comprehensively evaluated on 10,000 previously unseen validation scenarios across various vertical industries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13349v1">When Less Latent Leads to Better Relay: Information-Preserving Compression for Latent Multi-Agent LLM Collaboration</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Communication in Large Language Model (LLM)-based multi-agent systems is moving beyond discrete tokens to preserve richer context. Recent work such as LatentMAS enables agents to exchange latent messages through full key-value (KV) caches. However, full KV relay incurs high memory and communication cost. We adapt eviction-style KV compression to this setting and introduce Orthogonal Backfill (OBF) to mitigate information loss from hard eviction. OBF injects a low-rank orthogonal residual from discarded KV states into the retained KV states. We evaluate proposed method against full KV relay on nine standard benchmarks spanning mathematical reasoning, coding, and knowledge-intensive QA. It achieves performance comparable to full KV relay while reducing communication cost by 79.8%--89.4%. OBF further improves the performance and achieves the best results on 7 of the 9 benchmarks. This suggests that more information does not necessarily lead to better communication; preserving the most useful information matters more. Our codebase is publicly available on https://github.com/markli404/When-Less-Latent-Leads-to-Better-Relay.
    </details>
</div>
