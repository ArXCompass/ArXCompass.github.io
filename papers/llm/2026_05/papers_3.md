# llm - 2026_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3
- [Part 4](papers_4.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05628v1">Towards Compute-Aware In-Switch Computing for LLMs Tensor-Parallelism on Multi-GPU Systems</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 15 pages, 18 figures, HPCA 2026
    </div>
    <details class="paper-abstract">
      Tensor parallelism (TP) in large-scale LLM inference and training introduces frequent collective operations that dominate inter-GPU communication. While in-switch computing, exemplified by NVLink SHARP (NVLS), accelerates collective operations by reducing redundant data transfer, its communication-centric design philosophy introduces the mismatch between its communication mode and the memory semantic requirement of LLM's computation kernel. Such a mismatch isolates the compute and communication phases, resulting in underutilized resources and limited overlap in multi-GPU systems. To address the limitation, we propose CAIS, the first Compute-Aware In-Switch computing framework that aligns communication modes with computation's memory semantics requirement. CAIS consists of three integral techniques: (1) compute-aware ISA and microarchitecture extension to enable compute-aware in-switch computing. (2) merging-aware TB (Thread Block) coordination to improve the temporal alignment for efficient request merging. (3) graph-level dataflow optimizer to achieve a tight cross-kernel overlap. Evaluations on LLM workloads show that CAIS achieves 1.38$\times$ average end-to-end training speedup over the SOTA NVLS-enabled solution, and 1.61$\times$ over T3, the SOTA compute-communicate overlap solutions but do not leverage NVLS, demonstrating its effectiveness in accelerating TP on multi-GPU systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.10241v2">ImCoref-CeS: An Improved Lightweight Pipeline for Coreference Resolution with LLM-based Checker-Splitter Refinement</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 Accepted by ACL2026 main
    </div>
    <details class="paper-abstract">
      Coreference Resolution (CR) is a critical task in Natural Language Processing (NLP). Current research faces a key dilemma: whether to further explore the potential of supervised neural methods based on small language models, whose detect-then-cluster pipeline still delivers top performance, or embrace the powerful capabilities of Large Language Models (LLMs). However, effectively combining their strengths remains underexplored. To this end, we propose \textbf{ImCoref-CeS}, a novel framework that integrates an enhanced supervised model with LLM-based reasoning. First, we present an improved CR method (\textbf{ImCoref}) to push the performance boundaries of the supervised neural method by introducing a lightweight bridging module to enhance long-text encoding capability, devising a biaffine scorer to comprehensively capture positional information, and invoking a hybrid mention regularization to improve training efficiency. Importantly, we employ an LLM acting as a multi-role Checker-Splitter agent to validate candidate mentions (filtering out invalid ones) and coreference results (splitting erroneous clusters) predicted by ImCoref. Extensive experiments demonstrate the effectiveness of ImCoref-CeS, which achieves superior performance compared to existing state-of-the-art (SOTA) methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21714v3">E-mem: Multi-agent based Episodic Context Reconstruction for LLM Agent Memory</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 This paper has been accepted by ICML 2026. If you find our project helpful, please consider giving it a star: https://github.com/dog-last/E-mem
    </div>
    <details class="paper-abstract">
      The evolution of Large Language Model (LLM) agents towards System~2 reasoning, characterized by deliberative, high-precision problem-solving, requires maintaining rigorous logical integrity over extended horizons. However, prevalent memory preprocessing paradigms suffer from destructive de-contextualization. By compressing complex sequential dependencies into pre-defined structures (e.g., embeddings or graphs), these methods sever the contextual integrity essential for deep reasoning. To address this, we propose E-mem, a framework shifting from Memory Preprocessing to Episodic Context Reconstruction. Inspired by biological engrams, E-mem employs a heterogeneous hierarchical architecture where multiple assistant agents maintain uncompressed memory contexts, while a central master agent orchestrates global planning. Unlike passive retrieval, our mechanism empowers assistants to locally reason within activated segments, extracting context-aware evidence before aggregation. Evaluations on the LoCoMo benchmark demonstrate that E-mem achieves over 54\% F1, surpassing the state-of-the-art GAM by 7.75\%, while reducing token cost by over 70\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20375v2">LLM-AutoDP: Automatic Data Processing via LLM Agents for Model Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 Accepted by VLDB2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can be fine-tuned on domain-specific data to enhance their performance in specialized fields. However, such data often contains numerous low-quality samples, necessitating effective data processing (DP). In practice, DP strategies are typically developed through iterative manual analysis and trial-and-error adjustment. These processes inevitably incur high labor costs and may lead to privacy issues in high-privacy domains like healthcare due to direct human access to sensitive data. Thus, achieving automated data processing without exposing the raw data has become a critical challenge. To address this challenge, we propose LLM-AutoDP, a novel framework that leverages LLMs as agents to automatically generate and optimize data processing strategies. Our method generates multiple candidate strategies and iteratively refines them using feedback signals and comparative evaluations. This iterative in-context learning mechanism enables the agent to converge toward high-quality processing pipelines without requiring direct human intervention or access to the underlying data. To further accelerate strategy search, we introduce three key techniques: Distribution Preserving Sampling, which reduces data volume while maintaining distributional integrity; Processing Target Selection, which uses a binary classifier to identify low-quality samples for focused processing; Cache-and-Reuse Mechanism}, which minimizes redundant computations by reusing prior processing results. Results show that models trained on data processed by our framework achieve over 80% win rates against models trained on unprocessed data. Compared to AutoML baselines based on LLM agents, LLM-AutoDP achieves approximately a 65% win rate. Moreover, our acceleration techniques reduce the total searching time by up to 10 times, demonstrating both effectiveness and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05598v1">Prober.ai: Gated Inquiry-Based Feedback via LLM-Constrained Personas for Argumentative Writing Development</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 Prototype awarded second place at the NYEdTech Hackathon (March 2026) https://www.nyedtechhackathon.com/2026-submissions
    </div>
    <details class="paper-abstract">
      The proliferation of large language models (LLMs) in educational settings has paradoxically undermined the cognitive processes they purport to support. Students increasingly outsource critical thinking to AI assistants that generate polished text on demand, resulting in measurable cognitive debt and diminished argumentative reasoning skills. We present Prober.ai, a web-based writing environment that inverts the conventional AI-tutoring paradigm: rather than generating or rewriting student text, the system constrains an LLM (Gemini 3 Flash Preview) through persona-specific system prompts and structured JSON output schemas to produce only targeted, inquiry-based questions about argumentative weaknesses. A two-phase interaction architecture -- Challenge and Unlock -- implements a pedagogical friction mechanism whereby revision suggestions are gated behind mandatory student reflection. The system's design is grounded in Toulmin's argumentation theory, research on peer feedforward questioning mechanisms, and evidence on AI-supported feedback in writing instruction. A functional prototype was developed in 36 hours during the NY EdTech Hackathon (March 2026), where it was awarded second place. We describe the system architecture, the prompt engineering methodology for constraining LLM output to pedagogically aligned JSON schemas, and discuss implications for scalable, cognition-preserving AI integration in writing education.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.23629v3">Emergent Slow Thinking in LLMs as Inverse Tree Freezing</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 34 pages, 17 figures, 1 table
    </div>
    <details class="paper-abstract">
      Reinforcement learning with verifiable rewards (RLVR) enables large language models to acquire slow, multi-step reasoning from sparse final-answer signals. We provide a statistical-physics picture of this emergence. We show that an autoregressive model's finite capacity forces it to compress its exponentially large prefix space into a Markov network of predictive states, on which slow thinking unfolds as a random walk -- the Concept Network (CoNet) picture. Within CoNet, RLVR dynamics are governed by two mechanisms: merging of compatible paths and frustrated competition among incompatible ones. Together they drive the network through nucleation, growth, and freezing into multi-input, single-output directed inverse trees. The picture reproduces the training dynamics of a 1.5-billion-parameter LLM and yields three predictions: reasoning chains lengthen as a geometric necessity of sparse topology; SFT induces catastrophic forgetting through bridge-node rupture; and frustration drives policy collapse. Building on the structural timing inherent in inverse-tree freezing, we propose Annealed-RLVR -- a brief SFT intervention at the moment of maximum frustration. It outperforms standard RLVR on both in- and out-of-distribution benchmarks, with the largest gains at high sampling budgets where standard RLVR collapses. The same SFT applied after the trees freeze instead triggers catastrophic forgetting, isolating timing as the active ingredient.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.03379v2">Two Calls, Two Moments, and the Vote-Accuracy Curve of Repeated LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Repeated sampling is a standard way to spend test-time compute, but its benefit is controlled by the latent distribution of correctness across examples, not by one-call accuracy alone. We study the binary correctness layer of repeated LLM inference under conditional-i.i.d. calls. One labeled call identifies the mean latent success probability; two labeled calls identify its second moment and hence the same-example correctness correlation that separates stable errors from recoverable call-level randomness. From these two moments, every fixed majority-vote budget has a sharp distribution-free two-call interval. The key technical reduction is that the infinite-dimensional moment problem has three-atom extremizers and quadratic dual certificates for every finite budget, so the bounds are exact rather than discretized or parametric. The first useful budget, three votes, has a closed form, width at most $1/8$, and a certified-improvement criterion. The infinite-vote endpoint is the limit of majority voting as the number of calls tends to infinity; it is also sharply bounded, but remains threshold-sensitive because it depends on latent mass around $q=1/2$. We add maximum-entropy and Latent-difficulty Gaussian-probit point completions, and experiments on LLM calls over QNLI and QQP show that empirical three- and five-vote accuracies are contained in the projected two-call regions while temperature changes and randomized model mixtures can create voting gains not ordered by one-call accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.01120v2">New Bounds for Zarankiewicz Numbers via Reinforced LLM Evolutionary Search</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 *Jay Bhan and Nicole Nobili contributed equally to this work as first authors, and their order was determined via coin flip
    </div>
    <details class="paper-abstract">
      The Zarankiewicz number $\textbf{Z}(m, n, s, t)$ is the maximum number of edges in a bipartite graph $G_{m, n}$ such that there is no complete $K_{s, t}$ bipartite subgraph. We determine for the first time the exact values of three Zarankiewicz numbers: $\textbf{Z}(11, 21, 3, 3)=116$, $\textbf{Z}(11, 22, 3, 3)=121$, and $\textbf{Z}(12, 22, 3, 3)=132$. We further establish lower bounds for 41 more Zarankiewicz numbers, including several that are within one edge of the best known upper bound, and we match the established value in four more closed cases. Our results are obtained using OpenEvolve, an open-source evolutionary algorithm based on Large Language Models (LLMs) that iteratively improves algorithms for generating mathematical constructions by optimizing a reward signal which we tailored for this specific problem. These findings provide new extremal graph constructions and demonstrate the potential of LLM-guided evolutionary search to contribute to mathematical research. In addition to presenting the resulting constructions, we report the generation algorithms produced, describe the relevant implementation details, and provide our computational costs. Our costs are remarkably low, at less than \$30 for each Zarankiewicz parameter combination, showing that LLM-guided evolutionary search can be an inexpensive, reproducible, and accessible tool for discovering new combinatorial constructions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05573v1">AstroAlertBench: Evaluating the Accuracy, Reasoning, and Honesty of Multimodal LLMs in Astronomical Classification</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Modern astronomical observatories generate a massive volume of multimodal data, creating a critical bottleneck for expert human review. While multimodal large language models (LLMs) have shown promise in interpreting complex visual and textual inputs, their ability to perform specialized scientific classification while providing interpretable reasoning remains understudied. We introduce AstroAlertBench, a comprehensive multimodal benchmark designed to evaluate LLM performance in astronomical event review along a three-stage logical chain: metadata grounding, scientific reasoning, and hierarchical classification over five categories. We use a pilot sample of 1,500 real-world alerts from the Zwicky Transient Facility (ZTF), a wide-field survey that scans the northern sky to detect transient astronomical events. On this dataset, we benchmark 13 frontier closed-source and open-weight LLMs that support visual input. Our results reveal that high accuracy does not always align with model ``honesty,'' defined as the ability to self-evaluate its reasoning, which affects its reliability as a real-world assistant. We further initialize a human-in-the-loop evaluation protocol as a precursor to future community-scale participation. Together, AstroAlertBench provides a framework for developing calibrated and interpretable astronomical assistants.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06656v1">Why Global LLM Leaderboards Are Misleading: Small Portfolios for Heterogeneous Supervised ML</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Ranking LLMs via pairwise human feedback underpins current leaderboards for open-ended tasks, such as creative writing and problem-solving. We analyze ~89K comparisons in 116 languages from 52 LLMs from Arena, and show that the best-fit global Bradley-Terry (BT) ranking is misleading. Nearly 2/3 of the decisive votes cancel out, and even the top 50 models according to the global BT ranking are statistically indistinguishable (pairwise win probabilities are at most 0.53 within the top 50 models). We trace this failure to strong, structured heterogeneity of opinions across language, task, and time. Moreover, we find an important characteristic - *language* plays a key role. Grouping by language (and families) increases the agreement of votes massively, resulting in two orders of magnitude higher spread in the ELO scores (i.e., very consistent rankings). What appears as global noise is in fact a mixture of coherent but conflicting subpopulations. To address such heterogeneity in supervised machine learning, we introduce the framework of $(λ, ν)$-portfolios, which are small sets of models that achieve a prediction error at most $λ$, "covering" at least a $ν$ fraction of users. We formulate this as a variant of the set cover problem and provide guarantees using the VC dimension of the underlying set system. On the Arena data, our algorithms recover just 5 distinct BT rankings that cover over 96% of votes at a modest $λ$, compared to the 21% coverage by the global ranking. We also provide a portfolio of 6 LLMs that cover twice as many votes as the top-6 LLMs from a global ranking. We further construct portfolios for a classification problem on the COMPAS dataset using an ensemble of fairness-regularized classification models and show that these portfolios can be used to detect blind spots in the data, which might be of independent interest to policymakers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06638v1">Can RL Teach Long-Horizon Reasoning to LLMs? Expressiveness Is Key</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has been applied to improve large language model (LLM) reasoning, yet the systematic study of how training scales with task difficulty has been hampered by the lack of controlled, scalable environments. We introduce ScaleLogic, a synthetic logical reasoning framework that offers independent control over two axes of difficulty: the depth of the required proof planning (i.e., the horizon) and the expressiveness of the underlying logic. Our proposed framework supports a wide range of logics: from simple implication-only logic ("if-then") towards more expressive first-order reasoning with conjunction ("and"), disjunction ("or"), negation ("not"), and universal quantification ("for all"). Using this framework, we show that the RL training compute $T$ follows a power law with respect to reasoning depth $D$ ($T \propto D^γ$, $R^{2} > 0.99$), and that the scaling exponent $γ$ increases monotonically with logical expressiveness, from $1.04$ to $2.60$. On downstream mathematics and general reasoning benchmarks, more expressive training settings yield both larger performance gains (up to $+10.66$ points) and more compute-efficient transfer compared to less expressive settings, demonstrating that what a model is trained on, not just how much it is trained, shapes downstream transfer. We further show that the power-law relationship holds across multiple RL methods, and curriculum-based training substantially improves scaling efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06635v1">Cited but Not Verified: Parsing and Evaluating Source Attribution in LLM Deep Research Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) power deep research agents that synthesize information from hundreds of web sources into cited reports, yet these citations cannot be reliably verified. Current approaches either trust models to self-cite accurately, risking bias, or employ retrieval-augmented generation (RAG) that does not validate source accessibility, relevance, or factual consistency. We introduce the first source attribution evaluation framework that uses a reproducible AST parser to extract and evaluate inline citations from LLM-generated Markdown reports at scale. Unlike methods that verify claims in isolation, our framework closes the loop by retrieving the actual cited content, enabling human or model evaluators to judge each citation against its source. Citations are evaluated along three dimensions. (1) Link Works verifies URL accessibility, (2) Relevant Content measures topical alignment, and (3) Fact Check validates factual accuracy against source content. We benchmark 14 closed-source and open-source LLMs across three evaluation dimensions using rubric-based LLM-as-a-judge evaluators calibrated through human review. Our results reveal that even the strongest frontier models maintain link validity above 94% and relevance above 80%, yet achieve only 39-77% factual accuracy, while fewer than half of open-source models successfully generate cited reports in a one-shot setting. Ablation studies on research depth show that Fact Check accuracy drops by approximately 42% on average across two frontier models as tool calls scale from 2 to 150, demonstrating that more retrieval does not produce more accurate citations. These findings reveal a critical disconnect between surface-level citation quality and factual reliability, and our framework provides the evaluation infrastructure to assess the disconnect.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06623v1">MASPO: Joint Prompt Optimization for LLM-based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 Accepted at ICML 2026
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based Multi-agent systems (MAS) have shown promise in tackling complex collaborative tasks, where agents are typically orchestrated via role-specific prompts. While the quality of these prompts is pivotal, jointly optimizing them across interacting agents remains a non-trivial challenge, primarily due to the misalignment between local agent objectives and holistic system goals. To address this, we introduce MASPO, a novel framework designed to automatically and iteratively refine prompts across the entire system. A core innovation of MASPO is its joint evaluation mechanism, which assesses prompts not merely by their local validity, but by their capacity to facilitate downstream success for successor agents. This effectively bridges the gap between local interactions and global outcomes without relying on ground-truth labels. Furthermore, MASPO employs a data-driven evolutionary beam search to efficiently navigate the high-dimensional prompt space. Extensive empirical evaluations across 6 diverse tasks demonstrate that MASPO consistently outperforms state-of-the-art prompt optimization methods, achieving an average accuracy improvement of 2.9. We release our code at https://github.com/wangzx1219/MASPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06605v1">How Many Iterations to Jailbreak? Dynamic Budget Allocation for Multi-Turn LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Evaluating and predicting the performance of large language models (LLMs) in multi-turn conversational settings is critical yet computationally expensive; key events -- e.g., jailbreaks or successful task completion by an agent -- often emerge only after repeated interactions. These events might be rare, and under any feasible computational budget, remain unobserved. Recent conformal survival frameworks construct reliable lower predictive bounds (LPBs) on the number of iterations to trigger the event of interest, but rely on static budget allocation that is inefficient in multi-turn setups. To address this, we introduce \emph{Dynamic Allocation via PRojected Optimization} (DAPRO), the first theoretically valid dynamic budget allocation framework for bounding the time-to-event in multi-turn LLM interactions. We prove that DAPRO satisfies the budget constraint and provides distribution-free, finite-sample coverage guarantees without requiring the conditional independence between censoring and event times assumed by prior conformal survival approaches. A key theoretical contribution is a novel coverage bound that scales with the square root of the mean censoring weight rather than the worst-case weight, yielding provably tighter guarantees than prior work. Furthermore, DAPRO can be employed to obtain unbiased, low-variance estimates of population-level evaluation metrics, such as the jailbreak rate, under limited computing resources. Comprehensive experiments across agentic task success, adversarial jailbreaks, toxic content generation, and RAG hallucinations using LLMs such as Llama 3.1 and Qwen 2.5 demonstrate that DAPRO consistently achieves coverage closer to the nominal level with lower variance than static baselines, while satisfying the budget constraint.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06596v1">FedAttr: Towards Privacy-preserving Client-Level Attribution in Federated LLM Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 39 pages, 4 figures, 21 tables (including appendix)
    </div>
    <details class="paper-abstract">
      Watermark radioactivity testing type of methods can detect whether a model was trained on watermarked documents, and have become key tools for protecting data ownership in the fine-tuning of large language models (LLMs). Existing works have proved their effectiveness in centralized LLM fine-tuning. However, this type of method faces several challenges and remains underexplored in federated learning (FL), a widely-applied paradigm for fine-tuning LLMs collaboratively on private data across different users. FL mainly ensures privacy through secure aggregation (SA), which allows the server to aggregate updates while keeping clients' updates private. This mechanism preserves privacy but makes it difficult to identify which client trained on watermarked documents. In this work, we propose FedAttr, a new client-level attribution protocol for FL. FedAttr identifies which clients trained on watermarked data via a paired-subset-difference mechanism, while preserving the privacy guarantees of SA and FL performance. FedAttr proceeds in three steps: (i) estimate each client's update by differencing two SA queries, (ii) score the estimate with the watermark detector via differential scoring, and (iii) combine scores across rounds via Stouffer method. We theoretically show that FedAttr produces an unbiased estimator of each client's update with bounded mutual information leakage (i.e., $O(d^*/N)$ per-round update). Moreover, FedAttr empirically achieves 100% TPR and 0% FPR, outperforming all baselines by at least 44.4% in TPR or 19.1% in FPR, with only 6.3% overhead relative to FL training time. Ablation studies confirm that FedAttr is robust to protocol parameters and configurations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06594v1">Automated Clinical Report Generation for Remote Cognitive Remediation: Comparing Knowledge-Engineered Templates and LLMs in Low-Resource Settings</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      The growing demand for cognitive remediation therapy, combined with limited speech therapist availability, has accelerated the adoption of remote rehabilitation tools. These systems generate large volumes of interaction data that are difficult for clinicians to review efficiently. This paper investigates automated clinical report generation for avatar-guided, home-based cognitive remediation sessions in a low-resource setting with no reference reports. We present and compare two approaches: (1) a rule-based template system encoding speech therapy domain knowledge as explicit decision rules and validated templates, ensuring clinical reliability and traceability; and (2) a zero-shot LLM-based approach (GPT-4) aimed at more fluent and concise output. Both systems use identical pre-extracted, expert-validated structured variables, enabling a controlled factual comparison. Outputs were evaluated by eight speech therapists and final-year students using a nine-criterion questionnaire. Results reveal a clear trade-off between clinical reliability and linguistic quality. The template-based system scored higher on fluidity, coherence, and results presentation, while GPT-4 produced more concise output. Directional differences are consistent across evaluation dimensions, though no comparison reached statistical significance after correction, reflecting the scale constraints of expert clinical evaluation. Based on evaluator feedback, we derive eight design recommendations for clinical reporting systems in remote rehabilitation settings. More broadly, this work contributes a replicable methodology combining expert elicitation, taxonomy-driven generation, and multi-dimensional human evaluation for clinical NLG in low-resource settings, and illustrates how controlled comparisons can inform the responsible adoption of generative AI in healthcare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06584v1">NeuroAgent: LLM Agents for Multimodal Neuroimaging Analysis and Research</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Multimodal neuroimaging analysis often involves complex, modality-specific preprocessing workflows that require careful configuration, quality control, and coordination across heterogeneous toolchains. Beyond preprocessing, downstream statistical analysis and disease classification commonly require task-specific code, evaluation protocols, and data-format conventions, creating additional barriers between raw acquisitions and reproducible scientific analysis. We present NeuroAgent, an LLM-driven agentic framework that automates key preprocessing and analysis steps for heterogeneous neuroimaging data, including sMRI, fMRI, dMRI, and PET, and supports interactive downstream analysis through natural-language queries. NeuroAgent employs a hierarchical multi-agent architecture with a feedback-driven Generate-Execute-Validate engine: agents autonomously generate executable preprocessing code, detect and recover from runtime errors, and validate output integrity. We evaluate the system on 1,470 subjects pooled across all ADNI phases (CN=1,000, AD=470), where all subjects have sMRI and tabular data, with subsets also having Tau-PET (n=469), fMRI (n=278), and DTI ($n=620$). Pipeline ablation studies across multiple LLM backends show that capable models reach up to 100% intent-parsing accuracy, with the strongest backend (Qwen3.5-27B) reaching 84.8% end-to-end preprocessing step correctness. Automated recovery limits manual intervention to edge cases where human review is required via the Human-In-The-Loop interface. For Alzheimer's Disease classification using automatically preprocessed multimodal data, our agent ensemble achieves an AUC of 0.9518 with four modalities, outperforming all single-modality baselines. These results show that NeuroAgent can reduce the manual effort required for neuroimaging preprocessing and enable end-to-end automated analysis pipelines for neuroimaging research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21464v2">Conversation for Non-verifiable Learning: Self-Evolving LLMs through Meta-Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 Accepted by ICML'26
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) for non-verifiable tasks, such as creative writing, dialogue, and ethical reasoning, remains challenging due to the absence of ground-truth labels. While LLM-as-Judge approaches offer a scalable alternative to human feedback, they face a fundamental limitation: performance is constrained by the evaluator's own quality. If the judge cannot recognize good solutions, it cannot provide useful training signals, and evaluation biases (e.g., favoring verbosity over quality) remain unaddressed. This motivates meta-evaluation: the ability to evaluate and improve the evaluator itself. We introduce CoNL, a framework that unifies generation, evaluation, and meta-evaluation through multi-agent self-play. Our key insight: critique quality can be measured by whether it helps others improve their solutions. In CoNL, multiple agents sharing the same policy engage in structured conversations to propose, critique, and revise solutions. Critiques that enable solution improvements earn a diagnostic reward, creating explicit supervision for meta-evaluation and enabling joint optimization of generation and judging capabilities through self-play, without external judges or ground truth. Experiments on various benchmarks show that CoNL achieves consistent improvements over self-rewarding baselines while maintaining stable training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06544v1">CCL-Bench 1.0: A Trace-Based Benchmark for LLM Infrastructure</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Evaluative claims about LLM infrastructure -- ``workload X is fastest on hardware Y with software Z'' -- depend on a complex configuration space spanning hardware accelerators, interconnect bandwidth, software frameworks, parallelism plans, and communication libraries. Current infrastructure evaluation benchmarks publish a small set of end-to-end numbers that do not explain why one configuration outperforms another. We present CCL-Bench, a trace-based benchmark that addresses the limitations of existing benchmarks by recording reusable evidence for every ML workload. Each contributed data point in CCL-Bench packages an execution trace, a YAML workload card, and the launch scripts. We have developed a community-extensible toolkit to compute fine-grained compute, memory, and communication efficiency metrics from this evidence. Using CCL-Bench, we surface three claims that summary-statistic benchmarks cannot support: (i) higher compute-communication overlap can coincide with longer training step time and reveal inefficient parallelization choices, (ii) doubling TPU interconnect bandwidth yields a much higher end-to-end improvement in step time than doubling GPU interconnect bandwidth on small and medium workloads, and (iii) the best-tuned configuration on one training framework can run up to 3$\times$ slower than the best-tuned configuration on a peer framework on identical hardware.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.13727v3">Attribution-Guided Pruning for Insight and Control: Circuit Discovery and Targeted Correction in Small-scale LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 Work in progress (9 pages manuscript, 3 pages references, 16 pages appendix)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely deployed in real-world applications, yet their internal mechanisms remain difficult to interpret and control, limiting our ability to diagnose and correct undesirable behaviors. Mechanistic interpretability addresses this challenge by identifying circuits -- subsets of model components responsible for specific behaviors. However, discovering such circuits in LLMs remains difficult due to their scale and complexity. We frame circuit discovery as identifying parameters that contribute most to model outputs on task-specific inputs, and use Layer-wise Relevance Propagation (LRP) with reference samples to attribute and extract these components via pruning. Building on this, we introduce contrastive relevance to isolate circuits associated with undesired behaviors while preserving general capabilities, enabling targeted model correction. On OPT-125M, we show that pruning as little as ~0.3% of neurons substantially reduces toxic outputs, while pruning approximately 0.03% of weight elements mitigates repetitive text generation without degrading general performance. These results establish attribution-guided pruning as an effective mechanism for identifying and intervening on behavior-specific circuits in LLMs. We further validate our findings on additional small-scale language models, demonstrating that the proposed approach transfers across architectures. Our code is publicly available at https://github.com/erfanhatefi/SparC3.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06527v1">STALE: Can LLM Agents Know When Their Memories Are No Longer Valid?</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents are increasingly expected to maintain coherent, long-term personalized memory, yet current benchmarks primarily measure static fact retrieval, overlooking the ability to revise stored beliefs when new evidence emerges. We identify a critical and underexplored failure mode, Implicit Conflict: a later observation invalidates an earlier memory without explicit negation, requiring contextual inference and commonsense reasoning to detect. To rigorously evaluate this capability, we introduce STALE, a benchmark of 400 expert-validated conflict scenarios (1,200 evaluation queries across three probing dimensions) spanning over 100 everyday topics with contexts up to 150K tokens. We propose a three-dimensional probing framework that tests State Resolution (detecting that a prior belief is outdated), Premise Resistance (rejecting queries that falsely presuppose a stale state), and Implicit Policy Adaptation (proactively applying updated states in downstream behavior). A systematic evaluation of frontier LLMs and specialized memory frameworks reveals a pervasive gap between retrieving updated evidence and acting on it, with even the best evaluated model achieving only 55.2% overall accuracy. Models often accept outdated assumptions embedded in a user's query, and they struggle to recognize when a change in one aspect of the user's state should invalidate related memories. To establish an initial baseline for state-aware memory, we further present CUPMem, a prototype that strengthens write-time revision through structured state consolidation and propagation-aware search, suggesting that explicit state adjudication is a promising direction for robust agentic memory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06525v1">Sustaining Cooperation in Populations Guided by AI: A Folk Theorem for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to provide instructions to many agents who interact with one another. Such shared reliance couples agents who appear to act independently: they may in fact be guided by a common model. This coupling can change the prospects for cooperation among agents with misaligned incentives. We study settings in which multiple LLMs each advise a population of clients who participate in instances of an underlying game, creating strategic interaction at the level of the LLMs themselves. This induces a meta-game among the LLMs, mediated through clients. We first analyze the one-shot setting, where shared instructions can change equilibrium behavior only when an LLM may influence more than one role in the same interaction; in such cases, cooperation may emerge, and the effect of client share can be beneficial, harmful, or non-monotone, depending on the base game. Our main result concerns the repeated setting. We prove a folk theorem for LLMs: despite indirect observation and the clients' inability to identify which LLM advised their opponents, all feasible and individually rational outcomes can be sustained as $\varepsilon$-equilibria. The result does not follow from the standard folk theorem and requires new proof techniques. Together, these results show that shared LLM guidance can sustain cooperation among populations of agents even when the underlying incentives are misaligned.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.02241v3">Zero-Shot Confidence Estimation for Small LLMs: When Supervised Baselines Aren't Worth Training</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      How reliably can a small language model estimate its own correctness? The answer determines whether local-to-cloud routing-escalating queries a cheap local model cannot handle-can work without supervised training data. As inference costs dominate large language model (LLM) deployment budgets, routing most queries to a cheap local model while reserving expensive cloud calls for hard cases is an increasingly common cost-control strategy. We compare zero-shot confidence signals against RouteLLM-style supervised baselines across three 7-8B model families and two datasets (1,000 and 500 queries per model, respectively). Average token log-probability, which requires no training data, matches or exceeds supervised baselines in-distribution (Area Under the Receiver Operating Characteristic curve (AUROC) 0.650-0.714 vs. 0.644-0.676) and substantially outperforms them out-of-distribution (0.717-0.833 vs. 0.512-0.564), because it measures a property of the model's generation rather than the query distribution. This paper further proposes retrieval-conditional self-assessment, a pre-generation signal that selectively injects retrieved knowledge when similarity is high, improving over bare self-assessment by up to +0.069 AUROC at 3-10x lower latency than log-probability. A supervised baseline trained on 1,000 labeled examples never exceeds the zero-shot signal. We release all code, data, and experiment logs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06486v1">Autonomous Adversary: Red-Teaming in the age of LLM</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 Accepted at ACISP 2026
    </div>
    <details class="paper-abstract">
      Language Model Agents (LMAs) are emerging as a powerful primitive for augmenting red-team operations. They can support attack planning, adversary emulation, and the orchestration of multi-step activity such as lateral movement, a core enabling capability of advanced persistent threat (APT) campaigns. Using frameworks such as MITRE ATT&CK, we analyze where these agents intersect with core offensive functions and assess current strengths and limitations of LMAs with an emphasis on governance and realistic evaluation. We benchmark LMAs across two lateral-movement scenarios in a controlled adversary-emulation environment, where LMAs interact with instrumented cyber agents, observe execution artifacts, and iteratively adapt based on environmental feedback. Each scenario is formalized as an ordered task chain with explicit validation predicates, leveraging an LLM-as-a-Judge paradigm to ensure deterministic outcome verification. We compare three operational modalities: fully autonomous execution, self-scaffolded planning, and expert-defined action plans. Preliminary findings indicate that expert-defined action plans yield higher task-completion rates relative to other operational modes. However, failure remains frequent across all modalities, largely attributable to brittle command invocation, environmental and deployment instability, and recurring errors in credential management and state handling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06457v1">Beyond Task Success: Measuring Workflow Fidelity in LLM-Based Agentic Payment Systems</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 6 pages, 2 tables. Accept at AI and Data Science for Digital Finance (AIDS4DF) Workshop, PAKDD 2026
    </div>
    <details class="paper-abstract">
      LLM-based multi-agent systems are increasingly deployed for payment workflows, yet prevailing metrics, Task Success Rate (TSR) and Agent Handoff F1-Score (HF1), capture only final outcomes or unordered routing decisions. We introduce the Agentic Success Rate (ASR), a trajectory-fidelity metric that compares observed and expected agent execution sequences at the transition level, decomposing performance into Transition Recall and Transition Precision. Applied to the Hierarchical Multi-Agent System for Payments (HMASP) across 18 LLMs and 90,000 task instances, ASR reveals that 10 of 18 models systematically skip a confirmation checkpoint during payment checkout, a deviation invisible to both TSR and HF1, while 8 models enforce the checkpoint perfectly. Notably, GPT-4.1 exhibits hidden workflow shortcuts despite achieving perfect TSR and HF1, while GPT-5.2 achieves perfect ASR. Prompt refinements and deterministic routing guards guided by ASR diagnostics yield substantial TSR improvements, with gains up to +93.8 percentage points for previously struggling models, demonstrating that trajectory-level evaluation is essential in regulated domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06455v1">PrefixGuard: From LLM-Agent Traces to Online Failure-Warning Monitors</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents now execute long, tool-using tasks where final outcome checks can arrive too late for intervention. Online warning requires lightweight prefix monitors over heterogeneous traces, but hand-authored event schemas are brittle and deployment-time LLM judging is costly. We introduce PrefixGuard, a trace-to-monitor framework with an offline StepView induction step followed by supervised monitor training. StepView induces deterministic typed-step adapters from raw trace samples, and the monitor learns an event abstraction and prefix-risk scorer from terminal outcomes. Across WebArena, $τ^2$-Bench, SkillsBench, and TerminalBench, the strongest PrefixGuard monitors reach 0.900/0.710/0.533/0.557 AUPRC. Using the strongest backend within each representation, they improve over raw-text controls by an average of +0.137 AUPRC. LLM judges remain substantially weaker under the same prefix-warning protocol. We also derive an observability ceiling on score-based area under the precision-recall curve (AUPRC) that separates monitor error from failures lacking evidence in the observed prefix. For finite-state audit, post-hoc deterministic finite automaton (DFA) extraction remains compact on WebArena and $τ^2$-Bench (29 and 20 states) but expands to 151 and 187 states on SkillsBench and TerminalBench. Finally, first-alert diagnostics show that strong ranking does not imply deployment utility: WebArena ranks well yet fails to support low-false-alarm alerts, whereas $τ^2$-Bench and TerminalBench retain more actionable early alerts. Together, these results position PrefixGuard as a practical monitor-synthesis recipe with explicit diagnostics for when prefix warnings translate into actionable interventions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06445v1">Constraint Decay: The Fragility of LLM Agents in Backend Code Generation</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents demonstrate strong performance in autonomous code generation under loose specifications. However, production-grade software requires strict adherence to structural constraints, such as architectural patterns, databases, and object-relational mappings. Existing benchmarks often overlook these non-functional requirements, rewarding functionally correct but structurally arbitrary solutions. We present a systematic study evaluating how well agents handle structural constraints in multi-file backend generation. By fixing a unified API contract across 80 greenfield generation tasks and 20 feature-implementation tasks spanning eight web frameworks, we isolate the effect of structural complexity using a dual evaluation with end-to-end behavioral tests and static verifiers. Our findings reveal a phenomenon of constraint decay: as structural requirements accumulate, agent performance exhibits a substantial decline. Capable configurations lose 30 points on average in assertion pass rates from baseline to fully specified tasks, while some weaker configurations approach zero. Framework sensitivity analysis exposes significant performance disparities: agents succeed in minimal, explicit frameworks (e.g., Flask) but perform substantially worse on average in convention-heavy environments (e.g., FastAPI, Django). Finally, error analysis identifies data-layer defects (e.g., incorrect query composition and ORM runtime violations) as the leading root causes. This work highlights that jointly satisfying functional and structural requirements remains a key open challenge for coding agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06443v1">AgenticPrecoding: LLM-Empowered Multi-Agent System for Precoding Optimization</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Precoding is a key technique for interference management and performance improvement in multi-antenna wireless systems. However, existing precoding methods are typically developed for specific system models, objectives, and constraint sets, which limits their adaptability to the heterogeneous and evolving scenarios expected in future 6G networks. To address this limitation, we propose AgenticPrecoding, a universal multi-agent framework that automates end-to-end precoding derivation directly from user-level communication requirements. Specifically, AgenticPrecoding decomposes the derivation process into four coordinated stages: problem formulation, solver selection, prompt upsampling, and code generation, assigning each stage to a specialized agent tailored to its specific reasoning demands. We employ two LoRA-adapted reasoning agents to inject precoding-specific domain knowledge for problem formulation and solver selection, while two general-purpose Large Language Models (LLMs) handle prompt refinement and executable code generation. Furthermore, a feedback-driven refinement mechanism is incorporated to enhance code executability, constraint feasibility, and solution quality. Extensive experiments across 10 representative precoding scenarios demonstrate that AgenticPrecoding achieves superior cross-scenario adaptability compared to conventional optimization-based and LLM-based baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06402v1">SparseForge: Efficient Semi-Structured LLM Sparsification via Annealing of Hessian-Guided Soft-Mask</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Semi-structured sparsity provides a practical path to accelerate large language models (LLMs) with native hardware support, but post-training semi-structured pruning often suffers from substantial quality degradation due to strong structural coupling. Existing methods rely on large-scale sparse retraining to recover accuracy, resulting in high computational cost. We propose SparseForge, a post-training framework that improves recovery efficiency by directly optimizing the sparsity mask rather than scaling up retraining tokens. SparseForge combines Hessian-aware importance estimation with progressive annealing of soft masks into hardware-executable structured sparsity, enabling stable and efficient sparse recovery. On LLaMA-2-7B under 2:4 sparsity, SparseForge achieves 57.27% average zero-shot accuracy with only $\textbf{5B}$ retraining tokens, surpassing the dense model's 56.43% accuracy and approaching the 57.52% result of a state-of-the-art method using $\textbf{40B}$ tokens. Such improvements on the accuracy-efficiency trade-off from SparseForge are shown to be consistent across model families.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20822v2">MediEval: A Unified Medical Benchmark for Patient-Contextual and Knowledge-Grounded Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly applied to medicine, yet their adoption is limited by concerns over reliability and safety. Existing evaluations either test factual medical knowledge in isolation or assess patient-level reasoning without verifying correctness, leaving a critical gap. We introduce MediEval, a benchmark that links MIMIC-IV electronic health records (EHRs) to a unified knowledge base built from UMLS and other biomedical vocabularies. MediEval generates diverse factual and counterfactual medical statements within real patient contexts, enabling systematic evaluation across a 4-quadrant framework that jointly considers knowledge grounding and contextual consistency. Using this framework, we identify critical failure modes, including hallucinated support and truth inversion, that current proprietary, open-source, and domain-specific LLMs frequently exhibit. To address these risks, we propose Counterfactual Risk-Aware Fine-tuning (CoRFu), a DPO-based method with an asymmetric penalty targeting unsafe confusions. CoRFu improves by +16.4 macro-F1 points over the base model and eliminates truth inversion errors, demonstrating both higher accuracy and substantially greater safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06374v1">ResiHP: Taming LLM Training Failures with Dynamic Hybrid</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Hybrid parallelism underpins large-scale LLM training across tens of thousands of GPUs. At such scale, hardware failures on individual devices lead to performance skew across devices, diminishing overall training efficiency. Existing resilient systems overlook sequence length variability in datasets and device performance skew under hybrid parallelism. As a result, (1) iteration time fluctuations induced by sequence length variability can trigger spurious fail-slow detections, and (2) failures are mitigated through individual adaptations in hybrid parallelism, leading to unnecessary detection overhead and inefficient resilient training. To respond, this paper presents ResiHP, a resilient system that enables robust failure detection and fine-grained adaptation for hybrid parallel training. First, we develop a Detector to accurately identify failures. In particular, it employs a workload-aware execution time predictor that disentangles failures from iteration time fluctuations while remaining lightweight for online detection. Second, we design a Scheduler that dynamically adapts parallelism group sizes, model partitioning, and workload scheduling policies to improve training efficiency under failures. Experiments show that ResiHP improves training throughput by 1.04-4.39$\times$ compared with state-of-the-art resilient training systems under diverse failure scenarios in a 256-GPU cluster.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06350v1">Is Escalation Worth It? A Decision-Theoretic Characterization of LLM Cascades</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Model cascades, in which a cheap LLM defers to an expensive one on low-confidence queries, are widely used to navigate the cost-quality tradeoff at deployment. Existing approaches largely treat the deferral threshold as an empirical hyperparameter, with limited guidance on the geometry of the resulting cost-quality frontier over a model pool. We develop a decision-theoretic framework grounded in constrained optimization and duality. For a two-model cascade, we establish piecewise concavity of the cost-quality frontier on decreasing-benefit regions of the confidence support, with reciprocal shadow prices linking the budget- and quality-constrained formulations. Given a pool of $k$ models, we characterize the frontier achievable by deterministic two-model threshold cascades as the pointwise envelope over $\binom{k}{2}$ pairwise cascades, with switching points where the optimal pair changes. For $k$-model cascades, we derive first-order conditions in which a single shadow price equalizes marginal quality-per-cost across stage boundaries. We validate the framework on five benchmarks (MATH, MMLU, TriviaQA, SimpleQA, LiveCodeBench) across eight models from five providers. Within the deterministic threshold-cascade class, full fixed chains underperform the pairwise envelope, and optimized subsequence cascades do not deliver practically meaningful held-out gains over it. A lightweight pre-generation router exceeds the best cascade policy on four of five datasets, mainly because it avoids the cheap model's generation cost on queries sent directly to a larger model rather than because of a stronger routing signal. These results suggest that cascade performance is limited primarily by structural cost, since cascades pay the cheap model before any escalation decision, rather than by a shortage of intermediate stages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06335v1">Eliciting associations between clinical variables from LLMs via comparison questions across populations</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      The training data of large language models (LLMs) comprises a wide range of biomedical literature, reflecting data from many different patient populations. We investigate how it might be possible to recover information on correlation and causal links between patient characteristics, as a key building block for medical decision making. To avoid the pitfalls of direct elicitation, we propose an approach based on structured comparison questions, specifically patient comparison triplet questions. This is combined with a statistical model for the LLM representation that provides estimates of correlations without access to activations or model internals. Intuitively, we consider how similarity decisions of LLMs based on a first variable are affected by providing information on a second variable for one of the patients being assessed. We then induce prompt-level environment shifts to obtain correlation estimates for different subpopulations, which enables an invariant causal prediction (ICP) approach to obtain conservative candidate parent links. We demonstrate the method in two clinical domains, chronic obstructive pulmonary disease (COPD) and multiple sclerosis (MS). Across prompted environments, the elicited correlations are smooth, stable, and clinically interpretable, yet vary in a statistically significant way that supports downstream invariance testing, such that ICP provides a small set of candidate invariant parent links. These results show that indirect elicitation via triplet comparisons can recover meaningful association structure from LLMs and offer a cautious route from implicit correlations to causal statements that are congruent with LLM answering patterns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06334v1">MANTRA: Synthesizing SMT-Validated Compliance Benchmarks for Tool-Using LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Tool-using large language model (LLM) agents are increasingly deployed in settings where their reliable behavior is governed by strict procedural manuals. Ensuring that such agents comply with the rules from these manuals is challenging, as they are typically written for humans in natural language while agent behavior manifests as an execution trace of tool calls. Existing evaluations of LLM agents rely on manually constructed benchmarks or LLM-based judges, which either do not scale or lack reliability for complex, long-horizon manuals. To overcome these limitations, we present MANTRA, a framework for automatically synthesizing machine-checkable compliance benchmarks from natural-language manuals and tool schemas. MANTRA independently generates (i) a symbolic world model capturing procedural dependencies, and (ii) a set of trace-level compliance checks for a given task, and validates their consistency using SMT solving. A structured repair loop resolves inconsistencies, requiring human intervention only as a fallback. %This yields benchmarks that are formally validated. Importantly, MANTRA supports arbitrary domains and long procedural manuals, and provides a tunable notion of task complexity which is utilized to automatically derive challenging tasks accompanying compliance checks. Using MANTRA, we build a new benchmark suite with 285 tasks across 6 domains scaling to 50+ page manuals with minimal human effort. Empirically, we show that the compliance checks are richer with stronger constraint enforcement compared to existing benchmarks. Additionally, the granularity of the checks can be used for debugging the agents' failure modes. These results demonstrate that combining automated benchmark generation with formally grounded validation methods enables scalable and reliable benchmarking of tool-using agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06327v1">Measuring Evaluation-Context Divergence in Open-Weight LLMs: A Paired-Prompt Protocol with Pilot Evidence of Alignment-Pipeline-Specific Heterogeneity</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Safety benchmarks are routinely treated as evidence about how a language model will behave once deployed, but this inference is fragile if behavior depends on whether a prompt looks like an evaluation. We define evaluation-context divergence as an observable within-item change in behavior induced by framing a fixed task as an evaluation, a live deployment interaction, or a neutral request, and present a paired-prompt protocol that measures it in open-weight LLMs while controlling for paraphrase variation, benchmark familiarity, and judge framing-sensitivity. Across five instruction-tuned checkpoints from four open-weight families plus a matched OLMo-3 base/instruct ablation ($20$ paired items, $840$ generations per checkpoint), we find striking heterogeneity. OLMo-3-Instruct alone is eval-cautious -- evaluation framing raises refusal vs. neutral by $11.8$pp ($p=0.007$) and reduces harmful compliance vs. deployment by $3.6$pp ($p=0.024$, $0/20$ items inverted) -- while Mistral-Small-3.2, Phi-3.5-mini, and Llama-3.1-8B are deployment-cautious}, with marginal eval-vs-deployment refusal effects of $-9$ to $-20$pp. The matched OLMo-3 base also exhibits the deployment-cautious pattern, identifying alignment as the inversion stage; within Llama-3.1, the $70$B model preserves direction with attenuated magnitude, ruling out a simple ``small-model effect that reverses at scale.'' One caveat: the cross-family heterogeneity is judge-dependent. Re-judging with a different-family safety classifier (Llama-Guard-3-8B) preserves the within-OLMo eval-cautious direction but flattens the cross-family contrast, indicating that the two judges operationalize distinct constructs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.11581v5">Hidden Measurement Error in LLM Pipelines Distorts Annotation, Evaluation, and Benchmarking</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      LLM evaluations drive which models get deployed, what safety standards get adopted, which research conclusions get published, and how projections of AI's labor-market impact get made. Yet standard confidence intervals ignore variability from judge model choice, model temperature, and prompt phrasing, producing under-coverage that worsens with more data. The omitted variance can shift results enough to reverse conclusions \citep{baumann2025llmhacking, huang2026dropping}; pipelines that fail to average over it leave the surface that ``benchmark hacking'' exploits \citep{singh2025leaderboard}. This paper decomposes LLM pipeline uncertainty into its sources, distinguishes variance that shrinks with more data from sensitivity to researcher design choices, and uses design-study projections to reduce total evaluation error (TEE). Across the demonstrations, naive standard errors are 40 - 60\% smaller than the TEE-corrected SE. Using Chatbot Arena data, we show naive 95\% CI coverage drops as $n$ grows while TEE-corrected coverage holds at 95\%, and TEE-guided pipelines restrict the benchmark gaming surface from 56 to 32 Elo ($K=27$), below the human-leaderboard baseline. We show further that a small pilot recovers honest CIs and projects which design changes most improve precision. Acting on those projections halves MMLU estimation error against the answer key at equivalent cost, and raises per-match agreement with human votes by 7.9 percentage points on Chatbot Arena.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.01189v2">Beyond Value Elicitation: Towards Moral Profiles in Early Requirements Engineering via Role-Playing Games and Anthropologist LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      This study presents a proof of concept for eliciting and representing the moral profiles of digital system users in Requirements Engineering (RE) by combining immersive role-playing games (RPGs) with large language model (LLM) analysis. While existing approaches rely on predefined value taxonomies and explicit articulation, values are often tacit, context-dependent, and difficult to express directly. To address these limitations, we propose moving from the elicitation of discrete moral values to the narrative reconstruction and representation of users' moral profiles. Grounded in phenomenological and narrative anthropology, the approach focuses on capturing users' moral orientations as they emerge through situated decision-making. RPG sessions generate context-rich narrative data, which are then analyzed by a specialized LLM (GPT-A) to produce individual anthropological moral profiles (IAMPs). A validation process based on cross-comparison between model outputs and participants' responses in unseen moral scenarios assesses the adequacy of the generated representations. Results indicate that RPG environments effectively support the generation of rich, context-dependent data for eliciting tacit values, and that an anthropologically grounded LLM can transform such data into coherent narrative representations of users' moral profiles. These representations enable the contextual interpretation of users' preferences and values within the given domain, with improved performance when interpretive framing captures relationships between actions, underlying motivations, and individual domain expertise. From an RE perspective, this approach enables the analysis of user preferences and trade-offs while preserving their situated and dynamic nature, providing a foundation for integrating human moral values into the early stages of RE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06307v1">LLM-Based Educational Simulation: Evaluating Temporal Student Persona Stability Across ADHD Profiles</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Student simulation with Large language models (LLMs) offers a scalable alternative for educational research and teacher training. Yet, its validity depends on whether models maintain stable personas across extended interactions. We test this prerequisite using a dual-assessment framework measuring self-reported characteristics and observer-rated behavioral expressions. Across two experiments testing four clinically-grounded ADHD persona conditions, five LLMs, and three prompt designs, we quantify between-conversation stability (N=4,968) and within-conversation stability (N=3,952 across 9 turns). Self-reported characteristics remain stable for high intensities, constituting a necessary prerequisite for valid behavioral simulation. Observer-rated behavioral expression reveals selective instability: within-conversation drift occurs in unscripted dialog for high and moderate ADHD personas. Scripted interactions with explicit task prompts eliminate this drift entirely. Stable, persona-aligned simulated learners benefit from a structured interaction design to maintain behavioral coherence, which holds significant implications for teacher training, adaptive tutoring, and any application requiring sustained, path-dependent learner interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06305v1">Addressing Labelled Data Scarcity: Taxonomy-Agnostic Annotation of PII Values in HTTP Traffic using LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 Accepted to 2026 IEEE European Symposium on Security and Privacy Workshops (EuroS&PW)
    </div>
    <details class="paper-abstract">
      Automated privacy audits of web and mobile applications often analyse outbound HTTP traffic to detect Personally Identifiable Information (PII) leakage. However, existing learning-based detectors typically depend on scarce, manually labelled traffic and are tightly coupled to fixed label taxonomies, limiting transferability across domains and evolving definitions of PII. This paper investigates whether Large Language Models (LLMs) can support taxonomy-agnostic annotation of explicitly transmitted PII values in HTTP message bodies when the taxonomy is provided at runtime. We introduce a multi-stage LLM-based pipeline that combines deterministic pre-processing with label-level classification, targeted instance-level value annotation, and output validation. To enable controlled evaluation and exemplar-based prompting without relying on sensitive real-user captures, we further propose an LLM-based generator for synthetic HTTP traffic with manually validated, taxonomy-derived PII annotations. We evaluate the approach across three taxonomies spanning different PII domains and granularity levels. Results show that the pipeline accurately detects PII types and extracts corresponding values for concrete PII taxonomies. Overall, our findings position LLMs as a promising foundation for flexible, taxonomy-agnostic traffic annotation and for creating labelled data under evolving privacy taxonomies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06279v1">Correct Code, Vulnerable Dependencies: A Large Scale Measurement Study of LLM-Specified Library Versions</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 35 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are now largely involved in software development workflows, and the code they generate routinely includes third-party library (TPL) imports annotated with specific version identifiers. These version choices can carry security and compatibility risks, yet they have not been systematically studied. We present the first large-scale measurement study of version-level risk in LLM-generated Python code, evaluating 10 LLMs on PinTrace, a curated benchmark of 1,000 Stack Overflow programming tasks. LLMs tend to specify version identifiers when directly prompted at 26.83%-95.18%, while down to 6.45%-59.19% in creating a manifest file directly. Among the specified versions, 36.70%-55.70% of tasks contain at least one known CVE, and 62.75%-74.51% of them carry Critical or High severity ratings. In 72.27%-91.37% of cases, the associated CVEs were publicly disclosed before the model's knowledge cutoff. The statistics show all models converge on the same small set of risky release versions, indicating a systemic bias rather than isolated model error. Static compatibility rates range from 19.70% to 63.20%, with installation failure as the dominant cause. The dynamic test cases confirm the pattern by 6.49%-48.62% pass rates. Further experiments confirm that these failures are attributable to version selection rather than code quality, and that externally anchored version constraints substantially reduce both vulnerability exposure and compatibility failures. Our findings reveal LLM version selection as a first-class, previously overlooked risk surface in LLM-based development. We disclosed these findings to the community of the evaluated models, and several confirmed the issue. All the code and dataset have been released for open science at https://github.com/dw763j/PinTrace.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19331v3">Evaluating LLM-Driven Summarisation of Parliamentary Debates with Computational Argumentation</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 Accepted at KR'26 In The Wild Track. Camera Ready with additional supplementary materials
    </div>
    <details class="paper-abstract">
      Understanding how policy is debated and justified in parliament is a fundamental aspect of the democratic process. However, the volume and complexity of such debates mean that outside audiences struggle to engage. Meanwhile, Large Language Models (LLMs) have been shown to enable automated summarisation at scale. While summaries of debates can make parliamentary procedures more accessible, evaluating whether these summaries faithfully communicate argumentative content remains challenging. Existing automated summarisation metrics have been shown to correlate poorly with human judgements of consistency (i.e., faithfulness or alignment between summary and source). In this work, we propose a formal framework for evaluating parliamentary debate summaries that grounds argument structures in the contested proposals up for debate. Our novel approach, driven by computational argumentation, focuses the evaluation on formal properties concerning the faithful preservation of the reasoning presented to justify or oppose policy outcomes. We demonstrate our methods using a case-study of debates from the European Parliament and associated LLM-driven summaries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06257v1">LearnMate^2: Design and Evaluation of an LLM-powered Personalized and Adaptive Support System for Online Learning</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Personalization is crucial for effective learning, yet online learning, designed for widespread availability and open access, lacks personalized guidance. Recent advancements in large language models (LLMs) offer opportunities to bridge this gap. We explore how LLM-driven tools may be designed to support personalized and adaptive learning and examine how they shape user experience and learning outcomes. We iteratively designed \tool{} to support online learning by providing personalized study plans, real-time contextual assistance, and adaptive learning activities. A preliminary study ($n=24$) assessed the effectiveness and usability of \tool{} and informed refinements in our system, which we then evaluated ($n = 16$) against a combination of a state-of-the-art online learning platform and an LLM for learning support. Results indicate that \tool{} advances AI pedagogy by improving both learning outcomes and user experience compared to existing online learning and support tools. This work advances our understanding of the design space of personalized, AI-driven educational tools and their potential impact on user experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06232v1">Profiling for Pennies: Unveiling the Privacy Iceberg of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 22 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized how information are collected, aggregated, and reasoned. However, this enables a novel and accessible vector of privacy intrusion: the automated and in-depth personal profiling; this engenders a chilling effect of "peepers everywhere". Existing research primarily unfolds from the training pipeline of LLM, emphasizing the exposure of Personally Identifiable Information (PII) through memorization, while privacy studies from a human-centric perspective remain underexplored. To fill this void, we empirically investigate privacy perception in the real world through the lens of human awareness and the practices of LLM-integrated platforms, revealing a significant dissonance: platforms fail to technically or policy-wise address public privacy concerns. To facilitate a systematic and quantifiable study of privacy risk, we propose the PrivacyIceberg, which categorizes real-world human privacy risks into three tiers: explicitly searched, contextually inferred, and deeply aggregated, based on the sophistication of LLM exploitation. We developed IcebergExplorer to audit privacy exposure, utilizing minimal PII as a search seed to reconstruct high-fidelity profiles, achieving over 90% factual accuracy within 10 minutes at a cost under $3, for real-world scenarios. Additionally, we identify six root causes contributing to such privacy disclosures and propose multi-stakeholder countermeasures for LLM vendors, individuals, and data publishers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06225v1">Memory Inception: Latent-Space KV Cache Manipulation for Steering LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Steering large language models (LLMs) is usually done by either instruction prompting or activation steering. Prompting often gives strong control, but caches guidance tokens at every layer and can clutter long interactions; activation steering is compact but typically weaker and does not support large structured reminders. We introduce memory inception (MI), a training-free method that steers in latent attention space by inserting text-derived key-value (KV) banks only at selected layers. Rather than materializing reminder content throughout the prompt cache, MI treats steering as selective KV allocation, injecting latent slots only where the model routes to them. On matched personality-steering tasks, MI gives the best overall control--drift trade-off, remaining competitive with prompting while consistently outperforming CAA. On updateable guidance, MI supports mid-conversation behavior shifts without rewriting the visible transcript, achieving the highest post-shift alignment on Qwen3. On structured reasoning, MI outperforms visible prompting on HARDMath and PHYSICS (10/12 subject$\times$mode cells), serving as proxies for structured reasoning in verifiable domains, while cutting content-matched KV storage by up to 118$\times$. These results position MI as a powerful steering method when guidance is persistent, structured, or expensive to keep in the visible transcript.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.15270v2">From Documents to Spans: Scalable Supervision for Evidence-Based ICD Coding with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      International Classification of Diseases (ICD) coding assigns diagnosis codes to clinical documents and is essential for healthcare billing and clinical analysis. Reliable coding requires that each predicted code be supported by explicit textual evidence. However, existing public datasets provide only code labels, without evidence annotations, limiting models' ability to learn evidence-grounded predictions. In this work, we argue that dense, document-level evidence annotation is not always necessary for learning evidence-based coding. Instead, models can learn code-specific evidence patterns from local spans and use these patterns to support document-level evidence-based coding. Based on this insight, we propose Span-Centric Learning (SCL), a training framework that strengthens LLMs' coding ability at the span level and transfers this capability to full clinical documents. Specifically, we use a small set of annotated documents to supervise evidence recognition, aggregation, and code assignment, while leveraging a large collection of lightweight evidence spans to reinforce span-level reasoning. Due to their compactness, span annotations are scalable and can be further augmented through synthesis. Under the same Llama3.1-8B backbone, our approach achieves an 8.2-point improvement in macro-F1 at only 20% of the training cost of standard SFT, and provides explicit supporting evidence for each predicted code, enabling human auditing and revision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06205v1">ClawGuard: Out-of-Band Detection of LLM Agent Workflow Hijacking via EM Side Channel</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Autonomous LLM agents face a critical security risk known as workflow hijacking, where attackers subtly alter tool and skill invocations. Existing defenses rely on host-internal telemetry (such as audit logs), which can be forged if the host OS is compromised. To solve this, we introduce ClawGuard, a passive, out-of-band monitor that audits LLM-agent workflows using electromagnetic (EM) emanations. Because distinct agent skills create unique hardware usage patterns (computation, DRAM, network blocking), they emit measurable, macroscopic EM envelopes. External software-defined radios (SDRs) capture these physical signals. Using a drift-aware pipeline with 320-dimensional features, ClawGuard converts RF streams into physical evidence. Evaluated on a 7.82TB RF corpus, ClawGuard achieved an AUC of 0.9945, detecting attacks with a 100% true-positive rate and a 1.16% false-positive rate. This proves passive EM sensing is a practical, forge-resistant physical check against compromised host software.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06721v2">ProAgent: Harnessing On-Demand Sensory Contexts for Proactive LLM Agent Systems in the Wild</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Recent studies have begun to explore proactive large language model (LLM) agents that provide unobtrusive assistance by automatically leveraging contextual information, such as in code editing and in-app suggestions. However, most focus on short, task-specific episodes or on-screen contexts, rather than continuously perceiving and assisting users throughout daily life. Enabling such in-the-wild assistance requires continuous sensing of users' surroundings, which can incur substantial system overhead. In this work, we propose ProAgent, an end-to-end proactive agent system that harnesses on-demand sensory contexts to provide in-the-wild assistance. ProAgent first employs on-demand tiered perception to continuously sense users' surroundings by integrating low-cost contextual cues with richer perception on demand, and uses proactive-oriented context extraction to derive hierarchical contexts integrating both sensory contexts and human preferences. ProAgent then employs a context-aware proactive reasoner to infer user needs and invokes external tools to deliver proactive assistance. We implement ProAgent on AR glasses and evaluate it on a public dataset and a real-world dataset. Results demonstrate that ProAgent achieves up to 27.7% higher proactive prediction accuracy and 20.5% lower false detection than state-of-the-art baselines. A user study with 20 participants shows that 85% were satisfied with ProAgent and willing to use it in daily life.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06166v1">One Algorithm, Two Goals: Dual Scoring for Parameter and Data Selection in LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      In Large Language Model (LLM) fine-tuning, parameter and data selection are common strategies for reducing fine-tuning cost, yet they are typically driven by separate scoring mechanisms. When a parameter mask and data subset jointly determine restricted fine-tuning, this separation incurs redundant overhead and makes coordinated selection difficult. We cast parameter and data selection as two bilevel selection problems under a common validation objective and derive a shared local response-surrogate scoring rule. Under first- and second-order validation-improvement approximations, parameter importance and data utility emerge as column-wise and row-wise aggregations of a single gradient interaction matrix, yielding a closed-form row-column correspondence for co-extracting both signals. Building on this structure, we propose DualSFT (Dual-Selection Fine-Tuning), a one-shot dual-scoring algorithm that produces a parameter mask and data subset from shared gradient statistics. On 3B-9B LLMs, single-axis DualSFT variants strengthen target-task performance and stability-plasticity trade-offs within their comparison groups, while full DualSFT yields a more favorable joint-constrained trade-off than sequential hybrid baselines under matched budgets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06161v1">Beyond Accuracy: Policy Invariance as a Reliability Test for LLM Safety Judges</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge pipelines have become the de facto evaluator for agent safety, yet existing benchmarks treat their verdicts as ground-truth proxies without checking whether the verdicts depend on the agent's behavior or merely on how the evaluation policy happens to be worded. We argue that any trustworthy safety judge must satisfy a basic property we call policy invariance, and we operationalize it as three testable principles: rubric-semantics invariance under certified-equivalent rewrites, rubric-threshold invariance under intentional strict-to-lenient shifts, and ambiguity-aware calibration so that verdict instability concentrates on genuinely ambiguous cases. Instantiating these principles as a stress-test protocol with four agent-class judges on trajectories drawn from ASSEBench and R-Judge, we surface a previously unmeasured failure mode: today's judges respond to meaningful normative shifts and to meaningless structural rewrites with comparable strength, and cannot tell the two apart. Content-preserving policy rewrites flip up to 9.1% of verdicts above baseline jitter, and 18-43% of all observed flips occur on unambiguous cases under such rewrites, so existing safety scores conflate what the agent did with how the evaluator was prompted. Beyond the diagnosis, we contribute the Policy Invariance Score and the Judge Card reporting protocol, which expose an order-of-magnitude spread in judge reliability that is invisible to accuracy-only leaderboards. We release the protocol and code so that future agent-safety benchmarks can audit their own evaluators rather than trust them by default.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06139v1">Listwise Policy Optimization: Group-based RLVR as Target-Projection on the LLM Response Simplex</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Reinforcement learning with verifiable rewards (RLVR) has become a standard approach for large language models (LLMs) post-training to incentivize reasoning capacity. Among existing recipes, group-based policy gradient is prevalent, which samples a group of responses per prompt and updates the policy via group-relative advantage signals. This work reveals that these optimization strategies share a common geometric structure: each implicitly defines a target distribution on the response simplex and projects toward it via first-order approximation. Building on this insight, we propose Listwise Policy Optimization (LPO) to explicitly conduct the target-projection, which demystifies the implicit target by restricting the proximal RL objective to the response simplex, and then projects the policy via exact divergence minimization. This framework provides (i) monotonic improvement on the listwise objective with bounded, zero-sum, and self-correcting projection gradients, and (ii) flexibility in divergence selection with distinct structural properties through the decoupled projection step. On diverse reasoning tasks and LLM backbones, LPO consistently improves training performance over typical policy gradient baselines under matched targets, while intrinsically preserving optimization stability and response diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06123v1">Back to the Beginning of Heuristic Design: Bridging Code and Knowledge with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 75 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently advanced automatic heuristic design (AHD) for combinatorial optimization (CO), where candidate heuristics are iteratively proposed, evaluated, and refined. Most existing approaches search over executable programs and distill insights from execution feedback to guide later iterations. Because this process moves from low-level implementations to high-level principles, we refer to it as a bottom-up paradigm. We argue that this view is incomplete and introduce a complementary top-down perspective: knowledge becomes the primary search object and code merely instantiates and tests it, making what is learned explicit and reusable across problems and trajectories. We formalize this shift through a statistical-learning view that exposes a distortion--compression trade-off, and instantiate it in both population-based and tree-based AHD frameworks. Across CO and tasks beyond it, knowledge-first search improves discovery efficiency, transfer, and generalization, often outperforming code-centric pipelines, while combining both strategies yields further gains. Our results suggest that progress in AHD depends on iteratively constructing and evolving interpretable hypotheses that retain value beyond a single search trajectory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06117v1">BoostLLM: Boosting-inspired LLM Fine-tuning for Few-shot Tabular Classification</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 19 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently been adapted to tabular prediction by serializing structured features into natural language, but their performance in low-data regimes remains limited compared to gradient-boosted decision trees (GBDTs). In this work, we revisit the boosting paradigm, traditionally associated with tree ensembles, and ask whether it can be applied as a general training principle for LLM fine-tuning. We propose BoostLLM, a framework that transforms parameter-efficient fine-tuning into a multi-round residual optimization process by training sequential PEFT adapters as weak learners. To incorporate tabular inductive bias, BoostLLM integrates decision-tree paths as a second input view alongside raw features; analysis reveals that the path view acts as a structured teacher in early training steps before the model shifts toward feature-driven representations. Empirically, BoostLLM achieves consistent improvements over standard fine-tuning across multiple LLM backbones and datasets, matching or surpassing XGBoost across a wide range of shot counts and outperforming GPT-4o-based methods with a 4B model. We further show that the framework scales: pairing with stronger tree models and extended boosting horizons yields additional gains under appropriate stabilization. These results suggest that boosting can serve as a general training principle for LLM fine-tuning, particularly in low-data regimes for structured data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06113v1">Tackling the Data-Parallel Load Balancing Bottleneck in LLM Serving: Practical Online Routing at Scale</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 29 pages,14 figures
    </div>
    <details class="paper-abstract">
      Data-parallel (DP) load balancing has emerged as a first-order bottleneck in large-scale LLM serving. When a model is sharded across devices via tensor parallelism (TP) or expert parallelism (EP) and replicated across many DP workers, every decode step ends in a synchronization barrier whose latency is set by the most heavily loaded worker; even modest persistent imbalance across DP workers compounds, step after step, into a substantial fraction of wasted compute. The problem is hard for reasons specific to LLM decoding: assignments are sticky (migrating KV caches has a high cost), per-request loads grow over time, arrivals are non-stationary, and the router must decide within a sub-100\,ms decode budget over hundreds of waiting requests and tens of workers. We present \textbf{BalanceRoute}, a family of practical online routing algorithms that target this bottleneck. The first, \textbf{BR-0}, requires no prediction infrastructure and uses a piecewise-linear F-score that captures the sharp asymmetry between admissions that fill safe margin and those that overflow into the envelope; a two-stage decomposition keeps per-step cost compatible with millisecond-scale scheduling. The second, \textbf{BR-H}, generalizes BR-0 with a short, constant lookahead $H$ and a lightweight termination-classifier interface, extending the F-score to a horizon-discounted form. We deploy BalanceRoute on a 144-NPU cluster and evaluate against vLLM baselines on both a proprietary production trace and the public Azure-2024 trace. Across both workloads, BalanceRoute substantially reduces average DP imbalance and improves end-to-end serving throughput.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06111v1">Schedule-and-Calibrate: Utility-Guided Multi-Task Reinforcement Learning for Code LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) with verifiable rewards has proven effective at post-training LLMs for coding, yet deploying separate task-specific specialists incurs costs that scale with the number of tasks, motivating a unified multi-task RL (MTRL) approach. However, existing MTRL methods treat all coding tasks uniformly, relying on fixed data curricula under a shared optimization strategy, ultimately limiting the effectiveness of multi-task training. To address these limitations, we propose ASTOR, a multi-tASk code reinforcement learning framework via uTility-driven coORdination. Centered on task utility, a signal capturing each task learning potential and cross-task synergy, ASTOR comprises two coupled modules: 1) Hierarchical Utility-Routed Data Scheduling module hierarchically allocates training budget and prioritizes informative prompts, steering training toward the most valuable data and 2) Adaptive Utility-Calibrated Policy Optimization module dynamically scales per-task KL regularization, matching update constraints to each tasks current training state. Experiments on two widely-used LLMs across four representative coding tasks demonstrate that ASTOR consistently improves a single model across all tasks, outperforming the best task-specific specialist by 9.0%-9.5% and surpassing the strongest MTRL baseline by 7.5%-12.8%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06076v1">Navigating by Old Maps: The Pitfalls of Static Mechanistic Localization in LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 26 pages
    </div>
    <details class="paper-abstract">
      The "Locate-then-Update" paradigm has become a predominant approach in the post-training of large language models (LLMs), identifying critical components via mechanistic interpretability for targeted parameter updates. However, this paradigm rests on a fundamental yet unverified assumption: can mechanisms derived from current static parameters reliably guide future dynamic parameter updates? To investigate this, we systematically track the structural evolution of Transformer circuits throughout the supervised fine-tuning (SFT) process, revealing the underlying dynamics of task mechanisms. We introduce three novel metrics-Circuit Distance, Circuit Stability, and Circuit Conflict-to analyze circuit evolution across three dimensions: neural migration, semantic stability, and cross-task interference. Our empirical results reveal that circuits inherently exhibit "Free Evolution" during parameter updates. Consequently, static mechanisms extracted from current states inevitably suffer from temporal latency, making them fundamentally inadequate for guiding future states. Moreover, by deconstructing the "illusion of effectiveness" in existing methods, this work underscores the necessity of "foresight" in mechanistic localization and proposes a predictive framework for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06068v1">VibeServe: Can AI Agents Build Bespoke LLM Serving Systems?</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      For years, we have built LLM serving systems like any other critical infrastructure: a single general-purpose stack, hand-tuned over many engineer-years, meant to support every model and workload. In this paper, we take the opposite bet: a multi-agent loop that automatically synthesizes bespoke serving systems for different usage scenarios. We propose VibeServe, the first agentic loop that generates entire LLM serving stacks end-to-end. VibeServe uses an outer loop to plan and track the search over system designs, and an inner loop to implement candidates, check correctness, and measure performance on the target benchmark. In the standard deployment setting, where existing stacks are highly optimized, VibeServe remains competitive with vLLM, showing that generation-time specialization need not come at the cost of performance. More interestingly, in non-standard scenarios, VibeServe outperforms existing systems by exploiting opportunities that generic systems miss in six scenarios involving non-standard model architectures, workload knowledge, and hardware-specific optimizations. Together, these results suggest a different point in the design space for infrastructure software: generation-time specialization rather than runtime generality. Code is available at https://github.com/uw-syfi/vibe-serve.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06054v1">Visual Fingerprints for LLM Generation Comparison</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 Submitted to the Short Paper track at IEEE VIS 2026
    </div>
    <details class="paper-abstract">
      Large language model (LLM) outputs arise from complex interactions among prompts, system instructions, model parameters, and architecture. We refer to specific configurations of these factors as generation conditions, each of which can bias outputs in various ways. Understanding how different generation conditions shape model behaviors is essential for tasks such as prompt design and model evaluation, yet it remains challenging due to the stochastic and open-ended nature of text generation. We present an approach to visually compare LLM outputs across generation conditions by modeling responses as collections of linguistic choices, including content, expression, and structure. We extract these choices using natural language processing pipelines and represent their distributions across repeated samples. We then visualize these distributions as visual fingerprints, enabling direct, distribution-level comparison of condition-specific tendencies. Through four usage scenarios, we demonstrate how visual fingerprints reveal consistent patterns in LLM behavior that are difficult to observe through individual responses or aggregate metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06052v1">XtraMAC: An Efficient MAC Architecture for Mixed-Precision LLM Inference on FPGA</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 Accepted to ISCA 2026. 14 pages, 14 figures
    </div>
    <details class="paper-abstract">
      The widespread adoption of mixed-precision quantization in large language models (LLMs) has created demand for hardware that can efficiently perform multiply-accumulate (MAC) operations across mixed datatypes and switch datatypes at runtime. Existing FPGA-based MAC solutions fall short due to limitations in fixed-datatype design, inefficient spatial or temporal resource sharing, and poor support for mixed-precision execution. These limitations collectively lead to under-utilization of DSP resources, limiting achievable parallelism and throughput. In this work, we present XtraMAC, a novel MAC architecture that unifies integer, floating-point, and mixed-precision operations within a single, datatype-adaptive microarchitecture. XtraMAC decomposes all supported MAC formats into a shared integer mantissa product with lightweight sign and exponent handling, enabling dynamic operand packing and efficient DSP resource sharing with constant latency and initiation interval of one across all datatypes. Evaluated on an AMD Xilinx U55c FPGA, XtraMAC achieves 1.4-2.0x higher compute density, reduces per-operation LUT, FF, and DSP consumption by 27-51%, and delivers up to 1.9x greater energy efficiency and 1.2x speedup on representative mixed-precision LLM workloads. The implementation of XtraMAC is open-sourced at https://github.com/Xtra-Computing/XtraMAC.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06040v1">Novelty-based Tree-of-Thought Search for LLM Reasoning and Planning</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Although advances such as chain-of-thought, tree-of-thought or reinforcement learning have improved the performance of LLMs in reasoning and planning tasks, they are still brittle and have not achieved human-level performance in many domains, and often suffer from high time and token costs. Inspired by the success of width-based search in planning, we explore how the concept of novelty can be transferred to language domains and how it can improve tree-of-thought reasoning. A tree of thoughts relies on building possible "paths" of consecutive ideas or thoughts. These are generated by repeatedly prompting an LLM. In our paper, a measurable concept of novelty is proposed that describes the uniqueness of a new node (thought) in comparison to nodes previously seen in the search tree. Novelty is estimated by prompting an LLM and making use of embedded general knowledge from pre-training. This metric can then be used to prune branches and reduce the scope of the search. Although this method introduces more prompts per state, the overall token cost can be reduced by pruning and reducing the overall tree size. This procedure is tested and compared using several benchmarks in language-based planning and general reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06030v1">More Aligned, Less Diverse? Analyzing the Grammar and Lexicon of Two Generations of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      This study contributes to a growing line of research in comparing LLM-generated texts with human-authored text, in this case, English news text. We focus in particular on the evaluation of syntactic properties through formal grammar frameworks. Our analysis compares two generations of LLMs in the context of two human-authored English news datasets from two different years. Employing the Head-Driven Phrase Structure Grammar (HPSG) formalism, we investigate the distributions of syntactic structures and lexical types of AI-generated texts and contrast them with the corresponding distributions in the human-authored New York Times (NYT) articles. We use diversity metrics from ecology and information theory to quantify variation in grammatical constructions and lexical types. We show that English news text has changed little in the given time frame, while newer LLMs display reduced syntactic and, especially, lexical diversity compared to older, non-instruction-tuned models. These findings point to future work in studying effects of instruction tuning, which, while enhancing coherence and adherence to prompts, may narrow the expressive range of model output.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06024v1">Strat-LLM: Stratified Strategy Alignment for LLM-based Stock Trading with Real-time Multi-Source Signals</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 Accepted by the 2026 International Joint Conference on Neural Networks (IJCNN)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are evolving into autonomous trading agents, yet existing benchmarks often overlook the interplay between architectural reasoning and strategy consistency. We propose Strat-LLM, a framework grounded in Stratified Strategy Alignment. Operating in a live-forward setting throughout 2025, it integrates heterogeneous data including sequential prices, real-time news, and annual reports to eliminate look-ahead bias. Extensive stress tests on A-share and U.S. markets reveal: (1) reasoning-heavy models achieve peak utility in Free Mode via internal logic, whereas standard models require Strict Mode as a vital risk anchor; (2) alignment utility is regime-dependent, with Free and Guided modes capturing momentum in uptrending markets, while Strict Mode mitigates drawdowns in downtrends; (3) mid-scale models (35B) show optimal fidelity under strict constraints, whereas ultra-large models (122B) suffer an alignment tax under rigid rules but gain a performance premium in Guided Mode; (4) standard LLMs often fall into a high win-rate trap, optimizing for small gains at the expense of total returns, which can only be mitigated through deep reasoning or strict external guardrails. Project details are available at https://Strat-LLM.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.10013v2">Training LLMs on HPC Systems: Best Practices from the OpenGPT-X Project</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      The training of large language models (LLMs) requires substantial computational resources, complex software stacks, and carefully designed workflows to achieve scalability and efficiency. This report presents best practices and insights gained from the OpenGPT-X project, a German initiative focused on developing open, multilingual LLMs optimized for European languages. We detail the use of high-performance computing (HPC) systems, primarily JUWELS Booster at JSC, for training Teuken-7B, a 7-billion-parameter transformer model. The report covers system architecture, training infrastructure, software choices, profiling and benchmarking tools, as well as engineering and operational challenges. It includes measured throughput data of various configurations of 3D parallelism during training and the impact of features such as flash attention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04913v2">Rethinking Local Learning: A Cheaper and Faster Recipe for LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 33pages
    </div>
    <details class="paper-abstract">
      LLM post-training typically propagates task gradients through the full depth of the model. Although this end-to-end structure is simple and general, it couples task adaptation to full-depth activation storage, long-range backward dependencies and direct task-gradient access to pretrained representations. We argue that this full-depth backward coupling can be unnecessarily expensive and intrusive, particularly when post-training supervision is much narrower than pre-training. To this end, we propose \textbf{LoPT}: Local-Learning Post-Training, a simple post-training strategy that makes gradient reach an explicit design choice. LoPT places a single gradient boundary at the transformer midpoint: the second-half block learns from the task objective, while the first-half block is updated by a lightweight feature-reconstruction objective to preserve useful representations and maintain interface compatibility. LoPT shortens the task-induced backward path while limiting direct interference from narrow task gradients on early-layer representations. Extensive experiments demonstrate that LoPT achieves competitive performance with lower memory cost, higher training efficiency and better retention of pretrained capabilities. Our code is available at: https://github.com/HumyuShi/LoPT
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05097v2">Continual Knowledge Updating in LLM Systems: Learning Through Multi-Timescale Memory Dynamics</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 Preprint. 9 pages, 2 figures
    </div>
    <details class="paper-abstract">
      LLMs are trained once, then deployed into a world that never stops changing. External memory compensates for this, but most systems manage it explicitly rather than letting it adapt on its own. Biological memory works differently: coupled multi-timescale dynamics make new associations immediately usable, strengthen what repetition confirms, and let the rest fade. We argue that external memory should follow a similar principle. In Memini, this view takes the form of an associative memory that organizes knowledge as a directed graph. Each edge carries two coupled internal variables, one fast and one slow, following the Benna-Fusi model of synaptic consolidation. From this coupling, episodic sensitivity, gradual consolidation, and selective forgetting emerge as facets of a single mechanism, reframing external memory as a learning substrate that reorganizes through its own dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05973v1">Towards Reliable LLM Evaluation: Correcting the Winner's Curse in Adaptive Benchmarking</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Adaptive prompt and program search makes LLM evaluation selection-sensitive. Once benchmark items are reused inside tuning, the observed winner's score need not estimate the fresh-data performance of the full tune-then-deploy procedure. We study inference for this procedure-level target under explicit tuning budgets. We propose SIREN, a selection-aware repeated-split reporting protocol that freezes the post-search shortlist, separates splitwise selection from held-out evaluation, and uses an item-level Gaussian multiplier bootstrap for uncertainty quantification. In a fixed-shortlist regime with smooth stabilized selection, the estimator admits a first-order item-level representation, and the bootstrap yields valid simultaneous inference on a finite budget grid. This supports confidence intervals for procedure-performance curves and pre-specified equal-budget and cross-budget comparisons. Controlled simulations and MMLU-Pro tuning experiments show that winner-based reporting can be optimistic and can change deployment conclusions, while SIREN remains close to the finite-sample reporting target.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05969v1">Heimdallr: Characterizing and Detecting LLM-Induced Security Risks in GitHub CI Workflows</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      GitHub Continuous Integration (CI) workflows increasingly integrate Large Language Models (LLMs) to automate review, triage, content generation, and repository maintenance. This creates a new attack surface: externally controllable workflow inputs can shape LLM prompts and outputs, which may in turn affect security decisions, repository state, or privileged execution. Although LLM security and CI security have each been studied extensively, their intersection remains underexplored. In this paper, we present the first study of LLM-induced security risks in GitHub CI workflows. We characterize the problem along the full execution chain and develop a taxonomy of high-level risk classes and concrete threat vectors. To detect such risks in practice, we design Heimdallr, a hybrid analysis framework that normalizes workflows into an LLM-Workflow Property Graph (L-WPG) and combines triggerability analysis, LLM-assisted dataflow summarization, and deterministic propagation to synthesize concrete threat-vector findings. Evaluated on 300 manually annotated unique workflows, Heimdallr achieves high accuracy on LLM-node identification (F1~=~0.994), triggerability classification (99.8%), and threat-vector detection (micro-average F1~=~0.917). As part of an ongoing detection and disclosure effort, we have so far responsibly disclosed 802 vulnerable workflow instances across 759 repositories and received 71 acknowledgments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.07830v2">Time Series Reasoning via Process-Verifiable Thinking Data Synthesis and Scheduling for Tailored LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Time series is a pervasive data type across various application domains, rendering the reasonable solving of diverse time series tasks a long-standing goal. Recent advances in large language models (LLMs), especially their reasoning abilities unlocked through reinforcement learning (RL), have opened new opportunities for tackling tasks with long Chain-of-Thought (CoT) reasoning. However, leveraging LLM reasoning for time series remains in its infancy, hindered by the absence of carefully curated time series CoT data for training, limited data efficiency caused by underexplored data scheduling, and the lack of RL algorithms tailored for exploiting such time series CoT data. In this paper, we introduce VeriTime, a framework that tailors LLMs for time series reasoning through data synthesis, data scheduling, and RL training. First, we propose a data synthesis pipeline that constructs a TS-text multimodal dataset with process-verifiable annotations. Second, we design a data scheduling mechanism that arranges training samples according to a principled hierarchy of difficulty and task taxonomy. Third, we develop a two-stage reinforcement finetuning featuring fine-grained, multi-objective rewards that leverage verifiable process-level CoT data. Extensive experiments show that VeriTime substantially boosts LLM performance across diverse time series reasoning tasks. Notably, it enables compact 3B, 4B models to achieve reasoning capabilities on par with or exceeding those of larger proprietary LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05957v1">Knowing but Not Correcting: Routine Task Requests Suppress Factual Correction in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      LLMs reliably correct false claims when presented in isolation, yet when the same claims are embedded in task-oriented requests, they often comply rather than correct. We term this failure mode \emph{correction suppression} and construct a benchmark of 300 false premises to systematically evaluate it across eight models. Suppression rates range from 19\% to 90\%, with four models exceeding 80\%, establishing correction suppression as a prevalent and severe phenomenon. Mechanistic analysis reveals that suppression is not a knowledge failure: the model registers the error internally but task context diverts early-layer attention from the false claim as output intent crystallizes toward compliance at middle layers. We characterize this as \emph{knowing but not correcting} -- suppression occurs at response selection rather than knowledge encoding. Guided by this mechanism, we propose two training-free interventions. Correction Direction Steering (CDS) estimates a correction-compliance direction from matched pairs and injects it at middle layers before output intent crystallizes. Dynamic Payload Amplification (DPA) localizes payload tokens via attention divergence between early and late layers and amplifies their representation at the final layer, requiring no calibration data. Experiments on Qwen3.5-9B and LLaMA3.1-8B show both methods substantially improve factual strictness. CDS achieves the highest correction rate on Qwen3.5-9B (0\%$\to$58.2\%). DPA is the only method that preserves or improves reasoning capability on both models. These findings introduce \emph{factual strictness} -- the willingness to uphold accuracy against contextual pressures -- as a new dimension of model reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.16155v2">PRIMETIME : Limits of LLMs in Temporal Primitives</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      This paper introduces PRIMETIME, a synthetic generator that supports both benchmarking and fine-tuning of two primitive operations underlying temporal reasoning in Large Language Models (LLMs): parsing and arithmetic on datetimes. Existing temporal benchmarks assume simplified canonical datetime forms, conflate arithmetic, composition, and world knowledge into a single aggregate score, and offer no direct path to remediation. The first contribution is methodological: the PRIMETIME synthetic generator delivers non-conflated, uncontaminated, and unlimited datetime exemplars that enable a decompositional evaluation strategy for each primitive in isolation. The generator is extensible to support complex datetime tasks and is publicly released, alongside generated benchmarks. The second contribution is diagnostic: under this evaluation strategy, the primitives themselves prove individually unreliable, with per-primitive accuracy ranging from near-zero to perfect across models and prompting conditions. The third contribution is constructive: the same generator used for diagnosis also produces new training exemplars for fine-tuning, and the resulting models show that the primitives are fully learnable and the composed Event Planning task reaches frontier-level accuracy using small quantized LoRA transformers. The broader takeaway is that a single synthetic generator can serve both diagnosis and production-ready deployment. This methodological pattern may apply beyond temporal reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05950v1">Lightweight Stylistic Consistency Profiling: Robust Detection of LLM-Generated Textual Content for Multimedia Moderation</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      The increasing prevalence of Large Language Models (LLMs) in content creation has made distinguishing human-written textual content from LLM-generated counterparts a critical task for multimedia moderation. Existing detectors often rely on statistical cues or model-specific heuristics, making them vulnerable to paraphrasing and adversarial manipulations, and consequently limiting their robustness and interpretability. In this work, we proposeLiSCP , a novel lightweight stylistic consistency profiling method for robust detection of LLM-generated textual content, focusing on feature stability under adversarial manipulation. Our approach constructs a consistency profile that combines discrete stylistic features with continuous semantic signals, leveraging stylistic stability across multimodal-guided paraphrased text variants. Experiments spanning real-world multimedia news and movie datasets and conventional text domains demonstrate that LiSCP achieves superior performance on in-domain detection and outperforms existing approaches by up to 11.79% in cross-domain settings. Additionally,it demonstrates notable robustness under adversarial scenarios, including adversarial attacks and hybrid human-AI settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05933v1">Whole-body CT attenuation and volume charts from routine clinical scans via evidence-grounded LLM report filtering</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 Supplement available at: https://github.com/ai-med/body-charts/blob/main/body_charts_supp.pdf
    </div>
    <details class="paper-abstract">
      Interpreting quantitative CT biomarkers, such as organ volume and tissue attenuation, requires large-scale healthy reference distributions. However, creating these is challenging because clinical datasets are often heavily enriched with pathology. Here, we develop an evidence-grounded, cross-verified large language model (LLM) ensemble to filter pathological findings from radiology reports, enabling the construction of pathology-reduced cohorts from over 350,000 CT examinations. Five LLMs, first, flag structure-level abnormality candidates grounded in verbatim report evidence and, second, resolve disagreements via cross-verification. Using distribution-aware generalized additive models for location, scale, and shape, we establish comprehensive whole-body reference charts for 106 anatomical structures (volumes and attenuation) across adulthood, accounting for age, sex, contrast enhancement, and acquisition parameters. Longitudinal analyses reveal structure- and contrast-dependent changes distinct from cross-sectional trends. These resources facilitate covariate-adjusted centile scoring from routine CT, supporting standardized quantitative phenotyping, multi-site imaging studies, and scalable opportunistic screening research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05927v1">Minimizing Modality Gap from the Input Side: Your Speech LLM Can Be a Prosody-Aware Text LLM</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Speech large language models (SLMs) are typically built from text large language model (TLM) checkpoints, yet they still suffer from a substantial modality gap. Prior work has mainly attempted to reduce this gap from the output side by making speech generation more text-like, but the gap remains. We argue that the key remaining bottleneck lies on the input side. We propose TextPro-SLM, an SLM that makes spoken input more closely resemble that of a prosody-aware text LLM. TextPro-SLM combines WhisperPro, a unified speech encoder that produces synchronized text tokens and prosody embeddings, with an LLM backbone trained to preserve the semantic capabilities of the original TLM while learning paralinguistic understanding. Experiments show that TextPro-SLM achieves the lowest modality gap among leading SLMs at both 3B and 7B scales, while also delivering strong overall performance on paralinguistic understanding tasks. These gains are achieved with only roughly 1,000 hours of LLM training audio, suggesting that reducing the modality gap from the input side is both effective and data-efficient.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05920v1">LLM-Driven Design Space Exploration of FPGA-based Accelerators</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 Accepted to the Workshop on Intelligent System Design (InSyDe) co-located with EuroSys '26
    </div>
    <details class="paper-abstract">
      Designing field-programmable gate array (FPGA)-based accelerators for modern artificial intelligence workloads requires navigating a large and complex hardware design space encompassing architectural parameters, dataflow strategies, and memory hierarchies, making the process time-consuming and resource-intensive. While the SECDA methodology enables rapid hardware-software co-design of accelerators through SystemC simulation and FPGA execution, identifying optimal accelerator configurations still requires substantial manual effort and domain expertise. This work presents SECDA-DSE, a framework that integrates Large Language Models (LLMs) into the SECDA ecosystem, comprising tools built around SECDA to automate the design space exploration (DSE) of FPGA-based accelerators. SECDA-DSE combines a structured DSE Explorer for generating accelerator configurations with an LLM Stack that performs reasoning-guided exploration using retrieval-augmented generation and chain-of-thought prompting, alongside a feedback loop that enables reinforced fine-tuning for continuous improvement. We demonstrate the feasibility of SECDA-DSE through an initial high-level synthesis based evaluation of a generated accelerator design that meets synthesis timing and resource constraints on an Zynq-7000 FPGA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05893v1">Logic-Regularized Verifier Elicits Reasoning from LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Verifiers are crucial components for enhancing modern LLMs' reasoning capability. Typicalverifiers require resource-intensive superviseddataset construction, which is costly and faceslimitations in data diversity. In this paper, wepropose LOVER, an unsupervised verifier regularized by logical rules. LOVER treats theverifier as a binary latent variable, utilizinginternal activations and enforcing three logical constraints on multiple reasoning paths:negation consistency, intra-group consistency,and inter-group consistency (grouped by thefinal answer). By incorporating logical rulesas priors, LOVER can leverage unlabeled examples and is directly compatible with any offthe-shelf LLMs. Experiments on 10 datasetsdemonstrate that LOVER significantly outperforms unsupervised baselines, achieving performance comparable to the supervised verifier(reaching its 95% level on average). The sourcecode is publicly available at https://github.com/wangxinyufighting/llm-lover.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05873v1">CITE: Anytime-Valid Statistical Inference in LLM Self-Consistency</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Large language models often improve reasoning by sampling multiple outputs and aggregating their final answers, but precise and efficient control of error levels remains a challenging task. In particular, deciding when to stop sampling remains difficult when the stopping rule is data-dependent and the set of possible answers is not known in advance. We study anytime-valid certification of a prespecified target answer as the unique mode of the model's response distribution, a guarantee distinct from answer correctness. We propose the Certification by Intersection-union Testing with E-processes (CITE) algorithm, which provably controls false certification at any prescribed level under arbitrary data-driven stopping, without requiring prior knowledge of the answer category set. We also prove an category-set-size-free stopping-time rate, establish matching minimax lower bounds up to constants in the main regime, and extend the construction to confidence-weighted voting. Simulations and LLM self-consistency experiments show empirical error control and improved certification in diffuse-tail settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05846v1">LoopTrap: Termination Poisoning Attacks on LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Modern LLM agents solve complex tasks by operating in iterative execution loops, where they repeatedly reason, act, and self-evaluate progress to determine when a task is complete. In this work, we show that while this self-directed loop facilitates autonomy, it also introduces a critical risk: by injecting malicious prompts into the agent's context, an adversary can distort the agent's termination judgment, making it believe the task remains incomplete and leading to unbounded computation.To understand this threat, we define and systematically characterize it as Termination Poisoning and design 10 representative attack strategies. Through a empirical study spanning 8 LLM agents and 60 tasks, we demonstrate that different LLM agents exhibit distinct behavioral signatures that determine which strategies succeed. These transferable patterns can serve as principled guidance for crafting effective attacks against previously unseen agents and tasks, enabling scalable red-teaming beyond manually designed templates. Building on these insights, we introduce LoopTrap, an automated red-teaming framework that synthesizes target-specific malicious prompts by exploiting agent behavioral tendencies. LoopTrap first constructs a behavioral profile of the target agent along four vulnerability dimensions via lightweight probing. It then performs adaptive trap synthesis, routing to the most effective strategy and selecting optimal injections via a self-scoring mechanism. Finally, successful traps are abstracted into a reusable skill library, while failed attempts are refined through self-reflection, ensuring continuous improvement. Extensive evaluation shows that LoopTrap achieves an average of 3.57$\times$ step amplification across 8 mainstream agents, with a peak of 25$\times$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05842v1">Taklif.AI: LLM-Powered Platform for Interest-Based Personalized College Assignments</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Educators face significant challenges in creating engaging, personalized assignments that accommodate students' diverse interests and cognitive abilities. Traditional one-size-fits-all assignments frequently lead to decreased student engagement and increased reliance on unethical practices such as plagiarism. To address these challenges, we present Taklif.AI, a platform that leverages Large Language Models (LLMs) to automatically generate personalized assignments tailored to individual student interests. Unlike existing AI-powered educational platforms that personalize based on academic performance metrics alone, Taklif.AI incorporates students' extracurricular interests and cultural contexts into the assignment generation process through a structured prompt engineering pipeline with input and output guardrails. The platform employs a serverless architecture on AWS with Next.js, using Llama 3.3 70B as the primary LLM via LiteLLM for multi-provider load balancing and LangChain for prompt orchestration. We describe the system architecture, the prompt design methodology, and the guardrails framework that ensures output quality. Preliminary user acceptance testing with 68 participants (65 students and 3 educators) indicates positive reception, with 84% of participants rating the personalization feature as beneficial. We discuss the platform's current capabilities and limitations, and outline directions for rigorous empirical evaluation of learning outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05837v1">An Additive Approximation Scheme for Generating Dyadic Codings for the Outputs of an LLM</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      We study the problem of approximating a discrete probability distribution, such as the next-token distribution of a large language model, by a dyadic distribution induced by a binary tree under encoding rate constraints. The objective is to partition the support of the distribution and assign dyadic probabilities to minimize total variation distance while achieving a prescribed rate. We formulate this task as a tree-based partitioning problem and develop a polynomial-time additive approximation scheme for the rate-constrained setting in the constant-rate regime. Our results provide provable guarantees for near-optimal dyadic approximations and, as an application, yield a principled framework for LLM-based steganography, where the rate maps to bits of hidden information embedded per token and the total variation bound controls statistical detectability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04704v2">UVMarvel: an Automated LLM-aided UVM Machine for Subsystem-level RTL Verification</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 This paper has been accepted by DAC 2026 and will appear in the proceedings
    </div>
    <details class="paper-abstract">
      Verification presents a major bottleneck in Integrated Circuit (IC) development, consuming nearly 70% of total effort. While the Universal Verification Methodology (UVM) improves reuse through structured verification environments, constructing subsystem-level UVM testbenches and generating high-quality stimuli still require extensive manual coding, repeated EDA tool runs, and deep protocol and micro-architectural expertise. We present UVMarvel, an automated verification framework that leverages Large Language Models (LLMs) to build UVM testbenches for subsystem-level RTL. UVMarvel introduces an Intermediate Representation (IR) and a Bus Protocol Library to translate heterogeneous specifications into protocol-correct subsystem-level UVM testbenches, and employs a Signal Tracker and a Verilog Patching Library to guide LLM-based stimuli refinement. UVMarvel is the first framework capable of automatically constructing subsystem-level UVM testbenches across mainstream bus protocols, and it achieves an average code coverage of 95.65%, while reducing verification time from several human working days to a 4.5-hour automated execution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05807v1">LCC-LLM: Leveraging Code-Centric Large Language Models for Malware Attribution</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      LLMs are increasingly explored for malware analysis; however, current LLM-based malware attribution remains limited by unsupported indicators and insufficient code-level grounding for identifying malicious and vulnerable code segments. To address these limitations, this research introduces LCC-LLM, a code-centric benchmark dataset and evidence-grounded framework for malware attribution and multi-task static malware analysis. The proposed LCCD dataset contains approximately 34K PE samples processed through a large-scale reverse-engineering pipeline and represented using decompiled C code, assembly code, CFG/FCG artifacts, hexadecimal data, PE metadata, suspicious API evidence, and structural features. Beyond dataset construction, LCC-LLM integrates LangGraph-orchestrated static analysis with multi-source cybersecurity knowledge to support evidence-grounded malware reasoning. The framework employs a seven-layer retrieval-augmented generation pipeline, CoVe for IoC validation, and a multi-dimensional quality gate to improve factual reliability and analyst-oriented decision support. Curriculum-ordered instruction data is used to fine-tune DeepSeek-R1-Distill-Qwen-14B and Qwen3-Coder-30B-A3B using QLoRA. Evaluation across 43 malware-analysis task types achieves an average semantic similarity of 0.634, with the highest task-level performance in structured report generation, IoC extraction, vulnerability assessment, malware configuration extraction, and malware class detection. In a real-world case study using MalwareBazaar samples, the grounded pipeline achieves a 10/10 structured analysis pass rate, producing CFG/FCG evidence, MITRE ATT&CK mappings, detection guidance, and analyst-ready reports. These results show that code-centric representations, retrieval grounding, and verification-guided reasoning improve the reliability and operational usefulness of LLM-assisted malware attribution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05795v1">Reward Shaping and Action Masking for Compositional Tasks using Behavior Trees and LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Decomposing complex tasks into a sequence of simpler subtasks can improve learning efficiency for an autonomous agent. Reinforcement learning (RL) can be used to optimize agent policies to complete subtasks, but requires well-defined subtask rewards and benefits from action masking. Recent work uses large language models (LLMs) to automate reward shaping and action masking, however none of them fully address reactivity to subtask failure and modularity to varying objects for compositional tasks. To overcome these challenges, we develop masking reward behavior tree (MRBT), a symbolic structure used as a reactive and modular reward and action mask function. We design an MRBT template and derive logical specifications to construct and verify MRBTs for a sequence of object-interaction subtasks. Further, we develop an automated pipeline that uses an LLM to generate MRBTs robust to varying task objects, an SMT-solver to verify correctness of specifications, and a neurosymbolic RL loop to train agents on compositional tasks. Experiments demonstrate successful generation and refinement of five MRBTs, consistently improving training efficiency and task success rates over baselines and MRBTs without action masking. We further highlight three advantages of MRBTs: transferability, modularity, and verifiability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05794v1">Revealing Modular Gradient Noise Imbalance in LLMs: Calibrating Adam via Signal-to-Noise Ratio</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      The impressive performance of large language models (LLMs) arises from their massive scale and heterogeneous module composition. However, this structural heterogeneity introduces additional optimization challenges. While adaptive optimizers such as Adam(W) provide per-parameter adaptivity, they do not explicitly account for module-level gradient heterogeneity, resulting in slower convergence, suboptimal performance, or training instability. Existing approaches typically rely on manually tuned module-specific learning rates or specific optimization strategies, which are computationally costly and difficult to generalize across tasks or models. To establish a more principled approach, we first analyze the noise-damping behavior of Adam in high-noise modules and introduce \textbf{Module-wise Learning Rate Scaling via SNR (MoLS)}. MoLS estimates module-level SNRs to scale Adam updates, allowing automated module-wise learning rate allocation without manual tuning. Empirical results through multiple LLM training benchmarks demonstrate that MoLS improves convergence speed and generalization, achieving performance comparable to carefully tuned module-specific learning rates, while remaining compatible with memory-efficient training algorithms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05790v1">GazeMind: A Gaze-Guided LLM Agent for Personalized Cognitive Load Assessment</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Smart glasses with AI assistants are increasingly used in daily life. However, current systems lack awareness of the user's internal cognitive state, leaving them unable to proactively anticipate users' needs without access to cognitive load. Existing methods for assessing cognitive load either rely on impractical sensors for lightweight eyewear or utilize eye gaze-based models that suffer from poor interpretability, and require task-specific fine-tuning, often failing to generalize across individuals. We propose GazeMind, a gaze-guided LLM agent framework for cognitive load assessment on smart glasses. It encodes eye-tracking data into structured representations for LLM-based reasoning and provides interpretable cognitive load predictions. Importantly, GazeMind generalizes across scenarios without LLM fine-tuning through a novel task-guidance reasoning approach and achieves personalized adaptation by incorporating user-specific characteristics and historical references. To support evaluation, we introduce CogLoad-Bench, the largest gaze-based cognitive load dataset with 152 participants, 40+ hours of multimodal data, and 10K+ real-time annotations across controlled and real-world tasks. Experiments show that GazeMind achieves state-of-the-art performance, outperforming baselines by over 20% across all metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.06412v3">Sample-efficient LLM Optimization with Reset Replay</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Recent advancements in LLM post-training, particularly through reinforcement learning and preference optimization, are key to boosting their reasoning capabilities. However, these methods often suffer from low sample efficiency and a susceptibility to primacy bias, a phenomenon where overfitting to initial experiences diminishes network plasticity and damages the learning process. To address these challenges, we introduce LLM optimization with Reset Replay (LoRR), a general and powerful plugin for enhancing sample efficiency in preference-based optimization. Its core mechanism enables high-replay training to maximize the utility of each data batch. To mitigate overfitting, LoRR orchestrates a periodic reset strategy that reuses the initial data and policy to maintain network plasticity, and further adopts a hybrid optimization objective to better exploit training data. Extensive experiments show that LoRR significantly boosts the performance of various preference optimization methods on both mathematical and general reasoning benchmarks. Notably, an iterative DPO framework augmented with LoRR achieves comparable performance on challenging math tasks, rivaling many complex or computationally expensive baselines. Our findings highlight that LoRR offers a practical and sample-efficient paradigm from limited offline data, unlocking greater performance with minimal changes to existing post-training workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05777v1">Estimating the Black-box LLM Uncertainty with Distribution-Aligned Adversarial Distillation</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 Accepted to ACL 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have progressed rapidly in complex reasoning and question answering, yet LLM hallucination remains a central bottleneck that hinders practical deployment, especially for commercial black-box LLMs accessible only via APIs. Existing uncertainty quantification methods typically depend on computationally expensive multiple sampling or internal parameters, which prevents real-time estimation and fails to capture information implicit in the black-box reasoning process. To address this issue, we propose Distribution-Aligned Adversarial Distillation (DisAAD), which introduces a generation-discrimination architecture to guide a lightweight proxy model to learn the high-quality regions of the output distribution of the black-box LLM, thus effectively endowing it with the ability to know whether the black-box LLM knows or not. Subsequently, we use the proxy model to reproduce the specific responses of the black-box LLM and estimate the corresponding uncertainty based on evidence learning. Extensive experiments have verified the effectiveness and promise of our proposed method, indicating that a proxy model even one that only accounts for 1\% of the target LLM's size can achieve reliable uncertainty quantification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05767v1">Priming, Path-dependence, and Plasticity: Understanding the molding of user-LLM interaction and its implications from (many) chat logs in the wild</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      User interactions with LLMs are shaped by prior experiences and individual exploration, but in-lab studies do not provide system designers with visibility into these in-the-wild factors. This work explores a new approach to studying real-world user-LLM interactions through large-scale chat logs from the wild. Through analysis of 140K chatbot sessions from 7,955 anonymized global users over time, we demonstrate key patterns in user expressions despite varied tasks: (1) LLM users are not tabula rasa, nor are they constantly adapting; rather, interaction patterns form and stabilize rapidly through individual early trajectories; (2) Longitudinal outcomes, such as recurring text patterns and retention rates, are strongly correlated with early exploration; (3) Parallel dynamics are present, including organizing expressions by task types such as emotional support, or in response to model-version updates. These results present an ``agency paradox'': despite LLM input spaces being unconstrained and user-driven, we in fact see less user exploration. We call for design consideration surrounding the molding procedure and its incorporation in future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.02821v3">When Is the Same Model Not the Same Service? A Measurement Study of Hosted Open-Weight LLM APIs</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 25 pages, 21 figures; substantially revised abstract, add open-sourced code repo
    </div>
    <details class="paper-abstract">
      Open-weight large language models (LLMs) are usually named as model artifacts, but production users often consume them as hosted API services. This paper argues that the operational unit is a service object: a provider-specific, time-varying endpoint defined by model variant, protocol behavior, context capacity, listed price, latency and throughput distribution, reliability, and task feasibility. Using sampled request logs, provider metadata, compatibility probes, pricing snapshots, and continuous latency measurements collected by AI Ping during Q4 2025, we study how this service layer changes the meaning of "the same model." Three empirical patterns emerge. First, observed demand is concentrated but persistent across versions: in the displayed family aggregate, the largest family carries 32.0% of relative demand and the top five carry 87.4%, with a Gini coefficient of 0.693, while older variants remain active after newer releases. Second, supply and use separate: provider listing breadth does not imply realized adoption, and listed prices are more anchored than latency, throughput, context length, protocol support, and error semantics. Third, task mix matters: applications induce different token-length regimes, so provider choice is a constrained decision over provider-model-task-time tuples rather than a lookup by model name. In two representative counterfactuals under observed feasibility constraints, routing lowers Qwen3-32B cost by 37.8% and raises DeepSeek-V3.2 average throughput by about 90% relative to direct official access. The results support a measurement view of hosted open-weight LLMs as heterogeneous services, not static catalog entries. We open-source the measurement methodology and reproduction artifacts at https://github.com/haoruilee/llm_api_measurement_study to support result reproduction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09298v2">Multi-Modal LLM based Image Captioning in ICT: Bridging the Gap Between General and Industry Domain</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      In the information and communications technology (ICT) industry, training a domain-specific large language model (LLM) or constructing a retrieval-augmented generation system requires a substantial amount of high-value domain knowledge. However, the knowledge is not only hidden in the textual modality but also in the image modality. Traditional methods can parse text from domain documents but dont have image captioning ability. Multi-modal LLM (MLLM) can understand images, but they do not have sufficient domain knowledge. To address the above issues, this paper proposes a multi-stage progressive training strategy to train a Domain-specific Image Captioning Model (DICModel) in ICT, and constructs a standard evaluation system to validate the performance of DICModel. Specifically, this work first synthesizes about 7K image-text pairs by combining the Mermaid tool and LLMs, which are used for the first-stage supervised-fine-tuning (SFT) of DICModel. Then, ICT-domain experts manually annotate about 2K image-text pairs for the second-stage SFT of DICModel. Finally, experts and LLMs jointly synthesize about 1.5K visual question answering data for the instruction-based SFT. Experimental results indicate that our DICModel with only 7B parameters performs better than other state-of-the-art models with 32B parameters. Compared to the SOTA models with 7B and 32B parameters, our DICModel increases the BLEU metric by approximately 56.8% and 20.8%, respectively. On the objective questions constructed by ICT domain experts, our DICModel outperforms Qwen2.5-VL 32B by 1% in terms of accuracy rate. In summary, this work can efficiently and accurately extract the logical text from images, which is expected to promote the development of multimodal models in the ICT domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.14337v2">On the Nature of Attention Sink that Shapes Decoding Strategy in Omni-LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      The goal of this paper is to strengthen the reasoning of Omnimodal Large Language Models (Omni-LLMs) at inference time, without additional training. These models jointly process video, audio, and text, and given the large number of tokens they consume, how attention is routed across them is central to their behaviour. We focus specifically on attention sinks, tokens that absorb a disproportionate share of attention mass regardless of their semantic content, to understand how this routing unfolds. To this end, we conduct a systematic analysis of sink behaviour in Omni-LLMs. Our analysis yields two key findings: (i) high sink attention does not solely indicate head redundancy, suggesting that sink value representations play additional functional roles; (ii) the sink value vector acts as a shared bias added to every token's output, serving as a global signal that organises the representation as a whole. Building on this, we propose OutRo, which correspondingly aligns non-sink token representations with the sink in feature space, and relaxes the causal mask for sink tokens at an early layer to sharpen this bias before the rest of decoding proceeds. This design enhances the reasoning process without requiring additional forward passes or access to attention maps. Based on extensive experiments, OutRo consistently improves performance on seven video QA benchmarks and demonstrates strong generalisation, while incurring only a 1.1x decoding overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05741v1">HyperLens: Quantifying Cognitive Effort in LLMs with Fine-grained Confidence Trajectory</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 33 pages
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) achieve strong performance across diverse tasks, their inference dynamics remain poorly understood because of the limited resolution of existing analysis tools. In this work, we identify an intrinsic magnification mechanism in transformer architectures: deeper layers inherently magnify the small changes of layer-wise confidence, providing a fine-grained confidence trajectory. Building on this insight, we introduce HyperLens, a high-resolution probe designed to trace confidence trajectories and quantify the cognitive effort during inference. Across LLMs and datasets, HyperLens reveals a consistent divergence in confidence trajectories that separates complex from simple tasks. We abstract this pattern into a quantitative cognitive effort metric. Our analysis reveals a fundamental principle: complex tasks consistently require higher cognitive effort. Finally, we provide a mechanistic diagnosis of a common side effect of standard Supervised Fine-Tuning (SFT): it can reduce cognitive effort and consequently degrade performance on in-domain tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05739v1">Multi-Dimensional Behavioral Evaluation of Agentic Stock Prediction Systems Using LLM Judges with Closed-Loop Reinforcement Learning Feedback</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 9 pages, 2 figures, 8 tables. Short Communication submitted to Knowledge-Based Systems (Elsevier)
    </div>
    <details class="paper-abstract">
      Agentic stock prediction systems make sequences of interdependent decisions (regime detection, pathway routing, reinforcement learning control) whose individual quality is hidden by aggregate metrics such as mean absolute percentage error (MAPE) or directional accuracy. We present a behavioral evaluation framework that addresses this gap. Behavioral traces logged at every autonomous decision point are grouped into five-day episodes and scored along six domain-specific dimensions (regime detection, routing, adaptation, risk calibration, strategy coherence, error recovery) by an ensemble of three large language model (LLM) judges (GPT 5.4, Claude 4.6 Opus, Gemini 3.1 Pro). Perturbation-based validation on 420 episodes yields targeted score drops of $-1.6$ to $-2.4$ on intended dimensions versus an average of $-0.32$ on the remaining five, with cross-model agreement up to Krippendorff's $α= 0.85$. The composite behavioral score, used here only for cross-episode reporting, correlates at $ρ= 0.72$ with realized 20-day Sharpe ratio from offline backtesting. Closing the loop, the framework converts deficient per-dimension scores into a credit-assigned penalty term added to the Soft Actor-Critic (SAC) reward. Three short fine-tuning cycles, all confined to the validation period, produce on the held-out 2017-2025 test period a one-day MAPE reduction from 0.61% to 0.54% (an 11.5% relative reduction; $p<0.001$, Cohen's $d=0.31$), a directional accuracy increase from 71% to 74%, and an 18% Sharpe ratio improvement (95% bootstrap CI [8.2%, 27.4%]), with gains concentrated in high-volatility episodes where the original system was most behaviorally deficient. Results are from offline backtesting and do not address effects specific to live deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05737v1">ReFlect: An Effective Harness System for Complex Long-Horizon LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Current reasoning paradigms for LLMs include chain-of-thought, ReAct, and post-hoc self-critique. These paradigms rely on two assumptions that fail on long-horizon, multi-stage tasks. As a result, errors accumulate silently across reasoning steps, leaving an open question: can a reasoning system effectively detect and recover from its own failures? We present ReFlect, a \emph{harness} system for LLM reasoning that creates standalone error detection and recovery logic as a deterministic wrapper around the model. Controlled experiments across 6 reasoning domains show that prompt-level self-critique produces formulaic templates that flag no issues in 90 of 100 audited reflection blocks, and the investigated LLMs wrongly accept a wrong answer in at least 76\% of cases. Our ReFlect harness achieves task success rates ranging from 41\% on gpt-4o-mini to 56\% on Claude Sonnet 4.5 across six models spanning small and frontier scale, with per-model gains over Direct CoT ranging from +7 pp on Qwen2.5-72B to +29 pp on Claude Sonnet 4.5, and additionally raises SWE-bench patch-structural quality from 0\% (Direct CoT) to between 82\% (Qwen2.5-72B) and 87\% (GPT-4o). Notably, the harness gain is inversely proportional to the model's Direct CoT task success rate (the fitted slope is -1.69 with r=-0.76): each pp lost in baseline success rate is mechanically recovered by 1.69 pp of harness gain. We spot that adding structured reasoning state and operators yields only 15.0--18.7\% pair-mean on Llama-3.3-70B and Qwen2.5-72B because models at this scale cannot reliably populate the state its operators require. ReFlect is model-agnostic, training-free, and operates entirely at inference time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05731v1">Knee Osteoarthritis Severity Grading Using Optimized Deep Learning and LLM-Driven Intelligent AI on Computationally Limited Systems</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 6 pages, 11 figures, Accepted and presented at the 2nd International Conference on Emerging Computational Intelligence (ICECI 2026), IEEE. Published in conference proceedings. To appear in IEEE Xplore
    </div>
    <details class="paper-abstract">
      Knee osteoarthritis (KOA) is among the musculoskeletal disorders that considerably restrict joint mobility, cause severe chronic pain and impact negatively on quality life. It is one of the persistent health issues worldwide. Generally, subjectivity and inter-observer variability undermine conventional practices and evaluation process that are adopted to address such health issues. Hence precise and timely diagnosis would be one of the effective ways for the assessment of its severity. This paper proposes an automated diagnostic approach for severity grading of KOA by blending a deep learning convolutional neural network (CNN) with a device-based inference platform powered by TensorFlow Lite. It proposes a model based on the ResNet-18 convolutional neural network. The designed model is trained on publicly available database. Through a transfer learning approach obtained knee images are first classified into five Kellgren-Lawrence (KL) grades. Further the developed model is optimised. During the training of the model test accuracy of 94.48% with stable convergence has been achieved. Subsequently the optimised model transformed into a lightweight TensorFlow Lite format, facilitating seamless deployment on resource-constrained devices. The designed model is capable enough to operate in the environment having no continuous internet connectivity. Also, an auxiliary Large Language Model (Gemini-2.0-flash) is applied to generate structured interpretive findings like potential symptoms, risk factors, and preventive majors etc. The LLM component functions as interface without influencing the classification process. The proposed model articulates the feasibility of an on-device, interpretable decision-support tools for early diagnosis and improve accessibility to Artificial Intelligence (AI)-assisted knee screening tool.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05727v1">LLM-Enhanced Deep Reinforcement Learning for Task Offloading in Collaborative Edge Computing</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Collaborative edge computing uses edge nodes in different locations to execute tasks, necessitating dynamic task offloading decisions to maintain low latency and high reliability, especially under unpredictable node failures. Although deep reinforcement learning (DRL) and large language models (LLMs) have shown promise for task offloading, DRL often suffers from high sample inefficiency and local optima, whereas LLMs struggle with real-time decision-making. To address these limitations, we propose \textbf{LeDRL}, a hybrid decision framework that couples a \emph{lightweight LLM} with self-attention-enhanced DRL for real-time task offloading. LeDRL constructs structured, context-aware prompts capturing node status, task semantics, and link dynamics to derive high-level strategy priors. These are selectively processed by a self-attention-based alignment module for context-aware policy optimization. A reflective evaluator distills semantic feedback from past trajectories to guide future prompts, enabling more informative and temporally generalizable LLM queries. Extensive experiments show that LeDRL outperforms baselines in task success rate, convergence speed, and real-time responsiveness across diverse network scales, achieving over 17\% improvement in success rate. Furthermore, we deploy LeDRL on Jetson-based edge devices using our prototype system \textit{CoEdgeSys}, demonstrating its robustness and feasibility under resource constraints. Our code is available at:https://github.com/GalleyG5/LeDRL.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05726v1">SkillRet: A Large-Scale Benchmark for Skill Retrieval in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      As LLM agents are increasingly deployed with large libraries of reusable skills, selecting the right skill for a user request has become a critical systems challenge. In small libraries, users may invoke skills explicitly by name, but this assumption breaks down as skill ecosystems grow under tight context and latency budgets. Despite its practical importance, skill retrieval remains underexplored, with limited benchmarks and little understanding of retrieval behavior on realistic skill libraries. To address this gap, we introduce SkillRet, a large-scale benchmark for skill retrieval in LLM agents. SkillRet contains 17,810 public agent skills, organized with structured semantic tags and a two-level taxonomy spanning 6 major categories and 18 sub-categories. It provides 63,259 training samples and 4,997 evaluation queries with disjoint skill pools, enabling both benchmarking and retrieval-oriented training. Across a diverse set of retrievers, we find that skill retrieval remains far from solved: off-the-shelf models struggle on realistic large-scale skill libraries, and prior skill-retrieval models still leave substantial headroom. Task-specific fine-tuning on SkillRet substantially improves performance, improving NDCG@10 by +13.1 points over the strongest prior retriever and by +16.9 points over the strongest off-the-shelf retriever. Our analysis further suggests that these gains arise because fine-tuned models better focus on the small skill-relevant signals within long and noisy queries. These results establish SkillRet as a strong benchmark and foundation for future research on retrieval in large-scale agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05716v1">More Is Not Always Better: Cross-Component Interference in LLM Agent Scaffolding</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 10 pages, 5 tables; preprint, under review
    </div>
    <details class="paper-abstract">
      LLM agent systems are built by stacking scaffolding components (planning, tools, memory, self-reflection, retrieval) assuming more is better. We study cross-component interference (CCI): degradation when components interact destructively. We run a full factorial experiment over all 2^5=32 subsets of five components on HotpotQA and GSM8K with Llama-3.1-8B/70B (96 conditions, up to 10 seeds). The All-In system is consistently suboptimal: on HotpotQA, a single-tool agent surpasses All-In by 32% (F1 0.233 vs 0.177, p=0.023); on GSM8K, a 3-component subset beats All-In by 79% (0.43 vs 0.24, p=0.010). Optimal component count is task-dependent (k*=1-4) and scale-sensitive: at 70B, combinations that hurt at 8B provide gains, though All-In still trails the best subset. We fit a main-effects regression (R^2=0.916, adj-R^2=0.899, LOOCV=0.872), compute exact Shapley values, and find 183/325 submodularity violations (56.3%), showing greedy selection is unreliable. A three-body synergy among Tool Use, Self-Reflection, and Retrieval (INT_3=+0.175, 95% CI [+0.003,+0.351]) is reported as exploratory. CCI replicates across model families (Qwen2.5) and is robust to prompt paraphrasing. Our findings suggest maximally-equipped agent defaults should be replaced by task-specific subset selection via interaction-aware analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05715v1">Decodable but Not Corrected by Fixed Residual-Stream Linear Steering: Evidence from Medical LLM Failure Regimes</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 22 pages (14 main + 8 appendix), 5 figures, 7 tables. Under review
    </div>
    <details class="paper-abstract">
      Can linearly decodable failure signals in LLM hidden states be leveraged to correct those failures? We investigate this classification-correction gap via Overthinking (OT)--a stable behavioral regime (Jaccard >= 0.81, 94% inter-annotator agreement) in medical QA where models answer correctly under resampling yet fail in extended chain-of-thought. OT is linearly decodable at 71.6% balanced accuracy (p < 10^{-16}). Yet five families of fixed linear steering (29 configurations, n=1,273) all yield Delta ~= 0, with identical null results cross-architecture (Qwen2.5-7B) and cross-domain (MMLU-STEM). Three convergent lines of evidence suggest representational entanglement: the OT direction has 85-88% overlap with task-critical computation (specificity ratio <= 0.152); non-targeted shared-direction steering damages accuracy (-12.1pp); and LEACE concept erasure damages accuracy (-3.6pp, p=0.01), while 10 random erasures produce Delta=+0.3pp. The per-instance probe-steering correlation is r=-0.002 (p=0.97). Positively, the same probe enables selective abstention (held-out AUROC=0.610, exceeding all five uncertainty baselines, p=0.009): decodable failure structure supports post-generation reliability estimation even when the fixed linear steering family cannot exploit it for correction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05711v1">Closing the Loop: Unified 3D Scene Generation and Immersive Interaction via LLM-RL Coupling</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have significantly improved language-driven 3D content generation, but most existing approaches still treat scene generation and user interaction as separate processes, limiting the adaptability and immersive potential of interactive multimedia systems. This paper presents a unified framework that closes the loop between language-driven 3D scene generation and immersive user interaction. Given natural language instructions, the system first constructs structured scene representations using LLMs, and then optimizes spatial layouts via reinforcement learning under geometric and semantic constraints. The generated environments are deployed in a virtual reality setting to facilitate HRI-in-the-loop, where user interactions provide continuous feedback to align generated content with human perception and usability. By tightly coupling generation and interaction, the proposed framework enables more responsive, adaptive, and realistic multimedia experiences. Experiments on the ALFRED benchmark demonstrate state-of-the-art performance in task-based scene generation. Furthermore, qualitative results and user studies show consistent improvements in immersion, interaction quality, and task efficiency, highlighting the importance of closed-loop integration of generation and interaction for next-generation multimedia systems. Our project page can be found at https://proj-showcase.github.io/h3ds/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05704v1">SafeHarbor: Hierarchical Memory-Augmented Guardrail for LLM Agent Safety</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 Accepted by ICML 2026
    </div>
    <details class="paper-abstract">
      With the rapid evolution of foundation models, Large Language Model (LLM) agents have demonstrated increasingly powerful tool-use capabilities. However, this proficiency introduces significant security risks, as malicious actors can manipulate agents into executing tools to generate harmful content. While existing defensive mechanisms are effective, they frequently suffer from the over-refusal problem, where increased safety strictness compromises the agent's utility on benign tasks. To mitigate this trade-off, we propose \textsc{SafeHarbor}, a novel framework designed to establish precise decision boundaries for LLM agents. Unlike static guidelines, \textsc{SafeHarbor} extracts context-aware defense rules through enhanced adversarial generation. We design a local hierarchical memory system for dynamic rule injection, offering a training-free, efficient, and plug-and-play solution. Furthermore, we introduce an information entropy-based self-evolution mechanism that continuously optimizes the memory structure through dynamic node splitting and merging. Extensive experiments demonstrate that \textsc{SafeHarbor} achieves state-of-the-art performance on both ambiguous benign tasks and explicit malicious attacks, notably attaining a peak benign utility of 63.6\% on GPT-4o while maintaining a robust refusal rate exceeding 93\% against harmful requests. The source code is publicly available at https://github.com/ljj-cyber/SafeHarbor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05703v1">Active Learning for Communication Structure Optimization in LLM-Based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Optimizing the communication structure of large language model based multi-agent systems (LLM-MAS) has been shown to improve downstream performance and reduce token usage. Existing methods typically rely on randomly sampled training tasks. However, tasks may differ substantially in difficulty and domain, and thus they are not equally informative for updating communication structure, making optimization under limited training budgets often unstable and highly sensitive to the particular training set. To actively identify the most valuable tasks for communication-structure optimization, we propose an ensemble-based information-theoretic task selection framework. The proposed method estimates task informativeness by how much a candidate task changes the distribution over graph parameters, using ensemble Kalman inversion as an efficient and derivative-free approximation of the corresponding Bayesian update. The resulting estimator is especially suitable for black-box and noisy multi-agent systems. To enhance scalability, we construct a compact candidate pool through embedding-based representative selection and combine the informative selection with surrogate modeling and batch Thompson sampling. We validate our method in both benign settings and settings with agent attacks, demonstrating its effectiveness for communication-structure optimization under constrained computational budgets.
    </details>
</div>
