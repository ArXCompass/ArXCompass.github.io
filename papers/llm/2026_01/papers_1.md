# llm - 2026_01

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22159v1">RedSage: A Cybersecurity Generalist LLM</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Accepted on ICLR 2026; Project page: https://risys-lab.github.io/RedSage/
    </div>
    <details class="paper-abstract">
      Cybersecurity operations demand assistant LLMs that support diverse workflows without exposing sensitive data. Existing solutions either rely on proprietary APIs with privacy risks or on open models lacking domain adaptation. To bridge this gap, we curate 11.8B tokens of cybersecurity-focused continual pretraining data via large-scale web filtering and manual collection of high-quality resources, spanning 28.6K documents across frameworks, offensive techniques, and security tools. Building on this, we design an agentic augmentation pipeline that simulates expert workflows to generate 266K multi-turn cybersecurity samples for supervised fine-tuning. Combined with general open-source LLM data, these resources enable the training of RedSage, an open-source, locally deployable cybersecurity assistant with domain-aware pretraining and post-training. To rigorously evaluate the models, we introduce RedSage-Bench, a benchmark with 30K multiple-choice and 240 open-ended Q&A items covering cybersecurity knowledge, skills, and tool expertise. RedSage is further evaluated on established cybersecurity benchmarks (e.g., CTI-Bench, CyberMetric, SECURE) and general LLM benchmarks to assess broader generalization. At the 8B scale, RedSage achieves consistently better results, surpassing the baseline models by up to +5.59 points on cybersecurity benchmarks and +5.05 points on Open LLM Leaderboard tasks. These findings demonstrate that domain-aware agentic augmentation and pre/post-training can not only enhance cybersecurity-specific expertise but also help to improve general reasoning and instruction-following. All models, datasets, and code are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.13187v3">"Not in My Backyard": LLMs Uncover Online and Offline Social Biases Against Homelessness</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Homelessness is a persistent social challenge, impacting millions worldwide. Over 876,000 people experienced homelessness (PEH) in the U.S. in 2025. Social bias is a significant barrier to alleviation, shaping public perception and influencing policymaking. Given that online textual media and offline city council discourse reflect and influence part of public opinion, it provides valuable insights to identify and track social biases against PEH. We present a new, manually-annotated multi-domain dataset compiled from Reddit, X (formerly Twitter), news articles, and city council meeting minutes across ten U.S. cities. Our 16-category multi-label taxonomy creates a challenging long-tail classification problem: some categories appear in less than 1% of samples, while others exceed 70%. We find that small human-annotated datasets (1,702 samples) are insufficient for training effective classifiers, whether used to fine-tune encoder models or as few-shot examples for LLMs. To address this, we use GPT-4.1 to generate pseudo-labels on a larger unlabeled corpus. Training on this expanded dataset enables even small encoder models (ModernBERT, 150M parameters) to achieve 35.23 macro-F1, approaching GPT-4.1's 41.57. This demonstrates that \textbf{data quantity matters more than model size}, enabling low-cost, privacy-preserving deployment without relying on commercial APIs. Our results reveal that negative bias against PEH is prevalent both offline and online (especially on Reddit), with "not in my backyard" narratives showing the highest engagement. These findings uncover a type of ostracism that directly impacts poverty-reduction policymaking and provide actionable insights for practitioners addressing homelessness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22132v1">Pay for Hints, Not Answers: LLM Shepherding for Cost-Efficient Inference</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) deliver state-of-the-art performance on complex reasoning tasks, but their inference costs limit deployment at scale. Small Language Models (SLMs) offer dramatic cost savings yet lag substantially in accuracy. Existing approaches - routing and cascading - treat the LLM as an all-or-nothing resource: either the query bypasses the LLM entirely, or the LLM generates a complete response at full cost. We introduce LLM Shepherding, a framework that requests only a short prefix (a hint) from the LLM and provides it to SLM. This simple mechanism is surprisingly effective for math and coding tasks: even hints comprising 10-30% of the full LLM response improve SLM accuracy significantly. Shepherding generalizes both routing and cascading, and it achieves lower cost under oracle decision-making. We develop a two-stage predictor that jointly determines whether a hint is needed and how many tokens to request. On the widely-used mathematical reasoning (GSM8K, CNK12) and code generation (HumanEval, MBPP) benchmarks, Shepherding reduces costs by 42-94% relative to LLM-only inference. Compared to state-of-the-art routing and cascading baselines, shepherding delivers up to 2.8x cost reduction while matching accuracy. To our knowledge, this is the first work to exploit token-level budget control for SLM-LLM collaboration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.09277v4">NeuroFaith: Evaluating LLM Self-Explanation Faithfulness via Internal Representation Alignment</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can generate plausible free text self-explanations to justify their answers. However, these natural language explanations may not accurately reflect the model's actual reasoning process, pinpointing a lack of faithfulness. Existing faithfulness evaluation methods rely primarily on behavioral tests or computational block analysis without examining the semantic content of internal neural representations. This paper proposes NeuroFaith, a flexible framework that measures the faithfulness of LLM free text self-explanation by identifying key concepts within explanations and mechanistically testing whether these concepts actually influence the model's predictions. We show the versatility of NeuroFaith across 2-hop reasoning and classification tasks. Additionally, we develop a linear faithfulness probe based on NeuroFaith to detect unfaithful self-explanations from representation space and improve faithfulness through steering. NeuroFaith provides a principled approach to evaluating and enhancing the faithfulness of LLM free text self-explanations, addressing critical needs for trustworthy AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17915v2">Think Locally, Explain Globally: Graph-Guided LLM Investigations via Local Reasoning and Belief Propagation</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      LLM agents excel when environments are mostly static and the needed information fits in a model's context window, but they often fail in open-ended investigations where explanations must be constructed by iteratively mining evidence from massive, heterogeneous operational data. These investigations exhibit hidden dependency structure: entities interact, signals co-vary, and the importance of a fact may only become clear after other evidence is discovered. Because the context window is bounded, agents must summarize intermediate findings before their significance is known, increasing the risk of discarding key evidence. ReAct-style agents are especially brittle in this regime. Their retrieve-summarize-reason loop makes conclusions sensitive to exploration order and introduces run-to-run non-determinism, producing a reliability gap where Pass-at-k may be high but Majority-at-k remains low. Simply sampling more rollouts or generating longer reasoning traces does not reliably stabilize results, since hypotheses cannot be autonomously checked as new evidence arrives and there is no explicit mechanism for belief bookkeeping and revision. In addition, ReAct entangles semantic reasoning with controller duties such as tool orchestration and state tracking, so execution errors and plan drift degrade reasoning while consuming scarce context. We address these issues by formulating investigation as abductive reasoning over a dependency graph and proposing EoG (Explanations over Graphs), a disaggregated framework in which an LLM performs bounded local evidence mining and labeling (cause vs symptom) while a deterministic controller manages traversal, state, and belief propagation to compute a minimal explanatory frontier. On a representative ITBench diagnostics task, EoG improves both accuracy and run-to-run consistency over ReAct baselines, including a 7x average gain in Majority-at-k entity F1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22050v1">MasalBench: A Benchmark for Contextual and Cross-Cultural Understanding of Persian Proverbs in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      In recent years, multilingual Large Language Models (LLMs) have become an inseparable part of daily life, making it crucial for them to master the rules of conversational language in order to communicate effectively with users. While previous work has evaluated LLMs' understanding of figurative language in high-resource languages, their performance in low-resource languages remains underexplored. In this paper, we introduce MasalBench, a comprehensive benchmark for assessing LLMs' contextual and cross-cultural understanding of Persian proverbs, which are a key component of conversation in this low-resource language. We evaluate eight state-of-the-art LLMs on MasalBench and find that they perform well in identifying Persian proverbs in context, achieving accuracies above 0.90. However, their performance drops considerably when tasked with identifying equivalent English proverbs, with the best model achieving 0.79 accuracy. Our findings highlight the limitations of current LLMs in cultural knowledge and analogical reasoning, and they provide a framework for assessing cross-cultural understanding in other low-resource languages. MasalBench is available at https://github.com/kalhorghazal/MasalBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.04328v2">OD-Stega: LLM-Based Relatively Secure Steganography via Optimized Distributions</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Accepted to EACL 2026
    </div>
    <details class="paper-abstract">
      We consider coverless steganography where a Large Language Model (LLM) is used to generate stego-texts in combination with arithmetic coding. An efficient method should embed secret bits in as few language tokens as possible while keeping the stego-text as natural as possible. We show that this problem is equivalent to maximizing the entropy of a replacement probability distribution of the next token generation, subject to a constraint on the divergence between the new distribution and the original one produced by the LLM. A closed-form solution is provided under either the KL divergence or the total variation constraint. Several important practical issues are also tackled: 1) An often-overlooked tokenization mismatch issue is resolved with a simple prompt selection approach, 2) The combination of the optimized distribution and the vocabulary truncation technique is considered, and 3) The incorporation of the proposed approach with existing (potentially non arithmetic coding based) techniques, e.g., the Discop technique.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22028v1">From Logits to Latents: Contrastive Representation Shaping for LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Most LLM unlearning methods aim to approximate retrain-from-scratch behaviors with minimal distribution shift, often via alignment-style objectives defined in the prediction space. While effective at reducing forgotten content generation, such approaches may act as suppression: forgotten concepts can persist in representations and remain entangled with retained knowledge. We introduce CLReg, a contrastive representation regularizer that identifies forget features while pushing them away from retain features, explicitly reducing forget-retain interference with minimal shifts on retain features. We provide first theoretical insights that relate representation shaping to entanglement reduction. Across unlearning benchmarks and LLMs of different sizes, CLReg decreases forget-retain representation entanglement that facilitates mainstream unlearning methods without positing extra privacy risks, inspiring future work that reshapes the representation space to remove forget concepts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22027v1">CAR-bench: Evaluating the Consistency and Limit-Awareness of LLM Agents under Real-World Uncertainty</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Existing benchmarks for Large Language Model (LLM) agents focus on task completion under idealistic settings but overlook reliability in real-world, user-facing applications. In domains, such as in-car voice assistants, users often issue incomplete or ambiguous requests, creating intrinsic uncertainty that agents must manage through dialogue, tool use, and policy adherence. We introduce CAR-bench, a benchmark for evaluating consistency, uncertainty handling, and capability awareness in multi-turn, tool-using LLM agents in an in-car assistant domain. The environment features an LLM-simulated user, domain policies, and 58 interconnected tools spanning navigation, productivity, charging, and vehicle control. Beyond standard task completion, CAR-bench introduces Hallucination tasks that test agents' limit-awareness under missing tools or information, and Disambiguation tasks that require resolving uncertainty through clarification or internal information gathering. Baseline results reveal large gaps between occasional and consistent success on all task types. Even frontier reasoning LLMs achieve less than 50% consistent pass rate on Disambiguation tasks due to premature actions, and frequently violate policies or fabricate information to satisfy user requests in Hallucination tasks, underscoring the need for more reliable and self-aware LLM agents in real-world settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22025v1">When "Better" Prompts Hurt: Evaluation-Driven Iteration for LLM Applications</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Evaluating Large Language Model (LLM) applications differs from traditional software testing because outputs are stochastic, high-dimensional, and sensitive to prompt and model changes. We present an evaluation-driven workflow - Define, Test, Diagnose, Fix - that turns these challenges into a repeatable engineering loop. We introduce the Minimum Viable Evaluation Suite (MVES), a tiered set of recommended evaluation components for (i) general LLM applications, (ii) retrieval-augmented generation (RAG), and (iii) agentic tool-use workflows. We also synthesize common evaluation methods (automated checks, human rubrics, and LLM-as-judge) and discuss known judge failure modes. In reproducible local experiments (Ollama; Llama 3 8B Instruct and Qwen 2.5 7B Instruct), we observe that a generic "improved" prompt template can trade off behaviors: on our small structured suites, extraction pass rate decreased from 100% to 90% and RAG compliance from 93.3% to 80% for Llama 3 when replacing task-specific prompts with generic rules, while instruction-following improved. These findings motivate evaluation-driven prompt iteration and careful claim calibration rather than universal prompt recipes. All test suites, harnesses, and results are included for reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19120v2">RobustExplain: Evaluating Robustness of LLM-Based Explanation Agents for Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 8 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to generate natural-language explanations in recommender systems, acting as explanation agents that reason over user behavior histories. While prior work has focused on explanation fluency and relevance under fixed inputs, the robustness of LLM-generated explanations to realistic user behavior noise remains largely unexplored. In real-world web platforms, interaction histories are inherently noisy due to accidental clicks, temporal inconsistencies, missing values, and evolving preferences, raising concerns about explanation stability and user trust. We present RobustExplain, the first systematic evaluation framework for measuring the robustness of LLM-generated recommendation explanations. RobustExplain introduces five realistic user behavior perturbations evaluated across multiple severity levels and a multi-dimensional robustness metric capturing semantic, keyword, structural, and length consistency. Our goal is to establish a principled, task-level evaluation framework and initial robustness baselines, rather than to provide a comprehensive leaderboard across all available LLMs. Experiments on four representative LLMs (7B--70B) show that current models exhibit only moderate robustness, with larger models achieving up to 8% higher stability. Our results establish the first robustness benchmarks for explanation agents and highlight robustness as a critical dimension for trustworthy, agent-driven recommender systems at web scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22008v1">LANCER: LLM Reranking for Nugget Coverage</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 ECIR 2026
    </div>
    <details class="paper-abstract">
      Unlike short-form retrieval-augmented generation (RAG), such as factoid question answering, long-form RAG requires retrieval to provide documents covering a wide range of relevant information. Automated report generation exemplifies this setting: it requires not only relevant information but also a more elaborate response with comprehensive information. Yet, existing retrieval methods are primarily optimized for relevance ranking rather than information coverage. To address this limitation, we propose LANCER, an LLM-based reranking method for nugget coverage. LANCER predicts what sub-questions should be answered to satisfy an information need, predicts which documents answer these sub-questions, and reranks documents in order to provide a ranked list covering as many information nuggets as possible. Our empirical results show that LANCER enhances the quality of retrieval as measured by nugget coverage metrics, achieving higher $α$-nDCG and information coverage than other LLM-based reranking methods. Our oracle analysis further reveals that sub-question generation plays an essential role.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19121v2">LLMs as Orchestrators: Constraint-Compliant Multi-Agent Optimization for Recommendation Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 8 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Recommendation systems must optimize multiple objectives while satisfying hard business constraints such as fairness and coverage. For example, an e-commerce platform may require every recommendation list to include items from multiple sellers and at least one newly listed product; violating such constraints--even once--is unacceptable in production. Prior work on multi-objective recommendation and recent LLM-based recommender agents largely treat constraints as soft penalties or focus on item scoring and interaction, leading to frequent violations in real-world deployments. How to leverage LLMs for coordinating constrained optimization in recommendation systems remains underexplored. We propose DualAgent-Rec, an LLM-coordinated dual-agent framework for constrained multi-objective e-commerce recommendation. The framework separates optimization into an Exploitation Agent that prioritizes accuracy under hard constraints and an Exploration Agent that promotes diversity through unconstrained Pareto search. An LLM-based coordinator adaptively allocates resources between agents based on optimization progress and constraint satisfaction, while an adaptive epsilon-relaxation mechanism guarantees feasibility of final solutions. Experiments on the Amazon Reviews 2023 dataset demonstrate that DualAgent-Rec achieves 100% constraint satisfaction and improves Pareto hypervolume by 4-6% over strong baselines, while maintaining competitive accuracy-diversity trade-offs. These results indicate that LLMs can act as effective orchestration agents for deployable and constraint-compliant recommendation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21996v1">Mechanistic Data Attribution: Tracing the Training Origins of Interpretable LLM Units</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      While Mechanistic Interpretability has identified interpretable circuits in LLMs, their causal origins in training data remain elusive. We introduce Mechanistic Data Attribution (MDA), a scalable framework that employs Influence Functions to trace interpretable units back to specific training samples. Through extensive experiments on the Pythia family, we causally validate that targeted intervention--removing or augmenting a small fraction of high-influence samples--significantly modulates the emergence of interpretable heads, whereas random interventions show no effect. Our analysis reveals that repetitive structural data (e.g., LaTeX, XML) acts as a mechanistic catalyst. Furthermore, we observe that interventions targeting induction head formation induce a concurrent change in the model's in-context learning (ICL) capability. This provides direct causal evidence for the long-standing hypothesis regarding the functional link between induction heads and ICL. Finally, we propose a mechanistic data augmentation pipeline that consistently accelerates circuit convergence across model scales, providing a principled methodology for steering the developmental trajectories of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21986v1">SpecTran: Spectral-Aware Transformer-based Adapter for LLM-Enhanced Sequential Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Traditional sequential recommendation (SR) models learn low-dimensional item ID embeddings from user-item interactions, often overlooking textual information such as item titles or descriptions. Recent advances in Large Language Models (LLMs) have inspired a surge of research that encodes item textual information with high-dimensional semantic embeddings, and designs transformation methods to inject such embeddings into SR models. These embedding transformation strategies can be categorized into two types, both of which exhibits notable drawbacks: 1) adapter-based methods suffer from pronounced dimension collapse, concentrating information into a few dominant dimensions; 2) SVD-based methods are rigid and manual, considering only a few principal spectral components while discarding rich information in the remaining spectrum. To address these limitations, we propose SpecTran, a spectral-aware transformer-based adapter that operates in the spectral domain, attending to the full spectrum to select and aggregates informative components. A learnable spectral-position encoding injects singular-value cues as an inductive bias, guiding attention toward salient spectral components and promoting diversity across embedding dimensions. Across four real-world datasets and three SR backbones, it consistently outperforms strong baselines, achieving an average improvement of 9.17%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21972v1">Learning Decentralized LLM Collaboration with Multi-Agent Actor Critic</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Recent work has explored optimizing LLM collaboration through Multi-Agent Reinforcement Learning (MARL). However, most MARL fine-tuning approaches rely on predefined execution protocols, which often require centralized execution. Decentralized LLM collaboration is more appealing in practice, as agents can run inference in parallel with flexible deployments. Also, current approaches use Monte Carlo methods for fine-tuning, which suffer from high variance and thus require more samples to train effectively. Actor-critic methods are prevalent in MARL for dealing with these issues, so we developed Multi-Agent Actor-Critic (MAAC) methods to optimize decentralized LLM collaboration. In this paper, we analyze when and why these MAAC methods are beneficial. We propose 2 MAAC approaches, \textbf{CoLLM-CC} with a \textbf{C}entralized \textbf{C}ritic and \textbf{CoLLM-DC} with \textbf{D}ecentralized \textbf{C}ritics. Our experiments across writing, coding, and game-playing domains show that Monte Carlo methods and CoLLM-DC can achieve performance comparable to CoLLM-CC in short-horizon and dense-reward settings. However, they both underperform CoLLM-CC on long-horizon or sparse-reward tasks, where Monte Carlo methods require substantially more samples and CoLLM-DC struggles to converge. Our code is available at https://github.com/OpenMLRL/CoMLRL/releases/tag/v1.3.2.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21963v1">Industrialized Deception: The Collateral Effects of LLM-Generated Misinformation on Digital Ecosystems</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Accepted at ACM TheWebConf '26 Companion
    </div>
    <details class="paper-abstract">
      Generative AI and misinformation research has evolved since our 2024 survey. This paper presents an updated perspective, transitioning from literature review to practical countermeasures. We report on changes in the threat landscape, including improved AI-generated content through Large Language Models (LLMs) and multimodal systems. Central to this work are our practical contributions: JudgeGPT, a platform for evaluating human perception of AI-generated news, and RogueGPT, a controlled stimulus generation engine for research. Together, these tools form an experimental pipeline for studying how humans perceive and detect AI-generated misinformation. Our findings show that detection capabilities have improved, but the competition between generation and detection continues. We discuss mitigation strategies including LLM-based detection, inoculation approaches, and the dual-use nature of generative AI. This work contributes to research addressing the adverse impacts of AI on information quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.12481v3">SAC-GLAM: Improving Online RL for LLM agents with Soft Actor-Critic and Hindsight Relabeling</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 This work has been presented at the IMOL workshop at NeurIPS 2025 (https://neurips.cc/virtual/2024/101058)
    </div>
    <details class="paper-abstract">
      The past years have seen Large Language Models (LLMs) strive not only as generative models but also as agents solving textual sequential decision-making tasks. When facing complex environments where their zero-shot abilities are insufficient, recent work showed online Reinforcement Learning (RL) could be used for the LLM agent to discover and learn efficient strategies interactively. However, most prior work sticks to on-policy algorithms, which greatly reduces the scope of methods such agents could use for both exploration and exploitation, such as experience replay and hindsight relabeling. Yet, such methods may be key for LLM learning agents, and in particular when designing autonomous intrinsically motivated agents sampling and pursuing their own goals (i.e. autotelic agents). This paper presents and studies an adaptation of Soft Actor-Critic and hindsight relabeling to LLM agents. Our method not only paves the path towards autotelic LLM agents that learn online but can also outperform on-policy methods in more classic multi-goal RL environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.06725v3">WorldLLM: Improving LLMs' world modeling using curiosity-driven theory-making</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 This project's code can be found at https://github.com/flowersteam/WorldLLM. This project was presented at RLDM 2025 (https://rldm.org/)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) possess general world knowledge but often struggle to generate precise predictions in structured, domain-specific contexts such as simulations. These limitations arise from their inability to ground their broad, unstructured understanding in specific environments. To address this, we present WorldLLM, a framework that enhances LLM-based world modeling by combining Bayesian inference and autonomous active exploration with reinforcement learning. WorldLLM leverages the in-context learning abilities of LLMs to guide an LLM-based world model's predictions using natural language hypotheses given in its prompt. These hypotheses are iteratively refined through a Bayesian inference framework that leverages a second LLM as the proposal distribution given collected evidence. This evidence is collected using a curiosity-driven reinforcement learning policy that explores the environment to find transitions with a low log-likelihood under our LLM-based predictive model using the current hypotheses. By alternating between refining hypotheses and collecting new evidence, our framework autonomously drives continual improvement of the predictions. Our experiments demonstrate the effectiveness of WorldLLM in a textual game environment that requires agents to manipulate and combine objects. The framework not only enhances predictive accuracy, but also generates human-interpretable theories of environment dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.17254v2">Near-Optimal Online Deployment and Routing for Streaming LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      The rapid pace at which new large language models (LLMs) appear, and older ones become obsolete, forces providers to manage a streaming inventory under a strict concurrency cap and per-query cost budgets. We cast this as an online decision problem that couples stage-wise deployment (at fixed maintenance windows) with per-query routing among live models. We introduce StageRoute, a hierarchical algorithm that (i) optimistically selects up to $M_{\max}$ models for the next stage using reward upper-confidence and cost lower-confidence bounds, and (ii) routes each incoming query by solving a budget- and throughput-constrained bandit subproblem over the deployed set. We prove a regret of $\tilde{\mathcal{O}}(T^{2/3})$ with a matching lower bound, establishing near-optimality, and validate the theory empirically: StageRoute tracks a strong oracle under tight budgets across diverse workloads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11479v2">Health Facility Location in Ethiopia: Leveraging LLMs to Integrate Expert Knowledge into Algorithmic Planning</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Ethiopia's Ministry of Health is upgrading health posts to improve access to essential services, particularly in rural areas. Limited resources, however, require careful prioritization of which facilities to upgrade to maximize population coverage while accounting for diverse expert and stakeholder preferences. In collaboration with the Ethiopian Public Health Institute and Ministry of Health, we propose a hybrid framework that systematically integrates expert knowledge with optimization techniques. Classical optimization methods provide theoretical guarantees but require explicit, quantitative objectives, whereas stakeholder criteria are often articulated in natural language and difficult to formalize. To bridge these domains, we develop the Large language model and Extended Greedy (LEG) framework. Our framework combines a provable approximation algorithm for population coverage optimization with LLM-driven iterative refinement that incorporates human-AI alignment to ensure solutions reflect expert qualitative guidance while preserving coverage guarantees. Experiments on real-world data from three Ethiopian regions demonstrate the framework's effectiveness and its potential to inform equitable, data-driven health system planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21909v1">From Meta-Thought to Execution: Cognitively Aligned Post-Training for Generalizable and Reliable LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Current LLM post-training methods optimize complete reasoning trajectories through Supervised Fine-Tuning (SFT) followed by outcome-based Reinforcement Learning (RL). While effective, a closer examination reveals a fundamental gap: this approach does not align with how humans actually solve problems. Human cognition naturally decomposes problem-solving into two distinct stages: first acquiring abstract strategies (i.e., meta-knowledge) that generalize across problems, then adapting them to specific instances. In contrast, by treating complete trajectories as basic units, current methods are inherently problem-centric, entangling abstract strategies with problem-specific execution. To address this misalignment, we propose a cognitively-inspired framework that explicitly mirrors the two-stage human cognitive process. Specifically, Chain-of-Meta-Thought (CoMT) focuses supervised learning on abstract reasoning patterns without specific executions, enabling acquisition of generalizable strategies. Confidence-Calibrated Reinforcement Learning (CCRL) then optimizes task adaptation via confidence-aware rewards on intermediate steps, preventing overconfident errors from cascading and improving execution reliability. Experiments across four models and eight benchmarks show 2.19\% and 4.63\% improvements in-distribution and out-of-distribution respectively over standard methods, while reducing training time by 65-70% and token consumption by 50%, demonstrating that aligning post-training with human cognitive principles yields not only superior generalization but also enhanced training efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21895v1">Learn-to-Distance: Distance Learning for Detecting LLM-Generated Text</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Accepted by ICLR2026
    </div>
    <details class="paper-abstract">
      Modern large language models (LLMs) such as GPT, Claude, and Gemini have transformed the way we learn, work, and communicate. Yet, their ability to produce highly human-like text raises serious concerns about misinformation and academic integrity, making it an urgent need for reliable algorithms to detect LLM-generated content. In this paper, we start by presenting a geometric approach to demystify rewrite-based detection algorithms, revealing their underlying rationale and demonstrating their generalization ability. Building on this insight, we introduce a novel rewrite-based detection algorithm that adaptively learns the distance between the original and rewritten text. Theoretically, we demonstrate that employing an adaptively learned distance function is more effective for detection than using a fixed distance. Empirically, we conduct extensive experiments with over 100 settings, and find that our approach demonstrates superior performance over baseline algorithms in the majority of scenarios. In particular, it achieves relative improvements from 57.8\% to 80.6\% over the strongest baseline across different target LLMs (e.g., GPT, Claude, and Gemini).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21894v1">Not All Code Is Equal: A Data-Centric Study of Code Complexity and LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 16 pages, 5 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly exhibit strong reasoning abilities, often attributed to their capacity to generate chain-of-thought-style intermediate reasoning. Recent work suggests that exposure to code can further enhance these skills, but existing studies largely treat code as a generic training signal, leaving open the question of which properties of code actually contribute to improved reasoning. To address this gap, we study the structural complexity of code, which captures control flow and compositional structure that may shape how models internalise multi-step reasoning during fine-tuning. We examine two complementary settings: solution-driven complexity, where complexity varies across multiple solutions to the same problem, and problem-driven complexity, where complexity reflects variation in the underlying tasks. Using cyclomatic complexity and logical lines of code to construct controlled fine-tuning datasets, we evaluate a range of open-weight LLMs on diverse reasoning benchmarks. Our findings show that although code can improve reasoning, structural properties strongly determine its usefulness. In 83% of experiments, restricting fine-tuning data to a specific structural complexity range outperforms training on structurally diverse code, pointing to a data-centric path for improving reasoning beyond scaling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21879v1">astra-langchain4j: Experiences Combining LLMs and Agent Programming</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Given the emergence of Generative AI over the last two years and the increasing focus on Agentic AI as a form of Multi-Agent System it is important to explore both how such technologies can impact the use of traditional Agent Toolkits and how the wealth of experience encapsulated in those toolkits can influence the design of the new agentic platforms. This paper presents an overview of our experience developing a prototype large language model (LLM) integration for the ASTRA programming language. It presents a brief overview of the toolkit, followed by three example implementations, concluding with a discussion of the experiences garnered through the examples.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02553v3">SimpleMem: Efficient Lifelong Memory for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      To support long-term interaction in complex environments, LLM agents require memory systems that manage historical experiences. Existing approaches either retain full interaction histories via passive context extension, leading to substantial redundancy, or rely on iterative reasoning to filter noise, incurring high token costs. To address this challenge, we introduce SimpleMem, an efficient memory framework based on semantic lossless compression. We propose a three-stage pipeline designed to maximize information density and token utilization: (1) Semantic Structured Compression, which distills unstructured interactions into compact, multi-view indexed memory units; (2) Online Semantic Synthesis, an intra-session process that instantly integrates related context into unified abstract representations to eliminate redundancy; and (3) Intent-Aware Retrieval Planning, which infers search intent to dynamically determine retrieval scope and construct precise context efficiently. Experiments on benchmark datasets show that our method consistently outperforms baseline approaches in accuracy, retrieval efficiency, and inference cost, achieving an average F1 improvement of 26.4% in LoCoMo while reducing inference-time token consumption by up to 30-fold, demonstrating a superior balance between performance and efficiency. Code is available at https://github.com/aiming-lab/SimpleMem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07568v2">d3LLM: Ultra-Fast Diffusion LLM using Pseudo-Trajectory Distillation</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Diffusion large language models (dLLMs) offer capabilities beyond those of autoregressive (AR) LLMs, such as parallel decoding and random-order generation. However, realizing these benefits in practice is non-trivial, as dLLMs inherently face an accuracy-parallelism trade-off. Despite increasing interest, existing methods typically focus on only one-side of the coin, targeting either efficiency or performance. To address this limitation, we propose d3LLM (Pseudo-Distilled Diffusion Large Language Model), striking a balance between accuracy and parallelism: (i) during training, we introduce pseudo-trajectory distillation to teach the model which tokens can be decoded confidently at early steps, thereby improving parallelism; (ii) during inference, we employ entropy-based multi-block decoding with a KV-cache refresh mechanism to achieve high parallelism while maintaining accuracy. To better evaluate dLLMs, we also introduce AUP (Accuracy Under Parallelism), a new metric that jointly measures accuracy and parallelism. Experiments demonstrate that our d3LLM achieves up to 10$\times$ speedup over vanilla LLaDA/Dream and 5$\times$ speedup over AR models without much accuracy drop. Our code is available at https://github.com/hao-ai-lab/d3LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21876v1">LLM-Driven Scenario-Aware Planning for Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Hybrid planner switching framework (HPSF) for autonomous driving needs to reconcile high-speed driving efficiency with safe maneuvering in dense traffic. Existing HPSF methods often fail to make reliable mode transitions or sustain efficient driving in congested environments, owing to heuristic scene recognition and low-frequency control updates. To address the limitation, this paper proposes LAP, a large language model (LLM) driven, adaptive planning method, which switches between high-speed driving in low-complexity scenes and precise driving in high-complexity scenes, enabling high qualities of trajectory generation through confined gaps. This is achieved by leveraging LLM for scene understanding and integrating its inference into the joint optimization of mode configuration and motion planning. The joint optimization is solved using tree-search model predictive control and alternating minimization. We implement LAP by Python in Robot Operating System (ROS). High-fidelity simulation results show that the proposed LAP outperforms other benchmarks in terms of both driving time and success rate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21864v1">KnowBias: Mitigating Social Bias in LLMs via Know-Bias Neuron Enhancement</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit social biases that reinforce harmful stereotypes, limiting their safe deployment. Most existing debiasing methods adopt a suppressive paradigm by modifying parameters, prompts, or neurons associated with biased behavior; however, such approaches are often brittle, weakly generalizable, data-inefficient, and prone to degrading general capability. We propose \textbf{KnowBias}, a lightweight and conceptually distinct framework that mitigates bias by strengthening, rather than suppressing, neurons encoding bias-knowledge. KnowBias identifies neurons encoding bias knowledge using a small set of bias-knowledge questions via attribution-based analysis, and selectively enhances them at inference time. This design enables strong debiasing while preserving general capabilities, generalizes across bias types and demographics, and is highly data efficient, requiring only a handful of simple yes/no questions and no retraining. Experiments across multiple benchmarks and LLMs demonstrate consistent state-of-the-art debiasing performance with minimal utility degradation. Data and code are available at https://github.com/JP-25/KnowBias.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.05029v3">Reputation as a Solution to Cooperation Collapse in LLM-based MASs</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Published as a conference paper at AAMAS 2026
    </div>
    <details class="paper-abstract">
      Cooperation has long been a fundamental topic in both human society and AI systems. However, recent studies indicate that the collapse of cooperation may emerge in multi-agent systems (MASs) driven by large language models (LLMs). To address this challenge, we explore reputation systems as a remedy. We propose RepuNet, a dynamic, dual-level reputation framework that models both agent-level reputation dynamics and system-level network evolution. Specifically, driven by direct interactions and indirect gossip, agents form reputations for both themselves and their peers, and decide whether to connect or disconnect other agents for future interactions. Through three distinct scenarios, we show that RepuNet effectively avoids cooperation collapse, promoting and sustaining cooperation in LLM-based MASs. Moreover, we find that reputation systems can give rise to rich emergent behaviors in LLM-based MASs, such as the formation of cooperative clusters, the social isolation of exploitative agents, and the preference for sharing positive gossip rather than negative ones. The GitHub repository for our project can be accessed via the following link: https://github.com/RGB-0000FF/RepuNet.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.14038v6">OAEI-LLM: A Benchmark Dataset for Understanding Large Language Model Hallucinations in Ontology Matching</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 5 pages, 1 figure, 1 table, 1 code snippet
    </div>
    <details class="paper-abstract">
      Hallucinations of large language models (LLMs) commonly occur in domain-specific downstream tasks, with no exception in ontology matching (OM). The prevalence of using LLMs for OM raises the need for benchmarks to better understand LLM hallucinations. The OAEI-LLM dataset is an extended version of the Ontology Alignment Evaluation Initiative (OAEI) datasets that evaluate LLM-specific hallucinations in OM tasks. We outline the methodology used in dataset construction and schema extension, and provide examples of potential use cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21824v1">DASH: Deterministic Attention Scheduling for High-throughput Reproducible LLM Training</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Determinism is indispensable for reproducibility in large language model (LLM) training, yet it often exacts a steep performance cost. In widely used attention implementations such as FlashAttention-3, the deterministic backward pass can incur up to a 37.9% throughput reduction relative to its non-deterministic counterpart, primarily because gradient accumulation operations must be serialized to guarantee numerical consistency. This performance loss stems from suboptimal scheduling of compute and gradient-reduction phases, leading to significant hardware underutilization. To address this challenge, we formulate the backward pass of deterministic attention as a scheduling problem on a Directed Acyclic Graph (DAG) and derive schedules that minimize the critical path length. Building on this formulation, we present DASH (Deterministic Attention Scheduling for High-Throughput), which encapsulates two complementary scheduling strategies: (i) Descending Q-Tile Iteration, a reversed query-block traversal that shrinks pipeline stalls in causal attention, and (ii) Shift Scheduling, a theoretically optimal schedule within our DAG model that reduces pipeline stalls for both full and causal masks. Our empirical evaluations on NVIDIA H800 GPUs demonstrate that DASH narrows the performance gap of deterministic attention. The proposed strategies improve the throughput of the attention backward pass by up to 1.28$\times$ compared to the baseline, significantly advancing the efficiency of reproducible LLM training. Our code is open-sourced at https://github.com/SJTU-Liquid/deterministic-FA3.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18352v2">LLM-based Few-Shot Early Rumor Detection with Imitation Agent</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Accepted at KDD 2026
    </div>
    <details class="paper-abstract">
      Early Rumor Detection (EARD) aims to identify the earliest point at which a claim can be accurately classified based on a sequence of social media posts. This is especially challenging in data-scarce settings. While Large Language Models (LLMs) perform well in few-shot NLP tasks, they are not well-suited for time-series data and are computationally expensive for both training and inference. In this work, we propose a novel EARD framework that combines an autonomous agent and an LLM-based detection model, where the agent acts as a reliable decision-maker for \textit{early time point determination}, while the LLM serves as a powerful \textit{rumor detector}. This approach offers the first solution for few-shot EARD, necessitating only the training of a lightweight agent and allowing the LLM to remain training-free. Extensive experiments on four real-world datasets show our approach boosts performance across LLMs and surpasses existing EARD methods in accuracy and earliness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21816v1">Nonparametric LLM Evaluation from Preference Data</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Evaluating the performance of large language models (LLMs) from human preference data is crucial for obtaining LLM leaderboards. However, many existing approaches either rely on restrictive parametric assumptions or lack valid uncertainty quantification when flexible machine learning methods are used. In this paper, we propose a nonparametric statistical framework, DMLEval, for comparing and ranking LLMs from preference data using debiased machine learning (DML). For this, we introduce generalized average ranking scores (GARS), which generalize commonly used ranking models, including the Bradley-Terry model or PageRank/ Rank centrality, with complex human responses such as ties. DMLEval comes with the following advantages: (i) It produces statistically efficient estimates of GARS ranking scores. (ii) It naturally allows the incorporation of black-box machine learning methods for estimation. (iii) It can be combined with pre-trained LLM evaluators (e.g., using LLM-as-a-judge). (iv) It suggests optimal policies for collecting preference data under budget constraints. We demonstrate these advantages both theoretically and empirically using both synthetic and real-world preference datasets. In summary, our framework provides practitioners with powerful, state-of-the-art methods for comparing or ranking LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21802v1">A Unified XAI-LLM Approach for EndotrachealSuctioning Activity Recognition</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Endotracheal suctioning (ES) is an invasive yet essential clinical procedure that requires a high degree of skill to minimize patient risk - particularly in home care and educational settings, where consistent supervision may be limited. Despite its critical importance, automated recognition and feedback systems for ES training remain underexplored. To address this gap, this study proposes a unified, LLM-centered framework for video-based activity recognition benchmarked against conventional machine learning and deep learning approaches, and a pilot study on feedback generation. Within this framework, the Large Language Model (LLM) serves as the central reasoning module, performing both spatiotemporal activity recognition and explainable decision analysis from video data. Furthermore, the LLM is capable of verbalizing feedback in natural language, thereby translating complex technical insights into accessible, human-understandable guidance for trainees. Experimental results demonstrate that the proposed LLM-based approach outperforms baseline models, achieving an improvement of approximately 15-20\% in both accuracy and F1 score. Beyond recognition, the framework incorporates a pilot student-support module built upon anomaly detection and explainable AI (XAI) principles, which provides automated, interpretable feedback highlighting correct actions and suggesting targeted improvements. Collectively, these contributions establish a scalable, interpretable, and data-driven foundation for advancing nursing education, enhancing training efficiency, and ultimately improving patient safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21758v1">EWSJF: An Adaptive Scheduler with Hybrid Partitioning for Mixed-Workload LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Serving Large Language Models (LLMs) under mixed workloads--short, latency-sensitive interactive queries alongside long, throughput-oriented batch requests--poses a fundamental scheduling challenge. Standard First-Come, First-Served (FCFS) policies suffer from severe head-of-line blocking, leading to high tail latency and underutilized hardware. We introduce EWSJF (Effective Workload-based Shortest Job First), an adaptive request-level scheduler that learns workload structure in real time to jointly improve fairness and throughput. EWSJF operates upstream of execution-level schedulers and integrates four components: (1) Refine-and-Prune, an unsupervised partitioning algorithm that discovers performance-homogeneous request groups; (2) Dynamic Queue Routing for assigning requests to these groups; (3) Density-Weighted Scoring, a context-aware prioritization function balancing urgency and fairness; and (4) Bayesian Meta-Optimization, which continuously tunes scoring and partitioning parameters based on live performance feedback. Implemented in vLLM, EWSJF improves end-to-end throughput by over 30% and reduces average Time-To-First-Token for short requests by up to 4x compared to FCFS. These results demonstrate that adaptive, learning-based request scheduling is a critical missing layer for efficient and responsive LLM serving. Implementation available at https://anonymous.4open.science/r/vllm_0110-32D8.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2312.00326v25">Agent-OM: Leveraging LLM Agents for Ontology Matching</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 31 pages - VLDB 2025 (Page 1-20), OM 2025 (Page 21-31)
    </div>
    <details class="paper-abstract">
      Ontology matching (OM) enables semantic interoperability between different ontologies and resolves their conceptual heterogeneity by aligning related entities. OM systems currently have two prevailing design paradigms: conventional knowledge-based expert systems and newer machine learning-based predictive systems. While large language models (LLMs) and LLM agents have revolutionised data engineering and have been applied creatively in many domains, their potential for OM remains underexplored. This study introduces a novel agent-powered LLM-based design paradigm for OM systems. With consideration of several specific challenges in leveraging LLM agents for OM, we propose a generic framework, namely Agent-OM (Agent for Ontology Matching), consisting of two Siamese agents for retrieval and matching, with a set of OM tools. Our framework is implemented in a proof-of-concept system. Evaluations of three Ontology Alignment Evaluation Initiative (OAEI) tracks over state-of-the-art OM systems show that our system can achieve results very close to the long-standing best performance on simple OM tasks and can significantly improve the performance on complex and few-shot OM tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21742v1">Epistemic Context Learning: Building Trust the Right Way in LLM-Based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Codes and data are available at https://github.com/skyriver-2000/epistemic-context-learning
    </div>
    <details class="paper-abstract">
      Individual agents in multi-agent (MA) systems often lack robustness, tending to blindly conform to misleading peers. We show this weakness stems from both sycophancy and inadequate ability to evaluate peer reliability. To address this, we first formalize the learning problem of history-aware reference, introducing the historical interactions of peers as additional input, so that agents can estimate peer reliability and learn from trustworthy peers when uncertain. This shifts the task from evaluating peer reasoning quality to estimating peer reliability based on interaction history. We then develop Epistemic Context Learning (ECL): a reasoning framework that conditions predictions on explicitly-built peer profiles from history. We further optimize ECL by reinforcement learning using auxiliary rewards. Our experiments reveal that our ECL enables small models like Qwen 3-4B to outperform a history-agnostic baseline 8x its size (Qwen 3-30B) by accurately identifying reliable peers. ECL also boosts frontier models to near-perfect (100%) performance. We show that ECL generalizes well to various MA configurations and we find that trust is modeled well by LLMs, revealing a strong correlation in trust modeling accuracy and final answer quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20539v2">PathWise: Planning through World Model for Automated Heuristic Design via Self-Evolving LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have enabled automated heuristic design (AHD) for combinatorial optimization problems (COPs), but existing frameworks' reliance on fixed evolutionary rules and static prompt templates often leads to myopic heuristic generation, redundant evaluations, and limited reasoning about how new heuristics should be derived. We propose a novel multi-agent reasoning framework, referred to as Planning through World Model for Automated Heuristic Design via Self-Evolving LLMs (PathWise), which formulates heuristic generation as a sequential decision process over an entailment graph serving as a compact, stateful memory of the search trajectory. This approach allows the system to carry forward past decisions and reuse or avoid derivation information across generations. A policy agent plans evolutionary actions, a world model agent generates heuristic rollouts conditioned on those actions, and critic agents provide routed reflections summarizing lessons from prior steps, shifting LLM-based AHD from trial-and-error evolution toward state-aware planning through reasoning. Experiments across diverse COPs show that PathWise converges faster to better heuristics, generalizes across different LLM backbones, and scales to larger problem sizes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21733v1">CE-GOCD: Central Entity-Guided Graph Optimization for Community Detection to Augment LLM Scientific Question Answering</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Accepted by IEEE ICASSP 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used for question answering over scientific research papers. Existing retrieval augmentation methods often rely on isolated text chunks or concepts, but overlook deeper semantic connections between papers. This impairs the LLM's comprehension of scientific literature, hindering the comprehensiveness and specificity of its responses. To address this, we propose Central Entity-Guided Graph Optimization for Community Detection (CE-GOCD), a method that augments LLMs' scientific question answering by explicitly modeling and leveraging semantic substructures within academic knowledge graphs. Our approach operates by: (1) leveraging paper titles as central entities for targeted subgraph retrieval, (2) enhancing implicit semantic discovery via subgraph pruning and completion, and (3) applying community detection to distill coherent paper groups with shared themes. We evaluated the proposed method on three NLP literature-based question-answering datasets, and the results demonstrate its superiority over other retrieval-augmented baseline approaches, confirming the effectiveness of our framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21714v1">E-mem: Multi-agent based Episodic Context Reconstruction for LLM Agent Memory</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      The evolution of Large Language Model (LLM) agents towards System~2 reasoning, characterized by deliberative, high-precision problem-solving, requires maintaining rigorous logical integrity over extended horizons. However, prevalent memory preprocessing paradigms suffer from destructive de-contextualization. By compressing complex sequential dependencies into pre-defined structures (e.g., embeddings or graphs), these methods sever the contextual integrity essential for deep reasoning. To address this, we propose E-mem, a framework shifting from Memory Preprocessing to Episodic Context Reconstruction. Inspired by biological engrams, E-mem employs a heterogeneous hierarchical architecture where multiple assistant agents maintain uncompressed memory contexts, while a central master agent orchestrates global planning. Unlike passive retrieval, our mechanism empowers assistants to locally reason within activated segments, extracting context-aware evidence before aggregation. Evaluations on the LoCoMo benchmark demonstrate that E-mem achieves over 54\% F1, surpassing the state-of-the-art GAM by 7.75\%, while reducing token cost by over 70\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.09101v3">A Survey of LLM Alignment: Instruction Understanding, Intention Reasoning, and Reliable Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 37 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Large language models have demonstrated exceptional capabilities in understanding and generation. However, in real-world scenarios, users' natural language expressions are often inherently fuzzy, ambiguous, and uncertain, leading to challenges such as vagueness, polysemy, and contextual ambiguity. This paper focuses on three challenges in LLM-based text generation tasks: instruction understanding, intention reasoning, and reliable dialog generation. Regarding human complex instruction, LLMs have deficiencies in understanding long contexts and instructions in multi-round conversations. For intention reasoning, LLMs may have inconsistent command reasoning, difficulty reasoning about commands containing incorrect information, difficulty understanding user ambiguous language commands, and a weak understanding of user intention in commands. Besides, In terms of Reliable Dialog Generation, LLMs may have unstable generated content and unethical generation. To this end, we classify and analyze the performance of LLMs in challenging scenarios and conduct a comprehensive evaluation of existing solutions. Furthermore, we introduce benchmarks and categorize them based on the aforementioned three core challenges. Finally, we explore potential directions for future research to enhance the reliability and adaptability of LLMs in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21700v1">Toward Culturally Aligned LLMs through Ontology-Guided Multi-Agent Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 35 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly support culturally sensitive decision making, yet often exhibit misalignment due to skewed pretraining data and the absence of structured value representations. Existing methods can steer outputs, but often lack demographic grounding and treat values as independent, unstructured signals, reducing consistency and interpretability. We propose OG-MAR, an Ontology-Guided Multi-Agent Reasoning framework. OG-MAR summarizes respondent-specific values from the World Values Survey (WVS) and constructs a global cultural ontology by eliciting relations over a fixed taxonomy via competency questions. At inference time, it retrieves ontology-consistent relations and demographically similar profiles to instantiate multiple value-persona agents, whose outputs are synthesized by a judgment agent that enforces ontology consistency and demographic proximity. Experiments on regional social-survey benchmarks across four LLM backbones show that OG-MAR improves cultural alignment and robustness over competitive baselines, while producing more transparent reasoning traces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21682v1">FIT: Defying Catastrophic Forgetting in Continual LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 20 Pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate impressive capabilities across diverse tasks but raise concerns about privacy, copyright, and harmful materials. Existing LLM unlearning methods rarely consider the continual and high-volume nature of real-world deletion requests, which can cause utility degradation and catastrophic forgetting as requests accumulate. To address this challenge, we introduce \fit, a framework for continual unlearning that handles large numbers of deletion requests while maintaining robustness against both catastrophic forgetting and post-unlearning recovery. \fit mitigates degradation through rigorous data \underline{F}iltering, \underline{I}mportance-aware updates, and \underline{T}argeted layer attribution, enabling stable performance across long sequences of unlearning operations and achieving a favorable balance between forgetting effectiveness and utility retention. To support realistic evaluation, we present \textbf{PCH}, a benchmark covering \textbf{P}ersonal information, \textbf{C}opyright, and \textbf{H}armful content in sequential deletion scenarios, along with two symmetric metrics, Forget Degree (F.D.) and Retain Utility (R.U.), which jointly assess forgetting quality and utility preservation. Extensive experiments on four open-source LLMs with hundreds of deletion requests show that \fit achieves the strongest trade-off between F.D. and R.U., surpasses existing methods on MMLU, CommonsenseQA, and GSM8K, and remains resistant against both relearning and quantization recovery attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19423v2">UniRec: Unified Multimodal Encoding for LLM-Based Recommendations</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Large language models have recently shown promise for multimodal recommendation, particularly with text and image inputs. Yet real-world recommendation signals extend far beyond these modalities. To reflect this, we formalize recommendation features into four modalities: text, images, categorical features, and numerical attributes, and highlight the unique challenges this heterogeneity poses for LLMs in understanding multimodal information. In particular, these challenges arise not only across modalities but also within them, as attributes such as price, rating, and time may all be numeric yet carry distinct semantic meanings. Beyond this intra-modality ambiguity, another major challenge is the nested structure of recommendation signals, where user histories are sequences of items, each associated with multiple attributes. To address these challenges, we propose UniRec, a unified multimodal encoder for LLM-based recommendation. UniRec first employs modality-specific encoders to produce consistent embeddings across heterogeneous signals. It then adopts a triplet representation, comprising attribute name, type, and value, to separate schema from raw inputs and preserve semantic distinctions. Finally, a hierarchical Q-Former models the nested structure of user interactions while maintaining their layered organization. Across multiple real-world benchmarks, UniRec outperforms state-of-the-art multimodal and LLM-based recommenders by up to 15%, and extensive ablation studies further validate the contributions of each component.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19684v2">LLM-Assisted Authentication and Fraud Detection</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 20 pages, 7 figures, 2 tables
    </div>
    <details class="paper-abstract">
      User authentication and fraud detection face growing challenges as digital systems expand and adversaries adopt increasingly sophisticated tactics. Traditional knowledge-based authentication remains rigid, requiring exact word-for-word string matches that fail to accommodate natural human memory and linguistic variation. Meanwhile, fraud-detection pipelines struggle to keep pace with rapidly evolving scam behaviors, leading to high false-positive rates and frequent retraining cycles required. This work introduces two complementary LLM-enabled solutions, namely, an LLM-assisted authentication mechanism that evaluates semantic correctness rather than exact wording, supported by document segmentation and a hybrid scoring method combining LLM judgement with cosine-similarity metrics and a RAG-based fraud-detection pipeline that grounds LLM reasoning in curated evidence to reduce hallucinations and adapt to emerging scam patterns without model retraining. Experiments show that the authentication system accepts 99.5% of legitimate non-exact answers while maintaining a 0.1% false-acceptance rate, and that the RAG-enhanced fraud detection reduces false positives from 17.2% to 3.5%. Together, these findings demonstrate that LLMs can significantly improve both usability and robustness in security workflows, offering a more adaptive , explainable, and human-aligned approach to authentication and fraud detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02901v3">Fairy2i: Training Complex LLMs from Real LLMs with All Parameters in $\{\pm 1, \pm i\}$</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 15 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have revolutionized artificial intelligence, yet their massive memory and computational demands necessitate aggressive quantization, increasingly pushing representations toward the theoretical limit of a single bit. While complex-valued LLMs, such as iFairy, offer a superior chance for low-bit representation compared to real-valued counterparts, they require training from scratch, preventing the utilization of the vast ecosystem of pre-trained real-valued foundation models. Here we present Fairy2i, a universal framework that transforms pre-trained real-valued layers into an equivalent widely-linear complex form, enabling extremely low-bit quantization while reusing existing checkpoints. By proving a lossless mathematical equivalence between real and widely-linear maps, we convert standard Transformers into the complex domain and employ a phase-aware quantization scheme with a highly efficient codebook of fourth roots of unity. Furthermore, we introduce a recursive residual quantization mechanism that iteratively minimizes quantization error, allowing inference to proceed via efficient multiplication-free accumulation. We demonstrate that Fairy2i restores the performance of LLaMA-2 7B at an effective 2-bit precision to levels nearly comparable with full-precision baselines, significantly outperforming state-of-the-art real-valued binary and ternary quantization methods. This work bridges the gap between the representational efficiency of complex-valued arithmetic and the practical utility of pre-trained models, paving a new way for efficient inference on commodity hardware. We open-source the Fairy2i model and code at https://huggingface.co/PKU-DS-LAB/Fairy2i-W2 and https://github.com/PKULab1806/Fairy2i-W2.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.13258v3">Towards Transparent RAG: Fostering Evidence Traceability in LLM Generation via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) delivers substantial value in knowledge-intensive applications. However, its generated responses often lack transparent reasoning paths that trace back to source evidence from retrieved documents. This opacity not only compromises the interpretability of the output but also limits the model's ability to fully exploit the provided context. To address this, we propose TRACE (Transparent RAG with evidenCE tracing), a framework designed to enhance evidence traceability in Large Language Models (LLMs) through reinforcement learning (RL). TRACE guides LLMs to produce structured outputs with explicit evidence citations by prompting and rewarding evidence relevance and proper formatting, alongside accuracy, to optimize structured traceability. To ensure training stability with multiple reward signals, we further introduce an adaptive strategy for merging rewards and adopt a stabilized KL-divergence estimator. Experiments on three multi-hop QA datasets using Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct show that TRACE achieves both transparent, evidence-attributed outputs and accuracy improvements of 10-30%. The resulting performance is comparable to advanced commercial LLMs (e.g., OpenAI o1, DeepSeek-R1). Further analyses demonstrate strong generalization capabilities to unseen tasks. Our code is publicly available now.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10233v2">Bridging Synthetic and Real Routing Problems via LLM-Guided Instance Generation and Progressive Adaptation</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 21 pages; Published in AAAI 2026, Main Technical Track
    </div>
    <details class="paper-abstract">
      Recent advances in Neural Combinatorial Optimization (NCO) methods have significantly improved the capability of neural solvers to handle synthetic routing instances. Nonetheless, existing neural solvers typically struggle to generalize effectively from synthetic, uniformly-distributed training data to real-world VRP scenarios, including widely recognized benchmark instances from TSPLib and CVRPLib. To bridge this generalization gap, we present Evolutionary Realistic Instance Synthesis (EvoReal), which leverages an evolutionary module guided by large language models (LLMs) to generate synthetic instances characterized by diverse and realistic structural patterns. Specifically, the evolutionary module produces synthetic instances whose structural attributes statistically mimics those observed in authentic real-world instances. Subsequently, pre-trained NCO models are progressively refined, firstly aligning them with these structurally enriched synthetic distributions and then further adapting them through direct fine-tuning on actual benchmark instances. Extensive experimental evaluations demonstrate that EvoReal markedly improves the generalization capabilities of state-of-the-art neural solvers, yielding a notable reduced performance gap compared to the optimal solutions on the TSPLib (1.05%) and CVRPLib (2.71%) benchmarks across a broad spectrum of problem scales.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06411v2">SofT-GRPO: Surpassing Discrete-Token LLM Reinforcement Learning via Gumbel-Reparameterized Soft-Thinking Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      The soft-thinking paradigm for Large Language Model (LLM) reasoning can outperform the conventional discrete-token Chain-of-Thought (CoT) reasoning in some scenarios, underscoring its research and application value. However, while the discrete-token CoT reasoning pattern can be reinforced through policy optimization algorithms such as group relative policy optimization (GRPO), extending the soft-thinking pattern with Reinforcement Learning (RL) remains challenging. This difficulty stems from the complexities of injecting stochasticity into soft-thinking tokens and updating soft-thinking policies accordingly. As a result, previous attempts to combine soft-thinking with GRPO typically underperform their discrete-token GRPO counterparts. To fully unlock the potential of soft-thinking, this paper presents a novel policy optimization algorithm, SofT-GRPO, to reinforce LLMs under the soft-thinking reasoning pattern. SofT-GRPO injects the Gumbel noise into logits, employs the Gumbel-Softmax technique to avoid soft-thinking tokens outside the pre-trained embedding space, and leverages the reparameterization trick in policy gradient. We conduct experiments across base LLMs ranging from 1.5B to 7B parameters, and results demonstrate that SofT-GRPO enables soft-thinking LLMs to slightly outperform discrete-token GRPO on Pass@1 (+0.13% on average accuracy), while exhibiting a substantial uplift on Pass@32 (+2.19% on average accuracy). Codes and weights are available on https://github.com/zz1358m/SofT-GRPO-master
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21590v1">Scalable Power Sampling: Unlocking Efficient, Training-Free Reasoning for LLMs via Distribution Sharpening</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) post-training is a dominant approach for improving the reasoning performance of large language models (LLMs), yet growing evidence suggests that its gains arise primarily from distribution sharpening rather than the acquisition of new capabilities. Recent work has shown that sampling from the power distribution of LLMs using Markov chain Monte Carlo (MCMC) can recover performance comparable to RL post-training without relying on external rewards; however, the high computational cost of MCMC makes such approaches impractical for widespread adoption. In this work, we propose a theoretically grounded alternative that eliminates the need for iterative MCMC. We derive a novel formulation showing that the global power distribution can be approximated by a token-level scaled low-temperature one, where the scaling factor captures future trajectory quality. Leveraging this insight, we introduce a training-free and verifier-free algorithm that sharpens the base model's generative distribution autoregressively. Empirically, we evaluate our method on math, QA, and code tasks across four LLMs, and show that our method matches or surpasses one-shot GRPO without relying on any external rewards, while reducing inference latency by over 10x compared to MCMC-based sampling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.10996v3">RAS: Retrieval-And-Structuring for Knowledge-Intensive LLM Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved impressive performance on knowledge-intensive tasks, yet they often struggle with multi-step reasoning due to the unstructured nature of retrieved context. While retrieval-augmented generation (RAG) methods provide external information, the lack of explicit organization among retrieved passages limits their effectiveness, leading to brittle reasoning pathways. Recent interpretability studies highlighting the importance of structured intermediate reasoning further align with this perspective. We propose Retrieval-And-Structuring (RAS), a framework that dynamically constructs question-specific knowledge graphs through iterative retrieval and structured knowledge building. RAS interleaves targeted retrieval planning with incremental graph construction, enabling models to assemble and reason over evolving knowledge structures tailored to each query. On seven knowledge-intensive benchmarks, RAS consistently outperforms strong baselines, achieving up to 8.7\% and 7.0\% gains with proprietary and open-source LLMs, respectively. Our results demonstrate that dynamic, question-specific knowledge structuring offers a robust path to improving reasoning accuracy and robustness in language model generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2407.04503v4">When LLMs Play the Telephone Game: Cultural Attractors as Conceptual Tools to Evaluate LLMs in Multi-turn Settings</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Code available at https://github.com/jeremyperez2/TelephoneGameLLM. Companion website with a Data Explorer tool at https://sites.google.com/view/telephone-game-llm . This paper was published at the 2025 International Conference on Learning Representations (ICLR2025) https://iclr.cc/virtual/2025/poster/28880
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) start interacting with each other and generating an increasing amount of text online, it becomes crucial to better understand how information is transformed as it passes from one LLM to the next. While significant research has examined individual LLM behaviors, existing studies have largely overlooked the collective behaviors and information distortions arising from iterated LLM interactions. Small biases, negligible at the single output level, risk being amplified in iterated interactions, potentially leading the content to evolve towards attractor states. In a series of telephone game experiments, we apply a transmission chain design borrowed from the human cultural evolution literature: LLM agents iteratively receive, produce, and transmit texts from the previous to the next agent in the chain. By tracking the evolution of text toxicity, positivity, difficulty, and length across transmission chains, we uncover the existence of biases and attractors, and study their dependence on the initial text, the instructions, language model, and model size. For instance, we find that more open-ended instructions lead to stronger attraction effects compared to more constrained tasks. We also find that different text properties display different sensitivity to attraction effects, with toxicity leading to stronger attractors than length. These findings highlight the importance of accounting for multi-step transmission dynamics and represent a first step towards a more comprehensive understanding of LLM cultural dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21551v1">Note2Chat: Improving LLMs for Multi-Turn Clinical History Taking Using Medical Notes</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Accepted at AAAI-26
    </div>
    <details class="paper-abstract">
      Effective clinical history taking is a foundational yet underexplored component of clinical reasoning. While large language models (LLMs) have shown promise on static benchmarks, they often fall short in dynamic, multi-turn diagnostic settings that require iterative questioning and hypothesis refinement. To address this gap, we propose \method{}, a note-driven framework that trains LLMs to conduct structured history taking and diagnosis by learning from widely available medical notes. Instead of relying on scarce and sensitive dialogue data, we convert real-world medical notes into high-quality doctor-patient dialogues using a decision tree-guided generation and refinement pipeline. We then propose a three-stage fine-tuning strategy combining supervised learning, simulated data augmentation, and preference learning. Furthermore, we propose a novel single-turn reasoning paradigm that reframes history taking as a sequence of single-turn reasoning problems. This design enhances interpretability and enables local supervision, dynamic adaptation, and greater sample efficiency. Experimental results show that our method substantially improves clinical reasoning, achieving gains of +16.9 F1 and +21.0 Top-1 diagnostic accuracy over GPT-4o. Our code and dataset can be found at https://github.com/zhentingsheng/Note2Chat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21545v1">ShardMemo: Masked MoE Routing for Sharded Agentic LLM Memory</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Agentic large language model (LLM) systems rely on external memory for long-horizon state and concurrent multi-agent execution, but centralized indexes and heuristic partitions become bottlenecks as memory volume and parallel access grow. We present ShardMemo, a budgeted tiered memory service with Tier A per-agent working state, Tier B sharded evidence with shard-local approximate nearest neighbor (ANN) indexes, and Tier C, a versioned skill library. Tier B enforces scope-before-routing: structured eligibility constraints mask ineligible shards before routing or ANN search. We cast shard probing as masked mixture-of-experts (MoE) routing over eligible shards, probing up to $B_{\mathrm{probe}}$ shards via Top-$B_{\mathrm{probe}}$ or adaptive Top-$P$, and use cost-aware gating over profile/observation/session shard families; the router is trained from evidence-to-shard supervision. On LoCoMo, ShardMemo improves over the strongest baseline (GAM) by +5.11 to +6.82 F1 across question categories. Under a fixed-budget routing setting ($B_{\mathrm{probe}}=3$), ShardMemo improves over cosine-to-prototype shard routing by +6.87 F1 while reducing retrieval work (VecScan 521->414, -20.5%) and p95 latency (95->76 ms). On long-context HotpotQA, ShardMemo achieves 63.41/61.88/57.95 F1 at 56K/224K/448K tokens. On ToolBench, Tier C reaches 0.97 Precision@3 and 1.94 StepRed (+10.2% and +7.2% over embedding-similarity retrieval).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.23694v4">SafeSearch: Automated Red-Teaming of LLM-Based Search Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Preprint. Updated with new experiments
    </div>
    <details class="paper-abstract">
      Search agents connect LLMs to the Internet, enabling them to access broader and more up-to-date information. However, this also introduces a new threat surface: unreliable search results can mislead agents into producing unsafe outputs. Real-world incidents and our two in-the-wild observations show that such failures can occur in practice. To study this threat systematically, we propose SafeSearch, an automated red-teaming framework that is scalable, cost-efficient, and lightweight, enabling harmless safety evaluation of search agents. Using this, we generate 300 test cases spanning five risk categories (e.g., misinformation and prompt injection) and evaluate three search agent scaffolds across 17 representative LLMs. Our results reveal substantial vulnerabilities in LLM-based search agents, with the highest ASR reaching 90.5% for GPT-4.1-mini in a search-workflow setting. Moreover, we find that common defenses, such as reminder prompting, offer limited protection. Overall, SafeSearch provides a practical way to measure and improve the safety of LLM-based search agents. Our codebase and test cases are publicly available: https://github.com/jianshuod/SafeSearch.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16651v2">Select or Project? Evaluating Lower-dimensional Vectors for LLM Training Data Explanations</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Added acknowledgments section and related work on random projection. 8 pages
    </div>
    <details class="paper-abstract">
      Gradient-based methods for instance-based explanation for large language models (LLMs) are hindered by the immense dimensionality of model gradients. In practice, influence estimation is restricted to a subset of model parameters to make computation tractable, but this subset is often chosen ad hoc and rarely justified by systematic evaluation. This paper investigates if it is better to create low-dimensional representations by selecting a small, architecturally informed subset of model components or by projecting the full gradients into a lower-dimensional space. Using a novel benchmark, we show that a greedily selected subset of components captures the information about training data influence needed for a retrieval task more effectively than either the full gradient or random projection. We further find that this approach is more computationally efficient than random projection, demonstrating that targeted component selection is a practical strategy for making instance-based explanations of large models more computationally feasible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21533v1">ARGORA: Orchestrated Argumentation for Causally Grounded LLM Reasoning and Decision Making</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 58 pages
    </div>
    <details class="paper-abstract">
      Existing multi-expert LLM systems gather diverse perspectives but combine them through simple aggregation, obscuring which arguments drove the final decision. We introduce ARGORA, a framework that organizes multi-expert discussions into explicit argumentation graphs showing which arguments support or attack each other. By casting these graphs as causal models, ARGORA can systematically remove individual arguments and recompute outcomes, identifying which reasoning chains were necessary and whether decisions would change under targeted modifications. We further introduce a correction mechanism that aligns internal reasoning with external judgments when they disagree. Across diverse benchmarks and an open-ended use case, ARGORA achieves competitive accuracy and demonstrates corrective behavior: when experts initially disagree, the framework resolves disputes toward correct answers more often than it introduces new errors, while providing causal diagnostics of decisive arguments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20755v2">ProfInfer: An eBPF-based Fine-Grained LLM Inference Profiler</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Accepted in the 9th Annual Conference on Machine Learning and Systems (MLSys 2026)
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) move from research to production, understanding how inference engines behave in real time has become both essential and elusive. Unlike general-purpose engines such as ONNX Runtime, today's LLM inference systems offer little operator-level visibility, leaving developers blind to where time and resources go. Even basic questions -- is this workload memory-bound or compute-bound? -- often remain unanswered. To close this gap, we develop a fine-grained, non-intrusive profiling framework for modern LLM inference engines, exemplified by llama-cpp but applicable to similar runtime architectures. Built on extended Berkeley Packet Filter (eBPF) technology, our system dynamically attaches probes to runtime functions across multiple layers -- without modifying or recompiling the source. It transforms collected traces into rich visualizations of operators, graphs, timelines, and hardware counter trends, exposing how dense inference, Mixture-of-Experts routing, and operator offloading behave in practice. With less than 4% runtime overhead and high profiling fidelity, our framework makes LLM inference both transparent and diagnosable, turning performance profiling into a practical tool for optimization, scheduling, and resource-aware deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.23019v4">LLM Watermark Evasion via Bias Inversion</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Watermarking offers a promising solution for detecting LLM-generated content, yet its robustness under realistic query-free (black-box) evasion remains an open challenge. Existing query-free attacks often achieve limited success or severely distort semantic meaning. We bridge this gap by theoretically analyzing rewriting-based evasion, demonstrating that reducing the average conditional probability of sampling green tokens by a small margin causes the detection probability to decay exponentially. Guided by this insight, we propose the Bias-Inversion Rewriting Attack (BIRA), a practical query-free method that applies a negative logit bias to a proxy suppression set identified via token surprisal. Empirically, BIRA achieves state-of-the-art evasion rates (>99%) across diverse watermarking schemes while preserving semantic fidelity substantially better than prior baselines. Our findings reveal a fundamental vulnerability in current watermarking methods and highlight the need for rigorous stress tests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.04418v2">The Illusion of Certainty: Uncertainty Quantification for LLMs Fails under Ambiguity</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Accurate uncertainty quantification (UQ) in Large Language Models (LLMs) is critical for trustworthy deployment. While real-world language is inherently ambiguous, reflecting aleatoric uncertainty, existing UQ methods are typically benchmarked against tasks with no ambiguity. In this work, we demonstrate that while current uncertainty estimators perform well under the restrictive assumption of no ambiguity, they degrade to close-to-random performance on ambiguous data. To this end, we introduce MAQA* and AmbigQA*, the first ambiguous question-answering (QA) datasets equipped with ground-truth answer distributions estimated from factual co-occurrence. We find this performance deterioration to be consistent across different estimation paradigms: using the predictive distribution itself, internal representations throughout the model, and an ensemble of models. We show that this phenomenon can be theoretically explained, revealing that predictive-distribution and ensemble-based estimators are fundamentally limited under ambiguity. Overall, our study reveals a key shortcoming of current UQ methods for LLMs and motivates a rethinking of current modeling paradigms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21500v1">Task-Awareness Improves LLM Generations and Uncertainty</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      In many applications of LLMs, natural language responses often have an underlying structure such as representing discrete labels, numerical values, or graphs. Yet, existing decoding and uncertainty estimation methods operate only in language space and largely disregard structural information. We address this by modeling LLM outputs directly in a task-dependent latent structure. By equipping this structure with a dissimilarity measure, we can compute Bayes-optimal responses. These are not selected from sampled generations but are newly synthesized by combining individual responses in the latent space. Across different tasks, Bayes-optimal responses consistently outperform standard decoding methods like beam search. Moreover, quantifying uncertainty via the induced Bayesian risk captures variations in terms of the latent structure and improves alignment with output quality and correctness. Our decision-theoretic framework is applicable to any problem that admits a latent response structure and enables reliable task-aware LLM predictions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21471v1">Best Arm Identification with LLM Judges and Limited Human</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 22 pages, 3 figures
    </div>
    <details class="paper-abstract">
      We study fixed-confidence best-arm identification (BAI) where a cheap but potentially biased proxy (e.g., LLM judge) is available for every sample, while an expensive ground-truth label can only be acquired selectively when using a human for auditing. Unlike classical multi-fidelity BAI, the proxy is biased (arm- and context-dependent) and ground truth is selectively observed. Consequently, standard multi-fidelity methods can mis-select the best arm, and uniform auditing, though accurate, wastes scarce resources and is inefficient. We prove that without bias correction and propensity adjustment, mis-selection probability may not vanish (even with unlimited proxy data). We then develop an estimator for the mean of each arm that combines proxy scores with inverse-propensity-weighted residuals and form anytime-valid confidence sequences for that estimator. Based on the estimator and confidence sequence, we propose an algorithm that adaptively selects and audits arms. The algorithm concentrates audits on unreliable contexts and close arms and we prove that a plug-in Neyman rule achieves near-oracle audit efficiency. Numerical experiments confirm the theoretical guarantees and demonstrate the superior empirical performance of the proposed algorithm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21464v1">Conversation for Non-verifiable Learning: Self-Evolving LLMs through Meta-Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Work in Progress
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) for non-verifiable tasks, such as creative writing, dialogue, and ethical reasoning, remains challenging due to the absence of ground-truth labels. While LLM-as-Judge approaches offer a scalable alternative to human feedback, they face a fundamental limitation: performance is constrained by the evaluator's own quality. If the judge cannot recognize good solutions, it cannot provide useful training signals, and evaluation biases (e.g., favoring verbosity over quality) remain unaddressed. This motivates meta-evaluation: the ability to evaluate and improve the evaluator itself. We introduce CoNL, a framework that unifies generation, evaluation, and meta-evaluation through multi-agent self-play. Our key insight: critique quality can be measured by whether it helps others improve their solutions. In CoNL, multiple agents sharing the same policy engage in structured conversations to propose, critique, and revise solutions. Critiques that enable solution improvements earn a diagnostic reward, creating explicit supervision for meta-evaluation and enabling joint optimization of generation and judging capabilities through self-play, without external judges or ground truth. Experiments on five benchmarks show that CoNL achieves consistent improvements over self-rewarding baselines while maintaining stable training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21459v1">HER: Human-like Reasoning and Reinforcement Learning for LLM Role-playing</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 41pages, 10 figures
    </div>
    <details class="paper-abstract">
      LLM role-playing, i.e., using LLMs to simulate specific personas, has emerged as a key capability in various applications, such as companionship, content creation, and digital games. While current models effectively capture character tones and knowledge, simulating the inner thoughts behind their behaviors remains a challenge. Towards cognitive simulation in LLM role-play, previous efforts mainly suffer from two deficiencies: data with high-quality reasoning traces, and reliable reward signals aligned with human preferences. In this paper, we propose HER, a unified framework for cognitive-level persona simulation. HER introduces dual-layer thinking, which distinguishes characters' first-person thinking from LLMs' third-person thinking. To bridge these gaps, we curate reasoning-augmented role-playing data via reverse engineering and construct human-aligned principles and reward models. Leveraging these resources, we train \method models based on Qwen3-32B via supervised and reinforcement learning. Extensive experiments validate the effectiveness of our approach. Notably, our models significantly outperform the Qwen3-32B baseline, achieving a 30.26 improvement on the CoSER benchmark and a 14.97 gain on the Minimax Role-Play Bench. Our datasets, principles, and models will be released to facilitate future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21448v1">ChipBench: A Next-Step Benchmark for Evaluating LLM Performance in AI-Aided Chip Design</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) show significant potential in hardware engineering, current benchmarks suffer from saturation and limited task diversity, failing to reflect LLMs' performance in real industrial workflows. To address this gap, we propose a comprehensive benchmark for AI-aided chip design that rigorously evaluates LLMs across three critical tasks: Verilog generation, debugging, and reference model generation. Our benchmark features 44 realistic modules with complex hierarchical structures, 89 systematic debugging cases, and 132 reference model samples across Python, SystemC, and CXXRTL. Evaluation results reveal substantial performance gaps, with state-of-the-art Claude-4.5-opus achieving only 30.74\% on Verilog generation and 13.33\% on Python reference model generation, demonstrating significant challenges compared to existing saturated benchmarks where SOTA models achieve over 95\% pass rates. Additionally, to help enhance LLM reference model generation, we provide an automated toolbox for high-quality training data generation, facilitating future research in this underexplored domain. Our code is available at https://github.com/zhongkaiyu/ChipBench.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21437v1">Accurate Network Traffic Matrix Prediction via LEAD: an LLM-Enhanced Adapter-Based Conditional Diffusion Model</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Driven by the evolution toward 6G and AI-native edge intelligence, network operations increasingly require predictive and risk-aware adaptation under stringent computation and latency constraints. Network Traffic Matrix (TM), which characterizes flow volumes between nodes, is a fundamental signal for proactive traffic engineering. However, accurate TM forecasting remains challenging due to the stochastic, non-linear, and bursty nature of network dynamics. Existing discriminative models often suffer from over-smoothing and provide limited uncertainty awareness, leading to poor fidelity under extreme bursts. To address these limitations, we propose LEAD, a Large Language Model (LLM)-Enhanced Adapter-based conditional Diffusion model. First, LEAD adopts a "Traffic-to-Image" paradigm to transform traffic matrices into RGB images, enabling global dependency modeling via vision backbones. Then, we design a "Frozen LLM with Trainable Adapter" model, which efficiently captures temporal semantics with limited computational cost. Moreover, we propose a Dual-Conditioning Strategy to precisely guide a diffusion model to generate complex, dynamic network traffic matrices. Experiments on the Abilene and GEANT datasets demonstrate that LEAD outperforms all baselines. On the Abilene dataset, LEAD attains a remarkable 45.2% reduction in RMSE against the best baseline, with the error margin rising only marginally from 0.1098 at one-step to 0.1134 at 20-step predictions. Meanwhile, on the GEANT dataset, LEAD achieves a 0.0258 RMSE at 20-step prediction horizon which is 27.3% lower than the best baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.01027v2">DBellQuant: Breaking the Bell with Double-Bell Transformation for LLMs Post Training Binarization</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 19 pages; Appendix added
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate remarkable performance but face substantial computational and memory challenges that limit their practical deployment. Quantization has emerged as a promising solution; however, its effectiveness is often limited by quantization errors arising from weight distributions that are not quantization-friendly and the presence of activation outliers. To address these challenges, we introduce DBellQuant, an innovative post-training quantization (PTQ) framework that achieves nearly 1-bit weight compression and 6-bit activation quantization with minimal performance degradation. DBellQuant uses Learnable Transformation for Dual-Bell (LTDB) algorithm, which transforms single-bell weight distributions into dual-bell forms to reduce binarization errors and applies inverse transformations to smooth activations. DBellQuant sets a new state-of-the-art by preserving superior model performance under aggressive weight and activation quantization. For example, on the Wikitext2 dataset, DBellQuant achieves a perplexity of 14.39 on LLaMA2-13B with 6-bit activation quantization, significantly outperforming BiLLM's 21.35 without activation quantization, underscoring its potential in compressing LLMs for real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.14466v2">Toward Robust Multilingual Adaptation of LLMs for Low-Resource Languages</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) continue to struggle with low-resource languages, primarily due to limited training data, translation noise, and unstable cross-lingual alignment. To address these challenges, we propose LiRA (Linguistic Robust Anchoring for LLMs)-a plug-and-play framework that requires only lightweight fine-tuning on top of existing pretrained backbones. LiRA jointly optimizes representation stability and cross-lingual semantic consistency by combining two key components: Arca (Anchored Representation Composition Architecture), which aligns low-resource inputs to a shared English semantic space through anchor-based alignment and collaborative encoding; and LaSR (Language-coupled Semantic Reasoner), a lightweight, language-aware head that enforces consistency regularization for unified cross-lingual understanding, retrieval, and reasoning. We theoretically show that under controlled anchoring error and translation-induced bias, LiRA guarantees bounded representation deviation and stable downstream performance under local Lipschitz continuity. To facilitate research, we release a new multilingual product retrieval dataset covering five Southeast Asian and two South Asian languages. Extensive experiments across diverse low-resource benchmarks demonstrate consistent improvements in retrieval, ranking, question answering, and reasoning tasks. Code will be publicly available on GitHub, and the dataset will be hosted on Hugging Face.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21410v1">Statsformer: Validated Ensemble Learning with LLM-Derived Semantic Priors</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      We introduce Statsformer, a principled framework for integrating large language model (LLM)-derived knowledge into supervised statistical learning. Existing approaches are limited in adaptability and scope: they either inject LLM guidance as an unvalidated heuristic, which is sensitive to LLM hallucination, or embed semantic information within a single fixed learner. Statsformer overcomes both limitations through a guardrailed ensemble architecture. We embed LLM-derived feature priors within an ensemble of linear and nonlinear learners, adaptively calibrating their influence via cross-validation. This design yields a flexible system with an oracle-style guarantee that it performs no worse than any convex combination of its in-library base learners, up to statistical error. Empirically, informative priors yield consistent performance improvements, while uninformative or misspecified LLM guidance is automatically downweighted, mitigating the impact of hallucinations across a diverse range of prediction tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21380v1">RerouteGuard: Understanding and Mitigating Adversarial Risks for LLM Routing</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 15 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Recent advancements in multi-model AI systems have leveraged LLM routers to reduce computational cost while maintaining response quality by assigning queries to the most appropriate model. However, as classifiers, LLM routers are vulnerable to novel adversarial attacks in the form of LLM rerouting, where adversaries prepend specially crafted triggers to user queries to manipulate routing decisions. Such attacks can lead to increased computational cost, degraded response quality, and even bypass safety guardrails, yet their security implications remain largely underexplored. In this work, we bridge this gap by systematizing LLM rerouting threats based on the adversary's objectives (i.e., cost escalation, quality hijacking, and safety bypass) and knowledge. Based on the threat taxonomy, we conduct a measurement study of real-world LLM routing systems against existing LLM rerouting attacks. The results reveal that existing routing systems are vulnerable to rerouting attacks, especially in the cost escalation scenario. We then characterize existing rerouting attacks using interpretability techniques, revealing that they exploit router decision boundaries through confounder gadgets that prepend queries to force misrouting. To mitigate these risks, we introduce RerouteGuard, a flexible and scalable guardrail framework for LLM rerouting. RerouteGuard filters adversarial rerouting prompts via dynamic embedding-based detection and adaptive thresholding. Extensive evaluations in three attack settings and four benchmarks demonstrate that RerouteGuard achieves over 99% detection accuracy against state-of-the-art rerouting attacks, while maintaining negligible impact on legitimate queries. The experimental results indicate that RerouteGuard offers a principled and practical solution for safeguarding multi-model AI systems against adversarial rerouting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19402v2">PROTEUS: SLA-Aware Routing via Lagrangian RL for Multi-LLM Serving Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Production LLM deployments serve diverse workloads where cost and quality requirements vary by customer tier, time of day, and query criticality. Model serving systems accept latency SLOs directly. LLM routers do not. They force operators to tune parameters offline and guess what accuracy might result. The relationship between parameters and outcomes is indirect, non-monotonic, and dataset-dependent. Operators need to specify accuracy targets, not infer them from opaque settings. We present PROTEUS (Polymorphic Router for Operational Target Enforcement with Unified SLA), a router that accepts accuracy targets tau as runtime input. PROTEUS uses Lagrangian dual control. A learned dual variable lambda tracks constraint violations during training and conditions the policy network. This lets the router translate specified tau values into routing decisions that satisfy them. A single trained model serves the full accuracy spectrum without retraining.We evaluate on RouterBench (11 models, 405K queries) and SPROUT (14 models, 45K queries). PROTEUS achieves consistent floor compliance where accuracy meets or exceeds tau. The target-response correlation reaches 0.97 to 0.98. The closest baseline, OmniRouter, meets floors only 22% of the time despite also using Lagrangian optimization. PROTEUS operates across tau in [0.85, 0.95] from a single model. On RouterBench it achieves 90.1% accuracy, within 1.3% of oracle. On SPROUT it achieves 94.0% accuracy, within 4.6% of oracle. Cost savings reach 89.8% versus the best fixed model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2408.06787v5">FLAME: Empowering Frozen LLMs for Knowledge Graph Completion</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Traditional knowledge graph completion (KGC) methods rely solely on structural information and struggle with sparsity, while Large Language Models (LLMs) address these limitations through rich world knowledge and strong context modeling. Fine-tuning LLMs is effective but costly, while non-fine-tuned LLMs are efficient but suboptimal. To address this trade-off, we propose \textbf{FLAME}, a framework that extracts context-aware hidden states from intermediate layers of frozen LLMs to train data-efficient KGC classifiers. We bridge LLM-KG semantic gaps via subgraph-based entity descriptions and employ sliced mutual information (SMI) to quantify task-relevant information in representations. Experiments demonstrate that FLAME achieves 47\% improvement over non-fine-tuned LLM baselines and, to our knowledge, is the first to achieve fine-tuned performance with $188\times$ memory efficiency and $26.11\times$ speedup.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22137v4">HybridFlow: Resource-Adaptive Subtask Routing for Efficient Edge-Cloud LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Edge-cloud collaborative inference is becoming a practical necessity for LLM-powered edge devices: on-device models often cannot afford the required reasoning capability, while cloud-only inference could be prohibitively costly and slow under strict latency and token/API budgets. However, existing edge-cloud collaboration methods often route per query or fixed steps simply based-on the estimated difficulty. Such coarse and static heuristics overlook subtask dependencies, missing opportunities for parallel execution and budget-adaptive routing. To this end, we propose \textbf{HybridFlow}, a resource-adaptive edge-cloud inference framework that (i) builds a dependency-aware DAG for each query and executes newly unlocked subtasks in parallel, reducing end-to-end latency; (ii) routes each subtask online to the edge or cloud via a learned benefit--cost utility model that dynamically trades accuracy gains against token/API and latency budgets, thereby reducing unnecessary cloud usage while preserving reasoning quality. Across GPQA, MMLU-Pro, AIME24, and LiveBench-Reasoning, HybridFlow improves the cost-accuracy trade-off, reducing latency and cloud API usage while maintaining competitive accuracy against strong structured reasoning baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.24050v3">Bridging On-Device and Cloud LLMs for Collaborative Reasoning: A Unified Methodology for Local Routing and Post-Training</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 We propose a unified post-training framework that integrates routing optimization, enabling the on-device LLM to improve its problem-solving ability while learning routing strategies
    </div>
    <details class="paper-abstract">
      Device-cloud collaboration holds promise for deploying large language models (LLMs), leveraging lightweight on-device models for efficiency while relying on powerful cloud models for superior reasoning. A central challenge in this setting is determining, for each incoming query, whether it should be processed locally or offloaded to the cloud. Existing approaches typically rely on external routers, which often struggle to determine difficulty from the prompt itself, especially for tasks involving complex reasoning. Motivated by this limitation, we propose enabling on-device LLMs to decide internally whether to invoke cloud assistance at inference time, with this capability instilled through reinforcement learning based post-training. Casting on-device LLM post-training as a reward maximization problem, we design hierarchical rewards to encourage local problem solving and judicious cloud offloading. To solve the resulting problem, we develop an algorithm featuring a group-level policy gradient that stabilizes optimization, together with adaptive prompt filtering that provides complementary learning signals to mitigate policy collapse (i.e., exclusive local execution or exclusive cloud offloading). Extensive experiments on on-device-scale LLaMA and Qwen models across multiple reasoning benchmarks show that our method consistently outperforms baselines and significantly narrows the gap to full cloud LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21351v1">Theoretically Optimal Attention/FFN Ratios in Disaggregated LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Submitted to ICML 2026
    </div>
    <details class="paper-abstract">
      Attention-FFN disaggregation (AFD) is an emerging architecture for LLM decoding that separates state-heavy, KV-cache-dominated Attention computation from stateless, compute-intensive FFN computation, connected by per-step communication. While AFD enables independent scaling of memory and compute resources, its performance is highly sensitive to the Attention/FFN provisioning ratio: mis-sizing induces step-level blocking and costly device idle time. We develop a tractable analytical framework for sizing AFD bundles in an $r$A-$1$F topology, where the key difficulty is that Attention-side work is nonstationary-token context grows and requests are continuously replenished with random lengths-while FFN work is stable given the aggregated batch. Using a probabilistic workload model, we derive closed-form rules for the optimal A/F ratio that maximize average throughput per instance across the system. A trace-calibrated AFD simulator validates the theory: across workloads, the theoretical optimal A/F ratio matches the simulation-optimal within 10%, and consistently reduces idle time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.01203v3">HetGCoT: Heterogeneous Graph-Enhanced Chain-of-Thought LLM Reasoning for Academic Question Answering</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Findings of the Association for Computational Linguistics: EMNLP 2025
    </div>
    <details class="paper-abstract">
      Academic question answering (QA) in heterogeneous scholarly networks presents unique challenges requiring both structural understanding and interpretable reasoning. While graph neural networks (GNNs) capture structured graph information and large language models (LLMs) demonstrate strong capabilities in semantic comprehension, current approaches lack integration at the reasoning level. We propose HetGCoT, a framework enabling LLMs to effectively leverage and learn information from graphs to reason interpretable academic QA results. Our framework introduces three technical contributions: (1) a framework that transforms heterogeneous graph structural information into LLM-processable reasoning chains, (2) an adaptive metapath selection mechanism identifying relevant subgraphs for specific queries, and (3) a multi-step reasoning strategy systematically incorporating graph contexts into the reasoning process. Experiments on OpenAlex and DBLP datasets show our approach outperforms all sota baselines. The framework demonstrates adaptability across different LLM architectures and applicability to various scholarly question answering tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21347v1">Towards Robust Dysarthric Speech Recognition: LLM-Agent Post-ASR Correction Beyond WER</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Accepted to ICASSP 2026
    </div>
    <details class="paper-abstract">
      While Automatic Speech Recognition (ASR) is typically benchmarked by word error rate (WER), real-world applications ultimately hinge on semantic fidelity. This mismatch is particularly problematic for dysarthric speech, where articulatory imprecision and disfluencies can cause severe semantic distortions. To bridge this gap, we introduce a Large Language Model (LLM)-based agent for post-ASR correction: a Judge-Editor over the top-k ASR hypotheses that keeps high-confidence spans, rewrites uncertain segments, and operates in both zero-shot and fine-tuned modes. In parallel, we release SAP-Hypo5, the largest benchmark for dysarthric speech correction, to enable reproducibility and future exploration. Under multi-perspective evaluation, our agent achieves a 14.51% WER reduction alongside substantial semantic gains, including a +7.59 pp improvement in MENLI and +7.66 pp in Slot Micro F1 on challenging samples. Our analysis further reveals that WER is highly sensitive to domain shift, whereas semantic metrics correlate more closely with downstream task performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21344v1">Dynamic Framework for Collaborative Learning: Leveraging Advanced LLM with Adaptive Feedback Mechanisms</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Publication Link: https://ieeexplore.ieee.org/document/11118419
    </div>
    <details class="paper-abstract">
      This paper presents a framework for integrating LLM into collaborative learning platforms to enhance student engagement, critical thinking, and inclusivity. The framework employs advanced LLMs as dynamic moderators to facilitate real-time discussions and adapt to learners' evolving needs, ensuring diverse and inclusive educational experiences. Key innovations include robust feedback mechanisms that refine AI moderation, promote reflective learning, and balance participation among users. The system's modular architecture featuring ReactJS for the frontend, Flask for backend operations, and efficient question retrieval supports personalized and engaging interactions through dynamic adjustments to prompts and discussion flows. Testing demonstrates that the framework significantly improves student collaboration, fosters deeper comprehension, and scales effectively across various subjects and user groups. By addressing limitations in static moderation and personalization in existing systems, this work establishes a strong foundation for next-generation AI-driven educational tools, advancing equitable and impactful learning outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19139v2">Native LLM and MLLM Inference at Scale on Apple Silicon</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      The growing adoption of Apple Silicon for machine learning development has created demand for efficient inference solutions that leverage its unique unified memory architecture. However, existing tools either lack native optimization (PyTorch MPS) or focus solely on text models, leaving multimodal workloads underserved. We present vllm-mlx, a framework for efficient LLM and MLLM inference on Apple Silicon built natively on MLX. For text models, we achieve 21\% to 87\% higher throughput than llama-cpp across models ranging from Qwen3-0.6B to Nemotron-30B, while providing continuous batching that scales to 4.3x aggregate throughput at 16 concurrent requests. For multimodal models, we introduce content-based prefix caching that eliminates redundant vision encoding by identifying identical images through content hashing, regardless of input format. Our evaluation on Apple M4 Max demonstrates throughput of up to 525 tokens per second on text models and 28x speedup on repeated image queries, reducing multimodal latency from 21.7 seconds to under 1 second. Video analysis with up to 64 frames achieves 24.7x cache speedup. We release our implementation as open source to support efficient inference on consumer Apple hardware.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20090v2">Should I Have Expressed a Different Intent? Counterfactual Generation for LLM-Based Autonomous Control</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-powered agents can translate high-level user intents into plans and actions in an environment. Yet after observing an outcome, users may wonder: What if I had phrased my intent differently? We introduce a framework that enables such counterfactual reasoning in agentic LLM-driven control scenarios, while providing formal reliability guarantees. Our approach models the closed-loop interaction between a user, an LLM-based agent, and an environment as a structural causal model (SCM), and leverages test-time scaling to generate multiple candidate counterfactual outcomes via probabilistic abduction. Through an offline calibration phase, the proposed conformal counterfactual generation (CCG) yields sets of counterfactual outcomes that are guaranteed to contain the true counterfactual outcome with high probability. We showcase the performance of CCG on a wireless network control use case, demonstrating significant advantages compared to naive re-execution baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19510v2">ALRM: Agentic LLM for Robotic Manipulation</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently empowered agentic frameworks to exhibit advanced reasoning and planning capabilities. However, their integration in robotic control pipelines remains limited in two aspects: (1) prior \ac{llm}-based approaches often lack modular, agentic execution mechanisms, limiting their ability to plan, reflect on outcomes, and revise actions in a closed-loop manner; and (2) existing benchmarks for manipulation tasks focus on low-level control and do not systematically evaluate multistep reasoning and linguistic variation. In this paper, we propose Agentic LLM for Robot Manipulation (ALRM), an LLM-driven agentic framework for robotic manipulation. ALRM integrates policy generation with agentic execution through a ReAct-style reasoning loop, supporting two complementary modes: Code-asPolicy (CaP) for direct executable control code generation, and Tool-as-Policy (TaP) for iterative planning and tool-based action execution. To enable systematic evaluation, we also introduce a novel simulation benchmark comprising 56 tasks across multiple environments, capturing linguistically diverse instructions. Experiments with ten LLMs demonstrate that ALRM provides a scalable, interpretable, and modular approach for bridging natural language reasoning with reliable robotic execution. Results reveal Claude-4.1-Opus as the top closed-source model and Falcon-H1-7B as the top open-source model under CaP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21283v1">DUET: Distilled LLM Unlearning from an Efficiently Contextualized Teacher</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      LLM unlearning is a technique to remove the impacts of undesirable knowledge from the model without retraining from scratch, which is indispensable towards trustworthy AI. Existing unlearning methods face significant limitations: conventional tuning-based unlearning is computationally heavy and prone to catastrophic forgetting. In contrast, in-contextualized unlearning is lightweight for precise unlearning but vulnerable to prompt removal or reverse engineering attacks. In response, we propose Distilled Unlearning from an Efficient Teacher (DUET), a novel distillation-based unlearning method that combines the merits of these two lines of work. It learns a student model to imitate the behavior of a prompt-steered teacher that effectively refuses undesirable knowledge generation while preserving general domain knowledge. Extensive evaluations on existing benchmarks with our enriched evaluation protocols demonstrate that DUET achieves higher performance in both forgetting and utility preservation, while being orders of magnitude more data-efficient than state-of-the-art unlearning methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.20802v3">SPADE: Structured Pruning and Adaptive Distillation for Efficient LLM-TTS</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 ICASSP 2026
    </div>
    <details class="paper-abstract">
      The goal of this paper is to introduce SPADE, a framework for Structured Pruning and Adaptive Distillation for Efficient Large Language Model-based text-to-speech (LLM-TTS). Recent LLM-TTS systems achieve strong controllability and zero-shot generalization, but their large parameter counts and high latency limit real-world deployment. SPADE addresses this by combining (i) a pruning step guided by a word-error-rate-based layer importance index to remove non-essential Transformer layers, with (ii) multi-level knowledge distillation to restore autoregressive coherence. On zero-shot benchmarks, SPADE preserves near-parity perceptual quality while halving Transformer depth, reducing VRAM usage by up to 20%, and achieving up to 1.7x faster real-time factor with less than 5% of the original training data. These results show that compact LLM-TTS models can maintain naturalness and speaker similarity while enabling practical real-time speech generation. Audio samples are available at https://mm.kaist.ac.kr/projects/SPADE/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18987v3">LLMs versus the Halting Problem: Revisiting Program Termination Prediction</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Determining whether a program terminates is a central problem in computer science. Turing's foundational result established the Halting Problem as undecidable, showing that no algorithm can universally determine termination for all programs and inputs. Consequently, automatic verification tools approximate termination, sometimes failing to prove or disprove; these tools rely on problem-specific architectures and abstractions, and are usually tied to particular programming languages. Recent success and progress in large language models (LLMs) raises the following question: can LLMs reliably predict program termination? In this work, we evaluate LLMs on a diverse set of C programs from the Termination category of the International Competition on Software Verification (SV-Comp) 2025. Our results suggest that LLMs perform remarkably well at predicting program termination, where GPT-5 and Claude Sonnet-4.5 would rank just behind the top-ranked tool (using test-time-scaling), and Code World Model (CWM) would place just behind the second-ranked tool. While LLMs are effective at predicting program termination, they often fail to provide a valid witness as a proof. Moreover, LLMs performance drops as program length increases. We hope these insights motivate further research into program termination and the broader potential of LLMs for reasoning about undecidable problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18196v2">LogicReward: Incentivizing LLM Reasoning via Step-Wise Logical Supervision</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      Although LLMs exhibit strong reasoning capabilities, existing training methods largely depend on outcome-based feedback, which can produce correct answers with flawed reasoning. Prior work introduces supervision on intermediate steps but still lacks guarantees of logical soundness, which is crucial in high-stakes scenarios where logical consistency is paramount. To address this, we propose LogicReward, a novel reward system that guides model training by enforcing step-level logical correctness with a theorem prover. We further introduce Autoformalization with Soft Unification, which reduces natural language ambiguity and improves formalization quality, enabling more effective use of the theorem prover. An 8B model trained on data constructed with LogicReward surpasses GPT-4o and o4-mini by 11.6\% and 2\% on natural language inference and logical reasoning tasks with simple training procedures. Further analysis shows that LogicReward enhances reasoning faithfulness, improves generalizability to unseen tasks such as math and commonsense reasoning, and provides a reliable reward signal even without ground-truth labels. We will release all data and code at https://llm-symbol.github.io/LogicReward.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21261v1">User-Centric Phishing Detection: A RAG and LLM-Based Approach</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      The escalating sophistication of phishing emails necessitates a shift beyond traditional rule-based and conventional machine-learning-based detectors. Although large language models (LLMs) offer strong natural language understanding, using them as standalone classifiers often yields elevated falsepositive (FP) rates, which mislabel legitimate emails as phishing and create significant operational burden. This paper presents a personalized phishing detection framework that integrates LLMs with retrieval-augmented generation (RAG). For each message, the system constructs user-specific context by retrieving a compact set of the user's historical legitimate emails and enriching it with real-time domain and URL reputation from a cyber-threat intelligence platform, then conditions the LLM's decision on this evidence. We evaluate four open-source LLMs (Llama4-Scout, DeepSeek-R1, Mistral-Saba, and Gemma2) on an email dataset collected from public and institutional sources. Results show high performance; for example, Llama4-Scout attains an F1-score of 0.9703 and achieves a 66.7% reduction in FPs with RAG. These findings validate that a RAG-based, user-profiling approach is both feasible and effective for building high-precision, low-friction email security systems that adapt to individual communication patterns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20420v2">Concept Component Analysis: A Principled Approach for Concept Extraction in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Developing human understandable interpretation of large language models (LLMs) becomes increasingly critical for their deployment in essential domains. Mechanistic interpretability seeks to mitigate the issues through extracts human-interpretable process and concepts from LLMs' activations. Sparse autoencoders (SAEs) have emerged as a popular approach for extracting interpretable and monosemantic concepts by decomposing the LLM internal representations into a dictionary. Despite their empirical progress, SAEs suffer from a fundamental theoretical ambiguity: the well-defined correspondence between LLM representations and human-interpretable concepts remains unclear. This lack of theoretical grounding gives rise to several methodological challenges, including difficulties in principled method design and evaluation criteria. In this work, we show that, under mild assumptions, LLM representations can be approximated as a {linear mixture} of the log-posteriors over concepts given the input context, through the lens of a latent variable model where concepts are treated as latent variables. This motivates a principled framework for concept extraction, namely Concept Component Analysis (ConCA), which aims to recover the log-posterior of each concept from LLM representations through a {unsupervised} linear unmixing process. We explore a specific variant, termed sparse ConCA, which leverages a sparsity prior to address the inherent ill-posedness of the unmixing problem. We implement 12 sparse ConCA variants and demonstrate their ability to extract meaningful concepts across multiple LLMs, offering theory-backed advantages over SAEs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21239v1">TIDE: Tuning-Integrated Dynamic Evolution for LLM-Based Automated Heuristic Design</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Although Large Language Models have advanced Automated Heuristic Design, treating algorithm evolution as a monolithic text generation task overlooks the coupling between discrete algorithmic structures and continuous numerical parameters. Consequently, existing methods often discard promising algorithms due to uncalibrated constants and suffer from premature convergence resulting from simple similarity metrics. To address these limitations, we propose TIDE, a Tuning-Integrated Dynamic Evolution framework designed to decouple structural reasoning from parameter optimization. TIDE features a nested architecture where an outer parallel island model utilizes Tree Similarity Edit Distance to drive structural diversity, while an inner loop integrates LLM-based logic generation with a differential mutation operator for parameter tuning. Additionally, a UCB-based scheduler dynamically prioritizes high-yield prompt strategies to optimize resource allocation. Extensive experiments across nine combinatorial optimization problems demonstrate that TIDE discovers heuristics that significantly outperform state-of-the-art baselines in solution quality while achieving improved search efficiency and reduced computational costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21233v1">Just Ask: Curious Code Agents Reveal System Prompts in Frontier LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 24 pages, 6 figures, 17 tables
    </div>
    <details class="paper-abstract">
      Autonomous code agents built on large language models are reshaping software and AI development through tool use, long-horizon reasoning, and self-directed interaction. However, this autonomy introduces a previously unrecognized security risk: agentic interaction fundamentally expands the LLM attack surface, enabling systematic probing and recovery of hidden system prompts that guide model behavior. We identify system prompt extraction as an emergent vulnerability intrinsic to code agents and present \textbf{\textsc{JustAsk}}, a self-evolving framework that autonomously discovers effective extraction strategies through interaction alone. Unlike prior prompt-engineering or dataset-based attacks, \textsc{JustAsk} requires no handcrafted prompts, labeled supervision, or privileged access beyond standard user interaction. It formulates extraction as an online exploration problem, using Upper Confidence Bound-based strategy selection and a hierarchical skill space spanning atomic probes and high-level orchestration. These skills exploit imperfect system-instruction generalization and inherent tensions between helpfulness and safety. Evaluated on \textbf{41} black-box commercial models across multiple providers, \textsc{JustAsk} consistently achieves full or near-complete system prompt recovery, revealing recurring design- and architecture-level vulnerabilities. Our results expose system prompts as a critical yet largely unprotected attack surface in modern agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21210v1">Uncovering Hidden Correctness in LLM Causal Reasoning via Symbolic Verification</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 EACL 2026 Main
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly being applied to tasks that involve causal reasoning. However, current benchmarks often rely on string matching or surface-level metrics that do not capture whether the output of a model is formally valid under the semantics of causal reasoning. To address this, we propose DoVerifier, a simple symbolic verifier that checks whether LLM-generated causal expressions are derivable from a given causal graph using rules from do-calculus and probability theory. This allows us to recover correct answers to causal queries that would otherwise be marked incorrect due to superficial differences in their causal semantics. Our evaluations on synthetic data and causal QA benchmarks show that DoVerifier more accurately captures semantic correctness of causal reasoning traces, offering a more rigorous and informative way to evaluate LLMs on causal reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.18874v3">When Ads Become Profiles: Uncovering the Invisible Risk of Web Advertising at Scale with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 The ACM Web Conference 2026
    </div>
    <details class="paper-abstract">
      Regulatory limits on explicit targeting have not eliminated algorithmic profiling on the Web, as optimisation systems still adapt ad delivery to users' private attributes. The widespread availability of powerful zero-shot multimodal Large Language Models (LLMs) has dramatically lowered the barrier for exploiting these latent signals for adversarial inference. We investigate this emerging societal risk, specifically how adversaries can now exploit these signals to reverse-engineer private attributes from ad exposure alone. We introduce a novel pipeline that leverages LLMs as adversarial inference engines to perform natural language profiling. Applying this method to a longitudinal dataset comprising over 435,000 Facebook ad impressions collected from 891 users, we conducted a large-scale study to assess the feasibility and precision of inferring private attributes from passive online ad observations. Our results demonstrate that off-the-shelf LLMs can accurately reconstruct complex user private attributes, including party preference, employment status, and education level, consistently outperforming strong census-based priors and matching or exceeding human social perception at only a fraction of the cost (223x lower) and time (52x faster) required by humans. Critically, actionable profiling is feasible even within short observation windows, indicating that prolonged tracking is not a prerequisite for a successful attack. These findings provide the first empirical evidence that ad streams serve as a high-fidelity digital footprint, enabling off-platform profiling that inherently bypasses current platform safeguards, highlighting a systemic vulnerability in the ad ecosystem and the urgent need for responsible web AI governance in the generative AI era. The code is available at https://github.com/Breezelled/when-ads-become-profiles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20251v2">Efficient Evaluation of LLM Performance with Statistical Guarantees</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 24 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Exhaustively evaluating many large language models (LLMs) on a large suite of benchmarks is expensive. We cast benchmarking as finite-population inference and, under a fixed query budget, seek tight confidence intervals (CIs) for model accuracy with valid frequentist coverage. We propose Factorized Active Querying (FAQ), which (a) leverages historical information through a Bayesian factor model; (b) adaptively selects questions using a hybrid variance-reduction/active-learning sampling policy; and (c) maintains validity through Proactive Active Inference -- a finite-population extension of active inference (Zrnic & Candès, 2024) that enables direct question selection while preserving coverage. With negligible overhead cost, FAQ delivers up to $5\times$ effective sample size gains over strong baselines on two benchmark suites, across varying historical-data missingness levels: this means that it matches the CI width of uniform sampling while using up to $5\times$ fewer queries. We release our source code and our curated datasets to support reproducible evaluation and future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.10163v2">Compound-QA: A Benchmark for Evaluating LLMs on Compound Questions</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Accepted to ICASSP 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate remarkable performance across various tasks, prompting researchers to develop diverse evaluation benchmarks. However, most benchmarks typically measure the ability of LLMs to respond to individual questions, neglecting the complex interactions in real-world applications. We introduce Compound Question Synthesis (CQ-Syn) to build Compound-QA, a benchmark targeting questions composed of multiple interrelated sub-questions. This benchmark is derived from existing QA datasets, annotated with proprietary LLMs, and verified by humans for accuracy. It encompasses five categories: Factual-Statement, Cause-and-Effect, Hypothetical-Analysis, Comparison-and-Selection, and Evaluation-and-Suggestion. It evaluates the LLM capability in terms of three dimensions, including understanding, reasoning, and knowledge. Evaluating nine open-source LLMs on Compound-QA reveals that their performance on compound questions is notably lower than on non-compound questions. We further explore strategies to enhance LLMs' handling of compound questions, and our results show that these methods substantially improve models' comprehension and reasoning abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21169v1">Output-Space Search: Targeting LLM Generations in a Frozen Encoder-Defined Output Space</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      We introduce Output-Space Search (OS-Search), which turns LLM generation into endpoint search. An outer loop selects a target z* in a frozen encoder-defined 3D output space Z, and a retrieval-grounded policy trained with sequence-level RL generates outputs whose coordinates land near z* under standard autoregressive decoding. This enables parallel sweeps and black-box optimization in Z without path-dependent token/program search. On stories, sweeping Z (text) yields 3.1x higher LLM-scored diversity than prompt-chaining. On code, Bayesian optimization over Z (code) improves an objective withheld from the controller under matched inference budgets while preserving validity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21164v1">Concise Geometric Description as a Bridge: Unleashing the Potential of LLM for Plane Geometry Problem Solving</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Plane Geometry Problem Solving (PGPS) is a multimodal reasoning task that aims to solve a plane geometric problem based on a geometric diagram and problem textual descriptions. Although Large Language Models (LLMs) possess strong reasoning skills, their direct application to PGPS is hindered by their inability to process visual diagrams. Existing works typically fine-tune Multimodal LLMs (MLLMs) end-to-end on large-scale PGPS data to enhance visual understanding and reasoning simultaneously. However, such joint optimization may compromise base LLMs' inherent reasoning capability. In this work, we observe that LLM itself is potentially a powerful PGPS solver when appropriately formulating visual information as textual descriptions. We propose to train a MLLM Interpreter to generate geometric descriptions for the visual diagram, and an off-the-shelf LLM is utilized to perform reasoning. Specifically, we choose Conditional Declaration Language (CDL) as the geometric description as its conciseness eases the MLLM Interpreter training. The MLLM Interpreter is fine-tuned via CoT (Chain-of-Thought)-augmented SFT followed by GRPO to generate CDL. Instead of using a conventional solution-based reward that compares the reasoning result with the ground-truth answer, we design CDL matching rewards to facilitate more effective GRPO training, which provides more direct and denser guidance for CDL generation. To support training, we construct a new dataset, Formalgeo7k-Rec-CoT, by manually reviewing Formalgeo7k v2 and incorporating CoT annotations. Extensive experiments on Formalgeo7k-Rec-CoT, Unigeo, and MathVista show our method (finetuned on only 5.5k data) performs favorably against leading open-source and closed-source MLLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20900v2">When Experts Speak:Sequential LLM-Bayesian Learning for Startup Success Prediction</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Evaluating startups is inherently challenging in entrepreneurial finance, where investors confront severe information asymmetry and limited quantitative data. Leveraging a novel expert network call data, we develop an LLM-Bayesian model that analyzes these conversations at the question-answer turn level, extracting semantic and evaluative signals via large language models (LLMs) and aggregating them in a sequential Bayesian architecture. The model dynamically updates beliefs as additional expert calls occur and attenuates contradictory assessments, which are absent from existing text-based screening tools. Empirically, our model outperforms state-of-the-art benchmarks by 6.691% in F1-score and increases portfolio-level Return on Investment by 15.255%. Attention and ablation analyses reveal that conversational cues are particularly informative for technologically complex startups, young firms, diverse founding teams, and firms with low public visibility. By converting expert dialogue into continually updated probabilities, our model advances research in entrepreneurial finance and information systems and offers policy implications for improving funding outcomes for informationally disadvantaged startups.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.06160v4">Navigating the Rabbit Hole: Emergent Biases in LLM-Generated Attack Narratives Targeting Mental Health Groups</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been shown to demonstrate imbalanced biases against certain groups. However, the study of unprovoked targeted attacks by LLMs towards at-risk populations remains underexplored. Our paper presents three novel contributions: (1) the explicit evaluation of LLM-generated attacks on highly vulnerable mental health groups; (2) a network-based framework to study the propagation of relative biases; and (3) an assessment of the relative degree of stigmatization that emerges from these attacks. Our analysis of a recently released large-scale bias audit dataset reveals that mental health entities occupy central positions within attack narrative networks, as revealed by a significantly higher mean centrality of closeness (p-value = 4.06e-10) and dense clustering (Gini coefficient = 0.7). Drawing from an established stigmatization framework, our analysis indicates increased labeling components for mental health disorder-related targets relative to initial targets in generation chains. Taken together, these insights shed light on the structural predilections of large language models to heighten harmful discourse and highlight the need for suitable approaches for mitigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.03217v2">Abstain and Validate: A Dual-LLM Policy for Reducing Noise in Agentic Program Repair</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Accepted to the 2026 IEEE/ACM 48th International Conference on Software Engineering: Software Engineering in Practice (ICSE-SEIP '26)
    </div>
    <details class="paper-abstract">
      Agentic Automated Program Repair (APR) is increasingly tackling complex, repository-level bugs in industry, but ultimately these patches still need to be reviewed by a human before committing them to ensure they address the bug. Showing patches unlikely to be accepted can lead to substantial noise, wasting valuable developer time and eroding trust in automated code changes. We introduce two complementary LLM-based policies to reduce such noise: bug abstention and patch validation policies. Bug abstention excludes bugs that the agentic APR system is unlikely to fix. Patch validation rejects patches that are unlikely to be a good fix for the given bug. We evaluate both policies on three sets of bugs from Google's codebase, and their candidate patches generated by an internal agentic APR system. On a set of 174 human-reported bugs, removing bugs and patches rejected by our policies can raise success rates by up to 13 percentage points and 15 percentage points, respectively, and by up to 39 percentage points in combination. On null pointer exceptions and sanitizer-reported bugs with machine-generated bug reports, patch validation also improves average single-sample success rates. This two-policy approach provides a practical path to the reliable, industrial-scale deployment of agentic APR systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20861v1">Evolutionary Strategies lead to Catastrophic Forgetting in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      One of the biggest missing capabilities in current AI systems is the ability to learn continuously after deployment. Implementing such continually learning systems have several challenges, one of which is the large memory requirement of gradient-based algorithms that are used to train state-of-the-art LLMs. Evolutionary Strategies (ES) have recently re-emerged as a gradient-free alternative to traditional learning algorithms and have shown encouraging performance on specific tasks in LLMs. In this paper, we perform a comprehensive analysis of ES and specifically evaluate its forgetting curves when training for an increasing number of update steps. We first find that ES is able to reach performance numbers close to GRPO for math and reasoning tasks with a comparable compute budget. However, and most importantly for continual learning, the performance gains in ES is accompanied by significant forgetting of prior abilities, limiting its applicability for training models online. We also explore the reason behind this behavior and show that the updates made using ES are much less sparse and have orders of magnitude larger $\ell_2$ norm compared to corresponding GRPO updates, explaining the contrasting forgetting curves between the two algorithms. With this study, we aim to highlight the issue of forgetting in gradient-free algorithms like ES and hope to inspire future work to mitigate these issues.
    </details>
</div>
