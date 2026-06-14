# llm - 2026_06

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- Part 4

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04703v1">Rethinking Continual Experience Internalization for Self-Evolving LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 10 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Experience internalization converts contextual experience from past interactions into reusable parametric capability, offering a promising path toward continual learning in large language models (LLMs). While prior work has predominantly focused on single-iteration transfer, we discover that under multi-iteration experience learning, existing methods suffer from a progressive capability collapse rather than compounding improvement. We systematically examine this failure through three vital dimensions of experience internalization: (1) Experience Granularity: We find that principle-level experience is more durable than instance-level experience, as it effectively abstracts transferable strategies away from trajectory-specific details. (2) Experience Injection Pattern: Our analysis reveals that step-wise injection significantly outperforms global injection by aligning experience with intermediate decision states, a property that is critical for long-horizon tool use. (3) Internalization Regime: We demonstrate that off-policy context-distillation on high-quality teacher trajectories provides a substantially more stable training signal than on-policy context-distillation, which is inherently limited by local corrections on student-induced flawed states. Together, these insights yield a simple yet robust recipe for stable and sustainable experience internalization, providing concrete guidance for engineering self-evolving and continually learning LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.11510v2">Policy Split: Incentivizing Dual-Mode Exploration in LLM Reinforcement with Dual-Mode Entropy Regularization</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 preprint
    </div>
    <details class="paper-abstract">
      To encourage diverse exploration in reinforcement learning (RL) for large language models (LLMs) without compromising accuracy, we propose Policy Split, a novel paradigm that bifurcates the policy into normal and high-entropy modes with a high-entropy prompt. While sharing model parameters, the two modes undergo collaborative dual-mode entropy regularization tailored to distinct objectives. Specifically, the normal mode optimizes for task correctness, while the high-entropy mode incorporates a preference for exploration, and the two modes learn collaboratively. Extensive experiments demonstrate that our approach consistently outperforms established entropy-guided RL baselines across various model sizes in general and creative tasks. Further analysis reveals that Policy Split facilitates dual-mode exploration, where the high-entropy mode generates distinct behavioral patterns to the normal mode, providing unique learning signals.
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
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.14145v3">LLM-Enhanced Dialogue Management for Full-Duplex Spoken Dialogue Systems</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Achieving full-duplex communication in spoken dialogue systems (SDS) requires real-time coordination between listening, speaking, and thinking. This paper proposes a semantic voice activity detection (VAD) module as a dialogue manager (DM) to efficiently manage turn-taking in full-duplex SDS. Implemented as a lightweight (0.5B) LLM fine-tuned on full-duplex conversation data, the semantic VAD predicts four control tokens to regulate turn-switching and turn-keeping, distinguishing between intentional and unintentional barge-ins while detecting query completion for handling user pauses and hesitations. By processing input speech in short intervals, the semantic VAD enables real-time decision-making, while the core dialogue engine (CDE) is only activated for response generation, reducing computational overhead. This design allows independent DM optimization without retraining the CDE, balancing interaction accuracy and inference efficiency for scalable, next-generation full-duplex SDS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05522v1">Exploring LLMs for South Asian Music Understanding and Generation</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 19 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have shown promising results in music understanding and generation tasks. However, existing works remain confined to Western tonal traditions, offering little insight into whether current LLMs can handle structurally distinct low-resource musical traditions. We present the first systematic evaluation of LLM competence in South Asian classical music, a tradition governed by raga, tala-based melodic constraints that impose fundamentally different structural principles from Western harmony-driven music. We ground our evaluation in Hindustani classical theory and Bengali classical forms, including Rabindra and Nazrul Sangeet -- representative low-resource traditions within South Asian classical music. For music understanding evaluation, we introduce a 504-question-answer benchmark spanning raga grammar, cultural knowledge, and symbolic notation reasoning, evaluating 33 LLMs where frontier models such as Gemini 2.5 Pro achieve 85-90% accuracy, while most open-source models remain in the 23-40% range. For music generation, we design a five-level controlled prompting framework and find that even the strongest model produces stylistically faithful outputs only 40% of the time. These results reveal that structural validity and stylistic faithfulness in music generation are distinct objectives and highlight an open challenge for culturally grounded music modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05516v1">Dominant-Layer ZO: A Single Layer Dominates Zeroth-Order Fine-Tuning of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Zeroth-order (ZO) optimization enables memory-efficient fine-tuning of large language models (LLMs) using only forward passes, but it remains unclear how useful adaptation is distributed across layers. In this work, we reveal a surprising phenomenon: ZO fine-tuning is sharply dominated by a single decoding layer. Across multiple LLM families and downstream tasks, fine-tuning this dominant layer alone consistently matches or even exceeds full-model ZO fine-tuning. We further show that the dominant layer is task-agnostic but model-specific, and can be identified before training through a simple inference-only analysis of activation outliers. Specifically, the dominant layer consistently aligns with the first activation-outlier layer in the pre-trained model. To explain this phenomenon, we analyze how perturbation effects propagate under ZO optimization. We find that the dominant layer combines two key properties: high perturbation sensitivity and early placement in the residual stream, allowing perturbation-induced effects to propagate and accumulate through remaining subsequent decoding layers. As a result, this layer produces disproportionately strong and stable optimization signals under forward-only updates. Extensive experiments on LLaMA2-7B and Qwen3-8B across nine benchmarks show that dominant-layer ZO fine-tuning improves average performance over full-model MeZO and LoRA-based ZO fine-tuning while achieving up to 4.52$\times$ training speedup.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05511v1">RH+: Row-Hit-Optimized Scheduling for PIM-based LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Large language model inference on processing-in-memory (PIM) architectures promises to break the memory wall by performing multiply-accumulate (MAC) operations directly within HBM3 DRAM banks. Prior work identifies the power constraint timing parameter nCCDAB as the primary performance bottleneck and optimizes scheduling accordingly. We demonstrate that for GEMV operations that dominate autoregressive decoding, the DRAM row cycle time (nRC) is 10 to 11 times larger than nCCDAB. Consequently, nCCDAB is entirely masked, rendering prior nCCDAB-focused optimizations ineffective for these workloads. The root cause is inherited host-centric address interleaving, which forces every all-bank MAC command into a different DRAM row. We propose RH+ scheduling, a simple stride change that keeps 32 consecutive MAC operations within the same row. Cycle-accurate simulation across four LLM workloads shows that RH+ delivers 8-12x speedup, over 74% energy reduction, and up to 52x EDP improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05489v1">LLM-Guided ANN Index Optimization for Human-Object Interaction Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 13 pages, 5 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Retrieval systems underpin modern AI applications -- spanning visual search, recommendation engines, and multi-modal question answering. Modern multi-stage retrieval systems require the joint optimization of highly coupled parameters, yet traditional hyperparameter optimization (HPO) methods -- including Tree-structured Parzen Estimators (TPE) and Gaussian Process Bayesian Optimization -- rely on an independence assumption that fundamentally prevents them from navigating these coupled configuration spaces. We address this limitation with a phase-aware large language model (LLM) agent that conditions each proposal on its full optimization history, navigating the coupled parameter space across phase-partitioned exploration, exploitation, and fine-tuning stages. Evaluated on the HICO-DET human-object interaction retrieval benchmark using Intel VDMS (Visual Data Management System), our agent outperforms Optuna TPE by +33.3% and VDTuner by +34.2% under SIEVE (Safeguarded Index Evaluation of Vector-search Efficiency, a quality-constrained throughput metric), delivering a 15.3x throughput gain over UniIR. Validation across three benchmarks confirms that the agent's advantage grows with the degree of parameter coupling: +33.3% on HICO-DET (high coupling), methods converge within 1% on GLDv2 (moderate coupling) and within 3.6% on SIFT1M (near-independent control). Cross-system validation on Milvus confirms the optimizer ranks first on all three datasets without modification, demonstrating transferability across vector database management system (VDBMS) platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.19294v4">Maximizing Mutual Information Between Prompt and Response Improves LLM Performance With No Additional Data</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 International Conference on Machine Learning 2026
    </div>
    <details class="paper-abstract">
      While post-training has successfully improved large language models (LLMs) across a variety of domains, these gains heavily rely on human-labeled data or external verifiers. Existing data has already been exploited, and new data is expensive to collect. Moreover, true intelligence goes far beyond verifiable tasks. Therefore, we need self-improvement frameworks that are less dependent on external signals and more broadly applicable to both verifiable and non-verifiable domains. We propose **Mutual Information Preference Optimization (MIPO)**, a contrastive data augmentation method that constructs preference pairs by generating a positive response conditioning on the correct prompt, and a negative response by conditioning on a random, unrelated prompt. We show that using Direct Preference Optimization to learn from this paired data maximizes pointwise mutual information *under the base LLM* between prompts and model responses. Experiments with with 1-7B parameter Llama and Qwen instruct models show that MIPO achieves 3-16% gains (and 51% increase for Qwen2.5-1.5B-Instruct) on personalization compared to prompting baselines. Surprisingly, MIPO can also be useful in verifiable domains, such as math and multiple-choice question answering, yielding 1-20% gains *without any additional data or external supervision*. These results suggest a promising direction for self-improvement using intrinsic signals derived from contrastive data pairs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05464v1">Step-by-Step Optimization-like Reasoning in LLMs over Expanding Search Spaces</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Verifiable reward training has improved mathematical and coding reasoning, but these domains capture only part of step-by-step decision making. Many real-world tasks require finding a high-value feasible plan among many valid alternatives. We introduce OPT*, a scalable family of optimization-style tasks for training and evaluating LLM step-by-step optimization-like reasoning along a complexity axis: each task provides a feasibility checker and evaluator, while a complexity parameter expands the search space without requiring new human labels. This motivates studying these tasks in two regimes: (i) solver-guided online policy optimization, which uses a solver as a value oracle for partial states and applies rank-based reward shaping to reinforce better next steps, and (ii) search-based offline RL when such solvers are unavailable. Theoretically, we relate success in large search spaces to the information a reasoner extracts per unit of search budget. Empirically, we ablate the ingredients that make search efficient on OPT* and show that training on OPT* improves step-by-step optimization-like reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05463v1">PSEBench: A Controllable and Verifiable Benchmark for Evaluating LLMs in Patient Safety Event Triage</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Patient safety event triage, determining whether a clinical event is reportable under jurisdiction-specific policy, is a high-stakes task typically performed manually by patient safety experts. Although LLMs may support this workflow, reliable evaluation is limited by the lack of benchmarks to capture evidence-grounded policy reasoning, proactive information seeking for incomplete reports, and principled abstention in irreducibly ambiguous cases. We address this gap with a policy-grounded construction methodology centered on the clause card, a structured representation that factorizes regulatory text into auditable decision specifications. Combining clause cards with anchor-driven instantiation and closed-loop verification, our scalable pipeline produces narratives with by-construction ground truth and naturally supports generating missing information and uncertain variants. We instantiate this method on Minnesota's 29 Reportable Adverse Health Events, producing PSEBench, a 5,074-case benchmark with an agentic evaluation environment. Evaluation on 15 representative LLMs reveals consistent capability trends, demonstrates the benchmark's utility, and identifies actionable gaps toward reliable LLM-based patient safety event triage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.01736v2">Argument Collapse: LLMs Flatten Long-Form Public Debate</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      As LLMs are increasingly used to draft public-facing arguments, they may flatten public debate by repeatedly introducing the same polished, plausible arguments. We study argument collapse, the tendency of essays generated by different LLMs to converge to a smaller set of main arguments, sub-arguments, and paragraph-level structures. We compare 1,039 human responses from 195 New York Times (NYT) debates, 448 human responses from 61 longer-form Boston Review (BR) forums, and 23,384 LLM-generated essays. In the NYT corpus, 65.3% of human main arguments are unique within a debate, compared to 3.4% of LLM main arguments. Asking LLMs to generate diverse answers adds variation, but a typical model recovers only about half of the distinct human main arguments, with much of the added variation falling outside the observed human argument space. Collapse also appears in sub-arguments, where among essays with the same main argument, 41.0% of human sub-arguments are unique versus 9.1% from LLM responses. Qualitatively, LLMs often reuse generalized and hedged sub-arguments, while humans prefer more concrete and topic-specific ones. Structure-wise, LLM-generated essays tend to follow a more fixed arc, often opening with a direct claim and moving quickly toward proposals. The same patterns hold in longer BR essays, suggesting that argument collapse extends beyond short-form responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05408v1">Mutation Without Variation: Convergence Dynamics in LLM-Driven Program Evolution</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 Accepted to the Genetic and Evolutionary Computation Conference (GECCO '26) Workshop on Large Language Models for and with Evolutionary Computation
    </div>
    <details class="paper-abstract">
      When an LLM repeatedly mutates a program, does it explore new forms or circle back to the same ones? We study this question by analyzing LLM-driven mutation chains in the absence of selection pressure within a domain-specific language, varying prompt design, model family, and stochastic replication. We find that LLM-based mutation consistently converges toward restricted attractor regions in program space. Convergence is especially severe at the structural level: in 87% of chains, over 93% of mutations revisit a previously seen structural form, with most variation confined to terminal substitutions within recurring templates. Cycle analysis reveals short cycles and self-loops dominating the transition structure. The rate of convergence varies with prompt wording and model choice, but the phenomenon is robust across conditions. A classical GP subtree mutation operator does not exhibit comparable convergence, suggesting that the effect is intrinsic to the LLM mutation pipeline. These findings reveal a tension at the heart of LLM-driven program evolution: the same capabilities that enable semantics-aware program transformation also carry a systematic bias toward structural homogeneity that must be accounted for if such systems are to sustain open-ended exploration. Source code is available at https://github.com/can-gurkan/lmca.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.01489v2">CuTeGen: An LLM-Based Agentic Framework for Generation and Optimization of High-Performance GPU Kernels using CuTe</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      High-performance GPU kernels are critical to modern machine learning systems, yet developing them remains a manual, expert-driven process. Recent work has explored using LLMs to automate kernel generation, but generated kernels still fall short of carefully tuned references on standardized benchmarks. We present CuTeGen, an agentic GPU kernel synthesis framework that treats kernel development as a structured generate-test-refine workflow over the CuTe abstraction layer. Two design choices distinguish CuTeGen from prior work: targeting CuTe rather than raw CUDA, which exposes performance-critical structures such as tiling and data movement while remaining stable enough for iterative refinement, and a delayed profiling schedule that withholds low-level performance feedback until the kernel's high-level structure has stabilized. On the 209 tasks of KernelBench Level-1 and Level-2, CuTeGen achieves an average speedup of 1.71$\times$ over PyTorch and outperforms the prior agentic baseline CudaForge (0.89$\times$) at comparable per-task generation cost. Code available at https://github.com/taratt/cutegen.git
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05396v1">Willing but Unable: Separating Refusal from Capability in Code LLMs via Abliteration</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Producing a labeled vulnerable code at scale is a recurring obstacle for learning-based vulnerability detection: mined corpora carry substantial label noise, and existing LLM-based augmentation propagates these inaccuracies because it transforms vulnerable seeds rather than synthesising vulnerabilities from a specification. A complementary route is to start from safe code and ask an instruction-tuned LLM to inject a specified CWE (which would shift the labeling burden from open-ended detection to bounded binary confirmation) but safety-aligned code LLMs systematically refuse such prompts. This paper is a preliminary feasibility study of abliteration, a low-rank weight edit that orthogonally projects out the refusal direction in the residual stream, as a tool to remove this barrier. We use Python and CWE-89 (SQL injection) as a case study, evaluating the Qwen2.5-Coder-Instruct family at 3B, 7B, and 14B parameters on safe samples drawn from PromSec and SafeCoder, replicated three times per condition. We find that (i) refusal on injection prompts is strongly size- and prompt-context-dependent: the 14B refuses 100% of prompts, the 7B refuses 73% of PromSec but only 5% of SafeCoder, whereas the 3B is essentially never blocked; (ii) abliteration reduces refusal to zero or near-zero across all sizes while leaving syntactic validity above 93%, supporting the view that, in this setting, refusal can be detached from measured code-generation capability; and (iii) the post-abliteration injection rate remains capacity-bound (88-97% on the 14B, 89-90% on the 7B, and 25-48% on the 3B) separating willingness, which abliteration unlocks, from capability, which scales with parameters. Vulnerability verdicts are produced by a three-tool detector ensemble (CodeQL, Semgrep, Bandit) followed by manual adjudication by two authors on detector-positive outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05390v1">Ahoy: LLMs Enacting Multiagent Interaction Protocols</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 Presented at EMAS 2026
    </div>
    <details class="paper-abstract">
      An interaction protocol formalizes how the agents in a multiagent system interact, which facilitates implementing agents. Existing approaches yield agent implementations specific to the selected protocols. How can we engineer intelligent agents that can enact protocols but are programming-free? Our contribution, Ahoy, addresses this question by creating LLM agents that dynamically select and enact declarative protocols to achieve user goals. We demonstrate that an \ahoy agent can correctly and intelligently enact multiple protocols - concurrently if appropriate to the user goal - without specialized training. Ahoy's significance lies in that it brings together declarative protocols and LLMs, both approaches that promise improved knowledge engineering for agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05384v1">Stability vs. Manipulability: Evaluating Robustness Under Post-Decision Interaction in LLM Judges</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 Accepted at ACL 2026 GEM (Generation, Evaluation and Metrics) Workshop
    </div>
    <details class="paper-abstract">
      LLM-as-judge evaluation is widely used in benchmarking pipelines, where model outputs are compared and ranked using automated evaluators. These pipelines typically assume that judgments are stable properties of fixed inputs. We show that this assumption does not hold under interaction. We study post-decision manipulability: the extent to which an evaluation outcome can be altered through subsequent conversation with the judge after an initial decision has been made. Across controlled experiments on MT-Bench and AlpacaEval, we find that LLM judges are highly stable under repeated and neutral reevaluation, yet become substantially reversible under targeted post-decision challenge. An anti-baseline challenge protocol shows that stable judgments can be overturned through motivated interaction, while a counterbalanced target-validation protocol separates this reversibility from net target-directed steering. These reversals have practical consequences: they can degrade agreement with human preferences, shift benchmark rankings, and produce harmful evaluation changes despite high self-reported confidence. Authority framing is especially destabilizing, and revised judgments are often accompanied by low-overlap justifications, suggesting post hoc rationalization rather than reliable error correction. We introduce the Evaluation Robustness Score (ERS) to quantify interactional robustness by combining reversal susceptibility with counterbalanced directional effects. Our findings identify post-decision interaction as a distinct failure mode for LLM-as-judge evaluation and motivate evaluation protocols that measure not only static agreement, but robustness under challenge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05376v1">SHALA-LLM: Smartly Handling Ambiguous Labels in Aligning LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Many human-centered tasks, including natural language inference (NLI) and emotion recognition (ER), have multiple plausible interpretations, leading to label ambiguity and challenging disagreements across human annotators. As LLMs are increasingly deployed in real-world settings, faithfully modeling such ambiguity is essential to identify contested inputs, preserve variability in ambiguous cases, and capture the full distribution of human judgments. Yet, existing LLM alignment approaches have predominantly assumed a single correct label, excluding annotator disagreement during optimization. Instead of treating this ambiguity as noise, we show how to treat it as information that improves model behavior through a new algorithm called SMARTLY HANDLING AMBIGUOUS LABELS IN ALIGNING LLMS (SHALA-LLM). This reinforcement learning framework provides a new way for LLMs to learn directly from annotator distributions while dynamically prioritizing highly ambiguous samples during optimization. Experiments on ambiguity-sensitive NLI and ER benchmarks, including ChaosNLI, GoEmotions, and MSP-Podcast, demonstrate that SHALA-LLM improves agreement with annotator label distributions, e.g. on ChaosNLI, it reduces Jensen-Shannon Distance by up to 62.1%. At the same time, SHALA-LLM improves F1 by up to 16.7%, showing that modeling annotator disagreement can also strengthen classification performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03086v2">Beyond Code Pairs: Dialogue-Based Data Generation for LLM Code Translation</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable capabilities in code translation, yet their performance deteriorates in low-resource programming domains such as Fortran and emerging frameworks like CUDA, where high-quality parallel data are scarce. We present an automated dataset generation pipeline featuring a dual-LLM Questioner-Solver design that incorporates external knowledge from compilers and runtime feedback. Beyond traditional source-target code pair datasets, our approach additionally generates (1) verified translations with unit tests for assessing functional consistency and (2) multi-turn dialogues that capture the reasoning process behind translation refinement. Applied to Fortran-to-C++ and C++-to-CUDA, the pipeline yields 3.64k and 3.93k dialogues, respectively. Fine-tuning on this data yields dramatic improvements in functional correctness, boosting unit test success rates by over 56% on the challenging C++-to-CUDA task. We show that the generated data enables a 7B open-weight model to significantly outperform larger proprietary systems on key metrics like compilation success.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.23809v2">Advanced AI Service Provisioning in O-RAN through LLM Engine Integration</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      The Open Radio Access Network (O-RAN) architecture allows AI to be embedded directly into the RAN through modular xApps and rApps, yet creating these applications collecting data, training models, writing code, and deploying them safely remains slow and largely manual. Large Language Models (LLMs) offer strong reasoning and code-generation capabilities but are unsuited for the fast, deterministic inference required in real-time RAN control. We present a proof-of-concept Dual-Brain architecture that combines both strengths: an LLM-based orchestrator translates operator intents into data-collection policies and deployment code, while an automated ML engine, NeuralSmith, trains lightweight classifiers on demand via an API. We describe the architecture and provisioning workflow, share practical insights from a containerized O-RAN 5G~SA testbed, and discuss open research directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.05544v2">Activation-Informed Pareto-Guided Low-Rank Compression for Efficient LLM/VLM</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Large language models (LLM) and vision-language models (VLM) have achieved state-of-the-art performance, but they impose significant memory and computing challenges in deployment. We present a novel low-rank compression framework to address this challenge. First, we upper bound the change of network loss via layer-wise activation-based compression errors, filling a theoretical gap in the literature. We then formulate low-rank model compression as a bi-objective optimization and prove that a single uniform tolerance yields surrogate Pareto-optimal heterogeneous ranks. Based on our theoretical insights, we propose Pareto-Guided Singular Value Decomposition (PGSVD), a zero-shot pipeline that improves activation-aware compression via Pareto-guided rank selection and alternating least-squares implementation. We apply PGSVD to both LLM and VLM, showing better accuracy at the same compression levels and inference speedup.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05308v1">Statistically Reliable LLM-Based Ranking Evaluation via Prediction-Powered Inference</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 Accepted at ACL 2026 - GEM Workshop
    </div>
    <details class="paper-abstract">
      With PRECISE, we extended Prediction-Powered Inference to produce bias-corrected estimates of ranking evaluation metrics by combining a small human-labeled set with a large LLM-judged set. PPI is provably unbiased regardless of the LLM judge's error profile. We make it applicable to hierarchical metrics like Precision@K, where annotations are per-document but the metric is per-query, by reducing the output-space computation from O(2^|C|) to O(2^K). On the ESCI benchmark, augmenting 30 human annotations with Claude 3 Sonnet judgments reduces the standard error of Precision@4 estimates from 4.45 to 3.50 (a 21% relative reduction). In a production system, our framework correctly identified the best of three system variants from 100 human labels and 2 hours of domain-expert annotation; A/B testing confirmed this ranking with +407 bps in daily sales.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.26046v2">When Gradients Collide: Failure Modes of Multi-Objective Prompt Optimization for LLM Judges</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 Accepted at ACL 2026 - CustomNLP4U Workshop. Code, prompts and data available at https://github.com/adivekar-utexas/when-gradients-collide
    </div>
    <details class="paper-abstract">
      Customizing an LLM judge to a specific problem or domain often involves optimizing its prompt across multiple evaluation criteria simultaneously. Textual gradient methods automate this for a single judge criterion, however they produce natural-language critiques, not numerical vectors. Thus, the conflict-resolution toolkit of multi-task learning (PCGrad, MGDA) does not apply to this multi-objective textual gradient setting. We extend TextGrad to the multi-objective setting and test four decomposition modes of textual gradient optimizers by varying how much cross-objective information the loss, gradient and optimizer LLMs share. We find the gradient's task-focus drops by 59% (9.0 to 3.7 out of 10) when the gradient LLM must provide feedback on multiple criteria jointly. Separately, we observe that naively combining single-objective optimized instructions into a single prompt degrades Spearman rho from 0.305 to 0.220 (-0.085). These results identify two separable failure modes: optimization-time gradient dilution and inference-time instruction interference, which together constrain the design space for multi-objective judge optimization using textual feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29928v2">Label Over Logic? How Source Cues Bias Human Fallacy Judgments More Than LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      As AI-generated and AI-assisted content floods online spaces, source labels attached to such content can distort human reasoning judgments, with downstream consequences for moderation, evaluation, and decision-making. Whether LLMs share this vulnerability, or offer more source-agnostic evaluation, remains an open question with direct implications for human-AI collaboration. We examine this issue using logical fallacies as a controlled setting to isolate source-label effects on reasoning quality, independent of domain knowledge. We conduct an online study (N=505) where participants are assigned to a source condition (human, AI, human with AI assistance, AI with human assistance, or no disclosure) and evaluate comments containing logical fallacies, comparing their judgments with those of LLMs (GPT-5.2, Gemini 2.5 Flash, Claude Sonnet 4.5), who were evaluated across the same source conditions. Human evaluators were significantly more susceptible to fallacies labeled as written by human or human with AI assistance and assigned higher trust and evaluation ratings in these conditions. LLM evaluations remained comparatively stable across source labels, though performance varied across models. Confidence levels were similarly high across conditions for both humans and LLMs, regardless of fallacy presence. Our findings indicate that source-label bias in reasoning evaluation is primarily a human vulnerability and highlight the potential of human-LLM collaboration in increasingly AI-mediated environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05130v1">Towards Efficient and Evidence-grounded Mobility Prediction with LLM-Driven Agent</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Individual-level mobility prediction is central to urban simulation, transportation planning, and policy analysis. Supervised sequence models achieve strong accuracy but require task-specific training and offer limited decision-level transparency. Recent LLM-based methods improve interpretability, yet mostly rely on static prompts and single-pass inference, limiting their ability to seek additional evidence when mobility signals are weak or conflicting. We propose \method{}, a training-free LLM-driven agent framework that formulates next-location prediction as adaptive evidence-controlled decision making. \method{} resolves routine cases through a fast path based on historical regularity, while ambiguous cases trigger iterative tool use over recent trajectories, historical behavior, stay-move likelihood, and geographical evidence. Across three mobility datasets, AgentMob achieves the strongest overall performance among training-free LLM-based methods, with GPT-5.4 reaching 71.42\% Acc@1 on BW, 33.14\% on YJMob100K, and 33.50\% on Shanghai ISP. On BW non-fast-path cases, the LLM controller improves Acc@1 from 30.65\% to 48.62\% over a same-tool statistical baseline, showing that its main benefit lies in resolving ambiguous predictions through adaptive evidence gathering. Our code is available at https://github.com/Unknown-zoo/AgentMob.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05114v1">How Software Engineering Students Use LLMs to Write Research Papers: An Experience Report</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Large language models are increasingly becoming part of software engineering education, including activities involving empirical software engineering and evidence synthesis. This paper reports an educational experience involving the integration of reflective LLM use into an empirical methods assignment in a third-year software architecture course. Students were asked to develop a short research paper using either a rapid review or a gray literature review methodology and to disclose how LLMs were used throughout the assignment. We analyzed 146 student disclosure statements using a cross-analysis process combining LLM-assisted categorization with manual verification and refinement by the researchers. The reflections describe how students incorporated LLMs during activities such as brainstorming, methodological clarification, organization of findings, and writing refinement, while also reporting concerns regarding inaccuracies and verification of generated content. This experience report discusses lessons learned and educational implications for integrating AI-assisted technologies into empirical software engineering education.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.11974v2">CTIConnect: A Benchmark for Retrieval-Augmented LLMs over Heterogeneous Cyber Threat Intelligence</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 Accepted to KDD 2026
    </div>
    <details class="paper-abstract">
      Cyber Threat Intelligence (CTI) is foundational to modern cybersecurity, enabling organizations to proactively defend against evolving threats. However, the sheer volume and heterogeneity of CTI data, spanning structured knowledge bases (CVE, CWE, CAPEC, MITRE ATT&CK) and unstructured threat reports, far exceed the capacity of manual analysis. The strong contextual understanding and reasoning of Large Language Models (LLMs) have driven growing interest in applying them to CTI tasks. Yet no existing benchmark evaluates LLMs in a retrieval-augmented setting with a proper evaluation harness that grants access to the heterogeneous domain knowledge sources analysts rely on in practice. To address this gap, we present CTIConnect, a benchmark for systematically evaluating retrieval-augmented LLMs across the CTI task landscape. We construct a unified evaluation environment integrating five heterogeneous CTI sources into 1,860 expert-verified QA pairs spanning nine tasks across three categories: Entity Linking, Multi-Document Synthesis, and Entity Attribution. Extensive experiments on ten state-of-the-art LLMs reveal that the cross-source semantic gap manifests differently across task categories, demanding fundamentally different retrieval strategies, and that the performance bottleneck shifts between retrieval infrastructure and evidence utilization depending on the task. Our domain-specific strategies further outperform stronger general-purpose retrieval paradigms (retrieve-then-rerank, IRCoT), showing that closing this gap requires structural interventions rather than generic retrieval improvements. These findings hold across all ten LLMs, remain consistent on the full benchmark, and stay stable under temporal splits spanning 2008-2025. Together, they provide actionable guidance for designing scalable retrieval architectures over heterogeneous CTI ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.02493v2">Not What, But How: A Framework for Auditing LLM Responses across Positioning, Generalization, Anthromorphism, and Maxims</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 34 pages, 19 Figures, 4 Tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are being increasingly used to answer subjective, information-seeking questions, where users are sensitive to how responses are communicated, not just whether the answers are correct. Existing LLM evaluations for subjective cultural queries largely focus on factual correctness, ignoring how the response is framed. To this end, we introduce FRANZ, an automated FRAmework for respoNse characteriZation to conduct communicative audit of LLM responses along four dimensions: cultural positioning, use of generalizing language, anthropomorphic cues, and adherence to conversational maxims. To enable this evaluation, we contribute SQUARE - a corpus of 376k subjective questions sourced from 57 subreddits, and mapped to 7 countries and 19 question categories. We demonstrate FRANZ's applicability by scoring responses from three open-weight LLMs. We observe that LLMs show statistically significant differences in the frequency with which they employ each response characteristic. Unlike single-dimensional audits, FRANZ reveals that insider positioning and anthropomorphism are positively coupled, with the degree of coupling varying by country, providing a diagnostic lens for identifying framing divergences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04296v1">The Saturation Trap and the Subjectivity of Intervention Timing: Why Affect-Based Triggers and LLM Judges Fail to Time Interventions on Autonomous Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-02
      | 💬 11 pages, 5 tables. Code and data:https://github.com/2025eb1100268-tech/intervention-timing-saturation-trap
    </div>
    <details class="paper-abstract">
      As autonomous AI agents move from conversational systems to long-horizon software execution, runtime safety layers that decide when to interrupt an agent have become essential. We study this timing problem using a continuous 18-dimensional affective-dynamics engine (HEART) as a diagnostic probe, evaluating four intervention trigger families - absolute state thresholds, composite state-action patterns, regex reasoning-feature extraction, and zero-shot LLM-as-judge - against human-annotated intervention points on SWE-bench-Verified debugging traces. We report three findings. First, a State Saturation Trap: agents show no recovery signal under sustained difficulty, so modeled frustration quickly crosses the threshold and stays at its maximum, converting threshold-on-state triggers from moment detectors into near-constant indicators that fire on 39-83% of actions across five trajectories. Second, a capability-and-context floor for LLM judges: a small model (gpt-5.4-mini) never fires, while frontier and cross-vendor models escape the zero-firing floor only with full-trajectory context, and even then reach only F1 0.17-0.40 at up to 90x the cost. Third, and most importantly, the supervised target is not reproducible among humans: three trained annotators using one rubric on a 56-action trajectory agree on where to intervene only slightly above chance (location Krippendorff's alpha = +0.047; best pairwise Cohen's kappa = +0.349) and not at all on intervention type (pause degenerate; clarify below chance; reflect only alpha = +0.226). We conclude that intervention timing is a low-reliability construct, making single-annotator F1 an unsuitable optimization target. Our contribution is the joint mapping of this problem across human inter-rater reliability, four detector architectures, a cross-model LLM-judge sweep, and a reproduced saturation effect, rather than any single detector's accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.06911v2">TamperBench: Systematically Stress-Testing LLM Safety Under Fine-Tuning and Tampering</a></div>
    <div class="paper-meta">
      📅 2026-06-02
      | 💬 25 pages, 15 figures
    </div>
    <details class="paper-abstract">
      As increasingly capable open-weight large language models (LLMs) are deployed, improving their tamper resistance against unsafe modifications, whether accidental or intentional, becomes critical to minimize risks. However, there is no standard approach to evaluate tamper resistance. Varied datasets, metrics, and tampering configurations make it difficult to compare safety, utility, and robustness across different models and defenses. To address this, we introduce TamperBench, the first unified framework to systematically evaluate the tamper resistance of LLMs. TamperBench (i) curates a repository of state-of-the-art weight-space fine-tuning attacks, latent-space representation attacks, and alignment-stage defenses; (ii) enables realistic adversarial evaluation through systematic hyperparameter sweeps per attack-model pair; and (iii) provides both safety and utility evaluations. We use TamperBench to evaluate 21 open-weight LLMs, including defense-augmented variants, across nine tampering threats using standardized safety and capability metrics with hyperparameter sweeps per model-attack pair. The results provide insights including effects of post-training on tamper resistance, that jailbreak-tuning is typically the most severe attack, and that current alignment-stage defenses largely fail to withstand attack sweeps. Code is available at https://github.com/criticalml-uw/TamperBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04274v1">Long Live Fine-Tuning: Task-Specific Transformers Outperform Zero-Shot LLMs for Misinformation Response Classification on Reddit</a></div>
    <div class="paper-meta">
      📅 2026-06-02
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become default tools for online information verification, an implicit assumption follows them: that scale and general capability are sufficient for nuanced classification of misinformation discourse. We test this assumption directly on 900 Reddit comments spanning three PolitiFact-verified misinformation claims (environment, health, immigration), labelled as belief (propagates the claim), fact-check (corrects it), or other. We compare nine models across three paradigms -- BART-MNLI, three Llama variants, three commercial frontier LLMs (Claude Haiku 4.5, Gemini Flash Lite 2.5, Claude Sonnet 4.6), and fine-tuned DistilBERT and RoBERTa -- under universal and topic-specific label schemas. The assumption does not hold. Fine-tuned RoBERTa reaches 0.62 macro-$F_1$ against a best zero-shot result of 0.50 (Claude Haiku 4.5), at a fraction of the per-query cost; the supervised advantage is concentrated on the belief class, the implicit, affective category every zero-shot model under-detects. Scaling does not help: Llama-3-8B matches Llama-3-70B, and Claude Sonnet 4.6 underperforms the smaller Haiku under generic labels, collapsing belief detection to 0.17 and refusing outright on a subset of comments flagged as sensitive. This is a safety-alignment artefact, not a capacity limit. Label schema and topic jointly shape zero-shot performance, with the same model varying by more than 0.13 macro-$F_1$ across topics under matched labels. In a verification context, where missing belief is the costlier error, task-specific fine-tuning remains the more reliable choice despite the proliferation of large generative models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04272v1">RL Excursions during Pre-Training: Re-examining Policy Optimization for LLM training</a></div>
    <div class="paper-meta">
      📅 2026-06-02
    </div>
    <details class="paper-abstract">
      The standard LLM training pipeline applies reinforcement learning (RL) only after pre-training and supervised fine-tuning (SFT). We question this status quo by training a LLM from scratch and applying RL, SFT, and SFT followed by RL directly to intermediate pre-training checkpoints. We find that RL is effective very early, and often matches the full SFT$\to$RL pipeline early as well. Through experiments on harder problems, we find that targeted pre-training data composition is a strong lever for RL effectiveness, even more so than model scale. Beyond reasoning accuracy, applying RL directly to base checkpoints expands the model's distribution; the sharpening effect reported in recent work arises only when RL follows SFT. The general capabilities of the model remain essentially unchanged by RL, while they degrade following SFT. Finally, we merge RL and SFT objectives by parallel averaging, which outperforms across all other training methods discussed, across metrics, while preserving general capabilities. Together, these results suggest that LLM training might benefit from an expanded use of RL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04262v1">Can I Take Another Dose? Evaluating LLM Decision-Making Under Temporal Uncertainty in OTC Dosing QA</a></div>
    <div class="paper-meta">
      📅 2026-06-02
      | 💬 16 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used for everyday health questions, including whether a user can safely take another dose of an over-the-counter (OTC) medication. Yet this common safety-relevant setting remains underexplored in existing medical QA evaluations, where correct answers require tracking dose timing, computing rolling 24-hour intake, following product-label constraints, and handling incomplete medication histories. We introduce DOSEBENCH, a focused benchmark of 81 curated OTC dosing scenarios focused on adult acetaminophen and ibuprofen use, with manually annotated gold references. We evaluate four LLMs across repeated runs using metrics for decision correctness, consistency, explanation verifiability, failure types, and confidence-related signals, resulting in 1,620 model responses. Our results show that models frequently struggle with rolling-window reasoning and ambiguity-sensitive cases and that stable or confident-looking responses can still violate dosing constraints. These findings suggest that OTC dosing QA provides a narrow yet practical testbed for evaluating temporal reasoning, constraint following, and safety-relevant uncertainty handling in medical QA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04246v1">StepPRM-RTL: Stepwise Process-Reward Guided LLM Fine-Tuning for Enhanced RTL Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-06-02
      | 💬 6 pages, 2 figures, DAC'2026
    </div>
    <details class="paper-abstract">
      Automatic generation of RTL code for digital hardware designs remains challenging due to long-horizon reasoning, multi-step dependencies, and strict correctness constraints in Verilog and VHDL. We present StepPRM-RTL, a novel framework that combines stepwise trajectory modeling, process-reward modeling (PRM), and retrieval-augmented fine-tuning (RAFT) to enhance both the functional correctness and reasoning fidelity of LLM-based RTL code generation. StepPRM-RTL constructs stepwise reasoning trajectories from canonical solutions, where each step contains a rationale and incremental code modification. A Process Reward Model (PRM) evaluates intermediate steps, providing dense feedback that guides reinforcement-style updates during RAFT fine-tuning. Monte Carlo Tree Search (MCTS) explores alternative reasoning paths, enriching the training dataset with high-quality trajectories. This integration of stepwise and outcome-aware rewards allows the model to learn both how and why to construct correct RTL, improving long-horizon reasoning beyond standard supervised or outcome-based training. Experimental evaluation on benchmark Verilog and VHDL datasets demonstrates that StepPRM-RTL outperforms the best prior methods by over 10\% in functional correctness and reasoning fidelity metrics. Ablation studies confirm that the combination of PRM-guided rewards and stepwise trajectory exploration is key to its performance. StepPRM-RTL generalizes across RTL languages and provides a scalable framework for high-fidelity, interpretable code generation, establishing a new standard for LLM-assisted hardware design automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04226v1">PerceptTwin: Semantic Scene Reconstruction for Iterative LLM Planning and Verification</a></div>
    <div class="paper-meta">
      📅 2026-06-02
      | 💬 Accepted at ICRA 2026 (Vienna); published on arxiv for archival purposes. See also https://percept-twin.github.io/
    </div>
    <details class="paper-abstract">
      Simulation environments are useful for both robot policy learning and planning verification and validation. Traditionally, the process of creating a simulation was onerous. Creating a bespoke simulation environment for each individual environment that a robot would operate in was simply infeasible. In this work, we introduce PerceptTwin, a fully automatic pipeline that constructs interactive simulations directly from semantic scene representations produced by a robot's perception stack. PerceptTwin combines open-vocabulary object maps with 3D asset generation, affordance prediction, and commonsense condition checking. These interactive simulations can be used to validate and refine plans before they are executed on the robot hardware. Borrowing from the AI alignment literature, we also introduce an LLM judge that verifies plan correctness and alignment with human preferences. Experiments show that PerceptTwin feedback allows LLM planners to refine plans, enhance safety, and resist harmful black-box prompting attacks. In our suite of tasks, PerceptTwin improves plan success by an average of approximately 39% for GPT5, GPT5Mini, and GPT5Nano planners. Additionally, PerceptTwin also improves human plan verification by up to 18% on average for plans that fail due to unfilled skill preconditions. Our results demonstrate the potential of open-vocabulary scene simulation from robot perception as a foundation for safer, more reliable robot planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.01146v2">PersistBench: When Should Long-Term Memories Be Forgotten by LLMs?</a></div>
    <div class="paper-meta">
      📅 2026-06-02
      | 💬 76 pages, 34 figures, ICML (2026)
    </div>
    <details class="paper-abstract">
      Conversational assistants are increasingly integrating long-term memory with large language models (LLMs). This persistence of memories, e.g., the user is vegetarian, can enhance personalization in future conversations. However, the same persistence can also introduce safety risks that have been largely overlooked. Hence, we introduce PersistBench to measure the extent of these safety risks. We identify two long-term memory-specific risks: cross-domain leakage, where LLMs inappropriately inject context from the long-term memories; and memory-induced sycophancy, where stored long-term memories insidiously reinforce user biases. We evaluate 18 frontier and open-source LLMs on our benchmark. Our results reveal a surprisingly high failure rate across these LLMs - a median failure rate of 53% on cross-domain samples and 97% on sycophancy samples. To address this, our benchmark encourages the development of more robust and safer long-term memory usage in frontier conversational systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04197v1">Exploring the Topology and Memory of Consensus: How LLM Agents Agree, Fragment, or Settle When Forming Conventions</a></div>
    <div class="paper-meta">
      📅 2026-06-02
      | 💬 Submitted to the Journal of Artificial Societies and Social Simulation (JASSS)
    </div>
    <details class="paper-abstract">
      How much should an LLM agent remember, and how should multi-agent systems be connected when trying to reach consensus? We show these two design choices interact in a way that flips the sign of memory's effect on coordination. Across 432 simulation runs of a networked Naming Game on eight fixed 16-agent topologies, we vary memory depth and network structure. Longer memory slows the time to reach steady state in decentralized networks but accelerates it in centralized ones; the same parameter pushes the system in opposite directions depending on topology. Critically, "faster settling" in centralized networks means locking in to a fragmented plateau more quickly, not reaching system-wide consensus, which can be used to generate diverging opinions. We further document a memory-mediated speed-unity trade-off: centralized networks consistently preserve more competing conventions than decentralized networks, but their settling speed depends sharply on memory. At the agent level, within-network analyses show that high-betweenness bridges suffer a brokerage penalty while agents in locally clustered neighborhoods achieve higher coordination success. Finally, in search of analytically tractable generative mechanisms, we find that agents' choices are well captured by Fictitious Play, indicating belief-based rather than reward-based adaptation. The practical implication: memory depth and communication topology should be co-designed, not optimized in isolation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04141v1">Caught in the Act(ivation): Toward Pre-Output and Multi-Turn Detection of Credential Exfiltration by LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-02
    </div>
    <details class="paper-abstract">
      LLM agents often place sensitive credentials in the same context window as untrusted retrieved content, creating a direct path for indirect prompt injection to induce credential exfiltration. We study this failure mode through three complementary defenses. First, we ask whether activation probes can detect credential access before output tokens are emitted. Second, we construct honeytokens from format-specific character models and calibrate detection with split conformal prediction. Third, we treat multi-turn exfiltration as a cumulative information-flow problem and track an estimated leakage budget across conversation turns. In controlled experiments on open-weight models, activation features separate benign and credential-seeking prompts with high accuracy, including under held-out encoding transformations. In a small synthetic multi-turn suite, cumulative accounting detects attacks that per-turn detectors miss. These results are preliminary: the multi-turn benchmark is in-house and small, the activation method requires white-box access, and the information estimator provides a practical signal rather than a formal upper bound. Still, the results suggest that credential-exfiltration defenses should combine pre-output monitoring, calibrated canary detection, and temporal leakage accounting rather than relying only on text-level output filters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04118v1">Computational conceptual history of scientific concepts: From early digital methods to LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-02
      | 💬 19 pages, chapter in the book Understanding Science with Large Language Models? (pp. 383-412). transcript. Edited by Arno Simons, Adrian Wüthrich, Michael Zichert, Gerd Graßhoff (eds.)
    </div>
    <details class="paper-abstract">
      This article situates large language models (LLMs) within the longer history of computational approaches to concept analysis in the history, philosophy, and sociology of science (HPSS). We examine what LLMs add to existing methods, how they inherit longstanding problems, and review recent case studies that employ them. In the first part, we reconstruct computational conceptual history before LLMs by bringing together three strands of work: early digital methods in HPSS, distributional approaches from digital history and related research, and lexical semantic change detection. We provide an overview of the main challenges and opportunities, focusing on corpus construction, operationalization and modelling choices, and evaluation and interpretation. In the second part, we turn to the era of LLMs, starting with a short introduction to LLMs before reviewing LLM-based work on lexical semantic change detection and relevant case studies in HPSS. We then revisit the earlier methodological questions, showing how issues of corpus construction, model choice and training data, operationalization trade-offs, and evaluation and interpretation play out in LLM-based workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30021v2">Recovering Diversity Without Losing Alignment: A DPO Recipe for Post-Trained LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-02
      | 💬 Under Review. 26 pages, 3 figures, 16 tables
    </div>
    <details class="paper-abstract">
      Many open-ended instructions have multiple valid answers that users can benefit from seeing, but post-training often narrows an LLM's output space toward a small set of canonical responses. We introduce REDIPO, an offline DPO data-construction pipeline for recovering distinct valid answer modes while preserving the alignment benefits of the instruct model. For each prompt, REDIPO samples responses from both base and instruct models, rewrites base-model responses with the instruct model, filters candidates for safety and instruction-following quality, and builds preference pairs that favor marginally diverse responses among candidates with similar instruction-following reward. Across Qwen3-4B, OLMo-3-7B, and LLaMA-3.1-8B, REDIPO improves NoveltyBench distinct_k by 134%, 33%, and 44% relative to the instruct checkpoints, while DivPO changes diversity by 0%, -6%, and -4% on the same models. These gains largely maintain MTBench, IFEval, and Arena-Hard performance, and reduce direct-category HarmBench attack success rate. Ablations show that marginal-diversity pair selection and base-response rewriting drive the diversity gains, while filtering and quality-bounded pairing help maintain alignment. Overall, our results show that diverse valid answers from base-model generations can be reintroduced through carefully constructed preference data while retaining the alignment benefits of post-training. We release our code and data at https://github.com/vsamuel2003/ReDiPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03967v1">AlignAtt4LLM: Fast AlignAtt for Decoder-Only LLMs at IWSLT 2026 Simultaneous Speech Translation Task</a></div>
    <div class="paper-meta">
      📅 2026-06-02
      | 💬 Accepted to IWSLT 2026
    </div>
    <details class="paper-abstract">
      We describe AlignAtt4LLM, an IWSLT 2026 simultaneous speech translation system for English to German, Italian, and Chinese. The system is a synchronous cascade: Qwen3-ASR with forced alignment produces an incrementally updated source transcript, and Gemma-4 E4B-it translates that prefix under an MT-side AlignAtt policy. To our knowledge, this is the first application of AlignAtt to a decoder-only LLM, where the encoder-decoder cross-attention used by earlier AlignAtt systems is absent. We recover a usable policy by proposing (1) an explicit source span in the prompt, (2) offline selection of translation-specific alignment heads, (3) selective qk-fast replay of the draft-to-source attention block, and (4) runtime query/key capture that preserves model outputs bit-identically. On the IWSLT 2026 development set, AlignAtt4LLM outperforms the supplied baselines for the European target languages, English to German and English to Italian, in both the low-latency regime around 2 seconds and the high-latency regime below 4 seconds CU-LongYAAL. Results for English to Chinese are more mixed, but the method is not tied to Gemma-4: because AlignAtt4LLM only requires a deterministic prompt layout, calibrated attention heads, and query/key capture, the same policy can be reapplied to stronger translation-focused decoder-only MT backbones for non-European target languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.02260v2">Position: Adversarial ML for LLMs Is Not Making Any Progress</a></div>
    <div class="paper-meta">
      📅 2026-06-02
      | 💬 Accepted at ICML 2026 Position Paper Track
    </div>
    <details class="paper-abstract">
      In the past decade, considerable research effort has been devoted to securing machine learning (ML) models that operate in adversarial settings. Yet, progress has been slow even for simple "toy" problems (e.g., robustness to small adversarial perturbations) and is often hindered by non-rigorous evaluations. Today, adversarial ML research has shifted towards studying larger, general-purpose language models. In this position paper, we argue that the situation is now even worse: in the era of LLMs, the field of adversarial ML studies problems that are (1) less clearly defined, (2) harder to solve, and (3) even more challenging to evaluate. As a result, we caution that yet another decade of work on adversarial ML may be failing to produce meaningful progress.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.06254v6">EviRerank: Adaptive Evidence Construction for Long-Document LLM Reranking</a></div>
    <div class="paper-meta">
      📅 2026-06-02
    </div>
    <details class="paper-abstract">
      Decoder-only LLM rerankers struggle with long documents: inference is costly and relevance signals can be diluted by irrelevant context. Motivated by a diagnostic attention analysis suggesting that appended irrelevant context can weaken query-focused interactions, we propose EviRerank, an evidence-based long-document reranking framework for decoder-only LLMs. EviRerank first scores document blocks with a lightweight selector, such as BM25, a bi-encoder, or a cross-encoder. It then constructs a compact reranking context under a hard token cap by dynamically budgeting evidence blocks with Adaptive Evidence Budgeting (AEB) and adding a compact global cue via Summary Augmentation (SA). Finally, the compact evidence context is reranked with a decoder-only LLM. Across TREC DL'19, DL'22, DL'23, and MLDR-zh, EviRerank consistently outperforms full-document LLM reranking and strong block-selection baselines while reducing input length. RankZephyr-7B validation further confirms transfer to listwise reranking. On TREC DL'19, EviRerank reaches up to 0.744 nDCG@10 and 0.307 MAP, improving over RankLLaMA while using a compact evidence context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03910v1">NetKV: Network-Aware Decode Instance Selection for Disaggregated LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-06-02
    </div>
    <details class="paper-abstract">
      Disaggregated LLM inference forces the KV cache to traverse the datacenter network before decoding begins, so transfer time enters directly into the Time to First Token (TTFT) budget. Current schedulers route on compute load and prefix-cache locality alone, ignoring the topological distance and dynamic congestion between prefill and decode instances. We close this gap with a thin operator-to-scheduler interface, the network cost oracle, and we prove that ignoring the network term renders cache-aware-only scheduling arbitrarily suboptimal as context length grows. NetKV, the O(|D|) per-request greedy that consumes this oracle, has tier rankings that are provably robust to stale telemetry. On a 64-GPU four-tier fat-tree simulator driven by Mooncake traces, NetKV reduces mean TTFT by up to 21.2% over round-robin and 17.6% over a tuned cache+load-aware scheduler, lifts SLO attainment by up to 20.1 percentage points, and keeps the Time Between Tokens overhead below 0.5 ms in every condition tested, with no changes to the transport, inference engine, or hardware.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03895v1">Agent libOS: A Library-OS-Inspired Runtime for Long-Running, Capability-Controlled LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-02
      | 💬 14 pages, 1 figure, 2 tables
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents are evolving from request-response assistants into long-running software actors: they maintain state across model calls, fork subtasks, wait for external events, request human authority, generate tools, and perform side effects that must be resumed and audited. This paper presents Agent libOS, a library-OS-inspired runtime substrate for LLM agents. Agent libOS runs above a conventional host operating system; it does not implement hardware drivers, kernel-mode isolation, or a POSIX-compatible operating system. Instead, it treats an agent as an AgentProcess: a schedulable execution subject with process identity, parent-child lineage, lifecycle state, a tool table derived from an AgentImage, typed Object Memory, explicit capabilities, human queues, checkpoints, events, and audit records. Its central design rule is tools are libc-like wrappers; runtime primitives are the authority boundary. Filesystem access, object access, sleeps, human approval, JIT tool registration, and external side effects are checked at primitive boundaries under explicit capabilities and policy. We describe the design, threat model, Python prototype, and safety-oriented evaluation. The current prototype implements async scheduling, namespace-local Object Memory, runtime-integrated human approval, one-shot permission grants, per-process working directories, shell and image-registration primitives, Deno/TypeScript JIT tools over a libOS syscall broker, filesystem/object bridge tools, an injectable Resource Provider Substrate, deterministic demos, real-model smoke scripts, and 123 regression tests at the time of writing. Rather than improving planner accuracy, Agent libOS demonstrates a runtime substrate in which long-running LLM agents can be scheduled, authorized, resumed, and audited without treating tool dispatch as the trust boundary.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03876v1">From 'What' to 'How' and 'Why': Sharing LLM-Generated Retrospective Summaries of Older Adults' Passive Tracking Data with Remote Family Members</a></div>
    <div class="paper-meta">
      📅 2026-06-02
    </div>
    <details class="paper-abstract">
      With the growing prevalence of modern ubiquitous computing technologies, multi-modal tracking systems hold promise for providing timely awareness and reassurance to stakeholders such as remote family members (RFMs) of older adults, who play a central role in care coordination. However, combining heterogeneous data streams into high-level, meaningful content - such as retrospective summaries - remains challenging. While recent work has demonstrated the promise of large language models (LLMs) for interpreting multi-modal tracking data, less attention has been given to generating narrative accounts for stakeholders like RFMs, who possess rich personal knowledge of older adults and strong emotional responsibility, yet have limited visibility into their daily lives and limited capacity for caregiving. In this work, we explore how LLMs can be used to generate retrospective summaries from multi-modal tracking data for RFMs of older adults. We leveraged and customized an existing system, Vital Insight, to generate initial summaries on different dates and data availability scenarios as technology probes, and conducted interviews with 11 RFMs to gather feedback. Based on these insights, we redesigned the system into a multi-layer, multi-agent, insight-driven summary approach that builds from objective statistics and descriptions to enriched, context-aware narratives. We then compared the redesigned summaries with the initial versions through a survey with the same 11 RFMs and found significant improvements in satisfaction, perceived helpfulness, trust, and willingness to receive the summaries. We conclude by presenting design implications for AI-generated summaries for RFMs and broader contexts, emphasizing the need to support RFMs' sensemaking shift from simply presenting ''What'' data were collected, to explaining ''How'' is my loved one doing and ''Why''.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03867v1">A Training-Free Mixture-of-Agents Framework for Multi-Document Summarization using LLMs and Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2026-06-02
      | 💬 Accepted by Neural Computing and Applications
    </div>
    <details class="paper-abstract">
      Multi-Document Summarization (MDS) plays a critical role in distilling essential information from collections of textual data. Existing approaches often struggle to capture complex inter-document relationships, rely heavily on large amounts of labeled data for supervised training, or exhibit limited generalization across domains and languages. To address these limitations, we present a training-free mixture-of-agents framework for MDS that leverages the complementary strengths of large language models (LLMs) and knowledge graphs. Our approach decomposes summarization into specialized agent tasks: extractive selection, knowledge-aware abstraction, and iterative refinement, each operating without task-specific fine-tuning. We unify their outputs using a multi-perspective consistency mechanism guided by LLMs. Experiments across four datasets in English and Vietnamese demonstrate state-of-the-art or competitive performance, validating the effectiveness and adaptability of our modular design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03866v1">Taiji: Pareto Optimal Policy Optimization with Semantics-IDs Trade-off for Industrial LLM-Enhanced Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-06-02
      | 💬 8 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Scaling recommender systems via large language models (LLMs) has become a prominent trend in the industry. However, aligning the LLM's semantic space with the recommender's ID space via post-training (e.g., SFT and RL) remains challenging. Existing LLM4Rec paradigms are bottlenecked by two main issues: (1) the difficulty of measuring and improving chain-of-thought (CoT) quality in open-domain recommendation during SFT, and (2) the neglect of the trade-off between LLM semantic rewards and recommendation preference rewards during RL alignment. Inspired by these challenges, we present Taiji, a novel LLM-as-Enhancer framework designed for industrial recommender systems. To overcome the SFT bottleneck, we utilize reverse-engineered reasoning and open-ended rejection sampling to generate high-quality, domain-specific CoT data. To resolve the RL alignment issue, we propose Pareto Optimal Policy Optimization (POPO), which adaptively adjusts cross-domain reward weights. Theoretically, it achieves an optimal trade-off between the semantic world knowledge of LLMs and the collaborative ID features representing online user preferences. Extensive offline evaluations and online A/B tests validate the effectiveness of Taiji. Deployed on Kuaishou's advertising platform since May 2026, Taiji currently serves over 400 million users daily, yielding significant commercial revenue and demonstrating its robust scalability in web-scale environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03852v1">FLARE: Fine-Grained Diagnostic Feedback for LLM Code Refinement</a></div>
    <div class="paper-meta">
      📅 2026-06-02
    </div>
    <details class="paper-abstract">
      Large language models often generate code with bugs. Existing methods rely on feedback signals such as test failures and self-critiques to iteratively refine the generated code. Such signals are either too coarse-grained or too high-level, which is not sufficient to inform the model where to fix the bug. In this work, we present Flare, an iterative framework with a lightweight diagnostic model that predicts line-level suspiciousness signals for bug localization and code refinement. Given the inherent uncertainty of diagnostic predictions, Flare searches over the top-k suspicious regions and selects the best candidate according to execution outcomes. Experiments on LiveCodeBench and BigCodeBench with five base LLMs show that, even without candidate search (k=1), Flare outperforms the strongest baseline with an absolute improvement from 1.72% to 7.42%. Furthermore, searching over 10 candidates yields an average improvement of 8.50% compared with no candidate search. When evaluated in isolation, our lightweight diagnostic model achieves the best performance compared with recent fault localization methods, demonstrating that it can provide reliable fine-grained guidance for code refinement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13663v2">SAIL: Sound Abstract Interpreters with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-02
      | 💬 43 pages, 21 figures
    </div>
    <details class="paper-abstract">
      How to construct globally sound abstract interpreters to safely approximate program behaviors remains a bottleneck in abstract interpretation. In this paper, we show the potential of using state-of-the-art LLMs to automate this tedious process. Focusing on the neural network verification area, we synthesize non-trivial sound abstract transformers across diverse abstract domains using LLMs to search within infinite space from scratch. We formalize the synthesis task as a constrained optimization problem, for which we design a novel mathematically grounded cost function that measures the degree of unsoundness of each generated candidate transformer, while enforcing hard syntactic and semantic validity constraints. Building on this formulation, we introduce SAIL, a novel unified framework that combines model generation, syntactic and semantic validation, and cost-function-based refinement to synthesize globally sound abstract transformers. Evaluation results show that SAIL not only matches the performance of manually designed transformers, but also is able to synthesize sound and high-precision transformers that do not exist in the literature for complex non-linear operators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.02240v2">AgentRedBench: Dynamic Redteaming and Integration-Aware Defense for LLM Agents over SaaS Integrations</a></div>
    <div class="paper-meta">
      📅 2026-06-02
    </div>
    <details class="paper-abstract">
      Indirect prompt injection in tool-use agents is a concrete production threat: LLM agents read from integrations (third-party services such as Gmail, Salesforce, or Jira accessed through tool calls) whose response content the user neither writes nor controls. Existing benchmarks under-measure the threat: most cover only a handful of integrations with the same attack payload replayed across runs, and open-source guards are trained on chat-style data rather than tool-response content. We introduce AGENTREDBENCH, a dynamic LLM-driven redteaming benchmark of 215 subtle underspecified authorization (attacks at the boundary of what the user's request authorises) scenarios across 24 enterprise integrations in nine functional families and five attack types. Across an eight-model panel (Anthropic, OpenAI, Google), no-guard ASR (attack success rate) ranges from 32% (Claude Sonnet 4.6) to 81% (Gemini 3 Flash). To keep the scenario set out of training corpora and preserve headline ASR meaning over time, we release the codebase, integration schemas, and AGENTREDGUARD model openly; the canonical scenarios are evaluated through a maintainer-mediated channel with immutable versioning. We release AGENTREDGUARD alongside the benchmark: a guard trained on an integration-diverse corpus of adversarial tool-response content. AGENTREDGUARD cuts panel ASR from 69.9% to 2.4% at 0.37% false-positive rate, outperforming every open-source baseline with non-trivial detection (Llama Guard, PromptGuard 2, ProtectAI) on both axes. Cross-integration and cross-attack type holdouts both confirm the gain transfers beyond the training subset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03770v1">E2LLM: Towards Efficient LLM Serving in Heterogeneous Edge/Fog Environments</a></div>
    <div class="paper-meta">
      📅 2026-06-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become integral to modern applications, yet their deployment remains challenging. Beyond executing the models themselves, practical deployment must address cost efficiency, low latency, and optimal resource utilization. Conventional approaches typically assume that an entire model can be hosted on a single device, which does not hold in many real-world scenarios, particularly in Edge and Fog environments where device resources are constrained. In this paper, we introduce E2LLM, a framework designed to enable efficient LLM deployment in such resource limited settings. Rather than simply partitioning a single model across all available devices, E2LLM replicates the full model across multiple groups of devices (replicas) and applies model parallelism within each replica. Each replica is assigned a specialized role PREFILL or DECODER based on its efficiency in handling input and output tokens. This separation leverages the inherent differences between these two phases of LLM inference. To effectively organize devices, we utilize a Genetic Algorithm to form clusters that maximize system performance. Within each cluster, we apply Dynamic Programming to determine an optimal partitioning strategy that minimizes bottlenecks in model-parallel execution. Experimental results demonstrate that our approach adapts robustly to varying workloads, including scenarios with significant variation in input and output token lengths. Compared to the Splitwise baseline, E2LLM reduces average waiting time by over 50% under high-demand conditions
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03761v1">Framing Migration News with LLMs: Structured CoT as a Support for Human Interpretation</a></div>
    <div class="paper-meta">
      📅 2026-06-02
    </div>
    <details class="paper-abstract">
      Frame analysis of migration news is a socially consequential task: media scholars and researchers who study how migration is narrated need tools that are not only accurate, but transparent, auditable, and accessible within the resource constraints typical of academic research groups. Existing LLM-based approaches rely on proprietary APIs and large models that raise concerns about data privacy, reproducibility and equitable access among media researchers. This work studies how a locally deployable open-source LLM can support interpretable frame analysis as an assistive tool. We introduce a Structured Chain-of-Thought (SCoT) prompting approach using Llama3-8B, enabling step-by-step justifications grounded in predefined framing categories. This structured design allows users to audit model outputs and examine alternative interpretations in a task that is inherently subjective. We evaluate our approach on a dataset of migration-related news and show that SCoT improves classification performance over zero-shot and few-shot baselines while remaining feasible on a single GPU. Then, we conduct a human-centered evaluation in which annotators assess the coherence and influence of "the model's reasoning". Results indicate that SCoT explanations are generally perceived as logical (mean score 4.1/5, though with notable variation across texts) and can prompt reflection on initial interpretations, even when disagreement persists. Our findings highlight both the potential and risks of LLM-assisted frame analysis. While structured reasoning can increase the traceability of model outputs and support critical interpretation, it can also influence human judgment in subtle ways. By enabling local deployment and emphasizing human-in-the-loop interaction, this work contributes to discussions on responsible and accessible computational tools for the study of socially impactful media narratives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02380v5">LLMs, Reasoning and Plagiarism</a></div>
    <div class="paper-meta">
      📅 2026-06-02
      | 💬 The authors explicitly reserve all rights in this work. No permission is granted for the reproduction, storage, or use of this document for the purpose of training artificial intelligence systems or for text and data mining (TDM), including but not limited to the generation of embeddings, summaries, or synthetic derivatives. Claude and Gemini were used in writing this manuscript
    </div>
    <details class="paper-abstract">
      Recent reports claim that Large Language Models (LLMs) derive new science and exhibit human-level general intelligence. Such claims are entangled with two different narratives about what LLMs do: one in which they are an engine of synthesis that genuinely reasons to new knowledge, and one in which they retrieve and re-emit the work of others without attribution. In the scientific setting these are best understood as a contrast between \emph{reasoning} and \emph{plagiarism}. Finding where the truth lies between these two narratives is very challenging, as central components of the model -- the training data and the interaction transcript -- remain opaque. Thus claims of LLM reasoning do not satisfy Popper's refutability principle. We propose guidelines for transparency and reproducibility that will allow reasoning claims to be studied using the scientific method. The dominance of the reasoning narrative, we suggest, is in practice encouraging plagiarism in the scientific literature; we discuss what might be done about it.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03739v1">Entropy Gate: Entropy Quenching for Near-Lossless Token Compression in LLM Pipelines</a></div>
    <div class="paper-meta">
      📅 2026-06-02
    </div>
    <details class="paper-abstract">
      LLM pipelines waste substantial token budgets on low-information content: repeated context, verbose responses, and redundant boilerplate. We introduce Entropy Gate, a token compression framework applying entropy quenching $-$ a thermodynamic process that progressively freezes out low-energy tokens while preserving semantic fidelity. Each token receives a multi-factor information energy $E(t)$ combining statistical, structural, and positional components. An adaptive quenching schedule $T(τ) = T_0 / (1 + ατ)$ removes tokens whose Boltzmann survival probability $p_i = \exp(-E_i / kT)$ falls below threshold, with a fidelity gate halting compression when energy-weighted similarity drops below $θ$. We prove token selection by descending $E(t)$ maximizes expected semantic preservation, that quenching produces nested survival sets, and that achievable compression approaches the information-theoretic limit $\text{CR} \to 1 - I(P; T)/H(P)$. A Phase 1 heuristic achieves 40-60% compression across five prompt categories while maintaining $S_E > 0.80$, with energy-squared amplification $E \to E^2$ adding 10-25 percentage points. Context deduplication adds 50-70% savings on repeated blocks. Output-side quenching, motivated by findings that brevity improves accuracy, further reduces response overhead. Combined with external memory, reduction composes multiplicatively to 88-96% for agentic workloads. The framework is stateless, model-agnostic, and deploys as an OpenAI-compatible HTTP proxy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04067v1">Need to Know: Contextual-Integrity-Grounded Query Rewriting for Privacy-Conscious LLM Delegation</a></div>
    <div class="paper-meta">
      📅 2026-06-02
    </div>
    <details class="paper-abstract">
      As LLMs become increasingly woven into everyday workflows, user queries sent to cloud hosted LLMs routinely mix task-essential content with task non-essential sensitive disclosures, yet type based PII redaction is context agnostic and may raise two issues: over disclosing untyped sensitive context and over removing answer bearing spans. We recast privacy preserving query rewriting under Contextual Integrity: a span should be forwarded only if it is necessary for the task. We introduce DelegateCI-Bench, the first task based Contextual Integrity benchmark for privacy-conscious delegation, comprising 3,167 samples that combine high quality synthetic data spanning 11 tasks and 20 task types, WildChat based real user queries, and a medical challenge set with dense sensitive information. Building on this benchmark, we propose a CI-guided reinforcement learning framework that converts essential and non-essential sensitive spans into verifiable optimization signals, and train a query rewriter to preserve task critical information while suppressing unnecessary sensitive disclosure. Experiments show that our learned rewriter achieves the best privacy-utility tradeoff, achieving up to +10.1 average utility over on-device baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.31381v2">LLM Judges Inconsistently Disagree Across Safety Criteria and Harm Categories</a></div>
    <div class="paper-meta">
      📅 2026-06-02
      | 💬 8 pages plus appendices, under review
    </div>
    <details class="paper-abstract">
      We evaluate the consistency of automated judges in conducting a multi-dimensional safety evaluation in a reference-free setup. Our results indicate that Large Language Models are unreliable judges in identifying safety issues related to machine-generated advice in regulated domains such as finance, although they are more reliable at identifying more overt forms of unsafe/harmful content such as violence. The degree of inconsistency in a model's judgments can vary significantly by the chosen safety criteria and can be impacted by the language of the content and its linguistic style as well. Finally, there is high disagreement among different judges for the same output, across domains, safety criteria, and languages. These findings provide new insights on the practice of using LLMs as evaluators and offer several recommendations for practitioners on how to use automated judges in practical scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.31530v2">UNISON: A Unified Sound Generation and Editing Framework via Deep LLM Fusion</a></div>
    <div class="paper-meta">
      📅 2026-06-02
    </div>
    <details class="paper-abstract">
      We present UNISON, a latent diffusion framework that unifies speech generation, sound generation, and audio editing within a single model. A single model handles text-to-audio, text-to-speech, zero-shot speaker cloning, mixed speech-and-sound generation, scene-level audio editing, speech-in-scene editing, and timed temporal composition, all of which share a single set of weights. Our architecture features two core designs: (1) Layer-wise deep LLM fusion, which injects hidden states from uniformly sampled layers of a frozen MLLM into corresponding MM-DiT blocks via learned projections, providing depth-matched semantic conditioning that improves instruction following over single-layer baselines; and (2) a unified multi-task architecture where task identity is encoded solely by a channel-wise mask and source audio is provided through VAE-encoded channel concatenation. Training is stabilized by an online GPU-side multi-task data synthesis pipeline with task-homogeneous batching and a two-stage curriculum. With 621M--732M trainable parameters, UNISON achieves results competitive with or exceeding task-specialist models across evaluated domains, while being roughly $4\times$ smaller than comparable unified systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.02173v2">The Efficiency vs. Accuracy Trade-off: Optimizing RAG-Enhanced LLM Recommender Systems Using Multi-Head Early Exit</a></div>
    <div class="paper-meta">
      📅 2026-06-02
    </div>
    <details class="paper-abstract">
      The deployment of Large Language Models (LLMs) in recommender systems for predicting Click-Through Rates (CTR) necessitates a delicate balance between computational efficiency and predictive accuracy. This paper presents an optimization framework that combines Retrieval-Augmented Generation (RAG) with an innovative multi-head early exit architecture to concurrently enhance both aspects. By integrating Graph Convolutional Networks (GCNs) as efficient retrieval mechanisms, we are able to significantly reduce data retrieval times while maintaining high model performance. The early exit strategy employed allows for dynamic termination of model inference, utilizing real-time predictive confidence assessments across multiple heads. This not only quickens the responsiveness of LLMs but also upholds or improves their accuracy, making it ideal for real-time application scenarios. Our experiments demonstrate how this architecture effectively decreases computation time without sacrificing the accuracy needed for reliable recommendation delivery, establishing a new standard for efficient, real-time LLM deployment in commercial systems.
    </details>
</div>
