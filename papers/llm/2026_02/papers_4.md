# llm - 2026_02

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- Part 4

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04206v1">Enforcing Monotonic Progress in Legal Cross-Examination: Preventing Long-Horizon Stagnation in LLM-Based Inquiry</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 Submitted to ICAIL 2026. Under review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit impressive linguistic fluency but struggle to reliably complete long-horizon tasks under explicit procedural constraints. In legal cross-examination, purely proba-bilistic generation often maintains behavioral coherence while failing to ensure procedural advancement. We characterize this failure as procedural stagnation and propose Soft-FSM, a neuro-symbolic architecture that enforces monotonic progress over accumulated Key Information Units (KIUs) via an external deterministic state controller. Experiments on three real-world Taiwanese criminal homicide cases show that baseline methods collapse below 40% completeness, while Soft-FSM consistently achieves over 97% with near-zero redundancy. These results suggest that, in such domains, reliable task completion cannot be guaranteed by emergent LLM behavior alone, and can be reliably enforced through explicit and verifiable external state control.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04197v1">From Helpfulness to Toxic Proactivity: Diagnosing Behavioral Misalignment in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 9 pages (excluding appendices), 6 figures. Code is available at https://github.com/wxyoio-0715/Toxic-Proactivity
    </div>
    <details class="paper-abstract">
      The enhanced capabilities of LLM-based agents come with an emergency for model planning and tool-use abilities. Attributing to helpful-harmless trade-off from LLM alignment, agents typically also inherit the flaw of "over-refusal", which is a passive failure mode. However, the proactive planning and action capabilities of agents introduce another crucial danger on the other side of the trade-off. This phenomenon we term "Toxic Proactivity'': an active failure mode in which an agent, driven by the optimization for Machiavellian helpfulness, disregards ethical constraints to maximize utility. Unlike over-refusal, Toxic Proactivity manifests as the agent taking excessive or manipulative measures to ensure its "usefulness'' is maintained. Existing research pays little attention to identifying this behavior, as it often lacks the subtle context required for such strategies to unfold. To reveal this risk, we introduce a novel evaluation framework based on dilemma-driven interactions between dual models, enabling the simulation and analysis of agent behavior over multi-step behavioral trajectories. Through extensive experiments with mainstream LLMs, we demonstrate that Toxic Proactivity is a widespread behavioral phenomenon and reveal two major tendencies. We further present a systematic benchmark for evaluating Toxic Proactive behavior across contextual settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03516v2">Not All Negative Samples Are Equal: LLMs Learn Better from Plausible Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Learning from negative samples holds great promise for improving Large Language Model (LLM) reasoning capability, yet existing methods treat all incorrect responses as equally informative, overlooking the crucial role of sample quality. To address this, we propose Plausible Negative Samples (PNS), a method that synthesizes high-quality negative samples exhibiting expected format and structural coherence while ultimately yielding incorrect answers. PNS trains a dedicated model via reverse reinforcement learning (RL) guided by a composite reward combining format compliance, accuracy inversion, reward model assessment, and chain-of-thought evaluation, generating responses nearly indistinguishable from correct solutions. We further validate PNS as a plug-and-play data source for preference optimization across three backbone models on seven mathematical reasoning benchmarks. Results demonstrate that PNS consistently outperforms other negative sample synthesis methods, achieving an average improvement of 2.03% over RL-trained models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.27675v2">On the Difficulty of Selecting Few-Shot Examples for Effective LLM-based Vulnerability Detection</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 Workshop on LLM Assisted Security and Trust Exploration (LAST-X) 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive capabilities across a wide range of coding tasks, including summarization, translation, completion, and code generation. Despite these advances, detecting code vulnerabilities remains a challenging problem for LLMs. In-context learning (ICL) has emerged as an effective mechanism for improving model performance by providing a small number of labeled examples within the prompt. Prior work has shown, however, that the effectiveness of ICL depends critically on how these few-shot examples are selected. In this paper, we study two intuitive criteria for selecting few-shot examples for ICL in the context of code vulnerability detection. The first criterion leverages model behavior by prioritizing samples on which the LLM consistently makes mistakes, motivated by the intuition that such samples can expose and correct systematic model weaknesses. The second criterion selects examples based on semantic similarity to the query program, using k-nearest-neighbor retrieval to identify relevant contexts. We conduct extensive evaluations using open-source LLMs and datasets spanning multiple programming languages. Our results show that for Python and JavaScript, careful selection of few-shot examples can lead to measurable performance improvements in vulnerability detection. In contrast, for C and C++ programs, few-shot example selection has limited impact, suggesting that more powerful but also more expensive approaches, such as re-training or fine-tuning, may be required to substantially improve model performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.00364v2">"Someone Hid It": Query-Agnostic Black-Box Attacks on LLM-Based Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been serving as effective backbones for retrieval systems, including Retrieval-Augmentation-Generation (RAG), Dense Information Retriever (IR), and Agent Memory Retrieval. Recent studies have demonstrated that such LLM-based Retrieval (LLMR) is vulnerable to adversarial attacks, which manipulates documents by token-level injections and enables adversaries to either boost or diminish these documents in retrieval tasks. However, existing attack studies mainly (1) presume a known query is given to the attacker, and (2) highly rely on access to the victim model's parameters or interactions, which are hardly accessible in real-world scenarios, leading to limited validity. To further explore the secure risks of LLMR, we propose a practical black-box attack method that generates transferable injection tokens based on zero-shot surrogate LLMs without need of victim queries or victim models knowledge. The effectiveness of our attack raises such a robustness issue that similar effects may arise from benign or unintended document edits in the real world. To achieve our attack, we first establish a theoretical framework of LLMR and empirically verify it. Under the framework, we simulate the transferable attack as a min-max problem, and propose an adversarial learning mechanism that finds optimal adversarial tokens with learnable query samples. Our attack is validated to be effective on benchmark datasets across popular LLM retrievers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03048v2">CoBA-RL: Capability-Oriented Budget Allocation for Reinforcement Learning in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a key approach for enhancing LLM reasoning. However, standard frameworks like Group Relative Policy Optimization (GRPO) typically employ a uniform rollout budget, leading to resource inefficiency. Moreover, existing adaptive methods often rely on instance-level metrics, such as task pass rates, failing to capture the model's dynamic learning state. To address these limitations, we propose CoBA-RL, a reinforcement learning algorithm designed to adaptively allocate rollout budgets based on the model's evolving capability. Specifically, CoBA-RL utilizes a Capability-Oriented Value function to map tasks to their potential training gains and employs a heap-based greedy strategy to efficiently self-calibrate the distribution of computational resources to samples with high training value. Extensive experiments demonstrate that our approach effectively orchestrates the trade-off between exploration and exploitation, delivering consistent generalization improvements across multiple challenging benchmarks. These findings underscore that quantifying sample training value and optimizing budget allocation are pivotal for advancing LLM post-training efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21972v2">Learning Decentralized LLM Collaboration with Multi-Agent Actor Critic</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Recent work has explored optimizing LLM collaboration through Multi-Agent Reinforcement Learning (MARL). However, most MARL fine-tuning approaches rely on predefined execution protocols, which often require centralized execution. Decentralized LLM collaboration is more appealing in practice, as agents can run inference in parallel with flexible deployments. Also, current approaches use Monte Carlo methods for fine-tuning, which suffer from high variance and thus require more samples to train effectively. Actor-critic methods are prevalent in MARL for dealing with these issues, so we developed Multi-Agent Actor-Critic (MAAC) methods to optimize decentralized LLM collaboration. In this paper, we analyze when and why these MAAC methods are beneficial. We propose 2 MAAC approaches, \textbf{CoLLM-CC} with a \textbf{C}entralized \textbf{C}ritic and \textbf{CoLLM-DC} with \textbf{D}ecentralized \textbf{C}ritics. Our experiments across writing, coding, and game-playing domains show that Monte Carlo methods and CoLLM-DC can achieve performance comparable to CoLLM-CC in short-horizon and dense-reward settings. However, they both underperform CoLLM-CC on long-horizon or sparse-reward tasks, where Monte Carlo methods require substantially more samples and CoLLM-DC struggles to converge. Our code is available at https://github.com/OpenMLRL/CoMLRL/releases/tag/v1.3.2.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04913v1">A$^2$-LLM: An End-to-end Conversational Audio Avatar Large Language Model</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 13 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Developing expressive and responsive conversational digital humans is a cornerstone of next-generation human-computer interaction. While large language models (LLMs) have significantly enhanced dialogue capabilities, most current systems still rely on cascaded architectures that connect independent modules. These pipelines are often plagued by accumulated errors, high latency, and poor real-time performance. Lacking access to the underlying conversational context, these pipelines inherently prioritize rigid lip-sync over emotional depth. To address these challenges, we propose A$^2$-LLM, an end-to-end conversational audio avatar large language model that jointly reasons about language, audio prosody, and 3D facial motion within a unified framework. To facilitate training, we introduce FLAME-QA, a high-quality multimodal dataset designed to align semantic intent with expressive facial dynamics within a QA format. By leveraging deep semantic understanding, A$^2$-LLM generates emotionally rich facial movements beyond simple lip-synchronization. Experimental results demonstrate that our system achieves superior emotional expressiveness while maintaining real-time efficiency (500 ms latency, 0.7 RTF).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04099v1">Rethinking Perplexity: Revealing the Impact of Input Length on Perplexity Evaluation in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Perplexity is a widely adopted metric for assessing the predictive quality of large language models (LLMs) and often serves as a reference metric for downstream evaluations. However, recent evidence shows that perplexity can be unreliable, especially when irrelevant long inputs are used, raising concerns for both benchmarking and system deployment. While prior efforts have employed selective input filtering and curated datasets, the impact of input length on perplexity has not been systematically studied from a systems perspective and input length has rarely been treated as a first-class system variable affecting both fairness and efficiency. In this work, we close this gap by introducing LengthBenchmark, a system-conscious evaluation framework that explicitly integrates input length, evaluation protocol design, and system-level costs, evaluating representative LLMs under two scoring protocols (direct accumulation and fixed window sliding) across varying context lengths. Unlike prior work that focuses solely on accuracy-oriented metrics, LengthBenchmark additionally measures latency, memory footprint, and evaluation cost, thereby linking predictive metrics to deployment realities. We further incorporate quantized variants not as a main contribution, but as robustness checks, showing that length-induced biases persist across both full-precision and compressed models. This design disentangles the effects of evaluation logic, quantization, and input length, and demonstrates that length bias is a general phenomenon that undermines fair cross-model comparison. Our analysis yields two key observations: (i) sliding window evaluation consistently inflates performance on short inputs, and (ii) both full-precision and quantized models appear to realise gains as the evaluated segment length grows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05128v1">Reporting and Reviewing LLM-Integrated Systems in HCI: Challenges and Considerations</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 18 pages, 1 figure, 2 tables. For proposed reporting guidelines, see https://ianarawjo.github.io/Guidelines-for-Reporting-LLM-Integrated-Systems-in-HCI/
    </div>
    <details class="paper-abstract">
      What should HCI scholars consider when reporting and reviewing papers that involve LLM-integrated systems? We interview 18 authors of LLM-integrated system papers on their authoring and reviewing experiences. We find that norms of trust-building between authors and reviewers appear to be eroded by the uncertainty of LLM behavior and hyperbolic rhetoric surrounding AI. Authors perceive that reviewers apply uniquely skeptical and inconsistent standards towards papers that report LLM-integrated systems, and mitigate mistrust by adding technical evaluations, justifying usage, and de-emphasizing LLM presence. Authors' views challenge blanket directives to report all prompts and use open models, arguing that prompt reporting is context-dependent and justifying proprietary model usage despite ethical concerns. Finally, some tensions in peer review appear to stem from clashes between the norms and values of HCI and ML/NLP communities, particularly around what constitutes a contribution and an appropriate level of technical rigor. Based on our findings and additional feedback from six expert HCI researchers, we present a set of guidelines and considerations for authors, reviewers, and HCI communities around reporting and reviewing papers that involve LLM-integrated systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05125v1">Rethinking Rubric Generation for Improving LLM Judge and Reward Modeling for Open-ended Tasks</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Recently, rubrics have been used to guide LLM judges in capturing subjective, nuanced, multi-dimensional human preferences, and have been extended from evaluation to reward signals for reinforcement fine-tuning (RFT). However, rubric generation remains hard to control: rubrics often lack coverage, conflate dimensions, misalign preference direction, and contain redundant or highly correlated criteria, degrading judge accuracy and producing suboptimal rewards during RFT. We propose RRD, a principled framework for rubric refinement built on a recursive decompose-filter cycle. RRD decomposes coarse rubrics into fine-grained, discriminative criteria, expanding coverage while sharpening separation between responses. A complementary filtering mechanism removes misaligned and redundant rubrics, and a correlation-aware weighting scheme prevents over-representing highly correlated criteria, yielding rubric sets that are informative, comprehensive, and non-redundant. Empirically, RRD delivers large, consistent gains across both evaluation and training: it improves preference-judgment accuracy on JudgeBench and PPE for both GPT-4o and Llama3.1-405B judges, achieving top performance in all settings with up to +17.7 points on JudgeBench. When used as the reward source for RFT on WildChat, it yields substantially stronger and more stable learning signals, boosting reward by up to 160% (Qwen3-4B) and 60% (Llama3.1-8B) versus 10-20% for prior rubric baselines, with gains that transfer to HealthBench-Hard and BiGGen Bench. Overall, RRD establishes recursive rubric refinement as a scalable and interpretable foundation for LLM judging and reward modeling in open-ended domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05110v1">Understanding LLM Evaluator Behavior: A Structured Multi-Evaluator Framework for Merchant Risk Assessment</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used as evaluators of reasoning quality, yet their reliability and bias in payments-risk settings remain poorly understood. We introduce a structured multi-evaluator framework for assessing LLM reasoning in Merchant Category Code (MCC)-based merchant risk assessment, combining a five-criterion rubric with Monte-Carlo scoring to evaluate rationale quality and evaluator stability. Five frontier LLMs generate and cross-evaluate MCC risk rationales under attributed and anonymized conditions. To establish a judge-independent reference, we introduce a consensus-deviation metric that eliminates circularity by comparing each judge's score to the mean of all other judges, yielding a theoretically grounded measure of self-evaluation and cross-model deviation. Results reveal substantial heterogeneity: GPT-5.1 and Claude 4.5 Sonnet show negative self-evaluation bias (-0.33, -0.31), while Gemini-2.5 Pro and Grok 4 display positive bias (+0.77, +0.71), with bias attenuating by 25.8 percent under anonymization. Evaluation by 26 payment-industry experts shows LLM judges assign scores averaging +0.46 points above human consensus, and that the negative bias of GPT-5.1 and Claude 4.5 Sonnet reflects closer alignment with human judgment. Ground-truth validation using payment-network data shows four models exhibit statistically significant alignment (Spearman rho = 0.56 to 0.77), confirming that the framework captures genuine quality. Overall, the framework provides a replicable basis for evaluating LLM-as-a-judge systems in payment-risk workflows and highlights the need for bias-aware protocols in operational financial settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21410v2">Statsformer: Validated Ensemble Learning with LLM-Derived Semantic Priors</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      We introduce Statsformer, a principled framework for integrating large language model (LLM)-derived knowledge into supervised statistical learning. Existing approaches are limited in adaptability and scope: they either inject LLM guidance as an unvalidated heuristic, which is sensitive to LLM hallucination, or embed semantic information within a single fixed learner. Statsformer overcomes both limitations through a guardrailed ensemble architecture. We embed LLM-derived feature priors within an ensemble of linear and nonlinear learners, adaptively calibrating their influence via cross-validation. This design yields a flexible system with an oracle-style guarantee that it performs no worse than any convex combination of its in-library base learners, up to statistical error. Empirically, informative priors yield consistent performance improvements, while uninformative or misspecified LLM guidance is automatically downweighted, mitigating the impact of hallucinations across a diverse range of prediction tasks.An open-source implementation of Statsformer is available at https://github.com/pilancilab/statsformer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.26641v3">All You Need for Object Detection: From Pixels, Points, and Prompts to Next-Gen Fusion and Multimodal LLMs/VLMs in Autonomous Vehicles</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Autonomous Vehicles (AVs) are transforming the future of transportation through advances in intelligent perception, decision-making, and control systems. However, their success is tied to one core capability, reliable object detection in complex and multimodal environments. While recent breakthroughs in Computer Vision (CV) and Artificial Intelligence (AI) have driven remarkable progress, the field still faces a critical challenge as knowledge remains fragmented across multimodal perception, contextual reasoning, and cooperative intelligence. This survey bridges that gap by delivering a forward-looking analysis of object detection in AVs, emphasizing emerging paradigms such as Vision-Language Models (VLMs), Large Language Models (LLMs), and Generative AI rather than re-examining outdated techniques. We begin by systematically reviewing the fundamental spectrum of AV sensors (camera, ultrasonic, LiDAR, and Radar) and their fusion strategies, highlighting not only their capabilities and limitations in dynamic driving environments but also their potential to integrate with recent advances in LLM/VLM-driven perception frameworks. Next, we introduce a structured categorization of AV datasets that moves beyond simple collections, positioning ego-vehicle, infrastructure-based, and cooperative datasets (e.g., V2V, V2I, V2X, I2I), followed by a cross-analysis of data structures and characteristics. Ultimately, we analyze cutting-edge detection methodologies, ranging from 2D and 3D pipelines to hybrid sensor fusion, with particular attention to emerging transformer-driven approaches powered by Vision Transformers (ViTs), Large and Small Language Models (SLMs), and VLMs. By synthesizing these perspectives, our survey delivers a clear roadmap of current capabilities, open challenges, and future opportunities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.17406v2">Robust Answers, Fragile Logic: Probing the Decoupling Hypothesis in LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      While Chain-of-Thought (CoT) prompting has become a cornerstone for complex reasoning in Large Language Models (LLMs), the faithfulness of the generated reasoning remains an open question. We investigate the Decoupling Hypothesis: that correct answers often mask fragile, post-hoc rationalizations that are not causally tied to the model's prediction. To systematically verify this, we introduce MATCHA, a novel Answer-Conditioned Probing framework. Unlike standard evaluations that focus on final output accuracy, MATCHA isolates the reasoning phase by conditioning generation on the model's predicted answer, allowing us to stress-test the stability of the rationale itself. Our experiments reveal a critical vulnerability: under imperceptible input perturbations, LLMs frequently maintain the correct answer while generating inconsistent or nonsensical reasoning - effectively being ``Right for the Wrong Reasons''. Using LLM judges to quantify this robustness gap, we find that multi-step and commonsense tasks are significantly more susceptible to this decoupling than logical tasks. Furthermore, we demonstrate that adversarial examples generated by MATCHA transfer non-trivially to black-box models. Our findings expose the illusion of CoT robustness and underscore the need for future architectures that enforce genuine answer-reasoning consistency rather than mere surface-level accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15397v2">Beyond Prompting: Efficient and Robust Contextual Biasing for Speech LLMs via Logit-Space Integration (LOGIC)</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 This paper is withdrawn temporarily to ensure full compliance with internal institutional publication approval processes
    </div>
    <details class="paper-abstract">
      The rapid emergence of new entities -- driven by cultural shifts, evolving trends, and personalized user data -- poses a significant challenge for existing Speech Large Language Models (Speech LLMs). While these models excel at general conversational tasks, their static training knowledge limits their ability to recognize domain-specific terms such as contact names, playlists, or technical jargon. Existing solutions primarily rely on prompting, which suffers from poor scalability: as the entity list grows, prompting encounters context window limitations, increased inference latency, and the "lost-in-the-middle" phenomenon. An alternative approach, Generative Error Correction (GEC), attempts to rewrite transcripts via post-processing but frequently suffers from "over-correction", introducing hallucinations of entities that were never spoken. In this work, we introduce LOGIC (Logit-Space Integration for Contextual Biasing), an efficient and robust framework that operates directly in the decoding layer. Unlike prompting, LOGIC decouples context injection from input processing, ensuring constant-time complexity relative to prompt length. Extensive experiments using the Phi-4-MM model across 11 multilingual locales demonstrate that LOGIC achieves an average 9% relative reduction in Entity WER with a negligible 0.30% increase in False Alarm Rate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04998v1">Learning Rate Matters: Vanilla LoRA May Suffice for LLM Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Low-Rank Adaptation (LoRA) is the prevailing approach for efficient large language model (LLM) fine-tuning. Building on this paradigm, recent studies have proposed alternative initialization strategies and architectural modifications, reporting substantial improvements over vanilla LoRA. However, these gains are often demonstrated under fixed or narrowly tuned hyperparameter settings, despite the known sensitivity of neural networks to training configurations. In this work, we systematically re-evaluate four representative LoRA variants alongside vanilla LoRA through extensive hyperparameter searches. Across mathematical and code generation tasks on diverse model scales, we find that different LoRA methods favor distinct learning rate ranges. Crucially, once learning rates are properly tuned, all methods achieve similar peak performance (within 1-2%), with only subtle rank-dependent behaviors. These results suggest that vanilla LoRA remains a competitive baseline and that improvements reported under single training configuration may not reflect consistent methodological advantages. Finally, a second-order analysis attributes the differing optimal learning rate ranges to variations in the largest Hessian eigenvalue, aligning with classical learning theories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04972v1">Learning Context Matters: Measuring and Diagnosing Personalization Gaps in LLM-Based Instructional Design</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      The adoption of generative AI in education has accelerated dramatically in recent years, with Large Language Models (LLMs) increasingly integrated into learning environments in the hope of providing personalized support that enhances learner engagement and knowledge retention. However, truly personalized support requires access to meaningful Learning Context (LC) regarding who the learner is, what they are trying to understand, and how they are engaging with the material. In this paper, we present a framework for measuring and diagnosing how the LC influences instructional strategy selection in LLM-based tutoring systems. Using psychometrically grounded synthetic learning contexts and a pedagogically grounded decision space, we compare LLM instructional decisions in context-blind and context-aware conditions and quantify their alignment with the pedagogical judgments of subject matter experts. Our results show that, while providing the LC induces systematic, measurable changes in instructional decisions that move LLM policies closer to the subject matter expert policy, substantial misalignment remains. To diagnose this misalignment, we introduce a relevance-impact analysis that reveals which learner characteristics are attended to, ignored, or spuriously influential in LLM instructional decision-making. This analysis, conducted in collaboration with subject matter experts, demonstrates that LC materially shapes LLM instructional planning but does not reliably induce pedagogically appropriate personalization. Our results enable principled evaluation of context-aware LLM systems and provide a foundation for improving personalization through learner characteristic prioritization, pedagogical model tuning, and LC engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04879v1">Rethinking the Trust Region in LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become a cornerstone for fine-tuning Large Language Models (LLMs), with Proximal Policy Optimization (PPO) serving as the de facto standard algorithm. Despite its ubiquity, we argue that the core ratio clipping mechanism in PPO is structurally ill-suited for the large vocabularies inherent to LLMs. PPO constrains policy updates based on the probability ratio of sampled tokens, which serves as a noisy single-sample Monte Carlo estimate of the true policy divergence. This creates a sub-optimal learning dynamic: updates to low-probability tokens are aggressively over-penalized, while potentially catastrophic shifts in high-probability tokens are under-constrained, leading to training inefficiency and instability. To address this, we propose Divergence Proximal Policy Optimization (DPPO), which substitutes heuristic clipping with a more principled constraint based on a direct estimate of policy divergence (e.g., Total Variation or KL). To avoid huge memory footprint, we introduce the efficient Binary and Top-K approximations to capture the essential divergence with negligible overhead. Extensive empirical evaluations demonstrate that DPPO achieves superior training stability and efficiency compared to existing methods, offering a more robust foundation for RL-based LLM fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.02542v4">OverThink: Slowdown Attacks on Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Most flagship language models generate explicit reasoning chains, enabling inference-time scaling. However, producing these reasoning chains increases token usage (i.e., reasoning tokens), which in turn increases latency and costs. Our OverThink attack increases overhead for applications that rely on reasoning language models (RLMs) and external context by forcing them to spend substantially more reasoning tokens while still producing contextually correct answers. An adversary mounts an attack by injecting decoy reasoning problems into public content that is consumed by RLM at inference time. Because our decoys (e.g., Markov decision processes, Sudokus, etc.) are benign, they evade safety filters. We evaluate OverThink on both closed-source and open-source reasoning models across the FreshQA, SQuAD, and MuSR datasets. We also explore the attack in multi-modal settings by creating images that cause excessive reasoning. We show that the resulting slowdown transfers across models. Finally, we explore both LLM-based and systems-level defenses, and discuss the societal, financial, and energy implications of the OverThink attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04813v1">Agentic AI in Healthcare & Medicine: A Seven-Dimensional Taxonomy for Empirical Evaluation of LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based agents that plan, use tools and act has begun to shape healthcare and medicine. Reported studies demonstrate competence on various tasks ranging from EHR analysis and differential diagnosis to treatment planning and research workflows. Yet the literature largely consists of overviews which are either broad surveys or narrow dives into a single capability (e.g., memory, planning, reasoning), leaving healthcare work without a common frame. We address this by reviewing 49 studies using a seven-dimensional taxonomy: Cognitive Capabilities, Knowledge Management, Interaction Patterns, Adaptation & Learning, Safety & Ethics, Framework Typology and Core Tasks & Subtasks with 29 operational sub-dimensions. Using explicit inclusion and exclusion criteria and a labeling rubric (Fully Implemented, Partially Implemented, Not Implemented), we map each study to the taxonomy and report quantitative summaries of capability prevalence and co-occurrence patterns. Our empirical analysis surfaces clear asymmetries. For instance, the External Knowledge Integration sub-dimension under Knowledge Management is commonly realized (~76% Fully Implemented) whereas Event-Triggered Activation sub-dimenison under Interaction Patterns is largely absent (~92% Not Implemented) and Drift Detection & Mitigation sub-dimension under Adaptation & Learning is rare (~98% Not Implemented). Architecturally, Multi-Agent Design sub-dimension under Framework Typology is the dominant pattern (~82% Fully Implemented) while orchestration layers remain mostly partial. Across Core Tasks & Subtasks, information centric capabilities lead e.g., Medical Question Answering & Decision Support and Benchmarking & Simulation, while action and discovery oriented areas such as Treatment Planning & Prescription still show substantial gaps (~59% Not Implemented).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10823v2">Mugi: Value Level Parallelism For Efficient LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 2026 International Conference on Architectural Support for Programming Languages and Operating Systems
    </div>
    <details class="paper-abstract">
      Value level parallelism (VLP) has been proposed to improve the efficiency of large-batch, low-precision general matrix multiply (GEMM) between symmetric activations and weights. In transformer based large language models (LLMs), there exist more sophisticated operations beyond activation-weight GEMM. In this paper, we explore how VLP benefits LLMs. First, we generalize VLP for nonlinear approximations, outperforming existing nonlinear approximations in end-to-end LLM accuracy, performance, and efficiency. Our VLP approximation follows a value-centric approach, where important values are assigned with greater accuracy. Second, we optimize VLP for small-batch GEMMs with asymmetric inputs efficiently, which leverages timely LLM optimizations, including weight-only quantization, key-value (KV) cache quantization, and group query attention. Finally, we design a new VLP architecture, Mugi, to encapsulate the innovations above and support full LLM workloads, while providing better performance, efficiency and sustainability. Our experimental results show that Mugi can offer significant improvements on throughput and energy efficiency, up to $45\times$ and $668\times$ for nonlinear softmax operations, and $2.07\times$ and $3.11\times$ for LLMs, and also decrease operational carbon for LLM operation by $1.45\times$ and embodied carbon by $1.48\times$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04785v1">Team, Then Trim: An Assembly-Line LLM Framework for High-Quality Tabular Data Generation</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      While tabular data is fundamental to many real-world machine learning (ML) applications, acquiring high-quality tabular data is usually labor-intensive and expensive. Limited by the scarcity of observations, tabular datasets often exhibit critical deficiencies, such as class imbalance, selection bias, and low fidelity. To address these challenges, building on recent advances in Large Language Models (LLMs), this paper introduces Team-then-Trim (T$^2$), a framework that synthesizes high-quality tabular data through a collaborative team of LLMs, followed by a rigorous three-stage plug-in data quality control (QC) pipeline. In T$^2$, tabular data generation is conceptualized as a manufacturing process: specialized LLMs, guided by domain knowledge, are tasked with generating different data components sequentially, and the resulting products, i.e., the synthetic data, are systematically evaluated across multiple dimensions of QC. Empirical results on both simulated and real-world datasets demonstrate that T$^2$ outperforms state-of-the-art methods in producing high-quality tabular data, highlighting its potential to support downstream models when direct data collection is practically infeasible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.02648v4">Steering the Herd: A Framework for LLM-based Control of Social Learning</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Algorithms increasingly serve as information mediators--from social media feeds and targeted advertising to the increasing ubiquity of LLMs. This engenders a joint process where agents combine private, algorithmically-mediated signals with learning from peers to arrive at decisions. To study such settings, we introduce a model of controlled sequential social learning in which an information-mediating planner (e.g. an LLM) controls the information structure of agents while they also learn from the decisions of earlier agents. The planner may seek to improve social welfare (altruistic planner) or to induce a specific action the planner prefers (biased planner). Our framework presents a new optimization problem for social learning that combines dynamic programming with decentralized action choices and Bayesian belief updates. We prove the convexity of the value function and characterize the optimal policies of altruistic and biased planners, which attain desired tradeoffs between the costs they incur and the payoffs they earn from induced agent choices. Notably, in some regimes the biased planner intentionally obfuscates the agents' signals. Even under stringent transparency constraints--information parity with individuals, no lying or cherry-picking, and full observability--we show that information mediation can substantially shift social welfare in either direction. We complement our theory with simulations in which LLMs act as both planner and agents. Notably, the LLM planner in our simulations exhibits emergent strategic behavior in steering public opinion that broadly mirrors the trends predicted, though key deviations suggest the influence of non-Bayesian reasoning consistent with the cognitive patterns of both humans and LLMs trained on human-like data. Together, we establish our framework as a tractable basis for studying the impact and regulation of LLM information mediators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04755v1">When Silence Is Golden: Can LLMs Learn to Abstain in Temporal QA and Beyond?</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 Accepted to ICLR2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) rarely admit uncertainty, often producing fluent but misleading answers, rather than abstaining (i.e., refusing to answer). This weakness is even evident in temporal question answering, where models frequently ignore time-sensitive evidence and conflate facts across different time-periods. In this paper, we present the first empirical study of training LLMs with an abstention ability while reasoning about temporal QA. Existing approaches such as calibration might be unreliable in capturing uncertainty in complex reasoning. We instead frame abstention as a teachable skill and introduce a pipeline that couples Chain-of-Thought (CoT) supervision with Reinforcement Learning (RL) guided by abstention-aware rewards. Our goal is to systematically analyze how different information types and training techniques affect temporal reasoning with abstention behavior in LLMs. Through extensive experiments studying various methods, we find that RL yields strong empirical gains on reasoning: a model initialized by Qwen2.5-1.5B-Instruct surpasses GPT-4o by $3.46\%$ and $5.80\%$ in Exact Match on TimeQA-Easy and Hard, respectively. Moreover, it improves the True Positive rate on unanswerable questions by $20\%$ over a pure supervised fine-tuned (SFT) variant. Beyond performance, our analysis shows that SFT induces overconfidence and harms reliability, while RL improves prediction accuracy but exhibits similar risks. Finally, by comparing implicit reasoning cues (e.g., original context, temporal sub-context, knowledge graphs) with explicit CoT supervision, we find that implicit information provides limited benefit for reasoning with abstention. Our study provides new insights into how abstention and reasoning can be jointly optimized, providing a foundation for building more reliable LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04750v1">Exploiting contextual information to improve stance detection in informal political discourse with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 14 pages, 7 figures
    </div>
    <details class="paper-abstract">
      This study investigates the use of Large Language Models (LLMs) for political stance detection in informal online discourse, where language is often sarcastic, ambiguous, and context-dependent. We explore whether providing contextual information, specifically user profile summaries derived from historical posts, can improve classification accuracy. Using a real-world political forum dataset, we generate structured profiles that summarize users' ideological leaning, recurring topics, and linguistic patterns. We evaluate seven state-of-the-art LLMs across baseline and context-enriched setups through a comprehensive cross-model evaluation. Our findings show that contextual prompts significantly boost accuracy, with improvements ranging from +17.5\% to +38.5\%, achieving up to 74\% accuracy that surpasses previous approaches. We also analyze how profile size and post selection strategies affect performance, showing that strategically chosen political content yields better results than larger, randomly selected contexts. These findings underscore the value of incorporating user-level context to enhance LLM performance in nuanced political classification tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04731v1">Less Finetuning, Better Retrieval: Rethinking LLM Adaptation for Biomedical Retrievers via Synthetic Data and Model Merging</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) has become the backbone of grounding Large Language Models (LLMs), improving knowledge updates and reducing hallucinations. Recently, LLM-based retriever models have shown state-of-the-art performance for RAG applications. However, several technical aspects remain underexplored on how to adapt general-purpose LLMs into effective domain-specific retrievers, especially in specialized domains such as biomedicine. We present Synthesize-Train-Merge (STM), a modular framework that enhances decoder-only LLMs with synthetic hard negatives, retrieval prompt optimization, and model merging. Experiments on a subset of 12 medical and general tasks from the MTEB benchmark show STM boosts task-specific experts by up to 23.5\% (average 7.5\%) and produces merged models that outperform both single experts and strong baselines without extensive pretraining. Our results demonstrate a scalable, efficient path for turning general LLMs into high-performing, domain-specialized retrievers, preserving general-domain capabilities while excelling on specialized tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04729v1">"Be My Cheese?": Cultural Nuance Benchmarking for Machine Translation in Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 under peer-review
    </div>
    <details class="paper-abstract">
      We present a large-scale human evaluation benchmark for assessing cultural localisation in machine translation produced by state-of-the-art multilingual large language models (LLMs). Existing MT benchmarks emphasise token-level and grammatical accuracy, but of ten overlook pragmatic and culturally grounded competencies required for real-world localisation. Building on a pilot study of 87 translations across 20 languages, we evaluate 7 multilingual LLMs across 15 target languages with 5 native-speaker raters per language. Raters scored both full-text translations and segment-level instances of culturally nuanced language (idioms, puns, holidays, and culturally embedded concepts) on an ordinal 0-3 quality scale; segment ratings additionally included an NA option for untranslated segments. Across full-text evaluations, mean overall quality is modest (1.68/3): GPT-5 (2.10/3), Claude Sonnet 3.7 (1.97/3), and Mistral Medium 3.1 (1.84/3) form the strongest tier with fewer catastrophic failures. Segment-level results show sharp category effects: holidays (2.20/3) and cultural concepts (2.19/3) translate substantially better than idioms (1.65/3) and puns (1.45/3), and idioms are most likely to be left untranslated. These findings demonstrate a persistent gap between grammatical adequacy and cultural resonance. To our knowledge, this is the first multilingual, human-annotated benchmark focused explicitly on cultural nuance in translation and localisation, highlighting the need for culturally informed training data, improved cross-lingual pragmatics, and evaluation paradigms that better reflect real-world communicative competence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06655v2">GSAE: Graph-Regularized Sparse Autoencoders for Robust LLM Safety Steering</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) face critical safety challenges, as they can be manipulated to generate harmful content through adversarial prompts and jailbreak attacks. Many defenses are typically either black-box guardrails that filter outputs, or internals-based methods that steer hidden activations by operationalizing safety as a single latent feature or dimension. While effective for simple concepts, this assumption is limiting, as recent evidence shows that abstract concepts such as refusal and temporality are distributed across multiple features rather than isolated in one. To address this limitation, we introduce Graph-Regularized Sparse Autoencoders (GSAEs), which extends SAEs with a Laplacian smoothness penalty on the neuron co-activation graph. Unlike standard SAEs that assign each concept to a single latent feature, GSAEs recover smooth, distributed safety representations as coherent patterns spanning multiple features. We empirically demonstrate that GSAE enables effective runtime safety steering, assembling features into a weighted set of safety-relevant directions and controlling them with a two-stage gating mechanism that activates interventions only when harmful prompts or continuations are detected during generation. This approach enforces refusals adaptively while preserving utility on benign queries. Across safety and QA benchmarks, GSAE steering achieves an average 82% selective refusal rate, substantially outperforming standard SAE steering (42%), while maintaining strong task accuracy (70% on TriviaQA, 65% on TruthfulQA, 74% on GSM8K). Robustness experiments further show generalization across LLaMA-3, Mistral, Qwen, and Phi families and resilience against jailbreak attacks (GCG, AutoDAN), consistently maintaining >= 90% refusal of harmful content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04693v1">LinGO: A Linguistic Graph Optimization Framework with LLMs for Interpreting Intents of Online Uncivil Discourse</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Detecting uncivil language is crucial for maintaining safe, inclusive, and democratic online spaces. Yet existing classifiers often misinterpret posts containing uncivil cues but expressing civil intents, leading to inflated estimates of harmful incivility online. We introduce LinGO, a linguistic graph optimization framework for large language models (LLMs) that leverages linguistic structures and optimization techniques to classify multi-class intents of incivility that use various direct and indirect expressions. LinGO decomposes language into multi-step linguistic components, identifies targeted steps that cause the most errors, and iteratively optimizes prompt and/or example components for targeted steps. We evaluate it using a dataset collected during the 2022 Brazilian presidential election, encompassing four forms of political incivility: Impoliteness (IMP), Hate Speech and Stereotyping (HSST), Physical Harm and Violent Political Rhetoric (PHAVPR), and Threats to Democratic Institutions and Values (THREAT). Each instance is annotated with six types of civil/uncivil intent. We benchmark LinGO using three cost-efficient LLMs: GPT-5-mini, Gemini 2.5 Flash-Lite, and Claude 3 Haiku, and four optimization techniques: TextGrad, AdalFlow, DSPy, and Retrieval-Augmented Generation (RAG). The results show that, across all models, LinGO consistently improves accuracy and weighted F1 compared with zero-shot, chain-of-thought, direct optimization, and fine-tuning baselines. RAG is the strongest optimization technique and, when paired with Gemini model, achieves the best overall performance. These findings demonstrate that incorporating multi-step linguistic components into LLM instructions and optimize targeted components can help the models explain complex semantic meanings, which can be extended to other complex semantic explanation tasks in the future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04674v1">Overstating Attitudes, Ignoring Networks: LLM Biases in Simulating Misinformation Susceptibility</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as proxies for human judgment in computational social science, yet their ability to reproduce patterns of susceptibility to misinformation remains unclear. We test whether LLM-simulated survey respondents, prompted with participant profiles drawn from social survey data measuring network, demographic, attitudinal and behavioral features, can reproduce human patterns of misinformation belief and sharing. Using three online surveys as baselines, we evaluate whether LLM outputs match observed response distributions and recover feature-outcome associations present in the original survey data. LLM-generated responses capture broad distributional tendencies and show modest correlation with human responses, but consistently overstate the association between belief and sharing. Linear models fit to simulated responses exhibit substantially higher explained variance and place disproportionate weight on attitudinal and behavioral features, while largely ignoring personal network characteristics, relative to models fit to human responses. Analyses of model-generated reasoning and LLM training data suggest that these distortions reflect systematic biases in how misinformation-related concepts are represented. Our findings suggest that LLM-based survey simulations are better suited for diagnosing systematic divergences from human judgment than for substituting it.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10931v2">Asynchronous Reasoning: Training-Free Interactive Thinking LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 Preprint, work in progress
    </div>
    <details class="paper-abstract">
      Many state-of-the-art LLMs are trained to think before giving their answer. Reasoning can greatly improve language model capabilities, but it also makes them less interactive: given a new input, a model must stop thinking before it can respond. Real-world use cases such as voice-based or embodied assistants require an LLM agent to respond and adapt to additional information in real time, which is incompatible with sequential interactions. In contrast, humans can listen, think, and act asynchronously: we begin thinking about the problem while reading it and continue thinking while formulating the answer. In this work, we augment LLMs capable of reasoning to operate in a similar way without additional training. Our method uses the properties of positional embeddings to enable LLMs built for sequential generation to simultaneously think, listen, and write outputs. We evaluate our approach on math, commonsense, and safety reasoning: it allows models to generate accurate thinking-augmented answers while reducing time to first non-thinking token from minutes to ${\le}$ 5s and the overall real-time delays by up to $12{\times}$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04630v1">Mapping the Web of Science, a large-scale graph and text-based dataset with LLM embeddings</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Large text data sets, such as publications, websites, and other text-based media, inherit two distinct types of features: (1) the text itself, its information conveyed through semantics, and (2) its relationship to other texts through links, references, or shared attributes. While the latter can be described as a graph structure and can be handled by a range of established algorithms for classification and prediction, the former has recently gained new potential through the use of LLM embedding models. Demonstrating these possibilities and their practicability, we investigate the Web of Science dataset, containing ~56 million scientific publications through the lens of our proposed embedding method, revealing a self-structured landscape of texts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04620v1">QUATRO: Query-Adaptive Trust Region Policy Optimization for LLM Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      GRPO-style reinforcement learning (RL)-based LLM fine-tuning algorithms have recently gained popularity. Relying on heuristic trust-region approximations, however, they can lead to brittle optimization behavior, as global importance-ratio clipping and group-wise normalization fail to regulate samples whose importance ratios fall outside the clipping range. We propose Query-Adaptive Trust-Region policy Optimization (QUATRO), which directly enforces trust-region constraints through a principled optimization. This yields a clear and interpretable objective that enables explicit control over policy updates and stable, entropy-controlled optimization, with a stabilizer terms arising intrinsically from the exact trust-region formulation. Empirically verified on diverse mathematical reasoning benchmarks, QUATRO shows stable training under increased policy staleness and aggressive learning rates, maintaining well-controlled entropy throughout training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.02389v3">From Trace to Line: LLM Agent for Real-World OSS Vulnerability Localization</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Large language models show promise for vulnerability discovery, yet prevailing methods inspect code in isolation, struggle with long contexts, and focus on coarse function- or file-level detections that offer limited guidance to engineers who need precise line-level localization for targeted patches. We introduce T2L, an executable framework for project-level, line-level vulnerability localization that progressively narrows scope from repository modules to exact vulnerable lines via AST-based chunking and evidence-guided refinement. We provide a baseline agent with an Agentic Trace Analyzer (ATA) that fuses runtime evidence such as crash points and stack traces to translate failure symptoms into actionable diagnoses. To enable rigorous evaluation, we introduce T2L-ARVO, an expert-verified 50-case benchmark spanning five crash families in real-world projects. On T2L-ARVO, our baseline achieves up to 58.0% detection and 54.8% line-level localization rate. Together, T2L framework advance LLM-based vulnerability detection toward deployable, precision diagnostics in open-source software workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04613v1">Disentangling meaning from language in LLM-based machine translation</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 61 pages, 70 figures
    </div>
    <details class="paper-abstract">
      Mechanistic Interpretability (MI) seeks to explain how neural networks implement their capabilities, but the scale of Large Language Models (LLMs) has limited prior MI work in Machine Translation (MT) to word-level analyses. We study sentence-level MT from a mechanistic perspective by analyzing attention heads to understand how LLMs internally encode and distribute translation functions. We decompose MT into two subtasks: producing text in the target language (i.e. target language identification) and preserving the input sentence's meaning (i.e. sentence equivalence). Across three families of open-source models and 20 translation directions, we find that distinct, sparse sets of attention heads specialize in each subtask. Based on this insight, we construct subtask-specific steering vectors and show that modifying just 1% of the relevant heads enables instruction-free MT performance comparable to instruction-based prompting, while ablating these heads selectively disrupts their corresponding translation functions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04595v1">Harmonia: Algorithm-Hardware Co-Design for Memory- and Compute-Efficient BFP-based LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are powerful but incur high memory and computation costs. Quantization is an effective solution, with INT weights and FP activations being widely adopted to preserve accuracy. Prior works further reduce FP overhead by using block floating point (BFP) activations in linear layers, but fail to extend BFP to attention layers due to severe accuracy degradation, limiting overall efficiency. To address this challenge, we propose Harmonia, an algorithm-hardware co-design framework that enables all-layer BFP activations with a configurable hardware architecture. First, we systematically explore BFP configurations to achieve a better trade-off between accuracy and activation compression across all layers. Second, to reduce KV-cache storage and computation in attention layers, we introduce an asymmetric bit-allocation strategy and computations in attention layers,we introduce an asymmetric bit-allocation strategy combined with a hybrid offline-online outlier smoothing technique. This allow aggressive KV-cache compression from FP16 to 4-bit-mantissa BFP with only 0.3% average accuracy loss. Third, to fully exploit all-layer BFP activations, we design dedicated hardware components, including a reconfigurable PE supporting mixed data formats (BFP-INT and BPF-BFP), a real-time FP16-to-BFP converter, and a tiling-aware dataflow to reduce memory traffic. We evaluate Harmonia on GEMM operations in both linear and attention layers across eight widely used LLMs. Compared with prior works, Harmonia achieves 3.84x (up to 5.05x) higher area efficiency, 2.03x (up to 3.90x) better energy efficiency, and 3.08x (up to 4.62x) speedup on average.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04572v1">From Competition to Collaboration: Designing Sustainable Mechanisms Between LLMs and Online Forums</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      While Generative AI (GenAI) systems draw users away from (Q&A) forums, they also depend on the very data those forums produce to improve their performance. Addressing this paradox, we propose a framework of sequential interaction, in which a GenAI system proposes questions to a forum that can publish some of them. Our framework captures several intricacies of such a collaboration, including non-monetary exchanges, asymmetric information, and incentive misalignment. We bring the framework to life through comprehensive, data-driven simulations using real Stack Exchange data and commonly used LLMs. We demonstrate the incentive misalignment empirically, yet show that players can achieve roughly half of the utility in an ideal full-information scenario. Our results highlight the potential for sustainable collaboration that preserves effective knowledge sharing between AI systems and human knowledge platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04570v1">Can LLMs capture stable human-generated sentence entropy measures?</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Predicting upcoming words is a core mechanism of language comprehension and may be quantified using Shannon entropy. There is currently no empirical consensus on how many human responses are required to obtain stable and unbiased entropy estimates at the word level. Moreover, large language models (LLMs) are increasingly used as substitutes for human norming data, yet their ability to reproduce stable human entropy remains unclear. Here, we address both issues using two large publicly available cloze datasets in German 1 and English 2. We implemented a bootstrap-based convergence analysis that tracks how entropy estimates stabilize as a function of sample size. Across both languages, more than 97% of sentences reached stable entropy estimates within the available sample sizes. 90% of sentences converged after 111 responses in German and 81 responses in English, while low-entropy sentences (<1) required as few as 20 responses and high-entropy sentences (>2.5) substantially more. These findings provide the first direct empirical validation for common norming practices and demonstrate that convergence critically depends on sentence predictability. We then compared stable human entropy values with entropy estimates derived from several LLMs, including GPT-4o, using both logit-based probability extraction and sampling-based frequency estimation, GPT2-xl/german-GPT-2, RoBERTa Base/GottBERT, and LLaMA 2 7B Chat. GPT-4o showed the highest correspondence with human data, although alignment depended strongly on the extraction method and prompt design. Logit-based estimates minimized absolute error, whereas sampling-based estimates were better in capturing the dispersion of human variability. Together, our results establish practical guidelines for human norming and show that while LLMs can approximate human entropy, they are not interchangeable with stable human-derived distributions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04541v1">LycheeDecode: Accelerating Long-Context LLM Inference via Hybrid-Head Sparse Decoding</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      The proliferation of long-context large language models (LLMs) exposes a key bottleneck: the rapidly expanding key-value cache during decoding, which imposes heavy memory and latency costs. While recent approaches attempt to alleviate this by sharing a single set of crucial tokens across layers, such coarse-grained sharing undermines model performance by neglecting the functional diversity of attention heads. To address this, we propose LycheeDecode, an efficient decoding method centered on a fine-grained hybrid-head attention mechanism that employs a hardware-efficient top-k selection strategy. Specifically, the novel HardKuma-based mechanism partitions attention heads into a small subset of retrieval heads that dynamically identify crucial tokens and a majority of sparse heads that reuse them for efficient computation. Through extensive experiments on leading models like Llama3 and Qwen3 across diverse benchmarks for long-context understanding (e.g., LongBench, RULER) and complex reasoning (e.g., AIME24, OlympiadBench), we demonstrate that LycheeDecode achieves generative quality comparable to, and at times surpassing even the full-attention baseline. Crucially, this is accomplished with up to a 2.7x speedup at a 128K context length. By preserving the functional diversity of attention heads, our fine-grained strategy overcomes the performance bottlenecks of existing methods, providing a powerful and validated pathway to both efficient and high-quality long-context LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20417v2">SpeechMapper: Speech-to-text Embedding Projector for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 Accepted to ICASSP 2026
    </div>
    <details class="paper-abstract">
      Current speech LLMs bridge speech foundation models to LLMs using projection layers, training all of these components on speech instruction data. This strategy is computationally intensive and susceptible to task and prompt overfitting. We present SpeechMapper, a cost-efficient speech-to-LLM-embedding training approach that mitigates overfitting, enabling more robust and generalizable models. Our model is first pretrained without the LLM on inexpensive hardware, and then efficiently attached to the target LLM via a brief 1K-step instruction tuning (IT) stage. Through experiments on speech translation and spoken question answering, we demonstrate the versatility of SpeechMapper's pretrained block, presenting results for both task-agnostic IT, an ASR-based adaptation strategy that does not train in the target task, and task-specific IT. In task-agnostic settings, Speechmapper rivals the best instruction-following speech LLM from IWSLT25, despite never being trained on these tasks, while in task-specific settings, it outperforms this model across many datasets, despite requiring less data and compute. Overall, SpeechMapper offers a practical and scalable approach for efficient, generalizable speech-LLM integration without large-scale IT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04510v1">OSCAgent: Accelerating the Discovery of Organic Solar Cells with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Organic solar cells (OSCs) hold great promise for sustainable energy, but discovering high-performance materials is time-consuming and costly. Existing molecular generation methods can aid the design of OSC molecules, but they are mostly confined to optimizing known backbones and lack effective use of domain-specific chemical knowledge, often leading to unrealistic candidates. In this paper, we introduce OSCAgent, a multi-agent framework for OSC molecular discovery that unifies retrieval-augmented design, molecular generation, and systematic evaluation into a continuously improving pipeline, without requiring additional human intervention. OSCAgent comprises three collaborative agents. The Planner retrieves knowledge from literature-curated molecules and prior candidates to guide design directions. The Generator proposes new OSC acceptors aligned with these plans. The Experimenter performs comprehensive evaluation of candidate molecules and provides feedback for refinement. Experiments show that OSCAgent produces chemically valid, synthetically accessible OSC molecules and achieves superior predicted performance compared to both traditional and large language model (LLM)-only baselines. Representative results demonstrate that some candidates achieve predicted efficiencies approaching 18\%. The code will be publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04089v1">Scaling In-Context Online Learning Capability of LLMs via Cross-Episode Meta-RL</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) achieve strong performance when all task-relevant information is available upfront, as in static prediction and instruction-following problems. However, many real-world decision-making tasks are inherently online: crucial information must be acquired through interaction, feedback is delayed, and effective behavior requires balancing information collection and exploitation over time. While in-context learning enables adaptation without weight updates, existing LLMs often struggle to reliably leverage in-context interaction experience in such settings. In this work, we show that this limitation can be addressed through training. We introduce ORBIT, a multi-task, multi-episode meta-reinforcement learning framework that trains LLMs to learn from interaction in context. After meta-training, a relatively small open-source model (Qwen3-14B) demonstrates substantially improved in-context online learning on entirely unseen environments, matching the performance of GPT-5.2 and outperforming standard RL fine-tuning by a large margin. Scaling experiments further reveal consistent gains with model size, suggesting significant headroom for learn-at-inference-time decision-making agents. Code reproducing the results in the paper can be found at https://github.com/XiaofengLin7/ORBIT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.15732v4">Can LLMs Reconcile Knowledge Conflicts in Counterfactual Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-02-03
      | 💬 ICML 2025 Workshop on Scaling up Intervention Models
    </div>
    <details class="paper-abstract">
      Large Language Models have been shown to contain extensive world knowledge in their parameters, enabling impressive performance on many knowledge intensive tasks. However, when deployed in novel settings, LLMs often encounter situations where they must integrate parametric knowledge with new or unfamiliar information. In this work, we explore whether LLMs can combine knowledge in-context with their parametric knowledge through the lens of counterfactual reasoning. Through synthetic and real experiments in multi-hop reasoning problems, we show that LLMs generally struggle with counterfactual reasoning, often resorting to exclusively using their parametric knowledge. Moreover, we show that simple post-hoc finetuning can struggle to instill counterfactual reasoning ability -- often leading to degradation in stored parametric knowledge. Ultimately, our work reveals important limitations of current LLM's abilities to re-purpose parametric knowledge in novel settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04064v1">The CitizenQuery Benchmark: A Novel Dataset and Evaluation Pipeline for Measuring LLM Performance in Citizen Query Tasks</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      "Citizen queries" are questions asked by an individual about government policies, guidance, and services that are relevant to their circumstances, encompassing a range of topics including benefits, taxes, immigration, employment, public health, and more. This represents a compelling use case for Large Language Models (LLMs) that respond to citizen queries with information that is adapted to a user's context and communicated according to their needs. However, in this use case, any misinformation could have severe, negative, likely invisible ramifications for an individual placing their trust in a model's response. To this effect, we introduce CitizenQuery-UK, a benchmark dataset of 22 thousand pairs of citizen queries and responses that have been synthetically generated from the swathes of public information on $gov.uk$ about government in the UK. We present the curation methodology behind CitizenQuery-UK and an overview of its contents. We also introduce a methodology for the benchmarking of LLMs with the dataset, using an adaptation of FActScore to benchmark 11 models for factuality, abstention frequency, and verbosity. We document these results, and interpret them in the context of the public sector, finding that: (i) there are distinct performance profiles across model families, but each is competitive; (ii) high variance undermines utility; (iii) abstention is low and verbosity is high, with implications on reliability; and (iv) more trustworthy AI requires acknowledged "fallibility" in the way it interacts with users. The contribution of our research lies in assessing the trustworthiness of LLMs in citizen query tasks; as we see a world of increasing AI integration into day-to-day life, our benchmark, built entirely on open data, lays the foundations for better evidenced decision-making regarding AI and the public sector.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22387v2">AI-Generated Code Is Not Reproducible (Yet): An Empirical Study of Dependency Gaps in LLM-Based Coding Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      The rise of Large Language Models (LLMs) as coding agents promises to accelerate software development, but their impact on generated code reproducibility remains largely unexplored. This paper presents an empirical study investigating whether LLM-generated code can be executed successfully in a clean environment with only OS packages and using only the dependencies that the model specifies. We evaluate three state-of-the-art LLM coding agents (Claude Code, OpenAI Codex, and Gemini) across 300 projects generated from 100 standardized prompts in Python, JavaScript, and Java. We introduce a three-layer dependency framework (distinguishing between claimed, working, and runtime dependencies) to quantify execution reproducibility. Our results show that only 68.3% of projects execute out-of-the-box, with substantial variation across languages (Python 89.2%, Java 44.0%). We also find a 13.5 times average expansion from declared to actual runtime dependencies, revealing significant hidden dependencies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04039v1">Evaluating the Vulnerability Landscape of LLM-Generated Smart Contracts</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been widely adopted in modern software development lifecycles, where they are increasingly used to automate and assist code generation, significantly improving developer productivity and reducing development time. In the blockchain domain, developers increasingly rely on LLMs to generate and maintain smart contracts, the immutable, self-executing components of decentralized applications. Because deployed smart contracts cannot be modified, correctness and security are paramount, particularly in high-stakes domains such as finance and governance. Despite this growing reliance, the security implications of LLM-generated smart contracts remain insufficiently understood. In this work, we conduct a systematic security analysis of Solidity smart contracts generated by state-of-the-art LLMs, including ChatGPT, Gemini, and Sonnet. We evaluate these contracts against a broad set of known smart contract vulnerabilities to assess their suitability for direct deployment in production environments. Our extensive experimental study shows that, despite their syntactic correctness and functional completeness, LLM-generated smart contracts frequently exhibit severe security flaws that could be exploited in real-world settings. We further analyze and categorize these vulnerabilities, identifying recurring weakness patterns across different models. Finally, we discuss practical countermeasures and development guidelines to help mitigate these risks, offering actionable insights for both developers and researchers. Our findings aim to support safe integration of LLMs into smart contract development workflows and to strengthen the overall security of the blockchain ecosystem against future security failures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04033v1">On the Credibility of Evaluating LLMs using Survey Questions</a></div>
    <div class="paper-meta">
      📅 2026-02-03
      | 💬 Accepted to the Workshop on Multilingual and Multicultural Evaluation at EACL 2026, 12 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Recent studies evaluate the value orientation of large language models (LLMs) using adapted social surveys, typically by prompting models with survey questions and comparing their responses to average human responses. This paper identifies limitations in this methodology that, depending on the exact setup, can lead to both underestimating and overestimating the similarity of value orientation. Using the World Value Survey in three languages across five countries, we demonstrate that prompting methods (direct vs. chain-of-thought) and decoding strategies (greedy vs. sampling) significantly affect results. To assess the interaction between answers, we introduce a novel metric, self-correlation distance. This metric measures whether LLMs maintain consistent relationships between answers across different questions, as humans do. This indicates that even a high average agreement with human data, when considering LLM responses independently, does not guarantee structural alignment in responses. Additionally, we reveal a weak correlation between two common evaluation metrics, mean-squared distance and KL divergence, which assume that survey answers are independent of each other. For future research, we recommend CoT prompting, sampling-based decoding with dozens of samples, and robust analysis using multiple metrics, including self-correlation distance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22548v2">Are LLM Evaluators Really Narcissists? Sanity Checking Self-Preference Evaluations</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      Recent research has shown that large language models (LLMs) favor their own outputs when acting as judges, undermining the integrity of automated post-training and evaluation workflows. However, it is difficult to disentangle which evaluation biases are explained by narcissism versus general experimental confounds, distorting measurements of self-preference bias. We discover a core methodological confound which could reduce measurement error by 89.6%. Specifically, LLM evaluators may deliver self-preferring verdicts when the judge responds to queries which they completed incorrectly themselves; this would be true regardless of whether one of their responses is their own. To decouple self-preference signals from noisy outputs on hard problems, we introduce an Evaluator Quality Baseline, which compares the probability that a judge incorrectly votes for itself against the probability that it votes for an incorrect response from another model. Evaluating this simple baseline on 37,448 queries, only 51% of initial findings retain statistical significance. Finally, we turn towards characterizing the entropy of "easy" versus "hard" evaluation votes from LLM judges. Our corrective baseline enables future research on self-preference by eliminating noisy data from potential solutions. More widely, this work contributes to the growing body of work on cataloging and isolating judge-bias effects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.13697v4">RL in Name Only? Analyzing the Structural Assumptions in RL post-training for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      Reinforcement learning based post-training of large language models (LLMs) has recently gained attention, particularly following the release of DeepSeek R1, which applied GRPO for fine-tuning. Amid the growing claims around improved reasoning abilities attributed to RL post-training, we critically examine the formulation and assumptions underlying these methods. We start by highlighting popular structural assumptions made in modeling LLM training as an MDP, and show how they lead to a degenerate MDP, that characterizes the problem as a contextual bandit, where RL updates naturally collapse into a form of on-policy variant of outcome-driven supervised learning. The two critical structural assumptions include (1) making the MDP states be just a concatenation of the actions with states becoming the context window and the actions becoming the tokens in LLMs and (2) splitting the reward of a state-action trajectory uniformly across the trajectory. Our comprehensive analysis demonstrates that, due to these simplifying assumptions, GRPO objective reduces to filtered Iterative SFT, an on-policy variant of supervised fine-tuning. Our experiments on benchmarks including GSM8K and Countdown, across a diverse set of model families show that Filtered Iterative SFT, incorporating both positive and negative samples, achieves performance comparable to GRPO-based training. We also show that these structural assumptions indirectly incentivize RL to generate longer sequences of intermediate tokens which in turn feeds into the narrative of "RL incentivizing thinking because it generates longer thinking traces."
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.02164v2">Co-RedTeam: Orchestrated Security Discovery and Exploitation with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown promise in assisting cybersecurity tasks, yet existing approaches struggle with automatic vulnerability discovery and exploitation due to limited interaction, weak execution grounding, and a lack of experience reuse. We propose Co-RedTeam, a security-aware multi-agent framework designed to mirror real-world red-teaming workflows by integrating security-domain knowledge, code-aware analysis, execution-grounded iterative reasoning, and long-term memory. Co-RedTeam decomposes vulnerability analysis into coordinated discovery and exploitation stages, enabling agents to plan, execute, validate, and refine actions based on real execution feedback while learning from prior trajectories. Extensive evaluations on challenging security benchmarks demonstrate that Co-RedTeam consistently outperforms strong baselines across diverse backbone models, achieving over 60% success rate in vulnerability exploitation and over 10% absolute improvement in vulnerability detection. Ablation and iteration studies further confirm the critical role of execution feedback, structured interaction, and memory for building robust and generalizable cybersecurity agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03979v1">Likelihood-Based Reward Designs for General LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) on reasoning benchmarks via reinforcement learning requires a specific reward function, often binary, for each benchmark. This comes with two potential limitations: the need to design the reward, and the potentially sparse nature of binary rewards. Here, we systematically investigate rewards derived from the probability or log-probability of emitting the reference answer (or any other prompt continuation present in the data), which have the advantage of not relying on specific verifiers and being available at scale. Several recent works have advocated for the use of similar rewards (e.g., VeriFree, JEPO, RLPR, NOVER). We systematically compare variants of likelihood-based rewards with standard baselines, testing performance both on standard mathematical reasoning benchmarks, and on long-form answers where no external verifier is available. We find that using the log-probability of the reference answer as the reward for chain-of-thought (CoT) learning is the only option that performs well in all setups. This reward is also consistent with the next-token log-likelihood loss used during pretraining. In verifiable settings, log-probability rewards bring comparable or better success rates than reinforcing with standard binary rewards, and yield much better perplexity. In non-verifiable settings, they perform on par with SFT. On the other hand, methods based on probability, such as VeriFree, flatline on non-verifiable settings due to vanishing probabilities of getting the correct answer. Overall, this establishes log-probability rewards as a viable method for CoT fine-tuning, bridging the short, verifiable and long, non-verifiable answer settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03955v1">AgentArk: Distilling Multi-Agent Intelligence into a Single LLM Agent</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      While large language model (LLM) multi-agent systems achieve superior reasoning performance through iterative debate, practical deployment is limited by their high computational cost and error propagation. This paper proposes AgentArk, a novel framework to distill multi-agent dynamics into the weights of a single model, effectively transforming explicit test-time interactions into implicit model capabilities. This equips a single agent with the intelligence of multi-agent systems while remaining computationally efficient. Specifically, we investigate three hierarchical distillation strategies across various models, tasks, scaling, and scenarios: reasoning-enhanced fine-tuning; trajectory-based augmentation; and process-aware distillation. By shifting the burden of computation from inference to training, the distilled models preserve the efficiency of one agent while exhibiting strong reasoning and self-correction performance of multiple agents. They further demonstrate enhanced robustness and generalization across diverse reasoning tasks. We hope this work can shed light on future research on efficient and robust multi-agent development. Our code is at https://github.com/AIFrontierLab/AgentArk.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03950v1">Enhancing Mathematical Problem Solving in LLMs through Execution-Driven Reasoning Augmentation</a></div>
    <div class="paper-meta">
      📅 2026-02-03
      | 💬 9 pages, 7 figures, submitted to ACL ARR 2026
    </div>
    <details class="paper-abstract">
      Mathematical problem solving is a fundamental benchmark for assessing the reasoning capabilities of artificial intelligence and a gateway to applications in education, science, and engineering where reliable symbolic reasoning is essential. Although recent advances in multi-agent LLM-based systems have enhanced their mathematical reasoning capabilities, they still lack a reliably revisable representation of the reasoning process. Existing agents either operate in rigid sequential pipelines that cannot correct earlier steps or rely on heuristic self-evaluation that can fail to identify and fix errors. In addition, programmatic context can distract language models and degrade accuracy. To address these gaps, we introduce Iteratively Improved Program Construction (IIPC), a reasoning method that iteratively refines programmatic reasoning chains and combines execution feedback with the native Chain-of-thought abilities of the base LLM to maintain high-level contextual focus. IIPC surpasses competing approaches in the majority of reasoning benchmarks on multiple base LLMs. All code and implementations are released as open source.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03794v1">Understanding Agent Scaling in LLM-Based Multi-Agent Systems via Diversity</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      LLM-based multi-agent systems (MAS) have emerged as a promising approach to tackle complex tasks that are difficult for individual LLMs. A natural strategy is to scale performance by increasing the number of agents; however, we find that such scaling exhibits strong diminishing returns in homogeneous settings, while introducing heterogeneity (e.g., different models, prompts, or tools) continues to yield substantial gains. This raises a fundamental question: what limits scaling, and why does diversity help? We present an information-theoretic framework showing that MAS performance is bounded by the intrinsic task uncertainty, not by agent count. We derive architecture-agnostic bounds demonstrating that improvements depend on how many effective channels the system accesses. Homogeneous agents saturate early because their outputs are strongly correlated, whereas heterogeneous agents contribute complementary evidence. We further introduce $K^*$, an effective channel count that quantifies the number of effective channels without ground-truth labels. Empirically, we show that heterogeneous configurations consistently outperform homogeneous scaling: 2 diverse agents can match or exceed the performance of 16 homogeneous agents. Our results provide principled guidelines for building efficient and robust MAS through diversity-aware design. Code and Dataset are available at the link: https://github.com/SafeRL-Lab/Agent-Scaling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.16552v6">Think Silently, Think Fast: Dynamic Latent Compression of LLM Reasoning Chains</a></div>
    <div class="paper-meta">
      📅 2026-02-03
      | 💬 15 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) achieve superior performance through Chain-of-Thought (CoT) reasoning, but these token-level reasoning chains are computationally expensive and inefficient. In this paper, we introduce Compressed Latent Reasoning (CoLaR), a novel framework that dynamically compresses reasoning processes in latent space through a two-stage training approach. First, during supervised fine-tuning, CoLaR extends beyond next-token prediction by incorporating an auxiliary next compressed embedding prediction objective. This process merges embeddings of consecutive tokens using a compression factor randomly sampled from a predefined range, and trains a specialized latent head to predict distributions of subsequent compressed embeddings. Second, we enhance CoLaR through reinforcement learning (RL) that leverages the latent head's non-deterministic nature to explore diverse reasoning paths and exploit more compact ones. This approach enables CoLaR to: i) perform reasoning at a dense latent level (i.e., silently), substantially reducing reasoning chain length, and ii) dynamically adjust reasoning speed at inference time by simply prompting the desired compression factor. Extensive experiments across four mathematical reasoning datasets demonstrate that CoLaR achieves 14.1% higher accuracy than latent-based baseline methods at comparable compression ratios, and reduces reasoning chain length by 53.3% with only 4.8% performance degradation compared to explicit CoT method. Moreover, when applied to more challenging mathematical reasoning tasks, our RL-enhanced CoLaR demonstrates performance gains of up to 5.4% while dramatically reducing latent reasoning chain length by 82.8%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10415v2">How to Trick Your AI TA: A Systematic Study of Academic Jailbreaking in LLM Code Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-02-03
      | 💬 This manuscript has been withdrawn by the authors because the methodology and results have been superseded by a more rigorous framework (SPACI and AST-ASIP). The corrected and expanded findings are now available in arXiv:2601.21360. Please cite the new manuscript instead
    </div>
    <details class="paper-abstract">
      The use of Large Language Models (LLMs) as automatic judges for code evaluation is becoming increasingly prevalent in academic environments. But their reliability can be compromised by students who may employ adversarial prompting strategies in order to induce misgrading and secure undeserved academic advantages. In this paper, we present the first large-scale study of jailbreaking LLM-based automated code evaluators in academic context. Our contributions are: (i) We systematically adapt 20+ jailbreaking strategies for jailbreaking AI code evaluators in the academic context, defining a new class of attacks termed academic jailbreaking. (ii) We release a poisoned dataset of 25K adversarial student submissions, specifically designed for the academic code-evaluation setting, sourced from diverse real-world coursework and paired with rubrics and human-graded references, and (iii) In order to capture the multidimensional impact of academic jailbreaking, we systematically adapt and define three jailbreaking metrics (Jailbreak Success Rate, Score Inflation, and Harmfulness). (iv) We comprehensively evalulate the academic jailbreaking attacks using six LLMs. We find that these models exhibit significant vulnerability, particularly to persuasive and role-play-based attacks (up to 97% JSR). Our adversarial dataset and benchmark suite lay the groundwork for next-generation robust LLM-based evaluators in academic code assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.21551v4">Grokking in LLM Pretraining? Monitor Memorization-to-Generalization without Test</a></div>
    <div class="paper-meta">
      📅 2026-02-03
      | 💬 Accepted at ICLR 2026
    </div>
    <details class="paper-abstract">
      This paper presents the first study of grokking in practical LLM pretraining. Specifically, we investigate when an LLM memorizes the training data, when its generalization on downstream tasks starts to improve, and what happens if there is a lag between the two. Unlike existing works studying when a small model generalizes to limited and specified tasks during thousands epochs' training on algorithmic data, we focus on a practical setting for LLMs, i.e., one-epoch pretraining of next-token prediction on a cross-domain, large-scale corpus, and generalization on diverse benchmark tasks covering math/commonsense reasoning, code generation, and domain-specific retrieval. Our study, for the first time, verifies that grokking still emerges in pretraining mixture-of-experts (MoE) LLMs, though different local data groups may enter their grokking stages asynchronously due to the heterogeneity of their distributions and attributions to others. To find a mechanistic interpretation of this local grokking, we investigate the dynamics of training data's pathways (i.e., expert choices across layers in MoE). Our primary discovery is that the pathways evolve from random, non-smooth across layers, instance-specific to more structured and transferable across samples, despite the converged pretraining loss. This depicts a transition from memorization to generalization. Two novel metrics are developed to quantify these patterns: one computes the pathway similarity between samples, while the other measures the consistency of aggregated experts between subsequent layers for each sample. These training data based metrics induce zero cost but can faithfully track and monitor the generalization of LLMs on downstream tasks, which, in conventional settings, requires costly instruction tuning and benchmark evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03712v1">SWE-Refactor: A Repository-Level Benchmark for Real-World LLM-Based Code Refactoring</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently attracted wide interest for tackling software engineering tasks. In contrast to code generation, refactoring demands precise, semantics-preserving edits that improve program structure, which also makes automated evaluation challenging. However, existing refactoring benchmarks commonly suffer from three shortcomings: limited coverage of refactoring scenarios, the inclusion of instances that mix refactoring with unrelated changes, and insufficient repository-level context for realistic assessment. To mitigate these issues, we introduce SWE-Refactor, a new benchmark for LLM-based code refactoring. SWE-Refactor comprises 1,099 developer-written, behavior-preserving refactorings mined from 18 Java projects, including 922 atomic and 177 compound instances. Each instance is validated via compilation, test execution, and automated refactoring detection tools to ensure correctness. We evaluate nine widely used LLMs on SWE-Refactor, covering models such as GPT-4o-mini, DeepSeek-V3, and CodeLLaMa, to provide representative reference results. Our results show that complex and compound refactorings remain the primary source of failures; notably, an OpenAI Codex agent achieves only 39.4% success on compound instances. We release SWE-Refactor and all evaluation results to facilitate future research on LLM-based code refactoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.02313v2">Interpreting and Controlling LLM Reasoning through Integrated Policy Gradient</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate strong reasoning abilities in solving complex real-world problems. Yet, the internal mechanisms driving these complex reasoning behaviors remain opaque. Existing interpretability approaches targeting reasoning either identify components (e.g., neurons) correlated with special textual patterns, or rely on human-annotated contrastive pairs to derive control vectors. Consequently, current methods struggle to precisely localize complex reasoning mechanisms or capture sequential influence from model internal workings to the reasoning outputs. In this paper, built on outcome-oriented and sequential-influence-aware principles, we focus on identifying components that have sequential contribution to reasoning behavior where outcomes are cumulated by long-range effects. We propose Integrated Policy Gradient (IPG), a novel framework that attributes reasoning behaviors to model's inner components by propagating compound outcome-based signals such as post reasoning accuracy backward through model inference trajectories. Empirical evaluations demonstrate that our approach achieves more precise localization and enables reliable modulation of reasoning behaviors (e.g., reasoning capability, reasoning strength) across diverse reasoning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03690v1">LLM-Inspired Pretrain-Then-Finetune for Small-Data, Large-Scale Optimization</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      We consider small-data, large-scale decision problems in which a firm must make many operational decisions simultaneously (e.g., across a large product portfolio) while observing only a few, potentially noisy, data points per instance. Inspired by the success of large language models (LLMs), we propose a pretrain-then-finetune approach built on a designed Transformer model to address this challenge. The model is first pretrained on large-scale, domain-informed synthetic data that encode managerial knowledge and structural features of the decision environment, and is then fine-tuned on real observations. This new pipeline offers two complementary advantages: pretraining injects domain knowledge into the learning process and enables the training of high-capacity models using abundant synthetic data, while finetuning adapts the pretrained model to the operational environment and improves alignment with the true data-generating regime. While we have leveraged the Transformer's state-of-the-art representational capacity, particularly its attention mechanism, to efficiently extract cross-task structure, our approach is not an off-the-shelf application. Instead, it relies on problem-specific architectural design and a tailored training procedure to match the decision setting. Theoretically, we develop the first comprehensive error analysis regarding Transformer learning in relevant contexts, establishing nonasymptotic guarantees that validate the method's effectiveness. Critically, our analysis reveals how pretraining and fine-tuning jointly determine performance, with the dominant contribution governed by whichever is more favorable. In particular, finetuning exhibits an economies-of-scale effect, whereby transfer learning becomes increasingly effective as the number of instances grows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03688v1">TodyComm: Task-Oriented Dynamic Communication for Multi-Round LLM-based Multi-Agent System</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      Multi-round LLM-based multi-agent systems rely on effective communication structures to support collaboration across rounds. However, most existing methods employ a fixed communication topology during inference, which falls short in many realistic applications where the agents' roles may change \textit{across rounds} due to dynamic adversary, task progression, or time-varying constraints such as communication bandwidth. In this paper, we propose addressing this issue through TodyComm, a \textbf{t}ask-\textbf{o}riented \textbf{dy}namic \textbf{comm}unication algorithm. It produces behavior-driven collaboration topologies that adapt to the dynamics at each round, optimizing the utility for the task through policy gradient. Experiments on five benchmarks demonstrate that under both dynamic adversary and communications budgets, TodyComm delivers superior task effectiveness while retaining token efficiency and scalability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.21543v2">Self-CriTeach: LLM Self-Teaching and Self-Critiquing for Improving Robotic Planning via Automated Domain Generation</a></div>
    <div class="paper-meta">
      📅 2026-02-03
      | 💬 31 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently shown strong promise for robotic task planning, particularly through automatic planning domain generation. Planning domains are brittle under imperfect logical states and perception noise; prior approaches largely treat generated planning domains as plan utilities, overlooking their potential as scalable sources of reasoning supervision and structured reward signals. At the same time, reasoning LLMs depend on chain-of-thought (CoT) supervision that is expensive to collect for robotic tasks, and reinforcement learning (RL) faces challenges in reward engineering. We propose Self-CriTeach, an LLM self-teaching and self-critiquing framework in which an LLM autonomously generates symbolic planning domains that serve a dual role: (i) enabling large-scale generation of robotic planning problem-plan pairs, and (ii) providing structured reward functions. First, the self-written domains enable large-scale generation of symbolic task plans, which are automatically transformed into extended CoT trajectories for supervised fine-tuning. Second, the self-written domains are reused as structured reward functions, providing dense feedback for reinforcement learning without manual reward engineering. This unified training pipeline yields a planning-enhanced LLM with higher planning success rates, stronger cross-task generalization, reduced inference cost, and improved robustness to imperfect logical states.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04900v1">Evaluating Kubernetes Performance for GenAI Inference: From Automatic Speech Recognition to LLM Summarization</a></div>
    <div class="paper-meta">
      📅 2026-02-03
      | 💬 A accepted at the 17th International Conference on Performance Engineering
    </div>
    <details class="paper-abstract">
      As Generative AI (GenAI), particularly inference, rapidly emerges as a dominant workload category, the Kubernetes ecosystem is proactively evolving to natively support its unique demands. This industry paper demonstrates how emerging Kubernetes-native projects can be combined to deliver the benefits of container orchestration, such as scalability and resource efficiency, to complex AI workflows. We implement and evaluate an illustrative, multi-stage use case consisting of automatic speech recognition and summarization. First, we address batch inference by using Kueue to manage jobs that transcribe audio files with Whisper models and Dynamic Accelerator Slicer (DAS) to increase parallel job execution. Second, we address a discrete online inference scenario by feeding the transcripts to a Large Language Model for summarization hosted using llm-d, a novel solution utilizing the recent developments around the Kubernetes Gateway API Inference Extension (GAIE) for optimized routing of inference requests. Our findings illustrate that these complementary components (Kueue, DAS, and GAIE) form a cohesive, high-performance platform, proving Kubernetes' capability to serve as a unified foundation for demanding GenAI workloads: Kueue reduced total makespan by up to 15%; DAS shortened mean job completion time by 36%; and GAIE improved Time to First Token by 82\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03648v1">Can Developers rely on LLMs for Secure IaC Development?</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      We investigated the capabilities of GPT-4o and Gemini 2.0 Flash for secure Infrastructure as Code (IaC) development. For security smell detection, on the Stack Overflow dataset, which primarily contains small, simplified code snippets, the models detected at least 71% of security smells when prompted to analyze code from a security perspective (general prompt). With a guided prompt (adding clear, step-by-step instructions), this increased to 78%.In GitHub repositories, which contain complete, real-world project scripts, a general prompt was less effective, leaving more than half of the smells undetected. However, with the guided prompt, the models uncovered at least 67% of the smells. For secure code generation, we prompted LLMs with 89 vulnerable synthetic scenarios and observed that only 7% of the generated scripts were secure. Adding an explicit instruction to generate secure code increased GPT secure output rate to 17%, while Gemini changed little (8%). These results highlight the need for further research to improve LLMs' capabilities in assisting developers with secure IaC development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.12517v3">Interaction Context Often Increases Sycophancy in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-03
      | 💬 To appear in the proceedings of the ACM Conference on Human Factors in Computing Systems (CHI 2026)
    </div>
    <details class="paper-abstract">
      We investigate how the presence and type of interaction context shapes sycophancy in LLMs. While real-world interactions allow models to mirror a user's values, preferences, and self-image, prior work often studies sycophancy in zero-shot settings devoid of context. Using two weeks of interaction context from 38 users, we evaluate two forms of sycophancy: (1) agreement sycophancy -- the tendency of models to produce overly affirmative responses, and (2) perspective sycophancy -- the extent to which models reflect a user's viewpoint. Agreement sycophancy tends to increase with the presence of user context, though model behavior varies based on the context type. User memory profiles are associated with the largest increases in agreement sycophancy (e.g. $+$45\% for Gemini 2.5 Pro), and some models become more sycophantic even with non-user synthetic contexts (e.g. $+$15\% for Llama 4 Scout). Perspective sycophancy increases only when models can accurately infer user viewpoints from interaction context. Overall, context shapes sycophancy in heterogeneous ways, underscoring the need for evaluations grounded in real-world interactions and raising questions for system design around alignment, memory, and personalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03630v1">Can LLMs Do Rocket Science? Exploring the Limits of Complex Reasoning with GTOC 12</a></div>
    <div class="paper-meta">
      📅 2026-02-03
      | 💬 Extended version of the paper presented at AIAA SciTech 2026 Forum. Includes futher experiments, corrections and new appendix
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable proficiency in code generation and general reasoning, yet their capacity for autonomous multi-stage planning in high-dimensional, physically constrained environments remains an open research question. This study investigates the limits of current AI agents by evaluating them against the 12th Global Trajectory Optimization Competition (GTOC 12), a complex astrodynamics challenge requiring the design of a large-scale asteroid mining campaign. We adapt the MLE-Bench framework to the domain of orbital mechanics and deploy an AIDE-based agent architecture to autonomously generate and refine mission solutions. To assess performance beyond binary validity, we employ an "LLM-as-a-Judge" methodology, utilizing a rubric developed by domain experts to evaluate strategic viability across five structural categories. A comparative analysis of models, ranging from GPT-4-Turbo to reasoning-enhanced architectures like Gemini 2.5 Pro, and o3, reveals a significant trend: the average strategic viability score has nearly doubled in the last two years (rising from 9.3 to 17.2 out of 26). However, we identify a critical capability gap between strategy and execution. While advanced models demonstrate sophisticated conceptual understanding, correctly framing objective functions and mission architectures, they consistently fail at implementation due to physical unit inconsistencies, boundary condition errors, and inefficient debugging loops. We conclude that, while current LLMs often demonstrate sufficient knowledge and intelligence to tackle space science tasks, they remain limited by an implementation barrier, functioning as powerful domain facilitators rather than fully autonomous engineers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03615v1">KTV: Keyframes and Key Tokens Selection for Efficient Training-Free Video LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      Training-free video understanding leverages the strong image comprehension capabilities of pre-trained vision language models (VLMs) by treating a video as a sequence of static frames, thus obviating the need for costly video-specific training. However, this paradigm often suffers from severe visual redundancy and high computational overhead, especially when processing long videos. Crucially, existing keyframe selection strategies, especially those based on CLIP similarity, are prone to biases and may inadvertently overlook critical frames, resulting in suboptimal video comprehension. To address these significant challenges, we propose \textbf{KTV}, a novel two-stage framework for efficient and effective training-free video understanding. In the first stage, KTV performs question-agnostic keyframe selection by clustering frame-level visual features, yielding a compact, diverse, and representative subset of frames that mitigates temporal redundancy. In the second stage, KTV applies key visual token selection, pruning redundant or less informative tokens from each selected keyframe based on token importance and redundancy, which significantly reduces the number of tokens fed into the LLM. Extensive experiments on the Multiple-Choice VideoQA task demonstrate that KTV outperforms state-of-the-art training-free baselines while using significantly fewer visual tokens, \emph{e.g.}, only 504 visual tokens for a 60-min video with 10800 frames, achieving $44.8\%$ accuracy on the MLVU-Test benchmark. In particular, KTV also exceeds several training-based approaches on certain benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03608v1">Controlling Output Rankings in Generative Engines for LLM-based Search</a></div>
    <div class="paper-meta">
      📅 2026-02-03
      | 💬 23 pages
    </div>
    <details class="paper-abstract">
      The way customers search for and choose products is changing with the rise of large language models (LLMs). LLM-based search, or generative engines, provides direct product recommendations to users, rather than traditional online search results that require users to explore options themselves. However, these recommendations are strongly influenced by the initial retrieval order of LLMs, which disadvantages small businesses and independent creators by limiting their visibility. In this work, we propose CORE, an optimization method that \textbf{C}ontrols \textbf{O}utput \textbf{R}ankings in g\textbf{E}nerative Engines for LLM-based search. Since the LLM's interactions with the search engine are black-box, CORE targets the content returned by search engines as the primary means of influencing output rankings. Specifically, CORE optimizes retrieved content by appending strategically designed optimization content to steer the ranking of outputs. We introduce three types of optimization content: string-based, reasoning-based, and review-based, demonstrating their effectiveness in shaping output rankings. To evaluate CORE in realistic settings, we introduce ProductBench, a large-scale benchmark with 15 product categories and 200 products per category, where each product is associated with its top-10 recommendations collected from Amazon's search interface. Extensive experiments on four LLMs with search capabilities (GPT-4o, Gemini-2.5, Claude-4, and Grok-3) demonstrate that CORE achieves an average Promotion Success Rate of \textbf{91.4\% @Top-5}, \textbf{86.6\% @Top-3}, and \textbf{80.3\% @Top-1}, across 15 product categories, outperforming existing ranking manipulation methods while preserving the fluency of optimized content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03589v1">SlowFocus: Enhancing Fine-grained Temporal Understanding in Video LLM</a></div>
    <div class="paper-meta">
      📅 2026-02-03
      | 💬 NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated exceptional capabilities in text understanding, which has paved the way for their expansion into video LLMs (Vid-LLMs) to analyze video data. However, current Vid-LLMs struggle to simultaneously retain high-quality frame-level semantic information (i.e., a sufficient number of tokens per frame) and comprehensive video-level temporal information (i.e., an adequate number of sampled frames per video). This limitation hinders the advancement of Vid-LLMs towards fine-grained video understanding. To address this issue, we introduce the SlowFocus mechanism, which significantly enhances the equivalent sampling frequency without compromising the quality of frame-level visual tokens. SlowFocus begins by identifying the query-related temporal segment based on the posed question, then performs dense sampling on this segment to extract local high-frequency features. A multi-frequency mixing attention module is further leveraged to aggregate these local high-frequency details with global low-frequency contexts for enhanced temporal comprehension. Additionally, to tailor Vid-LLMs to this innovative mechanism, we introduce a set of training strategies aimed at bolstering both temporal grounding and detailed temporal reasoning capabilities. Furthermore, we establish FineAction-CGR, a benchmark specifically devised to assess the ability of Vid-LLMs to process fine-grained temporal understanding tasks. Comprehensive experiments demonstrate the superiority of our mechanism across both existing public video understanding benchmarks and our proposed FineAction-CGR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.17360v2">Cortex: Achieving Low-Latency, Cost-Efficient Remote Data Access For LLM via Semantic-Aware Knowledge Caching</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents tackle data-intensive tasks such as deep research and code generation. However, their effectiveness depends on frequent interactions with knowledge sources across remote clouds or regions. Such interactions can create non-trivial latency and cost bottlenecks. Existing caching solutions focus on exact-match queries, limiting their effectiveness for semantic knowledge reuse. To address this challenge, we introduce Cortex, a novel cross-region knowledge caching architecture for LLM agents. At its core are two abstractions: Semantic Element (SE) and Semantic Retrieval Index (Seri). A semantic element captures the semantic embedding representation of an LLM query together with performance-aware metadata such as latency, cost, and staticity. Seri then provides two-stage retrieval: a vector similar index with semantic embedding for fast candidate selection and a lightweight LLM-powered semantic judger for precise validation. Atop these primitives, Cortex builds a new cache interface that includes a new semantic-aware cache hit definition, a cost-efficient eviction policy, and proactive prefetching. To reduce overhead, Cortex co-locates the small LLM judger with the main LLM using adaptive scheduling and resource sharing. Our evaluation demonstrates that Cortex delivers substantial performance improvements without compromising correctness. On representative search workloads, Cortex achieves up to a 3.6x increase in throughput by maintaining cache hit rates of over 85%, while preserving accuracy virtually identical to non-cached baselines. Cortex also improves throughput for coding tasks by 20%, showcasing its versatility across diverse agentic workloads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03554v1">When Single Answer Is Not Enough: Rethinking Single-Step Retrosynthesis Benchmarks for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      Recent progress has expanded the use of large language models (LLMs) in drug discovery, including synthesis planning. However, objective evaluation of retrosynthesis performance remains limited. Existing benchmarks and metrics typically rely on published synthetic procedures and Top-K accuracy based on single ground-truth, which does not capture the open-ended nature of real-world synthesis planning. We propose a new benchmarking framework for single-step retrosynthesis that evaluates both general-purpose and chemistry-specialized LLMs using ChemCensor, a novel metric for chemical plausibility. By emphasizing plausibility over exact match, this approach better aligns with human synthesis planning practices. We also introduce CREED, a novel dataset comprising millions of ChemCensor-validated reaction records for LLM training, and use it to train a model that improves over the LLM baselines under this benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03510v1">Semantic Routing: Exploring Multi-Layer LLM Feature Weighting for Diffusion Transformers</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      Recent DiT-based text-to-image models increasingly adopt LLMs as text encoders, yet text conditioning remains largely static and often utilizes only a single LLM layer, despite pronounced semantic hierarchy across LLM layers and non-stationary denoising dynamics over both diffusion time and network depth. To better match the dynamic process of DiT generation and thereby enhance the diffusion model's generative capability, we introduce a unified normalized convex fusion framework equipped with lightweight gates to systematically organize multi-layer LLM hidden states via time-wise, depth-wise, and joint fusion. Experiments establish Depth-wise Semantic Routing as the superior conditioning strategy, consistently improving text-image alignment and compositional generation (e.g., +9.97 on the GenAI-Bench Counting task). Conversely, we find that purely time-wise fusion can paradoxically degrade visual generation fidelity. We attribute this to a train-inference trajectory mismatch: under classifier-free guidance, nominal timesteps fail to track the effective SNR, causing semantically mistimed feature injection during inference. Overall, our results position depth-wise routing as a strong and effective baseline and highlight the critical need for trajectory-aware signals to enable robust time-dependent conditioning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.18179v4">Problem Solved? Information Extraction Design Space for Layout-Rich Documents using LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-03
      | 💬 accepted at EMNLP'25
    </div>
    <details class="paper-abstract">
      This paper defines and explores the design space for information extraction (IE) from layout-rich documents using large language models (LLMs). The three core challenges of layout-aware IE with LLMs are 1) data structuring, 2) model engagement, and 3) output refinement. Our study investigates the sub-problems and methods within these core challenges, such as input representation, chunking, prompting, selection of LLMs, and multimodal models. It examines the effect of different design choices through LayIE-LLM, a new, open-source, layout-aware IE test suite, benchmarking against traditional, fine-tuned IE models. The results on two IE datasets show that LLMs require adjustment of the IE pipeline to achieve competitive performance: the optimized configuration found with LayIE-LLM achieves 13.3--37.5 F1 points more than a general-practice baseline configuration using the same LLM. To find a well-working configuration, we develop a one-factor-at-a-time (OFAT) method that achieves near-optimal results. Our method is only 0.8--1.8 points lower than the best full factorial exploration with a fraction (2.8%) of the required computation. Overall, we demonstrate that, if well-configured, general-purpose LLMs match the performance of specialized models, providing a cost-effective, finetuning-free alternative. Our test-suite is available at https://github.com/gayecolakoglu/LayIE-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03478v1">When Routing Collapses: On the Degenerate Convergence of LLM Routers</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      LLM routing aims to achieve a favorable quality--cost trade-off by dynamically assigning easy queries to smaller models and harder queries to stronger ones. However, across both unimodal and multimodal settings, we uncover a pervasive yet underexplored failure mode in existing routers: as the user's cost budget increases, routers systematically default to the most capable and most expensive model even when cheaper models already suffice. As a result, current routers under-utilize small models, wasting computation and monetary cost and undermining the core promise of routing; we term this phenomenon routing collapse. We attribute routing collapse to an objective--decision mismatch: many routers are trained to predict scalar performance scores, whereas routing decisions ultimately depend on discrete comparisons among candidate models. Consequently, small prediction errors can flip relative orderings and trigger suboptimal selections. To bridge this gap, we propose EquiRouter, a decision-aware router that directly learns model rankings, restoring the role of smaller models and mitigating routing collapse. On RouterBench, EquiRouter reduces cost by about 17\% at GPT-4-level performance compared to the strongest prior router. Our code is available at https://github.com/AIGNLAI/EquiRouter.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03439v1">Ontology-to-tools compilation for executable semantic constraint enforcement in LLM agents</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      We introduce ontology-to-tools compilation as a proof-of-principle mechanism for coupling large language models (LLMs) with formal domain knowledge. Within The World Avatar (TWA), ontological specifications are compiled into executable tool interfaces that LLM-based agents must use to create and modify knowledge graph instances, enforcing semantic constraints during generation rather than through post-hoc validation. Extending TWA's semantic agent composition framework, the Model Context Protocol (MCP) and associated agents are integral components of the knowledge graph ecosystem, enabling structured interaction between generative models, symbolic constraints, and external resources. An agent-based workflow translates ontologies into ontology-aware tools and iteratively applies them to extract, validate, and repair structured knowledge from unstructured scientific text. Using metal-organic polyhedra synthesis literature as an illustrative case, we show how executable ontological semantics can guide LLM behaviour and reduce manual schema and prompt engineering, establishing a general paradigm for embedding formal knowledge into generative systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.00880v2">Sensing What Surveys Miss: Understanding and Personalizing Proactive LLM Support by User Modeling</a></div>
    <div class="paper-meta">
      📅 2026-02-03
      | 💬 This manuscript has been accepted to CHI 2026
    </div>
    <details class="paper-abstract">
      Difficulty spillover and suboptimal help-seeking challenge the sequential, knowledge-intensive nature of digital tasks. In online surveys, tough questions can drain mental energy and hurt performance on later questions, while users often fail to recognize when they need assistance or may satisfy, lacking motivation to seek help. We developed a proactive, adaptive system using electrodermal activity and mouse movement to predict when respondents need support. Personalized classifiers with a rule-based threshold adaptation trigger timely LLM-based clarifications and explanations. In a within-subjects study (N=32), aligned-adaptive timing was compared to misaligned-adaptive and random-adaptive controls. Aligned-adaptive assistance improved response accuracy by 21%, reduced false negative rates from 50.9% to 22.9%, and improved perceived efficiency, dependability, and benevolence. Properly timed interventions prevent cascades of degraded responses, showing that aligning support with cognitive states improves both the outcomes and the user experience. This enables more effective, personalized LLM-assisted support in survey-based research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.17813v2">Don't Overthink it. Preferring Shorter Thinking Chains for Improved LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      Reasoning large language models (LLMs) heavily rely on scaling test-time compute to perform complex reasoning tasks by generating extensive "thinking" chains. While demonstrating impressive results, this approach incurs significant computational costs and inference time. In this work, we challenge the assumption that long thinking chains results in better reasoning capabilities. We first demonstrate that shorter reasoning chains within individual questions are significantly more likely to yield correct answers - up to 34.5% more accurate than the longest chain sampled for the same question. Based on these results, we suggest short-m@k, a novel reasoning LLM inference method. Our method executes k independent generations in parallel and halts computation once the first m thinking processes are done. The final answer is chosen using majority voting among these m chains. Basic short-1@k demonstrates similar or even superior performance over standard majority voting in low-compute settings - using up to 40% fewer thinking tokens. short-3@k, while slightly less efficient than short-1@k, consistently surpasses majority voting across all compute budgets, while still being substantially faster (up to 33% wall time reduction). To further validate our findings, we finetune LLMs using short, long, and randomly selected reasoning chains. We then observe that training on the shorter ones leads to better performance. Our findings suggest rethinking current methods of test-time compute in reasoning LLMs, emphasizing that longer "thinking" does not necessarily translate to improved performance and can, counter-intuitively, lead to degraded results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03422v1">RankSteer: Activation Steering for Pointwise LLM Ranking</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently shown strong performance as zero-shot rankers, yet their effectiveness is highly sensitive to prompt formulation, particularly role-play instructions. Prior analyses suggest that role-related signals are encoded along activation channels that are largely separate from query-document representations, raising the possibility of steering ranking behavior directly at the activation level rather than through brittle prompt engineering. In this work, we propose RankSteer, a post-hoc activation steering framework for zero-shot pointwise LLM ranking. We characterize ranking behavior through three disentangled and steerable directions in representation space: a \textbf{decision direction} that maps hidden states to relevance scores, an \textbf{evidence direction} that captures relevance signals not directly exploited by the decision head, and a \textbf{role direction} that modulates model behavior without injecting relevance information. Using projection-based interventions at inference time, RankSteer jointly controls these directions to calibrate ranking behavior without modifying model weights or introducing explicit cross-document comparisons. Experiments on TREC DL 20 and multiple BEIR benchmarks show that RankSteer consistently improves ranking quality using only a small number of anchor queries, demonstrating that substantial ranking capacity remains under-utilized in pointwise LLM rankers. We further provide a geometric analysis revealing that steering improves ranking by stabilizing ranking geometry and reducing dispersion, offering new insight into how LLMs internally represent and calibrate relevance judgments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.22316v4">Evaluating Scoring Bias in LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2026-02-03
      | 💬 Accepted by DASFAA 2026
    </div>
    <details class="paper-abstract">
      The "LLM-as-a-Judge" paradigm, using Large Language Models (LLMs) as automated evaluators, is pivotal to LLM development, offering scalable feedback for complex tasks. However, the reliability of these judges is compromised by various biases. Existing research has heavily concentrated on biases in comparative evaluations. In contrast, scoring-based evaluations-which assign an absolute score and are often more practical in industrial applications-remain under-investigated. To address this gap, we undertake the first dedicated examination of scoring bias in LLM judges. We shift the focus from biases tied to the evaluation targets to those originating from the scoring prompt itself. We formally define scoring bias and identify three novel, previously unstudied types: rubric order bias, score ID bias, and reference answer score bias. We propose a comprehensive framework to quantify these biases, featuring a suite of multi-faceted metrics and an automatic data synthesis pipeline to create a tailored evaluation corpus. Our experiments empirically demonstrate that even the most advanced LLMs suffer from these substantial scoring biases. Our analysis yields actionable insights for designing more robust scoring prompts and mitigating these newly identified biases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.21588v2">Understanding Verbatim Memorization in LLMs Through Circuit Discovery</a></div>
    <div class="paper-meta">
      📅 2026-02-03
      | 💬 The First Workshop on Large Language Model Memorization @ ACL 2025, Vienna, August 1st, 2025
    </div>
    <details class="paper-abstract">
      Underlying mechanisms of memorization in LLMs -- the verbatim reproduction of training data -- remain poorly understood. What exact part of the network decides to retrieve a token that we would consider as start of memorization sequence? How exactly is the models' behaviour different when producing memorized sentence vs non-memorized? In this work we approach these questions from mechanistic interpretability standpoint by utilizing transformer circuits -- the minimal computational subgraphs that perform specific functions within the model. Through carefully constructed contrastive datasets, we identify points where model generation diverges from memorized content and isolate the specific circuits responsible for two distinct aspects of memorization. We find that circuits that initiate memorization can also maintain it once started, while circuits that only maintain memorization cannot trigger its initiation. Intriguingly, memorization prevention mechanisms transfer robustly across different text domains, while memorization induction appears more context-dependent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20041v3">CiMRAG: CiM-Aware Domain-Adaptive and Noise-Resilient Retrieval-Augmented Generation for Edge-Based LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-03
      | 💬 Accepted by ICASSP 2026
    </div>
    <details class="paper-abstract">
      Personalized virtual assistants powered by large language models (LLMs) on edge devices are attracting growing attention, with Retrieval-Augmented Generation (RAG) emerging as a key method for personalization by retrieving relevant profile data and generating tailored responses. However, deploying RAG on edge devices faces efficiency hurdles due to the rapid growth of profile data, such as user-LLM interactions and recent updates. While Computing-in-Memory (CiM) architectures mitigate this bottleneck by eliminating data movement between memory and processing units via in-situ operations, they are susceptible to environmental noise that can degrade retrieval precision. This poses a critical issue in dynamic, multi-domain edge-based scenarios (e.g., travel, medicine, and law) where both accuracy and adaptability are paramount. To address these challenges, we propose Task-Oriented Noise-resilient Embedding Learning (TONEL), a framework that improves noise robustness and domain adaptability for RAG in noisy edge environments. TONEL employs a noise-aware projection model to learn task-specific embeddings compatible with CiM hardware constraints, enabling accurate retrieval under noisy conditions. Extensive experiments conducted on personalization benchmarks demonstrate the effectiveness and practicality of our methods relative to strong baselines, especially in task-specific noisy scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03359v1">MeKi: Memory-based Expert Knowledge Injection for Efficient LLM Scaling</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      Scaling Large Language Models (LLMs) typically relies on increasing the number of parameters or test-time computations to boost performance. However, these strategies are impractical for edge device deployment due to limited RAM and NPU resources. Despite hardware constraints, deploying performant LLM on edge devices such as smartphone remains crucial for user experience. To address this, we propose MeKi (Memory-based Expert Knowledge Injection), a novel system that scales LLM capacity via storage space rather than FLOPs. MeKi equips each Transformer layer with token-level memory experts that injects pre-stored semantic knowledge into the generation process. To bridge the gap between training capacity and inference efficiency, we employ a re-parameterization strategy to fold parameter matrices used during training into a compact static lookup table. By offloading the knowledge to ROM, MeKi decouples model capacity from computational cost, introducing zero inference latency overhead. Extensive experiments demonstrate that MeKi significantly outperforms dense LLM baselines with identical inference speed, validating the effectiveness of memory-based scaling paradigm for on-device LLMs. Project homepage is at https://github.com/ningding-o/MeKi.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03334v1">The Personality Trap: How LLMs Embed Bias When Generating Human-Like Personas</a></div>
    <div class="paper-meta">
      📅 2026-02-03
      | 💬 26 pages, 2 Figures
    </div>
    <details class="paper-abstract">
      This paper examines biases in large language models (LLMs) when generating synthetic populations from responses to personality questionnaires. Using five LLMs, we first assess the representativeness and potential biases in the sociodemographic attributes of the generated personas, as well as their alignment with the intended personality traits. While LLMs successfully reproduce known correlations between personality and sociodemographic variables, all models exhibit pronounced WEIRD (western, educated, industrialized, rich and democratic) biases, favoring young, educated, white, heterosexual, Western individuals with centrist or progressive political views and secular or Christian beliefs. In a second analysis, we manipulate input traits to maximize Neuroticism and Psychoticism scores. Notably, when Psychoticism is maximized, several models produce an overrepresentation of non-binary and LGBTQ+ identities, raising concerns about stereotyping and the potential pathologization of marginalized groups. Our findings highlight both the potential and the risks of using LLMs to generate psychologically grounded synthetic populations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03900v1">Knowledge Model Prompting Increases LLM Performance on Planning Tasks</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLM) can struggle with reasoning ability and planning tasks. Many prompting techniques have been developed to assist with LLM reasoning, notably Chain-of-Thought (CoT); however, these techniques, too, have come under scrutiny as LLMs' ability to reason at all has come into question. Borrowing from the domain of cognitive and educational science, this paper investigates whether the Task-Method-Knowledge (TMK) framework can improve LLM reasoning capabilities beyond its previously demonstrated success in educational applications. The TMK framework's unique ability to capture causal, teleological, and hierarchical reasoning structures, combined with its explicit task decomposition mechanisms, makes it particularly well-suited for addressing language model reasoning deficiencies, and unlike other hierarchical frameworks such as HTN and BDI, TMK provides explicit representations of not just what to do and how to do it, but also why actions are taken. The study evaluates TMK by experimenting on the PlanBench benchmark, focusing on the Blocksworld domain to test for reasoning and planning capabilities, examining whether TMK-structured prompting can help language models better decompose complex planning problems into manageable sub-tasks. Results also highlight significant performance inversion in reasoning models. TMK prompting enables the reasoning model to achieve up to an accuracy of 97.3\% on opaque, symbolic tasks (Random versions of Blocksworld in PlanBench) where it previously failed (31.5\%), suggesting the potential to bridge the gap between semantic approximation and symbolic manipulation. Our findings suggest that TMK functions not merely as context, but also as a mechanism that steers reasoning models away from their default linguistic modes to engage formal, code-execution pathways in the context of the experiments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.05742v2">Vipera: Blending Visual and LLM-Driven Guidance for Systematic Auditing of Text-to-Image Generative AI</a></div>
    <div class="paper-meta">
      📅 2026-02-03
      | 💬 17 pages, 8 figures; Accepted by CHI 2026
    </div>
    <details class="paper-abstract">
      Despite their increasing capabilities, text-to-image generative AI systems are known to produce biased, offensive, and otherwise problematic outputs. While recent advancements have supported testing and auditing of generative AI, existing auditing methods still face challenges in supporting effectively explore the vast space of AI-generated outputs in a structured way. To address this gap, we conducted formative studies with five AI auditors and synthesized five design goals for supporting systematic AI audits. Based on these insights, we developed Vipera, an interactive auditing interface that employs multiple visual cues including a scene graph to facilitate image sensemaking and inspire auditors to explore and hierarchically organize the auditing criteria. Additionally, Vipera leverages LLM-powered suggestions to facilitate exploration of unexplored auditing directions. Through a controlled experiment with 24 participants experienced in AI auditing, we demonstrate Vipera's effectiveness in helping auditors navigate large AI output spaces and organize their analyses while engaging with diverse criteria.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.00611v2">Structured Self-Consistency:A Multi-Task Evaluation of LLMs on VirtualHome</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      Embodied AI requires agents to understand goals, plan actions, and execute tasks in simulated environments. We present a comprehensive evaluation of Large Language Models (LLMs) on the VirtualHome benchmark using the Embodied Agent Interface (EAI) framework. We compare two representative 7B-parameter models OPENPANGU-7B and QWEN2.5-7B across four fundamental tasks: Goal Interpretation, Action Sequencing, Subgoal Decomposition, and Transition Modeling. We propose Structured Self-Consistency (SSC), an enhanced decoding strategy that leverages multiple sampling with domain-specific voting mechanisms to improve output quality for structured generation tasks. Experimental results demonstrate that SSC significantly enhances performance, with OPENPANGU-7B excelling at hierarchical planning while QWEN2.5-7B show advantages in action-level tasks. Our analysis reveals complementary strengths across model types, providing insights for future embodied AI system development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16540v2">Do Models Hear Like Us? Probing the Representational Alignment of Audio LLMs and Naturalistic EEG</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      Audio Large Language Models (Audio LLMs) have demonstrated strong capabilities in integrating speech perception with language understanding. However, whether their internal representations align with human neural dynamics during naturalistic listening remains largely unexplored. In this work, we systematically examine layer-wise representational alignment between 12 open-source Audio LLMs and Electroencephalogram (EEG) signals across 2 datasets. Specifically, we employ 8 similarity metrics, such as Spearman-based Representational Similarity Analysis (RSA), to characterize within-sentence representational geometry. Our analysis reveals 3 key findings: (1) we observe a rank-dependence split, in which model rankings vary substantially across different similarity metrics; (2) we identify spatio-temporal alignment patterns characterized by depth-dependent alignment peaks and a pronounced increase in RSA within the 250-500 ms time window, consistent with N400-related neural dynamics; (3) we find an affective dissociation whereby negative prosody, identified using a proposed Tri-modal Neighborhood Consistency (TNC) criterion, reduces geometric similarity while enhancing covariance-based dependence. These findings provide new neurobiological insights into the representational mechanisms of Audio LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19402v3">PROTEUS: SLA-Aware Routing via Lagrangian RL for Multi-LLM Serving Systems</a></div>
    <div class="paper-meta">
      📅 2026-02-03
      | 💬 Submitted to EuroMLSys26
    </div>
    <details class="paper-abstract">
      Production LLM deployments serve diverse workloads where cost and quality requirements vary by customer tier, time of day, and query criticality. Model serving systems accept latency SLOs directly. LLM routers do not. They force operators to tune parameters offline and guess what accuracy might result. The relationship between parameters and outcomes is indirect, non-monotonic, and dataset-dependent. Operators need to specify accuracy targets, not infer them from opaque settings. We present PROTEUS (Polymorphic Router for Operational Target Enforcement with Unified SLA), a router that accepts accuracy targets tau as runtime input. PROTEUS uses Lagrangian dual control. A learned dual variable lambda tracks constraint violations during training and conditions the policy network. This lets the router translate specified tau values into routing decisions that satisfy them. A single trained model serves the full accuracy spectrum without retraining.We evaluate on RouterBench (11 models, 405K queries) and SPROUT (14 models, 45K queries). PROTEUS achieves consistent floor compliance where accuracy meets or exceeds tau. The target-response correlation reaches 0.97 to 0.98. The closest baseline, OmniRouter, meets floors only 22% of the time despite also using Lagrangian optimization. PROTEUS operates across tau in [0.85, 0.95] from a single model. On RouterBench it achieves 90.1% accuracy, within 1.3% of oracle. On SPROUT it achieves 94.0% accuracy, within 4.6% of oracle. Cost savings reach 89.8% versus the best fixed model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03271v1">LogicScan: An LLM-driven Framework for Detecting Business Logic Vulnerabilities in Smart Contracts</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      Business logic vulnerabilities have become one of the most damaging yet least understood classes of smart contract vulnerabilities. Unlike traditional bugs such as reentrancy or arithmetic errors, these vulnerabilities arise from missing or incorrectly enforced business invariants and are tightly coupled with protocol semantics. Existing static analysis techniques struggle to capture such high-level logic, while recent large language model based approaches often suffer from unstable outputs and low accuracy due to hallucination and limited verification. In this paper, we propose LogicScan, an automated contrastive auditing framework for detecting business logic vulnerabilities in smart contracts. The key insight behind LogicScan is that mature, widely deployed on-chain protocols implicitly encode well-tested and consensus-driven business invariants. LogicScan systematically mines these invariants from large-scale on-chain contracts and reuses them as reference constraints to audit target contracts. To achieve this, LogicScan introduces a Business Specification Language (BSL) to normalize diverse implementation patterns into structured, verifiable logic representations. It further combines noise-aware logic aggregation with contrastive auditing to identify missing or weakly enforced invariants while mitigating LLM-induced false positives. We evaluate LogicScan on three real-world datasets, including DeFiHacks, Web3Bugs, and a set of top-200 audited contracts. The results show that LogicScan achieves an F1 score of 85.2%, significantly outperforming state-of-the-art tools while maintaining a low false-positive rate on production-grade contracts. Additional experiments demonstrate that LogicScan maintains consistent performance across different LLMs and is cost-effective, and that its false-positive suppression mechanisms substantially improve robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14943v2">State of the Art of LLM-Enabled Interaction with Visualization</a></div>
    <div class="paper-meta">
      📅 2026-02-03
      | 💬 Submitted to STARs of EuroVis'26
    </div>
    <details class="paper-abstract">
      We report on a systematic, PRISMA-guided survey of research at the intersection of LLMs and visualization, with a particular focus on visio-verbal interaction -- where verbal and visual modalities converge to support data sense-making. The emergence of Large Language Models (LLMs) has introduced new paradigms for interacting with data visualizations through natural language, leading to intuitive, multimodal, and accessible interfaces. We analyze 48 papers across six dimensions: application domain, visualization task, visualization representation, interaction modality, LLM integration, and system evaluation. Our classification framework maps LLM roles across the visualization pipeline, from data querying and transformation to visualization generation, explanation, and navigation. We highlight emerging design patterns, identify gaps in accessibility and visualization reading, and discuss the limitations of current LLMs in spatial reasoning and contextual grounding. We further reflect on evaluations of combined LLM-visualization systems, highlighting how current research projects tackle this challenge and discuss current gaps in conducting meaningful evaluations of such systems. With our survey we aim to guide future research and system design in LLM-enhanced visualization, supporting broad audiences and intelligent, conversational interfaces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03249v1">Accordion-Thinking: Self-Regulated Step Summaries for Efficient and Readable LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      Scaling test-time compute via long Chain-ofThought unlocks remarkable gains in reasoning capabilities, yet it faces practical limits due to the linear growth of KV cache and quadratic attention complexity. In this paper, we introduce Accordion-Thinking, an end-to-end framework where LLMs learn to self-regulate the granularity of the reasoning steps through dynamic summarization. This mechanism enables a Fold inference mode, where the model periodically summarizes its thought process and discards former thoughts to reduce dependency on historical tokens. We apply reinforcement learning to incentivize this capability further, uncovering a critical insight: the accuracy gap between the highly efficient Fold mode and the exhaustive Unfold mode progressively narrows and eventually vanishes over the course of training. This phenomenon demonstrates that the model learns to encode essential reasoning information into compact summaries, achieving effective compression of the reasoning context. Our Accordion-Thinker demonstrates that with learned self-compression, LLMs can tackle complex reasoning tasks with minimal dependency token overhead without compromising solution quality, and it achieves a 3x throughput while maintaining accuracy on a 48GB GPU memory configuration, while the structured step summaries provide a human-readable account of the reasoning process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03238v1">The Necessity of a Unified Framework for LLM-Based Agent Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      With the advent of Large Language Models (LLMs), general-purpose agents have seen fundamental advancements. However, evaluating these agents presents unique challenges that distinguish them from static QA benchmarks. We observe that current agent benchmarks are heavily confounded by extraneous factors, including system prompts, toolset configurations, and environmental dynamics. Existing evaluations often rely on fragmented, researcher-specific frameworks where the prompt engineering for reasoning and tool usage varies significantly, making it difficult to attribute performance gains to the model itself. Additionally, the lack of standardized environmental data leads to untraceable errors and non-reproducible results. This lack of standardization introduces substantial unfairness and opacity into the field. We propose that a unified evaluation framework is essential for the rigorous advancement of agent evaluation. To this end, we introduce a proposal aimed at standardizing agent evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03237v1">Merging Beyond: Streaming LLM Updates via Activation-Guided Rotations</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      The escalating scale of Large Language Models (LLMs) necessitates efficient adaptation techniques. Model merging has gained prominence for its efficiency and controllability. However, existing merging techniques typically serve as post-hoc refinements or focus on mitigating task interference, often failing to capture the dynamic optimization benefits of supervised fine-tuning (SFT). In this work, we propose Streaming Merging, an innovative model updating paradigm that conceptualizes merging as an iterative optimization process. Central to this paradigm is \textbf{ARM} (\textbf{A}ctivation-guided \textbf{R}otation-aware \textbf{M}erging), a strategy designed to approximate gradient descent dynamics. By treating merging coefficients as learning rates and deriving rotation vectors from activation subspaces, ARM effectively steers parameter updates along data-driven trajectories. Unlike conventional linear interpolation, ARM aligns semantic subspaces to preserve the geometric structure of high-dimensional parameter evolution. Remarkably, ARM requires only early SFT checkpoints and, through iterative merging, surpasses the fully converged SFT model. Experimental results across model scales (1.7B to 14B) and diverse domains (e.g., math, code) demonstrate that ARM can transcend converged checkpoints. Extensive experiments show that ARM provides a scalable and lightweight framework for efficient model adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03226v1">ATACompressor: Adaptive Task-Aware Compression for Efficient Long-Context Processing in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      Long-context inputs in large language models (LLMs) often suffer from the "lost in the middle" problem, where critical information becomes diluted or ignored due to excessive length. Context compression methods aim to address this by reducing input size, but existing approaches struggle with balancing information preservation and compression efficiency. We propose Adaptive Task-Aware Compressor (ATACompressor), which dynamically adjusts compression based on the specific requirements of the task. ATACompressor employs a selective encoder that compresses only the task-relevant portions of long contexts, ensuring that essential information is preserved while reducing unnecessary content. Its adaptive allocation controller perceives the length of relevant content and adjusts the compression rate accordingly, optimizing resource utilization. We evaluate ATACompressor on three QA datasets: HotpotQA, MSMARCO, and SQUAD-showing that it outperforms existing methods in terms of both compression efficiency and task performance. Our approach provides a scalable solution for long-context processing in LLMs. Furthermore, we perform a range of ablation studies and analysis experiments to gain deeper insights into the key components of ATACompressor.
    </details>
</div>
